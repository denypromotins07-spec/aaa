//! NEXUS-OMEGA Swarm Consensus Module
//! 
//! Implements Raft-based leader election, OMS state replication, and STONITH fencing
//! to ensure only one node executes trades at any time while maintaining high availability.

pub mod raft;
pub mod gating;
pub mod replication;
pub mod fencing;

pub use raft::{SwarmConsensusEngine, LeaderElection, HeartbeatMonitor};
pub use gating::{LeaderLeaseToken, CryptographicAuthorityGate};
pub use replication::{OMSStateReplicator, LockFreeRaftLog, LogCompactionSnapshot, OMSState, StateSnapshot};
pub use fencing::{STONITHFencer, FencingResult, QuorumWitness, WitnessState, KamikazeProtocol, KamikazeState};

use std::sync::Arc;
use tokio::sync::RwLock;

/// Unique identifier for a node in the swarm
pub type NodeId = u64;

/// Term number for Raft consensus
pub type Term = u64;

/// Log index in the Raft log
pub type LogIndex = u64;

/// Result type for consensus operations
pub type ConsensusResult<T> = Result<T, ConsensusError>;

/// Errors that can occur in the swarm consensus layer
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConsensusError {
    #[error("Not the leader: current leader is {0}")]
    NotLeader(NodeId),
    
    #[error("Lease expired: term {term}, expected valid until {valid_until}")]
    LeaseExpired { term: Term, valid_until: u64 },
    
    #[error("Quorum lost: cannot reach consensus")]
    QuorumLost,
    
    #[error("Split brain detected: multiple leaders")]
    SplitBrain,
    
    #[error("STONITH fencing failed: {0}")]
    StonithFailed(String),
    
    #[error("Log corruption detected at index {index}: {reason}")]
    LogCorruption { index: LogIndex, reason: String },
    
    #[error("Replication lag too high: {lag_ms}ms behind leader")]
    ReplicationLag { lag_ms: u64 },
    
    #[error("Kamikaze protocol initiated: self-terminating to protect portfolio")]
    KamikazeInitiated,
}

/// Configuration for the swarm consensus engine
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    pub node_id: NodeId,
    pub cluster_nodes: Vec<NodeId>,
    pub heartbeat_interval_ms: u64,
    pub election_timeout_ms: u64,
    pub lease_duration_ms: u64,
    pub stonith_enabled: bool,
    pub stonith_timeout_ms: u64,
    pub quorum_witness_urls: Vec<String>,
    pub max_log_entries_before_compaction: usize,
    pub snapshot_interval_ms: u64,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            node_id: 0,
            cluster_nodes: vec![],
            heartbeat_interval_ms: 50,
            election_timeout_ms: 150,
            lease_duration_ms: 500,
            stonith_enabled: true,
            stonith_timeout_ms: 2000,
            quorum_witness_urls: vec![],
            max_log_entries_before_compaction: 10000,
            snapshot_interval_ms: 60000,
        }
    }
}

/// State of a node in the swarm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    Follower,
    Candidate,
    Leader,
    Terminated,
}

/// Message types for inter-node communication
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum RaftMessage {
    RequestVote {
        term: Term,
        candidate_id: NodeId,
        last_log_index: LogIndex,
        last_log_term: Term,
    },
    VoteResponse {
        term: Term,
        vote_granted: bool,
        voter_id: NodeId,
    },
    AppendEntries {
        term: Term,
        leader_id: NodeId,
        prev_log_index: LogIndex,
        prev_log_term: Term,
        entries: Vec<LogEntry>,
        leader_commit: LogIndex,
    },
    AppendEntriesResponse {
        term: Term,
        success: bool,
        match_index: LogIndex,
        responder_id: NodeId,
    },
    Heartbeat {
        term: Term,
        leader_id: NodeId,
    },
    HeartbeatResponse {
        term: Term,
        follower_id: NodeId,
    },
    FenceRequest {
        term: Term,
        target_node_id: NodeId,
        requestor_id: NodeId,
    },
    FenceResponse {
        success: bool,
        message: String,
    },
}

/// A single entry in the Raft log representing a state mutation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogEntry {
    pub index: LogIndex,
    pub term: Term,
    pub timestamp_ns: u64,
    pub entry_type: LogEntryType,
    pub data: Vec<u8>,
    pub crc32: u32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum LogEntryType {
    NoOp,
    OrderSubmission { order_id: String },
    OrderCancellation { order_id: String },
    PositionUpdate { symbol: String, delta: i64 },
    PnLUpdate { delta: i64 },
    StateSnapshot { snapshot_id: u64 },
}

impl LogEntry {
    pub fn new(index: LogIndex, term: Term, entry_type: LogEntryType, data: Vec<u8>) -> Self {
        let crc32 = crc32fast::hash(&data);
        Self {
            index,
            term,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            entry_type,
            data,
            crc32,
        }
    }
    
    pub fn verify_integrity(&self) -> bool {
        self.crc32 == crc32fast::hash(&self.data)
    }
}

/// Creates and starts the swarm consensus engine with a dedicated runtime
pub async fn start_swarm_consensus(
    config: SwarmConfig,
) -> ConsensusResult<Arc<RwLock<SwarmConsensusEngine>>> {
    let engine = Arc::new(RwLock::new(SwarmConsensusEngine::new(config)?));
    
    // Start background tasks
    let engine_clone = Arc::clone(&engine);
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_millis(
                engine_clone.read().await.config.heartbeat_interval_ms
            )).await;
            
            let mut guard = engine_clone.write().await;
            if let Err(e) = guard.process_heartbeats().await {
                tracing::warn!("Heartbeat processing error: {:?}", e);
            }
        }
    });
    
    Ok(engine)
}
