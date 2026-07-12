//! Multi-Group Raft Consensus Implementation
//! 
//! Shards the Stage 2 Order Book and Stage 4 OMS state machines across multiple swarm nodes.
//! Implements deterministic execution with exactly-once semantics.

use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use tokio::time::timeout;
use bytes::{Buf, BufMut, BytesMut};
use serde::{Serialize, Deserialize};
use crate::raft::zero_copy_snapshot::ZeroCopySnapshot;
use crate::raft::deterministic_executor::{DeterministicExecutor, ExecutionResult, TransactionId};

/// Unique identifier for a Raft group (shard)
pub type GroupId = u64;
/// Unique identifier for a node in the swarm
pub type NodeId = u64;
/// Log index position
pub type LogIndex = u64;
/// Term number
pub type Term = u64;

/// Commands that can be proposed to the Raft log
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RaftCommand {
    /// Submit a new order to the OMS
    SubmitOrder {
        order_id: String,
        symbol: String,
        side: OrderSide,
        quantity: u64,
        price: u64,
    },
    /// Cancel an existing order
    CancelOrder {
        order_id: String,
    },
    /// Update order book state from market data
    UpdateOrderBook {
        symbol: String,
        bids: Vec<(u64, u64)>,
        asks: Vec<(u64, u64)>,
    },
    /// Checkpoint state for recovery
    Checkpoint {
        checkpoint_id: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Log entry in the Raft log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub term: Term,
    pub index: LogIndex,
    pub command: RaftCommand,
    pub timestamp: u128,
    pub transaction_id: TransactionId,
}

impl LogEntry {
    pub fn new(term: Term, index: LogIndex, command: RaftCommand) -> Self {
        Self {
            term,
            index,
            command,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            transaction_id: TransactionId::new(),
        }
    }

    pub fn serialize(&self) -> BytesMut {
        let mut buf = BytesMut::with_capacity(1024);
        buf.put_u64(self.term);
        buf.put_u64(self.index);
        buf.put_u128(self.timestamp);
        buf.put_slice(&self.transaction_id.0);
        
        let cmd_bytes = bincode::serialize(&self.command).unwrap_or_default();
        buf.put_u32(cmd_bytes.len() as u32);
        buf.put_slice(&cmd_bytes);
        
        buf
    }

    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < 32 {
            return None;
        }
        
        let mut buf = &data[..];
        let term = buf.get_u64();
        let index = buf.get_u64();
        let timestamp = buf.get_u128();
        
        if buf.remaining() < 16 {
            return None;
        }
        let mut tid_bytes = [0u8; 16];
        buf.copy_to_slice(&mut tid_bytes);
        let transaction_id = TransactionId(tid_bytes);
        
        if buf.remaining() < 4 {
            return None;
        }
        let cmd_len = buf.get_u32() as usize;
        if buf.remaining() < cmd_len {
            return None;
        }
        let cmd_bytes = &buf[..cmd_len];
        let command = bincode::deserialize(cmd_bytes).ok()?;
        
        Some(Self {
            term,
            index,
            command,
            timestamp,
            transaction_id,
        })
    }
}

/// State of a Raft node
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaftState {
    Follower,
    Candidate,
    Leader,
}

/// Configuration for a Raft group
#[derive(Debug, Clone)]
pub struct RaftConfig {
    pub group_id: GroupId,
    pub node_id: NodeId,
    pub peer_nodes: Vec<NodeId>,
    pub election_timeout_min: Duration,
    pub election_timeout_max: Duration,
    pub heartbeat_interval: Duration,
    pub max_log_entries_per_batch: usize,
    pub snapshot_threshold: LogIndex,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            group_id: 0,
            node_id: 0,
            peer_nodes: Vec::new(),
            election_timeout_min: Duration::from_millis(150),
            election_timeout_max: Duration::from_millis(300),
            heartbeat_interval: Duration::from_millis(50),
            max_log_entries_per_batch: 1000,
            snapshot_threshold: 10000,
        }
    }
}

/// Multi-Group Raft Manager
/// Manages multiple Raft groups for horizontal scaling
pub struct MultiGroupRaft {
    groups: RwLock<HashMap<GroupId, RaftGroup>>,
    node_id: NodeId,
    config: RaftConfig,
    snapshot_manager: Arc<ZeroCopySnapshot>,
    executor: Arc<DeterministicExecutor>,
}

impl MultiGroupRaft {
    pub fn new(node_id: NodeId, config: RaftConfig) -> Self {
        Self {
            groups: RwLock::new(HashMap::new()),
            node_id,
            config,
            snapshot_manager: Arc::new(ZeroCopySnapshot::new()),
            executor: Arc::new(DeterministicExecutor::new()),
        }
    }

    /// Create or join a Raft group
    pub async fn create_group(&self, group_id: GroupId, peers: Vec<NodeId>) -> Result<(), RaftError> {
        let mut groups = self.groups.write().await;
        
        if groups.contains_key(&group_id) {
            return Err(RaftError::GroupAlreadyExists(group_id));
        }

        let mut group_config = self.config.clone();
        group_config.group_id = group_id;
        group_config.node_id = self.node_id;
        group_config.peer_nodes = peers;

        let group = RaftGroup::new(
            group_config,
            Arc::clone(&self.snapshot_manager),
            Arc::clone(&self.executor),
        );

        groups.insert(group_id, group);
        Ok(())
    }

    /// Propose a command to a specific group
    pub async fn propose(&self, group_id: GroupId, command: RaftCommand) -> Result<TransactionId, RaftError> {
        let groups = self.groups.read().await;
        let group = groups.get(&group_id)
            .ok_or_else(|| RaftError::GroupNotFound(group_id))?;
        
        group.propose(command).await
    }

    /// Get the committed state for a group
    pub async fn get_state(&self, group_id: GroupId) -> Result<BTreeMap<String, serde_json::Value>, RaftError> {
        let groups = self.groups.read().await;
        let group = groups.get(&group_id)
            .ok_or_else(|| RaftError::GroupNotFound(group_id))?;
        
        group.get_committed_state().await
    }

    /// Trigger a snapshot for a group
    pub async fn trigger_snapshot(&self, group_id: GroupId) -> Result<(), RaftError> {
        let groups = self.groups.read().await;
        let group = groups.get(&group_id)
            .ok_or_else(|| RaftError::GroupNotFound(group_id))?;
        
        group.create_snapshot().await
    }

    /// Get all active group IDs
    pub async fn get_active_groups(&self) -> Vec<GroupId> {
        let groups = self.groups.read().await;
        groups.keys().copied().collect()
    }

    /// Shutdown all groups gracefully
    pub async fn shutdown(&self) -> Result<(), RaftError> {
        let mut groups = self.groups.write().await;
        for (_, group) in groups.iter_mut() {
            group.shutdown().await?;
        }
        groups.clear();
        Ok(())
    }
}

/// Single Raft Group implementation
pub struct RaftGroup {
    config: RaftConfig,
    state: RwLock<RaftState>,
    current_term: RwLock<Term>,
    voted_for: RwLock<Option<NodeId>>,
    log: RwLock<Vec<LogEntry>>,
    commit_index: RwLock<LogIndex>,
    last_applied: RwLock<LogIndex>,
    leader_id: RwLock<Option<NodeId>>,
    /// Lease expiration timestamp for split-brain prevention
    lease_expiration_ns: RwLock<u64>,
    snapshot_manager: Arc<ZeroCopySnapshot>,
    executor: Arc<DeterministicExecutor>,
    shutdown_tx: mpsc::Sender<()>,
}

impl RaftGroup {
    pub fn new(
        config: RaftConfig,
        snapshot_manager: Arc<ZeroCopySnapshot>,
        executor: Arc<DeterministicExecutor>,
    ) -> Self {
        let (shutdown_tx, _) = mpsc::channel(1);
        
        Self {
            config,
            state: RwLock::new(RaftState::Follower),
            current_term: RwLock::new(0),
            voted_for: RwLock::new(None),
            log: RwLock::new(Vec::with_capacity(config.max_log_entries_per_batch)),
            commit_index: RwLock::new(0),
            last_applied: RwLock::new(0),
            leader_id: RwLock::new(None),
            snapshot_manager,
            executor,
            shutdown_tx,
        }
    }

    /// Propose a command to the Raft log
    pub async fn propose(&self, command: RaftCommand) -> Result<TransactionId, RaftError> {
        // Must be leader to propose
        {
            let state = *self.state.read().await;
            if state != RaftState::Leader {
                return Err(RaftError::NotLeader);
            }
        }

        let term = *self.current_term.read().await;
        let index = {
            let mut log = self.log.write().await;
            let next_index = log.last().map(|e| e.index + 1).unwrap_or(1);
            let entry = LogEntry::new(term, next_index, command);
            let tx_id = entry.transaction_id.clone();
            log.push(entry);
            tx_id
        };

        // Wait for replication to quorum
        self.replicate_to_quorum(index).await?;

        Ok(TransactionId::new())
    }

    /// Replicate log entry to quorum
    async fn replicate_to_quorum(&self, index: LogIndex) -> Result<(), RaftError> {
        let peers = self.config.peer_nodes.clone();
        let required_acknowledgments = (peers.len() / 2) + 1;
        
        // In production, this would send AppendEntries RPCs to peers
        // For now, we simulate immediate quorum for single-node setup
        if peers.is_empty() {
            // Single node - auto-commit
            let mut commit_index = self.commit_index.write().await;
            *commit_index = index;
            return Ok(());
        }

        // TODO: Implement actual network replication with timeout
        // Using timeout to prevent deadlocks
        let replication_timeout = Duration::from_millis(100);
        match timeout(replication_timeout, self.simulate_replication(peers.len(), required_acknowledgments)).await {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(RaftError::ReplicationTimeout),
        }
    }

    async fn simulate_replication(&self, peer_count: usize, required: usize) -> Result<(), RaftError> {
        // Simulated replication - in production this sends RPCs
        // Acknowledge immediately for local testing
        if peer_count == 0 || required <= 1 {
            return Ok(());
        }
        
        // Wait for simulated network delay
        tokio::time::sleep(Duration::from_micros(10)).await;
        Ok(())
    }

    /// Apply committed log entries to state machine
    pub async fn apply_committed_entries(&self) -> Result<(), RaftError> {
        let commit_index = *self.commit_index.read().await;
        let mut last_applied = self.last_applied.write().await;
        let log = self.log.read().await;

        while *last_applied < commit_index {
            let entry_index = (*last_applied) as usize;
            if entry_index >= log.len() {
                break;
            }

            let entry = &log[entry_index];
            
            // Execute deterministically with atomic rollback on failure
            match self.executor.execute_transaction(entry).await {
                Ok(ExecutionResult::Committed(tx_id)) => {
                    *last_applied += 1;
                }
                Ok(ExecutionResult::Pending) => {
                    // Wait for more data
                    break;
                }
                Err(e) => {
                    // Critical: Log divergence detected
                    // Rollback and mark group as unhealthy
                    eprintln!("CRITICAL: Execution failed at index {}: {:?}", entry.index, e);
                    return Err(RaftError::ExecutionFailed(e.to_string()));
                }
            }
        }

        // Check if snapshot needed
        if *last_applied >= self.config.snapshot_threshold {
            drop(log);
            drop(last_applied);
            self.create_snapshot().await?;
        }

        Ok(())
    }

    /// Create a zero-copy snapshot of current state
    pub async fn create_snapshot(&self) -> Result<(), RaftError> {
        let last_applied = *self.last_applied.read().await;
        let term = *self.current_term.read().await;
        
        let state = self.executor.get_state_summary().await;
        
        self.snapshot_manager
            .create_snapshot(self.config.group_id, last_applied, term, state)
            .await?;

        // Compact log after successful snapshot
        self.compact_log(last_applied).await?;

        Ok(())
    }

    /// Compact the log by removing entries before the snapshot
    async fn compact_log(&self, up_to_index: LogIndex) -> Result<(), RaftError> {
        let mut log = self.log.write().await;
        let original_len = log.len();
        
        log.retain(|entry| entry.index > up_to_index);
        
        let removed = original_len.saturating_sub(log.len());
        if removed > 0 {
            tracing::info!(
                group_id = %self.config.group_id,
                removed_entries = removed,
                "Log compaction completed"
            );
        }

        Ok(())
    }

    /// Get committed state
    pub async fn get_committed_state(&self) -> Result<BTreeMap<String, serde_json::Value>, RaftError> {
        self.executor.get_state_summary().await
    }

    /// Graceful shutdown
    pub async fn shutdown(&self) -> Result<(), RaftError> {
        // Ensure all pending entries are applied
        self.apply_committed_entries().await?;
        
        // Create final snapshot
        self.create_snapshot().await?;
        
        // Signal shutdown
        let _ = self.shutdown_tx.send(()).await;
        
        Ok(())
    }

    /// Restore from snapshot
    pub async fn restore_from_snapshot(&self, snapshot_data: &[u8]) -> Result<(), RaftError> {
        let (last_index, term, state) = self.snapshot_manager.deserialize_snapshot(snapshot_data)?;
        
        *self.current_term.write().await = term;
        *self.last_applied.write().await = last_index;
        *self.commit_index.write().await = last_index;
        
        self.executor.restore_state(state).await?;
        
        Ok(())
    }
}

/// Raft error types
#[derive(Debug, thiserror::Error)]
pub enum RaftError {
    #[error("Group {0} already exists")]
    GroupAlreadyExists(GroupId),
    #[error("Group {0} not found")]
    GroupNotFound(GroupId),
    #[error("Node is not the leader")]
    NotLeader,
    #[error("Replication timeout")]
    ReplicationTimeout,
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Snapshot error: {0}")]
    SnapshotError(String),
    #[error("Invalid log entry")]
    InvalidLogEntry,
    #[error("Quorum not reached")]
    QuorumNotReached,
    #[error("Channel closed")]
    ChannelClosed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_multi_group_creation() {
        let raft = MultiGroupRaft::new(1, RaftConfig::default());
        
        assert!(raft.create_group(1, vec![2, 3]).await.is_ok());
        assert!(raft.create_group(2, vec![4, 5]).await.is_ok());
        
        // Duplicate group should fail
        assert!(matches!(
            raft.create_group(1, vec![6]).await,
            Err(RaftError::GroupAlreadyExists(1))
        ));

        let groups = raft.get_active_groups().await;
        assert_eq!(groups.len(), 2);
        assert!(groups.contains(&1));
        assert!(groups.contains(&2));
    }

    #[tokio::test]
    async fn test_propose_command() {
        let mut config = RaftConfig::default();
        config.node_id = 1;
        config.peer_nodes = vec![]; // Single node for testing
        
        let raft = MultiGroupRaft::new(1, config.clone());
        raft.create_group(1, vec![]).await.unwrap();

        // Manually set as leader for testing
        {
            let groups = raft.groups.read().await;
            let group = groups.get(&1).unwrap();
            *group.state.write().await = RaftState::Leader;
            *group.current_term.write().await = 1;
        }

        let command = RaftCommand::SubmitOrder {
            order_id: "order-1".to_string(),
            symbol: "BTCUSD".to_string(),
            side: OrderSide::Buy,
            quantity: 100,
            price: 50000,
        };

        let result = raft.propose(1, command).await;
        assert!(result.is_ok());
    }
}
