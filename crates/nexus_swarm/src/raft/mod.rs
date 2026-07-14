//! Raft Consensus Engine Module Declaration

pub mod leader_election;
pub mod heartbeat_monitor;

pub use leader_election::LeaderElection;
pub use heartbeat_monitor::HeartbeatMonitor;

use crate::{
    ConsensusError, ConsensusResult, LogEntry, LogIndex, NodeId, NodeState, RaftMessage, 
    SwarmConfig, Term, LockFreeRaftLog, STONITHFencer, LeaderLeaseToken,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, RwLock};

/// Main swarm consensus engine coordinating all Raft operations
pub struct SwarmConsensusEngine {
    pub config: SwarmConfig,
    pub node_id: NodeId,
    pub state: NodeState,
    pub current_term: Term,
    pub current_leader: Option<NodeId>,
    pub election: LeaderElection,
    pub heartbeat_monitor: HeartbeatMonitor,
    pub raft_log: Arc<LockFreeRaftLog>,
    pub lease_token: Option<LeaderLeaseToken>,
    pub stonith_fencer: Option<STONITHFencer>,
    last_heartbeat_sent: Instant,
    last_election_time: Option<Instant>,
}

impl SwarmConsensusEngine {
    pub fn new(config: SwarmConfig) -> ConsensusResult<Self> {
        let node_id = config.node_id;
        let election = LeaderElection::new(&config);
        let heartbeat_monitor = HeartbeatMonitor::new(&config);
        let raft_log = Arc::new(LockFreeRaftLog::new(&config));
        
        let stonith_fencer = if config.stonith_enabled {
            Some(STONITHFencer::new(&config)?)
        } else {
            None
        };

        Ok(Self {
            config,
            node_id,
            state: NodeState::Follower,
            current_term: 0,
            current_leader: None,
            election,
            heartbeat_monitor,
            raft_log,
            lease_token: None,
            stonith_fencer,
            last_heartbeat_sent: Instant::now(),
            last_election_time: None,
        })
    }

    /// Process heartbeats and check for leader timeout
    pub async fn process_heartbeats(&mut self) -> ConsensusResult<()> {
        match self.state {
            NodeState::Leader => {
                // Send heartbeats to all followers
                self.send_heartbeats().await?;
            }
            NodeState::Follower | NodeState::Candidate => {
                // Check if leader has timed out
                if self.heartbeat_monitor.check_leader_timeout() {
                    self.start_election().await?;
                }
            }
            NodeState::Terminated => {
                return Err(ConsensusError::KamikazeInitiated);
            }
        }
        Ok(())
    }

    /// Send heartbeats to all cluster nodes
    async fn send_heartbeats(&mut self) -> ConsensusResult<()> {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_heartbeat_sent);
        
        if elapsed < self.config.heartbeat_interval_ms.into() {
            return Ok(());
        }

        self.last_heartbeat_sent = now;
        self.current_term = self.heartbeat_monitor.current_term();
        
        // In production, this would broadcast via UDP/TCP to all followers
        tracing::debug!("Leader {} sending heartbeat for term {}", self.node_id, self.current_term);
        
        Ok(())
    }

    /// Start a new leader election
    async fn start_election(&mut self) -> ConsensusResult<()> {
        self.last_election_time = Some(Instant::now());
        let vote_request = self.election.start_election();
        
        tracing::info!(
            "Node {} starting election for term {}",
            self.node_id,
            self.election.current_term()
        );
        
        self.state = NodeState::Candidate;
        self.current_term = self.election.current_term();
        self.heartbeat_monitor.reset(self.current_term);
        
        // In production, broadcast vote request to all nodes
        // and collect responses
        
        Ok(())
    }

    /// Handle becoming the leader - includes STONITH fencing
    pub async fn become_leader(&mut self) -> ConsensusResult<()> {
        if self.state != NodeState::Candidate && self.state != NodeState::Follower {
            return Err(ConsensusError::NotLeader(self.node_id));
        }

        self.state = NodeState::Leader;
        self.current_leader = Some(self.node_id);
        self.current_term = self.election.current_term();

        // CRITICAL: Execute STONITH fencing before accepting leadership
        if let Some(ref mut fencer) = self.stonith_fencer {
            tracing::warn!("Executing STONITH fencing before assuming leadership");
            
            // Fence any previous leader
            let previous_term = self.current_term.saturating_sub(1);
            fencer.execute_fence(previous_term, self.node_id).await?;
        }

        // Generate new lease token for cryptographic authority
        self.lease_token = Some(LeaderLeaseToken::new(
            self.node_id,
            self.current_term,
            Duration::from_millis(self.config.lease_duration_ms),
        ));

        tracing::info!(
            "Node {} became leader for term {} with lease token",
            self.node_id,
            self.current_term
        );

        Ok(())
    }

    /// Handle stepping down from leadership
    pub fn step_down(&mut self, new_term: Term) {
        self.state = NodeState::Follower;
        self.current_term = new_term;
        self.current_leader = None;
        self.lease_token = None;
        self.heartbeat_monitor.reset(new_term);
        
        tracing::info!("Node {} stepped down to follower for term {}", self.node_id, new_term);
    }

    /// Check if this node can execute trades (is leader with valid lease)
    pub fn can_execute_trades(&self) -> bool {
        self.state == NodeState::Leader && 
        self.lease_token.as_ref().map_or(false, |t| t.is_valid())
    }

    /// Get the current lease token for order signing
    pub fn get_lease_token(&self) -> Option<&LeaderLeaseToken> {
        self.lease_token.as_ref()
    }

    /// Get current node state
    pub fn get_state(&self) -> NodeState {
        self.state
    }

    /// Get current term
    pub fn get_term(&self) -> Term {
        self.current_term
    }

    /// Get current leader ID
    pub fn get_leader(&self) -> Option<NodeId> {
        self.current_leader
    }

    /// Append an entry to the Raft log (leader only)
    pub fn append_log_entry(&self, entry: LogEntry) -> ConsensusResult<LogIndex> {
        if self.state != NodeState::Leader {
            return Err(ConsensusError::NotLeader(self.node_id));
        }
        self.raft_log.append(entry)
    }

    /// Commit entries up to the given index
    pub fn commit_entries(&self, up_to_index: LogIndex) -> ConsensusResult<()> {
        if self.state != NodeState::Leader {
            return Err(ConsensusError::NotLeader(self.node_id));
        }
        self.raft_log.commit(up_to_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_swarm_engine_creation() {
        let mut config = SwarmConfig::default();
        config.node_id = 1;
        config.cluster_nodes = vec![1, 2, 3];
        config.stonith_enabled = false; // Disable for unit test
        
        let engine = SwarmConsensusEngine::new(config).unwrap();
        
        assert_eq!(engine.node_id, 1);
        assert_eq!(engine.state, NodeState::Follower);
        assert!(!engine.can_execute_trades());
    }
}
