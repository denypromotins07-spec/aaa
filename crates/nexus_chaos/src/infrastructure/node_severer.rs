// STAGE 25: CHAPTER 1 - NODE SEVERER
// Tests Stage 22 Swarm Raft consensus and STONITH fencing
// Simulates complete node disconnection to force leader election

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use serde::{Deserialize, Serialize};

/// Node severing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeverConfig {
    pub sever_duration_ms: u64,
    pub target_node_ids: Vec<u64>,
    pub trigger_raft_election: bool,
    pub simulate_network_partition: bool,
}

/// Sever event types
#[derive(Debug, Clone, PartialEq)]
pub enum SeverEvent {
    NetworkDisconnected(u64), // node_id
    RaftLeadershipRevoked(u64),
    StonithTriggered(u64),
    ConnectionRestored(u64),
}

/// Node state tracking
pub struct NodeState {
    pub is_connected: AtomicBool,
    pub last_heartbeat: AtomicU64,
    pub sever_count: AtomicU64,
    pub is_raft_leader: AtomicBool,
}

impl Default for NodeState {
    fn default() -> Self {
        Self {
            is_connected: AtomicBool::new(true),
            last_heartbeat: AtomicU64::new(0),
            sever_count: AtomicU64::new(0),
            is_raft_leader: AtomicBool::new(false),
        }
    }
}

/// Node Severer for chaos testing
/// Forces node disconnection to test swarm resilience
pub struct NodeSeverer {
    nodes: std::collections::HashMap<u64, std::sync::Arc<NodeState>>,
    config: SeverConfig,
    chaos_mode_flag: AtomicBool,
    event_tx: mpsc::Sender<SeverEvent>,
}

impl NodeSeverer {
    pub fn new(config: SeverConfig) -> (Self, mpsc::Receiver<SeverEvent>) {
        let (tx, rx) = mpsc::channel(1024);
        let mut nodes = std::collections::HashMap::new();

        // Initialize nodes from config
        for &node_id in &config.target_node_ids {
            nodes.insert(node_id, std::sync::Arc::new(NodeState::default()));
        }

        (
            Self {
                nodes,
                config,
                chaos_mode_flag: AtomicBool::new(false),
                event_tx: tx,
            },
            rx,
        )
    }

    /// Enable chaos mode
    pub fn enable_chaos_mode(&self) {
        self.chaos_mode_flag.store(true, Ordering::SeqCst);
    }

    /// Disable chaos mode
    pub fn disable_chaos_mode(&self) {
        self.chaos_mode_flag.store(false, Ordering::SeqCst);
    }

    /// Check if chaos mode is active
    pub fn is_chaos_mode(&self) -> bool {
        self.chaos_mode_flag.load(Ordering::SeqCst)
    }

    /// Register a new node
    pub fn register_node(&mut self, node_id: u64) {
        self.nodes.insert(node_id, std::sync::Arc::new(NodeState::default()));
    }

    /// Sever a specific node's network connection
    /// Returns true if sever was successful
    pub async fn sever_node(&self, node_id: u64) -> Result<bool, SeverError> {
        if !self.chaos_mode_flag.load(Ordering::SeqCst) {
            return Err(SeverError::ChaosModeNotActive);
        }

        let node_state = match self.nodes.get(&node_id) {
            Some(state) => state.clone(),
            None => return Err(SeverError::NodeNotFound(node_id)),
        };

        // Mark node as disconnected
        node_state.is_connected.store(false, Ordering::SeqCst);
        node_state.sever_count.fetch_add(1, Ordering::Relaxed);

        // Send sever event
        let event = SeverEvent::NetworkDisconnected(node_id);
        let _ = self.event_tx.try_send(event);

        // If configured, trigger Raft election
        if self.config.trigger_raft_election && node_state.is_raft_leader.load(Ordering::Relaxed) {
            node_state.is_raft_leader.store(false, Ordering::SeqCst);
            let _ = self.event_tx.try_send(SeverEvent::RaftLeadershipRevoked(node_id));
        }

        // Schedule restoration after duration
        let restore_after = Duration::from_millis(self.config.sever_duration_ms);
        let event_tx = self.event_tx.clone();
        
        tokio::spawn(async move {
            tokio::time::sleep(restore_after).await;
            let _ = event_tx.try_send(SeverEvent::ConnectionRestored(node_id));
        });

        Ok(true)
    }

    /// Trigger STONITH fencing on a node
    /// CRITICAL: This simulates hardware power cut for split-brain prevention
    pub async fn trigger_stonith(&self, node_id: u64) -> Result<bool, SeverError> {
        if !self.chaos_mode_flag.load(Ordering::SeqCst) {
            return Err(SeverError::ChaosModeNotActive);
        }

        let node_state = match self.nodes.get(&node_id) {
            Some(state) => state.clone(),
            None => return Err(SeverError::NodeNotFound(node_id)),
        };

        // Mark as disconnected and not leader
        node_state.is_connected.store(false, Ordering::SeqCst);
        node_state.is_raft_leader.store(false, Ordering::SeqCst);

        // Send STONITH event
        let event = SeverEvent::StonithTriggered(node_id);
        match self.event_tx.try_send(event) {
            Ok(_) => Ok(true),
            Err(_) => Err(SeverError::EventChannelFull),
        }
    }

    /// Check if a node is currently connected
    pub fn is_node_connected(&self, node_id: u64) -> Option<bool> {
        self.nodes.get(&node_id).map(|n| n.is_connected.load(Ordering::Relaxed))
    }

    /// Check if a node is the Raft leader
    pub fn is_raft_leader(&self, node_id: u64) -> Option<bool> {
        self.nodes.get(&node_id).map(|n| n.is_raft_leader.load(Ordering::Relaxed))
    }

    /// Set a node as Raft leader (for testing)
    pub fn set_raft_leader(&self, node_id: u64, is_leader: bool) {
        if let Some(node_state) = self.nodes.get(&node_id) {
            node_state.is_raft_leader.store(is_leader, Ordering::SeqCst);
        }
    }

    /// Update heartbeat timestamp for a node
    pub fn update_heartbeat(&self, node_id: u64) {
        if let Some(node_state) = self.nodes.get(&node_id) {
            let now = Instant::now().as_millis() as u64;
            node_state.last_heartbeat.store(now, Ordering::Relaxed);
        }
    }

    /// Get time since last heartbeat in milliseconds
    pub fn time_since_heartbeat(&self, node_id: u64) -> Option<u64> {
        self.nodes.get(&node_id).map(|n| {
            let now = Instant::now().as_millis() as u64;
            let last = n.last_heartbeat.load(Ordering::Relaxed);
            now.saturating_sub(last)
        })
    }

    /// Get all connected nodes
    pub fn get_connected_nodes(&self) -> Vec<u64> {
        self.nodes
            .iter()
            .filter_map(|(&id, state)| {
                if state.is_connected.load(Ordering::Relaxed) {
                    Some(id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get sever statistics for a node
    pub fn get_sever_stats(&self, node_id: u64) -> Option<SeverStats> {
        self.nodes.get(&node_id).map(|n| SeverStats {
            sever_count: n.sever_count.load(Ordering::Relaxed),
            is_connected: n.is_connected.load(Ordering::Relaxed),
            is_raft_leader: n.is_raft_leader.load(Ordering::Relaxed),
            last_heartbeat_ms: n.last_heartbeat.load(Ordering::Relaxed),
        })
    }
}

/// Sever statistics
#[derive(Debug, Clone)]
pub struct SeverStats {
    pub sever_count: u64,
    pub is_connected: bool,
    pub is_raft_leader: bool,
    pub last_heartbeat_ms: u64,
}

/// Sever errors
#[derive(Debug, Clone, PartialEq)]
pub enum SeverError {
    ChaosModeNotActive,
    NodeNotFound(u64),
    EventChannelFull,
    InvalidConfiguration,
}

impl std::fmt::Display for SeverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SeverError::ChaosModeNotActive => write!(f, "Chaos mode not active"),
            SeverError::NodeNotFound(id) => write!(f, "Node {} not found", id),
            SeverError::EventChannelFull => write!(f, "Event channel full"),
            SeverError::InvalidConfiguration => write!(f, "Invalid configuration"),
        }
    }
}

impl std::error::Error for SeverError {}

/// Builder for sever configurations
pub struct SeverConfigBuilder {
    sever_duration_ms: u64,
    target_node_ids: Vec<u64>,
    trigger_raft_election: bool,
    simulate_network_partition: bool,
}

impl SeverConfigBuilder {
    pub fn new() -> Self {
        Self {
            sever_duration_ms: 5000,
            target_node_ids: vec![1, 2, 3],
            trigger_raft_election: true,
            simulate_network_partition: true,
        }
    }

    pub fn sever_duration(mut self, ms: u64) -> Self {
        self.sever_duration_ms = ms;
        self
    }

    pub fn target_node(mut self, node_id: u64) -> Self {
        self.target_node_ids.push(node_id);
        self
    }

    pub fn trigger_raft_election(mut self, trigger: bool) -> Self {
        self.trigger_raft_election = trigger;
        self
    }

    pub fn simulate_network_partition(mut self, simulate: bool) -> Self {
        self.simulate_network_partition = simulate;
        self
    }

    pub fn build(self) -> SeverConfig {
        SeverConfig {
            sever_duration_ms: self.sever_duration_ms,
            target_node_ids: self.target_node_ids,
            trigger_raft_election: self.trigger_raft_election,
            simulate_network_partition: self.simulate_network_partition,
        }
    }
}

impl Default for SeverConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_node_sever_without_chaos_mode() {
        let config = SeverConfigBuilder::new()
            .target_node(1)
            .build();

        let (severer, _rx) = NodeSeverer::new(config);

        // Should fail without chaos mode
        let result = severer.sever_node(1).await;
        assert!(matches!(result, Err(SeverError::ChaosModeNotActive)));
    }

    #[tokio::test]
    async fn test_node_sever_with_chaos_mode() {
        let config = SeverConfigBuilder::new()
            .sever_duration_ms(100)
            .target_node(1)
            .trigger_raft_election(true)
            .build();

        let (severer, mut rx) = NodeSeverer::new(config);
        severer.enable_chaos_mode();
        severer.set_raft_leader(1, true);

        // Sever should succeed
        let result = severer.sever_node(1).await;
        assert!(result.is_ok());

        // Node should be disconnected
        assert_eq!(severer.is_node_connected(1), Some(false));
        assert_eq!(severer.is_raft_leader(1), Some(false));

        // Wait for restoration event
        tokio::time::sleep(Duration::from_millis(150)).await;
        
        // Check for restoration event (may have been sent)
        while let Ok(event) = rx.try_recv() {
            if matches!(event, SeverEvent::ConnectionRestored(1)) {
                break;
            }
        }
    }

    #[tokio::test]
    async fn test_stonith_trigger() {
        let config = SeverConfigBuilder::new()
            .target_node(1)
            .build();

        let (severer, mut rx) = NodeSeverer::new(config);
        severer.enable_chaos_mode();

        let result = severer.trigger_stonith(1).await;
        assert!(result.is_ok());

        // Verify STONITH event was sent
        let event = rx.recv().await;
        assert!(matches!(event, Some(SeverEvent::StonithTriggered(1))));
    }
}
