//! Suspicion Timeout Management for SWIM Protocol
//! 
//! Implements adaptive suspicion timeout calculation based on network conditions,
//! node history, and failure probability estimation.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};

/// Unique node identifier
pub type NodeId = u64;

/// Configuration for suspicion timeout
#[derive(Debug, Clone)]
pub struct SuspicionConfig {
    /// Base timeout duration
    pub base_timeout: Duration,
    /// Minimum timeout (floor)
    pub min_timeout: Duration,
    /// Maximum timeout (ceiling)
    pub max_timeout: Duration,
    /// Number of samples for RTT estimation
    pub rtt_sample_count: usize,
    /// Confidence level for timeout calculation (0.0-1.0)
    pub confidence_level: f64,
    /// Adaptive scaling factor
    pub adaptive_factor: f64,
}

impl Default for SuspicionConfig {
    fn default() -> Self {
        Self {
            base_timeout: Duration::from_millis(500),
            min_timeout: Duration::from_millis(100),
            max_timeout: Duration::from_millis(5000),
            rtt_sample_count: 20,
            confidence_level: 0.95,
            adaptive_factor: 1.5,
        }
    }
}

/// Round-trip time sample
#[derive(Debug, Clone)]
struct RttSample {
    timestamp: Instant,
    rtt: Duration,
}

/// Node-specific suspicion state
#[derive(Debug, Clone)]
pub struct NodeSuspicionState {
    node_id: NodeId,
    rtt_samples: VecDeque<RttSample>,
    failure_count: u64,
    success_count: u64,
    last_suspicion_time: Option<Instant>,
    current_timeout: Duration,
    is_suspected: bool,
    suspicion_start: Option<Instant>,
}

impl NodeSuspicionState {
    fn new(node_id: NodeId, config: &SuspicionConfig) -> Self {
        Self {
            node_id,
            rtt_samples: VecDeque::with_capacity(config.rtt_sample_count),
            failure_count: 0,
            success_count: 0,
            last_suspicion_time: None,
            current_timeout: config.base_timeout,
            is_suspected: false,
            suspicion_start: None,
        }
    }

    /// Add an RTT sample
    fn add_rtt_sample(&mut self, rtt: Duration, max_samples: usize) {
        let sample = RttSample {
            timestamp: Instant::now(),
            rtt,
        };

        self.rtt_samples.push_back(sample);

        // Keep only recent samples
        while self.rtt_samples.len() > max_samples {
            self.rtt_samples.pop_front();
        }
    }

    /// Calculate mean RTT from samples
    fn mean_rtt(&self) -> Option<Duration> {
        if self.rtt_samples.is_empty() {
            return None;
        }

        let total: Duration = self.rtt_samples.iter().map(|s| s.rtt).sum();
        Some(total / self.rtt_samples.len() as u32)
    }

    /// Calculate standard deviation of RTT
    fn rtt_std_dev(&self) -> Option<Duration> {
        if self.rtt_samples.len() < 2 {
            return None;
        }

        let mean = self.mean_rtt()?;
        let mean_nanos = mean.as_nanos() as f64;

        let variance: f64 = self.rtt_samples.iter()
            .map(|s| {
                let diff = s.rtt.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>() / (self.rtt_samples.len() - 1) as f64;

        let std_dev_nanos = variance.sqrt();
        Some(Duration::from_nanos(std_dev_nanos as u64))
    }

    /// Record a successful ping/ack
    fn record_success(&mut self, rtt: Duration, config: &SuspicionConfig) {
        self.success_count += 1;
        self.add_rtt_sample(rtt, config.rtt_sample_count);
        
        if self.is_suspected {
            self.is_suspected = false;
            self.suspicion_start = None;
        }

        // Update timeout based on new RTT data
        self.current_timeout = self.calculate_timeout(config);
    }

    /// Record a failure (timeout or no response)
    fn record_failure(&mut self, config: &SuspicionConfig) {
        self.failure_count += 1;
        self.last_suspicion_time = Some(Instant::now());

        // Increase timeout after failure
        let increase_factor = 1.0 + (self.failure_count as f64 * 0.1);
        let new_timeout = self.current_timeout.mul_f64(increase_factor.min(2.0));
        self.current_timeout = new_timeout.min(config.max_timeout);
    }

    /// Calculate adaptive timeout
    fn calculate_timeout(&self, config: &SuspicionConfig) -> Duration {
        let base = if let Some(mean) = self.mean_rtt() {
            // Use mean RTT plus some multiple of std dev for confidence
            let std_dev = self.rtt_std_dev().unwrap_or(Duration::ZERO);
            let std_dev_nanos = std_dev.as_nanos() as f64;
            
            // Z-score for confidence level (approximate)
            let z_score = match config.confidence_level {
                x if x >= 0.99 => 2.576,
                x if x >= 0.95 => 1.96,
                x if x >= 0.90 => 1.645,
                _ => 1.0,
            };

            let timeout_nanos = mean.as_nanos() as f64 
                + (z_score * std_dev_nanos * config.adaptive_factor);
            
            Duration::from_nanos(timeout_nanos as u64)
        } else {
            config.base_timeout
        };

        // Clamp to min/max bounds
        base.clamp(config.min_timeout, config.max_timeout)
    }

    /// Mark node as suspected
    fn mark_suspected(&mut self) {
        self.is_suspected = true;
        self.suspicion_start = Some(Instant::now());
    }

    /// Check if suspicion timeout has elapsed
    fn is_suspicion_timeout_elapsed(&self) -> bool {
        if !self.is_suspected {
            return false;
        }

        if let Some(start) = self.suspicion_start {
            Instant::now().duration_since(start) >= self.current_timeout
        } else {
            false
        }
    }

    /// Get failure rate
    fn failure_rate(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            return 0.0;
        }
        self.failure_count as f64 / total as f64
    }
}

/// Suspicion Timeout Manager
pub struct SuspicionTimeoutManager {
    config: SuspicionConfig,
    nodes: RwLock<HashMap<NodeId, NodeSuspicionState>>,
    global_timeout: RwLock<Duration>,
}

impl SuspicionTimeoutManager {
    pub fn new(config: SuspicionConfig) -> Self {
        Self {
            config,
            nodes: RwLock::new(HashMap::new()),
            global_timeout: RwLock::new(config.base_timeout),
        }
    }

    /// Register a new node
    pub async fn register_node(&self, node_id: NodeId) {
        let mut nodes = self.nodes.write().await;
        if !nodes.contains_key(&node_id) {
            nodes.insert(node_id, NodeSuspicionState::new(node_id, &self.config));
        }
    }

    /// Record a successful ping response
    pub async fn record_success(&self, node_id: NodeId, rtt: Duration) {
        let mut nodes = self.nodes.write().await;
        if let Some(state) = nodes.get_mut(&node_id) {
            state.record_success(rtt, &self.config);
        }
    }

    /// Record a failure (no response)
    pub async fn record_failure(&self, node_id: NodeId) {
        let mut nodes = self.nodes.write().await;
        if let Some(state) = nodes.get_mut(&node_id) {
            state.record_failure(&self.config);
        }
    }

    /// Mark a node as suspected
    pub async fn mark_suspected(&self, node_id: NodeId) -> Result<(), SuspicionError> {
        let mut nodes = self.nodes.write().await;
        let state = nodes.get_mut(&node_id)
            .ok_or_else(|| SuspicionError::NodeNotFound(node_id))?;
        
        state.mark_suspected();
        Ok(())
    }

    /// Check if a node's suspicion timeout has elapsed
    pub async fn is_suspicion_timeout_elapsed(&self, node_id: NodeId) -> Result<bool, SuspicionError> {
        let nodes = self.nodes.read().await;
        let state = nodes.get(&node_id)
            .ok_or_else(|| SuspicionError::NodeNotFound(node_id))?;
        
        Ok(state.is_suspicion_timeout_elapsed())
    }

    /// Get current timeout for a node
    pub async fn get_timeout(&self, node_id: NodeId) -> Result<Duration, SuspicionError> {
        let nodes = self.nodes.read().await;
        let state = nodes.get(&node_id)
            .ok_or_else(|| SuspicionError::NodeNotFound(node_id))?;
        
        Ok(state.current_timeout)
    }

    /// Get all nodes that should be confirmed dead
    pub async fn get_nodes_to_confirm_dead(&self) -> Vec<NodeId> {
        let nodes = self.nodes.read().await;
        nodes.iter()
            .filter(|(_, state)| state.is_suspicion_timeout_elapsed())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get statistics for a node
    pub async fn get_node_stats(&self, node_id: NodeId) -> Result<SuspicionStats, SuspicionError> {
        let nodes = self.nodes.read().await;
        let state = nodes.get(&node_id)
            .ok_or_else(|| SuspicionError::NodeNotFound(node_id))?;

        Ok(SuspicionStats {
            node_id,
            current_timeout: state.current_timeout,
            mean_rtt: state.mean_rtt(),
            std_dev_rtt: state.rtt_std_dev(),
            failure_rate: state.failure_rate(),
            is_suspected: state.is_suspected,
            sample_count: state.rtt_samples.len(),
        })
    }

    /// Remove a node from tracking
    pub async fn remove_node(&self, node_id: NodeId) {
        let mut nodes = self.nodes.write().await;
        nodes.remove(&node_id);
    }

    /// Reset statistics for a node
    pub async fn reset_node_stats(&self, node_id: NodeId) -> Result<(), SuspicionError> {
        let mut nodes = self.nodes.write().await;
        let state = nodes.get_mut(&node_id)
            .ok_or_else(|| SuspicionError::NodeNotFound(node_id))?;

        *state = NodeSuspicionState::new(node_id, &self.config);
        Ok(())
    }

    /// Get global average timeout
    pub async fn get_global_timeout(&self) -> Duration {
        let nodes = self.nodes.read().await;
        if nodes.is_empty() {
            return *self.global_timeout.read().await;
        }

        let total: Duration = nodes.values().map(|s| s.current_timeout).sum();
        total / nodes.len() as u32
    }
}

/// Statistics about a node's suspicion state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuspicionStats {
    pub node_id: NodeId,
    pub current_timeout: Duration,
    pub mean_rtt: Option<Duration>,
    pub std_dev_rtt: Option<Duration>,
    pub failure_rate: f64,
    pub is_suspected: bool,
    pub sample_count: usize,
}

/// Suspicion manager errors
#[derive(Debug, thiserror::Error)]
pub enum SuspicionError {
    #[error("Node {0} not found")]
    NodeNotFound(NodeId),
    #[error("Invalid configuration")]
    InvalidConfig(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rtt_tracking() {
        let config = SuspicionConfig::default();
        let manager = SuspicionTimeoutManager::new(config);

        manager.register_node(1).await;

        // Record several RTT samples
        for i in 0..10 {
            let rtt = Duration::from_millis(50 + i);
            manager.record_success(1, rtt).await;
        }

        let stats = manager.get_node_stats(1).await.unwrap();
        assert!(stats.mean_rtt.is_some());
        assert_eq!(stats.sample_count, 10);
        assert!(stats.failure_rate == 0.0);
    }

    #[tokio::test]
    async fn test_failure_handling() {
        let config = SuspicionConfig::default();
        let manager = SuspicionTimeoutManager::new(config);

        manager.register_node(1).await;

        // Record some successes
        manager.record_success(1, Duration::from_millis(50)).await;
        manager.record_success(1, Duration::from_millis(55)).await;

        let initial_timeout = manager.get_timeout(1).await.unwrap();

        // Record failures
        manager.record_failure(1).await;
        manager.record_failure(1).await;

        let post_failure_timeout = manager.get_timeout(1).await.unwrap();

        // Timeout should have increased
        assert!(post_failure_timeout > initial_timeout);
    }

    #[tokio::test]
    async fn test_suspicion_lifecycle() {
        let mut config = SuspicionConfig::default();
        config.base_timeout = Duration::from_millis(100);
        config.min_timeout = Duration::from_millis(50);

        let manager = SuspicionTimeoutManager::new(config);
        manager.register_node(1).await;

        // Mark as suspected
        manager.mark_suspected(1).await.unwrap();

        // Should not be elapsed immediately
        let elapsed = manager.is_suspicion_timeout_elapsed(1).await.unwrap();
        assert!(!elapsed);

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should now be elapsed
        let elapsed = manager.is_suspicion_timeout_elapsed(1).await.unwrap();
        assert!(elapsed);
    }

    #[tokio::test]
    async fn test_adaptive_timeout_calculation() {
        let config = SuspicionConfig::default();
        let manager = SuspicionTimeoutManager::new(config);

        manager.register_node(1).await;

        // Record consistent RTT samples
        for _ in 0..20 {
            manager.record_success(1, Duration::from_millis(100)).await;
        }

        let stats = manager.get_node_stats(1).await.unwrap();
        
        // With low variance, timeout should be close to mean RTT
        assert!(stats.mean_rtt.is_some());
        assert!(stats.std_dev_rtt.is_some());
    }
}
