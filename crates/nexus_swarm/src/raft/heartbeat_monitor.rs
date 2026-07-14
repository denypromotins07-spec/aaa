//! Heartbeat Monitor for NEXUS-OMEGA Swarm
//! 
//! Monitors leader heartbeats and triggers elections when the leader fails.

use crate::{NodeId, NodeState, RaftMessage, SwarmConfig, Term};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Monitors heartbeats from the leader and detects failures
pub struct HeartbeatMonitor {
    node_id: NodeId,
    last_heartbeat_from_leader: HashMap<NodeId, Instant>,
    current_leader: Option<NodeId>,
    current_term: Term,
    election_timeout: Duration,
    heartbeat_interval: Duration,
    missed_heartbeat_count: u32,
    max_missed_heartbeats: u32,
}

impl HeartbeatMonitor {
    pub fn new(config: &SwarmConfig) -> Self {
        Self {
            node_id: config.node_id,
            last_heartbeat_from_leader: HashMap::new(),
            current_leader: None,
            current_term: 0,
            election_timeout: Duration::from_millis(config.election_timeout_ms),
            heartbeat_interval: Duration::from_millis(config.heartbeat_interval_ms),
            missed_heartbeat_count: 0,
            max_missed_heartbeats: 3,
        }
    }

    /// Record a heartbeat received from the leader
    pub fn record_heartbeat(&mut self, leader_id: NodeId, term: Term) {
        if term >= self.current_term {
            self.current_term = term;
            self.current_leader = Some(leader_id);
            self.last_heartbeat_from_leader.insert(leader_id, Instant::now());
            self.missed_heartbeat_count = 0;
        }
    }

    /// Check if the leader has timed out and election should be triggered
    pub fn check_leader_timeout(&mut self) -> bool {
        if let Some(leader_id) = self.current_leader {
            if let Some(last_heartbeat) = self.last_heartbeat_from_leader.get(&leader_id) {
                let elapsed = last_heartbeat.elapsed();
                
                if elapsed > self.election_timeout {
                    self.missed_heartbeat_count += 1;
                    
                    if self.missed_heartbeat_count >= self.max_missed_heartbeats {
                        // Leader has timed out, trigger election
                        self.current_leader = None;
                        return true;
                    }
                } else {
                    self.missed_heartbeat_count = 0;
                }
            }
        }
        false
    }

    /// Get the current known leader
    pub fn current_leader(&self) -> Option<NodeId> {
        self.current_leader
    }

    /// Get current term
    pub fn current_term(&self) -> Term {
        self.current_term
    }

    /// Reset monitor state (used after becoming leader or starting new term)
    pub fn reset(&mut self, term: Term) {
        self.current_term = term;
        self.current_leader = None;
        self.missed_heartbeat_count = 0;
        self.last_heartbeat_from_leader.clear();
    }

    /// Create a heartbeat message to send (if this node is leader)
    pub fn create_heartbeat(&self, term: Term) -> Option<RaftMessage> {
        if self.current_leader == Some(self.node_id) {
            Some(RaftMessage::Heartbeat {
                term,
                leader_id: self.node_id,
            })
        } else {
            None
        }
    }

    /// Process a heartbeat response from a follower
    pub fn process_heartbeat_response(&mut self, follower_id: NodeId, term: Term) {
        if term >= self.current_term && self.current_leader == Some(self.node_id) {
            self.last_heartbeat_from_leader.insert(follower_id, Instant::now());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_heartbeat_recording() {
        let config = SwarmConfig::default();
        let mut monitor = HeartbeatMonitor::new(&config);
        
        monitor.record_heartbeat(1, 1);
        assert_eq!(monitor.current_leader(), Some(1));
        assert_eq!(monitor.current_term(), 1);
    }

    #[test]
    fn test_leader_timeout_detection() {
        let mut config = SwarmConfig::default();
        config.election_timeout_ms = 50; // Very short for testing
        
        let mut monitor = HeartbeatMonitor::new(&config);
        monitor.record_heartbeat(1, 1);
        
        // Should not timeout immediately
        assert!(!monitor.check_leader_timeout());
        
        // Wait for timeout
        thread::sleep(Duration::from_millis(60));
        
        // Multiple checks to accumulate missed heartbeats
        for _ in 0..5 {
            if monitor.check_leader_timeout() {
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        
        // Leader should be cleared after timeout
        assert_eq!(monitor.current_leader(), None);
    }
}
