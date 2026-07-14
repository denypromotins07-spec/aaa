//! Quorum Witness for NEXUS-OMEGA Swarm
//! 
//! Provides external quorum confirmation for leader election and fencing decisions.

use crate::{ConsensusError, ConsensusResult, NodeId, Term};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Represents the state of a quorum witness
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WitnessState {
    Healthy,
    Degraded,
    Unreachable,
}

/// Manages communication with external quorum witnesses
pub struct QuorumWitness {
    witness_urls: Vec<String>,
    current_term: AtomicU64,
    known_leader: AtomicU64,
    is_healthy: AtomicBool,
    last_successful_contact: AtomicU64, // Timestamp in ms
}

unsafe impl Send for QuorumWitness {}
unsafe impl Sync for QuorumWitness {}

impl QuorumWitness {
    pub fn new(witness_urls: Vec<String>) -> Self {
        Self {
            witness_urls,
            current_term: AtomicU64::new(0),
            known_leader: AtomicU64::new(0),
            is_healthy: AtomicBool::new(false),
            last_successful_contact: AtomicU64::new(0),
        }
    }

    /// Check if we have quorum for leadership
    pub async fn check_quorum(&self, candidate_id: NodeId, term: Term) -> ConsensusResult<bool> {
        let timeout = Duration::from_millis(500);
        let mut success_count = 0;
        
        for url in &self.witness_urls {
            match self._contact_witness(url, candidate_id, term, timeout).await {
                Ok(true) => success_count += 1,
                Ok(false) | Err(_) => {}
            }
        }
        
        // Require majority
        let required = (self.witness_urls.len() / 2) + 1;
        let has_quorum = success_count >= required;
        
        if has_quorum {
            self.is_healthy.store(true, Ordering::Release);
            self.last_successful_contact.store(
                Instant::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64,
                Ordering::Release,
            );
        }
        
        Ok(has_quorum)
    }

    /// Contact a single witness
    async fn _contact_witness(
        &self,
        url: &str,
        candidate_id: NodeId,
        term: Term,
        timeout: Duration,
    ) -> Result<bool, ()> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|_| ())?;

        let request = serde_json::json!({
            "action": "check_quorum",
            "candidate_id": candidate_id,
            "term": term,
        });

        match tokio::time::timeout(timeout, async {
            client
                .post(format!("{}/api/v1/quorum", url))
                .json(&request)
                .send()
                .await
        }).await {
            Ok(Ok(response)) => {
                if response.status().is_success() {
                    let body: serde_json::Value = response.json().await.map_err(|_| ())?;
                    Ok(body.get("granted").and_then(|v| v.as_bool()).unwrap_or(false))
                } else {
                    Ok(false)
                }
            }
            _ => Err(()),
        }
    }

    /// Report a new leader to all witnesses
    pub async fn report_leader(&self, leader_id: NodeId, term: Term) -> ConsensusResult<()> {
        self.current_term.store(term, Ordering::Release);
        self.known_leader.store(leader_id, Ordering::Release);
        
        let timeout = Duration::from_millis(200);
        
        for url in &self.witness_urls {
            let _ = self._report_to_witness(url, leader_id, term, timeout).await;
        }
        
        Ok(())
    }

    async fn _report_to_witness(
        &self,
        url: &str,
        leader_id: NodeId,
        term: Term,
        timeout: Duration,
    ) -> Result<(), ()> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|_| ())?;

        let request = serde_json::json!({
            "action": "leader_report",
            "leader_id": leader_id,
            "term": term,
        });

        tokio::time::timeout(timeout, async {
            client
                .post(format!("{}/api/v1/leader", url))
                .json(&request)
                .send()
                .await
        })
        .await
        .map_err(|_| ())?
        .map_err(|_| ())?;

        Ok(())
    }

    /// Get current witness health status
    pub fn get_health(&self) -> WitnessState {
        let now = Instant::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        let last_contact = self.last_successful_contact.load(Ordering::Acquire);
        let elapsed_ms = now.saturating_sub(last_contact);
        
        if !self.is_healthy.load(Ordering::Acquire) {
            WitnessState::Unreachable
        } else if elapsed_ms > 5000 {
            WitnessState::Degraded
        } else {
            WitnessState::Healthy
        }
    }

    /// Get the known leader
    pub fn get_known_leader(&self) -> Option<NodeId> {
        let leader = self.known_leader.load(Ordering::Acquire);
        if leader == 0 {
            None
        } else {
            Some(leader)
        }
    }

    /// Get current term
    pub fn get_current_term(&self) -> Term {
        self.current_term.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_witness_creation() {
        let urls = vec![
            "http://witness1:8080".to_string(),
            "http://witness2:8080".to_string(),
        ];
        
        let witness = QuorumWitness::new(urls.clone());
        
        assert_eq!(witness.get_health(), WitnessState::Unreachable);
        assert_eq!(witness.get_known_leader(), None);
        assert_eq!(witness.get_current_term(), 0);
    }

    #[test]
    fn test_witness_state_transitions() {
        let urls = vec!["http://witness1:8080".to_string()];
        let witness = QuorumWitness::new(urls);
        
        // Initially unreachable
        assert_eq!(witness.get_health(), WitnessState::Unreachable);
        
        // Simulate healthy state
        witness.is_healthy.store(true, Ordering::Release);
        witness.last_successful_contact.store(
            Instant::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64,
            Ordering::Release,
        );
        
        assert_eq!(witness.get_health(), WitnessState::Healthy);
    }
}
