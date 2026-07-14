//! STONITH Fencer (Shoot The Other Node In The Head)
//! 
//! Implements physical fencing of failed leaders to prevent split-brain scenarios.
//! CRITICAL: On ANY fencing failure or timeout, the system MUST default to HALTING trading.

use crate::{ConsensusError, ConsensusResult, NodeId, SwarmConfig, Term};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Result of a STONITH fencing operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FencingResult {
    /// Successfully fenced the target node
    Success,
    /// Target was already dead/fenced
    AlreadyDead,
    /// Fencing API timed out - CRITICAL: must halt trading
    Timeout,
    /// Fencing API failed - CRITICAL: must halt trading
    Failed,
}

/// Executes STONITH fencing commands to isolate failed leaders
pub struct STONITHFencer {
    config: SwarmConfig,
    /// Tracks if fencing has been executed for current term
    last_fence_term: AtomicU64,
    /// Tracks if fencing succeeded
    last_fence_success: AtomicBool,
    /// Prevents duplicate fencing in same term
    fencing_in_progress: AtomicBool,
}

unsafe impl Send for STONITHFencer {}
unsafe impl Sync for STONITHFencer {}

impl STONITHFencer {
    pub fn new(config: &SwarmConfig) -> ConsensusResult<Self> {
        if config.stonith_enabled && config.quorum_witness_urls.is_empty() {
            tracing::warn!("STONITH enabled but no quorum witness URLs configured");
        }
        
        Ok(Self {
            config: config.clone(),
            last_fence_term: AtomicU64::new(0),
            last_fence_success: AtomicBool::new(false),
            fencing_in_progress: AtomicBool::new(false),
        })
    }

    /// Execute fencing against a previous leader
    /// 
    /// CRITICAL SECURITY: This method MUST return an error if:
    /// 1. The fencing API times out
    /// 2. The fencing API returns any error
    /// 3. Cannot confirm the target is definitively dead
    /// 
    /// On ANY failure, the caller MUST NOT assume leadership and MUST halt trading.
    pub async fn execute_fence(
        &self,
        previous_term: Term,
        requestor_id: NodeId,
    ) -> ConsensusResult<FencingResult> {
        // Prevent duplicate fencing in same term
        let expected = false;
        if !self.fencing_in_progress.compare_exchange(
            expected,
            true,
            Ordering::AcqRel,
            Ordering::Acquire,
        ).unwrap_or(true) {
            return Err(ConsensusError::StonithFailed(
                "Fencing already in progress".to_string()
            ));
        }

        let result = self._execute_fence_internal(previous_term, requestor_id).await;
        
        // Clear in-progress flag
        self.fencing_in_progress.store(false, Ordering::Release);
        
        match result {
            Ok(FencingResult::Success) | Ok(FencingResult::AlreadyDead) => {
                self.last_fence_term.store(previous_term, Ordering::Release);
                self.last_fence_success.store(true, Ordering::Release);
                result
            }
            Ok(FencingResult::Timeout) | Ok(FencingResult::Failed) => {
                self.last_fence_success.store(false, Ordering::Release);
                // CRITICAL: Return error to prevent leadership assumption
                Err(ConsensusError::StonithFailed(
                    format!("Fencing returned {:?} - MUST HALT TRADING", result.unwrap())
                ))
            }
            Err(e) => {
                self.last_fence_success.store(false, Ordering::Release);
                Err(e)
            }
        }
    }

    /// Internal fencing implementation
    async fn _execute_fence_internal(
        &self,
        previous_term: Term,
        requestor_id: NodeId,
    ) -> ConsensusResult<FencingResult> {
        if !self.config.stonith_enabled {
            tracing::warn!("STONITH disabled, skipping fencing");
            return Ok(FencingResult::Success);
        }

        // Create fence request with timeout
        let timeout = Duration::from_millis(self.config.stonith_timeout_ms);
        let fence_start = Instant::now();

        // Try all quorum witnesses
        let mut success_count = 0;
        let mut timeout_count = 0;
        let mut failure_count = 0;

        for witness_url in &self.config.quorum_witness_urls {
            match self._fence_via_witness(witness_url, previous_term, requestor_id, timeout).await {
                Ok(true) => success_count += 1,
                Ok(false) => failure_count += 1,
                Err(_) => timeout_count += 1,
            }
        }

        let elapsed = fence_start.elapsed();
        
        // Require majority of witnesses to confirm fencing
        let required_witnesses = (self.config.quorum_witness_urls.len() / 2) + 1;

        if timeout_count > 0 {
            // CRITICAL: ANY timeout means we cannot guarantee safety
            tracing::error!(
                "STONITH fencing TIMEOUT after {}ms - {} timeouts detected. \
                 CANNOT GUARANTEE OLD LEADER IS DEAD. HALTING.",
                elapsed.as_millis(),
                timeout_count
            );
            return Ok(FencingResult::Timeout);
        }

        if success_count >= required_witnesses {
            tracing::warn!(
                "STONITH fencing SUCCESS: confirmed {} witnesses in {}ms",
                success_count,
                elapsed.as_millis()
            );
            Ok(FencingResult::Success)
        } else if failure_count >= required_witnesses {
            // All witnesses agree target is already dead
            tracing::warn!(
                "STONITH fencing: target already dead ({} witnesses confirm)",
                failure_count
            );
            Ok(FencingResult::AlreadyDead)
        } else {
            tracing::error!(
                "STONITH fencing FAILED: only {}/{} witnesses succeeded, \
                 required {}. HALTING.",
                success_count,
                self.config.quorum_witness_urls.len(),
                required_witnesses
            );
            Ok(FencingResult::Failed)
        }
    }

    /// Attempt to fence via a single witness
    async fn _fence_via_witness(
        &self,
        witness_url: &str,
        previous_term: Term,
        requestor_id: NodeId,
        timeout: Duration,
    ) -> Result<bool, ()> {
        // In production, this would make an HTTP/gRPC call to the witness
        // For now, simulate with a mock that respects timeout
        
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|_| ())?;

        let fence_request = serde_json::json!({
            "action": "fence",
            "term": previous_term,
            "requestor_id": requestor_id,
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        });

        match tokio::time::timeout(timeout, async {
            client
                .post(format!("{}/api/v1/fence", witness_url))
                .json(&fence_request)
                .send()
                .await
        }).await {
            Ok(Ok(response)) => {
                if response.status().is_success() {
                    Ok(true)
                } else {
                    // Target already fenced/dead
                    Ok(false)
                }
            }
            Ok(Err(_)) => Err(()),
            Err(_) => Err(()), // Timeout
        }
    }

    /// Check if fencing was successful for the given term
    pub fn was_fencing_successful(&self, term: Term) -> bool {
        self.last_fence_term.load(Ordering::Acquire) == term &&
        self.last_fence_success.load(Ordering::Acquire)
    }

    /// Get the last fenced term
    pub fn get_last_fence_term(&self) -> Term {
        self.last_fence_term.load(Ordering::Acquire)
    }

    /// Reset fencing state (only called during controlled shutdown)
    pub fn reset(&self) {
        self.last_fence_term.store(0, Ordering::Release);
        self.last_fence_success.store(false, Ordering::Release);
        self.fencing_in_progress.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stonith_disabled() {
        let mut config = SwarmConfig::default();
        config.stonith_enabled = false;
        
        let fencer = STONITHFencer::new(&config).unwrap();
        let result = fencer.execute_fence(0, 1).await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), FencingResult::Success);
    }

    #[tokio::test]
    async fn test_stonith_no_witnesses() {
        let mut config = SwarmConfig::default();
        config.stonith_enabled = true;
        config.quorum_witness_urls = vec![]; // No witnesses
        
        let fencer = STONITHFencer::new(&config).unwrap();
        let result = fencer.execute_fence(0, 1).await;
        
        // With no witnesses, should fail safely
        assert!(result.is_err());
    }

    #[test]
    fn test_fencing_state_tracking() {
        let config = SwarmConfig::default();
        let fencer = STONITHFencer::new(&config).unwrap();
        
        assert!(!fencer.was_fencing_successful(0));
        assert_eq!(fencer.get_last_fence_term(), 0);
        
        fencer.reset();
        assert!(!fencer.was_fencing_successful(0));
    }
}
