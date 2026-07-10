//! Canary Router for Gradual Model Rollout
//!
//! Implements canary deployment with configurable traffic splitting
//! and automatic rollback on performance degradation.

use crate::MLOpsError;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

/// Canary deployment configuration
#[derive(Debug, Clone)]
pub struct CanaryConfig {
    /// Initial canary traffic percentage (0-100)
    pub initial_percentage: u8,
    /// Maximum canary traffic percentage before full rollout
    pub max_percentage: u8,
    /// Traffic increment step per promotion stage
    pub increment_step: u8,
    /// Minimum samples before evaluating canary performance
    pub min_samples_eval: u64,
    /// Performance degradation threshold (percentage)
    pub degradation_threshold: f64,
    /// Auto-rollback enabled
    pub auto_rollback: bool,
}

impl Default for CanaryConfig {
    fn default() -> Self {
        Self {
            initial_percentage: 5,
            max_percentage: 80,
            increment_step: 10,
            min_samples_eval: 1000,
            degradation_threshold: 0.05, // 5% degradation triggers rollback
            auto_rollback: true,
        }
    }
}

/// Canary router state
#[derive(Debug, Clone, PartialEq)]
pub enum CanaryState {
    /// Canary not started
    Idle,
    /// Canary running with traffic split
    Running,
    /// Canary promoted to next stage
    Promoted,
    /// Canary rolled back
    RolledBack,
    /// Full rollout complete
    FullyRolledOut,
}

/// Statistics for canary vs baseline comparison
#[derive(Debug, Clone, Default)]
pub struct CanaryStats {
    pub canary_requests: u64,
    pub baseline_requests: u64,
    pub canary_errors: u64,
    pub baseline_errors: u64,
    pub canary_latency_sum_ms: u64,
    pub baseline_latency_sum_ms: u64,
    pub canary_metric_sum: f64,
    pub baseline_metric_sum: f64,
}

/// Canary router for gradual model rollout
pub struct CanaryRouter {
    config: CanaryConfig,
    /// Current traffic percentage to canary
    traffic_percentage: AtomicU8,
    /// Canary state
    state: std::sync::Mutex<CanaryState>,
    /// Statistics
    stats: std::sync::RwLock<CanaryStats>,
    /// Request counter for deterministic routing
    request_counter: AtomicU64,
    /// Rollback flag
    rollback_requested: AtomicBool,
    /// Stage counter
    current_stage: AtomicU64,
}

use std::sync::atomic::AtomicU8;

impl CanaryRouter {
    /// Create new canary router
    pub fn new(config: CanaryConfig) -> Self {
        Self {
            config,
            traffic_percentage: AtomicU8::new(0),
            state: std::sync::Mutex::new(CanaryState::Idle),
            stats: std::sync::RwLock::new(CanaryStats::default()),
            request_counter: AtomicU64::new(0),
            rollback_requested: AtomicBool::new(false),
            current_stage: AtomicU64::new(0),
        }
    }

    /// Start canary deployment
    pub fn start_canary(&self) -> Result<(), MLOpsError> {
        let mut state = self.state.lock().map_err(|_| {
            MLOpsError::ModelSwapFailed("Canary state lock poisoned".to_string())
        })?;

        if *state != CanaryState::Idle {
            return Err(MLOpsError::ModelSwapFailed(
                "Canary already started".to_string()
            ));
        }

        self.traffic_percentage.store(self.config.initial_percentage, Ordering::Relaxed);
        *state = CanaryState::Running;
        self.current_stage.store(1, Ordering::Relaxed);

        Ok(())
    }

    /// Route request to canary or baseline
    /// Returns true if should route to canary
    pub fn should_route_to_canary(&self) -> bool {
        if self.rollback_requested.load(Ordering::Relaxed) {
            return false;
        }

        let percentage = self.traffic_percentage.load(Ordering::Relaxed);
        
        if percentage == 0 {
            return false;
        }

        // Deterministic routing based on request counter
        let counter = self.request_counter.fetch_add(1, Ordering::Relaxed);
        
        (counter % 100) < percentage as u64
    }

    /// Record canary observation
    pub fn record_canary_observation(
        &self,
        is_error: bool,
        latency_ms: u64,
        metric_value: f64,
    ) -> Result<(), MLOpsError> {
        let mut stats = self.stats.write().map_err(|_| {
            MLOpsError::ModelSwapFailed("Stats lock poisoned".to_string())
        })?;

        stats.canary_requests += 1;
        if is_error {
            stats.canary_errors += 1;
        }
        stats.canary_latency_sum_ms += latency_ms;
        stats.canary_metric_sum += metric_value;

        Ok(())
    }

    /// Record baseline observation
    pub fn record_baseline_observation(
        &self,
        is_error: bool,
        latency_ms: u64,
        metric_value: f64,
    ) -> Result<(), MLOpsError> {
        let mut stats = self.stats.write().map_err(|_| {
            MLOpsError::ModelSwapFailed("Stats lock poisoned".to_string())
        })?;

        stats.baseline_requests += 1;
        if is_error {
            stats.baseline_errors += 1;
        }
        stats.baseline_latency_sum_ms += latency_ms;
        stats.baseline_metric_sum += metric_value;

        Ok(())
    }

    /// Evaluate canary performance and decide next action
    pub fn evaluate_and_promote(&self) -> Result<CanaryAction, MLOpsError> {
        let stats = self.stats.read().map_err(|_| {
            MLOpsError::ModelSwapFailed("Stats lock poisoned".to_string())
        })?;

        // Check minimum samples
        if stats.canary_requests < self.config.min_samples_eval 
            || stats.baseline_requests < self.config.min_samples_eval
        {
            return Ok(CanaryAction::Continue);
        }

        // Calculate error rates
        let canary_error_rate = stats.canary_errors as f64 / stats.canary_requests as f64;
        let baseline_error_rate = stats.baseline_errors as f64 / stats.baseline_requests as f64;

        // Calculate average metrics
        let canary_avg_metric = stats.canary_metric_sum / stats.canary_requests as f64;
        let baseline_avg_metric = stats.baseline_metric_sum / stats.baseline_requests as f64;

        // Check for degradation
        let error_degradation = canary_error_rate - baseline_error_rate;
        let metric_degradation = if baseline_avg_metric.abs() > 1e-10 {
            (baseline_avg_metric - canary_avg_metric) / baseline_avg_metric.abs()
        } else {
            0.0
        };

        let max_degradation = error_degradation.max(metric_degradation.abs());

        if max_degradation > self.config.degradation_threshold {
            if self.config.auto_rollback {
                return Ok(CanaryAction::Rollback);
            } else {
                return Ok(CanaryAction::HoldForReview);
            }
        }

        // Check if we can promote
        let current_pct = self.traffic_percentage.load(Ordering::Relaxed);
        
        if current_pct >= self.config.max_percentage {
            return Ok(CanaryAction::FullRollout);
        }

        // Promote to next stage
        let new_pct = (current_pct + self.config.increment_step).min(self.config.max_percentage);
        
        Ok(CanaryAction::Promote(new_pct))
    }

    /// Execute promotion action
    pub fn execute_action(&self, action: CanaryAction) -> Result<(), MLOpsError> {
        match action {
            CanaryAction::Promote(new_percentage) => {
                self.traffic_percentage.store(new_percentage, Ordering::Relaxed);
                
                let mut state = self.state.lock().map_err(|_| {
                    MLOpsError::ModelSwapFailed("Canary state lock poisoned".to_string())
                })?;
                
                *state = CanaryState::Promoted;
                self.current_stage.fetch_add(1, Ordering::Relaxed);
                
                // Reset stats for next evaluation period
                let mut stats = self.stats.write().map_err(|_| {
                    MLOpsError::ModelSwapFailed("Stats lock poisoned".to_string())
                })?;
                *stats = CanaryStats::default();
            },
            CanaryAction::Rollback => {
                self.traffic_percentage.store(0, Ordering::Relaxed);
                self.rollback_requested.store(true, Ordering::Relaxed);
                
                let mut state = self.state.lock().map_err(|_| {
                    MLOpsError::ModelSwapFailed("Canary state lock poisoned".to_string())
                })?;
                
                *state = CanaryState::RolledBack;
            },
            CanaryAction::FullRollout => {
                self.traffic_percentage.store(100, Ordering::Relaxed);
                
                let mut state = self.state.lock().map_err(|_| {
                    MLOpsError::ModelSwapFailed("Canary state lock poisoned".to_string())
                })?;
                
                *state = CanaryState::FullyRolledOut;
            },
            CanaryAction::Continue | CanaryAction::HoldForReview => {
                // No action needed
            },
        }

        Ok(())
    }

    /// Get current canary state
    pub fn get_state(&self) -> CanaryState {
        self.state.lock().map(|s| s.clone()).unwrap_or(CanaryState::Idle)
    }

    /// Get current traffic percentage
    pub fn get_traffic_percentage(&self) -> u8 {
        self.traffic_percentage.load(Ordering::Relaxed)
    }

    /// Get current stage
    pub fn get_current_stage(&self) -> u64 {
        self.current_stage.load(Ordering::Relaxed)
    }

    /// Force rollback
    pub fn force_rollback(&self) -> Result<(), MLOpsError> {
        self.traffic_percentage.store(0, Ordering::Relaxed);
        self.rollback_requested.store(true, Ordering::Relaxed);
        
        let mut state = self.state.lock().map_err(|_| {
            MLOpsError::ModelSwapFailed("Canary state lock poisoned".to_string())
        })?;
        
        *state = CanaryState::RolledBack;
        
        Ok(())
    }

    /// Reset canary for new deployment
    pub fn reset(&self) {
        self.traffic_percentage.store(0, Ordering::Relaxed);
        self.rollback_requested.store(false, Ordering::Relaxed);
        self.request_counter.store(0, Ordering::Relaxed);
        self.current_stage.store(0, Ordering::Relaxed);
        
        let mut state = self.state.lock().unwrap();
        *state = CanaryState::Idle;
        
        let mut stats = self.stats.write().unwrap();
        *stats = CanaryStats::default();
    }
}

/// Action to take after canary evaluation
#[derive(Debug, Clone, PartialEq)]
pub enum CanaryAction {
    /// Continue monitoring
    Continue,
    /// Promote to next stage with specified percentage
    Promote(u8),
    /// Rollback to baseline
    Rollback,
    /// Hold for manual review
    HoldForReview,
    /// Complete full rollout
    FullRollout,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canary_routing() {
        let router = CanaryRouter::new(CanaryConfig {
            initial_percentage: 20,
            ..Default::default()
        });

        router.start_canary().unwrap();

        let mut canary_count = 0;
        for _ in 0..1000 {
            if router.should_route_to_canary() {
                canary_count += 1;
            }
        }

        // Should be approximately 20%
        assert!(canary_count > 150 && canary_count < 250);
    }

    #[test]
    fn test_canary_promotion_flow() {
        let router = CanaryRouter::new(CanaryConfig {
            initial_percentage: 10,
            max_percentage: 50,
            increment_step: 10,
            min_samples_eval: 100,
            degradation_threshold: 0.1,
            ..Default::default()
        });

        router.start_canary().unwrap();
        assert_eq!(router.get_traffic_percentage(), 10);

        // Simulate good performance
        for _ in 0..150 {
            router.record_canary_observation(false, 10, 0.95).unwrap();
            router.record_baseline_observation(false, 10, 0.90).unwrap();
        }

        let action = router.evaluate_and_promote().unwrap();
        
        match action {
            CanaryAction::Promote(pct) => {
                assert_eq!(pct, 20);
                router.execute_action(action).unwrap();
                assert_eq!(router.get_traffic_percentage(), 20);
            },
            _ => panic!("Expected promotion"),
        }
    }

    #[test]
    fn test_canary_rollback_on_degradation() {
        let router = CanaryRouter::new(CanaryConfig {
            initial_percentage: 10,
            min_samples_eval: 100,
            degradation_threshold: 0.05,
            auto_rollback: true,
            ..Default::default()
        });

        router.start_canary().unwrap();

        // Simulate bad canary performance
        for _ in 0..150 {
            router.record_canary_observation(true, 100, 0.5).unwrap(); // High errors, low metric
            router.record_baseline_observation(false, 10, 0.95).unwrap();
        }

        let action = router.evaluate_and_promote().unwrap();
        
        assert_eq!(action, CanaryAction::Rollback);
        router.execute_action(action).unwrap();
        
        assert_eq!(router.get_traffic_percentage(), 0);
        assert_eq!(router.get_state(), CanaryState::RolledBack);
    }
}
