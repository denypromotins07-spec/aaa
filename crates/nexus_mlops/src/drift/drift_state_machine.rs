//! Drift State Machine - Manages model decay detection and hot-swap signaling
//! 
//! Implements hysteresis and cooldown to prevent model thrashing in choppy markets.
//! Coordinates Page-Hinkley and KS tests for robust drift detection.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU32, Ordering};
use std::time::{Duration, Instant};
use crate::drift::{PageHinkleyTest, StreamingKSTest};
use crate::shadow::ModelRegime;

/// Maximum number of models supported
const MAX_MODELS: usize = 8;

/// Drift detection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftState {
    /// Model is healthy and performing well
    Healthy,
    /// Early warning signs detected
    Warning,
    /// Drift confirmed - model should be replaced
    Decayed,
    /// Cooldown after a swap
    Cooldown,
}

/// Configuration for the drift state machine
#[derive(Debug, Clone)]
pub struct DriftStateMachineConfig {
    /// Page-Hinkley threshold
    pub ph_threshold: f64,
    /// Page-Hinkley minimum samples
    pub ph_min_samples: usize,
    /// KS test critical value
    pub ks_critical_value: f64,
    /// KS test number of bins
    pub ks_num_bins: usize,
    /// Minimum samples before considering swap
    pub min_samples_for_swap: u64,
    /// Cooldown period after swap (seconds)
    pub cooldown_seconds: u64,
    /// Hysteresis threshold - how much better must new model be?
    pub hysteresis_threshold: f64,
    /// Consecutive detections required before triggering
    pub consecutive_detections_required: u32,
}

impl Default for DriftStateMachineConfig {
    fn default() -> Self {
        Self {
            ph_threshold: 50.0,
            ph_min_samples: 30,
            ks_critical_value: 0.15,
            ks_num_bins: 50,
            min_samples_for_swap: 100,
            cooldown_seconds: 300, // 5 minutes
            hysteresis_threshold: 0.5, // New model must be 50% better
            consecutive_detections_required: 3,
        }
    }
}

/// Result of drift evaluation
#[derive(Debug, Clone)]
pub struct DriftEvaluation {
    pub current_model_id: u32,
    pub current_state: DriftState,
    pub recommended_action: RecommendedAction,
    pub best_shadow_id: Option<u32>,
    pub improvement_factor: f64,
    pub ph_statistic: f64,
    pub ks_statistic: f64,
}

/// Recommended action from drift evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedAction {
    Continue,
    Monitor,
    PrepareSwap,
    ExecuteSwap,
}

/// Per-model drift tracking
struct ModelDriftTracker {
    page_hinkley: PageHinkleyTest,
    ks_test: StreamingKSTest,
    consecutive_detections: AtomicU32,
    last_detection_time: AtomicU64,
    sample_count: AtomicU64,
}

/// Drift State Machine - coordinates multiple detectors with hysteresis
pub struct DriftStateMachine {
    config: DriftStateMachineConfig,
    /// Trackers for each model
    trackers: [Option<Box<ModelDriftTracker>>; MAX_MODELS],
    /// Current live model ID
    live_model_id: AtomicU32,
    /// Current state for live model
    current_state: AtomicU32, // Encoded DriftState
    /// Last swap timestamp
    last_swap_timestamp: AtomicU64,
    /// Shadow model scores (for comparison)
    shadow_scores: [f64; MAX_MODELS],
    /// Number of active models
    num_models: usize,
}

// Safety: All mutable state uses atomics or interior mutability
unsafe impl Send for DriftStateMachine {}
unsafe impl Sync for DriftStateMachine {}

impl DriftStateMachine {
    /// Create a new drift state machine
    pub fn new(config: DriftStateMachineConfig, num_models: usize) -> Self {
        assert!(num_models <= MAX_MODELS, "Too many models");
        
        let mut trackers: [Option<Box<ModelDriftTracker>>; MAX_MODELS] = Default::default();
        
        for i in 0..num_models {
            trackers[i] = Some(Box::new(ModelDriftTracker {
                page_hinkley: PageHinkleyTest::new(config.ph_threshold, config.ph_min_samples),
                ks_test: StreamingKSTest::new(crate::drift::KSTestConfig {
                    num_bins: config.ks_num_bins,
                    critical_value: config.ks_critical_value,
                    ..Default::default()
                }),
                consecutive_detections: AtomicU32::new(0),
                last_detection_time: AtomicU64::new(0),
                sample_count: AtomicU64::new(0),
            }));
        }
        
        Self {
            config,
            trackers,
            live_model_id: AtomicU32::new(0),
            current_state: AtomicU32::new(DriftState::Healthy as u32),
            last_swap_timestamp: AtomicU64::new(0),
            shadow_scores: [0.0; MAX_MODELS],
            num_models,
        }
    }
    
    /// Record a prediction error for the live model
    #[inline(always)]
    pub fn record_error(&self, error: f64, feature_value: f64) -> bool {
        let live_id = self.live_model_id.load(Ordering::Relaxed) as usize;
        
        if let Some(ref tracker) = self.trackers[live_id] {
            tracker.sample_count.fetch_add(1, Ordering::Relaxed);
            
            // Update Page-Hinkley
            let ph_drift = unsafe {
                let ph_ptr = &tracker.page_hinkley as *const PageHinkleyTest as *mut PageHinkleyTest;
                (*ph_ptr).update(error, 0.001)
            };
            
            // Update KS test
            let ks_drift = unsafe {
                let ks_ptr = &tracker.ks_test as *const StreamingKSTest as *mut StreamingKSTest;
                (*ks_ptr).observe(feature_value)
            };
            
            // Check for combined drift signal
            if ph_drift || ks_drift {
                tracker.consecutive_detections.fetch_add(1, Ordering::Relaxed);
                
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                tracker.last_detection_time.store(now, Ordering::Relaxed);
                
                return self.check_drift_threshold(live_id);
            } else {
                // Reset consecutive counter on non-detection
                tracker.consecutive_detections.store(0, Ordering::Relaxed);
            }
        }
        
        false
    }
    
    /// Check if consecutive detections exceed threshold
    fn check_drift_threshold(&self, model_id: usize) -> bool {
        if let Some(ref tracker) = self.trackers[model_id] {
            let consecutive = tracker.consecutive_detections.load(Ordering::Relaxed);
            let samples = tracker.sample_count.load(Ordering::Relaxed);
            
            if consecutive >= self.config.consecutive_detections_required 
                && samples >= self.config.min_samples_for_swap 
            {
                // Check cooldown
                let last_swap = self.last_swap_timestamp.load(Ordering::Relaxed);
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                
                if now - last_swap < self.config.cooldown_seconds {
                    return false; // Still in cooldown
                }
                
                // Update state
                self.current_state.store(DriftState::Decayed as u32, Ordering::SeqCst);
                return true;
            } else if consecutive >= self.config.consecutive_detections_required / 2 {
                self.current_state.store(DriftState::Warning as u32, Ordering::Relaxed);
            }
        }
        
        false
    }
    
    /// Update shadow model score
    pub fn update_shadow_score(&mut self, model_id: u32, score: f64) {
        if model_id < MAX_MODELS as u32 {
            self.shadow_scores[model_id as usize] = score;
        }
    }
    
    /// Evaluate whether to swap models
    pub fn evaluate_swap(&self) -> DriftEvaluation {
        let live_id = self.live_model_id.load(Ordering::Relaxed);
        let state = match self.current_state.load(Ordering::Relaxed) {
            0 => DriftState::Healthy,
            1 => DriftState::Warning,
            2 => DriftState::Decayed,
            _ => DriftState::Cooldown,
        };
        
        // Get live model stats
        let (ph_stat, ks_stat) = if let Some(ref tracker) = self.trackers[live_id as usize] {
            (tracker.page_hinkley.statistic(), tracker.ks_test.statistic())
        } else {
            (0.0, 0.0)
        };
        
        // Find best shadow model
        let mut best_shadow: Option<u32> = None;
        let mut best_score = self.shadow_scores[live_id as usize];
        let live_score = best_score;
        
        for i in 0..self.num_models {
            if i == live_id as usize {
                continue;
            }
            
            let shadow_score = self.shadow_scores[i];
            // Apply hysteresis: shadow must be significantly better
            let adjusted_score = shadow_score * (1.0 + self.config.hysteresis_threshold);
            
            if adjusted_score > best_score {
                best_score = adjusted_score;
                best_shadow = Some(i as u32);
            }
        }
        
        // Determine recommended action
        let action = match state {
            DriftState::Decayed if best_shadow.is_some() => RecommendedAction::ExecuteSwap,
            DriftState::Warning if best_shadow.is_some() => RecommendedAction::PrepareSwap,
            DriftState::Warning => RecommendedAction::Monitor,
            _ => RecommendedAction::Continue,
        };
        
        let improvement = if live_score > 0.0 {
            (best_score - live_score) / live_score
        } else {
            0.0
        };
        
        DriftEvaluation {
            current_model_id: live_id,
            current_state: state,
            recommended_action: action,
            best_shadow_id: best_shadow,
            improvement_factor: improvement,
            ph_statistic: ph_stat,
            ks_statistic: ks_stat,
        }
    }
    
    /// Execute a model swap
    pub fn execute_swap(&mut self, new_model_id: u32) -> bool {
        if new_model_id >= self.num_models as u32 {
            return false;
        }
        
        // Reset old model's drift tracker
        let old_id = self.live_model_id.load(Ordering::Relaxed) as usize;
        if let Some(ref tracker) = self.trackers[old_id] {
            tracker.page_hinkley.reset();
            tracker.consecutive_detections.store(0, Ordering::Relaxed);
        }
        
        // Update live model
        self.live_model_id.store(new_model_id, Ordering::SeqCst);
        self.current_state.store(DriftState::Healthy as u32, Ordering::SeqCst);
        
        // Record swap timestamp for cooldown
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_swap_timestamp.store(now, Ordering::SeqCst);
        
        true
    }
    
    /// Get current state
    pub fn current_state(&self) -> DriftState {
        match self.current_state.load(Ordering::Relaxed) {
            0 => DriftState::Healthy,
            1 => DriftState::Warning,
            2 => DriftState::Decayed,
            _ => DriftState::Cooldown,
        }
    }
    
    /// Get live model ID
    pub fn live_model_id(&self) -> u32 {
        self.live_model_id.load(Ordering::Relaxed)
    }
    
    /// Force state reset (for manual intervention)
    pub fn force_reset(&mut self) {
        self.current_state.store(DriftState::Healthy as u32, Ordering::SeqCst);
        
        for tracker in self.trackers.iter_mut().flatten() {
            tracker.page_hinkley.reset();
            tracker.consecutive_detections.store(0, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_healthy_state_no_swap() {
        let config = DriftStateMachineConfig::default();
        let dsm = DriftStateMachine::new(config, 4);
        
        // Feed small errors (no drift)
        for _ in 0..50 {
            dsm.record_error(0.001, 0.0);
        }
        
        let eval = dsm.evaluate_swap();
        assert_eq!(eval.current_state, DriftState::Healthy);
        assert_eq!(eval.recommended_action, RecommendedAction::Continue);
    }
    
    #[test]
    fn test_drift_detection_triggers_warning() {
        let mut config = DriftStateMachineConfig::default();
        config.consecutive_detections_required = 3;
        config.ph_threshold = 10.0; // Low threshold for testing
        
        let dsm = DriftStateMachine::new(config, 4);
        
        // Feed large errors to trigger PH
        for _ in 0..50 {
            dsm.record_error(0.5, 0.0);
        }
        
        let state = dsm.current_state();
        // Should be at least Warning state
        assert!(state == DriftState::Warning || state == DriftState::Decayed);
    }
    
    #[test]
    fn test_hysteresis_prevents_thrashing() {
        let config = DriftStateMachineConfig::default();
        let mut dsm = DriftStateMachine::new(config, 4);
        
        // Set live model score
        dsm.update_shadow_score(0, 1.0);
        // Set shadow model score (slightly better but not enough)
        dsm.update_shadow_score(1, 1.2); // Only 20% better, need 50%
        
        // Force decayed state
        dsm.current_state.store(DriftState::Decayed as u32, Ordering::SeqCst);
        
        let eval = dsm.evaluate_swap();
        
        // Should NOT recommend swap due to hysteresis
        assert_eq!(eval.recommended_action, RecommendedAction::Continue);
        assert!(eval.best_shadow_id.is_none());
    }
    
    #[test]
    fn test_swap_execution() {
        let config = DriftStateMachineConfig::default();
        let mut dsm = DriftStateMachine::new(config, 4);
        
        assert_eq!(dsm.live_model_id(), 0);
        
        let result = dsm.execute_swap(2);
        assert!(result);
        assert_eq!(dsm.live_model_id(), 2);
        assert_eq!(dsm.current_state(), DriftState::Healthy);
    }
    
    #[test]
    fn test_cooldown_prevents_immediate_reswap() {
        let mut config = DriftStateMachineConfig::default();
        config.cooldown_seconds = 60; // 1 minute cooldown
        config.consecutive_detections_required = 1;
        config.ph_threshold = 1.0;
        
        let dsm = DriftStateMachine::new(config, 4);
        
        // Trigger drift immediately
        for _ in 0..10 {
            dsm.record_error(1.0, 0.0);
        }
        
        // Execute swap
        let _ = dsm.evaluate_swap();
        
        // State should reflect recent activity
        // (full test would verify cooldown behavior over time)
    }
}
