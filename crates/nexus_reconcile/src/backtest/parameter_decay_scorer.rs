//! Parameter Decay Scorer - Detects when Alpha parameters lose predictive power.
//! 
//! This module monitors the walk-forward backtest results and emits a RegimeDecay
//! signal when the live Sharpe ratio drops below the theoretical baseline,
//! triggering parameter re-optimization.

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};
use std::sync::Arc;

use super::walk_forward_micro_bt::WalkForwardResult;

/// Configuration for decay detection
#[derive(Debug, Clone)]
pub struct DecayScorerConfig {
    /// Baseline Sharpe ratio (scaled by 100) expected from backtests
    pub baseline_sharpe_x100: i64,
    
    /// Minimum Sharpe ratio before decay is triggered (scaled by 100)
    pub min_acceptable_sharpe_x100: i64,
    
    /// Number of consecutive decay signals required before emitting RegimeDecay
    pub consecutive_decay_threshold: u32,
    
    /// Cooldown period between decay emissions (in iterations)
    pub cooldown_iterations: u32,
    
    /// Hysteresis: new model must be this much better to trigger swap (scaled by 100)
    pub hysteresis_threshold_x100: i64,
}

impl Default for DecayScorerConfig {
    fn default() -> Self {
        Self {
            baseline_sharpe_x100: 150,  // Expected Sharpe of 1.5
            min_acceptable_sharpe_x100: 75,  // Minimum acceptable Sharpe of 0.75
            consecutive_decay_threshold: 3,  // Require 3 consecutive decay signals
            cooldown_iterations: 10,  // 10 iteration cooldown
            hysteresis_threshold_x100: 50,  // New model must be 0.5 Sharpe better
        }
    }
}

/// Result of decay scoring
#[derive(Debug, Clone, PartialEq)]
pub enum DecayState {
    /// Parameters are performing within expectations
    Healthy,
    
    /// Performance is degrading but not yet critical
    Warning { 
        current_sharpe_x100: i64,
        baseline_sharpe_x100: i64,
        decay_percentage: u32,
    },
    
    /// Critical decay detected - emit RegimeDecay signal
    RegimeDecay {
        current_sharpe_x100: i64,
        consecutive_count: u32,
        recommended_action: DecayAction,
    },
}

/// Recommended action when decay is detected
#[derive(Debug, Clone, PartialEq)]
pub enum DecayAction {
    /// Continue monitoring, no action needed
    Monitor,
    
    /// Trigger genetic optimizer to evolve new parameters
    EvolveParameters,
    
    /// Trigger MLOps hot-swap to promote shadow model
    HotSwapModel,
    
    /// Reduce position sizes due to uncertainty
    ReduceExposure,
}

/// Statistics about decay scoring
#[derive(Debug, Clone, Default)]
pub struct DecayStats {
    pub total_evaluations: u64,
    pub healthy_count: u64,
    pub warning_count: u64,
    pub decay_count: u64,
    pub regime_decay_emitted: u64,
    pub current_consecutive_decay: u32,
    pub iterations_in_cooldown: u32,
}

/// Parameter Decay Scorer
pub struct ParameterDecayScorer {
    config: DecayScorerConfig,
    
    /// Current consecutive decay counter
    consecutive_decay_count: AtomicU32,
    
    /// Iterations since last decay emission
    iterations_since_emission: AtomicU32,
    
    /// Whether currently in cooldown
    in_cooldown: AtomicBool,
    
    /// Last observed Sharpe ratio
    last_sharpe_x100: AtomicI64,
    
    /// Statistics
    stats: Arc<parking_lot::RwLock<DecayStats>>,
}

impl ParameterDecayScorer {
    pub fn new(config: DecayScorerConfig) -> Self {
        Self {
            config,
            consecutive_decay_count: AtomicU32::new(0),
            iterations_since_emission: AtomicU32::new(u32::MAX),  // Start out of cooldown
            in_cooldown: AtomicBool::new(false),
            last_sharpe_x100: AtomicI64::new(0),
            stats: Arc::new(parking_lot::RwLock::new(DecayStats::default())),
        }
    }
    
    /// Evaluate a new walk-forward result and return the decay state
    pub fn evaluate(&self, result: &WalkForwardResult) -> DecayState {
        let mut stats = self.stats.write();
        stats.total_evaluations += 1;
        
        let current_sharpe = result.sharpe_ratio_x10000 / 100;  // Convert to x100 scale
        self.last_sharpe_x100.store(current_sharpe, Ordering::Relaxed);
        
        // Check if in cooldown
        if self.in_cooldown.load(Ordering::Relaxed) {
            let since_emission = self.iterations_since_emission.fetch_add(1, Ordering::Relaxed) + 1;
            
            if since_emission >= self.config.cooldown_iterations {
                self.in_cooldown.store(false, Ordering::Relaxed);
                self.iterations_since_emission.store(0, Ordering::Relaxed);
                stats.iterations_in_cooldown = 0;
            } else {
                stats.iterations_in_cooldown = self.config.cooldown_iterations - since_emission;
                return DecayState::Healthy;  // Don't emit during cooldown
            }
        }
        
        // Calculate decay percentage
        let decay_pct = if self.config.baseline_sharpe_x100 > 0 {
            (((self.config.baseline_sharpe_x100 - current_sharpe) * 100) 
                / self.config.baseline_sharpe_x100).max(0) as u32
        } else {
            0
        };
        
        // Determine state
        let state = if current_sharpe >= self.config.baseline_sharpe_x100 {
            // Performing at or above baseline
            self.consecutive_decay_count.store(0, Ordering::Relaxed);
            stats.healthy_count += 1;
            stats.current_consecutive_decay = 0;
            DecayState::Healthy
        } else if current_sharpe >= self.config.min_acceptable_sharpe_x100 {
            // Below baseline but still acceptable
            self.consecutive_decay_count.store(0, Ordering::Relaxed);
            stats.warning_count += 1;
            stats.current_consecutive_decay = 0;
            DecayState::Warning {
                current_sharpe_x100: current_sharpe,
                baseline_sharpe_x100: self.config.baseline_sharpe_x100,
                decay_percentage: decay_pct,
            }
        } else {
            // Below minimum acceptable - potential decay
            let count = self.consecutive_decay_count.fetch_add(1, Ordering::Relaxed) + 1;
            stats.current_consecutive_decay = count;
            stats.decay_count += 1;
            
            if count >= self.config.consecutive_decay_threshold {
                // Threshold reached - emit RegimeDecay
                stats.regime_decay_emitted += 1;
                
                // Enter cooldown
                self.in_cooldown.store(true, Ordering::Relaxed);
                self.iterations_since_emission.store(0, Ordering::Relaxed);
                
                // Determine recommended action based on severity
                let action = if current_sharpe < 0 {
                    DecayAction::ReduceExposure
                } else if decay_pct > 70 {
                    DecayAction::HotSwapModel
                } else {
                    DecayAction::EvolveParameters
                };
                
                DecayState::RegimeDecay {
                    current_sharpe_x100: current_sharpe,
                    consecutive_count: count,
                    recommended_action: action,
                }
            } else {
                DecayState::Warning {
                    current_sharpe_x100: current_sharpe,
                    baseline_sharpe_x100: self.config.baseline_sharpe_x100,
                    decay_percentage: decay_pct,
                }
            }
        };
        
        state
    }
    
    /// Get the last observed Sharpe ratio
    pub fn get_last_sharpe(&self) -> i64 {
        self.last_sharpe_x100.load(Ordering::Relaxed)
    }
    
    /// Get current statistics
    pub fn get_stats(&self) -> DecayStats {
        self.stats.read().clone()
    }
    
    /// Check if currently in cooldown
    pub fn is_in_cooldown(&self) -> bool {
        self.in_cooldown.load(Ordering::Relaxed)
    }
    
    /// Get remaining cooldown iterations
    pub fn get_cooldown_remaining(&self) -> u32 {
        if !self.in_cooldown.load(Ordering::Relaxed) {
            return 0;
        }
        
        let since = self.iterations_since_emission.load(Ordering::Relaxed);
        self.config.cooldown_iterations.saturating_sub(since)
    }
    
    /// Reset the scorer (e.g., after parameter update)
    pub fn reset(&self) {
        self.consecutive_decay_count.store(0, Ordering::Relaxed);
        self.in_cooldown.store(false, Ordering::Relaxed);
        self.iterations_since_emission.store(u32::MAX, Ordering::Relaxed);
    }
}

// Need to add AtomicU32 import
use std::sync::atomic::AtomicU32;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_healthy_state() {
        let config = DecayScorerConfig::default();
        let scorer = ParameterDecayScorer::new(config);
        
        let result = WalkForwardResult {
            ticks_processed: 1000,
            total_pnl_scaled: 1_000_000,
            sharpe_ratio_x10000: 20000,  // Sharpe of 2.0
            max_drawdown_scaled: 100_000,
            win_rate_x10000: 6000,
            trade_count: 50,
        };
        
        let state = scorer.evaluate(&result);
        assert_eq!(state, DecayState::Healthy);
        assert!(!scorer.is_in_cooldown());
    }
    
    #[test]
    fn test_warning_state() {
        let config = DecayScorerConfig::default();
        let scorer = ParameterDecayScorer::new(config);
        
        let result = WalkForwardResult {
            ticks_processed: 1000,
            total_pnl_scaled: 500_000,
            sharpe_ratio_x10000: 10000,  // Sharpe of 1.0 (below baseline 1.5, above min 0.75)
            max_drawdown_scaled: 200_000,
            win_rate_x10000: 5500,
            trade_count: 40,
        };
        
        let state = scorer.evaluate(&result);
        assert!(matches!(state, DecayState::Warning { .. }));
    }
    
    #[test]
    fn test_regime_decay_after_consecutive() {
        let config = DecayScorerConfig {
            consecutive_decay_threshold: 3,
            ..Default::default()
        };
        let scorer = ParameterDecayScorer::new(config);
        
        let bad_result = WalkForwardResult {
            ticks_processed: 1000,
            total_pnl_scaled: -100_000,
            sharpe_ratio_x10000: 5000,  // Sharpe of 0.5 (below min 0.75)
            max_drawdown_scaled: 500_000,
            win_rate_x10000: 4000,
            trade_count: 30,
        };
        
        // First two evaluations should be warnings
        let state1 = scorer.evaluate(&bad_result);
        assert!(matches!(state1, DecayState::Warning { .. }));
        
        let state2 = scorer.evaluate(&bad_result);
        assert!(matches!(state2, DecayState::Warning { .. }));
        
        // Third evaluation should trigger RegimeDecay
        let state3 = scorer.evaluate(&bad_result);
        assert!(matches!(state3, DecayState::RegimeDecay { .. }));
        assert!(scorer.is_in_cooldown());
    }
    
    #[test]
    fn test_cooldown_prevents_rapid_emission() {
        let config = DecayScorerConfig {
            consecutive_decay_threshold: 2,
            cooldown_iterations: 5,
            ..Default::default()
        };
        let scorer = ParameterDecayScorer::new(config);
        
        let bad_result = WalkForwardResult {
            sharpe_ratio_x10000: 5000,
            ..Default::default()
        };
        
        // Trigger first decay
        scorer.evaluate(&bad_result);
        scorer.evaluate(&bad_result);  // This triggers RegimeDecay
        
        // Next evaluations should be Healthy due to cooldown
        for _ in 0..4 {
            let state = scorer.evaluate(&bad_result);
            assert_eq!(state, DecayState::Healthy);
        }
        
        // After cooldown expires, should detect decay again
        let state = scorer.evaluate(&bad_result);
        assert!(matches!(state, DecayState::RegimeDecay { .. }));
    }
}
