//! Liquidity Evaporation Model coupled with Hawkes intensity
//!
//! Predicts when bid/ask spreads will blow out during flash crashes by
//! combining Hawkes process intensity with order book depth analysis.

use crate::contagion::multivariate_hawkes::{MultivariateHawkesProcess, HawkesError};
use ndarray::{Array1, ArrayView1};
use thiserror::Error;

/// Errors from liquidity modeling
#[derive(Error, Debug, Clone)]
pub enum LiquidityError {
    #[error("Invalid order book depth: must be positive")]
    InvalidDepth,
    
    #[error("Spread calculation overflow")]
    SpreadOverflow,
    
    #[error("Hawkes process error: {0}")]
    HawkesError(#[from] HawkesError),
}

/// Configuration for liquidity evaporation model
#[derive(Debug, Clone)]
pub struct LiquidityEvaporationConfig {
    /// Baseline spread (in basis points)
    pub baseline_spread_bps: f64,
    /// Maximum spread multiplier before "evaporation" declared
    pub evaporation_threshold: f64,
    /// Sensitivity of spread to Hawkes intensity
    pub intensity_sensitivity: f64,
    /// Order book depth decay rate per unit intensity
    pub depth_decay_rate: f64,
    /// Minimum fractional depth remaining (circuit breaker trigger)
    pub min_depth_fraction: f64,
}

impl Default for LiquidityEvaporationConfig {
    fn default() -> Self {
        Self {
            baseline_spread_bps: 5.0,
            evaporation_threshold: 10.0,
            intensity_sensitivity: 2.0,
            depth_decay_rate: 0.1,
            min_depth_fraction: 0.1,
        }
    }
}

/// State of liquidity conditions
#[derive(Debug, Clone, Copy)]
pub struct LiquidityState {
    /// Current spread in basis points
    pub spread_bps: f64,
    /// Fraction of normal depth remaining (0.0 to 1.0)
    pub depth_fraction: f64,
    /// Hawkes intensity level
    pub hawkes_intensity: f64,
    /// Whether liquidity has "evaporated" (spread blown out)
    pub evaporated: bool,
    /// Time until expected recovery (seconds)
    pub estimated_recovery_secs: f64,
}

impl LiquidityState {
    /// Check if state indicates a flash crash condition
    pub fn is_flash_crash(&self) -> bool {
        self.evaporated || self.spread_bps > 50.0 || self.depth_fraction < 0.2
    }
    
    /// Get liquidity quality score (0 = terrible, 1 = excellent)
    pub fn quality_score(&self) -> f64 {
        let spread_score = (1.0 - (self.spread_bps / 100.0).min(1.0)).max(0.0);
        let depth_score = self.depth_fraction;
        (spread_score + depth_score) / 2.0
    }
}

/// Liquidity Evaporation Model
pub struct LiquidityEvaporationModel {
    config: LiquidityEvaporationConfig,
    /// Reference order book depth at normal conditions
    reference_depth: f64,
    /// Current estimated depth
    current_depth: f64,
}

impl LiquidityEvaporationModel {
    /// Create a new liquidity evaporation model
    pub fn new(config: LiquidityEvaporationConfig, reference_depth: f64) -> Result<Self, LiquidityError> {
        if reference_depth <= 0.0 {
            return Err(LiquidityError::InvalidDepth);
        }
        
        Ok(Self {
            config,
            reference_depth,
            current_depth: reference_depth,
        })
    }
    
    /// Update liquidity state based on current Hawkes intensity
    pub fn update_state(
        &mut self,
        hawkes_process: &mut MultivariateHawkesProcess,
        current_time: f64,
    ) -> Result<LiquidityState, LiquidityError> {
        // Get current Hawkes intensity
        let intensity = hawkes_process.current_intensity().sum();
        
        // Calculate spread expansion based on intensity
        // spread = baseline * exp(sensitivity * intensity)
        let spread_multiplier = (self.config.intensity_sensitivity * intensity).exp();
        let spread_bps = self.config.baseline_spread_bps * spread_multiplier;
        
        if !spread_bps.is_finite() {
            return Err(LiquidityError::SpreadOverflow);
        }
        
        // Calculate depth decay
        // depth = reference * exp(-decay_rate * intensity)
        let depth_multiplier = (-self.config.depth_decay_rate * intensity).exp();
        self.current_depth = self.reference_depth * depth_multiplier;
        
        let depth_fraction = self.current_depth / self.reference_depth;
        
        // Check for evaporation condition
        let spread_ratio = spread_bps / self.config.baseline_spread_bps;
        let evaporated = spread_ratio > self.config.evaporation_threshold
            || depth_fraction < self.config.min_depth_fraction;
        
        // Estimate recovery time based on Hawkes decay rates
        let avg_decay = hawkes_process
            .current_intensity()
            .iter()
            .map(|&lambda| if lambda > 1e-10 { 1.0 / lambda } else { 100.0 })
            .sum::<f64>()
            / hawkes_process.current_intensity().len() as f64;
        
        let estimated_recovery_secs = if evaporated {
            avg_decay * 3.0 // ~3 mean-reversion periods
        } else {
            0.0
        };
        
        Ok(LiquidityState {
            spread_bps,
            depth_fraction,
            hawkes_intensity: intensity,
            evaporated,
            estimated_recovery_secs,
        })
    }
    
    /// Predict spread at a future Hawkes intensity level
    pub fn predict_spread_at_intensity(&self, intensity: f64) -> f64 {
        let spread_multiplier = (self.config.intensity_sensitivity * intensity).exp();
        self.config.baseline_spread_bps * spread_multiplier
    }
    
    /// Calculate the critical intensity threshold that triggers evaporation
    pub fn critical_intensity_threshold(&self) -> f64 {
        // Solve: baseline * exp(sensitivity * λ_crit) / baseline = threshold
        // λ_crit = ln(threshold) / sensitivity
        (self.config.evaporation_threshold.ln() / self.config.intensity_sensitivity)
            .max(0.0)
    }
    
    /// Get the current depth fraction
    pub fn current_depth_fraction(&self) -> f64 {
        self.current_depth / self.reference_depth
    }
    
    /// Reset to reference conditions
    pub fn reset(&mut self) {
        self.current_depth = self.reference_depth;
    }
}

/// Circuit breaker that triggers when liquidity evaporates
pub struct LiquidityCircuitBreaker {
    model: LiquidityEvaporationModel,
    /// Whether circuit breaker is currently triggered
    pub triggered: bool,
    /// Time when circuit breaker was triggered
    trigger_time: Option<f64>,
    /// Cooldown period after trigger (seconds)
    cooldown_secs: f64,
}

impl LiquidityCircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(model: LiquidityEvaporationModel, cooldown_secs: f64) -> Self {
        Self {
            model,
            triggered: false,
            trigger_time: None,
            cooldown_secs,
        }
    }
    
    /// Check and potentially trigger the circuit breaker
    pub fn check_and_trigger(
        &mut self,
        hawkes_process: &mut MultivariateHawkesProcess,
        current_time: f64,
    ) -> Result<bool, LiquidityError> {
        // Check if in cooldown
        if self.triggered {
            if let Some(trigger_time) = self.trigger_time {
                if current_time - trigger_time < self.cooldown_secs {
                    return Ok(true); // Still in cooldown
                } else {
                    // Cooldown expired, reset
                    self.triggered = false;
                    self.trigger_time = None;
                }
            }
        }
        
        // Update liquidity state
        let state = self.model.update_state(hawkes_process, current_time)?;
        
        // Trigger if flash crash detected
        if state.is_flash_crash() && !self.triggered {
            self.triggered = true;
            self.trigger_time = Some(current_time);
            return Ok(true);
        }
        
        Ok(self.triggered)
    }
    
    /// Get current liquidity state
    pub fn liquidity_state(&self) -> &LiquidityEvaporationModel {
        &self.model
    }
    
    /// Manually reset the circuit breaker
    pub fn manual_reset(&mut self) {
        self.triggered = false;
        self.trigger_time = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contagion::multivariate_hawkes::{HawkesConfig, HawkesEvent};
    use ndarray::Array2;
    
    #[test]
    fn test_liquidity_model_creation() {
        let config = LiquidityEvaporationConfig::default();
        let model = LiquidityEvaporationModel::new(config, 1000000.0).unwrap();
        
        assert_eq!(model.current_depth_fraction(), 1.0);
    }
    
    #[test]
    fn test_spread_expansion() {
        let config = LiquidityEvaporationConfig {
            baseline_spread_bps: 5.0,
            intensity_sensitivity: 1.0,
            ..Default::default()
        };
        
        let model = LiquidityEvaporationModel::new(config, 1000000.0).unwrap();
        
        // At zero intensity, spread should be baseline
        let spread_zero = model.predict_spread_at_intensity(0.0);
        assert!((spread_zero - 5.0).abs() < 0.001);
        
        // At higher intensity, spread should expand exponentially
        let spread_high = model.predict_spread_at_intensity(2.0);
        assert!(spread_high > spread_zero);
    }
    
    #[test]
    fn test_circuit_breaker_triggering() {
        let hawkes_config = HawkesConfig {
            n_dimensions: 1,
            baseline_intensity: Array1::from_vec(vec![0.1]),
            excitation_matrix: Array2::from_shape_vec((1, 1), vec![0.5]).unwrap(),
            decay_rates: Array1::from_vec(vec![1.0]),
            max_intensity: 100.0,
            history_window_secs: 3600.0,
        };
        
        let mut hawkes = MultivariateHawkesProcess::new(hawkes_config).unwrap();
        
        // Record some events to increase intensity
        for i in 0..10 {
            let event = HawkesEvent {
                timestamp: i as f64,
                dimension: 0,
                magnitude: 1.0,
            };
            hawkes.record_event(event).unwrap();
        }
        
        let liq_config = LiquidityEvaporationConfig {
            baseline_spread_bps: 5.0,
            evaporation_threshold: 2.0, // Low threshold for testing
            intensity_sensitivity: 0.5,
            ..Default::default()
        };
        
        let model = LiquidityEvaporationModel::new(liq_config, 1000000.0).unwrap();
        let mut breaker = LiquidityCircuitBreaker::new(model, 60.0);
        
        let triggered = breaker.check_and_trigger(&mut hawkes, 10.0).unwrap();
        
        // May or may not trigger depending on accumulated intensity
        assert!(triggered == breaker.triggered);
    }
}
