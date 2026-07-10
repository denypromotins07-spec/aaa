//! Multivariate Hawkes Process for detecting spoofing and layering patterns.
//! 
//! Models cross-excitation between order submissions and cancellations at different price levels.
//! Uses self-exciting point processes to identify when cancellation rates diverge from baseline.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HawkesError {
    #[error("Intensity overflow: lambda exceeded maximum value")]
    IntensityOverflow,
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Signal indicating detected spoofing activity
#[derive(Debug, Clone)]
pub struct SpoofingSignal {
    pub price_level: i64,
    pub intensity: f64,
    pub confidence: f64,
    pub timestamp_ns: u64,
    pub is_layering: bool,
}

/// Configuration for the Hawkes spoofing detector
pub struct HawkesConfig {
    /// Base intensity (background rate)
    pub mu: f64,
    /// Excitation factor (how much each event increases intensity)
    pub alpha: f64,
    /// Decay rate (exponential decay of excitation)
    pub beta: f64,
    /// Maximum allowed intensity to prevent overflow
    pub max_intensity: f64,
    /// Minimum events required before signaling
    pub min_events: usize,
}

impl Default for HawkesConfig {
    fn default() -> Self {
        Self {
            mu: 0.1,
            alpha: 0.5,
            beta: 1.0,
            max_intensity: 1e10, // Strict clamp to prevent f64 overflow
            min_events: 10,
        }
    }
}

/// Per-price-level Hawkes process state
struct PriceLevelState {
    /// Current intensity λ(t)
    intensity: f64,
    /// Last event timestamp
    last_timestamp_ns: u64,
    /// Event count for this level
    event_count: usize,
    /// Sum of exp(-beta * (t - t_i)) for efficient updates
    decay_sum: f64,
}

impl PriceLevelState {
    fn new() -> Self {
        Self {
            intensity: 0.0,
            last_timestamp_ns: 0,
            event_count: 0,
            decay_sum: 0.0,
        }
    }
}

/// Multivariate Hawkes Process detector for spoofing and layering
pub struct HawkesSpoofingDetector {
    config: HawkesConfig,
    /// State per price level (bid/ask prices as keys)
    states: RwLock<HashMap<i64, PriceLevelState>>,
    /// Submission events counter
    submission_count: AtomicU64,
    /// Cancellation events counter
    cancellation_count: AtomicU64,
    /// Global timestamp for decay calculations
    global_time_ns: AtomicU64,
}

impl HawkesSpoofingDetector {
    /// Create a new Hawkes spoofing detector with given configuration
    pub fn new(config: HawkesConfig) -> Result<Self, HawkesError> {
        // Validate parameters to prevent numerical instability
        if config.alpha <= 0.0 || config.beta <= 0.0 || config.mu < 0.0 {
            return Err(HawkesError::InvalidParameter(
                "alpha, beta must be > 0, mu must be >= 0".to_string(),
            ));
        }
        if config.alpha >= config.beta {
            // Stationarity condition: alpha < beta for stable Hawkes process
            return Err(HawkesError::InvalidParameter(
                "Stationarity violation: alpha must be < beta".to_string(),
            ));
        }
        
        Ok(Self {
            config,
            states: RwLock::new(HashMap::with_capacity(256)),
            submission_count: AtomicU64::new(0),
            cancellation_count: AtomicU64::new(0),
            global_time_ns: AtomicU64::new(0),
        })
    }

    /// Record an order submission event at a specific price level
    #[inline]
    pub fn record_submission(&self, price_level: i64, timestamp_ns: u64) {
        self.submission_count.fetch_add(1, Ordering::Relaxed);
        self.update_intensity(price_level, timestamp_ns, false);
    }

    /// Record an order cancellation event at a specific price level
    #[inline]
    pub fn record_cancellation(&self, price_level: i64, timestamp_ns: u64) {
        self.cancellation_count.fetch_add(1, Ordering::Relaxed);
        self.update_intensity(price_level, timestamp_ns, true);
    }

    /// Update intensity for a price level using the Hawkes process formula:
    /// λ(t) = μ + α * Σ exp(-β * (t - t_i))
    /// 
    /// Uses efficient recursive update: λ(t) = λ(t_prev) * exp(-βΔt) + α * δ(t - t_event)
    #[inline]
    fn update_intensity(&self, price_level: i64, timestamp_ns: u64, is_cancellation: bool) {
        let mut states = self.states.write();
        
        let state = states.entry(price_level).or_insert_with(PriceLevelState::new);
        
        // Calculate time delta
        let dt_ns = if state.last_timestamp_ns > 0 {
            timestamp_ns.saturating_sub(state.last_timestamp_ns)
        } else {
            0
        };
        
        // Convert to seconds for decay calculation
        let dt_sec = dt_ns as f64 / 1e9;
        
        // Exponential decay of previous intensity
        let decay_factor = (-self.config.beta * dt_sec).exp();
        
        // Update decay_sum efficiently
        state.decay_sum *= decay_factor;
        
        // Add new event contribution
        state.decay_sum += 1.0;
        
        // Calculate new intensity: λ(t) = μ + α * decay_sum
        let new_intensity = self.config.mu + self.config.alpha * state.decay_sum;
        
        // CRITICAL: Clamp intensity to prevent f64 overflow (Audit Fix #1)
        state.intensity = new_intensity.min(self.config.max_intensity);
        state.last_timestamp_ns = timestamp_ns;
        
        if is_cancellation {
            state.event_count += 1;
        }
    }

    /// Check for spoofing signals at all price levels
    /// Returns Vec<SpoofingSignal> for levels with anomalous intensity
    pub fn detect_spoofing(&self, current_time_ns: u64, threshold_multiplier: f64) -> Vec<SpoofingSignal> {
        let states = self.states.read();
        let mut signals = Vec::with_capacity(states.len());
        
        // Calculate global baseline intensity
        let baseline = self.calculate_baseline_intensity();
        
        for (&price_level, state) in states.iter() {
            if state.event_count < self.config.min_events {
                continue;
            }
            
            // Apply decay to current time
            let dt_ns = current_time_ns.saturating_sub(state.last_timestamp_ns);
            let dt_sec = dt_ns as f64 / 1e9;
            let decay_factor = (-self.config.beta * dt_sec).exp();
            let current_intensity = state.intensity * decay_factor + self.config.mu * (1.0 - decay_factor);
            
            // Clamp again after decay
            let clamped_intensity = current_intensity.min(self.config.max_intensity);
            
            // Detect anomaly: intensity significantly above baseline
            let threshold = baseline * threshold_multiplier;
            if clamped_intensity > threshold {
                let confidence = ((clamped_intensity - baseline) / baseline).min(1.0);
                
                // Layering detection: multiple price levels with correlated high intensity
                let is_layering = self.detect_layering_pattern(&states, price_level, current_time_ns);
                
                signals.push(SpoofingSignal {
                    price_level,
                    intensity: clamped_intensity,
                    confidence,
                    timestamp_ns: current_time_ns,
                    is_layering,
                });
            }
        }
        
        signals
    }

    /// Calculate baseline intensity across all levels
    fn calculate_baseline_intensity(&self) -> f64 {
        let states = self.states.read();
        if states.is_empty() {
            return self.config.mu;
        }
        
        let sum: f64 = states.values().map(|s| s.intensity).sum();
        sum / states.len() as f64
    }

    /// Detect layering pattern: correlated high intensity across multiple adjacent price levels
    fn detect_layering_pattern(&self, states: &HashMap<i64, PriceLevelState>, center_price: i64, current_time_ns: u64) -> bool {
        let mut layered_count = 0;
        let threshold = self.calculate_baseline_intensity() * 2.0;
        
        // Check adjacent price levels (±5 ticks)
        for offset in -5..=5 {
            if offset == 0 {
                continue;
            }
            
            let check_price = center_price + offset;
            if let Some(state) = states.get(&check_price) {
                // Apply decay
                let dt_ns = current_time_ns.saturating_sub(state.last_timestamp_ns);
                let dt_sec = dt_ns as f64 / 1e9;
                let decayed = state.intensity * (-self.config.beta * dt_sec).exp();
                
                if decayed > threshold {
                    layered_count += 1;
                }
            }
        }
        
        // Layering requires at least 3 adjacent levels with high intensity
        layered_count >= 3
    }

    /// Get current intensity for a specific price level
    pub fn get_intensity(&self, price_level: i64, current_time_ns: u64) -> f64 {
        let states = self.states.read();
        if let Some(state) = states.get(&price_level) {
            let dt_ns = current_time_ns.saturating_sub(state.last_timestamp_ns);
            let dt_sec = dt_ns as f64 / 1e9;
            let decayed = state.intensity * (-self.config.beta * dt_sec).exp();
            decayed.min(self.config.max_intensity)
        } else {
            self.config.mu
        }
    }

    /// Reset detector state (e.g., after market close)
    pub fn reset(&self) {
        self.states.write().clear();
        self.submission_count.store(0, Ordering::Relaxed);
        self.cancellation_count.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hawkes_intensity_clamping() {
        let config = HawkesConfig {
            max_intensity: 1e6,
            ..Default::default()
        };
        let detector = HawkesSpoofingDetector::new(config).unwrap();
        
        // Simulate massive burst of events
        for i in 0..10000 {
            detector.record_cancellation(100, i * 1000);
        }
        
        let intensity = detector.get_intensity(100, 10000000);
        assert!(intensity.is_finite());
        assert!(intensity <= 1e6);
    }

    #[test]
    fn test_stationarity_check() {
        let config = HawkesConfig {
            alpha: 2.0,
            beta: 1.0, // alpha >= beta violates stationarity
            ..Default::default()
        };
        let result = HawkesSpoofingDetector::new(config);
        assert!(result.is_err());
    }
}
