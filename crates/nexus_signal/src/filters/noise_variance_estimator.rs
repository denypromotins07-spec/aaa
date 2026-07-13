//! Noise Variance Estimator for Kalman Filter Tuning
//! 
//! This module provides online estimation of measurement and process noise
//! variances, allowing the Kalman filter to adapt to changing market conditions.
//! 
//! Uses recursive algorithms to estimate variance without storing historical data.

use std::sync::atomic::{AtomicU64, Ordering};

/// Online variance estimator using Welford's algorithm
/// 
/// Zero-allocation, single-pass variance calculation
pub struct OnlineVarianceEstimator {
    count: u64,
    mean: f64,
    m2: f64, // Sum of squared differences from mean
    min_variance: f64,
    max_variance: f64,
}

impl OnlineVarianceEstimator {
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min_variance: 1e-10,
            max_variance: 1000.0,
        }
    }

    /// Create with variance bounds
    pub fn with_bounds(min_var: f64, max_var: f64) -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min_variance: min_var,
            max_variance: max_var,
        }
    }

    /// Add a new observation
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Get current variance estimate
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return self.max_variance; // Return max when insufficient data
        }
        
        let var = self.m2 / (self.count - 1) as f64;
        var.clamp(self.min_variance, self.max_variance)
    }

    /// Get current mean
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get sample count
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset estimator
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }

    /// Get standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Default for OnlineVarianceEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive noise estimator for Kalman filter parameters
/// 
/// Estimates both process noise (Q) and measurement noise (R) from residuals
pub struct AdaptiveNoiseEstimator {
    /// Estimator for measurement residuals
    measurement_estimator: OnlineVarianceEstimator,
    /// Estimator for process residuals (state changes)
    process_estimator: OnlineVarianceEstimator,
    /// Smoothing factor for adaptive updates [0, 1]
    adaptation_rate: f64,
    /// Minimum allowed noise values
    min_noise: f64,
    /// Maximum allowed noise values
    max_noise: f64,
    /// Last state estimate
    last_state: f64,
    /// Update count
    update_count: AtomicU64,
}

impl AdaptiveNoiseEstimator {
    pub fn new() -> Self {
        Self {
            measurement_estimator: OnlineVarianceEstimator::with_bounds(1e-6, 100.0),
            process_estimator: OnlineVarianceEstimator::with_bounds(1e-8, 10.0),
            adaptation_rate: 0.1,
            min_noise: 1e-8,
            max_noise: 100.0,
            last_state: 0.0,
            update_count: AtomicU64::new(0),
        }
    }

    /// Create with custom adaptation rate
    pub fn with_adaptation_rate(rate: f64) -> Self {
        Self {
            measurement_estimator: OnlineVarianceEstimator::with_bounds(1e-6, 100.0),
            process_estimator: OnlineVarianceEstimator::with_bounds(1e-8, 10.0),
            adaptation_rate: rate.clamp(0.01, 1.0),
            min_noise: 1e-8,
            max_noise: 100.0,
            last_state: 0.0,
            update_count: AtomicU64::new(0),
        }
    }

    /// Update with new measurement and state estimate
    pub fn update(&mut self, measurement: f64, state: f64) {
        self.update_count.fetch_add(1, Ordering::Relaxed);

        // Measurement residual (innovation)
        let measurement_residual = measurement - state;
        self.measurement_estimator.update(measurement_residual.abs());

        // Process residual (state change)
        let process_residual = (state - self.last_state).abs();
        self.process_estimator.update(process_residual);

        self.last_state = state;
    }

    /// Get estimated measurement noise (R)
    pub fn estimated_measurement_noise(&self) -> f64 {
        self.measurement_estimator.variance().clamp(self.min_noise, self.max_noise)
    }

    /// Get estimated process noise (Q)
    pub fn estimated_process_noise(&self) -> f64 {
        self.process_estimator.variance().clamp(self.min_noise, self.max_noise)
    }

    /// Get recommended Kalman config based on current estimates
    pub fn get_recommended_config(&self) -> crate::filters::simd_kalman_filter::KalmanConfig {
        use crate::filters::simd_kalman_filter::KalmanConfig;
        
        let r = self.estimated_measurement_noise();
        let q = self.estimated_process_noise();

        KalmanConfig {
            process_noise: q,
            measurement_noise: r,
            initial_covariance: r,
            covariance_floor: 1e-10,
        }
    }

    /// Check if estimates have stabilized
    pub fn is_stable(&self, window_size: usize) -> bool {
        self.measurement_estimator.count() >= window_size as u64
            && self.process_estimator.count() >= window_size as u64
    }

    /// Get update count
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }

    /// Reset estimators
    pub fn reset(&mut self) {
        self.measurement_estimator.reset();
        self.process_estimator.reset();
        self.last_state = 0.0;
    }
}

impl Default for AdaptiveNoiseEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_variance_basic() {
        let mut estimator = OnlineVarianceEstimator::new();
        
        // Known variance test
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        for v in values {
            estimator.update(v);
        }

        // Mean should be 5.0
        assert!((estimator.mean() - 5.0).abs() < 1e-10);
        
        // Population variance should be 4.0
        let var = estimator.variance();
        assert!((var - 4.0).abs() < 0.5);
    }

    #[test]
    fn test_online_variance_single_value() {
        let mut estimator = OnlineVarianceEstimator::new();
        estimator.update(5.0);
        
        // Should return max variance with only one sample
        assert_eq!(estimator.variance(), estimator.max_variance);
    }

    #[test]
    fn test_adaptive_noise_estimation() {
        let mut estimator = AdaptiveNoiseEstimator::new();
        
        // Simulate noisy measurements around a stable state
        let true_state = 10.0;
        for i in 0..100 {
            let measurement = true_state + (i as f64 * 0.1).sin() * 2.0;
            estimator.update(measurement, true_state);
        }

        // Should have reasonable noise estimates
        let r = estimator.estimated_measurement_noise();
        assert!(r > 0.0);
        assert!(r < 10.0);
    }

    #[test]
    fn test_adaptive_noise_stability() {
        let mut estimator = AdaptiveNoiseEstimator::new();
        
        assert!(!estimator.is_stable(10));
        
        for i in 0..20 {
            estimator.update(10.0 + i as f64, 10.0);
        }
        
        assert!(estimator.is_stable(10));
    }
}
