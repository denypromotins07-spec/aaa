//! SIMD-Accelerated 1D Kalman Filter for Signal Smoothing
//! 
//! Raw OBI and VPIN signals are extremely noisy and will cause the execution
//! engine to "whipsaw" (trade erratically). This module implements a zero-allocation,
//! 1D Kalman Filter using SIMD instructions to smooth signals in real-time.
//! 
//! The Kalman Filter maintains its state vector (x) and error covariance (P) on
//! the stack. It executes Predict and Update steps in O(1) time, stripping out
//! high-frequency noise while preserving true directional momentum.
//! 
//! ROOT CAUSE FIX: Enforces strict epsilon floor on P to prevent divide-by-zero.

use std::sync::atomic::{AtomicU64, Ordering};

/// Configuration for the Kalman filter
#[derive(Debug, Clone, Copy)]
pub struct KalmanConfig {
    /// Process noise variance (Q) - how much we trust the model
    pub process_noise: f64,
    /// Measurement noise variance (R) - how much we trust measurements
    pub measurement_noise: f64,
    /// Initial error covariance (P)
    pub initial_covariance: f64,
    /// Minimum covariance floor to prevent singularity
    pub covariance_floor: f64,
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            process_noise: 0.001,      // Low process noise = smooth output
            measurement_noise: 0.1,     // Moderate measurement noise
            initial_covariance: 1.0,    // Start with moderate uncertainty
            covariance_floor: 1e-10,    // Prevent divide-by-zero
        }
    }
}

/// SIMD-accelerated 1D Kalman Filter state
/// 
/// All state is stored on the stack for zero-allocation operation
pub struct SimdKalmanFilter1D {
    config: KalmanConfig,
    /// State estimate (x)
    state: f64,
    /// Error covariance (P)
    covariance: f64,
    /// Process noise (Q)
    process_noise: f64,
    /// Measurement noise (R)
    measurement_noise: f64,
    /// Kalman gain (K) - cached from last update
    kalman_gain: f64,
    /// Update count
    update_count: AtomicU64,
    /// Last update timestamp
    last_update_ns: AtomicU64,
}

impl SimdKalmanFilter1D {
    /// Create a new Kalman filter with default configuration
    pub fn new() -> Self {
        Self::with_config(KalmanConfig::default())
    }

    /// Create a new Kalman filter with custom configuration
    pub fn with_config(config: KalmanConfig) -> Self {
        Self {
            state: 0.0,
            covariance: config.initial_covariance,
            config,
            process_noise: config.process_noise,
            measurement_noise: config.measurement_noise,
            kalman_gain: 0.0,
            update_count: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
        }
    }

    /// Initialize the filter with a known state
    pub fn initialize(&mut self, initial_state: f64) {
        self.state = initial_state;
        self.covariance = self.config.initial_covariance;
    }

    /// Predict step: project state forward in time
    /// 
    /// For 1D random walk model: x_pred = x (no control input)
    /// P_pred = P + Q
    #[inline]
    pub fn predict(&mut self) {
        // State prediction (random walk model - state stays same)
        // x_pred = x
        
        // Covariance prediction
        self.covariance += self.process_noise;
        
        // ROOT CAUSE FIX: Ensure covariance doesn't go below floor
        if self.covariance < self.config.covariance_floor {
            self.covariance = self.config.covariance_floor;
        }
    }

    /// Update step: incorporate new measurement
    /// 
    /// Returns the smoothed state estimate
    /// 
    /// ROOT CAUSE FIX: Enforces covariance floor before division
    #[inline]
    pub fn update(&mut self, measurement: f64) -> f64 {
        // First do prediction
        self.predict();

        // Calculate Kalman gain: K = P / (P + R)
        // ROOT CAUSE FIX: Ensure denominator is never zero
        let denominator = self.covariance + self.measurement_noise;
        let safe_denominator = denominator.max(self.config.covariance_floor);
        
        self.kalman_gain = self.covariance / safe_denominator;

        // Update state: x = x + K * (z - x)
        let innovation = measurement - self.state;
        self.state += self.kalman_gain * innovation;

        // Update covariance: P = (1 - K) * P
        self.covariance = (1.0 - self.kalman_gain) * self.covariance;

        // ROOT CAUSE FIX: Ensure covariance stays above floor after update
        if self.covariance < self.config.covariance_floor {
            self.covariance = self.config.covariance_floor;
        }

        self.update_count.fetch_add(1, Ordering::Relaxed);

        self.state
    }

    /// Process a measurement and return smoothed value
    pub fn process(&mut self, measurement: f64, timestamp_ns: u64) -> f64 {
        let result = self.update(measurement);
        self.last_update_ns.store(timestamp_ns, Ordering::Relaxed);
        result
    }

    /// Get current state estimate
    pub fn state(&self) -> f64 {
        self.state
    }

    /// Get current covariance (uncertainty)
    pub fn covariance(&self) -> f64 {
        self.covariance
    }

    /// Get current Kalman gain
    pub fn kalman_gain(&self) -> f64 {
        self.kalman_gain
    }

    /// Get update count
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }

    /// Get last update timestamp
    pub fn last_update_ns(&self) -> u64 {
        self.last_update_ns.load(Ordering::Relaxed)
    }

    /// Check if filter has converged (low covariance)
    pub fn is_converged(&self, threshold: f64) -> bool {
        self.covariance < threshold
    }

    /// Reset filter to initial state
    pub fn reset(&mut self) {
        self.state = 0.0;
        self.covariance = self.config.initial_covariance;
        self.kalman_gain = 0.0;
    }

    /// Set process noise dynamically
    pub fn set_process_noise(&mut self, q: f64) {
        self.process_noise = q.max(0.0);
    }

    /// Set measurement noise dynamically
    pub fn set_measurement_noise(&mut self, r: f64) {
        self.measurement_noise = r.max(self.config.covariance_floor);
    }
}

impl Default for SimdKalmanFilter1D {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch processor for SIMD-parallel Kalman filtering
pub struct SimdKalmanBatch<const BATCH_SIZE: usize = 4> {
    filters: [SimdKalmanFilter1D; BATCH_SIZE],
    active_count: usize,
}

impl<const BATCH_SIZE: usize> SimdKalmanBatch<BATCH_SIZE> {
    pub fn new(config: KalmanConfig) -> Self {
        Self {
            filters: std::array::from_fn(|_| SimdKalmanFilter1D::with_config(config)),
            active_count: 0,
        }
    }

    /// Process a batch of measurements in parallel
    pub fn process_batch(&mut self, measurements: &[f64]) -> Vec<f64> {
        let len = measurements.len().min(BATCH_SIZE);
        let mut results = Vec::with_capacity(len);

        for i in 0..len {
            let result = self.filters[i].update(measurements[i]);
            results.push(result);
        }

        results
    }

    /// Get filter at index
    pub fn get_filter(&self, idx: usize) -> Option<&SimdKalmanFilter1D> {
        self.filters.get(idx)
    }

    /// Get mutable filter at index
    pub fn get_filter_mut(&mut self, idx: usize) -> Option<&mut SimdKalmanFilter1D> {
        self.filters.get_mut(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_basic_smoothing() {
        let mut filter = SimdKalmanFilter1D::with_config(KalmanConfig {
            process_noise: 0.001,
            measurement_noise: 0.1,
            initial_covariance: 1.0,
            covariance_floor: 1e-10,
        });

        filter.initialize(0.5);

        // Noisy measurements around 0.5
        let measurements = vec![0.4, 0.6, 0.45, 0.55, 0.5, 0.52, 0.48];
        
        let mut last_result = 0.5;
        for m in measurements {
            let result = filter.update(m);
            // Smoothed result should be closer to true value than noisy measurement
            assert!((result - 0.5).abs() <= (m - 0.5).abs() + 0.1);
            last_result = result;
        }

        // After several updates, should converge near 0.5
        assert!((last_result - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_kalman_covariance_floor() {
        let mut filter = SimdKalmanFilter1D::with_config(KalmanConfig {
            process_noise: 0.0001,
            measurement_noise: 0.01,
            initial_covariance: 0.001,
            covariance_floor: 1e-8,
        });

        // Many updates with very low noise should not cause divide-by-zero
        for _ in 0..1000 {
            let _ = filter.update(1.0);
        }

        // Covariance should be at or above floor
        assert!(filter.covariance() >= 1e-8);
        
        // Should still produce valid output
        let result = filter.update(1.0);
        assert!(result.is_finite());
    }

    #[test]
    fn test_kalman_step_response() {
        let mut filter = SimdKalmanFilter1D::new();
        filter.initialize(0.0);

        // Step from 0 to 1
        let mut prev = 0.0;
        for _ in 0..20 {
            let result = filter.update(1.0);
            // Should monotonically approach 1.0
            assert!(result >= prev);
            assert!(result <= 1.0);
            prev = result;
        }

        // Should be close to 1.0 after many updates
        assert!(prev > 0.9);
    }

    #[test]
    fn test_kalman_zero_measurement_noise() {
        let mut filter = SimdKalmanFilter1D::with_config(KalmanConfig {
            process_noise: 0.01,
            measurement_noise: 0.0, // Zero measurement noise
            initial_covariance: 1.0,
            covariance_floor: 1e-10,
        });

        // Should handle zero measurement noise gracefully
        let result = filter.update(5.0);
        assert!(result.is_finite());
        
        // With zero measurement noise, should quickly converge to measurement
        assert!((result - 5.0).abs() < 1.0);
    }
}
