//! Filters module for Signal Processing
//! 
//! Provides SIMD-accelerated Kalman filters and noise estimation for
//! smoothing alpha signals before routing to execution.

pub mod simd_kalman_filter;
pub mod noise_variance_estimator;

pub use simd_kalman_filter::{SimdKalmanFilter1D, SimdKalmanBatch, KalmanConfig};
pub use noise_variance_estimator::{
    OnlineVarianceEstimator, AdaptiveNoiseEstimator,
};

/// Signal smoother that combines Kalman filtering with signal routing
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use tracing::debug;

/// Configuration for signal smoothing pipeline
#[derive(Debug, Clone, Copy)]
pub struct SignalSmootherConfig {
    /// Enable Kalman filtering
    pub enable_kalman: bool,
    /// Enable adaptive noise estimation
    pub enable_adaptive: bool,
    /// Minimum signal magnitude to pass through
    pub signal_threshold: f64,
    /// Maximum rate of change (prevents whipsaw)
    pub max_delta_per_update: f64,
}

impl Default for SignalSmootherConfig {
    fn default() -> Self {
        Self {
            enable_kalman: true,
            enable_adaptive: false,
            signal_threshold: 0.01,
            max_delta_per_update: 0.1,
        }
    }
}

/// Unified signal smoother for OBI/VPIN signals
pub struct SignalSmoother {
    config: SignalSmootherConfig,
    kalman_filter: SimdKalmanFilter1D,
    noise_estimator: Option<AdaptiveNoiseEstimator>,
    last_smoothed: f64,
    last_raw: f64,
    update_count: AtomicU64,
    is_active: AtomicBool,
}

impl SignalSmoother {
    pub fn new() -> Self {
        Self::with_config(SignalSmootherConfig::default())
    }

    pub fn with_config(config: SignalSmootherConfig) -> Self {
        let kalman_config = if config.enable_adaptive {
            KalmanConfig::default()
        } else {
            // More aggressive smoothing for fixed params
            KalmanConfig {
                process_noise: 0.0005,
                measurement_noise: 0.05,
                initial_covariance: 0.5,
                covariance_floor: 1e-10,
            }
        };

        Self {
            config,
            kalman_filter: SimdKalmanFilter1D::with_config(kalman_config),
            noise_estimator: if config.enable_adaptive {
                Some(AdaptiveNoiseEstimator::new())
            } else {
                None
            },
            last_smoothed: 0.0,
            last_raw: 0.0,
            update_count: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
        }
    }

    /// Process a raw signal and return smoothed value
    pub fn process(&mut self, raw_signal: f64, timestamp_ns: u64) -> f64 {
        if !self.is_active.load(Ordering::Acquire) {
            return self.last_smoothed;
        }

        self.update_count.fetch_add(1, Ordering::Relaxed);

        // Apply signal threshold
        if raw_signal.abs() < self.config.signal_threshold {
            self.last_raw = raw_signal;
            return self.last_smoothed;
        }

        // Apply Kalman filter if enabled
        let kalman_output = if self.config.enable_kalman {
            // Update adaptive estimator if enabled
            if let Some(ref mut estimator) = self.noise_estimator {
                estimator.update(raw_signal, self.kalman_filter.state());
                
                // Dynamically update Kalman params
                let new_config = estimator.get_recommended_config();
                self.kalman_filter.set_process_noise(new_config.process_noise);
                self.kalman_filter.set_measurement_noise(new_config.measurement_noise);
            }

            self.kalman_filter.process(raw_signal, timestamp_ns)
        } else {
            raw_signal
        };

        // Rate limiting: prevent excessive delta
        let delta = kalman_output - self.last_smoothed;
        let limited_delta = delta.clamp(
            -self.config.max_delta_per_update,
            self.config.max_delta_per_update,
        );

        self.last_smoothed += limited_delta;
        self.last_raw = raw_signal;

        self.last_smoothed
    }

    /// Get current smoothed value
    pub fn smoothed(&self) -> f64 {
        self.last_smoothed
    }

    /// Get last raw input
    pub fn last_raw(&self) -> f64 {
        self.last_raw
    }

    /// Get smoothing delta (raw - smoothed)
    pub fn smoothing_delta(&self) -> f64 {
        self.last_raw - self.last_smoothed
    }

    /// Check if smoother is active
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }

    /// Activate/deactivate smoother
    pub fn set_active(&self, active: bool) {
        self.is_active.store(active, Ordering::Release);
    }

    /// Reset smoother state
    pub fn reset(&mut self) {
        self.kalman_filter.reset();
        if let Some(ref mut estimator) = self.noise_estimator {
            estimator.reset();
        }
        self.last_smoothed = 0.0;
        self.last_raw = 0.0;
    }

    /// Get update count
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }

    /// Get Kalman filter reference
    pub fn kalman_filter(&self) -> &SimdKalmanFilter1D {
        &self.kalman_filter
    }

    /// Get mutable Kalman filter reference
    pub fn kalman_filter_mut(&mut self) -> &mut SimdKalmanFilter1D {
        &mut self.kalman_filter
    }
}

impl Default for SignalSmoother {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoother_basic() {
        let mut smoother = SignalSmoother::new();
        
        // Noisy signal around 0.5
        for i in 0..100 {
            let raw = 0.5 + (i as f64 * 0.1).sin() * 0.2;
            let _smoothed = smoother.process(raw, i as u64 * 1000);
        }

        // Smoothed should be close to mean
        assert!((smoother.smoothed() - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_smoother_threshold() {
        let mut smoother = SignalSmoother::with_config(SignalSmootherConfig {
            signal_threshold: 0.1,
            ..Default::default()
        });

        // Small signals should be ignored
        let initial = smoother.smoothed();
        let _ = smoother.process(0.05, 1000);
        let _ = smoother.process(-0.03, 2000);
        
        // Should remain at initial value
        assert_eq!(smoother.smoothed(), initial);

        // Large signal should pass through
        let _ = smoother.process(0.5, 3000);
        assert!(smoother.smoothed().abs() > 0.1);
    }

    #[test]
    fn test_smoother_rate_limit() {
        let mut smoother = SignalSmoother::with_config(SignalSmootherConfig {
            max_delta_per_update: 0.05,
            ..Default::default()
        });

        // Step input
        let _ = smoother.process(1.0, 1000);
        
        // Should not jump immediately to 1.0
        assert!(smoother.smoothed() < 0.5);
    }

    #[test]
    fn test_smoother_adaptive() {
        let mut smoother = SignalSmoother::with_config(SignalSmootherConfig {
            enable_adaptive: true,
            ..Default::default()
        });

        // Process some data
        for i in 0..50 {
            let raw = (i as f64 * 0.1).sin();
            let _ = smoother.process(raw, i as u64 * 1000);
        }

        // Adaptive estimator should have data now
        assert!(smoother.update_count() > 0);
    }
}
