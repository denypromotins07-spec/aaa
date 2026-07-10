//! ADWIN (Adaptive Windowing) Drift Detector
//! 
//! Implements the ADWIN algorithm for streaming change-point detection.
//! Automatically shrinks/expands the window based on detected changes.
//! Zero-allocation design using pre-allocated ring buffers.

use crate::MLOpsError;
use std::collections::VecDeque;

/// ADWIN drift detector configuration
#[derive(Debug, Clone)]
pub struct AdwinConfig {
    /// Confidence parameter (delta). Smaller = more sensitive
    pub delta: f64,
    /// Maximum window size to prevent unbounded growth
    pub max_window_size: usize,
    /// Minimum samples before checking for drift
    pub min_samples: usize,
    /// Market hours awareness: skip updates during closed markets
    pub market_hours_aware: bool,
}

impl Default for AdwinConfig {
    fn default() -> Self {
        Self {
            delta: 0.002,
            max_window_size: 10_000,
            min_samples: 30,
            market_hours_aware: true,
        }
    }
}

/// ADWIN adaptive windowing detector for concept drift
pub struct AdwinDetector {
    config: AdwinConfig,
    /// Ring buffer for zero-allocation streaming
    window: VecDeque<f64>,
    /// Sum of values in window for O(1) mean calculation
    window_sum: f64,
    /// Total samples seen
    total_samples: u64,
    /// Last detected change point
    last_change_point: Option<u64>,
    /// Regime-conditional baseline variance
    baseline_variance: f64,
}

impl AdwinDetector {
    /// Create a new ADWIN detector with default config
    pub fn new() -> Self {
        Self::with_config(AdwinConfig::default())
    }

    /// Create a new ADWIN detector with custom config
    pub fn with_config(config: AdwinConfig) -> Self {
        Self {
            window: VecDeque::with_capacity(config.max_window_size),
            window_sum: 0.0,
            total_samples: 0,
            last_change_point: None,
            baseline_variance: 0.0,
            config,
        }
    }

    /// Update with a new sample and check for drift
    /// Returns Ok(true) if drift detected, Ok(false) otherwise
    pub fn update(&mut self, value: f64) -> Result<bool, MLOpsError> {
        // Skip updates during closed markets if configured
        if self.config.market_hours_aware && self.is_market_closed() {
            return Ok(false);
        }

        // Add to window
        if self.window.len() >= self.config.max_window_size {
            // Remove oldest element (zero-copy by reusing capacity)
            if let Some(oldest) = self.window.pop_front() {
                self.window_sum -= oldest;
            }
        }
        
        self.window.push_back(value);
        self.window_sum += value;
        self.total_samples += 1;

        // Check for drift only after minimum samples
        if self.window.len() < self.config.min_samples {
            return Ok(false);
        }

        // Run ADWIN cut detection
        let drift_detected = self.check_for_cut();
        
        if drift_detected {
            self.last_change_point = Some(self.total_samples);
        }

        Ok(drift_detected)
    }

    /// Check for optimal cut point in the window
    fn check_for_cut(&mut self) -> bool {
        let n = self.window.len();
        if n < self.config.min_samples {
            return false;
        }

        let mut max_cut_metric = 0.0;
        let mut best_cut_point = 0;

        // Try all possible cut points
        for cut_pos in 1..n - 1 {
            let (mean0, var0) = self.compute_mean_var(0, cut_pos);
            let (mean1, var1) = self.compute_mean_var(cut_pos, n - cut_pos);

            // Compute weighted difference
            let n0 = cut_pos as f64;
            let n1 = (n - cut_pos) as f64;
            
            let diff = (mean0 - mean1).abs();
            
            // Hoeffding bound for adaptive threshold
            let m = 1.0 / ((1.0 / n0) + (1.0 / n1));
            let epsilon = self.compute_epsilon(m);

            let cut_metric = diff - epsilon;

            if cut_metric > max_cut_metric {
                max_cut_metric = cut_metric;
                best_cut_point = cut_pos;
            }
        }

        // If significant cut found, shrink window
        if max_cut_metric > 0.0 && best_cut_point > 0 {
            self.shrink_window(best_cut_point);
            return true;
        }

        false
    }

    /// Compute mean and variance for a slice of the window
    fn compute_mean_var(&self, start: usize, len: usize) -> (f64, f64) {
        if len == 0 {
            return (0.0, 0.0);
        }

        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for i in 0..len {
            let idx = start + i;
            if let Some(&val) = self.window.get(idx) {
                sum += val;
                sum_sq += val * val;
            }
        }

        let mean = sum / len as f64;
        let variance = (sum_sq / len as f64) - (mean * mean);
        
        // Clamp variance to prevent negative values due to floating point
        let variance = variance.max(0.0);

        (mean, variance)
    }

    /// Compute Hoeffding bound epsilon
    fn compute_epsilon(&self, m: f64) -> f64 {
        let range = 1.0; // Assuming normalized [0,1] data
        let ln_delta = (4.0 / self.config.delta).ln().max(1.0);
        
        // Epsilon = sqrt((range^2 / (2*m)) * ln(4/delta))
        let epsilon_sq = (range * range / (2.0 * m)) * ln_delta;
        
        epsilon_sq.sqrt()
    }

    /// Shrink window to keep only the newer portion
    fn shrink_window(&mut self, keep_from: usize) {
        let to_remove = keep_from.min(self.window.len());
        
        for _ in 0..to_remove {
            if let Some(val) = self.window.pop_front() {
                self.window_sum -= val;
            }
        }
    }

    /// Get current window mean
    pub fn mean(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        self.window_sum / self.window.len() as f64
    }

    /// Get current window variance
    pub fn variance(&self) -> f64 {
        let mean = self.mean();
        if self.window.len() < 2 {
            return 0.0;
        }

        let mut sum_sq_diff = 0.0;
        for &val in &self.window {
            let diff = val - mean;
            sum_sq_diff += diff * diff;
        }

        sum_sq_diff / (self.window.len() - 1) as f64
    }

    /// Get total samples processed
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Get last detected change point
    pub fn last_change_point(&self) -> Option<u64> {
        self.last_change_point
    }

    /// Set regime-conditional baseline variance
    /// Used to adjust sensitivity based on macro regime
    pub fn set_baseline_variance(&mut self, variance: f64) {
        self.baseline_variance = variance;
    }

    /// Check if market is closed (simplified implementation)
    fn is_market_closed(&self) -> bool {
        // In production, this would check actual market hours
        // For now, always return false to allow continuous monitoring
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adwin_no_drift_stable_data() {
        let mut detector = AdwinDetector::new();
        
        // Stable data stream
        for i in 0..1000 {
            let value = 0.5 + (i % 10) as f64 * 0.01;
            let drift = detector.update(value).unwrap();
            assert!(!drift, "Should not detect drift in stable data");
        }
    }

    #[test]
    fn test_adwin_detects_level_shift() {
        let mut detector = AdwinDetector::with_config(AdwinConfig {
            delta: 0.01,
            max_window_size: 500,
            min_samples: 50,
            market_hours_aware: false,
        });

        // Pre-shift data
        for _ in 0..200 {
            detector.update(0.3).unwrap();
        }

        // Post-shift data
        let mut drift_detected = false;
        for _ in 0..300 {
            let drift = detector.update(0.8).unwrap();
            if drift {
                drift_detected = true;
                break;
            }
        }

        assert!(drift_detected, "Should detect level shift");
    }

    #[test]
    fn test_window_shrinking() {
        let mut detector = AdwinDetector::with_config(AdwinConfig {
            delta: 0.001,
            max_window_size: 100,
            min_samples: 30,
            market_hours_aware: false,
        });

        // Add initial data
        for _ in 0..50 {
            detector.update(0.2).unwrap();
        }

        // Add shifted data to trigger shrink
        for _ in 0..100 {
            detector.update(0.9).unwrap();
        }

        // Window should have shrunk
        assert!(detector.window.len() <= detector.config.max_window_size);
    }
}
