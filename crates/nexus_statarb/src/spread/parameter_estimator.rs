//! OU Parameter Estimator - Maximum Likelihood on Rolling Windows
//! 
//! Provides specialized MLE estimation for OU process parameters
//! with numerically stable algorithms.

use super::ou_process_modeler::{OUParameters, MAX_WINDOW};
use core::sync::atomic::{AtomicU64, Ordering};

/// Specialized parameter estimator using Welford's algorithm
/// for numerical stability in rolling window calculations
pub struct OUParameterEstimator {
    /// Running mean of spread
    mean_spread: f64,
    /// Running M2 for variance (Welford's algorithm)
    m2_spread: f64,
    /// Running mean of diff_spread
    mean_diff: f64,
    /// Running M2 for diff variance
    m2_diff: f64,
    /// Running covariance between spread and diff
    cov_spread_diff: f64,
    /// Count of observations
    count: usize,
    /// Circular buffer for removing old values
    spread_buffer: [f64; MAX_WINDOW],
    diff_buffer: [f64; MAX_WINDOW],
    /// Head position in circular buffer
    head: usize,
    /// Whether buffer is full (sliding window mode)
    is_full: bool,
    /// Update counter
    update_count: AtomicU64,
}

impl OUParameterEstimator {
    #[inline]
    pub fn new() -> Self {
        Self {
            mean_spread: 0.0,
            m2_spread: 0.0,
            mean_diff: 0.0,
            m2_diff: 0.0,
            cov_spread_diff: 0.0,
            count: 0,
            spread_buffer: [0.0; MAX_WINDOW],
            diff_buffer: [0.0; MAX_WINDOW],
            head: 0,
            is_full: false,
            update_count: AtomicU64::new(0),
        }
    }

    /// Update with new spread and difference values
    #[inline]
    pub fn update(&mut self, spread: f64, diff_spread: f64) {
        if !spread.is_finite() || !diff_spread.is_finite() {
            return;
        }

        let n = self.count as f64;
        
        if self.is_full {
            // Remove oldest value from statistics (sliding window)
            let old_spread = self.spread_buffer[self.head];
            let old_diff = self.diff_buffer[self.head];
            
            // Welford's algorithm reversal for removal
            let old_mean_spread = self.mean_spread;
            let old_mean_diff = self.mean_diff;
            
            self.mean_spread = (n * self.mean_spread - old_spread) / (n - 1.0);
            self.mean_diff = (n * self.mean_diff - old_diff) / (n - 1.0);
            
            // Update M2 and covariance (removal)
            let delta_old_spread = old_spread - old_mean_spread;
            let delta_new_spread = old_spread - self.mean_spread;
            let delta_old_diff = old_diff - old_mean_diff;
            let delta_new_diff = old_diff - self.mean_diff;
            
            self.m2_spread -= delta_old_spread * delta_new_spread;
            self.m2_diff -= delta_old_diff * delta_new_diff;
            self.cov_spread_diff -= (delta_old_spread * (old_diff - self.mean_diff) 
                                   + delta_new_diff * (old_spread - old_mean_spread)) * 0.5;
            
            self.m2_spread = self.m2_spread.max(0.0);
            self.m2_diff = self.m2_diff.max(0.0);
        } else {
            self.count += 1;
        }

        // Store values in circular buffer
        self.spread_buffer[self.head] = spread;
        self.diff_buffer[self.head] = diff_spread;
        self.head = (self.head + 1) % MAX_WINDOW;
        
        if self.count >= MAX_WINDOW {
            self.is_full = true;
        }

        // Welford's online algorithm for adding new value
        let delta_spread = spread - self.mean_spread;
        let delta_diff = diff_spread - self.mean_diff;
        
        let new_n = if self.is_full { n } else { self.count as f64 };
        
        self.mean_spread += delta_spread / new_n;
        self.mean_diff += delta_diff / new_n;
        
        if new_n > 1.0 {
            self.m2_spread += delta_spread * (spread - self.mean_spread);
            self.m2_diff += delta_diff * (diff_spread - self.mean_diff);
            self.cov_spread_diff += delta_spread * (diff_spread - self.mean_diff);
        }

        self.update_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Estimate OU parameters from current statistics
    #[inline]
    pub fn estimate(&self) -> Option<OUParameters> {
        let n = if self.is_full { MAX_WINDOW } else { self.count };
        
        if n < 3 {
            return None;
        }

        let n_f64 = n as f64;
        let variance_spread = self.m2_spread / (n_f64 - 1.0);
        let variance_diff = self.m2_diff / (n_f64 - 1.0);
        
        if variance_spread < 1e-15 {
            return None;
        }

        // Regression coefficient: beta = cov / var
        let beta = self.cov_spread_diff / self.m2_spread;
        
        // OU parameters
        let theta = -beta;
        
        if theta <= 0.0 {
            return Some(OUParameters::new(0.0, self.mean_spread, variance_diff.sqrt()));
        }

        // For OU: E[dX] = theta * (mu - X) * dt
        // Mean of dX = theta * (mu - mean_X)
        // mu = mean_X + mean_dX / theta
        let mu = self.mean_spread + self.mean_diff / theta;
        let sigma = variance_diff.sqrt();

        Some(OUParameters::new(theta, mu, sigma))
    }

    #[inline]
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn reset(&mut self) {
        self.mean_spread = 0.0;
        self.m2_spread = 0.0;
        self.mean_diff = 0.0;
        self.m2_diff = 0.0;
        self.cov_spread_diff = 0.0;
        self.count = 0;
        self.head = 0;
        self.is_full = false;
        self.update_count.store(0, Ordering::Relaxed);
    }
}

impl Default for OUParameterEstimator {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welford_stability() {
        let mut estimator = OUParameterEstimator::new();
        
        // Add many identical values - should have zero variance
        for _ in 0..100 {
            estimator.update(5.0, 0.0);
        }
        
        if let Some(params) = estimator.estimate() {
            assert!(params.sigma < 1e-10, "Variance should be near zero");
        }
    }

    #[test]
    fn test_sliding_window() {
        let mut estimator = OUParameterEstimator::new();
        
        // Fill the window
        for i in 0..MAX_WINDOW {
            estimator.update(i as f64, 1.0);
        }
        
        // Add more - should slide
        for i in 0..100 {
            estimator.update((MAX_WINDOW + i) as f64, 1.0);
        }
        
        // Mean should reflect recent values, not all-time
        if let Some(params) = estimator.estimate() {
            assert!(params.mu > MAX_WINDOW as f64, "Mean should reflect sliding window");
        }
    }
}
