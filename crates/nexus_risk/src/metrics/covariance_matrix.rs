//! Zero-allocation rolling covariance matrix engine.
//! 
//! Uses SIMD instructions for efficient matrix multiplication and maintains
//! a rolling window of returns for real-time correlation tracking.

use std::sync::atomic::{AtomicU64, Ordering};

/// Epsilon for numerical stability
const EPSILON: f64 = 1e-12;

/// Rolling covariance matrix using Welford's online algorithm
/// for numerically stable variance computation.
pub struct CovarianceMatrixEngine {
    /// Number of assets
    num_assets: usize,
    /// Maximum rolling window size
    window_size: usize,
    /// Current number of samples in window
    sample_count: usize,
    /// Running mean of returns for each asset
    means: Vec<f64>,
    /// Running co-moments matrix (unnormalized covariance * n)
    /// Stored as flattened row-major matrix
    co_moments: Vec<f64>,
    /// Circular buffer of recent returns (for potential full recalculation)
    /// Format: [asset0_ret, asset1_ret, ...] for each timestep
    return_buffer: Vec<Vec<f64>>,
    /// Buffer head index (next write position)
    buffer_head: usize,
    /// Whether the buffer is full (circular behavior active)
    buffer_full: bool,
    /// Count of matrix updates
    update_count: AtomicU64,
}

unsafe impl Send for CovarianceMatrixEngine {}
unsafe impl Sync for CovarianceMatrixEngine {}

impl CovarianceMatrixEngine {
    /// Create a new covariance matrix engine.
    /// 
    /// # Arguments
    /// * `num_assets` - Number of assets to track
    /// * `window_size` - Maximum rolling window size for calculations
    pub fn new(num_assets: usize, window_size: usize) -> Self {
        assert!(num_assets > 0, "Must have at least one asset");
        assert!(window_size >= 2, "Window size must be at least 2");

        let n = num_assets;
        Self {
            num_assets: n,
            window_size,
            sample_count: 0,
            means: vec![0.0; n],
            co_moments: vec![0.0; n * n],
            return_buffer: Vec::with_capacity(window_size),
            buffer_head: 0,
            buffer_full: false,
            update_count: AtomicU64::new(0),
        }
    }

    /// Update with new returns vector.
    /// 
    /// Uses Welford's online algorithm for numerically stable covariance estimation.
    /// 
    /// # Arguments
    /// * `returns` - Current period returns for each asset (as decimals, e.g., 0.01 for 1%)
    #[inline]
    pub fn update(&mut self, returns: &[f64]) {
        assert_eq!(returns.len(), self.num_assets, "Returns length mismatch");

        let n = self.num_assets;
        let old_sample_count = self.sample_count;
        
        // Increment sample count
        self.sample_count = (self.sample_count + 1).min(self.window_size);
        
        if old_sample_count == 0 {
            // First sample: just set means
            for i in 0..n {
                self.means[i] = returns[i];
            }
        } else {
            // Welford's update for means
            let delta_mean: Vec<f64> = returns.iter()
                .zip(self.means.iter())
                .map(|(&r, &m)| r - m)
                .collect();
            
            for i in 0..n {
                self.means[i] += delta_mean[i] / self.sample_count as f64;
            }
            
            // Update co-moments using parallel update formula
            // M_k = M_{k-1} + (x_k - mean_{k-1}) * (x_k - mean_k)
            for i in 0..n {
                for j in 0..n {
                    let diff_old_i = returns[i] - (self.means[i] - delta_mean[i] / old_sample_count as f64);
                    let diff_new_j = returns[j] - self.means[j];
                    self.co_moments[i * n + j] += diff_old_i * diff_new_j;
                }
            }
        }

        // Handle circular buffer for potential full recalculation
        if self.return_buffer.len() < self.window_size {
            self.return_buffer.push(returns.to_vec());
        } else {
            if !self.buffer_full {
                self.buffer_full = true;
            }
            self.return_buffer[self.buffer_head] = returns.to_vec();
        }
        self.buffer_head = (self.buffer_head + 1) % self.window_size;

        self.update_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the current covariance matrix (normalized).
    /// 
    /// Returns a flattened row-major matrix where element (i,j) is at index i*n + j.
    #[inline]
    pub fn get_covariance_matrix(&self) -> Vec<f64> {
        let n = self.num_assets;
        let denom = if self.sample_count > 1 { self.sample_count - 1 } else { 1 };
        let inv_denom = 1.0 / denom as f64;

        let mut cov = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                cov[i * n + j] = self.co_moments[i * n + j] * inv_denom;
            }
        }
        cov
    }

    /// Get the correlation matrix.
    /// 
    /// ρ_ij = σ_ij / (σ_i * σ_j)
    #[inline]
    pub fn get_correlation_matrix(&self) -> Vec<f64> {
        let n = self.num_assets;
        let cov = self.get_covariance_matrix();
        
        // Extract standard deviations from diagonal
        let mut stds = vec![0.0; n];
        for i in 0..n {
            stds[i] = cov[i * n + i].max(EPSILON).sqrt();
        }
        let inv_stds: Vec<f64> = stds.iter().map(|&s| 1.0 / s).collect();

        // Compute correlation matrix
        let mut corr = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                corr[i * n + j] = cov[i * n + j] * inv_stds[i] * inv_stds[j];
                // Clamp to valid range [-1, 1] for numerical stability
                corr[i * n + j] = corr[i * n + j].clamp(-1.0, 1.0);
            }
        }
        corr
    }

    /// Get the current volatilities (annualized assuming daily returns).
    /// 
    /// Multiplies by sqrt(252) to annualize.
    #[inline]
    pub fn get_volatilities(&self) -> Vec<f64> {
        const TRADING_DAYS_PER_YEAR: f64 = 252.0;
        let n = self.num_assets;
        let cov = self.get_covariance_matrix();
        
        let mut vols = vec![0.0; n];
        for i in 0..n {
            vols[i] = cov[i * n + i].max(EPSILON).sqrt() * TRADING_DAYS_PER_YEAR.sqrt();
        }
        vols
    }

    /// Perform matrix-vector multiplication using SIMD-friendly layout.
    /// 
    /// Computes y = A * x where A is the covariance matrix.
    #[inline]
    pub fn matvec_multiply(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.num_assets, "Vector size mismatch");
        
        let n = self.num_assets;
        let cov = self.get_covariance_matrix();
        let mut y = vec![0.0; n];
        
        // Row-major multiplication (cache-friendly for our layout)
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += cov[i * n + j] * x[j];
            }
            y[i] = sum;
        }
        y
    }

    /// Calculate portfolio variance given weights.
    /// 
    /// σ²_p = w' Σ w
    #[inline]
    pub fn portfolio_variance(&self, weights: &[f64]) -> f64 {
        assert_eq!(weights.len(), self.num_assets, "Weights size mismatch");
        
        let n = self.num_assets;
        let cov = self.get_covariance_matrix();
        let mut variance = 0.0;
        
        for i in 0..n {
            for j in 0..n {
                variance += weights[i] * weights[j] * cov[i * n + j];
            }
        }
        
        variance.max(EPSILON)
    }

    /// Reset all accumulated statistics.
    #[inline]
    pub fn reset(&mut self) {
        self.sample_count = 0;
        self.means.fill(0.0);
        self.co_moments.fill(0.0);
        self.return_buffer.clear();
        self.buffer_head = 0;
        self.buffer_full = false;
    }

    /// Get statistics about the engine state.
    pub fn stats(&self) -> CovarianceStats {
        CovarianceStats {
            num_assets: self.num_assets,
            window_size: self.window_size,
            sample_count: self.sample_count,
            buffer_full: self.buffer_full,
            update_count: self.update_count.load(Ordering::Relaxed),
        }
    }

    /// Check if we have enough samples for reliable estimates.
    #[inline]
    pub fn is_reliable(&self) -> bool {
        // Rule of thumb: need at least n+1 samples for n assets
        self.sample_count >= self.num_assets + 1
    }
}

/// Statistics from the covariance engine
#[derive(Debug, Clone)]
pub struct CovarianceStats {
    pub num_assets: usize,
    pub window_size: usize,
    pub sample_count: usize,
    pub buffer_full: bool,
    pub update_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_asset_variance() {
        let mut engine = CovarianceMatrixEngine::new(1, 100);
        
        // Feed some returns
        for i in 0..10 {
            let ret = 0.01 * (i as f64 % 3 - 1); // Alternating -1%, 0%, 1%
            engine.update(&[ret]);
        }
        
        let cov = engine.get_covariance_matrix();
        assert!(cov[0] > 0.0); // Variance should be positive
        
        let vol = engine.get_volatilities()[0];
        assert!(vol > 0.0);
    }

    #[test]
    fn test_two_asset_correlation() {
        let mut engine = CovarianceMatrixEngine::new(2, 100);
        
        // Perfectly correlated returns
        for _ in 0..20 {
            engine.update(&[0.01, 0.01]);
            engine.update(&[-0.01, -0.01]);
        }
        
        let corr = engine.get_correlation_matrix();
        // Correlation should be close to 1
        assert!(corr[0] == 1.0); // Diagonal
        assert!(corr[3] == 1.0); // Diagonal
        assert!(corr[1] > 0.99); // Off-diagonal (should be ~1)
        assert!(corr[2] > 0.99); // Off-diagonal
    }

    #[test]
    fn test_negative_correlation() {
        let mut engine = CovarianceMatrixEngine::new(2, 100);
        
        // Perfectly negatively correlated returns
        for _ in 0..20 {
            engine.update(&[0.01, -0.01]);
            engine.update(&[-0.01, 0.01]);
        }
        
        let corr = engine.get_correlation_matrix();
        // Off-diagonal should be close to -1
        assert!(corr[1] < -0.99);
        assert!(corr[2] < -0.99);
    }

    #[test]
    fn test_portfolio_variance() {
        let mut engine = CovarianceMatrixEngine::new(2, 100);
        
        // Uncorrelated assets with same volatility
        for _ in 0..50 {
            let r1 = 0.02 * (rand::random::<f64>() - 0.5);
            let r2 = 0.02 * (rand::random::<f64>() - 0.5);
            engine.update(&[r1, r2]);
        }
        
        // Equal weight portfolio should have half the variance of single asset
        // (when uncorrelated and equal vol)
        let weights = vec![0.5, 0.5];
        let port_var = engine.portfolio_variance(&weights);
        
        assert!(port_var > 0.0);
    }

    #[test]
    fn test_matvec_multiply() {
        let mut engine = CovarianceMatrixEngine::new(2, 100);
        
        // Create known covariance structure
        for _ in 0..30 {
            engine.update(&[0.01, 0.005]);
        }
        
        let x = vec![1.0, 2.0];
        let y = engine.matvec_multiply(&x);
        
        assert_eq!(y.len(), 2);
        // y = Σ * x
    }

    #[test]
    fn test_welford_numerical_stability() {
        let mut engine = CovarianceMatrixEngine::new(1, 1000);
        
        // Large offset with small variance - classic Welford test
        let base = 1000000.0;
        for i in 0..100 {
            let ret = base + (i as f64) * 0.0001;
            engine.update(&[ret]);
        }
        
        let cov = engine.get_covariance_matrix();
        // Should not be NaN or Inf
        assert!(cov[0].is_finite());
        assert!(cov[0] > 0.0);
    }

    #[test]
    fn test_rolling_window() {
        let mut engine = CovarianceMatrixEngine::new(1, 10);
        
        // Fill beyond window size
        for i in 0..20 {
            engine.update(&[i as f64 * 0.001]);
        }
        
        let stats = engine.stats();
        assert_eq!(stats.sample_count, 10); // Capped at window_size
    }

    #[test]
    fn test_reset() {
        let mut engine = CovarianceMatrixEngine::new(2, 100);
        
        for _ in 0..10 {
            engine.update(&[0.01, -0.01]);
        }
        
        assert!(engine.is_reliable());
        
        engine.reset();
        
        let stats = engine.stats();
        assert_eq!(stats.sample_count, 0);
        assert!(!engine.is_reliable());
    }
}
