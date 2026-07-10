//! Residual Alpha Calculator
//! 
//! Isolates idiosyncratic risk (spread between actual price and
//! PCA-reconstructed price) for mean-reversion trading.

use super::rolling_pca_engine::PCAResult;
use core::sync::atomic::{AtomicU64, Ordering};

/// Maximum assets supported
const MAX_ASSETS: usize = 128;

/// Result of residual calculation
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ResidualResult {
    /// Actual price
    pub actual_price: f64,
    /// PCA-reconstructed price
    pub reconstructed_price: f64,
    /// Residual (actual - reconstructed)
    pub residual: f64,
    /// Z-score of residual
    pub zscore: f64,
    /// Idiosyncratic variance
    pub idio_variance: f64,
}

impl Default for ResidualResult {
    #[inline]
    fn default() -> Self {
        Self {
            actual_price: 0.0,
            reconstructed_price: 0.0,
            residual: 0.0,
            zscore: 0.0,
            idio_variance: 0.0,
        }
    }
}

/// Residual Alpha Calculator
/// 
/// Computes the idiosyncratic component of asset returns after
/// removing exposure to principal factors.
pub struct ResidualAlphaCalculator {
    /// Factor loadings for each asset (asset x factor)
    loadings: [[f64; MAX_ASSETS]; 10], // Max 10 factors
    /// Residual values for each asset
    residuals: [f64; MAX_ASSETS],
    /// Running mean of residuals (for z-score)
    residual_means: [f64; MAX_ASSETS],
    /// Running M2 of residuals (Welford's algorithm)
    residual_m2: [f64; MAX_ASSETS],
    /// Number of assets
    n_assets: usize,
    /// Number of factors used
    n_factors: usize,
    /// Observation count
    count: u64,
    /// Update counter
    update_count: AtomicU64,
}

impl ResidualAlphaCalculator {
    /// Create a new residual calculator
    #[inline]
    pub fn new(n_assets: usize, n_factors: usize) -> Option<Self> {
        if n_assets == 0 || n_assets > MAX_ASSETS || n_factors == 0 || n_factors > 10 {
            return None;
        }

        Some(Self {
            loadings: [[0.0; MAX_ASSETS]; 10],
            residuals: [0.0; MAX_ASSETS],
            residual_means: [0.0; MAX_ASSETS],
            residual_m2: [0.0; MAX_ASSETS],
            n_assets,
            n_factors,
            count: 0,
            update_count: AtomicU64::new(0),
        })
    }

    /// Update factor loadings from PCA result
    /// 
    /// # Arguments
    /// * `pca_result` - PCA decomposition results
    /// * `n_factors` - Number of factors to use
    #[inline]
    pub fn update_loadings(&mut self, pca_result: &PCAResult, n_factors: usize) -> bool {
        let n_factors = n_factors.min(self.n_factors).min(pca_result.num_components);
        
        if n_factors == 0 {
            return false;
        }

        // Loadings are the eigenvectors (each row is a factor's loadings across assets)
        for k in 0..n_factors {
            for i in 0..self.n_assets {
                self.loadings[k][i] = pca_result.eigenvectors[k][i];
            }
        }

        self.n_factors = n_factors;
        true
    }

    /// Compute residual for a specific asset given factor returns
    /// 
    /// # Arguments
    /// * `asset_idx` - Index of the asset
    /// * `actual_return` - Actual return of the asset
    /// * `factor_returns` - Returns of each factor
    /// 
    /// Returns the residual calculation result
    #[inline]
    pub fn compute_residual(
        &mut self,
        asset_idx: usize,
        actual_return: f64,
        factor_returns: &[f64],
    ) -> Option<ResidualResult> {
        if asset_idx >= self.n_assets || factor_returns.len() < self.n_factors {
            return None;
        }

        if !actual_return.is_finite() {
            return None;
        }

        for &fr in factor_returns.iter().take(self.n_factors) {
            if !fr.is_finite() {
                return None;
            }
        }

        // Reconstruct expected return from factors
        let mut reconstructed = 0.0;
        for k in 0..self.n_factors {
            reconstructed += self.loadings[k][asset_idx] * factor_returns[k];
        }

        // Compute residual
        let residual = actual_return - reconstructed;
        self.residuals[asset_idx] = residual;

        // Update running statistics using Welford's algorithm
        self.count += 1;
        let n = self.count as f64;
        
        let delta = residual - self.residual_means[asset_idx];
        self.residual_means[asset_idx] += delta / n;
        
        let delta2 = residual - self.residual_means[asset_idx];
        self.residual_m2[asset_idx] += delta * delta2;

        // Compute z-score
        let zscore = if self.count > 1 {
            let variance = self.residual_m2[asset_idx] / (self.count - 1) as f64;
            let std = variance.sqrt();
            if std > 1e-15 {
                (residual - self.residual_means[asset_idx]) / std
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Idiosyncratic variance
        let idio_variance = if self.count > 1 {
            self.residual_m2[asset_idx] / (self.count - 1) as f64
        } else {
            0.0
        };

        Some(ResidualResult {
            actual_price: actual_return,
            reconstructed_price: reconstructed,
            residual,
            zscore,
            idio_variance,
        })
    }

    /// Get the current residual for an asset
    #[inline]
    pub fn get_residual(&self, asset_idx: usize) -> Option<f64> {
        if asset_idx < self.n_assets {
            Some(self.residuals[asset_idx])
        } else {
            None
        }
    }

    /// Get the z-score for an asset's residual
    #[inline]
    pub fn get_zscore(&self, asset_idx: usize) -> Option<f64> {
        if asset_idx >= self.n_assets || self.count <= 1 {
            return None;
        }

        let variance = self.residual_m2[asset_idx] / (self.count - 1) as f64;
        let std = variance.sqrt();
        
        if std > 1e-15 {
            Some((self.residuals[asset_idx] - self.residual_means[asset_idx]) / std)
        } else {
            Some(0.0)
        }
    }

    /// Get the number of observations processed
    #[inline]
    pub fn observation_count(&self) -> u64 {
        self.count
    }

    /// Get the update count
    #[inline]
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }

    /// Reset all statistics
    #[inline]
    pub fn reset(&mut self) {
        self.residuals = [0.0; MAX_ASSETS];
        self.residual_means = [0.0; MAX_ASSETS];
        self.residual_m2 = [0.0; MAX_ASSETS];
        self.count = 0;
        self.update_count.store(0, Ordering::Relaxed);
    }

    /// Check if residuals show mean-reverting behavior
    /// (variance of residuals < variance of original)
    #[inline]
    pub fn is_mean_reverting(&self, asset_idx: usize, total_variance: f64) -> bool {
        if asset_idx >= self.n_assets || self.count <= 1 {
            return false;
        }

        let idio_var = self.residual_m2[asset_idx] / (self.count - 1) as f64;
        
        // Idiosyncratic variance should be less than total variance
        // if factors explain meaningful variation
        idio_var < total_variance * 0.9
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual_computation() {
        let mut calc = ResidualAlphaCalculator::new(3, 1).unwrap();
        
        // Set up simple loading: asset 0 has 0.8 loading on factor 0
        calc.loadings[0][0] = 0.8;
        calc.loadings[0][1] = 0.6;
        calc.loadings[0][2] = 0.5;

        // Factor return is 1.0, actual return is 1.5
        // Expected: residual = 1.5 - 0.8*1.0 = 0.7
        let result = calc.compute_residual(0, 1.5, &[1.0]).unwrap();
        
        assert!((result.residual - 0.7).abs() < 1e-10);
        assert!((result.reconstructed_price - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_zscore_evolution() {
        let mut calc = ResidualAlphaCalculator::new(2, 1).unwrap();
        calc.loadings[0][0] = 1.0;

        // Add consistent residuals
        for i in 0..100 {
            let residual = (i % 10) as f64 - 5.0; // Oscillates around 0
            let _ = calc.compute_residual(0, residual + 1.0, &[1.0]);
        }

        let zscore = calc.get_zscore(0).unwrap();
        
        // After many observations, z-score should reflect deviation from mean
        assert!(zscore.is_finite());
    }

    #[test]
    fn test_invalid_inputs() {
        let mut calc = ResidualAlphaCalculator::new(3, 2).unwrap();
        
        // Invalid asset index
        assert!(calc.compute_residual(5, 1.0, &[1.0, 2.0]).is_none());
        
        // Not enough factor returns
        assert!(calc.compute_residual(0, 1.0, &[1.0]).is_none());
        
        // NaN input
        assert!(calc.compute_residual(0, f64::NAN, &[1.0, 2.0]).is_none());
    }

    #[test]
    fn test_reset() {
        let mut calc = ResidualAlphaCalculator::new(2, 1).unwrap();
        calc.loadings[0][0] = 1.0;
        
        // Add some data
        for _ in 0..50 {
            let _ = calc.compute_residual(0, 1.0, &[1.0]);
        }
        
        assert!(calc.observation_count() > 0);
        
        calc.reset();
        
        assert_eq!(calc.observation_count(), 0);
        assert_eq!(calc.get_residual(0), Some(0.0));
    }
}
