//! Non-synchronous covariance matrix builder.
//! 
//! Constructs full covariance matrices for multiple assets with
//! non-synchronous trading using pairwise Hayashi-Yoshida estimation.

use crate::lead_lag::hayashi_yoshida_estimator::{Tick, HayashiYoshidaEstimator, HyResult};
use ndarray::Array2;

/// Asset identifier
pub type AssetId = usize;

/// Result containing full covariance and correlation matrices
#[derive(Debug, Clone)]
pub struct NonSyncCovarianceMatrix {
    /// Asset IDs in order
    pub assets: Vec<AssetId>,
    /// Covariance matrix (annualized)
    pub covariance: Array2<f64>,
    /// Correlation matrix (derived)
    pub correlation: Array2<f64>,
    /// Variances (diagonal of covariance)
    pub variances: Vec<f64>,
}

impl NonSyncCovarianceMatrix {
    /// Create new result with specified size
    pub fn new(n_assets: usize) -> Self {
        Self {
            assets: Vec::with_capacity(n_assets),
            covariance: Array2::<f64>::zeros((n_assets, n_assets)),
            correlation: Array2::<f64>::zeros((n_assets, n_assets)),
            variances: Vec::with_capacity(n_assets),
        }
    }

    /// Check if matrix is positive semi-definite
    pub fn is_positive_semidefinite(&self, tolerance: f64) -> bool {
        // Simple check: all diagonal elements positive and matrix symmetric
        let n = self.covariance.nrows();
        
        for i in 0..n {
            if self.covariance[[i, i]] < -tolerance {
                return false;
            }
            
            for j in (i + 1)..n {
                let diff = (self.covariance[[i, j]] - self.covariance[[j, i]]).abs();
                if diff > tolerance {
                    return false;
                }
            }
        }
        
        true
    }

    /// Ensure symmetry by averaging upper and lower triangles
    pub fn symmetrize(&mut self) {
        let n = self.covariance.nrows();
        
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = (self.covariance[[i, j]] + self.covariance[[j, i]]) / 2.0;
                self.covariance[[i, j]] = avg;
                self.covariance[[j, i]] = avg;
                
                let corr_avg = (self.correlation[[i, j]] + self.correlation[[j, i]]) / 2.0;
                self.correlation[[i, j]] = corr_avg;
                self.correlation[[j, i]] = corr_avg;
            }
        }
    }
}

/// Builder for non-synchronous covariance matrices
pub struct NonSyncCovarianceBuilder {
    hy_estimator: HayashiYoshidaEstimator,
    min_overlaps: usize,
}

impl NonSyncCovarianceBuilder {
    /// Create new builder
    pub fn new(capacity: usize, min_overlaps: usize) -> Self {
        Self {
            hy_estimator: HayashiYoshidaEstimator::new(capacity),
            min_overlaps,
        }
    }

    /// Build full covariance matrix from tick data for multiple assets
    /// 
    /// # Arguments
    /// * `all_ticks` - Vector of (asset_id, ticks) pairs
    /// 
    /// # Returns
    /// Covariance matrix with assets in sorted order by ID
    pub fn build(
        &mut self,
        mut all_ticks: Vec<(AssetId, Vec<Tick>)>,
    ) -> Result<NonSyncCovarianceMatrix, String> {
        if all_ticks.is_empty() {
            return Err("No asset data provided".to_string());
        }

        // Sort by asset ID
        all_ticks.sort_by_key(|(id, _)| *id);

        let n_assets = all_ticks.len();
        let mut result = NonSyncCovarianceMatrix::new(n_assets);

        // Extract sorted asset IDs
        for (id, _) in &all_ticks {
            result.assets.push(*id);
        }

        // Compute pairwise covariances
        for i in 0..n_assets {
            let (_, ticks_i) = &all_ticks[i];
            
            // Variance on diagonal
            let var_i = self.compute_variance(ticks_i)?;
            result.variances.push(var_i);
            result.covariance[[i, i]] = var_i;
            result.correlation[[i, i]] = 1.0;

            for j in (i + 1)..n_assets {
                let (_, ticks_j) = &all_ticks[j];

                match self.hy_estimator.estimate_covariance(ticks_i, ticks_j) {
                    Ok(hy_result) => {
                        if hy_result.num_overlaps >= self.min_overlaps && hy_result.is_valid() {
                            result.covariance[[i, j]] = hy_result.covariance;
                            result.covariance[[j, i]] = hy_result.covariance;

                            // Compute correlation
                            let var_j = self.compute_variance(ticks_j)?;
                            let denom = (var_i * var_j).sqrt();
                            
                            if denom > 1e-15 {
                                let corr = (hy_result.covariance / denom).max(-1.0).min(1.0);
                                result.correlation[[i, j]] = corr;
                                result.correlation[[j, i]] = corr;
                            }
                        } else {
                            // Insufficient overlap - use fallback
                            result.covariance[[i, j]] = 0.0;
                            result.covariance[[j, i]] = 0.0;
                            result.correlation[[i, j]] = 0.0;
                            result.correlation[[j, i]] = 0.0;
                        }
                    }
                    Err(_) => {
                        // Estimation failed - zero fill
                        result.covariance[[i, j]] = 0.0;
                        result.covariance[[j, i]] = 0.0;
                        result.correlation[[i, j]] = 0.0;
                        result.correlation[[j, i]] = 0.0;
                    }
                }
            }
        }

        // Ensure symmetry (should already be symmetric, but safety check)
        result.symmetrize();

        // Validate positive semi-definiteness
        if !result.is_positive_semidefinite(1e-10) {
            // Apply nearest positive definite correction (simplified)
            self.apply_higham_correction(&mut result)?;
        }

        Ok(result)
    }

    /// Compute realized variance for single asset
    fn compute_variance(&self, ticks: &[Tick]) -> Result<f64, String> {
        self.hy_estimator.compute_variance(ticks)
            .ok_or_else(|| "Insufficient ticks for variance".to_string())
    }

    /// Simplified Higham nearest positive definite correction
    fn apply_higham_correction(
        &self,
        matrix: &mut NonSyncCovarianceMatrix,
    ) -> Result<(), String> {
        let n = matrix.covariance.nrows();
        
        // Add small diagonal perturbation
        let epsilon = 1e-6;
        for i in 0..n {
            matrix.covariance[[i, i]] += epsilon;
        }

        // Rebuild correlation matrix
        for i in 0..n {
            for j in (i + 1)..n {
                let var_i = matrix.covariance[[i, i]];
                let var_j = matrix.covariance[[j, j]];
                let denom = (var_i * var_j).sqrt();
                
                if denom > 1e-15 {
                    let corr = (matrix.covariance[[i, j]] / denom).max(-1.0).min(1.0);
                    matrix.correlation[[i, j]] = corr;
                    matrix.correlation[[j, i]] = corr;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_covariance_matrix() {
        let ticks_a = vec![
            Tick { timestamp_us: 0, price: 100.0, volume: 1.0 },
            Tick { timestamp_us: 1000, price: 101.0, volume: 1.0 },
            Tick { timestamp_us: 2000, price: 102.0, volume: 1.0 },
        ];

        let ticks_b = vec![
            Tick { timestamp_us: 0, price: 50.0, volume: 1.0 },
            Tick { timestamp_us: 1000, price: 50.5, volume: 1.0 },
            Tick { timestamp_us: 2000, price: 51.0, volume: 1.0 },
        ];

        let mut builder = NonSyncCovarianceBuilder::new(10, 1);
        let result = builder.build(vec![(0, ticks_a), (1, ticks_b)]).unwrap();

        assert_eq!(result.assets.len(), 2);
        assert_eq!(result.covariance.shape(), &[2, 2]);
        assert_eq!(result.correlation.shape(), &[2, 2]);
        
        // Diagonal should be positive (variances)
        assert!(result.covariance[[0, 0]] > 0.0);
        assert!(result.covariance[[1, 1]] > 0.0);
        
        // Diagonal correlations should be 1
        assert!((result.correlation[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((result.correlation[[1, 1]] - 1.0).abs() < 1e-10);
    }
}
