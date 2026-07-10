//! Rolling PCA Engine for Factor Extraction
//! 
//! Implements zero-allocation Principal Component Analysis
//! using power iteration and SIMD-accelerated Gram-Schmidt.

use super::simd_gram_schmidt::simd_gram_schmidt;
use core::sync::atomic::{AtomicU64, Ordering};

/// Maximum number of assets supported
const MAX_ASSETS: usize = 128;

/// Maximum number of principal components to extract
const MAX_COMPONENTS: usize = 10;

/// Result of PCA decomposition
#[repr(C)]
#[derive(Debug, Clone)]
pub struct PCAResult {
    /// Eigenvalues (variance explained by each component)
    pub eigenvalues: [f64; MAX_COMPONENTS],
    /// Eigenvectors (loadings for each component)
    pub eigenvectors: [[f64; MAX_ASSETS]; MAX_COMPONENTS],
    /// Number of valid components
    pub num_components: usize,
    /// Total variance in the data
    pub total_variance: f64,
}

impl PCAResult {
    #[inline]
    pub const fn new() -> Self {
        Self {
            eigenvalues: [0.0; MAX_COMPONENTS],
            eigenvectors: [[0.0; MAX_ASSETS]; MAX_COMPONENTS],
            num_components: 0,
            total_variance: 0.0,
        }
    }
}

impl Default for PCAResult {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Rolling PCA Engine using power iteration
/// 
/// Extracts principal components from a rolling covariance matrix
/// without heap allocations.
pub struct RollingPCAEngine {
    /// Pre-allocated data buffer (assets x time)
    data_buffer: [[f64; MAX_ASSETS]; MAX_ASSETS], // Simplified: square for covariance
    /// Current covariance matrix
    covariance: [[f64; MAX_ASSETS]; MAX_ASSETS],
    /// Number of assets
    n_assets: usize,
    /// Number of observations used
    n_obs: usize,
    /// Last PCA result
    last_result: PCAResult,
    /// Update counter
    update_count: AtomicU64,
    /// Whether covariance needs recomputation
    dirty: bool,
}

impl RollingPCAEngine {
    /// Create a new rolling PCA engine
    #[inline]
    pub fn new(n_assets: usize) -> Option<Self> {
        if n_assets == 0 || n_assets > MAX_ASSETS {
            return None;
        }

        Some(Self {
            data_buffer: [[0.0; MAX_ASSETS]; MAX_ASSETS],
            covariance: [[0.0; MAX_ASSETS]; MAX_ASSETS],
            n_assets,
            n_obs: 0,
            last_result: PCAResult::new(),
            update_count: AtomicU64::new(0),
            dirty: false,
        })
    }

    /// Add a new observation vector (prices/returns for all assets)
    /// 
    /// # Arguments
    /// * `observation` - Slice of returns for each asset
    /// * `decay_factor` - Exponential decay factor for rolling window (0 < lambda <= 1)
    #[inline]
    pub fn add_observation(&mut self, observation: &[f64], decay_factor: f64) -> bool {
        if observation.len() != self.n_assets {
            return false;
        }

        // Validate inputs
        for &v in observation.iter() {
            if !v.is_finite() {
                return false;
            }
        }

        let lambda = decay_factor.clamp(0.9, 1.0);

        // Update covariance using exponential weighted moving average
        // Cov_new = lambda * Cov_old + (1-lambda) * obs * obs^T
        
        if self.n_obs == 0 {
            // First observation - initialize covariance
            for i in 0..self.n_assets {
                for j in 0..self.n_assets {
                    self.covariance[i][j] = observation[i] * observation[j];
                }
            }
            self.n_obs = 1;
        } else {
            // EWMA update
            let one_minus_lambda = 1.0 - lambda;
            for i in 0..self.n_assets {
                for j in 0..self.n_assets {
                    self.covariance[i][j] = 
                        lambda * self.covariance[i][j] + 
                        one_minus_lambda * observation[i] * observation[j];
                }
            }
            self.n_obs += 1;
        }

        self.dirty = true;
        self.update_count.fetch_add(1, Ordering::Relaxed);
        
        true
    }

    /// Compute PCA using power iteration method
    /// 
    /// # Arguments
    /// * `num_components` - Number of principal components to extract
    /// * `max_iterations` - Maximum iterations per component
    /// * `tolerance` - Convergence tolerance
    #[inline]
    pub fn compute_pca(
        &mut self,
        num_components: usize,
        max_iterations: usize,
        tolerance: f64,
    ) -> Option<&PCAResult> {
        if !self.dirty && self.last_result.num_components > 0 {
            return Some(&self.last_result);
        }

        let num_components = num_components.min(MAX_COMPONENTS).min(self.n_assets);
        
        if num_components == 0 {
            return None;
        }

        let mut result = PCAResult::new();
        result.num_components = num_components;

        // Compute total variance (trace of covariance)
        let mut total_var = 0.0;
        for i in 0..self.n_assets {
            total_var += self.covariance[i][i];
        }
        result.total_variance = total_var;

        if total_var < 1e-15 {
            self.last_result = result;
            self.dirty = false;
            return Some(&self.last_result);
        }

        // Copy covariance for deflation
        let mut cov_work = self.covariance;

        // Extract components using power iteration
        for k in 0..num_components {
            // Initialize with random-ish vector (deterministic for reproducibility)
            let mut v = [0.0; MAX_ASSETS];
            for i in 0..self.n_assets {
                v[i] = ((k + i) as f64 * 0.1).sin();
            }

            // Normalize
            let mut norm = 0.0;
            for i in 0..self.n_assets {
                norm += v[i] * v[i];
            }
            norm = norm.sqrt();
            if norm < 1e-15 {
                continue;
            }
            for i in 0..self.n_assets {
                v[i] /= norm;
            }

            // Power iteration
            let mut eigenvalue = 0.0;
            for _iter in 0..max_iterations {
                // Matrix-vector multiply: w = Cov * v
                let mut w = [0.0; MAX_ASSETS];
                for i in 0..self.n_assets {
                    let mut sum = 0.0;
                    for j in 0..self.n_assets {
                        sum += cov_work[i][j] * v[j];
                    }
                    w[i] = sum;
                }

                // Compute eigenvalue estimate (Rayleigh quotient)
                let mut new_eigenvalue = 0.0;
                for i in 0..self.n_assets {
                    new_eigenvalue += v[i] * w[i];
                }

                // Normalize w
                let mut w_norm = 0.0;
                for i in 0..self.n_assets {
                    w_norm += w[i] * w[i];
                }
                w_norm = w_norm.sqrt();

                if w_norm < 1e-15 {
                    break;
                }

                for i in 0..self.n_assets {
                    w[i] /= w_norm;
                }

                // Check convergence
                if (new_eigenvalue - eigenvalue).abs() < tolerance {
                    eigenvalue = new_eigenvalue;
                    for i in 0..self.n_assets {
                        v[i] = w[i];
                    }
                    break;
                }

                eigenvalue = new_eigenvalue;
                for i in 0..self.n_assets {
                    v[i] = w[i];
                }
            }

            // Store results
            result.eigenvalues[k] = eigenvalue.max(0.0); // Ensure non-negative
            for i in 0..self.n_assets {
                result.eigenvectors[k][i] = v[i];
            }

            // Deflate covariance matrix: Cov = Cov - lambda * v * v^T
            for i in 0..self.n_assets {
                for j in 0..self.n_assets {
                    cov_work[i][j] -= eigenvalue * v[i] * v[j];
                }
            }
        }

        self.last_result = result;
        self.dirty = false;
        Some(&self.last_result)
    }

    /// Get the current number of assets
    #[inline]
    pub fn n_assets(&self) -> usize {
        self.n_assets
    }

    /// Get the number of observations processed
    #[inline]
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Get the update count
    #[inline]
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }

    /// Reset the engine
    #[inline]
    pub fn reset(&mut self) {
        self.covariance = [[0.0; MAX_ASSETS]; MAX_ASSETS];
        self.n_obs = 0;
        self.last_result = PCAResult::new();
        self.dirty = false;
        self.update_count.store(0, Ordering::Relaxed);
    }

    /// Get variance explained ratio for each component
    #[inline]
    pub fn variance_explained(&self) -> [f64; MAX_COMPONENTS] {
        let mut ratios = [0.0; MAX_COMPONENTS];
        if self.last_result.total_variance > 0.0 {
            for i in 0..self.last_result.num_components {
                ratios[i] = self.last_result.eigenvalues[i] / self.last_result.total_variance;
            }
        }
        ratios
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_single_component() {
        let mut pca = RollingPCAEngine::new(3).unwrap();
        
        // Add perfectly correlated observations
        for _ in 0..100 {
            pca.add_observation(&[1.0, 1.0, 1.0], 0.99);
        }

        let result = pca.compute_pca(1, 100, 1e-6).unwrap();
        
        assert!(result.num_components >= 1);
        assert!(result.eigenvalues[0] > 0.0);
    }

    #[test]
    fn test_pca_orthogonality() {
        let mut pca = RollingPCAEngine::new(5).unwrap();
        
        // Add varied observations
        for i in 0..200 {
            let obs = [
                (i as f64 * 0.1).sin(),
                (i as f64 * 0.1).cos(),
                (i as f64 * 0.05).sin(),
                (i as f64 * 0.07).cos(),
                (i as f64 * 0.03).sin(),
            ];
            pca.add_observation(&obs, 0.99);
        }

        let result = pca.compute_pca(3, 100, 1e-6).unwrap();
        
        // Check orthogonality of eigenvectors
        for i in 0..result.num_components {
            for j in (i + 1)..result.num_components {
                let mut dot = 0.0;
                for k in 0..5 {
                    dot += result.eigenvectors[i][k] * result.eigenvectors[j][k];
                }
                assert!(dot.abs() < 0.01, "Eigenvectors should be orthogonal");
            }
        }
    }

    #[test]
    fn test_invalid_asset_count() {
        let result = RollingPCAEngine::new(0);
        assert!(result.is_none());

        let result = RollingPCAEngine::new(MAX_ASSETS + 1);
        assert!(result.is_none());
    }

    #[test]
    fn test_variance_explained_sum() {
        let mut pca = RollingPCAEngine::new(4).unwrap();
        
        for i in 0..100 {
            let obs = [
                (i as f64 * 0.1).sin(),
                (i as f64 * 0.15).cos(),
                (i as f64 * 0.2).sin(),
                (i as f64 * 0.25).cos(),
            ];
            pca.add_observation(&obs, 0.99);
        }

        pca.compute_pca(4, 100, 1e-6);
        let ratios = pca.variance_explained();
        
        // Sum should be close to 1.0 (all variance explained)
        let sum: f64 = ratios.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Variance ratios should sum to ~1.0");
    }
}
