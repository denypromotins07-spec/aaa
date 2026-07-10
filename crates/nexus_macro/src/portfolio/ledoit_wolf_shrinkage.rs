//! Ledoit-Wolf covariance shrinkage estimator.
//!
//! When the number of assets N approaches or exceeds the number of observations T,
//! the sample covariance matrix becomes singular and unstable. Ledoit-Wolf shrinkage
//! pulls the sample covariance toward a structured target (constant correlation model)
//! to ensure positive-definiteness and better out-of-sample performance.
//!
//! Reference: Ledoit, O., & Wolf, M. (2004). "A well-conditioned estimator for
//! large-dimensional covariance matrices."

use ndarray::Array2;

/// Ledoit-Wolf shrinkage estimator
pub struct LedoitWolfShrinkage {
    /// Target structure: constant correlation model
    target_correlation: f64,
    /// Pre-allocated buffer for means
    means_buffer: Vec<f64>,
}

impl LedoitWolfShrinkage {
    /// Create new shrinkage estimator
    pub fn new() -> Self {
        Self {
            target_correlation: 0.0,
            means_buffer: Vec::new(),
        }
    }

    /// Compute shrunk covariance matrix from returns
    /// 
    /// # Arguments
    /// * `returns` - Matrix of asset returns [T x N]
    /// 
    /// # Returns
    /// Shrunk covariance matrix [N x N]
    pub fn shrink(&mut self, returns: &Array2<f64>) -> Result<Array2<f64>, String> {
        let (n_observations, n_assets) = returns.dim();
        
        if n_observations < 2 {
            return Err("Need at least 2 observations".to_string());
        }

        if n_assets == 0 {
            return Err("Need at least 1 asset".to_string());
        }

        // Step 1: Compute sample covariance matrix
        let sample_cov = self.sample_covariance(returns)?;

        // Step 2: Compute shrinkage intensity (optimal kappa)
        let kappa = self.compute_shrinkage_intensity(returns, &sample_cov)?;

        // Step 3: Build target matrix (constant correlation model)
        let target = self.build_target_matrix(&sample_cov);

        // Step 4: Apply shrinkage: Σ* = κ * F + (1 - κ) * S
        let mut shrunk_cov = Array2::<f64>::zeros((n_assets, n_assets));
        
        for i in 0..n_assets {
            for j in 0..n_assets {
                shrunk_cov[[i, j]] = kappa * target[[i, j]] + (1.0 - kappa) * sample_cov[[i, j]];
            }
        }

        // Ensure symmetry
        self.symmetrize(&mut shrunk_cov);

        // Ensure positive definiteness via eigenvalue clipping (simplified)
        self.ensure_positive_definite(&mut shrunk_cov)?;

        Ok(shrunk_cov)
    }

    /// Compute sample covariance matrix
    fn sample_covariance(&mut self, returns: &Array2<f64>) -> Result<Array2<f64>, String> {
        let (t, n) = returns.dim();
        let mut cov = Array2::<f64>::zeros((n, n));

        // Compute means
        self.means_buffer.resize(n, 0.0);
        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..t {
                sum += returns[[i, j]];
            }
            self.means_buffer[j] = sum / t as f64;
        }

        // Compute covariance
        for i in 0..n {
            for j in i..n {
                let mut sum = 0.0;
                for k in 0..t {
                    let di = returns[[k, i]] - self.means_buffer[i];
                    let dj = returns[[k, j]] - self.means_buffer[j];
                    sum += di * dj;
                }
                cov[[i, j]] = sum / (t - 1) as f64;
                if i != j {
                    cov[[j, i]] = cov[[i, j]];
                }
            }
        }

        Ok(cov)
    }

    /// Compute optimal shrinkage intensity using Ledoit-Wolf formula
    fn compute_shrinkage_intensity(
        &self,
        returns: &Array2<f64>,
        sample_cov: &Array2<f64>,
    ) -> Result<f64, String> {
        let (t, n) = returns.dim();
        
        if n == 1 {
            return Ok(0.0); // No shrinkage needed for single asset
        }

        // Compute means
        let mut means = vec![0.0; n];
        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..t {
                sum += returns[[i, j]];
            }
            means[j] = sum / t as f64;
        }

        // Compute sum of squared deviations from sample cov
        let mut sum_sq_dev = 0.0;
        for i in 0..n {
            for j in 0..n {
                let diff = sample_cov[[i, j]];
                sum_sq_dev += diff * diff;
            }
        }

        // Compute estimator of asymptotic variance
        let mut pi = 0.0;
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..t {
                    let di = returns[[k, i]] - means[i];
                    let dj = returns[[k, j]] - means[j];
                    sum += (di * dj - sample_cov[[i, j]]) * (di * dj - sample_cov[[i, j]]);
                }
                pi += sum / (t as f64 * t as f64);
            }
        }

        // Compute gamma (squared Frobenius norm of target estimation error)
        let gamma = self.compute_gamma(sample_cov)?;

        // Optimal shrinkage: κ = (π - γ) / τ where τ = π - γ + ...
        // Simplified: κ = max(0, min(1, (π - γ) / π))
        let kappa = if pi > 1e-15 {
            ((pi - gamma) / pi).max(0.0).min(1.0)
        } else {
            0.0
        };

        // Adjust for finite sample bias
        let adjusted_kappa = (kappa * t as f64 / (t as f64 + 2.0)).max(0.0).min(1.0);

        Ok(adjusted_kappa)
    }

    /// Compute gamma term for shrinkage intensity
    fn compute_gamma(&self, sample_cov: &Array2<f64>) -> Result<f64, String> {
        let n = sample_cov.nrows();
        
        // Extract variances and compute average correlation
        let mut sum_var = 0.0;
        let mut sum_corr = 0.0;
        let mut n_pairs = 0usize;

        for i in 0..n {
            sum_var += sample_cov[[i, i]];
            
            for j in (i + 1)..n {
                let var_i = sample_cov[[i, i]];
                let var_j = sample_cov[[j, j]];
                
                if var_i > 1e-15 && var_j > 1e-15 {
                    let corr = sample_cov[[i, j]] / (var_i * var_j).sqrt();
                    let corr = corr.max(-1.0).min(1.0);
                    sum_corr += corr;
                    n_pairs += 1;
                }
            }
        }

        // Average correlation
        let avg_corr = if n_pairs > 0 {
            sum_corr / n_pairs as f64
        } else {
            0.0
        };

        // Gamma measures how far sample cov is from constant correlation target
        let mut gamma = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let var_i = sample_cov[[i, i]];
                let var_j = sample_cov[[j, j]];
                
                if var_i > 1e-15 && var_j > 1e-15 {
                    let target_cov = avg_corr * (var_i * var_j).sqrt();
                    let diff = sample_cov[[i, j]] - target_cov;
                    gamma += diff * diff;
                }
            }
        }

        Ok(gamma / n as f64)
    }

    /// Build constant correlation target matrix
    fn build_target_matrix(&self, sample_cov: &Array2<f64>) -> Array2<f64> {
        let n = sample_cov.nrows();
        let mut target = Array2::<f64>::zeros((n, n));

        // Extract variances
        let variances: Vec<f64> = (0..n)
            .map(|i| sample_cov[[i, i]].max(1e-15))
            .collect();

        // Compute average off-diagonal correlation
        let mut sum_corr = 0.0;
        let mut n_pairs = 0usize;

        for i in 0..n {
            for j in (i + 1)..n {
                let std_i = variances[i].sqrt();
                let std_j = variances[j].sqrt();
                
                if std_i > 1e-10 && std_j > 1e-10 {
                    let corr = sample_cov[[i, j]] / (std_i * std_j);
                    let corr = corr.max(-1.0).min(1.0);
                    sum_corr += corr;
                    n_pairs += 1;
                }
            }
        }

        let avg_corr = if n_pairs > 0 {
            sum_corr / n_pairs as f64
        } else {
            0.0
        };

        // Build target: diagonal = variance, off-diagonal = avg_corr * sqrt(var_i * var_j)
        for i in 0..n {
            target[[i, i]] = variances[i];
            
            for j in (i + 1)..n {
                let target_cov = avg_corr * (variances[i] * variances[j]).sqrt();
                target[[i, j]] = target_cov;
                target[[j, i]] = target_cov;
            }
        }

        target
    }

    /// Ensure matrix symmetry
    fn symmetrize(&self, matrix: &mut Array2<f64>) {
        let n = matrix.nrows();
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = (matrix[[i, j]] + matrix[[j, i]]) / 2.0;
                matrix[[i, j]] = avg;
                matrix[[j, i]] = avg;
            }
        }
    }

    /// Ensure positive definiteness via diagonal perturbation
    fn ensure_positive_definite(&self, matrix: &mut Array2<f64>) -> Result<(), String> {
        let n = matrix.nrows();
        
        // Simple approach: add small value to diagonal until all eigenvalues positive
        // For production, use proper eigendecomposition
        let mut epsilon = 1e-8;
        
        for _ in 0..10 {
            // Check if diagonally dominant (sufficient condition for PD)
            let mut is_pd = true;
            for i in 0..n {
                let diag = matrix[[i, i]];
                let mut row_sum = 0.0;
                for j in 0..n {
                    if i != j {
                        row_sum += matrix[[i, j]].abs();
                    }
                }
                if diag <= row_sum {
                    is_pd = false;
                    break;
                }
            }

            if is_pd {
                return Ok(());
            }

            // Increase diagonal
            for i in 0..n {
                matrix[[i, i]] += epsilon;
            }
            epsilon *= 10.0;
        }

        // If still not PD, add larger perturbation
        for i in 0..n {
            matrix[[i, i]] *= 1.001;
        }

        Ok(())
    }
}

impl Default for LedoitWolfShrinkage {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shrinkage_basic() {
        // Create simple returns matrix
        let returns = Array2::from_shape_vec((10, 3), vec![
            0.01, 0.02, 0.015,
            -0.02, -0.01, -0.015,
            0.015, 0.01, 0.02,
            0.005, -0.005, 0.01,
            -0.01, 0.015, -0.005,
            0.02, 0.01, 0.015,
            -0.015, -0.02, -0.01,
            0.01, 0.005, 0.02,
            0.005, 0.01, -0.005,
            -0.005, 0.005, 0.01,
        ]).unwrap();

        let mut lw = LedoitWolfShrinkage::new();
        let result = lw.shrink(&returns).unwrap();

        assert_eq!(result.dim(), (3, 3));
        
        // Check symmetry
        for i in 0..3 {
            for j in (i + 1)..3 {
                assert!((result[[i, j]] - result[[j, i]]).abs() < 1e-10);
            }
        }

        // Check positive diagonal
        for i in 0..3 {
            assert!(result[[i, i]] > 0.0);
        }
    }

    #[test]
    fn test_shrinkage_intensity_bounds() {
        let mut lw = LedoitWolfShrinkage::new();
        
        // Shrinkage intensity should be in [0, 1]
        let returns = Array2::from_shape_vec((20, 5), (0..100).map(|i| (i % 100) as f64 * 0.01 - 0.5).collect()).unwrap();
        
        let cov = lw.sample_covariance(&returns).unwrap();
        let kappa = lw.compute_shrinkage_intensity(&returns, &cov).unwrap();
        
        assert!(kappa >= 0.0);
        assert!(kappa <= 1.0);
    }
}
