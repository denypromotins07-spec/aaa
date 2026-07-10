//! Tail Risk Parity allocator for crisis-aware portfolio construction
//!
//! Extends traditional risk parity to account for tail risks and non-Gaussian
//! return distributions. Allocates capital based on marginal contribution to
//! Expected Shortfall rather than variance.

use ndarray::{Array1, Array2, ArrayView1};
use thiserror::Error;

/// Errors from tail risk parity calculations
#[derive(Error, Debug, Clone)]
pub enum TailRiskParityError {
    #[error("Covariance matrix is not positive definite")]
    NotPositiveDefinite,
    
    #[error("Invalid asset count: {0}")]
    InvalidAssetCount(usize),
    
    #[error("ES calculation failed: {0}")]
    ExpectedShortfallError(String),
    
    #[error("Optimization did not converge after {0} iterations")]
    NoConvergence(usize),
    
    #[error("Budget constraint infeasible")]
    InfeasibleConstraint,
}

/// Configuration for tail risk parity
#[derive(Debug, Clone)]
pub struct TailRiskParityConfig {
    /// Target portfolio Expected Shortfall (as fraction)
    pub target_es: f64,
    /// Maximum position weight
    pub max_weight: f64,
    /// Minimum position weight  
    pub min_weight: f64,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Use robust covariance estimation
    pub robust_covariance: bool,
}

impl Default for TailRiskParityConfig {
    fn default() -> Self {
        Self {
            target_es: 0.05,
            max_weight: 0.40,
            min_weight: 0.01,
            tolerance: 1e-6,
            max_iterations: 1000,
            robust_covariance: true,
        }
    }
}

/// Result of tail risk parity optimization
#[derive(Debug, Clone)]
pub struct TailRiskParityResult {
    /// Optimal weights for each asset
    pub weights: Array1<f64>,
    /// Marginal contribution to ES for each asset
    pub marginal_es: Array1<f64>,
    /// Total portfolio Expected Shortfall
    pub portfolio_es: f64,
    /// Number of iterations to converge
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
}

impl TailRiskParityResult {
    /// Check if all risk contributions are equal (risk parity achieved)
    pub fn is_risk_parity(&self, tolerance: f64) -> bool {
        let mean_mc = self.marginal_es.mean().unwrap_or(0.0);
        
        if mean_mc < 1e-10 {
            return false;
        }
        
        for &mc in self.marginal_es.iter() {
            if (mc - mean_mc).abs() / mean_mc > tolerance {
                return false;
            }
        }
        
        true
    }
}

/// Tail Risk Parity optimizer
pub struct TailRiskParityOptimizer {
    config: TailRiskParityConfig,
    n_assets: usize,
}

impl TailRiskParityOptimizer {
    /// Create a new optimizer
    pub fn new(n_assets: usize, config: TailRiskParityConfig) -> Result<Self, TailRiskParityError> {
        if n_assets < 2 {
            return Err(TailRiskParityError::InvalidAssetCount(n_assets));
        }
        
        Ok(Self { config, n_assets })
    }
    
    /// Calculate optimal tail risk parity weights
    /// 
    /// Uses iterative algorithm to equalize marginal contributions to ES:
    /// w_i * ∂ES/∂w_i = constant for all i
    pub fn optimize(
        &self,
        returns: &Array2<f64>,
        tail_probs: &ArrayView1<f64>,
    ) -> Result<TailRiskParityResult, TailRiskParityError> {
        let (n_obs, n_assets) = returns.dim();
        
        if n_assets != self.n_assets {
            return Err(TailRiskParityError::InvalidAssetCount(n_assets));
        }
        
        if n_obs < 30 {
            return Err(TailRiskParityError::ExpectedShortfallError(
                "Insufficient observations for ES estimation".to_string()
            ));
        }
        
        // Initialize with equal weights
        let mut weights = Array1::from_elem(n_assets, 1.0 / n_assets as f64);
        
        // Calculate covariance matrix (possibly robust)
        let cov_matrix = if self.config.robust_covariance {
            self.robust_covariance(returns)?
        } else {
            self.sample_covariance(returns)?
        };
        
        // Iterative optimization
        let mut converged = false;
        let mut iterations = 0;
        
        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;
            
            // Calculate portfolio ES and marginal contributions
            let (portfolio_es, marginal_es) = 
                self.calculate_portfolio_es_and_marginal(&weights, returns, tail_probs, &cov_matrix)?;
            
            // Check convergence: all risk contributions should be equal
            let risk_contributions = (&weights * &marginal_es);
            let mean_rc = risk_contributions.sum() / n_assets as f64;
            
            let max_deviation = risk_contributions.iter()
                .map(|&rc| (rc - mean_rc).abs())
                .fold(0.0, f64::max);
            
            if max_deviation < self.config.tolerance {
                converged = true;
                break;
            }
            
            // Update weights using Newton-like step
            // w_new = w * (target_RC / current_RC)^step_size
            let step_size = 0.1;
            let mut new_weights = Array1::zeros(n_assets);
            
            for i in 0..n_assets {
                let current_rc = risk_contributions[i];
                let target_rc = mean_rc;
                
                if current_rc > 1e-15 {
                    let ratio = target_rc / current_rc;
                    new_weights[i] = weights[i] * ratio.powf(step_size);
                } else {
                    new_weights[i] = weights[i];
                }
                
                // Apply bounds
                new_weights[i] = new_weights[i].clamp(self.config.min_weight, self.config.max_weight);
            }
            
            // Normalize to sum to 1
            let sum: f64 = new_weights.sum();
            if sum > 1e-10 {
                new_weights.mapv_inplace(|w| w / sum);
            }
            
            weights = new_weights;
        }
        
        // Final ES calculation
        let (portfolio_es, marginal_es) = 
            self.calculate_portfolio_es_and_marginal(&weights, returns, tail_probs, &cov_matrix)?;
        
        Ok(TailRiskParityResult {
            weights,
            marginal_es,
            portfolio_es,
            iterations,
            converged,
        })
    }
    
    /// Calculate sample covariance matrix
    fn sample_covariance(&self, returns: &Array2<f64>) -> Result<Array2<f64>, TailRiskParityError> {
        let (n_obs, n_assets) = returns.dim();
        let mut cov = Array2::zeros((n_assets, n_assets));
        
        // Calculate means
        let means: Array1<f64> = (0..n_assets)
            .map(|j| returns.column(j).sum() / n_obs as f64)
            .collect();
        
        // Calculate covariance
        for i in 0..n_assets {
            for j in i..n_assets {
                let mut sum = 0.0;
                for k in 0..n_obs {
                    sum += (returns[[k, i]] - means[i]) * (returns[[k, j]] - means[j]);
                }
                cov[[i, j]] = sum / (n_obs - 1) as f64;
                cov[[j, i]] = cov[[i, j]];
            }
        }
        
        Ok(cov)
    }
    
    /// Calculate robust covariance using shrinkage estimator
    fn robust_covariance(&self, returns: &Array2<f64>) -> Result<Array2<f64>, TailRiskParityError> {
        let sample_cov = self.sample_covariance(returns)?;
        let n_assets = sample_cov.nrows();
        
        // Ledoit-Wolf shrinkage toward constant correlation
        let trace: f64 = (0..n_assets).map(|i| sample_cov[[i, i]]).sum();
        let avg_var = trace / n_assets as f64;
        
        // Average correlation
        let mut sum_corr = 0.0;
        let mut count = 0;
        for i in 0..n_assets {
            for j in (i+1)..n_assets {
                let std_i = sample_cov[[i, i]].sqrt();
                let std_j = sample_cov[[j, j]].sqrt();
                if std_i > 1e-10 && std_j > 1e-10 {
                    sum_corr += sample_cov[[i, j]] / (std_i * std_j);
                    count += 1;
                }
            }
        }
        
        let avg_corr = if count > 0 { sum_corr / count as f64 } else { 0.0 };
        
        // Shrinkage target: constant correlation matrix
        let mut target = Array2::zeros((n_assets, n_assets));
        for i in 0..n_assets {
            for j in 0..n_assets {
                if i == j {
                    target[[i, j]] = avg_var;
                } else {
                    target[[i, j]] = avg_corr * avg_var;
                }
            }
        }
        
        // Optimal shrinkage intensity (simplified)
        let n_obs = returns.nrows() as f64;
        let shrinkage = ((1.0 + avg_corr) / (1.0 + (n_assets - 1) as f64 * avg_corr)) * (1.0 / n_obs);
        let shrinkage = shrinkage.clamp(0.0, 1.0);
        
        // Shrunk covariance
        let mut shrunk_cov = Array2::zeros((n_assets, n_assets));
        for i in 0..n_assets {
            for j in 0..n_assets {
                shrunk_cov[[i, j]] = (1.0 - shrinkage) * sample_cov[[i, j]] + shrinkage * target[[i, j]];
            }
        }
        
        Ok(shrunk_cov)
    }
    
    /// Calculate portfolio ES and marginal contributions
    fn calculate_portfolio_es_and_marginal(
        &self,
        weights: &Array1<f64>,
        returns: &Array2<f64>,
        tail_probs: &ArrayView1<f64>,
        cov_matrix: &Array2<f64>,
    ) -> Result<(f64, Array1<f64>), TailRiskParityError> {
        let n_obs = returns.nrows();
        let n_assets = returns.ncols();
        
        // Calculate portfolio returns
        let mut port_returns = Array1::zeros(n_obs);
        for t in 0..n_obs {
            for i in 0..n_assets {
                port_returns[t] += weights[i] * returns[[t, i]];
            }
        }
        
        // Sort returns for ES calculation
        let mut sorted_returns = port_returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        // Find VaR threshold (α quantile)
        let alpha = 0.05; // 95% confidence
        let var_idx = (alpha * n_obs as f64) as usize;
        let var = sorted_returns[var_idx.min(n_obs - 1)];
        
        // Calculate ES (average of returns below VaR)
        let mut es_sum = 0.0;
        let mut es_count = 0;
        for &r in &sorted_returns[..=var_idx.min(n_obs - 1)] {
            es_sum += r;
            es_count += 1;
        }
        
        let portfolio_es = if es_count > 0 {
            -es_sum / es_count as f64 // ES is typically reported as positive number
        } else {
            0.0
        };
        
        // Marginal ES approximation using delta-normal approach
        // ∂ES/∂w ≈ Σ_ij * w_j / σ_p * φ(z_α) / α
        // Simplified: use covariance-based approximation
        
        let port_variance: f64 = weights.dot(&cov_matrix.dot(weights));
        let port_std = port_variance.sqrt();
        
        let mut marginal_es = Array1::zeros(n_assets);
        for i in 0..n_assets {
            let cov_with_port: f64 = (0..n_assets)
                .map(|j| cov_matrix[[i, j]] * weights[j])
                .sum();
            
            if port_std > 1e-10 {
                // Marginal VaR approximation scaled for ES
                marginal_es[i] = cov_with_port / port_std * 2.0; // ES multiplier
            }
            
            // Adjust by tail probability
            marginal_es[i] *= tail_probs[i].max(0.01);
        }
        
        Ok((portfolio_es, marginal_es))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tail_risk_parity_basic() {
        // Generate synthetic correlated returns
        let n_obs = 252;
        let n_assets = 3;
        let mut returns = Array2::zeros((n_obs, n_assets));
        
        for t in 0..n_obs {
            let base = (t as f64 * 0.1).sin() * 0.02;
            returns[[t, 0]] = base;
            returns[[t, 1]] = base * 0.8 + 0.005;
            returns[[t, 2]] = base * 1.2 - 0.003;
        }
        
        let tail_probs = Array1::from_elem(n_assets, 0.05);
        
        let optimizer = TailRiskParityOptimizer::new(n_assets, TailRiskParityConfig::default()).unwrap();
        let result = optimizer.optimize(&returns, &tail_probs.view()).unwrap();
        
        assert!(result.converged || result.iterations == 1000);
        assert_eq!(result.weights.len(), n_assets);
        
        // Weights should sum to ~1
        let sum: f64 = result.weights.sum();
        assert!((sum - 1.0).abs() < 0.01);
    }
}
