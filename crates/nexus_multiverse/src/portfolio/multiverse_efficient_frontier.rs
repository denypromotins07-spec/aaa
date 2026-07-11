//! Multiverse Portfolio Theory - Efficient Frontier
//! 
//! Extends classical Markowitz Mean-Variance optimization to the multiverse
//! by optimizing across all quantum branches weighted by their measure.

use alloc::vec::Vec;
use core::fmt;

/// Error types for multiverse portfolio optimization
#[derive(Debug, Clone, PartialEq)]
pub enum MultiversePortfolioError {
    InvalidAssetCount { count: usize },
    NegativeWeight { weight: f64 },
    WeightSumInvalid { sum: f64 },
    CovarianceMatrixInvalid { reason: &'static str },
    NumericalInstability { message: &'static str },
}

impl fmt::Display for MultiversePortfolioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiversePortfolioError::InvalidAssetCount { count } => {
                write!(f, "Invalid asset count: {}", count)
            }
            MultiversePortfolioError::NegativeWeight { weight } => {
                write!(f, "Negative weight: {}", weight)
            }
            MultiversePortfolioError::WeightSumInvalid { sum } => {
                write!(f, "Weight sum invalid: {}", sum)
            }
            MultiversePortfolioError::CovarianceMatrixInvalid { reason } => {
                write!(f, "Covariance matrix invalid: {}", reason)
            }
            MultiversePortfolioError::NumericalInstability { message } => {
                write!(f, "Numerical instability: {}", message)
            }
        }
    }
}

/// Asset allocation in a single branch
#[derive(Debug, Clone)]
pub struct BranchAllocation {
    pub branch_id: usize,
    pub weights: Vec<f64>,
    pub expected_return: f64,
    pub variance: f64,
    pub branch_probability: f64,
}

/// Multiverse Efficient Frontier result
#[derive(Debug, Clone)]
pub struct MultiverseEfficientFrontier {
    /// Portfolio weights (same across all branches - pre-measurement decision)
    pub weights: Vec<f64>,
    /// Measure-weighted expected return
    pub expected_return: f64,
    /// Measure-weighted variance
    pub variance: f64>,
    /// List of allocations per branch
    pub branch_allocations: Vec<BranchAllocation>,
    /// Sharpe ratio (measure-weighted)
    pub sharpe_ratio: f64,
}

/// Multiverse Portfolio Optimizer
pub struct MultiversePortfolioOptimizer {
    num_assets: usize,
    risk_free_rate: f64,
    weight_tolerance: f64,
}

impl MultiversePortfolioOptimizer {
    pub fn new(num_assets: usize, risk_free_rate: f64) -> Result<Self, MultiversePortfolioError> {
        if num_assets == 0 {
            return Err(MultiversePortfolioError::InvalidAssetCount { count: num_assets });
        }

        Ok(Self {
            num_assets,
            risk_free_rate,
            weight_tolerance: 1e-10,
        })
    }

    /// Calculate measure-weighted expected return across all branches
    pub fn calculate_measure_weighted_return(
        &self,
        branch_returns: &[f64],
        branch_probabilities: &[f64],
    ) -> Result<f64, MultiversePortfolioError> {
        if branch_returns.len() != branch_probabilities.len() {
            return Err(MultiversePortfolioError::NumericalInstability {
                message: "Branch returns and probabilities length mismatch",
            });
        }

        let mut weighted_return = 0.0_f64;
        
        for (ret, prob) in branch_returns.iter().zip(branch_probabilities.iter()) {
            // Check for valid probability
            if *prob < 0.0 || *prob > 1.0 {
                return Err(MultiversePortfolioError::NumericalInstability {
                    message: "Invalid branch probability",
                });
            }

            let contribution = ret * prob;
            
            if contribution.is_nan() || contribution.is_infinite() {
                return Err(MultiversePortfolioError::NumericalInstability {
                    message: "NaN or Inf in weighted return calculation",
                });
            }

            weighted_return += contribution;
        }

        Ok(weighted_return)
    }

    /// Calculate measure-weighted portfolio variance
    pub fn calculate_measure_weighted_variance(
        &self,
        branch_variances: &[f64],
        branch_probabilities: &[f64],
        branch_returns: &[f64],
        overall_return: f64,
    ) -> Result<f64, MultiversePortfolioError> {
        if branch_variances.len() != branch_probabilities.len() {
            return Err(MultiversePortfolioError::NumericalInstability {
                message: "Branch variances and probabilities length mismatch",
            });
        }

        let mut total_variance = 0.0_f64;
        
        for (var, prob, ret) in branch_variances
            .iter()
            .zip(branch_probabilities.iter())
            .zip(branch_returns.iter())
        {
            // Law of total variance: E[Var] + Var[E]
            // Within-branch variance contribution
            let within_contribution = *var * *prob;
            
            // Between-branch variance contribution (variance of means)
            let between_contribution = *prob * (*ret - overall_return).powi(2);
            
            let contribution = within_contribution + between_contribution;
            
            if contribution.is_nan() || contribution.is_infinite() {
                return Err(MultiversePortfolioError::NumericalInstability {
                    message: "NaN or Inf in variance calculation",
                });
            }

            total_variance += contribution;
        }

        if total_variance < 0.0 {
            return Err(MultiversePortfolioError::CovarianceMatrixInvalid {
                reason: "Negative total variance",
            });
        }

        Ok(total_variance)
    }

    /// Optimize portfolio for maximum measure-weighted Sharpe ratio
    pub fn optimize_sharpe_ratio(
        &self,
        branch_expected_returns: &[Vec<f64>],
        branch_covariances: &[Vec<Vec<f64>>],
        branch_probabilities: &[f64],
    ) -> Result<MultiverseEfficientFrontier, MultiversePortfolioError> {
        let num_branches = branch_expected_returns.len();
        
        if num_branches == 0 {
            return Err(MultiversePortfolioError::NumericalInstability {
                message: "No branches provided",
            });
        }

        // Validate inputs
        for (i, returns) in branch_expected_returns.iter().enumerate() {
            if returns.len() != self.num_assets {
                return Err(MultiversePortfolioError::InvalidAssetCount {
                    count: returns.len(),
                });
            }
        }

        // Simple equal-weight portfolio as baseline
        // In production, would use quadratic programming
        let mut weights = vec![1.0 / self.num_assets as f64; self.num_assets];

        // Calculate portfolio metrics for each branch
        let mut branch_allocations = Vec::with_capacity(num_branches);
        let mut branch_returns = Vec::with_capacity(num_branches);
        let mut branch_variances = Vec::with_capacity(num_branches);

        for (branch_idx, (returns, cov)) in branch_expected_returns
            .iter()
            .zip(branch_covariances.iter())
            .enumerate()
        {
            // Portfolio expected return in this branch
            let port_return = self.calculate_portfolio_return(&weights, returns)?;
            
            // Portfolio variance in this branch
            let port_variance = self.calculate_portfolio_variance(&weights, cov)?;

            branch_returns.push(port_return);
            branch_variances.push(port_variance);

            branch_allocations.push(BranchAllocation {
                branch_id: branch_idx,
                weights: weights.clone(),
                expected_return: port_return,
                variance: port_variance,
                branch_probability: branch_probabilities[branch_idx],
            });
        }

        // Calculate measure-weighted overall metrics
        let overall_return = self.calculate_measure_weighted_return(
            &branch_returns,
            branch_probabilities,
        )?;

        let overall_variance = self.calculate_measure_weighted_variance(
            &branch_variances,
            branch_probabilities,
            &branch_returns,
            overall_return,
        )?;

        let overall_std = overall_variance.sqrt();
        let sharpe_ratio = if overall_std > self.weight_tolerance {
            (overall_return - self.risk_free_rate) / overall_std
        } else {
            0.0
        };

        Ok(MultiverseEfficientFrontier {
            weights,
            expected_return: overall_return,
            variance: overall_variance,
            branch_allocations,
            sharpe_ratio,
        })
    }

    /// Calculate portfolio return given weights and asset returns
    fn calculate_portfolio_return(
        &self,
        weights: &[f64],
        asset_returns: &[f64],
    ) -> Result<f64, MultiversePortfolioError> {
        if weights.len() != asset_returns.len() {
            return Err(MultiversePortfolioError::NumericalInstability {
                message: "Weights and returns length mismatch",
            });
        }

        let mut port_return = 0.0_f64;
        
        for (w, r) in weights.iter().zip(asset_returns.iter()) {
            if *w < 0.0 {
                return Err(MultiversePortfolioError::NegativeWeight { weight: *w });
            }
            
            port_return += w * r;
        }

        Ok(port_return)
    }

    /// Calculate portfolio variance given weights and covariance matrix
    fn calculate_portfolio_variance(
        &self,
        weights: &[f64],
        covariance: &[Vec<f64>],
    ) -> Result<f64, MultiversePortfolioError> {
        let n = weights.len();
        
        if covariance.len() != n {
            return Err(MultiversePortfolioError::CovarianceMatrixInvalid {
                reason: "Covariance matrix dimension mismatch",
            });
        }

        let mut variance = 0.0_f64;
        
        for i in 0..n {
            for j in 0..n {
                if i >= covariance.len() || j >= covariance[i].len() {
                    return Err(MultiversePortfolioError::CovarianceMatrixInvalid {
                        reason: "Covariance matrix access out of bounds",
                    });
                }

                variance += weights[i] * weights[j] * covariance[i][j];
            }
        }

        if variance < 0.0 {
            return Err(MultiversePortfolioError::CovarianceMatrixInvalid {
                reason: "Negative portfolio variance",
            });
        }

        Ok(variance)
    }

    /// Verify weights sum to 1.0
    pub fn verify_weights(&self, weights: &[f64]) -> Result<(), MultiversePortfolioError> {
        let sum: f64 = weights.iter().sum();
        
        let deviation = (sum - 1.0).abs();
        if deviation > self.weight_tolerance {
            return Err(MultiversePortfolioError::WeightSumInvalid { sum });
        }

        // Check for negative weights
        for &w in weights {
            if w < -self.weight_tolerance {
                return Err(MultiversePortfolioError::NegativeWeight { weight: w });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let optimizer = MultiversePortfolioOptimizer::new(3, 0.02).unwrap();
        assert_eq!(optimizer.num_assets, 3);
        assert!((optimizer.risk_free_rate - 0.02).abs() < 1e-14);
    }

    #[test]
    fn test_measure_weighted_return() {
        let optimizer = MultiversePortfolioOptimizer::new(2, 0.0).unwrap();
        
        let returns = vec![0.1, 0.2];
        let probs = vec![0.5, 0.5];
        
        let weighted = optimizer
            .calculate_measure_weighted_return(&returns, &probs)
            .unwrap();
        
        assert!((weighted - 0.15).abs() < 1e-14);
    }

    #[test]
    fn test_weight_verification() {
        let optimizer = MultiversePortfolioOptimizer::new(3, 0.0).unwrap();
        
        let valid_weights = vec![0.3, 0.4, 0.3];
        assert!(optimizer.verify_weights(&valid_weights).is_ok());

        let invalid_sum = vec![0.5, 0.5, 0.5];
        assert!(optimizer.verify_weights(&invalid_sum).is_err());

        let negative = vec![0.6, -0.1, 0.5];
        assert!(optimizer.verify_weights(&negative).is_err());
    }
}
