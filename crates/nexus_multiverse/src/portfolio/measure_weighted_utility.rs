//! Measure-Weighted Utility Function for Multiverse Portfolio Optimization
//! 
//! Maximizes capital growth across highest-measure branches while hedging
//! against low-measure catastrophic branches (quantum black swans).

use alloc::vec::Vec;
use core::fmt;

/// Error types for measure-weighted utility
#[derive(Debug, Clone, PartialEq)]
pub enum MeasureWeightedUtilityError {
    InvalidRiskAversion { gamma: f64 },
    NegativeConsumption { value: f64 },
    MeasureSumInvalid { sum: f64 },
    NumericalOverflow { message: &'static str },
    CatastrophicBranchDetected { branch_id: usize, loss: f64 },
}

impl fmt::Display for MeasureWeightedUtilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeasureWeightedUtilityError::InvalidRiskAversion { gamma } => {
                write!(f, "Invalid risk aversion: {}", gamma)
            }
            MeasureWeightedUtilityError::NegativeConsumption { value } => {
                write!(f, "Negative consumption: {}", value)
            }
            MeasureWeightedUtilityError::MeasureSumInvalid { sum } => {
                write!(f, "Measure sum invalid: {}", sum)
            }
            MeasureWeightedUtilityError::NumericalOverflow { message } => {
                write!(f, "Numerical overflow: {}", message)
            }
            MeasureWeightedUtilityError::CatastrophicBranchDetected { branch_id, loss } => {
                write!(f, "Catastrophic branch {}: loss={}", branch_id, loss)
            }
        }
    }
}

/// CRRA (Constant Relative Risk Aversion) utility function parameters
pub struct CrraUtility {
    /// Risk aversion coefficient (γ > 0)
    gamma: f64,
    /// Minimum consumption floor
    consumption_floor: f64,
}

impl CrraUtility {
    pub fn new(gamma: f64, consumption_floor: f64) -> Result<Self, MeasureWeightedUtilityError> {
        if gamma <= 0.0 {
            return Err(MeasureWeightedUtilityError::InvalidRiskAversion { gamma });
        }

        if consumption_floor < 0.0 {
            return Err(MeasureWeightedUtilityError::NegativeConsumption {
                value: consumption_floor,
            });
        }

        Ok(Self {
            gamma,
            consumption_floor,
        })
    }

    /// Calculate CRRA utility: U(c) = c^(1-γ)/(1-γ) for γ ≠ 1, log(c) for γ = 1
    #[inline]
    pub fn utility(&self, consumption: f64) -> Result<f64, MeasureWeightedUtilityError> {
        let c = consumption.max(self.consumption_floor);

        if c <= 0.0 {
            return Err(MeasureWeightedUtilityError::NegativeConsumption {
                value: consumption,
            });
        }

        let util = if (self.gamma - 1.0).abs() < 1e-10 {
            // Log utility case (γ = 1)
            c.ln()
        } else {
            // CRRA case
            c.powf(1.0 - self.gamma) / (1.0 - self.gamma)
        };

        if util.is_nan() || util.is_infinite() {
            return Err(MeasureWeightedUtilityError::NumericalOverflow {
                message: "Utility calculation overflow",
            });
        }

        Ok(util)
    }

    /// Calculate marginal utility: U'(c) = c^(-γ)
    #[inline]
    pub fn marginal_utility(&self, consumption: f64) -> Result<f64, MeasureWeightedUtilityError> {
        let c = consumption.max(self.consumption_floor);

        if c <= 0.0 {
            return Err(MeasureWeightedUtilityError::NegativeConsumption {
                value: consumption,
            });
        }

        let mu = c.powf(-self.gamma);

        if mu.is_nan() || mu.is_infinite() {
            return Err(MeasureWeightedUtilityError::NumericalOverflow {
                message: "Marginal utility calculation overflow",
            });
        }

        Ok(mu)
    }
}

/// Branch outcome with measure and utility
#[derive(Debug, Clone)]
pub struct BranchOutcome {
    pub branch_id: usize,
    pub wealth: f64,
    pub measure: f64,
    pub is_catastrophic: bool,
}

/// Measure-Weighted Utility Calculator
pub struct MeasureWeightedUtilityCalculator {
    utility: CrraUtility,
    /// Threshold for catastrophic loss (e.g., 50% wealth loss)
    catastrophic_threshold: f64,
    /// Penalty multiplier for catastrophic branches
    catastrophe_penalty: f64,
}

impl MeasureWeightedUtilityCalculator {
    pub fn new(
        gamma: f64,
        consumption_floor: f64,
        catastrophic_threshold: f64,
        catastrophe_penalty: f64,
    ) -> Result<Self, MeasureWeightedUtilityError> {
        let utility = CrraUtility::new(gamma, consumption_floor)?;

        if catastrophic_threshold <= 0.0 || catastrophic_threshold > 1.0 {
            return Err(MeasureWeightedUtilityError::InvalidRiskAversion {
                gamma: catastrophic_threshold,
            });
        }

        if catastrophe_penalty < 1.0 {
            return Err(MeasureWeightedUtilityError::InvalidRiskAversion {
                gamma: catastrophe_penalty,
            });
        }

        Ok(Self {
            utility,
            catastrophic_threshold,
            catastrophe_penalty,
        })
    }

    /// Calculate measure-weighted expected utility across all branches
    pub fn calculate_expected_utility(
        &self,
        outcomes: &[BranchOutcome],
    ) -> Result<f64, MeasureWeightedUtilityError> {
        // Verify measure sums to 1
        let measure_sum: f64 = outcomes.iter().map(|o| o.measure).sum();
        if (measure_sum - 1.0).abs() > 1e-6 && measure_sum > 1e-6 {
            return Err(MeasureWeightedUtilityError::MeasureSumInvalid {
                sum: measure_sum,
            });
        }

        let mut expected_utility = 0.0_f64;

        for outcome in outcomes {
            // Check for catastrophic branch
            let loss_ratio = (outcome.wealth - 1.0).abs();
            let is_catastrophic = loss_ratio > self.catastrophic_threshold;

            // Calculate base utility
            let util = self.utility.utility(outcome.wealth)?;

            // Apply catastrophe penalty
            let adjusted_utility = if is_catastrophic {
                util * self.catastrophe_penalty
            } else {
                util
            };

            // Weight by quantum measure
            let contribution = adjusted_utility * outcome.measure;

            if contribution.is_nan() || contribution.is_infinite() {
                return Err(MeasureWeightedUtilityError::NumericalOverflow {
                    message: "Expected utility calculation overflow",
                });
            }

            expected_utility += contribution;
        }

        Ok(expected_utility)
    }

    /// Find optimal portfolio that maximizes measure-weighted utility
    pub fn optimize_portfolio(
        &self,
        candidate_portfolios: &[Vec<BranchOutcome>],
    ) -> Result<(usize, f64), MeasureWeightedUtilityError> {
        if candidate_portfolios.is_empty() {
            return Err(MeasureWeightedUtilityError::NumericalOverflow {
                message: "No candidate portfolios",
            });
        }

        let mut best_idx = 0;
        let mut best_utility = f64::NEG_INFINITY;

        for (idx, portfolio) in candidate_portfolios.iter().enumerate() {
            let util = self.calculate_expected_utility(portfolio)?;

            if util > best_utility {
                best_utility = util;
                best_idx = idx;
            }
        }

        Ok((best_idx, best_utility))
    }

    /// Calculate certainty equivalent wealth
    pub fn certainty_equivalent(
        &self,
        outcomes: &[BranchOutcome],
    ) -> Result<f64, MeasureWeightedUtilityError> {
        let expected_utility = self.calculate_expected_utility(outcomes)?;

        // Invert utility function to get certainty equivalent
        let ce = if (self.utility.gamma - 1.0).abs() < 1e-10 {
            // Log utility: CE = exp(EU)
            expected_utility.exp()
        } else {
            // CRRA: CE = ((1-γ) * EU)^(1/(1-γ))
            let gamma = self.utility.gamma;
            ((1.0 - gamma) * expected_utility).powf(1.0 / (1.0 - gamma))
        };

        if ce.is_nan() || ce.is_infinite() {
            return Err(MeasureWeightedUtilityError::NumericalOverflow {
                message: "Certainty equivalent calculation overflow",
            });
        }

        Ok(ce)
    }

    /// Detect and flag catastrophic branches
    pub fn detect_catastrophic_branches(
        &self,
        outcomes: &[BranchOutcome],
    ) -> Result<Vec<usize>, MeasureWeightedUtilityError> {
        let mut catastrophic = Vec::new();

        for outcome in outcomes {
            let loss_ratio = if outcome.wealth > 0.0 {
                (1.0 - outcome.wealth / 1.0).abs()
            } else {
                1.0
            };

            if loss_ratio > self.catastrophic_threshold {
                catastrophic.push(outcome.branch_id);
            }
        }

        Ok(catastrophic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crra_utility() {
        let util_fn = CrraUtility::new(2.0, 0.01).unwrap();

        // U(c) = c^(-1) / (-1) = -1/c for γ=2
        let u = util_fn.utility(1.0).unwrap();
        assert!((u - (-1.0)).abs() < 1e-10);

        let u = util_fn.utility(2.0).unwrap();
        assert!((u - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_log_utility() {
        let util_fn = CrraUtility::new(1.0, 0.01).unwrap();

        let u = util_fn.utility(1.0).unwrap();
        assert!(u.abs() < 1e-10); // ln(1) = 0

        let u = util_fn.utility(std::f64::consts::E).unwrap();
        assert!((u - 1.0).abs() < 1e-10); // ln(e) = 1
    }

    #[test]
    fn test_measure_weighted_utility() {
        let calc = MeasureWeightedUtilityCalculator::new(2.0, 0.01, 0.5, 10.0).unwrap();

        let outcomes = vec![
            BranchOutcome {
                branch_id: 0,
                wealth: 1.2,
                measure: 0.7,
                is_catastrophic: false,
            },
            BranchOutcome {
                branch_id: 1,
                wealth: 0.3,
                measure: 0.3,
                is_catastrophic: true,
            },
        ];

        let eu = calc.calculate_expected_utility(&outcomes).unwrap();
        assert!(eu.is_finite());
    }

    #[test]
    fn test_catastrophe_detection() {
        let calc = MeasureWeightedUtilityCalculator::new(2.0, 0.01, 0.5, 10.0).unwrap();

        let outcomes = vec![
            BranchOutcome {
                branch_id: 0,
                wealth: 1.0,
                measure: 0.8,
                is_catastrophic: false,
            },
            BranchOutcome {
                branch_id: 1,
                wealth: 0.3,
                measure: 0.2,
                is_catastrophic: false,
            },
        ];

        let catastrophic = calc.detect_catastrophic_branches(&outcomes).unwrap();
        assert_eq!(catastrophic.len(), 1);
        assert_eq!(catastrophic[0], 1);
    }
}
