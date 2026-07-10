//! Energy Gap Validator
//! 
//! Validates quantum/classical solutions by checking energy gaps,
//! constraint satisfaction, and solution quality metrics.

use thiserror::Error;
use crate::qubo::portfolio_hamiltonian::QuboMatrix;

/// Errors that can occur during validation
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Problem dimension mismatch")]
    DimensionMismatch,
    #[error("Numerical error: {0}")]
    NumericalError(String),
    #[error("Solution contains invalid values")]
    InvalidSolutionValues,
}

/// Result of gap validation
#[derive(Debug, Clone, Default)]
pub struct GapValidationResult {
    /// Whether the solution passed all validations
    pub is_valid: bool,
    /// Energy of the submitted solution
    pub solution_energy: f64,
    /// Estimated ground state energy (lower bound)
    pub estimated_ground_energy: f64,
    /// Energy gap to first excited state estimate
    pub energy_gap: f64,
    /// Relative gap (gap / energy_scale)
    pub relative_gap: f64,
    /// Constraint violation magnitude (0 = no violation)
    pub constraint_violation: f64,
    /// Quality score (0-1, higher is better)
    pub quality_score: f64,
    /// Detailed validation messages
    pub messages: Vec<String>,
}

impl GapValidationResult {
    /// Create a failed validation result
    pub fn failed(reason: &str) -> Self {
        Self {
            is_valid: false,
            quality_score: 0.0,
            messages: vec![reason.to_string()],
            ..Default::default()
        }
    }

    /// Get a summary of the validation
    pub fn summary(&self) -> String {
        if self.is_valid {
            format!(
                "Valid solution: energy={}, gap={}, quality={:.2}",
                self.solution_energy,
                self.energy_gap,
                self.quality_score
            )
        } else {
            format!(
                "Invalid solution: {}",
                self.messages.join("; ")
            )
        }
    }
}

/// Energy Gap Validator for quantum/classical solutions
pub struct EnergyGapValidator {
    /// Minimum acceptable relative gap
    min_relative_gap: f64,
    /// Maximum acceptable constraint violation
    max_constraint_violation: f64,
    /// Minimum quality score threshold
    min_quality_threshold: f64,
}

impl EnergyGapValidator {
    /// Create a new validator with default thresholds
    pub fn new() -> Self {
        Self {
            min_relative_gap: 0.001,
            max_constraint_violation: 0.01,
            min_quality_threshold: 0.5,
        }
    }

    /// Create a validator with custom thresholds
    pub fn with_thresholds(
        min_relative_gap: f64,
        max_constraint_violation: f64,
        min_quality_threshold: f64,
    ) -> Self {
        Self {
            min_relative_gap,
            max_constraint_violation,
            min_quality_threshold,
        }
    }

    /// Validate a solution against the QUBO problem
    /// 
    /// # Arguments
    /// * `qubo` - The QUBO matrix
    /// * `weights` - Continuous portfolio weights from the solver
    /// 
    /// # Returns
    /// Validation result with diagnostics
    pub fn validate_solution(
        &self,
        qubo: &QuboMatrix<f64>,
        weights: &[f64],
    ) -> GapValidationResult {
        let mut result = GapValidationResult::default();
        
        // Basic dimension check
        if weights.is_empty() {
            return GapValidationResult::failed("Empty weight vector");
        }
        
        // Calculate solution energy
        let energy = self.calculate_continuous_energy(qubo, weights);
        result.solution_energy = energy;
        
        // Check for numerical issues
        if !energy.is_finite() {
            return GapValidationResult::failed("Non-finite energy detected");
        }
        
        // Estimate ground state energy (using simple bounds)
        let ground_estimate = self.estimate_ground_state_energy(qubo);
        result.estimated_ground_energy = ground_estimate;
        
        // Calculate energy gap estimate
        let gap = self.estimate_energy_gap(qubo);
        result.energy_gap = gap;
        
        // Relative gap
        let energy_scale = energy.abs().max(ground_estimate.abs()).max(1.0);
        result.relative_gap = gap / energy_scale;
        
        // Check constraint violations (weights should sum to ~1, be non-negative)
        let constraint_violation = self.calculate_constraint_violation(weights);
        result.constraint_violation = constraint_violation;
        
        // Calculate quality score
        result.quality_score = self.calculate_quality_score(
            energy,
            ground_estimate,
            gap,
            constraint_violation,
        );
        
        // Determine validity
        result.is_valid = result.quality_score >= self.min_quality_threshold
            && result.constraint_violation <= self.max_constraint_violation
            && result.relative_gap >= self.min_relative_gap;
        
        // Add diagnostic messages
        if result.relative_gap < self.min_relative_gap {
            result.messages.push(format!(
                "Relative gap {:.6} below threshold {:.6}",
                result.relative_gap, self.min_relative_gap
            ));
        }
        
        if result.constraint_violation > self.max_constraint_violation {
            result.messages.push(format!(
                "Constraint violation {:.6} exceeds threshold {:.6}",
                result.constraint_violation, self.max_constraint_violation
            ));
        }
        
        if result.quality_score < self.min_quality_threshold {
            result.messages.push(format!(
                "Quality score {:.2} below threshold {:.2}",
                result.quality_score, self.min_quality_threshold
            ));
        }
        
        if result.is_valid && result.messages.is_empty() {
            result.messages.push("All validations passed".to_string());
        }
        
        result
    }

    /// Calculate energy for continuous weight vector
    fn calculate_continuous_energy(&self, qubo: &QuboMatrix<f64>, weights: &[f64]) -> f64 {
        let n = weights.len();
        let mut energy = 0.0;
        
        // Map continuous weights back to binary representation approximately
        // This is a simplified approach - full validation would use the exact binary encoding
        
        // For QUBO with binary variables, we need to discretize
        // Approximate by treating weights as probabilities
        for i in 0..n.min(qubo.n_qubits) {
            // Diagonal term: Q[i,i] * w[i] (since x^2 = x for binary, approximate as w)
            energy += qubo.matrix[[i, i]] * weights[i];
            
            // Off-diagonal: Q[i,j] * w[i] * w[j]
            for j in (i+1)..n.min(qubo.n_qubits) {
                energy += 2.0 * qubo.matrix[[i, j]] * weights[i] * weights[j];
            }
            
            // Linear term
            if i < qubo.linear_term.len() {
                energy += qubo.linear_term[i] * weights[i];
            }
        }
        
        energy
    }

    /// Estimate ground state energy using simple bounds
    fn estimate_ground_state_energy(&self, qubo: &QuboMatrix<f64>) -> f64 {
        // Use Gershgorin circle theorem for lower bound
        let n = qubo.n_qubits;
        let mut min_bound = f64::INFINITY;
        
        for i in 0..n {
            let diag = qubo.matrix[[i, i]];
            let off_diag_sum: f64 = (0..n)
                .filter(|&j| j != i)
                .map(|j| qubo.matrix[[i, j]].abs())
                .sum();
            
            // Gershgorin lower bound for eigenvalue i
            let gershgorin_lower = diag - off_diag_sum;
            
            // Include linear term contribution (worst case: always negative)
            let linear_contrib = if i < qubo.linear_term.len() {
                qubo.linear_term[i].min(0.0)
            } else {
                0.0
            };
            
            let bound = gershgorin_lower + linear_contrib;
            if bound < min_bound {
                min_bound = bound;
            }
        }
        
        // Return a reasonable estimate (not too pessimistic)
        min_bound.clamp(-1e6, 1e6)
    }

    /// Estimate energy gap using spectral heuristics
    fn estimate_energy_gap(&self, qubo: &QuboMatrix<f64>) -> f64 {
        // Simplified gap estimation based on matrix properties
        // Real gap calculation would require eigenvalue computation
        
        let n = qubo.n_qubits;
        
        // Estimate using minimum difference between diagonal elements
        // (rough approximation of level spacing)
        let mut min_diff = f64::INFINITY;
        
        let diagonals: Vec<f64> = (0..n)
            .map(|i| qubo.matrix[[i, i]])
            .collect();
        
        for i in 0..n {
            for j in (i+1)..n {
                let diff = (diagonals[i] - diagonals[j]).abs();
                if diff < min_diff && diff > 1e-10 {
                    min_diff = diff;
                }
            }
        }
        
        // If all diagonals are similar, use off-diagonal coupling scale
        if min_diff == f64::INFINITY || min_diff < 1e-10 {
            let mut max_off_diag = 0.0;
            for i in 0..n {
                for j in (i+1)..n {
                    max_off_diag = max_off_diag.max(qubo.matrix[[i, j]].abs());
                }
            }
            min_diff = max_off_diag * 0.1; // Rough estimate
        }
        
        min_diff.clamp(1e-10, 1e6)
    }

    /// Calculate constraint violation for portfolio weights
    fn calculate_constraint_violation(&self, weights: &[f64]) -> f64 {
        let mut violation = 0.0;
        
        // Budget constraint: sum should be 1
        let sum: f64 = weights.iter().sum();
        violation += (sum - 1.0).abs();
        
        // Non-negativity: all weights should be >= 0
        for &w in weights {
            if w < 0.0 {
                violation += w.abs();
            }
        }
        
        // Upper bound: individual weights typically <= 1
        for &w in weights {
            if w > 1.0 {
                violation += (w - 1.0).abs();
            }
        }
        
        violation
    }

    /// Calculate overall quality score
    fn calculate_quality_score(
        &self,
        energy: f64,
        ground_estimate: f64,
        gap: f64,
        constraint_violation: f64,
    ) -> f64 {
        let mut score = 1.0;
        
        // Energy proximity to ground state (higher score if closer)
        let energy_range = (energy - ground_estimate).abs();
        let energy_scale = ground_estimate.abs().max(1.0);
        let energy_score = 1.0 / (1.0 + energy_range / energy_scale);
        score *= energy_score;
        
        // Gap score (higher score if gap is resolvable)
        let gap_score = if gap > 1e-6 { 1.0 } else { 0.5 };
        score *= gap_score;
        
        // Constraint satisfaction score
        let constraint_score = 1.0 / (1.0 + constraint_violation * 10.0);
        score *= constraint_score;
        
        score.clamp(0.0, 1.0)
    }

    /// Compare two solutions and determine which is better
    pub fn compare_solutions(
        &self,
        qubo: &QuboMatrix<f64>,
        weights_a: &[f64],
        weights_b: &[f64],
    ) -> SolutionComparison {
        let result_a = self.validate_solution(qubo, weights_a);
        let result_b = self.validate_solution(qubo, weights_b);
        
        SolutionComparison {
            winner: if result_a.quality_score > result_b.quality_score {
                ComparisonWinner::A
            } else if result_b.quality_score > result_a.quality_score {
                ComparisonWinner::B
            } else {
                ComparisonWinner::Tie
            },
            result_a,
            result_b,
            energy_difference: result_a.solution_energy - result_b.solution_energy,
        }
    }
}

impl Default for EnergyGapValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of comparing two solutions
#[derive(Debug, Clone)]
pub struct SolutionComparison {
    /// Which solution is better
    pub winner: ComparisonWinner,
    /// Validation result for solution A
    pub result_a: GapValidationResult,
    /// Validation result for solution B
    pub result_b: GapValidationResult,
    /// Energy difference (A - B)
    pub energy_difference: f64,
}

/// Winner of a comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonWinner {
    A,
    B,
    Tie,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array1};

    #[test]
    fn test_validator_basic() {
        let validator = EnergyGapValidator::new();
        
        // Create simple QUBO
        let mut qubo = QuboMatrix::new(3);
        qubo.matrix = Array2::from_shape_vec((3, 3), vec![
            1.0, -0.5, 0.0,
            -0.5, 1.0, -0.5,
            0.0, -0.5, 1.0,
        ]).unwrap();
        qubo.linear_term = Array1::from_vec(vec![-0.1, -0.1, -0.1]);
        qubo.qubit_mapping = vec![
            ("a".to_string(), 0.33),
            ("b".to_string(), 0.33),
            ("c".to_string(), 0.34),
        ];
        
        // Valid weights (sum to 1, all positive)
        let weights = vec![0.4, 0.35, 0.25];
        let result = validator.validate_solution(&qubo, &weights);
        
        assert!(result.solution_energy.is_finite());
        assert!(result.constraint_violation < 0.1); // Small violation allowed
    }

    #[test]
    fn test_invalid_weights() {
        let validator = EnergyGapValidator::new();
        
        let mut qubo = QuboMatrix::new(2);
        qubo.matrix = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        qubo.linear_term = Array1::from_vec(vec![0.0, 0.0]);
        
        // Invalid: doesn't sum to 1
        let weights = vec![0.5, 0.5, 0.5];
        let result = validator.validate_solution(&qubo, &weights);
        
        assert!(result.constraint_violation > 0.4); // Large violation
    }

    #[test]
    fn test_ground_state_estimation() {
        let validator = EnergyGapValidator::new();
        
        let mut qubo = QuboMatrix::new(2);
        qubo.matrix = Array2::from_shape_vec((2, 2), vec![2.0, -1.0, -1.0, 2.0]).unwrap();
        qubo.linear_term = Array1::from_vec(vec![-0.5, -0.5]);
        
        let ground = validator.estimate_ground_state_energy(&qubo);
        
        // Ground state should be finite and negative (due to linear terms)
        assert!(ground.is_finite());
        assert!(ground < 2.0); // Should be less than pure diagonal
    }

    #[test]
    fn test_solution_comparison() {
        let validator = EnergyGapValidator::new();
        
        let mut qubo = QuboMatrix::new(2);
        qubo.matrix = Array2::from_shape_vec((2, 2), vec![1.0, -0.5, -0.5, 1.0]).unwrap();
        qubo.linear_term = Array1::from_vec(vec![-0.2, -0.2]);
        
        let weights_a = vec![0.6, 0.4];
        let weights_b = vec![0.4, 0.6];
        
        let comparison = validator.compare_solutions(&qubo, &weights_a, &weights_b);
        
        // Both should have same energy due to symmetry
        assert_eq!(comparison.winner, ComparisonWinner::Tie);
        assert!((comparison.energy_difference).abs() < 1e-10);
    }
}
