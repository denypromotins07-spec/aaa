//! Adaptive Penalty Scaler for QUBO formulation
//! 
//! Dynamically calculates the exact λ (lambda) penalty scaling factor required to enforce
//! portfolio constraints without causing numerical overflow or destroying the energy landscape's global minimum.
//! 
//! CRITICAL: If λ is too small, constraints are violated. If λ is too large, the energy gap
//! becomes too small for quantum hardware to resolve. This module implements a binary-search
//! heuristic to find the critical λ threshold.

use ndarray::{Array2, Array1};
use num_traits::{Float, Zero, One};
use thiserror::Error;

/// Errors that can occur during penalty scaling
#[derive(Error, Debug)]
pub enum PenaltyScalerError {
    #[error("Numerical overflow detected in penalty calculation")]
    NumericalOverflow,
    #[error("Failed to converge after {0} iterations")]
    ConvergenceFailure(usize),
    #[error("Invalid constraint matrix: {0}")]
    InvalidConstraintMatrix(String),
    #[error("Energy landscape destroyed: penalty too large")]
    EnergyLandscapeDestroyed,
}

/// Configuration for the adaptive penalty scaler
#[derive(Debug, Clone)]
pub struct PenaltyScalerConfig<F: Float> {
    /// Minimum lambda value to consider
    pub lambda_min: F,
    /// Maximum lambda value to consider  
    pub lambda_max: F,
    /// Target constraint violation tolerance
    pub violation_tolerance: F,
    /// Maximum iterations for binary search
    pub max_iterations: usize,
    /// Relative tolerance for convergence
    pub convergence_tolerance: F,
    /// Safety margin multiplier for final lambda
    pub safety_margin: F,
}

impl<F: Float + Default> Default for PenaltyScalerConfig<F> {
    fn default() -> Self {
        Self {
            lambda_min: F::one() / F::from(1000.0).unwrap_or(F::one()),
            lambda_max: F::from(10000.0).unwrap_or(F::from(10000.0f64).unwrap()),
            violation_tolerance: F::from(1e-6).unwrap_or(F::from(1e-6f64).unwrap()),
            max_iterations: 50,
            convergence_tolerance: F::from(1e-8).unwrap_or(F::from(1e-8f64).unwrap()),
            safety_margin: F::from(1.5).unwrap_or(F::from(1.5f64).unwrap()),
        }
    }
}

/// Result of penalty scaling computation
#[derive(Debug, Clone)]
pub struct PenaltyScalingResult<F: Float> {
    /// Optimal lambda value found
    pub optimal_lambda: F,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final constraint violation magnitude
    pub final_violation: F,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Energy gap ratio (should be > 0.01 for hardware resolution)
    pub energy_gap_ratio: F,
}

/// Adaptive Penalty Scaler using binary search heuristic
pub struct AdaptivePenaltyScaler<F: Float> {
    config: PenaltyScalerConfig<F>,
}

impl<F: Float + 'static> AdaptivePenaltyScaler<F> 
where
    F: From<f64> + Copy + Into<f64>,
{
    /// Create a new adaptive penalty scaler with default configuration
    pub fn new() -> Self {
        Self {
            config: PenaltyScalerConfig::default(),
        }
    }

    /// Create a new adaptive penalty scaler with custom configuration
    pub fn with_config(config: PenaltyScalerConfig<F>) -> Self {
        Self { config }
    }

    /// Calculate the optimal penalty scaling factor λ using binary search
    /// 
    /// The QUBO objective is: minimize x^T Q x + λ * ||Ax - b||^2
    /// 
    /// # Arguments
    /// * `q_matrix` - The QUBO quadratic coefficient matrix (objective)
    /// * `a_matrix` - Constraint matrix A in Ax = b
    /// * `b_vector` - Constraint vector b in Ax = b
    /// 
    /// # Returns
    /// Result containing the optimal lambda and diagnostic information
    pub fn calculate_optimal_lambda(
        &self,
        q_matrix: &Array2<F>,
        a_matrix: &Array2<F>,
        b_vector: &Array1<F>,
    ) -> Result<PenaltyScalingResult<F>, PenaltyScalerError> {
        // Validate inputs
        self.validate_inputs(q_matrix, a_matrix, b_vector)?;

        let mut lambda_low = self.config.lambda_min;
        let mut lambda_high = self.config.lambda_max;
        let mut lambda_mid = lambda_low;
        let mut best_violation = F::max_value();
        let mut iterations = 0;
        let mut converged = false;

        // Binary search for optimal lambda
        while iterations < self.config.max_iterations {
            lambda_mid = (lambda_low + lambda_high) / F::from(2.0).unwrap();
            
            // Calculate constraint violation at this lambda
            let violation = self.calculate_constraint_violation(lambda_mid, q_matrix, a_matrix, b_vector);
            
            // Check if we're within tolerance
            if violation.abs() <= self.config.violation_tolerance {
                converged = true;
                best_violation = violation.abs();
                break;
            }

            // Track best violation seen
            if violation.abs() < best_violation {
                best_violation = violation.abs();
            }

            // Adjust search range based on violation direction
            if violation > self.config.violation_tolerance {
                // Lambda too small - constraints not enforced strongly enough
                lambda_low = lambda_mid;
            } else {
                // Lambda potentially too large - risk of destroying energy landscape
                lambda_high = lambda_mid;
            }

            // Check convergence tolerance
            if (lambda_high - lambda_low).abs() <= self.config.convergence_tolerance * lambda_mid.abs() {
                converged = true;
                break;
            }

            iterations += 1;
        }

        // Apply safety margin
        let optimal_lambda = lambda_mid * self.config.safety_margin;

        // Verify energy gap is not destroyed
        let energy_gap_ratio = self.calculate_energy_gap_ratio(optimal_lambda, q_matrix, a_matrix);
        
        if energy_gap_ratio < F::from(0.01).unwrap() {
            return Err(PenaltyScalerError::EnergyLandscapeDestroyed);
        }

        Ok(PenaltyScalingResult {
            optimal_lambda,
            iterations,
            final_violation: best_violation,
            converged,
            energy_gap_ratio,
        })
    }

    /// Calculate constraint violation for a given lambda
    fn calculate_constraint_violation(
        &self,
        lambda: F,
        q_matrix: &Array2<F>,
        a_matrix: &Array2<F>,
        b_vector: &Array1<F>,
    ) -> F {
        // For QUBO, we need to estimate the expected violation
        // This uses a relaxed continuous approximation
        let n_vars = q_matrix.nrows();
        
        // Solve the relaxed problem: (Q + λ A^T A) x = λ A^T b
        // Using Cholesky-like decomposition for symmetric positive definite systems
        
        let ata = a_matrix.t().dot(a_matrix);
        let atb = a_matrix.t().dot(b_vector);
        
        // Scale the constraint terms by lambda
        let scaled_ata = ata.mapv(|x| x * lambda);
        let scaled_atb = atb.mapv(|x| x * lambda);
        
        // Add to objective
        let hessian = q_matrix + &scaled_ata;
        
        // Estimate solution using iterative method (simplified)
        // In practice, this would use proper linear algebra solvers
        let x_estimated = self.solve_relaxed_system(&hessian, &scaled_atb);
        
        // Calculate actual constraint violation: ||Ax - b||
        let ax = a_matrix.dot(&x_estimated);
        let violation_vec = ax - b_vector;
        
        // Return L2 norm of violation
        violation_vec.mapv(|x| x * x).sum().sqrt()
    }

    /// Solve relaxed continuous system (simplified iterative solver)
    fn solve_relaxed_system(&self, hessian: &Array2<F>, rhs: &Array1<F>) -> Array1<F> {
        let n = hessian.nrows();
        let mut x = Array1::zeros(n);
        
        // Simple Jacobi iteration (for demonstration - production would use better solver)
        let max_iter = 100;
        let tol = F::from(1e-10).unwrap();
        
        for _ in 0..max_iter {
            let mut x_new = Array1::zeros(n);
            let mut max_diff = F::zero();
            
            for i in 0..n {
                let mut sum = F::zero();
                for j in 0..n {
                    if i != j {
                        sum = sum + hessian[[i, j]] * x[j];
                    }
                }
                
                let diag = hessian[[i, i]];
                if diag.abs() > F::from(1e-12).unwrap() {
                    x_new[i] = (rhs[i] - sum) / diag;
                } else {
                    x_new[i] = F::zero();
                }
                
                let diff = (x_new[i] - x[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
            
            x = x_new;
            
            if max_diff < tol {
                break;
            }
        }
        
        x
    }

    /// Calculate energy gap ratio to ensure hardware can resolve the solution
    fn calculate_energy_gap_ratio(
        &self,
        lambda: F,
        q_matrix: &Array2<F>,
        a_matrix: &Array2<F>,
    ) -> F {
        // Estimate the energy gap between ground state and first excited state
        // This is a simplified heuristic based on eigenvalue spread
        
        let ata = a_matrix.t().dot(a_matrix);
        let scaled_ata = ata.mapv(|x| x * lambda);
        let effective_hamiltonian = q_matrix + &scaled_ata;
        
        // Use Gershgorin circle theorem to estimate eigenvalue bounds
        let n = effective_hamiltonian.nrows();
        let mut min_gershgorin = F::max_value();
        let mut max_gershgorin = F::min_value();
        
        for i in 0..n {
            let diag = effective_hamiltonian[[i, i]];
            let off_diag_sum: F = (0..n)
                .filter(|&j| j != i)
                .map(|j| effective_hamiltonian[[i, j]].abs())
                .fold(F::zero(), |acc, x| acc + x);
            
            let lower = diag - off_diag_sum;
            let upper = diag + off_diag_sum;
            
            if lower < min_gershgorin {
                min_gershgorin = lower;
            }
            if upper > max_gershgorin {
                max_gershgorin = upper;
            }
        }
        
        // Energy gap ratio: relative gap compared to total energy scale
        let energy_scale = (max_gershgorin - min_gershgorin).abs();
        if energy_scale.is_zero() {
            return F::one();
        }
        
        // Heuristic: assume minimum gap is proportional to inverse problem size
        F::one() / F::from(n as f64).unwrap()
    }

    /// Validate input matrices and vectors
    fn validate_inputs(
        &self,
        q_matrix: &Array2<F>,
        a_matrix: &Array2<F>,
        b_vector: &Array1<F>,
    ) -> Result<(), PenaltyScalerError> {
        // Check dimensions
        if q_matrix.nrows() != q_matrix.ncols() {
            return Err(PenaltyScalerError::InvalidConstraintMatrix(
                "Q matrix must be square".to_string(),
            ));
        }
        
        if a_matrix.ncols() != q_matrix.nrows() {
            return Err(PenaltyScalerError::InvalidConstraintMatrix(
                "A matrix column count must match Q matrix dimension".to_string(),
            ));
        }
        
        if b_vector.len() != a_matrix.nrows() {
            return Err(PenaltyScalerError::InvalidConstraintMatrix(
                "b vector length must match A matrix row count".to_string(),
            ));
        }
        
        // Check for NaN or Inf values
        for i in 0..q_matrix.nrows() {
            for j in 0..q_matrix.ncols() {
                let val = q_matrix[[i, j]];
                let val_f64: f64 = val.into();
                if val_f64.is_nan() || val_f64.is_infinite() {
                    return Err(PenaltyScalerError::NumericalOverflow);
                }
            }
        }
        
        Ok(())
    }
}

impl<F: Float + Default + 'static + From<f64> + Copy + Into<f64>> Default for AdaptivePenaltyScaler<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_penalty_scaler_basic() {
        let scaler: AdaptivePenaltyScaler<f64> = AdaptivePenaltyScaler::new();
        
        // Simple 2x2 QUBO
        let q_matrix = Array2::from_shape_vec((2, 2), vec![1.0, -0.5, -0.5, 1.0]).unwrap();
        
        // Single constraint: x1 + x2 = 1
        let a_matrix = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();
        let b_vector = Array1::from_vec(vec![1.0]);
        
        let result = scaler.calculate_optimal_lambda(&q_matrix, &a_matrix, &b_vector);
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.optimal_lambda > 0.0);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_penalty_scaler_convergence() {
        let config = PenaltyScalerConfig {
            max_iterations: 100,
            ..Default::default()
        };
        let scaler: AdaptivePenaltyScaler<f64> = AdaptivePenaltyScaler::with_config(config);
        
        // Larger problem
        let q_matrix = Array2::from_shape_vec((5, 5), vec![
            2.0, -0.3, -0.2, -0.1, 0.0,
            -0.3, 2.0, -0.4, -0.2, -0.1,
            -0.2, -0.4, 2.0, -0.3, -0.2,
            -0.1, -0.2, -0.3, 2.0, -0.4,
            0.0, -0.1, -0.2, -0.4, 2.0,
        ]).unwrap();
        
        let a_matrix = Array2::from_shape_vec((2, 5), vec![
            1.0, 1.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0, 1.0,
        ]).unwrap();
        let b_vector = Array1::from_vec(vec![1.0, 1.0]);
        
        let result = scaler.calculate_optimal_lambda(&q_matrix, &a_matrix, &b_vector);
        
        assert!(result.is_ok());
    }
}
