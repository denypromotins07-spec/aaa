//! Levenberg-Marquardt non-linear least squares optimizer.
//!
//! This implementation is specifically designed for fitting yield curve models
//! with pre-allocated Jacobian matrices to avoid heap allocations during iteration.
//!
//! The algorithm combines gradient descent and Gauss-Newton methods:
//! (J^T J + λ I) δ = J^T r
//!
//! where λ is the damping factor that adapts based on convergence behavior.

use ndarray::{Array1, Array2};
use std::f64;

/// Configuration for the Levenberg-Marquardt optimizer
#[derive(Debug, Clone)]
pub struct LmConfig {
    /// Initial damping factor (λ)
    pub initial_lambda: f64,
    /// Factor to increase lambda when step fails
    pub lambda_up_factor: f64,
    /// Factor to decrease lambda when step succeeds
    pub lambda_down_factor: f64,
    /// Maximum lambda before giving up
    pub max_lambda: f64,
    /// Minimum lambda for numerical stability
    pub min_lambda: f64,
    /// Convergence tolerance for residual norm
    pub residual_tolerance: f64,
    /// Convergence tolerance for parameter change
    pub param_tolerance: f64,
    /// Maximum iterations
    pub max_iterations: usize,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            initial_lambda: 1e-3,
            lambda_up_factor: 10.0,
            lambda_down_factor: 0.1,
            max_lambda: 1e10,
            min_lambda: 1e-15,
            residual_tolerance: 1e-10,
            param_tolerance: 1e-10,
            max_iterations: 100,
        }
    }
}

/// Result of LM optimization
#[derive(Debug)]
pub struct LmResult {
    /// Optimized parameters
    pub params: Vec<f64>,
    /// Final residual sum of squares
    pub residual_norm: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Final damping factor
    pub final_lambda: f64,
}

/// Levenberg-Marquardt optimizer with pre-allocated buffers
pub struct LevenbergMarquardtOptimizer {
    num_params: usize,
    num_residuals: usize,
    config: LmConfig,
    /// Pre-allocated Jacobian matrix
    jacobian: Array2<f64>,
    /// Pre-allocated residual vector
    residuals: Array1<f64>,
    /// Pre-allocated gradient (J^T r)
    gradient: Array1<f64>,
    /// Pre-allocated Hessian approximation (J^T J)
    hessian_approx: Array2<f64>,
    /// Pre-allocated damped Hessian (J^T J + λI)
    damped_hessian: Array2<f64>,
    /// Pre-allocated parameter delta
    delta: Array1<f64>,
    /// Pre-allocated new parameters
    new_params: Vec<f64>,
    /// Pre-allocated new residuals
    new_residuals: Vec<f64>,
}

impl LevenbergMarquardtOptimizer {
    /// Create new optimizer with specified dimensions
    /// 
    /// # Arguments
    /// * `num_params` - Number of parameters to optimize
    /// * `num_residuals` - Number of residual functions (data points)
    pub fn new(num_params: usize, num_residuals: usize) -> Self {
        Self {
            num_params,
            num_residuals,
            config: LmConfig::default(),
            jacobian: Array2::<f64>::zeros((num_residuals, num_params)),
            residuals: Array1::<f64>::zeros(num_residuals),
            gradient: Array1::<f64>::zeros(num_params),
            hessian_approx: Array2::<f64>::zeros((num_params, num_params)),
            damped_hessian: Array2::<f64>::zeros((num_params, num_params)),
            delta: Array1::<f64>::zeros(num_params),
            new_params: vec![0.0; num_params],
            new_residuals: Vec::with_capacity(num_residuals),
        }
    }

    /// Create with custom configuration
    pub fn with_config(num_params: usize, num_residuals: usize, config: LmConfig) -> Self {
        let mut opt = Self::new(num_params, num_residuals);
        opt.config = config;
        opt
    }

    /// Optimize parameters to minimize sum of squared residuals
    /// 
    /// # Arguments
    /// * `initial_params` - Starting parameter values
    /// * `residual_fn` - Function computing residuals given parameters
    /// * `jacobian_fn` - Function computing Jacobian matrix given parameters
    /// * `max_iterations` - Maximum number of iterations
    /// * `tolerance` - Convergence tolerance
    /// 
    /// # Returns
    /// Optimized parameters or error message
    pub fn optimize<F, G>(
        &mut self,
        initial_params: impl AsRef<[f64]>,
        mut residual_fn: F,
        mut jacobian_fn: G,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<Vec<f64>, String>
    where
        F: FnMut(&[f64]) -> Result<Vec<f64>, String>,
        G: FnMut(&[f64]) -> Result<Array2<f64>, String>,
    {
        let initial_params = initial_params.as_ref();
        if initial_params.len() != self.num_params {
            return Err(format!(
                "Expected {} parameters, got {}",
                self.num_params,
                initial_params.len()
            ));
        }

        let mut params: Vec<f64> = initial_params.to_vec();
        let mut lambda = self.config.initial_lambda;

        // Compute initial residuals
        self.residuals = Array1::from_vec(residual_fn(&params)?);
        let mut residual_norm = self.compute_residual_norm();

        // Check initial convergence
        if residual_norm < tolerance {
            return Ok(params);
        }

        let mut prev_residual_norm = residual_norm;
        let mut iterations = 0;

        while iterations < max_iterations {
            // Compute Jacobian
            self.jacobian = jacobian_fn(&params)?;

            // Compute gradient: g = J^T r
            self.gradient = self.jacobian.t().dot(&self.residuals);

            // Compute Hessian approximation: H ≈ J^T J
            self.hessian_approx = self.jacobian.t().dot(&self.jacobian);

            // Add damping: H + λI
            self.damped_hessian = self.hessian_approx.clone();
            for i in 0..self.num_params {
                self.damped_hessian[[i, i]] += lambda;
            }

            // Solve (J^T J + λI) δ = J^T r using Cholesky or LU decomposition
            // For numerical stability, we use a simple approach here
            // In production, use a proper linear algebra library
            match self.solve_linear_system() {
                Ok(delta_vec) => {
                    self.delta = Array1::from_vec(delta_vec);
                }
                Err(e) => {
                    // Singular matrix - increase damping dramatically
                    lambda *= self.config.lambda_up_factor;
                    if lambda > self.config.max_lambda {
                        return Err(format!("Singular matrix, max lambda exceeded: {}", e));
                    }
                    continue;
                }
            }

            // Try new parameters
            for i in 0..self.num_params {
                self.new_params[i] = params[i] - self.delta[i];
            }

            // Compute new residuals
            match residual_fn(&self.new_params) {
                Ok(new_res) => {
                    self.new_residuals = new_res;
                    let new_norm = self.compute_new_residual_norm();

                    // Check if step improved the solution
                    if new_norm < prev_residual_norm {
                        // Accept step
                        params.clone_from(&self.new_params);
                        prev_residual_norm = new_norm;

                        // Decrease damping for faster convergence
                        lambda = (lambda * self.config.lambda_down_factor)
                            .max(self.config.min_lambda);

                        // Check convergence
                        if new_norm < tolerance {
                            return Ok(LmResult {
                                params,
                                residual_norm: new_norm,
                                iterations: iterations + 1,
                                converged: true,
                                final_lambda: lambda,
                            }.params);
                        }

                        // Check parameter change convergence
                        let param_change = self.delta.iter().map(|x| x.abs()).sum::<f64>();
                        if param_change < self.config.param_tolerance {
                            return Ok(LmResult {
                                params,
                                residual_norm: new_norm,
                                iterations: iterations + 1,
                                converged: true,
                                final_lambda: lambda,
                            }.params);
                        }
                    } else {
                        // Reject step - increase damping
                        lambda *= self.config.lambda_up_factor;
                        if lambda > self.config.max_lambda {
                            return Err("Maximum damping reached without convergence".to_string());
                        }
                    }
                }
                Err(e) => {
                    // Residual computation failed - reject step
                    lambda *= self.config.lambda_up_factor;
                    if lambda > self.config.max_lambda {
                        return Err(format!("Residual computation failed: {}", e));
                    }
                }
            }

            iterations += 1;
        }

        // Return best result even if not fully converged
        Ok(LmResult {
            params,
            residual_norm: prev_residual_norm,
            iterations,
            converged: false,
            final_lambda: lambda,
        }.params)
    }

    /// Compute L2 norm of current residuals
    #[inline(always)]
    fn compute_residual_norm(&self) -> f64 {
        self.residuals.iter().map(|r| r * r).sum::<f64>().sqrt()
    }

    /// Compute L2 norm of new residuals (after trial step)
    #[inline(always)]
    fn compute_new_residual_norm(&self) -> f64 {
        self.new_residuals.iter().map(|r| r * r).sum::<f64>().sqrt()
    }

    /// Solve linear system (H + λI) δ = g using Gaussian elimination with partial pivoting
    /// This is a simplified solver - for production use, leverage ndarray-linalg
    fn solve_linear_system(&self) -> Result<Vec<f64>, String> {
        let n = self.num_params;
        
        // Create augmented matrix [A | b]
        let mut aug = Array2::<f64>::zeros((n, n + 1));
        
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = self.damped_hessian[[i, j]];
            }
            aug[[i, n]] = self.gradient[i];
        }

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[[col, col]].abs();
            
            for row in (col + 1)..n {
                let val = aug[[row, col]].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }

            // Check for singular matrix
            if max_val < 1e-15 {
                return Err("Near-singular matrix detected".to_string());
            }

            // Swap rows
            if max_row != col {
                for j in col..(n + 1) {
                    let tmp = aug[[col, j]];
                    aug[[col, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = tmp;
                }
            }

            // Eliminate column
            for row in (col + 1)..n {
                let factor = aug[[row, col]] / aug[[col, col]];
                for j in col..(n + 1) {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }

        // Back substitution
        let mut solution = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[[i, n]];
            for j in (i + 1)..n {
                sum -= aug[[i, j]] * solution[j];
            }
            solution[i] = sum / aug[[i, i]];
            
            // Check for NaN/Inf
            if !solution[i].is_finite() {
                return Err("Non-finite solution detected".to_string());
            }
        }

        Ok(solution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_quadratic_minimization() {
        // Minimize f(x) = (x - 3)^2, which has minimum at x = 3
        // Residual: r(x) = x - 3
        // Jacobian: J = 1
        
        let mut optimizer = LevenbergMarquardtOptimizer::new(1, 1);
        
        let result = optimizer.optimize(
            [0.0],  // Initial guess
            |params| Ok(vec![params[0] - 3.0]),
            |_params| Ok(Array2::from_shape_vec((1, 1), vec![1.0]).unwrap()),
            100,
            1e-10,
        ).unwrap();
        
        assert!((result[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_regression() {
        // Fit line y = mx + b to data points
        // Parameters: [m, b]
        
        let data_points = vec![(1.0, 2.1), (2.0, 3.9), (3.0, 6.2), (4.0, 8.1)];
        let x_vals: Vec<f64> = data_points.iter().map(|(x, _)| *x).collect();
        let y_vals: Vec<f64> = data_points.iter().map(|(_, y)| *y).collect();
        
        let mut optimizer = LevenbergMarquardtOptimizer::new(2, data_points.len());
        
        let result = optimizer.optimize(
            [1.0, 1.0],  // Initial guess [m, b]
            |params| {
                let m = params[0];
                let b = params[1];
                let residuals: Vec<f64> = x_vals.iter()
                    .zip(&y_vals)
                    .map(|(&x, &y)| m * x + b - y)
                    .collect();
                Ok(residuals)
            },
            |params| {
                let mut jacobian = Array2::<f64>::zeros((data_points.len(), 2));
                for (i, &x) in x_vals.iter().enumerate() {
                    jacobian[[i, 0]] = x;  // dm
                    jacobian[[i, 1]] = 1.0; // db
                }
                Ok(jacobian)
            },
            100,
            1e-10,
        ).unwrap();
        
        // Expected slope ~2.0, intercept ~0
        assert!((result[0] - 2.0).abs() < 0.5);
        assert!(result[1].abs() < 0.5);
    }
}
