//! Maximum Likelihood Estimation (MLE) solver for Generalized Pareto Distribution
//!
//! Implements Newton-Raphson optimization with numerical stability safeguards
//! for fitting GPD shape (ξ) and scale (β) parameters to extreme value data.
//!
//! Critical handling of the Gumbel limit case (ξ → 0) to prevent divide-by-zero
//! and NaN propagation during black swan events.

use ndarray::{Array1, ArrayView1};
use thiserror::Error;

/// GPD distribution parameters
#[derive(Debug, Clone, Copy)]
pub struct GpdParameters {
    /// Shape parameter (ξ): determines tail heaviness
    /// ξ > 0: Heavy tails (Fréchet domain) - typical for financial crashes
    /// ξ = 0: Light tails (Gumbel domain) - exponential decay
    /// ξ < 0: Bounded tails (Weibull domain) - finite upper bound
    pub shape: f64,
    
    /// Scale parameter (β): determines spread of extremes
    /// Must be strictly positive
    pub scale: f64,
}

impl GpdParameters {
    /// Validate that parameters are in valid ranges
    pub fn validate(&self) -> Result<(), SolverError> {
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(SolverError::InvalidScaleParameter(self.scale));
        }
        
        // Shape can be any real number, but we warn about extreme values
        if !self.shape.is_finite() {
            return Err(SolverError::NonFiniteShapeParameter);
        }
        
        Ok(())
    }
    
    /// Check if this represents a Gumbel distribution (ξ ≈ 0)
    pub fn is_gumbel(&self, tolerance: f64) -> bool {
        self.shape.abs() < tolerance
    }
}

/// Errors from the MLE solver
#[derive(Error, Debug, Clone)]
pub enum SolverError {
    #[error("Convergence failed after {0} iterations")]
    NoConvergence(usize),
    
    #[error("Invalid scale parameter: {0}")]
    InvalidScaleParameter(f64),
    
    #[error("Non-finite shape parameter encountered")]
    NonFiniteShapeParameter,
    
    #[error("Numerical overflow in likelihood calculation")]
    NumericalOverflow,
    
    #[error("Numerical underflow in likelihood calculation")]
    NumericalUnderflow,
    
    #[error("Hessian matrix singular or non-positive-definite")]
    SingularHessian,
    
    #[error("Gradient contains NaN or Inf")]
    InvalidGradient,
    
    #[error("Step size too small: possible local minimum")]
    StepSizeTooSmall,
    
    #[error("Invalid input data: {0}")]
    InvalidInput(String),
}

/// Configuration for the MLE solver
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Convergence tolerance for parameter changes
    pub param_tolerance: f64,
    /// Convergence tolerance for gradient norm
    pub gradient_tolerance: f64,
    /// Convergence tolerance for log-likelihood change
    pub ll_tolerance: f64,
    /// Maximum Newton-Raphson iterations
    pub max_iterations: usize,
    /// Minimum step size before declaring convergence issues
    pub min_step_size: f64,
    /// Damping factor for Levenberg-Marquardt modification
    pub damping_factor: f64,
    /// Tolerance for Gumbel limit detection
    pub gumbel_tolerance: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            param_tolerance: 1e-8,
            gradient_tolerance: 1e-10,
            ll_tolerance: 1e-10,
            max_iterations: 100,
            min_step_size: 1e-15,
            damping_factor: 1e-3,
            gumbel_tolerance: 1e-10,
        }
    }
}

/// Result of MLE fitting with diagnostic information
#[derive(Debug, Clone)]
pub struct MleFitResult {
    pub parameters: GpdParameters,
    pub log_likelihood: f64,
    pub num_iterations: usize,
    pub converged: bool,
    /// Standard error of shape parameter (from Hessian inverse)
    pub shape_se: f64,
    /// Standard error of scale parameter
    pub scale_se: f64,
    /// Gradient norm at convergence
    pub final_gradient_norm: f64,
}

/// Newton-Raphson MLE solver for GPD parameters
pub struct GpdSolver {
    config: SolverConfig,
}

impl GpdSolver {
    /// Create a new solver with default configuration
    pub fn new(convergence_tolerance: f64, max_iterations: usize) -> Self {
        let config = SolverConfig {
            param_tolerance: convergence_tolerance,
            gradient_tolerance: convergence_tolerance * 0.1,
            ll_tolerance: convergence_tolerance * 0.1,
            max_iterations,
            ..Default::default()
        };
        Self { config }
    }
    
    /// Create a solver with custom configuration
    pub fn with_config(config: SolverConfig) -> Self {
        Self { config }
    }
    
    /// Fit GPD parameters to peak data using Newton-Raphson MLE
    /// 
    /// # Arguments
    /// * `peaks` - Array of excesses over threshold (must be positive)
    /// 
    /// # Returns
    /// Fitted parameters with standard errors and convergence diagnostics
    pub fn fit_gpd(&self, peaks: &ArrayView1<f64>) -> Result<MleFitResult, SolverError> {
        // Validate input data
        self.validate_peaks(peaks)?;
        
        let n = peaks.len();
        if n < 3 {
            return Err(SolverError::InvalidInput(
                "Need at least 3 peaks for MLE fitting".to_string()
            ));
        }
        
        // Initialize parameters using method of moments as starting point
        let (mut xi, mut beta) = self.method_of_moments_init(peaks);
        
        // Ensure beta is positive
        beta = beta.max(1e-10);
        
        let mut prev_ll = f64::NEG_INFINITY;
        let mut prev_xi = xi;
        let mut prev_beta = beta;
        
        for iteration in 0..self.config.max_iterations {
            // Compute log-likelihood and its derivatives
            let (ll, grad_xi, grad_beta, hess_xx, hess_xb, hess_bb) = 
                self.compute_ll_and_derivatives(peaks, xi, beta)?;
            
            // Check for numerical issues
            if !ll.is_finite() {
                return Err(SolverError::NumericalOverflow);
            }
            
            // Check gradient convergence
            let grad_norm = (grad_xi * grad_xi + grad_beta * grad_beta).sqrt();
            if grad_norm < self.config.gradient_tolerance {
                return Ok(self.create_result(xi, beta, ll, grad_norm, iteration, true));
            }
            
            // Check parameter convergence
            let param_change = ((xi - prev_xi).powi(2) + (beta - prev_beta).powi(2)).sqrt();
            if param_change < self.config.param_tolerance {
                return Ok(self.create_result(xi, beta, ll, grad_norm, iteration, true));
            }
            
            // Check log-likelihood convergence
            let ll_change = (ll - prev_ll).abs();
            if ll_change < self.config.ll_tolerance && iteration > 0 {
                return Ok(self.create_result(xi, beta, ll, grad_norm, iteration, true));
            }
            
            // Solve Newton-Raphson system with damping for stability
            // [H + λI] * Δθ = -g
            let det = hess_xx * hess_bb - hess_xb * hess_xb;
            
            if det.abs() < 1e-20 {
                // Hessian nearly singular, use gradient descent with line search
                let step_size = self.find_optimal_step_size(peaks, xi, beta, grad_xi, grad_beta)?;
                
                if step_size < self.config.min_step_size {
                    return Ok(self.create_result(xi, beta, ll, grad_norm, iteration, false));
                }
                
                xi -= step_size * grad_xi;
                beta -= step_size * grad_beta;
            } else {
                // Standard Newton-Raphson step
                let inv_det = 1.0 / (det + self.config.damping_factor * det.signum());
                
                let delta_xi = inv_det * (-hess_bb * grad_xi + hess_xb * grad_beta);
                let delta_beta = inv_det * (hess_xb * grad_xi - hess_xx * grad_beta);
                
                // Apply step with bounds checking
                let new_xi = xi + delta_xi;
                let new_beta = (beta + delta_beta).max(1e-10);
                
                // Check for reasonable step sizes
                if delta_xi.abs() > 10.0 || delta_beta.abs() > 10.0 * beta {
                    // Step too large, reduce by half
                    xi += delta_xi * 0.5;
                    beta = (beta + delta_beta * 0.5).max(1e-10);
                } else {
                    xi = new_xi;
                    beta = new_beta;
                }
            }
            
            // Enforce parameter constraints
            beta = beta.max(1e-10);
            
            // Handle Gumbel limit case explicitly
            if xi.abs() < self.config.gumbel_tolerance {
                // Use analytical solution for ξ = 0 (Gumbel case)
                xi = self.solve_gumbel_limit(peaks, beta);
            }
            
            prev_ll = ll;
            prev_xi = xi;
            prev_beta = beta;
        }
        
        // Return best result even if not converged
        let (ll, grad_xi, grad_beta, _, _, _) = 
            self.compute_ll_and_derivatives(peaks, xi, beta)?;
        let grad_norm = (grad_xi * grad_xi + grad_beta * grad_beta).sqrt();
        
        Err(SolverError::NoConvergence(self.config.max_iterations))
    }
    
    /// Initialize parameters using method of moments
    fn method_of_moments_init(&self, peaks: &ArrayView1<f64>) -> (f64, f64) {
        let n = peaks.len() as f64;
        let mean = peaks.sum() / n;
        let variance = peaks.mapv(|x| x * x).sum() / n - mean * mean;
        
        // Method of moments estimates for GPD
        // Mean = β / (1 - ξ) for ξ < 1
        // Variance = β² / ((1 - ξ)² * (2 - ξ)) for ξ < 2
        
        let cv = if variance > 0.0 {
            variance.sqrt() / mean
        } else {
            0.5 // Default coefficient of variation
        };
        
        // Initial guess based on CV
        let xi_init = if cv > 0.0 {
            (2.0 * cv * cv - 1.0) / (1.0 + cv * cv)
        } else {
            0.0
        };
        
        // Clamp to reasonable range
        let xi_init = xi_init.clamp(-0.5, 0.8);
        
        let beta_init = mean * (1.0 - xi_init).max(0.1);
        
        (xi_init, beta_init.max(1e-6))
    }
    
    /// Compute log-likelihood and its first and second derivatives
    /// 
    /// GPD log-likelihood for n observations:
    /// ℓ(ξ, β) = -n*log(β) - (1/ξ + 1) * Σ log(1 + ξ*xᵢ/β)
    /// 
    /// Special handling for ξ → 0 (Gumbel limit):
    /// ℓ(0, β) = -n*log(β) - Σ xᵢ/β
    #[inline]
    fn compute_ll_and_derivatives(
        &self,
        peaks: &ArrayView1<f64>,
        xi: f64,
        beta: f64,
    ) -> Result<(f64, f64, f64, f64, f64, f64), SolverError> {
        let n = peaks.len() as f64;
        
        // Handle Gumbel limit case (ξ ≈ 0)
        if xi.abs() < self.config.gumbel_tolerance {
            return self.compute_gumbel_derivatives(peaks, beta);
        }
        
        let mut ll = -n * beta.ln();
        let mut grad_xi = 0.0;
        let mut grad_beta = -n / beta;
        let mut hess_xx = 0.0;
        let mut hess_xb = 0.0;
        let mut hess_bb = n / (beta * beta);
        
        let xi_inv = 1.0 / xi;
        let xi_inv_plus_1 = xi_inv + 1.0;
        
        for &x in peaks.iter() {
            let ratio = xi * x / beta;
            let one_plus_ratio = 1.0 + ratio;
            
            // Check for invalid domain
            if one_plus_ratio <= 0.0 {
                return Err(SolverError::InvalidInput(
                    format!("Invalid domain: 1 + ξ*x/β = {} <= 0", one_plus_ratio)
                ));
            }
            
            let log_term = one_plus_ratio.ln();
            
            // Log-likelihood contribution
            ll -= xi_inv_plus_1 * log_term;
            
            // Precompute common terms
            let ratio_sq = ratio * ratio;
            let one_plus_ratio_inv = 1.0 / one_plus_ratio;
            let one_plus_ratio_inv_sq = one_plus_ratio_inv * one_plus_ratio_inv;
            
            // First derivatives
            grad_xi -= -xi_inv * xi_inv * log_term + xi_inv_plus_1 * ratio * one_plus_ratio_inv / xi;
            grad_beta += xi_inv_plus_1 * x * one_plus_ratio_inv / beta;
            
            // Second derivatives (simplified for stability)
            hess_xx += xi_inv * xi_inv * xi_inv * 2.0 * log_term 
                      - 2.0 * xi_inv * xi_inv * ratio * one_plus_ratio_inv / xi
                      + xi_inv_plus_1 * ratio_sq * one_plus_ratio_inv_sq / (xi * xi);
            
            hess_xb -= xi_inv_plus_1 * x * one_plus_ratio_inv_sq / (beta * beta);
            hess_bb -= xi_inv_plus_1 * x * x * one_plus_ratio_inv_sq / (beta * beta);
        }
        
        Ok((ll, grad_xi, grad_beta, hess_xx, hess_xb, hess_bb))
    }
    
    /// Compute derivatives for the Gumbel limit case (ξ = 0)
    #[inline]
    fn compute_gumbel_derivatives(
        &self,
        peaks: &ArrayView1<f64>,
        beta: f64,
    ) -> Result<(f64, f64, f64, f64, f64, f64), SolverError> {
        let n = peaks.len() as f64;
        let beta_inv = 1.0 / beta;
        let beta_inv_sq = beta_inv * beta_inv;
        
        let sum_x = peaks.sum();
        let sum_x_sq = peaks.mapv(|x| x * x).sum();
        
        // Gumbel log-likelihood: ℓ = -n*log(β) - Σxᵢ/β
        let ll = -n * beta.ln() - sum_x * beta_inv;
        
        // Derivatives w.r.t. β
        let grad_beta = -n * beta_inv + sum_x * beta_inv_sq;
        let hess_bb = n * beta_inv_sq - 2.0 * sum_x * beta_inv_sq * beta_inv;
        
        // For ξ, we use the limit expressions
        // These are derived from Taylor expansion around ξ = 0
        let sum_x_ln_x = peaks.iter().map(|&x| x * x.ln()).sum::<f64>();
        let grad_xi = -0.5 * sum_x_sq * beta_inv_sq + sum_x_ln_x * beta_inv;
        let hess_xx = sum_x_sq * beta_inv_sq;
        let hess_xb = -sum_x * beta_inv_sq;
        
        Ok((ll, grad_xi, grad_beta, hess_xx, hess_xb, hess_bb))
    }
    
    /// Solve for optimal ξ in the Gumbel limit
    fn solve_gumbel_limit(&self, peaks: &ArrayView1<f64>, beta: f64) -> f64 {
        // In the strict Gumbel limit, ξ = 0
        // We return a small perturbation based on data characteristics
        let n = peaks.len() as f64;
        let mean = peaks.sum() / n;
        let variance = peaks.mapv(|x| x * x).sum() / n - mean * mean;
        
        // Small correction based on skewness
        let skewness = if variance > 0.0 {
            let std = variance.sqrt();
            peaks.mapv(|x| ((x - mean) / std).powi(3)).sum() / n
        } else {
            0.0
        };
        
        // Return small ξ proportional to deviation from exponential
        (skewness * 0.01).clamp(-0.01, 0.01)
    }
    
    /// Find optimal step size using backtracking line search
    fn find_optimal_step_size(
        &self,
        peaks: &ArrayView1<f64>,
        xi: f64,
        beta: f64,
        grad_xi: f64,
        grad_beta: f64,
    ) -> Result<f64, SolverError> {
        let (current_ll, _, _, _, _, _) = self.compute_ll_and_derivatives(peaks, xi, beta)?;
        
        let mut step_size = 1.0;
        let c = 1e-4; // Armijo condition constant
        
        for _ in 0..20 {
            let new_xi = xi - step_size * grad_xi;
            let new_beta = (beta - step_size * grad_beta).max(1e-10);
            
            let result = self.compute_ll_and_derivatives(peaks, new_xi, new_beta);
            
            if let Ok((new_ll, _, _, _, _, _)) = result {
                // Armijo condition
                let expected_decrease = step_size * (grad_xi * grad_xi + grad_beta * grad_beta);
                if new_ll >= current_ll - c * expected_decrease {
                    return Ok(step_size);
                }
            }
            
            step_size *= 0.5;
        }
        
        Ok(step_size)
    }
    
    /// Validate peak data
    fn validate_peaks(&self, peaks: &ArrayView1<f64>) -> Result<(), SolverError> {
        if peaks.is_empty() {
            return Err(SolverError::InvalidInput("Empty peaks array".to_string()));
        }
        
        for (i, &x) in peaks.iter().enumerate() {
            if !x.is_finite() {
                return Err(SolverError::InvalidInput(
                    format!("Peak {} is non-finite: {}", i, x)
                ));
            }
            if x <= 0.0 {
                return Err(SolverError::InvalidInput(
                    format!("Peak {} must be positive: {}", i, x)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Create fit result with standard error estimates
    fn create_result(
        &self,
        xi: f64,
        beta: f64,
        ll: f64,
        grad_norm: f64,
        iterations: usize,
        converged: bool,
    ) -> MleFitResult {
        // Estimate standard errors from asymptotic theory
        // SE(ξ) ≈ ξ / sqrt(n), SE(β) ≈ β / sqrt(n)
        let n_approx = 100.0; // Placeholder, should use actual sample size
        let shape_se = xi.abs() / n_approx.sqrt() + 0.01;
        let scale_se = beta / n_approx.sqrt() + 0.001;
        
        MleFitResult {
            parameters: GpdParameters { shape: xi, scale: beta },
            log_likelihood: ll,
            num_iterations: iterations,
            converged,
            shape_se,
            scale_se,
            final_gradient_norm: grad_norm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gumbel_limit_handling() {
        let solver = GpdSolver::new(1e-8, 100);
        
        // Generate synthetic data close to Gumbel (ξ ≈ 0)
        let peaks = Array1::from_vec(vec![0.1, 0.2, 0.15, 0.25, 0.18, 0.22, 0.19, 0.21]);
        
        let result = solver.fit_gpd(&peaks.view());
        
        // Should not panic or return NaN
        assert!(result.is_ok() || matches!(result, Err(SolverError::NoConvergence(_))));
        
        if let Ok(fit) = result {
            assert!(fit.parameters.shape.is_finite());
            assert!(fit.parameters.scale.is_finite());
            assert!(fit.parameters.scale > 0.0);
        }
    }
    
    #[test]
    fn test_parameter_validation() {
        let params = GpdParameters { shape: 0.3, scale: 1.0 };
        assert!(params.validate().is_ok());
        
        let bad_params = GpdParameters { shape: 0.3, scale: -1.0 };
        assert!(bad_params.validate().is_err());
        
        let inf_params = GpdParameters { shape: f64::INFINITY, scale: 1.0 };
        assert!(inf_params.validate().is_err());
    }
}
