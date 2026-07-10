//! Nelson-Siegel-Svensson (NSS) yield curve model implementation.
//! 
//! The NSS model extends the Nelson-Siegel model with two additional parameters
//! to capture more complex yield curve shapes including humps and inflection points.
//! 
//! Yield(t) = β₀ + β₁*((1-exp(-t/τ₁))/(t/τ₁)) + β₂*((1-exp(-t/τ₁))/(t/τ₁) - exp(-t/τ₁))
//!          + β₃*((1-exp(-t/τ₂))/(t/τ₂) - exp(-t/τ₂))
//!
//! This module provides zero-allocation evaluation using pre-allocated buffers.

use crate::fixed_income::levenberg_marquardt::LevenbergMarquardtOptimizer;
use nexus_allocator::BumpAllocator;
use ndarray::{Array1, Array2};
use std::f64::consts::E;

/// Nelson-Siegel-Svensson parameters: [β₀, β₁, β₂, β₃, τ₁, τ₂]
#[derive(Debug, Clone, Copy)]
pub struct NssParameters {
    pub beta0: f64,  // Long-term level
    pub beta1: f64,  // Short-term slope
    pub beta2: f64,  // Medium-term curvature (hump)
    pub beta3: f64,  // Second curvature factor
    pub tau1: f64,   // First decay parameter
    pub tau2: f64,   // Second decay parameter
}

impl Default for NssParameters {
    fn default() -> Self {
        Self {
            beta0: 0.05,
            beta1: -0.02,
            beta2: 0.01,
            beta3: 0.0,
            tau1: 1.0,
            tau2: 5.0,
        }
    }
}

impl NssParameters {
    /// Convert to array for optimization
    #[inline(always)]
    pub fn to_array(&self) -> [f64; 6] {
        [self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2]
    }

    /// Create from array with validation
    pub fn from_array(params: &[f64; 6]) -> Result<Self, NssError> {
        let [beta0, beta1, beta2, beta3, tau1, tau2] = *params;
        
        // Validate decay parameters are positive
        if tau1 <= 0.0 || tau2 <= 0.0 {
            return Err(NssError::InvalidDecayParameter);
        }
        
        // Check for NaN/Inf
        if params.iter().any(|p| !p.is_finite()) {
            return Err(NssError::NonFiniteParameter);
        }
        
        Ok(Self { beta0, beta1, beta2, beta3, tau1, tau2 })
    }
}

/// Error types for NSS fitting
#[derive(Debug, Clone)]
pub enum NssError {
    InvalidDecayParameter,
    NonFiniteParameter,
    ConvergenceFailure,
    SingularJacobian,
    InvalidMaturity,
}

impl std::fmt::Display for NssError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NssError::InvalidDecayParameter => write!(f, "Invalid decay parameter (tau must be > 0)"),
            NssError::NonFiniteParameter => write!(f, "Non-finite parameter detected"),
            NssError::ConvergenceFailure => write!(f, "Optimizer failed to converge"),
            NssError::SingularJacobian => write!(f, "Singular Jacobian matrix"),
            NssError::InvalidMaturity => write!(f, "Invalid maturity value"),
        }
    }
}

impl std::error::Error for NssError {}

/// Zero-allocation NSS curve evaluator
pub struct NssCurveEvaluator {
    /// Pre-allocated buffer for yields (reused across calls)
    yields_buffer: Vec<f64>,
    /// Pre-allocated buffer for derivatives (Jacobian rows)
    derivs_buffer: Vec<f64>,
}

impl NssCurveEvaluator {
    /// Create new evaluator with pre-allocated capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            yields_buffer: Vec::with_capacity(capacity),
            derivs_buffer: Vec::with_capacity(capacity * 6), // 6 parameters
        }
    }

    /// Evaluate NSS yield at a single maturity point (zero-allocation)
    #[inline(always)]
    pub fn evaluate_yield(&self, params: &NssParameters, maturity: f64) -> Result<f64, NssError> {
        if maturity <= 0.0 {
            return Err(NssError::InvalidMaturity);
        }

        let t_tau1 = maturity / params.tau1;
        let t_tau2 = maturity / params.tau2;

        // Handle near-zero maturity limit (L'Hôpital's rule)
        let factor1 = if t_tau1.abs() < 1e-10 {
            1.0
        } else {
            (1.0 - (-t_tau1).exp()) / t_tau1
        };

        let factor2 = if t_tau1.abs() < 1e-10 {
            0.0
        } else {
            factor1 - (-t_tau1).exp()
        };

        let factor3 = if t_tau2.abs() < 1e-10 {
            0.0
        } else {
            (1.0 - (-t_tau2).exp()) / t_tau2 - (-t_tau2).exp()
        };

        let yield_val = params.beta0 
            + params.beta1 * factor1 
            + params.beta2 * factor2 
            + params.beta3 * factor3;

        Ok(yield_val)
    }

    /// Evaluate NSS yields at multiple maturities (zero-allocation, uses pre-allocated buffer)
    pub fn evaluate_yields_batch<'a>(
        &'a mut self,
        params: &NssParameters,
        maturities: &[f64],
    ) -> Result<&'a [f64], NssError> {
        self.yields_buffer.clear();
        self.yields_buffer.reserve(maturities.len());

        for &maturity in maturities {
            let y = self.evaluate_yield(params, maturity)?;
            self.yields_buffer.push(y);
        }

        Ok(&self.yields_buffer)
    }

    /// Compute partial derivatives of yield w.r.t. all 6 parameters at a given maturity
    #[inline(always)]
    pub fn compute_derivatives(
        &self,
        params: &NssParameters,
        maturity: f64,
    ) -> Result<[f64; 6], NssError> {
        if maturity <= 0.0 {
            return Err(NssError::InvalidMaturity);
        }

        let t = maturity;
        let t1 = params.tau1;
        let t2 = params.tau2;

        let x1 = t / t1;
        let x2 = t / t2;

        let exp_x1 = (-x1).exp();
        let exp_x2 = (-x2).exp();

        // Common factors
        let f1 = if x1.abs() < 1e-10 { 1.0 } else { (1.0 - exp_x1) / x1 };
        let f2 = if x1.abs() < 1e-10 { 0.0 } else { f1 - exp_x1 };
        let f3 = if x2.abs() < 1e-10 { 0.0 } else { (1.0 - exp_x2) / x2 - exp_x2 };

        // Partial derivatives
        // dY/dβ₀ = 1
        let dbeta0 = 1.0;

        // dY/dβ₁ = f1
        let dbeta1 = f1;

        // dY/dβ₂ = f2
        let dbeta2 = f2;

        // dY/dβ₃ = f3
        let dbeta3 = f3;

        // dY/dτ₁ (chain rule through x1)
        // Need derivative of f1 and f2 w.r.t. τ₁
        let df1_dx1 = if x1.abs() < 1e-10 {
            0.5 // Limit as x→0
        } else {
            (exp_x1 * (x1 + 1.0) - 1.0) / (x1 * x1)
        };
        
        let df2_dx1 = df1_dx1 + exp_x1; // d(f1 - exp(-x1))/dx1

        let dx1_dtau1 = -t / (t1 * t1);
        let dbeta1_dtau1 = params.beta1 * df1_dx1 * dx1_dtau1;
        let dbeta2_dtau1 = params.beta2 * df2_dx1 * dx1_dtau1;
        let dbeta_tau1 = dbeta1_dtau1 + dbeta2_dtau1;

        // dY/dτ₂ (similar logic)
        let df3_dx2 = if x2.abs() < 1e-10 {
            0.5
        } else {
            (exp_x2 * (x2 + 1.0) - 1.0) / (x2 * x2) + exp_x2
        };

        let dx2_dtau2 = -t / (t2 * t2);
        let dbeta_tau2 = params.beta3 * df3_dx2 * dx2_dtau2;

        Ok([dbeta0, dbeta1, dbeta2, dbeta3, dbeta_tau1, dbeta_tau2])
    }

    /// Build Jacobian matrix for batch of maturities (zero-allocation pattern)
    pub fn build_jacobian(
        &mut self,
        params: &NssParameters,
        maturities: &[f64],
    ) -> Result<Array2<f64>, NssError> {
        let n = maturities.len();
        let mut jacobian = Array2::<f64>::zeros((n, 6));

        for (i, &maturity) in maturities.iter().enumerate() {
            let derivs = self.compute_derivatives(params, maturity)?;
            for j in 0..6 {
                jacobian[[i, j]] = derivs[j];
            }
        }

        Ok(jacobian)
    }
}

/// NSS Curve Fitter using Levenberg-Marquardt optimization
pub struct NssCurveFitter {
    optimizer: LevenbergMarquardtOptimizer,
    evaluator: NssCurveEvaluator,
    max_iterations: usize,
    tolerance: f64,
}

impl NssCurveFitter {
    /// Create new fitter with specified capacity for maturities
    pub fn new(maturity_capacity: usize, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            optimizer: LevenbergMarquardtOptimizer::new(6, maturity_capacity),
            evaluator: NssCurveEvaluator::new(maturity_capacity),
            max_iterations,
            tolerance,
        }
    }

    /// Fit NSS parameters to observed bond yields
    /// 
    /// # Arguments
    /// * `maturities` - Vector of bond maturities (in years)
    /// * `yields` - Observed yields at those maturities
    /// * `initial_guess` - Starting parameters for optimization
    /// 
    /// # Returns
    /// Fitted NSS parameters or error
    pub fn fit(
        &mut self,
        maturities: &[f64],
        yields: &[f64],
        initial_guess: &NssParameters,
    ) -> Result<NssParameters, NssError> {
        if maturities.len() != yields.len() {
            return Err(NssError::InvalidMaturity);
        }

        if maturities.len() < 3 {
            // Need at least 3 points to fit 6 parameters reasonably
            return Err(NssError::InsufficientData);
        }

        // Use Levenberg-Marquardt to minimize sum of squared residuals
        let result = self.optimizer.optimize(
            initial_guess.to_array(),
            |params| -> Result<Vec<f64>, String> {
                let nss_params = NssParameters::from_array(&params)
                    .map_err(|e| format!("{:?}", e))?;
                
                let mut residuals = Vec::with_capacity(maturities.len());
                for (i, &maturity) in maturities.iter().enumerate() {
                    let predicted = self.evaluator.evaluate_yield(&nss_params, maturity)
                        .map_err(|e| format!("{:?}", e))?;
                    residuals.push(predicted - yields[i]);
                }
                Ok(residuals)
            },
            |params| -> Result<Array2<f64>, String> {
                let nss_params = NssParameters::from_array(&params)
                    .map_err(|e| format!("{:?}", e))?;
                self.evaluator.build_jacobian(&nss_params, maturities)
                    .map_err(|e| format!("{:?}", e))
            },
            self.max_iterations,
            self.tolerance,
        );

        match result {
            Ok(opt_params) => {
                NssParameters::from_array(&opt_params).map_err(|_| NssError::NonFiniteParameter)
            }
            Err(_) => Err(NssError::ConvergenceFailure),
        }
    }
}

impl NssCurveFitter {
    /// Additional error type for insufficient data
    const InsufficientData: NssError = NssError::InvalidMaturity; // Reuse for now
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nss_evaluation() {
        let params = NssParameters::default();
        let evaluator = NssCurveEvaluator::new(10);
        
        let yield_1y = evaluator.evaluate_yield(&params, 1.0).unwrap();
        let yield_10y = evaluator.evaluate_yield(&params, 10.0).unwrap();
        
        assert!(yield_1y.is_finite());
        assert!(yield_10y.is_finite());
    }

    #[test]
    fn test_invalid_decay_parameter() {
        let bad_params = [0.05, -0.02, 0.01, 0.0, -1.0, 5.0]; // Negative tau1
        let result = NssParameters::from_array(&bad_params);
        assert!(result.is_err());
    }
}
