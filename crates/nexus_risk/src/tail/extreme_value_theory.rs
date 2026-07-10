//! Extreme Value Theory (EVT) Implementation using Peaks Over Threshold (POT) method
//! 
//! This module implements the Generalized Pareto Distribution (GPD) for modeling
//! extreme tails of return distributions. Critical for black swan detection and
//! tail risk estimation beyond standard Gaussian VaR.
//!
//! Mathematical Foundation:
//! - GPD CDF: F(x) = 1 - (1 + ξ * x/β)^(-1/ξ) for ξ ≠ 0
//! - Gumbel limit: F(x) = 1 - exp(-x/β) for ξ → 0
//! - Shape parameter ξ > 0 indicates heavy tails (Fréchet domain)

use crate::tail::gpd_mle_solver::{GpdParameters, GpdSolver, SolverError};
use crate::tail::simd_peaks_over_threshold::{SimdPeaksFilter, PeakThreshold};
use ndarray::{Array1, ArrayView1};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

/// Configuration for EVT analysis
#[derive(Debug, Clone)]
pub struct EvtConfig {
    /// Threshold percentile for peak selection (e.g., 0.95 for 95th percentile)
    pub threshold_percentile: f64,
    /// Minimum number of peaks required for fitting
    pub min_peaks: usize,
    /// Rolling window size for parameter estimation
    pub rolling_window_size: usize,
    /// Convergence tolerance for MLE solver
    pub convergence_tolerance: f64,
    /// Maximum iterations for Newton-Raphson
    pub max_iterations: usize,
}

impl Default for EvtConfig {
    fn default() -> Self {
        Self {
            threshold_percentile: 0.95,
            min_peaks: 30,
            rolling_window_size: 252, // ~1 trading year
            convergence_tolerance: 1e-8,
            max_iterations: 100,
        }
    }
}

/// Result of EVT fitting containing GPD parameters and diagnostics
#[derive(Debug, Clone)]
pub struct EvtFitResult {
    /// Fitted GPD parameters
    pub parameters: GpdParameters,
    /// Number of peaks used in fitting
    pub num_peaks: usize,
    /// Estimated threshold value
    pub threshold: f64,
    /// Log-likelihood of the fit
    pub log_likelihood: f64,
    /// Standard error of shape parameter
    pub shape_se: f64,
    /// Standard error of scale parameter
    pub scale_se: f64,
    /// Tail index (1/ξ) - infinite for ξ=0 (Gumbel)
    pub tail_index: Option<f64>,
    /// Return level estimates for various probabilities
    pub return_levels: Vec<(f64, f64)>,
}

impl EvtFitResult {
    /// Calculate Value-at-Risk at confidence level p using fitted GPD
    pub fn var_gpd(&self, p: f64) -> Result<f64, EvtError> {
        if p <= 0.0 || p >= 1.0 {
            return Err(EvtError::InvalidProbability(p));
        }
        
        let xi = self.parameters.shape;
        let beta = self.parameters.scale;
        let u = self.threshold;
        
        // Probability of exceeding threshold
        let zeta = (self.num_peaks as f64) / ((self.num_peaks + 1) as f64);
        
        // Return level calculation
        let m = 1.0 / (1.0 - p);
        let m_zeta = m * zeta;
        
        if m_zeta <= 1.0 {
            return Ok(u); // Below threshold
        }
        
        let log_term = (m_zeta).ln();
        
        if xi.abs() < 1e-10 {
            // Gumbel case (ξ → 0)
            Ok(u + beta * log_term)
        } else {
            // General GPD case
            let power_term = (-xi * log_term).exp();
            Ok(u + (beta / xi) * (power_term - 1.0))
        }
    }
    
    /// Calculate Expected Shortfall (ES) at confidence level p
    pub fn expected_shortfall(&self, p: f64) -> Result<f64, EvtError> {
        if p <= 0.0 || p >= 1.0 {
            return Err(EvtError::InvalidProbability(p));
        }
        
        let var = self.var_gpd(p)?;
        let xi = self.parameters.shape;
        let beta = self.parameters.scale;
        let u = self.threshold;
        
        if xi >= 1.0 {
            // ES undefined for ξ ≥ 1 (infinite mean)
            return Ok(f64::INFINITY);
        }
        
        let zeta = (self.num_peaks as f64) / ((self.num_peaks + 1) as f64);
        let m = 1.0 / (1.0 - p);
        let m_zeta = m * zeta;
        
        if m_zeta <= 1.0 {
            return Ok(u);
        }
        
        let log_term = (m_zeta).ln();
        let power_term = (-xi * log_term).exp();
        
        if xi.abs() < 1e-10 {
            // Gumbel case
            Ok(var + beta)
        } else {
            // General case: ES = (var / (1-ξ)) + ((β - ξ*u) / (1-ξ))
            let term1 = var / (1.0 - xi);
            let term2 = (beta - xi * u) / (1.0 - xi);
            Ok(term1 + term2)
        }
    }
}

/// Errors specific to EVT operations
#[derive(Error, Debug, Clone)]
pub enum EvtError {
    #[error("Insufficient peaks for fitting: got {0}, need at least {1}")]
    InsufficientPeaks(usize, usize),
    
    #[error("Invalid probability value: {0}")]
    InvalidProbability(f64),
    
    #[error("MLE solver failed: {0}")]
    SolverFailed(String),
    
    #[error("Threshold calculation failed: {0}")]
    ThresholdError(String),
    
    #[error("Numerical instability detected: {0}")]
    NumericalInstability(String),
    
    #[error("Tail index undefined for shape parameter near zero")]
    UndefinedTailIndex,
}

/// Main EVT engine for tail risk modeling
pub struct ExtremeValueTheory {
    config: EvtConfig,
    peaks_filter: SimdPeaksFilter,
    solver: GpdSolver,
    last_fit_result: Option<EvtFitResult>,
    fit_counter: AtomicU64,
}

impl ExtremeValueTheory {
    /// Create a new EVT engine with default configuration
    pub fn new() -> Self {
        Self::with_config(EvtConfig::default())
    }
    
    /// Create a new EVT engine with custom configuration
    pub fn with_config(config: EvtConfig) -> Self {
        let threshold = PeakThreshold::Percentile(config.threshold_percentile);
        Self {
            config: config.clone(),
            peaks_filter: SimdPeaksFilter::new(threshold, config.rolling_window_size),
            solver: GpdSolver::new(config.convergence_tolerance, config.max_iterations),
            last_fit_result: None,
            fit_counter: AtomicU64::new(0),
        }
    }
    
    /// Update the data stream and refit GPD parameters if enough new peaks
    pub fn update_and_fit(&mut self, returns: ArrayView1<f64>) -> Result<Option<EvtFitResult>, EvtError> {
        // Extract peaks above threshold using SIMD-accelerated filter
        let peaks = self.peaks_filter.extract_peaks(returns)?;
        
        if peaks.len() < self.config.min_peaks {
            return Ok(None); // Not enough data for reliable fitting
        }
        
        // Fit GPD using MLE
        let fit_result = self.solver.fit_gpd(&peaks)?;
        
        let result = EvtFitResult {
            parameters: fit_result.parameters,
            num_peaks: peaks.len(),
            threshold: self.peaks_filter.current_threshold(),
            log_likelihood: fit_result.log_likelihood,
            shape_se: fit_result.shape_se,
            scale_se: fit_result.scale_se,
            tail_index: if fit_result.parameters.shape.abs() > 1e-10 {
                Some(1.0 / fit_result.parameters.shape)
            } else {
                None
            },
            return_levels: self.calculate_return_levels(&fit_result.parameters)?,
        };
        
        self.last_fit_result = Some(result.clone());
        self.fit_counter.fetch_add(1, Ordering::Relaxed);
        
        Ok(Some(result))
    }
    
    /// Get the current tail index (1/ξ) if available
    pub fn tail_index(&self) -> Option<f64> {
        self.last_fit_result.as_ref().and_then(|r| r.tail_index)
    }
    
    /// Check if the distribution has heavy tails (ξ > 0)
    pub fn has_heavy_tails(&self) -> bool {
        self.last_fit_result
            .as_ref()
            .map(|r| r.parameters.shape > 0.0)
            .unwrap_or(false)
    }
    
    /// Get the fitted shape parameter
    pub fn shape_parameter(&self) -> Option<f64> {
        self.last_fit_result.as_ref().map(|r| r.parameters.shape)
    }
    
    /// Get the fitted scale parameter
    pub fn scale_parameter(&self) -> Option<f64> {
        self.last_fit_result.as_ref().map(|r| r.parameters.scale)
    }
    
    /// Calculate Value-at-Risk at confidence level p
    pub fn var_at_confidence(&self, p: f64) -> Result<f64, EvtError> {
        self.last_fit_result
            .as_ref()
            .ok_or_else(|| EvtError::SolverFailed("No fit result available".to_string()))?
            .var_gpd(p)
    }
    
    /// Calculate Expected Shortfall at confidence level p
    pub fn es_at_confidence(&self, p: f64) -> Result<f64, EvtError> {
        self.last_fit_result
            .as_ref()
            .ok_or_else(|| EvtError::SolverFailed("No fit result available".to_string()))?
            .expected_shortfall(p)
    }
    
    /// Calculate return levels for various return periods
    fn calculate_return_levels(
        &self,
        params: &GpdParameters,
    ) -> Result<Vec<(f64, f64)>, EvtError> {
        let periods = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0];
        let mut levels = Vec::with_capacity(periods.len());
        
        let zeta = 0.95; // Approximate probability of exceeding threshold
        
        for m in periods.iter() {
            let m_zeta = m * zeta;
            let log_term = m_zeta.ln();
            
            let return_level = if params.shape.abs() < 1e-10 {
                // Gumbel case
                self.peaks_filter.current_threshold() + params.scale * log_term
            } else {
                // General GPD case
                let power_term = (-params.shape * log_term).exp();
                self.peaks_filter.current_threshold()
                    + (params.scale / params.shape) * (power_term - 1.0)
            };
            
            levels.push((*m, return_level));
        }
        
        Ok(levels)
    }
    
    /// Get the number of times GPD has been fitted
    pub fn fit_count(&self) -> u64 {
        self.fit_counter.load(Ordering::Relaxed)
    }
    
    /// Reset the EVT engine state
    pub fn reset(&mut self) {
        self.peaks_filter.reset();
        self.last_fit_result = None;
        self.fit_counter.store(0, Ordering::Relaxed);
    }
}

impl Default for ExtremeValueTheory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::Distribution;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    
    #[test]
    fn test_evt_fitting_with_gpd_data() {
        let mut rng = StdRng::seed_from_u64(42);
        
        // Generate synthetic GPD data
        let shape = 0.3;
        let scale = 1.0;
        let n_samples = 1000;
        
        let mut returns = Array1::zeros(n_samples);
        for i in 0..n_samples {
            // Simplified GPD sampling
            let u = rng.gen::<f64>();
            let gpd_sample = scale * (1.0 - u.powf(-shape)) / shape;
            returns[i] = -gpd_sample; // Negative for losses
        }
        
        let mut evt = ExtremeValueTheory::with_config(EvtConfig {
            threshold_percentile: 0.90,
            min_peaks: 20,
            ..Default::default()
        });
        
        let result = evt.update_and_fit(returns.view()).unwrap();
        assert!(result.is_some());
        
        let fit = result.unwrap();
        // Shape should be positive for heavy tails
        assert!(fit.parameters.shape > 0.0);
    }
    
    #[test]
    fn test_var_calculation() {
        let mut evt = ExtremeValueTheory::new();
        
        // Create some test data
        let returns = Array1::from_vec(vec![-0.1, -0.05, -0.08, -0.15, -0.03]);
        
        // This will likely return None due to insufficient data
        let result = evt.update_and_fit(returns.view()).unwrap();
        assert!(result.is_none());
    }
}
