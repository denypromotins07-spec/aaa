//! Paradox Slippage Resolver using Deutsch CTC formalism
//! 
//! Resolves causal paradoxes in reflexive market execution where
//! the bot's trade causes the very slippage it was trying to avoid.

use crate::ctc::deutsch_density_matrix::{DensityMatrix, Complex, CTCResult};
use crate::ctc::fixed_point_iteration::{FixedPointSolver, FixedPointResult, ConvergenceMethod};

/// Minimum confidence threshold for paradox resolution
const MIN_CONFIDENCE_THRESHOLD: f64 = 0.95;

/// Maximum allowed slippage feedback ratio
const MAX_FEEDBACK_RATIO: f64 = 0.1;

/// Execution paradox type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParadoxType {
    /// Trade triggers flash crash invalidating alpha signal
    FlashCrash,
    /// Liquidity withdrawal due to order detection
    LiquidityWithdrawal,
    /// Self-fulfilling adverse selection
    AdverseSelection,
    /// Momentum exhaustion from own trading
    MomentumExhaustion,
    /// No paradox detected
    None,
}

/// Paradox resolution result
#[derive(Debug, Clone)]
pub struct ParadoxResolution {
    /// Type of paradox detected
    pub paradox_type: ParadoxType,
    /// Whether resolution was successful
    pub resolved: bool,
    /// Consistent execution state (CTC fixed point)
    pub consistent_state: DensityMatrix,
    /// Expected slippage after resolution
    pub resolved_slippage: f64,
    /// Original estimated slippage (before resolution)
    pub original_slippage: f64,
    /// Confidence in resolution
    pub confidence: f64,
    /// Number of iterations to resolve
    pub iterations: usize,
    /// Method used for resolution
    pub method: ConvergenceMethod,
}

/// Market execution parameters for paradox modeling
#[derive(Debug, Clone)]
pub struct ExecutionParams {
    /// Order size (base units)
    pub order_size: f64,
    /// Current market price
    pub market_price: f64,
    /// Estimated market impact coefficient
    pub impact_coefficient: f64,
    /// Order book depth at relevant levels
    pub order_book_depth: Vec<f64>,
    /// Historical volatility (annualized)
    pub volatility: f64,
    /// Time horizon for execution (nanoseconds)
    pub time_horizon_ns: u64,
}

impl ExecutionParams {
    /// Create new execution parameters
    pub fn new(order_size: f64, market_price: f64, impact_coef: f64, 
               depth: Vec<f64>, volatility: f64, time_horizon_ns: u64) -> Self {
        Self {
            order_size,
            market_price,
            impact_coefficient: impact_coef,
            order_book_depth: depth,
            volatility,
            time_horizon_ns,
        }
    }

    /// Estimate naive slippage (without paradox resolution)
    pub fn estimate_naive_slippage(&self) -> f64 {
        if self.order_book_depth.is_empty() {
            return self.impact_coefficient * self.order_size;
        }

        // Simple linear impact model
        let total_depth: f64 = self.order_book_depth.iter().sum();
        if total_depth < 1e-15 {
            return self.impact_coefficient * self.order_size;
        }

        let depth_ratio = self.order_size / total_depth;
        self.impact_coefficient * depth_ratio * self.market_price
    }
}

/// Paradox Slippage Resolver
pub struct ParadoxSlippageResolver {
    /// Fixed-point solver for CTC states
    solver: FixedPointSolver,
    /// Minimum confidence threshold
    confidence_threshold: f64,
    /// Maximum feedback ratio
    max_feedback_ratio: f64,
}

impl ParadoxSlippageResolver {
    /// Create a new paradox resolver with default parameters
    pub fn new() -> Self {
        Self {
            solver: FixedPointSolver::new(),
            confidence_threshold: MIN_CONFIDENCE_THRESHOLD,
            max_feedback_ratio: MAX_FEEDBACK_RATIO,
        }
    }

    /// Create resolver with custom thresholds
    pub fn with_thresholds(confidence: f64, feedback: f64) -> Self {
        Self {
            solver: FixedPointSolver::new(),
            confidence_threshold: confidence.clamp(0.5, 0.99),
            max_feedback_ratio: feedback.clamp(0.01, 0.5),
        }
    }

    /// Analyze and resolve execution paradox
    /// 
    /// # Arguments
    /// * `params` - Execution parameters
    /// * `alpha_signal` - Alpha signal strength vector
    /// 
    /// # Returns
    /// ParadoxResolution with consistent execution state
    pub fn resolve_paradox(&self, params: &ExecutionParams, alpha_signal: &[f64]) -> ParadoxResolution {
        // Detect paradox type
        let paradox_type = self.detect_paradox_type(params, alpha_signal);
        
        if paradox_type == ParadoxType::None {
            // No paradox detected, return early
            let naive_slippage = params.estimate_naive_slippage();
            return ParadoxResolution {
                paradox_type: ParadoxType::None,
                resolved: true,
                consistent_state: self.create_state_from_params(params),
                resolved_slippage: naive_slippage,
                original_slippage: naive_slippage,
                confidence: 1.0,
                iterations: 0,
                method: ConvergenceMethod::DirectIteration,
            };
        }

        // Build input density matrix from execution parameters
        let rho_in = match self.build_input_state(params, alpha_signal) {
            Some(dm) => dm,
            None => return ParadoxResolution {
                paradox_type,
                resolved: false,
                consistent_state: self.create_state_from_params(params),
                resolved_slippage: params.estimate_naive_slippage(),
                original_slippage: params.estimate_naive_slippage(),
                confidence: 0.0,
                iterations: 0,
                method: ConvergenceMethod::Failed,
            },
        };

        // Construct interaction unitary (simplified model)
        let unitary = self.construct_interaction_unitary(params);
        let unitary_dim = rho_in.dimension();

        // Find CTC fixed point
        let fp_result = self.solver.find_fixed_point(&rho_in, &unitary, unitary_dim);

        if !fp_result.converged {
            return ParadoxResolution {
                paradox_type,
                resolved: false,
                consistent_state: fp_result.rho_ctc,
                resolved_slippage: params.estimate_naive_slippage(),
                original_slippage: params.estimate_naive_slippage(),
                confidence: 0.0,
                iterations: fp_result.iterations,
                method: fp_result.method,
            };
        }

        // Calculate resolved slippage from fixed point
        let resolved_slippage = self.calculate_resolved_slippage(&fp_result.rho_ctc, params);
        let original_slippage = params.estimate_naive_slippage();
        
        // Verify slippage reduction
        let slippage_reduction = if original_slippage > 1e-15 {
            (original_slippage - resolved_slippage) / original_slippage
        } else {
            0.0
        };

        // Calculate confidence based on convergence quality and slippage reduction
        let confidence = self.calculate_confidence(&fp_result, slippage_reduction);

        ParadoxResolution {
            paradox_type,
            resolved: confidence >= self.confidence_threshold,
            consistent_state: fp_result.rho_ctc,
            resolved_slippage,
            original_slippage,
            confidence,
            iterations: fp_result.iterations,
            method: fp_result.method,
        }
    }

    /// Validate that resolved state is self-consistent
    pub fn validate_consistency(&self, resolution: &ParadoxResolution, 
                                params: &ExecutionParams, alpha_signal: &[f64]) -> bool {
        if !resolution.resolved {
            return false;
        }

        // Re-compute fixed point and compare
        let rho_in = match self.build_input_state(params, alpha_signal) {
            Some(dm) => dm,
            None => return false,
        };

        let unitary = self.construct_interaction_unitary(params);
        let unitary_dim = rho_in.dimension();

        // Verify fixed point
        match self.solver.verify_fixed_point(&resolution.consistent_state, &rho_in, &unitary, unitary_dim) {
            Some(residual) => residual < self.solver.tolerance,
            None => false,
        }
    }

    // Internal: Detect type of paradox
    fn detect_paradox_type(&self, params: &ExecutionParams, alpha_signal: &[f64]) -> ParadoxType {
        let naive_slippage = params.estimate_naive_slippage();
        
        // Check for flash crash conditions
        if params.volatility > 0.5 && params.order_size > 1000.0 {
            if alpha_signal.iter().any(|&s| s.abs() > 5.0) {
                return ParadoxType::FlashCrash;
            }
        }

        // Check for liquidity withdrawal
        if params.order_book_depth.len() >= 2 {
            let depth_gradient = params.order_book_depth[0] - 
                params.order_book_depth.last().copied().unwrap_or(0.0);
            if depth_gradient < 0.0 && depth_gradient.abs() > params.order_book_depth[0] * 0.5 {
                return ParadoxType::LiquidityWithdrawal;
            }
        }

        // Check for adverse selection
        let signal_variance = self.compute_variance(alpha_signal);
        if signal_variance > 10.0 {
            return ParadoxType::AdverseSelection;
        }

        // Check for momentum exhaustion
        if alpha_signal.len() >= 3 {
            let trend = alpha_signal[alpha_signal.len() - 1] - alpha_signal[0];
            if trend.abs() < signal_variance.sqrt() * 0.1 {
                return ParadoxType::MomentumExhaustion;
            }
        }

        ParadoxType::None
    }

    // Internal: Build input density matrix
    fn build_input_state(&self, params: &ExecutionParams, alpha_signal: &[f64]) -> Option<DensityMatrix> {
        // Combine execution params with alpha signal into state vector
        let mut state = Vec::with_capacity(alpha_signal.len().max(params.order_book_depth.len()));
        
        // Normalize order size contribution
        let size_factor = (params.order_size / 1000.0).clamp(-1.0, 1.0);
        state.push(size_factor);
        
        // Add normalized alpha signal components
        let alpha_norm = alpha_signal.iter().map(|&s| s.powi(2)).sum::<f64>().sqrt().max(1e-15);
        for &alpha in alpha_signal {
            state.push(alpha / alpha_norm);
        }
        
        // Pad or truncate to reasonable dimension
        while state.len() < 2 {
            state.push(0.0);
        }
        if state.len() > 64 {
            state.truncate(64);
        }

        DensityMatrix::from_pure_state(&state)
    }

    // Internal: Construct interaction unitary (simplified)
    fn construct_interaction_unitary(&self, params: &ExecutionParams) -> Vec<Complex> {
        let dim = params.order_book_depth.len().max(2).min(64);
        let mut unitary = vec![Complex::real(0.0); dim * dim];
        
        // Build approximate unitary from impact coefficient
        let phase = params.impact_coefficient * params.order_size;
        
        for i in 0..dim {
            for j in 0..dim {
                if i == j {
                    // Diagonal elements with phase rotation
                    unitary[i * dim + j] = Complex::new(phase.cos(), phase.sin());
                } else if (i as i64 - j as i64).abs() == 1 {
                    // Nearest neighbor coupling
                    let coupling = 0.1 * params.impact_coefficient;
                    unitary[i * dim + j] = Complex::real(coupling);
                }
            }
        }
        
        // Normalize columns (simplified unitarity check)
        for j in 0..dim {
            let col_norm: f64 = (0..dim).map(|i| unitary[i * dim + j].norm_squared()).sum::<f64>().sqrt();
            if col_norm > 1e-15 {
                for i in 0..dim {
                    unitary[i * dim + j].re /= col_norm;
                    unitary[i * dim + j].im /= col_norm;
                }
            }
        }
        
        unitary
    }

    // Internal: Calculate resolved slippage from CTC state
    fn calculate_resolved_slippage(&self, rho_ctc: &DensityMatrix, params: &ExecutionParams) -> f64 {
        // Extract diagonal elements (probabilities)
        let dim = rho_ctc.dimension();
        let mut probabilities = Vec::with_capacity(dim);
        
        for i in 0..dim {
            if let Some(elem) = rho_ctc.get(i, i) {
                probabilities.push(elem.re.max(0.0));
            } else {
                probabilities.push(0.0);
            }
        }
        
        // Weighted average slippage based on CTC probabilities
        let naive_slippage = params.estimate_naive_slippage();
        
        // Apply feedback suppression factor
        let feedback_factor = 1.0 - self.max_feedback_ratio;
        naive_slippage * feedback_factor * (probabilities.iter().sum::<f64>() / dim as f64)
    }

    // Internal: Calculate confidence score
    fn calculate_confidence(&self, fp_result: &FixedPointResult, slippage_reduction: f64) -> f64 {
        let convergence_factor = if fp_result.converged {
            1.0 - fp_result.residual.min(1.0)
        } else {
            0.0
        };
        
        let slippage_factor = slippage_reduction.clamp(0.0, 1.0);
        
        let method_bonus = match fp_result.method {
            ConvergenceMethod::DirectIteration => 0.1,
            ConvergenceMethod::DampedIteration => 0.05,
            ConvergenceMethod::SimulatedAnnealing => 0.0,
            ConvergenceMethod::Failed => -0.5,
        };
        
        (convergence_factor * 0.6 + slippage_factor * 0.4 + method_bonus).clamp(0.0, 1.0)
    }

    // Internal: Create state from params (fallback)
    fn create_state_from_params(&self, params: &ExecutionParams) -> DensityMatrix {
        let state = vec![
            (params.order_size / 1000.0).clamp(-1.0, 1.0),
            params.impact_coefficient,
            params.volatility,
        ];
        DensityMatrix::from_pure_state(&state).unwrap_or_else(|| {
            DensityMatrix::maximally_mixed(3).unwrap()
        })
    }

    // Internal: Compute variance
    fn compute_variance(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
    }
}

impl Default for ParadoxSlippageResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolver_creation() {
        let resolver = ParadoxSlippageResolver::new();
        assert!(resolver.confidence_threshold >= 0.5);
        assert!(resolver.max_feedback_ratio > 0.0);
    }

    #[test]
    fn test_no_paradox_detection() {
        let resolver = ParadoxSlippageResolver::new();
        let params = ExecutionParams::new(
            100.0,  // Small order
            100.0,  // Price
            0.001,  // Low impact
            vec![1000.0, 800.0, 600.0],  // Good depth
            0.2,    // Normal vol
            1_000_000,  // 1ms horizon
        );
        let alpha = vec![1.0, 1.1, 1.05];  // Stable signal
        
        let result = resolver.resolve_paradox(&params, &alpha);
        assert_eq!(result.paradox_type, ParadoxType::None);
        assert!(result.resolved);
    }

    #[test]
    fn test_custom_thresholds() {
        let resolver = ParadoxSlippageResolver::with_thresholds(0.9, 0.05);
        assert!((resolver.confidence_threshold - 0.9).abs() < 0.01);
        assert!((resolver.max_feedback_ratio - 0.05).abs() < 0.01);
    }
}
