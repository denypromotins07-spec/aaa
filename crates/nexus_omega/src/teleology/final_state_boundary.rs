//! Final State Boundary Conditions for eschatological pricing.
//! Implements Hartle-Hawking no-boundary proposal applied to financial derivatives.

use alloc::vec::Vec;
use core::fmt::Debug;

use super::wick_rotated_kolmogorov::{
    Complex, ModelParams, WickKolmogorovConfig, WickKolmogorovSolver, WickEvolutionResult,
    BoundaryCondition,
};

/// Types of final state boundaries
#[derive(Debug, Clone)]
pub enum FinalBoundaryType {
    /// Currency collapse (purchasing power -> 0)
    CurrencyCollapse { decay_rate: f64 },
    /// Market heat death (volatility -> 0)
    VolatilityExtinction { timescale: f64 },
    /// Regime transition (phase change at t_final)
    PhaseTransition { critical_time: f64 },
    /// Singularity (infinite growth rate)
    FiniteTimeSingularity { singularity_time: f64 },
    /// Cyclic boundary (returns to initial state)
    Cyclic { period: f64 },
}

/// Configuration for final state boundary problem
#[derive(Debug, Clone)]
pub struct FinalStateConfig {
    /// Type of final boundary condition
    pub boundary_type: FinalBoundaryType,
    /// Time horizon (real time)
    pub time_horizon: f64,
    /// Spatial domain for the problem
    pub spatial_range: (f64, f64),
    /// Grid resolution
    pub num_grid_points: usize,
}

impl Default for FinalStateConfig {
    fn default() -> Self {
        Self {
            boundary_type: FinalBoundaryType::CurrencyCollapse { decay_rate: 0.02 },
            time_horizon: 30.0,
            spatial_range: (0.1, 1000.0),
            num_grid_points: 256,
        }
    }
}

/// Result of final state boundary calculation
#[derive(Debug, Clone)]
pub struct FinalStateResult {
    /// Present value derived from final boundary
    pub present_value: Vec<f64>,
    /// Probability of reaching final state
    pub probability: f64,
    /// Expected time to boundary
    pub expected_time: f64,
    /// Whether solution is well-behaved
    pub is_stable: bool,
}

/// Final state boundary solver
pub struct FinalStateBoundarySolver {
    config: FinalStateConfig,
    wick_solver: WickKolmogorovSolver,
}

impl FinalStateBoundarySolver {
    pub fn new(config: FinalStateConfig) -> Self {
        let wick_config = WickKolmogorovConfig {
            num_grid_points: config.num_grid_points,
            dtau: 0.01,
            total_tau: config.time_horizon * 0.1,
            spatial_range: config.spatial_range,
            boundary: BoundaryCondition::DirichletZero,
        };

        Self {
            config,
            wick_solver: WickKolmogorovSolver::new(wick_config),
        }
    }

    /// Solve backward from final state boundary to present
    pub fn solve_backward(
        &self,
        terminal_payoff: &[f64],
    ) -> Result<FinalStateResult, &'static str> {
        if terminal_payoff.len() != self.config.num_grid_points {
            return Err("Terminal payoff size mismatch");
        }

        let params = self.derive_model_params();

        let wick_result = self.wick_solver.solve(
            terminal_payoff,
            params.drift,
            params.volatility,
            params.risk_free_rate,
        )?;

        let present_value = WickKolmogorovSolver::extract_real_solution(&wick_result);
        let probability = self.calculate_boundary_probability(&wick_result);
        let expected_time = self.estimate_first_passage_time(&present_value);
        let is_stable = self.verify_solution_stability(&present_value);

        Ok(FinalStateResult {
            present_value,
            probability,
            expected_time,
            is_stable,
        })
    }

    fn derive_model_params(&self) -> ModelParams {
        match &self.config.boundary_type {
            FinalBoundaryType::CurrencyCollapse { decay_rate } => ModelParams {
                drift: -decay_rate.abs(),
                volatility: 0.3,
                risk_free_rate: 0.05,
                dividend_yield: 0.0,
            },
            FinalBoundaryType::VolatilityExtinction { timescale } => ModelParams {
                drift: 0.0,
                volatility: 0.1 / timescale.abs().max(0.1),
                risk_free_rate: 0.02,
                dividend_yield: 0.0,
            },
            FinalBoundaryType::PhaseTransition { critical_time } => ModelParams {
                drift: 1.0 / critical_time.abs().max(1.0),
                volatility: 0.5,
                risk_free_rate: 0.03,
                dividend_yield: 0.0,
            },
            FinalBoundaryType::FiniteTimeSingularity { singularity_time } => ModelParams {
                drift: 2.0 / singularity_time.abs().max(1.0),
                volatility: 1.0,
                risk_free_rate: 0.01,
                dividend_yield: 0.0,
            },
            FinalBoundaryType::Cyclic { period } => ModelParams {
                drift: 0.0,
                volatility: 0.2,
                risk_free_rate: 0.025,
                dividend_yield: 1.0 / period.abs().max(1.0),
            },
        }
    }

    fn calculate_boundary_probability(&self, result: &WickEvolutionResult) -> f64 {
        let mut total_prob = 0.0;
        for psi in &result.wave_function {
            total_prob += psi.norm_sq();
        }
        total_prob.min(1.0).max(0.0)
    }

    fn estimate_first_passage_time(&self, present_value: &[f64]) -> f64 {
        if present_value.is_empty() {
            return self.config.time_horizon;
        }

        let max_idx = present_value
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let n = present_value.len() as f64;
        let fraction = max_idx as f64 / n;
        fraction * self.config.time_horizon
    }

    fn verify_solution_stability(&self, present_value: &[f64]) -> bool {
        if present_value.len() < 3 {
            return false;
        }

        let mut has_nan = false;
        let mut has_inf = false;
        let mut max_gradient = 0.0;

        for &v in present_value {
            if v.is_nan() {
                has_nan = true;
            }
            if v.is_infinite() {
                has_inf = true;
            }
        }

        for i in 1..present_value.len() {
            let grad = (present_value[i] - present_value[i - 1]).abs();
            if grad > max_gradient {
                max_gradient = grad;
            }
        }

        !has_nan && !has_inf && max_gradient < 1e6
    }

    /// Calculate eschatological option price
    pub fn price_eschatological_option(
        &self,
        strike: f64,
        option_type: OptionType,
    ) -> Result<FinalStateResult, &'static str> {
        let n = self.config.num_grid_points;
        let (x_min, x_max) = self.config.spatial_range;
        let dx = (x_max - x_min) / (n - 1) as f64;

        // Build terminal payoff
        let mut terminal_payoff = Vec::with_capacity(n);
        for i in 0..n {
            let s = (x_min + i as f64 * dx).exp();
            let payoff = match option_type {
                OptionType::Call => (s - strike).max(0.0),
                OptionType::Put => (strike - s).max(0.0),
            };
            terminal_payoff.push(payoff);
        }

        self.solve_backward(&terminal_payoff)
    }
}

/// Option type for eschatological derivatives
#[derive(Debug, Clone, Copy)]
pub enum OptionType {
    Call,
    Put,
}

/// Builder for complex final state scenarios
pub struct FinalStateBuilder {
    config: FinalStateConfig,
}

impl FinalStateBuilder {
    pub fn new() -> Self {
        Self {
            config: FinalStateConfig::default(),
        }
    }

    pub fn with_boundary_type(mut self, boundary: FinalBoundaryType) -> Self {
        self.config.boundary_type = boundary;
        self
    }

    pub fn with_time_horizon(mut self, horizon: f64) -> Self {
        if horizon > 0.0 {
            self.config.time_horizon = horizon;
        }
        self
    }

    pub fn with_spatial_range(mut self, min: f64, max: f64) -> Self {
        if min < max && min > 0.0 {
            self.config.spatial_range = (min, max);
        }
        self
    }

    pub fn build(self) -> FinalStateBoundarySolver {
        FinalStateBoundarySolver::new(self.config)
    }
}

impl Default for FinalStateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_final_state_builder() {
        let solver = FinalStateBuilder::new()
            .with_boundary_type(FinalBoundaryType::CurrencyCollapse { decay_rate: 0.03 })
            .with_time_horizon(20.0)
            .build();

        assert_eq!(solver.config.time_horizon, 20.0);
    }

    #[test]
    fn test_currency_collapse_params() {
        let config = FinalStateConfig {
            boundary_type: FinalBoundaryType::CurrencyCollapse { decay_rate: 0.05 },
            ..Default::default()
        };
        let solver = FinalStateBoundarySolver::new(config);
        let params = solver.derive_model_params();

        assert!(params.drift < 0.0);
    }

    #[test]
    fn test_solve_backward_basic() {
        let config = FinalStateConfig::default();
        let solver = FinalStateBoundarySolver::new(config);

        let terminal: Vec<f64> = (0..256).map(|i| (i as f64 / 256.0).powi(2)).collect();
        let result = solver.solve_backward(&terminal);

        assert!(result.is_ok());
        let res = result.unwrap();
        assert!(res.probability >= 0.0);
        assert!(res.probability <= 1.0);
    }
}
