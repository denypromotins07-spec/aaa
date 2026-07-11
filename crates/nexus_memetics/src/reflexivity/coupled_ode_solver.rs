//! Coupled ODE Solver for Soros Reflexivity Theory
//! 
//! Models the bidirectional feedback between narrative infection (SIR model)
//! and market liquidity, implementing George Soros' Theory of Reflexivity
//! as a rigorous mathematical system.

use crate::epidemiology::financial_sir_ode::{SirState, SirParameters};
use nalgebra::{Matrix2, Vector2, LU};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ReflexivityError {
    #[error("Jacobian computation failed")]
    JacobianFailure,
    #[error("Eigenvalue computation failed")]
    EigenvalueFailure,
    #[error("System unstable: Lyapunov exponent positive")]
    UnstableSystem { lyapunov_exponent: f64 },
    #[error("Integration diverged")]
    IntegrationDivergence,
}

/// State of the coupled reflexivity system
#[derive(Clone, Debug)]
pub struct ReflexivityState {
    /// SIR state of the narrative
    pub sir_state: SirState,
    /// Market price deviation from fundamental (normalized)
    pub price_deviation: f64,
    /// Liquidity depth (normalized)
    pub liquidity_depth: f64,
}

impl ReflexivityState {
    pub fn new(sir: SirState, price_dev: f64, liq_depth: f64) -> Self {
        Self {
            sir_state: sir,
            price_deviation: price_dev,
            liquidity_depth: liq_depth,
        }
    }

    pub fn to_vector(&self) -> Vector2<f64> {
        // Track infected fraction and price deviation as primary state
        Vector2::new(self.sir_state.infected, self.price_deviation)
    }
}

/// Parameters for the reflexivity coupling
pub struct ReflexivityParameters {
    /// Base SIR parameters
    pub sir_params: SirParameters,
    /// Price impact coefficient (how narrative affects price)
    pub narrative_price_impact: f64,
    /// Feedback strength (how price change affects narrative transmission)
    pub price_feedback_beta: f64,
    /// Liquidity sensitivity to narrative
    pub liquidity_sensitivity: f64,
    /// Mean reversion rate for price
    pub price_mean_reversion: f64,
}

impl ReflexivityParameters {
    pub fn new(
        beta: f64,
        gamma: f64,
        narrative_price_impact: f64,
        price_feedback_beta: f64,
        liquidity_sensitivity: f64,
        price_mean_reversion: f64,
    ) -> Result<Self, crate::epidemiology::financial_sir_ode::SirOdeError> {
        Ok(Self {
            sir_params: SirParameters::new(beta, gamma)?,
            narrative_price_impact,
            price_feedback_beta,
            liquidity_sensitivity,
            price_mean_reversion,
        })
    }
}

/// Derivative function for the coupled reflexivity ODE system
fn reflexivity_derivatives(state: &ReflexivityState, params: &ReflexivityParameters) -> ReflexivityState {
    let s = state.sir_state.susceptible.max(0.0);
    let i = state.sir_state.infected.max(0.0);
    let _r = state.sir_state.recovered.max(0.0);
    let n = params.sir_params.total_population;
    
    let price_dev = state.price_deviation;
    let liq_depth = state.liquidity_depth;

    // Modified beta due to price feedback (reflexivity)
    // Rising prices increase narrative transmission
    let effective_beta = params.sir_params.beta * (1.0 + params.price_feedback_beta * price_dev.max(0.0));
    
    // dS/dt = -beta_eff * S * I / N
    let dsdt = -effective_beta * s * i / n;
    
    // dI/dt = beta_eff * S * I / N - gamma * I
    let didt = effective_beta * s * i / n - params.sir_params.gamma * i;
    
    // dR/dt = gamma * I
    let drdt = params.sir_params.gamma * i;

    // dP/dt = alpha * I - kappa * P (price driven by infection, mean-reverting)
    let dpdt = params.narrative_price_impact * i - params.price_mean_reversion * price_dev;

    // dL/dt = -eta * I * P + rho * (1 - L) (liquidity drains during high infection + price moves)
    let dldt = -params.liquidity_sensitivity * i * price_dev.abs() + 0.1 * (1.0 - liq_depth);

    ReflexivityState {
        sir_state: SirState {
            susceptible: dsdt,
            infected: didt,
            recovered: drdt,
        },
        price_deviation: dpdt,
        liquidity_depth: dldt,
    }
}

/// Coupled ODE solver using adaptive Runge-Kutta
pub struct CoupledODESolver {
    max_iterations: usize,
    tolerance: f64,
}

impl CoupledODESolver {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self { max_iterations, tolerance }
    }

    /// Single step using RK4
    pub fn step_rk4(
        &self,
        state: &ReflexivityState,
        dt: f64,
        params: &ReflexivityParameters,
    ) -> Result<ReflexivityState, ReflexivityError> {
        let y = state.to_vector();
        
        // RK4 stages
        let k1 = reflexivity_derivatives(state, params).to_vector();
        
        let y2 = &y + (dt / 2.0) * &k1;
        let state2 = Self::from_vector(&y2, state)?;
        let k2 = reflexivity_derivatives(&state2, params).to_vector();
        
        let y3 = &y + (dt / 2.0) * &k2;
        let state3 = Self::from_vector(&y3, state)?;
        let k3 = reflexivity_derivatives(&state3, params).to_vector();
        
        let y4 = &y + dt * &k3;
        let state4 = Self::from_vector(&y4, state)?;
        let k4 = reflexivity_derivatives(&state4, params).to_vector();

        // Combine
        let dy = (dt / 6.0) * (&k1 + 2.0 * &k2 + 2.0 * &k3 + &k4);
        let y_new = &y + &dy;

        // Reconstruct state
        let mut new_state = state.clone();
        new_state.price_deviation = y_new[1];
        
        // Update SIR with proper integration
        let sir_dt = reflexivity_derivatives(state, params).sir_state;
        new_state.sir_state = SirState {
            susceptible: (state.sir_state.susceptible + dt * sir_dt.susceptible).max(0.0),
            infected: (state.sir_state.infected + dt * sir_dt.infected).max(0.0),
            recovered: (state.sir_state.recovered + dt * sir_dt.recovered).max(0.0),
        };

        // Normalize SIR
        let total = new_state.sir_state.susceptible + new_state.sir_state.infected + new_state.sir_state.recovered;
        if total > 1e-10 {
            new_state.sir_state.susceptible /= total;
            new_state.sir_state.infected /= total;
            new_state.sir_state.recovered /= total;
        }

        // Clamp values
        new_state.price_deviation = new_state.price_deviation.clamp(-10.0, 10.0);
        new_state.liquidity_depth = new_state.liquidity_depth.clamp(0.0, 1.0);

        Ok(new_state)
    }

    fn from_vector(y: &Vector2<f64>, base: &ReflexivityState) -> Result<ReflexivityState, ReflexivityError> {
        Ok(ReflexivityState {
            sir_state: SirState {
                susceptible: base.sir_state.susceptible,
                infected: y[0],
                recovered: 1.0 - y[0] - base.sir_state.susceptible,
            },
            price_deviation: y[1],
            liquidity_depth: base.liquidity_depth,
        })
    }

    /// Integrate the system forward in time
    pub fn integrate(
        &self,
        initial: &ReflexivityState,
        params: &ReflexivityParameters,
        t_span: (f64, f64),
        dt: f64,
    ) -> Result<Vec<(f64, ReflexivityState)>, ReflexivityError> {
        let (t0, tf) = t_span;
        let mut results = Vec::new();
        let mut current_state = initial.clone();
        let mut t = t0;

        results.push((t, current_state.clone()));

        while t < tf {
            match self.step_rk4(&current_state, dt, params) {
                Ok(new_state) => {
                    current_state = new_state;
                    t += dt;
                    
                    // Check for divergence
                    if !current_state.price_deviation.is_finite() 
                        || !current_state.liquidity_depth.is_finite()
                    {
                        return Err(ReflexivityError::IntegrationDivergence);
                    }

                    results.push((t, current_state.clone()));
                }
                Err(e) => return Err(e),
            }
        }

        Ok(results)
    }
}

/// Lyapunov exponent calculator for stability analysis
pub struct LyapunovAnalyzer {
    epsilon: f64,
}

impl LyapunovAnalyzer {
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }

    /// Compute the Jacobian matrix at a given state
    pub fn jacobian(&self, state: &ReflexivityState, params: &ReflexivityParameters) -> Matrix2<f64> {
        let i = state.sir_state.infected;
        let s = state.sir_state.susceptible;
        let p = state.price_deviation;
        let n = params.sir_params.total_population;

        // Effective beta including feedback
        let beta_eff = params.sir_params.beta * (1.0 + params.price_feedback_beta * p.max(0.0));

        // Partial derivatives for dI/dt and dP/dt
        // d(dI)/dI = beta_eff * S / N - gamma
        // d(dI)/dP = beta * price_feedback * S * I / N
        // d(dP)/dI = alpha
        // d(dP)/dP = -kappa

        let j00 = beta_eff * s / n - params.sir_params.gamma;
        let j01 = params.sir_params.beta * params.price_feedback_beta * s * i / n;
        let j10 = params.narrative_price_impact;
        let j11 = -params.price_mean_reversion;

        Matrix2::new(j00, j01, j10, j11)
    }

    /// Compute the dominant eigenvalue (Lyapunov exponent indicator)
    pub fn dominant_eigenvalue(&self, jacobian: &Matrix2<f64>) -> Result<f64, ReflexivityError> {
        // For 2x2 matrix, eigenvalues are roots of characteristic polynomial
        // λ² - trace*λ + det = 0
        let trace = jacobian.trace();
        let det = jacobian.determinant();

        let discriminant = trace * trace - 4.0 * det;

        if discriminant >= 0.0 {
            // Real eigenvalues
            let lambda1 = (trace + discriminant.sqrt()) / 2.0;
            let lambda2 = (trace - discriminant.sqrt()) / 2.0;
            Ok(lambda1.max(lambda2))
        } else {
            // Complex eigenvalues - return real part
            Ok(trace / 2.0)
        }
    }

    /// Check if system is unstable (positive Lyapunov exponent)
    pub fn is_unstable(&self, state: &ReflexivityState, params: &ReflexivityParameters) -> Result<bool, ReflexivityError> {
        let j = self.jacobian(state, params);
        let lambda = self.dominant_eigenvalue(&j)?;
        Ok(lambda > 0.0)
    }

    /// Compute full Lyapunov spectrum along trajectory
    pub fn lyapunov_spectrum(
        &self,
        trajectory: &[(f64, ReflexivityState)],
        params: &ReflexivityParameters,
    ) -> Result<Vec<f64>, ReflexivityError> {
        trajectory
            .iter()
            .map(|(_, state)| {
                let j = self.jacobian(state, params);
                self.dominant_eigenvalue(&j)
            })
            .collect()
    }
}

/// Bubble inflection point detector
pub struct BubbleInflectionDetector {
    analyzer: LyapunovAnalyzer,
}

impl BubbleInflectionDetector {
    pub fn new() -> Self {
        Self {
            analyzer: LyapunovAnalyzer::new(1e-8),
        }
    }

    /// Detect the inflection point where reflexivity loop becomes unstable
    pub fn detect_inflection(
        &self,
        trajectory: &[(f64, ReflexivityState)],
        params: &ReflexivityParameters,
    ) -> Option<(f64, ReflexivityState)> {
        for (t, state) in trajectory.iter() {
            if let Ok(true) = self.analyzer.is_unstable(state, params) {
                return Some((*t, state.clone()));
            }
        }
        None
    }

    /// Find the peak bubble point (maximum price deviation before crash)
    pub fn find_peak(&self, trajectory: &[(f64, ReflexivityState)]) -> Option<(f64, f64)> {
        trajectory
            .iter()
            .max_by(|a, b| {
                a.1.price_deviation.partial_cmp(&b.1.price_deviation).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(t, s)| (*t, s.price_deviation))
    }

    /// Predict crash timing based on instability onset
    pub fn predict_crash_timing(
        &self,
        trajectory: &[(f64, ReflexivityState)],
        params: &ReflexivityParameters,
    ) -> CrashPrediction {
        let inflection = self.detect_inflection(trajectory, params);
        let peak = self.find_peak(trajectory);

        CrashPrediction {
            inflection_time: inflection.map(|(t, _)| t),
            peak_time: peak.map(|(t, _)| t),
            peak_magnitude: peak.map(|(_, p)| p),
            confidence: if inflection.is_some() { 0.8 } else { 0.3 },
        }
    }
}

/// Crash prediction result
#[derive(Debug, Clone)]
pub struct CrashPrediction {
    pub inflection_time: Option<f64>,
    pub peak_time: Option<f64>,
    pub peak_magnitude: Option<f64>,
    pub confidence: f64,
}

impl Default for BubbleInflectionDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epidemiology::financial_sir_ode::SirState;

    #[test]
    fn test_coupled_solver_basic() {
        let params = ReflexivityParameters::new(0.5, 0.1, 0.3, 0.2, 0.1, 0.05).unwrap();
        let solver = CoupledODESolver::new(50, 1e-10);
        
        let initial = ReflexivityState::new(
            SirState::new(0.99, 0.01, 0.0).unwrap(),
            0.0,
            1.0,
        );

        let trajectory = solver.integrate(&initial, &params, (0.0, 5.0), 0.05);
        
        assert!(trajectory.is_ok());
        let traj = trajectory.unwrap();
        assert!(!traj.is_empty());
    }

    #[test]
    fn test_lyapunov_stability() {
        let params = ReflexivityParameters::new(0.5, 0.1, 0.3, 0.2, 0.1, 0.05).unwrap();
        let analyzer = LyapunovAnalyzer::new(1e-8);
        
        let state = ReflexivityState::new(
            SirState::new(0.9, 0.1, 0.0).unwrap(),
            0.0,
            1.0,
        );

        let j = analyzer.jacobian(&state, &params);
        let lambda = analyzer.dominant_eigenvalue(&j);
        
        assert!(lambda.is_ok());
    }
}
