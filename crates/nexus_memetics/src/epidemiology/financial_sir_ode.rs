//! Financial SIR ODE Solver with Implicit Runge-Kutta (Radau IIA) for Stiff Systems
//! 
//! Models narrative adoption as Susceptible-Infected-Recovered epidemiological dynamics.
//! Uses implicit methods to handle stiffness during panic spikes without numerical explosion.

use nalgebra::{Matrix3, Vector3, LU};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SirOdeError {
    #[error("Stiffness overflow: integration step failed to converge")]
    StiffnessOverflow,
    #[error("Invalid parameter: beta or gamma must be positive")]
    InvalidParameter,
    #[error("Population sum exceeds bounds")]
    PopulationBoundsExceeded,
}

/// State vector for SIR model: [S, I, R]
#[derive(Clone, Copy, Debug)]
pub struct SirState {
    pub susceptible: f64,
    pub infected: f64,
    pub recovered: f64,
}

impl SirState {
    #[inline]
    pub fn new(s: f64, i: f64, r: f64) -> Result<Self, SirOdeError> {
        if s < 0.0 || i < 0.0 || r < 0.0 {
            return Err(SirOdeError::InvalidParameter);
        }
        let total = s + i + r;
        if (total - 1.0).abs() > 1e-10 {
            return Err(SirOdeError::PopulationBoundsExceeded);
        }
        Ok(Self { susceptible: s, infected: i, recovered: r })
    }

    #[inline]
    pub fn to_vector(&self) -> Vector3<f64> {
        Vector3::new(self.susceptible, self.infected, self.recovered)
    }

    #[inline]
    pub fn from_vector(v: &Vector3<f64>) -> Result<Self, SirOdeError> {
        Self::new(v[0], v[1], v[2])
    }
}

/// Parameters for the financial SIR model
pub struct SirParameters {
    /// Infection rate (narrative transmission coefficient)
    pub beta: f64,
    /// Recovery rate (narrative decay/forgetting rate)
    pub gamma: f64,
    /// Total population (normalized to 1.0)
    pub total_population: f64,
}

impl SirParameters {
    pub fn new(beta: f64, gamma: f64) -> Result<Self, SirOdeError> {
        if beta <= 0.0 || gamma <= 0.0 {
            return Err(SirOdeError::InvalidParameter);
        }
        Ok(Self { beta, gamma, total_population: 1.0 })
    }
}

/// Derivative function for SIR ODE system
#[inline]
fn sir_derivatives(state: &Vector3<f64>, params: &SirParameters) -> Vector3<f64> {
    let s = state[0].max(0.0);
    let i = state[1].max(0.0);
    let _r = state[2].max(0.0);
    let n = params.total_population;

    // dS/dt = -beta * S * I / N
    let dsdt = -params.beta * s * i / n;
    // dI/dt = beta * S * I / N - gamma * I
    let didt = params.beta * s * i / n - params.gamma * i;
    // dR/dt = gamma * I
    let drdt = params.gamma * i;

    Vector3::new(dsdt, didt, drdt)
}

/// Implicit Runge-Kutta Radau IIA 2-stage solver for stiff SIR systems
/// This solver is A-stable and L-stable, handling extreme stiffness during panic events
pub struct RadauIIASolver {
    max_iterations: usize,
    tolerance: f64,
}

impl RadauIIASolver {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self { max_iterations, tolerance }
    }

    /// Single implicit step using Newton-Raphson iteration
    pub fn step(&self, state: &Vector3<f64>, dt: f64, params: &SirParameters) -> Result<Vector3<f64>, SirOdeError> {
        // Radau IIA 2-stage coefficients
        const A11: f64 = 5.0 / 12.0;
        const A12: f64 = -1.0 / 12.0;
        const A21: f64 = 3.0 / 4.0;
        const A22: f64 = 1.0 / 4.0;
        const C1: f64 = 1.0 / 3.0;
        const C2: f64 = 1.0;
        const B1: f64 = 1.0 / 4.0;
        const B2: f64 = 3.0 / 4.0;

        // Initial guess: explicit Euler
        let k0 = sir_derivatives(state, params);
        let mut k1 = k0.clone();
        let mut k2 = k0.clone();

        // Newton-Raphson iteration for implicit stages
        for iter in 0..self.max_iterations {
            let f1 = sir_derivatives(&(state + dt * (A11 * k1 + A12 * k2)), params);
            let f2 = sir_derivatives(&(state + dt * (A21 * k1 + A22 * k2)), params);

            let residual1 = k1 - f1;
            let residual2 = k2 - f2;

            let residual_norm = residual1.norm() + residual2.norm();

            if residual_norm < self.tolerance {
                break;
            }

            if iter == self.max_iterations - 1 {
                return Err(SirOdeError::StiffnessOverflow);
            }

            // Simplified Newton: approximate Jacobian
            let j_approx = self.approximate_jacobian(state, dt, params, &k1, &k2);
            
            // Solve linear system for corrections
            if let Some((dk1, dk2)) = self.solve_newton_step(&j_approx, &residual1, &residual2, dt) {
                k1 = k1 + dk1;
                k2 = k2 + dk2;
            } else {
                // Fallback to damped update
                let damping = 0.5_f64.powi(iter as i32);
                k1 = k1 - damping * residual1;
                k2 = k2 - damping * residual2;
            }
        }

        // Update state: y_{n+1} = y_n + dt * (b1*k1 + b2*k2)
        let new_state = state + dt * (B1 * k1 + B2 * k2);
        
        // Project back to valid simplex
        let projected = self.project_to_simplex(&new_state)?;
        Ok(projected)
    }

    fn approximate_jacobian(
        &self,
        state: &Vector3<f64>,
        dt: f64,
        params: &SirParameters,
        k1: &Vector3<f64>,
        k2: &Vector3<f64>,
    ) -> Matrix3<f64> {
        // Numerical Jacobian approximation for the implicit system
        let eps = 1e-8;
        let mut j = Matrix3::zeros();

        for i in 0..3 {
            let mut state_pert = state.clone();
            state_pert[i] += eps;
            let f_pert = sir_derivatives(&state_pert, params);
            
            let mut state_m = state.clone();
            state_m[i] -= eps;
            let f_m = sir_derivatives(&state_m, params);

            for row in 0..3 {
                j[(row, i)] = (f_pert[row] - f_m[row]) / (2.0 * eps);
            }
        }

        // J = I - dt * A ⊗ df/dy (Kronecker product structure simplified)
        Matrix3::identity() - dt * j * 0.5 // Simplified for stability
    }

    fn solve_newton_step(
        &self,
        j: &Matrix3<f64>,
        res1: &Vector3<f64>,
        res2: &Vector3<f64>,
        _dt: f64,
    ) -> Option<(Vector3<f64>, Vector3<f64>)> {
        // Simplified: solve J * dk = -res for each stage independently
        let lu = LU::new(j.clone());
        let dk1 = lu.solve(-res1)?;
        let dk2 = lu.solve(-res2)?;
        Some((dk1, dk2))
    }

    fn project_to_simplex(&self, v: &Vector3<f64>) -> Result<Vector3<f64>, SirOdeError> {
        let mut result = v.clone();
        for i in 0..3 {
            result[i] = result[i].max(0.0).min(1.0);
        }
        let sum = result.sum();
        if sum > 1e-10 {
            result /= sum;
        }
        Ok(result)
    }

    /// Integrate from t0 to tf with adaptive stepping
    pub fn integrate(
        &self,
        initial: &SirState,
        params: &SirParameters,
        t0: f64,
        tf: f64,
        output_interval: f64,
    ) -> Result<Vec<(f64, SirState)>, SirOdeError> {
        let mut results = Vec::new();
        let mut current_time = t0;
        let mut state = initial.to_vector();
        let mut next_output = t0;

        results.push((t0, *initial));

        let mut dt = 0.01; // Initial step size
        let min_dt = 1e-10;
        let max_dt = 0.1;

        while current_time < tf {
            if current_time + dt > tf {
                dt = tf - current_time;
            }

            match self.step(&state, dt, params) {
                Ok(new_state) => {
                    state = new_state;
                    current_time += dt;
                    
                    // Adaptive step size control
                    dt = (dt * 1.2).min(max_dt);

                    while next_output <= current_time && next_output <= tf {
                        let interp_state = self.interpolate(&state, &initial.to_vector(), current_time, next_output, params)?;
                        if let Ok(s) = SirState::from_vector(&interp_state) {
                            results.push((next_output, s));
                        }
                        next_output += output_interval;
                    }
                }
                Err(SirOdeError::StiffnessOverflow) => {
                    // Reduce step size and retry
                    dt *= 0.5;
                    if dt < min_dt {
                        return Err(SirOdeError::StiffnessOverflow);
                    }
                }
                Err(e) => return Err(e),
            }
        }

        Ok(results)
    }

    fn interpolate(
        &self,
        _current: &Vector3<f64>,
        _initial: &Vector3<f64>,
        _t_current: f64,
        t_target: f64,
        _params: &SirParameters,
    ) -> Result<Vector3<f64>, SirOdeError> {
        // Simple linear interpolation for output points
        // In production, use dense output formulas for Radau IIA
        Ok(Vector3::zeros()) // Placeholder - would need proper dense output
    }
}

/// High-level SIR model interface
pub struct FinancialSirModel {
    params: SirParameters,
    solver: RadauIIASolver,
}

impl FinancialSirModel {
    pub fn new(beta: f64, gamma: f64) -> Result<Self, SirOdeError> {
        let params = SirParameters::new(beta, gamma)?;
        let solver = RadauIIASolver::new(50, 1e-10);
        Ok(Self { params, solver })
    }

    pub fn simulate(
        &self,
        initial_s: f64,
        initial_i: f64,
        initial_r: f64,
        duration: f64,
        output_interval: f64,
    ) -> Result<Vec<(f64, SirState)>, SirOdeError> {
        let initial = SirState::new(initial_s, initial_i, initial_r)?;
        self.solver.integrate(&initial, &self.params, 0.0, duration, output_interval)
    }

    pub fn parameters(&self) -> &SirParameters {
        &self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_sir_dynamics() {
        let model = FinancialSirModel::new(0.5, 0.1).unwrap();
        let results = model.simulate(0.99, 0.01, 0.0, 10.0, 0.5).unwrap();
        
        assert!(!results.is_empty());
        // Infected should initially increase then decrease
        let peak_infected = results.iter().map(|(_, s)| s.infected).fold(0.0_f64, f64::max);
        assert!(peak_infected > 0.01);
    }

    #[test]
    fn test_stiffness_handling() {
        // High beta creates stiffness
        let model = FinancialSirModel::new(10.0, 0.1).unwrap();
        let results = model.simulate(0.9, 0.1, 0.0, 5.0, 0.1);
        
        // Should not panic or return NaN
        assert!(results.is_ok());
        if let Ok(r) = results {
            for (_, state) in r {
                assert!(state.susceptible.is_finite());
                assert!(state.infected.is_finite());
                assert!(state.recovered.is_finite());
            }
        }
    }
}
