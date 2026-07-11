//! Stiff Runge-Kutta Solver Implementations for Financial ODE Systems
//! 
//! Provides multiple implicit RK methods optimized for stiff epidemiological models
//! encountered during market panic events.

use nalgebra::{Matrix3, Vector3, LU, SVD};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum StiffRKError {
    #[error("Newton iteration failed to converge after {0} iterations")]
    NewtonNonConvergence(usize),
    #[error("Linear solver failed: singular matrix")]
    SingularMatrix,
    #[error("Step size underflow: cannot achieve requested tolerance")]
    StepSizeUnderflow,
    #[error("Invalid state: negative populations detected")]
    InvalidState,
}

/// Adaptive step size controller using embedded error estimates
pub struct StepController {
    initial_dt: f64,
    min_dt: f64,
    max_dt: f64,
    abs_tol: f64,
    rel_tol: f64,
    safety_factor: f64,
    max_step_increase: f64,
    max_step_decrease: f64,
}

impl StepController {
    pub fn new(
        initial_dt: f64,
        abs_tol: f64,
        rel_tol: f64,
    ) -> Self {
        Self {
            initial_dt,
            min_dt: 1e-15,
            max_dt: 1.0,
            abs_tol,
            rel_tol,
            safety_factor: 0.9,
            max_step_increase: 5.0,
            max_step_decrease: 0.2,
        }
    }

    /// Calculate optimal next step size based on error estimate
    #[inline]
    pub fn compute_next_step(&self, current_dt: f64, error_norm: f64) -> (f64, bool) {
        if error_norm < 1e-15 {
            // Error is essentially zero, can increase step significantly
            let new_dt = (current_dt * self.max_step_increase).min(self.max_dt);
            return (new_dt, true);
        }

        // Optimal step factor: (tol / error)^(1/(p+1)) where p is method order
        // For Radau IIA 2-stage, p = 3, so exponent = 1/4
        let exponent = 0.25;
        let tol = self.abs_tol + self.rel_tol * error_norm;
        let mut factor = self.safety_factor * (tol / error_norm).powf(exponent);

        // Limit step size changes
        factor = factor.clamp(self.max_step_decrease, self.max_step_increase);
        
        let new_dt = (current_dt * factor).clamp(self.min_dt, self.max_dt);
        let accepted = error_norm <= tol;

        (new_dt, accepted)
    }
}

/// Gauss-Legendre implicit Runge-Kutta method (order 4, 2 stages)
/// A-stable and symplectic, excellent for oscillatory systems
pub struct GaussLegendreRK {
    controller: StepController,
    max_newton_iter: usize,
}

impl GaussLegendreRK {
    pub fn new(abs_tol: f64, rel_tol: f64, max_newton_iter: usize) -> Self {
        Self {
            controller: StepController::new(0.01, abs_tol, rel_tol),
            max_newton_iter,
        }
    }

    /// Single step of GLRK4
    pub fn step<F>(
        &self,
        t: f64,
        y: &Vector3<f64>,
        dt: f64,
        f: &F,
    ) -> Result<(Vector3<f64>, f64), StiffRKError>
    where
        F: Fn(f64, &Vector3<f64>) -> Vector3<f64>,
    {
        // Gauss-Legendre nodes and weights for 2 stages
        const C1: f64 = 0.5 - std::f64::consts::FRAC_1_SQRT_3 / 6.0; // (3 - sqrt(3)) / 6
        const C2: f64 = 0.5 + std::f64::consts::FRAC_1_SQRT_3 / 6.0; // (3 + sqrt(3)) / 6
        const A11: f64 = 0.25;
        const A12: f64 = 0.25 - std::f64::consts::FRAC_1_SQRT_3 / 6.0;
        const A21: f64 = 0.25 + std::f64::consts::FRAC_1_SQRT_3 / 6.0;
        const A22: f64 = 0.25;
        const B1: f64 = 0.5;
        const B2: f64 = 0.5;

        // Initial guess: explicit Euler
        let k_init = f(t, y);
        let mut k1 = k_init.clone();
        let mut k2 = k_init.clone();

        // Newton iteration for implicit stages
        for iter in 0..self.max_newton_iter {
            let y1 = y + dt * (A11 * k1 + A12 * k2);
            let y2 = y + dt * (A21 * k1 + A22 * k2);

            let f1 = f(t + C1 * dt, &y1);
            let f2 = f(t + C2 * dt, &y2);

            let g1 = k1 - f1;
            let g2 = k2 - f2;

            let residual_norm = g1.norm() + g2.norm();

            if residual_norm < 1e-12 {
                break;
            }

            if iter == self.max_newton_iter - 1 {
                return Err(StiffRKError::NewtonNonConvergence(iter));
            }

            // Simplified Newton: use approximate Jacobian
            let j = self.approximate_jacobian(y, dt, f);
            
            match self.solve_newton_system(&j, &g1, &g2) {
                Some((dk1, dk2)) => {
                    k1 += dk1;
                    k2 += dk2;
                }
                None => {
                    // Damped update fallback
                    let damping = 0.5_f64.powi(iter as i32);
                    k1 -= damping * g1;
                    k2 -= damping * g2;
                }
            }
        }

        // Update solution
        let y_new = y + dt * (B1 * k1 + B2 * k2);
        
        // Error estimate via embedded method (simplified: compare with explicit Euler)
        let y_euler = y + dt * k_init;
        let error = (y_new - y_euler).norm();

        Ok((y_new, error))
    }

    fn approximate_jacobian<F>(
        &self,
        y: &Vector3<f64>,
        _dt: f64,
        f: &F,
    ) -> Matrix3<f64>
    where
        F: Fn(f64, &Vector3<f64>) -> Vector3<f64>,
    {
        let eps = 1e-8;
        let mut j = Matrix3::zeros();
        let fy = f(0.0, y);

        for i in 0..3 {
            let mut y_pert = y.clone();
            y_pert[i] += eps;
            let f_pert = f(0.0, &y_pert);

            for row in 0..3 {
                j[(row, i)] = (f_pert[row] - fy[row]) / eps;
            }
        }

        j
    }

    fn solve_newton_system(
        &self,
        j: &Matrix3<f64>,
        g1: &Vector3<f64>,
        g2: &Vector3<f64>,
    ) -> Option<(Vector3<f64>, Vector3<f64>)> {
        // For coupled system, would need 6x6 solve
        // Simplified: solve independently (approximation)
        let lu = LU::new(j.clone())?;
        let dk1 = lu.solve(-g1)?;
        let dk2 = lu.solve(-g2)?;
        Some((dk1, dk2))
    }
}

/// Radau IIA 3rd order (2 stages) - L-stable, excellent for very stiff problems
pub struct RadauIIA3 {
    controller: StepController,
    max_newton_iter: usize,
}

impl RadauIIA3 {
    pub fn new(abs_tol: f64, rel_tol: f64, max_newton_iter: usize) -> Self {
        Self {
            controller: StepController::new(0.01, abs_tol, rel_tol),
            max_newton_iter,
        }
    }

    pub fn step<F>(
        &self,
        t: f64,
        y: &Vector3<f64>,
        mut dt: f64,
        f: &F,
    ) -> Result<(Vector3<f64>, f64), StiffRKError>
    where
        F: Fn(f64, &Vector3<f64>) -> Vector3<f64>,
    {
        // Radau IIA coefficients (2 stages, order 3)
        const C1: f64 = 1.0 / 3.0;
        const C2: f64 = 1.0;
        const A11: f64 = 5.0 / 12.0;
        const A12: f64 = -1.0 / 12.0;
        const A21: f64 = 3.0 / 4.0;
        const A22: f64 = 1.0 / 4.0;
        const B1: f64 = 1.0 / 4.0;
        const B2: f64 = 3.0 / 4.0;

        let mut attempts = 0;
        let max_attempts = 10;

        while attempts < max_attempts {
            let k_init = f(t, y);
            let mut k1 = k_init.clone();
            let mut k2 = k_init.clone();

            // Newton iteration
            for iter in 0..self.max_newton_iter {
                let y1 = y + dt * (A11 * k1 + A12 * k2);
                let y2 = y + dt * (A21 * k1 + A22 * k2);

                let f1 = f(t + C1 * dt, &y1);
                let f2 = f(t + C2 * dt, &y2);

                let g1 = k1 - f1;
                let g2 = k2 - f2;

                let residual = g1.norm() + g2.norm();

                if residual < 1e-12 {
                    break;
                }

                if iter == self.max_newton_iter - 1 {
                    // Reduce step and retry
                    dt *= 0.5;
                    break;
                }

                // Simplified Newton with analytical Jacobian approximation
                let j = self.approximate_jacobian(y, dt, f);
                
                if let Some((dk1, dk2)) = self.solve_coupled(&j, &g1, &g2, dt) {
                    k1 += dk1;
                    k2 += dk2;
                } else {
                    dt *= 0.5;
                    break;
                }
            }

            let y_new = y + dt * (B1 * k1 + B2 * k2);
            
            // Validate state
            if y_new.iter().any(|&v| v < -1e-10) {
                dt *= 0.5;
                attempts += 1;
                continue;
            }

            // Error estimate
            let y_euler = y + dt * k_init;
            let error = (y_new - y_euler).norm();

            let (new_dt, accepted) = self.controller.compute_next_step(dt, error);
            
            if accepted {
                return Ok((y_new, error));
            }

            dt = new_dt;
            if dt < self.controller.min_dt {
                return Err(StiffRKError::StepSizeUnderflow);
            }
            attempts += 1;
        }

        Err(StiffRKError::StepSizeUnderflow)
    }

    fn approximate_jacobian<F>(&self, y: &Vector3<f64>, dt: f64, f: &F) -> Matrix3<f64>
    where
        F: Fn(f64, &Vector3<f64>) -> Vector3<f64>,
    {
        let eps = 1e-8;
        let mut j = Matrix3::zeros();
        let fy = f(0.0, y);

        for i in 0..3 {
            let mut yp = y.clone();
            yp[i] += eps;
            let fp = f(0.0, &yp);
            for row in 0..3 {
                j[(row, i)] = (fp[row] - fy[row]) / eps;
            }
        }

        // Scale by dt for implicit system
        Matrix3::identity() - dt * j * 0.5
    }

    fn solve_coupled(
        &self,
        j: &Matrix3<f64>,
        g1: &Vector3<f64>,
        g2: &Vector3<f64>,
        _dt: f64,
    ) -> Option<(Vector3<f64>, Vector3<f64>)> {
        let lu = LU::new(j.clone())?;
        let dk1 = lu.solve(-g1)?;
        let dk2 = lu.solve(-g2)?;
        Some((dk1, dk2))
    }
}

/// High-level adaptive integrator using Radau IIA
pub struct AdaptiveIntegrator {
    method: RadauIIA3,
}

impl AdaptiveIntegrator {
    pub fn new(abs_tol: f64, rel_tol: f64) -> Self {
        Self {
            method: RadauIIA3::new(abs_tol, rel_tol, 50),
        }
    }

    pub fn integrate<F>(
        &self,
        y0: &Vector3<f64>,
        t_span: (f64, f64),
        f: &F,
    ) -> Result<Vec<(f64, Vector3<f64>)>, StiffRKError>
    where
        F: Fn(f64, &Vector3<f64>) -> Vector3<f64>,
    {
        let (t0, tf) = t_span;
        let mut results = Vec::new();
        let mut t = t0;
        let mut y = y0.clone();
        
        results.push((t, y.clone()));

        let output_dt = (tf - t0) / 100.0; // 100 output points
        let mut next_output = t0 + output_dt;

        while t < tf {
            let remaining = tf - t;
            let dt = remaining.min(0.1);

            match self.method.step(t, &y, dt, f) {
                Ok((y_new, _error)) => {
                    y = y_new;
                    t += dt;

                    // Output at regular intervals
                    while t >= next_output && next_output <= tf {
                        results.push((next_output, y.clone()));
                        next_output += output_dt;
                    }
                }
                Err(e) => return Err(e),
            }
        }

        // Ensure final point
        if !results.last().map_or(false, |(t_, _)| (*t_ - tf).abs() < 1e-10) {
            results.push((tf, y));
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_decay() {
        // dy/dt = -y, solution: y = exp(-t)
        let f = |_t: f64, y: &Vector3<f64>| -y.clone();
        let integrator = AdaptiveIntegrator::new(1e-10, 1e-10);
        let y0 = Vector3::new(1.0, 0.0, 0.0);
        
        let results = integrator.integrate(&y0, (0.0, 1.0), &f).unwrap();
        
        let final_y = results.last().unwrap().1[0];
        let expected = (-1.0_f64).exp();
        
        assert!((final_y - expected).abs() < 1e-6);
    }

    #[test]
    fn test_stiff_system() {
        // Stiff system: dy1/dt = -100*y1 + y2, dy2/dt = -y2
        let f = |_t: f64, y: &Vector3<f64>| {
            Vector3::new(-100.0 * y[0] + y[1], -y[1], 0.0)
        };
        let integrator = AdaptiveIntegrator::new(1e-8, 1e-8);
        let y0 = Vector3::new(1.0, 1.0, 0.0);
        
        let results = integrator.integrate(&y0, (0.0, 1.0), &f);
        
        // Should not fail due to stiffness
        assert!(results.is_ok());
    }
}
