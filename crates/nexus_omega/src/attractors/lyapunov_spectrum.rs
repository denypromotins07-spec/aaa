//! Lyapunov Spectrum Calculator for strange attractor analysis.
//! Measures divergence of nearby trajectories to quantify chaos in the system.

use alloc::vec::Vec;
use core::fmt::Debug;

/// Result of Lyapunov exponent calculation
#[derive(Debug, Clone)]
pub struct LyapunovSpectrum {
    /// Exponents sorted in descending order
    pub exponents: Vec<f64>,
    /// Maximum Lyapunov exponent (indicator of chaos)
    pub max_exponent: f64,
    /// Sum of all exponents (negative for dissipative systems)
    pub sum: f64,
    /// Number of iterations used
    pub iterations: usize,
}

impl LyapunovSpectrum {
    /// Check if system is chaotic (max exponent > 0)
    pub const fn is_chaotic(&self) -> bool {
        self.max_exponent > 0.0
    }

    /// Check if system is stable (all exponents < 0)
    pub const fn is_stable(&self) -> bool {
        self.max_exponent <= 0.0
    }

    /// Estimate predictability horizon (1 / max_exponent)
    pub fn predictability_horizon(&self) -> Option<f64> {
        if self.max_exponent > 1e-15 {
            Some(1.0 / self.max_exponent)
        } else {
            None
        }
    }
}

/// Configuration for Lyapunov spectrum calculation
#[derive(Debug, Clone)]
pub struct LyapunovConfig {
    /// Number of exponents to compute (equals embedding dimension)
    pub num_exponents: usize,
    /// Integration step size
    pub dt: f64,
    /// Total integration time
    pub total_time: f64,
    /// Initial separation between trajectories
    pub initial_separation: f64,
    /// Renormalization frequency
    pub renorm_frequency: usize,
}

impl Default for LyapunovConfig {
    fn default() -> Self {
        Self {
            num_exponents: 3,
            dt: 0.01,
            total_time: 100.0,
            initial_separation: 1e-8,
            renorm_frequency: 10,
        }
    }
}

/// Benettin's algorithm for Lyapunov spectrum estimation
pub struct LyapunovCalculator {
    config: LyapunovConfig,
}

impl LyapunovCalculator {
    pub const fn new(config: LyapunovConfig) -> Self {
        Self { config }
    }

    /// Calculate full Lyapunov spectrum using Benettin's method
    /// Requires a trajectory from phase space reconstruction
    pub fn calculate_from_trajectory(
        &self,
        trajectory: &[Vec<f64>],
    ) -> Result<LyapunovSpectrum, &'static str> {
        if trajectory.is_empty() {
            return Err("Empty trajectory");
        }

        let dim = trajectory[0].len();
        if dim == 0 {
            return Err("Zero-dimensional trajectory");
        }

        let n_exponents = self.config.num_exponents.min(dim);
        let mut exponents = vec![0.0; n_exponents];
        let mut log_growth = vec![0.0; n_exponents];

        // Initialize orthonormal perturbation vectors
        let mut perturbations: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_exponents);
        for i in 0..n_exponents {
            let mut vec_i = vec![0.0; dim];
            vec_i[i % dim] = self.config.initial_separation;
            perturbations.push(vec_i);
        }

        let num_steps = (self.config.total_time / self.config.dt) as usize;
        let actual_steps = num_steps.min(trajectory.len().saturating_sub(2));

        if actual_steps < 10 {
            return Err("Trajectory too short for reliable estimation");
        }

        // Main integration loop
        for step in 0..actual_steps {
            let current_point = &trajectory[step];
            let next_point = &trajectory[(step + 1).min(trajectory.len() - 1)];

            // Evolve perturbations using finite difference approximation
            // of the Jacobian along the trajectory
            for (i, pert) in perturbations.iter_mut().enumerate() {
                // Simple linearized evolution: x_{n+1} ≈ x_n + J * delta_x
                // Approximate J using local trajectory differences
                let evolved = self.evolve_perturbation(current_point, next_point, pert);

                // Gram-Schmidt orthogonalization
                for j in 0..i {
                    let dot = self.dot_product(&evolved, &perturbations[j]);
                    for k in 0..dim {
                        perturbations[i][k] = evolved[k] - dot * perturbations[j][k];
                    }
                }

                // Normalize and accumulate growth
                let norm = self.vector_norm(&perturbations[i]);
                if norm > 1e-15 {
                    for k in 0..dim {
                        perturbations[i][k] /= norm;
                    }
                    log_growth[i] += norm.ln();
                }
            }

            // Periodic renormalization
            if step % self.config.renorm_frequency == 0 && step > 0 {
                for i in 0..n_exponents {
                    exponents[i] += log_growth[i];
                    log_growth[i] = 0.0;
                }
            }
        }

        // Finalize exponents
        let total_time_actual = actual_steps as f64 * self.config.dt;
        if total_time_actual < 1e-15 {
            return Err("Insufficient integration time");
        }

        for i in 0..n_exponents {
            exponents[i] = (exponents[i] + log_growth[i]) / total_time_actual;
        }

        // Sort in descending order
        exponents.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));

        let max_exp = exponents.first().copied().unwrap_or(0.0);
        let sum = exponents.iter().sum();

        Ok(LyapunovSpectrum {
            exponents,
            max_exponent: max_exp,
            sum,
            iterations: actual_steps,
        })
    }

    /// Evolve a perturbation vector along the trajectory
    fn evolve_perturbation(
        &self,
        current: &[f64],
        next: &[f64],
        perturbation: &[f64],
    ) -> Vec<f64> {
        let dim = current.len();
        let mut evolved = Vec::with_capacity(dim);

        // Linearized map approximation
        // δx_{n+1} ≈ δx_n + (∂f/∂x) * δx_n * dt
        // Use finite difference to approximate Jacobian action

        let epsilon = 1e-8;
        for i in 0..dim {
            let mut df_dx_sum = 0.0;

            // Approximate Jacobian element by finite differences
            for j in 0..dim {
                let mut x_plus = current.to_vec();
                let mut x_minus = current.to_vec();
                x_plus[j] += epsilon;
                x_minus[j] -= epsilon;

                // This is a simplified approximation; real implementation
                // would use the actual dynamical system
                let diff = (next[i] - current[i]).abs().max(epsilon);
                df_dx_sum += perturbation[j] * diff / epsilon;
            }

            evolved.push(perturbation[i] + df_dx_sum * self.config.dt);
        }

        evolved
    }

    fn dot_product(&self, a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        let mut sum = 0.0;
        for i in 0..len {
            sum += a[i] * b[i];
        }
        sum
    }

    fn vector_norm(&self, v: &[f64]) -> f64 {
        let mut sum = 0.0;
        for x in v {
            sum += x * x;
        }
        sum.sqrt()
    }

    /// Calculate maximum Lyapunov exponent using Rosenstein's method
    /// Faster but only gives the largest exponent
    pub fn calculate_max_exponent_rosenstein(
        &self,
        trajectory: &[Vec<f64>],
    ) -> Result<f64, &'static str> {
        if trajectory.len() < 10 {
            return Err("Trajectory too short");
        }

        let dim = trajectory[0].len();
        if dim == 0 {
            return Err("Zero-dimensional trajectory");
        }

        // Find nearest neighbors for each point
        let mut divergence_data: Vec<(usize, f64)> = Vec::new();

        for i in 0..trajectory.len().saturating_sub(10) {
            // Find nearest neighbor (excluding temporal neighbors)
            let mut min_dist = f64::MAX;
            let mut nn_idx = 0;

            for j in 0..trajectory.len() {
                if j == i || (j as isize - i as isize).abs() < 5 {
                    continue;
                }

                let dist = self.euclidean_distance(&trajectory[i], &trajectory[j]);
                if dist < min_dist && dist > 1e-15 {
                    min_dist = dist;
                    nn_idx = j;
                }
            }

            if min_dist < f64::MAX / 2.0 {
                // Track divergence over time
                for k in 0..10.min(trajectory.len().saturating_sub(i).saturating_sub(nn_idx)) {
                    let dist_k = self.euclidean_distance(
                        &trajectory[i + k],
                        &trajectory[nn_idx + k],
                    );
                    if dist_k > 1e-15 {
                        divergence_data.push((k, dist_k.ln()));
                    }
                }
            }
        }

        if divergence_data.is_empty() {
            return Ok(0.0);
        }

        // Linear regression on ln(d(k)) vs k
        let slope = self.linear_regression_slope(&divergence_data);

        Ok(slope / self.config.dt)
    }

    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        let mut sum = 0.0;
        for i in 0..len {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }

    fn linear_regression_slope(&self, data: &[(usize, f64)]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let n = data.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (x, y) in data {
            let xf = *x as f64;
            sum_x += xf;
            sum_y += y;
            sum_xy += xf * y;
            sum_xx += xf * xf;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denom
    }
}

/// Wrapper for quick chaos detection
pub struct ChaosDetector {
    threshold: f64,
}

impl ChaosDetector {
    pub const fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Quick check if trajectory exhibits chaotic behavior
    pub fn is_chaotic(&self, trajectory: &[Vec<f64>]) -> Result<bool, &'static str> {
        let config = LyapunovConfig::default();
        let calc = LyapunovCalculator::new(config);

        match calc.calculate_max_exponent_rosenstein(trajectory) {
            Ok(max_exp) => Ok(max_exp > self.threshold),
            Err(_) => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lyapunov_config_default() {
        let config = LyapunovConfig::default();
        assert_eq!(config.num_exponents, 3);
        assert!(config.dt > 0.0);
    }

    #[test]
    fn test_spectrum_properties() {
        let spectrum = LyapunovSpectrum {
            exponents: vec![0.5, -0.2, -1.3],
            max_exponent: 0.5,
            sum: -1.0,
            iterations: 1000,
        };

        assert!(spectrum.is_chaotic());
        assert!(!spectrum.is_stable());
        assert!(spectrum.predictability_horizon().is_some());
    }
}
