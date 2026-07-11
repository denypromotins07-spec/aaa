//! Imaginary Time Propagator for financial wave functions.
//! Evolves states through imaginary time using path integral methods.

use alloc::vec::Vec;
use core::fmt::Debug;

use super::wick_rotated_kolmogorov::{Complex, WickKolmogorovConfig, WickEvolutionResult};

/// Configuration for imaginary time propagation
#[derive(Debug, Clone)]
pub struct ImaginaryTimeConfig {
    /// Number of time slices for path integral
    pub num_time_slices: usize,
    /// Total imaginary time
    pub total_tau: f64,
    /// Spatial discretization points
    pub num_spatial_points: usize,
    /// Mass parameter (inverse volatility)
    pub mass: f64,
}

impl Default for ImaginaryTimeConfig {
    fn default() -> Self {
        Self {
            num_time_slices: 100,
            total_tau: 1.0,
            num_spatial_points: 128,
            mass: 1.0,
        }
    }
}

/// Result of imaginary time propagation
#[derive(Debug, Clone)]
pub struct PropagationResult {
    /// Propagated wave function
    pub wave_function: Vec<Complex>,
    /// Ground state energy estimate
    pub ground_state_energy: f64,
    /// Convergence indicator
    pub converged: bool,
    /// Number of iterations performed
    pub iterations: usize,
}

/// Imaginary time propagator using Feynman path integrals
pub struct ImaginaryTimePropagator {
    config: ImaginaryTimeConfig,
}

impl ImaginaryTimePropagator {
    pub const fn new(config: ImaginaryTimeConfig) -> Self {
        Self { config }
    }

    /// Propagate wave function in imaginary time
    /// 
    /// ψ(τ) = exp(-Ĥτ/ℏ) ψ(0)
    /// 
    /// In the limit τ → ∞, projects onto ground state
    pub fn propagate(
        &self,
        initial_state: &[f64],
        potential: &[f64],
    ) -> Result<PropagationResult, &'static str> {
        let n = self.config.num_spatial_points;
        
        if initial_state.len() != n {
            return Err("Initial state size mismatch");
        }
        
        if potential.len() != n {
            return Err("Potential size mismatch");
        }

        // Initialize wave function
        let mut psi: Vec<Complex> = Vec::with_capacity(n);
        for &val in initial_state {
            psi.push(Complex::real(val));
        }

        // Normalize initial state
        self.normalize(&mut psi);

        let dtau = self.config.total_tau / self.config.num_time_slices as f64;
        let mass = self.config.mass;
        
        // Kinetic energy prefactor
        let kinetic_prefactor = dtau / (2.0 * mass);

        let mut iterations = 0;
        let max_iterations = self.config.num_time_slices * 10;
        let mut prev_energy = f64::MAX;
        let mut converged = false;

        // Imaginary time evolution loop
        for slice in 0..self.config.num_time_slices {
            // Apply kinetic energy (Fourier space)
            psi = self.apply_kinetic(&psi, kinetic_prefactor);

            // Apply potential energy (real space)
            for i in 0..n {
                let factor = (-dtau * potential[i]).exp();
                psi[i] = psi[i].scale(factor);
            }

            // Normalize after each step
            self.normalize(&mut psi);

            iterations += 1;

            // Check convergence via energy estimate
            let current_energy = self.estimate_energy(&psi, potential, kinetic_prefactor);
            
            if iterations > 10 && (prev_energy - current_energy).abs() < 1e-8 {
                converged = true;
                break;
            }

            prev_energy = current_energy;

            if iterations >= max_iterations {
                break;
            }
        }

        // Estimate ground state energy
        let ground_energy = self.estimate_energy(&psi, potential, kinetic_prefactor);

        Ok(PropagationResult {
            wave_function: psi,
            ground_state_energy: ground_energy,
            converged,
            iterations,
        })
    }

    /// Apply kinetic energy operator in Fourier space
    fn apply_kinetic(&self, psi: &[Complex], prefactor: f64) -> Vec<Complex> {
        let n = psi.len();
        
        // Simple finite difference approximation for Laplacian
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let mut laplacian = Complex::zero();
            
            if i > 0 && i < n - 1 {
                // Second derivative: ψ'' ≈ (ψ_{i+1} - 2ψ_i + ψ_{i-1}) / dx²
                let second_diff = psi[i + 1].add(&psi[i - 1]).sub(&psi[i].scale(2.0));
                laplacian = second_diff.scale(prefactor);
            } else {
                // Boundary handling (Dirichlet zero)
                laplacian = psi[i].scale(-prefactor);
            }

            // exp(-T) ≈ 1 - T for small T
            result.push(psi[i].add(&laplacian.scale(-1.0)));
        }

        result
    }

    /// Normalize wave function
    fn normalize(&self, psi: &mut [Complex]) {
        let mut norm_sq = 0.0;
        for p in psi.iter() {
            norm_sq += p.norm_sq();
        }

        if norm_sq > 1e-15 {
            let norm_inv = 1.0 / norm_sq.sqrt();
            for p in psi.iter_mut() {
                *p = p.scale(norm_inv);
            }
        }
    }

    /// Estimate energy expectation value
    fn estimate_energy(&self, psi: &[Complex], potential: &[f64], kinetic_prefactor: f64) -> f64 {
        let n = psi.len();
        let mut energy = 0.0;

        for i in 0..n {
            // Potential energy contribution
            energy += psi[i].norm_sq() * potential[i];

            // Kinetic energy contribution
            if i > 0 && i < n - 1 {
                let gradient_sq = (psi[i + 1].re - psi[i - 1].re).powi(2) / 4.0;
                energy += gradient_sq / (2.0 * self.config.mass);
            }
        }

        energy
    }

    /// Project to ground state by long imaginary time evolution
    pub fn project_to_ground_state(
        &self,
        trial_state: &[f64],
        potential: &[f64],
    ) -> Result<(Vec<Complex>, f64), &'static str> {
        let result = self.propagate(trial_state, potential)?;
        
        if !result.converged {
            // Run additional iterations
            let mut extended_psi = result.wave_function.clone();
            self.normalize(&mut extended_psi);
            
            for _ in 0..100 {
                extended_psi = self.apply_kinetic(&extended_psi, 0.01);
                for i in 0..extended_psi.len() {
                    let factor = (-0.01 * potential[i]).exp();
                    extended_psi[i] = extended_psi[i].scale(factor);
                }
                self.normalize(&mut extended_psi);
            }

            let final_energy = self.estimate_energy(&extended_psi, potential, 0.01);
            return Ok((extended_psi, final_energy));
        }

        Ok((result.wave_function, result.ground_state_energy))
    }
}

/// Euclidean path integral calculator
pub struct PathIntegralCalculator {
    num_paths: usize,
}

impl PathIntegralCalculator {
    pub const fn new(num_paths: usize) -> Self {
        Self { num_paths }
    }

    /// Calculate transition amplitude using Monte Carlo path integral
    /// 
    /// K(x_f, τ_f; x_i, τ_i) = ∫ Dx(τ) exp(-S_E[x]/ℏ)
    /// 
    /// where S_E is the Euclidean action
    pub fn calculate_transition_amplitude(
        &self,
        x_initial: f64,
        x_final: f64,
        tau: f64,
        potential_fn: impl Fn(f64) -> f64,
    ) -> Complex {
        use core::f64::consts::PI;

        let mut sum = Complex::zero();
        let dt = tau / 100.0; // Discretize path

        // Simplified: single dominant path approximation
        // Full implementation would sample many paths
        
        // Classical path (straight line for free particle)
        let velocity = (x_final - x_initial) / tau;
        
        // Calculate Euclidean action along classical path
        let mut action = 0.0;
        for i in 0..100 {
            let t = i as f64 * dt;
            let x = x_initial + velocity * t;
            let v = velocity;
            
            // Euclidean Lagrangian: L_E = (1/2)m v² + V(x)
            let kinetic = 0.5 * v * v;
            let potential = potential_fn(x);
            action += (kinetic + potential) * dt;
        }

        // Amplitude ~ exp(-S_E)
        let amplitude = (-action).exp();
        Complex::real(amplitude)
    }

    /// Calculate partition function Z = Tr[exp(-βĤ)]
    pub fn calculate_partition_function(
        &self,
        beta: f64,
        energy_levels: &[f64],
    ) -> f64 {
        let mut z = 0.0;
        for &e in energy_levels {
            z += (-beta * e).exp();
        }
        z
    }
}

/// Thermal state preparation via imaginary time
pub struct ThermalStatePreparator {
    propagator: ImaginaryTimePropagator,
}

impl ThermalStatePreparator {
    pub fn new(config: ImaginaryTimeConfig) -> Self {
        Self {
            propagator: ImaginaryTimePropagator::new(config),
        }
    }

    /// Prepare thermal state at inverse temperature β
    pub fn prepare_thermal_state(
        &self,
        beta: f64,
        hamiltonian_diag: &[f64],
    ) -> Result<Vec<f64>, &'static str> {
        let n = hamiltonian_diag.len();
        let mut weights = Vec::with_capacity(n);

        // Boltzmann weights
        let mut partition = 0.0;
        for &e in hamiltonian_diag {
            let w = (-beta * e).exp();
            weights.push(w);
            partition += w;
        }

        // Normalize
        if partition > 1e-15 {
            for w in weights.iter_mut() {
                *w /= partition;
            }
        }

        Ok(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_propagator_initialization() {
        let config = ImaginaryTimeConfig::default();
        let propagator = ImaginaryTimePropagator::new(config);
        
        let initial: Vec<f64> = (0..128).map(|i| (i as f64 / 128.0 * PI).sin()).collect();
        let potential: Vec<f64> = vec![0.5; 128]; // Harmonic oscillator
        
        let result = propagator.propagate(&initial, &potential);
        assert!(result.is_ok());
    }

    #[test]
    fn test_path_integral_free_particle() {
        let calc = PathIntegralCalculator::new(1000);
        
        let amplitude = calc.calculate_transition_amplitude(
            0.0,
            1.0,
            1.0,
            |_| 0.0, // Free particle
        );
        
        assert!(amplitude.re > 0.0);
        assert!(amplitude.im.abs() < 1e-10);
    }

    #[test]
    fn test_partition_function() {
        let calc = PathIntegralCalculator::new(100);
        
        // Two-level system
        let energies = vec![0.0, 1.0];
        let beta = 1.0;
        
        let z = calc.calculate_partition_function(beta, &energies);
        
        // Z = 1 + exp(-β)
        let expected = 1.0 + (-1.0).exp();
        assert!((z - expected).abs() < 1e-10);
    }
}

// Import PI constant for tests
#[cfg(test)]
use core::f64::consts::PI;
