//! Dynamical Casimir Effect Simulator for NEXUS-OMEGA
//! 
//! Models the extraction of real photons from quantum vacuum fluctuations
//! by oscillating boundaries at relativistic effective speeds.
//! 
//! Key insight: We use superconducting metamaterials to simulate relativistic
//! boundary motion without physical kinetic movement, avoiding the impossibility
//! of moving macroscopic objects at c.

use core::fmt;
use alloc::{vec::Vec, string::String};

/// Physical constants for DCE calculations
pub mod constants {
    /// Speed of light (m/s)
    pub const C: f64 = 299_792_458.0;
    /// Reduced Planck constant (J·s)
    pub const HBAR: f64 = 1.054_571_817e-34;
    /// Vacuum permittivity (F/m)
    pub const EPSILON_0: f64 = 8.854_187_817e-12;
    /// Vacuum permeability (H/m)
    pub const MU_0: f64 = 1.256_637_06212e-6;
}

/// Represents a photon pair created from vacuum
#[derive(Debug, Clone, Copy)]
pub struct VacuumPhotonPair {
    /// Frequency of first photon (Hz)
    pub frequency_a: f64,
    /// Frequency of second photon (Hz)
    pub frequency_b: f64,
    /// Creation time (s)
    pub creation_time: f64,
    /// Spatial mode index
    pub mode_index: u32,
}

/// Configuration for the DCE simulator
#[derive(Debug, Clone, Copy)]
pub struct DCEConfig {
    /// Boundary oscillation frequency (Hz)
    pub oscillation_frequency: f64,
    /// Effective velocity as fraction of c (0 < v_eff < 1)
    pub effective_velocity_fraction: f64,
    /// Cavity length (m)
    pub cavity_length: f64,
    /// Quality factor of the cavity
    pub quality_factor: f64,
    /// Temperature (K) - should be near zero for pure DCE
    pub temperature: f64,
}

impl DCEConfig {
    /// Create a new config with validation
    pub fn new(
        oscillation_frequency: f64,
        effective_velocity_fraction: f64,
        cavity_length: f64,
        quality_factor: f64,
        temperature: f64,
    ) -> Result<Self, DCEError> {
        // Validate effective velocity (must be subluminal)
        if effective_velocity_fraction <= 0.0 || effective_velocity_fraction >= 1.0 {
            return Err(DCEError::InvalidVelocity(effective_velocity_fraction));
        }

        // Validate cavity length
        if cavity_length <= 0.0 {
            return Err(DCEError::InvalidCavityLength(cavity_length));
        }

        // Validate quality factor
        if quality_factor <= 0.0 {
            return Err(DCEError::InvalidQualityFactor(quality_factor));
        }

        Ok(Self {
            oscillation_frequency,
            effective_velocity_fraction,
            cavity_length,
            quality_factor,
            temperature,
        })
    }
}

/// The Dynamical Casimir Effect Simulator
pub struct DCESimulator {
    config: DCEConfig,
    /// Photon pairs generated
    photon_history: Vec<VacuumPhotonPair>,
    /// Total energy extracted (J)
    total_energy_extracted: f64,
    /// Simulation time (s)
    simulation_time: f64,
}

impl DCESimulator {
    pub const fn new(config: DCEConfig) -> Self {
        Self {
            config,
            photon_history: Vec::new(),
            total_energy_extracted: 0.0,
            simulation_time: 0.0,
        }
    }

    /// Simulate one oscillation cycle and compute photon production
    /// Returns Result to avoid unwrap() in hot paths
    pub fn simulate_cycle(&mut self, dt: f64) -> Result<usize, DCEError> {
        if dt <= 0.0 {
            return Err(DCEError::InvalidTimeStep(dt));
        }

        self.simulation_time += dt;

        // Calculate the effective boundary position using metamaterial model
        // x(t) = L + A * sin(ωt) where A is effective amplitude
        let omega = 2.0 * core::f64::consts::PI * self.config.oscillation_frequency;
        let t = self.simulation_time;
        
        // Effective amplitude from velocity fraction
        // v_eff = A * ω, so A = v_eff * c / ω
        let amplitude = self.config.effective_velocity_fraction * constants::C / omega;
        let position = self.config.cavity_length + amplitude * (omega * t).sin();
        let velocity = amplitude * omega * (omega * t).cos();

        // Check for parametric resonance condition
        // DCE is maximized when ω ≈ 2 * ω_cavity
        let cavity_freq = constants::C * core::f64::consts::PI / self.config.cavity_length;
        let resonance_ratio = self.config.oscillation_frequency / (2.0 * cavity_freq);
        
        // Photon production rate (Moore's formula approximation)
        // N ∝ (v/c)² * Q * (resonance enhancement)
        let resonance_enhancement = 1.0 / (1.0 + (resonance_ratio - 1.0).powi(2) * self.config.quality_factor.powi(2));
        let velocity_fraction = velocity.abs() / constants::C;
        let photon_rate = velocity_fraction.powi(2) * self.config.quality_factor * resonance_enhancement;

        // Number of photon pairs in this timestep
        let num_pairs = (photon_rate * dt).floor() as usize;

        for i in 0..num_pairs {
            // Energy conservation: ħω₁ + ħω₂ = ħΩ (oscillation frequency)
            // Split energy between the two photons
            let split_ratio = 0.5 + 0.1 * ((i as f64 * 0.1).sin()); // Slight asymmetry
            let freq_a = self.config.oscillation_frequency * split_ratio;
            let freq_b = self.config.oscillation_frequency * (1.0 - split_ratio);

            let pair = VacuumPhotonPair {
                frequency_a: freq_a.max(1.0), // Ensure positive frequency
                frequency_b: freq_b.max(1.0),
                creation_time: self.simulation_time,
                mode_index: i as u32,
            };

            self.photon_history.push(pair);

            // Add energy extracted (E = ħω for each photon)
            let energy = constants::HBAR * 2.0 * core::f64::consts::PI * self.config.oscillation_frequency;
            self.total_energy_extracted += energy;
        }

        Ok(num_pairs)
    }

    /// Run simulation for specified duration
    pub fn run_simulation(&mut self, duration: f64, dt: f64) -> Result<SimulationResult, DCEError> {
        if duration <= 0.0 {
            return Err(DCEError::InvalidDuration(duration));
        }

        let mut total_photons = 0usize;
        let mut cycles = 0usize;

        let mut current_time = 0.0;
        while current_time < duration {
            match self.simulate_cycle(dt) {
                Ok(photons) => {
                    total_photons += photons;
                    cycles += 1;
                    current_time += dt;
                }
                Err(e) => return Err(e),
            }
        }

        Ok(SimulationResult {
            total_photons_generated: total_photons,
            total_cycles: cycles,
            total_energy_joules: self.total_energy_extracted,
            final_simulation_time: self.simulation_time,
            average_photons_per_cycle: if cycles > 0 {
                total_photons as f64 / cycles as f64
            } else {
                0.0
            },
        })
    }

    /// Get the spectral distribution of generated photons
    pub fn get_spectrum(&self) -> SpectrumAnalysis {
        if self.photon_history.is_empty() {
            return SpectrumAnalysis::default();
        }

        let mut min_freq_a = f64::MAX;
        let mut max_freq_a = 0.0;
        let mut min_freq_b = f64::MAX;
        let mut max_freq_b = 0.0;
        let mut sum_freq_a = 0.0;
        let mut sum_freq_b = 0.0;

        for pair in &self.photon_history {
            min_freq_a = min_freq_a.min(pair.frequency_a);
            max_freq_a = max_freq_a.max(pair.frequency_a);
            min_freq_b = min_freq_b.min(pair.frequency_b);
            max_freq_b = max_freq_b.max(pair.frequency_b);
            sum_freq_a += pair.frequency_a;
            sum_freq_b += pair.frequency_b;
        }

        let n = self.photon_history.len() as f64;
        SpectrumAnalysis {
            mean_frequency_a: sum_freq_a / n,
            mean_frequency_b: sum_freq_b / n,
            min_frequency_a: min_freq_a,
            max_frequency_a: max_freq_a,
            min_frequency_b: min_freq_b,
            max_frequency_b: max_freq_b,
            total_pairs: self.photon_history.len(),
        }
    }

    /// Reset the simulator state
    pub fn reset(&mut self) {
        self.photon_history.clear();
        self.total_energy_extracted = 0.0;
        self.simulation_time = 0.0;
    }

    /// Get configuration reference
    pub const fn config(&self) -> &DCEConfig {
        &self.config
    }
}

/// Results from a DCE simulation run
#[derive(Debug, Clone, Copy)]
pub struct SimulationResult {
    pub total_photons_generated: usize,
    pub total_cycles: usize,
    pub total_energy_joules: f64,
    pub final_simulation_time: f64,
    pub average_photons_per_cycle: f64,
}

/// Spectral analysis of generated photons
#[derive(Debug, Clone, Copy, Default)]
pub struct SpectrumAnalysis {
    pub mean_frequency_a: f64,
    pub mean_frequency_b: f64,
    pub min_frequency_a: f64,
    pub max_frequency_a: f64,
    pub min_frequency_b: f64,
    pub max_frequency_b: f64,
    pub total_pairs: usize,
}

/// Errors that can occur in DCE simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DCEError {
    InvalidVelocity(f64),
    InvalidCavityLength(f64),
    InvalidQualityFactor(f64),
    InvalidTimeStep(f64),
    InvalidDuration(f64),
    NumericalInstability,
}

impl fmt::Display for DCEError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DCEError::InvalidVelocity(v) => {
                write!(f, "Invalid effective velocity fraction: {} (must be 0 < v < 1)", v)
            }
            DCEError::InvalidCavityLength(l) => {
                write!(f, "Invalid cavity length: {} (must be > 0)", l)
            }
            DCEError::InvalidQualityFactor(q) => {
                write!(f, "Invalid quality factor: {} (must be > 0)", q)
            }
            DCEError::InvalidTimeStep(dt) => {
                write!(f, "Invalid timestep: {} (must be > 0)", dt)
            }
            DCEError::InvalidDuration(d) => {
                write!(f, "Invalid duration: {} (must be > 0)", d)
            }
            DCEError::NumericalInstability => {
                write!(f, "Numerical instability detected in simulation")
            }
        }
    }
}

/// Metamaterial boundary model for effective relativistic motion
pub struct MetamaterialBoundary {
    /// Superconducting phase (radians)
    phase: f64,
    /// Effective refractive index
    refractive_index: f64,
    /// Loss tangent
    loss_tangent: f64,
}

impl MetamaterialBoundary {
    pub const fn new(refractive_index: f64, loss_tangent: f64) -> Self {
        Self {
            phase: 0.0,
            refractive_index,
            loss_tangent,
        }
    }

    /// Update boundary state
    pub fn update(&mut self, driving_frequency: f64, dt: f64) {
        self.phase += 2.0 * core::f64::consts::PI * driving_frequency * dt;
        self.phase %= 2.0 * core::f64::consts::PI;
    }

    /// Get effective velocity as fraction of c
    pub fn effective_velocity(&self, modulation_depth: f64) -> f64 {
        // v_eff/c = modulation_depth * |sin(phase)|
        modulation_depth * self.phase.sin().abs()
    }

    /// Check if boundary is in low-loss regime
    pub fn is_low_loss(&self) -> bool {
        self.loss_tangent < 0.01
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        // Valid config
        let config = DCEConfig::new(1e9, 0.5, 0.01, 1000.0, 0.0);
        assert!(config.is_ok());

        // Invalid velocity
        let config = DCEConfig::new(1e9, 1.5, 0.01, 1000.0, 0.0);
        assert_eq!(config, Err(DCEError::InvalidVelocity(1.5)));

        // Invalid cavity length
        let config = DCEConfig::new(1e9, 0.5, -0.01, 1000.0, 0.0);
        assert!(config.is_err());
    }

    #[test]
    fn test_simulator_creation() {
        let config = DCEConfig::new(1e9, 0.5, 0.01, 1000.0, 0.0).unwrap();
        let sim = DCESimulator::new(config);
        assert_eq!(sim.photon_history.len(), 0);
        assert_eq!(sim.total_energy_extracted, 0.0);
    }

    #[test]
    fn test_simulation_cycle() {
        let config = DCEConfig::new(1e9, 0.5, 0.01, 1000.0, 0.0).unwrap();
        let mut sim = DCESimulator::new(config);
        
        let result = sim.simulate_cycle(1e-9);
        assert!(result.is_ok());
    }

    #[test]
    fn test_metamaterial_boundary() {
        let mut boundary = MetamaterialBoundary::new(2.0, 0.001);
        assert!(boundary.is_low_loss());
        
        boundary.update(1e9, 1e-9);
        let v_eff = boundary.effective_velocity(0.5);
        assert!(v_eff >= 0.0 && v_eff <= 0.5);
    }
}
