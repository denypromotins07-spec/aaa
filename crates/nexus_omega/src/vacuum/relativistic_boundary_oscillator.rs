//! Relativistic Boundary Oscillator for NEXUS-OMEGA
//! 
//! Implements the metamaterial-based effective relativistic boundary motion
//! required for Dynamical Casimir Effect photon generation.
//! 
//! Key innovation: Instead of physically moving boundaries at impossible speeds,
//! we use superconducting quantum interference devices (SQUIDs) to modulate
//! the effective electrical length of a transmission line, simulating relativistic motion.

use core::fmt;
use alloc::{vec::Vec, string::String};

/// Physical constants
pub mod consts {
    /// Speed of light (m/s)
    pub const C: f64 = 299_792_458.0;
    /// Flux quantum (Wb)
    pub const PHI_0: f64 = 2.067_833_848e-15;
}

/// State of the boundary oscillator
#[derive(Debug, Clone, Copy)]
pub struct BoundaryState {
    /// Current position (effective electrical length in meters)
    pub position: f64,
    /// Current velocity (effective, as fraction of c)
    pub velocity_fraction: f64,
    /// Current acceleration (effective, m/s²)
    pub acceleration: f64,
    /// Phase of oscillation (radians)
    pub phase: f64,
    /// Time since start (s)
    pub time: f64,
}

impl Default for BoundaryState {
    fn default() -> Self {
        Self {
            position: 0.0,
            velocity_fraction: 0.0,
            acceleration: 0.0,
            phase: 0.0,
            time: 0.0,
        }
    }
}

/// Configuration for the relativistic boundary oscillator
#[derive(Debug, Clone, Copy)]
pub struct OscillatorConfig {
    /// Rest length of the cavity (m)
    pub rest_length: f64,
    /// Maximum modulation amplitude (fraction of rest length)
    pub modulation_depth: f64,
    /// Driving frequency (Hz)
    pub driving_frequency: f64,
    /// Quality factor of the resonator
    pub quality_factor: f64,
    /// SQUID inductance (H)
    pub squid_inductance: f64,
    /// Critical current of SQUID (A)
    pub critical_current: f64,
}

impl OscillatorConfig {
    pub fn new(
        rest_length: f64,
        modulation_depth: f64,
        driving_frequency: f64,
        quality_factor: f64,
        squid_inductance: f64,
        critical_current: f64,
    ) -> Result<Self, OscillatorError> {
        if rest_length <= 0.0 {
            return Err(OscillatorError::InvalidRestLength(rest_length));
        }
        if modulation_depth <= 0.0 || modulation_depth > 1.0 {
            return Err(OscillatorError::InvalidModulationDepth(modulation_depth));
        }
        if driving_frequency <= 0.0 {
            return Err(OscillatorError::InvalidFrequency(driving_frequency));
        }
        if quality_factor <= 0.0 {
            return Err(OscillatorError::InvalidQualityFactor(quality_factor));
        }

        Ok(Self {
            rest_length,
            modulation_depth,
            driving_frequency,
            quality_factor,
            squid_inductance,
            critical_current,
        })
    }
}

/// The Relativistic Boundary Oscillator
pub struct RelativisticOscillator {
    config: OscillatorConfig,
    state: BoundaryState,
    /// History of states for analysis
    state_history: Vec<BoundaryState>,
    /// Maximum velocity achieved (as fraction of c)
    max_velocity_fraction: f64,
}

impl RelativisticOscillator {
    pub fn new(config: OscillatorConfig) -> Self {
        Self {
            config,
            state: BoundaryState {
                position: config.rest_length,
                ..Default::default()
            },
            state_history: Vec::new(),
            max_velocity_fraction: 0.0,
        }
    }

    /// Update the oscillator state by one timestep
    /// Returns Result to avoid unwrap() in hot paths
    pub fn step(&mut self, dt: f64) -> Result<(), OscillatorError> {
        if dt <= 0.0 {
            return Err(OscillatorError::InvalidTimeStep(dt));
        }

        // Update time
        self.state.time += dt;

        // Calculate driven oscillation phase
        let omega = 2.0 * core::f64::consts::PI * self.config.driving_frequency;
        self.state.phase = omega * self.state.time;

        // Effective position from SQUID modulation
        // L_eff(t) = L_0 * (1 + δ * sin(ωt))
        // This simulates boundary motion without physical movement
        let delta = self.config.modulation_depth;
        let effective_length = self.config.rest_length * (1.0 + delta * self.state.phase.sin());
        
        // Effective velocity: v_eff = dL/dt = L_0 * δ * ω * cos(ωt)
        let effective_velocity = self.config.rest_length * delta * omega * self.state.phase.cos();
        
        // Normalize to fraction of c
        let velocity_fraction = effective_velocity / consts::C;
        
        // Check for numerical stability (should never exceed ~0.1 in practice)
        if velocity_fraction.abs() > 1.0 {
            return Err(OscillatorError::SuperluminalVelocity(velocity_fraction));
        }

        // Effective acceleration: a_eff = -L_0 * δ * ω² * sin(ωt)
        let acceleration = -self.config.rest_length * delta * omega.powi(2) * self.state.phase.sin();

        // Apply damping based on quality factor
        let damping_factor = (-core::f64::consts::PI / self.config.quality_factor).exp();
        let damped_velocity = velocity_fraction * damping_factor;

        // Update state
        self.state.position = effective_length;
        self.state.velocity_fraction = damped_velocity;
        self.state.acceleration = acceleration;

        // Track maximum velocity
        self.max_velocity_fraction = self.max_velocity_fraction.max(damped_velocity.abs());

        // Record state history (limit size)
        self.state_history.push(self.state);
        if self.state_history.len() > 10000 {
            self.state_history.remove(0);
        }

        Ok(())
    }

    /// Run simulation for specified duration
    pub fn run(&mut self, duration: f64, dt: f64) -> Result<OscillatorResult, OscillatorError> {
        if duration <= 0.0 {
            return Err(OscillatorError::InvalidDuration(duration));
        }

        let mut steps = 0usize;
        let mut current_time = 0.0;

        while current_time < duration {
            self.step(dt)?;
            steps += 1;
            current_time += dt;
        }

        Ok(OscillatorResult {
            total_steps: steps,
            final_state: self.state,
            max_velocity_fraction: self.max_velocity_fraction,
            total_simulation_time: self.state.time,
        })
    }

    /// Get the required SQUID bias current for target velocity
    pub fn required_bias_current(&self, target_velocity_fraction: f64) -> Result<f64, OscillatorError> {
        if target_velocity_fraction <= 0.0 || target_velocity_fraction >= 1.0 {
            return Err(OscillatorError::InvalidTargetVelocity(target_velocity_fraction));
        }

        // From SQUID physics: Φ = Φ_0 * I_bias / I_c
        // And L_eff ∝ 1/Φ for small modulations
        // So I_bias ≈ (v_eff/c) * I_c / (2π * modulation_depth)
        
        let omega = 2.0 * core::f64::consts::PI * self.config.driving_frequency;
        let required_modulation = target_velocity_fraction * consts::C / 
            (self.config.rest_length * omega);
        
        if required_modulation > self.config.modulation_depth {
            return Err(OscillatorError::ModulationLimitExceeded(required_modulation));
        }

        let bias_current = required_modulation * self.config.critical_current / 
            (2.0 * core::f64::consts::PI * self.config.modulation_depth);

        Ok(bias_current.min(self.config.critical_current))
    }

    /// Check if operating in quantum regime (single photon sensitivity)
    pub fn is_quantum_regime(&self) -> bool {
        // Quantum regime when zero-point fluctuations exceed thermal noise
        // ħω > k_B T
        let omega = 2.0 * core::f64::consts::PI * self.config.driving_frequency;
        let hbar_omega = 1.054e-34 * omega;
        let k_b_t = 1.38e-23 * 0.02; // Assuming 20 mK dilution refrigerator
        
        hbar_omega > k_b_t
    }

    /// Get current state
    pub const fn state(&self) -> &BoundaryState {
        &self.state
    }

    /// Get configuration
    pub const fn config(&self) -> &OscillatorConfig {
        &self.config
    }

    /// Reset oscillator to initial state
    pub fn reset(&mut self) {
        self.state = BoundaryState {
            position: self.config.rest_length,
            ..Default::default()
        };
        self.state_history.clear();
        self.max_velocity_fraction = 0.0;
    }
}

/// Results from an oscillator simulation
#[derive(Debug, Clone, Copy)]
pub struct OscillatorResult {
    pub total_steps: usize,
    pub final_state: BoundaryState,
    pub max_velocity_fraction: f64,
    pub total_simulation_time: f64,
}

/// Errors that can occur in oscillator operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OscillatorError {
    InvalidRestLength(f64),
    InvalidModulationDepth(f64),
    InvalidFrequency(f64),
    InvalidQualityFactor(f64),
    InvalidTimeStep(f64),
    InvalidDuration(f64),
    SuperluminalVelocity(f64),
    InvalidTargetVelocity(f64),
    ModulationLimitExceeded(f64),
    NumericalOverflow,
}

impl fmt::Display for OscillatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OscillatorError::InvalidRestLength(l) => {
                write!(f, "Invalid rest length: {} (must be > 0)", l)
            }
            OscillatorError::InvalidModulationDepth(d) => {
                write!(f, "Invalid modulation depth: {} (must be 0 < d ≤ 1)", d)
            }
            OscillatorError::InvalidFrequency(freq) => {
                write!(f, "Invalid frequency: {} (must be > 0)", freq)
            }
            OscillatorError::InvalidQualityFactor(q) => {
                write!(f, "Invalid quality factor: {} (must be > 0)", q)
            }
            OscillatorError::InvalidTimeStep(dt) => {
                write!(f, "Invalid timestep: {} (must be > 0)", dt)
            }
            OscillatorError::InvalidDuration(d) => {
                write!(f, "Invalid duration: {} (must be > 0)", d)
            }
            OscillatorError::SuperluminalVelocity(v) => {
                write!(f, "Superluminal velocity detected: {}c (numerical error)", v)
            }
            OscillatorError::InvalidTargetVelocity(v) => {
                write!(f, "Invalid target velocity: {} (must be 0 < v < 1)", v)
            }
            OscillatorError::ModulationLimitExceeded(required) => {
                write!(f, "Required modulation {} exceeds maximum", required)
            }
            OscillatorError::NumericalOverflow => {
                write!(f, "Numerical overflow in calculation")
            }
        }
    }
}

/// Transmission line model for the SQUID-terminated resonator
pub struct TransmissionLine {
    /// Characteristic impedance (Ω)
    impedance: f64,
    /// Phase velocity (m/s)
    phase_velocity: f64,
    /// Length (m)
    length: f64,
    /// Number of discrete segments
    segments: usize,
}

impl TransmissionLine {
    pub const fn new(impedance: f64, phase_velocity: f64, length: f64, segments: usize) -> Self {
        Self {
            impedance,
            phase_velocity,
            length,
            segments,
        }
    }

    /// Calculate resonance frequencies
    pub fn resonance_frequencies(&self, mode: usize) -> Option<f64> {
        if mode == 0 {
            return None;
        }
        // f_n = n * v_phase / (2 * L)
        Some((mode as f64) * self.phase_velocity / (2.0 * self.length))
    }

    /// Get fundamental frequency
    pub fn fundamental_frequency(&self) -> f64 {
        self.resonance_frequencies(1).unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = OscillatorConfig::new(0.01, 0.5, 1e9, 1000.0, 1e-9, 1e-6);
        assert!(config.is_ok());

        let config = OscillatorConfig::new(-0.01, 0.5, 1e9, 1000.0, 1e-9, 1e-6);
        assert!(config.is_err());

        let config = OscillatorConfig::new(0.01, 1.5, 1e9, 1000.0, 1e-9, 1e-6);
        assert!(config.is_err());
    }

    #[test]
    fn test_oscillator_step() {
        let config = OscillatorConfig::new(0.01, 0.5, 1e9, 1000.0, 1e-9, 1e-6).unwrap();
        let mut osc = RelativisticOscillator::new(config);

        assert!(osc.step(1e-12).is_ok());
        assert!(osc.state().time > 0.0);
    }

    #[test]
    fn test_max_velocity_tracking() {
        let config = OscillatorConfig::new(0.01, 0.5, 1e9, 1000.0, 1e-9, 1e-6).unwrap();
        let mut osc = RelativisticOscillator::new(config);

        // Run for quarter cycle to reach max velocity
        let quarter_period = 1.0 / (4.0 * config.driving_frequency);
        let result = osc.run(quarter_period, 1e-13).unwrap();

        assert!(result.max_velocity_fraction > 0.0);
        assert!(result.max_velocity_fraction < 1.0);
    }

    #[test]
    fn test_transmission_line_resonance() {
        let line = TransmissionLine::new(50.0, consts::C * 0.5, 0.01, 100);
        let f1 = line.fundamental_frequency();
        assert!(f1 > 0.0);

        let f2 = line.resonance_frequencies(2);
        assert!(f2.unwrap() > f1);
    }

    #[test]
    fn test_quantum_regime_check() {
        let config = OscillatorConfig::new(0.01, 0.5, 10e9, 1000.0, 1e-9, 1e-6).unwrap();
        let osc = RelativisticOscillator::new(config);
        // At 10 GHz and 20 mK, should be in quantum regime
        assert!(osc.is_quantum_regime());
    }
}
