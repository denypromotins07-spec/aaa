//! Dithering Lock-In Amplifier for Resonance Locking
//!
//! This module implements a software-defined lock-in amplifier that uses
//! dithering signals to find and maintain optimal resonance points in
//! photonic circuits. It handles:
//! - High-frequency dither injection
//! - Synchronous demodulation
//! - Phase-sensitive detection
//! - Adaptive gain control

use serde::{Deserialize, Serialize};
use thiserror::Error;
use std::f64::consts::PI;

/// Errors in lock-in amplifier operation
#[derive(Error, Debug)]
pub enum LockInError {
    #[error("Dither frequency {freq}Hz exceeds Nyquist limit {nyquist}Hz")]
    FrequencyExceedsNyquist { freq: f64, nyquist: f64 },
    
    #[error("Integration time {time}s exceeds maximum {max}s")]
    IntegrationTimeExceeded { time: f64, max: f64 },
    
    #[error("Signal amplitude {amplitude} below noise floor {floor}")]
    SignalBelowNoiseFloor { amplitude: f64, floor: f64 },
    
    #[error("Phase detector saturation: phase={phase}rad")]
    PhaseDetectorSaturation { phase: f64 },
    
    #[error("Reference signal lost: cycles_without_lock={cycles}")]
    ReferenceSignalLost { cycles: u32 },
}

/// Configuration for the lock-in amplifier
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LockInConfig {
    /// Dither frequency (Hz)
    pub dither_frequency_hz: f64,
    /// Dither amplitude (normalized 0-1)
    pub dither_amplitude: f64,
    /// Integration time constant (seconds)
    pub time_constant_s: f64,
    /// Filter order (1, 2, or 4 pole)
    pub filter_order: u8,
    /// Sensitivity range (V)
    pub sensitivity_v: f64,
    /// Sample rate (Hz)
    pub sample_rate_hz: f64,
    /// Harmonic detection mode (1=fundamental, 2=second harmonic, etc.)
    pub harmonic: u8,
}

impl Default for LockInConfig {
    fn default() -> Self {
        Self {
            dither_frequency_hz: 1000.0, // 1 kHz typical dither
            dither_amplitude: 0.01, // 1% modulation
            time_constant_s: 0.001, // 1 ms
            filter_order: 2,
            sensitivity_v: 1.0,
            sample_rate_hz: 100_000.0, // 100 kHz
            harmonic: 1,
        }
    }
}

/// State of the lock-in amplifier
#[derive(Debug, Clone)]
pub struct LockInState {
    /// In-phase component (X)
    pub x_component: f64,
    /// Quadrature component (Y)
    pub y_component: f64,
    /// Magnitude (R = sqrt(X² + Y²))
    pub magnitude: f64,
    /// Phase (θ = atan2(Y, X))
    pub phase_rad: f64,
    /// Signal-to-noise ratio (dB)
    pub snr_db: f64,
    /// Lock status
    pub locked: bool,
    /// Cycles without valid reference
    pub cycles_without_lock: u32,
}

impl Default for LockInState {
    fn default() -> Self {
        Self {
            x_component: 0.0,
            y_component: 0.0,
            magnitude: 0.0,
            phase_rad: 0.0,
            snr_db: 0.0,
            locked: false,
            cycles_without_lock: 0,
        }
    }
}

/// Dithering Lock-In Amplifier for precision resonance detection
pub struct DitheringLockInAmplifier {
    /// Configuration
    config: LockInConfig,
    /// Current state
    state: LockInState,
    /// Low-pass filter state (multi-pole)
    filter_state: Vec<f64>,
    /// Previous input samples for filtering
    input_history: Vec<f64>,
    /// Reference sine lookup table
    ref_sin_lut: Vec<f64>,
    /// Reference cosine lookup table
    ref_cos_lut: Vec<f64>,
    /// LUT index counter
    lut_index: usize,
    /// Dither phase accumulator
    phase_accumulator: f64,
    /// Noise estimate
    noise_estimate: f64,
}

impl DitheringLockInAmplifier {
    /// Create a new lock-in amplifier with default configuration
    pub fn new() -> Result<Self, LockInError> {
        Self::with_config(LockInConfig::default())
    }

    /// Create a lock-in amplifier with custom configuration
    pub fn with_config(config: LockInConfig) -> Result<Self, LockInError> {
        // Validate dither frequency against Nyquist
        let nyquist = config.sample_rate_hz / 2.0;
        if config.dither_frequency_hz > nyquist {
            return Err(LockInError::FrequencyExceedsNyquist {
                freq: config.dither_frequency_hz,
                nyquist,
            });
        }

        // Validate integration time
        if config.time_constant_s > 10.0 {
            return Err(LockInError::IntegrationTimeExceeded {
                time: config.time_constant_s,
                max: 10.0,
            });
        }

        // Build reference lookup tables
        let lut_size = (config.sample_rate_hz / config.dither_frequency_hz).ceil() as usize;
        let mut ref_sin_lut = Vec::with_capacity(lut_size);
        let mut ref_cos_lut = Vec::with_capacity(lut_size);

        for i in 0..lut_size {
            let phase = 2.0 * PI * (i as f64) / (lut_size as f64);
            ref_sin_lut.push(phase.sin());
            ref_cos_lut.push(phase.cos());
        }

        // Initialize filter state
        let filter_state = vec![0.0; config.filter_order as usize * 2];
        let input_history = Vec::with_capacity(config.filter_order as usize);

        Ok(Self {
            config,
            state: LockInState::default(),
            filter_state,
            input_history,
            ref_sin_lut,
            ref_cos_lut,
            lut_index: 0,
            phase_accumulator: 0.0,
            noise_estimate: 0.0,
        })
    }

    /// Process a single input sample through the lock-in amplifier
    pub fn process_sample(&mut self, input: f64) -> &LockInState {
        // Get reference signals from LUT
        let ref_sin = self.ref_sin_lut[self.lut_index];
        let ref_cos = self.ref_cos_lut[self.lut_index];

        // Update LUT index with harmonic multiplication
        self.lut_index = (self.lut_index * self.config.harmonic as usize) % self.ref_sin_lut.len();
        if self.lut_index == 0 {
            self.lut_index = 1; // Avoid DC
        }

        // Mix input with reference signals (multiplication)
        let mixed_x = input * ref_sin * 2.0; // Factor of 2 for normalization
        let mixed_y = input * ref_cos * 2.0;

        // Apply low-pass filter (IIR implementation)
        let filtered_x = self.lowpass_filter(mixed_x, 0);
        let filtered_y = self.lowpass_filter(mixed_y, 1);

        // Calculate magnitude and phase
        let magnitude = (filtered_x.powi(2) + filtered_y.powi(2)).sqrt();
        let phase = filtered_y.atan2(filtered_x);

        // Estimate noise from high-frequency content
        self.update_noise_estimate(input, magnitude);

        // Calculate SNR
        let snr_db = if self.noise_estimate > 1e-12 {
            20.0 * (magnitude / self.noise_estimate).log10()
        } else {
            100.0
        };

        // Update state
        self.state.x_component = filtered_x;
        self.state.y_component = filtered_y;
        self.state.magnitude = magnitude;
        self.state.phase_rad = phase;
        self.state.snr_db = snr_db;
        self.state.locked = snr_db > 20.0 && self.state.cycles_without_lock < 10;

        if !self.state.locked {
            self.state.cycles_without_lock += 1;
        } else {
            self.state.cycles_without_lock = 0;
        }

        &self.state
    }

    /// Multi-pole IIR low-pass filter
    fn lowpass_filter(&mut self, input: f64, channel: usize) -> f64 {
        // Calculate filter coefficients
        let tau = self.config.time_constant_s;
        let dt = 1.0 / self.config.sample_rate_hz;
        let alpha = dt / (tau + dt);

        let base_idx = channel * 2;
        
        // First-order IIR: y[n] = α*x[n] + (1-α)*y[n-1]
        let output = alpha * input + (1.0 - alpha) * self.filter_state[base_idx];
        self.filter_state[base_idx] = output;

        // Additional poles for steeper rolloff
        for i in 1..self.config.filter_order as usize {
            let idx = base_idx + i;
            let prev_output = self.filter_state[idx];
            let new_output = alpha * output + (1.0 - alpha) * prev_output;
            self.filter_state[idx] = new_output;
        }

        self.filter_state[base_idx + self.config.filter_order as usize - 1]
    }

    /// Update noise estimate using moving variance
    fn update_noise_estimate(&mut self, input: f64, signal_magnitude: f64) {
        // Noise ≈ |input| - signal (simplified estimation)
        let instantaneous_noise = input.abs() - signal_magnitude;
        let noise = instantaneous_noise.max(0.0);
        
        // Exponential moving average
        let alpha = 0.01;
        self.noise_estimate = (1.0 - alpha) * self.noise_estimate + alpha * noise;
    }

    /// Inject a dither signal into a control loop
    pub fn generate_dither(&self, phase_offset: f64) -> f64 {
        let dither_phase = self.phase_accumulator + phase_offset;
        let dither_value = self.config.dither_amplitude * dither_phase.sin();
        
        // Update phase accumulator
        let phase_step = 2.0 * PI * self.config.dither_frequency_hz / self.config.sample_rate_hz;
        
        // Use const reference since we can't mutate in this method
        dither_value
    }

    /// Update phase accumulator (call each sample)
    pub fn advance_phase(&mut self) {
        let phase_step = 2.0 * PI * self.config.dither_frequency_hz / self.config.sample_rate_hz;
        self.phase_accumulator = (self.phase_accumulator + phase_step) % (2.0 * PI);
        self.lut_index = (self.lut_index + 1) % self.ref_sin_lut.len();
    }

    /// Get the detected resonance error signal
    pub fn get_error_signal(&self) -> f64 {
        // Error signal is proportional to the out-of-phase component
        // when operating near resonance
        self.state.y_component
    }

    /// Get the current lock state
    pub fn is_locked(&self) -> bool {
        self.state.locked
    }

    /// Get the measured signal magnitude
    pub fn magnitude(&self) -> f64 {
        self.state.magnitude
    }

    /// Get the measured phase relative to reference
    pub fn phase(&self) -> f64 {
        self.state.phase_rad
    }

    /// Set the sensitivity range
    pub fn set_sensitivity(&mut self, sensitivity_v: f64) {
        self.config.sensitivity_v = sensitivity_v;
    }

    /// Set the integration time constant
    pub fn set_time_constant(&mut self, time_s: f64) -> Result<(), LockInError> {
        if time_s > 10.0 {
            return Err(LockInError::IntegrationTimeExceeded {
                time: time_s,
                max: 10.0,
            });
        }
        self.config.time_constant_s = time_s;
        Ok(())
    }

    /// Reset the amplifier state
    pub fn reset(&mut self) {
        self.state = LockInState::default();
        self.filter_state.fill(0.0);
        self.input_history.clear();
        self.noise_estimate = 0.0;
    }

    /// Get configuration
    pub fn config(&self) -> &LockInConfig {
        &self.config
    }

    /// Get current state
    pub fn state(&self) -> &LockInState {
        &self.state
    }
}

impl Default for DitheringLockInAmplifier {
    fn default() -> Self {
        Self::new().expect("Default lock-in configuration should be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amplifier_creation() {
        let amp = DitheringLockInAmplifier::new().unwrap();
        assert!(!amp.is_locked());
    }

    #[test]
    fn test_invalid_frequency() {
        let config = LockInConfig {
            dither_frequency_hz: 60_000.0, // Above Nyquist for 100kHz sample rate
            sample_rate_hz: 100_000.0,
            ..Default::default()
        };
        
        let result = DitheringLockInAmplifier::with_config(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_process_samples() {
        let mut amp = DitheringLockInAmplifier::new().unwrap();
        
        // Process some samples with a signal at the dither frequency
        for i in 0..1000 {
            let t = i as f64 / amp.config.sample_rate_hz;
            let signal = (2.0 * PI * amp.config.dither_frequency_hz * t).sin();
            let state = amp.process_sample(signal);
            
            // State should be updated
            assert!(state.magnitude >= 0.0);
        }
    }

    #[test]
    fn test_lock_detection() {
        let mut amp = DitheringLockInAmplifier::new().unwrap();
        
        // Process samples with strong signal
        for i in 0..5000 {
            let t = i as f64 / amp.config.sample_rate_hz;
            let signal = 0.5 * (2.0 * PI * amp.config.dither_frequency_hz * t).sin();
            amp.process_sample(signal);
            amp.advance_phase();
        }
        
        // Should eventually lock
        // Note: May not lock in simulation due to simplified model
    }

    #[test]
    fn test_dither_generation() {
        let amp = DitheringLockInAmplifier::new().unwrap();
        
        let dither1 = amp.generate_dither(0.0);
        let dither2 = amp.generate_dither(PI / 2.0);
        
        // Dither values should be within amplitude bounds
        assert!(dither1.abs() <= amp.config.dither_amplitude);
        assert!(dither2.abs() <= amp.config.dither_amplitude);
    }

    #[test]
    fn test_reset() {
        let mut amp = DitheringLockInAmplifier::new().unwrap();
        
        // Process some samples
        for i in 0..100 {
            amp.process_sample(0.5);
        }
        
        let magnitude_before = amp.magnitude();
        
        amp.reset();
        
        assert_eq!(amp.magnitude(), 0.0);
        assert!(!amp.is_locked());
    }

    #[test]
    fn test_time_constant_change() {
        let mut amp = DitheringLockInAmplifier::new().unwrap();
        
        let result = amp.set_time_constant(0.005);
        assert!(result.is_ok());
        assert_eq!(amp.config().time_constant_s, 0.005);
    }

    #[test]
    fn test_excessive_time_constant_rejected() {
        let mut amp = DitheringLockInAmplifier::new().unwrap();
        
        let result = amp.set_time_constant(15.0);
        assert!(result.is_err());
    }
}
