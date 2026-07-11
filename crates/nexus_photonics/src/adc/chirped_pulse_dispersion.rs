//! Chirped Pulse Dispersion Engine
//!
//! This module generates and manages chirped optical pulses for the
//! photonic time-stretch ADC. It handles:
//! - Supercontinuum generation modeling
//! - Group velocity dispersion (GVD) control
//! - Higher-order dispersion compensation
//! - Pulse shaping and characterization

use serde::{Deserialize, Serialize};
use thiserror::Error;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Errors in chirped pulse generation
#[derive(Error, Debug)]
pub enum DispersionError {
    #[error("Dispersion value {value}ps/nm exceeds physical limits [{min}, {max}]")]
    DispersionOutOfRange { value: f64, min: f64, max: f64 },
    
    #[error("Pulse duration {duration}fs below transform limit {limit}fs")]
    BelowTransformLimit { duration: f64, limit: f64 },
    
    #[error("Wavelength {wavelength}nm outside fiber transmission window")]
    InvalidWavelength { wavelength: f64 },
    
    #[error("Nonlinear phase shift {phase}rad exceeds SPM threshold")]
    ExcessiveNonlinearity { phase: f64 },
}

/// Configuration for dispersion management
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DispersionConfig {
    /// Second-order dispersion β₂ (ps²/km)
    pub beta2: f64,
    /// Third-order dispersion β₃ (ps³/km)
    pub beta3: f64,
    /// Fiber length (km)
    pub length_km: f64,
    /// Nonlinear coefficient γ (1/W/km)
    pub nonlinear_gamma: f64,
    /// Loss coefficient (dB/km)
    pub loss_db_km: f64,
}

impl Default for DispersionConfig {
    fn default() -> Self {
        Self {
            beta2: -20.0, // Typical SMF-28 at 1550nm
            beta3: 0.1,
            length_km: 1.0,
            nonlinear_gamma: 1.3,
            loss_db_km: 0.2,
        }
    }
}

/// Characteristics of a chirped pulse
#[derive(Debug, Clone)]
pub struct PulseCharacteristics {
    /// Full width at half maximum (fs)
    pub fwhm_fs: f64,
    /// Time-bandwidth product
    pub tbp: f64,
    /// Chirp parameter (dimensionless)
    pub chirp_parameter: f64,
    /// Peak power (W)
    pub peak_power_w: f64,
    /// Pulse energy (pJ)
    pub energy_pj: f64,
    /// Spectral bandwidth (nm)
    pub bandwidth_nm: f64,
}

/// Chirped Pulse Generator - creates stretched optical pulses
pub struct ChirpedPulseGenerator {
    /// Center wavelength (nm)
    center_wavelength_nm: f64,
    /// Initial pulse duration (fs)
    initial_duration_fs: f64,
    /// Applied dispersion (ps/nm)
    dispersion_ps_nm: f64,
    /// Dispersion configuration
    dispersion_config: DispersionConfig,
    /// Current pulse characteristics
    current_characteristics: Option<PulseCharacteristics>,
}

impl ChirpedPulseGenerator {
    /// Create a new chirped pulse generator
    pub fn new(center_wavelength_nm: f64, initial_duration_fs: f64, dispersion_ps_nm: f64) -> Self {
        Self {
            center_wavelength_nm,
            initial_duration_fs,
            dispersion_ps_nm,
            dispersion_config: DispersionConfig::default(),
            current_characteristics: None,
        }
    }

    /// Create with custom dispersion configuration
    pub fn with_dispersion_config(
        center_wavelength_nm: f64,
        initial_duration_fs: f64,
        config: DispersionConfig,
    ) -> Result<Self, DispersionError> {
        // Validate dispersion values
        let total_dispersion = config.beta2 * config.length_km;
        if total_dispersion < -1000.0 || total_dispersion > 1000.0 {
            return Err(DispersionError::DispersionOutOfRange {
                value: total_dispersion,
                min: -1000.0,
                max: 1000.0,
            });
        }

        Ok(Self {
            center_wavelength_nm,
            initial_duration_fs,
            dispersion_ps_nm: total_dispersion,
            dispersion_config: config,
            current_characteristics: None,
        })
    }

    /// Generate a chirped pulse envelope
    pub fn generate_chirped_pulse(&self, n_samples: usize) -> Result<Vec<Option<Complex64>>, DispersionError> {
        let mut pulse = Vec::with_capacity(n_samples);
        
        // Calculate chirped pulse parameters
        let stretched_duration = self.calculate_stretched_duration();
        let chirp_rate = self.dispersion_ps_nm / stretched_duration.powi(2);

        for i in 0..n_samples {
            let t = (i as f64 - n_samples as f64 / 2.0) / (n_samples as f64 / 4.0);
            
            // Gaussian envelope with chirp
            let envelope = (-t.powi(2) / 2.0).exp();
            
            // Quadratic phase from chirp
            let phase = chirp_rate * t.powi(2);
            let complex_amplitude = Complex64::new(phase.cos(), phase.sin()) * envelope;
            
            pulse.push(Some(complex_amplitude));
        }

        // Update characteristics
        self.current_characteristics = Some(self.measure_characteristics());

        Ok(pulse)
    }

    /// Calculate the stretched pulse duration after dispersion
    pub fn calculate_stretched_duration(&self) -> f64 {
        // For a Gaussian pulse with linear chirp:
        // τ_out = τ_in * sqrt(1 + (4*ln(2)*D*L/τ_in²)²)
        
        let wavelength_m = self.center_wavelength_nm * 1e-9;
        let tau0_s = self.initial_duration_fs * 1e-15;
        
        // Convert dispersion to SI units
        let d_total = self.dispersion_ps_nm * 1e-12; // ps/nm to s/m
        
        // Approximate stretched duration
        let chirp_factor = 1.0 + (4.0 * 0.693 * d_total / tau0_s.powi(2)).powi(2);
        let stretched_fs = self.initial_duration_fs * chirp_factor.sqrt();
        
        stretched_fs
    }

    /// Measure pulse characteristics
    pub fn measure_characteristics(&self) -> PulseCharacteristics {
        let stretched_duration = self.calculate_stretched_duration();
        
        // Calculate spectral bandwidth (transform-limited)
        // Δν * Δt ≈ 0.44 for Gaussian pulses
        let c = 299792458.0; // Speed of light (m/s)
        let frequency_hz = c / (self.center_wavelength_nm * 1e-9);
        
        let tbp_ideal = 0.44;
        let bandwidth_thz = tbp_ideal / (stretched_duration * 1e-12);
        
        // Convert to wavelength bandwidth
        let bandwidth_nm = (self.center_wavelength_nm.powi(2) * 1e-9 / c) * bandwidth_thz * 1e12;
        
        // Chirp parameter
        let chirp_param = self.dispersion_ps_nm / stretched_duration;
        
        PulseCharacteristics {
            fwhm_fs: stretched_duration,
            tbp: tbp_ideal * (1.0 + chirp_param.powi(2)).sqrt(),
            chirp_parameter: chirp_param,
            peak_power_w: 1.0, // Normalized
            energy_pj: stretched_duration * 1e-3, // Approximate
            bandwidth_nm,
        }
    }

    /// Measure pulse width at a given dispersion
    pub fn measure_pulse_width(&self, dispersion_ps_nm: f64) -> f64 {
        let temp_gen = ChirpedPulseGenerator::new(
            self.center_wavelength_nm,
            self.initial_duration_fs,
            dispersion_ps_nm,
        );
        temp_gen.calculate_stretched_duration()
    }

    /// Set the applied dispersion
    pub fn set_dispersion(&mut self, dispersion_ps_nm: f64) {
        self.dispersion_ps_nm = dispersion_ps_nm;
    }

    /// Get current dispersion
    pub fn dispersion(&self) -> f64 {
        self.dispersion_ps_nm
    }

    /// Apply dispersion compensation
    pub fn compensate_dispersion(&mut self, target_dispersion: f64) -> Result<(), DispersionError> {
        let residual = self.dispersion_ps_nm - target_dispersion;
        
        if residual.abs() > 10.0 {
            return Err(DispersionError::DispersionCompensationFailed {
                residual,
            });
        }

        self.dispersion_ps_nm = target_dispersion;
        Ok(())
    }

    /// Calculate nonlinear phase shift from self-phase modulation
    pub fn calculate_spm_phase(&self, peak_power_w: f64) -> f64 {
        let gamma = self.dispersion_config.nonlinear_gamma;
        let leff = self.effective_length();
        
        // φ_NL = γ * P_peak * L_eff
        gamma * peak_power_w * leff
    }

    /// Calculate effective fiber length accounting for loss
    fn effective_length(&self) -> f64 {
        let alpha = self.dispersion_config.loss_db_km / 4.343; // Convert dB/km to 1/km
        let l = self.dispersion_config.length_km;
        
        if alpha < 1e-6 {
            return l;
        }
        
        (1.0 - (-alpha * l).exp()) / alpha
    }

    /// Get pulse characteristics
    pub fn characteristics(&self) -> Option<&PulseCharacteristics> {
        self.current_characteristics.as_ref()
    }

    /// Verify pulse is above transform limit
    pub fn verify_transform_limit(&self) -> Result<(), DispersionError> {
        let characteristics = self.measure_characteristics();
        
        // Transform-limited duration for given bandwidth
        let c = 299792458.0;
        let delta_lambda_m = characteristics.bandwidth_nm * 1e-9;
        let lambda0_m = self.center_wavelength_nm * 1e-9;
        
        // Δt_min ≈ 0.44 * λ² / (c * Δλ)
        let transform_limit_fs = 0.44 * lambda0_m.powi(2) / (c * delta_lambda_m) * 1e15;
        
        if characteristics.fwhm_fs < transform_limit_fs {
            return Err(DispersionError::BelowTransformLimit {
                duration: characteristics.fwhm_fs,
                limit: transform_limit_fs,
            });
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_creation() {
        let gen = ChirpedPulseGenerator::new(1550.0, 200.0, 1000.0);
        assert_eq!(gen.dispersion(), 1000.0);
    }

    #[test]
    fn test_stretched_duration() {
        let gen = ChirpedPulseGenerator::new(1550.0, 200.0, 1000.0);
        let stretched = gen.calculate_stretched_duration();
        
        // Stretched duration should be larger than initial
        assert!(stretched > 200.0);
    }

    #[test]
    fn test_pulse_generation() {
        let gen = ChirpedPulseGenerator::new(1550.0, 200.0, 1000.0);
        let pulse = gen.generate_chirped_pulse(256).unwrap();
        
        assert_eq!(pulse.len(), 256);
        assert!(pulse.iter().all(|x| x.is_some()));
    }

    #[test]
    fn test_pulse_characteristics() {
        let gen = ChirpedPulseGenerator::new(1550.0, 200.0, 1000.0);
        let chars = gen.measure_characteristics();
        
        assert!(chars.fwhm_fs > 0.0);
        assert!(chars.tbp >= 0.44); // Minimum TBP for Gaussian
        assert!(chars.bandwidth_nm > 0.0);
    }

    #[test]
    fn test_dispersion_compensation() {
        let mut gen = ChirpedPulseGenerator::new(1550.0, 200.0, 1000.0);
        
        let result = gen.compensate_dispersion(995.0);
        assert!(result.is_ok());
        assert!((gen.dispersion() - 995.0).abs() < 0.01);
    }

    #[test]
    fn test_failed_compensation() {
        let mut gen = ChirpedPulseGenerator::new(1550.0, 200.0, 1000.0);
        
        // Large residual should fail
        let result = gen.compensate_dispersion(500.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_spm_phase() {
        let gen = ChirpedPulseGenerator::new(1550.0, 200.0, 1000.0);
        let phase = gen.calculate_spm_phase(1.0); // 1W peak power
        
        assert!(phase >= 0.0);
    }

    #[test]
    fn test_effective_length() {
        let gen = ChirpedPulseGenerator::new(1550.0, 200.0, 1000.0);
        let leff = gen.effective_length();
        
        // Effective length should be less than physical length due to loss
        assert!(leff <= gen.dispersion_config.length_km);
        assert!(leff > 0.0);
    }
}
