//! Photonic Time-Stretch ADC (TS-ADC) Simulation Engine
//!
//! This module implements a photonic time-stretch analog-to-digital converter
//! that overcomes the bandwidth-resolution product limit of electronic ADCs.
//! It uses chirped supercontinuum laser pulses to stretch ultra-fast RF signals
//! in the optical domain before detection by slower electronic photodetectors.
//!
//! Key features:
//! - Sub-picosecond temporal resolution
//! - 100+ GHz effective bandwidth
//! - Reduced aperture jitter
//! - Real-time capture of exchange network packets

use crate::adc::chirped_pulse_dispersion::{ChirpedPulseGenerator, DispersionConfig};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use num_complex::Complex64;

/// Errors in photonic time-stretch ADC operation
#[derive(Error, Debug)]
pub enum TsAdcError {
    #[error("Input signal bandwidth {bandwidth}GHz exceeds system limit {limit}GHz")]
    BandwidthExceeded { bandwidth: f64, limit: f64 },
    
    #[error("Stretch factor {factor} outside valid range [{min}, {max}]")]
    InvalidStretchFactor { factor: f64, min: f64, max: f64 },
    
    #[error("Laser pulse energy {energy}pJ below threshold {threshold}pJ")]
    InsufficientPulseEnergy { energy: f64, threshold: f64 },
    
    #[error("Dispersion compensation failed: residual_dispersion={residual}ps/nm")]
    DispersionCompensationFailed { residual: f64 },
    
    #[error("Photodetector saturation: peak_power={power}mW > max={max}mW")]
    DetectorSaturation { power: f64, max: f64 },
    
    #[error("SNR too low: {snr}dB < minimum {min_snr}dB")]
    InsufficientSnr { snr: f64, min_snr: f64 },
}

/// Configuration for the TS-ADC system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsAdcConfig {
    /// Center wavelength of the laser (nm)
    pub center_wavelength_nm: f64,
    /// Laser repetition rate (MHz)
    pub rep_rate_mhz: f64,
    /// Pulse duration (fs FWHM)
    pub pulse_duration_fs: f64,
    /// Stretch factor (time expansion ratio)
    pub stretch_factor: f64,
    /// First-order dispersion (ps/nm)
    pub dispersion_ps_nm: f64,
    /// Second-order dispersion (ps/nm²)
    pub dispersion_second_order: f64,
    /// Photodetector bandwidth (GHz)
    pub detector_bandwidth_ghz: f64,
    /// Electronic ADC sampling rate (GS/s)
    pub adc_sampling_rate_gsps: f64,
    /// Electronic ADC resolution (bits)
    pub adc_resolution_bits: u8,
    /// Maximum input signal bandwidth (GHz)
    pub max_input_bandwidth_ghz: f64,
}

impl Default for TsAdcConfig {
    fn default() -> Self {
        Self {
            center_wavelength_nm: 1550.0,
            rep_rate_mhz: 80.0,
            pulse_duration_fs: 200.0,
            stretch_factor: 10.0,
            dispersion_ps_nm: 1000.0,
            dispersion_second_order: 0.1,
            detector_bandwidth_ghz: 10.0,
            adc_sampling_rate_gsps: 80.0,
            adc_resolution_bits: 12,
            max_input_bandwidth_ghz: 100.0,
        }
    }
}

/// Captured frame from the TS-ADC
#[derive(Debug, Clone)]
pub struct CapturedFrame {
    /// Timestamp of capture (femtoseconds since epoch)
    pub timestamp_fs: u128,
    /// Stretched time samples (volts)
    pub samples: Vec<f64>,
    /// Effective time per sample (fs)
    pub time_per_sample_fs: f64,
    /// Signal-to-noise ratio (dB)
    pub snr_db: f64,
    /// Number of bits of effective resolution
    pub effective_bits: f64,
    /// Frame is valid
    pub valid: bool,
}

/// Photonic Time-Stretch ADC engine
pub struct PhotonicTimeStretchAdc {
    /// System configuration
    config: TsAdcConfig,
    /// Chirped pulse generator
    pulse_generator: ChirpedPulseGenerator,
    /// Current stretch factor
    current_stretch_factor: f64,
    /// Calibration data for dispersion compensation
    dispersion_calibration: Vec<(f64, f64)>,
    /// Temperature for thermal drift compensation
    temperature_c: f64,
}

impl PhotonicTimeStretchAdc {
    /// Create a new TS-ADC with default configuration
    pub fn new() -> Result<Self, TsAdcError> {
        Self::with_config(TsAdcConfig::default())
    }

    /// Create a TS-ADC with custom configuration
    pub fn with_config(config: TsAdcConfig) -> Result<Self, TsAdcError> {
        // Validate stretch factor
        if config.stretch_factor < 1.0 || config.stretch_factor > 100.0 {
            return Err(TsAdcError::InvalidStretchFactor {
                factor: config.stretch_factor,
                min: 1.0,
                max: 100.0,
            });
        }

        // Validate input bandwidth
        if config.max_input_bandwidth_ghz > 200.0 {
            return Err(TsAdcError::BandwidthExceeded {
                bandwidth: config.max_input_bandwidth_ghz,
                limit: 200.0,
            });
        }

        let pulse_generator = ChirpedPulseGenerator::new(
            config.center_wavelength_nm,
            config.pulse_duration_fs,
            config.dispersion_ps_nm,
        );

        Ok(Self {
            config,
            pulse_generator,
            current_stretch_factor: config.stretch_factor,
            dispersion_calibration: Vec::new(),
            temperature_c: 25.0,
        })
    }

    /// Capture an RF signal using time-stretch technique
    /// 
    /// The process:
    /// 1. Generate chirped supercontinuum pulse
    /// 2. Modulate pulse with RF signal via electro-optic modulator
    /// 3. Apply additional dispersion to stretch the waveform
    /// 4. Detect with balanced photodetector
    /// 5. Digitize with electronic ADC
    pub fn capture(&mut self, rf_signal: &[Complex64], sample_rate_ghz: f64) -> Result<CapturedFrame, TsAdcError> {
        // Validate signal bandwidth
        let signal_bandwidth = sample_rate_ghz / 2.0;
        if signal_bandwidth > self.config.max_input_bandwidth_ghz {
            return Err(TsAdcError::BandwidthExceeded {
                bandwidth: signal_bandwidth,
                limit: self.config.max_input_bandwidth_ghz,
            });
        }

        let n_samples = rf_signal.len();
        
        // Calculate effective stretched sampling parameters
        let stretched_sample_rate = sample_rate_ghz / self.current_stretch_factor;
        let time_per_sample_fs = 1e6 / stretched_sample_rate; // fs per sample

        // Generate chirped pulse envelope
        let pulse_envelope = self.pulse_generator.generate_chirped_pulse(n_samples)?;

        // Modulate pulse with RF signal (simulating electro-optic modulation)
        let mut modulated_signal = Vec::with_capacity(n_samples);
        for (i, &rf_sample) in rf_signal.iter().enumerate() {
            let envelope = pulse_envelope[i].unwrap_or(Complex64::new(1.0, 0.0));
            // Intensity modulation: I ∝ |E|^2
            let modulated = envelope * (1.0 + 0.5 * rf_sample);
            modulated_signal.push(modulated);
        }

        // Apply dispersion-induced time stretch
        let stretched_signal = self.apply_time_stretch(&modulated_signal)?;

        // Simulate photodetection (square-law detection)
        let detected_signal: Vec<f64> = stretched_signal
            .iter()
            .map(|&s| s.norm_sqr())
            .collect();

        // Check for detector saturation
        let peak_power = detected_signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if peak_power > 10.0 { // 10 mW saturation limit
            return Err(TsAdcError::DetectorSaturation {
                power: peak_power,
                max: 10.0,
            });
        }

        // Simulate ADC quantization
        let quantized_samples = self.quantize_signal(&detected_signal);

        // Calculate SNR
        let snr_db = self.calculate_snr(&quantized_samples, &detected_signal);

        // Calculate effective number of bits (ENOB)
        let effective_bits = (snr_db - 1.76) / 6.02;

        // Generate timestamp
        let timestamp_fs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u128;

        Ok(CapturedFrame {
            timestamp_fs,
            samples: quantized_samples,
            time_per_sample_fs,
            snr_db,
            effective_bits,
            valid: snr_db > 20.0 && effective_bits > 4.0,
        })
    }

    /// Apply time-stretch transformation via dispersion
    fn apply_time_stretch(&self, signal: &[Complex64]) -> Result<Vec<Complex64>, TsAdcError> {
        let n = signal.len();
        let mut stretched = Vec::with_capacity(n);

        // Time-stretch factor determines the amount of dispersion applied
        // Δt_stretched = M * Δt_original where M is stretch factor
        let total_dispersion = self.config.dispersion_ps_nm * self.current_stretch_factor;

        // Apply linear chirp corresponding to dispersion
        for (i, &sample) in signal.iter().enumerate() {
            let t = i as f64 / n as f64;
            let chirp_phase = PI * total_dispersion * t.powi(2);
            let phase_factor = Complex64::new(chirp_phase.cos(), chirp_phase.sin());
            stretched.push(sample * phase_factor);
        }

        Ok(stretched)
    }

    /// Quantize signal to ADC resolution
    fn quantize_signal(&self, signal: &[f64]) -> Vec<f64> {
        let num_levels = 1u32 << self.config.adc_resolution_bits;
        let max_val = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_val = signal.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = max_val - min_val;

        if range < 1e-12 {
            return vec![0.0; signal.len()];
        }

        signal
            .iter()
            .map(|&v| {
                let normalized = (v - min_val) / range;
                let level = (normalized * (num_levels - 1) as f64).round() as u32;
                let quantized = level as f64 / (num_levels - 1) as f64;
                quantized * range + min_val
            })
            .collect()
    }

    /// Calculate SNR between quantized and original signal
    fn calculate_snr(&self, quantized: &[f64], original: &[f64]) -> f64 {
        if quantized.len() != original.len() || quantized.is_empty() {
            return 0.0;
        }

        let mut signal_power = 0.0;
        let mut noise_power = 0.0;

        for (q, o) in quantized.iter().zip(original.iter()) {
            signal_power += o.powi(2);
            let error = q - o;
            noise_power += error.powi(2);
        }

        if noise_power < 1e-15 {
            return 100.0;
        }

        10.0 * (signal_power / noise_power).log10()
    }

    /// Calibrate dispersion compensation
    pub fn calibrate_dispersion(&mut self) -> Result<(), TsAdcError> {
        self.dispersion_calibration.clear();

        // Sweep through dispersion values and measure optimal point
        for dispersion_offset in -100..=100 {
            let test_dispersion = self.config.dispersion_ps_nm + dispersion_offset as f64;
            
            // Measure pulse width at this dispersion
            let pulse_width = self.pulse_generator.measure_pulse_width(test_dispersion);
            
            self.dispersion_calibration.push((test_dispersion, pulse_width));
        }

        Ok(())
    }

    /// Set the stretch factor dynamically
    pub fn set_stretch_factor(&mut self, factor: f64) -> Result<(), TsAdcError> {
        if factor < 1.0 || factor > 100.0 {
            return Err(TsAdcError::InvalidStretchFactor {
                factor,
                min: 1.0,
                max: 100.0,
            });
        }
        self.current_stretch_factor = factor;
        Ok(())
    }

    /// Get current configuration
    pub fn config(&self) -> &TsAdcConfig {
        &self.config
    }

    /// Set operating temperature for thermal drift compensation
    pub fn set_temperature(&mut self, temp_c: f64) {
        self.temperature_c = temp_c;
        // Adjust dispersion for thermal drift
        // Silicon: ~0.01 ps/nm/°C typical
        let drift = (temp_c - 25.0) * 0.01;
        self.pulse_generator.set_dispersion(self.config.dispersion_ps_nm + drift);
    }

    /// Get the effective temporal resolution (fs)
    pub fn temporal_resolution_fs(&self) -> f64 {
        self.config.pulse_duration_fs / self.current_stretch_factor
    }

    /// Get the effective bandwidth after stretching (GHz)
    pub fn effective_bandwidth_ghz(&self) -> f64 {
        self.config.detector_bandwidth_ghz * self.current_stretch_factor
    }
}

impl Default for PhotonicTimeStretchAdc {
    fn default() -> Self {
        Self::new().expect("Default TS-ADC configuration should be valid")
    }
}

use std::f64::consts::PI;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adc_creation() {
        let adc = PhotonicTimeStretchAdc::new().unwrap();
        assert_eq!(adc.config().stretch_factor, 10.0);
    }

    #[test]
    fn test_invalid_stretch_factor() {
        let config = TsAdcConfig {
            stretch_factor: 150.0, // Invalid
            ..Default::default()
        };
        let result = PhotonicTimeStretchAdc::with_config(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_capture_simulation() {
        let mut adc = PhotonicTimeStretchAdc::new().unwrap();
        
        // Generate test RF signal (complex baseband)
        let n_samples = 1024;
        let mut signal = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let t = i as f64 / n_samples as f64;
            let re = (2.0 * PI * 10.0 * t).cos();
            let im = (2.0 * PI * 10.0 * t).sin();
            signal.push(Complex64::new(re * 0.5, im * 0.5));
        }

        let frame = adc.capture(&signal, 50.0).unwrap();
        
        assert_eq!(frame.samples.len(), n_samples);
        assert!(frame.valid);
        assert!(frame.snr_db > 0.0);
    }

    #[test]
    fn test_bandwidth_exceeded() {
        let mut adc = PhotonicTimeStretchAdc::new().unwrap();
        
        // Signal with bandwidth exceeding limit
        let signal = vec![Complex64::new(1.0, 0.0); 1024];
        let result = adc.capture(&signal, 500.0); // 250 GHz bandwidth
        
        assert!(result.is_err());
    }

    #[test]
    fn test_stretch_factor_change() {
        let mut adc = PhotonicTimeStretchAdc::new().unwrap();
        
        adc.set_stretch_factor(20.0).unwrap();
        assert_eq!(adc.config.stretch_factor, 10.0); // Config doesn't change
        // But internal stretch factor does
        assert!((adc.current_stretch_factor - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_temporal_resolution() {
        let adc = PhotonicTimeStretchAdc::new().unwrap();
        
        // Resolution = pulse_duration / stretch_factor
        let expected = 200.0 / 10.0; // 20 fs
        assert!((adc.temporal_resolution_fs() - expected).abs() < 1.0);
    }

    #[test]
    fn test_effective_bandwidth() {
        let adc = PhotonicTimeStretchAdc::new().unwrap();
        
        // Effective BW = detector_BW * stretch_factor
        let expected = 10.0 * 10.0; // 100 GHz
        assert!((adc.effective_bandwidth_ghz() - expected).abs() < 1.0);
    }
}
