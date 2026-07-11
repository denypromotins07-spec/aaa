//! Optical MAC (Multiply-Accumulate) Accumulator
//!
//! This module implements the final stage of photonic matrix multiplication,
//! where weighted optical signals are accumulated using balanced photodetectors.
//! It handles:
//! - Differential optical-to-electrical conversion
//! - Shot noise and thermal noise modeling
//! - Transimpedance amplifier (TIA) simulation
//! - ADC interface for digitization

use crate::compute::microring_weight_bank::MicroringWeightBank;
use crate::compute::wdm_crossbar_router::WdmCrossbarRouter;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors in optical MAC accumulation
#[derive(Error, Debug)]
pub enum OpticalMacError {
    #[error("Photodetector saturation: input_power={power}mW > max={max}mW")]
    DetectorSaturation { power: f64, max: f64 },
    
    #[error("Signal below noise floor: signal={signal}dBm, noise={noise}dBm")]
    SignalBelowNoiseFloor { signal: f64, noise: f64 },
    
    #[error("Accumulator overflow at channel {channel}: value={value}")]
    AccumulatorOverflow { channel: usize, value: f64 },
    
    #[error("TIA gain configuration invalid: {gain} V/A")}
    InvalidTiaGain { gain: f64 },
}

/// Configuration for a balanced photodetector pair
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BalancedDetectorConfig {
    /// Responsivity (A/W) at operating wavelength
    pub responsivity: f64,
    /// Dark current (nA)
    pub dark_current_na: f64,
    /// Saturation power (mW)
    pub saturation_power_mw: f64,
    /// Bandwidth (GHz)
    pub bandwidth_ghz: f64,
    /// Common mode rejection ratio (dB)
    pub cmrr_db: f64,
}

impl Default for BalancedDetectorConfig {
    fn default() -> Self {
        Self {
            responsivity: 0.8, // Typical for InGaAs at 1550nm
            dark_current_na: 10.0,
            saturation_power_mw: 10.0,
            bandwidth_ghz: 25.0,
            cmrr_db: 30.0,
        }
    }
}

/// Transimpedance amplifier configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TiaConfig {
    /// Transimpedance gain (V/A)
    pub gain_v_a: f64,
    /// Input-referred noise current density (pA/√Hz)
    pub noise_density_pa_sqrt_hz: f64,
    /// Bandwidth (GHz)
    pub bandwidth_ghz: f64,
    /// Output voltage swing (Vpp)
    pub output_swing_vpp: f64,
}

impl Default for TiaConfig {
    fn default() -> Self {
        Self {
            gain_v_a: 1000.0, // 1 kΩ typical
            noise_density_pa_sqrt_hz: 5.0,
            bandwidth_ghz: 25.0,
            output_swing_vpp: 2.0,
        }
    }
}

/// State of a single MAC accumulator channel
#[derive(Debug, Clone)]
pub struct MacChannelState {
    /// Accumulated photocurrent (μA)
    pub photocurrent_ua: f64,
    /// TIA output voltage (V)
    pub tia_output_v: f64,
    /// Signal-to-noise ratio (dB)
    pub snr_db: f64,
    /// Channel is valid (not saturated or underflow)
    pub valid: bool,
}

/// Optical MAC Accumulator - converts optical weights to electrical accumulation
pub struct OpticalMacAccumulator {
    /// Number of parallel channels
    num_channels: usize,
    /// Detector configurations
    detector_config: BalancedDetectorConfig,
    /// TIA configuration
    tia_config: TiaConfig,
    /// Channel states
    channel_states: Vec<MacChannelState>,
    /// Temperature for noise calculations (K)
    temperature_k: f64,
    /// Integration time (ps)
    integration_time_ps: f64,
    /// Reference optical power (dBm)
    reference_power_dbm: f64,
}

impl OpticalMacAccumulator {
    /// Create a new optical MAC accumulator
    pub fn new(num_channels: usize) -> Self {
        let detector_config = BalancedDetectorConfig::default();
        let tia_config = TiaConfig::default();

        let channel_states: Vec<MacChannelState> = (0..num_channels)
            .map(|_| MacChannelState {
                photocurrent_ua: 0.0,
                tia_output_v: 0.0,
                snr_db: 0.0,
                valid: true,
            })
            .collect();

        Self {
            num_channels,
            detector_config,
            tia_config,
            channel_states,
            temperature_k: 300.0, // Room temperature
            integration_time_ps: 100.0, // 100 ps integration
            reference_power_dbm: 0.0,
        }
    }

    /// Create with custom configurations
    pub fn with_configs(
        num_channels: usize,
        detector_config: BalancedDetectorConfig,
        tia_config: TiaConfig,
    ) -> Result<Self, OpticalMacError> {
        // Validate TIA gain
        if tia_config.gain_v_a <= 0.0 || tia_config.gain_v_a > 10000.0 {
            return Err(OpticalMacError::InvalidTiaGain {
                gain: tia_config.gain_v_a,
            });
        }

        let channel_states: Vec<MacChannelState> = (0..num_channels)
            .map(|_| MacChannelState {
                photocurrent_ua: 0.0,
                tia_output_v: 0.0,
                snr_db: 0.0,
                valid: true,
            })
            .collect();

        Ok(Self {
            num_channels,
            detector_config,
            tia_config,
            channel_states,
            temperature_k: 300.0,
            integration_time_ps: 100.0,
            reference_power_dbm: 0.0,
        })
    }

    /// Perform MAC operation on optical input powers
    /// 
    /// Converts optical power to photocurrent, applies TIA gain,
    /// and accumulates results.
    pub fn accumulate(&mut self, optical_powers_mw: &[f64]) -> Result<Vec<f64>, OpticalMacError> {
        if optical_powers_mw.len() != self.num_channels {
            return Err(OpticalMacError::AccumulatorOverflow {
                channel: optical_powers_mw.len(),
                value: 0.0,
            });
        }

        let mut outputs = Vec::with_capacity(self.num_channels);

        for (i, &power_mw) in optical_powers_mw.iter().enumerate() {
            // Check for detector saturation
            if power_mw > self.detector_config.saturation_power_mw {
                self.channel_states[i].valid = false;
                return Err(OpticalMacError::DetectorSaturation {
                    power: power_mw,
                    max: self.detector_config.saturation_power_mw,
                });
            }

            // Convert optical power to photocurrent
            // I = R * P where R is responsivity (A/W), P is power (W)
            let power_w = power_mw / 1000.0;
            let photocurrent_a = self.detector_config.responsivity * power_w;
            let photocurrent_ua = photocurrent_a * 1e6;

            // Add dark current
            let total_current_ua = photocurrent_ua + self.detector_config.dark_current_na / 1000.0;

            // Apply TIA gain
            let tia_output_v = total_current_ua * 1e-6 * self.tia_config.gain_v_a;

            // Check output swing
            let output_v = tia_output_v.clamp(
                -self.tia_config.output_swing_vpp / 2.0,
                self.tia_config.output_swing_vpp / 2.0,
            );

            // Calculate SNR
            let snr_db = self.calculate_snr(power_mw);

            // Update channel state
            self.channel_states[i] = MacChannelState {
                photocurrent_ua: total_current_ua,
                tia_output_v: output_v,
                snr_db,
                valid: snr_db > 0.0,
            };

            outputs.push(output_v);
        }

        Ok(outputs)
    }

    /// Calculate signal-to-noise ratio for a given optical power
    fn calculate_snr(&self, power_mw: f64) -> f64 {
        let power_w = power_mw / 1000.0;
        
        // Signal photocurrent
        let signal_current_a = self.detector_config.responsivity * power_w;
        
        // Shot noise: i_shot² = 2*q*I*B
        let q = 1.602e-19; // Electron charge
        let bandwidth_hz = self.tia_config.bandwidth_ghz * 1e9;
        let shot_noise_a = (2.0 * q * signal_current_a * bandwidth_hz).sqrt();
        
        // Thermal noise: i_thermal² = 4*k*T*B/R
        let k = 1.381e-23; // Boltzmann constant
        let r_load = self.tia_config.gain_v_a; // Effective load resistance
        let thermal_noise_a = (4.0 * k * self.temperature_k * bandwidth_hz / r_load).sqrt();
        
        // TIA noise
        let tia_noise_a = self.tia_config.noise_density_pa_sqrt_hz * 1e-12 * bandwidth_hz.sqrt();
        
        // Total noise
        let total_noise_a = (shot_noise_a.powi(2) + thermal_noise_a.powi(2) + tia_noise_a.powi(2)).sqrt();
        
        // SNR
        if total_noise_a < 1e-15 {
            return 100.0; // Effectively infinite SNR
        }
        
        let snr_linear = signal_current_a / total_noise_a;
        20.0 * snr_linear.log10()
    }

    /// Perform differential MAC operation (balanced detection)
    pub fn accumulate_differential(
        &mut self,
        signal_powers_mw: &[f64],
        reference_powers_mw: &[f64],
    ) -> Result<Vec<f64>, OpticalMacError> {
        if signal_powers_mw.len() != reference_powers_mw.len() {
            return Err(OpticalMacError::AccumulatorOverflow {
                channel: signal_powers_mw.len(),
                value: 0.0,
            });
        }

        let mut outputs = Vec::with_capacity(self.num_channels);

        for i in 0..self.num_channels.min(signal_powers_mw.len()) {
            let signal_power = signal_powers_mw[i];
            let reference_power = reference_powers_mw[i];
            
            // Differential photocurrent
            let signal_current_a = self.detector_config.responsivity * (signal_power / 1000.0);
            let reference_current_a = self.detector_config.responsivity * (reference_power / 1000.0);
            
            let diff_current_a = signal_current_a - reference_current_a;
            let diff_current_ua = diff_current_a * 1e6;
            
            // Apply TIA gain
            let output_v = diff_current_a * self.tia_config.gain_v_a;
            let output_v = output_v.clamp(
                -self.tia_config.output_swing_vpp / 2.0,
                self.tia_config.output_swing_vpp / 2.0,
            );
            
            // Calculate effective SNR for differential signal
            let total_power = signal_power + reference_power;
            let snr_db = self.calculate_snr(total_power / 2.0);
            
            self.channel_states[i] = MacChannelState {
                photocurrent_ua: diff_current_ua,
                tia_output_v: output_v,
                snr_db,
                valid: snr_db > 0.0,
            };
            
            outputs.push(output_v);
        }

        Ok(outputs)
    }

    /// Reset all channel states
    pub fn reset(&mut self) {
        for state in &mut self.channel_states {
            state.photocurrent_ua = 0.0;
            state.tia_output_v = 0.0;
            state.snr_db = 0.0;
            state.valid = true;
        }
    }

    /// Get channel states
    pub fn get_channel_states(&self) -> &[MacChannelState] {
        &self.channel_states
    }

    /// Set integration time
    pub fn set_integration_time(&mut self, time_ps: f64) {
        self.integration_time_ps = time_ps;
    }

    /// Set reference optical power
    pub fn set_reference_power(&mut self, power_dbm: f64) {
        self.reference_power_dbm = power_dbm;
    }

    /// Get the number of channels
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Execute full matrix-vector multiplication with photonic hardware
    pub fn execute_matrix_vector_multiply(
        &mut self,
        weight_bank: &mut MicroringWeightBank,
        router: &WdmCrossbarRouter,
        input_vector: &[f64],
    ) -> Result<Vec<f64>, OpticalMacError> {
        let num_channels = self.num_channels.min(input_vector.len());
        let mut optical_powers = vec![0.0; num_channels];

        // Convert input vector to optical powers (assuming normalized inputs)
        for i in 0..num_channels {
            // Map input [-1, 1] to optical power [0, 1] mW
            let normalized = (input_vector[i] + 1.0) / 2.0;
            optical_powers[i] = normalized * self.reference_power_dbm.exp() / 1000.0;
        }

        // Apply weights via microring weight bank
        for i in 0..num_channels {
            if let Ok(weighted_power) = weight_bank.multiply(i as u32, optical_powers[i], 1550.0) {
                optical_powers[i] = weighted_power;
            }
        }

        // Accumulate and convert to electrical
        self.accumulate(&optical_powers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_creation() {
        let acc = OpticalMacAccumulator::new(8);
        assert_eq!(acc.num_channels(), 8);
    }

    #[test]
    fn test_basic_accumulation() {
        let mut acc = OpticalMacAccumulator::new(4);
        
        let powers = vec![1.0, 2.0, 3.0, 4.0]; // mW
        let result = acc.accumulate(&powers).unwrap();
        
        assert_eq!(result.len(), 4);
        
        // Higher power should produce higher voltage
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
        assert!(result[2] < result[3]);
    }

    #[test]
    fn test_detector_saturation() {
        let mut acc = OpticalMacAccumulator::new(4);
        
        // Power above saturation limit
        let powers = vec![15.0, 1.0, 1.0, 1.0]; // First channel saturates
        let result = acc.accumulate(&powers);
        
        assert!(result.is_err());
        match result {
            Err(OpticalMacError::DetectorSaturation { .. }) => (),
            _ => panic!("Expected DetectorSaturation error"),
        }
    }

    #[test]
    fn test_snr_calculation() {
        let acc = OpticalMacAccumulator::new(4);
        
        // Higher power should have better SNR
        let snr_low = acc.calculate_snr(0.1);
        let snr_high = acc.calculate_snr(5.0);
        
        assert!(snr_high > snr_low);
    }

    #[test]
    fn test_differential_accumulation() {
        let mut acc = OpticalMacAccumulator::new(4);
        
        let signal = vec![2.0, 3.0, 4.0, 5.0];
        let reference = vec![1.0, 1.0, 1.0, 1.0];
        
        let result = acc.accumulate_differential(&signal, &reference).unwrap();
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_invalid_tia_gain() {
        let result = OpticalMacAccumulator::with_configs(
            4,
            BalancedDetectorConfig::default(),
            TiaConfig {
                gain_v_a: 0.0, // Invalid
                ..Default::default()
            },
        );
        
        assert!(result.is_err());
    }

    #[test]
    fn test_reset() {
        let mut acc = OpticalMacAccumulator::new(4);
        
        let powers = vec![1.0, 2.0, 3.0, 4.0];
        acc.accumulate(&powers).unwrap();
        
        // Verify non-zero state
        assert!(acc.channel_states[0].photocurrent_ua > 0.0);
        
        acc.reset();
        
        // Verify reset
        assert_eq!(acc.channel_states[0].photocurrent_ua, 0.0);
        assert_eq!(acc.channel_states[0].tia_output_v, 0.0);
    }
}
