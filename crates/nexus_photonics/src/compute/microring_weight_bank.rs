//! Microring Resonator Weight Bank Simulator
//!
//! This module simulates microring resonator (MRR) weight banks used in photonic
//! matrix multiplication. Each MRR acts as a tunable optical filter that attenuates
//! specific wavelengths, performing the multiply-accumulate (MAC) operation via
//! optical interference in O(1) time.
//!
//! Key physics modeled:
//! - Lorentzian resonance lineshape
//! - Thermo-optic tuning coefficient
//! - Free spectral range (FSR)
//! - Quality factor (Q) and insertion loss
//! - Optical crosstalk between adjacent rings

use serde::{Deserialize, Serialize};
use thiserror::Error;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Errors that can occur in microring operations
#[derive(Error, Debug)]
pub enum MicroringError {
    #[error("Wavelength {wavelength}nm outside operational range [{min}, {max}]nm")]
    WavelengthOutOfRange { wavelength: f64, min: f64, max: f64 },
    
    #[error("Resonance detuning exceeded: detuning={detuning}pm, max={max_pm}pm")]
    ResonanceDetuningExceeded { detuning: f64, max_pm: f64 },
    
    #[error("Thermal runaway detected at ring {ring_id}: temperature={temp}°C")]
    ThermalRunaway { ring_id: u32, temp: f64 },
    
    #[error("Crosstalk limit exceeded: measured={measured}dB, threshold={threshold}dB")]
    CrosstalkLimitExceeded { measured: f64, threshold: f64 },
    
    #[error("Ring {ring_id} coupling coefficient out of bounds: {coeff}")]
    InvalidCouplingCoefficient { ring_id: u32, coeff: f64 },
}

/// Configuration for a single microring resonator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroringConfig {
    /// Unique identifier for this ring
    pub ring_id: u32,
    /// Resonant wavelength at reference temperature (nanometers)
    pub resonant_wavelength_nm: f64,
    /// Ring radius (micrometers)
    pub radius_um: f64,
    /// Coupling coefficient (0 to 1)
    pub coupling_coefficient: f64,
    /// Round-trip loss (dB/cm)
    pub propagation_loss_db_cm: f64,
    /// Thermo-optic coefficient (pm/°C)
    pub thermo_optic_coefficient: f64,
    /// Operating group index
    pub group_index: f64,
}

impl Default for MicroringConfig {
    fn default() -> Self {
        Self {
            ring_id: 0,
            resonant_wavelength_nm: 1550.0, // C-band standard
            radius_um: 5.0,
            coupling_coefficient: 0.1,
            propagation_loss_db_cm: 2.0,
            thermo_optic_coefficient: 80.0, // ~80 pm/°C for silicon
            group_index: 4.0,
        }
    }
}

/// State of a single microring resonator
#[derive(Debug, Clone)]
pub struct MicroringState {
    /// Current temperature offset from reference (°C)
    pub temperature_offset: f64,
    /// Current applied heater voltage (V)
    pub heater_voltage: f64,
    /// Current resonance wavelength shift (pm)
    pub wavelength_shift_pm: f64,
    /// Calculated quality factor
    pub quality_factor: f64,
}

/// Complete microring weight bank with multiple rings
pub struct MicroringWeightBank {
    /// Configuration for each ring
    rings: Vec<MicroringConfig>,
    /// Current state of each ring
    states: Vec<MicroringState>,
    /// Reference temperature (°C)
    reference_temperature: f64,
    /// Maximum allowable temperature rise (°C)
    max_temperature_rise: f64,
    /// WDM channel spacing (nm)
    channel_spacing_nm: f64,
    /// Minimum channel isolation (dB)
    min_channel_isolation_db: f64,
}

impl MicroringWeightBank {
    /// Create a new microring weight bank with specified number of rings
    pub fn new(num_rings: usize, base_wavelength_nm: f64, channel_spacing_nm: f64) -> Self {
        let mut rings = Vec::with_capacity(num_rings);
        let mut states = Vec::with_capacity(num_rings);

        for i in 0..num_rings {
            let wavelength = base_wavelength_nm + (i as f64) * channel_spacing_nm;
            
            rings.push(MicroringConfig {
                ring_id: i as u32,
                resonant_wavelength_nm: wavelength,
                ..Default::default()
            });

            states.push(MicroringState {
                temperature_offset: 0.0,
                heater_voltage: 0.0,
                wavelength_shift_pm: 0.0,
                quality_factor: 10000.0, // Typical Q for silicon MRR
            });
        }

        Self {
            rings,
            states,
            reference_temperature: 25.0,
            max_temperature_rise: 50.0,
            channel_spacing_nm,
            min_channel_isolation_db: 20.0,
        }
    }

    /// Create from explicit configurations
    pub fn from_configs(configs: Vec<MicroringConfig>) -> Result<Self, MicroringError> {
        // Validate all coupling coefficients
        for config in &configs {
            if config.coupling_coefficient < 0.0 || config.coupling_coefficient > 1.0 {
                return Err(MicroringError::InvalidCouplingCoefficient {
                    ring_id: config.ring_id,
                    coeff: config.coupling_coefficient,
                });
            }
        }

        let num_rings = configs.len();
        let mut states = Vec::with_capacity(num_rings);

        for _ in 0..num_rings {
            states.push(MicroringState {
                temperature_offset: 0.0,
                heater_voltage: 0.0,
                wavelength_shift_pm: 0.0,
                quality_factor: 10000.0,
            });
        }

        // Determine channel spacing from configurations
        let channel_spacing_nm = if num_rings > 1 {
            configs[1].resonant_wavelength_nm - configs[0].resonant_wavelength_nm
        } else {
            0.8 // Default 800pm spacing
        };

        Ok(Self {
            rings: configs,
            states,
            reference_temperature: 25.0,
            max_temperature_rise: 50.0,
            channel_spacing_nm,
            min_channel_isolation_db: 20.0,
        })
    }

    /// Calculate the transmission through a microring at a given wavelength
    /// 
    /// Uses the Lorentzian lineshape model:
    /// T(λ) = 1 - (κ² / (1 + (2(λ - λ₀)/Δλ)²))
    /// where κ is coupling coefficient, λ₀ is resonant wavelength, Δλ is linewidth
    pub fn calculate_transmission(&self, ring_id: u32, wavelength_nm: f64) -> Result<f64, MicroringError> {
        let ring = self.rings.iter().find(|r| r.ring_id == ring_id)
            .ok_or_else(|| MicroringError::WavelengthOutOfRange {
                wavelength: wavelength_nm,
                min: 0.0,
                max: 0.0,
            })?;

        let state = &self.states[ring_id as usize];
        
        // Calculate current resonant wavelength including thermal shift
        let current_resonance = ring.resonant_wavelength_nm + state.wavelength_shift_pm / 1000.0;

        // Validate wavelength is in operational range
        let fsr = self.calculate_fsr(ring_id)?;
        let operational_range = fsr / 2.0;
        
        if (wavelength_nm - current_resonance).abs() > operational_range {
            return Err(MicroringError::WavelengthOutOfRange {
                wavelength: wavelength_nm,
                min: current_resonance - operational_range,
                max: current_resonance + operational_range,
            });
        }

        // Calculate quality factor from coupling and loss
        let q_factor = self.calculate_quality_factor(ring);
        
        // Linewidth (FWHM) = λ₀ / Q
        let linewidth = current_resonance / q_factor;

        // Lorentzian transmission
        let detuning = wavelength_nm - current_resonance;
        let normalized_detuning = 2.0 * detuning / linewidth;
        
        let kappa_sq = ring.coupling_coefficient.powi(2);
        let transmission = 1.0 - (kappa_sq / (1.0 + normalized_detuning.powi(2)));

        // Apply propagation loss
        let loss_factor = 10.0_f64.powf(-ring.propagation_loss_db_cm * 2.0 * PI * ring.radius_um / 1e4 / 10.0);
        
        Ok(transmission * loss_factor)
    }

    /// Calculate free spectral range (FSR) for a ring
    pub fn calculate_fsr(&self, ring_id: u32) -> Result<f64, MicroringError> {
        let ring = self.rings.iter().find(|r| r.ring_id == ring_id)
            .ok_or_else(|| MicroringError::WavelengthOutOfRange {
                wavelength: 0.0,
                min: 0.0,
                max: 0.0,
            })?;

        // FSR = λ² / (n_g * L)
        // where L = 2πR is the ring circumference
        let wavelength_m = ring.resonant_wavelength_nm * 1e-9;
        let circumference_m = 2.0 * PI * ring.radius_um * 1e-6;
        
        let fsr_m = wavelength_m.powi(2) / (ring.group_index * circumference_m);
        let fsr_nm = fsr_m * 1e9;

        Ok(fsr_nm)
    }

    /// Calculate quality factor for a ring
    fn calculate_quality_factor(&self, ring: &MicroringConfig) -> f64 {
        // Q = (π * n_g * R) / (κ² * λ)
        // Simplified model including loss
        
        let round_trip_length_cm = 2.0 * PI * ring.radius_um * 1e-4;
        let loss_per_roundtrip = 10.0_f64.powf(-ring.propagation_loss_db_cm * round_trip_length_cm / 10.0);
        
        let total_loss = 1.0 - loss_per_roundtrip * (1.0 - ring.coupling_coefficient.powi(2));
        
        if total_loss <= 0.0 {
            return 100000.0; // Upper bound
        }

        let q = 2.0 * PI * ring.group_index * ring.radius_um * 1e-6 
            / (ring.resonant_wavelength_nm * 1e-9 * total_loss);
        
        q.clamp(1000.0, 100000.0)
    }

    /// Set the temperature offset for a specific ring (simulates heater control)
    pub fn set_ring_temperature(&mut self, ring_id: u32, temperature_offset: f64) -> Result<(), MicroringError> {
        if temperature_offset.abs() > self.max_temperature_rise {
            return Err(MicroringError::ThermalRunaway {
                ring_id,
                temp: temperature_offset,
            });
        }

        let ring = &self.rings[ring_id as usize];
        let state = &mut self.states[ring_id as usize];

        // Calculate wavelength shift from temperature
        // Δλ = thermo_optic_coefficient * ΔT
        state.wavelength_shift_pm = ring.thermo_optic_coefficient * temperature_offset;
        state.temperature_offset = temperature_offset;

        // Update quality factor based on temperature
        state.quality_factor = self.calculate_quality_factor(ring) * (1.0 - 0.001 * temperature_offset.abs());

        Ok(())
    }

    /// Set weight by adjusting ring resonance (wavelength-selective attenuation)
    pub fn set_weight(&mut self, ring_id: u32, target_transmission: f64) -> Result<(), MicroringError> {
        if target_transmission < 0.0 || target_transmission > 1.0 {
            return Err(MicroringError::WavelengthOutOfRange {
                wavelength: target_transmission,
                min: 0.0,
                max: 1.0,
            });
        }

        let ring = &self.rings[ring_id as usize];
        let state = &mut self.states[ring_id as usize];

        // Find temperature offset that achieves target transmission
        // Using Newton-Raphson iteration
        let mut temp_offset = 0.0;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 50;
        const TOLERANCE: f64 = 1e-6;

        while iterations < MAX_ITERATIONS {
            state.wavelength_shift_pm = ring.thermo_optic_coefficient * temp_offset;
            
            let transmission = self.calculate_transmission(ring_id, ring.resonant_wavelength_nm)?;
            let error = transmission - target_transmission;

            if error.abs() < TOLERANCE {
                break;
            }

            // Numerical derivative
            let delta = 0.1;
            state.wavelength_shift_pm = ring.thermo_optic_coefficient * (temp_offset + delta);
            let transmission_plus = self.calculate_transmission(ring_id, ring.resonant_wavelength_nm)?;
            
            let derivative = (transmission_plus - transmission) / delta;
            
            if derivative.abs() < 1e-10 {
                break; // Avoid division by zero
            }

            temp_offset -= error / derivative;
            iterations += 1;
        }

        // Check for thermal runaway
        if temp_offset.abs() > self.max_temperature_rise {
            return Err(MicroringError::ThermalRunaway {
                ring_id,
                temp: temp_offset,
            });
        }

        state.temperature_offset = temp_offset;
        state.wavelength_shift_pm = ring.thermo_optic_coefficient * temp_offset;

        Ok(())
    }

    /// Perform weighted multiplication on input optical power
    /// 
    /// Simulates the MAC operation: P_out = T(λ) * P_in
    /// where T(λ) is the wavelength-dependent transmission
    pub fn multiply(&self, ring_id: u32, input_power_mw: f64, wavelength_nm: f64) -> Result<f64, MicroringError> {
        let transmission = self.calculate_transmission(ring_id, wavelength_nm)?;
        Ok(transmission * input_power_mw)
    }

    /// Calculate crosstalk between adjacent channels
    pub fn calculate_crosstalk(&self, ring_id: u32, adjacent_ring_id: u32) -> Result<f64, MicroringError> {
        let ring = self.rings.iter().find(|r| r.ring_id == ring_id)
            .ok_or_else(|| MicroringError::WavelengthOutOfRange {
                wavelength: 0.0,
                min: 0.0,
                max: 0.0,
            })?;

        let adjacent_ring = self.rings.iter().find(|r| r.ring_id == adjacent_ring_id)
            .ok_or_else(|| MicroringError::WavelengthOutOfRange {
                wavelength: 0.0,
                min: 0.0,
                max: 0.0,
            })?;

        // Calculate transmission at adjacent ring's wavelength through this ring
        let unwanted_transmission = self.calculate_transmission(ring_id, adjacent_ring.resonant_wavelength_nm)?;
        let desired_transmission = self.calculate_transmission(ring_id, ring.resonant_wavelength_nm)?;

        // Crosstalk in dB = 10 * log10(unwanted / desired)
        if desired_transmission < 1e-12 {
            return Ok(-100.0); // Effectively infinite isolation
        }

        let crosstalk_db = 10.0 * (unwanted_transmission / desired_transmission).log10();
        Ok(crosstalk_db)
    }

    /// Verify all channels meet minimum isolation requirements
    pub fn verify_channel_isolation(&self) -> Result<(), MicroringError> {
        for i in 0..self.rings.len() {
            for j in 0..self.rings.len() {
                if i != j {
                    let crosstalk = self.calculate_crosstalk(self.rings[i].ring_id, self.rings[j].ring_id)?;
                    
                    if crosstalk > -self.min_channel_isolation_db {
                        return Err(MicroringError::CrosstalkLimitExceeded {
                            measured: crosstalk,
                            threshold: -self.min_channel_isolation_db,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Get the current state of a ring
    pub fn get_ring_state(&self, ring_id: u32) -> Option<&MicroringState> {
        self.states.get(ring_id as usize)
    }

    /// Get all ring configurations
    pub fn rings(&self) -> &[MicroringConfig] {
        &self.rings
    }

    /// Set the reference temperature
    pub fn set_reference_temperature(&mut self, temp: f64) {
        self.reference_temperature = temp;
    }

    /// Get the number of rings in the weight bank
    pub fn num_rings(&self) -> usize {
        self.rings.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_bank_creation() {
        let bank = MicroringWeightBank::new(8, 1550.0, 0.8);
        assert_eq!(bank.num_rings(), 8);
    }

    #[test]
    fn test_transmission_on_resonance() {
        let mut bank = MicroringWeightBank::new(4, 1550.0, 0.8);
        
        // On resonance, transmission should be minimum (high attenuation)
        let transmission = bank.calculate_transmission(0, 1550.0).unwrap();
        assert!(transmission < 1.0);
        assert!(transmission >= 0.0);
    }

    #[test]
    fn test_transmission_off_resonance() {
        let bank = MicroringWeightBank::new(4, 1550.0, 0.8);
        
        // Far from resonance, transmission should approach 1.0
        let transmission = bank.calculate_transmission(0, 1552.0).unwrap();
        assert!(transmission > 0.9);
    }

    #[test]
    fn test_thermal_tuning() {
        let mut bank = MicroringWeightBank::new(4, 1550.0, 0.8);
        
        // Apply thermal tuning
        bank.set_ring_temperature(0, 10.0).unwrap();
        
        let state = bank.get_ring_state(0).unwrap();
        assert!((state.temperature_offset - 10.0).abs() < 0.01);
        assert!(state.wavelength_shift_pm > 0.0);
    }

    #[test]
    fn test_thermal_runaway_protection() {
        let mut bank = MicroringWeightBank::new(4, 1550.0, 0.8);
        
        // Should reject excessive temperature
        let result = bank.set_ring_temperature(0, 100.0);
        assert!(result.is_err());
        match result {
            Err(MicroringError::ThermalRunaway { .. }) => (),
            _ => panic!("Expected ThermalRunaway error"),
        }
    }

    #[test]
    fn test_crosstalk_calculation() {
        let bank = MicroringWeightBank::new(4, 1550.0, 0.8);
        
        // Adjacent channels should have significant crosstalk
        let crosstalk = bank.calculate_crosstalk(0, 1).unwrap();
        assert!(crosstalk < 0.0); // Should be negative (attenuation)
        
        // Non-adjacent should have less crosstalk
        let crosstalk_far = bank.calculate_crosstalk(0, 3).unwrap();
        assert!(crosstalk_far < crosstalk);
    }

    #[test]
    fn test_multiply_operation() {
        let bank = MicroringWeightBank::new(4, 1550.0, 0.8);
        
        let input_power = 1.0; // 1 mW
        let output = bank.multiply(0, input_power, 1550.0).unwrap();
        
        assert!(output <= input_power);
        assert!(output >= 0.0);
    }

    #[test]
    fn test_invalid_coupling_coefficient() {
        let configs = vec![
            MicroringConfig {
                coupling_coefficient: 1.5, // Invalid: > 1.0
                ..Default::default()
            },
        ];

        let result = MicroringWeightBank::from_configs(configs);
        assert!(result.is_err());
    }
}
