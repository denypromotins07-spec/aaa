//! MZI Mesh Compiler - Compiles neural network weights to photonic hardware configurations
//!
//! This module takes the output from Clements decomposition and compiles it into
//! hardware-specific configurations for silicon photonic chips, including:
//! - DAC voltage mappings for thermo-optic phase shifters
//! - Calibration data for manufacturing variations
//! - Thermal crosstalk compensation matrices

use crate::onn::clements_decomposition::{ClementsDecomposer, ClementsConfig, MziMeshConfig, MziElement};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use std::sync::Arc;

/// Errors that can occur during MZI mesh compilation
#[derive(Error, Debug)]
pub enum MziCompilerError {
    #[error("Hardware configuration mismatch: expected={expected}, got={got}")]
    HardwareMismatch { expected: String, got: String },
    
    #[error("Phase shifter calibration failed at MZI row={row}, col={col}: {reason}")]
    CalibrationFailure { row: usize, col: usize, reason: String },
    
    #[error("Thermal crosstalk compensation diverged after {iterations} iterations")]
    ThermalCrosstalkDivergence { iterations: usize },
    
    #[error("DAC voltage out of range: requested={requested}V, max={max}V")]
    DacVoltageOutOfRange { requested: f64, max: f64 },
    
    #[error("Mesh dimension {dim} incompatible with hardware topology {topology}")]
    TopologyIncompatibility { dim: usize, topology: String },
}

/// Hardware-specific configuration for a photonic chip
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicHardwareConfig {
    /// Number of input/output ports
    pub mesh_dimension: usize,
    /// DAC resolution in bits
    pub dac_resolution_bits: u8,
    /// Maximum DAC output voltage (Volts)
    pub dac_max_voltage: f64,
    /// Minimum DAC output voltage (Volts)
    pub dac_min_voltage: f64,
    /// Phase shift per volt (radians/V) - device specific
    pub phase_shift_per_volt: f64,
    /// Thermal crosstalk coefficient between adjacent MZIs
    pub thermal_crosstalk_coefficient: f64,
    /// Operating temperature (Celsius)
    pub operating_temperature: f64,
    /// Temperature coefficient of phase (radians/°C)
    pub temp_coefficient: f64,
}

impl Default for PhotonicHardwareConfig {
    fn default() -> Self {
        Self {
            mesh_dimension: 64,
            dac_resolution_bits: 16,
            dac_max_voltage: 5.0,
            dac_min_voltage: 0.0,
            phase_shift_per_volt: std::f64::consts::PI, // π radians per volt typical
            thermal_crosstalk_coefficient: 0.02, // 2% crosstalk typical
            operating_temperature: 25.0,
            temp_coefficient: 0.001, // ~1 mrad/°C typical
        }
    }
}

/// Compiled configuration ready for hardware programming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledMziConfig {
    /// Original mesh configuration from Clements decomposition
    pub mesh_config: MziMeshConfig,
    /// DAC codes for each MZI theta parameter
    pub theta_dac_codes: Vec<u32>,
    /// DAC codes for each MZI phi parameter
    pub phi_dac_codes: Vec<u32>,
    /// DAC codes for output phase shifters
    pub output_phase_dac_codes: Vec<u32>,
    /// Thermal crosstalk compensation matrix
    pub crosstalk_compensation: Vec<f64>,
    /// Calibration metadata
    pub calibration_timestamp: u64,
}

/// MZI Mesh Compiler - transforms mathematical decompositions to hardware configurations
pub struct MziMeshCompiler {
    hardware_config: PhotonicHardwareConfig,
    clements_decomposer: ClementsDecomposer,
    /// Lookup table for phase-to-voltage conversion (calibrated)
    phase_to_voltage_lut: Vec<(f64, f64)>,
}

impl MziMeshCompiler {
    /// Create a new MZI mesh compiler with default hardware configuration
    pub fn new() -> Self {
        Self {
            hardware_config: PhotonicHardwareConfig::default(),
            clements_decomposer: ClementsDecomposer::new(),
            phase_to_voltage_lut: Self::build_phase_voltage_lut(
                PhotonicHardwareConfig::default().phase_shift_per_volt,
                PhotonicHardwareConfig::default().dac_max_voltage,
                PhotonicHardwareConfig::default().dac_min_voltage,
            ),
        }
    }

    /// Create a new MZI mesh compiler with custom hardware configuration
    pub fn with_hardware_config(config: PhotonicHardwareConfig) -> Result<Self, MziCompilerError> {
        // Validate hardware configuration
        if config.dac_max_voltage <= config.dac_min_voltage {
            return Err(MziCompilerError::HardwareMismatch {
                expected: "dac_max_voltage > dac_min_voltage".to_string(),
                got: format!("max={}V, min={}V", config.dac_max_voltage, config.dac_min_voltage),
            });
        }

        if config.phase_shift_per_volt <= 0.0 {
            return Err(MziCompilerError::HardwareMismatch {
                expected: "positive phase_shift_per_volt".to_string(),
                got: format!("{} rad/V", config.phase_shift_per_volt),
            });
        }

        let lut = Self::build_phase_voltage_lut(
            config.phase_shift_per_volt,
            config.dac_max_voltage,
            config.dac_min_voltage,
        );

        Ok(Self {
            hardware_config: config,
            clements_decomposer: ClementsDecomposer::new(),
            phase_to_voltage_lut: lut,
        })
    }

    /// Build a lookup table for phase-to-voltage conversion
    fn build_phase_voltage_lut(phase_per_volt: f64, max_voltage: f64, min_voltage: f64) -> Vec<(f64, f64)> {
        const LUT_SIZE: usize = 1024;
        let mut lut = Vec::with_capacity(LUT_SIZE);
        
        let voltage_range = max_voltage - min_voltage;
        let max_phase = phase_per_volt * voltage_range;
        
        for i in 0..LUT_SIZE {
            let phase = (i as f64 / LUT_SIZE as f64) * max_phase;
            let voltage = min_voltage + (phase / phase_per_volt).clamp(min_voltage, max_voltage);
            lut.push((phase, voltage));
        }
        
        lut
    }

    /// Compile a weight matrix directly to hardware configuration
    /// 
    /// This is the main entry point - takes neural network weights and produces
    /// the complete hardware programming sequence.
    pub fn compile_weights(
        &self,
        weights: &nalgebra::MatrixXcd,
    ) -> Result<CompiledMziConfig, MziCompilerError> {
        use nalgebra::Dyn;
        
        // Perform Clements decomposition
        let mesh_config = self.clements_decomposer
            .decompose(weights)
            .map_err(|e| MziCompilerError::CalibrationFailure {
                row: 0,
                col: 0,
                reason: format!("Clements decomposition failed: {}", e),
            })?;

        // Convert to hardware configuration
        self.compile_mesh_config(&mesh_config)
    }

    /// Compile an existing MZI mesh configuration to hardware-specific settings
    pub fn compile_mesh_config(
        &self,
        mesh_config: &MziMeshConfig,
    ) -> Result<CompiledMziConfig, MziCompilerError> {
        // Validate mesh dimension matches hardware
        if mesh_config.dimension != self.hardware_config.mesh_dimension {
            return Err(MziCompilerError::TopologyIncompatibility {
                dim: mesh_config.dimension,
                topology: format!("{}×{}", self.hardware_config.mesh_dimension, self.hardware_config.mesh_dimension),
            });
        }

        // Convert MZI parameters to DAC codes
        let theta_dac_codes: Vec<u32> = mesh_config.mzi_elements
            .iter()
            .map(|mzi| self.phase_to_dac_code(mzi.theta))
            .collect::<Result<Vec<_>, _>>()?;

        let phi_dac_codes: Vec<u32> = mesh_config.mzi_elements
            .iter()
            .map(|mzi| self.phase_to_dac_code(mzi.phi))
            .collect::<Result<Vec<_>, _>>()?;

        let output_phase_dac_codes: Vec<u32> = mesh_config.output_phases
            .iter()
            .map(|&phase| self.phase_to_dac_code(phase))
            .collect::<Result<Vec<_>, _>>()?;

        // Calculate thermal crosstalk compensation
        let crosstalk_compensation = self.calculate_crosstalk_compensation(&mesh_config.mzi_elements);

        Ok(CompiledMziConfig {
            mesh_config: mesh_config.clone(),
            theta_dac_codes,
            phi_dac_codes,
            output_phase_dac_codes,
            crosstalk_compensation,
            calibration_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }

    /// Convert a phase value (radians) to a DAC code
    fn phase_to_dac_code(&self, phase: f64) -> Result<u32, MziCompilerError> {
        // Normalize phase to [0, 2π)
        let normalized_phase = phase.rem_euclid(2.0 * std::f64::consts::PI);
        
        // Calculate required voltage
        let voltage = self.lookup_voltage_for_phase(normalized_phase);
        
        // Validate voltage range
        if voltage < self.hardware_config.dac_min_voltage 
            || voltage > self.hardware_config.dac_max_voltage {
            return Err(MziCompilerError::DacVoltageOutOfRange {
                requested: voltage,
                max: self.hardware_config.dac_max_voltage,
            });
        }

        // Convert voltage to DAC code
        let dac_range = (1u32 << self.hardware_config.dac_resolution_bits) - 1;
        let voltage_range = self.hardware_config.dac_max_voltage - self.hardware_config.dac_min_voltage;
        let dac_code = ((voltage - self.hardware_config.dac_min_voltage) / voltage_range * dac_range as f64).round() as u32;
        
        Ok(dac_code.clamp(0, dac_range))
    }

    /// Look up voltage for a given phase using the calibrated LUT
    fn lookup_voltage_for_phase(&self, phase: f64) -> f64 {
        if self.phase_to_voltage_lut.is_empty() {
            // Fallback to direct calculation
            return self.hardware_config.dac_min_voltage 
                + phase / self.hardware_config.phase_shift_per_volt;
        }

        // Binary search in LUT - use proper comparison that handles NaN
        let idx = match self.phase_to_voltage_lut.binary_search_by(|(p, _)| {
            p.partial_cmp(&phase).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(i) => i,
            Err(i) => {
                if i == 0 {
                    return self.phase_to_voltage_lut[0].1;
                }
                if i >= self.phase_to_voltage_lut.len() {
                    return self.phase_to_voltage_lut.last()
                        .map(|(_, v)| *v)
                        .unwrap_or(self.hardware_config.dac_min_voltage);
                }
                // Linear interpolation between adjacent entries
                let (p0, v0) = self.phase_to_voltage_lut[i - 1];
                let (p1, v1) = self.phase_to_voltage_lut[i];
                let denom = p1 - p0;
                if denom.abs() < 1e-12 {
                    return v0;
                }
                let t = (phase - p0) / denom;
                return v0 + t * (v1 - v0);
            }
        };

        self.phase_to_voltage_lut[idx].1
    }

    /// Calculate thermal crosstalk compensation values
    /// 
    /// When an MZI heater is activated, it heats neighboring waveguides,
    /// causing unintended phase shifts. This calculates compensation factors.
    fn calculate_crosstalk_compensation(&self, mzi_elements: &[MziElement]) -> Vec<f64> {
        let n = mzi_elements.len();
        let mut compensation = vec![0.0; n];
        let crosstalk_coef = self.hardware_config.thermal_crosstalk_coefficient;

        for (i, mzi) in mzi_elements.iter().enumerate() {
            // Calculate cumulative crosstalk from all other MZIs
            let mut crosstalk_sum = 0.0;
            
            for (j, other_mzi) in mzi_elements.iter().enumerate() {
                if i != j {
                    // Distance-based crosstalk model
                    let row_dist = (mzi.row as i32 - other_mzi.row as i32).abs();
                    let col_dist = (mzi.col as i32 - other_mzi.col as i32).abs();
                    let distance = (row_dist.pow(2) + col_dist.pow(2)) as f64;
                    
                    // Exponential decay with distance
                    let crosstalk = crosstalk_coef * (-distance / 4.0).exp();
                    crosstalk_sum += crosstalk * other_mzi.theta;
                }
            }

            compensation[i] = crosstalk_sum;
        }

        compensation
    }

    /// Apply crosstalk compensation to DAC codes
    pub fn apply_crosstalk_compensation(
        &self,
        compiled_config: &mut CompiledMziConfig,
        iterations: usize,
    ) -> Result<(), MziCompilerError> {
        let max_iterations = 100;
        let actual_iterations = iterations.min(max_iterations);

        for _ in 0..actual_iterations {
            let mut converged = true;
            let original_thetas = compiled_config.theta_dac_codes.clone();

            for (i, compensation) in compiled_config.crosstalk_compensation.iter().enumerate() {
                // Adjust DAC code based on crosstalk
                let adjustment = (*compensation / self.hardware_config.phase_shift_per_volt) 
                    * ((1u32 << self.hardware_config.dac_resolution_bits) - 1) as f64
                    / (self.hardware_config.dac_max_voltage - self.hardware_config.dac_min_voltage);
                
                let adjusted_code = original_thetas[i] as f64 - adjustment;
                let new_code = adjusted_code.round() as u32;

                if (new_code as i32 - original_thetas[i] as i32).abs() > 1 {
                    converged = false;
                }

                compiled_config.theta_dac_codes[i] = new_code.clamp(
                    0,
                    (1u32 << self.hardware_config.dac_resolution_bits) - 1,
                );
            }

            if converged {
                return Ok(());
            }
        }

        Err(MziCompilerError::ThermalCrosstalkDivergence {
            iterations: actual_iterations,
        })
    }

    /// Get the hardware configuration
    pub fn hardware_config(&self) -> &PhotonicHardwareConfig {
        &self.hardware_config
    }

    /// Update hardware configuration (requires recalibration)
    pub fn update_hardware_config(&mut self, config: PhotonicHardwareConfig) -> Result<(), MziCompilerError> {
        self.hardware_config = config;
        self.phase_to_voltage_lut = Self::build_phase_voltage_lut(
            self.hardware_config.phase_shift_per_volt,
            self.hardware_config.dac_max_voltage,
            self.hardware_config.dac_min_voltage,
        );
        Ok(())
    }
}

impl Default for MziMeshCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Complex, SMatrix};
    use num_complex::Complex64;

    #[test]
    fn test_compiler_creation() {
        let compiler = MziMeshCompiler::new();
        assert_eq!(compiler.hardware_config().mesh_dimension, 64);
        assert_eq!(compiler.hardware_config().dac_resolution_bits, 16);
    }

    #[test]
    fn test_custom_hardware_config() {
        let config = PhotonicHardwareConfig {
            mesh_dimension: 32,
            dac_resolution_bits: 14,
            dac_max_voltage: 3.3,
            dac_min_voltage: 0.0,
            phase_shift_per_volt: 2.0 * std::f64::consts::PI,
            ..Default::default()
        };

        let compiler = MziMeshCompiler::with_hardware_config(config).unwrap();
        assert_eq!(compiler.hardware_config().mesh_dimension, 32);
        assert_eq!(compiler.hardware_config().dac_max_voltage, 3.3);
    }

    #[test]
    fn test_invalid_hardware_config() {
        let config = PhotonicHardwareConfig {
            dac_max_voltage: 0.0,
            dac_min_voltage: 5.0, // Invalid: min > max
            ..Default::default()
        };

        let result = MziMeshCompiler::with_hardware_config(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_phase_to_dac_conversion() {
        let compiler = MziMeshCompiler::new();
        
        // Test 0 phase -> minimum DAC code
        let code_0 = compiler.phase_to_dac_code(0.0).unwrap();
        assert!(code_0 <= 100); // Should be near zero
        
        // Test π phase -> mid-range DAC code
        let code_pi = compiler.phase_to_dac_code(std::f64::consts::PI).unwrap();
        let mid_code = (1u32 << 15); // Half of 16-bit range
        assert!((code_pi as i32 - mid_code as i32).abs() < 1000);
    }

    #[test]
    fn test_full_compilation_pipeline() {
        use nalgebra::Dyn;
        
        let config = PhotonicHardwareConfig {
            mesh_dimension: 4,
            ..Default::default()
        };
        let compiler = MziMeshCompiler::with_hardware_config(config).unwrap();

        // Create a simple 4x4 unitary matrix
        let u: SMatrix<Complex64, 4, 4> = SMatrix::identity(4, 4);
        
        let result = compiler.compile_weights(&u);
        assert!(result.is_ok());
        
        let compiled = result.unwrap();
        assert_eq!(compiled.theta_dac_codes.len(), 4 * 3 / 2); // n(n-1)/2 MZIs
        assert_eq!(compiled.output_phase_dac_codes.len(), 4);
    }
}
