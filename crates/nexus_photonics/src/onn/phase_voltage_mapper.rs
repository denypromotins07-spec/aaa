//! Phase-to-Voltage Mapper for Thermo-Optic Phase Shifters
//!
//! This module translates decomposed MZI phase angles into precise DAC voltages
//! required to drive thermo-optic phase shifters on silicon photonic chips.
//! It handles:
//! - Non-linear phase-voltage characteristics
//! - Temperature-dependent calibration
//! - Hysteresis compensation
//! - Multi-point calibration curves

use serde::{Deserialize, Serialize};
use thiserror::Error;
use std::collections::HashMap;

/// Errors that can occur during phase-to-voltage mapping
#[derive(Error, Debug)]
pub enum PhaseVoltageError {
    #[error("Calibration point out of range: phase={phase} rad, valid_range=[{min}, {max}]")]
    CalibrationOutOfRange { phase: f64, min: f64, max: f64 },
    
    #[error("Insufficient calibration points: got={got}, minimum={minimum}")]
    InsufficientCalibrationPoints { got: usize, minimum: usize },
    
    #[error("Phase value {phase} exceeds calibrated maximum {max_phase}")]
    PhaseExceedsCalibration { phase: f64, max_phase: f64 },
    
    #[error("Temperature {temp}°C outside operating range [{min}, {max}]°C")]
    TemperatureOutOfRange { temp: f64, min: f64, max: f64 },
    
    #[error("Hysteresis correction failed: direction={direction}")]
    HysteresisCorrectionFailed { direction: String },
}

/// A single calibration point (phase, voltage) pair
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CalibrationPoint {
    /// Target phase shift in radians
    pub phase_rad: f64,
    /// Measured voltage required to achieve this phase
    pub voltage_v: f64,
    /// Temperature at which this calibration was taken
    pub temperature_c: f64,
}

/// Complete calibration curve for a phase shifter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseShifterCalibration {
    /// Unique identifier for this phase shifter
    pub shifter_id: u32,
    /// Ordered calibration points (sorted by phase)
    pub points: Vec<CalibrationPoint>,
    /// Reference temperature for this calibration
    pub reference_temperature: f64,
    /// Temperature coefficient (rad/V/°C)
    pub temp_coefficient: f64,
    /// Hysteresis parameters
    pub hysteresis: HysteresisParams,
}

/// Hysteresis parameters for phase shifter calibration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HysteresisParams {
    /// Maximum hysteresis width in volts
    pub hysteresis_width_v: f64,
    /// Direction-dependent offset when increasing voltage
    pub increasing_offset_v: f64,
    /// Direction-dependent offset when decreasing voltage
    pub decreasing_offset_v: f64,
}

impl Default for HysteresisParams {
    fn default() -> Self {
        Self {
            hysteresis_width_v: 0.01, // 10 mV typical
            increasing_offset_v: 0.0,
            decreasing_offset_v: -0.01,
        }
    }
}

/// Operating mode for the phase shifter
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShifterMode {
    /// Normal operation with full calibration
    Normal,
    /// Low-power mode with reduced accuracy
    LowPower,
    /// High-speed mode ignoring hysteresis
    HighSpeed,
    /// Safe mode - minimum power, no phase shift
    Safe,
}

/// Voltage mapper state including hysteresis tracking
#[derive(Debug, Clone)]
pub struct VoltageMapperState {
    /// Last commanded voltage for each shifter
    pub last_voltage: HashMap<u32, f64>,
    /// Current voltage direction (true = increasing, false = decreasing)
    pub voltage_direction: HashMap<u32, bool>,
    /// Current operating mode
    pub mode: ShifterMode,
    /// Current chip temperature
    pub chip_temperature: f64,
}

impl Default for VoltageMapperState {
    fn default() -> Self {
        Self {
            last_voltage: HashMap::new(),
            voltage_direction: HashMap::new(),
            mode: ShifterMode::Normal,
            chip_temperature: 25.0,
        }
    }
}

/// Phase-to-Voltage Mapper - converts phase targets to DAC voltages
pub struct PhaseVoltageMapper {
    /// Calibrations for each phase shifter
    calibrations: HashMap<u32, PhaseShifterCalibration>,
    /// Current mapper state
    state: VoltageMapperState,
    /// Valid temperature operating range
    temp_min: f64,
    temp_max: f64,
    /// Spline interpolation coefficients (cached for performance)
    spline_cache: HashMap<u32, Vec<SplineCoefficients>>,
}

/// Cubic spline coefficients for smooth interpolation
#[derive(Debug, Clone)]
struct SplineCoefficients {
    /// Coefficients for each interval
    pub a: Vec<f64>, // Constant term
    pub b: Vec<f64>, // Linear term
    pub c: Vec<f64>, // Quadratic term
    pub d: Vec<f64>, // Cubic term
    /// X values (phases) defining intervals
    pub x: Vec<f64>,
}

impl PhaseVoltageMapper {
    /// Create a new phase-to-voltage mapper
    pub fn new() -> Self {
        Self {
            calibrations: HashMap::new(),
            state: VoltageMapperState::default(),
            temp_min: 15.0,
            temp_max: 35.0,
            spline_cache: HashMap::new(),
        }
    }

    /// Register a phase shifter calibration
    pub fn register_calibration(
        &mut self,
        calibration: PhaseShifterCalibration,
    ) -> Result<(), PhaseVoltageError> {
        // Validate calibration has sufficient points
        if calibration.points.len() < 3 {
            return Err(PhaseVoltageError::InsufficientCalibrationPoints {
                got: calibration.points.len(),
                minimum: 3,
            });
        }

        // Sort points by phase
        let mut sorted_calib = calibration.clone();
        sorted_calib.points.sort_by(|a, b| {
            a.phase_rad.partial_cmp(&b.phase_rad).unwrap()
        });

        // Validate phase range
        let min_phase = sorted_calib.points.first().unwrap().phase_rad;
        let max_phase = sorted_calib.points.last().unwrap().phase_rad;
        
        if min_phase < 0.0 || max_phase > 4.0 * std::f64::consts::PI {
            return Err(PhaseVoltageError::CalibrationOutOfRange {
                phase: if min_phase < 0.0 { min_phase } else { max_phase },
                min: 0.0,
                max: 4.0 * std::f64::consts::PI,
            });
        }

        // Build spline coefficients
        let spline = Self::build_spline(&sorted_calib.points)?;
        
        self.spline_cache.insert(calibration.shifter_id, spline);
        self.calibrations.insert(calibration.shifter_id, sorted_calib);

        Ok(())
    }

    /// Build cubic spline interpolation from calibration points
    fn build_spline(points: &[CalibrationPoint]) -> Result<Vec<SplineCoefficients>, PhaseVoltageError> {
        let n = points.len();
        if n < 3 {
            return Err(PhaseVoltageError::InsufficientCalibrationPoints {
                got: n,
                minimum: 3,
            });
        }

        let mut coeffs = Vec::with_capacity(n - 1);
        
        // Extract x (phase) and y (voltage) values
        let x: Vec<f64> = points.iter().map(|p| p.phase_rad).collect();
        let y: Vec<f64> = points.iter().map(|p| p.voltage_v).collect();

        // Natural cubic spline (second derivative = 0 at endpoints)
        let h: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
        
        // Build tridiagonal system for second derivatives
        let mut alpha: Vec<f64> = vec![0.0; n];
        for i in 1..n - 1 {
            alpha[i] = 3.0 / h[i] * (y[i + 1] - y[i]) - 3.0 / h[i - 1] * (y[i] - y[i - 1]);
        }

        // Thomas algorithm for tridiagonal system
        let mut c: Vec<f64> = vec![0.0; n];
        let mut l: Vec<f64> = vec![1.0; n];
        let mut mu: Vec<f64> = vec![0.0; n];
        let mut z: Vec<f64> = vec![0.0; n];

        for i in 1..n - 1 {
            l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        // Back substitution
        for j in (0..n - 1).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
            let b_coef = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
            let d_coef = (c[j + 1] - c[j]) / (3.0 * h[j]);

            coeffs.push(SplineCoefficients {
                a: vec![y[j]],
                b: vec![b_coef],
                c: vec![c[j]],
                d: vec![d_coef],
                x: vec![x[j], x[j + 1]],
            });
        }

        Ok(coeffs)
    }

    /// Convert a target phase to voltage for a specific shifter
    pub fn phase_to_voltage(
        &self,
        shifter_id: u32,
        target_phase: f64,
    ) -> Result<f64, PhaseVoltageError> {
        let calibration = self.calibrations.get(&shifter_id)
            .ok_or_else(|| PhaseVoltageError::CalibrationOutOfRange {
                phase: target_phase,
                min: 0.0,
                max: 0.0,
            })?;

        // Validate temperature
        if self.state.chip_temperature < self.temp_min 
            || self.state.chip_temperature > self.temp_max {
            return Err(PhaseVoltageError::TemperatureOutOfRange {
                temp: self.state.chip_temperature,
                min: self.temp_min,
                max: self.temp_max,
            });
        }

        // Normalize phase to calibrated range
        let max_phase = calibration.points.last().unwrap().phase_rad;
        let normalized_phase = target_phase.rem_euclid(max_phase);

        // Apply hysteresis correction if in normal mode
        let voltage = if self.state.mode == ShifterMode::HighSpeed {
            self.interpolate_voltage(shifter_id, normalized_phase)?
        } else {
            self.interpolate_with_hysteresis(shifter_id, normalized_phase)?
        };

        // Apply temperature compensation
        let compensated_voltage = self.apply_temperature_compensation(
            voltage,
            target_phase,
            calibration.temp_coefficient,
        );

        // Update state
        let mut_state = &mut self.state as *const VoltageMapperState as *mut VoltageMapperState;
        unsafe {
            (*mut_state).last_voltage.insert(shifter_id, compensated_voltage);
        }

        Ok(compensated_voltage)
    }

    /// Interpolate voltage using cached spline coefficients
    fn interpolate_voltage(&self, shifter_id: u32, phase: f64) -> Result<f64, PhaseVoltageError> {
        let spline = self.spline_cache.get(&shifter_id)
            .ok_or_else(|| PhaseVoltageError::CalibrationOutOfRange {
                phase,
                min: 0.0,
                max: 0.0,
            })?;

        // Find the correct interval
        for segment in spline {
            if phase >= segment.x[0] && phase <= segment.x[1] {
                let dx = phase - segment.x[0];
                let voltage = segment.a[0] 
                    + segment.b[0] * dx 
                    + segment.c[0] * dx.powi(2) 
                    + segment.d[0] * dx.powi(3);
                return Ok(voltage);
            }
        }

        // Extrapolation fallback (use nearest endpoint)
        if phase < spline[0].x[0] {
            return Ok(spline[0].a[0]);
        }
        if let Some(last) = spline.last() {
            if phase > last.x[1] {
                let dx = phase - last.x[0];
                return Ok(last.a[0] + last.b[0] * dx + last.c[0] * dx.powi(2) + last.d[0] * dx.powi(3));
            }
        }

        Err(PhaseVoltageError::PhaseExceedsCalibration {
            phase,
            max_phase: spline.last().map(|s| s.x[1]).unwrap_or(0.0),
        })
    }

    /// Interpolate voltage with hysteresis correction
    fn interpolate_with_hysteresis(&self, shifter_id: u32, phase: f64) -> Result<f64, PhaseVoltageError> {
        let base_voltage = self.interpolate_voltage(shifter_id, phase)?;
        
        let last_voltage = self.state.last_voltage.get(&shifter_id).copied().unwrap_or(base_voltage);
        let is_increasing = phase.to_radians() >= last_voltage;
        
        let calibration = self.calibrations.get(&shifter_id).unwrap();
        let hysteresis = calibration.hysteresis;
        
        let offset = if is_increasing {
            hysteresis.increasing_offset_v
        } else {
            hysteresis.decreasing_offset_v
        };

        // Update direction state
        let mut_state = &mut self.state as *const VoltageMapperState as *mut VoltageMapperState;
        unsafe {
            (*mut_state).voltage_direction.insert(shifter_id, is_increasing);
        }

        Ok(base_voltage + offset)
    }

    /// Apply temperature compensation to voltage
    fn apply_temperature_compensation(
        &self,
        voltage: f64,
        phase: f64,
        temp_coefficient: f64,
    ) -> f64 {
        let delta_temp = self.state.chip_temperature - 25.0; // Reference temperature
        let phase_error = temp_coefficient * delta_temp * voltage;
        
        // Compensate by adjusting voltage
        voltage + phase_error / (std::f64::consts::PI / 5.0) // Approximate Vπ
    }

    /// Set the current chip temperature
    pub fn set_chip_temperature(&mut self, temperature: f64) -> Result<(), PhaseVoltageError> {
        if temperature < self.temp_min || temperature > self.temp_max {
            return Err(PhaseVoltageError::TemperatureOutOfRange {
                temp: temperature,
                min: self.temp_min,
                max: self.temp_max,
            });
        }
        self.state.chip_temperature = temperature;
        Ok(())
    }

    /// Set the operating mode
    pub fn set_mode(&mut self, mode: ShifterMode) {
        self.state.mode = mode;
    }

    /// Get all calibrated shifter IDs
    pub fn calibrated_shifters(&self) -> Vec<u32> {
        self.calibrations.keys().copied().collect()
    }

    /// Get calibration info for a shifter
    pub fn get_calibration(&self, shifter_id: u32) -> Option<&PhaseShifterCalibration> {
        self.calibrations.get(&shifter_id)
    }

    /// Batch convert multiple phases to voltages
    pub fn batch_phase_to_voltage(
        &self,
        requests: &[(u32, f64)],
    ) -> Result<Vec<(u32, f64)>, PhaseVoltageError> {
        let mut results = Vec::with_capacity(requests.len());
        
        for &(shifter_id, phase) in requests {
            let voltage = self.phase_to_voltage(shifter_id, phase)?;
            results.push((shifter_id, voltage));
        }
        
        Ok(results)
    }

    /// Calculate the expected phase for a given voltage (inverse mapping)
    pub fn voltage_to_phase(&self, shifter_id: u32, voltage: f64) -> Result<f64, PhaseVoltageError> {
        let calibration = self.calibrations.get(&shifter_id)
            .ok_or_else(|| PhaseVoltageError::CalibrationOutOfRange {
                phase: 0.0,
                min: 0.0,
                max: 0.0,
            })?;

        // Binary search through calibration points
        let points = &calibration.points;
        for i in 0..points.len() - 1 {
            if voltage >= points[i].voltage_v && voltage <= points[i + 1].voltage_v {
                // Linear interpolation within segment
                let t = (voltage - points[i].voltage_v) 
                    / (points[i + 1].voltage_v - points[i].voltage_v);
                let phase = points[i].phase_rad + t * (points[i + 1].phase_rad - points[i].phase_rad);
                return Ok(phase);
            }
        }

        // Extrapolation
        if voltage < points[0].voltage_v {
            return Ok(points[0].phase_rad);
        }
        Ok(points.last().unwrap().phase_rad)
    }
}

impl Default for PhaseVoltageMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_calibration(id: u32) -> PhaseShifterCalibration {
        PhaseShifterCalibration {
            shifter_id: id,
            points: vec![
                CalibrationPoint { phase_rad: 0.0, voltage_v: 0.0, temperature_c: 25.0 },
                CalibrationPoint { phase_rad: std::f64::consts::PI, voltage_v: 2.5, temperature_c: 25.0 },
                CalibrationPoint { phase_rad: 2.0 * std::f64::consts::PI, voltage_v: 5.0, temperature_c: 25.0 },
            ],
            reference_temperature: 25.0,
            temp_coefficient: 0.001,
            hysteresis: HysteresisParams::default(),
        }
    }

    #[test]
    fn test_mapper_creation() {
        let mapper = PhaseVoltageMapper::new();
        assert_eq!(mapper.calibrated_shifters().len(), 0);
    }

    #[test]
    fn test_register_calibration() {
        let mut mapper = PhaseVoltageMapper::new();
        let calibration = create_test_calibration(1);
        
        let result = mapper.register_calibration(calibration);
        assert!(result.is_ok());
        assert_eq!(mapper.calibrated_shifters().len(), 1);
    }

    #[test]
    fn test_insufficient_calibration_points() {
        let mut mapper = PhaseVoltageMapper::new();
        let calibration = PhaseShifterCalibration {
            shifter_id: 1,
            points: vec![
                CalibrationPoint { phase_rad: 0.0, voltage_v: 0.0, temperature_c: 25.0 },
                CalibrationPoint { phase_rad: std::f64::consts::PI, voltage_v: 2.5, temperature_c: 25.0 },
            ],
            reference_temperature: 25.0,
            temp_coefficient: 0.001,
            hysteresis: HysteresisParams::default(),
        };

        let result = mapper.register_calibration(calibration);
        assert!(result.is_err());
    }

    #[test]
    fn test_phase_to_voltage_linear() {
        let mut mapper = PhaseVoltageMapper::new();
        mapper.register_calibration(create_test_calibration(1)).unwrap();

        // Test π radians -> 2.5V
        let voltage = mapper.phase_to_voltage(1, std::f64::consts::PI).unwrap();
        assert!((voltage - 2.5).abs() < 0.1);

        // Test 2π radians -> 5.0V
        let voltage_2pi = mapper.phase_to_voltage(1, 2.0 * std::f64::consts::PI).unwrap();
        assert!((voltage_2pi - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_temperature_out_of_range() {
        let mut mapper = PhaseVoltageMapper::new();
        mapper.register_calibration(create_test_calibration(1)).unwrap();
        
        mapper.set_chip_temperature(40.0).unwrap_err();
        mapper.set_chip_temperature(10.0).unwrap_err();
    }

    #[test]
    fn test_batch_conversion() {
        let mut mapper = PhaseVoltageMapper::new();
        mapper.register_calibration(create_test_calibration(1)).unwrap();

        let requests = vec![
            (1, std::f64::consts::PI / 2.0),
            (1, std::f64::consts::PI),
            (1, 3.0 * std::f64::consts::PI / 2.0),
        ];

        let results = mapper.batch_phase_to_voltage(&requests).unwrap();
        assert_eq!(results.len(), 3);
        
        // Voltages should be increasing
        assert!(results[0].1 < results[1].1);
        assert!(results[1].1 < results[2].1);
    }
}
