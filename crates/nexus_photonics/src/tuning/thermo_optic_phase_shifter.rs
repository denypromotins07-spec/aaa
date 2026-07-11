//! Thermo-Optic Phase Shifter Controller
//!
//! This module implements control for thermo-optic phase shifters used in
//! silicon photonic circuits. It handles:
//! - Temperature-dependent refractive index tuning
//! - Micro-heater current control with hardware limits
//! - Thermal runaway prevention
//! - Integration with Stage 31 microfluidic cooling

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors in thermo-optic phase shifter operation
#[derive(Error, Debug)]
pub enum PhaseShifterError {
    #[error("Heater current {current}mA exceeds maximum {max}mA")]
    CurrentExceeded { current: f64, max: f64 },
    
    #[error("Temperature {temp}°C exceeds safe limit {limit}°C")]
    TemperatureExceeded { temp: f64, limit: f64 },
    
    #[error("Thermal abort triggered at heater {heater_id}: critical temperature reached")]
    ThermalAbort { heater_id: u32 },
    
    #[error("Phase shifter {id} calibration invalid: resistance={resistance}Ω")]
    CalibrationInvalid { id: u32, resistance: f64 },
    
    #[error("Microfluidic cooling failure detected: flow_rate={flow}mL/min")]
    CoolingFailure { flow: f64 },
    
    #[error("DAC code {code} out of range [0, {max}]")]
    DacCodeOutOfRange { code: u32, max: u32 },
}

/// Configuration for a single phase shifter heater
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HeaterConfig {
    /// Unique identifier
    pub heater_id: u32,
    /// Resistance at reference temperature (Ω)
    pub resistance_ohms: f64,
    /// Temperature coefficient of resistance (1/°C)
    pub tcr: f64,
    /// Thermal resistance to ambient (°C/W)
    pub thermal_resistance_c_w: f64,
    /// Thermal capacitance (J/°C)
    pub thermal_capacitance_j_c: f64,
    /// Maximum safe operating temperature (°C)
    pub max_temperature_c: f64,
    /// Maximum heater current (mA)
    pub max_current_ma: f64,
    /// Phase shift efficiency (rad/mW)
    pub phase_efficiency_rad_mw: f64,
}

impl Default for HeaterConfig {
    fn default() -> Self {
        Self {
            heater_id: 0,
            resistance_ohms: 500.0, // Typical for silicon heaters
            tcr: 0.001, // ~0.1%/°C for doped silicon
            thermal_resistance_c_w: 100.0,
            thermal_capacitance_j_c: 1e-6,
            max_temperature_c: 80.0,
            max_current_ma: 20.0,
            phase_efficiency_rad_mw: 0.1, // ~0.1 rad/mW typical
        }
    }
}

/// State of a phase shifter heater
#[derive(Debug, Clone)]
pub struct HeaterState {
    /// Current temperature (°C)
    pub temperature_c: f64,
    /// Applied current (mA)
    pub current_ma: f64,
    /// Applied voltage (V)
    pub voltage_v: f64,
    /// Dissipated power (mW)
    pub power_mw: f64,
    /// Achieved phase shift (rad)
    pub phase_shift_rad: f64,
    /// Heater is enabled
    pub enabled: bool,
    /// Thermal abort flag
    pub thermal_abort: bool,
}

impl Default for HeaterState {
    fn default() -> Self {
        Self {
            temperature_c: 25.0,
            current_ma: 0.0,
            voltage_v: 0.0,
            power_mw: 0.0,
            phase_shift_rad: 0.0,
            enabled: false,
            thermal_abort: false,
        }
    }
}

/// Thermo-Optic Phase Shifter Controller
pub struct ThermoOpticPhaseShifter {
    /// Heater configurations
    heaters: Vec<HeaterConfig>,
    /// Current state of each heater
    states: Vec<HeaterState>,
    /// Ambient temperature from sensors (°C)
    ambient_temperature_c: f64,
    /// Microfluidic cooling flow rate (mL/min)
    cooling_flow_ml_min: f64,
    /// Minimum required cooling flow (mL/min)
    min_cooling_flow_ml_min: f64,
    /// DAC resolution (bits)
    dac_resolution_bits: u8,
    /// Maximum DAC output voltage (V)
    dac_max_voltage: f64,
}

impl ThermoOpticPhaseShifter {
    /// Create a new phase shifter controller
    pub fn new(num_heaters: usize) -> Self {
        let heaters: Vec<HeaterConfig> = (0..num_heaters)
            .map(|i| HeaterConfig {
                heater_id: i as u32,
                ..Default::default()
            })
            .collect();

        let states: Vec<HeaterState> = (0..num_heaters)
            .map(|_| HeaterState::default())
            .collect();

        Self {
            heaters,
            states,
            ambient_temperature_c: 25.0,
            cooling_flow_ml_min: 10.0,
            min_cooling_flow_ml_min: 5.0,
            dac_resolution_bits: 16,
            dac_max_voltage: 5.0,
        }
    }

    /// Create with custom heater configurations
    pub fn with_configs(configs: Vec<HeaterConfig>) -> Result<Self, PhaseShifterError> {
        // Validate all heater configurations
        for config in &configs {
            if config.resistance_ohms < 10.0 || config.resistance_ohms > 10000.0 {
                return Err(PhaseShifterError::CalibrationInvalid {
                    id: config.heater_id,
                    resistance: config.resistance_ohms,
                });
            }
        }

        let num_heaters = configs.len();
        let states: Vec<HeaterState> = (0..num_heaters)
            .map(|_| HeaterState::default())
            .collect();

        Ok(Self {
            heaters: configs,
            states,
            ambient_temperature_c: 25.0,
            cooling_flow_ml_min: 10.0,
            min_cooling_flow_ml_min: 5.0,
            dac_resolution_bits: 16,
            dac_max_voltage: 5.0,
        })
    }

    /// Set target phase shift for a specific heater
    pub fn set_phase_target(&mut self, heater_id: u32, target_phase_rad: f64) -> Result<(), PhaseShifterError> {
        let heater_idx = heater_id as usize;
        
        if heater_idx >= self.heaters.len() {
            return Err(PhaseShifterError::DacCodeOutOfRange {
                code: heater_id,
                max: (self.heaters.len() - 1) as u32,
            });
        }

        // Check for thermal abort condition
        if self.states[heater_idx].thermal_abort {
            return Err(PhaseShifterError::ThermalAbort {
                heater_id,
            });
        }

        // Check cooling system status
        if self.cooling_flow_ml_min < self.min_cooling_flow_ml_min {
            return Err(PhaseShifterError::CoolingFailure {
                flow: self.cooling_flow_ml_min,
            });
        }

        let heater = &self.heaters[heater_idx];
        
        // Calculate required power for target phase
        // P = φ / η where η is phase efficiency
        let required_power_mw = target_phase_rad / heater.phase_efficiency_rad_mw;
        
        // Calculate required current: P = I²R
        let resistance = self.get_current_resistance(heater_idx);
        let required_current_ma = (required_power_mw / resistance).sqrt() * 1000.0;

        // Validate current limit
        if required_current_ma > heater.max_current_ma {
            return Err(PhaseShifterError::CurrentExceeded {
                current: required_current_ma,
                max: heater.max_current_ma,
            });
        }

        // Calculate resulting temperature rise
        let temp_rise_c = required_power_mw * heater.thermal_resistance_c_w / 1000.0;
        let predicted_temp = self.ambient_temperature_c + temp_rise_c;

        // Validate temperature limit
        if predicted_temp > heater.max_temperature_c {
            return Err(PhaseShifterError::TemperatureExceeded {
                temp: predicted_temp,
                limit: heater.max_temperature_c,
            });
        }

        // Apply the settings
        self.apply_heater_settings(heater_idx, required_current_ma)?;

        Ok(())
    }

    /// Get current resistance accounting for temperature
    fn get_current_resistance(&self, heater_idx: usize) -> f64 {
        let heater = &self.heaters[heater_idx];
        let state = &self.states[heater_idx];
        
        // R(T) = R₀ * (1 + α * ΔT)
        let delta_t = state.temperature_c - self.ambient_temperature_c;
        heater.resistance_ohms * (1.0 + heater.tcr * delta_t)
    }

    /// Apply heater settings with safety checks
    fn apply_heater_settings(&mut self, heater_idx: usize, current_ma: f64) -> Result<(), PhaseShifterError> {
        let heater = &self.heaters[heater_idx];
        let state = &mut self.states[heater_idx];

        // Validate DAC code range
        let dac_code = self.current_to_dac_code(current_ma);
        let max_dac = (1u32 << self.dac_resolution_bits) - 1;
        
        if dac_code > max_dac {
            return Err(PhaseShifterError::DacCodeOutOfRange {
                code: dac_code,
                max: max_dac,
            });
        }

        // Calculate voltage: V = IR
        let resistance = self.get_current_resistance(heater_idx);
        let voltage_v = current_ma * resistance / 1000.0;

        // Calculate power: P = IV
        let power_mw = current_ma * voltage_v;

        // Update state
        state.current_ma = current_ma;
        state.voltage_v = voltage_v;
        state.power_mw = power_mw;
        state.enabled = true;

        // Calculate steady-state temperature
        let temp_rise = power_mw * heater.thermal_resistance_c_w / 1000.0;
        state.temperature_c = self.ambient_temperature_c + temp_rise;

        // Calculate achieved phase shift
        state.phase_shift_rad = power_mw * heater.phase_efficiency_rad_mw;

        // Final safety check
        if state.temperature_c > heater.max_temperature_c {
            state.thermal_abort = true;
            state.enabled = false;
            state.current_ma = 0.0;
            return Err(PhaseShifterError::ThermalAbort {
                heater_id: heater.heater_id,
            });
        }

        Ok(())
    }

    /// Convert current to DAC code
    fn current_to_dac_code(&self, current_ma: f64) -> u32 {
        let max_current = self.dac_max_voltage / 
            (self.heaters.first().map(|h| h.resistance_ohms).unwrap_or(500.0) / 1000.0);
        
        let normalized = (current_ma / max_current).clamp(0.0, 1.0);
        let max_dac = (1u32 << self.dac_resolution_bits) - 1;
        
        (normalized * max_dac as f64).round() as u32
    }

    /// Emergency shutdown of all heaters (thermal abort)
    pub fn emergency_shutdown(&mut self) {
        for state in &mut self.states {
            state.thermal_abort = true;
            state.enabled = false;
            state.current_ma = 0.0;
            state.voltage_v = 0.0;
            state.power_mw = 0.0;
        }
    }

    /// Reset thermal abort flags after cooldown
    pub fn reset_thermal_abort(&mut self) -> Result<(), PhaseShifterError> {
        // Verify all heaters have cooled down
        for (i, state) in self.states.iter().enumerate() {
            if state.temperature_c > self.ambient_temperature_c + 10.0 {
                return Err(PhaseShifterError::TemperatureExceeded {
                    temp: state.temperature_c,
                    limit: self.ambient_temperature_c + 10.0,
                });
            }
        }

        // Clear abort flags
        for state in &mut self.states {
            state.thermal_abort = false;
        }

        Ok(())
    }

    /// Update ambient temperature reading
    pub fn set_ambient_temperature(&mut self, temp_c: f64) {
        self.ambient_temperature_c = temp_c;
        
        // Recalculate all heater temperatures
        for (i, heater) in self.heaters.iter().enumerate() {
            let state = &mut self.states[i];
            if state.enabled {
                let temp_rise = state.power_mw * heater.thermal_resistance_c_w / 1000.0;
                state.temperature_c = self.ambient_temperature_c + temp_rise;
            } else {
                state.temperature_c = self.ambient_temperature_c;
            }
        }
    }

    /// Update cooling flow rate
    pub fn set_cooling_flow(&mut self, flow_ml_min: f64) -> Result<(), PhaseShifterError> {
        self.cooling_flow_ml_min = flow_ml_min;
        
        if flow_ml_min < self.min_cooling_flow_ml_min {
            // Trigger emergency shutdown
            self.emergency_shutdown();
            return Err(PhaseShifterError::CoolingFailure {
                flow: flow_ml_min,
            });
        }
        
        Ok(())
    }

    /// Get heater state
    pub fn get_heater_state(&self, heater_id: u32) -> Option<&HeaterState> {
        self.states.get(heater_id as usize)
    }

    /// Get all heater states
    pub fn all_states(&self) -> &[HeaterState] {
        &self.states
    }

    /// Simulate one time step of thermal dynamics
    pub fn simulate_step(&mut self, dt_ms: f64) {
        for (i, heater) in self.heaters.iter().enumerate() {
            let state = &mut self.states[i];
            
            if !state.enabled {
                // Cool down toward ambient
                let tau = heater.thermal_capacitance_j_c * heater.thermal_resistance_c_w * 1000.0;
                let alpha = (-dt_ms / tau).exp();
                state.temperature_c = self.ambient_temperature_c 
                    + (state.temperature_c - self.ambient_temperature_c) * alpha;
                continue;
            }

            // Thermal dynamics: C * dT/dt = P - (T - T_amb)/R
            let heat_input = state.power_mw / 1000.0; // Convert to W
            let heat_loss = (state.temperature_c - self.ambient_temperature_c) 
                / heater.thermal_resistance_c_w;
            
            let net_power = heat_input - heat_loss;
            let temp_change = net_power * dt_ms / (heater.thermal_capacitance_j_c * 1000.0);
            
            state.temperature_c += temp_change;

            // Check for thermal runaway
            if state.temperature_c > heater.max_temperature_c {
                state.thermal_abort = true;
                state.enabled = false;
                state.current_ma = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_creation() {
        let controller = ThermoOpticPhaseShifter::new(8);
        assert_eq!(controller.all_states().len(), 8);
    }

    #[test]
    fn test_set_phase_target() {
        let mut controller = ThermoOpticPhaseShifter::new(4);
        
        let result = controller.set_phase_target(0, std::f64::consts::PI);
        assert!(result.is_ok());
        
        let state = controller.get_heater_state(0).unwrap();
        assert!(state.enabled);
        assert!(state.phase_shift_rad > 0.0);
    }

    #[test]
    fn test_current_limit_enforcement() {
        let configs = vec![HeaterConfig {
            max_current_ma: 5.0, // Very low limit
            phase_efficiency_rad_mw: 0.01, // Low efficiency requires more power
            ..Default::default()
        }];
        
        let mut controller = ThermoOpticPhaseShifter::with_configs(configs).unwrap();
        
        // Request large phase shift that would exceed current limit
        let result = controller.set_phase_target(0, 10.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_temperature_limit_enforcement() {
        let configs = vec![HeaterConfig {
            max_temperature_c: 30.0, // Very low limit (near ambient)
            thermal_resistance_c_w: 1000.0, // High thermal resistance
            ..Default::default()
        }];
        
        let mut controller = ThermoOpticPhaseShifter::with_configs(configs).unwrap();
        
        let result = controller.set_phase_target(0, std::f64::consts::PI);
        assert!(result.is_err());
    }

    #[test]
    fn test_emergency_shutdown() {
        let mut controller = ThermoOpticPhaseShifter::new(4);
        
        // Enable some heaters
        controller.set_phase_target(0, 1.0).unwrap();
        controller.set_phase_target(1, 1.0).unwrap();
        
        // Emergency shutdown
        controller.emergency_shutdown();
        
        // All heaters should be disabled
        for state in controller.all_states() {
            assert!(!state.enabled);
            assert!(state.thermal_abort);
            assert_eq!(state.current_ma, 0.0);
        }
    }

    #[test]
    fn test_cooling_failure_triggers_shutdown() {
        let mut controller = ThermoOpticPhaseShifter::new(4);
        
        controller.set_phase_target(0, 1.0).unwrap();
        
        // Simulate cooling failure
        let result = controller.set_cooling_flow(2.0); // Below minimum
        assert!(result.is_err());
        
        // Heaters should be shut down
        let state = controller.get_heater_state(0).unwrap();
        assert!(!state.enabled);
    }

    #[test]
    fn test_thermal_simulation() {
        let mut controller = ThermoOpticPhaseShifter::new(4);
        controller.set_phase_target(0, 1.0).unwrap();
        
        let initial_temp = controller.get_heater_state(0).unwrap().temperature_c;
        
        // Simulate heating over time
        for _ in 0..100 {
            controller.simulate_step(1.0);
        }
        
        let final_temp = controller.get_heater_state(0).unwrap().temperature_c;
        assert!(final_temp >= initial_temp);
    }
}
