//! Two-Phase Dielectric Immersion Cooling Controller
//! 
//! Manages two-phase immersion cooling systems using 3M Fluorinert or similar
//! dielectric fluids, monitoring boiling crisis prevention and vapor condensation.

use core::fmt;
use crate::cooling::microfluidic_pid::{PidError, ThermalSensor};

/// Critical heat flux threshold (W/cm²) - point where boiling becomes unstable
const CRITICAL_HEAT_FLUX: f64 = 25.0;
/// Maximum safe subcooling temperature difference (°C)
const MAX_SUBCOOLING: f64 = 15.0;
/// Minimum fluid level percentage
const MIN_FLUID_LEVEL: f64 = 20.0;
/// Maximum fluid temperature before pump shutdown (°C)
const MAX_FLUID_TEMP: f64 = 85.0;
/// Boiling point of Fluorinert FC-72 at 1 atm (°C)
const FLUORINERT_BOILING_POINT: f64 = 56.0;

/// Errors specific to two-phase immersion cooling
#[derive(Debug, Clone, PartialEq)]
pub enum ImmersionError {
    CriticalHeatFluxExceeded,
    FluidLevelLow,
    FluidOverheated,
    VaporPressureHigh,
    CondenserFailure,
    PumpFailure,
    SensorMalfunction,
}

impl fmt::Display for ImmersionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImmersionError::CriticalHeatFluxExceeded => write!(f, "Critical heat flux exceeded - boiling crisis imminent"),
            ImmersionError::FluidLevelLow => write!(f, "Dielectric fluid level below minimum"),
            ImmersionError::FluidOverheated => write!(f, "Fluid temperature exceeds safe operating limit"),
            ImmersionError::VaporPressureHigh => write!(f, "Vapor pressure in containment vessel too high"),
            ImmersionError::CondenserFailure => write!(f, "Condenser unit failure detected"),
            ImmersionError::PumpFailure => write!(f, "Circulation pump failure"),
            ImmersionError::SensorMalfunction => write!(f, "Immersion sensor malfunction"),
        }
    }
}

/// State of the two-phase immersion system
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImmersionState {
    /// Current fluid temperature (°C)
    pub fluid_temp: f64,
    /// Current vapor temperature (°C)
    pub vapor_temp: f64,
    /// Fluid level percentage (0-100)
    pub fluid_level: f64,
    /// Vapor pressure (kPa)
    pub vapor_pressure: f64,
    /// Heat flux estimate (W/cm²)
    pub heat_flux: f64,
    /// Subcooling margin (°C)
    pub subcooling: f64,
    /// Pump flow rate (L/min)
    pub flow_rate: f64,
    /// Condenser efficiency (0-1)
    pub condenser_efficiency: f64,
}

impl Default for ImmersionState {
    fn default() -> Self {
        Self {
            fluid_temp: 40.0,
            vapor_temp: 45.0,
            fluid_level: 80.0,
            vapor_pressure: 101.3,
            heat_flux: 5.0,
            subcooling: 10.0,
            flow_rate: 10.0,
            condenser_efficiency: 0.9,
        }
    }
}

/// Two-phase immersion cooling controller
pub struct TwoPhaseImmersionController {
    state: ImmersionState,
    /// Safety margin factor (0.5 = 50% below critical limits)
    safety_margin: f64,
    /// Emergency shutdown flag
    emergency_shutdown: bool,
}

impl TwoPhaseImmersionController {
    /// Create a new immersion controller with specified safety margin
    pub fn new(safety_margin: f64) -> Result<Self, ImmersionError> {
        if safety_margin <= 0.0 || safety_margin >= 1.0 {
            return Err(ImmersionError::SensorMalfunction);
        }

        Ok(Self {
            state: ImmersionState::default(),
            safety_margin,
            emergency_shutdown: false,
        })
    }

    /// Update system state from sensor readings
    pub fn update_state(
        &mut self,
        fluid_temp: f64,
        vapor_temp: f64,
        fluid_level: f64,
        vapor_pressure: f64,
        flow_rate: f64,
    ) -> Result<(), ImmersionError> {
        // Validate inputs
        if fluid_temp < 0.0 || fluid_temp > 150.0 {
            return Err(ImmersionError::SensorMalfunction);
        }
        if vapor_temp < 0.0 || vapor_temp > 150.0 {
            return Err(ImmersionError::SensorMalfunction);
        }
        if fluid_level < 0.0 || fluid_level > 100.0 {
            return Err(ImmersionError::SensorMalfunction);
        }

        // Calculate derived values
        let subcooling = FLUORINERT_BOILING_POINT - fluid_temp;
        
        // Estimate heat flux based on temperature differential and flow rate
        let delta_t = vapor_temp - fluid_temp;
        let heat_flux = if flow_rate > 0.0 {
            delta_t * flow_rate * 0.042 // Simplified heat transfer coefficient
        } else {
            0.0
        };

        self.state = ImmersionState {
            fluid_temp,
            vapor_temp,
            fluid_level,
            vapor_pressure,
            heat_flux,
            subcooling,
            flow_rate,
            condenser_efficiency: self.state.condenser_efficiency,
        };

        // Run safety checks
        self.safety_check()?;

        Ok(())
    }

    /// Perform comprehensive safety checks
    fn safety_check(&mut self) -> Result<(), ImmersionError> {
        // Check for critical heat flux
        let safe_heat_flux = CRITICAL_HEAT_FLUX * (1.0 - self.safety_margin);
        if self.state.heat_flux > safe_heat_flux {
            self.emergency_shutdown = true;
            return Err(ImmersionError::CriticalHeatFluxExceeded);
        }

        // Check fluid level
        if self.state.fluid_level < MIN_FLUID_LEVEL {
            self.emergency_shutdown = true;
            return Err(ImmersionError::FluidLevelLow);
        }

        // Check fluid temperature
        if self.state.fluid_temp > MAX_FLUID_TEMP {
            self.emergency_shutdown = true;
            return Err(ImmersionError::FluidOverheated);
        }

        // Check subcooling margin
        if self.state.subcooling < 0.0 || self.state.subcooling > MAX_SUBCOOLING {
            // Negative subcooling means fluid is boiling uncontrollably
            if self.state.subcooling < 0.0 {
                self.emergency_shutdown = true;
                return Err(ImmersionError::CriticalHeatFluxExceeded);
            }
        }

        // Check vapor pressure (simplified - should be relative to vessel rating)
        if self.state.vapor_pressure > 200.0 {
            self.emergency_shutdown = true;
            return Err(ImmersionError::VaporPressureHigh);
        }

        Ok(())
    }

    /// Get recommended pump speed adjustment
    pub fn get_pump_adjustment(&self) -> f64 {
        if self.emergency_shutdown {
            return -1.0; // Full stop
        }

        let mut adjustment = 0.0;

        // Increase flow if heat flux is high
        let heat_flux_ratio = self.state.heat_flux / CRITICAL_HEAT_FLUX;
        if heat_flux_ratio > 0.5 {
            adjustment += (heat_flux_ratio - 0.5) * 2.0;
        }

        // Decrease flow if subcooling is too high (overcooling)
        if self.state.subcooling > MAX_SUBCOOLING * 0.8 {
            adjustment -= 0.2;
        }

        adjustment.clamp(-1.0, 1.0)
    }

    /// Get recommended condenser setpoint
    pub fn get_condenser_setpoint(&self) -> f64 {
        // Target fluid temperature slightly below boiling point
        let target = FLUORINERT_BOILING_POINT - 5.0;
        
        // Adjust based on current state
        if self.state.fluid_temp > target {
            target - 2.0 // Increase cooling
        } else if self.state.subcooling > MAX_SUBCOOLING * 0.5 {
            target + 2.0 // Reduce cooling to save energy
        } else {
            target
        }
    }

    /// Check if emergency shutdown is active
    pub fn is_emergency_shutdown(&self) -> bool {
        self.emergency_shutdown
    }

    /// Reset emergency shutdown (requires manual intervention)
    pub fn reset_emergency_shutdown(&mut self) -> Result<(), ImmersionError> {
        // Only allow reset if conditions are safe
        if self.state.fluid_temp < MAX_FLUID_TEMP * 0.8 
            && self.state.heat_flux < CRITICAL_HEAT_FLUX * 0.5
            && self.state.fluid_level > MIN_FLUID_LEVEL * 1.5 
        {
            self.emergency_shutdown = false;
            Ok(())
        } else {
            Err(ImmersionError::CriticalHeatFluxExceeded)
        }
    }

    /// Get current state for monitoring
    pub fn get_state(&self) -> &ImmersionState {
        &self.state
    }

    /// Calculate boiling crisis probability (0-1)
    pub fn boiling_crisis_probability(&self) -> f64 {
        let heat_flux_risk = (self.state.heat_flux / CRITICAL_HEAT_FLUX).clamp(0.0, 1.0);
        let subcooling_risk = if self.state.subcooling < 5.0 {
            (5.0 - self.state.subcooling) / 5.0
        } else {
            0.0
        };
        
        (heat_flux_risk * 0.7 + subcooling_risk * 0.3).clamp(0.0, 1.0)
    }
}

/// Trait for immersion-specific sensors
pub trait ImmersionSensor: ThermalSensor {
    fn read_fluid_level(&self) -> Result<f64, ImmersionError>;
    fn read_vapor_pressure(&self) -> Result<f64, ImmersionError>;
    fn read_flow_rate(&self) -> Result<f64, ImmersionError>;
}

/// Trait for immersion actuators
pub trait CondenserActuator {
    fn set_temperature(&mut self, temp: f64) -> Result<(), ImmersionError>;
}

pub trait CirculationPump {
    fn set_flow_rate(&mut self, rate: f64) -> Result<(), ImmersionError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_creation() {
        let controller = TwoPhaseImmersionController::new(0.3);
        assert!(controller.is_ok());
    }

    #[test]
    fn test_invalid_safety_margin() {
        let controller = TwoPhaseImmersionController::new(1.5);
        assert_eq!(controller.unwrap_err(), ImmersionError::SensorMalfunction);
    }

    #[test]
    fn test_normal_operation() {
        let mut controller = TwoPhaseImmersionController::new(0.3).unwrap();
        
        let result = controller.update_state(45.0, 50.0, 75.0, 105.0, 12.0);
        assert!(result.is_ok());
        assert!(!controller.is_emergency_shutdown());
    }

    #[test]
    fn test_critical_heat_flux() {
        let mut controller = TwoPhaseImmersionController::new(0.3).unwrap();
        
        // High heat flux should trigger emergency
        let result = controller.update_state(55.0, 65.0, 70.0, 120.0, 25.0);
        assert_eq!(result.unwrap_err(), ImmersionError::CriticalHeatFluxExceeded);
        assert!(controller.is_emergency_shutdown());
    }

    #[test]
    fn test_low_fluid_level() {
        let mut controller = TwoPhaseImmersionController::new(0.3).unwrap();
        
        let result = controller.update_state(40.0, 45.0, 15.0, 100.0, 10.0);
        assert_eq!(result.unwrap_err(), ImmersionError::FluidLevelLow);
    }

    #[test]
    fn test_boiling_crisis_probability() {
        let mut controller = TwoPhaseImmersionController::new(0.3).unwrap();
        
        // Normal conditions - low risk
        controller.update_state(45.0, 50.0, 75.0, 105.0, 12.0).unwrap();
        let prob = controller.boiling_crisis_probability();
        assert!(prob < 0.3);

        // High risk conditions
        controller.update_state(54.0, 60.0, 60.0, 130.0, 20.0).unwrap();
        let prob = controller.boiling_crisis_probability();
        assert!(prob > 0.7);
    }
}
