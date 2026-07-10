//! Microfluidic PID Controller for Direct-to-Chip Liquid Cooling
//! 
//! Implements zero-allocation embedded-hal PID controllers with anti-windup logic
//! to manage micro-pump RPMs and solenoid valve duty cycles based on die temperature.

use core::fmt;
use core::time::Duration;

/// Maximum allowable integral term to prevent windup
const MAX_INTEGRAL: f64 = 1000.0;
/// Minimum allowable integral term
const MIN_INTEGRAL: f64 = -1000.0;
/// Maximum pump RPM limit
const MAX_PUMP_RPM: u32 = 5000;
/// Minimum pump RPM (off)
const MIN_PUMP_RPM: u32 = 0;
/// Maximum solenoid duty cycle (0-100%)
const MAX_DUTY_CYCLE: u8 = 100;
/// Minimum solenoid duty cycle
const MIN_DUTY_CYCLE: u8 = 0;
/// Derivative kick filter coefficient
const DERIVATIVE_FILTER_ALPHA: f64 = 0.1;

/// Errors that can occur in the microfluidic PID system
#[derive(Debug, Clone, PartialEq)]
pub enum PidError {
    SensorReadFailure,
    ActuatorWriteFailure,
    InvalidParameter,
    IntegralWindup,
    TemperatureOutOfRange,
}

impl fmt::Display for PidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PidError::SensorReadFailure => write!(f, "Failed to read temperature sensor"),
            PidError::ActuatorWriteFailure => write!(f, "Failed to write to actuator"),
            PidError::InvalidParameter => write!(f, "Invalid PID parameter"),
            PidError::IntegralWindup => write!(f, "Integral term windup detected"),
            PidError::TemperatureOutOfRange => write!(f, "Temperature reading out of valid range"),
        }
    }
}

/// PID controller state for microfluidic cooling
pub struct MicrofluidicPidController {
    /// Proportional gain
    kp: f64,
    /// Integral gain
    ki: f64,
    /// Derivative gain
    kd: f64,
    /// Integral accumulator (anti-windup protected)
    integral: f64,
    /// Previous error value
    prev_error: f64,
    /// Filtered derivative term
    filtered_derivative: f64,
    /// Target temperature setpoint (°C)
    setpoint: f64,
    /// Last update timestamp (microseconds)
    last_update_us: u64,
    /// Minimum time between updates (microseconds)
    min_update_interval_us: u64,
}

impl MicrofluidicPidController {
    /// Create a new PID controller with specified gains
    pub fn new(kp: f64, ki: f64, kd: f64, setpoint: f64) -> Result<Self, PidError> {
        if kp < 0.0 || ki < 0.0 || kd < 0.0 {
            return Err(PidError::InvalidParameter);
        }
        if setpoint < 0.0 || setpoint > 150.0 {
            return Err(PidError::TemperatureOutOfRange);
        }

        Ok(Self {
            kp,
            ki,
            kd,
            integral: 0.0,
            prev_error: 0.0,
            filtered_derivative: 0.0,
            setpoint,
            last_update_us: 0,
            min_update_interval_us: 100, // 10kHz minimum update rate
        })
    }

    /// Update PID parameters dynamically
    pub fn update_params(&mut self, kp: Option<f64>, ki: Option<f64>, kd: Option<f64>) -> Result<(), PidError> {
        if let Some(new_kp) = kp {
            if new_kp < 0.0 {
                return Err(PidError::InvalidParameter);
            }
            self.kp = new_kp;
        }
        if let Some(new_ki) = ki {
            if new_ki < 0.0 {
                return Err(PidError::InvalidParameter);
            }
            self.ki = new_ki;
        }
        if let Some(new_kd) = kd {
            if new_kd < 0.0 {
                return Err(PidError::InvalidParameter);
            }
            self.kd = new_kd;
        }
        Ok(())
    }

    /// Compute PID output from current temperature reading
    /// Returns (pump_rpm, duty_cycle) tuple
    pub fn compute(&mut self, current_temp: f64, timestamp_us: u64) -> Result<(u32, u8), PidError> {
        if current_temp < -40.0 || current_temp > 150.0 {
            return Err(PidError::TemperatureOutOfRange);
        }

        // Check minimum update interval
        if timestamp_us - self.last_update_us < self.min_update_interval_us {
            // Return last computed values or defaults
            return Ok((MIN_PUMP_RPM, MIN_DUTY_CYCLE));
        }

        // Calculate time delta in seconds
        let dt_us = timestamp_us - self.last_update_us;
        let dt = if dt_us == 0 { 0.001 } else { dt_us as f64 / 1_000_000.0 };

        // Calculate error
        let error = self.setpoint - current_temp;

        // Proportional term
        let p_term = self.kp * error;

        // Integral term with anti-windup clamping
        let mut integral_increment = self.ki * error * dt;
        
        // Anti-windup: only integrate if output is not saturated
        let predicted_output = p_term + self.integral + integral_increment;
        if predicted_output > 100.0 || predicted_output < -100.0 {
            // Clamp integral increment to prevent windup
            if integral_increment > 0.0 {
                integral_increment = 0.0;
            }
        }
        
        self.integral += integral_increment;
        
        // Hard clamp integral term
        self.integral = self.integral.clamp(MIN_INTEGRAL, MAX_INTEGRAL);

        // Derivative term with filtering (derivative kick prevention)
        let raw_derivative = (error - self.prev_error) / dt;
        self.filtered_derivative = DERIVATIVE_FILTER_ALPHA * raw_derivative 
            + (1.0 - DERIVATIVE_FILTER_ALPHA) * self.filtered_derivative;
        let d_term = self.kd * self.filtered_derivative;

        // Update previous error
        self.prev_error = error;
        self.last_update_us = timestamp_us;

        // Combine terms
        let output = p_term + self.integral + d_term;

        // Map output to pump RPM and duty cycle
        let (pump_rpm, duty_cycle) = self.map_output_to_actuators(output);

        Ok((pump_rpm, duty_cycle))
    }

    /// Map PID output to actuator commands
    fn map_output_to_actuators(&self, output: f64) -> (u32, u8) {
        // Normalize output to 0-100 range
        let normalized = (output / 100.0).clamp(-1.0, 1.0);
        
        // Map to pump RPM (linear mapping for simplicity)
        let pump_rpm = if normalized <= 0.0 {
            MIN_PUMP_RPM
        } else {
            (MIN_PUMP_RPM as f64 + normalized * (MAX_PUMP_RPM - MIN_PUMP_RPM) as f64) as u32
        };

        // Map to duty cycle (separate mapping for solenoid valves)
        let duty_cycle = if normalized <= 0.0 {
            MIN_DUTY_CYCLE
        } else {
            (MIN_DUTY_CYCLE as f64 + normalized * (MAX_DUTY_CYCLE - MIN_DUTY_CYCLE) as f64) as u8
        };

        (pump_rpm, duty_cycle)
    }

    /// Reset integral term (useful during startup or mode changes)
    pub fn reset_integral(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
        self.filtered_derivative = 0.0;
    }

    /// Get current integral value for monitoring
    pub fn get_integral(&self) -> f64 {
        self.integral
    }

    /// Check if integral is approaching windup limits
    pub fn is_approaching_windup(&self, threshold: f64) -> bool {
        self.integral.abs() > (MAX_INTEGRAL.abs() * threshold)
    }
}

/// Trait for hardware abstraction layer (embedded-hal compatible)
pub trait ThermalSensor {
    fn read_temperature(&self) -> Result<f64, PidError>;
}

pub trait PumpActuator {
    fn set_rpm(&mut self, rpm: u32) -> Result<(), PidError>;
}

pub trait SolenoidActuator {
    fn set_duty_cycle(&mut self, duty: u8) -> Result<(), PidError>;
}

/// Complete cooling control loop
pub struct CoolingControlLoop<S, P, V> 
where
    S: ThermalSensor,
    P: PumpActuator,
    V: SolenoidActuator,
{
    pid: MicrofluidicPidController,
    sensor: S,
    pump: P,
    valve: V,
    timestamp_us: u64,
}

impl<S, P, V> CoolingControlLoop<S, P, V>
where
    S: ThermalSensor,
    P: PumpActuator,
    V: SolenoidActuator,
{
    pub fn new(pid: MicrofluidicPidController, sensor: S, pump: P, valve: V) -> Self {
        Self {
            pid,
            sensor,
            pump,
            valve,
            timestamp_us: 0,
        }
    }

    /// Execute one control cycle
    pub fn step(&mut self) -> Result<(), PidError> {
        // Read temperature
        let temp = self.sensor.read_temperature()?;
        
        // Update timestamp
        self.timestamp_us += 100; // Assume 100μs cycle time
        
        // Compute PID output
        let (rpm, duty) = self.pid.compute(temp, self.timestamp_us)?;
        
        // Apply to actuators
        self.pump.set_rpm(rpm)?;
        self.valve.set_duty_cycle(duty)?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_creation() {
        let pid = MicrofluidicPidController::new(1.0, 0.5, 0.1, 60.0);
        assert!(pid.is_ok());
    }

    #[test]
    fn test_invalid_gains() {
        let pid = MicrofluidicPidController::new(-1.0, 0.5, 0.1, 60.0);
        assert_eq!(pid.unwrap_err(), PidError::InvalidParameter);
    }

    #[test]
    fn test_anti_windup() {
        let mut pid = MicrofluidicPidController::new(1.0, 10.0, 0.0, 60.0).unwrap();
        
        // Simulate large persistent error
        for i in 0..1000 {
            let _ = pid.compute(100.0, i * 1000);
        }
        
        // Integral should be clamped
        assert!(pid.get_integral() <= MAX_INTEGRAL);
        assert!(pid.get_integral() >= MIN_INTEGRAL);
    }

    #[test]
    fn test_temperature_bounds() {
        let mut pid = MicrofluidicPidController::new(1.0, 0.5, 0.1, 60.0).unwrap();
        
        let result = pid.compute(200.0, 1000);
        assert_eq!(result.unwrap_err(), PidError::TemperatureOutOfRange);
        
        let result = pid.compute(-50.0, 1000);
        assert_eq!(result.unwrap_err(), PidError::TemperatureOutOfRange);
    }
}
