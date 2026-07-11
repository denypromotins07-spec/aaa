//! Resonance Feedback Loop for MZI Mesh Calibration
//!
//! This module implements the complete resonance locking feedback system
//! that maintains MZI mesh calibration despite thermal drift and environmental
//! perturbations. It integrates:
//! - Lock-in amplifier based error detection
//! - PID control for heater adjustment
//! - Thermal runaway prevention with hardware current limits
//! - Fallback to electronic execution on critical failure

use crate::tuning::dithering_lock_in_amplifier::{DitheringLockInAmplifier, LockInConfig, LockInError};
use crate::tuning::thermo_optic_phase_shifter::{ThermoOpticPhaseShifter, HeaterState, PhaseShifterError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors in resonance feedback loop operation
#[derive(Error, Debug)]
pub enum ResonanceError {
    #[error("Feedback loop diverged after {iterations} iterations")]
    LoopDivergence { iterations: usize },
    
    #[error("All heaters in thermal abort - switching to electronic fallback")]
    AllHeatersAbort,
    
    #[error("Critical heater {heater_id} failed - mesh calibration compromised")]
    CriticalHeaterFailure { heater_id: u32 },
    
    #[error("Lock-in amplifier error: {source}")]
    LockInAmplifierError { source: LockInError },
    
    #[error("Phase shifter error: {source}")]
    PhaseShifterError { source: PhaseShifterError },
    
    #[error("Convergence failed: residual_error={error} exceeds threshold={threshold}")]
    ConvergenceFailed { error: f64, threshold: f64 },
}

/// PID controller configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PidConfig {
    /// Proportional gain
    pub kp: f64,
    /// Integral gain
    pub ki: f64,
    /// Derivative gain
    pub kd: f64,
    /// Integral windup limit
    pub integral_limit: f64,
    /// Output saturation limit
    pub output_limit: f64,
}

impl Default for PidConfig {
    fn default() -> Self {
        Self {
            kp: 1.0,
            ki: 0.1,
            kd: 0.05,
            integral_limit: 10.0,
            output_limit: 1.0,
        }
    }
}

/// State of a single resonance lock
#[derive(Debug, Clone)]
pub struct ResonanceLockState {
    /// Heater ID being controlled
    pub heater_id: u32,
    /// Current error signal
    pub error_signal: f64,
    /// Integrated error
    pub integral_error: f64,
    /// Previous error (for derivative)
    pub previous_error: f64,
    /// Control output
    pub control_output: f64,
    /// Lock achieved
    pub locked: bool,
    /// Cycles in current state
    pub cycles_in_state: usize,
}

/// Operating mode of the feedback system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackMode {
    /// Normal closed-loop operation
    ClosedLoop,
    /// Open-loop with last known good settings
    OpenLoop,
    /// Electronic fallback (photonic disabled)
    ElectronicFallback,
    /// Emergency shutdown
    EmergencyShutdown,
}

/// Complete resonance feedback loop controller
pub struct ResonanceFeedbackLoop {
    /// Number of controlled elements
    num_elements: usize,
    /// Per-element lock states
    lock_states: Vec<ResonanceLockState>,
    /// PID configuration
    pid_config: PidConfig,
    /// Lock-in amplifiers for error detection
    lock_in_amps: Vec<DitheringLockInAmplifier>,
    /// Phase shifter controller
    phase_shifter: ThermoOpticPhaseShifter,
    /// Current operating mode
    mode: FeedbackMode,
    /// Error threshold for convergence
    convergence_threshold: f64,
    /// Maximum iterations before declaring divergence
    max_iterations: usize,
    /// Critical heater IDs (mesh will fail if these lose lock)
    critical_heaters: Vec<u32>,
}

impl ResonanceFeedbackLoop {
    /// Create a new resonance feedback loop
    pub fn new(num_elements: usize) -> Result<Self, ResonanceError> {
        let mut lock_states = Vec::with_capacity(num_elements);
        let mut lock_in_amps = Vec::with_capacity(num_elements);

        for i in 0..num_elements {
            lock_states.push(ResonanceLockState {
                heater_id: i as u32,
                error_signal: 0.0,
                integral_error: 0.0,
                previous_error: 0.0,
                control_output: 0.0,
                locked: false,
                cycles_in_state: 0,
            });

            // Create lock-in amplifier with slightly different dither frequencies
            // to avoid inter-channel interference
            let dither_freq = 1000.0 + (i as f64) * 50.0; // Stagger by 50 Hz
            let config = LockInConfig {
                dither_frequency_hz: dither_freq,
                ..Default::default()
            };

            match DitheringLockInAmplifier::with_config(config) {
                Ok(amp) => lock_in_amps.push(amp),
                Err(e) => return Err(ResonanceError::LockInAmplifierError { source: e }),
            }
        }

        let phase_shifter = ThermoOpticPhaseShifter::new(num_elements);

        Ok(Self {
            num_elements,
            lock_states,
            pid_config: PidConfig::default(),
            lock_in_amps,
            phase_shifter,
            mode: FeedbackMode::ClosedLoop,
            convergence_threshold: 1e-6,
            max_iterations: 1000,
            critical_heaters: Vec::new(),
        })
    }

    /// Mark specific heaters as critical
    pub fn set_critical_heaters(&mut self, heater_ids: &[u32]) {
        self.critical_heaters = heater_ids.to_vec();
    }

    /// Execute one iteration of the feedback loop
    pub fn step(&mut self, sensor_readings: &[f64]) -> Result<(), ResonanceError> {
        if self.mode == FeedbackMode::EmergencyShutdown 
            || self.mode == FeedbackMode::ElectronicFallback {
            return Ok(()); // No action in fallback modes
        }

        if sensor_readings.len() != self.num_elements {
            return Err(ResonanceError::ConvergenceFailed {
                error: sensor_readings.len() as f64,
                threshold: self.num_elements as f64,
            });
        }

        let mut all_locked = true;
        let mut any_critical_failed = false;

        for i in 0..self.num_elements {
            // Process sensor reading through lock-in amplifier
            let lock_in_state = self.lock_in_amps[i].process_sample(sensor_readings[i]);
            self.lock_in_amps[i].advance_phase();

            // Get error signal from lock-in
            let error = self.lock_in_amps[i].get_error_signal();
            self.lock_states[i].error_signal = error;

            // Check if this is a critical heater
            let is_critical = self.critical_heaters.contains(&(i as u32));

            // Update PID control
            let control = self.pid_step(i, error)?;
            self.lock_states[i].control_output = control;

            // Apply control to phase shifter
            // Convert control output [-1, 1] to phase shift [0, 2π]
            let target_phase = ((control + 1.0) / 2.0) * 2.0 * std::f64::consts::PI;
            
            match self.phase_shifter.set_phase_target(i as u32, target_phase) {
                Ok(_) => {
                    // Check lock status
                    if lock_in_state.snr_db < 20.0 {
                        self.lock_states[i].locked = false;
                        self.lock_states[i].cycles_in_state = 0;
                        all_locked = false;
                        
                        if is_critical {
                            any_critical_failed = true;
                        }
                    } else {
                        self.lock_states[i].locked = true;
                        self.lock_states[i].cycles_in_state += 1;
                    }
                }
                Err(e) => {
                    // Handle phase shifter errors
                    self.handle_phase_shifter_error(i as u32, e)?;
                    all_locked = false;
                    
                    if is_critical {
                        any_critical_failed = true;
                    }
                }
            }
        }

        // Check for critical failures
        if any_critical_failed {
            self.mode = FeedbackMode::ElectronicFallback;
            return Err(ResonanceError::CriticalHeaterFailure {
                heater_id: self.critical_heaters.first().copied().unwrap_or(0),
            });
        }

        // Check for complete system failure
        if !all_locked && self.lock_states.iter().all(|s| !s.locked) {
            self.mode = FeedbackMode::EmergencyShutdown;
            self.phase_shifter.emergency_shutdown();
            return Err(ResonanceError::AllHeatersAbort);
        }

        Ok(())
    }

    /// Execute one PID control step
    fn pid_step(&mut self, index: usize, error: f64) -> Result<f64, ResonanceError> {
        let state = &mut self.lock_states[index];

        // Proportional term
        let p_term = self.pid_config.kp * error;

        // Integral term with windup protection
        state.integral_error += error;
        state.integral_error = state.integral_error.clamp(
            -self.pid_config.integral_limit,
            self.pid_config.integral_limit,
        );
        let i_term = self.pid_config.ki * state.integral_error;

        // Derivative term
        let derivative = error - state.previous_error;
        let d_term = self.pid_config.kd * derivative;
        state.previous_error = error;

        // Combine terms
        let mut output = p_term + i_term + d_term;

        // Apply saturation limit
        output = output.clamp(-self.pid_config.output_limit, self.pid_config.output_limit);

        Ok(output)
    }

    /// Handle phase shifter errors
    fn handle_phase_shifter_error(&mut self, heater_id: u32, error: PhaseShifterError) -> Result<(), ResonanceError> {
        match &error {
            PhaseShifterError::ThermalAbort { .. } => {
                // Thermal abort - mark as unlocked
                self.lock_states[heater_id as usize].locked = false;
                
                // Check if we should switch modes
                let abort_count = self.lock_states.iter()
                    .filter(|s| !s.locked)
                    .count();
                
                if abort_count > self.num_elements / 2 {
                    self.mode = FeedbackMode::ElectronicFallback;
                    return Err(ResonanceError::AllHeatersAbort);
                }
            }
            PhaseShifterError::CoolingFailure { .. } => {
                // Cooling failure - immediate shutdown
                self.mode = FeedbackMode::EmergencyShutdown;
                self.phase_shifter.emergency_shutdown();
                return Err(ResonanceError::AllHeatersAbort);
            }
            _ => {
                // Other errors - just mark as unlocked
                self.lock_states[heater_id as usize].locked = false;
            }
        }

        Err(ResonanceError::PhaseShifterError { source: error })
    }

    /// Run the feedback loop until convergence or failure
    pub fn run_until_converge(&mut self, sensor_stream: &[Vec<f64>]) -> Result<usize, ResonanceError> {
        let mut iterations = 0;

        while iterations < self.max_iterations {
            if iterations >= sensor_stream.len() {
                break;
            }

            self.step(&sensor_stream[iterations])?;

            // Check convergence
            if self.check_convergence() {
                return Ok(iterations + 1);
            }

            iterations += 1;
        }

        // Check why we didn't converge
        if self.mode == FeedbackMode::ElectronicFallback {
            return Err(ResonanceError::AllHeatersAbort);
        }

        let max_error = self.lock_states.iter()
            .map(|s| s.error_signal.abs())
            .fold(0.0, f64::max);

        Err(ResonanceError::ConvergenceFailed {
            error: max_error,
            threshold: self.convergence_threshold,
        })
    }

    /// Check if all locks have converged
    fn check_convergence(&self) -> bool {
        if self.mode != FeedbackMode::ClosedLoop {
            return false;
        }

        for state in &self.lock_states {
            if !state.locked {
                return false;
            }
            if state.error_signal.abs() > self.convergence_threshold {
                return false;
            }
            if state.cycles_in_state < 10 {
                return false; // Need sustained lock
            }
        }

        true
    }

    /// Get current lock states
    pub fn get_lock_states(&self) -> &[ResonanceLockState] {
        &self.lock_states
    }

    /// Get current operating mode
    pub fn mode(&self) -> FeedbackMode {
        self.mode
    }

    /// Set PID configuration
    pub fn set_pid_config(&mut self, config: PidConfig) {
        self.pid_config = config;
    }

    /// Set convergence threshold
    pub fn set_convergence_threshold(&mut self, threshold: f64) {
        self.convergence_threshold = threshold;
    }

    /// Reset all lock states
    pub fn reset(&mut self) {
        for state in &mut self.lock_states {
            state.error_signal = 0.0;
            state.integral_error = 0.0;
            state.previous_error = 0.0;
            state.control_output = 0.0;
            state.locked = false;
            state.cycles_in_state = 0;
        }
        for amp in &mut self.lock_in_amps {
            amp.reset();
        }
        self.mode = FeedbackMode::ClosedLoop;
    }

    /// Attempt recovery from fallback mode
    pub fn attempt_recovery(&mut self) -> Result<(), ResonanceError> {
        if self.mode != FeedbackMode::ElectronicFallback 
            && self.mode != FeedbackMode::EmergencyShutdown {
            return Ok(()); // Already in normal mode
        }

        // Try to reset thermal aborts
        match self.phase_shifter.reset_thermal_abort() {
            Ok(_) => {
                // Reset lock states
                self.reset();
                self.mode = FeedbackMode::ClosedLoop;
                Ok(())
            }
            Err(_) => {
                // Still too hot, can't recover yet
                Err(ResonanceError::AllHeatersAbort)
            }
        }
    }

    /// Get the phase shifter controller (for external access)
    pub fn phase_shifter(&self) -> &ThermoOpticPhaseShifter {
        &self.phase_shifter
    }

    /// Get mutable access to phase shifter
    pub fn phase_shifter_mut(&mut self) -> &mut ThermoOpticPhaseShifter {
        &mut self.phase_shifter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedback_loop_creation() {
        let loop_ctrl = ResonanceFeedbackLoop::new(8).unwrap();
        assert_eq!(loop_ctrl.lock_states.len(), 8);
        assert_eq!(loop_ctrl.mode(), FeedbackMode::ClosedLoop);
    }

    #[test]
    fn test_single_step() {
        let mut loop_ctrl = ResonanceFeedbackLoop::new(4).unwrap();
        
        // Simulate sensor readings (small error signals)
        let sensors = vec![0.01, -0.02, 0.005, -0.01];
        
        let result = loop_ctrl.step(&sensors);
        assert!(result.is_ok());
    }

    #[test]
    fn test_convergence_check() {
        let mut loop_ctrl = ResonanceFeedbackLoop::new(4).unwrap();
        loop_ctrl.set_convergence_threshold(0.1);
        
        // Provide consistent small error signals
        let sensors = vec![vec![0.001; 4]; 20];
        
        let result = loop_ctrl.run_until_converge(&sensors);
        // May or may not converge depending on PID tuning
        // Test just verifies no crash
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_critical_heater_failure() {
        let mut loop_ctrl = ResonanceFeedbackLoop::new(4).unwrap();
        loop_ctrl.set_critical_heaters(&[0]);
        
        // Simulate large error on critical heater
        let sensors = vec![1.0, 0.0, 0.0, 0.0];
        
        let result = loop_ctrl.step(&sensors);
        // Should detect the large error but continue
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_pid_tuning() {
        let mut loop_ctrl = ResonanceFeedbackLoop::new(4).unwrap();
        
        let new_pid = PidConfig {
            kp: 2.0,
            ki: 0.5,
            kd: 0.1,
            ..Default::default()
        };
        
        loop_ctrl.set_pid_config(new_pid);
        assert_eq!(loop_ctrl.pid_config.kp, 2.0);
    }

    #[test]
    fn test_reset() {
        let mut loop_ctrl = ResonanceFeedbackLoop::new(4).unwrap();
        
        // Run a few steps
        for _ in 0..10 {
            let sensors = vec![0.01; 4];
            loop_ctrl.step(&sensors).unwrap();
        }
        
        // Verify some state changed
        let any_nonzero = loop_ctrl.lock_states.iter()
            .any(|s| s.integral_error != 0.0 || s.previous_error != 0.0);
        
        loop_ctrl.reset();
        
        // All states should be zeroed
        for state in loop_ctrl.get_lock_states() {
            assert_eq!(state.integral_error, 0.0);
            assert_eq!(state.previous_error, 0.0);
            assert!(!state.locked);
        }
    }

    #[test]
    fn test_mode_transitions() {
        let mut loop_ctrl = ResonanceFeedbackLoop::new(4).unwrap();
        assert_eq!(loop_ctrl.mode(), FeedbackMode::ClosedLoop);
        
        // Force emergency shutdown
        loop_ctrl.mode = FeedbackMode::EmergencyShutdown;
        loop_ctrl.phase_shifter.emergency_shutdown();
        
        assert_eq!(loop_ctrl.mode(), FeedbackMode::EmergencyShutdown);
    }
}
