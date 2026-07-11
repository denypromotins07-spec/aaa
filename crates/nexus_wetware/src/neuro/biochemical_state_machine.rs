//! Biochemical State Machine for Neuromodulation Control
//! 
//! Manages state transitions between different biochemical/neuromodulatory
//! regimes in response to market conditions and organoid activity.

use crate::neuro::synaptic_gain_modulator::{NetworkState, GainError};
use crate::neuro::microfluidic_pump_controller::{BiochemicalAgent, MicrofluidicPumpController, PumpError};

/// Maximum number of state history entries
pub const MAX_HISTORY_SIZE: usize = 64;

/// Biochemical regime types
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum BiochemicalRegime {
    /// Normal baseline operation
    Baseline = 0,
    /// Enhanced dopamine (risk-seeking, learning)
    Dopaminergic = 1,
    /// Enhanced serotonin (stable, low variance)
    Serotonergic = 2,
    /// Enhanced cortisol (risk-averse, conservative)
    CortisolHigh = 3,
    /// Mixed excitation (glutamate dominant)
    Excitatory = 4,
    /// Mixed inhibition (GABA dominant)
    Inhibitory = 5,
    /// Emergency quenching state
    EmergencyQuench = 6,
}

/// Market condition inputs
#[derive(Debug, Clone, Copy)]
#[repr(C, align(32))]
pub struct MarketConditions {
    /// VIX or volatility index
    pub volatility: f32,
    /// Market direction (-1 to 1)
    pub trend: f32,
    /// Liquidity measure (0 to 1)
    pub liquidity: f32,
    /// Tail risk indicator
    pub tail_risk: f32,
}

impl Default for MarketConditions {
    fn default() -> Self {
        Self {
            volatility: 0.2,
            trend: 0.0,
            liquidity: 0.8,
            tail_risk: 0.0,
        }
    }
}

/// Organoid activity metrics
#[derive(Debug, Clone, Copy)]
#[repr(C, align(32))]
pub struct OrganoidMetrics {
    /// Mean firing rate (Hz)
    pub mean_firing_rate: f32,
    /// Burst rate (bursts/min)
    pub burst_rate: f32,
    /// Synchrony index (0 to 1)
    pub synchrony: f32,
    /// LFP power in gamma band
    pub gamma_power: f32,
    /// Seizure probability (0 to 1)
    pub seizure_probability: f32,
}

impl Default for OrganoidMetrics {
    fn default() -> Self {
        Self {
            mean_firing_rate: 10.0,
            burst_rate: 5.0,
            synchrony: 0.3,
            gamma_power: 1.0,
            seizure_probability: 0.0,
        }
    }
}

/// Error types for state machine operations
#[derive(Debug, Clone, Copy)]
pub enum StateMachineError {
    InvalidTransition,
    RegimeLocked,
    MetricOutOfRange,
    PumpCommunicationFailed,
    NotInitialized,
}

/// State transition record
#[repr(C)]
pub struct TransitionRecord {
    /// From regime
    pub from: BiochemicalRegime,
    /// To regime
    pub to: BiochemicalRegime,
    /// Timestamp (ns)
    pub timestamp_ns: u64,
    /// Trigger reason
    pub reason: u8,
}

/// Biochemical State Machine
pub struct BiochemicalStateMachine {
    /// Current regime
    current_regime: BiochemicalRegime,
    /// Target regime (pending transition)
    target_regime: Option<BiochemicalRegime>,
    /// Regime lock (prevents changes during critical operations)
    regime_locked: bool,
    /// Transition history
    history: [TransitionRecord; MAX_HISTORY_SIZE],
    history_idx: usize,
    /// Dwell time in current regime (ms)
    dwell_time_ms: u64,
    /// Minimum dwell time per regime
    min_dwell_times: [u64; 7],
    /// Last market conditions
    last_market: MarketConditions,
    /// Last organoid metrics
    last_metrics: OrganoidMetrics,
    /// Pump controller reference
    pump_controller: Option<MicrofluidicPumpController>,
    /// System enabled
    enabled: bool,
}

impl BiochemicalStateMachine {
    /// Create a new state machine
    pub fn new(initial_regime: BiochemicalRegime) -> Self {
        let mut sm = Self {
            current_regime: initial_regime,
            target_regime: None,
            regime_locked: false,
            history: [TransitionRecord {
                from: BiochemicalRegime::Baseline,
                to: BiochemicalRegime::Baseline,
                timestamp_ns: 0,
                reason: 0,
            }; MAX_HISTORY_SIZE],
            history_idx: 0,
            dwell_time_ms: 0,
            min_dwell_times: [100, 500, 500, 1000, 500, 500, 100], // ms per regime
            last_market: MarketConditions::default(),
            last_metrics: OrganoidMetrics::default(),
            pump_controller: None,
            enabled: false,
        };

        // Set default minimum dwell times
        sm.min_dwell_times[BiochemicalRegime::EmergencyQuench as usize] = 5000; // 5s minimum
        
        sm
    }

    /// Attach pump controller
    pub fn attach_pump_controller(&mut self, controller: MicrofluidicPumpController) {
        self.pump_controller = Some(controller);
    }

    /// Update market conditions
    pub fn update_market_conditions(&mut self, conditions: MarketConditions) {
        self.last_market = conditions;
        
        // Auto-transition based on market conditions
        self.evaluate_market_transitions();
    }

    /// Update organoid metrics
    pub fn update_organoid_metrics(&mut self, metrics: OrganoidMetrics) -> Result<(), StateMachineError> {
        if metrics.seizure_probability < 0.0 || metrics.seizure_probability > 1.0 {
            return Err(StateMachineError::MetricOutOfRange);
        }
        if metrics.synchrony < 0.0 || metrics.synchrony > 1.0 {
            return Err(StateMachineError::MetricOutOfRange);
        }

        self.last_metrics = metrics;

        // Check for seizure risk
        if metrics.seizure_probability > 0.8 {
            self.trigger_emergency_quench()?;
        }

        // Evaluate other transitions
        self.evaluate_metric_transitions();

        Ok(())
    }

    /// Evaluate transitions based on market conditions
    fn evaluate_market_transitions(&mut self) {
        let m = self.last_market;

        // High volatility + high tail risk -> cortisol regime
        if m.volatility > 0.5 && m.tail_risk > 0.5 {
            let _ = self.request_transition(BiochemicalRegime::CortisolHigh, 1);
        }
        // Low volatility + positive trend -> dopaminergic
        else if m.volatility < 0.2 && m.trend > 0.3 && m.liquidity > 0.5 {
            let _ = self.request_transition(BiochemicalRegime::Dopaminergic, 2);
        }
        // High liquidity, neutral trend -> serotonergic
        else if m.liquidity > 0.8 && m.trend.abs() < 0.1 {
            let _ = self.request_transition(BiochemicalRegime::Serotonergic, 3);
        }
    }

    /// Evaluate transitions based on organoid metrics
    fn evaluate_metric_transitions(&mut self) {
        let met = self.last_metrics;

        // High synchrony + high firing -> inhibitory
        if met.synchrony > 0.7 && met.mean_firing_rate > 50.0 {
            let _ = self.request_transition(BiochemicalRegime::Inhibitory, 4);
        }
        // Very low activity -> excitatory
        else if met.mean_firing_rate < 1.0 {
            let _ = self.request_transition(BiochemicalRegime::Excitatory, 5);
        }
    }

    /// Request a regime transition
    pub fn request_transition(
        &mut self,
        target: BiochemicalRegime,
        reason: u8,
    ) -> Result<(), StateMachineError> {
        if self.regime_locked {
            return Err(StateMachineError::RegimeLocked);
        }

        if target == self.current_regime {
            return Ok(()); // Already in target regime
        }

        // Validate transition
        if !self.is_valid_transition(self.current_regime, target) {
            return Err(StateMachineError::InvalidTransition);
        }

        // Check dwell time
        let min_dwell = self.min_dwell_times[self.current_regime as usize];
        if self.dwell_time_ms < min_dwell {
            return Err(StateMachineError::InvalidTransition); // Too soon
        }

        self.target_regime = Some(target);
        
        // Record transition request in history
        self.record_transition(self.current_regime, target, 0, reason);

        Ok(())
    }

    /// Execute pending transition
    pub fn execute_transition(&mut self, timestamp_ns: u64) -> Result<(), StateMachineError> {
        let target = match self.target_regime {
            Some(t) => t,
            None => return Ok(()), // No pending transition
        };

        // Apply regime change
        let old_regime = self.current_regime;
        self.current_regime = target;
        self.target_regime = None;
        self.dwell_time_ms = 0;

        // Record completed transition
        self.record_transition(old_regime, target, timestamp_ns, 0xFF);

        // Apply biochemical changes
        self.apply_regime_effects()?;

        Ok(())
    }

    /// Record transition in history
    fn record_transition(
        &mut self,
        from: BiochemicalRegime,
        to: BiochemicalRegime,
        timestamp_ns: u64,
        reason: u8,
    ) {
        self.history[self.history_idx] = TransitionRecord {
            from,
            to,
            timestamp_ns,
            reason,
        };
        self.history_idx = (self.history_idx + 1) % MAX_HISTORY_SIZE;
    }

    /// Check if transition is valid
    fn is_valid_transition(&self, from: BiochemicalRegime, to: BiochemicalRegime) -> bool {
        match (from, to) {
            // Emergency quench can be triggered from any state
            (_, BiochemicalRegime::EmergencyQuench) => true,
            // Emergency quench can only go to inhibitory or baseline
            (BiochemicalRegime::EmergencyQuench, BiochemicalRegime::Inhibitory) => true,
            (BiochemicalRegime::EmergencyQuench, BiochemicalRegime::Baseline) => true,
            (BiochemicalRegime::EmergencyQuench, _) => false,
            // Default: allow most transitions
            _ => true,
        }
    }

    /// Apply biochemical effects for current regime
    fn apply_regime_effects(&mut self) -> Result<(), StateMachineError> {
        let pump = match &mut self.pump_controller {
            Some(p) => p,
            None => return Ok(()), // No pump attached
        };

        match self.current_regime {
            BiochemicalRegime::Dopaminergic => {
                // Start dopamine infusion
                let _ = pump.configure_channel(0, BiochemicalAgent::Dopamine, 10.0);
                let _ = pump.start_pump(0);
            }
            BiochemicalRegime::Serotonergic => {
                // Start serotonin infusion
                let _ = pump.configure_channel(1, BiochemicalAgent::Serotonin, 5.0);
                let _ = pump.start_pump(1);
            }
            BiochemicalRegime::CortisolHigh => {
                // Start cortisol analogue
                let _ = pump.configure_channel(3, BiochemicalAgent::Cortisol, 50.0);
                let _ = pump.start_pump(3);
            }
            BiochemicalRegime::Inhibitory => {
                // GABA infusion
                let _ = pump.configure_channel(5, BiochemicalAgent::GABA, 20.0);
                let _ = pump.start_pump(5);
            }
            BiochemicalRegime::Excitatory => {
                // Glutamate infusion
                let _ = pump.configure_channel(4, BiochemicalAgent::Glutamate, 15.0);
                let _ = pump.start_pump(4);
            }
            BiochemicalRegime::EmergencyQuench => {
                // Maximum GABA + stop all excitatory agents
                let _ = pump.stop_pump(0); // Stop dopamine
                let _ = pump.stop_pump(4); // Stop glutamate
                let _ = pump.deliver_bolus(5, 1000.0, 100.0); // GABA bolus
            }
            BiochemicalRegime::Baseline => {
                // Return to saline perfusion
                let _ = pump.configure_channel(7, BiochemicalAgent::Saline, 10.0);
            }
        }

        Ok(())
    }

    /// Trigger emergency quench (seizure response)
    pub fn trigger_emergency_quench(&mut self) -> Result<(), StateMachineError> {
        if self.regime_locked {
            return Err(StateMachineError::RegimeLocked);
        }

        // Force transition to emergency quench
        let old_regime = self.current_regime;
        self.current_regime = BiochemicalRegime::EmergencyQuench;
        self.dwell_time_ms = 0;
        self.target_regime = None;

        self.record_transition(old_regime, BiochemicalRegime::EmergencyQuench, 0, 0xFE);
        self.apply_regime_effects()?;

        Ok(())
    }

    /// Lock regime (prevent changes)
    pub fn lock_regime(&mut self) {
        self.regime_locked = true;
    }

    /// Unlock regime
    pub fn unlock_regime(&mut self) {
        self.regime_locked = false;
    }

    /// Update dwell time (call periodically)
    pub fn update_dwell_time(&mut self, elapsed_ms: u64) {
        self.dwell_time_ms += elapsed_ms;
    }

    /// Get current regime
    pub fn get_regime(&self) -> BiochemicalRegime {
        self.current_regime
    }

    /// Get pending transition target
    pub fn get_pending_transition(&self) -> Option<BiochemicalRegime> {
        self.target_regime
    }

    /// Get dwell time in current regime
    pub fn get_dwell_time(&self) -> u64 {
        self.dwell_time_ms
    }

    /// Enable state machine
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable state machine (return to baseline)
    pub fn disable(&mut self) {
        self.enabled = false;
        let _ = self.request_transition(BiochemicalRegime::Baseline, 0);
        let _ = self.execute_transition(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_machine_initialization() {
        let sm = BiochemicalStateMachine::new(BiochemicalRegime::Baseline);
        assert_eq!(sm.get_regime(), BiochemicalRegime::Baseline);
        assert!(!sm.regime_locked);
    }

    #[test]
    fn test_valid_transitions() {
        let mut sm = BiochemicalStateMachine::new(BiochemicalRegime::Baseline);
        sm.dwell_time_ms = 1000; // Satisfy minimum dwell

        let result = sm.request_transition(BiochemicalRegime::Dopaminergic, 1);
        assert!(result.is_ok());

        let result = sm.execute_transition(0);
        assert!(result.is_ok());
        assert_eq!(sm.get_regime(), BiochemicalRegime::Dopaminergic);
    }

    #[test]
    fn test_emergency_quench() {
        let mut sm = BiochemicalStateMachine::new(BiochemicalRegime::Dopaminergic);

        let result = sm.trigger_emergency_quench();
        assert!(result.is_ok());
        assert_eq!(sm.get_regime(), BiochemicalRegime::EmergencyQuench);
    }

    #[test]
    fn test_invalid_transition_timing() {
        let mut sm = BiochemicalStateMachine::new(BiochemicalRegime::Baseline);
        sm.dwell_time_ms = 10; // Less than minimum

        let result = sm.request_transition(BiochemicalRegime::Dopaminergic, 1);
        assert!(matches!(result, Err(StateMachineError::InvalidTransition)));
    }

    #[test]
    fn test_regime_lock() {
        let mut sm = BiochemicalStateMachine::new(BiochemicalRegime::Baseline);
        sm.lock_regime();

        let result = sm.request_transition(BiochemicalRegime::Dopaminergic, 1);
        assert!(matches!(result, Err(StateMachineError::RegimeLocked)));
    }
}
