//! Doomsday State Machine for crisis mode transitions
//!
//! Monitors EVT, Copula, and Hawkes signals to detect when the system
//! should transition into Survival Mode, activating convex hedges and
//! de-risking the portfolio.

use crate::tail::extreme_value_theory::{EvtFitResult, ExtremeValueTheory};
use crate::dependence::tail_dependence_metric::TailDependenceResult;
use crate::contagion::multivariate_hawkes::MultivariateHawkesProcess;
use thiserror::Error;

/// Errors from state machine operations
#[derive(Error, Debug, Clone)]
pub enum StateMachineError {
    #[error("Invalid threshold configuration")]
    InvalidThreshold,
    
    #[error("Signal not available: {0}")]
    SignalNotAvailable(String),
    
    #[error("Transition rejected: {reason}")]
    TransitionRejected { reason: String },
}

/// System state in the doomsday hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoomsdayState {
    /// Normal operation - all systems green
    Normal,
    /// Elevated vigilance - monitoring closely
    Vigilant,
    /// Warning signs detected - preparing defenses
    Warning,
    /// Critical threat - partial hedge activation
    Critical,
    /// Full survival mode - maximum protection
    Survival,
}

impl DoomsdayState {
    /// Get numeric severity level (higher = worse)
    pub fn severity(&self) -> u8 {
        match self {
            DoomsdayState::Normal => 0,
            DoomsdayState::Vigilant => 1,
            DoomsdayState::Warning => 2,
            DoomsdayState::Critical => 3,
            DoomsdayState::Survival => 4,
        }
    }
    
    /// Check if state indicates crisis conditions
    pub fn is_crisis(&self) -> bool {
        matches!(self, DoomsdayState::Critical | DoomsdayState::Survival)
    }
}

/// Thresholds for state transitions
#[derive(Debug, Clone)]
pub struct TransitionThresholds {
    /// EVT shape parameter threshold (ξ > value triggers escalation)
    pub evt_shape_threshold: f64,
    /// Tail dependence coefficient threshold
    pub tail_dependence_threshold: f64,
    /// Hawkes intensity multiplier threshold (current/baseline)
    pub hawkes_intensity_threshold: f64,
    /// Combined signal threshold for Survival mode
    pub survival_combined_threshold: f64,
    /// Cooldown period before downgrading state (seconds)
    pub downgrade_cooldown_secs: f64,
}

impl Default for TransitionThresholds {
    fn default() -> Self {
        Self {
            evt_shape_threshold: 0.3,
            tail_dependence_threshold: 0.5,
            hawkes_intensity_threshold: 3.0,
            survival_combined_threshold: 0.8,
            downgrade_cooldown_secs: 300.0, // 5 minutes
        }
    }
}

/// Aggregated risk signals for decision making
#[derive(Debug, Clone)]
pub struct RiskSignals {
    /// EVT shape parameter (ξ)
    pub evt_shape: Option<f64>,
    /// EVT-based VaR estimate
    pub evt_var: Option<f64>,
    /// Tail dependence coefficient
    pub tail_dependence: Option<f64>,
    /// Hawkes intensity ratio (current/baseline)
    pub hawkes_intensity_ratio: Option<f64>,
    /// Liquidity evaporation flag
    pub liquidity_evaporated: Option<bool>,
    /// Combined risk score (0-1)
    pub combined_risk_score: f64,
}

impl RiskSignals {
    /// Calculate combined risk score from individual signals
    pub fn calculate_combined(&mut self) -> f64 {
        let mut score = 0.0;
        let mut weight_sum = 0.0;
        
        // EVT shape contribution (heavy tails = higher risk)
        if let Some(xi) = self.evt_shape {
            let xi_score = (xi / 0.5).min(1.0).max(0.0);
            score += xi_score * 0.3;
            weight_sum += 0.3;
        }
        
        // Tail dependence contribution
        if let Some(lambda) = self.tail_dependence {
            let td_score = lambda.min(1.0).max(0.0);
            score += td_score * 0.3;
            weight_sum += 0.3;
        }
        
        // Hawkes intensity contribution
        if let Some(ratio) = self.hawkes_intensity_ratio {
            let hawkes_score = (ratio / 5.0).min(1.0).max(0.0);
            score += hawkes_score * 0.2;
            weight_sum += 0.2;
        }
        
        // Liquidity evaporation (binary but heavily weighted)
        if let Some(evaporated) = self.liquidity_evaporated {
            if evaporated {
                score += 0.2;
            }
            weight_sum += 0.2;
        }
        
        if weight_sum > 0.0 {
            self.combined_risk_score = score / weight_sum;
        }
        
        self.combined_risk_score
    }
}

/// Doomsday State Machine for crisis detection and response
pub struct DoomsdayStateMachine {
    current_state: DoomsdayState,
    thresholds: TransitionThresholds,
    /// Time of last state transition
    last_transition_time: f64,
    /// History of states for hysteresis
    state_history: Vec<(f64, DoomsdayState)>,
    /// Maximum history size
    max_history: usize,
}

impl DoomsdayStateMachine {
    /// Create a new state machine with default thresholds
    pub fn new() -> Self {
        Self::with_thresholds(TransitionThresholds::default())
    }
    
    /// Create a new state machine with custom thresholds
    pub fn with_thresholds(thresholds: TransitionThresholds) -> Self {
        Self {
            current_state: DoomsdayState::Normal,
            thresholds,
            last_transition_time: 0.0,
            state_history: Vec::new(),
            max_history: 100,
        }
    }
    
    /// Evaluate current signals and potentially transition state
    pub fn evaluate_and_transition(
        &mut self,
        signals: &RiskSignals,
        current_time: f64,
    ) -> Result<DoomsdayState, StateMachineError> {
        // Calculate combined risk score
        let risk_score = signals.combined_risk_score;
        
        // Determine target state based on signals
        let target_state = self.determine_target_state(signals, risk_score)?;
        
        // Apply hysteresis: harder to downgrade than upgrade
        let new_state = if target_state.severity() > self.current_state.severity() {
            // Upgrade immediately if signals warrant
            target_state
        } else if target_state.severity() < self.current_state.severity() {
            // Downgrade only after cooldown
            let time_since_transition = current_time - self.last_transition_time;
            
            if time_since_transition >= self.thresholds.downgrade_cooldown_secs {
                // Also require sustained low risk
                let recent_avg = self.recent_average_risk();
                if recent_avg < risk_score * 0.7 {
                    target_state
                } else {
                    self.current_state // Stay put
                }
            } else {
                self.current_state // Still in cooldown
            }
        } else {
            target_state // Same state
        };
        
        // Record transition if state changed
        if new_state != self.current_state {
            self.current_state = new_state;
            self.last_transition_time = current_time;
            self.record_state(current_time, new_state);
        }
        
        Ok(new_state)
    }
    
    /// Determine target state based on risk signals
    fn determine_target_state(
        &self,
        signals: &RiskSignals,
        risk_score: f64,
    ) -> Result<DoomsdayState, StateMachineError> {
        // Check for Survival mode: multiple extreme signals
        let survival_conditions = [
            signals.evt_shape.map(|x| x > self.thresholds.evt_shape_threshold * 1.5).unwrap_or(false),
            signals.tail_dependence.map(|x| x > self.thresholds.tail_dependence_threshold * 1.5).unwrap_or(false),
            signals.hawkes_intensity_ratio.map(|x| x > self.thresholds.hawkes_intensity_threshold * 2.0).unwrap_or(false),
            signals.liquidity_evaporated.unwrap_or(false),
        ];
        
        let survival_count = survival_conditions.iter().filter(|&&x| x).count();
        
        if survival_count >= 2 || risk_score > self.thresholds.survival_combined_threshold {
            return Ok(DoomsdayState::Survival);
        }
        
        // Check for Critical mode
        let critical_conditions = [
            signals.evt_shape.map(|x| x > self.thresholds.evt_shape_threshold).unwrap_or(false),
            signals.tail_dependence.map(|x| x > self.thresholds.tail_dependence_threshold).unwrap_or(false),
            signals.hawkes_intensity_ratio.map(|x| x > self.thresholds.hawkes_intensity_threshold).unwrap_or(false),
        ];
        
        let critical_count = critical_conditions.iter().filter(|&&x| x).count();
        
        if critical_count >= 2 || risk_score > self.thresholds.survival_combined_threshold * 0.7 {
            return Ok(DoomsdayState::Critical);
        }
        
        // Check for Warning mode
        let warning_conditions = [
            signals.evt_shape.map(|x| x > self.thresholds.evt_shape_threshold * 0.5).unwrap_or(false),
            signals.tail_dependence.map(|x| x > self.thresholds.tail_dependence_threshold * 0.5).unwrap_or(false),
            signals.hawkes_intensity_ratio.map(|x| x > 1.5).unwrap_or(false),
        ];
        
        let warning_count = warning_conditions.iter().filter(|&&x| x).count();
        
        if warning_count >= 2 || risk_score > 0.3 {
            return Ok(DoomsdayState::Warning);
        }
        
        // Check for Vigilant mode
        if signals.evt_shape.is_some() && signals.evt_shape.unwrap() > 0.1
            || signals.tail_dependence.is_some() && signals.tail_dependence.unwrap() > 0.2
            || risk_score > 0.1
        {
            return Ok(DoomsdayState::Vigilant);
        }
        
        Ok(DoomsdayState::Normal)
    }
    
    /// Record state in history
    fn record_state(&mut self, time: f64, state: DoomsdayState) {
        self.state_history.push((time, state));
        
        // Trim history if needed
        while self.state_history.len() > self.max_history {
            self.state_history.remove(0);
        }
    }
    
    /// Calculate average risk score over recent history
    fn recent_average_risk(&self) -> f64 {
        if self.state_history.is_empty() {
            return 0.0;
        }
        
        let recent_count = self.state_history.len().min(10);
        let start_idx = self.state_history.len() - recent_count;
        
        let sum: f64 = self.state_history[start_idx..]
            .iter()
            .map(|(_, s)| s.severity() as f64)
            .sum();
        
        sum / recent_count as f64
    }
    
    /// Get current state
    pub fn current_state(&self) -> DoomsdayState {
        self.current_state
    }
    
    /// Check if system is in survival mode
    pub fn is_survival_mode(&self) -> bool {
        self.current_state == DoomsdayState::Survival
    }
    
    /// Get recommended hedge allocation fraction based on state
    pub fn recommended_hedge_fraction(&self) -> f64 {
        match self.current_state {
            DoomsdayState::Normal => 0.02,   // 2% baseline
            DoomsdayState::Vigilant => 0.05, // 5%
            DoomsdayState::Warning => 0.10,  // 10%
            DoomsdayState::Critical => 0.20, // 20%
            DoomsdayState::Survival => 0.35, // 35% maximum
        }
    }
    
    /// Reset state machine to normal
    pub fn reset(&mut self) {
        self.current_state = DoomsdayState::Normal;
        self.last_transition_time = 0.0;
        self.state_history.clear();
    }
}

impl Default for DoomsdayStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_state_machine_transitions() {
        let mut sm = DoomsdayStateMachine::new();
        
        assert_eq!(sm.current_state(), DoomsdayState::Normal);
        
        // Test transition to Vigilant
        let signals = RiskSignals {
            evt_shape: Some(0.15),
            tail_dependence: Some(0.25),
            hawkes_intensity_ratio: Some(1.2),
            liquidity_evaporated: Some(false),
            combined_risk_score: 0.15,
        };
        
        let new_state = sm.evaluate_and_transition(&signals, 100.0).unwrap();
        assert_eq!(new_state, DoomsdayState::Vigilant);
    }
    
    #[test]
    fn test_survival_mode_detection() {
        let mut sm = DoomsdayStateMachine::new();
        
        // Strong crisis signals
        let signals = RiskSignals {
            evt_shape: Some(0.6),
            tail_dependence: Some(0.8),
            hawkes_intensity_ratio: Some(8.0),
            liquidity_evaporated: Some(true),
            combined_risk_score: 0.9,
        };
        
        let new_state = sm.evaluate_and_transition(&signals, 100.0).unwrap();
        assert_eq!(new_state, DoomsdayState::Survival);
        assert!(sm.is_survival_mode());
    }
    
    #[test]
    fn test_recommended_hedge_allocation() {
        let sm = DoomsdayStateMachine::new();
        assert!((sm.recommended_hedge_fraction() - 0.02).abs() < 0.001);
        
        // Simulate being in survival mode
        let mut sm_survival = DoomsdayStateMachine::with_thresholds(TransitionThresholds::default());
        sm_survival.current_state = DoomsdayState::Survival;
        assert!((sm_survival.recommended_hedge_fraction() - 0.35).abs() < 0.001);
    }
}
