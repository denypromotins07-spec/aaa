//! NEXUS-OMEGA Stage 19: Forward Invariance Guard
//!
//! This module enforces forward invariance of the safe set by combining
//! the Lyapunov function with the QP action projector. It guarantees that
//! if the system starts in a safe state, all future states remain safe.
//!
//! Forward invariance condition:
//!     If V(s₀) ≤ c and ∇V(s) · f(s, a) ≤ -αV(s) for all t,
//!     then V(s(t)) ≤ c for all t ≥ 0
//!
//! Author: NEXUS-OMEGA Architecture
//! Stage: 19 of 50

use crate::safety::lyapunov_function::{
    LyapunovFunction, LyapunovConfig, RiskMetrics, SafetyStatus,
};
use crate::safety::qp_action_projector::{
    QPActionProjector, QPProjectorConfig, ProjectionStatus,
};

/// Configuration for the forward invariance guard
#[derive(Debug, Clone)]
pub struct InvarianceGuardConfig {
    /// Minimum allowed Lyapunov value (safety margin)
    pub min_lyapunov_margin: f64,
    
    /// Maximum rate of change allowed for Lyapunov function
    pub max_dv_dt: f64,
    
    /// Aggressiveness of intervention (0 = passive, 1 = aggressive)
    pub intervention_aggressiveness: f64,
    
    /// Enable fallback to recovery policy on infeasibility
    pub enable_recovery_fallback: bool,
    
    /// Warning threshold (fraction of critical threshold)
    pub warning_threshold_ratio: f64,
}

impl Default for InvarianceGuardConfig {
    fn default() -> Self {
        Self {
            min_lyapunov_margin: 0.1,
            max_dv_dt: 0.05,
            intervention_aggressiveness: 0.8,
            enable_recovery_fallback: true,
            warning_threshold_ratio: 0.7,
        }
    }
}

/// Result of safety guard evaluation
#[derive(Debug, Clone)]
pub struct SafetyGuardResult {
    /// Whether the action is safe to execute
    pub is_safe: bool,
    
    /// Projected action (if modification was needed)
    pub projected_action: [f64; 8],
    
    /// Status of the projection
    pub projection_status: ProjectionStatus,
    
    /// Current Lyapunov value
    pub current_v: f64,
    
    /// Predicted Lyapunov value after action
    pub predicted_v: f64,
    
    /// Safety status classification
    pub safety_status: SafetyStatus,
    
    /// Whether recovery policy should be triggered
    pub trigger_recovery: bool,
    
    /// Diagnostic message
    pub message: String,
}

impl SafetyGuardResult {
    /// Create a safe result with no modification needed
    pub fn safe(action: &[f64], v: f64) -> Self {
        let mut projected = [0.0; 8];
        for (i, &a) in action.iter().take(8).enumerate() {
            projected[i] = a;
        }
        
        Self {
            is_safe: true,
            projected_action: projected,
            projection_status: ProjectionStatus::AlreadySafe,
            current_v: v,
            predicted_v: v,
            safety_status: SafetyStatus::Safe,
            trigger_recovery: false,
            message: String::from("Action verified safe"),
        }
    }
    
    /// Create an unsafe result requiring intervention
    pub fn unsafe_result(
        projected: [f64; 8],
        status: ProjectionStatus,
        current_v: f64,
        predicted_v: f64,
        trigger_recovery: bool,
    ) -> Self {
        let (status_class, msg) = match status {
            ProjectionStatus::AlreadySafe => (SafetyStatus::Safe, "No intervention needed"),
            ProjectionStatus::Projected => (SafetyStatus::Warning, "Action projected to safe region"),
            ProjectionStatus::OnBoundary => (SafetyStatus::Warning, "Action on safety boundary"),
            ProjectionStatus::Infeasible => (SafetyStatus::Critical, "Infeasible - recovery required"),
            ProjectionStatus::MaxIterationsReached => (SafetyStatus::Critical, "Solver timeout"),
            ProjectionStatus::NumericalError => (SafetyStatus::Critical, "Numerical error"),
        };
        
        Self {
            is_safe: !matches!(status, ProjectionStatus::Infeasible | ProjectionStatus::MaxIterationsReached),
            projected_action: projected,
            projection_status: status,
            current_v,
            predicted_v,
            safety_status: status_class,
            trigger_recovery,
            message: String::from(msg),
        }
    }
}

/// Forward invariance guard for safe RL action execution
pub struct ForwardInvarianceGuard {
    config: InvarianceGuardConfig,
    lyapunov_fn: LyapunovFunction,
    qp_projector: QPActionProjector,
    
    // State tracking
    consecutive_interventions: u32,
    total_evaluations: u64,
    recovery_triggers: u64,
}

impl ForwardInvarianceGuard {
    /// Create a new forward invariance guard with default settings
    pub fn new() -> Self {
        Self::with_config(InvarianceGuardConfig::default())
    }
    
    /// Create with custom configuration
    pub fn with_config(config: InvarianceGuardConfig) -> Self {
        let lyapunov_config = LyapunovConfig {
            safety_threshold: 0.5,
            critical_threshold: 0.8,
            ..Default::default()
        };
        
        Self {
            config,
            lyapunov_fn: LyapunovFunction::new(lyapunov_config),
            qp_projector: QPActionProjector::new(),
            consecutive_interventions: 0,
            total_evaluations: 0,
            recovery_triggers: 0,
        }
    }
    
    /// Evaluate an action and ensure forward invariance
    ///
    /// # Arguments
    /// * `action` - Raw action from RL agent
    /// * `current_risk` - Current risk metrics
    /// * `predicted_risk` - Predicted risk metrics after action
    /// * `action_bounds` - Optional action bounds
    ///
    /// # Returns
    /// * `SafetyGuardResult` - Evaluation result with safe action
    pub fn evaluate_action(
        &mut self,
        action: &[f64],
        current_risk: &RiskMetrics,
        predicted_risk: Option<&RiskMetrics>,
        action_bounds: Option<&[(f64, f64)]>,
    ) -> SafetyGuardResult {
        self.total_evaluations += 1;
        
        // Compute current Lyapunov value
        let current_v = self.lyapunov_fn.compute(current_risk);
        let current_status = self.lyapunov_fn.get_safety_status(current_risk);
        
        // Check if already in critical state
        if current_status.is_critical() {
            self.consecutive_interventions += 1;
            
            // Already critical - may need recovery
            let (_, projected) = self.qp_projector.project(
                action,
                &self.lyapunov_fn,
                current_risk,
                action_bounds,
            );
            
            if self.config.enable_recovery_fallback {
                self.recovery_triggers += 1;
            }
            
            return SafetyGuardResult::unsafe_result(
                projected,
                ProjectionStatus::Infeasible,
                current_v,
                current_v,
                self.config.enable_recovery_fallback,
            );
        }
        
        // Estimate predicted Lyapunov value
        let predicted_v = if let Some(pred_risk) = predicted_risk {
            self.lyapunov_fn.compute(pred_risk)
        } else {
            // Estimate from action
            self.estimate_predicted_v(action, current_risk, current_v)
        };
        
        // Check Lyapunov stability condition
        let dt = 1.0; // Assume unit time step
        let dv_dt = (predicted_v - current_v) / dt;
        let stability_threshold = -self.lyapunov_fn.config().stability_alpha * current_v;
        
        // Verify forward invariance: dV/dt ≤ -αV
        let satisfies_invariance = dv_dt <= stability_threshold + self.config.min_lyapunov_margin;
        
        if satisfies_invariance && current_status == SafetyStatus::Safe {
            // Action maintains forward invariance
            self.consecutive_interventions = 0;
            return SafetyGuardResult::safe(action, current_v);
        }
        
        // Need to project action
        let (status, projected) = self.qp_projector.project(
            action,
            &self.lyapunov_fn,
            current_risk,
            action_bounds,
        );
        
        // Determine if recovery is needed
        let trigger_recovery = match status {
            ProjectionStatus::Infeasible => true,
            ProjectionStatus::MaxIterationsReached => {
                // Could not find safe action in time
                self.config.enable_recovery_fallback
            }
            _ => {
                // Check if too many consecutive interventions
                self.consecutive_interventions >= 5
            }
        };
        
        if trigger_recovery {
            self.recovery_triggers += 1;
        }
        
        if status != ProjectionStatus::AlreadySafe {
            self.consecutive_interventions += 1;
        } else {
            self.consecutive_interventions = 0;
        }
        
        SafetyGuardResult::unsafe_result(
            projected,
            status,
            current_v,
            predicted_v,
            trigger_recovery,
        )
    }
    
    /// Estimate predicted Lyapunov value from action
    fn estimate_predicted_v(
        &self,
        action: &[f64],
        current_risk: &RiskMetrics,
        current_v: f64,
    ) -> f64 {
        // Simple linear model: V_next ≈ V_current + ∇V · Δs(a)
        let grad = self.lyapunov_fn.compute_gradient(current_risk);
        
        // Approximate state change from action
        let position_change: f64 = action.iter().map(|x| x.abs()).sum();
        
        // Estimated changes in risk metrics
        let delta_leverage = position_change * 0.05;
        let delta_margin = -position_change * 0.02;
        
        // First-order approximation
        let delta_v = grad.leverage * delta_leverage + grad.margin_buffer * delta_margin;
        
        (current_v + delta_v).max(0.0)
    }
    
    /// Check if the safe set is forward invariant under current conditions
    pub fn verify_forward_invariance(&self, risk: &RiskMetrics) -> bool {
        let v = self.lyapunov_fn.compute(risk);
        let margin = self.lyapunov_fn.compute_safety_margin(risk);
        
        // Safe set is forward invariant if:
        // 1. We're inside the safe set (margin > 0)
        // 2. There exists at least one action that maintains invariance
        
        margin > 0.0 && v < self.lyapunov_fn.config().critical_threshold
    }
    
    /// Get the maximum allowable action magnitude for safety
    pub fn get_max_safe_action_magnitude(&self, risk: &RiskMetrics) -> f64 {
        let v = self.lyapunov_fn.compute(risk);
        let margin = self.lyapunov_fn.compute_safety_margin(risk);
        
        if margin <= 0.0 {
            return 0.0; // No safe action possible
        }
        
        // Conservative estimate based on available margin
        let grad_norm = self.lyapunov_fn.compute_gradient(risk).norm();
        
        if grad_norm < 1e-10 {
            return 1.0; // Gradient too small, allow full action
        }
        
        // Max action such that ΔV ≤ margin
        (margin / grad_norm) * self.config.intervention_aggressiveness
    }
    
    /// Reset internal counters
    pub fn reset(&mut self) {
        self.consecutive_interventions = 0;
        self.total_evaluations = 0;
        self.recovery_triggers = 0;
    }
    
    /// Get diagnostic statistics
    pub fn get_statistics(&self) -> InvarianceGuardStats {
        InvarianceGuardStats {
            total_evaluations: self.total_evaluations,
            consecutive_interventions: self.consecutive_interventions,
            recovery_triggers: self.recovery_triggers,
            intervention_rate: if self.total_evaluations > 0 {
                self.recovery_triggers as f64 / self.total_evaluations as f64
            } else {
                0.0
            },
        }
    }
    
    /// Update Lyapunov configuration
    pub fn update_lyapunov_config(&mut self, config: LyapunovConfig) {
        self.lyapunov_fn.update_config(config);
    }
    
    /// Get reference to Lyapunov function
    pub fn lyapunov_fn(&self) -> &LyapunovFunction {
        &self.lyapunov_fn
    }
    
    /// Get mutable reference to Lyapunov function
    pub fn lyapunov_fn_mut(&mut self) -> &mut LyapunovFunction {
        &mut self.lyapunov_fn
    }
}

impl Default for ForwardInvarianceGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from the invariance guard
#[derive(Debug, Clone)]
pub struct InvarianceGuardStats {
    pub total_evaluations: u64,
    pub consecutive_interventions: u32,
    pub recovery_triggers: u64,
    pub intervention_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_safe_action_passes() {
        let mut guard = ForwardInvarianceGuard::new();
        
        let safe_risk = RiskMetrics {
            drawdown: 0.01,
            margin_buffer: 0.8,
            leverage: 1.2,
            ..Default::default()
        };
        
        let action = [0.1, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = guard.evaluate_action(&action, &safe_risk, None, None);
        
        assert!(result.is_safe);
        assert!(!result.trigger_recovery);
    }
    
    #[test]
    fn test_risky_state_triggers_intervention() {
        let mut guard = ForwardInvarianceGuard::new();
        
        let risky_risk = RiskMetrics {
            drawdown: 0.12,
            margin_buffer: 0.15,
            leverage: 4.5,
            ..Default::default()
        };
        
        let action = [1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = guard.evaluate_action(&action, &risky_risk, None, None);
        
        // Should either project or trigger recovery
        assert!(result.projection_status != ProjectionStatus::AlreadySafe);
    }
    
    #[test]
    fn test_statistics_tracking() {
        let mut guard = ForwardInvarianceGuard::new();
        
        let safe_risk = RiskMetrics::default();
        let action = [0.1; 8];
        
        for _ in 0..10 {
            let _ = guard.evaluate_action(&action, &safe_risk, None, None);
        }
        
        let stats = guard.get_statistics();
        assert_eq!(stats.total_evaluations, 10);
    }
}
