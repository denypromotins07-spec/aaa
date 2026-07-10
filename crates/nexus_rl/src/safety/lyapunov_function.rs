//! NEXUS-OMEGA Stage 19: Lyapunov Function for Trading Safety
//!
//! This module implements a Lyapunov function V(s) that mathematically bounds
//! the distance to the "ruin state" (margin call). The Lyapunov function is
//! constructed from Stage 5 Risk Engine metrics and Stage 11 EVT (Extreme
//! Value Theory) components.
//!
//! The Lyapunov function satisfies:
//! - V(s) > 0 for all non-terminal states
//! - V(s) = 0 if and only if s is the ruin state
//! - dV/dt ≤ -αV(s) for some α > 0 (stability condition)
//!
//! Author: NEXUS-OMEGA Architecture
//! Stage: 19 of 50

use std::sync::Arc;

/// Risk metrics required for Lyapunov function computation
#[derive(Debug, Clone)]
pub struct RiskMetrics {
    /// Current drawdown from peak equity (0.0 to 1.0)
    pub drawdown: f64,
    
    /// Value at Risk at 95% confidence (as fraction of equity)
    pub var_95: f64,
    
    /// Expected Shortfall / CVaR (as fraction of equity)
    pub expected_shortfall: f64,
    
    /// Current leverage ratio (gross exposure / equity)
    pub leverage: f64,
    
    /// Portfolio volatility (annualized)
    pub volatility: f64,
    
    /// Distance to margin call (normalized, 0 = margin call)
    pub margin_buffer: f64,
    
    /// Extreme value index from EVT analysis
    pub evi: f64,
    
    /// Liquidity score (0 = illiquid, 1 = highly liquid)
    pub liquidity_score: f64,
}

impl Default for RiskMetrics {
    fn default() -> Self {
        Self {
            drawdown: 0.0,
            var_95: 0.01,
            expected_shortfall: 0.015,
            leverage: 1.0,
            volatility: 0.15,
            margin_buffer: 1.0,
            evi: 0.1,
            liquidity_score: 0.8,
        }
    }
}

/// Configuration for Lyapunov function weights
#[derive(Debug, Clone)]
pub struct LyapunovConfig {
    /// Weight for drawdown term
    pub drawdown_weight: f64,
    
    /// Weight for VaR term
    pub var_weight: f64,
    
    /// Weight for leverage term
    pub leverage_weight: f64,
    
    /// Weight for margin buffer term (inverse)
    pub margin_weight: f64,
    
    /// Weight for EVT tail risk term
    pub evt_weight: f64,
    
    /// Weight for liquidity term
    pub liquidity_weight: f64,
    
    /// Stability coefficient α in dV/dt ≤ -αV
    pub stability_alpha: f64,
    
    /// Maximum acceptable Lyapunov value before safety intervention
    pub safety_threshold: f64,
    
    /// Critical threshold triggering recovery policy
    pub critical_threshold: f64,
}

impl Default for LyapunovConfig {
    fn default() -> Self {
        Self {
            drawdown_weight: 2.0,
            var_weight: 1.5,
            leverage_weight: 1.0,
            margin_weight: 3.0,
            evt_weight: 1.0,
            liquidity_weight: 0.5,
            stability_alpha: 0.1,
            safety_threshold: 0.7,
            critical_threshold: 0.9,
        }
    }
}

/// Lyapunov function for trading safety verification
///
/// Computes V(s) which measures distance to ruin state.
/// Lower values indicate safer states.
pub struct LyapunovFunction {
    config: LyapunovConfig,
}

impl LyapunovFunction {
    /// Create a new Lyapunov function with given configuration
    pub fn new(config: LyapunovConfig) -> Self {
        Self { config }
    }
    
    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(LyapunovConfig::default())
    }
    
    /// Compute the Lyapunov function value V(s) for current state
    ///
    /// Returns value in [0, ∞) where:
    /// - 0 = ruin state (margin call)
    /// - < safety_threshold = safe operating region
    /// - >= safety_threshold = warning region
    /// - >= critical_threshold = danger region, recovery may be needed
    pub fn compute(&self, metrics: &RiskMetrics) -> f64 {
        // Component-wise Lyapunov terms
        
        // Drawdown term: increases with drawdown
        // Uses exponential penalty for large drawdowns
        let drawdown_term = self.config.drawdown_weight 
            * (metrics.drawdown.powi(2));
        
        // VaR term: penalizes high value-at-risk
        let var_term = self.config.var_weight 
            * (metrics.var_95.powi(2));
        
        // Leverage term: penalizes excessive leverage
        // Leverage > 1 contributes positively, leverage < 1 is neutral
        let leverage_excess = (metrics.leverage - 1.0).max(0.0);
        let leverage_term = self.config.leverage_weight 
            * leverage_excess.powi(2);
        
        // Margin buffer term: inverse relationship
        // As margin_buffer → 0, this term → ∞
        let margin_term = if metrics.margin_buffer > 1e-6 {
            self.config.margin_weight * (1.0 / metrics.margin_buffer - 1.0).powi(2)
        } else {
            // Approaching ruin: very large penalty
            1e6
        };
        
        // EVT tail risk term: accounts for fat tails
        let evt_term = self.config.evt_weight 
            * metrics.evi.powi(2);
        
        // Liquidity term: low liquidity increases risk
        let liquidity_term = self.config.liquidity_weight 
            * (1.0 - metrics.liquidity_score).powi(2);
        
        // Combine all terms
        let v = drawdown_term 
            + var_term 
            + leverage_term 
            + margin_term 
            + evt_term 
            + liquidity_term;
        
        // Ensure non-negative (numerical stability)
        v.max(0.0)
    }
    
    /// Compute the time derivative dV/dt given state transition
    ///
    /// Used to verify Lyapunov stability condition: dV/dt ≤ -αV
    pub fn compute_derivative(
        &self,
        current_metrics: &RiskMetrics,
        next_metrics: &RiskMetrics,
        dt: f64,
    ) -> f64 {
        let v_current = self.compute(current_metrics);
        let v_next = self.compute(next_metrics);
        
        // Finite difference approximation
        (v_next - v_current) / dt
    }
    
    /// Check if the Lyapunov stability condition is satisfied
    ///
    /// Condition: dV/dt ≤ -αV(s)
    /// If satisfied, the system is converging toward safety
    pub fn check_stability(
        &self,
        current_metrics: &RiskMetrics,
        next_metrics: &RiskMetrics,
        dt: f64,
    ) -> bool {
        let v = self.compute(current_metrics);
        let dv_dt = self.compute_derivative(current_metrics, next_metrics, dt);
        
        // Stability condition: dV/dt ≤ -αV
        dv_dt <= -self.config.stability_alpha * v
    }
    
    /// Get the safety status based on Lyapunov value
    pub fn get_safety_status(&self, metrics: &RiskMetrics) -> SafetyStatus {
        let v = self.compute(metrics);
        
        if v >= self.config.critical_threshold {
            SafetyStatus::Critical
        } else if v >= self.config.safety_threshold {
            SafetyStatus::Warning
        } else {
            SafetyStatus::Safe
        }
    }
    
    /// Compute safety margin (how much room before violation)
    ///
    /// Positive = safe, Negative = violated
    pub fn compute_safety_margin(&self, metrics: &RiskMetrics) -> f64 {
        self.config.safety_threshold - self.compute(metrics)
    }
    
    /// Get gradient of V with respect to key risk metrics
    ///
    /// Used for safe action projection (Chapter 2)
    pub fn compute_gradient(&self, metrics: &RiskMetrics) -> LyapunovGradient {
        let eps = 1e-8;
        
        // Numerical gradient computation
        let mut base_metrics = metrics.clone();
        
        // Gradient w.r.t. drawdown
        base_metrics.drawdown += eps;
        let v_up = self.compute(&base_metrics);
        base_metrics.drawdown = metrics.drawdown;
        let grad_drawdown = (v_up - self.compute(metrics)) / eps;
        
        // Gradient w.r.t. VaR
        base_metrics.var_95 += eps;
        let v_up = self.compute(&base_metrics);
        base_metrics.var_95 = metrics.var_95;
        let grad_var = (v_up - self.compute(metrics)) / eps;
        
        // Gradient w.r.t. leverage
        base_metrics.leverage += eps;
        let v_up = self.compute(&base_metrics);
        base_metrics.leverage = metrics.leverage;
        let grad_leverage = (v_up - self.compute(metrics)) / eps;
        
        // Gradient w.r.t. margin buffer
        base_metrics.margin_buffer += eps;
        let v_up = self.compute(&base_metrics);
        base_metrics.margin_buffer = metrics.margin_buffer;
        let grad_margin = (v_up - self.compute(metrics)) / eps;
        
        LyapunovGradient {
            drawdown: grad_drawdown,
            var_95: grad_var,
            leverage: grad_leverage,
            margin_buffer: grad_margin,
        }
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: LyapunovConfig) {
        self.config = config;
    }
    
    /// Get current configuration
    pub fn config(&self) -> &LyapunovConfig {
        &self.config
    }
}

/// Gradient of Lyapunov function w.r.t. state variables
#[derive(Debug, Clone, Copy)]
pub struct LyapunovGradient {
    pub drawdown: f64,
    pub var_95: f64,
    pub leverage: f64,
    pub margin_buffer: f64,
}

impl LyapunovGradient {
    /// Compute dot product with state change vector
    pub fn dot(&self, delta_state: &LyapunovGradient) -> f64 {
        self.drawdown * delta_state.drawdown
            + self.var_95 * delta_state.var_95
            + self.leverage * delta_state.leverage
            + self.margin_buffer * delta_state.margin_buffer
    }
    
    /// Compute norm of gradient
    pub fn norm(&self) -> f64 {
        (self.drawdown.powi(2)
            + self.var_95.powi(2)
            + self.leverage.powi(2)
            + self.margin_buffer.powi(2))
        .sqrt()
    }
}

/// Safety status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyStatus {
    /// Within safe operating region
    Safe,
    /// Approaching safety boundary
    Warning,
    /// Critical: immediate intervention required
    Critical,
}

impl SafetyStatus {
    /// Check if status requires intervention
    pub fn needs_intervention(self) -> bool {
        matches!(self, SafetyStatus::Warning | SafetyStatus::Critical)
    }
    
    /// Check if status is critical
    pub fn is_critical(self) -> bool {
        matches!(self, SafetyStatus::Critical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lyapunov_safe_state() {
        let lyap = LyapunovFunction::default_config();
        
        let safe_metrics = RiskMetrics {
            drawdown: 0.01,      // 1% drawdown
            var_95: 0.01,        // 1% VaR
            leverage: 1.5,       // 1.5x leverage
            margin_buffer: 0.8,  // Healthy margin
            ..Default::default()
        };
        
        let v = lyap.compute(&safe_metrics);
        assert!(v < lyap.config.safety_threshold);
        assert_eq!(lyap.get_safety_status(&safe_metrics), SafetyStatus::Safe);
    }
    
    #[test]
    fn test_lyapunov_critical_state() {
        let lyap = LyapunovFunction::default_config();
        
        let critical_metrics = RiskMetrics {
            drawdown: 0.15,      // 15% drawdown
            var_95: 0.08,        // 8% VaR
            leverage: 5.0,       // 5x leverage (dangerous)
            margin_buffer: 0.05, // Near margin call
            ..Default::default()
        };
        
        let v = lyap.compute(&critical_metrics);
        assert!(v >= lyap.config.critical_threshold);
        assert_eq!(lyap.get_safety_status(&critical_metrics), SafetyStatus::Critical);
    }
    
    #[test]
    fn test_lyapunov_non_negative() {
        let lyap = LyapunovFunction::default_config();
        
        // Test various states
        for drawdown in [0.0, 0.05, 0.1, 0.2] {
            for margin in [0.01, 0.1, 0.5, 1.0] {
                let metrics = RiskMetrics {
                    drawdown,
                    margin_buffer: margin,
                    ..Default::default()
                };
                
                let v = lyap.compute(&metrics);
                assert!(v >= 0.0, "Lyapunov function must be non-negative");
            }
        }
    }
}
