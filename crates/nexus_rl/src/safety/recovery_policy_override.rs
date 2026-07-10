//! NEXUS-OMEGA Stage 19: Recovery Policy Override
//!
//! This module implements a deterministic recovery policy that takes over
//! when the primary C-PPO agent cannot find safe actions. It executes
//! aggressive but controlled position flattening using TWAP/Iceberg algos
//! from Stage 4 to minimize slippage during emergency liquidation.
//!
//! Trigger conditions:
//! - QP projection infeasible (safe set is empty)
//! - Lyapunov value exceeds critical threshold
//! - Consecutive safety interventions > threshold
//! - Market gap breach of safe set
//!
//! Author: NEXUS-OMEGA Architecture
//! Stage: 19 of 50

use std::time::{Duration, Instant};

/// Recovery mode enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryMode {
    /// No recovery needed
    Inactive,
    /// Gradual position reduction
    Deleverage,
    /// Aggressive partial liquidation
    PartialLiquidate,
    /// Full emergency liquidation
    FullLiquidate,
    /// Kill-switch: close all positions immediately
    KillSwitch,
}

/// Configuration for recovery policy
#[derive(Debug, Clone)]
pub struct RecoveryPolicyConfig {
    /// Maximum consecutive interventions before triggering recovery
    pub max_consecutive_interventions: u32,
    
    /// Lyapunov value threshold for recovery trigger
    pub lyapunov_critical_threshold: f64,
    
    /// Minimum time between recovery actions (ms)
    pub min_action_interval_ms: u64,
    
    /// Target position reduction per step (fraction)
    pub deleverage_step_size: f64,
    
    /// TWAP duration for liquidation (seconds)
    pub liquidation_twap_duration_secs: u64,
    
    /// Iceberg order size fraction
    pub iceberg_display_fraction: f64,
    
    /// Maximum slippage tolerance (basis points)
    pub max_slippage_bps: u32,
}

impl Default for RecoveryPolicyConfig {
    fn default() -> Self {
        Self {
            max_consecutive_interventions: 5,
            lyapunov_critical_threshold: 0.9,
            min_action_interval_ms: 100,
            deleverage_step_size: 0.2,
            liquidation_twap_duration_secs: 60,
            iceberg_display_fraction: 0.1,
            max_slippage_bps: 50,
        }
    }
}

/// Recovery action to execute
#[derive(Debug, Clone)]
pub struct RecoveryAction {
    /// Asset identifier
    pub asset_id: String,
    
    /// Side of the trade
    pub side: TradeSide,
    
    /// Quantity to trade
    pub quantity: f64,
    
    /// Order type
    pub order_type: RecoveryOrderType,
    
    /// Urgency level (0 = normal, 1 = emergency)
    pub urgency: f64,
}

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Order type for recovery
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryOrderType {
    /// Standard market order
    Market,
    /// Limit order at mid price
    LimitMid,
    /// Iceberg order
    Iceberg,
    /// TWAP slice
    TwapSlice,
}

/// State machine for recovery execution
pub struct RecoveryStateMachine {
    config: RecoveryPolicyConfig,
    current_mode: RecoveryMode,
    
    // Execution tracking
    last_action_time: Option<Instant>,
    actions_executed: u64,
    total_quantity_liquidated: f64,
    
    // TWAP state
    twap_start_time: Option<Instant>,
    twap_total_quantity: f64,
    twap_executed_quantity: f64,
    
    // Position tracking
    initial_position: f64,
    current_position: f64,
}

impl RecoveryStateMachine {
    /// Create a new recovery state machine
    pub fn new(config: RecoveryPolicyConfig) -> Self {
        Self {
            config,
            current_mode: RecoveryMode::Inactive,
            last_action_time: None,
            actions_executed: 0,
            total_quantity_liquidated: 0.0,
            twap_start_time: None,
            twap_total_quantity: 0.0,
            twap_executed_quantity: 0.0,
            initial_position: 0.0,
            current_position: 0.0,
        }
    }
    
    /// Evaluate whether recovery should be triggered
    pub fn evaluate_trigger(
        &mut self,
        lyapunov_value: f64,
        consecutive_interventions: u32,
        qp_infeasible: bool,
        current_position: f64,
    ) -> RecoveryMode {
        self.current_position = current_position;
        
        // Check for immediate kill-switch conditions
        if qp_infeasible && lyapunov_value >= self.config.lyapunov_critical_threshold {
            self.current_mode = RecoveryMode::KillSwitch;
            return self.current_mode;
        }
        
        // Check for full liquidation
        if lyapunov_value >= self.config.lyapunov_critical_threshold * 0.9 {
            self.current_mode = RecoveryMode::FullLiquidate;
            return self.current_mode;
        }
        
        // Check for consecutive interventions
        if consecutive_interventions >= self.config.max_consecutive_interventions {
            if current_position.abs() > 0.5 * self.initial_position.abs() {
                self.current_mode = RecoveryMode::PartialLiquidate;
            } else {
                self.current_mode = RecoveryMode::Deleverage;
            }
            return self.current_mode;
        }
        
        // Check for gradual deleveraging
        if lyapunov_value >= self.config.lyapunov_critical_threshold * 0.7 {
            self.current_mode = RecoveryMode::Deleverage;
            return self.current_mode;
        }
        
        // No recovery needed
        if lyapunov_value < self.config.lyapunov_critical_threshold * 0.5 
            && consecutive_interventions == 0 
        {
            self.current_mode = RecoveryMode::Inactive;
        }
        
        self.current_mode
    }
    
    /// Generate recovery action based on current mode
    pub fn generate_action(&mut self, asset_id: &str) -> Option<RecoveryAction> {
        // Check rate limit
        if let Some(last_time) = self.last_action_time {
            if last_time.elapsed().as_millis() < self.config.min_action_interval_ms as u128 {
                return None;
            }
        }
        
        let action = match self.current_mode {
            RecoveryMode::Inactive => None,
            
            RecoveryMode::Deleverage => {
                Some(self.generate_deleverage_action(asset_id))
            }
            
            RecoveryMode::PartialLiquidate => {
                Some(self.generate_partial_liquidation_action(asset_id))
            }
            
            RecoveryMode::FullLiquidate => {
                Some(self.generate_full_liquidation_action(asset_id))
            }
            
            RecoveryMode::KillSwitch => {
                Some(self.generate_kill_switch_action(asset_id))
            }
        };
        
        if action.is_some() {
            self.last_action_time = Some(Instant::now());
            self.actions_executed += 1;
        }
        
        action
    }
    
    fn generate_deleverage_action(&self, asset_id: &str) -> RecoveryAction {
        let reduce_qty = self.current_position.abs() * self.config.deleverage_step_size;
        let side = if self.current_position > 0.0 {
            TradeSide::Sell
        } else {
            TradeSide::Buy
        };
        
        RecoveryAction {
            asset_id: asset_id.to_string(),
            side,
            quantity: reduce_qty,
            order_type: RecoveryOrderType::LimitMid,
            urgency: 0.3,
        }
    }
    
    fn generate_partial_liquidation_action(&self, asset_id: &str) -> RecoveryAction {
        let reduce_qty = self.current_position.abs() * 0.5;
        let side = if self.current_position > 0.0 {
            TradeSide::Sell
        } else {
            TradeSide::Buy
        };
        
        RecoveryAction {
            asset_id: asset_id.to_string(),
            side,
            quantity: reduce_qty,
            order_type: RecoveryOrderType::Iceberg,
            urgency: 0.7,
        }
    }
    
    fn generate_full_liquidation_action(&self, asset_id: &str) -> RecoveryAction {
        let side = if self.current_position > 0.0 {
            TradeSide::Sell
        } else {
            TradeSide::Buy
        };
        
        // Initialize TWAP
        let remaining = self.current_position.abs() - self.twap_executed_quantity;
        
        RecoveryAction {
            asset_id: asset_id.to_string(),
            side,
            quantity: remaining / 10.0, // Split into 10 slices
            order_type: RecoveryOrderType::TwapSlice,
            urgency: 0.9,
        }
    }
    
    fn generate_kill_switch_action(&self, asset_id: &str) -> RecoveryAction {
        let side = if self.current_position > 0.0 {
            TradeSide::Sell
        } else {
            TradeSide::Buy
        };
        
        RecoveryAction {
            asset_id: asset_id.to_string(),
            side,
            quantity: self.current_position.abs(),
            order_type: RecoveryOrderType::Market,
            urgency: 1.0,
        }
    }
    
    /// Update TWAP execution state
    pub fn update_twap_execution(&mut self, executed_qty: f64) {
        self.twap_executed_quantity += executed_qty;
        self.total_quantity_liquidated += executed_qty;
    }
    
    /// Start TWAP liquidation
    pub fn start_twap(&mut self, total_qty: f64, duration_secs: u64) {
        self.twap_start_time = Some(Instant::now());
        self.twap_total_quantity = total_qty;
        self.twap_executed_quantity = 0.0;
    }
    
    /// Check if TWAP is complete
    pub fn is_twap_complete(&self) -> bool {
        self.twap_executed_quantity >= self.twap_total_quantity * 0.99
    }
    
    /// Get TWAP progress
    pub fn get_twap_progress(&self) -> f64 {
        if self.twap_total_quantity < 1e-8 {
            return 1.0;
        }
        self.twap_executed_quantity / self.twap_total_quantity
    }
    
    /// Reset state machine
    pub fn reset(&mut self, new_position: f64) {
        self.current_mode = RecoveryMode::Inactive;
        self.last_action_time = None;
        self.twap_start_time = None;
        self.twap_total_quantity = 0.0;
        self.twap_executed_quantity = 0.0;
        self.initial_position = new_position;
        self.current_position = new_position;
    }
    
    /// Get current mode
    pub fn current_mode(&self) -> RecoveryMode {
        self.current_mode
    }
    
    /// Get statistics
    pub fn get_statistics(&self) -> RecoveryStats {
        RecoveryStats {
            actions_executed: self.actions_executed,
            total_liquidated: self.total_quantity_liquidated,
            twap_progress: self.get_twap_progress(),
            current_mode: self.current_mode,
        }
    }
}

/// Recovery statistics
#[derive(Debug, Clone)]
pub struct RecoveryStats {
    pub actions_executed: u64,
    pub total_liquidated: f64,
    pub twap_progress: f64,
    pub current_mode: RecoveryMode,
}

/// Recovery policy manager integrating with main trading loop
pub struct RecoveryPolicyManager {
    state_machine: RecoveryStateMachine,
    is_active: bool,
    trigger_count: u64,
}

impl RecoveryPolicyManager {
    pub fn new() -> Self {
        Self::with_config(RecoveryPolicyConfig::default())
    }
    
    pub fn with_config(config: RecoveryPolicyConfig) -> Self {
        Self {
            state_machine: RecoveryStateMachine::new(config),
            is_active: false,
            trigger_count: 0,
        }
    }
    
    /// Check and potentially trigger recovery
    pub fn check_and_trigger(
        &mut self,
        lyapunov_value: f64,
        consecutive_interventions: u32,
        qp_infeasible: bool,
        current_position: f64,
    ) -> bool {
        let mode = self.state_machine.evaluate_trigger(
            lyapunov_value,
            consecutive_interventions,
            qp_infeasible,
            current_position,
        );
        
        let was_inactive = !self.is_active;
        self.is_active = mode != RecoveryMode::Inactive;
        
        if self.is_active && was_inactive {
            self.trigger_count += 1;
            self.state_machine.reset(current_position);
        }
        
        self.is_active
    }
    
    /// Get next recovery action
    pub fn get_next_action(&mut self, asset_id: &str) -> Option<RecoveryAction> {
        if !self.is_active {
            return None;
        }
        
        let action = self.state_machine.generate_action(asset_id);
        
        // Check if we should deactivate
        if let Some(mode) = action.as_ref().map(|_| self.state_machine.current_mode()) {
            if mode == RecoveryMode::Inactive {
                self.is_active = false;
            }
        }
        
        action
    }
    
    /// Report execution result
    pub fn report_execution(&mut self, executed_qty: f64) {
        self.state_machine.update_twap_execution(executed_qty);
    }
    
    /// Check if recovery is active
    pub fn is_active(&self) -> bool {
        self.is_active
    }
    
    /// Force deactivation
    pub fn deactivate(&mut self) {
        self.is_active = false;
        self.state_machine.reset(0.0);
    }
    
    /// Get trigger count
    pub fn trigger_count(&self) -> u64 {
        self.trigger_count
    }
}

impl Default for RecoveryPolicyManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_recovery_trigger_on_critical_lyapunov() {
        let mut sm = RecoveryStateMachine::new(RecoveryPolicyConfig::default());
        
        let mode = sm.evaluate_trigger(
            0.95,  // Critical Lyapunov
            0,
            false,
            100.0,
        );
        
        assert_eq!(mode, RecoveryMode::FullLiquidate);
    }
    
    #[test]
    fn test_kill_switch_on_infeasible_qp() {
        let mut sm = RecoveryStateMachine::new(RecoveryPolicyConfig::default());
        
        let mode = sm.evaluate_trigger(
            0.92,  // High Lyapunov
            0,
            true,  // QP infeasible
            100.0,
        );
        
        assert_eq!(mode, RecoveryMode::KillSwitch);
    }
    
    #[test]
    fn test_deleverage_on_consecutive_interventions() {
        let mut sm = RecoveryStateMachine::new(RecoveryPolicyConfig::default());
        sm.initial_position = 100.0;
        
        let mode = sm.evaluate_trigger(
            0.5,   // Moderate Lyapunov
            5,     // Max interventions reached
            false,
            80.0,
        );
        
        assert_eq!(mode, RecoveryMode::PartialLiquidate);
    }
}
