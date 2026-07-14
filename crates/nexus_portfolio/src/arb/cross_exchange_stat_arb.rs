//! Cross-Exchange Statistical Arbitrage Engine.
//! 
//! Identifies cointegrated pairs across venues and executes dual-leg trades.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use crate::arb::atomic_legging_resolver::{AtomicLeggingResolver, LegState};

/// State of the stat arb engine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatArbState {
    Idle,
    Scanning,
    SignalDetected,
    Executing,
    PositionOpen,
    Closing,
}

/// A stat arb signal
#[derive(Debug, Clone)]
pub struct StatArbSignal {
    pub symbol_a: [u8; 16],
    pub symbol_b: [u8; 16],
    pub exchange_a: u8,
    pub exchange_b: u8,
    /// Z-score of the spread
    pub z_score: i128, // Scaled by 1e12
    /// Expected mean reversion profit (scaled)
    pub expected_profit_scaled: i128,
}

/// Command to execute stat arb
#[derive(Debug, Clone)]
pub struct StatArbCommand {
    pub signal: StatArbSignal,
    pub notional_scaled: i128,
    pub max_slippage_bps: i128,
}

/// Cross-exchange stat arb engine
pub struct CrossExchangeStatArb {
    current_state: AtomicU8,
    /// The legging resolver for handling orphan legs
    legging_resolver: AtomicLeggingResolver,
    /// Active position flag
    position_active: AtomicBool,
}

impl CrossExchangeStatArb {
    pub fn new() -> Self {
        Self {
            current_state: AtomicU8::new(StatArbState::Idle as u8),
            legging_resolver: AtomicLeggingResolver::new(),
            position_active: AtomicBool::new(false),
        }
    }

    /// Check if a signal meets entry criteria
    pub fn validate_signal(&self, signal: &StatArbSignal, min_z_score: i128) -> bool {
        // Z-score must exceed threshold (e.g., |z| > 2.0)
        let abs_z = signal.z_score.abs();
        abs_z > min_z_score
    }

    /// Initiate stat arb execution
    pub fn initiate_execution(&self, command: &StatArbCommand) -> bool {
        let expected = StatArbState::SignalDetected as u8;
        let desired = StatArbState::Executing as u8;
        
        match self.current_state.compare_exchange(
            expected,
            desired,
            Ordering::SeqCst,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                // Initialize the legging resolver for this trade
                self.legging_resolver.reset_for_new_trade(command.notional_scaled);
                true
            }
            Err(_) => false,
        }
    }

    /// Record fill on leg A
    pub fn record_leg_a_fill(&self, fill_size: i128) {
        self.legging_resolver.record_leg_fill(0, fill_size);
    }

    /// Record fill on leg B
    pub fn record_leg_b_fill(&self, fill_size: i128) {
        self.legging_resolver.record_leg_fill(1, fill_size);
    }

    /// Check if both legs filled successfully
    pub fn check_both_legs_filled(&self) -> bool {
        self.legging_resolver.both_legs_filled()
    }

    /// Handle orphan leg - triggers SOR to flatten
    pub fn handle_orphan_leg(&self) -> bool {
        if self.legging_resolver.detect_orphan_leg() {
            // Trigger the SOR router to flatten
            return self.legging_resolver.trigger_sor_flatten();
        }
        false
    }

    /// Transition to position open state
    pub fn confirm_position_open(&self) {
        self.current_state.store(StatArbState::PositionOpen as u8, Ordering::SeqCst);
        self.position_active.store(true, Ordering::SeqCst);
    }

    /// Initiate close
    pub fn initiate_close(&self) -> bool {
        let expected = StatArbState::PositionOpen as u8;
        let desired = StatArbState::Closing as u8;
        
        match self.current_state.compare_exchange(
            expected,
            desired,
            Ordering::SeqCst,
            Ordering::Relaxed,
        ) {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    /// Confirm position closed
    pub fn confirm_position_closed(&self) {
        self.current_state.store(StatArbState::Idle as u8, Ordering::SeqCst);
        self.position_active.store(false, Ordering::SeqCst);
        self.legging_resolver.reset_for_new_trade(0);
    }

    /// Get current state
    pub fn get_state(&self) -> StatArbState {
        self.u8_to_state(self.current_state.load(Ordering::Relaxed))
    }

    /// Check if position is active
    pub fn is_position_active(&self) -> bool {
        self.position_active.load(Ordering::Relaxed)
    }

    /// Get reference to legging resolver
    pub fn get_legging_resolver(&self) -> &AtomicLeggingResolver {
        &self.legging_resolver
    }

    fn u8_to_state(&self, val: u8) -> StatArbState {
        match val {
            0 => StatArbState::Idle,
            1 => StatArbState::Scanning,
            2 => StatArbState::SignalDetected,
            3 => StatArbState::Executing,
            4 => StatArbState::PositionOpen,
            5 => StatArbState::Closing,
            _ => StatArbState::Idle,
        }
    }
}

impl Default for CrossExchangeStatArb {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stat_arb_signal_validation() {
        let arb = CrossExchangeStatArb::new();
        
        let strong_signal = StatArbSignal {
            symbol_a: [0; 16],
            symbol_b: [0; 16],
            exchange_a: 1,
            exchange_b: 2,
            z_score: 250_000_000_000i128, // 2.5 sigma
            expected_profit_scaled: 1_000_000_000_000_000_000i128,
        };
        
        assert!(arb.validate_signal(&strong_signal, 2_000_000_000_000i128)); // |z| > 2.0
        
        let weak_signal = StatArbSignal {
            z_score: 1_500_000_000_000i128, // 1.5 sigma
            ..strong_signal.clone()
        };
        
        assert!(!arb.validate_signal(&weak_signal, 2_000_000_000_000i128));
    }
}
