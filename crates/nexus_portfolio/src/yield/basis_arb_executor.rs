//! Basis Arbitrage Executor - Executes delta-neutral spot/perp arb trades.
//! 
//! When funding rate > threshold, executes: Buy Spot + Short Perp

use std::sync::atomic::{AtomicBool, AtomicI128, Ordering};
use crate::yield::funding_rate_harvester::AnnualizedFunding;

/// Thresholds
const MIN_SPOT_PERP_SPREAD_BPS: i128 = 50; // 0.5% minimum spread

/// State of a basis arb position
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisArbState {
    Idle,
    PendingEntry,
    ActiveLongSpotShortPerp,
    PendingExit,
    Closed,
}

/// Command to execute basis arb
#[derive(Debug, Clone)]
pub struct BasisArbCommand {
    pub symbol: [u8; 16],
    pub notional_scaled: i128,
    pub spot_exchange_id: u8,
    pub perp_exchange_id: u8,
    pub max_slippage_bps: i128,
}

/// Result of arb execution attempt
#[derive(Debug, Clone)]
pub enum BasisArbResult {
    Success,
    PartialFill { spot_filled: bool, perp_filled: bool },
    Failed(&'static str),
}

/// Lock-free basis arb executor
pub struct BasisArbExecutor {
    /// Current state
    current_state: AtomicU8,
    /// Active position size (scaled)
    position_size: AtomicI128,
    /// Accumulated arb yield (scaled)
    accumulated_yield: AtomicI128,
    /// Flag indicating entry in progress
    entry_in_progress: AtomicBool,
}

// Map u8 to BasisArbState
impl BasisArbExecutor {
    pub fn new() -> Self {
        Self {
            current_state: AtomicU8::new(BasisArbState::Idle as u8),
            position_size: AtomicI128::new(0),
            accumulated_yield: AtomicI128::new(0),
            entry_in_progress: AtomicBool::new(false),
        }
    }

    /// Check if conditions are met for basis arb
    pub fn check_arb_conditions(
        &self,
        annualized_funding: &AnnualizedFunding,
        spot_perp_spread_bps: i128,
    ) -> bool {
        // Need funding > 15% APY AND positive spot-perp spread
        annualized_funding.apy_bps > 1500 && spot_perp_spread_bps > MIN_SPOT_PERP_SPREAD_BPS
    }

    /// Initiate basis arb entry
    pub fn initiate_entry(&self, command: &BasisArbCommand) -> bool {
        let expected = BasisArbState::Idle as u8;
        let desired = BasisArbState::PendingEntry as u8;
        
        // CAS to transition from Idle -> PendingEntry
        match self.current_state.compare_exchange(
            expected,
            desired,
            Ordering::SeqCst,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                self.entry_in_progress.store(true, Ordering::SeqCst);
                true
            }
            Err(_) => false, // Already in a position or entry in progress
        }
    }

    /// Confirm both legs filled
    pub fn confirm_both_legs_filled(&self, size_scaled: i128) -> BasisArbResult {
        let current = self.current_state.load(Ordering::Relaxed);
        let state = self.u8_to_state(current);
        
        if state != BasisArbState::PendingEntry {
            return BasisArbResult::Failed("Not in pending entry state");
        }
        
        self.position_size.store(size_scaled, Ordering::SeqCst);
        self.current_state.store(
            BasisArbState::ActiveLongSpotShortPerp as u8,
            Ordering::SeqCst,
        );
        self.entry_in_progress.store(false, Ordering::SeqCst);
        
        BasisArbResult::Success
    }

    /// Record funding yield received
    pub fn record_funding_yield(&self, yield_scaled: i128) {
        self.accumulated_yield.fetch_add(yield_scaled, Ordering::Relaxed);
    }

    /// Initiate exit (close both legs)
    pub fn initiate_exit(&self) -> bool {
        let expected = BasisArbState::ActiveLongSpotShortPerp as u8;
        let desired = BasisArbState::PendingExit as u8;
        
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

    /// Confirm both legs closed
    pub fn confirm_both_legs_closed(&self) -> BasisArbResult {
        let current = self.current_state.load(Ordering::Relaxed);
        let state = self.u8_to_state(current);
        
        if state != BasisArbState::PendingExit {
            return BasisArbResult::Failed("Not in pending exit state");
        }
        
        self.position_size.store(0, Ordering::SeqCst);
        self.current_state.store(BasisArbState::Closed as u8, Ordering::SeqCst);
        
        // Transition back to idle after a brief period (caller should handle timing)
        self.current_state.store(BasisArbState::Idle as u8, Ordering::SeqCst);
        
        BasisArbResult::Success
    }

    /// Get current state
    pub fn get_state(&self) -> BasisArbState {
        self.u8_to_state(self.current_state.load(Ordering::Relaxed))
    }

    /// Get position size
    pub fn get_position_size(&self) -> i128 {
        self.position_size.load(Ordering::Relaxed)
    }

    /// Get accumulated yield
    pub fn get_accumulated_yield(&self) -> i128 {
        self.accumulated_yield.load(Ordering::Relaxed)
    }

    fn u8_to_state(&self, val: u8) -> BasisArbState {
        match val {
            0 => BasisArbState::Idle,
            1 => BasisArbState::PendingEntry,
            2 => BasisArbState::ActiveLongSpotShortPerp,
            3 => BasisArbState::PendingExit,
            4 => BasisArbState::Closed,
            _ => BasisArbState::Idle,
        }
    }
}

impl Default for BasisArbExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_arb_state_transitions() {
        let executor = BasisArbExecutor::new();
        
        assert_eq!(executor.get_state(), BasisArbState::Idle);
        
        // Initiate entry
        let cmd = BasisArbCommand {
            symbol: [0; 16],
            notional_scaled: 1_000_000_000_000_000_000i128,
            spot_exchange_id: 1,
            perp_exchange_id: 2,
            max_slippage_bps: 10,
        };
        
        assert!(executor.initiate_entry(&cmd));
        assert_eq!(executor.get_state(), BasisArbState::PendingEntry);
        assert!(executor.entry_in_progress.load(Ordering::Relaxed));
        
        // Confirm fills
        let result = executor.confirm_both_legs_filled(1_000_000_000_000_000_000i128);
        assert!(matches!(result, BasisArbResult::Success));
        assert_eq!(executor.get_state(), BasisArbState::ActiveLongSpotShortPerp);
        assert!(!executor.entry_in_progress.load(Ordering::Relaxed));
        
        // Initiate exit
        assert!(executor.initiate_exit());
        assert_eq!(executor.get_state(), BasisArbState::PendingExit);
        
        // Confirm close
        let result = executor.confirm_both_legs_closed();
        assert!(matches!(result, BasisArbResult::Success));
        assert_eq!(executor.get_state(), BasisArbState::Idle);
    }
}
