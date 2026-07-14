//! Portfolio Flatten Finite State Machine
//! 
//! When the Kill Switch trips, this FSM manages the orderly liquidation
//! of all open positions. It transitions through states:
//! 
//! Trading -> CancelAllOpenOrders -> TWAP_Liquidation -> FlatAndHalted
//! 
//! CRITICAL: The flatten process uses limit orders with strict slippage
//! tolerance to avoid catastrophic fills during flash crashes.

use std::sync::atomic::{AtomicU8, AtomicBool, AtomicU64, Ordering};

/// Flatten FSM states
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlattenState {
    /// Normal trading operation
    Trading = 0,
    /// Kill switch tripped - cancelling all open orders
    CancellingOrders = 1,
    /// Actively liquidating positions via TWAP
    TWAPLiquidation = 2,
    /// All positions flat, system halted
    FlatAndHalted = 3,
}

impl FlattenState {
    fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Trading),
            1 => Some(Self::CancellingOrders),
            2 => Some(Self::TWAPLiquidation),
            3 => Some(Self::FlatAndHalted),
            _ => None,
        }
    }
}

/// Result of state transition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionResult {
    Success(FlattenState),
    Rejected { current: FlattenState, requested: FlattenState },
}

/// Configuration for the flatten FSM
#[derive(Debug, Clone)]
pub struct FlattenConfig {
    /// Maximum slippage in basis points (e.g., 500 = 5%)
    pub max_slippage_bps: u16,
    /// TWAP interval in milliseconds
    pub twap_interval_ms: u64,
    /// Minimum order size (base units)
    pub min_order_size: u64,
}

impl Default for FlattenConfig {
    fn default() -> Self {
        Self {
            max_slippage_bps: 500, // 5% max slippage
            twap_interval_ms: 1000, // 1 second between slices
            min_order_size: 10_000, // Minimum order size
        }
    }
}

/// Statistics from the flatten FSM
#[derive(Debug, Clone)]
pub struct FlattenStats {
    pub current_state: FlattenState,
    pub positions_remaining: i64,
    pub total_liquidated: u64,
    pub cancelled_orders: u64,
    pub liquidation_trips: u64,
}

/// Portfolio Flatten FSM
/// 
/// Manages the orderly exit from all positions when the kill switch trips.
/// Uses a strict state machine to ensure proper sequencing.
pub struct FlattenFSM {
    config: FlattenConfig,
    /// Current state
    state: AtomicU8,
    /// Whether we're currently in a liquidation sequence
    is_liquidating: AtomicBool,
    /// Total position at start of liquidation
    initial_position: AtomicU64,
    /// Remaining position to liquidate (absolute value)
    remaining_position: AtomicU64,
    /// Count of orders cancelled during liquidation
    cancelled_orders: AtomicU64,
    /// Count of liquidation sequences initiated
    liquidation_trips: AtomicU64,
    /// Total quantity liquidated
    total_liquidated: AtomicU64,
}

unsafe impl Send for FlattenFSM {}
unsafe impl Sync for FlattenFSM {}

impl FlattenFSM {
    /// Create a new flatten FSM
    pub fn new(config: FlattenConfig) -> Self {
        Self {
            config,
            state: AtomicU8::new(FlattenState::Trading as u8),
            is_liquidating: AtomicBool::new(false),
            initial_position: AtomicU64::new(0),
            remaining_position: AtomicU64::new(0),
            cancelled_orders: AtomicU64::new(0),
            liquidation_trips: AtomicU64::new(0),
            total_liquidated: AtomicU64::new(0),
        }
    }

    /// Get current state
    #[inline]
    pub fn get_state(&self) -> FlattenState {
        let value = self.state.load(Ordering::Relaxed);
        FlattenState::from_u8(value).unwrap_or(FlattenState::FlatAndHalted)
    }

    /// Initiate the flatten sequence (called when kill switch trips)
    /// 
    /// Transitions from Trading -> CancellingOrders
    #[inline]
    pub fn initiate_flatten(&self, total_position: i64) -> TransitionResult {
        let current = self.get_state();
        
        if current != FlattenState::Trading {
            return TransitionResult::Rejected {
                current,
                requested: FlattenState::CancellingOrders,
            };
        }

        // Store initial position (as absolute value)
        let abs_position = total_position.unsigned_abs();
        self.initial_position.store(abs_position, Ordering::Relaxed);
        self.remaining_position.store(abs_position, Ordering::Relaxed);

        // Transition to CancellingOrders
        self.state.store(FlattenState::CancellingOrders as u8, Ordering::SeqCst);
        self.is_liquidating.store(true, Ordering::Relaxed);
        self.liquidation_trips.fetch_add(1, Ordering::Relaxed);

        TransitionResult::Success(FlattenState::CancellingOrders)
    }

    /// Mark an order as cancelled during liquidation
    #[inline]
    pub fn record_cancel(&self) {
        self.cancelled_orders.fetch_add(1, Ordering::Relaxed);
    }

    /// Transition to TWAP liquidation after all orders cancelled
    #[inline]
    pub fn begin_twap_liquidation(&self) -> TransitionResult {
        let current = self.get_state();
        
        if current != FlattenState::CancellingOrders {
            return TransitionResult::Rejected {
                current,
                requested: FlattenState::TWAPLiquidation,
            };
        }

        self.state.store(FlattenState::TWAPLiquidation as u8, Ordering::SeqCst);
        TransitionResult::Success(FlattenState::TWAPLiquidation)
    }

    /// Record a filled liquidation order
    /// 
    /// # Arguments
    /// * `quantity` - Quantity filled
    #[inline]
    pub fn record_liquidation_fill(&self, quantity: u64) {
        self.remaining_position.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| current.checked_sub(quantity),
        ).ok();
        
        self.total_liquidated.fetch_add(quantity, Ordering::Relaxed);
    }

    /// Check if liquidation is complete and transition to FlatAndHalted
    #[inline]
    pub fn check_liquidation_complete(&self) -> TransitionResult {
        let current = self.get_state();
        
        if current != FlattenState::TWAPLiquidation {
            return TransitionResult::Rejected {
                current,
                requested: FlattenState::FlatAndHalted,
            };
        }

        if self.remaining_position.load(Ordering::Relaxed) == 0 {
            self.state.store(FlattenState::FlatAndHalted as u8, Ordering::SeqCst);
            self.is_liquidating.store(false, Ordering::Relaxed);
            TransitionResult::Success(FlattenState::FlatAndHalted)
        } else {
            TransitionResult::Rejected {
                current,
                requested: FlattenState::FlatAndHalted,
            }
        }
    }

    /// Reset the FSM (only after manual intervention)
    #[inline]
    pub fn reset(&self) {
        self.state.store(FlattenState::Trading as u8, Ordering::SeqCst);
        self.is_liquidating.store(false, Ordering::Relaxed);
        self.initial_position.store(0, Ordering::Relaxed);
        self.remaining_position.store(0, Ordering::Relaxed);
    }

    /// Get maximum slippage in basis points
    #[inline]
    pub fn max_slippage_bps(&self) -> u16 {
        self.config.max_slippage_bps
    }

    /// Get TWAP interval in milliseconds
    #[inline]
    pub fn twap_interval_ms(&self) -> u64 {
        self.config.twap_interval_ms
    }

    /// Get statistics
    pub fn stats(&self) -> FlattenStats {
        FlattenStats {
            current_state: self.get_state(),
            positions_remaining: self.remaining_position.load(Ordering::Relaxed) as i64,
            total_liquidated: self.total_liquidated.load(Ordering::Relaxed),
            cancelled_orders: self.cancelled_orders.load(Ordering::Relaxed),
            liquidation_trips: self.liquidation_trips.load(Ordering::Relaxed),
        }
    }

    /// Check if currently liquidating
    #[inline]
    pub fn is_liquidating(&self) -> bool {
        self.is_liquidating.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_state() {
        let fsm = FlattenFSM::new(FlattenConfig::default());
        assert_eq!(fsm.get_state(), FlattenState::Trading);
        assert!(!fsm.is_liquidating());
    }

    #[test]
    fn test_flatten_sequence() {
        let fsm = FlattenFSM::new(FlattenConfig::default());
        
        // Start with 1 BTC position (100M satoshis)
        let result = fsm.initiate_flatten(100_000_000);
        assert!(matches!(result, TransitionResult::Success(FlattenState::CancellingOrders)));
        
        // Record some cancels
        fsm.record_cancel();
        fsm.record_cancel();
        
        // Begin TWAP
        let result = fsm.begin_twap_liquidation();
        assert!(matches!(result, TransitionResult::Success(FlattenState::TWAPLiquidation)));
        
        // Record partial fill
        fsm.record_liquidation_fill(50_000_000);
        assert_eq!(fsm.stats().positions_remaining, 50_000_000);
        
        // Record rest of fill
        fsm.record_liquidation_fill(50_000_000);
        
        // Should be complete now
        let result = fsm.check_liquidation_complete();
        assert!(matches!(result, TransitionResult::Success(FlattenState::FlatAndHalted)));
    }

    #[test]
    fn test_invalid_transition() {
        let fsm = FlattenFSM::new(FlattenConfig::default());
        
        // Can't go directly to TWAP from Trading
        let result = fsm.begin_twap_liquidation();
        assert!(matches!(result, TransitionResult::Rejected { .. }));
    }
}
