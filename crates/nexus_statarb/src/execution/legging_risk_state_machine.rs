//! Legging Risk State Machine
//! 
//! Tracks the exact fill status of both legs in a pair trade
//! and manages state transitions without deadlocking.

use core::sync::atomic::{AtomicU64, AtomicU8, Ordering};

/// State of an individual leg in the pair trade
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegState {
    /// Order submitted, awaiting response
    Pending,
    /// Order acknowledged by exchange
    Acknowledged,
    /// Partially filled
    Partial,
    /// Fully filled
    Filled,
    /// Cancelled by user/system
    Cancelled,
    /// Rejected by exchange
    Rejected,
    /// Failed after retries
    Failed,
    /// Being retried
    Retrying,
}

impl Default for LegState {
    #[inline]
    fn default() -> Self {
        Self::Pending
    }
}

/// Overall state of the pair execution
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairExecutionStateType {
    /// Initial state - no orders submitted
    Init,
    /// Both orders submitted
    Submitted,
    /// One leg filled, waiting for other
    HalfFilled,
    /// Both legs filled successfully
    Complete,
    /// One leg failed, hedging in progress
    Hedging,
    /// Emergency flattening
    Flattening,
    /// Execution failed
    Failed,
    /// Cancelled
    Cancelled,
}

impl Default for PairExecutionStateType {
    #[inline]
    fn default() -> Self {
        Self::Init
    }
}

/// Snapshot of pair execution state
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PairExecutionState {
    pub leg_a_state: LegState,
    pub leg_b_state: LegState,
    pub leg_a_filled: bool,
    pub leg_b_filled: bool,
    pub leg_a_qty: i64,
    pub leg_b_qty: i64,
    pub leg_a_price: f64,
    pub leg_b_price: f64,
}

impl Default for PairExecutionState {
    #[inline]
    fn default() -> Self {
        Self {
            leg_a_state: LegState::Pending,
            leg_b_state: LegState::Pending,
            leg_a_filled: false,
            leg_b_filled: false,
            leg_a_qty: 0,
            leg_b_qty: 0,
            leg_a_price: 0.0,
            leg_b_price: 0.0,
        }
    }
}

/// Legging Risk State Machine
/// 
/// Manages state transitions for paired trades with strict
/// handling of partial fills and race conditions.
pub struct LeggingRiskStateMachine {
    /// Current state of leg A
    leg_a: LegState,
    /// Current state of leg B
    leg_b: LegState,
    /// Overall pair state
    pair_state: PairExecutionStateType,
    /// Quantity imbalance (leg_a_qty - expected_hedge_qty)
    quantity_imbalance: i64,
    /// Price slippage in basis points
    slippage_bps: u16,
    /// Number of state transitions
    transition_count: AtomicU64,
    /// Last error code
    last_error_code: AtomicU8,
    /// Timestamp of last state change (microseconds)
    last_transition_ts: AtomicU64,
}

impl LeggingRiskStateMachine {
    /// Create a new state machine
    #[inline]
    pub fn new() -> Self {
        Self {
            leg_a: LegState::Pending,
            leg_b: LegState::Pending,
            pair_state: PairExecutionStateType::Init,
            quantity_imbalance: 0,
            slippage_bps: 0,
            transition_count: AtomicU64::new(0),
            last_error_code: AtomicU8::new(0),
            last_transition_ts: AtomicU64::new(0),
        }
    }

    /// Transition leg A to a new state
    /// 
    /// Returns true if transition is valid, false otherwise
    #[inline]
    pub fn transition_leg_a(&mut self, new_state: LegState, timestamp_us: u64) -> bool {
        if !self.is_valid_leg_transition(self.leg_a, new_state) {
            self.last_error_code.store(1, Ordering::Relaxed);
            return false;
        }

        self.leg_a = new_state;
        self.update_pair_state(timestamp_us);
        true
    }

    /// Transition leg B to a new state
    #[inline]
    pub fn transition_leg_b(&mut self, new_state: LegState, timestamp_us: u64) -> bool {
        if !self.is_valid_leg_transition(self.leg_b, new_state) {
            self.last_error_code.store(2, Ordering::Relaxed);
            return false;
        }

        self.leg_b = new_state;
        self.update_pair_state(timestamp_us);
        true
    }

    /// Check if a leg state transition is valid
    #[inline]
    fn is_valid_leg_transition(&self, current: LegState, new: LegState) -> bool {
        match current {
            LegState::Pending => matches!(
                new,
                LegState::Acknowledged | LegState::Partial | LegState::Filled | LegState::Rejected | LegState::Cancelled
            ),
            LegState::Acknowledged => matches!(
                new,
                LegState::Partial | LegState::Filled | LegState::Rejected | LegState::Cancelled | LegState::Retrying
            ),
            LegState::Partial => matches!(
                new,
                LegState::Filled | LegState::Cancelled | LegState::Retrying
            ),
            LegState::Filled => false, // Terminal state
            LegState::Cancelled => false, // Terminal state
            LegState::Rejected => matches!(new, LegState::Retrying),
            LegState::Failed => false, // Terminal state
            LegState::Retrying => matches!(
                new,
                LegState::Acknowledged | LegState::Partial | LegState::Filled | LegState::Failed
            ),
        }
    }

    /// Update the overall pair state based on leg states
    #[inline]
    fn update_pair_state(&mut self, timestamp_us: u64) {
        let old_pair_state = self.pair_state;

        self.pair_state = match (self.leg_a, self.leg_b) {
            // Both filled - complete
            (LegState::Filled, LegState::Filled) => PairExecutionStateType::Complete,
            
            // One filled, one in progress - half filled
            (LegState::Filled, LegState::Pending | LegState::Acknowledged | LegState::Partial) |
            (LegState::Pending | LegState::Acknowledged | LegState::Partial, LegState::Filled) => {
                PairExecutionStateType::HalfFilled
            }
            
            // One failed, need to hedge
            (LegState::Failed, LegState::Filled) |
            (LegState::Filled, LegState::Failed) => PairExecutionStateType::Hedging,
            
            // Both failed or rejected
            (LegState::Failed, LegState::Failed) |
            (LegState::Rejected, LegState::Rejected) => PairExecutionStateType::Failed,
            
            // One cancelled
            (LegState::Cancelled, _) | (_, LegState::Cancelled) => PairExecutionStateType::Cancelled,
            
            // Both submitted/pending
            _ => PairExecutionStateType::Submitted,
        };

        // Record transition
        if old_pair_state != self.pair_state {
            self.transition_count.fetch_add(1, Ordering::Relaxed);
            self.last_transition_ts.store(timestamp_us, Ordering::Relaxed);
        }
    }

    /// Update quantity imbalance after a fill
    #[inline]
    pub fn update_imbalance(&mut self, leg_a_qty: i64, leg_b_qty: i64) {
        self.quantity_imbalance = leg_a_qty + leg_b_qty; // Should be ~0 for market-neutral
        self.slippage_bps = ((leg_a_qty as f64 - (-leg_b_qty) as f64).abs() / 
            (leg_a_qty.abs().max(leg_b_qty.abs()) as f64) * 10000.0) as u16;
    }

    /// Get the current quantity imbalance
    #[inline]
    pub fn quantity_imbalance(&self) -> i64 {
        self.quantity_imbalance
    }

    /// Get the current slippage in basis points
    #[inline]
    pub fn slippage_bps(&self) -> u16 {
        self.slippage_bps
    }

    /// Check if there's dangerous legging risk (imbalance > threshold)
    #[inline]
    pub fn has_dangerous_imbalance(&self, threshold_bps: u16) -> bool {
        self.slippage_bps > threshold_bps
    }

    /// Get current pair state
    #[inline]
    pub fn pair_state(&self) -> PairExecutionStateType {
        self.pair_state
    }

    /// Get leg A state
    #[inline]
    pub fn leg_a_state(&self) -> LegState {
        self.leg_a
    }

    /// Get leg B state
    #[inline]
    pub fn leg_b_state(&self) -> LegState {
        self.leg_b
    }

    /// Get transition count
    #[inline]
    pub fn transition_count(&self) -> u64 {
        self.transition_count.load(Ordering::Relaxed)
    }

    /// Get last error code
    #[inline]
    pub fn last_error_code(&self) -> u8 {
        self.last_error_code.load(Ordering::Relaxed)
    }

    /// Get timestamp of last transition
    #[inline]
    pub fn last_transition_timestamp(&self) -> u64 {
        self.last_transition_ts.load(Ordering::Relaxed)
    }

    /// Reset to initial state
    #[inline]
    pub fn reset(&mut self) {
        self.leg_a = LegState::Pending;
        self.leg_b = LegState::Pending;
        self.pair_state = PairExecutionStateType::Init;
        self.quantity_imbalance = 0;
        self.slippage_bps = 0;
        self.transition_count.fetch_add(1, Ordering::Relaxed);
        self.last_error_code.store(0, Ordering::Relaxed);
    }

    /// Force transition to hedging state (for emergency scenarios)
    #[inline]
    pub fn force_hedging(&mut self, timestamp_us: u64) {
        self.pair_state = PairExecutionStateType::Hedging;
        self.transition_count.fetch_add(1, Ordering::Relaxed);
        self.last_transition_ts.store(timestamp_us, Ordering::Relaxed);
    }

    /// Force transition to flattening state
    #[inline]
    pub fn force_flattening(&mut self, timestamp_us: u64) {
        self.pair_state = PairExecutionStateType::Flattening;
        self.transition_count.fetch_add(1, Ordering::Relaxed);
        self.last_transition_ts.store(timestamp_us, Ordering::Relaxed);
    }
}

impl Default for LeggingRiskStateMachine {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_transitions() {
        let mut sm = LeggingRiskStateMachine::new();
        
        assert!(sm.transition_leg_a(LegState::Acknowledged, 1000));
        assert!(sm.transition_leg_a(LegState::Filled, 1100));
        
        assert_eq!(sm.leg_a_state(), LegState::Filled);
    }

    #[test]
    fn test_invalid_transition_filled_to_pending() {
        let mut sm = LeggingRiskStateMachine::new();
        
        sm.transition_leg_a(LegState::Filled, 1000);
        
        // Cannot go from Filled back to Pending
        assert!(!sm.transition_leg_a(LegState::Pending, 1100));
    }

    #[test]
    fn test_pair_state_complete() {
        let mut sm = LeggingRiskStateMachine::new();
        
        sm.transition_leg_a(LegState::Filled, 1000);
        sm.transition_leg_b(LegState::Filled, 1050);
        
        assert_eq!(sm.pair_state(), PairExecutionStateType::Complete);
    }

    #[test]
    fn test_half_filled_state() {
        let mut sm = LeggingRiskStateMachine::new();
        
        sm.transition_leg_a(LegState::Filled, 1000);
        sm.transition_leg_b(LegState::Acknowledged, 1050);
        
        assert_eq!(sm.pair_state(), PairExecutionStateType::HalfFilled);
    }

    #[test]
    fn test_hedging_state_on_failure() {
        let mut sm = LeggingRiskStateMachine::new();
        
        sm.transition_leg_a(LegState::Filled, 1000);
        sm.transition_leg_b(LegState::Failed, 1050);
        
        assert_eq!(sm.pair_state(), PairExecutionStateType::Hedging);
    }

    #[test]
    fn test_quantity_imbalance() {
        let mut sm = LeggingRiskStateMachine::new();
        
        // Perfect hedge: +100 and -100
        sm.update_imbalance(100, -100);
        assert_eq!(sm.quantity_imbalance(), 0);
        assert_eq!(sm.slippage_bps(), 0);
        
        // Imperfect hedge: +100 and -90
        sm.update_imbalance(100, -90);
        assert_eq!(sm.quantity_imbalance(), 10);
        assert!(sm.slippage_bps() > 0);
    }

    #[test]
    fn test_dangerous_imbalance_detection() {
        let mut sm = LeggingRiskStateMachine::new();
        
        // 10% imbalance = 1000 bps
        sm.update_imbalance(100, -90);
        
        assert!(sm.has_dangerous_imbalance(500)); // 500 bps threshold
        assert!(!sm.has_dangerous_imbalance(1500)); // 1500 bps threshold
    }

    #[test]
    fn test_force_states() {
        let mut sm = LeggingRiskStateMachine::new();
        
        sm.force_hedging(1000);
        assert_eq!(sm.pair_state(), PairExecutionStateType::Hedging);
        
        sm.force_flattening(1100);
        assert_eq!(sm.pair_state(), PairExecutionStateType::Flattening);
    }
}
