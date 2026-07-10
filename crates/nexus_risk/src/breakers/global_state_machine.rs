//! Global System State Machine using atomic state transitions.
//! 
//! All execution threads poll this state via AtomicU8 loads with Relaxed ordering
//! for zero-cost state checks. The state machine ensures coordinated behavior
//! across all system components.

use std::sync::atomic::{AtomicU8, Ordering};

/// System state representation as u8 for atomic operations.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SystemState {
    /// Normal trading operation - all systems go
    Trading = 0,
    /// Halting: accepting no new orders, existing orders still active
    Halting = 1,
    /// Flattening: actively closing all positions
    Flattening = 2,
    /// Dead: complete shutdown, no network activity allowed
    Dead = 3,
    /// Paused: temporary pause (e.g., during maintenance)
    Paused = 4,
    /// Recovery: attempting to recover from an error state
    Recovery = 5,
}

impl SystemState {
    /// Convert from u8, returning None for invalid values
    #[inline]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Trading),
            1 => Some(Self::Halting),
            2 => Some(Self::Flattening),
            3 => Some(Self::Dead),
            4 => Some(Self::Paused),
            5 => Some(Self::Recovery),
            _ => None,
        }
    }

    /// Check if this is a terminal state (no further transitions allowed except to Dead)
    #[inline]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Dead)
    }

    /// Check if new orders are allowed in this state
    #[inline]
    pub fn allows_new_orders(&self) -> bool {
        matches!(self, Self::Trading)
    }

    /// Check if order cancellations are allowed
    #[inline]
    pub fn allows_cancellations(&self) -> bool {
        !matches!(self, Self::Dead)
    }

    /// Check if position flattening is active
    #[inline]
    pub fn is_flattening(&self) -> bool {
        matches!(self, Self::Flattening)
    }
}

/// Valid state transitions for the state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StateTransition {
    TradingToHalting,
    TradingToPaused,
    HaltingToFlattening,
    HaltingToTrading,
    FlatteningToDead,
    FlatteningToTrading,
    PausedToTrading,
    PausedToDead,
    RecoveryToTrading,
    RecoveryToDead,
    AnyToDead,
}

impl StateTransition {
    /// Check if a transition is valid
    #[inline]
    fn is_valid(from: SystemState, to: SystemState) -> bool {
        matches!(
            (from, to),
            (SystemState::Trading, SystemState::Halting) |
            (SystemState::Trading, SystemState::Paused) |
            (SystemState::Halting, SystemState::Flattening) |
            (SystemState::Halting, SystemState::Trading) |
            (SystemState::Flattening, SystemState::Dead) |
            (SystemState::Flattening, SystemState::Trading) |
            (SystemState::Paused, SystemState::Trading) |
            (SystemState::Paused, SystemState::Dead) |
            (SystemState::Recovery, SystemState::Trading) |
            (SystemState::Recovery, SystemState::Dead) |
            (_, SystemState::Dead) // Any state can go to Dead
        )
    }
}

/// Result of a state transition attempt
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionResult {
    /// Transition succeeded
    Success(SystemState),
    /// Transition rejected (invalid transition)
    Rejected {
        current_state: SystemState,
        requested_state: SystemState,
    },
    /// Transition failed due to concurrent modification
    ConcurrentModification {
        current_state: SystemState,
        requested_state: SystemState,
    },
}

/// Global System State Machine
/// 
/// Uses a single AtomicU8 for lock-free state management.
/// All threads can read the state with zero overhead using Relaxed ordering.
pub struct GlobalStateMachine {
    /// Current system state (stored as u8 for atomic operations)
    state: AtomicU8,
    /// Count of successful transitions
    transition_count: AtomicU8,
    /// Count of rejected transitions
    rejection_count: AtomicU8,
    /// Timestamp of last state change (nanoseconds)
    last_transition_timestamp_ns: AtomicU8,
}

unsafe impl Send for GlobalStateMachine {}
unsafe impl Sync for GlobalStateMachine {}

impl GlobalStateMachine {
    /// Create a new state machine starting in Trading state.
    pub fn new() -> Self {
        Self {
            state: AtomicU8::new(SystemState::Trading as u8),
            transition_count: AtomicU8::new(0),
            rejection_count: AtomicU8::new(0),
            last_transition_timestamp_ns: AtomicU8::new(0),
        }
    }

    /// Get the current system state.
    /// 
    /// This is a zero-cost operation using Relaxed ordering.
    /// Safe to call from any thread at any frequency.
    #[inline]
    pub fn get_state(&self) -> SystemState {
        let value = self.state.load(Ordering::Relaxed);
        SystemState::from_u8(value).unwrap_or(SystemState::Dead)
    }

    /// Attempt to transition to a new state.
    /// 
    /// Uses CAS loop to ensure atomic transition.
    /// Validates that the transition is allowed before proceeding.
    /// 
    /// # Arguments
    /// * `target_state` - The desired new state
    /// * `timestamp_ns` - Current timestamp for tracking
    #[inline]
    pub fn transition(&self, target_state: SystemState, timestamp_ns: u64) -> TransitionResult {
        let mut current_value = self.state.load(Ordering::Acquire);
        
        loop {
            let current_state = SystemState::from_u8(current_value)
                .unwrap_or(SystemState::Dead);
            
            // Validate transition
            if !StateTransition::is_valid(current_state, target_state) {
                self.rejection_count.fetch_add(1, Ordering::Relaxed);
                return TransitionResult::Rejected {
                    current_state,
                    requested_state: target_state,
                };
            }
            
            // Attempt CAS
            match self.state.compare_exchange(
                current_value,
                target_state as u8,
                Ordering::SeqCst,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // Success!
                    self.transition_count.fetch_add(1, Ordering::Relaxed);
                    // Store timestamp (truncated to fit in u64 stored as two u32s)
                    // For simplicity, we just track that a transition occurred
                    return TransitionResult::Success(target_state);
                }
                Err(actual) => {
                    // Another thread modified the state, retry
                    current_value = actual;
                    
                    // Check if we've reached a terminal state
                    if let Some(s) = SystemState::from_u8(current_value) {
                        if s.is_terminal() {
                            return TransitionResult::Rejected {
                                current_state: s,
                                requested_state: target_state,
                            };
                        }
                    }
                }
            }
        }
    }

    /// Quick check if system is in Trading state.
    /// 
    /// Optimized for the hot path - uses Relaxed ordering.
    #[inline]
    pub fn is_trading(&self) -> bool {
        self.state.load(Ordering::Relaxed) == SystemState::Trading as u8
    }

    /// Quick check if system is halted (any non-Trading state).
    #[inline]
    pub fn is_halted(&self) -> bool {
        self.state.load(Ordering::Relaxed) != SystemState::Trading as u8
    }

    /// Quick check if system is dead.
    #[inline]
    pub fn is_dead(&self) -> bool {
        self.state.load(Ordering::Relaxed) == SystemState::Dead as u8
    }

    /// Emergency transition to Dead state.
    /// 
    /// This transition is always allowed from any state.
    /// Uses SeqCst ordering to ensure immediate visibility.
    #[inline]
    pub fn emergency_stop(&self, timestamp_ns: u64) {
        self.state.store(SystemState::Dead as u8, Ordering::SeqCst);
        self.transition_count.fetch_add(1, Ordering::Relaxed);
        let _ = timestamp_ns; // Could store this for audit
    }

    /// Initiate graceful halt (stop new orders).
    #[inline]
    pub fn halt(&self, timestamp_ns: u64) -> TransitionResult {
        self.transition(SystemState::Halting, timestamp_ns)
    }

    /// Resume trading from Halting or Paused state.
    #[inline]
    pub fn resume(&self, timestamp_ns: u64) -> TransitionResult {
        self.transition(SystemState::Trading, timestamp_ns)
    }

    /// Start position flattening.
    #[inline]
    pub fn flatten(&self, timestamp_ns: u64) -> TransitionResult {
        self.transition(SystemState::Flattening, timestamp_ns)
    }

    /// Enter paused state (for maintenance).
    #[inline]
    pub fn pause(&self, timestamp_ns: u64) -> TransitionResult {
        self.transition(SystemState::Paused, timestamp_ns)
    }

    /// Get statistics about state transitions.
    pub fn stats(&self) -> StateMachineStats {
        let current_value = self.state.load(Ordering::Relaxed);
        StateMachineStats {
            current_state: SystemState::from_u8(current_value).unwrap_or(SystemState::Dead),
            transition_count: self.transition_count.load(Ordering::Relaxed),
            rejection_count: self.rejection_count.load(Ordering::Relaxed),
        }
    }

    /// Reset counters (for testing).
    #[inline]
    pub fn reset_counters(&self) {
        self.transition_count.store(0, Ordering::Relaxed);
        self.rejection_count.store(0, Ordering::Relaxed);
    }
}

impl Default for GlobalStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from the state machine
#[derive(Debug, Clone)]
pub struct StateMachineStats {
    pub current_state: SystemState,
    pub transition_count: u8,
    pub rejection_count: u8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::sync::Arc;

    #[test]
    fn test_initial_state() {
        let sm = GlobalStateMachine::new();
        assert_eq!(sm.get_state(), SystemState::Trading);
        assert!(sm.is_trading());
        assert!(!sm.is_halted());
        assert!(!sm.is_dead());
    }

    #[test]
    fn test_valid_transitions() {
        let sm = GlobalStateMachine::new();
        
        // Trading -> Halting
        let result = sm.halt(1000);
        assert!(matches!(result, TransitionResult::Success(SystemState::Halting)));
        assert_eq!(sm.get_state(), SystemState::Halting);
        
        // Halting -> Flattening
        let result = sm.flatten(2000);
        assert!(matches!(result, TransitionResult::Success(SystemState::Flattening)));
        
        // Flattening -> Dead
        let result = sm.transition(SystemState::Dead, 3000);
        assert!(matches!(result, TransitionResult::Success(SystemState::Dead)));
        assert!(sm.is_dead());
    }

    #[test]
    fn test_invalid_transition() {
        let sm = GlobalStateMachine::new();
        
        // Trading -> Flattening (invalid, must go through Halting first)
        let result = sm.transition(SystemState::Flattening, 1000);
        assert!(matches!(result, TransitionResult::Rejected { .. }));
        
        // State should still be Trading
        assert_eq!(sm.get_state(), SystemState::Trading);
    }

    #[test]
    fn test_emergency_stop() {
        let sm = GlobalStateMachine::new();
        
        // Can emergency stop from any state
        sm.emergency_stop(1000);
        assert!(sm.is_dead());
        
        // Even when already halted
        let sm2 = GlobalStateMachine::new();
        let _ = sm2.halt(1000);
        sm2.emergency_stop(2000);
        assert!(sm2.is_dead());
    }

    #[test]
    fn test_any_to_dead_transition() {
        let sm = GlobalStateMachine::new();
        
        // Any state can transition to Dead
        let result = sm.transition(SystemState::Dead, 1000);
        assert!(matches!(result, TransitionResult::Success(SystemState::Dead)));
    }

    #[test]
    fn test_concurrent_access() {
        let sm = Arc::new(GlobalStateMachine::new());
        let success_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        
        let mut handles = vec![];
        
        // Multiple threads trying to halt
        for _ in 0..10 {
            let sm_clone = Arc::clone(&sm);
            let success_clone = Arc::clone(&success_count);
            handles.push(thread::spawn(move || {
                let result = sm_clone.halt(1000);
                if matches!(result, TransitionResult::Success(_)) {
                    success_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }));
        }
        
        for h in handles {
            let _ = h.join();
        }
        
        // Exactly one should succeed
        assert_eq!(success_count.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(sm.get_state(), SystemState::Halting);
    }

    #[test]
    fn test_state_allows_methods() {
        assert!(SystemState::Trading.allows_new_orders());
        assert!(!SystemState::Halting.allows_new_orders());
        assert!(!SystemState::Dead.allows_new_orders());
        
        assert!(SystemState::Trading.allows_cancellations());
        assert!(SystemState::Halting.allows_cancellations());
        assert!(!SystemState::Dead.allows_cancellations());
        
        assert!(SystemState::Flattening.is_flattening());
        assert!(!SystemState::Trading.is_flattening());
    }

    #[test]
    fn test_from_u8_roundtrip() {
        for i in 0u8..6 {
            if let Some(state) = SystemState::from_u8(i) {
                // Valid states should roundtrip
                assert_eq!(SystemState::from_u8(state as u8), Some(state));
            }
        }
        
        // Invalid values should return None
        assert_eq!(SystemState::from_u8(100), None);
        assert_eq!(SystemState::from_u8(255), None);
    }

    #[test]
    fn test_stats() {
        let sm = GlobalStateMachine::new();
        
        let initial_stats = sm.stats();
        assert_eq!(initial_stats.current_state, SystemState::Trading);
        assert_eq!(initial_stats.transition_count, 0);
        assert_eq!(initial_stats.rejection_count, 0);
        
        // Make some transitions
        let _ = sm.halt(1000);
        let _ = sm.resume(2000);
        
        // Try invalid transition
        let _ = sm.transition(SystemState::Flattening, 3000); // Invalid from Trading
        
        let final_stats = sm.stats();
        assert_eq!(final_stats.transition_count, 2);
        assert!(final_stats.rejection_count >= 1);
    }
}
