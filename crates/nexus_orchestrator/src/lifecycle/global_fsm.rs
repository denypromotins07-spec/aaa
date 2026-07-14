//! Global Lifecycle Finite State Machine for NEXUS-OMEGA
//! Governs transitions: Bootstrapping -> ShadowMode -> PreFlightChecks -> LiveTrading -> GracefulShutdown -> CatastrophicHalt

use std::sync::atomic::{AtomicU8, Ordering};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrchestratorState {
    Bootstrapping = 0,
    ShadowMode = 1,
    PreFlightChecks = 2,
    LiveTrading = 3,
    GracefulShutdown = 4,
    CatastrophicHalt = 5,
}

impl From<u8> for OrchestratorState {
    fn from(value: u8) -> Self {
        match value {
            0 => OrchestratorState::Bootstrapping,
            1 => OrchestratorState::ShadowMode,
            2 => OrchestratorState::PreFlightChecks,
            3 => OrchestratorState::LiveTrading,
            4 => OrchestratorState::GracefulShutdown,
            5 => OrchestratorState::CatastrophicHalt,
            _ => OrchestratorState::CatastrophicHalt,
        }
    }
}

impl fmt::Display for OrchestratorState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrchestratorState::Bootstrapping => write!(f, "Bootstrapping"),
            OrchestratorState::ShadowMode => write!(f, "ShadowMode"),
            OrchestratorState::PreFlightChecks => write!(f, "PreFlightChecks"),
            OrchestratorState::LiveTrading => write!(f, "LiveTrading"),
            OrchestratorState::GracefulShutdown => write!(f, "GracefulShutdown"),
            OrchestratorState::CatastrophicHalt => write!(f, "CatastrophicHalt"),
        }
    }
}

/// Thread-safe global FSM using atomic operations for lock-free state transitions
pub struct GlobalLifecycleFSM {
    state: AtomicU8,
}

impl GlobalLifecycleFSM {
    pub fn new() -> Self {
        Self {
            state: AtomicU8::new(OrchestratorState::Bootstrapping as u8),
        }
    }

    /// Get current state
    pub fn current_state(&self) -> OrchestratorState {
        OrchestratorState::from(self.state.load(Ordering::Acquire))
    }

    /// Attempt to transition to a new state
    /// Returns true if transition was successful, false if invalid transition
    pub fn transition(&self, from: OrchestratorState, to: OrchestratorState) -> bool {
        // Validate transition rules
        if !Self::is_valid_transition(from, to) {
            tracing::warn!(
                "Invalid state transition attempted: {} -> {}",
                from,
                to
            );
            return false;
        }

        // Atomic compare-and-swap
        let expected = from as u8;
        match self
            .state
            .compare_exchange(expected, to as u8, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {
                tracing::info!("FSM Transition: {} -> {}", from, to);
                true
            }
            Err(actual) => {
                tracing::warn!(
                    "State changed concurrently: expected {}, found {}",
                    from,
                    OrchestratorState::from(actual)
                );
                false
            }
        }
    }

    /// Force transition (used only for catastrophic halt)
    pub fn force_transition(&self, to: OrchestratorState) {
        self.state.store(to as u8, Ordering::Release);
        tracing::warn!("Force transition to: {}", to);
    }

    /// Check if currently in live trading mode
    pub fn is_live_trading(&self) -> bool {
        self.current_state() == OrchestratorState::LiveTrading
    }

    /// Check if in catastrophic halt
    pub fn is_halted(&self) -> bool {
        matches!(
            self.current_state(),
            OrchestratorState::CatastrophicHalt | OrchestratorState::GracefulShutdown
        )
    }

    fn is_valid_transition(from: OrchestratorState, to: OrchestratorState) -> bool {
        match from {
            OrchestratorState::Bootstrapping => {
                matches!(to, OrchestratorState::ShadowMode | OrchestratorState::CatastrophicHalt)
            }
            OrchestratorState::ShadowMode => {
                matches!(
                    to,
                    OrchestratorState::PreFlightChecks
                        | OrchestratorState::GracefulShutdown
                        | OrchestratorState::CatastrophicHalt
                )
            }
            OrchestratorState::PreFlightChecks => {
                matches!(
                    to,
                    OrchestratorState::LiveTrading
                        | OrchestratorState::ShadowMode
                        | OrchestratorState::CatastrophicHalt
                )
            }
            OrchestratorState::LiveTrading => {
                matches!(
                    to,
                    OrchestratorState::GracefulShutdown | OrchestratorState::CatastrophicHalt
                )
            }
            OrchestratorState::GracefulShutdown => {
                matches!(to, OrchestratorState::CatastrophicHalt)
            }
            OrchestratorState::CatastrophicHalt => false, // Terminal state
        }
    }
}

impl Default for GlobalLifecycleFSM {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_transitions() {
        let fsm = GlobalLifecycleFSM::new();
        assert_eq!(fsm.current_state(), OrchestratorState::Bootstrapping);

        assert!(fsm.transition(
            OrchestratorState::Bootstrapping,
            OrchestratorState::ShadowMode
        ));
        assert_eq!(fsm.current_state(), OrchestratorState::ShadowMode);

        assert!(fsm.transition(
            OrchestratorState::ShadowMode,
            OrchestratorState::PreFlightChecks
        ));
        assert_eq!(fsm.current_state(), OrchestratorState::PreFlightChecks);
    }

    #[test]
    fn test_invalid_transition() {
        let fsm = GlobalLifecycleFSM::new();
        // Cannot jump directly to LiveTrading
        assert!(!fsm.transition(
            OrchestratorState::Bootstrapping,
            OrchestratorState::LiveTrading
        ));
    }
}
