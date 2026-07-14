//! Maintenance Margin Finite State Machine.
//! 
//! Tracks portfolio margin state and emits MarginCallImminent events when
//! the margin ratio drops below safe thresholds.

use crate::margin::fixed_point_pnl::{FixedPoint, calculate_margin_ratio};
use std::sync::atomic::{AtomicI128, AtomicBool, Ordering};

/// Margin states for the FSM
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarginState {
    /// Healthy: margin ratio > 2.0x
    Healthy,
    /// Warning: 1.5x < margin ratio <= 2.0x
    Warning,
    /// Critical: 1.0x < margin ratio <= 1.5x - emit MarginCallImminent
    Critical,
    /// LiquidationImminent: margin ratio <= 1.0x
    LiquidationImminent,
}

/// Thresholds (scaled by 1e18)
const HEALTHY_THRESHOLD: i128 = 2_000_000_000_000_000_000; // 2.0x
const WARNING_THRESHOLD: i128 = 1_500_000_000_000_000_000; // 1.5x
const CRITICAL_THRESHOLD: i128 = 1_000_000_000_000_000_000; // 1.0x

/// Event emitted when margin state changes
#[derive(Debug, Clone, Copy)]
pub enum MarginEvent {
    StateChanged(MarginState),
    MarginCallImminent,
    LiquidationWarning,
}

/// Lock-free maintenance margin FSM using atomic operations.
pub struct MaintenanceMarginFsm {
    /// Current margin ratio (scaled i128)
    current_ratio: AtomicI128,
    /// Current FSM state
    current_state: AtomicU8,
    /// Flag indicating a margin call was emitted
    margin_call_emitted: AtomicBool,
}

// Map u8 to MarginState
impl MaintenanceMarginFsm {
    pub fn new() -> Self {
        Self {
            current_ratio: AtomicI128::new(HEALTHY_THRESHOLD),
            current_state: AtomicU8::new(MarginState::Healthy as u8),
            margin_call_emitted: AtomicBool::new(false),
        }
    }

    /// Update the margin ratio and transition state if needed.
    /// Returns an event if state changed or margin call triggered.
    pub fn update_ratio(&self, new_ratio: FixedPoint) -> Option<MarginEvent> {
        let ratio_scaled = new_ratio.to_scaled();
        self.current_ratio.store(ratio_scaled, Ordering::SeqCst);

        let new_state = self.calculate_state(ratio_scaled);
        let old_state_u8 = self.current_state.load(Ordering::Relaxed);
        let old_state = self.u8_to_state(old_state_u8);

        if new_state != old_state {
            self.current_state.store(new_state as u8, Ordering::SeqCst);
            
            let event = match new_state {
                MarginState::Critical => {
                    self.margin_call_emitted.store(true, Ordering::SeqCst);
                    Some(MarginEvent::MarginCallImminent)
                }
                MarginState::LiquidationImminent => {
                    self.margin_call_emitted.store(true, Ordering::SeqCst);
                    Some(MarginEvent::LiquidationWarning)
                }
                _ => Some(MarginEvent::StateChanged(new_state)),
            };
            return event;
        }

        None
    }

    fn calculate_state(&self, ratio: i128) -> MarginState {
        if ratio > HEALTHY_THRESHOLD {
            MarginState::Healthy
        } else if ratio > WARNING_THRESHOLD {
            MarginState::Warning
        } else if ratio > CRITICAL_THRESHOLD {
            MarginState::Critical
        } else {
            MarginState::LiquidationImminent
        }
    }

    fn u8_to_state(&self, val: u8) -> MarginState {
        match val {
            0 => MarginState::Healthy,
            1 => MarginState::Warning,
            2 => MarginState::Critical,
            3 => MarginState::LiquidationImminent,
            _ => MarginState::Healthy,
        }
    }

    pub fn get_current_state(&self) -> MarginState {
        self.u8_to_state(self.current_state.load(Ordering::Relaxed))
    }

    pub fn get_current_ratio(&self) -> i128 {
        self.current_ratio.load(Ordering::Relaxed)
    }

    pub fn is_margin_call_active(&self) -> bool {
        self.margin_call_emitted.load(Ordering::Relaxed)
    }

    /// Reset margin call flag after deleveraging
    pub fn reset_margin_call(&self) {
        self.margin_call_emitted.store(false, Ordering::SeqCst);
    }
}

impl Default for MaintenanceMarginFsm {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fsm_transitions() {
        let fsm = MaintenanceMarginFsm::new();
        
        // Start healthy
        assert_eq!(fsm.get_current_state(), MarginState::Healthy);
        
        // Drop to warning
        let warning_ratio = FixedPoint::from_scaled(WARNING_THRESHOLD + 1);
        let event = fsm.update_ratio(warning_ratio);
        assert!(matches!(event, Some(MarginEvent::StateChanged(MarginState::Warning))));
        
        // Drop to critical - should emit MarginCallImminent
        let critical_ratio = FixedPoint::from_scaled(CRITICAL_THRESHOLD + 1);
        let event = fsm.update_ratio(critical_ratio);
        assert!(matches!(event, Some(MarginEvent::MarginCallImminent)));
        assert!(fsm.is_margin_call_active());
        
        // Reset
        fsm.reset_margin_call();
        assert!(!fsm.is_margin_call_active());
    }
}
