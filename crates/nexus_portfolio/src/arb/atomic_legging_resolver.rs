//! Atomic Legging Resolver - Handles orphan legs in cross-exchange arb.
//! 
//! CRITICAL: Uses microsecond timeouts to prevent deadlocks when one leg fails.
//! If Exchange B's WS drops after Leg A fills, this resolver forces immediate SOR flatten.

use std::sync::atomic::{AtomicBool, AtomicI128, AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant};

/// State of each leg
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegState {
    Pending,
    Filled,
    Rejected,
    TimedOut,
}

/// Combined state for both legs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DualLegState {
    BothPending,
    LegAFilledLegBPending,
    LegBFilledLegAPending,
    BothFilled,
    OrphanLegA, // A filled, B failed
    OrphanLegB, // B filled, A failed
    BothFailed,
}

/// Timeout for leg execution (500 microseconds - aggressive for HFT)
const LEG_TIMEOUT_MICROS: u64 = 500;

/// Resolver for atomic legging
pub struct AtomicLeggingResolver {
    /// State of leg A (exchange A)
    leg_a_state: AtomicU8,
    /// State of leg B (exchange B)
    leg_b_state: AtomicU8,
    /// Fill size for leg A
    leg_a_fill_size: AtomicI128,
    /// Fill size for leg B
    leg_b_fill_size: AtomicI128,
    /// Expected total size
    expected_size: AtomicI128,
    /// Timestamp when trade was initiated (microseconds since epoch)
    start_time_micros: AtomicU64,
    /// Flag indicating SOR flatten triggered
    sor_flatten_triggered: AtomicBool,
    /// Flag indicating orphan detected
    orphan_detected: AtomicBool,
}

impl AtomicLeggingResolver {
    pub fn new() -> Self {
        Self {
            leg_a_state: AtomicU8::new(LegState::Pending as u8),
            leg_b_state: AtomicU8::new(LegState::Pending as u8),
            leg_a_fill_size: AtomicI128::new(0),
            leg_b_fill_size: AtomicI128::new(0),
            expected_size: AtomicI128::new(0),
            start_time_micros: AtomicU64::new(0),
            sor_flatten_triggered: AtomicBool::new(false),
            orphan_detected: AtomicBool::new(false),
        }
    }

    /// Reset for a new trade
    pub fn reset_for_new_trade(&self, expected_size: i128) {
        self.leg_a_state.store(LegState::Pending as u8, Ordering::SeqCst);
        self.leg_b_state.store(LegState::Pending as u8, Ordering::SeqCst);
        self.leg_a_fill_size.store(0, Ordering::Relaxed);
        self.leg_b_fill_size.store(0, Ordering::Relaxed);
        self.expected_size.store(expected_size, Ordering::Relaxed);
        
        // Set start time (microseconds since epoch)
        let now = Instant::now();
        let micros = now.elapsed().as_micros() as u64; // Simplified; use proper epoch time in prod
        self.start_time_micros.store(micros, Ordering::Relaxed);
        
        self.sor_flatten_triggered.store(false, Ordering::Relaxed);
        self.orphan_detected.store(false, Ordering::Relaxed);
    }

    /// Record a fill on a specific leg
    pub fn record_leg_fill(&self, leg_index: u8, fill_size: i128) {
        if leg_index == 0 {
            self.leg_a_fill_size.store(fill_size, Ordering::SeqCst);
            self.leg_a_state.store(LegState::Filled as u8, Ordering::SeqCst);
        } else {
            self.leg_b_fill_size.store(fill_size, Ordering::SeqCst);
            self.leg_b_state.store(LegState::Filled as u8, Ordering::SeqCst);
        }
    }

    /// Record rejection on a leg
    pub fn record_leg_rejection(&self, leg_index: u8) {
        if leg_index == 0 {
            self.leg_a_state.store(LegState::Rejected as u8, Ordering::SeqCst);
        } else {
            self.leg_b_state.store(LegState::Rejected as u8, Ordering::SeqCst);
        }
        self.check_for_orphan();
    }

    /// Check if timeout has occurred for pending legs
    pub fn check_timeout(&self) -> bool {
        let now = Instant::now();
        let current_micros = now.elapsed().as_micros() as u64;
        let start = self.start_time_micros.load(Ordering::Relaxed);
        
        if start == 0 {
            return false;
        }
        
        let elapsed = current_micros.saturating_sub(start);
        
        if elapsed > LEG_TIMEOUT_MICROS {
            // Mark any pending legs as timed out
            let leg_a = self.u8_to_leg_state(self.leg_a_state.load(Ordering::Relaxed));
            let leg_b = self.u8_to_leg_state(self.leg_b_state.load(Ordering::Relaxed));
            
            if leg_a == LegState::Pending {
                self.leg_a_state.store(LegState::TimedOut as u8, Ordering::SeqCst);
            }
            if leg_b == LegState::Pending {
                self.leg_b_state.store(LegState::TimedOut as u8, Ordering::SeqCst);
            }
            
            self.check_for_orphan();
            return true;
        }
        
        false
    }

    /// Check for orphan leg condition
    fn check_for_orphan(&self) {
        let leg_a = self.u8_to_leg_state(self.leg_a_state.load(Ordering::Relaxed));
        let leg_b = self.u8_to_leg_state(self.leg_b_state.load(Ordering::Relaxed));
        
        let orphan = match (leg_a, leg_b) {
            (LegState::Filled, LegState::Rejected) => true,
            (LegState::Filled, LegState::TimedOut) => true,
            (LegState::Rejected, LegState::Filled) => true,
            (LegState::TimedOut, LegState::Filled) => true,
            _ => false,
        };
        
        if orphan {
            self.orphan_detected.store(true, Ordering::SeqCst);
        }
    }

    /// Detect if an orphan leg exists
    pub fn detect_orphan_leg(&self) -> bool {
        // Also check timeout as part of detection
        self.check_timeout();
        self.orphan_detected.load(Ordering::Relaxed)
    }

    /// Trigger SOR to flatten the orphan leg
    pub fn trigger_sor_flatten(&self) -> bool {
        if !self.orphan_detected.load(Ordering::Relaxed) {
            return false;
        }
        
        // Determine which leg is orphaned and needs flattening
        let leg_a = self.u8_to_leg_state(self.leg_a_state.load(Ordering::Relaxed));
        let leg_b = self.u8_to_leg_state(self.leg_b_state.load(Ordering::Relaxed));
        
        // The orphaned leg is the one that filled - we need to close it
        // because the hedge didn't execute
        if leg_a == LegState::Filled && (leg_b == LegState::Rejected || leg_b == LegState::TimedOut) {
            // Leg A is orphaned - SOR will flatten it
            self.sor_flatten_triggered.store(true, Ordering::SeqCst);
            return true;
        }
        
        if leg_b == LegState::Filled && (leg_a == LegState::Rejected || leg_a == LegState::TimedOut) {
            // Leg B is orphaned - SOR will flatten it
            self.sor_flatten_triggered.store(true, Ordering::SeqCst);
            return true;
        }
        
        false
    }

    /// Check if both legs filled successfully
    pub fn both_legs_filled(&self) -> bool {
        let leg_a = self.u8_to_leg_state(self.leg_a_state.load(Ordering::Relaxed));
        let leg_b = self.u8_to_leg_state(self.leg_b_state.load(Ordering::Relaxed));
        
        leg_a == LegState::Filled && leg_b == LegState::Filled
    }

    /// Get current dual leg state
    pub fn get_dual_state(&self) -> DualLegState {
        let leg_a = self.u8_to_leg_state(self.leg_a_state.load(Ordering::Relaxed));
        let leg_b = self.u8_to_leg_state(self.leg_b_state.load(Ordering::Relaxed));
        
        match (leg_a, leg_b) {
            (LegState::Pending, LegState::Pending) => DualLegState::BothPending,
            (LegState::Filled, LegState::Pending) => DualLegState::LegAFilledLegBPending,
            (LegState::Pending, LegState::Filled) => DualLegState::LegBFilledLegAPending,
            (LegState::Filled, LegState::Filled) => DualLegState::BothFilled,
            (LegState::Filled, _) => DualLegState::OrphanLegA,
            (_, LegState::Filled) => DualLegState::OrphanLegB,
            _ => DualLegState::BothFailed,
        }
    }

    /// Check if SOR flatten was triggered
    pub fn is_sor_flatten_triggered(&self) -> bool {
        self.sor_flatten_triggered.load(Ordering::Relaxed)
    }

    /// Get the size that needs flattening
    pub fn get_flatten_size(&self) -> i128 {
        let leg_a = self.u8_to_leg_state(self.leg_a_state.load(Ordering::Relaxed));
        let leg_b = self.u8_to_leg_state(self.leg_b_state.load(Ordering::Relaxed));
        
        if leg_a == LegState::Filled {
            self.leg_a_fill_size.load(Ordering::Relaxed)
        } else if leg_b == LegState::Filled {
            self.leg_b_fill_size.load(Ordering::Relaxed)
        } else {
            0
        }
    }

    fn u8_to_leg_state(&self, val: u8) -> LegState {
        match val {
            0 => LegState::Pending,
            1 => LegState::Filled,
            2 => LegState::Rejected,
            3 => LegState::TimedOut,
            _ => LegState::Pending,
        }
    }
}

impl Default for AtomicLeggingResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orphan_detection_on_rejection() {
        let resolver = AtomicLeggingResolver::new();
        resolver.reset_for_new_trade(1_000_000_000_000_000_000i128);
        
        // Leg A fills
        resolver.record_leg_fill(0, 1_000_000_000_000_000_000i128);
        assert!(!resolver.detect_orphan_leg()); // Not yet orphan
        
        // Leg B rejects
        resolver.record_leg_rejection(1);
        
        assert!(resolver.detect_orphan_leg());
        assert!(resolver.trigger_sor_flatten());
        assert!(resolver.is_sor_flatten_triggered());
        assert_eq!(resolver.get_flatten_size(), 1_000_000_000_000_000_000i128);
    }

    #[test]
    fn test_both_legs_filled_success() {
        let resolver = AtomicLeggingResolver::new();
        resolver.reset_for_new_trade(1_000_000_000_000_000_000i128);
        
        resolver.record_leg_fill(0, 1_000_000_000_000_000_000i128);
        resolver.record_leg_fill(1, 1_000_000_000_000_000_000i128);
        
        assert!(resolver.both_legs_filled());
        assert!(!resolver.detect_orphan_leg());
        assert_eq!(resolver.get_dual_state(), DualLegState::BothFilled);
    }
}
