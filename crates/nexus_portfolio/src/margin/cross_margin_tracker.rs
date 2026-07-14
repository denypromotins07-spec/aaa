//! Cross-Margin Tracker - Aggregates positions and balances across multiple exchanges.
//! 
//! Uses lock-free atomic accumulators for real-time margin calculations.

use crate::margin::fixed_point_pnl::{FixedPoint, calculate_unrealized_pnl, calculate_maintenance_margin, calculate_margin_ratio};
use crate::margin::maintenance_margin_fsm::{MaintenanceMarginFsm, MarginEvent, MarginState};
use std::sync::atomic::{AtomicI128, AtomicUsize, Ordering};
use std::collections::HashMap;

/// Represents a single position on an exchange
#[derive(Debug, Clone)]
pub struct Position {
    pub exchange_id: u8,
    pub symbol: [u8; 16], // Fixed-size symbol buffer
    pub size: i128, // Positive = long, negative = short
    pub entry_price_scaled: i128,
    pub mark_price_scaled: i128,
    pub notional_scaled: i128,
}

/// Balance for a single exchange
#[derive(Debug, Clone)]
pub struct ExchangeBalance {
    pub exchange_id: u8,
    pub available_scaled: i128,
    pub total_scaled: i128,
    pub unrealized_pnl_scaled: i128,
}

/// Aggregated portfolio state
pub struct CrossMarginTracker {
    /// Total equity across all exchanges (scaled)
    total_equity: AtomicI128,
    /// Total maintenance margin required (scaled)
    total_maintenance_margin: AtomicI128,
    /// Number of active positions
    position_count: AtomicUsize,
    /// The margin FSM
    margin_fsm: MaintenanceMarginFsm,
    /// Flag indicating if new entries should be halted
    halt_new_entries: std::sync::atomic::AtomicBool,
}

impl CrossMarginTracker {
    pub fn new() -> Self {
        Self {
            total_equity: AtomicI128::new(0),
            total_maintenance_margin: AtomicI128::new(0),
            position_count: AtomicUsize::new(0),
            margin_fsm: MaintenanceMarginFsm::new(),
            halt_new_entries: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Update total equity atomically
    pub fn update_equity(&self, new_equity_scaled: i128) {
        self.total_equity.store(new_equity_scaled, Ordering::SeqCst);
        self.recalculate_margin_state();
    }

    /// Update total maintenance margin atomically
    pub fn update_maintenance_margin(&self, new_mm_scaled: i128) {
        self.total_maintenance_margin.store(new_mm_scaled, Ordering::SeqCst);
        self.recalculate_margin_state();
    }

    /// Add a position to the tracker
    pub fn add_position(&self, position: &Position) {
        self.position_count.fetch_add(1, Ordering::Relaxed);
        
        // Calculate position's contribution to maintenance margin
        let notional = FixedPoint::from_scaled(position.notional_scaled);
        // Assume 0.5% MM rate for perps (exchange-specific rates should be passed in)
        let mm_rate = FixedPoint::from_scaled(5_000_000_000_000_000); // 0.005
        
        if let Ok(mm_required) = calculate_maintenance_margin(notional, mm_rate) {
            let current_mm = self.total_maintenance_margin.load(Ordering::Relaxed);
            self.total_maintenance_margin.store(
                current_mm.saturating_add(mm_required.to_scaled()),
                Ordering::SeqCst
            );
        }
        
        self.recalculate_margin_state();
    }

    /// Remove a position from the tracker
    pub fn remove_position(&self, position: &Position) {
        self.position_count.fetch_sub(1, Ordering::Relaxed);
        
        // Subtract position's MM contribution
        let notional = FixedPoint::from_scaled(position.notional_scaled.abs());
        let mm_rate = FixedPoint::from_scaled(5_000_000_000_000_000);
        
        if let Ok(mm_required) = calculate_maintenance_margin(notional, mm_rate) {
            let current_mm = self.total_maintenance_margin.load(Ordering::Relaxed);
            self.total_maintenance_margin.store(
                current_mm.saturating_sub(mm_required.to_scaled()),
                Ordering::SeqCst
            );
        }
        
        self.recalculate_margin_state();
    }

    /// Recalculate margin ratio and update FSM
    fn recalculate_margin_state(&self) {
        let equity = self.total_equity.load(Ordering::Relaxed);
        let mm = self.total_maintenance_margin.load(Ordering::Relaxed);
        
        if mm == 0 {
            return; // No positions, skip
        }
        
        let equity_fp = FixedPoint::from_scaled(equity);
        let mm_fp = FixedPoint::from_scaled(mm);
        
        if let Ok(ratio) = calculate_margin_ratio(equity_fp, mm_fp) {
            if let Some(event) = self.margin_fsm.update_ratio(ratio) {
                match event {
                    MarginEvent::MarginCallImminent | MarginEvent::LiquidationWarning => {
                        self.halt_new_entries.store(true, Ordering::SeqCst);
                    }
                    MarginEvent::StateChanged(MarginState::Healthy) => {
                        self.halt_new_entries.store(false, Ordering::SeqCst);
                        self.margin_fsm.reset_margin_call();
                    }
                    _ => {}
                }
            }
        }
    }

    /// Check if new entries should be halted
    pub fn should_halt_new_entries(&self) -> bool {
        self.halt_new_entries.load(Ordering::Relaxed)
    }

    /// Get current margin state
    pub fn get_margin_state(&self) -> MarginState {
        self.margin_fsm.get_current_state()
    }

    /// Get current margin ratio (scaled)
    pub fn get_margin_ratio(&self) -> i128 {
        self.margin_fsm.get_current_ratio()
    }

    /// Manually reset halt flag after deleveraging
    pub fn reset_halt_flag(&self) {
        self.halt_new_entries.store(false, Ordering::SeqCst);
        self.margin_fsm.reset_margin_call();
    }
}

impl Default for CrossMarginTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_margin_tracker_halt_on_critical() {
        let tracker = CrossMarginTracker::new();
        
        // Set initial equity: $100k
        tracker.update_equity(100_000 * 1_000_000_000_000_000_000i128);
        
        // Add position requiring $40k MM (ratio = 2.5x, healthy)
        tracker.update_maintenance_margin(40_000 * 1_000_000_000_000_000_000i128);
        assert!(!tracker.should_halt_new_entries());
        
        // Increase MM to $80k (ratio = 1.25x, critical)
        tracker.update_maintenance_margin(80_000 * 1_000_000_000_000_000_000i128);
        assert!(tracker.should_halt_new_entries());
        assert_eq!(tracker.get_margin_state(), MarginState::Critical);
    }
}
