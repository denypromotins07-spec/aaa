//! Lock-free Position and Margin Tracker using atomic operations.
//! All operations use AtomicI64 for thread-safe updates without locks.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use crate::fixed_point_math::FixedPoint;

/// Position representation with fixed-point math
#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub symbol_id: u32,
    pub quantity: FixedPoint,
    pub avg_entry_price: FixedPoint,
    pub realized_pnl: FixedPoint,
    pub unrealized_pnl: FixedPoint,
}

impl Position {
    #[inline]
    pub const fn new(symbol_id: u32) -> Self {
        Self {
            symbol_id,
            quantity: FixedPoint::from_raw(0),
            avg_entry_price: FixedPoint::from_raw(0),
            realized_pnl: FixedPoint::from_raw(0),
            unrealized_pnl: FixedPoint::from_raw(0),
        }
    }

    #[inline]
    pub fn is_flat(&self) -> bool {
        self.quantity.is_zero()
    }

    #[inline]
    pub fn is_long(&self) -> bool {
        self.quantity.is_positive()
    }

    #[inline]
    pub fn is_short(&self) -> bool {
        self.quantity.is_negative()
    }
}

/// Lock-free position tracker using atomic operations
/// Tracks position, margin, and P&L atomically without locks
pub struct LockFreePositionTracker {
    /// Net position in base units (scaled by 10^8)
    position_qty: AtomicI64,
    /// Average entry price (scaled by 10^8)
    avg_entry_price: AtomicI64,
    /// Realized P&L (scaled by 10^8)
    realized_pnl: AtomicI64,
    /// Initial margin required (scaled by 10^8)
    initial_margin: AtomicI64,
    /// Maintenance margin required (scaled by 10^8)
    maintenance_margin: AtomicI64,
    /// Account balance (scaled by 10^8)
    account_balance: AtomicI64,
    /// Available balance for trading (scaled by 10^8)
    available_balance: AtomicI64,
    /// Symbol ID
    symbol_id: AtomicU64,
    /// Update sequence number for consistency checks
    sequence: AtomicU64,
}

impl LockFreePositionTracker {
    /// Create a new position tracker with initial balance
    #[inline]
    pub fn new(symbol_id: u32, initial_balance: FixedPoint) -> Self {
        Self {
            position_qty: AtomicI64::new(0),
            avg_entry_price: AtomicI64::new(0),
            realized_pnl: AtomicI64::new(0),
            initial_margin: AtomicI64::new(0),
            maintenance_margin: AtomicI64::new(0),
            account_balance: AtomicI64::new(initial_balance.raw()),
            available_balance: AtomicI64::new(initial_balance.raw()),
            symbol_id: AtomicU64::new(symbol_id as u64),
            sequence: AtomicU64::new(0),
        }
    }

    /// Get current position quantity
    #[inline]
    pub fn get_position_qty(&self) -> FixedPoint {
        let raw = self.position_qty.load(Ordering::Acquire);
        FixedPoint::from_raw(raw)
    }

    /// Get average entry price
    #[inline]
    pub fn get_avg_entry_price(&self) -> FixedPoint {
        let raw = self.avg_entry_price.load(Ordering::Acquire);
        FixedPoint::from_raw(raw)
    }

    /// Get realized P&L
    #[inline]
    pub fn get_realized_pnl(&self) -> FixedPoint {
        let raw = self.realized_pnl.load(Ordering::Acquire);
        FixedPoint::from_raw(raw)
    }

    /// Get account balance
    #[inline]
    pub fn get_account_balance(&self) -> FixedPoint {
        let raw = self.account_balance.load(Ordering::Acquire);
        FixedPoint::from_raw(raw)
    }

    /// Get available balance
    #[inline]
    pub fn get_available_balance(&self) -> FixedPoint {
        let raw = self.available_balance.load(Ordering::Acquire);
        FixedPoint::from_raw(raw)
    }

    /// Get initial margin
    #[inline]
    pub fn get_initial_margin(&self) -> FixedPoint {
        let raw = self.initial_margin.load(Ordering::Acquire);
        FixedPoint::from_raw(raw)
    }

    /// Get maintenance margin
    #[inline]
    pub fn get_maintenance_margin(&self) -> FixedPoint {
        let raw = self.maintenance_margin.load(Ordering::Acquire);
        FixedPoint::from_raw(raw)
    }

    /// Atomically update position on fill
    /// Uses CAS loop for lock-free update
    /// Returns the new sequence number on success
    #[inline]
    pub fn update_on_fill(
        &self,
        fill_qty: FixedPoint,
        fill_price: FixedPoint,
        is_buy: bool,
    ) -> Result<u64, &'static str> {
        let qty_raw = fill_qty.raw();
        let price_raw = fill_price.raw();

        // Adjust quantity sign based on side
        let signed_qty = if is_buy { qty_raw } else { -qty_raw };

        // CAS loop for position update
        let mut seq;
        loop {
            let current_seq = self.sequence.load(Ordering::Acquire);
            let current_qty = self.position_qty.load(Ordering::Acquire);
            let current_avg = self.avg_entry_price.load(Ordering::Acquire);
            let current_margin = self.initial_margin.load(Ordering::Acquire);

            // Calculate new position
            let new_qty = current_qty + signed_qty;
            
            // Calculate new average entry price
            // If same direction, weighted average; if opposite, realize P&L
            let new_avg = if (current_qty >= 0 && signed_qty > 0) || 
                          (current_qty < 0 && signed_qty < 0) {
                // Same direction: weighted average
                let total_value = (current_qty as i128 * current_avg as i128) + 
                                  (signed_qty as i128 * price_raw as i128);
                let new_qty_abs = new_qty.abs() as i128;
                if new_qty_abs == 0 {
                    0i64
                } else {
                    ((total_value / new_qty_abs) as i64).min(i64::MAX).max(i64::MIN)
                }
            } else {
                // Opposite direction: keep existing avg for remaining position
                // or zero if fully closed
                if new_qty == 0 { 0 } else { current_avg }
            };

            // Calculate new margin requirement (simplified: 10% of notional)
            let notional = (new_qty.abs() as i128 * new_avg as i128) / 100_000_000i128;
            let new_margin = ((notional / 10) as i64).max(0);

            // Attempt CAS
            if self.position_qty.compare_exchange_weak(
                current_qty, new_qty, Ordering::AcqRel, Ordering::Acquire
            ).is_ok() {
                self.avg_entry_price.store(new_avg, Ordering::Release);
                self.initial_margin.store(new_margin, Ordering::Release);
                seq = self.sequence.fetch_add(1, Ordering::AcqRel) + 1;
                return Ok(seq);
            }
            // CAS failed, retry
        }
    }

    /// Atomically update realized P&L
    #[inline]
    pub fn add_realized_pnl(&self, pnl: FixedPoint) -> Result<u64, &'static str> {
        let pnl_raw = pnl.raw();
        
        loop {
            let current = self.realized_pnl.load(Ordering::Acquire);
            let new_val = current.saturating_add(pnl_raw);
            
            if self.realized_pnl.compare_exchange_weak(
                current, new_val, Ordering::AcqRel, Ordering::Acquire
            ).is_ok() {
                // Also update account balance
                self.update_account_balance(pnl)?;
                return Ok(self.sequence.load(Ordering::Relaxed));
            }
        }
    }

    /// Atomically update account balance
    #[inline]
    fn update_account_balance(&self, delta: FixedPoint) -> Result<(), &'static str> {
        let delta_raw = delta.raw();
        
        loop {
            let current = self.account_balance.load(Ordering::Acquire);
            let new_val = current.saturating_add(delta_raw);
            
            if self.account_balance.compare_exchange_weak(
                current, new_val, Ordering::AcqRel, Ordering::Acquire
            ).is_ok() {
                return Ok(());
            }
        }
    }

    /// Atomically reserve margin for a new order
    /// Returns true if margin was successfully reserved
    #[inline]
    pub fn try_reserve_margin(&self, required: FixedPoint) -> bool {
        let required_raw = required.raw();
        
        loop {
            let available = self.available_balance.load(Ordering::Acquire);
            
            if available < required_raw {
                return false; // Insufficient margin
            }
            
            let new_available = available - required_raw;
            
            if self.available_balance.compare_exchange_weak(
                available, new_available, Ordering::AcqRel, Ordering::Acquire
            ).is_ok() {
                return true;
            }
        }
    }

    /// Atomically release reserved margin
    #[inline]
    pub fn release_margin(&self, amount: FixedPoint) {
        let amount_raw = amount.raw();
        
        loop {
            let current = self.available_balance.load(Ordering::Acquire);
            let new_val = current.saturating_add(amount_raw);
            
            if self.available_balance.compare_exchange_weak(
                current, new_val, Ordering::AcqRel, Ordering::Acquire
            ).is_ok() {
                return;
            }
        }
    }

    /// Get current position snapshot
    #[inline]
    pub fn get_position_snapshot(&self) -> Position {
        let qty_raw = self.position_qty.load(Ordering::Acquire);
        let avg_raw = self.avg_entry_price.load(Ordering::Acquire);
        let pnl_raw = self.realized_pnl.load(Ordering::Acquire);
        let sym = self.symbol_id.load(Ordering::Relaxed) as u32;

        Position {
            symbol_id: sym,
            quantity: FixedPoint::from_raw(qty_raw),
            avg_entry_price: FixedPoint::from_raw(avg_raw),
            realized_pnl: FixedPoint::from_raw(pnl_raw),
            unrealized_pnl: FixedPoint::from_raw(0), // Calculated externally
        }
    }

    /// Check if position is flat
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.position_qty.load(Ordering::Acquire) == 0
    }

    /// Get sequence number for consistency checks
    #[inline]
    pub fn get_sequence(&self) -> u64 {
        self.sequence.load(Ordering::Acquire)
    }
}

// SAFETY: All operations are atomic and thread-safe
unsafe impl Send for LockFreePositionTracker {}
unsafe impl Sync for LockFreePositionTracker {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_creation() {
        let tracker = LockFreePositionTracker::new(1, FixedPoint::from_int(10000));
        
        assert!(tracker.is_flat());
        assert_eq!(tracker.get_account_balance().to_f64(), 10000.0);
        assert_eq!(tracker.get_available_balance().to_f64(), 10000.0);
    }

    #[test]
    fn test_update_on_fill() {
        let tracker = LockFreePositionTracker::new(1, FixedPoint::from_int(10000));
        
        // Buy 10 @ 100
        let result = tracker.update_on_fill(
            FixedPoint::from_int(10),
            FixedPoint::from_int(100),
            true,
        );
        
        assert!(result.is_ok());
        assert_eq!(tracker.get_position_qty().to_f64(), 10.0);
        assert_eq!(tracker.get_avg_entry_price().to_f64(), 100.0);
    }

    #[test]
    fn test_margin_reservation() {
        let tracker = LockFreePositionTracker::new(1, FixedPoint::from_int(10000));
        
        // Try to reserve 1000
        assert!(tracker.try_reserve_margin(FixedPoint::from_int(1000)));
        assert_eq!(tracker.get_available_balance().to_f64(), 9000.0);
        
        // Try to reserve more than available
        assert!(!tracker.try_reserve_margin(FixedPoint::from_int(10000)));
        
        // Release margin
        tracker.release_margin(FixedPoint::from_int(1000));
        assert_eq!(tracker.get_available_balance().to_f64(), 10000.0);
    }
}
