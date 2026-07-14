//! Atomic Position Accumulator - Lock-Free Position Tracking
//! 
//! Tracks net position per symbol using atomic operations and scaled integer math.
//! This prevents floating-point drift over thousands of trades.
//! 
//! CRITICAL: Uses satoshi/wei precision (scaled integers) to ensure exact
//! reconciliation even after 10,000+ partial fills and cancellations.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// Maximum number of symbols to track
const MAX_SYMBOLS: usize = 1024;

/// Cache-line padded atomic value to prevent false sharing
#[repr(align(64))]
struct PaddedAtomicI64 {
    value: AtomicI64,
    _padding: [u8; 56], // Padding to reach 64 bytes
}

impl PaddedAtomicI64 {
    fn new(value: i64) -> Self {
        Self {
            value: AtomicI64::new(value),
            _padding: [0; 56],
        }
    }

    #[inline]
    fn load(&self, ordering: Ordering) -> i64 {
        self.value.load(ordering)
    }

    #[inline]
    fn fetch_add(&self, delta: i64, ordering: Ordering) -> i64 {
        self.value.fetch_add(delta, ordering)
    }

    #[inline]
    fn store(&self, value: i64, ordering: Ordering) {
        self.value.store(value, ordering);
    }
}

/// Result of a position limit check
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionCheckResult {
    /// Position would be within limits
    Ok,
    /// Position would exceed maximum allowed
    WouldExceed {
        current: i64,
        requested_delta: i64,
        new_position: i64,
        max_allowed: i64,
    },
}

/// Atomic Position Accumulator
/// 
/// Maintains lock-free tracking of net positions per symbol.
/// All operations use atomic fetch-add for thread safety without locks.
/// 
/// ZERO-ALLOCATION GUARANTEE:
/// - Pre-allocated array for all symbol positions
/// - No heap allocations in hot path
/// - Scaled integer math prevents drift
pub struct AtomicPositionAccumulator {
    /// Per-symbol net positions (in base units, e.g., satoshis)
    positions: Box<[PaddedAtomicI64; MAX_SYMBOLS]>,
    /// Maximum absolute position size per symbol
    max_position: i64,
    /// Total count of position updates
    update_count: AtomicU64,
    /// Count of position limit violations
    violation_count: AtomicU64,
}

unsafe impl Send for AtomicPositionAccumulator {}
unsafe impl Sync for AtomicPositionAccumulator {}

impl AtomicPositionAccumulator {
    /// Create a new atomic position accumulator
    /// 
    /// # Arguments
    /// * `max_position` - Maximum absolute position size (in base units)
    pub fn new(max_position: i64) -> Self {
        // Initialize all positions to zero
        let positions: Box<[PaddedAtomicI64; MAX_SYMBOLS]> = 
            Box::new(std::array::from_fn(|_| PaddedAtomicI64::new(0)));
        
        Self {
            positions,
            max_position,
            update_count: AtomicU64::new(0),
            violation_count: AtomicU64::new(0),
        }
    }

    /// Check if a position update would exceed the limit WITHOUT applying it.
    /// 
    /// This is the CRITICAL function called by the PreTradeRiskGatekeeper
    /// before approving any order.
    /// 
    /// # Arguments
    /// * `symbol_hash` - Hash of the symbol identifier
    /// * `delta` - Change in position (positive for buy, negative for sell)
    /// 
    /// # Returns
    /// * `Ok(())` - Position would remain within limits
    /// * `Err(PositionCheckResult::WouldExceed)` - Position would exceed limit
    #[inline]
    pub fn check_position_would_exceed(
        &self,
        symbol_hash: u64,
        delta: i64,
    ) -> Result<(), PositionCheckResult> {
        let idx = (symbol_hash as usize) % MAX_SYMBOLS;
        let current = self.positions[idx].load(Ordering::Relaxed);
        
        // Calculate new position with overflow protection
        let new_position = match current.checked_add(delta) {
            Some(pos) => pos,
            None => {
                // Overflow occurred - this is definitely a violation
                self.violation_count.fetch_add(1, Ordering::Relaxed);
                return Err(PositionCheckResult::WouldExceed {
                    current,
                    requested_delta: delta,
                    new_position: current, // Use current as placeholder
                    max_allowed: self.max_position,
                });
            }
        };
        
        // Check absolute value against limit
        if new_position.abs() > self.max_position {
            self.violation_count.fetch_add(1, Ordering::Relaxed);
            return Err(PositionCheckResult::WouldExceed {
                current,
                requested_delta: delta,
                new_position,
                max_allowed: self.max_position,
            });
        }
        
        Ok(())
    }

    /// Update the position for a symbol (after a fill).
    /// 
    /// MUST be called atomically after an order fill is confirmed.
    /// Uses fetch-add for lock-free operation.
    /// 
    /// # Arguments
    /// * `symbol_hash` - Hash of the symbol identifier
    /// * `delta` - Change in position (positive for buy, negative for sell)
    #[inline]
    pub fn update_position(&self, symbol_hash: u64, delta: i64) {
        let idx = (symbol_hash as usize) % MAX_SYMBOLS;
        self.positions[idx].fetch_add(delta, Ordering::Relaxed);
        self.update_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the current net position for a symbol.
    /// 
    /// # Arguments
    /// * `symbol_hash` - Hash of the symbol identifier
    /// 
    /// # Returns
    /// Current net position (positive = long, negative = short)
    #[inline]
    pub fn get_position(&self, symbol_hash: u64) -> i64 {
        let idx = (symbol_hash as usize) % MAX_SYMBOLS;
        self.positions[idx].load(Ordering::Relaxed)
    }

    /// Set the position for a symbol (for initialization/reconciliation).
    /// 
    /// USE WITH CAUTION: Only call during system startup or manual reconciliation.
    /// 
    /// # Arguments
    /// * `symbol_hash` - Hash of the symbol identifier
    /// * `position` - New position value
    #[inline]
    pub fn set_position(&self, symbol_hash: u64, position: i64) {
        let idx = (symbol_hash as usize) % MAX_SYMBOLS;
        self.positions[idx].store(position, Ordering::SeqCst);
    }

    /// Get total count of position updates
    #[inline]
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }

    /// Get count of position limit violations
    #[inline]
    pub fn violation_count(&self) -> u64 {
        self.violation_count.load(Ordering::Relaxed)
    }

    /// Reset all positions to zero (for testing/system reset)
    /// 
    /// WARNING: Only use during system initialization or emergency reset.
    #[inline]
    pub fn reset_all_positions(&self) {
        for i in 0..MAX_SYMBOLS {
            self.positions[i].store(0, Ordering::SeqCst);
        }
    }

    /// Get the maximum allowed position size
    #[inline]
    pub fn max_position(&self) -> i64 {
        self.max_position
    }
}

/// Helper for converting between different quantity scales
pub mod scale_conversion {
    /// Convert BTC to satoshis
    #[inline]
    pub const fn btc_to_satoshi(btc: u64) -> u64 {
        btc.saturating_mul(100_000_000)
    }

    /// Convert satoshis to BTC (truncates)
    #[inline]
    pub const fn satoshi_to_btc(satoshis: u64) -> u64 {
        satoshis / 100_000_000
    }

    /// Convert ETH to wei
    #[inline]
    pub const fn eth_to_wei(eth: u64) -> u128 {
        (eth as u128).saturating_mul(1_000_000_000_000_000_000)
    }

    /// Convert wei to ETH (truncates)
    #[inline]
    pub const fn wei_to_eth(wei: u128) -> u64 {
        (wei / 1_000_000_000_000_000_000) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_position_tracking() {
        let accumulator = AtomicPositionAccumulator::new(10_000_000_000); // 100 BTC
        
        assert_eq!(accumulator.get_position(42), 0);
        
        // Buy 1 BTC (100M satoshis)
        accumulator.update_position(42, 100_000_000);
        assert_eq!(accumulator.get_position(42), 100_000_000);
        
        // Sell 0.5 BTC (50M satoshis)
        accumulator.update_position(42, -50_000_000);
        assert_eq!(accumulator.get_position(42), 50_000_000);
    }

    #[test]
    fn test_position_limit_check() {
        let accumulator = AtomicPositionAccumulator::new(1_000_000_000); // 10 BTC max
        
        // Should pass: buying 5 BTC when limit is 10 BTC
        let result = accumulator.check_position_would_exceed(42, 500_000_000);
        assert_eq!(result, Ok(()));
        
        // Should fail: buying 15 BTC when limit is 10 BTC
        let result = accumulator.check_position_would_exceed(42, 1_500_000_000);
        assert!(matches!(result, Err(PositionCheckResult::WouldExceed { .. })));
    }

    #[test]
    fn test_short_position_limit() {
        let accumulator = AtomicPositionAccumulator::new(1_000_000_000); // 10 BTC max
        
        // Should pass: shorting 5 BTC
        let result = accumulator.check_position_would_exceed(42, -500_000_000);
        assert_eq!(result, Ok(()));
        
        // Should fail: shorting 15 BTC
        let result = accumulator.check_position_would_exceed(42, -1_500_000_000);
        assert!(matches!(result, Err(PositionCheckResult::WouldExceed { .. })));
    }

    #[test]
    fn test_partial_fill_reconciliation() {
        let accumulator = AtomicPositionAccumulator::new(10_000_000_000);
        
        // Simulate partial fills over many iterations
        for i in 0..1000 {
            accumulator.update_position(42, 1_000_000); // Buy 0.01 BTC each
        }
        
        // Should have exactly 1000 * 1M = 1B satoshis (10 BTC)
        assert_eq!(accumulator.get_position(42), 1_000_000_000);
        
        // Verify no drift after many operations
        for i in 0..1000 {
            accumulator.update_position(42, -1_000_000); // Sell back
        }
        
        // Should be exactly zero (no floating-point drift)
        assert_eq!(accumulator.get_position(42), 0);
    }

    #[test]
    fn test_overflow_protection() {
        let accumulator = AtomicPositionAccumulator::new(i64::MAX);
        
        // Try to add to near-max value
        accumulator.set_position(42, i64::MAX - 100);
        
        // Adding 200 would overflow
        let result = accumulator.check_position_would_exceed(42, 200);
        assert!(matches!(result, Err(PositionCheckResult::WouldExceed { .. })));
    }

    #[test]
    fn test_scale_conversions() {
        use scale_conversion::*;
        
        // BTC <-> Satoshi
        assert_eq!(btc_to_satoshi(1), 100_000_000);
        assert_eq!(satoshi_to_btc(100_000_000), 1);
        
        // ETH <-> Wei
        assert_eq!(eth_to_wei(1), 1_000_000_000_000_000_000);
        assert_eq!(wei_to_eth(1_000_000_000_000_000_000), 1);
    }

    #[test]
    fn test_concurrent_updates() {
        use std::sync::Arc;
        use std::thread;
        
        let accumulator = Arc::new(AtomicPositionAccumulator::new(10_000_000_000));
        let num_threads = 10;
        let updates_per_thread = 1000;
        
        let mut handles = vec![];
        
        for t in 0..num_threads {
            let acc = Arc::clone(&accumulator);
            handles.push(thread::spawn(move || {
                for _ in 0..updates_per_thread {
                    acc.update_position(42, 1_000_000);
                }
            }));
        }
        
        for h in handles {
            let _ = h.join();
        }
        
        // Should have exactly num_threads * updates_per_thread * 1M
        let expected = (num_threads * updates_per_thread * 1_000_000) as i64;
        assert_eq!(accumulator.get_position(42), expected);
    }
}
