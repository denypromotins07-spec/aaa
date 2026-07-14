//! Price Collar Validator - Fat-Finger Protection
//! 
//! Validates that order prices are within acceptable bounds relative to
//! the current micro-price. Uses atomic operations for zero-lock price checks.
//! 
//! This implements the "Price Collars" requirement from Chapter 1:
//! - Reject any limit order > X% away from Stage 2 Micro-Price
//! - Uses basis points for precision without floating-point

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::collections::HashMap;
use parking_lot::RwLock;

/// Maximum number of symbols to track (tunable)
const MAX_SYMBOLS: usize = 1024;

/// Result of price validation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriceValidationResult {
    /// Price is within acceptable bounds
    Valid,
    /// Price too high (above upper collar)
    TooHigh {
        submitted: u64,
        max_allowed: u64,
    },
    /// Price too low (below lower collar)
    TooLow {
        submitted: u64,
        min_allowed: u64,
    },
    /// Micro-price is stale or uninitialized
    StalePrice,
}

/// Per-symbol price state with cache-line padding
#[repr(align(64))]
struct SymbolPriceState {
    /// Current micro-price in quote units
    micro_price: AtomicU64,
    /// Last update timestamp in nanoseconds
    last_update_ns: AtomicU64,
    /// Padding to prevent false sharing
    _padding: [u8; 48],
}

impl SymbolPriceState {
    fn new() -> Self {
        Self {
            micro_price: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
            _padding: [0; 48],
        }
    }
}

impl Default for SymbolPriceState {
    fn default() -> Self {
        Self::new()
    }
}

/// Price Collar Validator
/// 
/// Implements fat-finger protection by rejecting orders with prices
/// outside an acceptable collar around the current micro-price.
/// 
/// ZERO-ALLOCATION DESIGN:
/// - Uses pre-allocated array for symbol tracking
/// - All math uses integer arithmetic (basis points)
/// - No heap allocations in hot path
pub struct PriceCollarValidator {
    /// Collar width in basis points (e.g., 200 = 2%)
    collar_bps: u16,
    /// Stale price threshold in nanoseconds
    stale_threshold_ns: u64,
    /// Per-symbol price states (pre-allocated)
    symbol_states: Box<[SymbolPriceState; MAX_SYMBOLS]>,
    /// Count of price violations
    violation_count: AtomicUsize,
}

unsafe impl Send for PriceCollarValidator {}
unsafe impl Sync for PriceCollarValidator {}

impl PriceCollarValidator {
    /// Create a new price collar validator
    /// 
    /// # Arguments
    /// * `collar_bps` - Maximum allowed deviation in basis points (e.g., 200 = 2%)
    /// * `stale_threshold_ns` - Time after which micro-price is considered stale
    pub fn new(collar_bps: u16, stale_threshold_ns: u64) -> Self {
        // Initialize all symbol states to zero
        let symbol_states: Box<[SymbolPriceState; MAX_SYMBOLS]> = 
            Box::new(std::array::from_fn(|_| SymbolPriceState::new()));
        
        Self {
            collar_bps,
            stale_threshold_ns,
            symbol_states,
            violation_count: AtomicUsize::new(0),
        }
    }

    /// Validate a price against the micro-price
    /// 
    /// # Arguments
    /// * `price` - The submitted order price
    /// * `is_buy` - True if this is a buy order
    /// * `micro_price` - Current market micro-price
    /// * `timestamp_ns` - Current timestamp
    /// 
    /// # Returns
    /// * `Ok(())` - Price is valid
    /// * `Err(PriceValidationResult)` - Price violated collar
    #[inline]
    pub fn validate(
        &self,
        price: u64,
        is_buy: bool,
        micro_price: u64,
        timestamp_ns: u64,
    ) -> Result<(), PriceValidationResult> {
        // Handle zero micro-price (uninitialized)
        if micro_price == 0 {
            return Err(PriceValidationResult::StalePrice);
        }

        // Calculate allowed deviation using integer math
        // deviation = micro_price * collar_bps / 10000
        let deviation = (micro_price as u128 * self.collar_bps as u128 / 10000) as u64;

        if is_buy {
            // For buys: price must not exceed micro_price + collar
            let max_allowed = micro_price.saturating_add(deviation);
            
            if price > max_allowed {
                self.violation_count.fetch_add(1, Ordering::Relaxed);
                return Err(PriceValidationResult::TooHigh {
                    submitted: price,
                    max_allowed,
                });
            }
        } else {
            // For sells: price must not be below micro_price - collar
            let min_allowed = micro_price.saturating_sub(deviation);
            
            // Special case: if min_allowed would be 0, allow any positive price
            if min_allowed > 0 && price < min_allowed {
                self.violation_count.fetch_add(1, Ordering::Relaxed);
                return Err(PriceValidationResult::TooLow {
                    submitted: price,
                    min_allowed,
                });
            }
        }

        Ok(())
    }

    /// Update the micro-price for a symbol
    /// 
    /// # Arguments
    /// * `symbol_hash` - Hash of the symbol identifier
    /// * `price` - New micro-price
    /// * `timestamp_ns` - Timestamp of the price update
    #[inline]
    pub fn update_price(&self, symbol_hash: u64, price: u64, timestamp_ns: u64) {
        let idx = (symbol_hash as usize) % MAX_SYMBOLS;
        let state = &self.symbol_states[idx];
        state.micro_price.store(price, Ordering::Relaxed);
        state.last_update_ns.store(timestamp_ns, Ordering::Relaxed);
    }

    /// Get the current micro-price for a symbol
    #[inline]
    pub fn get_micro_price(&self, symbol_hash: u64) -> u64 {
        let idx = (symbol_hash as usize) % MAX_SYMBOLS;
        self.symbol_states[idx].micro_price.load(Ordering::Relaxed)
    }

    /// Check if the price for a symbol is stale
    #[inline]
    pub fn is_price_stale(&self, symbol_hash: u64, current_time_ns: u64) -> bool {
        let idx = (symbol_hash as usize) % MAX_SYMBOLS;
        let last_update = self.symbol_states[idx].last_update_ns.load(Ordering::Relaxed);
        
        if last_update == 0 {
            return true; // Never updated
        }
        
        current_time_ns.saturating_sub(last_update) > self.stale_threshold_ns
    }

    /// Get count of price violations
    #[inline]
    pub fn violation_count(&self) -> usize {
        self.violation_count.load(Ordering::Relaxed)
    }

    /// Reset violation counter (for testing/metrics)
    #[inline]
    pub fn reset_violations(&self) -> usize {
        self.violation_count.swap(0, Ordering::Relaxed)
    }
}

/// Helper functions for calculating price bounds
pub mod price_bounds {
    /// Calculate upper price bound without overflow
    #[inline]
    pub fn upper_bound(mid_price: u64, collar_bps: u16) -> u64 {
        let deviation = (mid_price as u128 * collar_bps as u128 / 10000) as u64;
        mid_price.saturating_add(deviation)
    }

    /// Calculate lower price bound without underflow
    #[inline]
    pub fn lower_bound(mid_price: u64, collar_bps: u16) -> u64 {
        let deviation = (mid_price as u128 * collar_bps as u128 / 10000) as u64;
        mid_price.saturating_sub(deviation)
    }

    /// Check if price is within bounds
    #[inline]
    pub fn is_within_bounds(price: u64, mid_price: u64, collar_bps: u16, is_buy: bool) -> bool {
        if is_buy {
            price <= upper_bound(mid_price, collar_bps)
        } else {
            let lower = lower_bound(mid_price, collar_bps);
            lower == 0 || price >= lower
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_buy_price() {
        let validator = PriceCollarValidator::new(200, 1_000_000_000); // 2% collar
        
        // Valid buy at micro-price
        assert!(validator.validate(50_000_000_000, true, 50_000_000_000, 1000).is_ok());
        
        // Valid buy slightly above (within 2%)
        assert!(validator.validate(51_000_000_000, true, 50_000_000_000, 1000).is_ok());
        
        // Valid buy at upper bound (exactly 2%)
        assert!(validator.validate(51_000_000_000, true, 50_000_000_000, 1000).is_ok());
    }

    #[test]
    fn test_invalid_buy_price() {
        let validator = PriceCollarValidator::new(200, 1_000_000_000); // 2% collar
        
        // Invalid buy too high (3% above)
        let result = validator.validate(51_500_000_000, true, 50_000_000_000, 1000);
        assert!(matches!(result, Err(PriceValidationResult::TooHigh { .. })));
    }

    #[test]
    fn test_valid_sell_price() {
        let validator = PriceCollarValidator::new(200, 1_000_000_000);
        
        // Valid sell at micro-price
        assert!(validator.validate(50_000_000_000, false, 50_000_000_000, 1000).is_ok());
        
        // Valid sell slightly below (within 2%)
        assert!(validator.validate(49_000_000_000, false, 50_000_000_000, 1000).is_ok());
    }

    #[test]
    fn test_invalid_sell_price() {
        let validator = PriceCollarValidator::new(200, 1_000_000_000);
        
        // Invalid sell too low (5% below)
        let result = validator.validate(47_500_000_000, false, 50_000_000_000, 1000);
        assert!(matches!(result, Err(PriceValidationResult::TooLow { .. })));
    }

    #[test]
    fn test_zero_micro_price() {
        let validator = PriceCollarValidator::new(200, 1_000_000_000);
        
        // Zero micro-price should return StalePrice
        let result = validator.validate(50_000_000_000, true, 0, 1000);
        assert_eq!(result, Err(PriceValidationResult::StalePrice));
    }

    #[test]
    fn test_price_bounds_helpers() {
        use price_bounds::*;
        
        let mid = 50_000_000_000;
        let collar = 200; // 2%
        
        assert_eq!(upper_bound(mid, collar), 51_000_000_000);
        assert_eq!(lower_bound(mid, collar), 49_000_000_000);
        
        assert!(is_within_bounds(50_500_000_000, mid, collar, true));
        assert!(!is_within_bounds(52_000_000_000, mid, collar, true));
        
        assert!(is_within_bounds(49_500_000_000, mid, collar, false));
        assert!(!is_within_bounds(48_000_000_000, mid, collar, false));
    }

    #[test]
    fn test_overflow_protection() {
        use price_bounds::*;
        
        // Test with very large prices
        let mid = u64::MAX / 2;
        let collar = 500; // 5%
        
        // Should not overflow
        let upper = upper_bound(mid, collar);
        assert!(upper > mid);
    }
}
