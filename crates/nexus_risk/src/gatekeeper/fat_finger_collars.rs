//! Fat-finger price collar validation.
//! 
//! Validates that order prices are within acceptable bounds relative to
//! the current market mid-price to prevent catastrophic fat-finger errors.

use std::sync::atomic::{AtomicU64, Ordering};

/// Represents a validated price with basis point precision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PriceBps(u64);

impl PriceBps {
    #[inline]
    pub const fn from_bps(bps: u16) -> Self {
        Self(bps as u64)
    }
    
    #[inline]
    pub const fn as_bps(&self) -> u16 {
        self.0 as u16
    }
}

/// Fat-finger validation result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FatFingerResult {
    /// Price is within acceptable bounds
    Valid,
    /// Price is too high (above upper collar)
    TooHigh { 
        submitted_price: u64, 
        max_allowed: u64,
        deviation_bps: u16,
    },
    /// Price is too low (below lower collar)
    TooLow { 
        submitted_price: u64, 
        min_allowed: u64,
        deviation_bps: u16,
    },
    /// Mid-price is stale (not updated recently)
    StaleMidPrice,
}

/// Fat-finger validator using atomic operations for zero-lock price checks.
/// 
/// This validator maintains an atomic snapshot of the mid-price that can be
/// updated by the market data thread and read by the risk validation thread
/// without any locking overhead.
pub struct FatFingerValidator {
    /// Current mid-price in quote units (atomic for lock-free reads)
    mid_price: AtomicU64,
    /// Timestamp of last mid-price update in nanoseconds
    mid_price_timestamp_ns: AtomicU64,
    /// Maximum allowed deviation from mid-price in basis points
    collar_bps: u16,
    /// Maximum age of mid-price before considered stale (nanoseconds)
    max_stale_ns: u64,
    /// Count of rejected orders due to fat-finger violations
    rejection_count: AtomicU64,
}

unsafe impl Send for FatFingerValidator {}
unsafe impl Sync for FatFingerValidator {}

impl FatFingerValidator {
    /// Create a new fat-finger validator.
    /// 
    /// # Arguments
    /// * `collar_bps` - Maximum allowed deviation from mid-price in basis points (e.g., 200 = 2%)
    /// * `max_stale_ns` - Maximum age of mid-price before considered stale
    pub fn new(collar_bps: u16, max_stale_ns: u64) -> Self {
        Self {
            mid_price: AtomicU64::new(0),
            mid_price_timestamp_ns: AtomicU64::new(0),
            collar_bps,
            max_stale_ns,
            rejection_count: AtomicU64::new(0),
        }
    }

    /// Update the mid-price atomically.
    /// 
    /// This should be called by the market data thread whenever a new
    /// mid-price is available. Uses relaxed ordering since we only need
    /// eventual consistency for risk checks.
    #[inline]
    pub fn update_mid_price(&self, price: u64, timestamp_ns: u64) {
        self.mid_price.store(price, Ordering::Relaxed);
        self.mid_price_timestamp_ns.store(timestamp_ns, Ordering::Relaxed);
    }

    /// Validate a limit order price against the current mid-price.
    /// 
    /// Returns `FatFingerResult::Valid` if the price is acceptable,
    /// or an error variant describing the violation.
    /// 
    /// # Arguments
    /// * `submitted_price` - The price submitted in the order
    /// * `is_buy` - True if this is a buy order, false for sell
    /// * `current_time_ns` - Current timestamp in nanoseconds
    #[inline]
    pub fn validate_limit_price(
        &self,
        submitted_price: u64,
        is_buy: bool,
        current_time_ns: u64,
    ) -> FatFingerResult {
        let mid_price = self.mid_price.load(Ordering::Relaxed);
        
        // Handle uninitialized mid-price
        if mid_price == 0 {
            return FatFingerResult::StaleMidPrice;
        }
        
        // Check if mid-price is stale
        let price_age_ns = current_time_ns.saturating_sub(
            self.mid_price_timestamp_ns.load(Ordering::Relaxed)
        );
        
        if price_age_ns > self.max_stale_ns {
            return FatFingerResult::StaleMidPrice;
        }
        
        // Calculate allowed deviation in price units
        // Using integer arithmetic: deviation = mid_price * collar_bps / 10000
        let deviation = (mid_price as u128 * self.collar_bps as u128 / 10000) as u64;
        
        if is_buy {
            // For buys, price must not exceed mid + collar
            let max_allowed = mid_price.saturating_add(deviation);
            
            if submitted_price > max_allowed {
                let actual_deviation_bps = if mid_price > 0 {
                    ((submitted_price as u128 * 10000 / mid_price as u128 - 10000) as u64).min(u16::MAX as u64) as u16
                } else {
                    u16::MAX
                };
                
                self.rejection_count.fetch_add(1, Ordering::Relaxed);
                return FatFingerResult::TooHigh {
                    submitted_price,
                    max_allowed,
                    deviation_bps: actual_deviation_bps,
                };
            }
        } else {
            // For sells, price must not be below mid - collar
            let min_allowed = mid_price.saturating_sub(deviation);
            
            if submitted_price < min_allowed && min_allowed > 0 {
                let actual_deviation_bps = if mid_price > 0 {
                    ((10000 - submitted_price as u128 * 10000 / mid_price as u128) as u64).min(u16::MAX as u64) as u16
                } else {
                    u16::MAX
                };
                
                self.rejection_count.fetch_add(1, Ordering::Relaxed);
                return FatFingerResult::TooLow {
                    submitted_price,
                    min_allowed,
                    deviation_bps: actual_deviation_bps,
                };
            }
        }
        
        FatFingerResult::Valid
    }

    /// Validate a market order (no price check needed, but verify mid-price is fresh)
    #[inline]
    pub fn validate_market_order(&self, current_time_ns: u64) -> bool {
        let mid_price = self.mid_price.load(Ordering::Relaxed);
        if mid_price == 0 {
            return false;
        }
        
        let price_age_ns = current_time_ns.saturating_sub(
            self.mid_price_timestamp_ns.load(Ordering::Relaxed)
        );
        
        price_age_ns <= self.max_stale_ns
    }

    /// Get the current mid-price
    #[inline]
    pub fn get_mid_price(&self) -> u64 {
        self.mid_price.load(Ordering::Relaxed)
    }

    /// Get count of rejected orders
    #[inline]
    pub fn rejection_count(&self) -> u64 {
        self.rejection_count.load(Ordering::Relaxed)
    }

    /// Reset the rejection counter (for testing/metrics reset)
    #[inline]
    pub fn reset_rejection_count(&self) -> u64 {
        self.rejection_count.swap(0, Ordering::Relaxed)
    }
}

/// Helper for calculating safe price bounds without overflow
pub mod price_bounds {
    /// Calculate upper price bound with overflow protection
    #[inline]
    pub fn upper_bound(mid_price: u64, collar_bps: u16) -> u64 {
        let deviation = (mid_price as u128 * collar_bps as u128 / 10000) as u64;
        mid_price.saturating_add(deviation)
    }

    /// Calculate lower price bound with underflow protection
    #[inline]
    pub fn lower_bound(mid_price: u64, collar_bps: u16) -> u64 {
        let deviation = (mid_price as u128 * collar_bps as u128 / 10000) as u64;
        mid_price.saturating_sub(deviation)
    }

    /// Check if price is within bounds (returns true if valid)
    #[inline]
    pub fn is_within_bounds(price: u64, mid_price: u64, collar_bps: u16, is_buy: bool) -> bool {
        if is_buy {
            price <= upper_bound(mid_price, collar_bps)
        } else {
            price >= lower_bound(mid_price, collar_bps) || lower_bound(mid_price, collar_bps) == 0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_buy_price() {
        let validator = FatFingerValidator::new(200, 1_000_000_000); // 2% collar
        validator.update_mid_price(100_000_000, 1000); // $100,000
        
        // Valid buy at mid
        assert_eq!(
            validator.validate_limit_price(100_000_000, true, 2000),
            FatFingerResult::Valid
        );
        
        // Valid buy slightly above mid (within 2%)
        assert_eq!(
            validator.validate_limit_price(101_000_000, true, 2000),
            FatFingerResult::Valid
        );
        
        // Valid buy at upper bound (exactly 2% above)
        assert_eq!(
            validator.validate_limit_price(102_000_000, true, 2000),
            FatFingerResult::Valid
        );
    }

    #[test]
    fn test_invalid_buy_price() {
        let validator = FatFingerValidator::new(200, 1_000_000_000); // 2% collar
        validator.update_mid_price(100_000_000, 1000);
        
        // Invalid buy too high (3% above mid)
        match validator.validate_limit_price(103_000_000, true, 2000) {
            FatFingerResult::TooHigh { submitted_price, max_allowed, deviation_bps } => {
                assert_eq!(submitted_price, 103_000_000);
                assert_eq!(max_allowed, 102_000_000);
                assert!(deviation_bps >= 294); // ~3%
            }
            _ => panic!("Expected TooHigh result"),
        }
    }

    #[test]
    fn test_valid_sell_price() {
        let validator = FatFingerValidator::new(200, 1_000_000_000);
        validator.update_mid_price(100_000_000, 1000);
        
        // Valid sell at mid
        assert_eq!(
            validator.validate_limit_price(100_000_000, false, 2000),
            FatFingerResult::Valid
        );
        
        // Valid sell slightly below mid (within 2%)
        assert_eq!(
            validator.validate_limit_price(99_000_000, false, 2000),
            FatFingerResult::Valid
        );
    }

    #[test]
    fn test_invalid_sell_price() {
        let validator = FatFingerValidator::new(200, 1_000_000_000);
        validator.update_mid_price(100_000_000, 1000);
        
        // Invalid sell too low (5% below mid)
        match validator.validate_limit_price(95_000_000, false, 2000) {
            FatFingerResult::TooLow { .. } => {}
            _ => panic!("Expected TooLow result"),
        }
    }

    #[test]
    fn test_stale_mid_price() {
        let validator = FatFingerValidator::new(200, 100_000_000); // 100ms stale threshold
        validator.update_mid_price(100_000_000, 1000);
        
        // Should be stale after 200ms
        assert_eq!(
            validator.validate_limit_price(100_000_000, true, 200_000_000),
            FatFingerResult::StaleMidPrice
        );
    }

    #[test]
    fn test_zero_mid_price() {
        let validator = FatFingerValidator::new(200, 1_000_000_000);
        // Mid price not set
        
        assert_eq!(
            validator.validate_limit_price(100_000_000, true, 2000),
            FatFingerResult::StaleMidPrice
        );
    }

    #[test]
    fn test_price_bounds_helpers() {
        use price_bounds::*;
        
        let mid = 100_000_000;
        let collar = 200; // 2%
        
        assert_eq!(upper_bound(mid, collar), 102_000_000);
        assert_eq!(lower_bound(mid, collar), 98_000_000);
        
        assert!(is_within_bounds(101_000_000, mid, collar, true));
        assert!(!is_within_bounds(103_000_000, mid, collar, true));
        
        assert!(is_within_bounds(99_000_000, mid, collar, false));
        assert!(!is_within_bounds(97_000_000, mid, collar, false));
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
        
        // Test with small prices
        let mid = 1;
        let collar = 1; // 0.01%
        
        let lower = lower_bound(mid, collar);
        assert!(lower <= mid);
    }
}
