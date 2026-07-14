//! TWAP Liquidation Router
//! 
//! Implements a Time-Weighted Average Price liquidation algorithm for
//! orderly position exit during kill switch events.
//! 
//! CRITICAL SAFETY FEATURE:
//! This router ONLY uses limit orders with strict slippage tolerance.
//! It will NOT dump market orders during flash crashes, preventing
//! catastrophic fills at terrible prices.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Result of TWAP slice calculation
#[derive(Debug, Clone)]
pub struct TWAPSlice {
    /// Quantity to trade in this slice
    pub quantity: u64,
    /// Limit price (maximum acceptable for buys, minimum for sells)
    pub limit_price: u64,
    /// Whether this is a buy or sell
    pub is_buy: bool,
}

/// Configuration for TWAP liquidation
#[derive(Debug, Clone)]
pub struct TWAPConfig {
    /// Number of slices to divide the position into
    pub num_slices: u64,
    /// Interval between slices in milliseconds
    pub interval_ms: u64,
    /// Maximum slippage in basis points from reference price
    pub max_slippage_bps: u16,
    /// Minimum order size (below this, send remaining)
    pub min_order_size: u64,
}

impl Default for TWAPConfig {
    fn default() -> Self {
        Self {
            num_slices: 10,
            interval_ms: 1000,
            max_slippage_bps: 500, // 5% max slippage
            min_order_size: 10_000,
        }
    }
}

/// TWAP Liquidation Router
/// 
/// Breaks large positions into smaller slices executed over time.
/// Uses LIMIT ORDERS ONLY with strict slippage bounds to prevent
//! catastrophic fills during liquidity voids.
pub struct TWAPLiquidationRouter {
    config: TWAPConfig,
    /// Total position to liquidate
    total_position: AtomicU64,
    /// Remaining position to liquidate
    remaining_position: AtomicU64,
    /// Current slice index
    current_slice: AtomicU64,
    /// Reference price for slippage calculation
    reference_price: AtomicU64,
    /// Whether we're liquidating a long (false = short)
    is_long: AtomicBool,
    /// Count of slices executed
    slices_executed: AtomicU64,
    /// Total quantity liquidated
    total_liquidated: AtomicU64,
}

unsafe impl Send for TWAPLiquidationRouter {}
unsafe impl Sync for TWAPLiquidationRouter {}

impl TWAPLiquidationRouter {
    /// Create a new TWAP liquidation router
    pub fn new(config: TWAPConfig) -> Self {
        Self {
            config,
            total_position: AtomicU64::new(0),
            remaining_position: AtomicU64::new(0),
            current_slice: AtomicU64::new(0),
            reference_price: AtomicU64::new(0),
            is_long: AtomicBool::new(true),
            slices_executed: AtomicU64::new(0),
            total_liquidated: AtomicU64::new(0),
        }
    }

    /// Initialize the TWAP liquidation
    /// 
    /// # Arguments
    /// * `position` - Total position to liquidate (absolute value)
    /// * `is_long` - True if liquidating a long position
    /// * `reference_price` - Current market price for slippage calc
    #[inline]
    pub fn initialize(&self, position: u64, is_long: bool, reference_price: u64) {
        self.total_position.store(position, Ordering::Relaxed);
        self.remaining_position.store(position, Ordering::Relaxed);
        self.current_slice.store(0, Ordering::Relaxed);
        self.reference_price.store(reference_price, Ordering::Relaxed);
        self.is_long.store(is_long, Ordering::Relaxed);
    }

    /// Calculate the next TWAP slice
    /// 
    /// Returns None if liquidation is complete.
    /// 
    /// CRITICAL: The returned limit price includes slippage protection.
    /// For longs (selling): limit_price = reference * (1 - max_slippage)
    /// For shorts (buying): limit_price = reference * (1 + max_slippage)
    #[inline]
    pub fn next_slice(&self, current_price: u64) -> Option<TWAPSlice> {
        let remaining = self.remaining_position.load(Ordering::Relaxed);
        
        if remaining == 0 {
            return None;
        }

        let slice_idx = self.current_slice.load(Ordering::Relaxed);
        let total = self.total_position.load(Ordering::Relaxed);
        let num_slices = self.config.num_slices;

        // Calculate slice size
        let slice_size = if slice_idx >= num_slices - 1 {
            // Last slice: send all remaining
            remaining
        } else {
            // Equal division with remainder handling
            total / num_slices
        };

        // Respect minimum order size
        let quantity = if slice_size < self.config.min_order_size && remaining >= self.config.min_order_size {
            self.config.min_order_size
        } else {
            slice_size.min(remaining)
        };

        if quantity == 0 {
            return None;
        }

        // Calculate limit price with slippage protection
        let ref_price = self.reference_price.load(Ordering::Relaxed);
        let is_long = self.is_long.load(Ordering::Relaxed);

        let limit_price = if is_long {
            // Selling: set minimum acceptable price
            // limit = ref * (1 - slippage/10000)
            let slippage = (ref_price as u128 * self.config.max_slippage_bps as u128 / 10000) as u64;
            ref_price.saturating_sub(slippage)
        } else {
            // Buying: set maximum acceptable price
            // limit = ref * (1 + slippage/10000)
            let slippage = (ref_price as u128 * self.config.max_slippage_bps as u128 / 10000) as u64;
            ref_price.saturating_add(slippage)
        };

        Some(TWAPSlice {
            quantity,
            limit_price,
            is_buy: !is_long, // If long, we're selling; if short, we're buying
        })
    }

    /// Mark a slice as executed
    /// 
    /// # Arguments
    /// * `filled_quantity` - Actual quantity filled
    #[inline]
    pub fn mark_slice_executed(&self, filled_quantity: u64) {
        self.current_slice.fetch_add(1, Ordering::Relaxed);
        self.slices_executed.fetch_add(1, Ordering::Relaxed);
        
        self.remaining_position.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| current.checked_sub(filled_quantity),
        ).ok();
        
        self.total_liquidated.fetch_add(filled_quantity, Ordering::Relaxed);
    }

    /// Update reference price (should be called periodically)
    #[inline]
    pub fn update_reference_price(&self, price: u64) {
        self.reference_price.store(price, Ordering::Relaxed);
    }

    /// Check if liquidation is complete
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.remaining_position.load(Ordering::Relaxed) == 0
    }

    /// Get remaining position
    #[inline]
    pub fn remaining_position(&self) -> u64 {
        self.remaining_position.load(Ordering::Relaxed)
    }

    /// Get current slice index
    #[inline]
    pub fn current_slice(&self) -> u64 {
        self.current_slice.load(Ordering::Relaxed)
    }

    /// Get statistics
    pub fn stats(&self) -> TWAPStats {
        TWAPStats {
            total_position: self.total_position.load(Ordering::Relaxed),
            remaining_position: self.remaining_position.load(Ordering::Relaxed),
            slices_executed: self.slices_executed.load(Ordering::Relaxed),
            total_liquidated: self.total_liquidated.load(Ordering::Relaxed),
            reference_price: self.reference_price.load(Ordering::Relaxed),
            is_complete: self.is_complete(),
        }
    }

    /// Reset the router
    #[inline]
    pub fn reset(&self) {
        self.total_position.store(0, Ordering::Relaxed);
        self.remaining_position.store(0, Ordering::Relaxed);
        self.current_slice.store(0, Ordering::Relaxed);
        self.reference_price.store(0, Ordering::Relaxed);
        self.slices_executed.store(0, Ordering::Relaxed);
        self.total_liquidated.store(0, Ordering::Relaxed);
    }
}

/// Statistics from the TWAP router
#[derive(Debug, Clone)]
pub struct TWAPStats {
    pub total_position: u64,
    pub remaining_position: u64,
    pub slices_executed: u64,
    pub total_liquidated: u64,
    pub reference_price: u64,
    pub is_complete: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twap_slice_calculation() {
        let config = TWAPConfig::default();
        let router = TWAPLiquidationRouter::new(config);

        // Initialize with 1 BTC (100M satoshis), long position
        router.initialize(100_000_000, true, 50_000_000_000);

        // Get first slice
        let slice = router.next_slice(50_000_000_000).unwrap();
        
        // Should be ~10M (100M / 10 slices)
        assert!(slice.quantity >= 9_000_000 && slice.quantity <= 11_000_000);
        
        // Should be selling (is_buy = false for long liquidation)
        assert!(!slice.is_buy);
        
        // Limit price should have slippage protection (max 5% below ref)
        let min_limit = 50_000_000_000 * 95 / 100; // 5% below
        assert!(slice.limit_price >= min_limit);
        assert!(slice.limit_price <= 50_000_000_000);
    }

    #[test]
    fn test_short_liquidation() {
        let config = TWAPConfig::default();
        let router = TWAPLiquidationRouter::new(config);

        // Short position (need to buy back)
        router.initialize(50_000_000, false, 40_000_000_000);

        let slice = router.next_slice(40_000_000_000).unwrap();
        
        // Should be buying
        assert!(slice.is_buy);
        
        // Limit price should have upside protection
        let max_limit = 40_000_000_000 * 105 / 100; // 5% above
        assert!(slice.limit_price <= max_limit);
        assert!(slice.limit_price >= 40_000_000_000);
    }

    #[test]
    fn test_liquidation_completion() {
        let config = TWAPConfig::default();
        let router = TWAPLiquidationRouter::new(config);

        router.initialize(100_000_000, true, 50_000_000_000);

        assert!(!router.is_complete());

        // Execute all slices
        while let Some(slice) = router.next_slice(50_000_000_000) {
            router.mark_slice_executed(slice.quantity);
        }

        assert!(router.is_complete());
        assert_eq!(router.remaining_position(), 0);
    }

    #[test]
    fn test_zero_slippage_config() {
        let mut config = TWAPConfig::default();
        config.max_slippage_bps = 0; // No slippage allowed
        let router = TWAPLiquidationRouter::new(config);

        router.initialize(100_000_000, true, 50_000_000_000);

        let slice = router.next_slice(50_000_000_000).unwrap();
        
        // Limit should equal reference exactly
        assert_eq!(slice.limit_price, 50_000_000_000);
    }
}
