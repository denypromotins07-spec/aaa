//! Chapter 3: Micro-Price and Order Book Imbalance Calculator
//!
//! This module calculates the micro-price (volume-weighted mid price)
//! and order book imbalance using SIMD acceleration.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use nexus_core::memory::cache_padder::CachePadded64;
use wide::f64x8;

/// Order book imbalance result
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OrderBookImbalance {
    /// Bid volume at best
    pub bid_volume: f64,
    /// Ask volume at best
    pub ask_volume: f64,
    /// Imbalance ratio: (bid_vol - ask_vol) / (bid_vol + ask_vol)
    pub imbalance_ratio: f64,
    /// Weighted imbalance (with depth consideration)
    pub weighted_imbalance: f64,
}

impl OrderBookImbalance {
    #[inline]
    pub const fn new() -> Self {
        Self {
            bid_volume: 0.0,
            ask_volume: 0.0,
            imbalance_ratio: 0.0,
            weighted_imbalance: 0.0,
        }
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        self.bid_volume > 0.0 || self.ask_volume > 0.0
    }
}

impl Default for OrderBookImbalance {
    fn default() -> Self {
        Self::new()
    }
}

/// Micro-price calculator with SIMD support
#[repr(C)]
pub struct MicroPriceCalculator {
    /// Best bid price (nanodollars)
    best_bid: CachePadded64<AtomicU64>,
    /// Best ask price (nanodollars)
    best_ask: CachePadded64<AtomicU64>,
    /// Best bid volume
    bid_volume: CachePadded64<AtomicU64>,
    /// Best ask volume
    ask_volume: CachePadded64<AtomicU64>,
    /// Second-level bid prices (for depth calculation)
    bid_depth_prices: CachePadded64<Box<[u64]>>,
    /// Second-level ask prices
    ask_depth_prices: CachePadded64<Box<[u64]>>,
    /// Second-level bid volumes
    bid_depth_volumes: CachePadded64<Box<[u64]>>,
    /// Second-level ask volumes
    ask_depth_volumes: CachePadded64<Box<[u64]>>,
    /// Depth levels tracked
    depth_levels: usize,
    /// Last micro-price (cached)
    last_micro_price: CachePadded64<AtomicU64>,
    /// Update count
    update_count: CachePadded64<AtomicUsize>,
}

// SAFETY: MicroPriceCalculator is single-threaded hot-path
unsafe impl Send for MicroPriceCalculator {}
unsafe impl Sync for MicroPriceCalculator {}

impl MicroPriceCalculator {
    /// Create a new micro-price calculator
    #[inline]
    pub fn new(depth_levels: usize) -> Self {
        let aligned_depth = ((depth_levels + 7) / 8) * 8; // SIMD alignment
        
        Self {
            best_bid: CachePadded64::new(AtomicU64::new(0)),
            best_ask: CachePadded64::new(AtomicU64::new(0)),
            bid_volume: CachePadded64::new(AtomicU64::new(0)),
            ask_volume: CachePadded64::new(AtomicU64::new(0)),
            bid_depth_prices: CachePadded64::new(vec![0u64; aligned_depth].into_boxed_slice()),
            ask_depth_prices: CachePadded64::new(vec![0u64; aligned_depth].into_boxed_slice()),
            bid_depth_volumes: CachePadded64::new(vec![0u64; aligned_depth].into_boxed_slice()),
            ask_depth_volumes: CachePadded64::new(vec![0u64; aligned_depth].into_boxed_slice()),
            depth_levels: aligned_depth,
            last_micro_price: CachePadded64::new(AtomicU64::new(0)),
            update_count: CachePadded64::new(AtomicUsize::new(0)),
        }
    }

    /// Update best bid/ask
    #[inline]
    pub fn update_top_of_book(&self, bid_price: u64, bid_vol: u64, ask_price: u64, ask_vol: u64) {
        self.best_bid.0.store(bid_price, Ordering::Release);
        self.best_ask.0.store(ask_price, Ordering::Release);
        self.bid_volume.0.store(bid_vol, Ordering::Release);
        self.ask_volume.0.store(ask_vol, Ordering::Release);
        
        self.update_count.0.fetch_add(1, Ordering::Relaxed);
        
        // Recalculate micro-price
        let _ = self.micro_price();
    }

    /// Update depth level
    #[inline]
    pub fn update_depth_level(&self, level: usize, bid_price: u64, bid_vol: u64, 
                               ask_price: u64, ask_vol: u64) {
        if level >= self.depth_levels {
            return;
        }
        
        self.bid_depth_prices.0[level] = bid_price;
        self.bid_depth_volumes.0[level] = bid_vol;
        self.ask_depth_prices.0[level] = ask_price;
        self.ask_depth_volumes.0[level] = ask_vol;
    }

    /// Calculate micro-price: weighted average of bid/ask based on volume
    /// Formula: (bid_price * ask_vol + ask_price * bid_vol) / (bid_vol + ask_vol)
    #[inline]
    pub fn micro_price(&self) -> u64 {
        let bid_price = self.best_bid.0.load(Ordering::Acquire);
        let ask_price = self.best_ask.0.load(Ordering::Acquire);
        let bid_vol = self.bid_volume.0.load(Ordering::Acquire) as f64;
        let ask_vol = self.ask_volume.0.load(Ordering::Acquire) as f64;
        
        if bid_vol == 0.0 && ask_vol == 0.0 {
            return (bid_price + ask_price) / 2;
        }
        
        let total_vol = bid_vol + ask_vol;
        if total_vol == 0.0 {
            return (bid_price + ask_price) / 2;
        }
        
        // Micro-price formula: closer to side with less volume (pressure indicator)
        let micro = (bid_price as f64 * ask_vol + ask_price as f64 * bid_vol) / total_vol;
        let result = micro.round() as u64;
        
        self.last_micro_price.0.store(result, Ordering::Release);
        result
    }

    /// Get cached micro-price
    #[inline]
    pub fn get_cached_micro_price(&self) -> u64 {
        self.last_micro_price.0.load(Ordering::Acquire)
    }

    /// Calculate order book imbalance
    #[inline]
    pub fn calculate_imbalance(&self) -> OrderBookImbalance {
        let bid_vol = self.bid_volume.0.load(Ordering::Acquire) as f64;
        let ask_vol = self.ask_volume.0.load(Ordering::Acquire) as f64;
        
        let total = bid_vol + ask_vol;
        
        let imbalance_ratio = if total == 0.0 {
            0.0
        } else {
            (bid_vol - ask_vol) / total
        };
        
        // Calculate weighted imbalance using depth levels
        let weighted_imbalance = self.calculate_weighted_imbalance();
        
        OrderBookImbalance {
            bid_volume: bid_vol,
            ask_volume: ask_vol,
            imbalance_ratio,
            weighted_imbalance,
        }
    }

    /// Calculate weighted imbalance using SIMD
    #[inline]
    fn calculate_weighted_imbalance(&self) -> f64 {
        let mut bid_weighted = 0.0f64;
        let mut ask_weighted = 0.0f64;
        
        // Process in SIMD lanes
        let mut i = 0;
        while i + 8 <= self.depth_levels {
            // Load volumes into SIMD registers
            let bid_vols = f64x8::from_slice_unaligned(&self.bid_depth_volumes.0[i..]);
            let ask_vols = f64x8::from_slice_unaligned(&self.ask_depth_volumes.0[i..]);
            
            // Apply decay weights (closer levels have higher weight)
            let weights = f64x8::from([
                1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3,
            ]);
            
            let weighted_bids = bid_vols * weights;
            let weighted_asks = ask_vols * weights;
            
            // Horizontal sum
            bid_weighted += weighted_bids.reduce_sum();
            ask_weighted += weighted_asks.reduce_sum();
            
            i += 8;
        }
        
        // Handle remainder
        for j in i..self.depth_levels {
            let weight = 1.0 - (j as f64 * 0.1);
            if weight > 0.0 {
                bid_weighted += self.bid_depth_volumes.0[j] as f64 * weight;
                ask_weighted += self.ask_depth_volumes.0[j] as f64 * weight;
            }
        }
        
        // Add top of book (highest weight)
        let top_bid = self.bid_volume.0.load(Ordering::Acquire) as f64;
        let top_ask = self.ask_volume.0.load(Ordering::Acquire) as f64;
        
        bid_weighted += top_bid;
        ask_weighted += top_ask;
        
        let total = bid_weighted + ask_weighted;
        if total == 0.0 {
            0.0
        } else {
            (bid_weighted - ask_weighted) / total
        }
    }

    /// Get bid-ask spread in nanodollars
    #[inline]
    pub fn spread(&self) -> u64 {
        let ask = self.best_ask.0.load(Ordering::Acquire);
        let bid = self.best_bid.0.load(Ordering::Acquire);
        
        if ask > bid {
            ask - bid
        } else {
            0
        }
    }

    /// Get mid price
    #[inline]
    pub fn mid_price(&self) -> u64 {
        let bid = self.best_bid.0.load(Ordering::Acquire);
        let ask = self.best_ask.0.load(Ordering::Acquire);
        (bid + ask) / 2
    }

    /// Clear all state
    #[inline]
    pub fn clear(&self) {
        self.best_bid.0.store(0, Ordering::Release);
        self.best_ask.0.store(0, Ordering::Release);
        self.bid_volume.0.store(0, Ordering::Release);
        self.ask_volume.0.store(0, Ordering::Release);
        
        for i in 0..self.depth_levels {
            self.bid_depth_prices.0[i] = 0;
            self.bid_depth_volumes.0[i] = 0;
            self.ask_depth_prices.0[i] = 0;
            self.ask_depth_volumes.0[i] = 0;
        }
        
        self.last_micro_price.0.store(0, Ordering::Release);
        self.update_count.0.store(0, Ordering::Release);
    }

    /// Get update count
    #[inline]
    pub fn update_count(&self) -> usize {
        self.update_count.0.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_micro_price_basic() {
        let calc = MicroPriceCalculator::new(5);
        
        // Symmetric book: micro-price should be mid
        calc.update_top_of_book(99_000_000_000, 1_000_000_000, 
                                100_000_000_000, 1_000_000_000);
        
        let micro = calc.micro_price();
        assert_eq!(micro, 99_500_000_000); // Mid price
    }

    #[test]
    fn test_micro_price_imbalanced() {
        let calc = MicroPriceCalculator::new(5);
        
        // More bid pressure: micro-price closer to ask
        calc.update_top_of_book(99_000_000_000, 3_000_000_000,
                                100_000_000_000, 1_000_000_000);
        
        let micro = calc.micro_price();
        // Formula: (99B * 1B + 100B * 3B) / 4B = (99 + 300) / 4 = 99.75B
        assert_eq!(micro, 99_750_000_000);
    }

    #[test]
    fn test_order_book_imbalance() {
        let calc = MicroPriceCalculator::new(5);
        
        calc.update_top_of_book(99_000_000_000, 2_000_000_000,
                                100_000_000_000, 1_000_000_000);
        
        let obi = calc.calculate_imbalance();
        
        assert_eq!(obi.bid_volume, 2_000_000_000.0);
        assert_eq!(obi.ask_volume, 1_000_000_000.0);
        // (2B - 1B) / (2B + 1B) = 1/3 ≈ 0.333
        assert!((obi.imbalance_ratio - 0.333).abs() < 0.001);
    }

    #[test]
    fn test_spread_and_mid() {
        let calc = MicroPriceCalculator::new(5);
        
        calc.update_top_of_book(99_000_000_000, 1_000_000_000,
                                100_000_000_000, 1_000_000_000);
        
        assert_eq!(calc.spread(), 1_000_000_000);
        assert_eq!(calc.mid_price(), 99_500_000_000);
    }

    #[test]
    fn test_cached_micro_price() {
        let calc = MicroPriceCalculator::new(5);
        
        calc.update_top_of_book(99_000_000_000, 1_000_000_000,
                                100_000_000_000, 1_000_000_000);
        
        // First call calculates
        let micro1 = calc.micro_price();
        assert_eq!(calc.update_count(), 1);
        
        // Cached value
        let micro2 = calc.get_cached_micro_price();
        assert_eq!(micro1, micro2);
    }
}
