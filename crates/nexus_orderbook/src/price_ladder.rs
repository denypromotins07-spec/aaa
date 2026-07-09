//! Chapter 2: Cache-Aligned Price Ladder for O(1) Lookups
//!
//! This module implements a custom pre-allocated, cache-aligned array-based
//! price ladder that replaces standard BTreeMap/HashMap for price levels.
//! It guarantees O(1) lookups and zero heap allocations during updates.

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use nexus_core::memory::arena::BumpAllocator;
use nexus_core::memory::cache_padder::CachePadded64;

/// Maximum number of price levels per side (configurable based on exchange)
pub const MAX_PRICE_LEVELS: usize = 256;

/// Minimum price increment in nanodollars (1e-9)
pub const MIN_PRICE_INCREMENT: i64 = 1;

/// Side of the order book
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Bid = 0,
    Ask = 1,
}

impl Side {
    #[inline]
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Side::Bid),
            1 => Some(Side::Ask),
            _ => None,
        }
    }

    #[inline]
    pub fn is_bid(self) -> bool {
        self == Side::Bid
    }

    #[inline]
    pub fn is_ask(self) -> bool {
        self == Side::Ask
    }

    #[inline]
    pub fn opposite(self) -> Self {
        match self {
            Side::Bid => Side::Ask,
            Side::Ask => Side::Bid,
        }
    }
}

/// Price level entry with volume and order count
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PriceLevel {
    /// Price in nanodollars (fixed-point for precision)
    pub price: CachePadded64<AtomicU64>,
    /// Volume at this price level (in base units * 1e9)
    pub volume: CachePadded64<AtomicU64>,
    /// Number of orders at this level
    pub order_count: CachePadded64<AtomicU16>,
    /// Index into L3 slab for first order
    pub first_order_index: CachePadded64<AtomicUsize>,
    /// Whether this level is active
    pub active: CachePadded64<AtomicBool>,
    /// Padding to ensure 64-byte alignment
    pub _padding: [u8; 45],
}

use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU16;

// SAFETY: PriceLevel is used in lock-free contexts with atomic operations
unsafe impl Send for PriceLevel {}
unsafe impl Sync for PriceLevel {}

impl PriceLevel {
    #[inline]
    pub const fn new() -> Self {
        Self {
            price: CachePadded64::new(AtomicU64::new(0)),
            volume: CachePadded64::new(AtomicU64::new(0)),
            order_count: CachePadded64::new(AtomicU16::new(0)),
            first_order_index: CachePadded64::new(AtomicUsize::new(usize::MAX)),
            active: CachePadded64::new(AtomicBool::new(false)),
            _padding: [0; 45],
        }
    }

    #[inline]
    pub fn set(&self, price: u64, volume: u64, order_count: u16) {
        self.price.0.store(price, Ordering::Release);
        self.volume.0.store(volume, Ordering::Release);
        self.order_count.0.store(order_count, Ordering::Release);
        self.active.0.store(true, Ordering::Release);
    }

    #[inline]
    pub fn get_price(&self) -> u64 {
        self.price.0.load(Ordering::Acquire)
    }

    #[inline]
    pub fn get_volume(&self) -> u64 {
        self.volume.0.load(Ordering::Acquire)
    }

    #[inline]
    pub fn get_order_count(&self) -> u16 {
        self.order_count.0.load(Ordering::Acquire)
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        self.active.0.load(Ordering::Acquire)
    }

    #[inline]
    pub fn deactivate(&self) {
        self.active.0.store(false, Ordering::Release);
        self.order_count.0.store(0, Ordering::Release);
        self.volume.0.store(0, Ordering::Release);
    }

    #[inline]
    pub fn add_volume(&self, delta: i64) -> u64 {
        let current = self.volume.0.load(Ordering::Acquire) as i64;
        let new_volume = (current + delta).max(0) as u64;
        self.volume.0.store(new_volume, Ordering::Release);
        
        // Update active status
        if new_volume == 0 {
            self.deactivate();
        } else {
            self.active.0.store(true, Ordering::Release);
        }
        
        new_volume
    }

    #[inline]
    pub fn increment_order_count(&self) -> u16 {
        self.order_count.0.fetch_add(1, Ordering::AcqRel) + 1
    }

    #[inline]
    pub fn decrement_order_count(&self) -> u16 {
        let new_count = self.order_count.0.fetch_sub(1, Ordering::AcqRel).saturating_sub(1);
        if new_count == 0 {
            self.deactivate();
        }
        new_count
    }
}

impl Default for PriceLevel {
    fn default() -> Self {
        Self::new()
    }
}

/// Price ladder for one side of the book (bids or asks)
/// Uses direct indexing instead of HashMap for O(1) access
pub struct PriceLadder {
    /// Side of the book
    side: Side,
    /// Pre-allocated price levels (direct indexed by price bucket)
    levels: CachePadded64<[PriceLevel; MAX_PRICE_LEVELS]>,
    /// Price-to-index mapping (simple hash for direct lookup)
    /// In production, this would be a perfect hash based on tick size
    price_map: CachePadded64<[AtomicU16; MAX_PRICE_LEVELS]>,
    /// Best price index (cached for fast access)
    best_index: CachePadded64<AtomicUsize>,
    /// Worst price index (cached for fast access)
    worst_index: CachePadded64<AtomicUsize>,
    /// Total active levels
    active_count: CachePadded64<AtomicUsize>,
    /// Tick size in nanodollars
    tick_size: u64,
    /// Base price for bucket calculation
    base_price: u64,
}

// SAFETY: PriceLadder uses atomic operations for thread safety
unsafe impl Send for PriceLadder {}
unsafe impl Sync for PriceLadder {}

impl PriceLadder {
    /// Create a new price ladder
    #[inline]
    pub fn new(side: Side, tick_size: u64, base_price: u64) -> Self {
        Self {
            side,
            levels: CachePadded64::new([PriceLevel::new(); MAX_PRICE_LEVELS]),
            price_map: CachePadded64::new(std::array::from_fn(|_| AtomicU16::new(u16::MAX))),
            best_index: CachePadded64::new(AtomicUsize::new(usize::MAX)),
            worst_index: CachePadded64::new(AtomicUsize::new(usize::MAX)),
            active_count: CachePadded64::new(AtomicUsize::new(0)),
            tick_size,
            base_price,
        }
    }

    /// Calculate price bucket index from raw price
    #[inline]
    fn price_to_bucket(&self, price: u64) -> Option<usize> {
        if price < self.base_price || self.tick_size == 0 {
            return None;
        }
        
        let offset = price - self.base_price;
        let bucket = (offset / self.tick_size) as usize;
        
        if bucket >= MAX_PRICE_LEVELS {
            None
        } else {
            Some(bucket)
        }
    }

    /// Get or create price level for given price
    #[inline]
    pub fn get_or_create_level(&self, price: u64) -> Option<&PriceLevel> {
        let bucket = self.price_to_bucket(price)?;
        let level = &self.levels.0[bucket];
        
        // If not active, initialize it
        if !level.is_active() {
            level.set(price, 0, 0);
            
            // Update price map
            self.price_map.0[bucket].store(bucket as u16, Ordering::Release);
            
            // Update active count
            self.active_count.0.fetch_add(1, Ordering::AcqRel);
            
            // Update best/worst
            self.update_best_worst(bucket);
        }
        
        Some(level)
    }

    /// Update best and worst price indices
    #[inline]
    fn update_best_worst(&self, new_index: usize) {
        let mut best = self.best_index.0.load(Ordering::Acquire);
        let mut worst = self.worst_index.0.load(Ordering::Acquire);
        
        match self.side {
            Side::Bid => {
                // For bids, higher price is better
                if best == usize::MAX || new_index > best {
                    self.best_index.0.store(new_index, Ordering::Release);
                }
                if worst == usize::MAX || new_index < worst {
                    self.worst_index.0.store(new_index, Ordering::Release);
                }
            }
            Side::Ask => {
                // For asks, lower price is better
                if best == usize::MAX || new_index < best {
                    self.best_index.0.store(new_index, Ordering::Release);
                }
                if worst == usize::MAX || new_index > worst {
                    self.worst_index.0.store(new_index, Ordering::Release);
                }
            }
        }
    }

    /// Get best bid/ask price
    #[inline]
    pub fn get_best_price(&self) -> Option<u64> {
        let index = self.best_index.0.load(Ordering::Acquire);
        if index == usize::MAX {
            return None;
        }
        
        let level = &self.levels.0[index];
        if level.is_active() {
            Some(level.get_price())
        } else {
            None
        }
    }

    /// Get best bid/ask volume
    #[inline]
    pub fn get_best_volume(&self) -> Option<u64> {
        let index = self.best_index.0.load(Ordering::Acquire);
        if index == usize::MAX {
            return None;
        }
        
        let level = &self.levels.0[index];
        if level.is_active() {
            Some(level.get_volume())
        } else {
            None
        }
    }

    /// Get volume at specific price
    #[inline]
    pub fn get_volume_at_price(&self, price: u64) -> Option<u64> {
        let bucket = self.price_to_bucket(price)?;
        let level = &self.levels.0[bucket];
        
        if level.is_active() {
            Some(level.get_volume())
        } else {
            None
        }
    }

    /// Add volume at price level
    #[inline]
    pub fn add_volume(&self, price: u64, volume: u64) -> Result<(), &'static str> {
        let level = self.get_or_create_level(price)
            .ok_or("Price out of range")?;
        
        level.add_volume(volume as i64);
        level.increment_order_count();
        
        Ok(())
    }

    /// Remove volume at price level
    #[inline]
    pub fn remove_volume(&self, price: u64, volume: u64) -> Result<(), &'static str> {
        let bucket = self.price_to_bucket(price)
            .ok_or("Price out of range")?;
        
        let level = &self.levels.0[bucket];
        if !level.is_active() {
            return Err("Level not active");
        }
        
        level.add_volume(-(volume as i64));
        level.decrement_order_count();
        
        Ok(())
    }

    /// Clear all levels
    #[inline]
    pub fn clear(&self) {
        for i in 0..MAX_PRICE_LEVELS {
            self.levels.0[i].deactivate();
            self.price_map.0[i].store(u16::MAX, Ordering::Release);
        }
        
        self.best_index.0.store(usize::MAX, Ordering::Release);
        self.worst_index.0.store(usize::MAX, Ordering::Release);
        self.active_count.0.store(0, Ordering::Release);
    }

    /// Get active level count
    #[inline]
    pub fn active_count(&self) -> usize {
        self.active_count.0.load(Ordering::Relaxed)
    }

    /// Get side
    #[inline]
    pub fn side(&self) -> Side {
        self.side
    }

    /// Iterate over active levels (for snapshot purposes)
    #[inline]
    pub fn for_each_active<F>(&self, mut f: F)
    where
        F: FnMut(u64, u64, u16), // price, volume, order_count
    {
        for i in 0..MAX_PRICE_LEVELS {
            let level = &self.levels.0[i];
            if level.is_active() {
                f(level.get_price(), level.get_volume(), level.get_order_count());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_side_conversion() {
        assert_eq!(Side::from_u8(0), Some(Side::Bid));
        assert_eq!(Side::from_u8(1), Some(Side::Ask));
        assert_eq!(Side::from_u8(2), None);
        
        assert!(Side::Bid.is_bid());
        assert!(!Side::Bid.is_ask());
        assert_eq!(Side::Bid.opposite(), Side::Ask);
    }

    #[test]
    fn test_price_level_operations() {
        let level = PriceLevel::new();
        assert!(!level.is_active());
        
        level.set(100_000_000_000, 1_000_000_000, 5);
        assert!(level.is_active());
        assert_eq!(level.get_price(), 100_000_000_000);
        assert_eq!(level.get_volume(), 1_000_000_000);
        assert_eq!(level.get_order_count(), 5);
        
        let new_vol = level.add_volume(500_000_000);
        assert_eq!(new_vol, 1_500_000_000);
        
        level.deactivate();
        assert!(!level.is_active());
    }

    #[test]
    fn test_price_ladder_basics() {
        let ladder = PriceLadder::new(Side::Bid, 100_000_000, 99_000_000_000);
        assert_eq!(ladder.side(), Side::Bid);
        assert_eq!(ladder.active_count(), 0);
        assert!(ladder.get_best_price().is_none());
    }

    #[test]
    fn test_price_ladder_add_remove() {
        let ladder = PriceLadder::new(Side::Bid, 100_000_000, 99_000_000_000);
        
        // Add volume at price
        assert!(ladder.add_volume(99_100_000_000, 1_000_000_000).is_ok());
        assert_eq!(ladder.active_count(), 1);
        
        // Verify volume
        assert_eq!(
            ladder.get_volume_at_price(99_100_000_000),
            Some(1_000_000_000)
        );
        
        // Remove volume
        assert!(ladder.remove_volume(99_100_000_000, 500_000_000).is_ok());
        assert_eq!(
            ladder.get_volume_at_price(99_100_000_000),
            Some(500_000_000)
        );
    }

    #[test]
    fn test_price_ladder_clear() {
        let ladder = PriceLadder::new(Side::Ask, 50_000_000, 100_000_000_000);
        
        ladder.add_volume(100_050_000_000, 2_000_000_000).unwrap();
        ladder.add_volume(100_100_000_000, 3_000_000_000).unwrap();
        assert_eq!(ladder.active_count(), 2);
        
        ladder.clear();
        assert_eq!(ladder.active_count(), 0);
        assert!(ladder.get_best_price().is_none());
    }

    #[test]
    fn test_price_out_of_range() {
        let ladder = PriceLadder::new(Side::Bid, 100_000_000, 99_000_000_000);
        
        // Price too low
        assert!(ladder.add_volume(98_000_000_000, 1_000_000_000).is_err());
        
        // Price too high (beyond MAX_PRICE_LEVELS * tick_size)
        let max_price = 99_000_000_000 + (MAX_PRICE_LEVELS as u64 * 100_000_000);
        assert!(ladder.add_volume(max_price, 1_000_000_000).is_err());
    }
}
