//! Chapter 2: L2/L3 Order Book Reconstructor
//!
//! This module consumes OrderBookDelta events from the ring buffer and
//! maintains the exact state of the exchange order book using zero-allocation
//! data structures.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use nexus_core::concurrency::spsc_ring::Consumer;
use nexus_core::memory::cache_padder::CachePadded64;
use tracing::{debug, error, info, warn};

use crate::price_ladder::{PriceLadder, Side, MAX_PRICE_LEVELS};
use crate::l3_slab_allocator::{L3SlabAllocator, INVALID_ORDER_INDEX};

/// Type of delta operation
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaType {
    /// New order added
    Add = 0,
    /// Order modified (volume change)
    Modify = 1,
    /// Order removed/canceled
    Cancel = 2,
    /// Order executed (trade)
    Trade = 3,
    /// Snapshot/clear book
    Snapshot = 4,
}

impl DeltaType {
    #[inline]
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(DeltaType::Add),
            1 => Some(DeltaType::Modify),
            2 => Some(DeltaType::Cancel),
            3 => Some(DeltaType::Trade),
            4 => Some(DeltaType::Snapshot),
            _ => None,
        }
    }
}

/// Order book delta event
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OrderBookDelta {
    /// Event timestamp in nanoseconds
    pub timestamp_ns: CachePadded64<AtomicU64>,
    /// Order ID
    pub order_id: CachePadded64<AtomicU64>,
    /// Price in nanodollars
    pub price: CachePadded64<AtomicU64>,
    /// Volume in base units * 1e9
    pub volume: CachePadded64<AtomicU64>,
    /// Side (0=Bid, 1=Ask)
    pub side: CachePadded64<AtomicU8>,
    /// Delta type
    pub delta_type: CachePadded64<AtomicU8>,
    /// Sequence number for ordering
    pub sequence: CachePadded64<AtomicU64>,
    /// Exchange ID
    pub exchange_id: CachePadded64<AtomicU16>,
    /// Padding
    pub _padding: [u8; 50],
}

use std::sync::atomic::AtomicU8;
use std::sync::atomic::AtomicU16;

// SAFETY: OrderBookDelta is used in lock-free contexts
unsafe impl Send for OrderBookDelta {}
unsafe impl Sync for OrderBookDelta {}

impl OrderBookDelta {
    #[inline]
    pub const fn new() -> Self {
        Self {
            timestamp_ns: CachePadded64::new(AtomicU64::new(0)),
            order_id: CachePadded64::new(AtomicU64::new(0)),
            price: CachePadded64::new(AtomicU64::new(0)),
            volume: CachePadded64::new(AtomicU64::new(0)),
            side: CachePadded64::new(AtomicU8::new(0)),
            delta_type: CachePadded64::new(AtomicU8::new(0)),
            sequence: CachePadded64::new(AtomicU64::new(0)),
            exchange_id: CachePadded64::new(AtomicU16::new(0)),
            _padding: [0; 50],
        }
    }

    #[inline]
    pub fn get_side(&self) -> Side {
        match self.side.0.load(Ordering::Acquire) {
            0 => Side::Bid,
            1 => Side::Ask,
            _ => Side::Bid, // Default
        }
    }

    #[inline]
    pub fn get_delta_type(&self) -> Option<DeltaType> {
        DeltaType::from_u8(self.delta_type.0.load(Ordering::Acquire))
    }
}

impl Default for OrderBookDelta {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for order book operations
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct OrderBookStats {
    /// Total deltas processed
    pub deltas_processed: CachePadded64<AtomicU64>,
    /// Adds processed
    pub adds_processed: CachePadded64<AtomicU64>,
    /// Modifies processed
    pub modifies_processed: CachePadded64<AtomicU64>,
    /// Cancels processed
    pub cancels_processed: CachePadded64<AtomicU64>,
    /// Trades processed
    pub trades_processed: CachePadded64<AtomicU64>,
    /// Invalid deltas
    pub invalid_deltas: CachePadded64<AtomicU64>,
    /// Last sequence number
    pub last_sequence: CachePadded64<AtomicU64>,
    /// Sequence gaps detected
    pub sequence_gaps: CachePadded64<AtomicU64>,
}

impl OrderBookStats {
    #[inline]
    pub fn new() -> Self {
        Self {
            deltas_processed: CachePadded64::new(AtomicU64::new(0)),
            adds_processed: CachePadded64::new(AtomicU64::new(0)),
            modifies_processed: CachePadded64::new(AtomicU64::new(0)),
            cancels_processed: CachePadded64::new(AtomicU64::new(0)),
            trades_processed: CachePadded64::new(AtomicU64::new(0)),
            invalid_deltas: CachePadded64::new(AtomicU64::new(0)),
            last_sequence: CachePadded64::new(AtomicU64::new(0)),
            sequence_gaps: CachePadded64::new(AtomicU64::new(0)),
        }
    }
}

/// Full order book reconstructor
pub struct LobReconstructor {
    /// Bid side ladder
    bids: PriceLadder,
    /// Ask side ladder
    asks: PriceLadder,
    /// L3 order tracking
    l3_slab: Arc<L3SlabAllocator>,
    /// Statistics
    stats: CachePadded64<OrderBookStats>,
    /// Tick size in nanodollars
    tick_size: u64,
    /// Base price for bucket calculation
    base_price: u64,
    /// Last bid price (cached)
    last_bid: CachePadded64<AtomicU64>,
    /// Last ask price (cached)
    last_ask: CachePadded64<AtomicU64>,
}

// SAFETY: LobReconstructor is designed for single-threaded consumption
unsafe impl Send for LobReconstructor {}
unsafe impl Sync for LobReconstructor {}

impl LobReconstructor {
    /// Create a new order book reconstructor
    #[inline]
    pub fn new(tick_size: u64, base_price: u64) -> Self {
        Self {
            bids: PriceLadder::new(Side::Bid, tick_size, base_price),
            asks: PriceLadder::new(Side::Ask, tick_size, base_price),
            l3_slab: Arc::new(L3SlabAllocator::new()),
            stats: CachePadded64::new(OrderBookStats::new()),
            tick_size,
            base_price,
            last_bid: CachePadded64::new(AtomicU64::new(0)),
            last_ask: CachePadded64::new(AtomicU64::new(0)),
        }
    }

    /// Process a single delta event
    #[inline]
    pub fn process_delta(&mut self, delta: &OrderBookDelta) -> Result<(), &'static str> {
        self.stats.0.deltas_processed.0.fetch_add(1, Ordering::Relaxed);
        
        // Check sequence ordering
        let seq = delta.sequence.0.load(Ordering::Acquire);
        let last_seq = self.stats.0.last_sequence.0.load(Ordering::Acquire);
        
        if seq > 0 && seq != last_seq + 1 {
            self.stats.0.sequence_gaps.0.fetch_add(1, Ordering::Relaxed);
            debug!("Sequence gap detected: expected {}, got {}", last_seq + 1, seq);
        }
        
        self.stats.0.last_sequence.0.store(seq, Ordering::Release);
        
        let side = delta.get_side();
        let delta_type = delta.get_delta_type()
            .ok_or("Invalid delta type")?;
        
        let price = delta.price.0.load(Ordering::Acquire);
        let volume = delta.volume.0.load(Ordering::Acquire);
        let order_id = delta.order_id.0.load(Ordering::Acquire);
        
        match delta_type {
            DeltaType::Add => {
                self.stats.0.adds_processed.0.fetch_add(1, Ordering::Relaxed);
                self.handle_add(side, price, volume, order_id)?;
            }
            DeltaType::Modify => {
                self.stats.0.modifies_processed.0.fetch_add(1, Ordering::Relaxed);
                self.handle_modify(side, price, volume, order_id)?;
            }
            DeltaType::Cancel => {
                self.stats.0.cancels_processed.0.fetch_add(1, Ordering::Relaxed);
                self.handle_cancel(side, order_id)?;
            }
            DeltaType::Trade => {
                self.stats.0.trades_processed.0.fetch_add(1, Ordering::Relaxed);
                self.handle_trade(side, price, volume, order_id)?;
            }
            DeltaType::Snapshot => {
                self.clear();
                info!("Order book cleared via snapshot");
            }
        }
        
        // Update cached best prices
        self.update_cached_prices();
        
        Ok(())
    }

    /// Handle ADD delta
    #[inline]
    fn handle_add(&self, side: Side, price: u64, volume: u64, order_id: u64) -> Result<(), &'static str> {
        let ladder = match side {
            Side::Bid => &self.bids,
            Side::Ask => &self.asks,
        };
        
        // Add to L2
        ladder.add_volume(price, volume)?;
        
        // Add to L3
        let side_u8 = match side {
            Side::Bid => 0,
            Side::Ask => 1,
        };
        
        if let Some(order_idx) = self.l3_slab.allocate(order_id, price, volume, side_u8) {
            // Get tail of price level for intrusive list
            let tail_idx = self.get_level_tail(side, price);
            let _ = self.l3_slab.insert_into_level(order_idx, tail_idx);
        }
        
        Ok(())
    }

    /// Handle MODIFY delta
    #[inline]
    fn handle_modify(&self, side: Side, price: u64, volume: u64, order_id: u64) -> Result<(), &'static str> {
        let ladder = match side {
            Side::Bid => &self.bids,
            Side::Ask => &self.asks,
        };
        
        // Find existing order
        if let Some(order_idx) = self.l3_slab.find_by_order_id(order_id) {
            let record = self.l3_slab.get_record(order_idx)
                .ok_or("Order not found")?;
            
            let old_volume = record.get_volume();
            let old_price = record.get_price();
            
            // Update L3
            record.volume.0.store(volume, Ordering::Release);
            record.price.0.store(price, Ordering::Release);
            
            // Update L2 - remove old, add new
            if old_price != price {
                ladder.remove_volume(old_price, old_volume)?;
                ladder.add_volume(price, volume)?;
            } else {
                let vol_delta = volume as i64 - old_volume as i64;
                let level = ladder.get_or_create_level(price)
                    .ok_or("Price out of range")?;
                level.add_volume(vol_delta);
            }
        } else {
            // Order not found, treat as add
            self.handle_add(side, price, volume, order_id)?;
        }
        
        Ok(())
    }

    /// Handle CANCEL delta
    #[inline]
    fn handle_cancel(&self, side: Side, order_id: u64) -> Result<(), &'static str> {
        // Find order in L3
        if let Some(order_idx) = self.l3_slab.find_by_order_id(order_id) {
            let record = self.l3_slab.get_record(order_idx)
                .ok_or("Order not found")?;
            
            let price = record.get_price();
            let volume = record.get_volume();
            
            // Remove from L3 intrusive list
            self.l3_slab.remove_from_level(order_idx)?;
            
            // Free the slot
            self.l3_slab.free(order_idx)?;
            
            // Update L2
            let ladder = match side {
                Side::Bid => &self.bids,
                Side::Ask => &self.asks,
            };
            
            ladder.remove_volume(price, volume)?;
        }
        
        Ok(())
    }

    /// Handle TRADE delta
    #[inline]
    fn handle_trade(&self, side: Side, price: u64, volume: u64, order_id: u64) -> Result<(), &'static str> {
        // Trades reduce volume at price level
        let ladder = match side {
            Side::Bid => &self.bids,
            Side::Ask => &self.asks,
        };
        
        // Try to find and update the order
        if let Some(order_idx) = self.l3_slab.find_by_order_id(order_id) {
            let record = self.l3_slab.get_record(order_idx)
                .ok_or("Order not found")?;
            
            let current_vol = record.get_volume();
            let new_vol = current_vol.saturating_sub(volume);
            
            record.volume.0.store(new_vol, Ordering::Release);
            
            // Update L2
            ladder.remove_volume(price, volume)?;
            
            // If fully filled, cancel the order
            if new_vol == 0 {
                self.l3_slab.remove_from_level(order_idx)?;
                self.l3_slab.free(order_idx)?;
            }
        } else {
            // Aggressor order not tracked, just update L2
            ladder.remove_volume(price, volume)?;
        }
        
        Ok(())
    }

    /// Get tail index of a price level's intrusive list
    #[inline]
    fn get_level_tail(&self, side: Side, price: u64) -> usize {
        // Simplified: would need to track tail per level in production
        INVALID_ORDER_INDEX
    }

    /// Update cached best prices
    #[inline]
    fn update_cached_prices(&self) {
        if let Some(bid) = self.bids.get_best_price() {
            self.last_bid.0.store(bid, Ordering::Release);
        }
        if let Some(ask) = self.asks.get_best_price() {
            self.last_ask.0.store(ask, Ordering::Release);
        }
    }

    /// Get best bid price
    #[inline]
    pub fn best_bid(&self) -> Option<u64> {
        self.bids.get_best_price()
    }

    /// Get best ask price
    #[inline]
    pub fn best_ask(&self) -> Option<u64> {
        self.asks.get_best_price()
    }

    /// Get spread in nanodollars
    #[inline]
    pub fn spread(&self) -> Option<u64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                if ask > bid {
                    Some(ask - bid)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get mid price
    #[inline]
    pub fn mid_price(&self) -> Option<u64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2),
            _ => None,
        }
    }

    /// Get bid volume at best price
    #[inline]
    pub fn best_bid_volume(&self) -> Option<u64> {
        self.bids.get_best_volume()
    }

    /// Get ask volume at best price
    #[inline]
    pub fn best_ask_volume(&self) -> Option<u64> {
        self.asks.get_best_volume()
    }

    /// Clear the order book
    #[inline]
    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.l3_slab.clear();
        self.last_bid.0.store(0, Ordering::Release);
        self.last_ask.0.store(0, Ordering::Release);
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> OrderBookStats {
        self.stats.0.clone()
    }

    /// Get L3 slab reference
    #[inline]
    pub fn l3_slab(&self) -> &Arc<L3SlabAllocator> {
        &self.l3_slab
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_type_conversion() {
        assert_eq!(DeltaType::from_u8(0), Some(DeltaType::Add));
        assert_eq!(DeltaType::from_u8(1), Some(DeltaType::Modify));
        assert_eq!(DeltaType::from_u8(2), Some(DeltaType::Cancel));
        assert_eq!(DeltaType::from_u8(3), Some(DeltaType::Trade));
        assert_eq!(DeltaType::from_u8(4), Some(DeltaType::Snapshot));
        assert_eq!(DeltaType::from_u8(5), None);
    }

    #[test]
    fn test_reconstructor_basic() {
        let mut recon = LobReconstructor::new(100_000_000, 99_000_000_000);
        
        assert!(recon.best_bid().is_none());
        assert!(recon.best_ask().is_none());
        assert!(recon.spread().is_none());
    }

    #[test]
    fn test_process_add_delta() {
        let mut recon = LobReconstructor::new(100_000_000, 99_000_000_000);
        
        let mut delta = OrderBookDelta::new();
        delta.order_id.0.store(12345, Ordering::Release);
        delta.price.0.store(99_100_000_000, Ordering::Release);
        delta.volume.0.store(1_000_000_000, Ordering::Release);
        delta.side.0.store(0, Ordering::Release); // Bid
        delta.delta_type.0.store(0, Ordering::Release); // Add
        delta.sequence.0.store(1, Ordering::Release);
        
        assert!(recon.process_delta(&delta).is_ok());
        
        assert_eq!(recon.best_bid(), Some(99_100_000_000));
        assert_eq!(recon.best_bid_volume(), Some(1_000_000_000));
    }

    #[test]
    fn test_process_cancel_delta() {
        let mut recon = LobReconstructor::new(100_000_000, 99_000_000_000);
        
        // Add order
        let mut delta = OrderBookDelta::new();
        delta.order_id.0.store(12345, Ordering::Release);
        delta.price.0.store(99_100_000_000, Ordering::Release);
        delta.volume.0.store(1_000_000_000, Ordering::Release);
        delta.side.0.store(0, Ordering::Release);
        delta.delta_type.0.store(0, Ordering::Release);
        delta.sequence.0.store(1, Ordering::Release);
        
        assert!(recon.process_delta(&delta).is_ok());
        
        // Cancel order
        let mut cancel = OrderBookDelta::new();
        cancel.order_id.0.store(12345, Ordering::Release);
        cancel.side.0.store(0, Ordering::Release);
        cancel.delta_type.0.store(2, Ordering::Release); // Cancel
        cancel.sequence.0.store(2, Ordering::Release);
        
        assert!(recon.process_delta(&cancel).is_ok());
        
        assert!(recon.best_bid().is_none());
    }

    #[test]
    fn test_spread_and_mid() {
        let mut recon = LobReconstructor::new(100_000_000, 99_000_000_000);
        
        // Add bid
        let mut bid = OrderBookDelta::new();
        bid.order_id.0.store(1, Ordering::Release);
        bid.price.0.store(99_100_000_000, Ordering::Release);
        bid.volume.0.store(1_000_000_000, Ordering::Release);
        bid.side.0.store(0, Ordering::Release);
        bid.delta_type.0.store(0, Ordering::Release);
        bid.sequence.0.store(1, Ordering::Release);
        recon.process_delta(&bid).unwrap();
        
        // Add ask
        let mut ask = OrderBookDelta::new();
        ask.order_id.0.store(2, Ordering::Release);
        ask.price.0.store(99_200_000_000, Ordering::Release);
        ask.volume.0.store(1_000_000_000, Ordering::Release);
        ask.side.0.store(1, Ordering::Release);
        ask.delta_type.0.store(0, Ordering::Release);
        ask.sequence.0.store(2, Ordering::Release);
        recon.process_delta(&ask).unwrap();
        
        assert_eq!(recon.spread(), Some(100_000_000));
        assert_eq!(recon.mid_price(), Some(99_150_000_000));
    }

    #[test]
    fn test_statistics() {
        let mut recon = LobReconstructor::new(100_000_000, 99_000_000_000);
        
        // Add an order
        let mut delta = OrderBookDelta::new();
        delta.order_id.0.store(1, Ordering::Release);
        delta.price.0.store(99_100_000_000, Ordering::Release);
        delta.volume.0.store(1_000_000_000, Ordering::Release);
        delta.side.0.store(0, Ordering::Release);
        delta.delta_type.0.store(0, Ordering::Release);
        delta.sequence.0.store(1, Ordering::Release);
        recon.process_delta(&delta).unwrap();
        
        let stats = recon.get_stats();
        assert_eq!(*stats.deltas_processed.0.get(), 1);
        assert_eq!(*stats.adds_processed.0.get(), 1);
    }
}
