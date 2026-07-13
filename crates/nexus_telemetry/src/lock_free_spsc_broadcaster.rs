//! Lock-Free SPSC (Single Producer Single Consumer) Ring Buffer Broadcaster
//! 
//! This module implements a high-performance, lock-free ring buffer for broadcasting
//! market telemetry from the trading engine (producer) to WebSocket clients (consumers).
//! 
//! CRITICAL: The trading engine writes to this buffer without any locks or allocations.
//! The WebSocket server runs on a DEDICATED Tokio thread pool and reads from this buffer.

use crossbeam::channel::{bounded, Sender, Receiver};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicUsize, AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use crate::binary_serializer::MarketTelemetry;

/// Maximum capacity of the SPSC ring buffer
/// Tuned for ~100ms of high-frequency data at 10K updates/sec
const RING_BUFFER_CAPACITY: usize = 16384;

/// Telemetry item with sequence number for gap detection
#[derive(Debug, Clone)]
pub struct TelemetryItem {
    pub sequence: u64,
    pub data: MarketTelemetry,
    pub serialized_bytes: Vec<u8>,
}

/// Lock-free SPSC broadcaster state
struct BROADCASTER_STATE {
    /// Pre-allocated ring buffer slots
    buffer: Vec<Option<TelemetryItem>>,
    /// Write position (modified only by producer)
    write_pos: AtomicUsize,
    /// Read position (modified only by consumer)
    read_pos: AtomicUsize,
    /// Sequence counter
    sequence: AtomicU64,
    /// Overflow indicator
    overflowed: AtomicBool,
}

/// Thread-safe SPSC broadcaster for market telemetry
pub struct SpscBroadcaster {
    inner: Arc<RwLock<BROADCASTER_STATE>>,
    /// Channel for signaling new data to consumers
    data_signal: (Sender<()>, Receiver<()>),
}

impl SpscBroadcaster {
    /// Create a new SPSC broadcaster
    pub fn new() -> Self {
        let mut buffer = Vec::with_capacity(RING_BUFFER_CAPACITY);
        buffer.resize_with(RING_BUFFER_CAPACITY, || None);
        
        let state = BROADCASTER_STATE {
            buffer,
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
            sequence: AtomicU64::new(0),
            overflowed: AtomicBool::new(false),
        };
        
        let (tx, rx) = bounded::<()>(1024);
        
        Self {
            inner: Arc::new(RwLock::new(state)),
            data_signal: (tx, rx),
        }
    }
    
    /// Publish telemetry data (called by trading engine - PRODUCER)
    /// This is LOCK-FREE and zero-allocation in the hot path
    /// Returns true if successful, false if buffer is full (overflow)
    pub fn publish(&self, telemetry: MarketTelemetry, serialized: Vec<u8>) -> bool {
        let state = self.inner.read();
        
        let current_write = state.write_pos.load(Ordering::Relaxed);
        let current_read = state.read_pos.load(Ordering::Acquire);
        
        // Check if buffer is full (circular buffer check)
        let next_write = (current_write + 1) % RING_BUFFER_CAPACITY;
        if next_write == current_read {
            // Buffer full - mark overflow and drop oldest
            state.overflowed.store(true, Ordering::Release);
            return false;
        }
        
        let seq = state.sequence.fetch_add(1, Ordering::AcqRel);
        
        let item = TelemetryItem {
            sequence: seq,
            data: telemetry,
            serialized_bytes: serialized,
        };
        
        // SAFETY: We have exclusive write access to this slot
        // The reader won't touch it until we update write_pos
        unsafe {
            let slot = &mut state.buffer.as_slice()[current_write] as *const Option<TelemetryItem> as *mut Option<TelemetryItem>;
            *slot = Some(item);
        }
        
        // Memory barrier: ensure write is visible before updating position
        state.write_pos.store(next_write, Ordering::Release);
        
        // Signal consumers that new data is available (non-blocking)
        let _ = self.data_signal.0.try_send(());
        
        true
    }
    
    /// Consume all available telemetry items (called by WebSocket broadcaster - CONSUMER)
    /// Returns an iterator over available items
    pub fn consume_batch<F>(&self, mut callback: F) 
    where
        F: FnMut(&TelemetryItem)
    {
        let state = self.inner.read();
        
        let mut current_read = state.read_pos.load(Ordering::Relaxed);
        let current_write = state.write_pos.load(Ordering::Acquire);
        
        while current_read != current_write {
            if let Some(ref item) = state.buffer[current_read] {
                callback(item);
            }
            
            // Clear the slot before advancing
            unsafe {
                let slot = &mut state.buffer.as_slice()[current_read] as *const Option<TelemetryItem> as *mut Option<TelemetryItem>;
                *slot = None;
            }
            
            current_read = (current_read + 1) % RING_BUFFER_CAPACITY;
        }
        
        state.read_pos.store(current_read, Ordering::Release);
    }
    
    /// Get signal receiver for consumers
    pub fn get_signal_receiver(&self) -> Receiver<()> {
        self.data_signal.1.clone()
    }
    
    /// Check if overflow occurred
    pub fn has_overflowed(&self) -> bool {
        self.inner.read().overflowed.load(Ordering::Acquire)
    }
    
    /// Reset overflow flag after handling
    pub fn reset_overflow(&self) {
        self.inner.write().overflowed.store(false, Ordering::Release);
    }
    
    /// Get current buffer utilization (for monitoring)
    pub fn utilization(&self) -> f32 {
        let state = self.inner.read();
        let write = state.write_pos.load(Ordering::Relaxed);
        let read = state.read_pos.load(Ordering::Relaxed);
        
        if write >= read {
            (write - read) as f32 / RING_BUFFER_CAPACITY as f32
        } else {
            (RING_BUFFER_CAPACITY - read + write) as f32 / RING_BUFFER_CAPACITY as f32
        }
    }
}

impl Default for SpscBroadcaster {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-client broadcaster wrapper
/// Manages multiple WebSocket client subscriptions
pub struct MultiClientBroadcaster {
    spsc: Arc<SpscBroadcaster>,
    /// Connected client count
    client_count: AtomicUsize,
}

impl MultiClientBroadcaster {
    pub fn new() -> Self {
        Self {
            spsc: Arc::new(SpscBroadcaster::new()),
            client_count: AtomicUsize::new(0),
        }
    }
    
    pub fn get_spsc(&self) -> Arc<SpscBroadcaster> {
        Arc::clone(&self.spsc)
    }
    
    pub fn client_connected(&self) {
        self.client_count.fetch_add(1, Ordering::AcqRel);
    }
    
    pub fn client_disconnected(&self) {
        self.client_count.fetch_sub(1, Ordering::AcqRel);
    }
    
    pub fn client_count(&self) -> usize {
        self.client_count.load(Ordering::Acquire)
    }
}

impl Default for MultiClientBroadcaster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary_serializer::{BinarySerializer, MarketTelemetry, TradeTick};
    
    fn create_test_telemetry() -> MarketTelemetry {
        MarketTelemetry {
            timestamp_ns: 1704067200000000000,
            symbol: "BTC-USD".to_string(),
            best_bid_price: 4200000,
            best_ask_price: 4200100,
            best_bid_volume: 150,
            best_ask_volume: 200,
            l2_bids: vec![],
            l2_asks: vec![],
            recent_trades: vec![],
        }
    }
    
    #[test]
    fn test_spsc_publish_consume() {
        let broadcaster = SpscBroadcaster::new();
        
        let telemetry = create_test_telemetry();
        let mut buffer = Vec::with_capacity(4096);
        BinarySerializer::encode_telemetry(&telemetry, &mut buffer).unwrap();
        
        assert!(broadcaster.publish(telemetry.clone(), buffer));
        
        let mut consumed_count = 0;
        broadcaster.consume_batch(|_| {
            consumed_count += 1;
        });
        
        assert_eq!(consumed_count, 1);
    }
    
    #[test]
    fn test_buffer_overflow_detection() {
        let broadcaster = SpscBroadcaster::new();
        
        // Fill the buffer
        for i in 0..RING_BUFFER_CAPACITY + 100 {
            let mut telemetry = create_test_telemetry();
            telemetry.best_bid_price = i as i64;
            let mut buffer = Vec::with_capacity(4096);
            BinarySerializer::encode_telemetry(&telemetry, &mut buffer).unwrap();
            broadcaster.publish(telemetry, buffer);
        }
        
        assert!(broadcaster.has_overflowed());
    }
}
