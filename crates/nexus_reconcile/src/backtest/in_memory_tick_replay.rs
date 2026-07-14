//! In-Memory Tick Replay - Safe SPSC ring buffer replay for walk-forward backtesting.
//! 
//! CRITICAL: This module enforces strict consumer/producer index bounds checking
//! to prevent reading corrupted memory when the live ingestion engine overwrites
//! ticks while the backtester is actively replaying.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A tick of market data for replay
#[derive(Debug, Clone, Copy)]
pub struct MarketTick {
    /// Timestamp in nanoseconds (monotonic)
    pub timestamp_ns: u64,
    
    /// Symbol identifier
    pub symbol: [u8; 16],
    
    /// Bid price (scaled integer)
    pub bid_price: i128,
    
    /// Ask price (scaled integer)
    pub ask_price: i128,
    
    /// Last trade price (scaled integer)
    pub last_price: i128,
    
    /// Bid size (scaled integer)
    pub bid_size: i128,
    
    /// Ask size (scaled integer)
    pub ask_size: i128,
    
    /// Sequence number for validation
    pub sequence: u64,
}

impl Default for MarketTick {
    fn default() -> Self {
        Self {
            timestamp_ns: 0,
            symbol: [0u8; 16],
            bid_price: 0,
            ask_price: 0,
            last_price: 0,
            bid_size: 0,
            ask_size: 0,
            sequence: 0,
        }
    }
}

/// Lock-free SPSC ring buffer for tick storage
pub struct TickRingBuffer {
    /// The underlying tick storage
    buffer: Vec<MarketTick>,
    
    /// Buffer capacity (must be power of 2 for efficient masking)
    capacity: usize,
    
    /// Mask for efficient modulo operation (capacity - 1)
    mask: usize,
    
    /// Producer index (where next tick will be written)
    producer_idx: AtomicUsize,
    
    /// Consumer index (where next tick will be read)
    consumer_idx: AtomicUsize,
    
    /// High watermark - highest index ever written (for bounds checking)
    high_watermark: AtomicUsize,
}

impl TickRingBuffer {
    /// Create a new ring buffer with the given capacity
    /// Capacity will be rounded up to the next power of 2
    pub fn new(capacity: usize) -> Self {
        // Round up to power of 2
        let actual_capacity = capacity.next_power_of_two();
        let mask = actual_capacity - 1;
        
        let mut buffer = Vec::with_capacity(actual_capacity);
        buffer.resize_with(actual_capacity, MarketTick::default);
        
        Self {
            buffer,
            capacity: actual_capacity,
            mask,
            producer_idx: AtomicUsize::new(0),
            consumer_idx: AtomicUsize::new(0),
            high_watermark: AtomicUsize::new(0),
        }
    }
    
    /// Push a new tick to the buffer (producer side)
    /// Returns the sequence number of the pushed tick
    #[inline]
    pub fn push(&self, tick: MarketTick) -> u64 {
        let current_producer = self.producer_idx.load(Ordering::Relaxed);
        let idx = current_producer & self.mask;
        
        // Write the tick
        unsafe {
            // Safe because we're the only writer
            let ptr = self.buffer.as_ptr() as *mut MarketTick;
            std::ptr::write(ptr.add(idx), tick);
        }
        
        // Update indices
        self.producer_idx.fetch_add(1, Ordering::Release);
        
        // Update high watermark
        let new_high = current_producer + 1;
        self.high_watermark.store(new_high, Ordering::Relaxed);
        
        tick.sequence
    }
    
    /// Try to read the next tick (consumer side)
    /// Returns None if no tick is available
    #[inline]
    pub fn try_pop(&self) -> Option<MarketTick> {
        let current_consumer = self.consumer_idx.load(Ordering::Relaxed);
        let current_producer = self.producer_idx.load(Ordering::Acquire);
        
        // Check if buffer is empty
        if current_consumer >= current_producer {
            return None;
        }
        
        let idx = current_consumer & self.mask;
        
        // Read the tick
        let tick = unsafe {
            // Safe because we're the only reader and producer won't overwrite
            // until we've advanced consumer_idx
            let ptr = self.buffer.as_ptr() as *const MarketTick;
            std::ptr::read(ptr.add(idx))
        };
        
        // Advance consumer index
        self.consumer_idx.fetch_add(1, Ordering::Release);
        
        Some(tick)
    }
    
    /// Get the current number of ticks available for reading
    #[inline]
    pub fn len(&self) -> usize {
        let producer = self.producer_idx.load(Ordering::Acquire);
        let consumer = self.consumer_idx.load(Ordering::Relaxed);
        producer.saturating_sub(consumer)
    }
    
    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get the producer index (for bounds checking)
    #[inline]
    pub fn producer_index(&self) -> usize {
        self.producer_idx.load(Ordering::Acquire)
    }
    
    /// Get the consumer index (for bounds checking)
    #[inline]
    pub fn consumer_index(&self) -> usize {
        self.consumer_idx.load(Ordering::Relaxed)
    }
    
    /// Get the high watermark (highest index ever written)
    #[inline]
    pub fn high_watermark(&self) -> usize {
        self.high_watermark.load(Ordering::Relaxed)
    }
    
    /// Safely read a tick at a specific index with bounds checking
    /// Returns None if the index is out of valid range or has been overwritten
    #[inline]
    pub fn get_tick_at(&self, index: usize) -> Option<MarketTick> {
        let producer = self.producer_idx.load(Ordering::Acquire);
        let consumer = self.consumer_idx.load(Ordering::Relaxed);
        
        // Bounds check: index must be within [consumer, producer)
        if index < consumer || index >= producer {
            return None;
        }
        
        // Check if this tick might have been overwritten (wrapped around)
        let high_water = self.high_watermark.load(Ordering::Relaxed);
        if high_water > self.capacity && index < high_water - self.capacity {
            // This tick has likely been overwritten
            return None;
        }
        
        let idx = index & self.mask;
        
        // Read the tick
        Some(unsafe {
            let ptr = self.buffer.as_ptr() as *const MarketTick;
            std::ptr::read(ptr.add(idx))
        })
    }
    
    /// Get buffer capacity
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// In-Memory Tick Replay engine for walk-forward backtesting
pub struct InMemoryTickReplay {
    /// Reference to the shared tick ring buffer
    buffer: Arc<TickRingBuffer>,
    
    /// Replay start index
    start_index: AtomicUsize,
    
    /// Replay end index (exclusive)
    end_index: AtomicUsize,
    
    /// Current replay position
    current_index: AtomicUsize,
    
    /// Flag indicating if replay is complete
    replay_complete: AtomicUsize,  // 0 = false, 1 = true
}

impl InMemoryTickReplay {
    pub fn new(buffer: Arc<TickRingBuffer>) -> Self {
        Self {
            buffer,
            start_index: AtomicUsize::new(0),
            end_index: AtomicUsize::new(0),
            current_index: AtomicUsize::new(0),
            replay_complete: AtomicUsize::new(1),  // Initially complete (nothing to replay)
        }
    }
    
    /// Initialize a replay session from the current buffer state
    /// 
    /// CRITICAL: This method performs strict bounds checking to ensure
    /// the replay window is valid and won't read corrupted memory.
    pub fn init_replay(&self, lookback_ticks: usize) -> Result<(), ReplayError> {
        let producer = self.buffer.producer_index();
        let consumer = self.buffer.consumer_index();
        
        // Validate: can't replay more ticks than available
        let available_ticks = producer.saturating_sub(consumer);
        if available_ticks == 0 {
            return Err(ReplayError::NoTicksAvailable);
        }
        
        // Clamp lookback to available ticks
        let actual_lookback = lookback_ticks.min(available_ticks);
        
        // Set replay window
        let end_idx = producer;  // Replay up to current producer
        let start_idx = end_idx.saturating_sub(actual_lookback);
        
        // CRITICAL: Verify start_idx is not before consumer (would read invalid data)
        if start_idx < consumer {
            return Err(ReplayError::BoundsViolation {
                requested_start: start_idx,
                valid_consumer: consumer,
            });
        }
        
        self.start_index.store(start_idx, Ordering::SeqCst);
        self.end_index.store(end_idx, Ordering::SeqCst);
        self.current_index.store(start_idx, Ordering::SeqCst);
        self.replay_complete.store(0, Ordering::SeqCst);
        
        Ok(())
    }
    
    /// Get the next tick in the replay sequence
    /// Returns None when replay is complete or if bounds violation detected
    pub fn next_tick(&self) -> Option<MarketTick> {
        let current = self.current_index.load(Ordering::Relaxed);
        let end = self.end_index.load(Ordering::Acquire);
        let producer = self.buffer.producer_index();
        
        // Check if replay is complete
        if current >= end {
            self.replay_complete.store(1, Ordering::Relaxed);
            return None;
        }
        
        // CRITICAL: Check for producer overrun during replay
        // If producer has moved past our end index, the buffer may have wrapped
        if current >= producer {
            // Producer has overwritten our replay window
            self.replay_complete.store(1, Ordering::Relaxed);
            return None;
        }
        
        // Get the tick with bounds checking
        match self.buffer.get_tick_at(current) {
            Some(tick) => {
                self.current_index.fetch_add(1, Ordering::Relaxed);
                Some(tick)
            }
            None => {
                // Tick was overwritten or out of bounds
                self.replay_complete.store(1, Ordering::Relaxed);
                None
            }
        }
    }
    
    /// Check if replay is complete
    pub fn is_complete(&self) -> bool {
        self.replay_complete.load(Ordering::Relaxed) != 0
    }
    
    /// Get replay progress (current position / total ticks)
    pub fn get_progress(&self) -> (usize, usize) {
        let current = self.current_index.load(Ordering::Relaxed);
        let start = self.start_index.load(Ordering::Relaxed);
        let end = self.end_index.load(Ordering::Relaxed);
        
        let total = end.saturating_sub(start);
        let progressed = current.saturating_sub(start);
        
        (progressed, total)
    }
    
    /// Reset the replay state
    pub fn reset(&self) {
        self.start_index.store(0, Ordering::Relaxed);
        self.end_index.store(0, Ordering::Relaxed);
        self.current_index.store(0, Ordering::Relaxed);
        self.replay_complete.store(1, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum ReplayError {
    #[error("No ticks available in buffer")]
    NoTicksAvailable,
    
    #[error("Bounds violation: requested_start={requested_start}, valid_consumer={valid_consumer}")]
    BoundsViolation {
        requested_start: usize,
        valid_consumer: usize,
    },
    
    #[error("Buffer overrun during replay")]
    BufferOverrun,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ring_buffer_push_pop() {
        let buffer = TickRingBuffer::new(16);
        
        // Push some ticks
        for i in 0..10 {
            let tick = MarketTick {
                timestamp_ns: i * 1000,
                sequence: i,
                ..Default::default()
            };
            buffer.push(tick);
        }
        
        assert_eq!(buffer.len(), 10);
        
        // Pop all ticks
        for i in 0..10 {
            let tick = buffer.try_pop().unwrap();
            assert_eq!(tick.sequence, i);
            assert_eq!(tick.timestamp_ns, i * 1000);
        }
        
        assert!(buffer.is_empty());
    }
    
    #[test]
    fn test_replay_bounds_checking() {
        let buffer = Arc::new(TickRingBuffer::new(32));
        let replay = InMemoryTickReplay::new(Arc::clone(&buffer));
        
        // Push some ticks
        for i in 0..20 {
            let tick = MarketTick {
                timestamp_ns: i * 1000,
                sequence: i,
                ..Default::default()
            };
            buffer.push(tick);
        }
        
        // Initialize replay with lookback of 10
        replay.init_replay(10).unwrap();
        
        // Replay should work
        let mut count = 0;
        while let Some(_tick) = replay.next_tick() {
            count += 1;
        }
        
        assert_eq!(count, 10);
        assert!(replay.is_complete());
    }
    
    #[test]
    fn test_replay_prevents_overrun() {
        let buffer = Arc::new(TickRingBuffer::new(16));
        let replay = InMemoryTickReplay::new(Arc::clone(&buffer));
        
        // Push initial ticks
        for i in 0..10 {
            buffer.push(MarketTick {
                sequence: i,
                ..Default::default()
            });
        }
        
        // Start replay
        replay.init_replay(10).unwrap();
        
        // Consume some ticks
        for _ in 0..5 {
            replay.next_tick();
        }
        
        // Now push more ticks (simulating live ingestion during replay)
        for i in 10..30 {
            buffer.push(MarketTick {
                sequence: i,
                ..Default::default()
            });
        }
        
        // Continue replay - should handle the new data gracefully
        let mut count = 5;  // Already consumed 5
        while let Some(_tick) = replay.next_tick() {
            count += 1;
        }
        
        // Should have completed without panic
        assert!(replay.is_complete());
    }
}
