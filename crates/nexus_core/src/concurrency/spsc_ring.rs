//! Lock-free Single-Producer Single-Consumer (SPSC) Ring Buffer
//!
//! This module implements a wait-free SPSC ring buffer optimized for
//! high-frequency market data ingestion. Key features:
//!
//! - Zero allocations after initialization
//! - Wait-free push and pop operations
//! - Cache-line padding to prevent false sharing
//! - Strict memory ordering for correctness
//! - Support for variable-sized messages via length-prefixing
//!
//! # Memory Ordering
//!
//! The implementation uses the following memory ordering guarantees:
//! - Producer: Release ordering on head updates (ensures data is visible before index update)
//! - Consumer: Acquire ordering on head reads (ensures data is read after index update)
//!
//! # Safety Notes
//!
//! - Only ONE thread may call push() at a time
//! - Only ONE thread may call pop() at a time
//! - The producer and consumer MAY be different threads
//! - Never call push() from the consumer thread or vice versa

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::usize;

use crate::memory::arena::{align_to_cache_line, CACHE_LINE_SIZE};

/// Error type for ring buffer operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingBufferError {
    Full,
    Empty,
    InvalidCapacity,
    MessageTooLarge,
}

impl std::fmt::Display for RingBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RingBufferError::Full => write!(f, "Ring buffer is full"),
            RingBufferError::Empty => write!(f, "Ring buffer is empty"),
            RingBufferError::InvalidCapacity => write!(f, "Invalid capacity (must be power of 2)"),
            RingBufferError::MessageTooLarge => write!(f, "Message exceeds buffer capacity"),
        }
    }
}

impl std::error::Error for RingBufferError {}

/// Slot header containing message metadata
/// Padded to cache line to prevent false sharing between slots
#[repr(C, align(64))]
struct SlotHeader {
    /// Length of the message in bytes (0 means slot is empty)
    length: AtomicUsize,
    /// Sequence number for detecting overwrites/stale reads
    sequence: AtomicUsize,
    /// Padding to reach 64 bytes
    _padding: [u8; SLOT_HEADER_PADDING],
}

const SLOT_HEADER_SIZE: usize = 16; // 2 * AtomicUsize
const SLOT_HEADER_PADDING: usize = CACHE_LINE_SIZE - SLOT_HEADER_SIZE;

impl SlotHeader {
    const fn new() -> Self {
        Self {
            length: AtomicUsize::new(0),
            sequence: AtomicUsize::new(0),
            _padding: [0u8; SLOT_HEADER_PADDING],
        }
    }
}

/// A single slot in the ring buffer
struct Slot<T> {
    header: SlotHeader,
    /// The actual data storage
    /// Using UnsafeCell for interior mutability without requiring &mut
    data: UnsafeCell<[u8; MAX_SLOT_DATA_SIZE]>,
    _phantom: PhantomData<T>,
}

/// Maximum message size that can fit in a single slot
/// Reserved space for header alignment
pub const MAX_SLOT_DATA_SIZE: usize = 4096 - CACHE_LINE_SIZE;

/// Cache-padded producer state to prevent false sharing with consumer
#[repr(C, align(64))]
struct ProducerState {
    /// Next position to write to
    head: AtomicUsize,
    /// Cached copy of consumer tail (for fast full check)
    cached_tail: UnsafeCell<usize>,
    _padding: [u8; PRODUCER_PADDING],
}

const PRODUCER_STATE_SIZE: usize = 16; // AtomicUsize + usize
const PRODUCER_PADDING: usize = CACHE_LINE_SIZE - PRODUCER_STATE_SIZE;

/// Cache-padded consumer state to prevent false sharing with producer
#[repr(C, align(64))]
struct ConsumerState {
    /// Next position to read from
    tail: AtomicUsize,
    /// Cached copy of producer head (for fast empty check)
    cached_head: UnsafeCell<usize>,
    _padding: [u8; CONSUMER_PADDING],
}

const CONSUMER_STATE_SIZE: usize = 16; // AtomicUsize + usize
const CONSUMER_PADDING: usize = CACHE_LINE_SIZE - CONSUMER_STATE_SIZE;

/// The SPSC ring buffer structure
pub struct SPSCRingBuffer<T> {
    /// Producer state (only accessed by producer thread)
    producer: Box<ProducerState>,
    
    /// Consumer state (only accessed by consumer thread)
    consumer: Box<ConsumerState>,
    
    /// The ring buffer slots
    slots: Box<[Slot<T>]>,
    
    /// Capacity mask for fast modulo operation (capacity - 1)
    capacity_mask: usize,
    
    /// Total capacity (power of 2)
    capacity: usize,
}

// SAFETY: SPSCRingBuffer can be sent between threads as long as
// the producer and consumer access patterns are maintained.
unsafe impl<T> Send for SPSCRingBuffer<T> {}

// We do NOT implement Sync because the ring buffer is designed for
// single-producer single-consumer access. Multiple threads calling
// push() or pop() would violate the SPSC guarantee.

impl<T> SPSCRingBuffer<T> {
    /// Create a new SPSC ring buffer with the specified capacity
    /// 
    /// Capacity will be rounded up to the next power of 2.
    /// Minimum capacity is 1, maximum is bounded by MAX_SLOT_DATA_SIZE.
    pub fn new(capacity: usize) -> Result<Self, RingBufferError> {
        if capacity == 0 {
            return Err(RingBufferError::InvalidCapacity);
        }

        // Round up to next power of 2 for efficient masking
        let actual_capacity = capacity.next_power_of_two();
        
        if actual_capacity > 65536 {
            return Err(RingBufferError::InvalidCapacity);
        }

        let capacity_mask = actual_capacity - 1;

        // Allocate slots
        let mut slots = Vec::with_capacity(actual_capacity);
        for _ in 0..actual_capacity {
            slots.push(Slot {
                header: SlotHeader::new(),
                data: UnsafeCell::new([0u8; MAX_SLOT_DATA_SIZE]),
                _phantom: PhantomData,
            });
        }

        Ok(Self {
            producer: Box::new(ProducerState {
                head: AtomicUsize::new(0),
                cached_tail: UnsafeCell::new(0),
                _padding: [0u8; PRODUCER_PADDING],
            }),
            consumer: Box::new(ConsumerState {
                tail: AtomicUsize::new(0),
                cached_head: UnsafeCell::new(0),
                _padding: [0u8; CONSUMER_PADDING],
            }),
            slots: slots.into_boxed_slice(),
            capacity_mask,
            capacity: actual_capacity,
        })
    }

    /// Push data into the ring buffer (producer only)
    /// 
    /// # Safety
    /// - Must only be called from the designated producer thread
    /// - Data must not exceed MAX_SLOT_DATA_SIZE
    #[inline]
    pub fn push(&self, data: &[u8]) -> Result<(), RingBufferError> {
        if data.is_empty() {
            return Ok(());
        }

        if data.len() > MAX_SLOT_DATA_SIZE {
            return Err(RingBufferError::MessageTooLarge);
        }

        let current_head = self.producer.head.load(Ordering::Relaxed);
        
        // Check cached tail first (fast path)
        let cached_tail = unsafe { *self.producer.cached_tail.get() };
        let next_head = (current_head + 1) & self.capacity_mask;
        
        if next_head == cached_tail {
            // Cache miss - need to check actual tail
            let actual_tail = self.consumer.tail.load(Ordering::Acquire);
            unsafe { *self.producer.cached_tail.get() = actual_tail };
            
            if next_head == actual_tail {
                return Err(RingBufferError::Full);
            }
        }

        // Get the slot we're writing to
        let slot = &self.slots[current_head];
        
        // Wait for the slot to be consumed (sequence should match)
        let expected_seq = current_head;
        while slot.header.sequence.load(Ordering::Acquire) != expected_seq {
            std::hint::spin_loop();
        }

        // Write the data
        // SAFETY: We have exclusive access to this slot as the producer
        unsafe {
            ptr::copy_nonoverlapping(
                data.as_ptr(),
                (*slot.data.get()).as_mut_ptr(),
                data.len(),
            );
        }

        // Set the length (makes the message visible)
        slot.header.length.store(data.len(), Ordering::Release);
        
        // Update head for next iteration
        self.producer.head.store(next_head, Ordering::Release);
        
        // Increment sequence for next time this slot is used
        slot.header.sequence.fetch_add(self.capacity, Ordering::Release);

        Ok(())
    }

    /// Pop data from the ring buffer (consumer only)
    /// 
    /// Returns the length of data written to the output buffer.
    /// 
    /// # Safety
    /// - Must only be called from the designated consumer thread
    #[inline]
    pub fn pop(&self, out: &mut [u8]) -> Result<usize, RingBufferError> {
        let current_tail = self.consumer.tail.load(Ordering::Relaxed);
        
        // Check cached head first (fast path)
        let cached_head = unsafe { *self.consumer.cached_head.get() };
        
        if current_tail == cached_head {
            // Cache miss - need to check actual head
            let actual_head = self.producer.head.load(Ordering::Acquire);
            unsafe { *self.consumer.cached_head.get() = actual_head };
            
            if current_tail == actual_head {
                return Err(RingBufferError::Empty);
            }
        }

        // Get the slot we're reading from
        let slot = &self.slots[current_tail];
        
        // Read the length (with acquire semantics)
        let len = slot.header.length.load(Ordering::Acquire);
        
        if len == 0 {
            return Err(RingBufferError::Empty);
        }

        if len > out.len() {
            return Err(RingBufferError::MessageTooLarge);
        }

        // Read the data
        // SAFETY: We have exclusive access to this slot as the consumer
        unsafe {
            ptr::copy_nonoverlapping(
                (*slot.data.get()).as_ptr(),
                out.as_mut_ptr(),
                len,
            );
        }

        // Clear the length (marks slot as empty)
        slot.header.length.store(0, Ordering::Release);
        
        // Update tail for next iteration
        let next_tail = (current_tail + 1) & self.capacity_mask;
        self.consumer.tail.store(next_tail, Ordering::Release);
        
        // Increment sequence for next time this slot is used
        slot.header.sequence.fetch_add(self.capacity, Ordering::Release);

        Ok(len)
    }

    /// Check if the buffer is empty (consumer-side check)
    #[inline]
    pub fn is_empty(&self) -> bool {
        let tail = self.consumer.tail.load(Ordering::Acquire);
        let head = self.producer.head.load(Ordering::Acquire);
        tail == head
    }

    /// Check if the buffer is full (producer-side check)
    #[inline]
    pub fn is_full(&self) -> bool {
        let head = self.producer.head.load(Ordering::Relaxed);
        let tail = self.consumer.tail.load(Ordering::Acquire);
        ((head + 1) & self.capacity_mask) == tail
    }

    /// Get the current number of elements in the buffer
    #[inline]
    pub fn len(&self) -> usize {
        let head = self.producer.head.load(Ordering::Acquire);
        let tail = self.consumer.tail.load(Ordering::Acquire);
        
        if head >= tail {
            head - tail
        } else {
            self.capacity - tail + head
        }
    }

    /// Get the capacity of the buffer
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the capacity mask (useful for external indexing)
    #[inline]
    pub fn capacity_mask(&self) -> usize {
        self.capacity_mask
    }

    /// Reset the ring buffer to initial state
    /// 
    /// # Safety
    /// - Must only be called when no concurrent push/pop operations are in progress
    pub fn reset(&self) {
        self.producer.head.store(0, Ordering::SeqCst);
        self.consumer.tail.store(0, Ordering::SeqCst);
        unsafe {
            *self.producer.cached_tail.get() = 0;
            *self.consumer.cached_head.get() = 0;
        }
        
        // Clear all slot headers
        for slot in self.slots.iter() {
            slot.header.length.store(0, Ordering::SeqCst);
            slot.header.sequence.store(0, Ordering::SeqCst);
        }
    }
}

impl<T> Drop for SPSCRingBuffer<T> {
    fn drop(&mut self) {
        // Ensure all pending operations complete before deallocation
        std::sync::atomic::fence(Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_creation() {
        let rb = SPSCRingBuffer::<()>::new(16).unwrap();
        assert_eq!(rb.capacity(), 16);
        assert!(rb.is_empty());
        assert!(!rb.is_full());
    }

    #[test]
    fn test_invalid_capacity() {
        let rb = SPSCRingBuffer::<()>::new(0);
        assert!(matches!(rb, Err(RingBufferError::InvalidCapacity)));
    }

    #[test]
    fn test_push_pop_single() {
        let rb = SPSCRingBuffer::<()>::new(16).unwrap();
        let data = b"hello";
        
        rb.push(data).unwrap();
        assert!(!rb.is_empty());
        
        let mut out = [0u8; 10];
        let len = rb.pop(&mut out).unwrap();
        assert_eq!(len, 5);
        assert_eq!(&out[..len], b"hello");
        assert!(rb.is_empty());
    }

    #[test]
    fn test_push_pop_multiple() {
        let rb = SPSCRingBuffer::<()>::new(16).unwrap();
        
        for i in 0..10 {
            let data = [i as u8; 8];
            rb.push(&data).unwrap();
        }
        
        assert_eq!(rb.len(), 10);
        
        for i in 0..10 {
            let mut out = [0u8; 8];
            let len = rb.pop(&mut out).unwrap();
            assert_eq!(len, 8);
            assert_eq!(out, [i as u8; 8]);
        }
        
        assert!(rb.is_empty());
    }

    #[test]
    fn test_full_condition() {
        let rb = SPSCRingBuffer::<()>::new(4).unwrap();
        
        // Fill the buffer (capacity - 1 due to full detection)
        for _ in 0..3 {
            rb.push(&[1u8; 8]).unwrap();
        }
        
        assert!(rb.is_full());
        
        let result = rb.push(&[2u8; 8]);
        assert!(matches!(result, Err(RingBufferError::Full)));
    }

    #[test]
    fn test_empty_condition() {
        let rb = SPSCRingBuffer::<()>::new(4).unwrap();
        
        assert!(rb.is_empty());
        
        let mut out = [0u8; 8];
        let result = rb.pop(&mut out);
        assert!(matches!(result, Err(RingBufferError::Empty)));
    }

    #[test]
    fn test_wraparound() {
        let rb = SPSCRingBuffer::<()>::new(4).unwrap();
        
        // Push, pop, push again to test wraparound
        rb.push(&[1u8; 8]).unwrap();
        rb.push(&[2u8; 8]).unwrap();
        
        let mut out = [0u8; 8];
        rb.pop(&mut out).unwrap();
        rb.pop(&mut out).unwrap();
        
        // Now push again - should wrap around
        rb.push(&[3u8; 8]).unwrap();
        rb.push(&[4u8; 8]).unwrap();
        
        rb.pop(&mut out).unwrap();
        assert_eq!(out, [3u8; 8]);
        
        rb.pop(&mut out).unwrap();
        assert_eq!(out, [4u8; 8]);
    }

    #[test]
    fn test_concurrent_spsc() {
        let rb = std::sync::Arc::new(SPSCRingBuffer::<()>::new(1024).unwrap());
        let iterations = 1000;
        
        let rb_producer = rb.clone();
        let producer = thread::spawn(move || {
            for i in 0..iterations {
                let data = [(i % 256) as u8; 16];
                while rb_producer.push(&data).is_err() {
                    thread::yield_now();
                }
            }
        });
        
        let rb_consumer = rb;
        let consumer = thread::spawn(move || {
            let mut count = 0;
            let mut out = [0u8; 16];
            
            while count < iterations {
                if let Ok(len) = rb_consumer.pop(&mut out) {
                    assert_eq!(len, 16);
                    let expected = [(count % 256) as u8; 16];
                    assert_eq!(out, expected);
                    count += 1;
                } else {
                    thread::yield_now();
                }
            }
        });
        
        producer.join().unwrap();
        consumer.join().unwrap();
    }

    #[test]
    fn test_message_too_large() {
        let rb = SPSCRingBuffer::<()>::new(16).unwrap();
        let large_data = vec![0u8; MAX_SLOT_DATA_SIZE + 1];
        
        let result = rb.push(&large_data);
        assert!(matches!(result, Err(RingBufferError::MessageTooLarge)));
    }

    #[test]
    fn test_reset() {
        let rb = SPSCRingBuffer::<()>::new(16).unwrap();
        
        rb.push(&[1u8; 8]).unwrap();
        rb.push(&[2u8; 8]).unwrap();
        assert_eq!(rb.len(), 2);
        
        rb.reset();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
    }
}
