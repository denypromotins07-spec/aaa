//! Lock-free MPMC order queue for risk evaluation.
//! 
//! Uses atomic compare-and-swap (CAS) operations to enable zero-blocking
//! order submission from the OMS thread while a dedicated risk thread
//! evaluates orders against risk limits.

use crossbeam_utils::atomic::AtomicCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::cell::UnsafeCell;

/// Cache-line padded value to prevent false sharing
#[repr(align(64))]
struct CachePadded<T> {
    value: T,
}

impl<T> CachePadded<T> {
    #[inline]
    fn new(value: T) -> Self {
        Self { value }
    }
}

/// Lock-free ring buffer slot
#[repr(align(64))]
struct RingSlot<T> {
    /// Sequence number for this slot
    sequence: AtomicU64,
    /// Data stored in this slot (uninitialized until written)
    data: UnsafeCell<Option<T>>,
}

impl<T> RingSlot<T> {
    #[inline]
    fn new(seq: u64) -> Self {
        Self {
            sequence: AtomicU64::new(seq),
            data: UnsafeCell::new(None),
        }
    }
}

/// Lock-free MPMC queue for order validation.
/// 
/// This queue allows the OMS thread to submit orders without blocking,
/// while a dedicated risk evaluation thread consumes and validates them.
/// Invalid orders are dropped atomically before reaching the network.
pub struct LockFreeOrderQueue<T> {
    /// Ring buffer storage
    buffer: Vec<RingSlot<T>>,
    /// Buffer size (must be power of 2)
    capacity: usize,
    /// Mask for modulo operation (capacity - 1)
    mask: usize,
    /// Producer sequence (cache-padded)
    head: CachePadded<AtomicU64>,
    /// Consumer sequence (cache-padded)
    tail: CachePadded<AtomicU64>,
    /// Count of items currently in queue
    count: AtomicUsize,
    /// Count of dropped (rejected) orders
    dropped_count: AtomicUsize,
}

unsafe impl<T: Send> Send for LockFreeOrderQueue<T> {}
unsafe impl<T: Send> Sync for LockFreeOrderQueue<T> {}

impl<T> LockFreeOrderQueue<T> {
    /// Create a new lock-free queue with the specified capacity.
    /// Capacity will be rounded up to the next power of 2.
    /// 
    /// # Panics
    /// Panics if capacity is 0 or exceeds 2^30.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be positive");
        assert!(capacity <= 1 << 30, "Capacity too large");
        
        // Round up to power of 2
        let actual_capacity = capacity.next_power_of_two();
        let mask = actual_capacity - 1;
        
        let mut buffer = Vec::with_capacity(actual_capacity);
        for i in 0..actual_capacity {
            buffer.push(RingSlot::new(i as u64));
        }
        
        Self {
            buffer,
            capacity: actual_capacity,
            mask,
            head: CachePadded::new(AtomicU64::new(0)),
            tail: CachePadded::new(AtomicU64::new(0)),
            count: AtomicUsize::new(0),
            dropped_count: AtomicUsize::new(0),
        }
    }

    /// Try to enqueue an item. Returns Ok(()) on success, Err(item) if queue is full.
    /// 
    /// This is a lock-free operation that uses CAS for coordination.
    /// The caller retains ownership of the item if enqueue fails.
    #[inline]
    pub fn try_enqueue(&self, item: T) -> Result<(), T> {
        loop {
            let head = self.head.value.load(Ordering::Relaxed);
            let tail = self.tail.value.load(Ordering::Acquire);
            
            // Check if queue is full
            if head.wrapping_sub(tail) >= self.capacity as u64 {
                return Err(item);
            }
            
            let index = (head as usize) & self.mask;
            let slot = &self.buffer[index];
            let seq = slot.sequence.load(Ordering::Acquire);
            
            // Check if this slot is ready for our sequence number
            match seq.wrapping_sub(head) {
                0 => {
                    // Slot is ready, try to claim it
                    if self.head.value.compare_exchange_weak(
                        head,
                        head.wrapping_add(1),
                        Ordering::SeqCst,
                        Ordering::Relaxed,
                    ).is_ok() {
                        // Successfully claimed the slot, write data
                        unsafe {
                            *slot.data.get() = Some(item);
                        }
                        // Release the slot for consumers
                        slot.sequence.store(head.wrapping_add(1), Ordering::Release);
                        self.count.fetch_add(1, Ordering::Relaxed);
                        return Ok(());
                    }
                    // CAS failed, another producer claimed it, retry
                }
                _ => {
                    // Slot is not ready, another producer is working on it
                    // Yield and retry (in production, might want to spin with backoff)
                    std::hint::spin_loop();
                }
            }
        }
    }

    /// Try to dequeue an item. Returns Ok(Some(item)) on success, 
    /// Ok(None) if queue is empty, Err(dropped_item) if item should be rejected.
    /// 
    /// The consumer can decide to drop invalid items by returning them via Err.
    #[inline]
    pub fn try_dequeue<F>(&self, validator: F) -> Result<Option<T>, T>
    where
        F: FnOnce(&T) -> bool,
    {
        loop {
            let tail = self.tail.value.load(Ordering::Relaxed);
            let head = self.head.value.load(Ordering::Acquire);
            
            // Check if queue is empty
            if tail >= head {
                return Ok(None);
            }
            
            let index = (tail as usize) & self.mask;
            let slot = &self.buffer[index];
            let seq = slot.sequence.load(Ordering::Acquire);
            
            // Check if this slot has data ready
            match seq.wrapping_sub(tail.wrapping_add(1)) {
                0 => {
                    // Data is ready, try to claim the slot
                    if self.tail.value.compare_exchange_weak(
                        tail,
                        tail.wrapping_add(1),
                        Ordering::SeqCst,
                        Ordering::Relaxed,
                    ).is_ok() {
                        // Successfully claimed, read data
                        let item = unsafe { (*slot.data.get()).take() };
                        
                        // Release the slot for producers
                        slot.sequence.store(tail.wrapping_add(self.capacity as u64), Ordering::Release);
                        self.count.fetch_sub(1, Ordering::Relaxed);
                        
                        match item {
                            Some(data) => {
                                if validator(&data) {
                                    return Ok(Some(data));
                                } else {
                                    // Item failed validation, mark as dropped
                                    self.dropped_count.fetch_add(1, Ordering::Relaxed);
                                    return Err(data);
                                }
                            }
                            None => {
                                // Should not happen in correct implementation
                                return Ok(None);
                            }
                        }
                    }
                    // CAS failed, another consumer claimed it, retry
                }
                _ => {
                    // Data not ready yet
                    std::hint::spin_loop();
                }
            }
        }
    }

    /// Get approximate number of items in the queue
    #[inline]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Check if queue is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get count of dropped (rejected) orders
    #[inline]
    pub fn dropped_count(&self) -> usize {
        self.dropped_count.load(Ordering::Relaxed)
    }

    /// Get queue capacity
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::sync::Arc;

    #[test]
    fn test_basic_enqueue_dequeue() {
        let queue = Arc::new(LockFreeOrderQueue::new(16));
        
        // Enqueue some items
        for i in 0..5 {
            assert!(queue.try_enqueue(i).is_ok());
        }
        
        assert_eq!(queue.len(), 5);
        
        // Dequeue and validate
        let mut received = Vec::new();
        while let Ok(Some(item)) = queue.try_dequeue(|_| true) {
            received.push(item);
        }
        
        assert_eq!(received, vec![0, 1, 2, 3, 4]);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_rejection() {
        let queue = Arc::new(LockFreeOrderQueue::new(16));
        
        queue.try_enqueue(10).unwrap();
        queue.try_enqueue(20).unwrap();
        queue.try_enqueue(30).unwrap();
        
        // Reject odd numbers
        let mut accepted = Vec::new();
        let mut rejected = Vec::new();
        
        loop {
            match queue.try_dequeue(|&x| x % 2 == 0) {
                Ok(Some(item)) => accepted.push(item),
                Ok(None) => break,
                Err(item) => rejected.push(item),
            }
        }
        
        assert_eq!(accepted, vec![10, 20]);
        assert_eq!(rejected, vec![30]);
        assert_eq!(queue.dropped_count(), 1);
    }

    #[test]
    fn test_concurrent_producers_consumers() {
        let queue = Arc::new(LockFreeOrderQueue::new(1024));
        let produced = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let consumed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        
        let mut handles = vec![];
        
        // Spawn producer threads
        for _ in 0..4 {
            let q = Arc::clone(&queue);
            let p = Arc::clone(&produced);
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    while q.try_enqueue(i).is_err() {
                        thread::yield_now();
                    }
                    p.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }));
        }
        
        // Spawn consumer threads
        for _ in 0..2 {
            let q = Arc::clone(&queue);
            let c = Arc::clone(&consumed);
            handles.push(thread::spawn(move || {
                loop {
                    match q.try_dequeue(|_| true) {
                        Ok(Some(_)) => {
                            c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        Ok(None) => {
                            if produced.load(std::sync::atomic::Ordering::Relaxed) >= 4000 
                                && q.is_empty() 
                            {
                                break;
                            }
                            thread::yield_now();
                        }
                        Err(_) => unreachable!(),
                    }
                }
            }));
        }
        
        for h in handles {
            let _ = h.join();
        }
        
        assert_eq!(produced.load(std::sync::atomic::Ordering::Relaxed), 4000);
        assert_eq!(consumed.load(std::sync::atomic::Ordering::Relaxed), 4000);
    }
}
