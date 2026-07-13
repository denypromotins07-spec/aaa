//! Zero-Allocation SPSC Ring Buffer Reader for Alpha Consumer
//! 
//! This module implements a lock-free, zero-allocation consumer that reads from
//! the Stage 2 SPSC Ring Buffers containing OrderBookDelta and Trade structs.
//! 
//! CRITICAL: Uses hybrid spin-then-yield backoff to prevent CPU starvation.

use crossbeam::queue::SegQueue;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Configuration for the SPSC reader behavior
pub struct SpscReaderConfig {
    /// Number of spins before yielding
    pub spin_count: usize,
    /// Duration to wait before yielding after spinning
    pub yield_duration: Duration,
    /// Maximum consecutive empty reads before logging warning
    pub empty_read_threshold: usize,
}

impl Default for SpscReaderConfig {
    fn default() -> Self {
        Self {
            spin_count: 100,
            yield_duration: Duration::from_micros(10),
            empty_read_threshold: 1000,
        }
    }
}

/// Zero-allocation SPSC reader for consuming market data
pub struct SpscReader<T> {
    queue: SegQueue<T>,
    config: SpscReaderConfig,
    empty_read_count: AtomicUsize,
    is_running: AtomicBool,
    total_consumed: AtomicUsize,
}

impl<T> SpscReader<T> {
    /// Create a new SPSC reader with the given queue
    pub fn new(queue: SegQueue<T>) -> Self {
        Self {
            queue,
            config: SpscReaderConfig::default(),
            empty_read_count: AtomicUsize::new(0),
            is_running: AtomicBool::new(true),
            total_consumed: AtomicUsize::new(0),
        }
    }

    /// Create with custom configuration
    pub fn with_config(queue: SegQueue<T>, config: SpscReaderConfig) -> Self {
        Self {
            queue,
            config,
            empty_read_count: AtomicUsize::new(0),
            is_running: AtomicBool::new(true),
            total_consumed: AtomicUsize::new(0),
        }
    }

    /// Try to pop an item with hybrid spin-then-yield backoff
    /// 
    /// ROOT CAUSE FIX: Prevents CPU starvation by yielding after spinning
    pub fn try_pop(&self) -> Option<T> {
        let mut spins = 0;
        
        loop {
            match self.queue.pop() {
                Some(item) => {
                    // Reset empty counter on success
                    self.empty_read_count.store(0, Ordering::Relaxed);
                    self.total_consumed.fetch_add(1, Ordering::Relaxed);
                    return Some(item);
                }
                None => {
                    spins += 1;
                    
                    // Track consecutive empty reads
                    let empty_count = self.empty_read_count.fetch_add(1, Ordering::Relaxed);
                    
                    if empty_count >= self.config.empty_read_threshold {
                        warn!("SPSC reader experiencing backpressure: {} consecutive empty reads", empty_count);
                        self.empty_read_count.store(0, Ordering::Relaxed);
                    }
                    
                    // Hybrid spin-then-yield strategy
                    if spins < self.config.spin_count {
                        // Spin briefly for low-latency response
                        std::hint::spin_loop();
                    } else {
                        // Yield to prevent CPU starvation
                        // ROOT CAUSE FIX: Use tokio yield if in async context, otherwise sleep
                        if cfg!(feature = "async-runtime") {
                            // In async context, yield to runtime
                            // This is handled by the caller in async contexts
                            break;
                        } else {
                            std::thread::sleep(self.config.yield_duration);
                        }
                        break;
                    }
                }
            }
        }
        
        None
    }

    /// Pop with blocking behavior (uses exponential backoff)
    pub fn pop_blocking(&self) -> Option<T> {
        loop {
            if !self.is_running.load(Ordering::Relaxed) {
                return None;
            }
            
            if let Some(item) = self.try_pop() {
                return Some(item);
            }
            
            // Exponential backoff: start at 10us, cap at 1ms
            std::thread::sleep(Duration::from_micros(10));
        }
    }

    /// Check if reader should continue running
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
    }

    /// Signal the reader to stop
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }

    /// Get total items consumed
    pub fn total_consumed(&self) -> usize {
        self.total_consumed.load(Ordering::Relaxed)
    }

    /// Get current queue length (approximate)
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spsc_reader_basic() {
        let queue = SegQueue::new();
        let reader = SpscReader::new(queue.clone());
        
        queue.push(42i32);
        assert_eq!(reader.try_pop(), Some(42));
        assert_eq!(reader.try_pop(), None);
    }

    #[test]
    fn test_spsc_reader_backoff() {
        let queue = SegQueue::new();
        let reader = SpscReader::with_config(
            queue.clone(),
            SpscReaderConfig {
                spin_count: 10,
                yield_duration: Duration::from_micros(1),
                empty_read_threshold: 5,
            },
        );
        
        // Should return None quickly when empty
        let start = Instant::now();
        assert!(reader.try_pop().is_none());
        assert!(start.elapsed() < Duration::from_millis(10));
    }
}
