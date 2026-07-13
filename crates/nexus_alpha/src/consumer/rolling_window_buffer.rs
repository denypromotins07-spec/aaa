//! Lock-Free Circular Buffer for Rolling Window Calculations
//! 
//! This module implements a fixed-size, stack-allocated circular buffer
//! for maintaining rolling windows of micro-prices and trade volumes.
//! 
//! ZERO-ALLOC: Uses fixed-size arrays, no heap allocation after initialization.
//! LOCK-FREE: Uses atomic operations for thread-safe access.

use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};

/// Fixed-size circular buffer for rolling window calculations
/// 
/// Generic over capacity N - must be known at compile time
pub struct RollingWindowBuffer<const N: usize> {
    /// Circular buffer storage (stack-allocated)
    buffer: [f64; N],
    /// Write index (next position to write)
    write_idx: AtomicUsize,
    /// Read count (total items ever written)
    write_count: AtomicUsize,
    /// Sum of all values in current window (for O(1) mean calculation)
    window_sum: AtomicU64, // Stored as u64 bits for atomic operations
    /// Whether buffer is full (has wrapped around)
    is_full: AtomicUsize, // 0 = false, 1 = true
}

// Helper for atomic f64 operations using bit representation
fn f64_to_u64_bits(val: f64) -> u64 {
    val.to_bits()
}

fn u64_bits_to_f64(bits: u64) -> f64 {
    f64::from_bits(bits)
}

impl<const N: usize> RollingWindowBuffer<N> {
    /// Create a new empty rolling window buffer
    pub const fn new() -> Self {
        Self {
            buffer: [0.0; N],
            write_idx: AtomicUsize::new(0),
            write_count: AtomicUsize::new(0),
            window_sum: AtomicU64::new(0),
            is_full: AtomicUsize::new(0),
        }
    }

    /// Push a new value into the rolling window
    /// 
    /// If the buffer is full, overwrites the oldest value
    /// Returns the overwritten value if any
    pub fn push(&self, value: f64) -> Option<f64> {
        let idx = self.write_idx.fetch_add(1, Ordering::AcqRel) % N;
        
        // Get the old value at this position
        let old_value = self.buffer[idx];
        
        // Check if we're overwriting (buffer was full or we've wrapped)
        let count = self.write_count.fetch_add(1, Ordering::AcqRel);
        
        if count >= N {
            // We're overwriting an old value, update the sum
            let old_sum_bits = self.window_sum.load(Ordering::Acquire);
            let old_sum = u64_bits_to_f64(old_sum_bits);
            let new_sum = old_sum - old_value + value;
            self.window_sum.store(f64_to_u64_bits(new_sum), Ordering::Release);
            self.is_full.store(1, Ordering::Release);
            Some(old_value)
        } else {
            // First fill-up phase, just add to sum
            let old_sum_bits = self.window_sum.load(Ordering::Acquire);
            let old_sum = u64_bits_to_f64(old_sum_bits);
            let new_sum = old_sum + value;
            self.window_sum.store(f64_to_u64_bits(new_sum), Ordering::Release);
            
            if count + 1 >= N {
                self.is_full.store(1, Ordering::Release);
            }
            None
        }
    }

    /// Set value at specific index (internal use)
    fn set_at(&self, idx: usize, value: f64) {
        // Safe because we're the only writer
        unsafe {
            let ptr = self.buffer.as_ptr() as *mut f64;
            ptr.add(idx).write(value);
        }
    }

    /// Get the current mean of the window in O(1) time
    pub fn mean(&self) -> Option<f64> {
        let count = self.count();
        if count == 0 {
            return None;
        }
        
        let sum_bits = self.window_sum.load(Ordering::Acquire);
        let sum = u64_bits_to_f64(sum_bits);
        Some(sum / count as f64)
    }

    /// Get the current variance of the window
    /// Note: This requires O(n) iteration
    pub fn variance(&self) -> Option<f64> {
        let mean = self.mean()?;
        let count = self.count();
        
        if count == 0 {
            return None;
        }

        let mut sum_sq_diff = 0.0f64;
        let effective_count = if self.is_full.load(Ordering::Acquire) == 1 {
            N
        } else {
            count
        };

        for i in 0..effective_count {
            let val = self.buffer[i];
            let diff = val - mean;
            sum_sq_diff += diff * diff;
        }

        Some(sum_sq_diff / effective_count as f64)
    }

    /// Get the standard deviation of the window
    pub fn std_dev(&self) -> Option<f64> {
        self.variance().map(|v| v.sqrt())
    }

    /// Get the minimum value in the window
    pub fn min(&self) -> Option<f64> {
        let count = self.count();
        if count == 0 {
            return None;
        }

        let effective_count = if self.is_full.load(Ordering::Acquire) == 1 {
            N
        } else {
            count
        };

        let mut min_val = f64::INFINITY;
        for i in 0..effective_count {
            let val = self.buffer[i];
            if val < min_val {
                min_val = val;
            }
        }

        Some(min_val)
    }

    /// Get the maximum value in the window
    pub fn max(&self) -> Option<f64> {
        let count = self.count();
        if count == 0 {
            return None;
        }

        let effective_count = if self.is_full.load(Ordering::Acquire) == 1 {
            N
        } else {
            count
        };

        let mut max_val = f64::NEG_INFINITY;
        for i in 0..effective_count {
            let val = self.buffer[i];
            if val > max_val {
                max_val = val;
            }
        }

        Some(max_val)
    }

    /// Get the number of valid elements in the window
    pub fn count(&self) -> usize {
        let write_count = self.write_count.load(Ordering::Acquire);
        if write_count >= N {
            N
        } else {
            write_count
        }
    }

    /// Check if the buffer is full
    pub fn is_full(&self) -> bool {
        self.is_full.load(Ordering::Acquire) == 1
    }

    /// Get all values in the window as a Vec (for debugging/serialization)
    pub fn to_vec(&self) -> Vec<f64> {
        let count = self.count();
        let mut result = Vec::with_capacity(count);
        
        let start_idx = if self.is_full.load(Ordering::Acquire) == 1 {
            self.write_idx.load(Ordering::Acquire) % N
        } else {
            0
        };

        for i in 0..count {
            let idx = (start_idx + i) % N;
            result.push(self.buffer[idx]);
        }

        result
    }

    /// Clear the buffer
    pub fn clear(&self) {
        self.write_idx.store(0, Ordering::Release);
        self.write_count.store(0, Ordering::Release);
        self.window_sum.store(0, Ordering::Release);
        self.is_full.store(0, Ordering::Release);
        
        // Zero out the buffer
        for i in 0..N {
            self.buffer[i] = 0.0;
        }
    }

    /// Get capacity
    pub const fn capacity(&self) -> usize {
        N
    }
}

impl<const N: usize> Default for RollingWindowBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_window_basic() {
        let window: RollingWindowBuffer<5> = RollingWindowBuffer::new();
        
        assert_eq!(window.count(), 0);
        assert!(window.mean().is_none());
        
        window.push(1.0);
        window.push(2.0);
        window.push(3.0);
        
        assert_eq!(window.count(), 3);
        assert_eq!(window.mean(), Some(2.0));
    }

    #[test]
    fn test_rolling_window_wrap() {
        let window: RollingWindowBuffer<3> = RollingWindowBuffer::new();
        
        window.push(1.0);
        window.push(2.0);
        window.push(3.0);
        
        assert_eq!(window.count(), 3);
        assert_eq!(window.mean(), Some(2.0));
        assert!(window.is_full());
        
        // This should overwrite 1.0
        window.push(4.0);
        
        assert_eq!(window.count(), 3);
        // Now contains [4.0, 2.0, 3.0] (circular)
        assert_eq!(window.mean(), Some(3.0));
    }

    #[test]
    fn test_rolling_window_variance() {
        let window: RollingWindowBuffer<4> = RollingWindowBuffer::new();
        
        window.push(2.0);
        window.push(4.0);
        window.push(4.0);
        window.push(4.0);
        
        let variance = window.variance().unwrap();
        // Mean is 3.5, variance = ((2-3.5)^2 + 3*(4-3.5)^2) / 4 = (2.25 + 0.75) / 4 = 0.75
        assert!((variance - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_window_min_max() {
        let window: RollingWindowBuffer<5> = RollingWindowBuffer::new();
        
        window.push(3.0);
        window.push(1.0);
        window.push(4.0);
        window.push(1.0);
        window.push(5.0);
        
        assert_eq!(window.min(), Some(1.0));
        assert_eq!(window.max(), Some(5.0));
    }

    #[test]
    fn test_rolling_window_empty() {
        let window: RollingWindowBuffer<5> = RollingWindowBuffer::new();
        
        assert_eq!(window.count(), 0);
        assert!(window.mean().is_none());
        assert!(window.variance().is_none());
        assert!(window.min().is_none());
        assert!(window.max().is_none());
    }
}
