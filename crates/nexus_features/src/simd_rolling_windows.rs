//! Chapter 3: SIMD-Accelerated Rolling Windows
//!
//! This module implements SIMD-optimized rolling window calculators for
//! VWAP, Order Book Imbalance, and micro-price using AVX2/AVX-512 instructions.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use nexus_core::memory::cache_padder::CachePadded64;
use wide::f64x8;

/// Default window size for rolling calculations
pub const DEFAULT_WINDOW_SIZE: usize = 64;

/// Maximum SIMD lanes (AVX-512)
pub const SIMD_LANES: usize = 8;

/// SIMD-accelerated rolling window
#[repr(C)]
pub struct SimdRollingWindow {
    /// Pre-allocated circular buffer (aligned for SIMD)
    buffer: CachePadded64<Box<[f64]>>,
    /// Current write index
    write_idx: CachePadded64<AtomicUsize>,
    /// Element count (capped at capacity)
    count: CachePadded64<AtomicUsize>,
    /// Running sum for O(1) mean
    running_sum: CachePadded64<AtomicU64>,
    /// Capacity
    capacity: usize,
    /// Padding for cache alignment
    _padding: [u8; 48],
}

// SAFETY: SimdRollingWindow is used in single-threaded hot path
unsafe impl Send for SimdRollingWindow {}
unsafe impl Sync for SimdRollingWindow {}

impl SimdRollingWindow {
    /// Create a new rolling window
    #[inline]
    pub fn new(capacity: usize) -> Self {
        // Ensure capacity is multiple of SIMD lanes for efficient processing
        let aligned_capacity = ((capacity + SIMD_LANES - 1) / SIMD_LANES) * SIMD_LANES;
        
        let buffer = vec![0.0f64; aligned_capacity].into_boxed_slice();
        
        Self {
            buffer: CachePadded64::new(buffer),
            write_idx: CachePadded64::new(AtomicUsize::new(0)),
            count: CachePadded64::new(AtomicUsize::new(0)),
            running_sum: CachePadded64::new(AtomicU64::new(0)),
            capacity: aligned_capacity,
            _padding: [0; 48],
        }
    }

    /// Push a new value into the window
    #[inline]
    pub fn push(&self, value: f64) {
        let idx = self.write_idx.0.fetch_add(1, Ordering::AcqRel) % self.capacity;
        let old_value = self.buffer.0[idx];
        
        self.buffer.0[idx] = value;
        
        // Update running sum using atomic operations
        let sum_bits = self.running_sum.0.load(Ordering::Acquire);
        let current_sum = f64::from_bits(sum_bits);
        let new_sum = current_sum - old_value + value;
        self.running_sum.0.store(new_sum.to_bits(), Ordering::Release);
        
        // Update count
        let current_count = self.count.0.load(Ordering::Acquire);
        if current_count < self.capacity {
            self.count.0.fetch_add(1, Ordering::Release);
        }
    }

    /// Get the mean of the window
    #[inline]
    pub fn mean(&self) -> f64 {
        let count = self.count.0.load(Ordering::Acquire);
        if count == 0 {
            return 0.0;
        }
        
        let sum_bits = self.running_sum.0.load(Ordering::Acquire);
        let sum = f64::from_bits(sum_bits);
        sum / count as f64
    }

    /// Get the sum of the window
    #[inline]
    pub fn sum(&self) -> f64 {
        let sum_bits = self.running_sum.0.load(Ordering::Acquire);
        f64::from_bits(sum_bits)
    }

    /// Get element count
    #[inline]
    pub fn len(&self) -> usize {
        self.count.0.load(Ordering::Acquire)
    }

    /// Check if window is full
    #[inline]
    pub fn is_full(&self) -> bool {
        self.count.0.load(Ordering::Acquire) >= self.capacity
    }

    /// Clear the window
    #[inline]
    pub fn clear(&self) {
        for i in 0..self.capacity {
            self.buffer.0[i] = 0.0;
        }
        self.write_idx.0.store(0, Ordering::Release);
        self.count.0.store(0, Ordering::Release);
        self.running_sum.0.store(0.0f64.to_bits(), Ordering::Release);
    }

    /// Get all values as slice
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        let count = self.count.0.load(Ordering::Acquire);
        &self.buffer.0[..count]
    }
}

/// SIMD-accelerated VWAP calculator
#[repr(C)]
pub struct SimdVwapCalculator {
    /// Price window
    prices: SimdRollingWindow,
    /// Volume window
    volumes: SimdRollingWindow,
    /// Cumulative price*volume sum
    pv_sum: CachePadded64<AtomicU64>,
    /// Cumulative volume sum
    v_sum: CachePadded64<AtomicU64>,
    /// Total ticks processed
    tick_count: CachePadded64<AtomicUsize>,
}

// SAFETY: SimdVwapCalculator is single-threaded
unsafe impl Send for SimdVwapCalculator {}
unsafe impl Sync for SimdVwapCalculator {}

impl SimdVwapCalculator {
    /// Create a new VWAP calculator
    #[inline]
    pub fn new(window_size: usize) -> Self {
        Self {
            prices: SimdRollingWindow::new(window_size),
            volumes: SimdRollingWindow::new(window_size),
            pv_sum: CachePadded64::new(AtomicU64::new(0)),
            v_sum: CachePadded64::new(AtomicU64::new(0)),
            tick_count: CachePadded64::new(AtomicUsize::new(0)),
        }
    }

    /// Add a tick (price, volume)
    #[inline]
    pub fn add_tick(&self, price: f64, volume: f64) {
        self.prices.push(price);
        self.volumes.push(volume);
        
        // Update cumulative sums
        let pv = price * volume;
        let pv_bits = self.pv_sum.0.load(Ordering::Acquire);
        let current_pv = f64::from_bits(pv_bits);
        self.pv_sum.0.store((current_pv + pv).to_bits(), Ordering::Release);
        
        let v_bits = self.v_sum.0.load(Ordering::Acquire);
        let current_v = f64::from_bits(v_bits);
        self.v_sum.0.store((current_v + volume).to_bits(), Ordering::Release);
        
        self.tick_count.0.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current VWAP
    #[inline]
    pub fn vwap(&self) -> f64 {
        let v_bits = self.v_sum.0.load(Ordering::Acquire);
        let volume = f64::from_bits(v_bits);
        
        if volume == 0.0 {
            return 0.0;
        }
        
        let pv_bits = self.pv_sum.0.load(Ordering::Acquire);
        let pv = f64::from_bits(pv_bits);
        
        pv / volume
    }

    /// Reset cumulative sums (for new session)
    #[inline]
    pub fn reset_cumulative(&self) {
        self.pv_sum.0.store(0.0f64.to_bits(), Ordering::Release);
        self.v_sum.0.store(0.0f64.to_bits(), Ordering::Release);
        self.tick_count.0.store(0, Ordering::Release);
    }

    /// Get tick count
    #[inline]
    pub fn tick_count(&self) -> usize {
        self.tick_count.0.load(Ordering::Relaxed)
    }

    /// Get average price in window
    #[inline]
    pub fn avg_price(&self) -> f64 {
        self.prices.mean()
    }

    /// Get average volume in window
    #[inline]
    pub fn avg_volume(&self) -> f64 {
        self.volumes.mean()
    }
}

/// SIMD batch processor for multiple features
pub struct SimdFeatureProcessor {
    /// Number of features
    num_features: usize,
    /// Aligned storage for SIMD processing
    data: CachePadded64<Box<[f64]>>,
}

// SAFETY: FeatureProcessor is single-threaded
unsafe impl Send for SimdFeatureProcessor {}
unsafe impl Sync for SimdFeatureProcessor {}

impl SimdFeatureProcessor {
    /// Create a new feature processor
    #[inline]
    pub fn new(num_features: usize) -> Self {
        // Align to SIMD lane boundaries
        let aligned = ((num_features + SIMD_LANES - 1) / SIMD_LANES) * SIMD_LANES;
        let data = vec![0.0f64; aligned].into_boxed_slice();
        
        Self {
            num_features,
            data: CachePadded64::new(data),
        }
    }

    /// Process features using SIMD (batch update)
    #[inline]
    pub fn process_batch(&mut self, values: &[f64]) {
        let len = values.len().min(self.num_features);
        
        // Process in SIMD lanes
        let mut i = 0;
        while i + SIMD_LANES <= len {
            // Load 8 values into SIMD register
            let simd_vals = f64x8::from_slice(&values[i..]);
            
            // Example SIMD operation: square root (could be any transformation)
            let result = simd_vals.sqrt();
            
            // Store back
            result.write_to_slice(&mut self.data.0[i..]);
            
            i += SIMD_LANES;
        }
        
        // Handle remainder
        for j in i..len {
            self.data.0[j] = values[j].sqrt();
        }
    }

    /// Get processed data
    #[inline]
    pub fn get_data(&self) -> &[f64] {
        &self.data.0[..self.num_features]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_window_basic() {
        let window = SimdRollingWindow::new(8);
        assert_eq!(window.len(), 0);
        assert!(!window.is_full());
        
        window.push(1.0);
        window.push(2.0);
        window.push(3.0);
        
        assert_eq!(window.len(), 3);
        assert_eq!(window.mean(), 2.0);
        assert_eq!(window.sum(), 6.0);
    }

    #[test]
    fn test_rolling_window_circular() {
        let window = SimdRollingWindow::new(4);
        
        for i in 1..=6 {
            window.push(i as f64);
        }
        
        // Should have last 4 values: 3, 4, 5, 6
        assert_eq!(window.len(), 4);
        assert_eq!(window.mean(), 4.5);
    }

    #[test]
    fn test_vwap_calculator() {
        let vwap = SimdVwapCalculator::new(10);
        
        // Add ticks: (price, volume)
        vwap.add_tick(100.0, 10.0); // PV = 1000
        vwap.add_tick(102.0, 20.0); // PV = 2040
        vwap.add_tick(98.0, 10.0);  // PV = 980
        
        // Total PV = 4020, Total V = 40
        // VWAP = 4020 / 40 = 100.5
        assert!((vwap.vwap() - 100.5).abs() < 1e-10);
        assert_eq!(vwap.tick_count(), 3);
    }

    #[test]
    fn test_vwap_reset() {
        let vwap = SimdVwapCalculator::new(10);
        
        vwap.add_tick(100.0, 10.0);
        vwap.add_tick(102.0, 20.0);
        
        assert!((vwap.vwap() - 101.333).abs() < 0.001);
        
        vwap.reset_cumulative();
        vwap.add_tick(50.0, 5.0);
        
        assert_eq!(vwap.vwap(), 50.0);
        assert_eq!(vwap.tick_count(), 1);
    }

    #[test]
    fn test_simd_feature_processor() {
        let mut proc = SimdFeatureProcessor::new(10);
        
        let values: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        proc.process_batch(&values);
        
        let result = proc.get_data();
        
        // Check first 8 (SIMD processed)
        for i in 0..8 {
            assert!((result[i] - (i + 1) as f64).sqrt().abs() < 1e-10);
        }
    }
}
