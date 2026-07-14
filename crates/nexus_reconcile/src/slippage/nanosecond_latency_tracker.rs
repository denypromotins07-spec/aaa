//! Nanosecond Latency Tracker - Monotonic clock-based latency measurement.
//! 
//! CRITICAL: This module uses std::time::Instant (monotonic clock) for all
//! latency measurements, NOT wall-clock time. This eliminates issues with
//! NTP adjustments, leap seconds, and system clock changes.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;
use std::sync::Arc;

/// Tracks latency between two events using monotonic nanosecond clock
pub struct NanosecondLatencyTracker {
    /// Start time reference ( Instant at tracker creation )
    start_time: Instant,
    
    /// Latest measured latency in nanoseconds
    latest_latency_ns: AtomicU64,
    
    /// Minimum observed latency in nanoseconds
    min_latency_ns: AtomicU64,
    
    /// Maximum observed latency in nanoseconds
    max_latency_ns: AtomicU64,
    
    /// Sum of all latencies for average calculation
    sum_latency_ns: AtomicU64,
    
    /// Count of measurements
    measurement_count: AtomicUsize,
}

impl NanosecondLatencyTracker {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            latest_latency_ns: AtomicU64::new(0),
            min_latency_ns: AtomicU64::new(u64::MAX),
            max_latency_ns: AtomicU64::new(0),
            sum_latency_ns: AtomicU64::new(0),
            measurement_count: AtomicUsize::new(0),
        }
    }
    
    /// Record a new latency measurement
    #[inline]
    pub fn record_latency_ns(&self, latency_ns: u64) {
        self.latest_latency_ns.store(latency_ns, Ordering::Relaxed);
        
        // Update min (using compare-exchange loop for lock-free update)
        let mut current_min = self.min_latency_ns.load(Ordering::Relaxed);
        while latency_ns < current_min {
            match self.min_latency_ns.compare_exchange_weak(
                current_min,
                latency_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_current) => current_min = new_current,
            }
        }
        
        // Update max
        let mut current_max = self.max_latency_ns.load(Ordering::Relaxed);
        while latency_ns > current_max {
            match self.max_latency_ns.compare_exchange_weak(
                current_max,
                latency_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_current) => current_max = new_current,
            }
        }
        
        // Update sum and count
        self.sum_latency_ns.fetch_add(latency_ns, Ordering::Relaxed);
        self.measurement_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Mark the start of a latency measurement window
    /// Returns the Instant to be used with mark_end()
    #[inline]
    pub fn mark_start(&self) -> Instant {
        Instant::now()
    }
    
    /// Mark the end of a latency measurement window and record the latency
    /// Returns the measured latency in nanoseconds
    #[inline]
    pub fn mark_end(&self, start: Instant) -> u64 {
        let latency_ns = start.elapsed().as_nanos() as u64;
        self.record_latency_ns(latency_ns);
        latency_ns
    }
    
    /// Get the latest recorded latency in nanoseconds
    #[inline]
    pub fn latest_ns(&self) -> u64 {
        self.latest_latency_ns.load(Ordering::Relaxed)
    }
    
    /// Get the minimum observed latency in nanoseconds
    #[inline]
    pub fn min_ns(&self) -> u64 {
        let min = self.min_latency_ns.load(Ordering::Relaxed);
        if min == u64::MAX { 0 } else { min }
    }
    
    /// Get the maximum observed latency in nanoseconds
    #[inline]
    pub fn max_ns(&self) -> u64 {
        self.max_latency_ns.load(Ordering::Relaxed)
    }
    
    /// Get the average latency in nanoseconds
    #[inline]
    pub fn avg_ns(&self) -> u64 {
        let sum = self.sum_latency_ns.load(Ordering::Relaxed);
        let count = self.measurement_count.load(Ordering::Relaxed);
        if count == 0 { 0 } else { sum / count as u64 }
    }
    
    /// Get the total number of measurements
    #[inline]
    pub fn count(&self) -> usize {
        self.measurement_count.load(Ordering::Relaxed)
    }
    
    /// Reset all statistics
    pub fn reset(&self) {
        self.latest_latency_ns.store(0, Ordering::Relaxed);
        self.min_latency_ns.store(u64::MAX, Ordering::Relaxed);
        self.max_latency_ns.store(0, Ordering::Relaxed);
        self.sum_latency_ns.store(0, Ordering::Relaxed);
        self.measurement_count.store(0, Ordering::Relaxed);
    }
    
    /// Get a snapshot of all statistics
    pub fn get_stats(&self) -> LatencyStats {
        LatencyStats {
            latest_ns: self.latest_ns(),
            min_ns: self.min_ns(),
            max_ns: self.max_ns(),
            avg_ns: self.avg_ns(),
            count: self.count(),
        }
    }
}

impl Default for NanosecondLatencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LatencyStats {
    pub latest_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub avg_ns: u64,
    pub count: usize,
}

impl LatencyStats {
    /// Convert to microseconds for easier reading
    pub fn to_micros(&self) -> LatencyStatsMicros {
        LatencyStatsMicros {
            latest_us: self.latest_ns / 1000,
            min_us: self.min_ns / 1000,
            max_us: self.max_ns / 1000,
            avg_us: self.avg_ns / 1000,
            count: self.count,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LatencyStatsMicros {
    pub latest_us: u64,
    pub min_us: u64,
    pub max_us: u64,
    pub avg_us: u64,
    pub count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_latency_measurement_accuracy() {
        let tracker = NanosecondLatencyTracker::new();
        
        // Measure a known sleep duration
        let start = tracker.mark_start();
        thread::sleep(Duration::from_millis(10));
        let latency_ns = tracker.mark_end(start);
        
        // Should be at least 10ms (10,000,000 ns), allow some overhead
        assert!(latency_ns >= 10_000_000, "Latency {}ns should be >= 10ms", latency_ns);
        assert!(latency_ns < 100_000_000, "Latency {}ns should be < 100ms", latency_ns);
        
        assert_eq!(tracker.count(), 1);
        assert_eq!(tracker.latest_ns(), latency_ns);
        assert_eq!(tracker.min_ns(), latency_ns);
        assert_eq!(tracker.max_ns(), latency_ns);
        assert_eq!(tracker.avg_ns(), latency_ns);
    }
    
    #[test]
    fn test_min_max_tracking() {
        let tracker = NanosecondLatencyTracker::new();
        
        // Record varying latencies
        tracker.record_latency_ns(1000);
        tracker.record_latency_ns(5000);
        tracker.record_latency_ns(2000);
        tracker.record_latency_ns(10000);
        tracker.record_latency_ns(500);
        
        assert_eq!(tracker.min_ns(), 500);
        assert_eq!(tracker.max_ns(), 10000);
        assert_eq!(tracker.avg_ns(), 3700);  // (1000+5000+2000+10000+500)/5
        assert_eq!(tracker.count(), 5);
    }
    
    #[test]
    fn test_monotonic_clock_stability() {
        let tracker = NanosecondLatencyTracker::new();
        
        // Take multiple measurements in quick succession
        let mut latencies = Vec::new();
        for _ in 0..100 {
            let start = tracker.mark_start();
            // Tiny operation
            let _x = 1 + 1;
            latencies.push(tracker.mark_end(start));
        }
        
        // All latencies should be positive and reasonable (< 1ms for simple add)
        for &lat in &latencies {
            assert!(lat > 0, "Latency should be positive");
            assert!(lat < 1_000_000, "Latency {}ns should be < 1ms", lat);
        }
    }
}
