//! Drop Policy Module - Strategies for Handling Overflow
//!
//! This module defines drop policies used throughout the telemetry system
//! to handle buffer overflows and backpressure scenarios.

use std::sync::atomic::{AtomicU64, Ordering};

/// Global drop policy configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropPolicy {
    /// Drop the newest item when buffer is full
    /// Use this when you want to preserve order and oldest data
    DropNewest,
    
    /// Drop the oldest item when buffer is full
    /// Use this when you always want the latest data
    DropOldest,
    
    /// Block until space is available
    /// WARNING: This can backpressure into the trading engine - use with extreme caution
    Block,
    
    /// Return error immediately without dropping
    /// Caller must handle the overflow
    ReturnError,
}

impl Default for DropPolicy {
    fn default() -> Self {
        // Default to DropNewest to protect the trading engine
        Self::DropNewest
    }
}

/// Metrics tracker for drop events
pub struct DropMetrics {
    /// Total items dropped
    dropped_count: AtomicU64,
    /// Total items processed (sent + dropped)
    total_count: AtomicU64,
}

impl DropMetrics {
    pub fn new() -> Self {
        Self {
            dropped_count: AtomicU64::new(0),
            total_count: AtomicU64::new(0),
        }
    }

    /// Record a dropped item
    #[inline]
    pub fn record_drop(&self) {
        self.dropped_count.fetch_add(1, Ordering::Relaxed);
        self.total_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a successfully sent item
    #[inline]
    pub fn record_sent(&self) {
        self.total_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get drop count
    pub fn dropped(&self) -> u64 {
        self.dropped_count.load(Ordering::Relaxed)
    }

    /// Get total count
    pub fn total(&self) -> u64 {
        self.total_count.load(Ordering::Relaxed)
    }

    /// Get drop rate (0.0 to 1.0)
    pub fn drop_rate(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        self.dropped() as f64 / total as f64
    }

    /// Reset metrics
    pub fn reset(&self) {
        self.dropped_count.store(0, Ordering::Relaxed);
        self.total_count.store(0, Ordering::Relaxed);
    }
}

impl Default for DropMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive drop policy that adjusts based on load
pub struct AdaptiveDropPolicy {
    /// Current policy
    current_policy: std::sync::atomic::AtomicU8,
    /// Threshold for switching to more aggressive dropping
    high_load_threshold: f64,
    /// Metrics for adaptation
    metrics: DropMetrics,
}

impl AdaptiveDropPolicy {
    pub fn new(high_load_threshold: f64) -> Self {
        Self {
            current_policy: std::sync::atomic::AtomicU8::new(DropPolicy::DropNewest as u8),
            high_load_threshold,
            metrics: DropMetrics::new(),
        }
    }

    /// Get the current policy based on observed load
    pub fn get_policy(&self) -> DropPolicy {
        let drop_rate = self.metrics.drop_rate();
        
        if drop_rate > self.high_load_threshold {
            // High load - switch to dropping oldest to keep latest data fresh
            DropPolicy::DropOldest
        } else {
            DropPolicy::DropNewest
        }
    }

    /// Record a send attempt result
    pub fn record_result(&self, success: bool) {
        if success {
            self.metrics.record_sent();
        } else {
            self.metrics.record_drop();
        }
    }

    /// Get current drop rate
    pub fn drop_rate(&self) -> f64 {
        self.metrics.drop_rate()
    }

    /// Reset adaptive state
    pub fn reset(&self) {
        self.metrics.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drop_metrics() {
        let metrics = DropMetrics::new();
        
        metrics.record_sent();
        metrics.record_sent();
        metrics.record_drop();
        metrics.record_sent();
        
        assert_eq!(metrics.dropped(), 1);
        assert_eq!(metrics.total(), 4);
        assert!((metrics.drop_rate() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_adaptive_policy() {
        let policy = AdaptiveDropPolicy::new(0.5);
        
        // Initially should be DropNewest
        assert_eq!(policy.get_policy(), DropPolicy::DropNewest);
        
        // Simulate high drop rate
        for _ in 0..10 {
            policy.record_result(false); // All drops
        }
        
        // Should now switch to DropOldest
        assert_eq!(policy.get_policy(), DropPolicy::DropOldest);
    }

    #[test]
    fn test_drop_policy_default() {
        // Verify default protects trading engine
        assert_eq!(DropPolicy::default(), DropPolicy::DropNewest);
    }
}
