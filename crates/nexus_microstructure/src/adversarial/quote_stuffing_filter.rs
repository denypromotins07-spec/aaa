//! Quote Stuffing Filter using lock-free sliding window.
//! 
//! Detects micro-bursts of order updates that attempt to flood the exchange gateway
//! and increase processing latency.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use ringbuf::{HeapRb, Rb};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum QuoteStuffingError {
    #[error("Invalid window configuration")]
    InvalidWindowConfig,
    #[error("Buffer overflow detected")]
    BufferOverflow,
}

/// Detection result for quote stuffing analysis
#[derive(Debug, Clone)]
pub struct QuoteStuffingAlert {
    pub is_stuffing_detected: bool,
    pub events_per_ms: f64,
    pub burst_ratio: f64,
    pub recommended_action: RecommendedAction,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecommendedAction {
    Normal,
    Throttle,
    PauseUpdates,
    EmergencyStop,
}

/// Lock-free sliding window for event rate monitoring
pub struct QuoteStuffingFilter {
    /// Window size in milliseconds
    window_size_ms: u64,
    /// Maximum allowed events per millisecond
    max_events_per_ms: f64,
    /// Burst detection threshold (ratio of peak to average)
    burst_threshold: f64,
    /// Ring buffer storing event timestamps (nanoseconds)
    event_buffer: parking_lot::Mutex<HeapRb<u64>>,
    /// Current event count in window
    event_count: AtomicUsize,
    /// Last cleanup timestamp
    last_cleanup_ns: AtomicU64,
    /// Peak events observed in any 1ms sub-window
    peak_events_1ms: AtomicUsize,
}

impl QuoteStuffingFilter {
    /// Create a new quote stuffing filter
    pub fn new(
        window_size_ms: u64,
        max_events_per_ms: f64,
        burst_threshold: f64,
        max_buffer_size: usize,
    ) -> Result<Self, QuoteStuffingError> {
        if window_size_ms == 0 || max_events_per_ms <= 0.0 || burst_threshold <= 1.0 {
            return Err(QuoteStuffingError::InvalidWindowConfig);
        }
        
        Ok(Self {
            window_size_ms,
            max_events_per_ms,
            burst_threshold,
            event_buffer: parking_lot::Mutex::new(HeapRb::new(max_buffer_size)),
            event_count: AtomicUsize::new(0),
            last_cleanup_ns: AtomicU64::new(0),
            peak_events_1ms: AtomicUsize::new(0),
        })
    }

    /// Record an incoming order book update event
    #[inline]
    pub fn record_event(&self, timestamp_ns: u64) -> Result<(), QuoteStuffingError> {
        let mut buffer = self.event_buffer.lock();
        
        // Try to push the event timestamp
        match buffer.push(timestamp_ns) {
            Ok(_) => {
                self.event_count.fetch_add(1, Ordering::Relaxed);
                self.update_peak_estimate(timestamp_ns);
                Ok(())
            }
            Err(_) => Err(QuoteStuffingError::BufferOverflow),
        }
    }

    /// Update peak events estimate using sliding 1ms windows
    fn update_peak_estimate(&self, current_ns: u64) {
        let window_start_ns = current_ns.saturating_sub(1_000_000); // 1ms ago
        
        // Count events in the last 1ms
        let buffer = self.event_buffer.lock();
        let mut count_1ms = 0;
        
        // Iterate backwards through buffer
        for i in 0..buffer.len() {
            let idx = (buffer.write_pos() - 1 - i + buffer.len()) % buffer.len();
            if let Some(&ts) = buffer.get(idx) {
                if ts < window_start_ns {
                    break;
                }
                count_1ms += 1;
            }
        }
        
        // Update peak if higher
        let current_peak = self.peak_events_1ms.load(Ordering::Relaxed);
        if count_1ms > current_peak {
            self.peak_events_1ms.store(count_1ms, Ordering::Relaxed);
        }
        
        // Decay peak over time (simple exponential decay approximation)
        if count_1ms < current_peak / 2 {
            self.peak_events_1ms.fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |p| Some((p * 95) / 100), // 5% decay
            ).ok();
        }
    }

    /// Analyze current event rate for quote stuffing patterns
    pub fn analyze(&self, current_time_ns: u64) -> QuoteStuffingAlert {
        let buffer = self.event_buffer.lock();
        let event_count = self.event_count.load(Ordering::Relaxed);
        
        if event_count == 0 {
            return QuoteStuffingAlert {
                is_stuffing_detected: false,
                events_per_ms: 0.0,
                burst_ratio: 0.0,
                recommended_action: RecommendedAction::Normal,
                timestamp_ns: current_time_ns,
            };
        }
        
        // Clean old events outside window
        let window_start_ns = current_time_ns.saturating_sub(self.window_size_ms * 1_000_000);
        let mut valid_count = 0;
        
        for i in 0..buffer.len() {
            let idx = (buffer.write_pos() - 1 - i + buffer.len()) % buffer.len();
            if let Some(&ts) = buffer.get(idx) {
                if ts >= window_start_ns {
                    valid_count += 1;
                }
            }
        }
        
        // Calculate events per millisecond
        let events_per_ms = valid_count as f64 / self.window_size_ms as f64;
        
        // Calculate burst ratio (peak / average)
        let peak = self.peak_events_1ms.load(Ordering::Relaxed) as f64;
        let burst_ratio = if events_per_ms > 0.0 {
            peak / events_per_ms
        } else {
            0.0
        };
        
        // Determine recommended action based on severity
        let recommended_action = if events_per_ms > self.max_events_per_ms * 3.0 || burst_ratio > self.burst_threshold * 2.0 {
            RecommendedAction::EmergencyStop
        } else if events_per_ms > self.max_events_per_ms * 2.0 || burst_ratio > self.burst_threshold * 1.5 {
            RecommendedAction::PauseUpdates
        } else if events_per_ms > self.max_events_per_ms || burst_ratio > self.burst_threshold {
            RecommendedAction::Throttle
        } else {
            RecommendedAction::Normal
        };
        
        let is_stuffing_detected = recommended_action != RecommendedAction::Normal;
        
        QuoteStuffingAlert {
            is_stuffing_detected,
            events_per_ms,
            burst_ratio,
            recommended_action,
            timestamp_ns: current_time_ns,
        }
    }

    /// Reset the filter state
    pub fn reset(&self) {
        self.event_buffer.lock().clear();
        self.event_count.store(0, Ordering::Relaxed);
        self.peak_events_1ms.store(0, Ordering::Relaxed);
        self.last_cleanup_ns.store(0, Ordering::Relaxed);
    }

    /// Get current event rate
    pub fn get_events_per_ms(&self, current_time_ns: u64) -> f64 {
        let alert = self.analyze(current_time_ns);
        alert.events_per_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quote_stuffing_detection() {
        let filter = QuoteStuffingFilter::new(100, 50.0, 3.0, 10000).unwrap();
        
        // Normal traffic
        for i in 0..100 {
            filter.record_event(i * 1_000_000).unwrap(); // 1 event per ms
        }
        
        let alert = filter.analyze(100_000_000);
        assert!(!alert.is_stuffing_detected);
        assert_eq!(alert.recommended_action, RecommendedAction::Normal);
    }

    #[test]
    fn test_burst_detection() {
        let filter = QuoteStuffingFilter::new(100, 10.0, 2.0, 10000).unwrap();
        
        // Burst of events in short time
        for i in 0..500 {
            filter.record_event(i * 1000).unwrap(); // 500 events in 0.5ms
        }
        
        let alert = filter.analyze(1_000_000);
        assert!(alert.is_stuffing_detected);
        assert!(alert.recommended_action != RecommendedAction::Normal);
    }
}
