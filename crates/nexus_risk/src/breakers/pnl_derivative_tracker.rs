//! PnL Derivative Tracker
//! 
//! Tracks the first and second derivatives of PnL over time to detect
//! acceleration in losses. This provides early warning before the
//! VelocityCircuitBreaker trips.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// Number of samples for derivative calculation
const DERIVATIVE_SAMPLES: usize = 10;

/// Stack-allocated buffer for derivative calculation
#[repr(align(64))]
struct DerivativeBuffer {
    timestamps_ns: [u64; DERIVATIVE_SAMPLES],
    pnl_micro_usd: [i64; DERIVATIVE_SAMPLES],
    head: usize,
    count: usize,
}

impl DerivativeBuffer {
    const fn new() -> Self {
        Self {
            timestamps_ns: [0; DERIVATIVE_SAMPLES],
            pnl_micro_usd: [0; DERIVATIVE_SAMPLES],
            head: 0,
            count: 0,
        }
    }

    #[inline]
    fn push(&mut self, ts: u64, pnl: i64) {
        self.timestamps_ns[self.head] = ts;
        self.pnl_micro_usd[self.head] = pnl;
        self.head = (self.head + 1) % DERIVATIVE_SAMPLES;
        if self.count < DERIVATIVE_SAMPLES {
            self.count += 1;
        }
    }

    /// Calculate first derivative (velocity) - dPnL/dt
    #[inline]
    fn first_derivative(&self) -> i64 {
        if self.count < 2 {
            return 0;
        }

        let newest_idx = if self.head == 0 {
            DERIVATIVE_SAMPLES - 1
        } else {
            self.head - 1
        };
        let oldest_idx = (self.head + DERIVATIVE_SAMPLES - self.count) % DERIVATIVE_SAMPLES;

        let delta_pnl = self.pnl_micro_usd[newest_idx] - self.pnl_micro_usd[oldest_idx];
        let delta_ts = self.timestamps_ns[newest_idx].saturating_sub(self.timestamps_ns[oldest_idx]);

        if delta_ts == 0 {
            return 0;
        }

        // Return micro-USD per ms
        delta_pnl / (delta_ts / 1_000_000) as i64
    }

    /// Calculate second derivative (acceleration) - d²PnL/dt²
    #[inline]
    fn second_derivative(&self) -> i64 {
        if self.count < 3 {
            return 0;
        }

        // Calculate velocity over first half and second half
        let mid_idx = (self.head + DERIVATIVE_SAMPLES - self.count / 2) % DERIVATIVE_SAMPLES;
        let newest_idx = if self.head == 0 {
            DERIVATIVE_SAMPLES - 1
        } else {
            self.head - 1
        };
        let oldest_idx = (self.head + DERIVATIVE_SAMPLES - self.count) % DERIVATIVE_SAMPLES;

        // First half velocity
        let delta_pnl_1 = self.pnl_micro_usd[mid_idx] - self.pnl_micro_usd[oldest_idx];
        let delta_ts_1 = self.timestamps_ns[mid_idx].saturating_sub(self.timestamps_ns[oldest_idx]);
        let vel_1 = if delta_ts_1 > 0 {
            delta_pnl_1 / (delta_ts_1 / 1_000_000) as i64
        } else {
            0
        };

        // Second half velocity
        let delta_pnl_2 = self.pnl_micro_usd[newest_idx] - self.pnl_micro_usd[mid_idx];
        let delta_ts_2 = self.timestamps_ns[newest_idx].saturating_sub(self.timestamps_ns[mid_idx]);
        let vel_2 = if delta_ts_2 > 0 {
            delta_pnl_2 / (delta_ts_2 / 1_000_000) as i64
        } else {
            0
        };

        // Acceleration = change in velocity
        vel_2 - vel_1
    }
}

impl Default for DerivativeBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// PnL Derivative Tracker statistics
#[derive(Debug, Clone)]
pub struct DerivativeStats {
    pub first_derivative_usd_per_ms: f64,
    pub second_derivative_usd_per_ms2: f64,
    pub current_pnl_usd: f64,
}

/// PnL Derivative Tracker
/// 
/// Monitors both velocity (first derivative) and acceleration (second derivative)
/// of PnL changes. Provides early warning of deteriorating conditions.
pub struct PnLDerivativeTracker {
    buffer: parking_lot::Mutex<DerivativeBuffer>,
    last_pnl_micro_usd: AtomicI64,
}

unsafe impl Send for PnLDerivativeTracker {}
unsafe impl Sync for PnLDerivativeTracker {}

impl PnLDerivativeTracker {
    /// Create a new PnL derivative tracker
    pub fn new() -> Self {
        Self {
            buffer: parking_lot::Mutex::new(DerivativeBuffer::new()),
            last_pnl_micro_usd: AtomicI64::new(0),
        }
    }

    /// Update with new PnL reading
    #[inline]
    pub fn update(&self, pnl_usd: f64, timestamp_ns: u64) {
        let pnl_micro_usd = (pnl_usd * 1_000_000.0) as i64;
        self.last_pnl_micro_usd.store(pnl_micro_usd, Ordering::Relaxed);

        let mut buffer = self.buffer.lock();
        buffer.push(timestamp_ns, pnl_micro_usd);
    }

    /// Get current derivative statistics
    pub fn get_stats(&self) -> DerivativeStats {
        let buffer = self.buffer.lock();
        let pnl = self.last_pnl_micro_usd.load(Ordering::Relaxed);

        DerivativeStats {
            first_derivative_usd_per_ms: buffer.first_derivative() as f64 / 1_000_000.0,
            second_derivative_usd_per_ms2: buffer.second_derivative() as f64 / 1_000_000.0,
            current_pnl_usd: pnl as f64 / 1_000_000.0,
        }
    }

    /// Get current PnL in USD
    #[inline]
    pub fn get_pnl_usd(&self) -> f64 {
        self.last_pnl_micro_usd.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }
}

impl Default for PnLDerivativeTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivative_calculation() {
        let tracker = PnLDerivativeTracker::new();

        // Add some samples with consistent loss rate
        for i in 0..10 {
            let pnl = -(i as f64) * 100.0; // -$100/ms
            let ts = i * 1_000_000;
            tracker.update(pnl, ts);
        }

        let stats = tracker.get_stats();
        
        // Should show negative first derivative (losing money)
        assert!(stats.first_derivative_usd_per_ms < 0);
    }

    #[test]
    fn test_acceleration_detection() {
        let tracker = PnLDerivativeTracker::new();

        // Start slow, then accelerate losses
        for i in 0..5 {
            let pnl = -(i as f64) * 50.0; // -$50/ms
            tracker.update(pnl, i * 1_000_000);
        }
        for i in 5..10 {
            let pnl = -250.0 - ((i - 5) as f64) * 200.0; // -$200/ms
            tracker.update(pnl, i * 1_000_000);
        }

        let stats = tracker.get_stats();
        
        // Second derivative should be negative (accelerating losses)
        assert!(stats.second_derivative_usd_per_ms2 < 0);
    }
}
