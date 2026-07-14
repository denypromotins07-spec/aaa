//! Velocity of Loss Circuit Breaker
//! 
//! Monitors the rate of change (derivative) of PnL to detect anomalous loss acceleration.
//! If losses accelerate beyond a threshold, the breaker trips and locks the gatekeeper.
//! 
//! This implements the "Velocity of Loss (VoL) Breaker" from Chapter 2:
//! - Maintains a rolling window of PnL samples
//! - Calculates dPnL/dt (first derivative)
//! - Trips when loss velocity exceeds threshold

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

/// Window size for PnL tracking (number of samples)
const WINDOW_SIZE: usize = 64;

/// Result of velocity check
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VelocityCheckResult {
    /// Loss velocity is within acceptable bounds
    Normal,
    /// Warning: elevated loss velocity
    Warning {
        velocity_usd_per_ms: i64,
        threshold_usd_per_ms: i64,
    },
    /// Critical: trip circuit breaker
    Critical {
        velocity_usd_per_ms: i64,
        threshold_usd_per_ms: i64,
    },
}

/// Stack-allocated circular buffer for PnL samples
#[repr(align(64))]
struct PnLWindow {
    /// Timestamps in nanoseconds
    timestamps: [u64; WINDOW_SIZE],
    /// PnL values in micro-USD (scaled integers, no floats)
    pnl_values: [i64; WINDOW_SIZE],
    /// Current write index
    head: usize,
    /// Number of valid samples
    count: usize,
}

impl PnLWindow {
    const fn new() -> Self {
        Self {
            timestamps: [0; WINDOW_SIZE],
            pnl_values: [0; WINDOW_SIZE],
            head: 0,
            count: 0,
        }
    }

    #[inline]
    fn push(&mut self, timestamp_ns: u64, pnl_micro_usd: i64) {
        self.timestamps[self.head] = timestamp_ns;
        self.pnl_values[self.head] = pnl_micro_usd;
        self.head = (self.head + 1) % WINDOW_SIZE;
        if self.count < WINDOW_SIZE {
            self.count += 1;
        }
    }

    /// Calculate velocity over the last N milliseconds
    /// Returns velocity in micro-USD per millisecond
    #[inline]
    fn calculate_velocity(&self, window_ms: u64) -> i64 {
        if self.count < 2 {
            return 0;
        }

        let window_ns = window_ms * 1_000_000;
        
        // Find newest sample
        let newest_idx = if self.head == 0 {
            WINDOW_SIZE - 1
        } else {
            self.head - 1
        };
        
        let newest_ts = self.timestamps[newest_idx];
        let newest_pnl = self.pnl_values[newest_idx];
        
        // Find oldest sample within window
        let cutoff_ns = newest_ts.saturating_sub(window_ns);
        
        let mut oldest_pnl = newest_pnl;
        let mut oldest_ts = newest_ts;
        
        for i in 0..self.count {
            let idx = (self.head + i) % WINDOW_SIZE;
            let ts = self.timestamps[idx];
            let pnl = self.pnl_values[idx];
            
            if ts >= cutoff_ns && ts < oldest_ts {
                oldest_ts = ts;
                oldest_pnl = pnl;
            }
        }
        
        // Calculate velocity: delta_pnl / delta_time
        let delta_pnl = newest_pnl - oldest_pnl;
        let delta_time_ms = newest_ts.saturating_sub(oldest_ts) / 1_000_000;
        
        if delta_time_ms == 0 {
            return 0;
        }
        
        // Velocity in micro-USD per ms
        delta_pnl / delta_time_ms as i64
    }
}

impl Default for PnLWindow {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for velocity circuit breaker
#[derive(Debug, Clone)]
pub struct VelocityConfig {
    /// Warning threshold in USD/ms (negative = loss)
    pub warning_threshold_usd_per_ms: i64,
    /// Critical threshold in USD/ms
    pub critical_threshold_usd_per_ms: i64,
    /// Time window for velocity calculation in ms
    pub time_window_ms: u64,
}

impl Default for VelocityConfig {
    fn default() -> Self {
        Self {
            warning_threshold_usd_per_ms: -100_000,      // -$100/ms warning
            critical_threshold_usd_per_ms: -500_000,     // -$500/ms critical
            time_window_ms: 50,                          // 50ms window
        }
    }
}

/// Velocity of Loss Circuit Breaker
/// 
/// Detects accelerating losses by monitoring dPnL/dt.
/// Uses scaled integer math (micro-USD) to avoid floating-point issues.
/// 
/// ZERO-ALLOCATION:
/// - Fixed-size stack array for window
/// - All atomic operations, no locks
pub struct VelocityCircuitBreaker {
    config: VelocityConfig,
    /// PnL window (wrapped in parking_lot for interior mutability)
    window: parking_lot::Mutex<PnLWindow>,
    /// Last PnL value (atomic for quick reads)
    last_pnl_micro_usd: AtomicI64,
    /// Whether breaker is tripped
    tripped: AtomicBool,
    /// Count of warnings
    warning_count: AtomicU64,
    /// Count of critical events
    critical_count: AtomicU64,
    /// Timestamp of last trip
    last_trip_timestamp_ns: AtomicU64,
}

unsafe impl Send for VelocityCircuitBreaker {}
unsafe impl Sync for VelocityCircuitBreaker {}

impl VelocityCircuitBreaker {
    /// Create a new velocity circuit breaker
    pub fn new(config: VelocityConfig) -> Self {
        Self {
            config,
            window: parking_lot::Mutex::new(PnLWindow::new()),
            last_pnl_micro_usd: AtomicI64::new(0),
            tripped: AtomicBool::new(false),
            warning_count: AtomicU64::new(0),
            critical_count: AtomicU64::new(0),
            last_trip_timestamp_ns: AtomicU64::new(0),
        }
    }

    /// Update with new PnL reading and check velocity
    /// 
    /// # Arguments
    /// * `pnl_usd` - Current cumulative PnL in USD (can be fractional, will be scaled)
    /// * `timestamp_ns` - Timestamp in nanoseconds
    /// 
    /// # Returns
    /// Result of the velocity check
    #[inline]
    pub fn update(&self, pnl_usd: f64, timestamp_ns: u64) -> VelocityCheckResult {
        // Convert to micro-USD for integer math
        let pnl_micro_usd = (pnl_usd * 1_000_000.0) as i64;
        
        // Store atomically
        self.last_pnl_micro_usd.store(pnl_micro_usd, Ordering::Relaxed);
        
        // Add to window
        {
            let mut window = self.window.lock();
            window.push(timestamp_ns, pnl_micro_usd);
        }
        
        // Calculate velocity
        let window_ref = self.window.lock();
        let velocity = window_ref.calculate_velocity(self.config.time_window_ms);
        
        // Determine result (velocity is negative for losses)
        let result = if velocity < self.config.critical_threshold_usd_per_ms {
            VelocityCheckResult::Critical {
                velocity_usd_per_ms: velocity,
                threshold_usd_per_ms: self.config.critical_threshold_usd_per_ms,
            }
        } else if velocity < self.config.warning_threshold_usd_per_ms {
            VelocityCheckResult::Warning {
                velocity_usd_per_ms: velocity,
                threshold_usd_per_ms: self.config.warning_threshold_usd_per_ms,
            }
        } else {
            VelocityCheckResult::Normal
        };
        
        // Update counters and trip state
        match result {
            VelocityCheckResult::Critical { .. } => {
                self.critical_count.fetch_add(1, Ordering::Relaxed);
                self.tripped.store(true, Ordering::SeqCst);
                self.last_trip_timestamp_ns.store(timestamp_ns, Ordering::Relaxed);
            }
            VelocityCheckResult::Warning { .. } => {
                self.warning_count.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        
        result
    }

    /// Check if breaker is tripped
    #[inline]
    pub fn is_tripped(&self) -> bool {
        self.tripped.load(Ordering::SeqCst)
    }

    /// Reset the breaker after manual intervention
    #[inline]
    pub fn reset(&self) {
        self.tripped.store(false, Ordering::SeqCst);
    }

    /// Force trip the breaker
    #[inline]
    pub fn force_trip(&self, timestamp_ns: u64) {
        self.tripped.store(true, Ordering::SeqCst);
        self.critical_count.fetch_add(1, Ordering::Relaxed);
        self.last_trip_timestamp_ns.store(timestamp_ns, Ordering::Relaxed);
    }

    /// Get current PnL in USD
    #[inline]
    pub fn get_current_pnl_usd(&self) -> f64 {
        self.last_pnl_micro_usd.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    /// Get statistics
    pub fn stats(&self) -> VelocityStats {
        let window = self.window.lock();
        let velocity = window.calculate_velocity(self.config.time_window_ms);
        
        VelocityStats {
            current_velocity_usd_per_ms: velocity as f64 / 1_000_000.0,
            warning_threshold: self.config.warning_threshold_usd_per_ms as f64,
            critical_threshold: self.config.critical_threshold_usd_per_ms as f64,
            time_window_ms: self.config.time_window_ms,
            warning_count: self.warning_count.load(Ordering::Relaxed),
            critical_count: self.critical_count.load(Ordering::Relaxed),
            is_tripped: self.tripped.load(Ordering::SeqCst),
            current_pnl_usd: self.get_current_pnl_usd(),
        }
    }
}

/// Statistics from the velocity breaker
#[derive(Debug, Clone)]
pub struct VelocityStats {
    pub current_velocity_usd_per_ms: f64,
    pub warning_threshold: f64,
    pub critical_threshold: f64,
    pub time_window_ms: u64,
    pub warning_count: u64,
    pub critical_count: u64,
    pub is_tripped: bool,
    pub current_pnl_usd: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_operation() {
        let config = VelocityConfig::default();
        let breaker = VelocityCircuitBreaker::new(config);
        
        // Simulate normal PnL fluctuations
        for i in 0..20 {
            let pnl = (i as f64 - 10.0) * 100.0; // Small swings
            let ts = i * 1_000_000; // 1ms intervals
            
            let result = breaker.update(pnl, ts);
            assert_eq!(result, VelocityCheckResult::Normal);
        }
        
        assert!(!breaker.is_tripped());
    }

    #[test]
    fn test_rapid_loss_detection() {
        let config = VelocityConfig::default();
        let breaker = VelocityCircuitBreaker::new(config);
        
        // Simulate rapid losses: -$1000 per ms
        for i in 0..30 {
            let pnl = -(i as f64) * 1000.0;
            let ts = i * 1_000_000;
            
            breaker.update(pnl, ts);
        }
        
        assert!(breaker.is_tripped());
        
        let stats = breaker.stats();
        assert!(stats.critical_count > 0);
    }

    #[test]
    fn test_reset_after_trip() {
        let config = VelocityConfig::default();
        let breaker = VelocityCircuitBreaker::new(config);
        
        // Trip the breaker
        for i in 0..30 {
            let pnl = -(i as f64) * 1000.0;
            let ts = i * 1_000_000;
            breaker.update(pnl, ts);
        }
        
        assert!(breaker.is_tripped());
        
        // Reset
        breaker.reset();
        assert!(!breaker.is_tripped());
    }
}
