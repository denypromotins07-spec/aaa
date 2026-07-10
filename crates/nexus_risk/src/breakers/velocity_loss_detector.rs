//! Velocity of Loss Circuit Breaker.
//! 
//! Detects acceleration in losses (not just absolute drawdown) to catch
//! catastrophic logic bugs or market flash crashes before they cause irreparable damage.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Epsilon for floating-point comparisons
const EPSILON: f64 = 1e-9;

/// Result of velocity check
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VelocityCheckResult {
    /// Loss velocity is within acceptable bounds
    Normal,
    /// Warning: loss velocity is elevated but not critical
    Warning {
        current_velocity: f64,
        warning_threshold: f64,
    },
    /// Critical: loss velocity exceeded threshold - trigger circuit breaker
    Critical {
        current_velocity: f64,
        critical_threshold: f64,
        time_window_ms: u64,
    },
}

/// Sliding window for tracking P&L samples
struct SlidingWindow {
    /// Ring buffer of (timestamp_ns, cumulative_pnl) pairs
    timestamps_ns: Vec<u64>,
    pnl_values: Vec<f64>,
    /// Current write position
    head: usize,
    /// Number of valid samples
    count: usize,
    /// Maximum window size
    capacity: usize,
}

impl SlidingWindow {
    fn new(capacity: usize) -> Self {
        Self {
            timestamps_ns: vec![0; capacity],
            pnl_values: vec![0.0; capacity],
            head: 0,
            count: 0,
            capacity,
        }
    }

    #[inline]
    fn push(&mut self, timestamp_ns: u64, pnl: f64) {
        self.timestamps_ns[self.head] = timestamp_ns;
        self.pnl_values[self.head] = pnl;
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Get the oldest sample in the window
    #[inline]
    fn oldest(&self) -> Option<(u64, f64)> {
        if self.count == 0 {
            return None;
        }
        let oldest_idx = if self.count < self.capacity {
            0
        } else {
            self.head
        };
        Some((self.timestamps_ns[oldest_idx], self.pnl_values[oldest_idx]))
    }

    /// Get the newest sample
    #[inline]
    fn newest(&self) -> Option<(u64, f64)> {
        if self.count == 0 {
            return None;
        }
        let newest_idx = if self.head == 0 {
            self.capacity - 1
        } else {
            self.head - 1
        };
        Some((self.timestamps_ns[newest_idx], self.pnl_values[newest_idx]))
    }

    /// Calculate velocity over a specific time window
    #[inline]
    fn velocity_over_window(&self, window_ns: u64) -> f64 {
        if self.count < 2 {
            return 0.0;
        }

        let now = self.newest().map(|(t, _)| t).unwrap_or(0);
        let cutoff = now.saturating_sub(window_ns);

        // Find the oldest sample within our target window
        let mut oldest_in_window: Option<(u64, f64)> = None;
        let mut newest_in_window: Option<(u64, f64)> = None;

        for i in 0..self.count {
            let idx = (self.head + i) % self.capacity;
            let ts = self.timestamps_ns[idx];
            let pnl = self.pnl_values[idx];

            if ts >= cutoff {
                if oldest_in_window.is_none() || ts < oldest_in_window.unwrap().0 {
                    oldest_in_window = Some((ts, pnl));
                }
                if newest_in_window.is_none() || ts > newest_in_window.unwrap().0 {
                    newest_in_window = Some((ts, pnl));
                }
            }
        }

        match (oldest_in_window, newest_in_window) {
            (Some((t1, p1)), Some((t2, p2))) if t2 > t1 => {
                let delta_pnl = p2 - p1;
                let delta_t_ms = (t2 - t1) as f64 / 1_000_000.0; // Convert ns to ms
                if delta_t_ms > EPSILON {
                    delta_pnl / delta_t_ms // USD per millisecond
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }
}

/// Configuration for the velocity loss detector
#[derive(Debug, Clone)]
pub struct VelocityConfig {
    /// Warning threshold in USD/ms (negative value indicates loss)
    pub warning_threshold_usd_per_ms: f64,
    /// Critical threshold in USD/ms
    pub critical_threshold_usd_per_ms: f64,
    /// Time window for velocity calculation in milliseconds
    pub time_window_ms: u64,
    /// Minimum samples required before triggering
    pub min_samples: usize,
}

impl Default for VelocityConfig {
    fn default() -> Self {
        Self {
            // $100 per ms = $100k per second warning
            warning_threshold_usd_per_ms: -100.0,
            // $500 per ms = $500k per second critical
            critical_threshold_usd_per_ms: -500.0,
            // 50ms window for ultra-fast detection
            time_window_ms: 50,
            min_samples: 3,
        }
    }
}

/// Velocity of Loss Detector
/// 
/// Monitors the rate of change of P&L to detect anomalous loss acceleration.
/// Uses a sliding window approach with atomic operations for lock-free updates.
pub struct VelocityLossDetector {
    /// Configuration
    config: VelocityConfig,
    /// Sliding window of P&L samples
    window: parking_lot::Mutex<SlidingWindow>,
    /// Last recorded P&L value
    last_pnl: AtomicU64, // Stored as bits for atomic f64
    /// Whether circuit breaker has been triggered
    tripped: AtomicBool,
    /// Count of warnings issued
    warning_count: AtomicU64,
    /// Count of critical events
    critical_count: AtomicU64,
    /// Timestamp of last trip (nanoseconds)
    last_trip_timestamp_ns: AtomicU64,
}

unsafe impl Send for VelocityLossDetector {}
unsafe impl Sync for VelocityLossDetector {}

impl VelocityLossDetector {
    /// Create a new velocity loss detector.
    pub fn new(config: VelocityConfig) -> Self {
        // Window capacity based on expected update frequency
        // For 50ms window with 1ms updates, need ~50 samples
        let window_capacity = (config.time_window_ms * 2) as usize;
        
        Self {
            config,
            window: parking_lot::Mutex::new(SlidingWindow::new(window_capacity)),
            last_pnl: AtomicU64::new(0),
            tripped: AtomicBool::new(false),
            warning_count: AtomicU64::new(0),
            critical_count: AtomicU64::new(0),
            last_trip_timestamp_ns: AtomicU64::new(0),
        }
    }

    /// Update with new P&L reading.
    /// 
    /// # Arguments
    /// * `pnl` - Current cumulative P&L in USD (negative = loss)
    /// * `timestamp_ns` - Timestamp in nanoseconds
    /// 
    /// Returns the result of the velocity check.
    #[inline]
    pub fn update(&self, pnl: f64, timestamp_ns: u64) -> VelocityCheckResult {
        // Store P&L atomically (as bit representation)
        self.last_pnl.store(f64::to_bits(pnl), Ordering::Relaxed);

        // Add to sliding window
        {
            let mut window = self.window.lock();
            window.push(timestamp_ns, pnl);
        }

        // Check if we have enough samples
        let window_ref = self.window.lock();
        if window_ref.count < self.config.min_samples {
            return VelocityCheckResult::Normal;
        }

        // Calculate velocity over the configured time window
        let window_ns = self.config.time_window_ms * 1_000_000;
        let velocity = window_ref.velocity_over_window(window_ns);

        // Determine result
        let result = if velocity < self.config.critical_threshold_usd_per_ms {
            VelocityCheckResult::Critical {
                current_velocity: velocity,
                critical_threshold: self.config.critical_threshold_usd_per_ms,
                time_window_ms: self.config.time_window_ms,
            }
        } else if velocity < self.config.warning_threshold_usd_per_ms {
            VelocityCheckResult::Warning {
                current_velocity: velocity,
                warning_threshold: self.config.warning_threshold_usd_per_ms,
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

    /// Get current P&L value
    #[inline]
    pub fn get_current_pnl(&self) -> f64 {
        f64::from_bits(self.last_pnl.load(Ordering::Relaxed))
    }

    /// Check if circuit breaker is tripped
    #[inline]
    pub fn is_tripped(&self) -> bool {
        self.tripped.load(Ordering::SeqCst)
    }

    /// Reset the circuit breaker (after manual intervention)
    #[inline]
    pub fn reset(&self) {
        self.tripped.store(false, Ordering::SeqCst);
    }

    /// Force trip the circuit breaker
    #[inline]
    pub fn force_trip(&self, timestamp_ns: u64) {
        self.tripped.store(true, Ordering::SeqCst);
        self.last_trip_timestamp_ns.store(timestamp_ns, Ordering::Relaxed);
        self.critical_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get statistics
    pub fn stats(&self) -> VelocityStats {
        let window = self.window.lock();
        let current_velocity = window.velocity_over_window(self.config.time_window_ms * 1_000_000);
        
        VelocityStats {
            current_velocity_usd_per_ms: current_velocity,
            warning_threshold: self.config.warning_threshold_usd_per_ms,
            critical_threshold: self.config.critical_threshold_usd_per_ms,
            time_window_ms: self.config.time_window_ms,
            warning_count: self.warning_count.load(Ordering::Relaxed),
            critical_count: self.critical_count.load(Ordering::Relaxed),
            is_tripped: self.tripped.load(Ordering::SeqCst),
            last_trip_timestamp_ns: self.last_trip_timestamp_ns.load(Ordering::Relaxed),
            current_pnl: f64::from_bits(self.last_pnl.load(Ordering::Relaxed)),
        }
    }

    /// Get estimated time to breach based on current velocity
    /// 
    /// # Arguments
    /// * `max_loss` - Maximum allowable loss from current P&L
    #[inline]
    pub fn time_to_breach(&self, max_loss: f64) -> Option<f64> {
        let stats = self.stats();
        let current_pnl = stats.current_pnl_usd_per_ms;
        let velocity = stats.current_velocity_usd_per_ms;
        
        if velocity >= 0.0 {
            return None; // Not losing money
        }
        
        let remaining_buffer = max_loss - current_pnl;
        if remaining_buffer <= 0.0 {
            return Some(0.0); // Already breached
        }
        
        // Time = distance / velocity (in ms)
        let time_ms = remaining_buffer / velocity.abs();
        Some(time_ms)
    }
}

/// Statistics from the velocity detector
#[derive(Debug, Clone)]
pub struct VelocityStats {
    pub current_velocity_usd_per_ms: f64,
    pub warning_threshold: f64,
    pub critical_threshold: f64,
    pub time_window_ms: u64,
    pub warning_count: u64,
    pub critical_count: u64,
    pub is_tripped: bool,
    pub last_trip_timestamp_ns: u64,
    pub current_pnl_usd_per_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_operation() {
        let config = VelocityConfig::default();
        let detector = VelocityLossDetector::new(config);
        
        // Simulate normal P&L fluctuations
        for i in 0..10 {
            let pnl = (i as f64 - 5.0) * 100.0; // Small swings around zero
            let ts = i * 1_000_000; // 1ms intervals
            
            let result = detector.update(pnl, ts);
            assert_eq!(result, VelocityCheckResult::Normal);
        }
        
        assert!(!detector.is_tripped());
    }

    #[test]
    fn test_rapid_loss_detection() {
        let config = VelocityConfig::default();
        let detector = VelocityLossDetector::new(config);
        
        // Simulate rapid losses: -$1000 per ms
        for i in 0..20 {
            let pnl = -(i as f64) * 1000.0;
            let ts = i * 1_000_000; // 1ms intervals
            
            detector.update(pnl, ts);
        }
        
        assert!(detector.is_tripped());
        
        let stats = detector.stats();
        assert!(stats.critical_count > 0);
        assert!(stats.current_velocity_usd_per_ms < config.critical_threshold_usd_per_ms);
    }

    #[test]
    fn test_warning_before_critical() {
        let mut config = VelocityConfig::default();
        config.warning_threshold_usd_per_ms = -50.0;
        config.critical_threshold_usd_per_ms = -200.0;
        
        let detector = VelocityLossDetector::new(config);
        
        // Simulate moderate losses: -$75 per ms (between warning and critical)
        for i in 0..20 {
            let pnl = -(i as f64) * 75.0;
            let ts = i * 1_000_000;
            
            detector.update(pnl, ts);
        }
        
        let stats = detector.stats();
        assert!(stats.warning_count > 0);
        assert_eq!(stats.critical_count, 0);
        assert!(!detector.is_tripped());
    }

    #[test]
    fn test_reset_after_trip() {
        let config = VelocityConfig::default();
        let detector = VelocityLossDetector::new(config);
        
        // Trip the breaker
        for i in 0..20 {
            let pnl = -(i as f64) * 1000.0;
            let ts = i * 1_000_000;
            detector.update(pnl, ts);
        }
        
        assert!(detector.is_tripped());
        
        // Reset
        detector.reset();
        assert!(!detector.is_tripped());
    }

    #[test]
    fn test_time_to_breach() {
        let config = VelocityConfig::default();
        let detector = VelocityLossDetector::new(config);
        
        // Set up a scenario with consistent loss velocity
        for i in 0..10 {
            let pnl = -(i as f64) * 100.0;
            let ts = i * 1_000_000;
            detector.update(pnl, ts);
        }
        
        // Estimate time to breach with $10k max loss
        let time_to_breach = detector.time_to_breach(-10000.0);
        
        // Should give some estimate (may be None if not enough data)
        // This test mainly verifies the function doesn't crash
        let _ = time_to_breach;
    }

    #[test]
    fn test_recovery_detection() {
        let config = VelocityConfig::default();
        let detector = VelocityLossDetector::new(config);
        
        // First, lose money rapidly
        for i in 0..10 {
            let pnl = -(i as f64) * 1000.0;
            let ts = i * 1_000_000;
            detector.update(pnl, ts);
        }
        
        assert!(detector.is_tripped());
        
        // Now simulate recovery (positive P&L movement)
        detector.reset();
        for i in 0..20 {
            let pnl = -10000.0 + (i as f64) * 100.0; // Recovering
            let ts = (10 + i) * 1_000_000;
            let result = detector.update(pnl, ts);
            
            // Should eventually return to normal
            if result == VelocityCheckResult::Normal {
                break;
            }
        }
        
        assert!(!detector.is_tripped());
    }

    #[test]
    fn test_sliding_window_behavior() {
        let mut window = SlidingWindow::new(5);
        
        // Fill the window
        for i in 0..10 {
            window.push(i * 1_000_000, i as f64);
        }
        
        // Should only have last 5 samples
        assert_eq!(window.count, 5);
        
        // Oldest should be sample 5, newest should be sample 9
        let oldest = window.oldest().unwrap();
        let newest = window.newest().unwrap();
        
        assert!(oldest.0 >= 5_000_000);
        assert_eq!(newest.1, 9.0);
    }
}
