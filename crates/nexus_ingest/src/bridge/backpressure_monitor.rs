//! Chapter 3: Backpressure Monitor
//!
//! This module provides monitoring and alerting for backpressure conditions
//! in the telemetry bridge. It tracks queue depths, drop rates, and provides
//! early warning signals before data loss occurs.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tracing::{debug, error, info, warn};

/// Alert level for backpressure conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertLevel {
    /// Normal operation
    Normal,
    /// Elevated but acceptable
    Elevated,
    /// Warning - approaching limits
    Warning,
    /// Critical - immediate action needed
    Critical,
    /// Data being dropped
    Dropping,
}

/// Backpressure metrics snapshot
#[derive(Debug, Clone)]
pub struct BackpressureMetrics {
    pub delta_queue_depth: usize,
    pub trade_queue_depth: usize,
    pub delta_drop_rate: f64, // Drops per second
    pub trade_drop_rate: f64,
    pub avg_latency_ns: u64,
    pub max_latency_ns: u64,
    pub consumer_lag_ms: u64,
    pub alert_level: AlertLevel,
    pub timestamp_ns: u64,
}

/// Configuration for backpressure monitor
#[derive(Debug, Clone)]
pub struct BackpressureMonitorConfig {
    /// Sample interval in milliseconds
    pub sample_interval_ms: u64,
    /// Alert log interval in seconds
    pub alert_log_interval_secs: u64,
    /// Drop rate threshold for warning (drops/sec)
    pub warning_drop_rate: f64,
    /// Drop rate threshold for critical (drops/sec)
    pub critical_drop_rate: f64,
    /// Maximum acceptable consumer lag in milliseconds
    pub max_consumer_lag_ms: u64,
}

impl Default for BackpressureMonitorConfig {
    fn default() -> Self {
        Self {
            sample_interval_ms: 100,
            alert_log_interval_secs: 10,
            warning_drop_rate: 10.0,
            critical_drop_rate: 100.0,
            max_consumer_lag_ms: 50,
        }
    }
}

/// Internal state for rate calculation
struct RateCalculator {
    last_count: u64,
    last_timestamp_ns: u64,
    current_rate: f64,
}

impl RateCalculator {
    fn new() -> Self {
        Self {
            last_count: 0,
            last_timestamp_ns: 0,
            current_rate: 0.0,
        }
    }

    fn update(&mut self, count: u64, timestamp_ns: u64) -> f64 {
        if self.last_timestamp_ns == 0 {
            self.last_count = count;
            self.last_timestamp_ns = timestamp_ns;
            return 0.0;
        }

        let elapsed_secs = (timestamp_ns - self.last_timestamp_ns) as f64 / 1_000_000_000.0;
        if elapsed_secs > 0.0 {
            let delta = count as f64 - self.last_count as f64;
            self.current_rate = delta / elapsed_secs;
        }

        self.last_count = count;
        self.last_timestamp_ns = timestamp_ns;
        self.current_rate
    }
}

/// Backpressure Monitor
pub struct BackpressureMonitor {
    config: BackpressureMonitorConfig,
    /// Current alert level
    alert_level: Arc<RwLock<AlertLevel>>,
    /// Delta drops counter
    delta_drops: Arc<AtomicU64>,
    /// Trade drops counter
    trade_drops: Arc<AtomicU64>,
    /// Total deltas pushed
    total_deltas: Arc<AtomicU64>,
    /// Total trades pushed
    total_trades: Arc<AtomicU64>,
    /// Last producer timestamp
    last_produce_ts: Arc<AtomicU64>,
    /// Last consumer timestamp
    last_consume_ts: Arc<AtomicU64>,
    /// Rate calculators
    delta_drop_rate: Arc<RwLock<RateCalculator>>,
    trade_drop_rate: Arc<RwLock<RateCalculator>>,
    /// Running flag
    is_running: Arc<AtomicBool>,
    /// Last alert log timestamp
    last_alert_log_ns: Arc<AtomicU64>,
}

// SAFETY: Uses atomics and RwLock for thread safety
unsafe impl Send for BackpressureMonitor {}
unsafe impl Sync for BackpressureMonitor {}

impl BackpressureMonitor {
    pub fn new(config: BackpressureMonitorConfig) -> Self {
        Self {
            config,
            alert_level: Arc::new(RwLock::new(AlertLevel::Normal)),
            delta_drops: Arc::new(AtomicU64::new(0)),
            trade_drops: Arc::new(AtomicU64::new(0)),
            total_deltas: Arc::new(AtomicU64::new(0)),
            total_trades: Arc::new(AtomicU64::new(0)),
            last_produce_ts: Arc::new(AtomicU64::new(0)),
            last_consume_ts: Arc::new(AtomicU64::new(0)),
            delta_drop_rate: Arc::new(RwLock::new(RateCalculator::new())),
            trade_drop_rate: Arc::new(RwLock::new(RateCalculator::new())),
            is_running: Arc::new(AtomicBool::new(true)),
            last_alert_log_ns: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Record a delta push
    pub fn record_delta_push(&self) {
        self.total_deltas.fetch_add(1, Ordering::Relaxed);
        self.update_producer_timestamp();
    }

    /// Record a delta drop
    pub fn record_delta_drop(&self) {
        self.delta_drops.fetch_add(1, Ordering::Relaxed);
        self.update_alert_level();
    }

    /// Record a trade push
    pub fn record_trade_push(&self) {
        self.total_trades.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a trade drop
    pub fn record_trade_drop(&self) {
        self.trade_drops.fetch_add(1, Ordering::Relaxed);
        self.update_alert_level();
    }

    /// Update producer timestamp
    fn update_producer_timestamp(&self) {
        let now = current_time_ns();
        self.last_produce_ts.store(now, Ordering::Relaxed);
    }

    /// Update consumer timestamp (called by consumer)
    pub fn update_consumer_timestamp(&self) {
        let now = current_time_ns();
        self.last_consume_ts.store(now, Ordering::Relaxed);
    }

    /// Calculate consumer lag
    fn get_consumer_lag_ms(&self) -> u64 {
        let produce_ts = self.last_produce_ts.load(Ordering::Relaxed);
        let consume_ts = self.last_consume_ts.load(Ordering::Relaxed);
        
        if produce_ts == 0 || consume_ts == 0 {
            return 0;
        }
        
        ((produce_ts - consume_ts) / 1_000_000) as u64
    }

    /// Update alert level based on current conditions
    fn update_alert_level(&self) {
        let now = current_time_ns();
        
        // Calculate drop rates
        let delta_drops = self.delta_drops.load(Ordering::Relaxed);
        let trade_drops = self.trade_drops.load(Ordering::Relaxed);
        
        let delta_rate = self.delta_drop_rate.write().update(delta_drops, now);
        let trade_rate = self.trade_drop_rate.write().update(trade_drops, now);
        
        let combined_rate = delta_rate + trade_rate;
        let consumer_lag = self.get_consumer_lag_ms();
        
        // Determine alert level
        let new_level = if combined_rate >= self.config.critical_drop_rate || 
                         consumer_lag >= self.config.max_consumer_lag_ms * 2 {
            AlertLevel::Dropping
        } else if combined_rate >= self.config.warning_drop_rate ||
                   consumer_lag >= self.config.max_consumer_lag_ms {
            AlertLevel::Critical
        } else if combined_rate > 0.0 {
            AlertLevel::Warning
        } else {
            AlertLevel::Normal
        };

        let mut current = self.alert_level.write();
        if *current != new_level {
            debug!("Alert level changed: {:?} -> {:?}", *current, new_level);
            *current = new_level;
            
            // Log significant alerts
            if new_level == AlertLevel::Critical || new_level == AlertLevel::Dropping {
                self.maybe_log_alert(new_level, combined_rate, consumer_lag);
            }
        }
    }

    /// Log alert with rate limiting
    fn maybe_log_alert(&self, level: AlertLevel, drop_rate: f64, lag_ms: u64) {
        let now = current_time_ns();
        let last_log = self.last_alert_log_ns.load(Ordering::Relaxed);
        let interval_ns = Duration::from_secs(self.config.alert_log_interval_secs).as_nanos() as u64;

        if now - last_log >= interval_ns {
            match level {
                AlertLevel::Critical => {
                    error!(
                        "BACKPRESSURE CRITICAL: drop_rate={:.2}/s, consumer_lag={}ms",
                        drop_rate, lag_ms
                    );
                }
                AlertLevel::Dropping => {
                    error!(
                        "BACKPRESSURE DROPPING: DATA LOSS IN PROGRESS! drop_rate={:.2}/s, lag={}ms",
                        drop_rate, lag_ms
                    );
                }
                _ => {}
            }
            self.last_alert_log_ns.store(now, Ordering::Relaxed);
        }
    }

    /// Get current alert level
    pub fn get_alert_level(&self) -> AlertLevel {
        *self.alert_level.read()
    }

    /// Get current metrics snapshot
    pub fn get_metrics(&self, delta_queue_depth: usize, trade_queue_depth: usize) -> BackpressureMetrics {
        let now = current_time_ns();
        
        BackpressureMetrics {
            delta_queue_depth,
            trade_queue_depth,
            delta_drop_rate: self.delta_drop_rate.read().current_rate,
            trade_drop_rate: self.trade_drop_rate.read().current_rate,
            avg_latency_ns: 0, // Would need additional tracking
            max_latency_ns: 0,
            consumer_lag_ms: self.get_consumer_lag_ms(),
            alert_level: self.get_alert_level(),
            timestamp_ns: now,
        }
    }

    /// Get total drops since start
    pub fn get_total_drops(&self) -> (u64, u64) {
        (
            self.delta_drops.load(Ordering::Relaxed),
            self.trade_drops.load(Ordering::Relaxed),
        )
    }

    /// Reset counters
    pub fn reset(&self) {
        self.delta_drops.store(0, Ordering::Relaxed);
        self.trade_drops.store(0, Ordering::Relaxed);
        self.total_deltas.store(0, Ordering::Relaxed);
        self.total_trades.store(0, Ordering::Relaxed);
        *self.delta_drop_rate.write() = RateCalculator::new();
        *self.trade_drop_rate.write() = RateCalculator::new();
        *self.alert_level.write() = AlertLevel::Normal;
        info!("BackpressureMonitor reset");
    }

    /// Stop monitoring
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Release);
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }
}

/// Get current time in nanoseconds
fn current_time_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let monitor = BackpressureMonitor::new(BackpressureMonitorConfig::default());
        
        assert_eq!(monitor.get_alert_level(), AlertLevel::Normal);
        assert_eq!(monitor.get_total_drops(), (0, 0));
    }

    #[test]
    fn test_drop_recording() {
        let monitor = BackpressureMonitor::new(BackpressureMonitorConfig::default());
        
        monitor.record_delta_push();
        monitor.record_delta_drop();
        
        let (delta_drops, trade_drops) = monitor.get_total_drops();
        assert_eq!(delta_drops, 1);
        assert_eq!(trade_drops, 0);
        
        // Alert level should change due to drops
        let level = monitor.get_alert_level();
        assert!(level == AlertLevel::Warning || level == AlertLevel::Elevated);
    }

    #[test]
    fn test_consumer_lag_calculation() {
        let monitor = BackpressureMonitor::new(BackpressureMonitorConfig::default());
        
        // Initially no lag
        assert_eq!(monitor.get_consumer_lag_ms(), 0);
        
        // Simulate producer activity
        monitor.record_delta_push();
        
        // Small lag should exist now
        std::thread::sleep(Duration::from_millis(10));
        let lag = monitor.get_consumer_lag_ms();
        assert!(lag >= 5); // Allow some tolerance
    }

    #[test]
    fn test_reset() {
        let monitor = BackpressureMonitor::new(BackpressureMonitorConfig::default());
        
        monitor.record_delta_drop();
        monitor.record_trade_drop();
        
        monitor.reset();
        
        assert_eq!(monitor.get_total_drops(), (0, 0));
        assert_eq!(monitor.get_alert_level(), AlertLevel::Normal);
    }
}
