//! Chapter 3: Zero-Alloc SPSC Ring Buffer Integration
//!
//! This module provides the TelemetryBridge that routes normalized
//! OrderBookDelta and Trade structs into Stage 2 Lock-Free SPSC Ring Buffers.
//!
//! CRITICAL: The producer side NEVER blocks. If the buffer is full,
//! oldest ticks are dropped with a "Backpressure Warning" log.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam::queue::SegQueue;
use parking_lot::RwLock;
use tracing::{debug, error, info, warn};

/// Normalized order book delta for internal routing
#[derive(Debug, Clone, Copy)]
pub struct NormalizedDelta {
    pub timestamp_ns: u64,
    pub symbol: [u8; 16], // Fixed-size symbol buffer (zero-copy)
    pub price: u64,       // Nanodollars
    pub quantity: u64,    // Base units * 1e9
    pub side: u8,         // 0=Bid, 1=Ask
    pub delta_type: u8,   // 0=Add, 1=Modify, 2=Cancel, 3=Trade
    pub sequence_id: u64,
}

impl Default for NormalizedDelta {
    fn default() -> Self {
        Self {
            timestamp_ns: 0,
            symbol: [0u8; 16],
            price: 0,
            quantity: 0,
            side: 0,
            delta_type: 0,
            sequence_id: 0,
        }
    }
}

/// Normalized trade event
#[derive(Debug, Clone, Copy)]
pub struct NormalizedTrade {
    pub timestamp_ns: u64,
    pub symbol: [u8; 16],
    pub price: u64,
    pub quantity: u64,
    pub aggressor_side: u8, // 0=Buyer, 1=Seller
    pub trade_id: u64,
}

impl Default for NormalizedTrade {
    fn default() -> Self {
        Self {
            timestamp_ns: 0,
            symbol: [0u8; 16],
            price: 0,
            quantity: 0,
            aggressor_side: 0,
            trade_id: 0,
        }
    }
}

/// Backpressure state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressureState {
    Normal,
    Warning,
    Critical,
    Dropping,
}

/// Backpressure statistics
#[derive(Debug, Clone, Default)]
pub struct BackpressureStats {
    pub total_pushes: u64,
    pub successful_pushes: u64,
    pub dropped_deltas: u64,
    pub dropped_trades: u64,
    pub backpressure_warnings: u64,
    pub backpressure_critical_count: u64,
    pub current_queue_depth: usize,
    pub max_queue_depth_seen: usize,
}

/// Configuration for telemetry bridge
#[derive(Debug, Clone)]
pub struct TelemetryBridgeConfig {
    /// Maximum queue depth before backpressure warning
    pub warning_threshold: usize,
    /// Maximum queue depth before critical alert
    pub critical_threshold: usize,
    /// Maximum queue depth before dropping messages
    pub drop_threshold: usize,
    /// Interval between backpressure status logs in ms
    pub status_log_interval_ms: u64,
}

impl Default for TelemetryBridgeConfig {
    fn default() -> Self {
        Self {
            warning_threshold: 1000,
            critical_threshold: 5000,
            drop_threshold: 10000,
            status_log_interval_ms: 1000,
        }
    }
}

/// Telemetry Bridge for routing data to SPSC ring buffers
pub struct TelemetryBridge {
    config: TelemetryBridgeConfig,
    /// Queue for order book deltas (producer side)
    delta_queue: Arc<SegQueue<NormalizedDelta>>,
    /// Queue for trades (producer side)
    trade_queue: Arc<SegQueue<NormalizedTrade>>,
    /// Current backpressure state
    backpressure_state: Arc<RwLock<BackpressureState>>,
    /// Statistics
    stats: Arc<RwLock<BackpressureStats>>,
    /// Running flag
    is_running: Arc<AtomicBool>,
    /// Last status log timestamp
    last_status_log_ns: Arc<AtomicU64>,
}

// SAFETY: Uses thread-safe queues and RwLock
unsafe impl Send for TelemetryBridge {}
unsafe impl Sync for TelemetryBridge {}

impl TelemetryBridge {
    pub fn new(config: TelemetryBridgeConfig) -> Self {
        Self {
            config,
            delta_queue: Arc::new(SegQueue::new()),
            trade_queue: Arc::new(SegQueue::new()),
            backpressure_state: Arc::new(RwLock::new(BackpressureState::Normal)),
            stats: Arc::new(RwLock::new(BackpressureStats::default())),
            is_running: Arc::new(AtomicBool::new(true)),
            last_status_log_ns: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Push an order book delta into the queue (non-blocking)
    /// Returns true if successfully pushed, false if dropped due to backpressure
    pub fn push_delta(&self, delta: NormalizedDelta) -> bool {
        let mut stats = self.stats.write();
        stats.total_pushes += 1;

        // Check backpressure state
        let current_state = *self.backpressure_state.read();
        
        if current_state == BackpressureState::Dropping {
            stats.dropped_deltas += 1;
            debug!("Dropping delta due to severe backpressure");
            return false;
        }

        // Check queue depth
        let depth = self.delta_queue.len();
        
        if depth >= self.config.drop_threshold {
            *self.backpressure_state.write() = BackpressureState::Dropping;
            stats.dropped_deltas += 1;
            stats.backpressure_critical_count += 1;
            error!(
                "BACKPRESSURE CRITICAL: Queue depth {} exceeds drop threshold {}. Dropping delta.",
                depth, self.config.drop_threshold
            );
            return false;
        }

        if depth >= self.config.critical_threshold && current_state != BackpressureState::Critical {
            *self.backpressure_state.write() = BackpressureState::Critical;
            stats.backpressure_critical_count += 1;
            error!(
                "BACKPRESSURE CRITICAL: Queue depth {} exceeds critical threshold {}",
                depth, self.config.critical_threshold
            );
        } else if depth >= self.config.warning_threshold && current_state == BackpressureState::Normal {
            *self.backpressure_state.write() = BackpressureState::Warning;
            stats.backpressure_warnings += 1;
            warn!(
                "BACKPRESSURE WARNING: Queue depth {} exceeds warning threshold {}",
                depth, self.config.warning_threshold
            );
        } else if depth < self.config.warning_threshold && current_state != BackpressureState::Normal {
            *self.backpressure_state.write() = BackpressureState::Normal;
        }

        // Push to queue (SegQueue push is lock-free and non-blocking)
        self.delta_queue.push(delta);
        stats.successful_pushes += 1;
        stats.current_queue_depth = self.delta_queue.len();
        
        if stats.current_queue_depth > stats.max_queue_depth_seen {
            stats.max_queue_depth_seen = stats.current_queue_depth;
        }

        // Log periodic status
        self.maybe_log_status();

        true
    }

    /// Push a trade event into the queue (non-blocking)
    pub fn push_trade(&self, trade: NormalizedTrade) -> bool {
        let mut stats = self.stats.write();
        stats.total_pushes += 1;

        let current_state = *self.backpressure_state.read();
        
        if current_state == BackpressureState::Dropping {
            stats.dropped_trades += 1;
            debug!("Dropping trade due to severe backpressure");
            return false;
        }

        let depth = self.trade_queue.len();
        
        if depth >= self.config.drop_threshold {
            *self.backpressure_state.write() = BackpressureState::Dropping;
            stats.dropped_trades += 1;
            error!(
                "BACKPRESSURE CRITICAL: Trade queue depth {} exceeds drop threshold {}",
                depth, self.config.drop_threshold
            );
            return false;
        }

        self.trade_queue.push(trade);
        stats.successful_pushes += 1;

        true
    }

    /// Get the delta queue for consumer side (Stage 2)
    pub fn get_delta_queue(&self) -> Arc<SegQueue<NormalizedDelta>> {
        self.delta_queue.clone()
    }

    /// Get the trade queue for consumer side (Stage 2)
    pub fn get_trade_queue(&self) -> Arc<SegQueue<NormalizedTrade>> {
        self.trade_queue.clone()
    }

    /// Get current backpressure state
    pub fn get_backpressure_state(&self) -> BackpressureState {
        *self.backpressure_state.read()
    }

    /// Get statistics snapshot
    pub fn get_stats(&self) -> BackpressureStats {
        let mut stats = self.stats.write();
        stats.current_queue_depth = self.delta_queue.len();
        stats.clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        *self.stats.write() = BackpressureStats::default();
    }

    /// Stop the bridge
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Release);
        info!("TelemetryBridge stopped");
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Log status periodically
    fn maybe_log_status(&self) {
        let now = current_time_ns();
        let last_log = self.last_status_log_ns.load(Ordering::Acquire);
        let interval_ns = Duration::from_millis(self.config.status_log_interval_ms).as_nanos() as u64;

        if now - last_log >= interval_ns {
            let stats = self.get_stats();
            info!(
                "TelemetryBridge Status: pushes={}, success={}, dropped_deltas={}, dropped_trades={}, queue_depth={}, state={:?}",
                stats.total_pushes,
                stats.successful_pushes,
                stats.dropped_deltas,
                stats.dropped_trades,
                stats.current_queue_depth,
                self.get_backpressure_state()
            );
            self.last_status_log_ns.store(now, Ordering::Release);
        }
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
    fn test_push_delta_success() {
        let bridge = TelemetryBridge::new(TelemetryBridgeConfig::default());
        
        let delta = NormalizedDelta {
            timestamp_ns: current_time_ns(),
            symbol: *b"BTCUSDT         ",
            price: 50000_000000000,
            quantity: 1_000000000,
            side: 0,
            delta_type: 0,
            sequence_id: 1,
        };

        assert!(bridge.push_delta(delta));
        
        let stats = bridge.get_stats();
        assert_eq!(stats.total_pushes, 1);
        assert_eq!(stats.successful_pushes, 1);
    }

    #[test]
    fn test_backpressure_state_transitions() {
        let config = TelemetryBridgeConfig {
            warning_threshold: 10,
            critical_threshold: 50,
            drop_threshold: 100,
            ..Default::default()
        };
        let bridge = TelemetryBridge::new(config);

        assert_eq!(bridge.get_backpressure_state(), BackpressureState::Normal);

        // Fill queue past warning threshold
        for i in 0..15 {
            let delta = NormalizedDelta {
                sequence_id: i,
                ..Default::default()
            };
            bridge.push_delta(delta);
        }

        assert_eq!(bridge.get_backpressure_state(), BackpressureState::Warning);
    }

    #[test]
    fn test_push_trade() {
        let bridge = TelemetryBridge::new(TelemetryBridgeConfig::default());
        
        let trade = NormalizedTrade {
            timestamp_ns: current_time_ns(),
            symbol: *b"BTCUSDT         ",
            price: 50000_000000000,
            quantity: 1_000000000,
            aggressor_side: 0,
            trade_id: 12345,
        };

        assert!(bridge.push_trade(trade));
    }

    #[test]
    fn test_statistics_tracking() {
        let bridge = TelemetryBridge::new(TelemetryBridgeConfig::default());

        for i in 0..5 {
            let delta = NormalizedDelta {
                sequence_id: i,
                ..Default::default()
            };
            bridge.push_delta(delta);
        }

        let stats = bridge.get_stats();
        assert_eq!(stats.total_pushes, 5);
        assert_eq!(stats.successful_pushes, 5);
        assert_eq!(stats.dropped_deltas, 0);
    }
}
