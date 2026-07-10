// STAGE 23: Computational Law, RegTech Compliance & Immutable Audit Trails
// Chapter 1: Real-Time Market Abuse Regulation (MAR) & Wash Trade Detection
// File: crates/nexus_legal/src/mar/spoofing_self_check.rs

//! Spoofing Self-Check module for monitoring cancel-to-trade and order-to-trade ratios.
//! Ensures the bot's own algorithms don't cross into illegal layering patterns.
//! Tracks per-symbol, per-venue metrics with sliding window aggregation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use crossbeam::channel::{bounded, Sender, Receiver};

use crate::mar::wash_trade_graph::{ExecutionNode, Side};

/// Configuration for spoofing detection thresholds
#[derive(Debug, Clone)]
pub struct SpoofingConfig {
    /// Maximum allowed cancel-to-trade ratio (e.g., 5.0 = 5 cancels per trade)
    pub max_cancel_ratio: f64,
    /// Maximum allowed order-to-trade ratio
    pub max_otr: f64,
    /// Time window for ratio calculation (nanoseconds)
    pub window_size_ns: u64,
    /// Minimum trades before enforcement kicks in
    pub min_trades_threshold: u32,
    /// Enable automatic order cancellation if limits exceeded
    pub auto_halt: bool,
}

impl Default for SpoofingConfig {
    fn default() -> Self {
        Self {
            max_cancel_ratio: 10.0, // Typical exchange limit
            max_otr: 20.0,
            window_size_ns: Duration::from_secs(300).as_nanos() as u64, // 5 minutes
            min_trades_threshold: 10,
            auto_halt: true,
        }
    }
}

/// Metrics tracked per symbol/venue combination
#[derive(Debug, Clone)]
struct SymbolMetrics {
    /// Total orders submitted in window
    total_orders: u64,
    /// Total trades executed in window
    total_trades: u64,
    /// Total cancellations in window
    total_cancellations: u64,
    /// Order IDs still active (not filled or cancelled)
    active_orders: Vec<u64>,
    /// Timestamp of first event in current window
    window_start_ns: u64,
    /// Last update time
    last_update_ns: u64,
}

impl SymbolMetrics {
    fn new(current_time_ns: u64) -> Self {
        Self {
            total_orders: 0,
            total_trades: 0,
            total_cancellations: 0,
            active_orders: Vec::with_capacity(256),
            window_start_ns: current_time_ns,
            last_update_ns: current_time_ns,
        }
    }

    fn cancel_ratio(&self) -> f64 {
        if self.total_trades == 0 {
            f64::INFINITY
        } else {
            self.total_cancellations as f64 / self.total_trades as f64
        }
    }

    fn otr(&self) -> f64 {
        if self.total_trades == 0 {
            f64::INFINITY
        } else {
            self.total_orders as f64 / self.total_trades as f64
        }
    }

    fn slide_window(&mut self, current_time_ns: u64, window_size_ns: u64) {
        if current_time_ns.saturating_sub(self.window_start_ns) > window_size_ns {
            // Reset counters for new window
            self.total_orders = 0;
            self.total_trades = 0;
            self.total_cancellations = 0;
            self.active_orders.clear();
            self.window_start_ns = current_time_ns;
        }
        self.last_update_ns = current_time_ns;
    }
}

/// Alert for spoofing-related violations
#[derive(Debug, Clone)]
pub struct SpoofingAlert {
    pub alert_id: u64,
    pub timestamp_ns: u64,
    pub symbol: String,
    pub venue_id: u32,
    pub alert_type: SpoofingAlertType,
    pub current_ratio: f64,
    pub threshold: f64,
    pub severity: AlertSeverity,
    pub auto_halt_triggered: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpoofingAlertType {
    CancelRatioExceeded,
    OtrExceeded,
    LayeringPatternDetected,
    RapidCancelSequence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Lock-free spoofing monitor with async alerting
pub struct SpoofingMonitor {
    config: SpoofingConfig,
    /// Metrics partitioned by (symbol, venue_id)
    metrics: DashMap<(String, u32), SymbolMetrics>,
    /// Alert counter
    alert_counter: AtomicU64,
    /// Async alert channel
    alert_sender: Sender<SpoofingAlert>,
    alert_receiver: Receiver<SpoofingAlert>,
    /// Global halt flag (set if any symbol exceeds limits)
    global_halt: AtomicBool,
    /// Layering pattern detector state
    layering_state: DashMap<(String, u32), LayeringState>,
}

/// State for detecting layering patterns
#[derive(Debug, Clone)]
struct LayeringState {
    /// Sequence of order sizes on each side
    buy_order_sizes: Vec<i64>,
    sell_order_sizes: Vec<i64>,
    /// Timestamps of recent orders
    order_timestamps: Vec<u64>,
    /// Detected imbalance score
    imbalance_score: f64,
}

impl LayeringState {
    fn new() -> Self {
        Self {
            buy_order_sizes: Vec::with_capacity(50),
            sell_order_sizes: Vec::with_capacity(50),
            order_timestamps: Vec::with_capacity(50),
            imbalance_score: 0.0,
        }
    }

    fn add_order(&mut self, side: Side, quantity: i64, timestamp_ns: u64) {
        match side {
            Side::Buy => self.buy_order_sizes.push(quantity),
            Side::Sell => self.sell_order_sizes.push(quantity),
        }
        self.order_timestamps.push(timestamp_ns);

        // Keep only last 50 orders
        if self.buy_order_sizes.len() > 50 {
            self.buy_order_sizes.remove(0);
        }
        if self.sell_order_sizes.len() > 50 {
            self.sell_order_sizes.remove(0);
        }
        if self.order_timestamps.len() > 50 {
            self.order_timestamps.remove(0);
        }

        // Calculate imbalance score
        let buy_total: i64 = self.buy_order_sizes.iter().sum();
        let sell_total: i64 = self.sell_order_sizes.iter().sum();
        
        if buy_total + sell_total > 0 {
            self.imbalance_score = 
                ((buy_total - sell_total) as f64 / (buy_total + sell_total) as f64).abs();
        }
    }
}

impl SpoofingMonitor {
    pub fn new() -> Self {
        Self::new_with_config(SpoofingConfig::default())
    }

    pub fn new_with_config(config: SpoofingConfig) -> Self {
        let (tx, rx) = bounded(10_000);
        Self {
            config,
            metrics: DashMap::new(),
            alert_counter: AtomicU64::new(0),
            alert_sender: tx,
            alert_receiver: rx,
            global_halt: AtomicBool::new(false),
            layering_state: DashMap::new(),
        }
    }

    /// Record an execution event (called from hot-path, must be non-blocking)
    pub fn record_execution(&self, exec: &ExecutionNode) {
        let key = (exec.symbol.clone(), exec.venue_id);
        let current_time_ns = exec.timestamp_ns;

        // Get or create metrics for this symbol/venue
        let mut metrics = self.metrics
            .entry(key.clone())
            .or_insert_with(|| SymbolMetrics::new(current_time_ns));

        // Slide window if needed
        metrics.slide_window(current_time_ns, self.config.window_size_ns);

        // Update counters
        metrics.total_orders += 1;
        metrics.total_trades += 1;
        metrics.active_orders.retain(|&id| id != exec.order_id);

        // Update layering state
        if let Some(mut layer_state) = self.layering_state.get_mut(&key) {
            layer_state.add_order(exec.side, exec.quantity, current_time_ns);
            
            // Check for layering pattern
            if layer_state.imbalance_score > 0.8 && layer_state.buy_order_sizes.len() > 10 {
                self.check_layering_pattern(&key, &layer_state, current_time_ns);
            }
        } else {
            let mut new_state = LayeringState::new();
            new_state.add_order(exec.side, exec.quantity, current_time_ns);
            self.layering_state.insert(key.clone(), new_state);
        }

        // Check ratios and potentially raise alerts
        self.check_ratios(&key, &metrics, current_time_ns);
    }

    /// Record a cancelled order
    pub fn record_cancellation(&self, symbol: &str, venue_id: u32, order_id: u64, timestamp_ns: u64) {
        let key = (symbol.to_string(), venue_id);
        
        let mut metrics = self.metrics
            .entry(key.clone())
            .or_insert_with(|| SymbolMetrics::new(timestamp_ns));

        metrics.slide_window(timestamp_ns, self.config.window_size_ns);
        metrics.total_cancellations += 1;
        metrics.active_orders.retain(|&id| id != order_id);

        self.check_ratios(&key, &metrics, timestamp_ns);
    }

    /// Record a new order submission (before execution)
    pub fn record_order_submission(&self, symbol: &str, venue_id: u32, order_id: u64, timestamp_ns: u64) {
        let key = (symbol.to_string(), venue_id);
        
        let mut metrics = self.metrics
            .entry(key.clone())
            .or_insert_with(|| SymbolMetrics::new(timestamp_ns));

        metrics.slide_window(timestamp_ns, self.config.window_size_ns);
        metrics.total_orders += 1;
        metrics.active_orders.push(order_id);
    }

    fn check_ratios(&self, key: &(String, u32), metrics: &SymbolMetrics, timestamp_ns: u64) {
        // Only check if we have enough trades
        if metrics.total_trades < self.config.min_trades_threshold as u64 {
            return;
        }

        let cancel_ratio = metrics.cancel_ratio();
        let otr = metrics.otr();

        // Check cancel ratio
        if cancel_ratio > self.config.max_cancel_ratio {
            self.raise_alert(
                key.0.clone(),
                key.1,
                SpoofingAlertType::CancelRatioExceeded,
                cancel_ratio,
                self.config.max_cancel_ratio,
                timestamp_ns,
            );
        }

        // Check OTR
        if otr > self.config.max_otr {
            self.raise_alert(
                key.0.clone(),
                key.1,
                SpoofingAlertType::OtrExceeded,
                otr,
                self.config.max_otr,
                timestamp_ns,
            );
        }
    }

    fn check_layering_pattern(&self, key: &(String, u32), state: &LayeringState, timestamp_ns: u64) {
        // Detect rapid sequence of orders on one side with quick cancellations
        // This is a simplified heuristic - production would use ML models
        
        if state.order_timestamps.len() < 20 {
            return;
        }

        // Check for rapid order sequence (< 1ms between orders)
        let mut rapid_count = 0;
        for i in 1..state.order_timestamps.len() {
            if state.order_timestamps[i].saturating_sub(state.order_timestamps[i-1]) < 1_000_000 {
                rapid_count += 1;
            }
        }

        if rapid_count > 15 && state.imbalance_score > 0.7 {
            self.raise_alert(
                key.0.clone(),
                key.1,
                SpoofingAlertType::LayeringPatternDetected,
                state.imbalance_score,
                0.7,
                timestamp_ns,
            );
        }
    }

    fn raise_alert(
        &self,
        symbol: String,
        venue_id: u32,
        alert_type: SpoofingAlertType,
        current_ratio: f64,
        threshold: f64,
        timestamp_ns: u64,
    ) {
        let alert_id = self.alert_counter.fetch_add(1, Ordering::SeqCst);
        
        let severity = match alert_type {
            SpoofingAlertType::CancelRatioExceeded => AlertSeverity::High,
            SpoofingAlertType::OtrExceeded => AlertSeverity::High,
            SpoofingAlertType::LayeringPatternDetected => AlertSeverity::Critical,
            SpoofingAlertType::RapidCancelSequence => AlertSeverity::Medium,
        };

        let auto_halt = self.config.auto_halt && matches!(severity, AlertSeverity::Critical);

        let alert = SpoofingAlert {
            alert_id,
            timestamp_ns,
            symbol,
            venue_id,
            alert_type,
            current_ratio,
            threshold,
            severity,
            auto_halt_triggered: auto_halt,
        };

        // Non-blocking send
        let _ = self.alert_sender.try_send(alert);

        if auto_halt {
            self.global_halt.store(true, Ordering::SeqCst);
            log::error!("CRITICAL: Spoofing detected - global halt triggered");
        }
    }

    /// Poll for alerts (called by compliance daemon)
    pub fn poll_alerts(&self) -> Vec<SpoofingAlert> {
        let mut alerts = Vec::new();
        while let Ok(alert) = self.alert_receiver.try_recv() {
            alerts.push(alert);
        }
        alerts
    }

    /// Calculate current OTR for a symbol
    pub fn calculate_otr(&self, symbol: &str, venue_id: u32) -> f64 {
        let key = (symbol.to_string(), venue_id);
        self.metrics
            .get(&key)
            .map(|m| m.otr())
            .unwrap_or(0.0)
    }

    /// Check if cancel ratio exceeds limit
    pub fn check_cancel_ratio(&self, symbol: &str, venue_id: u32, limit: f64) -> bool {
        let key = (symbol.to_string(), venue_id);
        self.metrics
            .get(&key)
            .map(|m| m.cancel_ratio() <= limit)
            .unwrap_or(true)
    }

    /// Check if global halt is active
    pub fn is_halted(&self) -> bool {
        self.global_halt.load(Ordering::Relaxed)
    }

    /// Reset halt flag (after manual intervention)
    pub fn reset_halt(&self) {
        self.global_halt.store(false, Ordering::SeqCst);
    }

    /// Get statistics for a symbol
    pub fn get_symbol_stats(&self, symbol: &str, venue_id: u32) -> Option<SymbolStats> {
        let key = (symbol.to_string(), venue_id);
        self.metrics.get(&key).map(|m| {
            SymbolStats {
                total_orders: m.total_orders,
                total_trades: m.total_trades,
                total_cancellations: m.total_cancellations,
                active_orders: m.active_orders.len() as u64,
                cancel_ratio: m.cancel_ratio(),
                otr: m.otr(),
            }
        })
    }
}

#[derive(Debug, Clone)]
pub struct SymbolStats {
    pub total_orders: u64,
    pub total_trades: u64,
    pub total_cancellations: u64,
    pub active_orders: u64,
    pub cancel_ratio: f64,
    pub otr: f64,
}

impl Default for SpoofingMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancel_ratio_tracking() {
        let config = SpoofingConfig {
            min_trades_threshold: 5,
            ..Default::default()
        };
        let monitor = SpoofingMonitor::new_with_config(config);

        let base_time = 1000000000u64;

        // Submit orders and executions
        for i in 0..10 {
            let exec = ExecutionNode {
                id: crate::mar::wash_trade_graph::ExecutionId(i),
                symbol: "BTCUSD".to_string(),
                asset_class: crate::mar::wash_trade_graph::AssetClass::CryptoSpot,
                venue_id: 1,
                side: Side::Buy,
                quantity: 100,
                price: 50000,
                timestamp_ns: base_time + i * 1000,
                strategy_id: 1,
                order_id: 100 + i,
                is_maker: false,
            };
            monitor.record_execution(&exec);
        }

        // Record many cancellations
        for i in 0..50 {
            monitor.record_cancellation("BTCUSD", 1, 200 + i, base_time + 50000 + i * 100);
        }

        let stats = monitor.get_symbol_stats("BTCUSD", 1).unwrap();
        assert!(stats.cancel_ratio > 5.0);
    }

    #[test]
    fn test_otr_calculation() {
        let monitor = SpoofingMonitor::new();
        
        // OTR should be infinite when no trades
        assert_eq!(monitor.calculate_otr("BTCUSD", 1), f64::INFINITY);
    }
}
