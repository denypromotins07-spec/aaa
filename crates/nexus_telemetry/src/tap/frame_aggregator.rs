//! Frame Aggregator - Microsecond to 60fps UI Frame Conversion
//!
//! This module aggregates microsecond-level trading events into 16ms "UI Frames"
//! for efficient frontend rendering. The UI only renders at 60fps, so there's no
//! need to send every single microsecond update.
//!
//! CRITICAL: This aggregator buffers events and produces a single dense payload
//! per 16ms window, dramatically reducing serialization overhead.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::Mutex;

use super::lock_free_event_subscriber::{TelemetryEvent, LockFreeEventSubscriber};
use crate::binary_serializer::{TelemetryFrame, SystemHealth};

/// Configuration for the frame aggregator
pub struct FrameAggregatorConfig {
    /// Target FPS (typically 60)
    pub target_fps: u32,
    /// Maximum order book levels to include per frame
    pub max_orderbook_levels: usize,
    /// Maximum trades to include per frame
    pub max_trades_per_frame: usize,
    /// Maximum alpha signals to include per frame
    pub max_alpha_signals: usize,
}

impl Default for FrameAggregatorConfig {
    fn default() -> Self {
        Self {
            target_fps: 60,
            max_orderbook_levels: 50,
            max_trades_per_frame: 100,
            max_alpha_signals: 20,
        }
    }
}

/// Aggregated state for a single symbol
#[derive(Debug, Clone, Default)]
pub struct SymbolState {
    /// Current bid levels (price, volume)
    pub bids: Vec<(i64, u64)>,
    /// Current ask levels (price, volume)
    pub asks: Vec<(i64, u64)>,
    /// Recent trades in this frame window
    pub trades: Vec<(i64, u64, u8)>,
    /// Latest timestamp seen
    pub last_update_ns: u64,
}

/// Active alpha signal
#[derive(Debug, Clone)]
pub struct ActiveAlphaSignal {
    pub strategy_id: u32,
    pub symbol: [u8; 8],
    pub signal_strength: f64,
    pub direction: i8,
    pub timestamp_ns: u64,
}

/// Internal aggregation state
struct AggregatorState {
    /// Per-symbol aggregated state
    symbols: Mutex<std::collections::HashMap<[u8; 8], SymbolState>>,
    /// Active alpha signals
    alpha_signals: Mutex<Vec<ActiveAlphaSignal>>,
    /// Current PnL state
    total_pnl_cents: AtomicI64,
    realized_pnl_cents: AtomicI64,
    unrealized_pnl_cents: AtomicI64,
    /// OMS state
    active_orders: AtomicU64,
    pending_orders: AtomicU64,
    filled_orders: AtomicU64,
    /// System health
    latency_us: AtomicU64,
    ops: AtomicU64,
    memory_mb: AtomicU64,
    /// Frame counter
    frames_produced: AtomicU64,
    /// Events consumed counter
    events_consumed: AtomicU64,
    /// Active flag
    active: AtomicBool,
}

/// Frame Aggregator - converts microsecond events to 60fps UI frames
pub struct FrameAggregator {
    state: Arc<AggregatorState>,
    subscriber: LockFreeEventSubscriber,
    config: FrameAggregatorConfig,
    /// Frame interval in nanoseconds
    frame_interval_ns: u64,
}

impl FrameAggregator {
    /// Create a new frame aggregator
    pub fn new(subscriber: LockFreeEventSubscriber, config: FrameAggregatorConfig) -> Self {
        let frame_interval_ns = Duration::from_secs(1).as_nanos() as u64 / config.target_fps as u64;
        
        Self {
            state: Arc::new(AggregatorState {
                symbols: Mutex::new(std::collections::HashMap::new()),
                alpha_signals: Mutex::new(Vec::new()),
                total_pnl_cents: AtomicI64::new(0),
                realized_pnl_cents: AtomicI64::new(0),
                unrealized_pnl_cents: AtomicI64::new(0),
                active_orders: AtomicU64::new(0),
                pending_orders: AtomicU64::new(0),
                filled_orders: AtomicU64::new(0),
                latency_us: AtomicU64::new(0),
                ops: AtomicU64::new(0),
                memory_mb: AtomicU64::new(0),
                frames_produced: AtomicU64::new(0),
                events_consumed: AtomicU64::new(0),
                active: AtomicBool::new(true),
            }),
            subscriber,
            config,
            frame_interval_ns,
        }
    }

    /// Process incoming events and aggregate them
    /// Call this in a loop at high frequency
    #[inline]
    pub fn process_events(&self) {
        while let Ok(event) = self.subscriber.try_recv() {
            self.state.events_consumed.fetch_add(1, Ordering::Relaxed);
            
            match event {
                TelemetryEvent::OrderBookUpdate { symbol, bids, asks, timestamp_ns: _ } => {
                    let mut symbols = self.state.symbols.lock();
                    let entry = symbols.entry(symbol).or_insert_with(SymbolState::default);
                    
                    // Keep only top N levels
                    entry.bids = bids.into_iter().take(self.config.max_orderbook_levels).collect();
                    entry.asks = asks.into_iter().take(self.config.max_orderbook_levels).collect();
                }
                TelemetryEvent::TradeExecuted { symbol, price, volume, side, timestamp_ns: _ } => {
                    let mut symbols = self.state.symbols.lock();
                    let entry = symbols.entry(symbol).or_insert_with(SymbolState::default);
                    
                    if entry.trades.len() < self.config.max_trades_per_frame {
                        entry.trades.push((price, volume, side));
                    }
                }
                TelemetryEvent::AlphaSignal { strategy_id, symbol, signal_strength, direction, timestamp_ns } => {
                    let mut signals = self.state.alpha_signals.lock();
                    if signals.len() < self.config.max_alpha_signals {
                        signals.push(ActiveAlphaSignal {
                            strategy_id,
                            symbol,
                            signal_strength,
                            direction,
                            timestamp_ns,
                        });
                    }
                }
                TelemetryEvent::PnLUpdate { total_pnl_cents, realized_pnl_cents, unrealized_pnl_cents, timestamp_ns: _ } => {
                    self.state.total_pnl_cents.store(total_pnl_cents, Ordering::Relaxed);
                    self.state.realized_pnl_cents.store(realized_pnl_cents, Ordering::Relaxed);
                    self.state.unrealized_pnl_cents.store(unrealized_pnl_cents, Ordering::Relaxed);
                }
                TelemetryEvent::OmsStateChange { active_orders, pending_orders, filled_orders, timestamp_ns: _ } => {
                    self.state.active_orders.store(active_orders, Ordering::Relaxed);
                    self.state.pending_orders.store(pending_orders, Ordering::Relaxed);
                    self.state.filled_orders.store(filled_orders, Ordering::Relaxed);
                }
                TelemetryEvent::SystemHealth { latency_us, ops, memory_mb, timestamp_ns: _ } => {
                    self.state.latency_us.store(latency_us as u64, Ordering::Relaxed);
                    self.state.ops.store(ops, Ordering::Relaxed);
                    self.state.memory_mb.store(memory_mb, Ordering::Relaxed);
                }
            }
        }
    }

    /// Produce a UI frame from the current aggregated state
    /// Call this at the target FPS rate (e.g., every 16ms for 60fps)
    pub fn produce_frame(&self) -> Option<TelemetryFrame> {
        if !self.state.active.load(Ordering::Relaxed) {
            return None;
        }

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Snapshot the current state
        let symbols = self.state.symbols.lock();
        let alpha_signals = self.state.alpha_signals.lock();
        
        // For simplicity, we'll use the first symbol (in production, you'd want multi-symbol support)
        let (symbol, symbol_state) = symbols.iter().next()?;
        
        let frame = TelemetryFrame {
            timestamp_ns: now_ns,
            symbol: *symbol,
            bids: symbol_state.bids.clone(),
            asks: symbol_state.asks.clone(),
            trades: symbol_state.trades.clone(),
            health: SystemHealth {
                latency_us: self.state.latency_us.load(Ordering::Relaxed) as u32,
                ops: self.state.ops.load(Ordering::Relaxed),
                pnl_cents: self.state.total_pnl_cents.load(Ordering::Relaxed),
                active_strategies: alpha_signals.len() as u8,
                memory_mb: self.state.memory_mb.load(Ordering::Relaxed) as u32,
            },
        };

        // Clear trades for next frame (they're ephemeral)
        drop(symbols);
        let mut symbols = self.state.symbols.lock();
        for (_, state) in symbols.iter_mut() {
            state.trades.clear();
        }
        
        // Clear alpha signals for next frame
        drop(alpha_signals);
        *self.state.alpha_signals.lock() = Vec::new();

        self.state.frames_produced.fetch_add(1, Ordering::Relaxed);
        
        Some(frame)
    }

    /// Run the aggregation loop
    /// This should be spawned as a tokio task
    pub async fn run_aggregation_loop(self: Arc<Self>) {
        let frame_duration = Duration::from_nanos(self.frame_interval_ns);
        let mut interval = tokio::time::interval(frame_duration);
        
        while self.state.active.load(Ordering::Relaxed) {
            interval.tick().await;
            
            // Process all pending events
            self.process_events();
            
            // Produce and emit a frame
            if let Some(_frame) = self.produce_frame() {
                // In production, this would send to the broadcaster
                // For now, we just count it
            }
        }
    }

    /// Check if aggregator is active
    pub fn is_active(&self) -> bool {
        self.state.active.load(Ordering::Relaxed)
    }

    /// Shutdown the aggregator
    pub fn shutdown(&self) {
        self.state.active.store(false, Ordering::SeqCst);
        self.subscriber.shutdown();
    }

    /// Get frames produced count
    pub fn frames_produced(&self) -> u64 {
        self.state.frames_produced.load(Ordering::Relaxed)
    }

    /// Get events consumed count
    pub fn events_consumed(&self) -> u64 {
        self.state.events_consumed.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_aggregator_basic() {
        let subscriber = LockFreeEventSubscriber::new(Default::default());
        let config = FrameAggregatorConfig::default();
        let aggregator = FrameAggregator::new(subscriber, config);
        
        // Emit some test events
        let _ = aggregator.subscriber.emit(TelemetryEvent::OrderBookUpdate {
            symbol: *b"BTCUSD   ",
            bids: vec![(95000000, 150000), (94999000, 250000)],
            asks: vec![(95001000, 100000), (95002000, 200000)],
            timestamp_ns: 1000000,
        });
        
        // Process events
        aggregator.process_events();
        
        // Produce a frame
        let frame = aggregator.produce_frame();
        assert!(frame.is_some());
        
        let frame = frame.unwrap();
        assert_eq!(&frame.symbol, b"BTCUSD   ");
        assert_eq!(frame.bids.len(), 2);
        assert_eq!(frame.asks.len(), 2);
    }
}
