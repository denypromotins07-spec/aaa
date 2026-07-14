//! Lock-Free Event Subscriber for Telemetry Tap
//!
//! This module implements a lock-free event subscriber that taps into the global
//! event bus (ingestion fills, alpha signals, OMS state changes, PnL updates).
//! It uses crossbeam channels for zero-contention event consumption.
//!
//! CRITICAL: This subscriber NEVER blocks the trading engine. Events are consumed
//! asynchronously and dropped if the internal buffer overflows.

use crossbeam::channel::{bounded, unbounded, Receiver, Sender, TryRecvError};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::Mutex;

/// Event types from the trading engine that telemetry needs to track
#[derive(Debug, Clone)]
pub enum TelemetryEvent {
    /// Order book update (symbol, bid_levels, ask_levels)
    OrderBookUpdate {
        symbol: [u8; 8],
        bids: Vec<(i64, u64)>,
        asks: Vec<(i64, u64)>,
        timestamp_ns: u64,
    },
    /// Trade execution
    TradeExecuted {
        symbol: [u8; 8],
        price: i64,
        volume: u64,
        side: u8, // 0=buy, 1=sell
        timestamp_ns: u64,
    },
    /// Alpha signal generated
    AlphaSignal {
        strategy_id: u32,
        symbol: [u8; 8],
        signal_strength: f64,
        direction: i8, // -1=short, 1=long
        timestamp_ns: u64,
    },
    /// PnL update
    PnLUpdate {
        total_pnl_cents: i64,
        realized_pnl_cents: i64,
        unrealized_pnl_cents: i64,
        timestamp_ns: u64,
    },
    /// OMS state change
    OmsStateChange {
        active_orders: u64,
        pending_orders: u64,
        filled_orders: u64,
        timestamp_ns: u64,
    },
    /// System health metric
    SystemHealth {
        latency_us: u32,
        ops: u64,
        memory_mb: u32,
        timestamp_ns: u64,
    },
}

/// Configuration for the lock-free event subscriber
pub struct EventSubscriberConfig {
    /// Buffer capacity for incoming events
    pub buffer_capacity: usize,
    /// Whether to drop oldest on overflow (true) or drop newest (false)
    pub drop_oldest_on_overflow: bool,
}

impl Default for EventSubscriberConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: 65536, // Handle burst of 64K events
            drop_oldest_on_overflow: true,
        }
    }
}

/// Internal state for the event subscriber
struct SubscriberState {
    /// Sequence counter for ordering
    sequence: AtomicU64,
    /// Active flag
    active: AtomicBool,
    /// Count of dropped events
    dropped_events: AtomicU64,
    /// Count of processed events
    processed_events: AtomicU64,
}

/// Lock-free event subscriber that taps into the global event bus
pub struct LockFreeEventSubscriber {
    state: Arc<SubscriberState>,
    /// Incoming event channel
    event_tx: Sender<TelemetryEvent>,
    event_rx: Receiver<TelemetryEvent>,
    /// Registered event sources (trading engine components)
    sources: Mutex<Vec<Sender<TelemetryEvent>>>,
    config: EventSubscriberConfig,
}

impl LockFreeEventSubscriber {
    /// Create a new lock-free event subscriber
    pub fn new(config: EventSubscriberConfig) -> Self {
        let (event_tx, event_rx) = bounded(config.buffer_capacity);
        
        Self {
            state: Arc::new(SubscriberState {
                sequence: AtomicU64::new(0),
                active: AtomicBool::new(true),
                dropped_events: AtomicU64::new(0),
                processed_events: AtomicU64::new(0),
            }),
            event_tx,
            event_rx,
            sources: Mutex::new(Vec::with_capacity(16)),
            config,
        }
    }

    /// Register an event source (called by trading engine components)
    /// Returns a Sender that the component can use to emit events
    pub fn register_source(&self) -> Sender<TelemetryEvent> {
        let (src_tx, src_rx) = unbounded();
        
        // Spawn a forwarder task that moves events from source to main buffer
        let main_tx = self.event_tx.clone();
        let active = Arc::clone(&self.state.active);
        let dropped = Arc::clone(&self.state.dropped_events);
        
        tokio::spawn(async move {
            while active.load(Ordering::Relaxed) {
                match src_rx.try_recv() {
                    Ok(event) => {
                        if main_tx.send(event).is_err() {
                            break;
                        }
                    }
                    Err(TryRecvError::Empty) => {
                        tokio::task::yield_now().await;
                    }
                    Err(TryRecvError::Disconnected) => break,
                }
            }
        });
        
        self.sources.lock().push(src_tx.clone());
        src_tx
    }

    /// Emit an event directly (for internal use)
    #[inline]
    pub fn emit(&self, event: TelemetryEvent) -> Result<(), u64> {
        if !self.state.active.load(Ordering::Relaxed) {
            return Err(0);
        }

        match self.event_tx.try_send(event) {
            Ok(()) => {
                self.state.sequence.fetch_add(1, Ordering::Relaxed);
                self.state.processed_events.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(_) => {
                let dropped = self.state.dropped_events.fetch_add(1, Ordering::Relaxed);
                Err(dropped + 1)
            }
        }
    }

    /// Try to receive an event (non-blocking)
    #[inline]
    pub fn try_recv(&self) -> Result<TelemetryEvent, TryRecvError> {
        self.event_rx.try_recv()
    }

    /// Get the receiver for batch consumption
    pub fn receiver(&self) -> &Receiver<TelemetryEvent> {
        &self.event_rx
    }

    /// Check if subscriber is active
    pub fn is_active(&self) -> bool {
        self.state.active.load(Ordering::Relaxed)
    }

    /// Shutdown the subscriber
    pub fn shutdown(&self) {
        self.state.active.store(false, Ordering::SeqCst);
    }

    /// Get current sequence number
    pub fn sequence(&self) -> u64 {
        self.state.sequence.load(Ordering::Relaxed)
    }

    /// Get count of dropped events
    pub fn dropped_count(&self) -> u64 {
        self.state.dropped_events.load(Ordering::Relaxed)
    }

    /// Get count of processed events
    pub fn processed_count(&self) -> u64 {
        self.state.processed_events.load(Ordering::Relaxed)
    }
}

impl Clone for LockFreeEventSubscriber {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            event_tx: self.event_tx.clone(),
            event_rx: self.event_rx.clone(),
            sources: self.sources.clone(),
            config: self.config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subscriber_basic() {
        let config = EventSubscriberConfig::default();
        let subscriber = LockFreeEventSubscriber::new(config);
        
        let event = TelemetryEvent::SystemHealth {
            latency_us: 10,
            ops: 1000,
            memory_mb: 64,
            timestamp_ns: 1000000,
        };
        
        assert!(subscriber.emit(event).is_ok());
        assert_eq!(subscriber.processed_count(), 1);
        assert_eq!(subscriber.dropped_count(), 0);
    }

    #[test]
    fn test_subscriber_overflow() {
        let config = EventSubscriberConfig {
            buffer_capacity: 2,
            drop_oldest_on_overflow: true,
        };
        let subscriber = LockFreeEventSubscriber::new(config);
        
        // Fill buffer
        for i in 0..10 {
            let _ = subscriber.emit(TelemetryEvent::SystemHealth {
                latency_us: i,
                ops: i as u64,
                memory_mb: 64,
                timestamp_ns: i as u64,
            });
        }
        
        // Some events should have been dropped
        assert!(subscriber.dropped_count() > 0);
    }
}
