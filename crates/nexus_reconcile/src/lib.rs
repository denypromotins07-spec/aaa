//! Shadow Reconciliation Ledger, Implementation Shortfall & Walk-Forward Micro-Backtesting
//! 
//! This crate provides the "Reality Anchor" for NEXUS-OMEGA:
//! - ShadowStatePoller: Periodically queries exchange REST API for ground-truth state
//! - LockFreeOMSSnapshot: Epoch-based atomic snapshots without blocking live OMS
//! - PhantomFillDetector: Detects dropped WebSocket packets and triggers kill-switch
//! - ImplementationShortfallTracker: Measures slippage between signal and fill in nanoseconds
//! - WalkForwardMicroBacktester: Continuous in-memory backtest against recent tick data

pub mod ledger;
pub mod slippage;
pub mod backtest;

// Re-exports for convenience
pub use ledger::{
    shadow_state_poller::ShadowStatePoller,
    lock_free_oms_snapshot::LockFreeOMSSnapshot,
    phantom_fill_detector::PhantomFillDetector,
};

pub use slippage::{
    implementation_shortfall::ImplementationShortfallTracker,
    nanosecond_latency_tracker::NanosecondLatencyTracker,
    market_impact_feedback::MarketImpactFeedback,
};

pub use backtest::{
    walk_forward_micro_bt::WalkForwardMicroBacktester,
    parameter_decay_scorer::ParameterDecayScorer,
    in_memory_tick_replay::InMemoryTickReplay,
};

/// Core error types for reconciliation operations
#[derive(Debug, thiserror::Error)]
pub enum ReconcileError {
    #[error("Critical state drift detected: local={local:?}, exchange={exchange:?}")]
    CriticalStateDrift { local: String, exchange: String },
    
    #[error("Phantom fill detected: order_id={order_id}, expected={expected:?}, actual={actual:?}")]
    PhantomFill { 
        order_id: String, 
        expected: Option<FillReport>, 
        actual: Option<FillReport> 
    },
    
    #[error("REST API rate limit exceeded: retry_after={retry_after}s")]
    RateLimitExceeded { retry_after: u64 },
    
    #[error("Snapshot tear detected: epoch mismatch")]
    SnapshotTear,
    
    #[error("Tick buffer overrun: consumer_index={consumer}, producer={producer}")]
    TickBufferOverrun { consumer: usize, producer: usize },
    
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Fill report structure for reconciliation
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FillReport {
    pub order_id: String,
    pub executed_qty: i128,  // Scaled integer (wei/satoshis)
    pub avg_price: i128,     // Scaled integer (wei/satoshis)
    pub commission: i128,    // Scaled integer
    pub timestamp_ns: u64,   // Nanosecond timestamp
}

/// Exchange state snapshot for comparison
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExchangeStateSnapshot {
    pub balances: std::collections::HashMap<String, i128>,  // Asset -> Balance (scaled)
    pub positions: std::collections::HashMap<String, PositionState>,
    pub active_orders: Vec<OrderState>,
    pub server_time_ms: u64,
    pub epoch_id: u64,  // For tear-free reads
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PositionState {
    pub symbol: String,
    pub side: i8,  // 1=long, -1=short, 0=flat
    pub qty: i128,  // Scaled integer
    pub entry_price: i128,  // Scaled integer
    pub unrealized_pnl: i128,  // Scaled integer
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrderState {
    pub order_id: String,
    pub symbol: String,
    pub side: i8,  // 1=buy, -1=sell
    pub qty: i128,
    pub filled_qty: i128,
    pub price: i128,
    pub status: String,  // "NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED"
}
