//! Nexus Ingest - Live Market Data Ingestion Engine
//!
//! This crate provides production-grade WebSocket adapters for connecting
//! to real exchange market data feeds, with sequence healing and SPSC
//! ring buffer integration for zero-allocation internal routing.

pub mod adapters;
pub mod bridge;
pub mod decoders;

// Re-export main components
pub use adapters::{
    BinanceWsConfig, ConnectionState, LiveExchangeAdapter, ReconnectPhase,
    ReconnectStateMachine, SubscriptionChannel, WsStats,
};
pub use bridge::{
    AlertLevel, BackpressureState, BackpressureStats, NormalizedDelta,
    NormalizedTrade, TelemetryBridge, TelemetryBridgeConfig,
};
