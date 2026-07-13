//! Nexus Telemetry Crate - Zero-Alloc WebSocket Telemetry Bridge
//!
//! This crate provides a high-performance telemetry system for the NEXUS-OMEGA
//! trading bot. It uses:
//! - MessagePack binary serialization (zero heap allocations in hot path)
//! - Lock-free SPSC ring buffer for producer-consumer communication
//! - Dedicated Axum WebSocket server on isolated Tokio runtime
//!
//! CRITICAL: No serde_json in the high-frequency data path!

pub mod binary_serializer;
pub mod lock_free_spsc_broadcaster;
pub mod axum_ws_server;

// Re-export main types for convenience
pub use binary_serializer::{TelemetryFrame, SystemHealth, WsMessage, ControlMessage, Command};
pub use lock_free_spsc_broadcaster::{
    TelemetryBroadcaster, ProducerHandle, ConsumerHandle, 
    split_broadcaster, BroadcasterConfig,
};
pub use axum_ws_server::{WsServerConfig, WsServerHandle, start_telemetry_server};
