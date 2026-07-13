//! Nexus Telemetry Crate - Root Module
//! 
//! This crate provides a zero-allocation WebSocket telemetry bridge
//! for the NEXUS-OMEGA trading bot. It uses MessagePack binary serialization
//! for high-frequency market data and runs on a dedicated Tokio thread pool.

pub mod binary_serializer;
pub mod lock_free_spsc_broadcaster;
pub mod axum_ws_server;

pub use binary_serializer::*;
pub use lock_free_spsc_broadcaster::*;
pub use axum_ws_server::*;
