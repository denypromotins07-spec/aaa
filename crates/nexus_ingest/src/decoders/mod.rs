//! Chapter 3: Bridge Module
//!
//! This module provides the telemetry bridge for routing data
//! to SPSC ring buffers with backpressure monitoring.

pub mod spsc_telemetry_router;
pub mod backpressure_monitor;

pub use spsc_telemetry_router::{
    BackpressureState, BackpressureStats, NormalizedDelta, NormalizedTrade,
    TelemetryBridge, TelemetryBridgeConfig,
};
pub use backpressure_monitor::{
    AlertLevel, BackpressureMetrics, BackpressureMonitor, BackpressureMonitorConfig,
};
