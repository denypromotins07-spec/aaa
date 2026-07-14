//! Nexus Telemetry Crate - Zero-Alloc WebSocket Telemetry Bridge
//!
//! This crate provides a high-performance telemetry system for the NEXUS-OMEGA
//! trading bot. It uses:
//! - MessagePack binary serialization (zero heap allocations in hot path)
//! - Lock-free SPSC ring buffer for producer-consumer communication
//! - Dedicated Axum WebSocket server on isolated Tokio runtime
//! - Frame aggregation (microseconds to 60fps UI frames)
//! - Backpressure handling with slow client guillotine
//!
//! CRITICAL: No serde_json in the high-frequency data path!

// Tap module - Event subscription and frame aggregation
pub mod tap;

// Broadcast module - Binary serialization and schema registry
pub mod broadcast;

// Backpressure module - Drop policies and slow client handling
pub mod backpressure;

// Legacy modules (kept for backward compatibility)
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

// Re-export tap types
pub use tap::{
    LockFreeEventSubscriber, TelemetryEvent, EventSubscriberConfig,
    FrameAggregator, FrameAggregatorConfig,
    fps_to_frame_duration_ns, FRAME_60FPS_NS,
};

// Re-export broadcast types
pub use broadcast::{
    ZeroAllocSerializer, SerializationResult, SerializationError,
    BinarySchemaRegistry, FrameHeader, MessageType,
    MAX_BUFFER_SIZE, PROTOCOL_VERSION,
};

// Re-export backpressure types
pub use backpressure::{
    BoundedMpscQueue, BoundedQueueConfig, QueueDropPolicy,
    SlowClientGuillotine, GuillotineConfig, DisconnectReason,
    DropPolicy, DropMetrics,
};
