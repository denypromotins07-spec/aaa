//! Broadcast Module - Zero-Alloc Binary WebSocket Broadcaster
//!
//! This module provides the broadcast layer for telemetry data:
//! - Zero-allocation binary serialization
//! - Binary schema registry for protocol versioning
//! - Axum WebSocket server integration

pub mod zero_alloc_serializer;
pub mod binary_schema_registry;

pub use zero_alloc_serializer::{
    ZeroAllocSerializer, SerializationResult, SerializationError,
    MAX_BUFFER_SIZE, CHUNK_SIZE,
};
pub use binary_schema_registry::{
    BinarySchemaRegistry, FrameHeader, MessageType, MessageSchema, SchemaField,
    PROTOCOL_VERSION, PROTOCOL_MAGIC, SchemaError,
};
