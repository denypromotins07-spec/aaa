//! Binary Schema Registry for WebSocket Telemetry
//!
//! This module manages binary schema versioning to ensure compatibility between
//! the Rust backend and JavaScript frontend. It handles schema negotiation,
//! version headers, and backward compatibility.

use std::collections::HashMap;

/// Current telemetry protocol version
pub const PROTOCOL_VERSION: u32 = 1;

/// Magic bytes for protocol identification
pub const PROTOCOL_MAGIC: [u8; 4] = [b'N', b'E', b'T', b'L']; // NETL = Nexus Telemetry

/// Message types in the binary protocol
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    /// Full telemetry frame (orderbook + health)
    TelemetryFrame = 0x01,
    /// Order book delta update
    OrderBookDelta = 0x02,
    /// Trade notification
    TradeNotification = 0x03,
    /// Alpha signal
    AlphaSignal = 0x04,
    /// PnL update
    PnLUpdate = 0x05,
    /// System health only
    SystemHealth = 0x06,
    /// Control message (JSON)
    ControlMessage = 0x10,
    /// Acknowledgment
    Acknowledgment = 0x20,
    /// Error response
    ErrorResponse = 0x21,
    /// Schema negotiation request
    SchemaNegotiation = 0xFE,
    /// Schema negotiation response
    SchemaNegotiationResponse = 0xFF,
}

impl MessageType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x01 => Some(Self::TelemetryFrame),
            0x02 => Some(Self::OrderBookDelta),
            0x03 => Some(Self::TradeNotification),
            0x04 => Some(Self::AlphaSignal),
            0x05 => Some(Self::PnLUpdate),
            0x06 => Some(Self::SystemHealth),
            0x10 => Some(Self::ControlMessage),
            0x20 => Some(Self::Acknowledgment),
            0x21 => Some(Self::ErrorResponse),
            0xFE => Some(Self::SchemaNegotiation),
            0xFF => Some(Self::SchemaNegotiationResponse),
            _ => None,
        }
    }
}

/// Binary frame header structure
/// 
/// Layout:
/// - Bytes 0-3: Protocol magic (NETL)
/// - Byte 4: Protocol version
/// - Byte 5: Message type
/// - Bytes 6-7: Payload length (big-endian u16, or 0xFFFF for extended)
/// - Bytes 8+: Payload (MessagePack serialized)
#[derive(Debug, Clone)]
pub struct FrameHeader {
    pub magic: [u8; 4],
    pub version: u8,
    pub message_type: MessageType,
    pub payload_length: u16,
}

impl FrameHeader {
    pub const HEADER_SIZE: usize = 8;

    pub fn new(message_type: MessageType, payload_length: u16) -> Self {
        Self {
            magic: PROTOCOL_MAGIC,
            version: PROTOCOL_VERSION as u8,
            message_type,
            payload_length,
        }
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> [u8; Self::HEADER_SIZE] {
        let mut bytes = [0u8; Self::HEADER_SIZE];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4] = self.version;
        bytes[5] = self.message_type as u8;
        bytes[6..8].copy_from_slice(&self.payload_length.to_be_bytes());
        bytes
    }

    /// Deserialize header from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SchemaError> {
        if bytes.len() < Self::HEADER_SIZE {
            return Err(SchemaError::InvalidHeaderLength(bytes.len()));
        }

        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != PROTOCOL_MAGIC {
            return Err(SchemaError::InvalidMagic(magic));
        }

        let version = bytes[4];
        let message_type = MessageType::from_u8(bytes[5])
            .ok_or(SchemaError::UnknownMessageType(bytes[5]))?;
        let payload_length = u16::from_be_bytes([bytes[6], bytes[7]]);

        Ok(Self {
            magic,
            version,
            message_type,
            payload_length,
        })
    }
}

/// Schema field definition for documentation and validation
#[derive(Debug, Clone)]
pub struct SchemaField {
    pub name: &'static str,
    pub field_type: &'static str,
    pub required: bool,
    pub description: &'static str,
}

/// Schema definition for a message type
#[derive(Debug, Clone)]
pub struct MessageSchema {
    pub message_type: MessageType,
    pub version: u32,
    pub fields: Vec<SchemaField>,
}

/// Binary schema registry managing all message schemas
pub struct BinarySchemaRegistry {
    /// Registered schemas by message type and version
    schemas: HashMap<(MessageType, u32), MessageSchema>,
    /// Current version for each message type
    current_versions: HashMap<MessageType, u32>,
}

impl BinarySchemaRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            schemas: HashMap::new(),
            current_versions: HashMap::new(),
        };
        
        // Register default schemas
        registry.register_default_schemas();
        registry
    }

    fn register_default_schemas(&mut self) {
        // TelemetryFrame schema v1
        self.register_schema(MessageSchema {
            message_type: MessageType::TelemetryFrame,
            version: 1,
            fields: vec![
                SchemaField {
                    name: "timestamp_ns",
                    field_type: "u64",
                    required: true,
                    description: "Timestamp in nanoseconds since epoch",
                },
                SchemaField {
                    name: "symbol",
                    field_type: "[u8; 8]",
                    required: true,
                    description: "Symbol identifier (8 bytes, space-padded)",
                },
                SchemaField {
                    name: "bids",
                    field_type: "Vec<(i64, u64)>",
                    required: true,
                    description: "Bid price levels (price, volume)",
                },
                SchemaField {
                    name: "asks",
                    field_type: "Vec<(i64, u64)>",
                    required: true,
                    description: "Ask price levels (price, volume)",
                },
                SchemaField {
                    name: "trades",
                    field_type: "Vec<(i64, u64, u8)>",
                    required: true,
                    description: "Recent trades (price, volume, side)",
                },
                SchemaField {
                    name: "health",
                    field_type: "SystemHealth",
                    required: true,
                    description: "System health metrics",
                },
            ],
        });

        // SystemHealth schema v1
        self.register_schema(MessageSchema {
            message_type: MessageType::SystemHealth,
            version: 1,
            fields: vec![
                SchemaField {
                    name: "latency_us",
                    field_type: "u32",
                    required: true,
                    description: "Latency in microseconds",
                },
                SchemaField {
                    name: "ops",
                    field_type: "u64",
                    required: true,
                    description: "Orders per second",
                },
                SchemaField {
                    name: "pnl_cents",
                    field_type: "i64",
                    required: true,
                    description: "Total PnL in cents",
                },
                SchemaField {
                    name: "active_strategies",
                    field_type: "u8",
                    required: true,
                    description: "Number of active strategies",
                },
                SchemaField {
                    name: "memory_mb",
                    field_type: "u32",
                    required: true,
                    description: "Memory usage in MB",
                },
            ],
        });
    }

    pub fn register_schema(&mut self, schema: MessageSchema) {
        let key = (schema.message_type, schema.version);
        self.schemas.insert(key, schema.clone());
        
        self.current_versions
            .entry(schema.message_type)
            .and_modify(|v| *v = (*v).max(schema.version))
            .or_insert(schema.version);
    }

    pub fn get_schema(&self, message_type: MessageType, version: u32) -> Option<&MessageSchema> {
        self.schemas.get(&(message_type, version))
    }

    pub fn get_current_version(&self, message_type: MessageType) -> u32 {
        *self.current_versions.get(&message_type).unwrap_or(&1)
    }

    /// Generate schema documentation for frontend developers
    pub fn generate_docs(&self) -> String {
        let mut docs = String::from("# NEXUS-OMEGA Telemetry Binary Schema\n\n");
        docs.push_str("## Protocol Header Format\n\n");
        docs.push_str("```\n");
        docs.push_str("| Offset | Size | Field          | Description                          |\n");
        docs.push_str("|--------|------|----------------|--------------------------------------|\n");
        docs.push_str("| 0      | 4    | Magic          | Protocol magic bytes (NETL)          |\n");
        docs.push_str("| 4      | 1    | Version        | Protocol version                     |\n");
        docs.push_str("| 5      | 1    | Message Type   | Message type identifier              |\n");
        docs.push_str("| 6      | 2    | Payload Length | Payload size in bytes (big-endian)   |\n");
        docs.push_str("| 8      | N    | Payload        | MessagePack serialized data          |\n");
        docs.push_str("```\n\n");

        docs.push_str("## Message Types\n\n");
        for (key, schema) in &self.schemas {
            docs.push_str(&format!("### {:?} (v{})\n\n", key.0, key.1));
            for field in &schema.fields {
                docs.push_str(&format!(
                    "- `{}`: {} ({}) - {}\n",
                    field.name,
                    field.field_type,
                    if field.required { "required" } else { "optional" },
                    field.description
                ));
            }
            docs.push('\n');
        }

        docs
    }
}

impl Default for BinarySchemaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Schema-related errors
#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    #[error("Invalid header length: {0}")]
    InvalidHeaderLength(usize),
    #[error("Invalid magic bytes: {0:?}")]
    InvalidMagic([u8; 4]),
    #[error("Unknown message type: {0}")]
    UnknownMessageType(u8),
    #[error("Unsupported protocol version: {0}")]
    UnsupportedVersion(u8),
    #[error("Schema not found for message type {0:?} version {1}")]
    SchemaNotFound(MessageType, u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = FrameHeader::new(MessageType::TelemetryFrame, 1024);
        let bytes = header.to_bytes();
        let decoded = FrameHeader::from_bytes(&bytes).unwrap();

        assert_eq!(header.magic, decoded.magic);
        assert_eq!(header.version, decoded.version);
        assert_eq!(header.message_type, decoded.message_type);
        assert_eq!(header.payload_length, decoded.payload_length);
    }

    #[test]
    fn test_message_type_conversion() {
        for i in 0u8..=255 {
            let result = MessageType::from_u8(i);
            if let Some(mt) = result {
                assert_eq!(MessageType::from_u8(mt as u8), Some(mt));
            }
        }
    }

    #[test]
    fn test_registry_default_schemas() {
        let registry = BinarySchemaRegistry::new();
        
        let schema = registry.get_schema(MessageType::TelemetryFrame, 1);
        assert!(schema.is_some());
        
        let schema = schema.unwrap();
        assert_eq!(schema.fields.len(), 6);
    }
}
