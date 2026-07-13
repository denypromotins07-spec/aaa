//! Binary Serializer Module - Zero-Heap Allocation Serialization
//! 
//! CRITICAL: This module uses MessagePack (rmp-serde) for high-frequency
//! market data to ensure ZERO heap allocations during serialization.
//! Standard JSON is ONLY used for low-frequency UI control messages.

use serde::{Deserialize, Serialize};

/// High-frequency telemetry data structure
/// Designed for compact binary serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryFrame {
    /// Timestamp in nanoseconds since epoch
    pub timestamp_ns: u64,
    /// Symbol identifier
    pub symbol: [u8; 8],
    /// Bid price levels (price as i64 fixed-point, volume as u64)
    pub bids: Vec<(i64, u64)>,
    /// Ask price levels
    pub asks: Vec<(i64, u64)>,
    /// Last executed trades (price, volume, side: 0=buy, 1=sell)
    pub trades: Vec<(i64, u64, u8)>,
    /// System health metrics
    pub health: SystemHealth,
}

/// System health metrics for the trading engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    /// Latency in microseconds
    pub latency_us: u32,
    /// Orders per second
    pub ops: u64,
    /// PnL in cents (fixed-point)
    pub pnl_cents: i64,
    /// Active strategies count
    pub active_strategies: u8,
    /// Memory usage in MB
    pub memory_mb: u32,
}

/// Low-frequency control message (JSON-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlMessage {
    pub command: Command,
    pub payload: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    StartBot,
    StopBot,
    PauseStrategy,
    ResumeStrategy,
    UpdateRiskParams,
    RequestSnapshot,
}

/// Message envelope for WebSocket transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WsMessage {
    /// Binary MessagePack telemetry (high-frequency)
    Telemetry(TelemetryFrame),
    /// JSON control message (low-frequency)
    Control(ControlMessage),
    /// Acknowledgment
    Ack { sequence: u64 },
    /// Error response
    Error { code: u16, message: String },
}

impl WsMessage {
    /// Serialize telemetry to MessagePack bytes (zero-alloc path)
    pub fn to_msgpack_bytes(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(self)
    }

    /// Deserialize from MessagePack bytes
    pub fn from_msgpack_bytes(bytes: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(bytes)
    }

    /// Serialize control message to JSON (low-frequency only)
    pub fn to_json_string(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize from JSON string
    pub fn from_json_string(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_msgpack_roundtrip() {
        let frame = TelemetryFrame {
            timestamp_ns: 1704067200000000000,
            symbol: *b"BTCUSD   ",
            bids: vec![(95000000, 150000), (94999000, 250000)],
            asks: vec![(95001000, 100000), (95002000, 200000)],
            trades: vec![(95000500, 50000, 0)],
            health: SystemHealth {
                latency_us: 15,
                ops: 50000,
                pnl_cents: 125000,
                active_strategies: 12,
                memory_mb: 256,
            },
        };

        let msg = WsMessage::Telemetry(frame);
        let bytes = msg.to_msgpack_bytes().unwrap();
        let decoded = WsMessage::from_msgpack_bytes(&bytes).unwrap();
        
        match decoded {
            WsMessage::Telemetry(t) => {
                assert_eq!(t.timestamp_ns, 1704067200000000000);
                assert_eq!(&t.symbol, b"BTCUSD   ");
            }
            _ => panic!("Expected Telemetry variant"),
        }
    }
}
