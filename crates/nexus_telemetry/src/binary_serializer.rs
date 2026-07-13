//! Binary Serializer Module
//! Uses MessagePack (rmp-serde) for zero-allocation serialization of high-frequency market data.
//! Standard JSON is reserved ONLY for low-frequency control messages.

use serde::{Deserialize, Serialize};
use rmp_serde::{encode::Writer, decode::ReadReader};
use std::io::Cursor;

/// High-frequency market telemetry data structure
/// Designed for compact binary representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTelemetry {
    /// Timestamp in nanoseconds since epoch
    pub timestamp_ns: u64,
    /// Symbol identifier (e.g., "BTC-USD")
    pub symbol: String,
    /// Best bid price (fixed-point representation to avoid float issues)
    pub best_bid_price: i64,
    /// Best ask price
    pub best_ask_price: i64,
    /// Bid volume at best level
    pub best_bid_volume: u64,
    /// Ask volume at best level
    pub best_ask_volume: u64,
    /// L2 orderbook snapshot (compressed)
    pub l2_bids: Vec<(i64, u64)>, // (price, volume)
    pub l2_asks: Vec<(i64, u64)>,
    /// Recent trades for micro-price tape
    pub recent_trades: Vec<TradeTick>,
}

/// Individual trade tick for the micro-price tape
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeTick {
    pub timestamp_ns: u64,
    pub price: i64,
    pub volume: u64,
    pub is_aggressive_buy: bool,
}

/// System health metrics (lower frequency)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub latency_us: u32,
    pub pnl_usd: f64,
    pub swarm_status: SwarmState,
    pub cpu_usage: f32,
    pub memory_mb: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmState {
    Idle,
    Running,
    Paused,
    EmergencyStop,
}

/// Control message from UI to backend (JSON only)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlMessage {
    pub command: ControlCommand,
    pub payload: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlCommand {
    StartBot,
    StopBot,
    PauseBot,
    ResumeBot,
    SetRiskLimit { limit_usd: f64 },
}

/// Telemetry envelope for WebSocket transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TelemetryEnvelope {
    MarketData(MarketTelemetry),
    HealthUpdate(SystemHealth),
    Error { code: u16, message: String },
}

/// Zero-allocation MessagePack encoder
pub struct BinarySerializer;

impl BinarySerializer {
    /// Serialize telemetry to MessagePack bytes without heap allocation in the hot path
    /// The buffer is pre-allocated and reused
    pub fn encode_telemetry(telemetry: &MarketTelemetry, buffer: &mut Vec<u8>) -> Result<(), rmp_serde::encode::Error> {
        buffer.clear();
        let mut writer = Writer::new(Cursor::new(buffer));
        rmp_serde::encode::write(&mut writer, telemetry)?;
        Ok(())
    }

    /// Decode MessagePack bytes to telemetry
    pub fn decode_telemetry(bytes: &[u8]) -> Result<MarketTelemetry, rmp_serde::decode::Error> {
        let reader = ReadReader::new(Cursor::new(bytes));
        rmp_serde::from_read(reader)
    }

    /// Encode system health update
    pub fn encode_health(health: &SystemHealth, buffer: &mut Vec<u8>) -> Result<(), rmp_serde::encode::Error> {
        buffer.clear();
        let mut writer = Writer::new(Cursor::new(buffer));
        rmp_serde::encode::write(&mut writer, health)?;
        Ok(())
    }

    /// Encode error message
    pub fn encode_error(code: u16, message: &str, buffer: &mut Vec<u8>) -> Result<(), rmp_serde::encode::Error> {
        let envelope = TelemetryEnvelope::Error {
            code,
            message: message.to_string(),
        };
        buffer.clear();
        let mut writer = Writer::new(Cursor::new(buffer));
        rmp_serde::encode::write(&mut writer, &envelope)?;
        Ok(())
    }
}

/// Message type discriminator for WebSocket frames
/// First byte indicates message type
pub mod msg_types {
    pub const MARKET_DATA: u8 = 0x01;
    pub const HEALTH_UPDATE: u8 = 0x02;
    pub const CONTROL_MSG: u8 = 0x03;
    pub const ERROR_MSG: u8 = 0xFF;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_serialization_roundtrip() {
        let telemetry = MarketTelemetry {
            timestamp_ns: 1704067200000000000,
            symbol: "BTC-USD".to_string(),
            best_bid_price: 4200000, // $42,000.00 in cents
            best_ask_price: 4200100,
            best_bid_volume: 150,
            best_ask_volume: 200,
            l2_bids: vec![(4200000, 150), (4199900, 300)],
            l2_asks: vec![(4200100, 200), (4200200, 250)],
            recent_trades: vec![
                TradeTick {
                    timestamp_ns: 1704067200000000000,
                    price: 4200050,
                    volume: 50,
                    is_aggressive_buy: true,
                }
            ],
        };

        let mut buffer = Vec::with_capacity(4096);
        BinarySerializer::encode_telemetry(&telemetry, &mut buffer).unwrap();
        
        let decoded = BinarySerializer::decode_telemetry(&buffer).unwrap();
        assert_eq!(telemetry.symbol, decoded.symbol);
        assert_eq!(telemetry.best_bid_price, decoded.best_bid_price);
    }
}
