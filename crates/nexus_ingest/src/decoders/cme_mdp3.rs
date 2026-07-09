//! Chapter 4: CME MDP 3.0 Binary Decoder
//!
//! This module implements a zero-copy binary decoder for CME Market Data Platform 3.0
//! protocol, reading directly from byte slices using pointer arithmetic.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use nexus_core::memory::cache_padder::CachePadded64;

/// CME MDP 3.0 message types
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mdp3MessageType {
    /// Channel reset
    ChannelReset = 5049,
    /// Instrument definition
    InstrumentDefinition = 35,
    /// Security definition
    SecurityDefinition = 32,
    /// Incremental refresh (order book updates)
    MarketDataIncrementalRefresh = 347,
    /// Snapshot full refresh
    MarketDataSnapshotFullRefresh = 322,
    /// Trade summary
    MDIncGrpTradeSummary = 801,
    /// Book clear
    BookClear = 805,
}

impl Mdp3MessageType {
    #[inline]
    pub fn from_u16(val: u16) -> Option<Self> {
        match val {
            5049 => Some(Mdp3MessageType::ChannelReset),
            35 => Some(Mdp3MessageType::InstrumentDefinition),
            32 => Some(Mdp3MessageType::SecurityDefinition),
            347 => Some(Mdp3MessageType::MarketDataIncrementalRefresh),
            322 => Some(Mdp3MessageType::MarketDataSnapshotFullRefresh),
            801 => Some(Mdp3MessageType::MDIncGrpTradeSummary),
            805 => Some(Mdp3MessageType::BookClear),
            _ => None,
        }
    }
}

/// MDP3 message header (SBE - Simple Binary Encoding)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Mdp3Header {
    /// Message length (excluding header)
    pub message_length: u16,
    /// Message type
    pub message_type: u16,
    /// Session ID
    pub session_id: u32,
}

// SAFETY: Mdp3Header is POD (plain old data)
unsafe impl Send for Mdp3Header {}
unsafe impl Sync for Mdp3Header {}

impl Mdp3Header {
    #[inline]
    pub const fn new() -> Self {
        Self {
            message_length: 0,
            message_type: 0,
            session_id: 0,
        }
    }

    /// Parse header from raw bytes (little-endian)
    #[inline]
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 8 {
            return None;
        }

        Some(Self {
            message_length: u16::from_le_bytes([data[0], data[1]]),
            message_type: u16::from_le_bytes([data[2], data[3]]),
            session_id: u32::from_le_bytes([data[4], data[5], data[6], data[7]]),
        })
    }
}

impl Default for Mdp3Header {
    fn default() -> Self {
        Self::new()
    }
}

/// Order book delta entry
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OrderBookEntry {
    /// Order ID
    pub order_id: u64,
    /// Price (scaled)
    pub price: u64,
    /// Quantity
    pub quantity: u32,
    /// Side (0=Buy, 1=Sell)
    pub side: u8,
    /// Level (price level)
    pub level: u8,
    /// Action type
    pub action: u8,
}

// SAFETY: OrderBookEntry is POD
unsafe impl Send for OrderBookEntry {}
unsafe impl Sync for OrderBookEntry {}

impl OrderBookEntry {
    #[inline]
    pub const fn new() -> Self {
        Self {
            order_id: 0,
            price: 0,
            quantity: 0,
            side: 0,
            level: 0,
            action: 0,
        }
    }
}

impl Default for OrderBookEntry {
    fn default() -> Self {
        Self::new()
    }
}

/// Parsed MDP3 message
#[repr(C)]
pub struct Mdp3Message<'a> {
    /// Message type
    pub message_type: Mdp3MessageType,
    /// Session ID
    pub session_id: u32,
    /// Sequence number
    pub sequence: u64,
    /// Transaction time (nanoseconds)
    pub transact_time: u64,
    /// Raw payload reference (zero-copy)
    pub payload: &'a [u8],
    /// Number of entries
    pub entry_count: usize,
}

// SAFETY: Mdp3Message contains borrowed data but is short-lived
unsafe impl<'a> Send for Mdp3Message<'a> {}

/// CME MDP 3.0 decoder
#[repr(C)]
pub struct CmeMdp3Decoder {
    /// Messages decoded
    messages_decoded: CachePadded64<AtomicU64>,
    /// Bytes decoded
    bytes_decoded: CachePadded64<AtomicU64>,
    /// Decode errors
    decode_errors: CachePadded64<AtomicU64>,
    /// Last sequence number
    last_sequence: CachePadded64<AtomicU64>,
}

// SAFETY: CmeMdp3Decoder is single-threaded
unsafe impl Send for CmeMdp3Decoder {}
unsafe impl Sync for CmeMdp3Decoder {}

impl CmeMdp3Decoder {
    /// Create a new MDP3 decoder
    #[inline]
    pub fn new() -> Self {
        Self {
            messages_decoded: CachePadded64::new(AtomicU64::new(0)),
            bytes_decoded: CachePadded64::new(AtomicU64::new(0)),
            decode_errors: CachePadded64::new(AtomicU64::new(0)),
            last_sequence: CachePadded64::new(AtomicU64::new(0)),
        }
    }

    /// Decode a single MDP3 message from raw bytes
    #[inline]
    pub fn decode_message<'a>(&self, data: &'a [u8]) -> Result<Mdp3Message<'a>, &'static str> {
        if data.is_empty() {
            return Err("Empty data");
        }

        // Parse header
        let header = Mdp3Header::from_bytes(data)
            .ok_or("Invalid header")?;

        // Validate we have enough data
        let total_len = 8 + header.message_length as usize;
        if data.len() < total_len {
            return Err("Incomplete message");
        }

        // Get message type
        let msg_type = Mdp3MessageType::from_u16(header.message_type)
            .ok_or("Unknown message type")?;

        // Extract payload (skip header)
        let payload = &data[8..total_len];

        self.bytes_decoded.0.fetch_add(total_len as u64, Ordering::Relaxed);
        self.messages_decoded.0.fetch_add(1, Ordering::Relaxed);

        Ok(Mdp3Message {
            message_type: msg_type,
            session_id: header.session_id,
            sequence: 0, // Would be extracted from message body
            transact_time: 0, // Would be extracted from message body
            payload,
            entry_count: 0, // Would be counted during parsing
        })
    }

    /// Decode incremental refresh message
    #[inline]
    pub fn decode_incremental_refresh<'a>(
        &self,
        data: &'a [u8],
    ) -> Result<(Mdp3Message<'a>, Vec<OrderBookEntry>), &'static str> {
        let msg = self.decode_message(data)?;

        if msg.message_type != Mdp3MessageType::MarketDataIncrementalRefresh {
            return Err("Not an incremental refresh message");
        }

        // Parse SBE repeating group
        let entries = self.parse_repeating_group(msg.payload)?;

        Ok((msg, entries))
    }

    /// Parse SBE repeating group for order book entries
    #[inline]
    fn parse_repeating_group(&self, payload: &[u8]) -> Result<Vec<OrderBookEntry>, &'static str> {
        if payload.is_empty() {
            return Ok(Vec::new());
        }

        let mut entries = Vec::with_capacity(16);
        let mut offset = 0;

        // Skip block header (group size encoding)
        if payload.len() < 2 {
            return Err("Payload too small");
        }
        
        let num_in_group = payload[0] as usize;
        offset += 1; // Simplified - actual SBE has more complex header

        // Parse each entry
        for _ in 0..num_in_group {
            if offset + 24 > payload.len() {
                break; // Not enough data for full entry
            }

            let entry = OrderBookEntry {
                order_id: u64::from_le_bytes([
                    payload[offset],
                    payload[offset + 1],
                    payload[offset + 2],
                    payload[offset + 3],
                    payload[offset + 4],
                    payload[offset + 5],
                    payload[offset + 6],
                    payload[offset + 7],
                ]),
                price: u64::from_le_bytes([
                    payload[offset + 8],
                    payload[offset + 9],
                    payload[offset + 10],
                    payload[offset + 11],
                    payload[offset + 12],
                    payload[offset + 13],
                    payload[offset + 14],
                    payload[offset + 15],
                ]),
                quantity: u32::from_le_bytes([
                    payload[offset + 16],
                    payload[offset + 17],
                    payload[offset + 18],
                    payload[offset + 19],
                ]),
                side: payload[offset + 20],
                level: payload[offset + 21],
                action: payload[offset + 22],
            };

            entries.push(entry);
            offset += 23; // Entry size
        }

        Ok(entries)
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> (u64, u64, u64) {
        (
            self.messages_decoded.0.load(Ordering::Relaxed),
            self.bytes_decoded.0.load(Ordering::Relaxed),
            self.decode_errors.0.load(Ordering::Relaxed),
        )
    }

    /// Update last sequence
    #[inline]
    pub fn update_sequence(&self, seq: u64) {
        self.last_sequence.0.store(seq, Ordering::Release);
    }

    /// Get last sequence
    #[inline]
    pub fn last_sequence(&self) -> u64 {
        self.last_sequence.0.load(Ordering::Acquire)
    }
}

impl Default for CmeMdp3Decoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mdp3_header_parsing() {
        let data = vec![
            0x10, 0x00, // message_length = 16
            0x59, 0x01, // message_type = 347 (0x0159 little-endian)
            0x01, 0x00, 0x00, 0x00, // session_id = 1
        ];

        let header = Mdp3Header::from_bytes(&data).unwrap();
        assert_eq!(header.message_length, 16);
        assert_eq!(header.message_type, 347);
        assert_eq!(header.session_id, 1);
    }

    #[test]
    fn test_message_type_conversion() {
        assert_eq!(Mdp3MessageType::from_u16(347), Some(Mdp3MessageType::MarketDataIncrementalRefresh));
        assert_eq!(Mdp3MessageType::from_u16(322), Some(Mdp3MessageType::MarketDataSnapshotFullRefresh));
        assert_eq!(Mdp3MessageType::from_u16(9999), None);
    }

    #[test]
    fn test_decoder_basic() {
        let decoder = CmeMdp3Decoder::new();
        
        // Create minimal valid message
        let mut data = vec![
            0x04, 0x00, // message_length = 4
            0x59, 0x01, // message_type = 347
            0x01, 0x00, 0x00, 0x00, // session_id = 1
            0x00, 0x00, 0x00, 0x00, // 4 bytes payload
        ];

        let result = decoder.decode_message(&data);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert_eq!(msg.message_type, Mdp3MessageType::MarketDataIncrementalRefresh);
        assert_eq!(msg.session_id, 1);
    }

    #[test]
    fn test_decoder_statistics() {
        let decoder = CmeMdp3Decoder::new();

        let data = vec![
            0x04, 0x00,
            0x59, 0x01,
            0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];

        let _ = decoder.decode_message(&data);
        let _ = decoder.decode_message(&data);

        let (msgs, bytes, errors) = decoder.get_stats();
        assert_eq!(msgs, 2);
        assert_eq!(bytes, 24); // 12 bytes per message * 2
        assert_eq!(errors, 0);
    }

    #[test]
    fn test_incomplete_message() {
        let decoder = CmeMdp3Decoder::new();

        // Header says 100 bytes but we only have 4
        let data = vec![
            0x64, 0x00, // message_length = 100
            0x59, 0x01,
            0x01, 0x00, 0x00, 0x00,
        ];

        let result = decoder.decode_message(&data);
        assert!(result.is_err());
    }
}
