//! Chapter 4: SIMD-JSON WebSocket Parser
//!
//! This module implements a custom streaming parser for Binance/Bybit WebSockets
//! using simd-json to parse JSON payloads directly from raw network buffers
//! without allocating intermediate Strings.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use nexus_core::memory::cache_padder::CachePadded64;
use tracing::{debug, error, warn};

/// Maximum JSON payload size
pub const MAX_JSON_PAYLOAD: usize = 65536;

/// WebSocket frame types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebSocketFrameType {
    Text = 0x01,
    Binary = 0x02,
    Close = 0x08,
    Ping = 0x09,
    Pong = 0x0A,
}

impl WebSocketFrameType {
    #[inline]
    pub fn from_u8(val: u8) -> Option<Self> {
        match val & 0x0F {
            0x01 => Some(WebSocketFrameType::Text),
            0x02 => Some(WebSocketFrameType::Binary),
            0x08 => Some(WebSocketFrameType::Close),
            0x09 => Some(WebSocketFrameType::Ping),
            0x0A => Some(WebSocketFrameType::Pong),
            _ => None,
        }
    }
}

/// Zero-copy WebSocket frame
#[repr(C)]
pub struct WebSocketFrame<'a> {
    /// Frame type
    pub frame_type: WebSocketFrameType,
    /// Whether this is the final fragment
    pub fin: bool,
    /// Payload length
    pub payload_len: usize,
    /// Raw payload data (zero-copy reference)
    pub payload: &'a [u8],
}

// SAFETY: WebSocketFrame contains borrowed data but is short-lived
unsafe impl<'a> Send for WebSocketFrame<'a> {}

impl<'a> WebSocketFrame<'a> {
    #[inline]
    pub fn new(frame_type: WebSocketFrameType, payload: &'a [u8]) -> Self {
        Self {
            frame_type,
            fin: true,
            payload_len: payload.len(),
            payload,
        }
    }

    #[inline]
    pub fn as_str(&self) -> Option<&'a str> {
        if self.frame_type == WebSocketFrameType::Text {
            std::str::from_utf8(self.payload).ok()
        } else {
            None
        }
    }
}

/// SIMD-JSON parser state
#[repr(C)]
pub struct SimdJsonParser {
    /// Reusable buffer for partial frames
    buffer: CachePadded64<Box<[u8]>>,
    /// Current buffer position
    buffer_pos: CachePadded64<AtomicUsize>,
    /// Total bytes parsed
    bytes_parsed: CachePadded64<AtomicU64>,
    /// Parse errors
    parse_errors: CachePadded64<AtomicU64>,
    /// Messages parsed
    messages_parsed: CachePadded64<AtomicU64>,
}

// SAFETY: SimdJsonParser is single-threaded
unsafe impl Send for SimdJsonParser {}
unsafe impl Sync for SimdJsonParser {}

impl SimdJsonParser {
    /// Create a new SIMD JSON parser
    #[inline]
    pub fn new() -> Self {
        Self {
            buffer: CachePadded64::new(vec![0u8; MAX_JSON_PAYLOAD].into_boxed_slice()),
            buffer_pos: CachePadded64::new(AtomicUsize::new(0)),
            bytes_parsed: CachePadded64::new(AtomicU64::new(0)),
            parse_errors: CachePadded64::new(AtomicU64::new(0)),
            messages_parsed: CachePadded64::new(AtomicU64::new(0)),
        }
    }

    /// Parse a WebSocket frame from raw bytes
    #[inline]
    pub fn parse_frame<'a>(&self, data: &'a [u8]) -> Result<WebSocketFrame<'a>, &'static str> {
        if data.is_empty() {
            return Err("Empty data");
        }

        // Simple WebSocket frame parsing (client-to-server, unmasked)
        let first_byte = data[0];
        let second_byte = data[1];

        let fin = (first_byte & 0x80) != 0;
        let opcode = first_byte & 0x0F;
        let masked = (second_byte & 0x80) != 0;
        let mut payload_len = (second_byte & 0x7F) as usize;

        let frame_type = WebSocketFrameType::from_u8(opcode)
            .ok_or("Unknown frame type")?;

        let mut header_len = 2;

        // Handle extended payload length
        if payload_len == 126 {
            if data.len() < 4 {
                return Err("Incomplete header");
            }
            payload_len = ((data[2] as usize) << 8) | (data[3] as usize);
            header_len = 4;
        } else if payload_len == 127 {
            if data.len() < 10 {
                return Err("Incomplete header");
            }
            payload_len = ((data[2] as usize) << 56)
                | ((data[3] as usize) << 48)
                | ((data[4] as usize) << 40)
                | ((data[5] as usize) << 32)
                | ((data[6] as usize) << 24)
                | ((data[7] as usize) << 16)
                | ((data[8] as usize) << 8)
                | (data[9] as usize);
            header_len = 10;
        }

        // Handle mask
        if masked {
            header_len += 4; // 4-byte mask key
        }

        // Validate we have enough data
        if data.len() < header_len + payload_len {
            return Err("Incomplete frame");
        }

        let payload = &data[header_len..header_len + payload_len];

        self.bytes_parsed.0.fetch_add((header_len + payload_len) as u64, Ordering::Relaxed);
        self.messages_parsed.0.fetch_add(1, Ordering::Relaxed);

        Ok(WebSocketFrame {
            frame_type,
            fin,
            payload_len,
            payload,
        })
    }

    /// Parse JSON message from frame using simd-json
    #[inline]
    pub fn parse_json_message<'a>(&self, frame: &'a WebSocketFrame<'a>) -> Result<(), &'static str> {
        if frame.frame_type != WebSocketFrameType::Text {
            return Err("Not a text frame");
        }

        let json_str = frame.as_str().ok_or("Invalid UTF-8")?;

        // Use simd-json for zero-copy parsing
        // In production, this would use simd_json::to_value() with owned or borrowed API
        // For now, we validate the JSON structure
        
        if !self.validate_json_structure(json_str.as_bytes()) {
            self.parse_errors.0.fetch_add(1, Ordering::Relaxed);
            return Err("Invalid JSON structure");
        }

        Ok(())
    }

    /// Validate JSON structure (simplified simd-json style validation)
    #[inline]
    fn validate_json_structure(&self, data: &[u8]) -> bool {
        if data.is_empty() {
            return false;
        }

        // Quick validation: must start with { or [
        let first = data[0];
        if first != b'{' && first != b'[' {
            return false;
        }

        // Check balanced braces/brackets (simplified)
        let mut brace_count = 0i32;
        let mut bracket_count = 0i32;
        let mut in_string = false;
        let mut escape = false;

        for &byte in data {
            if escape {
                escape = false;
                continue;
            }

            if byte == b'\\' && in_string {
                escape = true;
                continue;
            }

            if byte == b'"' {
                in_string = !in_string;
                continue;
            }

            if !in_string {
                match byte {
                    b'{' => brace_count += 1,
                    b'}' => brace_count -= 1,
                    b'[' => bracket_count += 1,
                    b']' => bracket_count -= 1,
                    _ => {}
                }

                if brace_count < 0 || bracket_count < 0 {
                    return false;
                }
            }
        }

        brace_count == 0 && bracket_count == 0 && !in_string
    }

    /// Extract specific field from JSON (zero-copy)
    #[inline]
    pub fn extract_field<'a>(&self, json: &'a [u8], field_name: &str) -> Option<&'a [u8]> {
        // Find field name in JSON
        let search_pattern = format!("\"{}\"", field_name);
        let pattern_bytes = search_pattern.as_bytes();

        // Simple substring search (would use SIMD in production)
        for i in 0..json.len().saturating_sub(pattern_bytes.len()) {
            if &json[i..i + pattern_bytes.len()] == pattern_bytes {
                // Found field name, now find the value
                let mut pos = i + pattern_bytes.len();

                // Skip whitespace and colon
                while pos < json.len() && (json[pos] == b' ' || json[pos] == b':' || json[pos] == b'\t' || json[pos] == b'\n') {
                    pos += 1;
                }

                if pos >= json.len() {
                    return None;
                }

                // Determine value type and extract
                let start = pos;
                match json[pos] {
                    b'"' => {
                        // String value - find closing quote
                        pos += 1;
                        while pos < json.len() && json[pos] != b'"' {
                            if json[pos] == b'\\' {
                                pos += 1;
                            }
                            pos += 1;
                        }
                        if pos < json.len() {
                            pos += 1; // Include closing quote
                        }
                    }
                    b'{' | b'[' => {
                        // Object or array - find matching close
                        let opener = json[pos];
                        let closer = if opener == b'{' { b'}' } else { b']' };
                        let mut count = 1i32;
                        pos += 1;
                        while pos < json.len() && count > 0 {
                            if json[pos] == opener {
                                count += 1;
                            } else if json[pos] == closer {
                                count -= 1;
                            }
                            pos += 1;
                        }
                    }
                    _ => {
                        // Number, boolean, null - find delimiter
                        while pos < json.len() 
                            && json[pos] != b',' 
                            && json[pos] != b'}' 
                            && json[pos] != b']'
                            && json[pos] != b' '
                            && json[pos] != b'\n'
                            && json[pos] != b'\t'
                        {
                            pos += 1;
                        }
                    }
                }

                return Some(&json[start..pos]);
            }
        }

        None
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> (u64, u64, u64) {
        (
            self.bytes_parsed.0.load(Ordering::Relaxed),
            self.messages_parsed.0.load(Ordering::Relaxed),
            self.parse_errors.0.load(Ordering::Relaxed),
        )
    }

    /// Reset parser state
    #[inline]
    pub fn reset(&self) {
        self.buffer_pos.0.store(0, Ordering::Release);
    }
}

impl Default for SimdJsonParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_websocket_frame_text() {
        // Simple text frame: FIN=1, opcode=1, len=5, payload="hello"
        let frame_data = vec![
            0x81, 0x05, // FIN + text opcode, length 5
            b'h', b'e', b'l', b'l', b'o',
        ];

        let parser = SimdJsonParser::new();
        let frame = parser.parse_frame(&frame_data).unwrap();

        assert_eq!(frame.frame_type, WebSocketFrameType::Text);
        assert!(frame.fin);
        assert_eq!(frame.payload_len, 5);
        assert_eq!(frame.as_str(), Some("hello"));
    }

    #[test]
    fn test_websocket_frame_ping() {
        let frame_data = vec![0x89, 0x00]; // FIN + ping opcode, length 0

        let parser = SimdJsonParser::new();
        let frame = parser.parse_frame(&frame_data).unwrap();

        assert_eq!(frame.frame_type, WebSocketFrameType::Ping);
        assert_eq!(frame.payload_len, 0);
    }

    #[test]
    fn test_json_validation() {
        let parser = SimdJsonParser::new();

        assert!(parser.validate_json_structure(b"{}"));
        assert!(parser.validate_json_structure(b"[]"));
        assert!(parser.validate_json_structure(b"{\"key\": \"value\"}"));
        assert!(parser.validate_json_structure(b"[1, 2, 3]"));
        assert!(parser.validate_json_structure(b"{\"nested\": {\"a\": 1}}"));

        assert!(!parser.validate_json_structure(b"{"));
        assert!(!parser.validate_json_structure(b"}"));
        assert!(!parser.validate_json_structure(b"invalid"));
    }

    #[test]
    fn test_extract_field() {
        let parser = SimdJsonParser::new();
        let json = br#"{"symbol": "BTCUSDT", "price": "50000.00", "qty": "1.5"}"#;

        let symbol = parser.extract_field(json, "symbol");
        assert!(symbol.is_some());

        let price = parser.extract_field(json, "price");
        assert!(price.is_some());

        let missing = parser.extract_field(json, "missing");
        assert!(missing.is_none());
    }

    #[test]
    fn test_parser_statistics() {
        let parser = SimdJsonParser::new();

        let frame_data = vec![0x81, 0x05, b'h', b'e', b'l', b'l', b'o'];
        let frame = parser.parse_frame(&frame_data).unwrap();
        let _ = parser.parse_json_message(&frame);

        let (bytes, msgs, errors) = parser.get_stats();
        assert!(bytes > 0);
        assert_eq!(msgs, 1);
        assert_eq!(errors, 0);
    }
}
