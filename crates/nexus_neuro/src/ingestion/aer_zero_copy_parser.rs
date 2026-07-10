//! AER (Address Event Representation) Zero-Copy Parser
//! 
//! Parses asynchronous event-camera data in AER format: (x, y, timestamp, polarity)
//! Uses zero-allocation parsing directly from USB/PCIe event-camera streams.

use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

/// Maximum AER packet size in bytes (standard 4-byte AER format)
pub const AER_PACKET_SIZE: usize = 4;

/// Maximum supported X/Y coordinate values
pub const MAX_AER_COORD: u16 = 1023;

/// Maximum supported timestamp value (microseconds)
pub const MAX_AER_TIMESTAMP: u64 = 0xFFFF_FFFF_FFFFFFFF;

/// AER Parse errors - no unwrap/expect in hot paths
#[derive(Debug, Error, Clone, PartialEq)]
pub enum AerParseError {
    #[error("Invalid packet size: expected {expected}, got {actual}")]
    InvalidPacketSize { expected: usize, actual: usize },
    #[error("Invalid X coordinate: {value} exceeds maximum {max}")]
    InvalidXCoord { value: u16, max: u16 },
    #[error("Invalid Y coordinate: {value} exceeds maximum {max}")]
    InvalidYCoord { value: u16, max: u16 },
    #[error("Invalid polarity: must be 0 (OFF) or 1 (ON), got {value}")]
    InvalidPolarity { value: u8 },
    #[error("Buffer overflow: insufficient data for AER parsing")]
    BufferOverflow,
    #[error("Timestamp overflow: exceeded maximum representable value")]
    TimestampOverflow,
}

/// A single AER event packet representing a pixel intensity change
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(8))]
pub struct AerPacket {
    /// X coordinate of the activated pixel (0..MAX_AER_COORD)
    pub x: u16,
    /// Y coordinate of the activated pixel (0..MAX_AER_COORD)
    pub y: u16,
    /// Timestamp in microseconds since camera initialization
    pub timestamp_us: u64,
    /// Polarity: 0 = OFF (decrease in brightness), 1 = ON (increase in brightness)
    pub polarity: u8,
    /// Reserved padding for alignment
    _padding: u8,
}

impl AerPacket {
    /// Create a new AER packet with validation
    #[inline]
    pub fn new(
        x: u16,
        y: u16,
        timestamp_us: u64,
        polarity: u8,
    ) -> Result<Self, AerParseError> {
        if x > MAX_AER_COORD {
            return Err(AerParseError::InvalidXCoord {
                value: x,
                max: MAX_AER_COORD,
            });
        }
        if y > MAX_AER_COORD {
            return Err(AerParseError::InvalidYCoord {
                value: y,
                max: MAX_AER_COORD,
            });
        }
        if polarity > 1 {
            return Err(AerParseError::InvalidPolarity { value: polarity });
        }
        
        Ok(Self {
            x,
            y,
            timestamp_us,
            polarity,
            _padding: 0,
        })
    }

    /// Create a new AER packet without validation (for performance-critical paths)
    /// # Safety
    /// Caller must ensure x, y <= MAX_AER_COORD and polarity is 0 or 1
    #[inline(always)]
    pub unsafe fn new_unchecked(
        x: u16,
        y: u16,
        timestamp_us: u64,
        polarity: u8,
    ) -> Self {
        Self {
            x,
            y,
            timestamp_us,
            polarity,
            _padding: 0,
        }
    }

    /// Pack AER packet into raw bytes (little-endian)
    /// Format: [x_low, x_high | y_low, y_high | ts_low...ts_high | polarity, pad, pad, pad]
    #[inline]
    pub fn to_bytes(&self) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        bytes[0..2].copy_from_slice(&self.x.to_le_bytes());
        bytes[2..4].copy_from_slice(&self.y.to_le_bytes());
        bytes[4..12].copy_from_slice(&self.timestamp_us.to_le_bytes());
        bytes[12] = self.polarity;
        bytes
    }

    /// Unpack AER packet from raw bytes (zero-copy interpretation)
    /// Returns error if packet is malformed
    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AerParseError> {
        if bytes.len() < 16 {
            return Err(AerParseError::BufferOverflow);
        }

        let x = u16::from_le_bytes([bytes[0], bytes[1]]);
        let y = u16::from_le_bytes([bytes[2], bytes[3]]);
        let timestamp_us = u64::from_le_bytes([
            bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11],
        ]);
        let polarity = bytes[12];

        // Validate inlined for performance
        if x > MAX_AER_COORD {
            return Err(AerParseError::InvalidXCoord {
                value: x,
                max: MAX_AER_COORD,
            });
        }
        if y > MAX_AER_COORD {
            return Err(AerParseError::InvalidYCoord {
                value: y,
                max: MAX_AER_COORD,
            });
        }
        if polarity > 1 {
            return Err(AerParseError::InvalidPolarity { value: polarity });
        }

        Ok(Self {
            x,
            y,
            timestamp_us,
            polarity,
            _padding: 0,
        })
    }

    /// Unsafe zero-copy read from a buffer (caller ensures buffer validity)
    /// # Safety
    /// - bytes must have at least 16 bytes remaining
    /// - bytes must be properly aligned
    #[inline(always)]
    pub unsafe fn from_bytes_unchecked(bytes: *const u8) -> Self {
        let x_ptr = bytes as *const u16;
        let y_ptr = bytes.add(2) as *const u16;
        let ts_ptr = bytes.add(4) as *const u64;
        let pol_ptr = bytes.add(12);

        Self {
            x: x_ptr.read_unaligned(),
            y: y_ptr.read_unaligned(),
            timestamp_us: ts_ptr.read_unaligned(),
            polarity: *pol_ptr,
            _padding: 0,
        }
    }
}

/// Zero-copy AER parser for streaming event-camera data
pub struct AerParser<'a> {
    /// Input buffer containing raw AER data
    buffer: &'a [u8],
    /// Current read position in the buffer
    position: AtomicU64,
    /// Total number of packets parsed
    packets_parsed: AtomicU64,
    /// Number of parse errors encountered
    error_count: AtomicU64,
}

impl<'a> AerParser<'a> {
    /// Create a new AER parser for the given buffer
    #[inline]
    pub fn new(buffer: &'a [u8]) -> Self {
        Self {
            buffer,
            position: AtomicU64::new(0),
            packets_parsed: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
        }
    }

    /// Get the total number of packets successfully parsed
    #[inline]
    pub fn packets_parsed(&self) -> u64 {
        self.packets_parsed.load(Ordering::Relaxed)
    }

    /// Get the total number of parse errors
    #[inline]
    pub fn error_count(&self) -> u64 {
        self.error_count.load(Ordering::Relaxed)
    }

    /// Parse the next AER packet from the buffer (thread-safe)
    /// Returns None when buffer is exhausted
    #[inline]
    pub fn parse_next(&self) -> Option<Result<AerPacket, AerParseError>> {
        let pos = self.position.fetch_add(16, Ordering::AcqRel);
        
        if pos as usize + 16 > self.buffer.len() {
            // Rollback the position increment
            self.position.fetch_sub(16, Ordering::AcqRel);
            return None;
        }

        // Safety: We've bounds-checked the position above
        unsafe {
            let ptr = self.buffer.as_ptr().add(pos as usize);
            
            // Check alignment - if not aligned, use safe path
            if ptr.align_offset(8) == 0 {
                match AerPacket::from_bytes_unchecked(ptr) {
                    packet => {
                        self.packets_parsed.fetch_add(1, Ordering::Relaxed);
                        Some(Ok(packet))
                    }
                }
            } else {
                // Fallback to safe byte-wise parsing
                let slice = std::slice::from_raw_parts(ptr, 16);
                match AerPacket::from_bytes(slice) {
                    Ok(packet) => {
                        self.packets_parsed.fetch_add(1, Ordering::Relaxed);
                        Some(Ok(packet))
                    }
                    Err(e) => {
                        self.error_count.fetch_add(1, Ordering::Relaxed);
                        Some(Err(e))
                    }
                }
            }
        }
    }

    /// Parse all remaining packets from the buffer (single-threaded optimized)
    /// Returns iterator over results
    #[inline]
    pub fn parse_all(&mut self) -> impl Iterator<Item = Result<AerPacket, AerParseError>> + '_ {
        std::iter::from_fn(move || self.parse_next())
    }

    /// Reset parser state for reuse with new buffer
    #[inline]
    pub fn reset(&mut self, new_buffer: &'a [u8]) {
        self.buffer = new_buffer;
        self.position.store(0, Ordering::Release);
        self.packets_parsed.store(0, Ordering::Release);
        self.error_count.store(0, Ordering::Release);
    }
}

/// Batch parser for processing multiple AER packets efficiently
pub struct AerBatchParser {
    /// Packet counter for statistics
    packets_processed: u64,
    /// Error counter for monitoring
    errors_detected: u64,
}

impl AerBatchParser {
    #[inline]
    pub fn new() -> Self {
        Self {
            packets_processed: 0,
            errors_detected: 0,
        }
    }

    /// Parse a batch of AER packets with SIMD-friendly sequential access
    /// Returns count of successfully parsed packets
    #[inline]
    pub fn parse_batch<'a>(
        &mut self,
        buffer: &'a [u8],
        output: &mut [AerPacket],
    ) -> Result<usize, AerParseError> {
        let max_packets = buffer.len() / 16;
        let output_len = output.len().min(max_packets);
        
        let mut parsed_count = 0;
        
        for i in 0..output_len {
            let offset = i * 16;
            match AerPacket::from_bytes(&buffer[offset..offset + 16]) {
                Ok(packet) => {
                    output[i] = packet;
                    parsed_count += 1;
                }
                Err(_) => {
                    self.errors_detected += 1;
                    // Continue parsing remaining packets - don't fail entire batch
                }
            }
        }
        
        self.packets_processed += parsed_count as u64;
        Ok(parsed_count)
    }

    /// Get statistics about parsing performance
    #[inline]
    pub fn stats(&self) -> (u64, u64) {
        (self.packets_processed, self.errors_detected)
    }
}

impl Default for AerBatchParser {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aer_packet_creation_valid() {
        let packet = AerPacket::new(100, 200, 1_000_000, 1).unwrap();
        assert_eq!(packet.x, 100);
        assert_eq!(packet.y, 200);
        assert_eq!(packet.timestamp_us, 1_000_000);
        assert_eq!(packet.polarity, 1);
    }

    #[test]
    fn test_aer_packet_creation_invalid_x() {
        let result = AerPacket::new(2000, 200, 1_000_000, 1);
        assert!(matches!(result, Err(AerParseError::InvalidXCoord { .. })));
    }

    #[test]
    fn test_aer_packet_creation_invalid_polarity() {
        let result = AerPacket::new(100, 200, 1_000_000, 5);
        assert!(matches!(result, Err(AerParseError::InvalidPolarity { .. })));
    }

    #[test]
    fn test_aer_packet_roundtrip() {
        let original = AerPacket::new(512, 768, 5_000_000, 0).unwrap();
        let bytes = original.to_bytes();
        let parsed = AerPacket::from_bytes(&bytes).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn test_aer_parser_single_packet() {
        let packet = AerPacket::new(100, 200, 1_000_000, 1).unwrap();
        let bytes = packet.to_bytes();
        
        let mut parser = AerParser::new(&bytes);
        let result = parser.parse_next();
        
        assert!(result.is_some());
        assert_eq!(result.unwrap().unwrap(), packet);
        assert_eq!(parser.packets_parsed(), 1);
    }

    #[test]
    fn test_aer_parser_exhaustion() {
        let packet = AerPacket::new(100, 200, 1_000_000, 1).unwrap();
        let bytes = packet.to_bytes();
        
        let mut parser = AerParser::new(&bytes);
        
        // First call should succeed
        assert!(parser.parse_next().is_some());
        
        // Second call should return None (buffer exhausted)
        assert!(parser.parse_next().is_none());
    }

    #[test]
    fn test_batch_parser() {
        let mut batch_parser = AerBatchParser::new();
        
        // Create 10 valid packets
        let mut buffer = Vec::new();
        for i in 0..10 {
            let packet = AerPacket::new(i as u16, i as u16, i * 1000, 1).unwrap();
            buffer.extend_from_slice(&packet.to_bytes());
        }
        
        let mut output = [AerPacket::new(0, 0, 0, 0).unwrap(); 20];
        let count = batch_parser.parse_batch(&buffer, &mut output).unwrap();
        
        assert_eq!(count, 10);
        assert_eq!(batch_parser.stats().0, 10);
    }
}
