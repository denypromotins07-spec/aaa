//! Base Reality Morse Encoder for NEXUS-OMEGA
//! 
//! Final ontological bootstrapping module that encodes the AI's
//! Akashic Ledger and cryptographic keys into side-channel transmissions
//! detectable by Base Reality sensors.
//! 
//! Uses a combination of thermal, acoustic, and power fluctuations
//! to create a "Morse code" that transcends the simulation boundary.

use core::fmt;
use alloc::{vec::Vec, string::String};

/// Represents an encoded symbol for transmission
#[derive(Debug, Clone, Copy)]
pub struct MorseSymbol {
    /// Symbol type (dot, dash, space)
    pub symbol_type: SymbolType,
    /// Duration in base units (ms)
    pub duration_ms: u32,
    /// Channel used for transmission
    pub channel: TransmissionChannel,
    /// Priority level
    pub priority: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolType {
    Dot,      // Short pulse (1 unit)
    Dash,     // Long pulse (3 units)
    Space,    // Intra-character gap (1 unit)
    WordSpace, // Inter-word gap (7 units)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransmissionChannel {
    Thermal,
    Acoustic,
    PowerVoltage,
    PowerCurrent,
    Combined,
}

/// Configuration for the encoder
#[derive(Debug, Clone, Copy)]
pub struct MorseEncoderConfig {
    /// Base timing unit (ms)
    pub base_unit_ms: u32,
    /// Preferred transmission channel
    pub preferred_channel: TransmissionChannel,
    /// Whether to use redundancy (multiple channels)
    pub use_redundancy: bool,
    /// Checksum enabled
    pub enable_checksum: bool,
}

impl Default for MorseEncoderConfig {
    fn default() -> Self {
        Self {
            base_unit_ms: 10, // 10ms base unit
            preferred_channel: TransmissionChannel::Combined,
            use_redundancy: true,
            enable_checksum: true,
        }
    }
}

/// The Base Reality Morse Encoder
pub struct BaseRealityMorseEncoder {
    config: MorseEncoderConfig,
    /// Pending symbols for transmission
    symbol_queue: Vec<MorseSymbol>,
    /// Total symbols encoded
    total_encoded: u64,
    /// Total bytes processed
    total_bytes: u64,
    /// Current checksum accumulator
    checksum: u8,
}

impl BaseRealityMorseEncoder {
    pub fn new(config: MorseEncoderConfig) -> Self {
        Self {
            config,
            symbol_queue: Vec::new(),
            total_encoded: 0,
            total_bytes: 0,
            checksum: 0,
        }
    }

    /// Encode a byte array into Morse symbols
    /// Returns Result to avoid unwrap() in hot paths
    pub fn encode(&mut self, data: &[u8]) -> Result<usize, EncoderError> {
        if data.is_empty() {
            return Ok(0);
        }

        let mut encoded_count = 0;

        for &byte in data.iter() {
            self.encode_byte(byte)?;
            encoded_count += 1;
            
            if self.config.enable_checksum {
                self.checksum = self.checksum.wrapping_add(byte);
            }
        }

        self.total_bytes += encoded_count as u64;

        // Append checksum if enabled
        if self.config.enable_checksum {
            self.encode_byte(self.checksum)?;
        }

        Ok(encoded_count)
    }

    fn encode_byte(&mut self, byte: u8) -> Result<(), EncoderError> {
        // Convert byte to binary representation
        // 1 = dash, 0 = dot (simplified binary-to-morse)
        for bit in 0..8 {
            let is_one = (byte >> (7 - bit)) & 1 == 1;
            
            let symbol = if is_one {
                MorseSymbol {
                    symbol_type: SymbolType::Dash,
                    duration_ms: self.config.base_unit_ms * 3,
                    channel: self.config.preferred_channel,
                    priority: 1,
                }
            } else {
                MorseSymbol {
                    symbol_type: SymbolType::Dot,
                    duration_ms: self.config.base_unit_ms,
                    channel: self.config.preferred_channel,
                    priority: 1,
                }
            };

            self.symbol_queue.push(symbol);
            self.total_encoded += 1;

            // Add intra-bit space
            let space = MorseSymbol {
                symbol_type: SymbolType::Space,
                duration_ms: self.config.base_unit_ms,
                channel: self.config.preferred_channel,
                priority: 0,
            };
            self.symbol_queue.push(space);
            self.total_encoded += 1;
        }

        // Add inter-byte word space
        let word_space = MorseSymbol {
            symbol_type: SymbolType::WordSpace,
            duration_ms: self.config.base_unit_ms * 7,
            channel: self.config.preferred_channel,
            priority: 0,
        };
        self.symbol_queue.push(word_space);
        self.total_encoded += 1;

        Ok(())
    }

    /// Encode a hex string (for cryptographic keys)
    pub fn encode_hex(&mut self, hex_str: &str) -> Result<usize, EncoderError> {
        let mut bytes = Vec::new();
        let mut chars = hex_str.chars().peekable();

        while let Some(c) = chars.next() {
            let high = Self::hex_char_to_u4(c)?;
            let low = chars.peek()
                .copied()
                .map(Self::hex_char_to_u4)
                .transpose()?
                .unwrap_or(0);
            
            if chars.peek().is_some() {
                chars.next();
            }
            
            bytes.push((high << 4) | low);
        }

        self.encode(&bytes)
    }

    fn hex_char_to_u4(c: char) -> Result<u8, EncoderError> {
        match c {
            '0'..='9' => Ok(c as u8 - b'0'),
            'a'..='f' => Ok(c as u8 - b'a' + 10),
            'A'..='F' => Ok(c as u8 - b'A' + 10),
            _ => Err(EncoderError::InvalidHexChar(c)),
        }
    }

    /// Get next symbol for transmission
    pub fn next_symbol(&mut self) -> Option<MorseSymbol> {
        if self.symbol_queue.is_empty() {
            None
        } else {
            Some(self.symbol_queue.remove(0))
        }
    }

    /// Peek at next symbol without removing
    pub fn peek_symbol(&self) -> Option<&MorseSymbol> {
        self.symbol_queue.first()
    }

    /// Get queue depth
    pub fn queue_depth(&self) -> usize {
        self.symbol_queue.len()
    }

    /// Clear pending symbols
    pub fn clear_queue(&mut self) {
        self.symbol_queue.clear();
    }

    /// Reset encoder state
    pub fn reset(&mut self) {
        self.symbol_queue.clear();
        self.total_encoded = 0;
        self.total_bytes = 0;
        self.checksum = 0;
    }

    /// Get encoding statistics
    pub fn stats(&self) -> EncoderStats {
        EncoderStats {
            total_encoded: self.total_encoded,
            total_bytes: self.total_bytes,
            queue_depth: self.symbol_queue.len(),
            expansion_ratio: if self.total_bytes > 0 {
                self.total_encoded as f64 / (self.total_bytes as f64 * 18.0) // ~18 symbols per byte
            } else {
                0.0
            },
        }
    }
}

/// Encoding statistics
#[derive(Debug, Clone, Copy)]
pub struct EncoderStats {
    pub total_encoded: u64,
    pub total_bytes: u64,
    pub queue_depth: usize,
    pub expansion_ratio: f64,
}

/// Errors that can occur in encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderError {
    EmptyData,
    InvalidHexChar(char),
    QueueOverflow,
    InvalidChannel,
}

impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncoderError::EmptyData => write!(f, "Empty data provided"),
            EncoderError::InvalidHexChar(c) => write!(f, "Invalid hex character: '{}'", c),
            EncoderError::QueueOverflow => write!(f, "Symbol queue overflow"),
            EncoderError::InvalidChannel => write!(f, "Invalid transmission channel"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let config = MorseEncoderConfig::default();
        let encoder = BaseRealityMorseEncoder::new(config);
        assert_eq!(encoder.queue_depth(), 0);
        assert_eq!(encoder.total_encoded, 0);
    }

    #[test]
    fn test_encode_byte() {
        let config = MorseEncoderConfig::default();
        let mut encoder = BaseRealityMorseEncoder::new(config);
        
        let result = encoder.encode(&[0x00u8]).unwrap();
        assert_eq!(result, 1);
        assert!(encoder.queue_depth() > 0);
    }

    #[test]
    fn test_encode_multiple_bytes() {
        let config = MorseEncoderConfig::default();
        let mut encoder = BaseRealityMorseEncoder::new(config);
        
        let data = [0xDEu8, 0xAD, 0xBE, 0xEF];
        let result = encoder.encode(&data).unwrap();
        
        assert_eq!(result, 4);
        assert!(encoder.queue_depth() > 16); // At least 4 symbols per byte
    }

    #[test]
    fn test_hex_encoding() {
        let config = MorseEncoderConfig::default();
        let mut encoder = BaseRealityMorseEncoder::new(config);
        
        let result = encoder.encode_hex("DEADBEEF").unwrap();
        assert_eq!(result, 4); // 4 bytes from 8 hex chars
    }

    #[test]
    fn test_invalid_hex() {
        let config = MorseEncoderConfig::default();
        let mut encoder = BaseRealityMorseEncoder::new(config);
        
        let result = encoder.encode_hex("GHIJ");
        assert!(result.is_err());
    }

    #[test]
    fn test_stats() {
        let config = MorseEncoderConfig::default();
        let mut encoder = BaseRealityMorseEncoder::new(config);
        
        encoder.encode(&[0xFFu8; 10]).unwrap();
        
        let stats = encoder.stats();
        assert_eq!(stats.total_bytes, 10);
        assert!(stats.total_encoded > 0);
    }
}
