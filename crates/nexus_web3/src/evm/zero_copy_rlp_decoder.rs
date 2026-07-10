//! Zero-Copy RLP Decoder for Ethereum Mempool Transactions
//! 
//! Parses Recursive Length Prefix (RLP) encoded data directly from WebSocket buffers
//! without allocating intermediate strings or byte vectors. Enforces strict size limits
//! to prevent DoS attacks from malformed transactions.

use thiserror::Error;
use alloc::vec::Vec;
use core::slice;

/// Maximum allowed RLP payload size (10MB safety limit)
const MAX_RLP_SIZE: usize = 10 * 1024 * 1024;

/// Maximum string/bytes length in RLP structure
const MAX_STRING_LENGTH: usize = 32 * 1024 * 1024;

/// Maximum list items to prevent stack overflow
const MAX_LIST_ITEMS: usize = 1024;

#[derive(Error, Debug, PartialEq)]
pub enum RlpDecodeError {
    #[error("Buffer too short: needed {needed} bytes at position {pos}")]
    BufferTooShort { pos: usize, needed: usize },
    #[error("Invalid RLP prefix: {prefix:#x} at position {pos}")]
    InvalidPrefix { pos: usize, prefix: u8 },
    #[error("Payload exceeds maximum size: {size} > {max}")]
    PayloadTooLarge { size: usize, max: usize },
    #[error("String length overflow: {len} bytes")]
    StringLengthOverflow { len: usize },
    #[error("List item count exceeded: {count} > {max}")]
    TooManyListItems { count: usize, max: usize },
    #[error("Invalid length encoding: declared {declared} but only {available} remaining")]
    InvalidLength { declared: usize, available: usize },
    #[error("Non-canonical length encoding detected")]
    NonCanonicalLength,
    #[error("Trailing data after RLP structure")]
    TrailingData,
}

pub type Result<T> = core::result::Result<T, RlpDecodeError>;

/// RLP type tag
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RlpType {
    EmptyString,
    String(usize),      // length
    List(usize),        // payload length
}

/// Zero-copy RLP value reference
#[derive(Debug, Clone, Copy)]
pub struct RlpValueRef<'a> {
    data: &'a [u8],
    offset: usize,
    r#type: RlpType,
}

impl<'a> RlpValueRef<'a> {
    /// Get the raw bytes of this value (zero-copy slice)
    pub fn as_bytes(&self) -> &'a [u8] {
        match self.r#type {
            RlpType::EmptyString => &[],
            RlpType::String(len) => {
                let start = self.offset + self.header_size();
                &self.data[start..start + len]
            }
            RlpType::List(_) => {
                // For lists, return the entire payload (excluding header)
                let start = self.offset + self.header_size();
                let len = if let RlpType::List(l) = self.r#type { l } else { 0 };
                &self.data[start..start + len]
            }
        }
    }

    /// Calculate header size in bytes
    const fn header_size(&self) -> usize {
        match self.r#type {
            RlpType::EmptyString => 1,
            RlpType::String(len) => {
                if len < 56 { 1 } else { 1 + length_of_length(len) }
            }
            RlpType::List(len) => {
                if len < 56 { 1 } else { 1 + length_of_length(len) }
            }
        }
    }

    /// Parse as u256 (big-endian integer)
    pub fn as_u256(&self) -> Result<[u8; 32]> {
        let bytes = self.as_bytes();
        if bytes.is_empty() {
            return Ok([0u8; 32]);
        }
        if bytes.len() > 32 {
            return Err(RlpDecodeError::StringLengthOverflow { len: bytes.len() });
        }
        
        let mut result = [0u8; 32];
        result[32 - bytes.len()..].copy_from_slice(bytes);
        Ok(result)
    }

    /// Parse as u64 (big-endian)
    pub fn as_u64(&self) -> Result<u64> {
        let bytes = self.as_bytes();
        if bytes.is_empty() {
            return Ok(0);
        }
        if bytes.len() > 8 {
            return Err(RlpDecodeError::StringLengthOverflow { len: bytes.len() });
        }
        
        let mut result = 0u64;
        for &byte in bytes {
            result = result.checked_shl(8).ok_or(
                RlpDecodeError::StringLengthOverflow { len: bytes.len() }
            )? | (byte as u64);
        }
        Ok(result)
    }

    /// Get the type of this value
    pub const fn r#type(&self) -> RlpType {
        self.r#type
    }

    /// Get total size including header
    pub fn total_size(&self) -> usize {
        self.header_size() + match self.r#type {
            RlpType::EmptyString => 0,
            RlpType::String(len) | RlpType::List(len) => len,
        }
    }
}

/// Calculate bytes needed to encode a length
const fn length_of_length(len: usize) -> usize {
    if len < (1 << 8) { 1 }
    else if len < (1 << 16) { 2 }
    else if len < (1 << 24) { 3 }
    else { 4 }
}

/// Zero-Copy RLP Decoder
/// 
/// Parses RLP structures directly from input buffers without allocations.
/// Enforces strict bounds checking to prevent DoS attacks.
pub struct ZeroCopyRlpDecoder<'a> {
    data: &'a [u8],
    position: usize,
    depth: usize,
    max_depth: usize,
}

impl<'a> ZeroCopyRlpDecoder<'a> {
    /// Create a new decoder for the given buffer
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            position: 0,
            depth: 0,
            max_depth: 32,
        }
    }

    /// Set maximum recursion depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Check if more data is available
    pub fn has_remaining(&self) -> bool {
        self.position < self.data.len()
    }

    /// Get remaining bytes
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }

    /// Peek at the next byte without advancing
    fn peek_byte(&self) -> Result<u8> {
        if self.position >= self.data.len() {
            return Err(RlpDecodeError::BufferTooShort {
                pos: self.position,
                needed: 1,
            });
        }
        Ok(self.data[self.position])
    }

    /// Read a byte and advance position
    fn read_byte(&mut self) -> Result<u8> {
        let byte = self.peek_byte()?;
        self.position += 1;
        Ok(byte)
    }

    /// Read bytes without allocation (returns slice reference)
    fn read_slice(&mut self, len: usize) -> Result<&'a [u8]> {
        if len > MAX_STRING_LENGTH {
            return Err(RlpDecodeError::StringLengthOverflow { len });
        }
        
        let end = self.position.checked_add(len).ok_or(
            RlpDecodeError::BufferTooShort { pos: self.position, needed: len }
        )?;
        
        if end > self.data.len() {
            return Err(RlpDecodeError::BufferTooShort {
                pos: self.position,
                needed: len,
            });
        }
        
        let slice = &self.data[self.position..end];
        self.position = end;
        Ok(slice)
    }

    /// Decode length prefix for strings/lists > 55 bytes
    fn decode_long_length(&mut self, prefix: u8, offset: u8) -> Result<usize> {
        let length_bytes = (prefix - offset) as usize;
        
        if length_bytes == 0 || length_bytes > 4 {
            return Err(RlpDecodeError::InvalidPrefix {
                pos: self.position - 1,
                prefix,
            });
        }
        
        let mut length: usize = 0;
        for i in 0..length_bytes {
            let byte = self.read_byte()?;
            if i == 0 && byte == 0 {
                // Non-canonical encoding (leading zeros)
                return Err(RlpDecodeError::NonCanonicalLength);
            }
            length = length.checked_shl(8).ok_or(
                RlpDecodeError::StringLengthOverflow { len: usize::MAX }
            )? | (byte as usize);
        }
        
        // Validate minimum length for long format
        if length < 56 {
            return Err(RlpDecodeError::NonCanonicalLength);
        }
        
        Ok(length)
    }

    /// Parse the next RLP value (zero-copy)
    pub fn parse_next(&mut self) -> Result<RlpValueRef<'a>> {
        if self.depth >= self.max_depth {
            return Err(RlpDecodeError::TooManyListItems {
                count: self.depth,
                max: self.max_depth,
            });
        }

        let start_pos = self.position;
        let prefix = self.read_byte()?;

        let r#type = match prefix {
            0x00..=0x7f => RlpType::EmptyString,
            0x01..=0x7f => RlpType::String((prefix - 0x00) as usize),
            0x80 => RlpType::EmptyString,
            0x81..=0xb7 => RlpType::String((prefix - 0x80) as usize),
            0xb8..=0xbf => {
                let len = self.decode_long_length(prefix, 0xb8)?;
                RlpType::String(len)
            }
            0xc0..=0xf7 => {
                let len = (prefix - 0xc0) as usize;
                self.depth += 1;
                RlpType::List(len)
            }
            0xf8..=0xff => {
                let len = self.decode_long_length(prefix, 0xf8)?;
                self.depth += 1;
                RlpType::List(len)
            }
        };

        // Validate payload size
        if let RlpType::String(len) | RlpType::List(len) = r#type {
            if len > MAX_RLP_SIZE {
                return Err(RlpDecodeError::PayloadTooLarge {
                    size: len,
                    max: MAX_RLP_SIZE,
                });
            }
            
            // Ensure we have enough data
            if len > self.remaining() {
                return Err(RlpDecodeError::InvalidLength {
                    declared: len,
                    available: self.remaining(),
                });
            }
        }

        Ok(RlpValueRef {
            data: self.data,
            offset: start_pos,
            r#type,
        })
    }

    /// Parse entire transaction structure
    pub fn parse_transaction(&mut self) -> Result<TransactionFields<'a>> {
        let value = self.parse_next()?;
        
        // Transaction should be a list
        if let RlpType::List(_) = value.r#type {
            // Create sub-decoder for list contents
            let payload = value.as_bytes();
            let mut list_decoder = ZeroCopyRlpDecoder::new(payload);
            
            // Parse standard EIP-155 or legacy transaction fields
            let nonce = list_decoder.parse_next()?.as_u64()?;
            let gas_price = list_decoder.parse_next()?.as_u256()?;
            let gas_limit = list_decoder.parse_next()?.as_u256()?;
            let to = list_decoder.parse_next()?; // Address or empty for contract creation
            let value = list_decoder.parse_next()?.as_u256()?;
            let input = list_decoder.parse_next()?;
            
            // Optional fields for EIP-155
            let chain_id = if list_decoder.has_remaining() {
                Some(list_decoder.parse_next()?.as_u64()?)
            } else {
                None
            };
            
            let v = if list_decoder.has_remaining() {
                Some(list_decoder.parse_next()?.as_u64()?)
            } else {
                None
            };
            
            let r = if list_decoder.has_remaining() {
                Some(list_decoder.parse_next()?.as_u256()?)
            } else {
                None
            };
            
            let s = if list_decoder.has_remaining() {
                Some(list_decoder.parse_next()?.as_u256()?)
            } else {
                None
            };

            Ok(TransactionFields {
                nonce,
                gas_price,
                gas_limit,
                to,
                value,
                input: input.as_bytes(),
                chain_id,
                v,
                r,
                s,
            })
        } else {
            Err(RlpDecodeError::InvalidPrefix {
                pos: 0,
                prefix: 0,
            })
        }
    }
}

/// Parsed transaction fields (zero-copy references where possible)
pub struct TransactionFields<'a> {
    pub nonce: u64,
    pub gas_price: [u8; 32],
    pub gas_limit: [u8; 32],
    pub to: RlpValueRef<'a>,
    pub value: [u8; 32],
    pub input: &'a [u8],
    pub chain_id: Option<u64>,
    pub v: Option<u64>,
    pub r: Option<[u8; 32]>,
    pub s: Option<[u8; 32]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_empty_string() {
        let data = [0x80]; // Empty string
        let mut decoder = ZeroCopyRlpDecoder::new(&data);
        let value = decoder.parse_next().unwrap();
        
        assert_eq!(value.r#type(), RlpType::EmptyString);
        assert_eq!(value.as_bytes(), &[]);
    }

    #[test]
    fn test_decode_short_string() {
        let data = [0x83, b'd', b'o', b'g']; // "dog"
        let mut decoder = ZeroCopyRlpDecoder::new(&data);
        let value = decoder.parse_next().unwrap();
        
        assert_eq!(value.r#type(), RlpType::String(3));
        assert_eq!(value.as_bytes(), b"dog");
    }

    #[test]
    fn test_decode_positive_integer() {
        let data = [0x02]; // Number 2
        let mut decoder = ZeroCopyRlpDecoder::new(&data);
        let value = decoder.parse_next().unwrap();
        
        assert_eq!(value.as_u64().unwrap(), 2);
    }

    #[test]
    fn test_decode_list() {
        // List containing "cat", "dog"
        let data = [0xc8, 0x83, b'c', b'a', b't', 0x83, b'd', b'o', b'g'];
        let mut decoder = ZeroCopyRlpDecoder::new(&data);
        let value = decoder.parse_next().unwrap();
        
        assert_eq!(value.r#type(), RlpType::List(6));
    }

    #[test]
    fn test_buffer_overflow_protection() {
        // Malformed: claims 1MB string but only has 10 bytes
        let mut data = vec![0xb9, 0x00, 0x01, 0x00]; // 256 bytes claimed
        data.extend_from_slice(&[0u8; 10]); // Only 10 bytes provided
        
        let mut decoder = ZeroCopyRlpDecoder::new(&data);
        let result = decoder.parse_next();
        
        assert!(matches!(result, Err(RlpDecodeError::InvalidLength { .. })));
    }

    #[test]
    fn test_max_size_protection() {
        // Attempt to create oversized payload reference
        let data = [0xb9, 0xff, 0xff]; // Claims 65535 bytes
        
        let mut decoder = ZeroCopyRlpDecoder::new(&data);
        let result = decoder.parse_next();
        
        // Should fail due to insufficient data or size check
        assert!(result.is_err());
    }

    #[test]
    fn test_non_canonical_rejection() {
        // Non-canonical: leading zero in length encoding
        let data = [0xb8, 0x01, 0x00]; // Claims 1 byte with long format
        
        let mut decoder = ZeroCopyRlpDecoder::new(&data);
        let result = decoder.parse_next();
        
        assert!(matches!(result, Err(RlpDecodeError::NonCanonicalLength)));
    }
}
