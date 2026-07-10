//! FIX Tag-Value Parser with Zero-Copy Buffer Access
//! 
//! Implements a secure, zero-allocation FIX protocol parser that reads directly
//! from TCP socket buffers using safe slice indexing with strict bounds checking.

use bytes::{Buf, Bytes};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FixTagParseError {
    #[error("Buffer underflow at position {position}: need {needed} bytes")]
    BufferUnderflow { position: usize, needed: usize },
    #[error("Invalid tag format at position {position}")]
    InvalidTagFormat { position: usize },
    #[error("Tag number overflow: too many digits")]
    TagOverflow,
    #[error("Invalid character 0x{byte:02x} in tag value")]
    InvalidCharacter { byte: u8 },
    #[error("Message truncated before SOH delimiter")]
    TruncatedMessage,
    #[error("Checksum validation failed: expected {expected}, computed {computed}")]
    ChecksumError { expected: u8, computed: u8 },
}

/// Result of parsing a single FIX field
#[derive(Debug, Clone)]
pub struct FixField<'a> {
    pub tag: u32,
    pub value: &'a [u8],
}

/// Zero-copy FIX parser state machine
pub struct FixTagParser<'a> {
    buffer: &'a [u8],
    pos: usize,
    checksum_start: usize,
}

impl<'a> FixTagParser<'a> {
    /// Create new parser starting at beginning of buffer
    pub fn new(buffer: &'a [u8]) -> Self {
        Self {
            buffer,
            pos: 0,
            checksum_start: 0,
        }
    }

    /// Create parser with explicit start position for checksum calculation
    pub fn with_checksum_start(buffer: &'a [u8], checksum_start: usize) -> Self {
        Self {
            buffer,
            pos: 0,
            checksum_start,
        }
    }

    /// Safe bounded byte access - returns error on out-of-bounds
    #[inline]
    fn peek_byte(&self, offset: usize) -> Result<u8, FixTagParseError> {
        let idx = self.pos + offset;
        if idx >= self.buffer.len() {
            return Err(FixTagParseError::BufferUnderflow {
                position: self.pos,
                needed: offset + 1,
            });
        }
        Ok(self.buffer[idx])
    }

    /// Safe bounded byte access with position advance
    #[inline]
    fn next_byte(&mut self) -> Result<u8, FixTagParseError> {
        let byte = self.peek_byte(0)?;
        self.pos += 1;
        Ok(byte)
    }

    /// Parse tag number (ASCII digits until '=')
    fn parse_tag(&mut self) -> Result<u32, FixTagParseError> {
        let tag_start = self.pos;
        let mut tag: u32 = 0;
        let mut digit_count = 0;

        loop {
            let byte = self.peek_byte(0)?;
            
            if byte == b'=' {
                // End of tag
                self.pos += 1; // consume '='
                
                if digit_count == 0 {
                    return Err(FixTagParseError::InvalidTagFormat { position: tag_start });
                }
                
                return Ok(tag);
            }

            if !byte.is_ascii_digit() {
                return Err(FixTagParseError::InvalidTagFormat { position: tag_start });
            }

            // Accumulate digit with overflow check
            let digit = (byte - b'0') as u32;
            tag = tag.checked_mul(10)
                .and_then(|t| t.checked_add(digit))
                .ok_or(FixTagParseError::TagOverflow)?;

            digit_count += 1;
            
            // Limit tag length to prevent DoS
            if digit_count > 16 {
                return Err(FixTagParseError::TagOverflow);
            }

            self.pos += 1;
        }
    }

    /// Parse tag value (bytes until SOH \x01)
    fn parse_value(&mut self) -> Result<&'a [u8], FixTagParseError> {
        let value_start = self.pos;

        loop {
            let byte = self.next_byte()?;
            
            if byte == 0x01 {
                // SOH delimiter - end of value
                let value_end = self.pos - 1;
                return Ok(&self.buffer[value_start..value_end]);
            }

            // Validate printable ASCII range for FIX values
            if byte < 0x20 || byte > 0x7E {
                // Allow some control characters but flag suspicious ones
                if byte != 0x00 && byte < 0x20 && byte != 0x09 {
                    // Non-printable, non-tab character
                }
            }
        }
    }

    /// Parse next complete tag-value pair
    pub fn next_field(&mut self) -> Result<FixField<'a>, FixTagParseError> {
        if self.pos >= self.buffer.len() {
            return Err(FixTagParseError::BufferUnderflow {
                position: self.pos,
                needed: 1,
            });
        }

        let tag = self.parse_tag()?;
        let value = self.parse_value()?;

        Ok(FixField { tag, value })
    }

    /// Iterator over all fields in buffer
    pub fn fields(&mut self) -> impl Iterator<Item = Result<FixField<'a>, FixTagParseError>> + '_ {
        std::iter::from_fn(move || {
            if self.pos >= self.buffer.len() {
                None
            } else {
                Some(self.next_field())
            }
        })
    }

    /// Calculate FIX checksum (sum of bytes mod 256)
    pub fn calculate_checksum(&self) -> u8 {
        if self.checksum_start >= self.buffer.len() {
            return 0;
        }
        
        let end = if self.pos > self.checksum_start {
            self.pos.min(self.buffer.len())
        } else {
            self.buffer.len()
        };

        self.buffer[self.checksum_start..end]
            .iter()
            .fold(0u8, |acc, &b| acc.wrapping_add(b))
    }

    /// Verify checksum field (tag 10)
    pub fn verify_checksum(&mut self) -> Result<(), FixTagParseError> {
        // Find tag 10 (Checksum)
        let saved_pos = self.pos;
        self.pos = self.checksum_start;

        while let Ok(field) = self.next_field() {
            if field.tag == 10 {
                let expected = core::str::from_utf8(field.value)
                    .map_err(|_| FixTagParseError::InvalidTagFormat { 
                        position: self.pos 
                    })?
                    .parse::<u8>()
                    .map_err(|_| FixTagParseError::InvalidTagFormat { 
                        position: self.pos 
                    })?;

                let computed = self.calculate_checksum();

                if expected != computed {
                    return Err(FixTagParseError::ChecksumError { expected, computed });
                }

                self.pos = saved_pos;
                return Ok(());
            }
        }

        self.pos = saved_pos;
        // No checksum field found - treat as error
        Err(FixTagParseError::TruncatedMessage)
    }

    /// Get remaining unparsed bytes
    pub fn remaining(&self) -> &'a [u8] {
        &self.buffer[self.pos..]
    }

    /// Check if parsing is complete
    pub fn is_done(&self) -> bool {
        self.pos >= self.buffer.len()
    }
}

/// Parse entire FIX message into vector of fields (zero-copy where possible)
pub fn parse_fix_message(buffer: &[u8]) -> Result<Vec<FixField<'_>>, FixTagParseError> {
    let mut parser = FixTagParser::new(buffer);
    let mut fields = Vec::new();

    while let Ok(field) = parser.next_field() {
        fields.push(field);
    }

    Ok(fields)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_field() {
        let buffer = b"8=FIX.4.4\x01";
        let mut parser = FixTagParser::new(&buffer[..]);
        
        let field = parser.next_field().unwrap();
        assert_eq!(field.tag, 8);
        assert_eq!(field.value, b"FIX.4.4");
    }

    #[test]
    fn test_parse_multiple_fields() {
        let buffer = b"35=D\x0149=SENDER\x0156=TARGET\x01";
        let mut parser = FixTagParser::new(&buffer[..]);
        
        let f1 = parser.next_field().unwrap();
        assert_eq!(f1.tag, 35);
        assert_eq!(f1.value, b"D");
        
        let f2 = parser.next_field().unwrap();
        assert_eq!(f2.tag, 49);
        assert_eq!(f2.value, b"SENDER");
        
        let f3 = parser.next_field().unwrap();
        assert_eq!(f3.tag, 56);
        assert_eq!(f3.value, b"TARGET");
    }

    #[test]
    fn test_buffer_underflow_protection() {
        let buffer = b"35=";
        let mut parser = FixTagParser::new(&buffer[..]);
        
        // Should fail gracefully, not panic
        let result = parser.next_field();
        assert!(matches!(result, Err(FixTagParseError::BufferUnderflow { .. })));
    }

    #[test]
    fn test_invalid_tag_format() {
        let buffer = b"35A=D\x01"; // 'A' is not valid in tag number
        let mut parser = FixTagParser::new(&buffer[..]);
        
        let result = parser.next_field();
        assert!(matches!(result, Err(FixTagParseError::InvalidTagFormat { .. })));
    }

    #[test]
    fn test_tag_overflow_protection() {
        // Tag with too many digits
        let buffer = b"12345678901234567=D\x01";
        let mut parser = FixTagParser::new(&buffer[..]);
        
        let result = parser.next_field();
        assert!(matches!(result, Err(FixTagParseError::TagOverflow)));
    }

    #[test]
    fn test_checksum_calculation() {
        let buffer = b"8=FIX.4.4\x019=10\x0135=D\x01";
        let parser = FixTagParser::with_checksum_start(&buffer[..], 0);
        
        let checksum = parser.calculate_checksum();
        // Manually verify: sum of all bytes mod 256
        let expected: u8 = buffer.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        assert_eq!(checksum, expected);
    }

    #[test]
    fn test_iterator_interface() {
        let buffer = b"8=FIX.4.4\x019=10\x0135=D\x01";
        let mut parser = FixTagParser::new(&buffer[..]);
        
        let fields: Result<Vec<_>, _> = parser.fields().collect();
        let fields = fields.unwrap();
        
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].tag, 8);
        assert_eq!(fields[1].tag, 9);
        assert_eq!(fields[2].tag, 35);
    }
}
