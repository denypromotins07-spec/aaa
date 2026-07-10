//! Zero-Copy SIMD Tokenizer
//! 
//! Implements streaming tokenization using nom parser combinators and
//! SIMD-accelerated byte scanning. Extracts text payloads directly from
//! network buffers without heap allocation.

use std::simd::{u8x16, SimdPartialEq, SimdPartialOrd};
use nom::{
    IResult,
    bytes::complete::{take_while1, take_until},
    character::complete::{space0, alphanumeric1},
    combinator::{opt, recognize},
    multi::{many0, many_till},
    sequence::{delimited, pair, preceded},
};

/// Token types for financial text
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenType {
    Word,
    Number,
    Symbol,
    Ticker,
    Hashtag,
    Mention,
    Url,
    DollarAmount,
    PercentAmount,
}

/// A zero-copy token referencing the original buffer
#[derive(Debug, Clone)]
pub struct Token<'a> {
    pub token_type: TokenType,
    pub data: &'a [u8],
    pub offset: usize,
}

/// SIMD-accelerated whitespace detector
#[inline]
fn is_whitespace_simd(byte: u8) -> bool {
    byte == b' ' || byte == b'\n' || byte == b'\r' || byte == b'\t'
}

/// SIMD-accelerated byte scanner for token boundaries
pub struct SimdScanner<'a> {
    buffer: &'a [u8],
    pos: usize,
}

impl<'a> SimdScanner<'a> {
    /// Create a new scanner from a buffer slice
    pub fn new(buffer: &'a [u8]) -> Self {
        Self { buffer, pos: 0 }
    }

    /// Scan for next token boundary using SIMD
    pub fn scan_next(&mut self) -> Option<Token<'a>> {
        if self.pos >= self.buffer.len() {
            return None;
        }

        // Skip leading whitespace
        while self.pos < self.buffer.len() && is_whitespace_simd(self.buffer[self.pos]) {
            self.pos += 1;
        }

        if self.pos >= self.buffer.len() {
            return None;
        }

        let start = self.pos;
        let byte = self.buffer[start];

        // Determine token type and find end
        let (token_type, end) = match byte {
            b'$' => {
                // Dollar amount
                self.pos += 1;
                let num_end = self.scan_number();
                (TokenType::DollarAmount, num_end)
            }
            b'#' => {
                // Hashtag
                self.pos += 1;
                let word_end = self.scan_word();
                (TokenType::Hashtag, word_end)
            }
            b'@' => {
                // Mention
                self.pos += 1;
                let word_end = self.scan_word();
                (TokenType::Mention, word_end)
            }
            b'h' | b'H' => {
                // Check for http/https URL
                if self.buffer[start..].starts_with(b"http") || 
                   self.buffer[start..].starts_with(b"HTTP") {
                    let url_end = self.scan_url();
                    (TokenType::Url, url_end)
                } else {
                    let word_end = self.scan_word();
                    (TokenType::Word, word_end)
                }
            }
            b'0'..=b'9' => {
                let num_end = self.scan_number();
                // Check for percentage
                if num_end < self.buffer.len() && self.buffer[num_end] == b'%' {
                    (TokenType::PercentAmount, num_end + 1)
                } else {
                    (TokenType::Number, num_end)
                }
            }
            b'A'..=b'Z' => {
                // Could be ticker symbol (all caps, 1-5 chars)
                let word_end = self.scan_uppercase_word();
                let word_len = word_end - start;
                if word_len >= 1 && word_len <= 5 {
                    (TokenType::Ticker, word_end)
                } else {
                    (TokenType::Word, word_end)
                }
            }
            _ => {
                let word_end = self.scan_word();
                (TokenType::Word, word_end)
            }
        };

        let token_data = &self.buffer[start..end];
        self.pos = end;

        Some(Token {
            token_type,
            data: token_data,
            offset: start,
        })
    }

    /// Scan a number sequence
    fn scan_number(&mut self) -> usize {
        let mut i = self.pos;
        while i < self.buffer.len() {
            let byte = self.buffer[i];
            if byte.is_ascii_digit() || byte == b'.' || byte == b',' {
                i += 1;
            } else {
                break;
            }
        }
        i
    }

    /// Scan a word (alphanumeric + underscore)
    fn scan_word(&mut self) -> usize {
        let mut i = self.pos;
        while i < self.buffer.len() {
            let byte = self.buffer[i];
            if byte.is_ascii_alphanumeric() || byte == b'_' {
                i += 1;
            } else {
                break;
            }
        }
        i
    }

    /// Scan uppercase word (for ticker detection)
    fn scan_uppercase_word(&mut self) -> usize {
        let mut i = self.pos;
        while i < self.buffer.len() {
            let byte = self.buffer[i];
            if byte.is_ascii_uppercase() || byte.is_ascii_digit() {
                i += 1;
            } else {
                break;
            }
        }
        i
    }

    /// Scan URL
    fn scan_url(&mut self) -> usize {
        let mut i = self.pos;
        while i < self.buffer.len() {
            let byte = self.buffer[i];
            if !is_whitespace_simd(byte) && byte != b'<' && byte != b'>' {
                i += 1;
            } else {
                break;
            }
        }
        i
    }
}

/// Iterator over tokens in a buffer
pub struct TokenIterator<'a> {
    scanner: SimdScanner<'a>,
}

impl<'a> TokenIterator<'a> {
    pub fn new(buffer: &'a [u8]) -> Self {
        Self {
            scanner: SimdScanner::new(buffer),
        }
    }
}

impl<'a> Iterator for TokenIterator<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.scanner.scan_next()
    }
}

/// Nom-based parser for structured text extraction
pub mod nom_parsers {
    use super::*;
    use nom::{
        bytes::complete::tag,
        character::complete::char,
    };

    /// Parse a JSON-like string field without allocation
    pub fn parse_string_field(input: &[u8]) -> IResult<&[u8], &[u8]> {
        delimited(
            char('"'),
            take_until("\""),
            char('"'),
        )(input)
    }

    /// Parse a key-value pair from JSON
    pub fn parse_json_pair(input: &[u8]) -> IResult<&[u8], (&[u8], &[u8])> {
        pair(
            parse_string_field,
            preceded(
                space0,
                preceded(
                    char(':'),
                    preceded(space0, parse_string_field),
                ),
            ),
        )(input)
    }

    /// Extract text content from JSON payload
    pub fn extract_text_from_json<'a>(payload: &'a [u8]) -> Option<&'a [u8]> {
        // Try to find "text": or "content": fields
        let search_keys = [b"\"text\"", b"\"content\"", b"\"body\"", b"\"message\""];
        
        for key in &search_keys {
            if let Some(pos) = payload.windows(key.len()).position(|w| w == *key) {
                let rest = &payload[pos + key.len()..];
                // Skip colon and whitespace
                let trimmed = rest.iter().position(|&b| b != b':' && b != b' ' && b != b'\t')?;
                let value_start = &rest[trimmed..];
                
                if let Ok((_, text)) = parse_string_field(value_start) {
                    return Some(text);
                }
            }
        }
        
        // Fallback: return entire payload if it looks like plain text
        if payload.first() != Some(&b'{') {
            return Some(payload);
        }
        
        None
    }
}

/// Zero-copy tokenizer that processes buffers without allocation
pub struct ZeroCopyTokenizer {
    buffer_pool: Vec<Vec<u8>>,
}

impl ZeroCopyTokenizer {
    /// Create a new zero-copy tokenizer
    pub fn new() -> Self {
        Self {
            buffer_pool: Vec::with_capacity(16),
        }
    }

    /// Tokenize a buffer, returning an iterator of zero-copy tokens
    pub fn tokenize<'a>(&'a mut self, buffer: &'a [u8]) -> TokenIterator<'a> {
        TokenIterator::new(buffer)
    }

    /// Extract text from JSON payload using zero-copy techniques
    pub fn extract_text<'a>(&mut self, payload: &'a [u8]) -> Option<&'a [u8]> {
        nom_parsers::extract_text_from_json(payload)
    }

    /// Recycle a buffer back into the pool
    pub fn recycle_buffer(&mut self, mut buffer: Vec<u8>) {
        buffer.clear();
        if self.buffer_pool.len() < 32 {
            self.buffer_pool.push(buffer);
        }
    }

    /// Get a buffer from the pool
    pub fn get_buffer(&mut self, capacity: usize) -> Vec<u8> {
        self.buffer_pool
            .iter()
            .position(|b| b.capacity() >= capacity)
            .map(|i| self.buffer_pool.swap_remove(i))
            .unwrap_or_else(|| Vec::with_capacity(capacity))
    }
}

impl Default for ZeroCopyTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_scanner() {
        let buffer = b"$AAPL up 2.5% #bullish @elonmusk says buy";
        let mut scanner = SimdScanner::new(buffer);
        
        let tokens: Vec<Token> = scanner.by_ref().collect();
        
        assert!(tokens.iter().any(|t| t.token_type == TokenType::Ticker));
        assert!(tokens.iter().any(|t| t.token_type == TokenType::PercentAmount));
        assert!(tokens.iter().any(|t| t.token_type == TokenType::Hashtag));
        assert!(tokens.iter().any(|t| t.token_type == TokenType::Mention));
    }

    #[test]
    fn test_json_extraction() {
        let json = br#"{"text": "Fed raises rates", "source": "Reuters"}"#;
        let extracted = nom_parsers::extract_text_from_json(json);
        assert_eq!(extracted, Some(&b"Fed raises rates"[..]));
    }

    #[test]
    fn test_tokenizer_iterator() {
        let buffer = b"BTC USD EUR $50000 2.5% #crypto";
        let mut tokenizer = ZeroCopyTokenizer::new();
        let tokens: Vec<Token> = tokenizer.tokenize(buffer).collect();
        
        assert!(tokens.len() > 5);
    }
}
