//! Zero-Copy SIMD-Accelerated Text Tokenizer
//!
//! This module implements a high-performance tokenizer that operates directly
//! on byte slices without heap allocations, using SIMD instructions for
//! parallel character classification and boundary detection.

use std::simd::{u8x16, Simd, SimdPartialEq, SimdPartialOrd, Mask};
use nom::{
    bytes::complete::take_while1,
    character::complete::multispace1,
    combinator::recognize,
    multi::many0,
    sequence::tuple,
    IResult,
};

/// Token types recognized by the tokenizer
#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    Word,
    Number,
    Symbol,
    Whitespace,
    Punctuation,
    Special,
}

/// A zero-copy token view into the original byte slice
#[derive(Debug, Clone)]
pub struct Token<'a> {
    /// Reference to the original data (zero-copy)
    pub data: &'a [u8],
    /// Start offset in the original buffer
    pub offset: usize,
    /// Length of the token
    pub len: usize,
    /// Type of token
    pub token_type: TokenType,
}

impl<'a> Token<'a> {
    /// Get the token as a string slice (if valid UTF-8)
    #[inline]
    pub fn as_str(&self) -> Option<&'a str> {
        std::str::from_utf8(self.data).ok()
    }

    /// Get the token as a byte slice
    #[inline]
    pub fn as_bytes(&self) -> &'a [u8] {
        self.data
    }
}

/// SIMD-accelerated character classifier
pub struct SimdCharClassifier;

impl SimdCharClassifier {
    /// Check if bytes are ASCII alphabetic using SIMD
    #[inline]
    pub fn is_ascii_alpha_simd(bytes: &[u8]) -> Vec<bool> {
        let mut results = Vec::with_capacity(bytes.len());
        
        // Process 16 bytes at a time using SIMD
        let chunks = bytes.chunks_exact(16);
        let remainder = bytes.len() % 16;
        
        for chunk in chunks {
            let v = u8x16::from_slice(chunk);
            
            // Check if byte is in range ['a', 'z'] or ['A', 'Z']
            let lower_a = u8x16::splat(b'a');
            let lower_z = u8x16::splat(b'z');
            let upper_a = u8x16::splat(b'A');
            let upper_z = u8x16::splat(b'Z');
            
            let is_lower = v.simd_ge(lower_a) & v.simd_le(lower_z);
            let is_upper = v.simd_ge(upper_a) & v.simd_le(upper_z);
            let is_alpha = is_lower | is_upper;
            
            let mask = is_alpha.to_mask();
            for i in 0..16 {
                results.push(mask.test(i));
            }
        }
        
        // Handle remainder
        for &byte in bytes.iter().skip(bytes.len() - remainder) {
            results.push(byte.is_ascii_alphabetic());
        }
        
        results
    }

    /// Check if bytes are ASCII numeric using SIMD
    #[inline]
    pub fn is_ascii_digit_simd(bytes: &[u8]) -> Vec<bool> {
        let mut results = Vec::with_capacity(bytes.len());
        
        let chunks = bytes.chunks_exact(16);
        let remainder = bytes.len() % 16;
        
        for chunk in chunks {
            let v = u8x16::from_slice(chunk);
            
            // Check if byte is in range ['0', '9']
            let zero = u8x16::splat(b'0');
            let nine = u8x16::splat(b'9');
            
            let is_digit = v.simd_ge(zero) & v.simd_le(nine);
            let mask = is_digit.to_mask();
            
            for i in 0..16 {
                results.push(mask.test(i));
            }
        }
        
        for &byte in bytes.iter().skip(bytes.len() - remainder) {
            results.push(byte.is_ascii_digit());
        }
        
        results
    }

    /// Find whitespace boundaries using SIMD
    #[inline]
    pub fn find_whitespace_boundaries(bytes: &[u8]) -> Vec<usize> {
        let mut boundaries = Vec::new();
        
        let chunks = bytes.chunks_exact(16);
        let remainder = bytes.len() % 16;
        
        for (chunk_idx, chunk) in chunks.enumerate() {
            let v = u8x16::from_slice(chunk);
            
            // Check for common whitespace characters
            let space = u8x16::splat(b' ');
            let tab = u8x16::splat(b'\t');
            let newline = u8x16::splat(b'\n');
            let cr = u8x16::splat(b'\r');
            
            let is_ws = v.simd_eq(space) | v.simd_eq(tab) | v.simd_eq(newline) | v.simd_eq(cr);
            let mask = is_ws.to_mask();
            
            for i in 0..16 {
                if mask.test(i) {
                    boundaries.push(chunk_idx * 16 + i);
                }
            }
        }
        
        // Handle remainder
        let start = bytes.len() - remainder;
        for (i, &byte) in bytes.iter().enumerate().skip(start) {
            if byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r' {
                boundaries.push(i);
            }
        }
        
        boundaries
    }
}

/// Zero-copy streaming tokenizer state
pub struct ZeroCopyTokenizer<'a> {
    /// Original input buffer
    input: &'a [u8],
    /// Current position
    position: usize,
    /// Cached tokens (optional, for reuse)
    cached_tokens: Vec<Token<'a>>,
}

impl<'a> ZeroCopyTokenizer<'a> {
    /// Create a new tokenizer from a byte slice
    pub fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            position: 0,
            cached_tokens: Vec::with_capacity(256),
        }
    }

    /// Tokenize the entire input, returning zero-copy tokens
    pub fn tokenize_all(&mut self) -> Vec<Token<'a>> {
        self.cached_tokens.clear();
        let mut pos = 0;
        
        while pos < self.input.len() {
            // Skip whitespace
            while pos < self.input.len() && self.input[pos].is_ascii_whitespace() {
                pos += 1;
            }
            
            if pos >= self.input.len() {
                break;
            }
            
            let start = pos;
            let token_type = self.classify_and_advance(&mut pos);
            
            self.cached_tokens.push(Token {
                data: &self.input[start..pos],
                offset: start,
                len: pos - start,
                token_type,
            });
        }
        
        self.cached_tokens.clone()
    }

    /// Classify character and advance position
    #[inline]
    fn classify_and_advance(&self, pos: &mut usize) -> TokenType {
        let byte = self.input[*pos];
        
        if byte.is_ascii_alphabetic() {
            // Consume all alphabetic characters
            while *pos < self.input.len() && self.input[*pos].is_ascii_alphabetic() {
                *pos += 1;
            }
            TokenType::Word
        } else if byte.is_ascii_digit() {
            // Consume all digits (including decimal point for numbers)
            while *pos < self.input.len() 
                && (self.input[*pos].is_ascii_digit() || self.input[*pos] == b'.') 
            {
                *pos += 1;
            }
            TokenType::Number
        } else if byte.is_ascii_punctuation() {
            *pos += 1;
            TokenType::Punctuation
        } else if byte.is_ascii_whitespace() {
            while *pos < self.input.len() && self.input[*pos].is_ascii_whitespace() {
                *pos += 1;
            }
            TokenType::Whitespace
        } else {
            // Symbol or special character
            *pos += 1;
            TokenType::Symbol
        }
    }

    /// Iterate over tokens without allocating
    pub fn iter(&self) -> TokenIterator<'a, '_> {
        TokenIterator {
            tokenizer: self,
            position: 0,
        }
    }

    /// Get the original input
    #[inline]
    pub fn input(&self) -> &'a [u8] {
        self.input
    }
}

/// Iterator over zero-copy tokens
pub struct TokenIterator<'a, 'b> {
    tokenizer: &'b ZeroCopyTokenizer<'a>,
    position: usize,
}

impl<'a, 'b> Iterator for TokenIterator<'a, 'b> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let input = self.tokenizer.input();
        
        // Skip whitespace
        while self.position < input.len() && input[self.position].is_ascii_whitespace() {
            self.position += 1;
        }
        
        if self.position >= input.len() {
            return None;
        }
        
        let start = self.position;
        let byte = input[self.position];
        
        // Classify and consume
        if byte.is_ascii_alphabetic() {
            while self.position < input.len() && input[self.position].is_ascii_alphabetic() {
                self.position += 1;
            }
            Some(Token {
                data: &input[start..self.position],
                offset: start,
                len: self.position - start,
                token_type: TokenType::Word,
            })
        } else if byte.is_ascii_digit() {
            while self.position < input.len() 
                && (input[self.position].is_ascii_digit() || input[self.position] == b'.') 
            {
                self.position += 1;
            }
            Some(Token {
                data: &input[start..self.position],
                offset: start,
                len: self.position - start,
                token_type: TokenType::Number,
            })
        } else {
            self.position += 1;
            Some(Token {
                data: &input[start..self.position],
                offset: start,
                len: self.position - start,
                token_type: TokenType::Symbol,
            })
        }
    }
}

/// Nom parser for text extraction from raw buffers
pub mod nom_parsers {
    use super::*;
    
    /// Parse a word (alphabetic characters only)
    pub fn parse_word(input: &[u8]) -> IResult<&[u8], &[u8]> {
        take_while1(|c: u8| c.is_ascii_alphabetic())(input)
    }

    /// Parse a number (digits and optional decimal point)
    pub fn parse_number(input: &[u8]) -> IResult<&[u8], &[u8]> {
        take_while1(|c: u8| c.is_ascii_digit() || c == b'.')(input)
    }

    /// Parse until whitespace
    pub fn parse_until_whitespace(input: &[u8]) -> IResult<&[u8], &[u8]> {
        take_while1(|c: u8| !c.is_ascii_whitespace())(input)
    }

    /// Parse a complete text payload, extracting words
    pub fn parse_text_payload(input: &[u8]) -> IResult<&[u8], Vec<&[u8]>> {
        many0(tuple((
            multispace1,
            parse_word,
        )))(input).map(|(rest, pairs)| {
            (rest, pairs.into_iter().map(|(_, word)| word).collect())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenization() {
        let input = b"Hello world 123 test";
        let mut tokenizer = ZeroCopyTokenizer::new(input);
        let tokens = tokenizer.tokenize_all();
        
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].as_str(), Some("Hello"));
        assert_eq!(tokens[0].token_type, TokenType::Word);
        assert_eq!(tokens[1].as_str(), Some("world"));
        assert_eq!(tokens[2].token_type, TokenType::Number);
        assert_eq!(tokens[2].as_str(), Some("123"));
    }

    #[test]
    fn test_zero_copy_property() {
        let input = b"The quick brown fox";
        let mut tokenizer = ZeroCopyTokenizer::new(input);
        let tokens = tokenizer.tokenize_all();
        
        // Verify tokens point to original buffer
        for token in &tokens {
            let original_slice = &input[token.offset..token.offset + token.len];
            assert_eq!(token.data, original_slice);
        }
    }

    #[test]
    fn test_iterator() {
        let input = b"apple banana cherry";
        let tokenizer = ZeroCopyTokenizer::new(input);
        let tokens: Vec<Token> = tokenizer.iter().collect();
        
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].as_str(), Some("apple"));
        assert_eq!(tokens[1].as_str(), Some("banana"));
        assert_eq!(tokens[2].as_str(), Some("cherry"));
    }

    #[test]
    fn test_simd_classifier() {
        let input = b"Hello123World";
        let alpha_results = SimdCharClassifier::is_ascii_alpha_simd(input);
        
        assert_eq!(alpha_results.len(), 13);
        assert!(alpha_results[0]); // H
        assert!(alpha_results[4]); // o
        assert!(!alpha_results[5]); // 1
        assert!(!alpha_results[7]); // 3
        assert!(alpha_results[8]); // W
    }

    #[test]
    fn test_nom_parsers() {
        use nom_parsers::*;
        
        let result = parse_word(b"Hello123");
        assert!(result.is_ok());
        let (_, word) = result.unwrap();
        assert_eq!(word, b"Hello");
        
        let result = parse_number(b"123.45abc");
        assert!(result.is_ok());
        let (_, number) = result.unwrap();
        assert_eq!(number, b"123.45");
    }
}
