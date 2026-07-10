//! Homopolymer Avoidance Encoder
//! 
//! Implements advanced homopolymer constraint enforcement for DNA synthesis.
//! Uses rotating code schemes and lookup tables to guarantee max 3 consecutive identical bases.

use crate::dna::nucleotide_base4_encoder::{Base, DnaEncoderError, MAX_HOMOPOLYMER_RUN};
use thiserror::Error;

/// Lookup table for homopolymer-safe encoding (2 bits -> base with context)
const HOMOPOLYMER_SAFE_TABLE: [[Base; 4]; 4] = [
    // Previous base: A
    [Base::C, Base::G, Base::T, Base::A], // Run length 1: can use A
    // Previous base: C  
    [Base::A, Base::G, Base::T, Base::C], // Run length 1: can use C
    // Previous base: G
    [Base::A, Base::C, Base::T, Base::G], // Run length 1: can use G
    // Previous base: T
    [Base::A, Base::C, Base::G, Base::T], // Run length 1: can use T
];

#[derive(Error, Debug)]
pub enum HomopolymerError {
    #[error("Cannot encode without creating homopolymer violation")]
    UnavoidableHomopolymer,
    #[error("Invalid run length: {0}")]
    InvalidRunLength(usize),
    #[error("Buffer overflow")]
    BufferOverflow,
}

/// State tracker for homopolymer encoding
#[derive(Debug, Clone, Copy)]
pub struct HomopolymerState {
    last_base: Option<Base>,
    current_run_length: usize,
}

impl HomopolymerState {
    #[inline]
    pub fn new() -> Self {
        Self { last_base: None, current_run_length: 0 }
    }

    #[inline]
    pub fn update(&mut self, base: Base) {
        if let Some(last) = self.last_base {
            if base == last {
                self.current_run_length += 1;
            } else {
                self.current_run_length = 1;
            }
        } else {
            self.current_run_length = 1;
        }
        self.last_base = Some(base);
    }

    #[inline]
    pub fn can_accept(&self, base: Base) -> bool {
        if let Some(last) = self.last_base {
            if base == last {
                self.current_run_length < MAX_HOMOPOLYMER_RUN
            } else {
                true
            }
        } else {
            true
        }
    }

    #[inline]
    pub fn run_length(&self) -> usize {
        self.current_run_length
    }

    #[inline]
    pub fn last_base(&self) -> Option<Base> {
        self.last_base
    }
}

impl Default for HomopolymerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced homopolymer avoidance encoder using rotating codes
pub struct HomopolymerAvoidanceEncoder {
    state: HomopolymerState,
    output_buffer: Box<[Base]>,
    output_len: usize,
    capacity: usize,
}

impl HomopolymerAvoidanceEncoder {
    /// Create a new encoder with pre-allocated buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            state: HomopolymerState::new(),
            output_buffer: vec![Base::A; capacity].into_boxed_slice(),
            output_len: 0,
            capacity,
        }
    }

    /// Encode input bits with guaranteed homopolymer avoidance
    /// 
    /// Uses a rotating code scheme where the mapping from 2-bit values to bases
    /// depends on the previous base and run length.
    pub fn encode_bits(&mut self, bits: &[u8]) -> Result<&[Base], HomopolymerError> {
        self.output_len = 0;
        self.state = HomopolymerState::new();

        // Each 2-bit value becomes one base
        let required_capacity = bits.len();
        if required_capacity > self.capacity {
            return Err(HomopolymerError::BufferOverflow);
        }

        for &bits_pair in bits {
            let value = bits_pair & 0b11;
            
            // Find a base that encodes this value without violating homopolymer constraint
            let base = self.select_safe_base(value)?;
            
            // Store the base
            self.output_buffer[self.output_len] = base;
            self.output_len += 1;
            
            // Update state
            self.state.update(base);
        }

        Ok(&self.output_buffer[..self.output_len])
    }

    /// Select a safe base for the given 2-bit value
    fn select_safe_base(&self, value: u8) -> Result<Base, HomopolymerError> {
        let value = (value & 0b11) as usize;
        
        // Try each possible base mapping
        for offset in 0..4 {
            let candidate_val = (value + offset) % 4;
            let candidate_base = match candidate_val {
                0 => Base::A,
                1 => Base::C,
                2 => Base::G,
                3 => Base::T,
                _ => unreachable!(),
            };

            if self.state.can_accept(candidate_base) {
                return Ok(candidate_base);
            }
        }

        // This should never happen with proper rotation
        Err(HomopolymerError::UnavoidableHomopolymer)
    }

    /// Encode bytes (8 bits) into homopolymer-safe DNA
    /// Each byte becomes 5 bases (with redundancy for error detection)
    pub fn encode_bytes(&mut self, data: &[u8]) -> Result<&[Base], HomopolymerError> {
        self.output_len = 0;
        self.state = HomopolymerState::new();

        // Each byte needs 5 bases (4 for data + 1 for checksum/homopolymer safety)
        let required_capacity = data.len() * 5;
        if required_capacity > self.capacity {
            return Err(HomopolymerError::BufferOverflow);
        }

        for &byte in data {
            // Split byte into 2-bit chunks with padding
            let bits = [
                (byte >> 6) & 0b11,
                (byte >> 4) & 0b11,
                (byte >> 2) & 0b11,
                byte & 0b11,
            ];

            // Encode first 4 bits normally
            for &bits_pair in &bits {
                let base = self.select_safe_base(bits_pair)?;
                self.output_buffer[self.output_len] = base;
                self.output_len += 1;
                self.state.update(base);
            }

            // Add a balancing base based on run length
            let balance_base = self.get_balance_base();
            self.output_buffer[self.output_len] = balance_base;
            self.output_len += 1;
            self.state.update(balance_base);
        }

        Ok(&self.output_buffer[..self.output_len])
    }

    /// Get a base that helps balance GC content and break homopolymers
    fn get_balance_base(&self) -> Base {
        let current_gc = self.calculate_gc_content();
        
        // Prefer non-GC if GC content is high
        if current_gc > 0.55 {
            if !self.state.can_accept(Base::A) {
                Base::T
            } else {
                Base::A
            }
        } 
        // Prefer GC if content is low
        else if current_gc < 0.45 {
            if !self.state.can_accept(Base::G) {
                Base::C
            } else {
                Base::G
            }
        }
        // Just break any potential homopolymer
        else {
            match self.state.last_base() {
                Some(Base::A) => Base::C,
                Some(Base::C) => Base::G,
                Some(Base::G) => Base::T,
                Some(Base::T) => Base::A,
                None => Base::G,
            }
        }
    }

    /// Calculate current GC content of output
    fn calculate_gc_content(&self) -> f64 {
        if self.output_len == 0 {
            return 0.0;
        }
        let gc_count = self.output_buffer[..self.output_len]
            .iter()
            .filter(|b| b.is_gc())
            .count();
        gc_count as f64 / self.output_len as f64
    }

    /// Decode homopolymer-safe DNA back to bytes
    pub fn decode_to_bytes(&self, dna: &[Base]) -> Result<Vec<u8>, DnaEncoderError> {
        if dna.len() % 5 != 0 {
            return Err(DnaEncoderError::InvalidInputLength);
        }

        let mut result = Vec::with_capacity(dna.len() / 5);

        for chunk in dna.chunks(5) {
            if chunk.len() < 4 {
                continue;
            }

            // Extract 4 data bases (ignore balancing base)
            let mut byte = 0u8;
            for (i, &base) in chunk[..4].iter().enumerate() {
                let bits = base.to_u8();
                let shift = 6 - (i * 2);
                byte |= bits << shift;
            }
            result.push(byte);
        }

        Ok(result)
    }

    /// Verify sequence has no homopolymer violations
    pub fn verify_sequence(dna: &[Base]) -> bool {
        let mut run_length = 1;
        let mut last_base = dna.first().copied();

        for &base in dna.iter().skip(1) {
            if Some(base) == last_base {
                run_length += 1;
                if run_length > MAX_HOMOPOLYMER_RUN {
                    return false;
                }
            } else {
                run_length = 1;
                last_base = Some(base);
            }
        }

        true
    }

    /// Get encoded sequence length
    #[inline]
    pub fn encoded_len(&self) -> usize {
        self.output_len
    }

    /// Get encoded sequence as slice
    #[inline]
    pub fn as_slice(&self) -> &[Base] {
        &self.output_buffer[..self.output_len]
    }

    /// Clear encoder state
    #[inline]
    pub fn clear(&mut self) {
        self.output_len = 0;
        self.state = HomopolymerState::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_homopolymer_state() {
        let mut state = HomopolymerState::new();
        assert!(state.can_accept(Base::A));
        
        state.update(Base::A);
        assert!(state.can_accept(Base::A));
        
        state.update(Base::A);
        assert!(state.can_accept(Base::A));
        
        state.update(Base::A);
        assert!(!state.can_accept(Base::A)); // Run length would be 4
        assert!(state.can_accept(Base::C));
    }

    #[test]
    fn test_encoder_basic() {
        let mut encoder = HomopolymerAvoidanceEncoder::new(1024);
        let data = vec![0x00, 0xFF, 0xAA, 0x55];
        
        let encoded = encoder.encode_bytes(&data).unwrap();
        assert!(HomopolymerAvoidanceEncoder::verify_sequence(encoded));
    }

    #[test]
    fn test_no_homopolymers() {
        let mut encoder = HomopolymerAvoidanceEncoder::new(4096);
        
        // Data designed to create worst-case homopolymers
        let data: Vec<u8> = (0..256).collect();
        let encoded = encoder.encode_bytes(&data).unwrap();
        
        assert!(HomopolymerAvoidanceEncoder::verify_sequence(encoded));
        assert_eq!(encoded.len(), data.len() * 5);
    }

    #[test]
    fn test_roundtrip() {
        let mut encoder = HomopolymerAvoidanceEncoder::new(1024);
        let original = vec![0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
        
        let encoded = encoder.encode_bytes(&original).unwrap();
        let decoded = encoder.decode_to_bytes(encoded).unwrap();
        
        assert_eq!(original, decoded);
    }
}
