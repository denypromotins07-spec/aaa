//! Nucleotide Base-4 Encoder
//! 
//! Translates binary financial data into DNA base-4 sequences (A, C, G, T).
//! Implements strict homopolymer avoidance (max 3 consecutive identical bases).

use thiserror::Error;

/// DNA nucleotide bases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Base {
    A = 0,
    C = 1,
    G = 2,
    T = 3,
}

impl Base {
    #[inline]
    pub fn from_u8(value: u8) -> Result<Self, DnaEncoderError> {
        match value {
            0 => Ok(Base::A),
            1 => Ok(Base::C),
            2 => Ok(Base::G),
            3 => Ok(Base::T),
            _ => Err(DnaEncoderError::InvalidBaseValue(value)),
        }
    }

    #[inline]
    pub fn to_u8(self) -> u8 {
        self as u8
    }

    #[inline]
    pub fn from_char(c: char) -> Result<Self, DnaEncoderError> {
        match c {
            'A' | 'a' => Ok(Base::A),
            'C' | 'c' => Ok(Base::C),
            'G' | 'g' => Ok(Base::G),
            'T' | 't' => Ok(Base::T),
            _ => Err(DnaEncoderError::InvalidBaseChar(c)),
        }
    }

    #[inline]
    pub fn to_char(self) -> char {
        match self {
            Base::A => 'A',
            Base::C => 'C',
            Base::G => 'G',
            Base::T => 'T',
        }
    }

    /// Check if this base is G or C (for GC-content calculation)
    #[inline]
    pub fn is_gc(self) -> bool {
        matches!(self, Base::G | Base::C)
    }
}

#[derive(Error, Debug)]
pub enum DnaEncoderError {
    #[error("Invalid base value: {0}")]
    InvalidBaseValue(u8),
    #[error("Invalid base character: {0}")]
    InvalidBaseChar(char),
    #[error("Homopolymer constraint violated: {0}")]
    HomopolymerViolation(String),
    #[error("GC content out of range: {0}%")]
    GCContentOutOfRange(f64),
    #[error("Buffer overflow")]
    BufferOverflow,
    #[error("Invalid input length")]
    InvalidInputLength,
}

/// Maximum homopolymer run length (biological constraint)
pub const MAX_HOMOPOLYMER_RUN: usize = 3;

/// Target GC content range (50% ± 10%)
pub const MIN_GC_CONTENT: f64 = 0.40;
pub const MAX_GC_CONTENT: f64 = 0.60;
pub const TARGET_GC_CONTENT: f64 = 0.50;

/// Zero-allocation buffer for DNA sequences
pub struct DnaSequenceBuffer {
    data: Box<[Base]>,
    len: usize,
    capacity: usize,
}

impl DnaSequenceBuffer {
    /// Create a pre-allocated DNA sequence buffer
    pub fn with_capacity(capacity: usize) -> Self {
        let data = vec![Base::A; capacity].into_boxed_slice();
        Self { data, len: 0, capacity }
    }

    #[inline]
    pub fn push(&mut self, base: Base) -> Result<(), DnaEncoderError> {
        if self.len >= self.capacity {
            return Err(DnaEncoderError::BufferOverflow);
        }
        self.data[self.len] = base;
        self.len += 1;
        Ok(())
    }

    #[inline]
    pub fn extend_from_slice(&mut self, bases: &[Base]) -> Result<(), DnaEncoderError> {
        if self.len + bases.len() > self.capacity {
            return Err(DnaEncoderError::BufferOverflow);
        }
        for &base in bases {
            self.data[self.len] = base;
            self.len += 1;
        }
        Ok(())
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[Base] {
        &self.data[..self.len]
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        self.as_slice().iter().map(|b| b.to_char()).collect()
    }

    /// Calculate GC content
    pub fn gc_content(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        let gc_count = self.as_slice().iter().filter(|b| b.is_gc()).count();
        gc_count as f64 / self.len as f64
    }

    /// Check for homopolymer violations
    pub fn has_homopolymer_violation(&self) -> Option<(usize, Base)> {
        let slice = self.as_slice();
        if slice.len() < MAX_HOMOPOLYMER_RUN + 1 {
            return None;
        }

        let mut run_length = 1;
        let mut current_base = slice[0];

        for i in 1..slice.len() {
            if slice[i] == current_base {
                run_length += 1;
                if run_length > MAX_HOMOPOLYMER_RUN {
                    return Some((i - run_length + 1, current_base));
                }
            } else {
                run_length = 1;
                current_base = slice[i];
            }
        }

        None
    }
}

/// Nucleotide Encoder for binary-to-DNA conversion
pub struct NucleotideEncoder {
    buffer: DnaSequenceBuffer,
    homopolymer_check: bool,
    gc_balance: bool,
}

impl NucleotideEncoder {
    /// Create a new encoder with specified options
    pub fn new(capacity: usize, homopolymer_check: bool, gc_balance: bool) -> Self {
        Self {
            buffer: DnaSequenceBuffer::with_capacity(capacity),
            homopolymer_check,
            gc_balance,
        }
    }

    /// Encode 2 bits into a single DNA base
    #[inline]
    pub fn encode_bits(bits: u8) -> Result<Base, DnaEncoderError> {
        if bits > 0b11 {
            return Err(DnaEncoderError::InvalidBaseValue(bits));
        }
        Base::from_u8(bits)
    }

    /// Decode a DNA base back to 2 bits
    #[inline]
    pub fn decode_base(base: Base) -> u8 {
        base.to_u8()
    }

    /// Encode a byte array into DNA sequence with homopolymer avoidance
    pub fn encode_bytes(&mut self, data: &[u8]) -> Result<&[Base], DnaEncoderError> {
        self.buffer.clear();

        // Each byte becomes 4 bases (2 bits per base)
        let required_capacity = data.len() * 4;
        if required_capacity > self.buffer.capacity {
            return Err(DnaEncoderError::BufferOverflow);
        }

        let mut last_base: Option<Base> = None;
        let mut run_length = 0;

        for &byte in data {
            // Extract 4 pairs of 2 bits
            for shift in [6, 4, 2, 0] {
                let bits = (byte >> shift) & 0b11;
                let mut base = Base::from_u8(bits)?;

                // Homopolymer avoidance: rotate base if needed
                if self.homopolymer_check {
                    if let Some(last) = last_base {
                        if base == last {
                            run_length += 1;
                            if run_length >= MAX_HOMOPOLYMER_RUN {
                                // Rotate to next base to break homopolymer
                                base = self.rotate_base(base, run_length);
                                run_length = 1;
                            }
                        } else {
                            run_length = 1;
                        }
                    } else {
                        run_length = 1;
                    }
                }

                // GC balancing: adjust if content drifts too far
                if self.gc_balance && self.buffer.len() > 10 {
                    let current_gc = self.buffer.gc_content();
                    if current_gc > MAX_GC_CONTENT && base.is_gc() {
                        // Prefer A or T
                        base = if base == Base::G { Base::A } else { Base::T };
                    } else if current_gc < MIN_GC_CONTENT && !base.is_gc() {
                        // Prefer G or C
                        base = if base == Base::A { Base::G } else { Base::C };
                    }
                }

                self.buffer.push(base)?;
                last_base = Some(base);
            }
        }

        // Final validation
        if self.homopolymer_check {
            if let Some((pos, base)) = self.buffer.has_homopolymer_violation() {
                return Err(DnaEncoderError::HomopolymerViolation(
                    format!("Position {}: {} repeated > {} times", pos, base.to_char(), MAX_HOMOPOLYMER_RUN)
                ));
            }
        }

        if self.gc_balance {
            let gc = self.buffer.gc_content();
            if gc < MIN_GC_CONTENT || gc > MAX_GC_CONTENT {
                return Err(DnaEncoderError::GCContentOutOfRange(gc * 100.0));
            }
        }

        Ok(self.buffer.as_slice())
    }

    /// Rotate base to avoid homopolymers
    fn rotate_base(&self, base: Base, rotation: usize) -> Base {
        // Use a deterministic rotation scheme
        let offset = (rotation % 3) + 1; // Never rotate to same base
        let base_val = base.to_u8() as usize;
        let new_val = (base_val + offset) % 4;
        match Base::from_u8(new_val as u8) {
            Ok(b) => b,
            Err(_) => base, // Fallback to original on error
        }
    }

    /// Decode DNA sequence back to bytes
    pub fn decode_to_bytes(&self, dna: &[Base]) -> Result<Vec<u8>, DnaEncoderError> {
        if dna.len() % 4 != 0 {
            return Err(DnaEncoderError::InvalidInputLength);
        }

        let mut result = Vec::with_capacity(dna.len() / 4);
        
        for chunk in dna.chunks(4) {
            let mut byte = 0u8;
            for (i, &base) in chunk.iter().enumerate() {
                let bits = base.to_u8();
                let shift = 6 - (i * 2);
                byte |= bits << shift;
            }
            result.push(byte);
        }

        Ok(result)
    }

    /// Get the encoded sequence as a string
    pub fn encoded_string(&self) -> String {
        self.buffer.to_string()
    }

    /// Get current GC content
    pub fn current_gc_content(&self) -> f64 {
        self.buffer.gc_content()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_conversion() {
        assert_eq!(Base::A.to_u8(), 0);
        assert_eq!(Base::C.to_u8(), 1);
        assert_eq!(Base::G.to_u8(), 2);
        assert_eq!(Base::T.to_u8(), 3);

        assert_eq!(Base::from_u8(0).unwrap(), Base::A);
        assert_eq!(Base::from_u8(3).unwrap(), Base::T);
        assert!(Base::from_u8(4).is_err());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut encoder = NucleotideEncoder::new(1024, true, true);
        let data = vec![0x48, 0x65, 0x6C, 0x6C, 0x6F]; // "Hello"
        
        let encoded = encoder.encode_bytes(&data).unwrap();
        let decoded = encoder.decode_to_bytes(encoded).unwrap();
        
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_homopolymer_avoidance() {
        let mut encoder = NucleotideEncoder::new(1024, true, false);
        
        // Data that would normally create homopolymers
        let data = vec![0xFF, 0xFF, 0xFF, 0xFF];
        let encoded = encoder.encode_bytes(&data).unwrap();
        
        // Verify no homopolymer violations
        assert!(encoder.buffer.has_homopolymer_violation().is_none());
    }

    #[test]
    fn test_gc_content() {
        let mut encoder = NucleotideEncoder::new(1024, false, true);
        let data: Vec<u8> = (0..100).collect();
        
        let _ = encoder.encode_bytes(&data).unwrap();
        let gc = encoder.current_gc_content();
        
        assert!(gc >= MIN_GC_CONTENT);
        assert!(gc <= MAX_GC_CONTENT);
    }
}
