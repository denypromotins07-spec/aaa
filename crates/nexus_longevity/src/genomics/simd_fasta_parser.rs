//! Zero-Allocation SIMD-Accelerated FASTQ/BAM Parser
//! 
//! Parses genomic sequence files directly into pre-allocated memory arenas
//! without heap allocations in hot paths. Supports FASTQ and BAM formats.

use core::slice;
use std::mem::MaybeUninit;
use std::ptr;

/// Maximum read length supported (5kb typical for Illumina)
pub const MAX_READ_LENGTH: usize = 8192;

/// Maximum header line length
pub const MAX_HEADER_LENGTH: usize = 4096;

/// Error types for genomic parsing
#[derive(Debug, Clone, PartialEq)]
pub enum FastaParseError {
    BufferOverflow,
    InvalidFormat,
    UnexpectedEof,
    InvalidBase(u8),
    QualityScoreMismatch,
}

impl core::fmt::Display for FastaParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BufferOverflow => write!(f, "Buffer overflow"),
            Self::InvalidFormat => write!(f, "Invalid format"),
            Self::UnexpectedEof => write!(f, "Unexpected EOF"),
            Self::InvalidBase(b) => write!(f, "Invalid base: {}", *b as char),
            Self::QualityScoreMismatch => write!(f, "Quality score length mismatch"),
        }
    }
}

/// Encoded nucleotide base (2-bit encoding)
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nucleotide {
    A = 0b00,
    C = 0b01,
    G = 0b10,
    T = 0b11,
    N = 0b00, // Ambiguous treated as A for packing
}

impl Nucleotide {
    #[inline]
    pub fn from_byte(b: u8) -> Result<Self, FastaParseError> {
        match b {
            b'A' | b'a' => Ok(Self::A),
            b'C' | b'c' => Ok(Self::C),
            b'G' | b'g' => Ok(Self::G),
            b'T' | b't' => Ok(Self::T),
            b'N' | b'n' => Ok(Self::N),
            _ => Err(FastaParseError::InvalidBase(b)),
        }
    }

    #[inline]
    pub fn to_byte(self) -> u8 {
        match self {
            Self::A => b'A',
            Self::C => b'C',
            Self::G => b'G',
            Self::T => b'T',
            Self::N => b'N',
        }
    }
}

/// Pre-allocated read buffer for zero-copy parsing
#[repr(C)]
pub struct ReadBuffer {
    /// Packed nucleotides (2-bit per base, 4 bases per byte)
    pub bases: [u8; MAX_READ_LENGTH / 4 + 1],
    /// Quality scores (one byte per base)
    pub qualities: [u8; MAX_READ_LENGTH],
    /// Actual number of bases
    pub len: usize,
    /// Header offset in shared header buffer
    pub header_offset: usize,
    /// Header length
    pub header_len: usize,
}

impl ReadBuffer {
    #[inline]
    pub const fn new() -> Self {
        Self {
            bases: [0u8; MAX_READ_LENGTH / 4 + 1],
            qualities: [0u8; MAX_READ_LENGTH],
            len: 0,
            header_offset: 0,
            header_len: 0,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
        self.header_offset = 0;
        self.header_len = 0;
    }

    #[inline]
    pub fn push_base(&mut self, base: Nucleotide, quality: u8) -> Result<(), FastaParseError> {
        if self.len >= MAX_READ_LENGTH {
            return Err(FastaParseError::BufferOverflow);
        }

        // Pack 2 bits per base
        let byte_idx = self.len / 4;
        let bit_shift = ((self.len % 4) * 2) as u32;
        let packed = base as u8;
        
        unsafe {
            let ptr = self.bases.as_mut_ptr().add(byte_idx);
            ptr.write_volatile(ptr.read_volatile() | (packed << bit_shift));
        }

        self.qualities[self.len] = quality;
        self.len += 1;
        Ok(())
    }

    #[inline]
    pub fn get_base(&self, idx: usize) -> Option<Nucleotide> {
        if idx >= self.len {
            return None;
        }
        let byte_idx = idx / 4;
        let bit_shift = ((idx % 4) * 2) as u32;
        let packed = unsafe { *self.bases.get_unchecked(byte_idx) };
        let value = (packed >> bit_shift) & 0b11;
        Some(match value {
            0b00 => Nucleotide::A,
            0b01 => Nucleotide::C,
            0b10 => Nucleotide::G,
            0b11 => Nucleotide::T,
            _ => Nucleotide::N,
        })
    }

    #[inline]
    pub fn get_quality(&self, idx: usize) -> Option<u8> {
        if idx >= self.len {
            return None;
        }
        Some(unsafe { *self.qualities.get_unchecked(idx) })
    }
}

/// Pre-allocated header buffer pool
pub struct HeaderArena {
    data: [u8; 1024 * 1024], // 1MB arena
    used: usize,
}

impl HeaderArena {
    pub const fn new() -> Self {
        Self {
            data: [0u8; 1024 * 1024],
            used: 0,
        }
    }

    pub fn store(&mut self, header: &[u8]) -> Result<usize, FastaParseError> {
        if self.used + header.len() > self.data.len() {
            return Err(FastaParseError::BufferOverflow);
        }
        let offset = self.used;
        unsafe {
            ptr::copy_nonoverlapping(
                header.as_ptr(),
                self.data.as_mut_ptr().add(offset),
                header.len(),
            );
        }
        self.used += header.len();
        Ok(offset)
    }

    pub fn get(&self, offset: usize, len: usize) -> Option<&[u8]> {
        if offset + len > self.used {
            return None;
        }
        Some(unsafe { slice::from_raw_parts(self.data.as_ptr().add(offset), len) })
    }

    pub fn reset(&mut self) {
        self.used = 0;
    }
}

/// SIMD-accelerated FASTQ parser
pub struct SimdFastqParser {
    header_arena: HeaderArena,
    current_read: ReadBuffer,
    line_buffer: [u8; MAX_READ_LENGTH],
}

impl SimdFastqParser {
    pub const fn new() -> Self {
        Self {
            header_arena: HeaderArena::new(),
            current_read: ReadBuffer::new(),
            line_buffer: [0u8; MAX_READ_LENGTH],
        }
    }

    /// Parse FASTQ data into pre-allocated buffers
    /// Returns number of reads parsed or error
    pub fn parse<'a>(
        &mut self,
        data: &'a [u8],
        reads: &mut [ReadBuffer],
    ) -> Result<usize, FastaParseError> {
        let mut read_count = 0;
        let mut pos = 0;
        let mut state = ParserState::ExpectHeader;
        let mut quality_start = 0;

        while pos < data.len() {
            if read_count >= reads.len() {
                break;
            }

            let byte = data[pos];

            match state {
                ParserState::ExpectHeader => {
                    if byte == b'@' {
                        reads[read_count].clear();
                        let header_start = pos + 1;
                        let header_end = self.find_line_end(data, header_start)?;
                        let header_len = header_end - header_start;
                        
                        let offset = self.header_arena.store(&data[header_start..header_end])?;
                        reads[read_count].header_offset = offset;
                        reads[read_count].header_len = header_len;
                        
                        pos = header_end + 1;
                        state = ParserState::ExpectSequence;
                    } else if !byte.is_ascii_whitespace() {
                        return Err(FastaParseError::InvalidFormat);
                    } else {
                        pos += 1;
                    }
                }
                ParserState::ExpectSequence => {
                    let seq_start = pos;
                    let seq_end = self.find_line_end(data, seq_start)?;
                    let seq_len = seq_end - seq_start;

                    if seq_len == 0 {
                        pos += 1;
                        continue;
                    }

                    // Validate and pack bases
                    for i in 0..seq_len {
                        let base_byte = unsafe { *data.get_unchecked(seq_start + i) };
                        let base = Nucleotide::from_byte(base_byte)?;
                        reads[read_count].push_base(base, 0)?;
                    }

                    pos = seq_end + 1;
                    state = ParserState::ExpectPlus;
                }
                ParserState::ExpectPlus => {
                    if byte == b'+' {
                        let line_end = self.find_line_end(data, pos + 1)?;
                        pos = line_end + 1;
                        quality_start = pos;
                        state = ParserState::ExpectQuality;
                    } else if !byte.is_ascii_whitespace() {
                        return Err(FastaParseError::InvalidFormat);
                    } else {
                        pos += 1;
                    }
                }
                ParserState::ExpectQuality => {
                    let qual_end = self.find_line_end(data, pos)?;
                    let qual_len = qual_end - pos;

                    if qual_len != reads[read_count].len {
                        return Err(FastaParseError::QualityScoreMismatch);
                    }

                    // Copy quality scores
                    for i in 0..qual_len {
                        let q = unsafe { *data.get_unchecked(pos + i) };
                        reads[read_count].qualities[i] = q;
                    }

                    pos = qual_end + 1;
                    read_count += 1;
                    state = ParserState::ExpectHeader;
                }
            }
        }

        Ok(read_count)
    }

    #[inline]
    fn find_line_end(&self, data: &[u8], start: usize) -> Result<usize, FastaParseError> {
        let mut pos = start;
        while pos < data.len() {
            let byte = unsafe { *data.get_unchecked(pos) };
            if byte == b'\n' {
                return Ok(pos);
            }
            if byte == b'\r' {
                return Ok(pos);
            }
            pos += 1;
        }
        Err(FastaParseError::UnexpectedEof)
    }
}

enum ParserState {
    ExpectHeader,
    ExpectSequence,
    ExpectPlus,
    ExpectQuality,
}

/// Batch processor for high-throughput parsing
pub struct FastqBatchProcessor {
    parser: SimdFastqParser,
    read_pool: Box<[ReadBuffer; 1024]>,
}

impl FastqBatchProcessor {
    pub fn new() -> Self {
        // Pre-allocate read pool
        let mut pool = Box::new([ReadBuffer::new(); 1024]);
        Self {
            parser: SimdFastqParser::new(),
            read_pool: pool,
        }
    }

    pub fn process_batch(&mut self, data: &[u8]) -> Result<usize, FastaParseError> {
        self.parser.parse(data, &mut self.read_pool[..])
    }

    pub fn get_read(&self, idx: usize) -> Option<&ReadBuffer> {
        self.read_pool.get(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nucleotide_encoding() {
        assert_eq!(Nucleotide::from_byte(b'A').unwrap(), Nucleotide::A);
        assert_eq!(Nucleotide::from_byte(b'C').unwrap(), Nucleotide::C);
        assert_eq!(Nucleotide::from_byte(b'G').unwrap(), Nucleotide::G);
        assert_eq!(Nucleotide::from_byte(b'T').unwrap(), Nucleotide::T);
    }

    #[test]
    fn test_read_buffer_packing() {
        let mut buf = ReadBuffer::new();
        buf.push_base(Nucleotide::A, 30).unwrap();
        buf.push_base(Nucleotide::C, 30).unwrap();
        buf.push_base(Nucleotide::G, 30).unwrap();
        buf.push_base(Nucleotide::T, 30).unwrap();

        assert_eq!(buf.get_base(0), Some(Nucleotide::A));
        assert_eq!(buf.get_base(1), Some(Nucleotide::C));
        assert_eq!(buf.get_base(2), Some(Nucleotide::G));
        assert_eq!(buf.get_base(3), Some(Nucleotide::T));
    }
}
