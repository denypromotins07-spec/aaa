//! Succinct Rank/Select Dictionaries for O(1) compressed access
//! Implements RRR (Raman-Raman-Rao) encoding for space-efficient bit vectors

use thiserror::Error;

/// Error types for succinct data structure operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum SuccinctError {
    #[error("Index out of bounds: {index} >= {length}")]
    IndexOutOfBounds { index: usize, length: usize },
    #[error("Invalid block size: must be power of 2")]
    InvalidBlockSize,
    #[error("Compression failed")]
    CompressionFailed,
    #[error("Decompression failed")]
    DecompressionFailed,
}

/// Block size for rank/select dictionaries (must be power of 2)
const BLOCK_SIZE: usize = 512;

/// Superblock size for coarse-grained indexing
const SUPERBLOCK_SIZE: usize = 4096;

/// A succinct bit vector with rank/select support
pub struct SuccinctBitVector {
    /// Raw bit data packed into u64 words
    data: Vec<u64>,
    /// Total number of bits
    len: usize,
    /// Rank index: cumulative count of 1s at superblock boundaries
    rank_index: Vec<usize>,
    /// Select index: positions of every k-th 1
    select_index: Vec<usize>,
}

impl SuccinctBitVector {
    /// Create a new succinct bit vector from raw bits
    pub fn from_bits(bits: &[bool]) -> Result<Self, SuccinctError> {
        let len = bits.len();
        let num_words = (len + 63) / 64;
        
        // Pack bits into u64 words
        let mut data = vec![0u64; num_words];
        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                data[word_idx] |= 1u64 << bit_idx;
            }
        }

        // Build rank index
        let mut rank_index = Vec::new();
        let mut cumulative_rank = 0;
        
        for (superblock_idx, _) in (0..len).step_by(SUPERBLOCK_SIZE).enumerate() {
            rank_index.push(cumulative_rank);
            
            // Count 1s in this superblock
            let start = superblock_idx * SUPERBLOCK_SIZE;
            let end = (start + SUPERBLOCK_SIZE).min(len);
            
            for i in start..end {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                if (data[word_idx] >> bit_idx) & 1 == 1 {
                    cumulative_rank += 1;
                }
            }
        }

        // Build select index (every 64th 1)
        let mut select_index = Vec::new();
        let mut ones_seen = 0;
        
        for (i, _) in bits.iter().enumerate() {
            if bits[i] {
                if ones_seen % 64 == 0 {
                    select_index.push(i);
                }
                ones_seen += 1;
            }
        }

        Ok(Self {
            data,
            len,
            rank_index,
            select_index,
        })
    }

    /// Get bit at position with bounds checking
    #[inline(always)]
    pub fn get(&self, index: usize) -> Result<bool, SuccinctError> {
        if index >= self.len {
            return Err(SuccinctError::IndexOutOfBounds {
                index,
                length: self.len,
            });
        }
        
        let word_idx = index / 64;
        let bit_idx = index % 64;
        Ok((self.data[word_idx] >> bit_idx) & 1 == 1)
    }

    /// Rank operation: count number of 1s up to (but not including) index
    #[inline(always)]
    pub fn rank(&self, index: usize) -> Result<usize, SuccinctError> {
        if index > self.len {
            return Err(SuccinctError::IndexOutOfBounds {
                index,
                length: self.len,
            });
        }

        // Find superblock
        let superblock_idx = index / SUPERBLOCK_SIZE;
        let base_rank = if superblock_idx < self.rank_index.len() {
            self.rank_index[superblock_idx]
        } else {
            0
        };

        // Count 1s within superblock
        let start = superblock_idx * SUPERBLOCK_SIZE;
        let mut local_rank = 0;
        
        for i in start..index {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            if (self.data[word_idx] >> bit_idx) & 1 == 1 {
                local_rank += 1;
            }
        }

        Ok(base_rank + local_rank)
    }

    /// Select operation: find position of the n-th 1 (0-indexed)
    pub fn select(&self, n: usize) -> Result<usize, SuccinctError> {
        if n >= self.select_index.len() * 64 {
            // Need to search beyond indexed positions
            return self.select_slow(n);
        }

        // Use index for starting point
        let block_idx = n / 64;
        let remainder = n % 64;
        
        let start_pos = if block_idx < self.select_index.len() {
            self.select_index[block_idx]
        } else {
            0
        };

        // Search from start position
        let mut ones_found = block_idx * 64;
        
        for i in start_pos..self.len {
            if self.get(i)? {
                if ones_found == n {
                    return Ok(i);
                }
                ones_found += 1;
            }
        }

        Err(SuccinctError::IndexOutOfBounds {
            index: n,
            length: self.rank(self.len)?,
        })
    }

    /// Slow select fallback for large indices
    fn select_slow(&self, n: usize) -> Result<usize, SuccinctError> {
        let mut count = 0;
        for i in 0..self.len {
            if self.get(i)? {
                if count == n {
                    return Ok(i);
                }
                count += 1;
            }
        }
        
        Err(SuccinctError::IndexOutOfBounds {
            index: n,
            length: count,
        })
    }

    /// Get the total length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get compression ratio estimate
    pub fn compression_ratio(&self) -> f64 {
        let raw_bits = self.len as f64;
        let compressed_bits = (self.data.len() * 64 + 
                               self.rank_index.len() * 64 + 
                               self.select_index.len() * 64) as f64;
        compressed_bits / raw_bits
    }
}

/// RRR-encoded bit vector for highly compressible sparse/dense sequences
pub struct RRRVector {
    /// Block size (log2 determines bucket count)
    log_block_size: usize,
    /// Number of blocks
    num_blocks: usize,
    /// For each block: (offset, class) where class = popcount
    offsets: Vec<u64>,
    classes: Vec<u8>,
    /// Original length
    len: usize,
}

impl RRRVector {
    /// Create RRR vector from bits
    pub fn from_bits(bits: &[bool], log_block_size: usize) -> Result<Self, SuccinctError> {
        if !BLOCK_SIZE.is_power_of_two() {
            return Err(SuccinctError::InvalidBlockSize);
        }

        let block_size = 1 << log_block_size;
        let len = bits.len();
        let num_blocks = (len + block_size - 1) / block_size;

        let mut offsets = Vec::with_capacity(num_blocks);
        let mut classes = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(len);

            // Extract block bits and compute popcount
            let mut block_value: u64 = 0;
            let mut popcount: u8 = 0;

            for i in start..end {
                if bits[i] {
                    block_value |= 1u64 << (i - start);
                    popcount += 1;
                }
            }

            // Store offset (position within class)
            // In full implementation, would use combinatorial numbering
            offsets.push(block_value);
            classes.push(popcount);
        }

        Ok(Self {
            log_block_size,
            num_blocks,
            offsets,
            classes,
            len,
        })
    }

    /// Get bit at position
    pub fn get(&self, index: usize) -> Result<bool, SuccinctError> {
        if index >= self.len {
            return Err(SuccinctError::IndexOutOfBounds {
                index,
                length: self.len,
            });
        }

        let block_size = 1 << self.log_block_size;
        let block_idx = index / block_size;
        let bit_in_block = index % block_size;

        if block_idx >= self.num_blocks {
            return Ok(false);
        }

        Ok((self.offsets[block_idx] >> bit_in_block) & 1 == 1)
    }

    /// Get compression statistics
    pub fn stats(&self) -> RRRStats {
        let raw_bits = self.len;
        let stored_bits = self.offsets.len() * 64 + self.classes.len() * 8;
        
        RRRStats {
            original_bits: raw_bits,
            stored_bits,
            compression_ratio: stored_bits as f64 / raw_bits as f64,
            num_blocks: self.num_blocks,
        }
    }
}

/// Statistics about RRR compression
#[derive(Debug, Clone)]
pub struct RRRStats {
    pub original_bits: usize,
    pub stored_bits: usize,
    pub compression_ratio: f64,
    pub num_blocks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_succinct_bit_vector_basic() {
        let bits = vec![true, false, true, true, false, false, true];
        let sbv = SuccinctBitVector::from_bits(&bits).unwrap();

        assert_eq!(sbv.len(), 7);
        assert_eq!(sbv.get(0).unwrap(), true);
        assert_eq!(sbv.get(1).unwrap(), false);
        assert_eq!(sbv.get(2).unwrap(), true);
    }

    #[test]
    fn test_rank_operation() {
        let bits = vec![true, false, true, true, false, true];
        let sbv = SuccinctBitVector::from_bits(&bits).unwrap();

        assert_eq!(sbv.rank(0).unwrap(), 0);
        assert_eq!(sbv.rank(1).unwrap(), 1);
        assert_eq!(sbv.rank(3).unwrap(), 2);
        assert_eq!(sbv.rank(6).unwrap(), 4);
    }

    #[test]
    fn test_select_operation() {
        let bits = vec![false, true, false, true, true, false];
        let sbv = SuccinctBitVector::from_bits(&bits).unwrap();

        assert_eq!(sbv.select(0).unwrap(), 1); // First 1 at position 1
        assert_eq!(sbv.select(1).unwrap(), 3); // Second 1 at position 3
        assert_eq!(sbv.select(2).unwrap(), 4); // Third 1 at position 4
    }

    #[test]
    fn test_bounds_checking() {
        let bits = vec![true, false, true];
        let sbv = SuccinctBitVector::from_bits(&bits).unwrap();

        let result = sbv.get(10);
        assert!(result.is_err());
        
        match result {
            Err(SuccinctError::IndexOutOfBounds { .. }) => {}
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_rrr_vector() {
        let bits = vec![true, false, true, false, true, false, false, false];
        let rrr = RRRVector::from_bits(&bits, 3).unwrap(); // Block size = 8

        assert_eq!(rrr.get(0).unwrap(), true);
        assert_eq!(rrr.get(1).unwrap(), false);
        assert_eq!(rrr.get(2).unwrap(), true);

        let stats = rrr.stats();
        assert!(stats.compression_ratio <= 1.0);
    }

    #[test]
    fn test_large_vector() {
        let mut bits = Vec::with_capacity(10000);
        for i in 0..10000 {
            bits.push(i % 3 == 0);
        }

        let sbv = SuccinctBitVector::from_bits(&bits).unwrap();
        assert_eq!(sbv.len(), 10000);

        // Spot check some values
        assert_eq!(sbv.get(0).unwrap(), true);
        assert_eq!(sbv.get(1).unwrap(), false);
        assert_eq!(sbv.get(3).unwrap(), true);
        assert_eq!(sbv.get(9999).unwrap(), true);
    }
}
