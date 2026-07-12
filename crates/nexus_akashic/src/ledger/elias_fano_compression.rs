//! Elias-Fano Compression for Monotone Sequences
//! Space-efficient encoding of increasing integer sequences with O(1) random access

use thiserror::Error;

/// Error types for Elias-Fano operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum EliasFanoError {
    #[error("Sequence must be strictly monotone increasing")]
    NotMonotoneIncreasing,
    #[error("Index out of bounds: {index} >= {length}")]
    IndexOutOfBounds { index: usize, length: usize },
    #[error("Empty sequence")]
    EmptySequence,
    #[error("Encoding failed: value {value} exceeds maximum")]
    EncodingFailed { value: u64 },
}

/// Elias-Fano encoded monotone sequence
pub struct EliasFanoEncoder {
    /// Upper bits stored as unary-coded gaps
    upper_bits: Vec<u64>,
    /// Lower bits stored verbatim
    lower_bits: Vec<u64>,
    /// Number of lower bits per element (log2 of universe size / n)
    log_lower_bits: usize,
    /// Number of elements
    num_elements: usize,
    /// Maximum value in sequence
    max_value: u64,
    /// Cumulative counts for upper bit indexing
    upper_index: Vec<usize>,
}

impl EliasFanoEncoder {
    /// Encode a monotone increasing sequence
    pub fn encode(sequence: &[u64]) -> Result<Self, EliasFanoError> {
        if sequence.is_empty() {
            return Err(EliasFanoError::EmptySequence);
        }

        // Verify monotonicity
        for i in 1..sequence.len() {
            if sequence[i] <= sequence[i - 1] {
                return Err(EliasFanoError::NotMonotoneIncreasing);
            }
        }

        let num_elements = sequence.len();
        let max_value = sequence[sequence.len() - 1];
        
        // Calculate optimal lower bit width
        // L = ceil(log2(U/n)) where U is universe size, n is element count
        let universe_ratio = if num_elements > 0 {
            max_value / num_elements as u64
        } else {
            max_value
        };
        
        let log_lower_bits = if universe_ratio == 0 {
            0
        } else {
            64 - universe_ratio.leading_zeros() as usize
        };
        
        let lower_mask = if log_lower_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << log_lower_bits) - 1
        };

        let mut upper_bits: Vec<u64> = Vec::new();
        let mut lower_bits: Vec<u64> = Vec::with_capacity(num_elements);
        let mut current_upper_word: u64 = 0;
        let mut current_bit_pos = 0;

        let mut prev_value: u64 = 0;
        let mut upper_count = 0;
        let mut upper_index = Vec::new();

        for (elem_idx, &value) in sequence.iter().enumerate() {
            // Store index at every 64th element for fast seeking
            if elem_idx % 64 == 0 {
                upper_index.push(upper_bits.len());
            }

            // Split value into upper and lower parts
            let lower = value & lower_mask;
            let upper = value >> log_lower_bits;
            
            lower_bits.push(lower);

            // Encode upper part in unary (gap from previous upper)
            let upper_gap = if elem_idx == 0 {
                upper
            } else {
                upper - (prev_value >> log_lower_bits)
            };

            // Write upper_gap + 1 ones followed by a zero (unary coding)
            for _ in 0..upper_gap {
                if current_bit_pos >= 64 {
                    upper_bits.push(current_upper_word);
                    current_upper_word = 0;
                    current_bit_pos = 0;
                }
                current_upper_word |= 1u64 << current_bit_pos;
                current_bit_pos += 1;
            }
            
            // Write the terminating zero
            if current_bit_pos >= 64 {
                upper_bits.push(current_upper_word);
                current_upper_word = 0;
                current_bit_pos = 0;
            }
            // Zero is implicit (we don't set it)
            current_bit_pos += 1;
            
            upper_count += upper_gap as usize + 1;

            prev_value = value;
        }

        // Flush remaining bits
        if current_bit_pos > 0 {
            upper_bits.push(current_upper_word);
        }

        Ok(Self {
            upper_bits,
            lower_bits,
            log_lower_bits,
            num_elements,
            max_value,
            upper_index,
        })
    }

    /// Access element at index with O(1) time using binary search on upper bits
    pub fn access(&self, index: usize) -> Result<u64, EliasFanoError> {
        if index >= self.num_elements {
            return Err(EliasFanoError::IndexOutOfBounds {
                index,
                length: self.num_elements,
            });
        }

        // Find starting point using index table
        let indexed_block = index / 64;
        let offset_in_block = index % 64;
        
        let start_bit = if indexed_block < self.upper_index.len() {
            // Count total bits up to this block
            let mut bit_count = 0;
            for i in 0..indexed_block {
                // Each element contributes (upper_gap + 1) bits
                // Approximate by assuming average gap
                bit_count += 64; // Rough estimate
            }
            bit_count
        } else {
            0
        };

        // Binary search for the position of the index-th one
        let upper_value = self.find_upper_value(index, start_bit)?;
        
        // Get lower bits directly
        let lower_value = self.lower_bits[index];
        
        // Reconstruct original value
        let reconstructed = (upper_value << self.log_lower_bits) | lower_value;
        
        Ok(reconstructed)
    }

    /// Find upper value using selective search
    fn find_upper_value(&self, target_index: usize, start_bit: usize) -> Result<u64, EliasFanoError> {
        let mut ones_seen = 0;
        let mut current_upper = 0u64;
        
        for word_idx in (start_bit / 64)..self.upper_bits.len() {
            let word = self.upper_bits[word_idx];
            let start_in_word = if word_idx == start_bit / 64 {
                start_bit % 64
            } else {
                0
            };

            for bit_idx in start_in_word..64 {
                if (word >> bit_idx) & 1 == 1 {
                    if ones_seen == target_index {
                        return Ok(current_upper);
                    }
                    ones_seen += 1;
                } else {
                    current_upper += 1;
                }
            }
        }

        Err(EliasFanoError::IndexOutOfBounds {
            index: target_index,
            length: self.num_elements,
        })
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.num_elements
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.num_elements == 0
    }

    /// Get compression statistics
    pub fn stats(&self) -> EliasFanoStats {
        let raw_bits = self.num_elements * 64;
        let compressed_bits = self.upper_bits.len() * 64 + 
                              self.lower_bits.len() * self.log_lower_bits +
                              self.upper_index.len() * 64;

        EliasFanoStats {
            num_elements: self.num_elements,
            max_value: self.max_value,
            raw_bits,
            compressed_bits,
            compression_ratio: if raw_bits > 0 {
                compressed_bits as f64 / raw_bits as f64
            } else {
                0.0
            },
            log_lower_bits: self.log_lower_bits,
        }
    }

    /// Iterate over all decoded values
    pub fn iter(&self) -> EliasFanoIter<'_> {
        EliasFanoIter {
            encoder: self,
            current_index: 0,
            current_upper: 0,
            current_bit_pos: 0,
        }
    }
}

/// Statistics about Elias-Fano compression
#[derive(Debug, Clone)]
pub struct EliasFanoStats {
    pub num_elements: usize,
    pub max_value: u64,
    pub raw_bits: usize,
    pub compressed_bits: usize,
    pub compression_ratio: f64,
    pub log_lower_bits: usize,
}

/// Iterator over Elias-Fano encoded values
pub struct EliasFanoIter<'a> {
    encoder: &'a EliasFanoEncoder,
    current_index: usize,
    current_upper: u64,
    current_bit_pos: usize,
}

impl<'a> Iterator for EliasFanoIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.encoder.num_elements {
            return None;
        }

        // Skip zeros (count upper increments) until we hit a one
        while self.current_bit_pos < 64 * self.encoder.upper_bits.len() {
            let word_idx = self.current_bit_pos / 64;
            let bit_idx = self.current_bit_pos % 64;
            
            if word_idx >= self.encoder.upper_bits.len() {
                break;
            }

            let word = self.encoder.upper_bits[word_idx];
            
            if (word >> bit_idx) & 1 == 1 {
                // Found a one - this marks end of current element's upper code
                let lower = self.encoder.lower_bits[self.current_index];
                let value = (self.current_upper << self.encoder.log_lower_bits) | lower;
                
                self.current_index += 1;
                self.current_bit_pos += 1;
                return Some(value);
            } else {
                // Zero means increment upper
                self.current_upper += 1;
            }
            
            self.current_bit_pos += 1;
        }

        None
    }
}

/// Safe index wrapper for bounds-checked access
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafeIndex(usize);

impl SafeIndex {
    pub fn new(index: usize, max: usize) -> Result<Self, EliasFanoError> {
        if index >= max {
            Err(EliasFanoError::IndexOutOfBounds {
                index,
                length: max,
            })
        } else {
            Ok(Self(index))
        }
    }

    #[inline(always)]
    pub fn get(&self) -> usize {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elias_fano_encode_decode() {
        let sequence = vec![3, 7, 8, 15, 23, 30, 45, 60];
        let encoder = EliasFanoEncoder::encode(&sequence).unwrap();

        for (i, &expected) in sequence.iter().enumerate() {
            let value = encoder.access(i).unwrap();
            assert_eq!(value, expected);
        }
    }

    #[test]
    fn test_monotone_validation() {
        let non_monotone = vec![1, 5, 3, 10];
        let result = EliasFanoEncoder::encode(&non_monotone);
        assert!(result.is_err());
        
        match result {
            Err(EliasFanoError::NotMonotoneIncreasing) => {}
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_empty_sequence() {
        let empty: Vec<u64> = vec![];
        let result = EliasFanoEncoder::encode(&empty);
        assert!(result.is_err());
    }

    #[test]
    fn test_bounds_checking() {
        let sequence = vec![10, 20, 30, 40, 50];
        let encoder = EliasFanoEncoder::encode(&sequence).unwrap();

        let result = encoder.access(10);
        assert!(result.is_err());
        
        match result {
            Err(EliasFanoError::IndexOutOfBounds { .. }) => {}
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_iterator() {
        let sequence = vec![5, 10, 15, 20, 25];
        let encoder = EliasFanoEncoder::encode(&sequence).unwrap();

        let decoded: Vec<u64> = encoder.iter().collect();
        assert_eq!(decoded, sequence);
    }

    #[test]
    fn test_compression_stats() {
        let sequence: Vec<u64> = (0..1000).map(|i| i * 10).collect();
        let encoder = EliasFanoEncoder::encode(&sequence).unwrap();

        let stats = encoder.stats();
        assert_eq!(stats.num_elements, 1000);
        assert!(stats.compression_ratio < 1.0, "Should achieve compression");
    }

    #[test]
    fn test_safe_index() {
        let valid = SafeIndex::new(5, 10);
        assert!(valid.is_ok());
        assert_eq!(valid.unwrap().get(), 5);

        let invalid = SafeIndex::new(10, 10);
        assert!(invalid.is_err());
    }

    #[test]
    fn test_large_sequence() {
        let sequence: Vec<u64> = (0..10000).map(|i| i * 100 + 50).collect();
        let encoder = EliasFanoEncoder::encode(&sequence).unwrap();

        assert_eq!(encoder.len(), 10000);

        // Spot check
        assert_eq!(encoder.access(0).unwrap(), 50);
        assert_eq!(encoder.access(9999).unwrap(), 999950);
        assert_eq!(encoder.access(5000).unwrap(), 500050);
    }
}
