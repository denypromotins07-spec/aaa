//! SIMD-accelerated Hamming Distance Cleanup Memory
//! Snaps noisy queries to nearest stored "Platonic" vectors

use crate::hdc::bipolar_vector_generator::{BipolarVector, HDC_DIMENSION, BipolarVectorError};
use thiserror::Error;

/// Error types for cleanup memory operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum CleanupMemoryError {
    #[error("No items stored in cleanup memory")]
    Empty,
    #[error("No match found within threshold")]
    NoMatch,
    #[error("Invalid threshold: {value}. Must be between 0 and {max}")]
    InvalidThreshold { value: f64, max: f64 },
}

/// Configuration for cleanup memory
#[derive(Debug, Clone)]
pub struct CleanupMemoryConfig {
    /// Maximum Hamming distance for a valid match
    pub max_hamming_distance: u32,
    /// Minimum similarity threshold (alternative to max_hamming)
    pub min_similarity: f64,
    /// Use SIMD optimization when available
    pub use_simd: bool,
}

impl CleanupMemoryConfig {
    pub fn new(max_hamming_distance: u32, min_similarity: f64) -> Result<Self, CleanupMemoryError> {
        if min_similarity < -1.0 || min_similarity > 1.0 {
            return Err(CleanupMemoryError::InvalidThreshold {
                value: min_similarity,
                max: 1.0,
            });
        }

        Ok(Self {
            max_hamming_distance,
            min_similarity,
            use_simd: true,
        })
    }

    /// Get recommended configuration
    pub fn recommended() -> Self {
        // For D=10000, expected random distance is 5000
        // Set threshold at ~3 standard deviations for reliable retrieval
        let std_dev = (HDC_DIMENSION as f64 / 4.0).sqrt(); // sqrt(D/4) for bipolar vectors
        let max_dist = (HDC_DIMENSION as f64 / 2.0 - 3.0 * std_dev) as u32;
        
        Self {
            max_hamming_distance: max_dist.max(100),
            min_similarity: 0.7,
            use_simd: true,
        }
    }
}

/// SIMD-accelerated Hamming distance calculator
#[inline(always)]
pub fn hamming_distance_simd(a: &BipolarVector, b: &BipolarVector) -> u32 {
    // Process 8 u64 words at a time for better instruction-level parallelism
    let a_bits = a.as_bits();
    let b_bits = b.as_bits();
    
    let mut distance: u32 = 0;
    
    // Unroll loop for better performance
    let chunks = a_bits.chunks_exact(8);
    let remainder = chunks.remainder();
    
    for chunk in chunks {
        distance += (chunk[0] ^ b_bits[chunk.as_ptr() as usize - a_bits.as_ptr() as usize]).count_ones();
        distance += (chunk[1] ^ b_bits[chunk.as_ptr() as usize - a_bits.as_ptr() as usize + 1]).count_ones();
        distance += (chunk[2] ^ b_bits[chunk.as_ptr() as usize - a_bits.as_ptr() as usize + 2]).count_ones();
        distance += (chunk[3] ^ b_bits[chunk.as_ptr() as usize - a_bits.as_ptr() as usize + 3]).count_ones();
        distance += (chunk[4] ^ b_bits[chunk.as_ptr() as usize - a_bits.as_ptr() as usize + 4]).count_ones();
        distance += (chunk[5] ^ b_bits[chunk.as_ptr() as usize - a_bits.as_ptr() as usize + 5]).count_ones();
        distance += (chunk[6] ^ b_bits[chunk.as_ptr() as usize - a_bits.as_ptr() as usize + 6]).count_ones();
        distance += (chunk[7] ^ b_bits[chunk.as_ptr() as usize - a_bits.as_ptr() as usize + 7]).count_ones();
    }
    
    // Handle remainder
    for i in 0..remainder.len() {
        let base_idx = chunks.len() * 8 + i;
        distance += (remainder[i] ^ b_bits[base_idx]).count_ones();
    }
    
    distance
}

/// Standard Hamming distance (fallback)
#[inline(always)]
pub fn hamming_distance_standard(a: &BipolarVector, b: &BipolarVector) -> u32 {
    a.hamming_distance(b)
}

/// Cleanup Memory for snapping noisy vectors to stored prototypes
pub struct SimdHammingCleanupMemory {
    config: CleanupMemoryConfig,
    /// Stored prototype vectors ("Platonic" forms)
    prototypes: Vec<BipolarVector>,
    /// Optional metadata for each prototype
    metadata: Vec<Vec<u8>>,
}

impl SimdHammingCleanupMemory {
    /// Create a new cleanup memory
    pub fn new(config: CleanupMemoryConfig) -> Self {
        Self {
            config,
            prototypes: Vec::new(),
            metadata: Vec::new(),
        }
    }

    /// Add a prototype vector
    pub fn add_prototype(&mut self, vector: BipolarVector, metadata: Option<Vec<u8>>) -> usize {
        let id = self.prototypes.len();
        self.prototypes.push(vector);
        self.metadata.push(metadata.unwrap_or_default());
        id
    }

    /// Clean up a noisy query vector by finding nearest prototype
    pub fn cleanup(&self, query: &BipolarVector) -> Result<(usize, &BipolarVector, f64), CleanupMemoryError> {
        if self.prototypes.is_empty() {
            return Err(CleanupMemoryError::Empty);
        }

        let mut best_idx = 0;
        let mut best_distance = u32::MAX;

        for (idx, prototype) in self.prototypes.iter().enumerate() {
            let distance = if self.config.use_simd {
                hamming_distance_simd(query, prototype)
            } else {
                hamming_distance_standard(query, prototype)
            };

            if distance < best_distance {
                best_distance = distance;
                best_idx = idx;
            }
        }

        // Check if best match meets threshold
        let similarity = 1.0 - 2.0 * (best_distance as f64 / HDC_DIMENSION as f64);
        
        if best_distance > self.config.max_hamming_distance 
            || similarity < self.config.min_similarity {
            return Err(CleanupMemoryError::NoMatch);
        }

        Ok((best_idx, &self.prototypes[best_idx], similarity))
    }

    /// Clean up and return only the prototype (convenience method)
    pub fn cleanup_vector(&self, query: &BipolarVector) -> Result<&BipolarVector, CleanupMemoryError> {
        self.cleanup(query).map(|(_, vec, _)| vec)
    }

    /// Batch cleanup multiple queries
    pub fn batch_cleanup<'a>(
        &'a self,
        queries: &[&'a BipolarVector],
    ) -> Vec<Result<(usize, &'a BipolarVector, f64), CleanupMemoryError>> {
        queries.iter().map(|q| self.cleanup(q)).collect()
    }

    /// Get all prototypes
    pub fn prototypes(&self) -> &[BipolarVector] {
        &self.prototypes
    }

    /// Get metadata for a prototype
    pub fn get_metadata(&self, idx: usize) -> Option<&[u8]> {
        self.metadata.get(idx).map(|v| v.as_slice())
    }

    /// Clear all prototypes
    pub fn clear(&mut self) {
        self.prototypes.clear();
        self.metadata.clear();
    }

    /// Get number of stored prototypes
    pub fn len(&self) -> usize {
        self.prototypes.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.prototypes.is_empty()
    }
}

/// Streaming cleanup memory with sliding window
pub struct StreamingCleanupMemory {
    base_memory: SimdHammingCleanupMemory,
    /// Maximum number of prototypes before eviction
    max_prototypes: usize,
    /// Access counts for LRU-style eviction
    access_counts: Vec<u64>,
}

impl StreamingCleanupMemory {
    pub fn new(config: CleanupMemoryConfig, max_prototypes: usize) -> Self {
        Self {
            base_memory: SimdHammingCleanupMemory::new(config),
            max_prototypes,
            access_counts: Vec::new(),
        }
    }

    /// Add or update a prototype
    pub fn add(&mut self, vector: BipolarVector, metadata: Option<Vec<u8>>) -> Result<usize, CleanupMemoryError> {
        // Check if similar prototype exists
        if let Ok((idx, _, sim)) = self.base_memory.cleanup(&vector) {
            if sim > 0.99 {
                // Nearly identical, just increment access count
                if idx < self.access_counts.len() {
                    self.access_counts[idx] += 1;
                }
                return Ok(idx);
            }
        }

        // Need to add new prototype
        if self.base_memory.len() >= self.max_prototypes {
            // Evict least accessed
            if let Some(min_idx) = self.access_counts
                .iter()
                .enumerate()
                .min_by_key(|(_, count)| **count)
                .map(|(idx, _)| idx)
            {
                // In production, would properly remove and reindex
                // For now, just append (simplified implementation)
            }
        }

        let id = self.base_memory.add_prototype(vector, metadata);
        self.access_counts.push(0);
        Ok(id)
    }

    /// Clean up a query
    pub fn cleanup(&mut self, query: &BipolarVector) -> Result<(usize, &BipolarVector, f64), CleanupMemoryError> {
        let result = self.base_memory.cleanup(query);
        
        if let Ok((idx, _, _)) = result {
            if idx < self.access_counts.len() {
                self.access_counts[idx] += 1;
            }
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::bipolar_vector_generator::BipolarVectorGenerator;

    #[test]
    fn test_cleanup_memory_basic() {
        let config = CleanupMemoryConfig::recommended();
        let mut memory = SimdHammingCleanupMemory::new(config);

        let mut gen = BipolarVectorGenerator::new(42);
        let proto = gen.generate().unwrap();
        
        memory.add_prototype(proto.clone(), Some(vec![1, 2, 3]));

        // Query with exact same vector
        let (idx, retrieved, sim) = memory.cleanup(&proto).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(sim, 1.0);
    }

    #[test]
    fn test_cleanup_with_noise() {
        let config = CleanupMemoryConfig::recommended();
        let mut memory = SimdHammingCleanupMemory::new(config);

        let mut gen = BipolarVectorGenerator::new(123);
        let proto = gen.generate().unwrap();
        memory.add_prototype(proto.clone(), None);

        // Create slightly noisy version by bundling
        let noise = gen.generate().unwrap();
        let refs = [&proto, &proto, &noise];
        
        use crate::hdc::simd_binding_bundling::bundle_vectors;
        let noisy = bundle_vectors(&refs).unwrap();

        // Should still clean up to original prototype
        let result = memory.cleanup(&noisy);
        assert!(result.is_ok());
        let (_, retrieved, sim) = result.unwrap();
        assert!(sim > 0.8);
    }

    #[test]
    fn test_no_match_threshold() {
        let config = CleanupMemoryConfig::new(100, 0.95).unwrap(); // Very strict threshold
        let mut memory = SimdHammingCleanupMemory::new(config);

        let mut gen = BipolarVectorGenerator::new(456);
        let proto = gen.generate().unwrap();
        memory.add_prototype(proto, None);

        // Query with completely different vector
        let different = gen.generate().unwrap();
        let result = memory.cleanup(&different);
        
        // Should fail due to strict threshold
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_cleanup() {
        let config = CleanupMemoryConfig::recommended();
        let mut memory = SimdHammingCleanupMemory::new(config);

        let mut gen = BipolarVectorGenerator::new(789);
        
        for i in 0..5 {
            let v = gen.generate().unwrap();
            memory.add_prototype(v, Some(vec![i as u8]));
        }

        let queries: Vec<BipolarVector> = (0..3)
            .map(|_| gen.generate().unwrap())
            .collect();
        let query_refs: Vec<&BipolarVector> = queries.iter().collect();

        let results = memory.batch_cleanup(&query_refs);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_streaming_cleanup() {
        let config = CleanupMemoryConfig::recommended();
        let mut memory = StreamingCleanupMemory::new(config, 100);

        let mut gen = BipolarVectorGenerator::new(111);
        let v = gen.generate().unwrap();
        
        let id = memory.add(v.clone(), None).unwrap();
        assert_eq!(id, 0);

        let result = memory.cleanup(&v);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hamming_distance_symmetry() {
        let mut gen = BipolarVectorGenerator::new(222);
        let v1 = gen.generate().unwrap();
        let v2 = gen.generate().unwrap();

        let d1 = hamming_distance_simd(&v1, &v2);
        let d2 = hamming_distance_simd(&v2, &v1);

        assert_eq!(d1, d2);
    }

    #[test]
    fn test_simd_vs_standard_consistency() {
        let mut gen = BipolarVectorGenerator::new(333);
        let v1 = gen.generate().unwrap();
        let v2 = gen.generate().unwrap();

        let simd_dist = hamming_distance_simd(&v1, &v2);
        let std_dist = hamming_distance_standard(&v1, &v2);

        assert_eq!(simd_dist, std_dist);
    }
}
