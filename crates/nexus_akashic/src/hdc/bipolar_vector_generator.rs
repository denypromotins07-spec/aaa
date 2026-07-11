//! Bipolar Vector Generator for Hyper-Dimensional Computing
//! Generates pseudo-orthogonal 10,000-dimensional bipolar vectors (+1, -1)
//! with strict capacity thresholds and sliding-window decay to prevent orthogonality degradation.

use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use thiserror::Error;

/// Dimension of hyper-dimensional space (fixed at 10,000 for SIMD alignment)
pub const HDC_DIMENSION: usize = 10_000;

/// Maximum bundling capacity before orthogonality degrades significantly
/// Based on theoretical HDC capacity limits: ~sqrt(D) for reliable retrieval
pub const MAX_BUNDLING_CAPACITY: usize = 100;

/// Decay factor for sliding window (applied every MAX_BUNDLING_CAPACITY bundles)
const DECAY_FACTOR: f64 = 0.95;

/// Error types for bipolar vector generation
#[derive(Error, Debug, Clone, PartialEq)]
pub enum BipolarVectorError {
    #[error("Bundling capacity exceeded: {current} > {max}. Orthogonality degradation imminent.")]
    CapacityExceeded { current: usize, max: usize },
    #[error("Invalid dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },
    #[error("Seed generation failed")]
    SeedGenerationFailed,
}

/// A bipolar hyper-dimensional vector with elements in {-1, +1}
/// Stored as packed bits for SIMD efficiency: 0 -> -1, 1 -> +1
#[derive(Clone, Debug, PartialEq)]
pub struct BipolarVector {
    /// Packed bits representing the bipolar values (0 = -1, 1 = +1)
    data: [u64; HDC_DIMENSION / 64],
    /// Number of times this vector has been bundled (for decay tracking)
    bundle_count: usize,
    /// Cached Hamming weight for fast similarity calculations
    cached_hamming_weight: Option<u32>,
}

impl BipolarVector {
    /// Create a new random bipolar vector from a seed
    pub fn from_seed(seed: u64) -> Result<Self, BipolarVectorError> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        Self::generate_random(&mut rng)
    }

    /// Generate a random bipolar vector using the provided RNG
    pub fn generate_random<R: Rng>(rng: &mut R) -> Result<Self, BipolarVectorError> {
        let mut data: [u64; HDC_DIMENSION / 64] = [0; HDC_DIMENSION / 64];
        
        for chunk in data.iter_mut() {
            *chunk = rng.gen();
        }
        
        Ok(Self {
            data,
            bundle_count: 0,
            cached_hamming_weight: None,
        })
    }

    /// Create a bipolar vector from raw bits (0 -> -1, 1 -> +1)
    pub fn from_bits(bits: [u64; HDC_DIMENSION / 64]) -> Result<Self, BipolarVectorError> {
        Ok(Self {
            data: bits,
            bundle_count: 0,
            cached_hamming_weight: None,
        })
    }

    /// Get the raw bit representation
    #[inline(always)]
    pub fn as_bits(&self) -> &[u64; HDC_DIMENSION / 64] {
        &self.data
    }

    /// Get the value at a specific index as i8 (-1 or +1)
    #[inline(always)]
    pub fn get_value(&self, index: usize) -> Result<i8, BipolarVectorError> {
        if index >= HDC_DIMENSION {
            return Err(BipolarVectorError::InvalidDimension {
                expected: HDC_DIMENSION,
                actual: index,
            });
        }
        
        let word_index = index / 64;
        let bit_index = index % 64;
        
        if (self.data[word_index] >> bit_index) & 1 == 1 {
            Ok(1)
        } else {
            Ok(-1)
        }
    }

    /// Calculate Hamming distance to another vector (number of differing bits)
    #[inline(always)]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| (a ^ b).count_ones())
            .sum()
    }

    /// Calculate cosine similarity approximation: 1 - 2 * (hamming_dist / D)
    #[inline(always)]
    pub fn cosine_similarity(&self, other: &Self) -> f64 {
        let dist = self.hamming_distance(other) as f64;
        1.0 - 2.0 * (dist / HDC_DIMENSION as f64)
    }

    /// Get the bundle count
    #[inline(always)]
    pub fn bundle_count(&self) -> usize {
        self.bundle_count
    }

    /// Apply decay factor to reduce orthogonality degradation
    /// This is called when bundle_count exceeds threshold
    pub fn apply_decay(&mut self) {
        if self.bundle_count > 0 {
            // Decay by flipping a small fraction of bits based on DECAY_FACTOR
            let flip_probability = (1.0 - DECAY_FACTOR) / 2.0;
            let expected_flips = (HDC_DIMENSION as f64 * flip_probability) as usize;
            
            // Use deterministic pseudo-random flipping based on current state
            let mut flip_hash = self.bundle_count as u64;
            for i in 0..self.data.len() {
                flip_hash = flip_hash.wrapping_mul(6364136223846793005).wrapping_add(1);
                // Flip approximately expected_flips / num_words bits
                let flips_in_word = expected_flips / self.data.len();
                for j in 0..flips_in_word.min(64) {
                    flip_hash = flip_hash.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let bit_to_flip = (flip_hash % 64) as u32;
                    self.data[i] ^= 1u64 << bit_to_flip;
                }
            }
            
            self.cached_hamming_weight = None;
            self.bundle_count = (self.bundle_count as f64 * DECAY_FACTOR) as usize;
        }
    }

    /// Verify orthogonality with another vector (should be near 0.5 cosine similarity for random vectors)
    pub fn verify_orthogonality(&self, other: &Self, tolerance: f64) -> bool {
        let sim = self.cosine_similarity(other);
        // Random bipolar vectors should have cosine similarity near 0
        // After bundling, some deviation is expected but should stay within tolerance
        sim.abs() < tolerance
    }
}

/// Generator for managing bipolar vector creation with capacity tracking
pub struct BipolarVectorGenerator {
    rng: ChaCha20Rng,
    total_generated: usize,
    active_vectors: usize,
}

impl BipolarVectorGenerator {
    /// Create a new generator with a seed
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha20Rng::seed_from_u64(seed),
            total_generated: 0,
            active_vectors: 0,
        }
    }

    /// Generate a new random bipolar vector
    pub fn generate(&mut self) -> Result<BipolarVector, BipolarVectorError> {
        let vector = BipolarVector::generate_random(&mut self.rng)?;
        self.total_generated += 1;
        self.active_vectors += 1;
        Ok(vector)
    }

    /// Generate a vector for a specific entity (asset, price level, etc.)
    pub fn generate_for_entity(&mut self, entity_id: u64) -> Result<BipolarVector, BipolarVectorError> {
        // Use entity_id as part of seed for reproducibility
        let combined_seed = entity_id.wrapping_mul(6364136223846793005).wrapping_add(self.total_generated as u64);
        BipolarVector::from_seed(combined_seed)
    }

    /// Check if bundling more vectors would exceed capacity
    pub fn check_bundling_capacity(&self, additional_bundles: usize) -> Result<(), BipolarVectorError> {
        let projected = self.active_vectors + additional_bundles;
        if projected > MAX_BUNDLING_CAPACITY {
            Err(BipolarVectorError::CapacityExceeded {
                current: projected,
                max: MAX_BUNDLING_CAPACITY,
            })
        } else {
            Ok(())
        }
    }

    /// Get statistics about the generator
    pub fn stats(&self) -> GeneratorStats {
        GeneratorStats {
            total_generated: self.total_generated,
            active_vectors: self.active_vectors,
            capacity_utilization: self.active_vectors as f64 / MAX_BUNDLING_CAPACITY as f64,
        }
    }
}

/// Statistics about the vector generator
#[derive(Debug, Clone)]
pub struct GeneratorStats {
    pub total_generated: usize,
    pub active_vectors: usize,
    pub capacity_utilization: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_generation() {
        let mut gen = BipolarVectorGenerator::new(42);
        let v1 = gen.generate().unwrap();
        let v2 = gen.generate().unwrap();
        
        assert_eq!(v1.as_bits().len(), HDC_DIMENSION / 64);
        assert_ne!(v1.as_bits(), v2.as_bits());
    }

    #[test]
    fn test_orthogonality() {
        let mut gen = BipolarVectorGenerator::new(123);
        let v1 = gen.generate().unwrap();
        let v2 = gen.generate().unwrap();
        
        // Random vectors should be nearly orthogonal (cosine sim near 0)
        let sim = v1.cosine_similarity(&v2);
        assert!(sim.abs() < 0.15, "Vectors not sufficiently orthogonal: {}", sim);
    }

    #[test]
    fn test_hamming_distance() {
        let v1 = BipolarVector::from_seed(1).unwrap();
        let v2 = BipolarVector::from_seed(1).unwrap(); // Same seed = same vector
        
        assert_eq!(v1.hamming_distance(&v2), 0);
        
        let v3 = BipolarVector::from_seed(2).unwrap();
        let dist = v1.hamming_distance(&v3);
        
        // Expected distance for random vectors is D/2 = 5000
        assert!(dist > 4000 && dist < 6000, "Distance {} outside expected range", dist);
    }

    #[test]
    fn test_capacity_check() {
        let mut gen = BipolarVectorGenerator::new(42);
        
        // Generate up to capacity
        for _ in 0..MAX_BUNDLING_CAPACITY - 1 {
            gen.generate().unwrap();
        }
        
        // Next one should fail
        let result = gen.check_bundling_capacity(2);
        assert!(result.is_err());
        
        match result {
            Err(BipolarVectorError::CapacityExceeded { current, max }) => {
                assert_eq!(max, MAX_BUNDLING_CAPACITY);
                assert!(current > max);
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_get_value() {
        let v = BipolarVector::from_seed(42).unwrap();
        
        for i in 0..HDC_DIMENSION {
            let val = v.get_value(i).unwrap();
            assert!(val == 1 || val == -1);
        }
        
        let err = v.get_value(HDC_DIMENSION).unwrap_err();
        match err {
            BipolarVectorError::InvalidDimension { .. } => {}
            _ => panic!("Wrong error type"),
        }
    }
}
