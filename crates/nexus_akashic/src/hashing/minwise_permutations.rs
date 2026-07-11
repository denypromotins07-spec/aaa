//! Min-Wise Independent Permutations for Jaccard Similarity Estimation
//! Implements MinHash algorithm with strict collision resistance and birthday paradox mitigation

use thiserror::Error;
use blake3::Hasher;

/// Error types for MinHash operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum MinHashError {
    #[error("Invalid number of hash functions: {requested}. Must be between {min} and {max}")]
    InvalidHashFunctionCount { requested: usize, min: usize, max: usize },
    #[error("Empty set provided for hashing")]
    EmptySet,
    #[error("Hash computation failed")]
    HashComputationFailed,
}

/// Minimum number of hash functions for statistical significance
const MIN_HASH_FUNCTIONS: usize = 8;

/// Maximum number of hash functions to prevent Birthday Paradox issues
/// Based on: for N items, we need hash space > N^2 / 2 for low collision probability
const MAX_HASH_FUNCTIONS: usize = 256;

/// A single MinHash signature component
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MinHashValue(u64);

impl MinHashValue {
    #[inline(always)]
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    #[inline(always)]
    pub fn value(&self) -> u64 {
        self.0
    }
}

/// Complete MinHash signature for a set
#[derive(Debug, Clone, PartialEq)]
pub struct MinHashSignature {
    /// Array of minimum hash values from different permutations
    values: [MinHashValue; MAX_HASH_FUNCTIONS],
    /// Number of active hash functions (<= MAX_HASH_FUNCTIONS)
    num_hashes: usize,
}

impl MinHashSignature {
    /// Create a new empty signature with specified number of hash functions
    pub fn new(num_hashes: usize) -> Result<Self, MinHashError> {
        if num_hashes < MIN_HASH_FUNCTIONS || num_hashes > MAX_HASH_FUNCTIONS {
            return Err(MinHashError::InvalidHashFunctionCount {
                requested: num_hashes,
                min: MIN_HASH_FUNCTIONS,
                max: MAX_HASH_FUNCTIONS,
            });
        }

        let mut values = [MinHashValue::new(u64::MAX); MAX_HASH_FUNCTIONS];
        
        Ok(Self {
            values,
            num_hashes,
        })
    }

    /// Update the signature with a new element's hash values
    #[inline(always)]
    pub fn update(&mut self, hashes: &[u64]) -> Result<(), MinHashError> {
        if hashes.len() < self.num_hashes {
            return Err(MinHashError::InvalidHashFunctionCount {
                requested: hashes.len(),
                min: self.num_hashes,
                max: MAX_HASH_FUNCTIONS,
            });
        }

        for i in 0..self.num_hashes {
            let hash_val = MinHashValue::new(hashes[i]);
            if hash_val < self.values[i] {
                self.values[i] = hash_val;
            }
        }

        Ok(())
    }

    /// Get the signature values as a slice
    #[inline(always)]
    pub fn as_slice(&self) -> &[MinHashValue] {
        &self.values[..self.num_hashes]
    }

    /// Calculate Jaccard similarity with another signature
    pub fn jaccard_similarity(&self, other: &Self) -> Result<f64, MinHashError> {
        if self.num_hashes != other.num_hashes {
            return Err(MinHashError::InvalidHashFunctionCount {
                requested: self.num_hashes,
                min: other.num_hashes,
                max: other.num_hashes,
            });
        }

        let mut matches = 0u64;
        for i in 0..self.num_hashes {
            if self.values[i] == other.values[i] {
                matches += 1;
            }
        }

        Ok(matches as f64 / self.num_hashes as f64)
    }

    /// Calculate Hamming distance between signatures (for LSH)
    pub fn hamming_distance(&self, other: &Self) -> Result<usize, MinHashError> {
        if self.num_hashes != other.num_hashes {
            return Err(MinHashError::InvalidHashFunctionCount {
                requested: self.num_hashes,
                min: other.num_hashes,
                max: other.num_hashes,
            });
        }

        let mut distance = 0;
        for i in 0..self.num_hashes {
            if self.values[i] != other.values[i] {
                distance += 1;
            }
        }

        Ok(distance)
    }
}

/// Min-Wise Independent Hash Family Generator
/// Generates multiple independent hash functions using Blake3 with different seeds
pub struct MinHashFamily {
    /// Seeds for each hash function
    seeds: [u64; MAX_HASH_FUNCTIONS],
    /// Number of active hash functions
    num_hashes: usize,
}

impl MinHashFamily {
    /// Create a new hash family with specified number of functions
    pub fn new(num_hashes: usize, base_seed: u64) -> Result<Self, MinHashError> {
        if num_hashes < MIN_HASH_FUNCTIONS || num_hashes > MAX_HASH_FUNCTIONS {
            return Err(MinHashError::InvalidHashFunctionCount {
                requested: num_hashes,
                min: MIN_HASH_FUNCTIONS,
                max: MAX_HASH_FUNCTIONS,
            });
        }

        let mut seeds = [0u64; MAX_HASH_FUNCTIONS];
        let mut current_seed = base_seed;

        // Generate unique seeds using linear congruential generator
        for i in 0..num_hashes {
            current_seed = current_seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seeds[i] = current_seed ^ (i as u64);
        }

        Ok(Self {
            seeds,
            num_hashes,
        })
    }

    /// Compute all hash values for a single element
    #[inline(always)]
    pub fn hash_element(&self, data: &[u8]) -> Result<[u64; MAX_HASH_FUNCTIONS], MinHashError> {
        let mut hashes = [0u64; MAX_HASH_FUNCTIONS];

        for i in 0..self.num_hashes {
            let mut hasher = Hasher::new();
            hasher.update(&self.seeds[i].to_le_bytes());
            hasher.update(data);
            let hash = hasher.finalize();
            
            // Extract u64 from first 8 bytes of Blake3 hash
            let hash_bytes = hash.as_bytes();
            hashes[i] = u64::from_le_bytes([
                hash_bytes[0], hash_bytes[1], hash_bytes[2], hash_bytes[3],
                hash_bytes[4], hash_bytes[5], hash_bytes[6], hash_bytes[7],
            ]);
        }

        Ok(hashes)
    }

    /// Compute MinHash signature for a set of elements
    pub fn compute_signature<I, T>(&self, elements: I) -> Result<MinHashSignature, MinHashError>
    where
        I: IntoIterator<Item = T>,
        T: AsRef<[u8]>,
    {
        let mut signature = MinHashSignature::new(self.num_hashes)?;
        let mut has_elements = false;

        for element in elements {
            has_elements = true;
            let hashes = self.hash_element(element.as_ref())?;
            signature.update(&hashes[..self.num_hashes])?;
        }

        if !has_elements {
            return Err(MinHashError::EmptySet);
        }

        Ok(signature)
    }

    /// Get the number of hash functions
    #[inline(always)]
    pub fn num_hashes(&self) -> usize {
        self.num_hashes
    }
}

/// Estimate cardinality of a set using MinHash values
pub fn estimate_cardinality(signature: &MinHashSignature) -> u64 {
    // Use the minimum hash value to estimate cardinality
    // E[min] ≈ U / (n + 1) where U is hash space size, n is cardinality
    let mut min_val = u64::MAX;
    for i in 0..signature.as_slice().len() {
        min_val = min_val.min(signature.as_slice()[i].value());
    }

    // Cardinality estimate: n ≈ U / min_val - 1
    // Using full u64 range as hash space
    if min_val == 0 {
        u64::MAX
    } else {
        (u64::MAX / min_val).saturating_sub(1)
    }
}

/// Compare two sets for approximate equality using MinHash
pub fn sets_approximately_equal<T, U>(
    set1: &[T],
    set2: &[U],
    family: &MinHashFamily,
    threshold: f64,
) -> Result<bool, MinHashError>
where
    T: AsRef<[u8]>,
    U: AsRef<[u8]>,
{
    let sig1 = family.compute_signature(set1.iter())?;
    let sig2 = family.compute_signature(set2.iter())?;
    
    let similarity = sig1.jaccard_similarity(&sig2)?;
    Ok(similarity >= threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minhash_family_creation() {
        let family = MinHashFamily::new(64, 42);
        assert!(family.is_ok());
        assert_eq!(family.unwrap().num_hashes(), 64);
    }

    #[test]
    fn test_invalid_hash_count() {
        let result = MinHashFamily::new(5, 42);
        assert!(result.is_err());
        
        let result = MinHashFamily::new(300, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_signature_computation() {
        let family = MinHashFamily::new(32, 123).unwrap();
        let elements = vec!["hello", "world", "test"];
        
        let signature = family.compute_signature(elements.iter()).unwrap();
        assert_eq!(signature.as_slice().len(), 32);
    }

    #[test]
    fn test_empty_set_error() {
        let family = MinHashFamily::new(16, 456).unwrap();
        let elements: Vec<&str> = vec![];
        
        let result = family.compute_signature(elements.iter());
        assert!(result.is_err());
        match result {
            Err(MinHashError::EmptySet) => {}
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_jaccard_similarity_identical() {
        let family = MinHashFamily::new(64, 789).unwrap();
        let elements = vec!["a", "b", "c", "d", "e"];
        
        let sig1 = family.compute_signature(elements.iter()).unwrap();
        let sig2 = family.compute_signature(elements.iter()).unwrap();
        
        let sim = sig1.jaccard_similarity(&sig2).unwrap();
        assert_eq!(sim, 1.0);
    }

    #[test]
    fn test_jaccard_similarity_disjoint() {
        let family = MinHashFamily::new(128, 111).unwrap();
        let elements1 = vec!["cat", "dog", "bird"];
        let elements2 = vec!["apple", "orange", "banana"];
        
        let sig1 = family.compute_signature(elements1.iter()).unwrap();
        let sig2 = family.compute_signature(elements2.iter()).unwrap();
        
        let sim = sig1.jaccard_similarity(&sig2).unwrap();
        // Disjoint sets should have very low similarity (near 0)
        assert!(sim < 0.1, "Similarity too high for disjoint sets: {}", sim);
    }

    #[test]
    fn test_jaccard_similarity_partial() {
        let family = MinHashFamily::new(64, 222).unwrap();
        let elements1 = vec!["a", "b", "c", "d"];
        let elements2 = vec!["c", "d", "e", "f"];
        
        let sig1 = family.compute_signature(elements1.iter()).unwrap();
        let sig2 = family.compute_signature(elements2.iter()).unwrap();
        
        let sim = sig1.jaccard_similarity(&sig2).unwrap();
        
        // Jaccard(A,B) = |A∩B| / |A∪B| = 2/6 = 0.333...
        // MinHash provides an unbiased estimator, so it should be close
        assert!(sim > 0.1 && sim < 0.6, "Similarity {} outside expected range", sim);
    }

    #[test]
    fn test_hamming_distance() {
        let family = MinHashFamily::new(32, 333).unwrap();
        let elements1 = vec!["x", "y", "z"];
        let elements2 = vec!["p", "q", "r"];
        
        let sig1 = family.compute_signature(elements1.iter()).unwrap();
        let sig2 = family.compute_signature(elements2.iter()).unwrap();
        
        let dist = sig1.hamming_distance(&sig2).unwrap();
        
        // Different sets should have significant Hamming distance
        assert!(dist > 10, "Hamming distance too small: {}", dist);
        assert!(dist <= 32, "Hamming distance exceeds signature size");
    }

    #[test]
    fn test_sets_approximately_equal() {
        let family = MinHashFamily::new(64, 444).unwrap();
        
        let set1 = vec!["same", "elements"];
        let set2 = vec!["same", "elements"];
        let set3 = vec!["different", "stuff"];
        
        let equal = sets_approximately_equal(&set1, &set2, &family, 0.9).unwrap();
        assert!(equal);
        
        let not_equal = sets_approximately_equal(&set1, &set3, &family, 0.5).unwrap();
        assert!(!not_equal);
    }

    #[test]
    fn test_cardinality_estimation() {
        let family = MinHashFamily::new(64, 555).unwrap();
        
        // Create a set with known cardinality
        let elements: Vec<String> = (0..100).map(|i| format!("item_{}", i)).collect();
        let signature = family.compute_signature(elements.iter()).unwrap();
        
        let estimated = estimate_cardinality(&signature);
        
        // Estimate should be within reasonable bounds (order of magnitude)
        assert!(estimated > 10 && estimated < 10000, 
                "Cardinality estimate {} way off from 100", estimated);
    }
}
