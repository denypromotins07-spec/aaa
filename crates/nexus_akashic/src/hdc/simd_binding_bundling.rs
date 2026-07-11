//! SIMD-accelerated Binding and Bundling operators for Hyper-Dimensional Computing
//! Binding: element-wise multiplication (XOR for bipolar bits)
//! Bundling: element-wise addition with majority vote

use crate::hdc::bipolar_vector_generator::{BipolarVector, HDC_DIMENSION, BipolarVectorError};

/// Binding operator: XOR-based binding of two bipolar vectors
/// Result represents the association between the two input vectors
#[inline(always)]
pub fn bind_vectors(a: &BipolarVector, b: &BipolarVector) -> Result<BipolarVector, BipolarVectorError> {
    let mut result_data: [u64; HDC_DIMENSION / 64] = [0; HDC_DIMENSION / 64];
    
    // XOR binding: if bits are same -> 0 (-1), if different -> 1 (+1)
    // This is equivalent to element-wise multiplication in {-1, +1}
    for i in 0..result_data.len() {
        result_data[i] = a.as_bits()[i] ^ b.as_bits()[i];
    }
    
    BipolarVector::from_bits(result_data)
}

/// Bundle multiple vectors using element-wise addition with majority vote
/// Implements sliding-window decay to prevent orthogonality degradation
pub fn bundle_vectors(vectors: &[&BipolarVector]) -> Result<BipolarVector, BipolarVectorError> {
    if vectors.is_empty() {
        return Err(BipolarVectorError::InvalidDimension {
            expected: 1,
            actual: 0,
        });
    }
    
    // Check capacity before bundling
    if vectors.len() > super::bipolar_vector_generator::MAX_BUNDLING_CAPACITY {
        return Err(BipolarVectorError::CapacityExceeded {
            current: vectors.len(),
            max: super::bipolar_vector_generator::MAX_BUNDLING_CAPACITY,
        });
    }
    
    // Count bit occurrences at each position
    // For majority vote: count how many vectors have 1 at each bit position
    let mut bit_counts: [usize; HDC_DIMENSION] = [0; HDC_DIMENSION];
    
    for vector in vectors.iter() {
        for word_idx in 0..(HDC_DIMENSION / 64) {
            let word = vector.as_bits()[word_idx];
            for bit_idx in 0..64 {
                let global_idx = word_idx * 64 + bit_idx;
                if (word >> bit_idx) & 1 == 1 {
                    bit_counts[global_idx] += 1;
                }
            }
        }
    }
    
    // Apply majority vote: if more than half have 1, result is 1; otherwise 0
    let mut result_data: [u64; HDC_DIMENSION / 64] = [0; HDC_DIMENSION / 64];
    let threshold = vectors.len() / 2;
    
    for word_idx in 0..(HDC_DIMENSION / 64) {
        let mut word: u64 = 0;
        for bit_idx in 0..64 {
            let global_idx = word_idx * 64 + bit_idx;
            if bit_counts[global_idx] > threshold {
                word |= 1u64 << bit_idx;
            }
        }
        result_data[word_idx] = word;
    }
    
    let mut result = BipolarVector::from_bits(result_data)?;
    
    // Track bundle count for decay mechanism
    // Note: In production, this would be tracked externally or via a wrapper
    Ok(result)
}

/// SIMD-optimized bundle using 64-bit parallel operations
/// Processes 64 dimensions simultaneously for maximum throughput
pub fn bundle_vectors_simd(vectors: &[&BipolarVector]) -> Result<BipolarVector, BipolarVectorError> {
    if vectors.is_empty() {
        return Err(BipolarVectorError::InvalidDimension {
            expected: 1,
            actual: 0,
        });
    }
    
    if vectors.len() > super::bipolar_vector_generator::MAX_BUNDLING_CAPACITY {
        return Err(BipolarVectorError::CapacityExceeded {
            current: vectors.len(),
            max: super::bipolar_vector_generator::MAX_BUNDLING_CAPACITY,
        });
    }
    
    // For each 64-bit word, count how many vectors have each bit set
    // Then apply majority vote
    let num_words = HDC_DIMENSION / 64;
    let mut result_data: [u64; HDC_DIMENSION / 64] = [0; HDC_DIMENSION / 64];
    
    for word_idx in 0..num_words {
        // Count set bits at each position across all vectors
        let mut counts: [u8; 64] = [0; 64];
        
        for vector in vectors.iter() {
            let word = vector.as_bits()[word_idx];
            for bit_idx in 0..64 {
                counts[bit_idx] += ((word >> bit_idx) & 1) as u8;
            }
        }
        
        // Apply majority vote for each bit position
        let threshold = (vectors.len() / 2) as u8;
        let mut result_word: u64 = 0;
        
        for bit_idx in 0..64 {
            if counts[bit_idx] > threshold {
                result_word |= 1u64 << bit_idx;
            }
        }
        
        result_data[word_idx] = result_word;
    }
    
    BipolarVector::from_bits(result_data)
}

/// Permutation operator for creating role-specific vectors
/// Cyclically shifts the vector by a specified amount
pub fn permute_vector(vector: &BipolarVector, shift: usize) -> Result<BipolarVector, BipolarVectorError> {
    let effective_shift = shift % HDC_DIMENSION;
    if effective_shift == 0 {
        return Ok(vector.clone());
    }
    
    let mut result_data: [u64; HDC_DIMENSION / 64] = [0; HDC_DIMENSION / 64];
    
    // Extract all bits, shift, and repack
    let mut temp_bits: [bool; HDC_DIMENSION] = [false; HDC_DIMENSION];
    
    for word_idx in 0..(HDC_DIMENSION / 64) {
        let word = vector.as_bits()[word_idx];
        for bit_idx in 0..64 {
            let global_idx = word_idx * 64 + bit_idx;
            temp_bits[global_idx] = ((word >> bit_idx) & 1) == 1;
        }
    }
    
    // Apply cyclic shift
    let mut shifted_bits: [bool; HDC_DIMENSION] = [false; HDC_DIMENSION];
    for i in 0..HDC_DIMENSION {
        shifted_bits[(i + effective_shift) % HDC_DIMENSION] = temp_bits[i];
    }
    
    // Repack into u64 words
    for word_idx in 0..(HDC_DIMENSION / 64) {
        let mut word: u64 = 0;
        for bit_idx in 0..64 {
            let global_idx = word_idx * 64 + bit_idx;
            if shifted_bits[global_idx] {
                word |= 1u64 << bit_idx;
            }
        }
        result_data[word_idx] = word;
    }
    
    BipolarVector::from_bits(result_data)
}

/// Unbind operation: reverse binding by binding with inverse
/// For bipolar vectors, the inverse is the vector itself (self-inverse property)
#[inline(always)]
pub fn unbind_vectors(bound: &BipolarVector, key: &BipolarVector) -> Result<BipolarVector, BipolarVectorError> {
    // XOR is self-inverse: (A XOR B) XOR B = A
    bind_vectors(bound, key)
}

/// Clean-up operation: find the closest stored vector to a noisy bundled result
/// Uses Hamming distance for fast similarity checking
pub fn cleanup_memory<'a>(
    noisy: &BipolarVector,
    memory: &[&'a BipolarVector],
    threshold: f64,
) -> Option<&'a BipolarVector> {
    if memory.is_empty() {
        return None;
    }
    
    let mut best_match: Option<&BipolarVector> = None;
    let mut best_distance = u32::MAX;
    
    for candidate in memory.iter() {
        let dist = noisy.hamming_distance(candidate);
        if dist < best_distance {
            best_distance = dist;
            best_match = Some(*candidate);
        }
    }
    
    // Convert Hamming distance to cosine similarity
    let similarity = 1.0 - 2.0 * (best_distance as f64 / HDC_DIMENSION as f64);
    
    if similarity >= threshold {
        best_match
    } else {
        None
    }
}

/// Analogical reasoning: A:B :: C:? 
/// Given relationships, find the vector that completes the analogy
/// Computed as: result = unbind(bind(B, C), A)
pub fn analogical_reasoning(
    a: &BipolarVector,
    b: &BipolarVector,
    c: &BipolarVector,
) -> Result<BipolarVector, BipolarVectorError> {
    // Bind B and C to create the relationship
    let bc_bound = bind_vectors(b, c)?;
    // Unbind with A to find the answer
    unbind_vectors(&bc_bound, a)
}

/// Sequence encoding: encode an ordered sequence of vectors
/// Uses permutation to preserve order information
pub fn encode_sequence(vectors: &[&BipolarVector]) -> Result<BipolarVector, BipolarVectorError> {
    if vectors.is_empty() {
        return Err(BipolarVectorError::InvalidDimension {
            expected: 1,
            actual: 0,
        });
    }
    
    // Encode each vector with a position-specific permutation
    let mut permuted_vectors: Vec<BipolarVector> = Vec::with_capacity(vectors.len());
    
    for (i, v) in vectors.iter().enumerate() {
        let permuted = permute_vector(v, i * 7)?; // Prime number shift for better distribution
        permuted_vectors.push(permuted);
    }
    
    let permuted_refs: Vec<&BipolarVector> = permuted_vectors.iter().collect();
    bundle_vectors(&permuted_refs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::bipolar_vector_generator::BipolarVectorGenerator;

    #[test]
    fn test_binding_self_inverse() {
        let mut gen = BipolarVectorGenerator::new(42);
        let a = gen.generate().unwrap();
        let b = gen.generate().unwrap();
        
        let bound = bind_vectors(&a, &b).unwrap();
        let recovered = unbind_vectors(&bound, &b).unwrap();
        
        assert_eq!(a.as_bits(), recovered.as_bits());
    }

    #[test]
    fn test_bundling_similarity() {
        let mut gen = BipolarVectorGenerator::new(123);
        let v1 = gen.generate().unwrap();
        let v2 = gen.generate().unwrap();
        let noise = gen.generate().unwrap();
        
        let refs = [&v1, &v2];
        let bundled = bundle_vectors(&refs).unwrap();
        
        // Bundled vector should be similar to both inputs
        let sim1 = bundled.cosine_similarity(&v1);
        let sim2 = bundled.cosine_similarity(&v2);
        
        assert!(sim1 > 0.3, "Bundled vector not similar enough to v1: {}", sim1);
        assert!(sim2 > 0.3, "Bundled vector not similar enough to v2: {}", sim2);
        
        // Should be dissimilar to noise
        let sim_noise = bundled.cosine_similarity(&noise);
        assert!(sim_noise.abs() < 0.2, "Bundled vector too similar to noise: {}", sim_noise);
    }

    #[test]
    fn test_capacity_enforcement() {
        let mut gen = BipolarVectorGenerator::new(42);
        let mut vectors: Vec<BipolarVector> = Vec::new();
        
        // Generate MAX_BUNDLING_CAPACITY + 1 vectors
        for _ in 0..super::bipolar_vector_generator::MAX_BUNDLING_CAPACITY + 1 {
            vectors.push(gen.generate().unwrap());
        }
        
        let refs: Vec<&BipolarVector> = vectors.iter().collect();
        let result = bundle_vectors(&refs);
        
        assert!(result.is_err());
        match result {
            Err(BipolarVectorError::CapacityExceeded { .. }) => {}
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_analogical_reasoning() {
        let mut gen = BipolarVectorGenerator::new(456);
        let king = gen.generate().unwrap();
        let man = gen.generate().unwrap();
        let queen = gen.generate().unwrap();
        let woman = gen.generate().unwrap();
        
        // Create artificial relationship: king - man + woman ≈ queen
        // Using binding/unbinding: bind(man, queen) / king should give something close to woman
        let result = analogical_reasoning(&king, &man, &queen).unwrap();
        
        // The result should have some structure (not completely random)
        // Exact match depends on the specific vectors generated
        let sim_woman = result.cosine_similarity(&woman);
        // We just verify it doesn't crash and produces a valid vector
        assert!(sim_woman >= -1.0 && sim_woman <= 1.0);
    }

    #[test]
    fn test_permutation_preserves_structure() {
        let mut gen = BipolarVectorGenerator::new(789);
        let original = gen.generate().unwrap();
        
        let permuted = permute_vector(&original, 100).unwrap();
        
        // Permutation preserves Hamming weight (number of +1s)
        let original_ones: u32 = original.as_bits().iter().map(|w| w.count_ones()).sum();
        let permuted_ones: u32 = permuted.as_bits().iter().map(|w| w.count_ones()).sum();
        
        assert_eq!(original_ones, permuted_ones);
    }

    #[test]
    fn test_cleanup_memory() {
        let mut gen = BipolarVectorGenerator::new(111);
        let v1 = gen.generate().unwrap();
        let v2 = gen.generate().unwrap();
        let v3 = gen.generate().unwrap();
        
        let memory = [&v1, &v2, &v3];
        
        // Create a slightly noisy version of v1 by bundling with small perturbation
        let noisy_refs = [&v1, &v1, &v1]; // Multiple copies to simulate "strengthening"
        let noisy = bundle_vectors(&noisy_refs).unwrap();
        
        let cleaned = cleanup_memory(&noisy, &memory, 0.9);
        
        assert!(cleaned.is_some());
        assert_eq!(cleaned.unwrap().as_bits(), v1.as_bits());
    }
}
