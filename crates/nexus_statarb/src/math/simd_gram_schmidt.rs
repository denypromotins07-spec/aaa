//! SIMD-accelerated Gram-Schmidt Orthogonalization
//! 
//! Ensures extracted factors remain strictly orthogonal
//! using AVX2 vectorization.

use core::arch::x86_64::*;

/// Maximum vector size for SIMD operations
const MAX_VECTOR_SIZE: usize = 256;

/// Perform Gram-Schmidt orthogonalization on a set of vectors
/// 
/// # Arguments
/// * `vectors` - Mutable slice of vectors to orthogonalize (each vector is &[f64])
/// * `n_vectors` - Number of vectors to process
/// * `vector_len` - Length of each vector
/// 
/// Returns true if successful, false if dimensions are invalid
#[inline]
pub fn gram_schmidt(vectors: &mut [&mut [f64]], n_vectors: usize, vector_len: usize) -> bool {
    if n_vectors == 0 || vector_len == 0 {
        return false;
    }

    // Use SIMD for vectors longer than 4 elements
    if vector_len >= 4 {
        unsafe {
            simd_gram_schmidt(vectors, n_vectors, vector_len)
        }
    } else {
        scalar_gram_schmidt(vectors, n_vectors, vector_len)
    }
}

/// Scalar fallback for small vectors
#[inline]
fn scalar_gram_schmidt(vectors: &mut [&mut [f64]], n_vectors: usize, vector_len: usize) -> bool {
    for i in 0..n_vectors {
        // Orthogonalize against all previous vectors
        for j in 0..i {
            // Compute dot product: proj = v_i · v_j
            let mut dot = 0.0;
            for k in 0..vector_len {
                dot += vectors[i][k] * vectors[j][k];
            }

            // Subtract projection: v_i = v_i - proj * v_j
            for k in 0..vector_len {
                vectors[i][k] -= dot * vectors[j][k];
            }
        }

        // Normalize v_i
        let mut norm_sq = 0.0;
        for k in 0..vector_len {
            norm_sq += vectors[i][k] * vectors[i][k];
        }

        let norm = norm_sq.sqrt();
        if norm > 1e-15 {
            for k in 0..vector_len {
                vectors[i][k] /= norm;
            }
        } else {
            // Vector collapsed to zero - numerical instability
            return false;
        }
    }

    true
}

/// SIMD-accelerated Gram-Schmidt using AVX2
/// 
/// # Safety
/// Caller must ensure vectors have valid memory layout
#[inline]
pub unsafe fn simd_gram_schmidt(
    vectors: &mut [&mut [f64]],
    n_vectors: usize,
    vector_len: usize,
) -> bool {
    let simd_width = 4; // AVX2 processes 4 doubles at once
    let simd_limit = (vector_len / simd_width) * simd_width;

    for i in 0..n_vectors {
        // Orthogonalize against all previous vectors
        for j in 0..i {
            // Compute dot product using SIMD
            let mut dot_acc = _mm256_setzero_pd();

            for k in (0..simd_limit).step_by(simd_width) {
                let v_i = _mm256_loadu_pd(vectors[i].as_ptr().add(k));
                let v_j = _mm256_loadu_pd(vectors[j].as_ptr().add(k));

                let prod = _mm256_mul_pd(v_i, v_j);
                dot_acc = _mm256_add_pd(dot_acc, prod);
            }

            // Horizontal sum of accumulator
            let mut dot_vals = [0.0f64; 4];
            _mm256_storeu_pd(dot_vals.as_mut_ptr(), dot_acc);
            let mut dot = dot_vals[0] + dot_vals[1] + dot_vals[2] + dot_vals[3];

            // Process remaining elements
            for k in simd_limit..vector_len {
                dot += vectors[i][k] * vectors[j][k];
            }

            // Subtract projection using SIMD
            for k in (0..simd_limit).step_by(simd_width) {
                let v_i = _mm256_loadu_pd(vectors[i].as_ptr().add(k));
                let v_j = _mm256_loadu_pd(vectors[j].as_ptr().add(k));
                let scale = _mm256_set1_pd(dot);

                let proj = _mm256_mul_pd(scale, v_j);
                let result = _mm256_sub_pd(v_i, proj);

                _mm256_storeu_pd(vectors[i].as_mut_ptr().add(k), result);
            }

            // Process remaining elements
            for k in simd_limit..vector_len {
                vectors[i][k] -= dot * vectors[j][k];
            }
        }

        // Normalize using SIMD
        let mut norm_sq_acc = _mm256_setzero_pd();

        for k in (0..simd_limit).step_by(simd_width) {
            let v_i = _mm256_loadu_pd(vectors[i].as_ptr().add(k));
            let sq = _mm256_mul_pd(v_i, v_i);
            norm_sq_acc = _mm256_add_pd(norm_sq_acc, sq);
        }

        let mut norm_vals = [0.0f64; 4];
        _mm256_storeu_pd(norm_vals.as_mut_ptr(), norm_sq_acc);
        let mut norm_sq = norm_vals[0] + norm_vals[1] + norm_vals[2] + norm_vals[3];

        // Process remaining elements
        for k in simd_limit..vector_len {
            norm_sq += vectors[i][k] * vectors[i][k];
        }

        let norm = norm_sq.sqrt();
        if norm < 1e-15 {
            return false; // Numerical instability
        }

        let inv_norm = 1.0 / norm;
        let inv_norm_vec = _mm256_set1_pd(inv_norm);

        for k in (0..simd_limit).step_by(simd_width) {
            let v_i = _mm256_loadu_pd(vectors[i].as_ptr().add(k));
            let normalized = _mm256_mul_pd(v_i, inv_norm_vec);
            _mm256_storeu_pd(vectors[i].as_mut_ptr().add(k), normalized);
        }

        for k in simd_limit..vector_len {
            vectors[i][k] *= inv_norm;
        }
    }

    true
}

/// Re-orthogonalize a single vector against a basis
/// 
/// Uses modified Gram-Schmidt for better numerical stability
#[inline]
pub fn reorthogonalize_vector(
    target: &mut [f64],
    basis: &[&[f64]],
) -> bool {
    let n = target.len();
    
    if n == 0 {
        return false;
    }

    // Modified Gram-Schmidt: project out each basis vector
    for b in basis.iter() {
        if b.len() != n {
            return false;
        }

        // Compute dot product
        let mut dot = 0.0;
        for i in 0..n {
            dot += target[i] * b[i];
        }

        // Subtract projection
        for i in 0..n {
            target[i] -= dot * b[i];
        }
    }

    // Normalize
    let mut norm_sq = 0.0;
    for i in 0..n {
        norm_sq += target[i] * target[i];
    }

    let norm = norm_sq.sqrt();
    if norm < 1e-15 {
        return false;
    }

    for i in 0..n {
        target[i] /= norm;
    }

    true
}

/// Check orthogonality of a set of vectors
/// 
/// Returns the maximum absolute dot product between any pair
#[inline]
pub fn check_orthogonality(vectors: &[&[f64]], threshold: f64) -> bool {
    let n = vectors.len();
    if n < 2 {
        return true;
    }

    let len = vectors[0].len();
    
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dot = 0.0;
            for k in 0..len {
                dot += vectors[i][k] * vectors[j][k];
            }
            
            if dot.abs() > threshold {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_gram_schmidt() {
        let mut v1 = vec![1.0, 0.0, 0.0];
        let mut v2 = vec![1.0, 1.0, 0.0];
        let mut v3 = vec![1.0, 1.0, 1.0];

        let mut vectors: Vec<&mut [f64]> = vec![&mut v1, &mut v2, &mut v3];
        
        let result = gram_schmidt(&mut vectors, 3, 3);
        assert!(result);

        // Check orthogonality
        let mut dot12 = 0.0;
        let mut dot13 = 0.0;
        let mut dot23 = 0.0;
        for i in 0..3 {
            dot12 += v1[i] * v2[i];
            dot13 += v1[i] * v3[i];
            dot23 += v2[i] * v3[i];
        }

        assert!(dot12.abs() < 1e-10);
        assert!(dot13.abs() < 1e-10);
        assert!(dot23.abs() < 1e-10);
    }

    #[test]
    fn test_simd_gram_schmidt() {
        // Create larger vectors to trigger SIMD path
        let n = 16;
        let mut vectors_data: Vec<Vec<f64>> = (0..4)
            .map(|i| (0..n).map(|j| ((i + j) as f64) * 0.1).collect())
            .collect();
        
        let mut vectors: Vec<&mut [f64]> = vectors_data.iter_mut().map(|v| v.as_mut_slice()).collect();
        
        let result = gram_schmidt(&mut vectors, 4, n);
        assert!(result);

        // Verify orthogonality
        for i in 0..4 {
            for j in (i + 1)..4 {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += vectors[i][k] * vectors[j][k];
                }
                assert!(dot.abs() < 1e-10, "Vectors {} and {} not orthogonal", i, j);
            }
        }
    }

    #[test]
    fn test_reorthogonalize() {
        let basis1: Vec<f64> = vec![1.0, 0.0, 0.0];
        let basis2: Vec<f64> = vec![0.0, 1.0, 0.0];
        let mut target = vec![1.0, 1.0, 1.0];

        let basis: Vec<&[f64]> = vec![&basis1, &basis2];
        
        let result = reorthogonalize_vector(&mut target, &basis);
        assert!(result);

        // Target should now be [0, 0, 1] (orthogonal to both basis vectors)
        assert!((target[0]).abs() < 1e-10);
        assert!((target[1]).abs() < 1e-10);
        assert!((target[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_check_orthogonality() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        let v3 = vec![0.0, 0.0, 1.0];

        let vectors: Vec<&[f64]> = vec![&v1, &v2, &v3];
        
        assert!(check_orthogonality(&vectors, 1e-10));

        // Add non-orthogonal vector
        let v4 = vec![1.0, 1.0, 0.0];
        let vectors_bad: Vec<&[f64]> = vec![&v1, &v2, &v4];
        
        assert!(!check_orthogonality(&vectors_bad, 1e-10));
    }

    #[test]
    fn test_empty_input() {
        let mut data: Vec<f64> = vec![];
        let mut vectors: Vec<&mut [f64]> = vec![&mut data];
        
        assert!(!gram_schmidt(&mut vectors, 1, 0));
    }
}
