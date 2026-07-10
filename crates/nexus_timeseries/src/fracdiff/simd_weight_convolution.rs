//! SIMD-Accelerated Weight Convolution for Fractional Differentiation
//! Pre-computes and caches fractional weights using bump allocation.
//! Provides zero-allocation convolution operations.

use std::arch::x86_64::*;

/// Pre-computed fractional differentiation weights
/// Stored in a cache-aligned structure for SIMD operations
#[derive(Debug, Clone)]
pub struct FracDiffWeights {
    /// The differencing coefficient d used to generate these weights
    d: f64,
    /// Number of weights (window size)
    len: usize,
    /// Padded length for SIMD (multiple of 4 for AVX)
    padded_len: usize,
    /// Weight values (padded with zeros for SIMD alignment)
    weights: Vec<f64>,
}

impl FracDiffWeights {
    /// Compute fractional differentiation weights for given d and window size
    /// 
    /// Uses the binomial series expansion:
    /// w_k = (-1)^k * Gamma(d+1) / (Gamma(k+1) * Gamma(d-k+1))
    /// 
    /// # Arguments
    /// * `d` - Differencing coefficient
    /// * `window_size` - Number of weights to compute
    /// 
    /// # Returns
    /// * `Some(Self)` on success
    /// * `None` if computation fails (e.g., numerical overflow)
    pub fn compute(d: f64, window_size: usize) -> Option<Self> {
        if window_size == 0 {
            return None;
        }

        let mut weights = Vec::with_capacity(window_size);
        
        // First weight is always 1.0
        weights.push(1.0);

        // Compute remaining weights using recurrence relation:
        // w_k = w_{k-1} * (k - 1 - d) / k
        for k in 1..window_size {
            let prev = *weights.last()?;
            let wk = prev * (k as f64 - 1.0 - d) / k as f64;
            
            // Truncate weights that are numerically insignificant
            // This saves computation without introducing discontinuity
            if wk.abs() < 1e-16 {
                // Pad remaining weights with zeros and break
                while weights.len() < window_size {
                    weights.push(0.0);
                }
                break;
            }
            
            weights.push(wk);
        }

        // Pad to multiple of 4 for AVX SIMD operations
        let padded_len = ((weights.len() + 3) / 4) * 4;
        while weights.len() < padded_len {
            weights.push(0.0);
        }

        Some(Self {
            d,
            len: window_size,
            padded_len,
            weights,
        })
    }

    /// Get the raw weight slice (unpadded)
    pub fn weights(&self) -> &[f64] {
        &self.weights[..self.len]
    }

    /// Get the padded weight slice (for SIMD)
    pub fn weights_padded(&self) -> &[f64] {
        &self.weights[..self.padded_len]
    }

    /// Get the differencing coefficient
    pub fn d(&self) -> f64 {
        self.d
    }

    /// Get the actual number of non-zero weights
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if weights are empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// SIMD-accelerated convolution of values with pre-computed weights
/// 
/// Uses AVX2 instructions to process 4 doubles per cycle.
/// 
/// # Arguments
/// * `values` - Circular buffer of values (most recent at wrap_pos - 1)
/// * `wrap_pos` - Position where next value will be written (current head)
/// * `weights` - Pre-computed fractional weights
/// 
/// # Returns
/// Convolution result (fractional difference)
#[inline(always)]
pub fn simd_convolve(values: &[f64], wrap_pos: usize, weights: &FracDiffWeights) -> f64 {
    let n = weights.len();
    if n == 0 || values.is_empty() {
        return 0.0;
    }

    let weights_slice = weights.weights_padded();
    
    // Ensure we have enough values
    let available = values.len().min(n);
    if available == 0 {
        return 0.0;
    }

    // Use AVX2 for blocks of 4
    let mut result = 0.0;
    let mut i = 0;

    unsafe {
        let mut acc = _mm256_setzero_pd();
        
        // Process 4 elements at a time
        while i + 3 < available {
            // Load 4 weights
            let w = _mm256_loadu_pd(weights_slice.as_ptr().add(i));
            
            // Load 4 values from circular buffer
            // Values are arranged: [wrap_pos, wrap_pos+1, ..., end, 0, 1, ...]
            let mut v_vals = [0.0; 4];
            for j in 0..4 {
                let idx = (wrap_pos + j) % values.len();
                v_vals[j] = values[idx];
            }
            let v = _mm256_loadu_pd(v_vals.as_ptr());
            
            // Multiply and accumulate
            let prod = _mm256_mul_pd(w, v);
            acc = _mm256_add_pd(acc, prod);
            
            i += 4;
        }

        // Horizontal sum of accumulator
        let mut sum_arr = [0.0; 4];
        _mm256_storeu_pd(sum_arr.as_mut_ptr(), acc);
        result = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];

        // Handle remaining elements
        while i < available {
            let w = weights_slice[i];
            let v_idx = (wrap_pos + i) % values.len();
            let v = values[v_idx];
            result += w * v;
            i += 1;
        }
    }

    // Apply sign flip for fractional difference (convention)
    // The weights alternate in sign, so we may need to negate
    if n > 0 && weights_slice[0] < 0.0 {
        -result
    } else {
        result
    }
}

/// Scalar fallback convolution (for testing or when SIMD unavailable)
#[inline(always)]
pub fn scalar_convolve(values: &[f64], wrap_pos: usize, weights: &FracDiffWeights) -> f64 {
    let n = weights.len().min(values.len());
    if n == 0 {
        return 0.0;
    }

    let mut result = 0.0;
    for i in 0..n {
        let w = weights.weights()[i];
        let v_idx = (wrap_pos + i) % values.len();
        let v = values[v_idx];
        result += w * v;
    }

    result
}

/// Pre-compute weights for multiple d values and cache them
/// Useful for adaptive d selection algorithms
pub struct WeightCache {
    caches: Vec<FracDiffWeights>,
    window_size: usize,
}

impl WeightCache {
    /// Create a cache of weights for different d values
    pub fn new(window_size: usize, d_values: &[f64]) -> Self {
        let mut caches = Vec::with_capacity(d_values.len());
        
        for &d in d_values {
            if let Some(weights) = FracDiffWeights::compute(d, window_size) {
                caches.push(weights);
            }
        }

        Self {
            caches,
            window_size,
        }
    }

    /// Find the best cached weights for a target d
    pub fn get_closest(&self, target_d: f64) -> Option<&FracDiffWeights> {
        self.caches.iter().min_by(|a, b| {
            (a.d() - target_d).abs().partial_cmp(&(b.d() - target_d).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get all cached weights
    pub fn all_weights(&self) -> &[FracDiffWeights] {
        &self.caches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_computation() {
        let weights = FracDiffWeights::compute(0.5, 10);
        assert!(weights.is_some());
        
        let w = weights.unwrap();
        assert_eq!(w.len(), 10);
        assert!((w.weights()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weight_truncation() {
        // For small d, weights should decay quickly
        let weights = FracDiffWeights::compute(0.1, 1000);
        assert!(weights.is_some());
        
        let w = weights.unwrap();
        // Many weights should be truncated to zero
        let non_zero = w.weights().iter().filter(|&&x| x.abs() > 1e-16).count();
        assert!(non_zero < 100); // Should truncate significantly
    }

    #[test]
    fn test_simd_vs_scalar() {
        let weights = FracDiffWeights::compute(0.4, 16).unwrap();
        let values: Vec<f64> = (0..16).map(|i| i as f64 * 0.5).collect();
        
        let simd_result = simd_convolve(&values, 0, &weights);
        let scalar_result = scalar_convolve(&values, 0, &weights);
        
        assert!((simd_result - scalar_result).abs() < 1e-10);
    }

    #[test]
    fn test_circular_buffer_convolution() {
        let weights = FracDiffWeights::compute(0.3, 8).unwrap();
        let values: Vec<f64> = (0..8).map(|i| i as f64).collect();
        
        // Test with different wrap positions
        for wrap_pos in 0..8 {
            let result = simd_convolve(&values, wrap_pos, &weights);
            assert!(result.is_finite());
        }
    }

    #[test]
    fn test_weight_cache() {
        let d_values = vec![0.2, 0.4, 0.6, 0.8];
        let cache = WeightCache::new(50, &d_values);
        
        assert_eq!(cache.all_weights().len(), 4);
        
        let closest = cache.get_closest(0.41);
        assert!(closest.is_some());
        assert!((closest.unwrap().d() - 0.4).abs() < 0.01);
    }
}
