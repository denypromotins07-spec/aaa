//! SIMD-accelerated Pheromone Decay Operator.
//! 
//! Implements vectorized pheromone evaporation using AVX2/AVX-512 instructions
//! for high-throughput decay operations on large pheromone matrices.

use std::arch::x86_64::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// SIMD lane width for AVX2 (256-bit / 64-bit per f64 = 4 lanes)
pub const SIMD_LANE_WIDTH_AVX2: usize = 4;

/// SIMD lane width for AVX-512 (512-bit / 64-bit per f64 = 8 lanes)
#[cfg(target_feature = "avx512f")]
pub const SIMD_LANE_WIDTH_AVX512: usize = 8;

/// Padded buffer for SIMD alignment
#[derive(Debug, Clone)]
pub struct SimdPheromoneBuffer {
    data: Vec<f64>,
    padded_len: usize,
    original_len: usize,
}

impl SimdPheromoneBuffer {
    /// Create a new SIMD-aligned buffer from pheromone data
    pub fn new(pheromones: &[f64]) -> Self {
        let original_len = pheromones.len();
        let padded_len = Self::pad_to_simd_width(original_len);
        
        let mut data = Vec::with_capacity(padded_len);
        data.extend_from_slice(pheromones);
        
        // Pad with minimum pheromone value to avoid affecting calculations
        while data.len() < padded_len {
            data.push(0.001); // MIN_PHEROMONE equivalent
        }

        Self {
            data,
            padded_len,
            original_len,
        }
    }

    /// Calculate padded length for SIMD alignment
    fn pad_to_simd_width(len: usize) -> usize {
        let remainder = len % SIMD_LANE_WIDTH_AVX2;
        if remainder == 0 {
            len
        } else {
            len + (SIMD_LANE_WIDTH_AVX2 - remainder)
        }
    }

    /// Get reference to underlying data
    pub fn as_slice(&self) -> &[f64] {
        &self.data[..self.original_len]
    }

    /// Get mutable reference to underlying data
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data[..self.original_len]
    }

    /// Get full padded slice (including padding)
    pub fn as_padded_slice(&self) -> &[f64] {
        &self.data
    }

    /// Get mutable padded slice
    pub fn as_mut_padded_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Check if buffer is properly aligned for SIMD
    pub fn is_simd_aligned(&self) -> bool {
        self.padded_len % SIMD_LANE_WIDTH_AVX2 == 0
    }
}

/// SIMD-accelerated decay operator for pheromone evaporation
pub struct SimdDecayOperator {
    decay_factor: f64,
    min_bound: f64,
    max_bound: f64,
    processed_count: AtomicU64,
}

impl SimdDecayOperator {
    pub fn new(decay_factor: f64, min_bound: f64, max_bound: f64) -> Self {
        Self {
            decay_factor: decay_factor.clamp(0.0, 1.0),
            min_bound,
            max_bound,
            processed_count: AtomicU64::new(0),
        }
    }

    /// Apply decay using AVX2 SIMD instructions
    /// 
    /// Safety: Requires CPU with AVX2 support
    pub unsafe fn apply_avx2(&mut self, buffer: &mut SimdPheromoneBuffer) {
        if !buffer.is_simd_aligned() {
            // Fallback to scalar if not aligned
            self.apply_scalar(buffer.as_mut_slice());
            return;
        }

        let decay_vec = _mm256_set1_pd(self.decay_factor);
        let min_vec = _mm256_set1_pd(self.min_bound);
        let max_vec = _mm256_set1_pd(self.max_bound);

        let data = buffer.as_mut_padded_slice();
        let len = data.len();
        let simd_len = len - (len % SIMD_LANE_WIDTH_AVX2);

        // Process full SIMD lanes
        for i in (0..simd_len).step_by(SIMD_LANE_WIDTH_AVX2) {
            let ptr = data.as_mut_ptr().add(i);
            
            // Load 4 f64 values
            let mut v = _mm256_loadu_pd(ptr);
            
            // Multiply by decay factor
            v = _mm256_mul_pd(v, decay_vec);
            
            // Clamp to [min, max]
            v = _mm256_max_pd(v, min_vec);
            v = _mm256_min_pd(v, max_vec);
            
            // Store back
            _mm256_storeu_pd(ptr, v);
        }

        self.processed_count.fetch_add(simd_len as u64, Ordering::Relaxed);
    }

    /// Apply decay using AVX-512 SIMD instructions (when available)
    #[cfg(target_feature = "avx512f")]
    pub unsafe fn apply_avx512(&mut self, buffer: &mut SimdPheromoneBuffer) {
        use std::arch::x86_64::*;

        let decay_vec = _mm512_set1_pd(self.decay_factor);
        let min_vec = _mm512_set1_pd(self.min_bound);
        let max_vec = _mm512_set1_pd(self.max_bound);

        let data = buffer.as_mut_padded_slice();
        let len = data.len();
        let simd_len = len - (len % SIMD_LANE_WIDTH_AVX512);

        for i in (0..simd_len).step_by(SIMD_LANE_WIDTH_AVX512) {
            let ptr = data.as_mut_ptr().add(i);
            
            let mut v = _mm512_loadu_pd(ptr);
            v = _mm512_mul_pd(v, decay_vec);
            v = _mm512_max_pd(v, min_vec);
            v = _mm512_min_pd(v, max_vec);
            _mm512_storeu_pd(ptr, v);
        }

        self.processed_count.fetch_add(simd_len as u64, Ordering::Relaxed);
    }

    /// Scalar fallback implementation
    pub fn apply_scalar(&mut self, pheromones: &mut [f64]) {
        for p in pheromones.iter_mut() {
            *p = (*p * self.decay_factor).clamp(self.min_bound, self.max_bound);
        }
        self.processed_count.fetch_add(pheromones.len() as u64, Ordering::Relaxed);
    }

    /// Auto-select best SIMD implementation based on CPU features
    pub fn apply_auto(&mut self, buffer: &mut SimdPheromoneBuffer) {
        #[cfg(target_feature = "avx512f")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe {
                    self.apply_avx512(buffer);
                }
                return;
            }
        }

        if is_x86_feature_detected!("avx2") {
            unsafe {
                self.apply_avx2(buffer);
            }
        } else {
            self.apply_scalar(buffer.as_mut_slice());
        }
    }

    /// Update decay factor
    pub fn set_decay_factor(&mut self, factor: f64) {
        self.decay_factor = factor.clamp(0.0, 1.0);
    }

    /// Get current decay factor
    pub fn decay_factor(&self) -> f64 {
        self.decay_factor
    }

    /// Get total processed count
    pub fn processed_count(&self) -> u64 {
        self.processed_count.load(Ordering::Relaxed)
    }

    /// Reset processed count
    pub fn reset_count(&mut self) {
        self.processed_count.store(0, Ordering::Relaxed);
    }
}

/// Parallel pheromone decay executor using rayon
pub struct ParallelDecayExecutor {
    operators: Vec<SimdDecayOperator>,
}

impl ParallelDecayExecutor {
    pub fn new(num_threads: usize, decay_factor: f64, min_bound: f64, max_bound: f64) -> Self {
        let operators = (0..num_threads)
            .map(|_| SimdDecayOperator::new(decay_factor, min_bound, max_bound))
            .collect();

        Self { operators }
    }

    /// Execute decay in parallel across multiple buffers
    pub fn execute_parallel(&mut self, buffers: &mut [SimdPheromoneBuffer]) {
        use rayon::prelude::*;

        let num_threads = self.operators.len();
        let chunk_size = (buffers.len() + num_threads - 1) / num_threads;

        buffers
            .par_chunks_mut(chunk_size.max(1))
            .enumerate()
            .for_each(|(idx, chunk)| {
                if idx < self.operators.len() {
                    for buffer in chunk {
                        self.operators[idx].apply_auto(buffer);
                    }
                }
            });
    }

    /// Get total processed elements across all operators
    pub fn total_processed(&self) -> u64 {
        self.operators.iter().map(|op| op.processed_count()).sum()
    }

    /// Reset all operator counts
    pub fn reset_all_counts(&mut self) {
        for op in &mut self.operators {
            op.reset_count();
        }
    }
}

/// Batch decay statistics for monitoring
#[derive(Debug, Clone, Copy, Default)]
pub struct DecayStatistics {
    pub elements_processed: u64,
    pub mean_before: f64,
    pub mean_after: f64,
    pub min_after: f64,
    pub max_after: f64,
    pub nanoseconds_elapsed: u64,
}

impl DecayStatistics {
    pub fn calculate(before: &[f64], after: &[f64], elapsed_ns: u64) -> Self {
        let mean_before = if before.is_empty() {
            0.0
        } else {
            before.iter().sum::<f64>() / before.len() as f64
        };

        let mean_after = if after.is_empty() {
            0.0
        } else {
            after.iter().sum::<f64>() / after.len() as f64
        };

        let (min_after, max_after) = if after.is_empty() {
            (0.0, 0.0)
        } else {
            let mut min = f64::MAX;
            let mut max = f64::MIN;
            for &v in after {
                min = min.min(v);
                max = max.max(v);
            }
            (min, max)
        };

        Self {
            elements_processed: after.len() as u64,
            mean_before,
            mean_after,
            min_after,
            max_after,
            nanoseconds_elapsed: elapsed_ns,
        }
    }

    /// Get throughput in elements per second
    pub fn throughput_eps(&self) -> f64 {
        if self.nanoseconds_elapsed == 0 {
            return 0.0;
        }
        self.elements_processed as f64 * 1e9 / self.nanoseconds_elapsed as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_buffer_creation() {
        let pheromones = vec![0.5, 0.6, 0.7];
        let buffer = SimdPheromoneBuffer::new(&pheromones);

        assert_eq!(buffer.original_len, 3);
        assert!(buffer.padded_len >= 3);
        assert!(buffer.padded_len % SIMD_LANE_WIDTH_AVX2 == 0);
        assert!(buffer.is_simd_aligned());
    }

    #[test]
    fn test_simd_buffer_data_integrity() {
        let pheromones = vec![0.5, 0.6, 0.7, 0.8, 0.9];
        let buffer = SimdPheromoneBuffer::new(&pheromones);

        assert_eq!(buffer.as_slice(), &pheromones);
    }

    #[test]
    fn test_decay_operator_scalar() {
        let mut op = SimdDecayOperator::new(0.5, 0.001, 1.0);
        let mut data = vec![0.8, 0.6, 0.4, 0.2];

        op.apply_scalar(&mut data);

        assert!((data[0] - 0.4).abs() < 1e-10);
        assert!((data[1] - 0.3).abs() < 1e-10);
        assert!((data[2] - 0.2).abs() < 1e-10);
        assert!((data[3] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_decay_operator_bounds() {
        let mut op = SimdDecayOperator::new(0.5, 0.1, 0.5);
        let mut data = vec![0.02, 0.8, 0.3];

        op.apply_scalar(&mut data);

        // Should be clamped to [0.1, 0.5]
        assert!(data[0] >= 0.1);
        assert!(data[1] <= 0.5);
        assert!(data[2] >= 0.1 && data[2] <= 0.5);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_decay_avx2() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }

        let mut op = SimdDecayOperator::new(0.5, 0.001, 1.0);
        let mut buffer = SimdPheromoneBuffer::new(&vec![0.8, 0.6, 0.4, 0.2]);

        unsafe {
            op.apply_avx2(&mut buffer);
        }

        let result = buffer.as_slice();
        assert!((result[0] - 0.4).abs() < 1e-10);
        assert!((result[1] - 0.3).abs() < 1e-10);
        assert!((result[2] - 0.2).abs() < 1e-10);
        assert!((result[3] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_decay_statistics() {
        let before = vec![0.8, 0.6, 0.4, 0.2];
        let after = vec![0.4, 0.3, 0.2, 0.1];

        let stats = DecayStatistics::calculate(&before, &after, 1000);

        assert_eq!(stats.elements_processed, 4);
        assert!((stats.mean_before - 0.5).abs() < 1e-10);
        assert!((stats.mean_after - 0.25).abs() < 1e-10);
        assert!((stats.min_after - 0.1).abs() < 1e-10);
        assert!((stats.max_after - 0.4).abs() < 1e-10);
    }
}
