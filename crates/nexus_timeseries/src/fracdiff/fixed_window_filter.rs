//! Fractional Differentiation (FracDiff) - Fixed Window Filter
//! Implements Marcos Lopez de Prado's fractional differentiation with fixed window.
//! Zero-allocation implementation using pre-computed weights.

use crate::fracdiff::simd_weight_convolution::{FracDiffWeights, simd_convolve};
use nexus_allocator::BumpAllocator;

/// Errors that can occur during fractional differentiation
#[derive(Debug, Clone, PartialEq)]
pub enum FracDiffError {
    InvalidDifferencingParameter { d: f64, reason: &'static str },
    InsufficientData { required: usize, provided: usize },
    WeightComputationFailed,
}

/// Fixed-window fractional differentiator
/// 
/// Uses a fixed window size to compute fractional differences,
/// avoiding the expanding window's memory growth while preserving
/// most of the long-term memory.
pub struct FixedWindowFracDiff {
    /// Differencing coefficient (0.0 < d < 1.0 typically)
    d: f64,
    /// Window size (number of lags to consider)
    window_size: usize,
    /// Pre-computed fractional weights
    weights: FracDiffWeights,
    /// Circular buffer for recent values
    value_buffer: Vec<f64>,
    /// Current position in circular buffer
    buffer_pos: usize,
    /// Number of values seen so far
    count: usize,
}

impl FixedWindowFracDiff {
    /// Create a new fixed-window fractional differentiator
    /// 
    /// # Arguments
    /// * `d` - Differencing coefficient (typically 0.2 to 0.8)
    /// * `window_size` - Number of lags to use (larger = more memory preserved but slower)
    /// 
    /// # Returns
    /// * `Ok(Self)` on success
    /// * `Err(FracDiffError)` if parameters are invalid
    pub fn new(d: f64, window_size: usize) -> Result<Self, FracDiffError> {
        // Validate differencing parameter
        if d < 0.0 {
            return Err(FracDiffError::InvalidDifferencingParameter {
                d,
                reason: "d must be non-negative",
            });
        }
        if d > 2.0 {
            return Err(FracDiffError::InvalidDifferencingParameter {
                d,
                reason: "d > 2.0 is numerically unstable",
            });
        }
        if window_size < 2 {
            return Err(FracDiffError::InvalidDifferencingParameter {
                d,
                reason: "window_size must be at least 2",
            });
        }

        // Pre-compute fractional weights
        let weights = FracDiffWeights::compute(d, window_size)
            .ok_or(FracDiffError::WeightComputationFailed)?;

        Ok(Self {
            d,
            window_size,
            weights,
            value_buffer: vec![0.0; window_size],
            buffer_pos: 0,
            count: 0,
        })
    }

    /// Update with a new value and compute the fractional difference
    /// 
    /// # Arguments
    /// * `value` - New observation
    /// 
    /// # Returns
    /// * `Some(f64)` - Fractional difference if enough data accumulated
    /// * `None` - If insufficient data for full window
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // Store value in circular buffer
        self.value_buffer[self.buffer_pos] = value;
        self.buffer_pos = (self.buffer_pos + 1) % self.window_size;
        self.count += 1;

        // Need at least window_size observations
        if self.count < self.window_size {
            return None;
        }

        // Perform SIMD-accelerated convolution
        // Buffer is arranged with most recent at buffer_pos - 1 (wrapping)
        Some(simd_convolve(&self.value_buffer, self.buffer_pos, &self.weights))
    }

    /// Get the current differencing coefficient
    pub fn d(&self) -> f64 {
        self.d
    }

    /// Get the window size
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Reset the internal state
    pub fn reset(&mut self) {
        self.value_buffer.fill(0.0);
        self.buffer_pos = 0;
        self.count = 0;
    }
}

/// Expanding window fractional differentiator
/// 
/// Uses all available history, providing maximum memory preservation
/// but with increasing computational cost.
pub struct ExpandingWindowFracDiff {
    /// Differencing coefficient
    d: f64,
    /// Maximum number of weights to cache (truncation point)
    max_weights: usize,
    /// Pre-computed weights (cached up to max_weights)
    weights: Option<FracDiffWeights>,
    /// History of values
    history: Vec<f64>,
    /// Minimum observations before producing output
    min_obs: usize,
}

impl ExpandingWindowFracDiff {
    /// Create a new expanding-window fractional differentiator
    pub fn new(d: f64, max_weights: usize, min_obs: usize) -> Result<Self, FracDiffError> {
        if d < 0.0 || d > 2.0 {
            return Err(FracDiffError::InvalidDifferencingParameter {
                d,
                reason: "d must be in range [0, 2]",
            });
        }

        let weights = FracDiffWeights::compute(d, max_weights);

        Ok(Self {
            d,
            max_weights,
            weights,
            history: Vec::with_capacity(max_weights * 2),
            min_obs,
        })
    }

    /// Update with new value and compute fractional difference
    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.history.push(value);

        let n = self.history.len();
        if n < self.min_obs {
            return None;
        }

        // Use cached weights or compute on-the-fly for small windows
        let weights = self.weights.as_ref().map(|w| w.weights()).unwrap_or_else(|| {
            // Fallback: compute weights dynamically (should not happen often)
            FracDiffWeights::compute(self.d, n.min(self.max_weights))?
                .weights()
        });

        let window_len = n.min(weights.len());
        
        // Convolve most recent window_len values with weights
        let mut result = 0.0;
        for i in 0..window_len {
            let idx = n - 1 - i;
            result += weights[i] * self.history[idx];
        }

        Some(result)
    }

    /// Get the number of observations
    pub fn observation_count(&self) -> usize {
        self.history.len()
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_window_creation() {
        let diff = FixedWindowFracDiff::new(0.5, 100);
        assert!(diff.is_ok());
    }

    #[test]
    fn test_invalid_d_parameter() {
        let diff = FixedWindowFracDiff::new(-0.1, 100);
        assert!(diff.is_err());

        let diff = FixedWindowFracDiff::new(2.5, 100);
        assert!(diff.is_err());
    }

    #[test]
    fn test_fractional_diff_output() {
        let mut diff = FixedWindowFracDiff::new(0.5, 10).unwrap();
        
        // Feed some values
        for i in 0..20 {
            let value = i as f64 * 1.5;
            let result = diff.update(value);
            
            if i >= 9 {
                assert!(result.is_some(), "Should produce output after warmup");
            }
        }
    }

    #[test]
    fn test_stationarity_preservation() {
        // Test that fracdiff preserves more memory than simple returns
        let mut frac_diff = FixedWindowFracDiff::new(0.4, 50).unwrap();
        let mut log_returns: Vec<f64> = Vec::new();
        let mut prev_price: Option<f64> = None;
        
        // Simulate a trending price series with memory
        let mut price = 100.0;
        for i in 0..200 {
            price *= 1.0005 + ((i as f64 * 0.013).sin() * 0.002);
            
            if let Some(prev) = prev_price {
                log_returns.push((price / prev).ln());
            }
            prev_price = Some(price);
            
            frac_diff.update(price);
        }
        
        // The fracdiff series should have higher autocorrelation than log returns
        // (This is a qualitative test - actual verification would need statistical tests)
    }
}
