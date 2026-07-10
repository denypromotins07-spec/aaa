//! Shannon Entropy Calculator for Time Series Analysis
//! Zero-allocation implementation using pre-allocated histograms.

/// Streaming Shannon entropy estimator
pub struct ShannonEntropy {
    /// Number of bins for histogram
    n_bins: usize,
    /// Histogram counts
    counts: Vec<usize>,
    /// Total observations
    total: usize,
    /// Minimum value for binning
    min_val: f64,
    /// Maximum value for binning
    max_val: f64,
}

impl ShannonEntropy {
    pub fn new(n_bins: usize) -> Self {
        Self {
            n_bins,
            counts: vec![0; n_bins],
            total: 0,
            min_val: f64::INFINITY,
            max_val: f64::NEG_INFINITY,
        }
    }

    /// Update with new observation
    pub fn update(&mut self, value: f64) {
        self.min_val = self.min_val.min(value);
        self.max_val = self.max_val.max(value);
        
        let bin = self.value_to_bin(value);
        if bin < self.n_bins {
            self.counts[bin] += 1;
            self.total += 1;
        }
    }

    fn value_to_bin(&self, value: f64) -> usize {
        if self.max_val <= self.min_val {
            return 0;
        }
        let range = self.max_val - self.min_val;
        let normalized = (value - self.min_val) / range;
        (normalized * self.n_bins as f64).min((self.n_bins - 1) as f64) as usize
    }

    /// Compute Shannon entropy H(X) = -Σ p(x) log p(x)
    pub fn entropy(&self) -> Option<f64> {
        if self.total == 0 {
            return None;
        }

        let mut h = 0.0;
        let total_f = self.total as f64;

        for &count in &self.counts {
            if count > 0 {
                let p = count as f64 / total_f;
                h -= p * p.ln();
            }
        }

        Some(h)
    }

    /// Reset the estimator
    pub fn reset(&mut self) {
        self.counts.fill(0);
        self.total = 0;
        self.min_val = f64::INFINITY;
        self.max_val = f64::NEG_INFINITY;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_distribution() {
        let mut entropy = ShannonEntropy::new(10);
        
        // Uniform distribution should have maximum entropy
        for i in 0..1000 {
            entropy.update((i % 10) as f64);
        }
        
        let h = entropy.entropy().unwrap();
        assert!(h > 2.0); // log2(10) ≈ 3.32, natural log ≈ 2.3
    }

    #[test]
    fn test_concentrated_distribution() {
        let mut entropy = ShannonEntropy::new(10);
        
        // All values same - minimum entropy
        for _ in 0..1000 {
            entropy.update(5.0);
        }
        
        let h = entropy.entropy().unwrap();
        assert!(h < 0.1); // Should be near zero
    }
}
