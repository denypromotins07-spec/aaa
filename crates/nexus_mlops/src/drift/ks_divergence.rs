//! Kolmogorov-Smirnov Divergence for Streaming Distribution Comparison
//!
//! Implements zero-allocation K-S divergence calculation using pre-allocated
//! streaming histograms. Compares live feature distributions against baselines.

use crate::MLOpsError;
use std::cmp::{max, min};

/// Streaming histogram with fixed bins for zero-allocation operation
#[derive(Clone)]
pub struct StreamingHistogram {
    /// Pre-allocated bin counts
    bins: Vec<u64>,
    /// Total samples seen
    count: u64,
    /// Minimum value for bin scaling
    min_val: f64,
    /// Maximum value for bin scaling
    max_val: f64,
    /// Whether bounds are fixed or adaptive
    fixed_bounds: bool,
}

impl StreamingHistogram {
    /// Create new streaming histogram with specified number of bins
    pub fn new(n_bins: usize) -> Self {
        Self {
            bins: vec![0; n_bins],
            count: 0,
            min_val: f64::INFINITY,
            max_val: f64::NEG_INFINITY,
            fixed_bounds: false,
        }
    }

    /// Create with fixed bounds (no adaptation)
    pub fn with_fixed_bounds(n_bins: usize, min_val: f64, max_val: f64) -> Self {
        Self {
            bins: vec![0; n_bins],
            count: 0,
            min_val,
            max_val,
            fixed_bounds: true,
        }
    }

    /// Update histogram with new value
    pub fn update(&mut self, value: f64) -> Result<(), MLOpsError> {
        if !self.fixed_bounds {
            // Adaptively expand bounds
            self.min_val = min(self.min_val, value);
            self.max_val = max(self.max_val, value);
            
            // Prevent zero range
            if (self.max_val - self.min_val).abs() < 1e-10 {
                self.min_val -= 0.5;
                self.max_val += 0.5;
            }
        }

        // Find bin index
        let bin_idx = self.value_to_bin(value);
        
        if bin_idx < self.bins.len() {
            self.bins[bin_idx] += 1;
            self.count += 1;
        }

        Ok(())
    }

    /// Convert value to bin index
    fn value_to_bin(&self, value: f64) -> usize {
        let range = self.max_val - self.min_val;
        if range.abs() < 1e-10 {
            return 0;
        }
        
        let normalized = (value - self.min_val) / range;
        let bin = (normalized * self.bins.len() as f64) as usize;
        
        // Clamp to valid range
        min(bin, self.bins.len().saturating_sub(1))
    }

    /// Get cumulative distribution function at given value
    pub fn cdf_at(&self, value: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        let bin_idx = self.value_to_bin(value);
        let mut cumsum = 0u64;
        
        for i in 0..=min(bin_idx, self.bins.len().saturating_sub(1)) {
            cumsum += self.bins[i];
        }

        cumsum as f64 / self.count as f64
    }

    /// Get total count
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset histogram
    pub fn reset(&mut self) {
        for count in &mut self.bins {
            *count = 0;
        }
        self.count = 0;
        if !self.fixed_bounds {
            self.min_val = f64::INFINITY;
            self.max_val = f64::NEG_INFINITY;
        }
    }

    /// Get bin counts slice
    pub fn bins(&self) -> &[u64] {
        &self.bins
    }
}

/// Kolmogorov-Smirnov divergence calculator
pub struct KSDivergence {
    /// Reference histogram (baseline distribution)
    reference: StreamingHistogram,
    /// Current/live histogram
    current: StreamingHistogram,
    /// Number of bins
    n_bins: usize,
}

impl KSDivergence {
    /// Create new K-S divergence calculator
    pub fn new(n_bins: usize) -> Self {
        Self {
            reference: StreamingHistogram::new(n_bins),
            current: StreamingHistogram::new(n_bins),
            n_bins,
        }
    }

    /// Create with fixed bounds
    pub fn with_fixed_bounds(n_bins: usize, min_val: f64, max_val: f64) -> Self {
        Self {
            reference: StreamingHistogram::with_fixed_bounds(n_bins, min_val, max_val),
            current: StreamingHistogram::with_fixed_bounds(n_bins, min_val, max_val),
            n_bins,
        }
    }

    /// Set reference distribution from baseline data
    pub fn set_reference(&mut self, baseline_data: &[f64]) -> Result<(), MLOpsError> {
        self.reference.reset();
        for &value in baseline_data {
            self.reference.update(value)?;
        }
        Ok(())
    }

    /// Update current distribution with new sample
    pub fn update(&mut self, value: f64) -> Result<(), MLOpsError> {
        self.current.update(value)
    }

    /// Compute K-S statistic (maximum CDF difference)
    pub fn compute_statistic(&self) -> Result<f64, MLOpsError> {
        if self.reference.count() == 0 || self.current.count() == 0 {
            return Ok(0.0);
        }

        let mut max_diff = 0.0;

        // Check at each bin boundary
        for i in 0..self.n_bins {
            let bin_edge = self.bin_index_to_value(i);
            
            let cdf_ref = self.reference.cdf_at(bin_edge);
            let cdf_cur = self.current.cdf_at(bin_edge);
            
            let diff = (cdf_ref - cdf_cur).abs();
            max_diff = max(max_diff, diff);
        }

        Ok(max_diff)
    }

    /// Convert bin index back to approximate value
    fn bin_index_to_value(&self, bin_idx: usize) -> f64 {
        let range = self.current.max_val - self.current.min_val;
        self.current.min_val + (bin_idx as f64 / self.n_bins as f64) * range
    }

    /// Compute p-value approximation for K-S statistic
    /// Uses asymptotic distribution for large samples
    pub fn compute_pvalue(&self) -> Result<f64, MLOpsError> {
        let statistic = self.compute_statistic()?;
        
        let n1 = self.reference.count() as f64;
        let n2 = self.current.count() as f64;
        
        if n1 < 1.0 || n2 < 1.0 {
            return Ok(1.0); // Can't reject with no data
        }

        // Effective sample size
        let n_eff = (n1 * n2) / (n1 + n2);
        
        // K-S asymptotic p-value approximation
        let lambda = statistic * ((n1 + n2) as f64).sqrt();
        
        // Approximate p-value using Kolmogorov distribution
        // P(K > lambda) ≈ 2 * exp(-2 * lambda^2) for large lambda
        let p_value = 2.0 * (-2.0 * lambda * lambda).exp();
        
        Ok(p_value.clamp(0.0, 1.0))
    }

    /// Check if distributions differ significantly at given alpha level
    pub fn is_significant(&self, alpha: f64) -> Result<bool, MLOpsError> {
        let p_value = self.compute_pvalue()?;
        Ok(p_value < alpha)
    }

    /// Get reference histogram
    pub fn reference(&self) -> &StreamingHistogram {
        &self.reference
    }

    /// Get current histogram
    pub fn current(&self) -> &StreamingHistogram {
        &self.current
    }

    /// Reset current histogram only (keep reference)
    pub fn reset_current(&mut self) {
        self.current.reset();
    }

    /// Reset both histograms
    pub fn reset_all(&mut self) {
        self.reference.reset();
        self.current.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ks_same_distribution() {
        let mut ks = KSDivergence::new(50);
        
        // Same distribution for both
        let baseline: Vec<f64> = (0..1000).map(|i| (i % 100) as f64 / 100.0).collect();
        ks.set_reference(&baseline).unwrap();
        
        for i in 0..1000 {
            let value = (i % 100) as f64 / 100.0;
            ks.update(value).unwrap();
        }

        let statistic = ks.compute_statistic().unwrap();
        assert!(statistic < 0.1, "Same distribution should have low K-S statistic");
    }

    #[test]
    fn test_ks_different_distributions() {
        let mut ks = KSDivergence::with_fixed_bounds(50, 0.0, 1.0);
        
        // Baseline: uniform [0, 0.5]
        let baseline: Vec<f64> = (0..1000).map(|i| (i % 50) as f64 / 100.0).collect();
        ks.set_reference(&baseline).unwrap();
        
        // Current: uniform [0.5, 1.0]
        for i in 0..1000 {
            let value = 0.5 + (i % 50) as f64 / 100.0;
            ks.update(value).unwrap();
        }

        let statistic = ks.compute_statistic().unwrap();
        assert!(statistic > 0.5, "Different distributions should have high K-S statistic");
        
        let is_sig = ks.is_significant(0.05).unwrap();
        assert!(is_sig, "Should detect significant difference");
    }

    #[test]
    fn test_streaming_histogram_adaptive() {
        let mut hist = StreamingHistogram::new(20);
        
        for i in 0..100 {
            hist.update(i as f64).unwrap();
        }

        assert!(hist.min_val < 50.0);
        assert!(hist.max_val > 50.0);
        assert_eq!(hist.count(), 100);
    }
}
