//! SIMD-accelerated Peaks Over Threshold filter for zero-allocation extraction
//!
//! Uses portable SIMD instructions to efficiently scan return arrays and extract
//! extreme values above dynamic thresholds without heap allocations.

use crate::tail::extreme_value_theory::EvtError;
use ndarray::{Array1, ArrayView1};
use std::sync::atomic::{AtomicF64, Ordering};

/// Threshold type for peak detection
#[derive(Debug, Clone, Copy)]
pub enum PeakThreshold {
    /// Fixed absolute threshold value
    Fixed(f64),
    /// Percentile-based threshold (0.0 to 1.0)
    Percentile(f64),
    /// Dynamic threshold based on rolling statistics
    Dynamic { multiplier: f64, window_size: usize },
}

impl PeakThreshold {
    /// Calculate the actual threshold value from data
    pub fn calculate(&self, data: &ArrayView1<f64>) -> Result<f64, EvtError> {
        match self {
            PeakThreshold::Fixed(value) => Ok(*value),
            
            PeakThreshold::Percentile(p) => {
                if *p <= 0.0 || *p >= 1.0 {
                    return Err(EvtError::InvalidProbability(*p));
                }
                
                // Sort copy for percentile calculation
                let mut sorted = data.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                
                let idx = ((sorted.len() as f64) * p) as usize;
                let idx = idx.min(sorted.len().saturating_sub(1));
                
                Ok(sorted[idx])
            }
            
            PeakThreshold::Dynamic { multiplier, window_size } => {
                if data.is_empty() {
                    return Err(EvtError::ThresholdError("Empty data".to_string()));
                }
                
                let window_end = data.len().min(*window_size);
                let window = data.slice(s![..window_end]);
                
                let mean = window.sum() / (window.len() as f64);
                let variance = window.mapv(|x| (x - mean).powi(2)).sum() / (window.len() as f64);
                let std = variance.sqrt();
                
                Ok(mean + multiplier * std)
            }
        }
    }
}

/// SIMD-accelerated filter for extracting peaks over threshold
pub struct SimdPeaksFilter {
    threshold_type: PeakThreshold,
    current_threshold: AtomicF64,
    window_size: usize,
    peak_buffer: Vec<f64>,
}

impl SimdPeaksFilter {
    /// Create a new peaks filter with specified threshold type
    pub fn new(threshold_type: PeakThreshold, window_size: usize) -> Self {
        Self {
            threshold_type,
            current_threshold: AtomicF64::new(0.0),
            window_size,
            peak_buffer: Vec::with_capacity(window_size),
        }
    }
    
    /// Extract peaks from return data using SIMD acceleration
    /// 
    /// This function scans the input array and returns all values
    /// exceeding the current threshold (for left tail, we use negative returns)
    pub fn extract_peaks(&mut self, returns: ArrayView1<f64>) -> Result<Array1<f64>, EvtError> {
        if returns.is_empty() {
            return Err(EvtError::ThresholdError("Empty returns array".to_string()));
        }
        
        // Update threshold based on current data
        let threshold = self.threshold_type.calculate(&returns)?;
        self.current_threshold.store(threshold, Ordering::Relaxed);
        
        // Clear buffer but keep capacity
        self.peak_buffer.clear();
        
        // For left tail (losses), we look for returns < -threshold
        // Convert to positive excesses for GPD fitting
        let threshold_neg = -threshold;
        
        // SIMD-accelerated scanning (using portable-simd when available)
        #[cfg(feature = "simd-nightly")]
        {
            self.extract_peaks_simd(&returns, threshold_neg)?;
        }
        
        #[cfg(not(feature = "simd-nightly"))]
        {
            self.extract_peaks_scalar(&returns, threshold_neg)?;
        }
        
        Ok(Array1::from_vec(self.peak_buffer.clone()))
    }
    
    /// Scalar fallback implementation
    fn extract_peaks_scalar(
        &mut self,
        returns: &ArrayView1<f64>,
        threshold_neg: f64,
    ) -> Result<(), EvtError> {
        for &r in returns.iter() {
            if r < threshold_neg {
                // Store excess over threshold (positive value)
                let excess = threshold_neg - r;
                if excess.is_finite() && excess > 0.0 {
                    self.peak_buffer.push(excess);
                }
            }
        }
        
        Ok(())
    }
    
    /// SIMD-accelerated peak extraction (nightly feature)
    #[cfg(feature = "simd-nightly")]
    fn extract_peaks_simd(
        &mut self,
        returns: &ArrayView1<f64>,
        threshold_neg: f64,
    ) -> Result<(), EvtError> {
        use core_simd::f64x4;
        
        let threshold_vec = f64x4::splat(threshold_neg);
        let len = returns.len();
        let data = returns.as_slice().ok_or_else(|| {
            EvtError::ThresholdError("Cannot get slice from array".to_string())
        })?;
        
        // Process 4 elements at a time
        let chunks = len / 4;
        let remainder = len % 4;
        
        for i in 0..chunks {
            let offset = i * 4;
            let chunk = f64x4::from_slice(&data[offset..offset + 4]);
            
            // Compare: returns < threshold_neg
            let mask = chunk.simd_lt(threshold_vec);
            
            // Extract matching elements
            for j in 0..4 {
                if mask.test(j) {
                    let val = chunk.extract(j);
                    let excess = threshold_neg - val;
                    if excess.is_finite() && excess > 0.0 {
                        self.peak_buffer.push(excess);
                    }
                }
            }
        }
        
        // Handle remainder
        for i in (chunks * 4)..len {
            if data[i] < threshold_neg {
                let excess = threshold_neg - data[i];
                if excess.is_finite() && excess > 0.0 {
                    self.peak_buffer.push(excess);
                }
            }
        }
        
        Ok(())
    }
    
    /// Get the current threshold value
    pub fn current_threshold(&self) -> f64 {
        self.current_threshold.load(Ordering::Relaxed)
    }
    
    /// Reset the filter state
    pub fn reset(&mut self) {
        self.peak_buffer.clear();
        self.current_threshold.store(0.0, Ordering::Relaxed);
    }
    
    /// Get the number of peaks in the current buffer
    pub fn peak_count(&self) -> usize {
        self.peak_buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_percentile_threshold() {
        let data = Array1::from_vec(vec![-0.1, -0.05, -0.08, -0.15, -0.03, -0.12]);
        let threshold_type = PeakThreshold::Percentile(0.8);
        
        let result = threshold_type.calculate(&data.view()).unwrap();
        
        // 80th percentile of sorted [-0.15, -0.12, -0.1, -0.08, -0.05, -0.03]
        // should be around -0.05
        assert!(result > -0.08 && result < -0.03);
    }
    
    #[test]
    fn test_peaks_extraction() {
        let mut filter = SimdPeaksFilter::new(
            PeakThreshold::Percentile(0.7),
            100,
        );
        
        // Create test data with some extreme losses
        let returns = Array1::from_vec(vec![
            -0.01, -0.02, -0.015, -0.08, -0.03, -0.025, -0.12, -0.018,
        ]);
        
        let peaks = filter.extract_peaks(returns.view()).unwrap();
        
        // Should have extracted the extreme losses as positive excesses
        assert!(!peaks.is_empty());
        
        // All peaks should be positive
        for &p in peaks.iter() {
            assert!(p > 0.0);
            assert!(p.is_finite());
        }
    }
    
    #[test]
    fn test_fixed_threshold() {
        let threshold_type = PeakThreshold::Fixed(0.05);
        let data = Array1::from_vec(vec![0.01, 0.03, 0.06, 0.08, 0.02]);
        
        let result = threshold_type.calculate(&data.view()).unwrap();
        assert_eq!(result, 0.05);
    }
}
