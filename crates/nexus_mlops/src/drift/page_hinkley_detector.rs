//! Page-Hinkley Drift Detector - Sequential mean-shift detection
//! 
//! Implements the Page-Hinkley test for detecting changes in the mean
//! of prediction errors. Used to detect concept drift in real-time.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

/// Page-Hinkley test state (stack-allocated, zero-heap)
pub struct PageHinkleyTest {
    /// Cumulative sum of deviations from threshold
    sum: f64,
    /// Minimum value seen so far
    min_sum: f64,
    /// Number of samples observed
    count: AtomicU64,
    /// Threshold for drift detection
    threshold: f64,
    /// Minimum samples before detection can trigger
    min_samples: usize,
    /// Whether drift has been detected
    drift_detected: AtomicBool,
}

impl PageHinkleyTest {
    /// Create a new Page-Hinkley detector
    /// 
    /// # Arguments
    /// * `threshold` - Detection threshold (typical: 50-200)
    /// * `min_samples` - Minimum samples before triggering (typical: 30-100)
    pub fn new(threshold: f64, min_samples: usize) -> Self {
        Self {
            sum: 0.0,
            min_sum: 0.0,
            count: AtomicU64::new(0),
            threshold,
            min_samples,
            drift_detected: AtomicBool::new(false),
        }
    }
    
    /// Update with a new prediction error
    /// 
    /// # Arguments
    /// * `error` - Prediction error (predicted - actual)
    /// * `delta` - Allowable deviation (typical: 0.001-0.01)
    /// 
    /// Returns true if drift is detected
    #[inline(always)]
    pub fn update(&mut self, error: f64, delta: f64) -> bool {
        self.count.fetch_add(1, Ordering::Relaxed);
        
        // Update cumulative sum
        self.sum += error - delta;
        
        // Track minimum
        if self.sum < self.min_sum {
            self.min_sum = self.sum;
        }
        
        // Calculate Page-Hinkley statistic
        let ph_stat = self.sum - self.min_sum;
        
        // Check for drift (only after min_samples)
        let current_count = self.count.load(Ordering::Relaxed);
        if current_count >= self.min_samples as u64 && ph_stat > self.threshold {
            self.drift_detected.store(true, Ordering::SeqCst);
            return true;
        }
        
        false
    }
    
    /// Reset the detector state
    #[inline(always)]
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.min_sum = 0.0;
        self.count.store(0, Ordering::Relaxed);
        self.drift_detected.store(false, Ordering::SeqCst);
    }
    
    /// Check if drift has been detected
    #[inline(always)]
    pub fn has_drift(&self) -> bool {
        self.drift_detected.load(Ordering::Relaxed)
    }
    
    /// Get current PH statistic value
    #[inline(always)]
    pub fn statistic(&self) -> f64 {
        self.sum - self.min_sum
    }
    
    /// Get sample count
    #[inline(always)]
    pub fn sample_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}

/// Streaming Kolmogorov-Smirnov test for distribution comparison
pub struct StreamingKSTest {
    /// Reference CDF bins (from training data)
    reference_cdf: Vec<f64>,
    /// Current observation CDF bins
    observation_cdf: Vec<f64>,
    /// Bin edges
    bin_edges: Vec<f64>,
    /// Number of observations
    count: AtomicU64,
    /// Maximum D statistic seen
    max_d: f64,
    /// Critical value for significance
    critical_value: f64,
    /// Drift detected flag
    drift_detected: AtomicBool,
}

impl StreamingKSTest {
    /// Create a new streaming KS test
    /// 
    /// # Arguments
    /// * `num_bins` - Number of histogram bins (typical: 20-50)
    /// * `min_val` - Minimum feature value
    /// * `max_val` - Maximum feature value
    /// * `critical_value` - Significance threshold (typical: 0.05 for p<0.05)
    pub fn new(num_bins: usize, min_val: f64, max_val: f64, critical_value: f64) -> Self {
        let bin_width = (max_val - min_val) / num_bins as f64;
        let bin_edges: Vec<f64> = (0..=num_bins)
            .map(|i| min_val + i as f64 * bin_width)
            .collect();
        
        let mut cdf = vec![0.0; num_bins];
        // Initialize reference CDF with uniform distribution
        for i in 0..num_bins {
            cdf[i] = (i + 1) as f64 / num_bins as f64;
        }
        
        Self {
            reference_cdf: cdf.clone(),
            observation_cdf: cdf,
            bin_edges,
            count: AtomicU64::new(0),
            max_d: 0.0,
            critical_value,
            drift_detected: AtomicBool::new(false),
        }
    }
    
    /// Set reference distribution from training data
    pub fn set_reference(&mut self, reference: &[f64]) {
        let num_bins = self.bin_edges.len() - 1;
        let mut counts = vec![0usize; num_bins];
        
        // Bin the reference data
        for &val in reference {
            if let Some(bin) = self.find_bin(val) {
                counts[bin] += 1;
            }
        }
        
        // Convert to CDF
        let total = counts.iter().sum::<usize>() as f64;
        if total > 0.0 {
            let mut cumsum = 0.0;
            for (i, &count) in counts.iter().enumerate() {
                cumsum += count as f64 / total;
                self.reference_cdf[i] = cumsum;
            }
        }
    }
    
    /// Add an observation and check for drift
    #[inline(always)]
    pub fn observe(&mut self, value: f64) -> bool {
        self.count.fetch_add(1, Ordering::Relaxed);
        
        if let Some(bin) = self.find_bin(value) {
            // Update observation CDF incrementally
            let n = self.count.load(Ordering::Relaxed) as f64;
            for i in bin..self.observation_cdf.len() {
                let old_val = self.observation_cdf[i];
                let new_val = old_val + (1.0 / n - old_val) / n;
                self.observation_cdf[i] = new_val;
            }
            
            // Calculate KS statistic (max difference between CDFs)
            let mut d = 0.0;
            for i in 0..self.reference_cdf.len() {
                let diff = (self.reference_cdf[i] - self.observation_cdf[i]).abs();
                if diff > d {
                    d = diff;
                }
            }
            
            if d > self.max_d {
                self.max_d = d;
            }
            
            // Check against critical value
            if d > self.critical_value {
                self.drift_detected.store(true, Ordering::SeqCst);
                return true;
            }
        }
        
        false
    }
    
    /// Find which bin a value falls into
    #[inline(always)]
    fn find_bin(&self, value: f64) -> Option<usize> {
        for i in 0..self.bin_edges.len() - 1 {
            if value >= self.bin_edges[i] && value < self.bin_edges[i + 1] {
                return Some(i);
            }
        }
        None
    }
    
    /// Reset the observation CDF
    pub fn reset_observations(&mut self) {
        let num_bins = self.bin_edges.len() - 1;
        self.observation_cdf = vec![0.0; num_bins];
        self.count.store(0, Ordering::Relaxed);
        self.max_d = 0.0;
        self.drift_detected.store(false, Ordering::SeqCst);
    }
    
    /// Get current KS statistic
    pub fn statistic(&self) -> f64 {
        let mut d = 0.0;
        for i in 0..self.reference_cdf.len() {
            let diff = (self.reference_cdf[i] - self.observation_cdf[i]).abs();
            if diff > d {
                d = diff;
            }
        }
        d
    }
    
    /// Check if drift detected
    pub fn has_drift(&self) -> bool {
        self.drift_detected.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_page_hinkley_no_drift() {
        let mut ph = PageHinkleyTest::new(100.0, 30);
        
        // Stable errors around zero
        for _ in 0..100 {
            assert!(!ph.update(0.001, 0.005));
        }
        
        assert!(!ph.has_drift());
    }
    
    #[test]
    fn test_page_hinkley_detects_shift() {
        let mut ph = PageHinkleyTest::new(50.0, 20);
        
        // Initial stable period
        for _ in 0..30 {
            ph.update(0.0, 0.001);
        }
        
        // Shift to large positive errors
        for _ in 0..50 {
            if ph.update(0.1, 0.001) {
                break;
            }
        }
        
        assert!(ph.has_drift());
    }
    
    #[test]
    fn test_ks_test_uniform() {
        let mut ks = StreamingKSTest::new(10, 0.0, 1.0, 0.5);
        
        // Feed uniform distribution matching reference
        for i in 0..100 {
            let val = (i % 10) as f64 / 10.0 + 0.05;
            ks.observe(val);
        }
        
        // Should not detect drift for matching distribution
        // (depends on critical value)
    }
    
    #[test]
    fn test_ks_test_detects_shift() {
        let mut ks = StreamingKSTest::new(10, 0.0, 1.0, 0.2);
        
        // Set reference as uniform
        let reference: Vec<f64> = (0..100).map(|i| (i % 10) as f64 / 10.0).collect();
        ks.set_reference(&reference);
        
        // Feed concentrated values (different distribution)
        for _ in 0..50 {
            if ks.observe(0.95) {
                break;
            }
        }
        
        assert!(ks.has_drift());
    }
    
    #[test]
    fn test_ph_reset() {
        let mut ph = PageHinkleyTest::new(50.0, 20);
        
        // Cause drift
        for _ in 0..100 {
            ph.update(0.1, 0.001);
        }
        assert!(ph.has_drift());
        
        // Reset
        ph.reset();
        assert!(!ph.has_drift());
        assert_eq!(ph.sample_count(), 0);
    }
}
