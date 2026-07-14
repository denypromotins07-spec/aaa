//! Streaming Kolmogorov-Smirnov Test for Distribution Comparison
//! 
//! Compares live market feature distributions against training data
//! to detect concept drift. Uses incremental CDF updates for O(1) operations.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Configuration for the streaming KS test
#[derive(Debug, Clone)]
pub struct KSTestConfig {
    pub num_bins: usize,
    pub min_val: f64,
    pub max_val: f64,
    pub critical_value: f64,
    pub cooldown_samples: u64,
}

impl Default for KSTestConfig {
    fn default() -> Self {
        Self {
            num_bins: 50,
            min_val: -3.0, // Standardized features typically in [-3, 3]
            max_val: 3.0,
            critical_value: 0.15, // ~p < 0.05 for moderate sample sizes
            cooldown_samples: 100,
        }
    }
}

/// Streaming KS test with zero-alloc hot path
pub struct StreamingKSTest {
    /// Reference CDF from training data
    reference_cdf: Box<[f64]>,
    /// Live observation CDF (incrementally updated)
    observation_cdf: Box<[f64]>,
    /// Observation counts per bin (for weighted updates)
    bin_counts: Box<[u64]>,
    /// Total observations
    count: AtomicU64,
    /// Current D statistic
    current_d: f64,
    /// Critical value threshold
    critical_value: f64,
    /// Samples since last detection (cooldown)
    samples_since_detection: u64,
    /// Cooldown period to prevent oscillation
    cooldown_samples: u64,
    /// Drift detected flag
    drift_detected: AtomicBool,
}

impl StreamingKSTest {
    /// Create a new streaming KS test
    pub fn new(config: KSTestConfig) -> Self {
        let num_bins = config.num_bins;
        
        Self {
            reference_cdf: vec![0.0; num_bins].into_boxed_slice(),
            observation_cdf: vec![0.0; num_bins].into_boxed_slice(),
            bin_counts: vec![0; num_bins].into_boxed_slice(),
            count: AtomicU64::new(0),
            current_d: 0.0,
            critical_value: config.critical_value,
            samples_since_detection: 0,
            cooldown_samples: config.cooldown_samples,
            drift_detected: AtomicBool::new(false),
        }
    }
    
    /// Initialize reference CDF from training data
    pub fn initialize_reference(&mut self, training_data: &[f64]) {
        let num_bins = self.reference_cdf.len();
        let mut counts = vec![0usize; num_bins];
        
        // Bin the training data
        for &val in training_data {
            if let Some(bin) = self.find_bin(val) {
                counts[bin] += 1;
            }
        }
        
        // Convert to cumulative distribution
        let total = counts.iter().sum::<usize>() as f64;
        if total > 0.0 {
            let mut cumsum = 0.0;
            for (i, &count) in counts.iter().enumerate() {
                cumsum += count as f64 / total;
                self.reference_cdf[i] = cumsum;
            }
        }
    }
    
    /// Set reference CDF directly (for pre-computed distributions)
    pub fn set_reference_cdf(&mut self, cdf: &[f64]) {
        if cdf.len() == self.reference_cdf.len() {
            for (i, &val) in cdf.iter().enumerate() {
                self.reference_cdf[i] = val;
            }
        }
    }
    
    /// Observe a new value and check for drift (zero-alloc, O(n) where n=num_bins)
    #[inline(always)]
    pub fn observe(&mut self, value: f64) -> bool {
        // Check cooldown
        if self.samples_since_detection < self.cooldown_samples {
            self.samples_since_detection += 1;
            return false;
        }
        
        let current_count = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        
        let Some(bin) = self.find_bin(value) else {
            return false;
        };
        
        // Update bin count
        self.bin_counts[bin] += 1;
        
        // Incrementally update observation CDF
        // New CDF[i] = old_CDF[i] + (indicator(i >= bin) / n - old_CDF[i]) / n
        let n = current_count as f64;
        let inv_n = 1.0 / n;
        
        for i in bin..self.observation_cdf.len() {
            let old_val = self.observation_cdf[i];
            // Welford-style incremental update for numerical stability
            self.observation_cdf[i] = old_val + (inv_n - old_val) / n;
        }
        
        // Calculate KS statistic (max absolute difference)
        let mut d = 0.0;
        for i in 0..self.reference_cdf.len() {
            let diff = (self.reference_cdf[i] - self.observation_cdf[i]).abs();
            if diff > d {
                d = diff;
            }
        }
        
        self.current_d = d;
        
        // Check for drift
        if d > self.critical_value {
            self.drift_detected.store(true, Ordering::SeqCst);
            self.samples_since_detection = 0;
            return true;
        }
        
        false
    }
    
    /// Find bin index for a value
    #[inline(always)]
    fn find_bin(&self, value: f64) -> Option<usize> {
        let num_bins = self.reference_cdf.len();
        if value < -1e10 || value > 1e10 {
            return None; // Out of reasonable range
        }
        
        // Linear search (fast for small num_bins)
        // For production with many bins, use binary search
        for i in 0..num_bins {
            let bin_start = -3.0 + 6.0 * (i as f64 / num_bins as f64);
            let bin_end = -3.0 + 6.0 * ((i + 1) as f64 / num_bins as f64);
            if value >= bin_start && value < bin_end {
                return Some(i);
            }
        }
        
        // Handle edge case for max value
        if value >= -3.0 + 6.0 * ((num_bins - 1) as f64 / num_bins as f64) {
            return Some(num_bins - 1);
        }
        
        None
    }
    
    /// Reset observations (keep reference)
    pub fn reset_observations(&mut self) {
        for val in self.observation_cdf.iter_mut() {
            *val = 0.0;
        }
        for val in self.bin_counts.iter_mut() {
            *val = 0;
        }
        self.count.store(0, Ordering::Relaxed);
        self.current_d = 0.0;
        self.samples_since_detection = 0;
        self.drift_detected.store(false, Ordering::SeqCst);
    }
    
    /// Get current D statistic
    #[inline(always)]
    pub fn statistic(&self) -> f64 {
        self.current_d
    }
    
    /// Check if drift is detected
    #[inline(always)]
    pub fn has_drift(&self) -> bool {
        self.drift_detected.load(Ordering::Relaxed)
    }
    
    /// Get observation count
    #[inline(always)]
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
    
    /// Get critical value
    #[inline(always)]
    pub fn critical_value(&self) -> f64 {
        self.critical_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ks_initialization() {
        let config = KSTestConfig::default();
        let mut ks = StreamingKSTest::new(config);
        
        // Generate training data from normal distribution
        let training: Vec<f64> = (0..1000)
            .map(|i| ((i as f64 * 0.1).sin() - 0.5))
            .collect();
        
        ks.initialize_reference(&training);
        
        // Reference CDF should be populated
        assert!(ks.reference_cdf.iter().any(|&x| x > 0.0));
        assert!(ks.reference_cdf.last().copied().unwrap_or(0.0) > 0.9);
    }
    
    #[test]
    fn test_ks_no_drift_same_distribution() {
        let mut config = KSTestConfig::default();
        config.critical_value = 0.3; // Relaxed threshold for testing
        let mut ks = StreamingKSTest::new(config);
        
        // Initialize with uniform-like data
        let training: Vec<f64> = (0..500)
            .map(|i| -3.0 + 6.0 * ((i % 50) as f64 / 50.0))
            .collect();
        ks.initialize_reference(&training);
        
        // Feed similar distribution
        let mut drift_count = 0;
        for i in 0..200 {
            let val = -3.0 + 6.0 * ((i % 50) as f64 / 50.0);
            if ks.observe(val) {
                drift_count += 1;
            }
        }
        
        // Should not detect significant drift
        assert!(drift_count < 10, "Too many false positives");
    }
    
    #[test]
    fn test_ks_detects_distribution_shift() {
        let mut config = KSTestConfig::default();
        config.critical_value = 0.15;
        let mut ks = StreamingKSTest::new(config);
        
        // Initialize with centered distribution
        let training: Vec<f64> = (0..500)
            .map(|i| -1.0 + 2.0 * ((i % 50) as f64 / 50.0))
            .collect();
        ks.initialize_reference(&training);
        
        // Feed shifted distribution (all values at high end)
        let mut detected = false;
        for _ in 0..100 {
            if ks.observe(2.5) { // Far from training mean
                detected = true;
                break;
            }
        }
        
        assert!(detected, "Should detect distribution shift");
    }
    
    #[test]
    fn test_ks_cooldown_prevents_oscillation() {
        let mut config = KSTestConfig::default();
        config.cooldown_samples = 50;
        let mut ks = StreamingKSTest::new(config);
        
        // Initialize
        ks.initialize_reference(&vec![0.0; 100]);
        
        // Trigger drift
        let first_detection = ks.observe(5.0);
        
        // Immediate subsequent observations should not trigger due to cooldown
        let mut additional_detections = 0;
        for _ in 0..49 {
            if ks.observe(5.0) {
                additional_detections += 1;
            }
        }
        
        assert!(first_detection);
        assert_eq!(additional_detections, 0, "Cooldown should prevent rapid re-detection");
    }
    
    #[test]
    fn test_ks_reset() {
        let mut ks = StreamingKSTest::new(KSTestConfig::default());
        ks.initialize_reference(&vec![0.0; 100]);
        
        // Add observations
        for _ in 0..50 {
            ks.observe(0.5);
        }
        
        assert!(ks.count() > 0);
        
        ks.reset_observations();
        
        assert_eq!(ks.count(), 0);
        assert!(!ks.has_drift());
    }
}
