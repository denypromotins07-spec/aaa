//! Page-Hinkley Test for Streaming Drift Detection
//!
//! Implements the Page-Hinkley sequential change detection test.
//! More sensitive to gradual drift than ADWIN, with lower memory footprint.

use crate::MLOpsError;

/// Page-Hinkley test configuration
#[derive(Debug, Clone)]
pub struct PageHinkleyConfig {
    /// Minimum mean value threshold
    pub delta: f64,
    /// Alarm threshold - higher = less sensitive
    pub threshold: f64,
    /// Minimum samples before checking
    pub min_samples: usize,
    /// Forgetting factor for exponential weighting
    pub alpha: f64,
}

impl Default for PageHinkleyConfig {
    fn default() -> Self {
        Self {
            delta: 0.005,
            threshold: 50.0,
            min_samples: 30,
            alpha: 0.9999, // Near 1 for slow forgetting
        }
    }
}

/// Page-Hinkley sequential drift detector
pub struct PageHinkleyTest {
    config: PageHinkleyConfig,
    /// Running sum of deviations
    sum_deviation: f64,
    /// Minimum sum encountered (for cumulative sum)
    min_sum: f64,
    /// Running mean estimate
    running_mean: f64,
    /// Sample count
    n_samples: u64,
    /// Last alarm triggered
    last_alarm: Option<u64>,
}

impl PageHinkleyTest {
    /// Create new Page-Hinkley test with default config
    pub fn new() -> Self {
        Self::with_config(PageHinkleyConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: PageHinkleyConfig) -> Self {
        Self {
            config,
            sum_deviation: 0.0,
            min_sum: f64::INFINITY,
            running_mean: 0.0,
            n_samples: 0,
            last_alarm: None,
        }
    }

    /// Update with new sample, returns true if drift detected
    pub fn update(&mut self, value: f64) -> Result<bool, MLOpsError> {
        self.n_samples += 1;

        // Update running mean with exponential weighting
        let weight = if self.n_samples == 1 {
            1.0
        } else {
            self.config.alpha
        };
        
        self.running_mean = weight * self.running_mean + (1.0 - weight) * value;

        // Compute deviation from expected mean
        let deviation = value - self.running_mean - self.config.delta;
        
        // Update cumulative sum
        self.sum_deviation += deviation;
        
        // Track minimum sum for PH statistic
        if self.sum_deviation < self.min_sum {
            self.min_sum = self.sum_deviation;
        }

        // Check for drift only after minimum samples
        if self.n_samples < self.config.min_samples as u64 {
            return Ok(false);
        }

        // Page-Hinkley statistic
        let ph_statistic = self.sum_deviation - self.min_sum;

        if ph_statistic > self.config.threshold {
            self.last_alarm = Some(self.n_samples);
            // Reset after alarm to detect subsequent changes
            self.reset_partial();
            return Ok(true);
        }

        Ok(false)
    }

    /// Partial reset after alarm (keeps running mean)
    fn reset_partial(&mut self) {
        self.sum_deviation = 0.0;
        self.min_sum = f64::INFINITY;
    }

    /// Full reset of all state
    pub fn reset(&mut self) {
        self.sum_deviation = 0.0;
        self.min_sum = f64::INFINITY;
        self.running_mean = 0.0;
        self.n_samples = 0;
        self.last_alarm = None;
    }

    /// Get current running mean
    pub fn running_mean(&self) -> f64 {
        self.running_mean
    }

    /// Get current PH statistic value
    pub fn statistic(&self) -> f64 {
        self.sum_deviation - self.min_sum
    }

    /// Get sample count
    pub fn n_samples(&self) -> u64 {
        self.n_samples
    }

    /// Get last alarm sample index
    pub fn last_alarm(&self) -> Option<u64> {
        self.last_alarm
    }

    /// Adjust threshold dynamically based on regime
    pub fn set_threshold(&mut self, threshold: f64) {
        self.config.threshold = threshold.max(1.0); // Prevent too-low threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ph_stable_data() {
        let mut ph = PageHinkleyTest::new();
        
        for i in 0..500 {
            let value = 10.0 + (i % 20) as f64 * 0.1;
            let drift = ph.update(value).unwrap();
            assert!(!drift, "Should not detect drift in stable data");
        }
    }

    #[test]
    fn test_ph_detects_gradual_drift() {
        let mut ph = PageHinkleyTest::with_config(PageHinkleyConfig {
            delta: 0.001,
            threshold: 20.0,
            min_samples: 50,
            alpha: 0.999,
        });

        // Stable period
        for _ in 0..100 {
            ph.update(5.0).unwrap();
        }

        // Gradual drift
        let mut drift_detected = false;
        for i in 0..300 {
            let value = 5.0 + (i as f64) * 0.02;
            if ph.update(value).unwrap() {
                drift_detected = true;
                break;
            }
        }

        assert!(drift_detected, "Should detect gradual drift");
    }

    #[test]
    fn test_ph_reset_after_alarm() {
        let mut ph = PageHinkleyTest::with_config(PageHinkleyConfig {
            delta: 0.001,
            threshold: 10.0,
            min_samples: 30,
            alpha: 0.999,
        });

        // Cause an alarm
        for _ in 0..100 {
            ph.update(1.0).unwrap();
        }
        for i in 0..200 {
            let value = 1.0 + (i as f64) * 0.05;
            if ph.update(value).unwrap() {
                break;
            }
        }

        // Statistic should be reset
        assert!(ph.statistic() < ph.config.threshold);
    }
}
