//! Intensity Spike Classifier for detecting anomalous cancellation rates.
//! 
//! Identifies when the cancellation rate at a specific price level mathematically
//! diverges from the baseline arrival rate, indicating phantom liquidity.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use parking_lot::RwLock;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ClassificationError {
    #[error("Insufficient data for classification")]
    InsufficientData,
    #[error("Invalid threshold configuration")]
    InvalidThreshold,
}

/// Classification result for intensity spike analysis
#[derive(Debug, Clone)]
pub struct IntensityClassification {
    pub price_level: i64,
    pub is_spike: bool,
    pub severity: f64,  // 0.0 to 1.0
    pub z_score: f64,
    pub deviation_from_baseline: f64,
    pub timestamp_ns: u64,
}

/// Rolling statistics for a price level using Welford's online algorithm
struct RollingStats {
    count: usize,
    mean: f64,
    m2: f64,  // Sum of squared differences from mean
    min_value: f64,
    max_value: f64,
}

impl RollingStats {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min_value: f64::MAX,
            max_value: f64::MIN,
        }
    }

    #[inline]
    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        
        if value < self.min_value {
            self.min_value = value;
        }
        if value > self.max_value {
            self.max_value = value;
        }
    }

    #[inline]
    fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    #[inline]
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Configuration for the intensity spike classifier
pub struct SpikeClassifierConfig {
    /// Number of standard deviations for spike detection
    pub z_threshold: f64,
    /// Minimum samples required before classification
    pub min_samples: usize,
    /// Decay factor for old samples (exponential weighting)
    pub decay_factor: f64,
    /// Maximum intensity value to consider
    pub max_intensity: f64,
}

impl Default for SpikeClassifierConfig {
    fn default() -> Self {
        Self {
            z_threshold: 3.0,  // 3 sigma event
            min_samples: 50,
            decay_factor: 0.99,
            max_intensity: 1e8,
        }
    }
}

/// Per-price-level tracking state
struct PriceLevelTracker {
    stats: RollingStats,
    last_intensity: f64,
    spike_count: AtomicUsize,
}

impl PriceLevelTracker {
    fn new() -> Self {
        Self {
            stats: RollingStats::new(),
            last_intensity: 0.0,
            spike_count: AtomicUsize::new(0),
        }
    }
}

/// Intensity Spike Classifier for detecting phantom liquidity
pub struct IntensitySpikeClassifier {
    config: SpikeClassifierConfig,
    trackers: RwLock<std::collections::HashMap<i64, PriceLevelTracker>>,
    global_sample_count: AtomicU64,
}

impl IntensitySpikeClassifier {
    /// Create a new intensity spike classifier
    pub fn new(config: SpikeClassifierConfig) -> Result<Self, ClassificationError> {
        if config.z_threshold <= 0.0 || config.decay_factor <= 0.0 || config.decay_factor > 1.0 {
            return Err(ClassificationError::InvalidThreshold);
        }
        
        Ok(Self {
            config,
            trackers: RwLock::new(std::collections::HashMap::with_capacity(256)),
            global_sample_count: AtomicU64::new(0),
        })
    }

    /// Record an intensity observation for a price level
    #[inline]
    pub fn record_intensity(&self, price_level: i64, intensity: f64) {
        // Clamp intensity to prevent numerical issues
        let clamped = intensity.min(self.config.max_intensity).max(0.0);
        
        let mut trackers = self.trackers.write();
        let tracker = trackers.entry(price_level).or_insert_with(PriceLevelTracker::new);
        
        // Apply exponential decay to existing statistics if configured
        if self.config.decay_factor < 1.0 && tracker.stats.count > 0 {
            // Weighted update: give more weight to recent observations
            let weight = 1.0 - self.config.decay_factor;
            tracker.stats.mean = tracker.stats.mean * self.config.decay_factor + clamped * weight;
            tracker.stats.m2 *= self.config.decay_factor; // Approximate decay for M2
        } else {
            tracker.stats.update(clamped);
        }
        
        tracker.last_intensity = clamped;
        self.global_sample_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Classify whether current intensity represents a spike
    pub fn classify(&self, price_level: i64, current_time_ns: u64) -> Result<IntensityClassification, ClassificationError> {
        let trackers = self.trackers.read();
        
        let tracker = trackers.get(&price_level)
            .ok_or(ClassificationError::InsufficientData)?;
        
        if tracker.stats.count < self.config.min_samples {
            return Err(ClassificationError::InsufficientData);
        }
        
        let std_dev = tracker.stats.std_dev();
        let mean = tracker.stats.mean;
        let current = tracker.last_intensity;
        
        // Calculate Z-score with protection against division by zero
        let z_score = if std_dev > 1e-10 {
            (current - mean) / std_dev
        } else {
            0.0
        };
        
        let is_spike = z_score > self.config.z_threshold;
        let severity = if is_spike {
            ((z_score - self.config.z_threshold) / self.config.z_threshold).min(1.0)
        } else {
            0.0
        };
        
        if is_spike {
            tracker.spike_count.fetch_add(1, Ordering::Relaxed);
        }
        
        Ok(IntensityClassification {
            price_level,
            is_spike,
            severity,
            z_score,
            deviation_from_baseline: current - mean,
            timestamp_ns: current_time_ns,
        })
    }

    /// Get all current spikes across all price levels
    pub fn detect_all_spikes(&self, current_time_ns: u64) -> Vec<IntensityClassification> {
        let trackers = self.trackers.read();
        let mut spikes = Vec::new();
        
        for (&price_level, tracker) in trackers.iter() {
            if tracker.stats.count >= self.config.min_samples {
                let std_dev = tracker.stats.std_dev();
                let mean = tracker.stats.mean;
                let current = tracker.last_intensity;
                
                let z_score = if std_dev > 1e-10 {
                    (current - mean) / std_dev
                } else {
                    0.0
                };
                
                if z_score > self.config.z_threshold {
                    let severity = ((z_score - self.config.z_threshold) / self.config.z_threshold).min(1.0);
                    spikes.push(IntensityClassification {
                        price_level,
                        is_spike: true,
                        severity,
                        z_score,
                        deviation_from_baseline: current - mean,
                        timestamp_ns: current_time_ns,
                    });
                }
            }
        }
        
        spikes
    }

    /// Get historical spike count for a price level
    pub fn get_spike_count(&self, price_level: i64) -> usize {
        let trackers = self.trackers.read();
        trackers.get(&price_level)
            .map(|t| t.spike_count.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Reset classifier state
    pub fn reset(&self) {
        self.trackers.write().clear();
        self.global_sample_count.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_z_score_calculation() {
        let classifier = IntensitySpikeClassifier::new(SpikeClassifierConfig::default()).unwrap();
        
        // Add normal observations
        for i in 0..100 {
            classifier.record_intensity(100, 10.0 + (i % 5) as f64);
        }
        
        // Add a spike
        classifier.record_intensity(100, 50.0);
        
        let result = classifier.classify(100, 1000000).unwrap();
        assert!(result.is_spike || result.z_score > 2.0);
    }

    #[test]
    fn test_insufficient_data() {
        let classifier = IntensitySpikeClassifier::new(SpikeClassifierConfig::default()).unwrap();
        
        // Only add a few samples
        for i in 0..10 {
            classifier.record_intensity(100, 10.0);
        }
        
        let result = classifier.classify(100, 1000000);
        assert!(result.is_err());
    }
}
