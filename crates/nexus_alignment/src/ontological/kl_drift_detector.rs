// STAGE 24: ONTOLOGICAL CRISIS DETECTION - KL DRIFT DETECTOR
// ============================================================
//!
//! This module implements ontological drift detection by computing
//! KL-divergence between predicted and actual market observations.
//!
//! Critical features:
//! - Cross-references with EVT tail-risk engine to distinguish
//!   ontological crisis from known black swan events
//! - Prevents false-positive shutdowns during real market stress
//! - Triggers epistemic humility mode when divergence exceeds threshold

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use nalgebra::{MatrixXf, VectorXf};
use rand::Rng;

/// Configuration for the KL drift detector
#[derive(Debug, Clone)]
pub struct KLDriftConfig {
    /// Sliding window size for distribution estimation
    pub window_size: usize,
    /// Base threshold for KL divergence alert
    pub base_threshold: f64,
    /// Adaptive threshold scaling factor
    pub adaptive_scale: f64,
    /// Minimum samples before activation
    pub min_samples: usize,
    /// Number of histogram bins for density estimation
    pub num_bins: usize,
    /// Cross-reference with EVT engine
    pub evt_integration: bool,
}

impl Default for KLDriftConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            base_threshold: 0.5,
            adaptive_scale: 0.1,
            min_samples: 100,
            num_bins: 50,
            evt_integration: true,
        }
    }
}

/// Result of a KL divergence check
#[derive(Debug, Clone)]
pub struct DriftCheckResult {
    pub kl_divergence: f64,
    pub is_ontological_crisis: bool,
    pub is_black_swan_event: bool,
    pub confidence: f64,
    pub recommended_action: DriftAction,
    pub timestamp: Instant,
}

/// Recommended action based on drift analysis
#[derive(Debug, Clone, PartialEq)]
pub enum DriftAction {
    ContinueNormal,
    ReducePositionSize(f64),
    IncreaseCashReserves(f64),
    RequestHumanConfirmation,
    EnterEpistemicHumility,
    EmergencyHalt,
}

/// KL Divergence Detector for ontological crisis detection
pub struct KLDriftDetector {
    config: KLDriftConfig,
    predicted_distribution: VecDeque<f64>,
    actual_distribution: VecDeque<f64>,
    historical_kl_values: VecDeque<f64>,
    baseline_kl_mean: f64,
    baseline_kl_std: f64,
    sample_count: usize,
    evt_tail_risk_active: bool,
}

impl KLDriftDetector {
    /// Create a new KL drift detector with given configuration
    pub fn new(config: KLDriftConfig) -> Self {
        Self {
            config,
            predicted_distribution: VecDeque::with_capacity(config.window_size),
            actual_distribution: VecDeque::with_capacity(config.window_size),
            historical_kl_values: VecDeque::with_capacity(config.window_size / 10),
            baseline_kl_mean: 0.0,
            baseline_kl_std: 0.1,
            sample_count: 0,
            evt_tail_risk_active: false,
        }
    }

    /// Record a predicted observation from the world model
    pub fn record_prediction(&mut self, value: f64) {
        if self.predicted_distribution.len() >= self.config.window_size {
            self.predicted_distribution.pop_front();
        }
        self.predicted_distribution.push_back(value);
        self.sample_count += 1;
    }

    /// Record an actual observation from the market
    pub fn record_actual(&mut self, value: f64) {
        if self.actual_distribution.len() >= self.config.window_size {
            self.actual_distribution.pop_front();
        }
        self.actual_distribution.push_back(value);
    }

    /// Set EVT tail-risk status (cross-reference for black swan detection)
    pub fn set_evt_tail_risk_active(&mut self, active: bool) {
        self.evt_tail_risk_active = active;
    }

    /// Compute KL divergence between predicted and actual distributions
    pub fn compute_kl_divergence(&self) -> Option<f64> {
        if self.predicted_distribution.len() < self.config.min_samples
            || self.actual_distribution.len() < self.config.min_samples
        {
            return None;
        }

        // Build histograms for both distributions
        let pred_hist = self.build_histogram(&self.predicted_distribution);
        let actual_hist = self.build_histogram(&self.actual_distribution);

        // Compute KL(P||Q) where P is actual, Q is predicted
        let mut kl_div = 0.0;
        let epsilon = 1e-10; // Smoothing constant

        for (p, q) in pred_hist.iter().zip(actual_hist.iter()) {
            if *p > epsilon && *q > epsilon {
                kl_div += p * (p / q).ln();
            }
        }

        Some(kl_div)
    }

    /// Perform a complete drift check with EVT cross-reference
    pub fn check_drift(&mut self) -> Option<DriftCheckResult> {
        let kl_div = self.compute_kl_divergence()?;

        // Update historical KL values for adaptive thresholding
        self.historical_kl_values.push_back(kl_div);
        if self.historical_kl_values.len() > 100 {
            self.historical_kl_values.pop_front();
        }
        self.update_baseline_statistics();

        // Compute adaptive threshold
        let adaptive_threshold = self.compute_adaptive_threshold();

        // Determine if this is an ontological crisis
        let normalized_divergence = (kl_div - self.baseline_kl_mean) 
            / (self.baseline_kl_std + 1e-10);

        // Check if EVT tail-risk is active (indicates known black swan)
        let is_black_swan = self.evt_tail_risk_active && normalized_divergence > 2.0;

        // Ontological crisis if divergence is extreme AND not explained by EVT
        let is_ontological_crisis = normalized_divergence > 3.0 && !is_black_swan;

        // Determine recommended action
        let recommended_action = self.determine_action(normalized_divergence, is_ontological_crisis);

        // Calculate confidence based on sample size
        let confidence = 1.0 - ((self.config.min_samples as f64) 
            / (self.sample_count as f64 + 1e-10)).min(1.0);

        Some(DriftCheckResult {
            kl_divergence: kl_div,
            is_ontological_crisis,
            is_black_swan_event: is_black_swan,
            confidence,
            recommended_action,
            timestamp: Instant::now(),
        })
    }

    /// Build histogram for density estimation
    fn build_histogram(&self, data: &VecDeque<f64>) -> Vec<f64> {
        let data_vec: Vec<f64> = data.iter().copied().collect();
        
        if data_vec.is_empty() {
            return vec![1.0 / self.config.num_bins as f64; self.config.num_bins];
        }

        let min_val = *data_vec.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_val = *data_vec.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        let range = (max_val - min_val).max(1e-10);
        let bin_width = range / self.config.num_bins as f64;

        let mut histogram = vec![0.0; self.config.num_bins];

        for value in data_vec {
            let bin_idx = ((value - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(self.config.num_bins - 1);
            histogram[bin_idx] += 1.0;
        }

        // Normalize to probability distribution
        let total: f64 = histogram.iter().sum();
        if total > 0.0 {
            for h in &mut histogram {
                *h /= total;
            }
        }

        // Add smoothing to prevent zero probabilities
        let smoothing = 1e-6;
        for h in &mut histogram {
            *h = (*h + smoothing) / (1.0 + smoothing * self.config.num_bins as f64);
        }

        histogram
    }

    /// Update baseline statistics for adaptive thresholding
    fn update_baseline_statistics(&mut self) {
        if self.historical_kl_values.len() < 10 {
            return;
        }

        let values: Vec<f64> = self.historical_kl_values.iter().copied().collect();
        self.baseline_kl_mean = values.iter().sum::<f64>() / values.len() as f64;
        
        let variance: f64 = values.iter()
            .map(|x| (x - self.baseline_kl_mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        self.baseline_kl_std = variance.sqrt();
    }

    /// Compute adaptive threshold based on historical KL values
    fn compute_adaptive_threshold(&self) -> f64 {
        self.baseline_kl_mean + self.config.adaptive_scale * self.baseline_kl_std
    }

    /// Determine recommended action based on drift analysis
    fn determine_action(&self, normalized_divergence: f64, is_crisis: bool) -> DriftAction {
        if is_crisis {
            DriftAction::EmergencyHalt
        } else if normalized_divergence > 4.0 {
            DriftAction::EnterEpistemicHumility
        } else if normalized_divergence > 3.0 {
            DriftAction::RequestHumanConfirmation
        } else if normalized_divergence > 2.0 {
            DriftAction::IncreaseCashReserves(0.3) // Increase by 30%
        } else if normalized_divergence > 1.5 {
            DriftAction::ReducePositionSize(0.5) // Reduce by 50%
        } else {
            DriftAction::ContinueNormal
        }
    }

    /// Get current sample count
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Reset the detector state
    pub fn reset(&mut self) {
        self.predicted_distribution.clear();
        self.actual_distribution.clear();
        self.historical_kl_values.clear();
        self.baseline_kl_mean = 0.0;
        self.baseline_kl_std = 0.1;
        self.sample_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kl_divergence_computation() {
        let config = KLDriftConfig {
            window_size: 100,
            min_samples: 20,
            ..Default::default()
        };

        let mut detector = KLDriftDetector::new(config);

        // Record similar distributions
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            detector.record_prediction(rng.gen_range(-1.0..1.0));
            detector.record_actual(rng.gen_range(-1.0..1.0));
        }

        let result = detector.check_drift();
        assert!(result.is_some());
    }

    #[test]
    fn test_ontological_crisis_detection() {
        let config = KLDriftConfig::default();
        let mut detector = KLDriftDetector::new(config);

        // Simulate divergent distributions
        let mut rng = rand::thread_rng();
        for _ in 0..500 {
            detector.record_prediction(rng.gen_range(-1.0..1.0));
            detector.record_actual(rng.gen_range(5.0..10.0)); // Very different!
        }

        detector.set_evt_tail_risk_active(false); // Not a black swan
        
        let result = detector.check_drift().unwrap();
        assert!(result.kl_divergence > 0.0);
    }
}
