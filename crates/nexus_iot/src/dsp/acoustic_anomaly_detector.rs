//! Acoustic Anomaly Detector for Industrial IoT
//! 
//! Detects anomalous acoustic patterns indicating equipment failure.
//! Uses statistical baseline comparison and spectral analysis.

use std::collections::VecDeque;

#[derive(Debug)]
pub struct AcousticAnomalyDetector {
    baseline_spectrum: Vec<f64>,
    recent_spectra: VecDeque<Vec<f64>>,
    max_window_size: usize,
    anomaly_threshold_std: f64,
}

impl AcousticAnomalyDetector {
    pub fn new(max_window_size: usize, threshold_std: f64) -> Self {
        Self {
            baseline_spectrum: Vec::new(),
            recent_spectra: VecDeque::with_capacity(max_window_size),
            max_window_size,
            anomaly_threshold_std: threshold_std,
        }
    }

    /// Set baseline from normal operating conditions
    pub fn set_baseline(&mut self, spectrum: Vec<f64>) {
        self.baseline_spectrum = spectrum;
    }

    /// Add new spectrum sample
    pub fn add_sample(&mut self, spectrum: Vec<f64>) {
        if self.recent_spectra.len() >= self.max_window_size {
            self.recent_spectra.pop_front();
        }
        self.recent_spectra.push_back(spectrum);
    }

    /// Detect anomaly using statistical deviation from baseline
    pub fn detect_anomaly(&self) -> Option<AnomalyReport> {
        if self.baseline_spectrum.is_empty() || self.recent_spectra.is_empty() {
            return None;
        }

        let current = self.recent_spectra.back().unwrap();
        
        if current.len() != self.baseline_spectrum.len() {
            return None;
        }

        // Calculate Mahalanobis-like distance
        let mut total_deviation = 0.0;
        let mut max_bin_deviation = 0.0;
        let mut max_bin_index = 0;

        for (i, (baseline, current)) in self.baseline_spectrum.iter().zip(current.iter()).enumerate() {
            let deviation = (current - baseline).abs();
            total_deviation += deviation;
            
            if deviation > max_bin_deviation {
                max_bin_deviation = deviation;
                max_bin_index = i;
            }
        }

        let avg_deviation = total_deviation / current.len() as f64;

        if avg_deviation > self.anomaly_threshold_std {
            Some(AnomalyReport {
                severity: avg_deviation / self.anomaly_threshold_std,
                primary_frequency_bin: max_bin_index,
                deviation_pattern: "broadband".to_string(),
            })
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct AnomalyReport {
    pub severity: f64,
    pub primary_frequency_bin: usize,
    pub deviation_pattern: String,
}
