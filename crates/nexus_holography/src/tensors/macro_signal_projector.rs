//! Macro Signal Projector
//! 
//! Projects coarse-grained MERA output to trading signals.
//! Extracts long-range entanglement (institutional signals) from HFT noise.

use crate::tensors::{MeraRenormalizer, MeraConfig, TensorConfig};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors specific to signal projection
#[derive(Error, Debug, Clone, PartialEq)]
pub enum SignalProjectionError {
    #[error("Invalid signal threshold: {0}")]
    InvalidThreshold(f64),
    #[error("Renormalization failed: {0}")]
    RenormalizationFailed(String),
    #[error("Signal extraction failed: {0}")]
    SignalExtractionFailed(String),
    #[error("No macro signal detected")]
    NoMacroSignal,
}

/// Extracted macro signal with confidence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroSignal {
    /// Signal direction: +1 for buy, -1 for sell
    pub direction: i8,
    /// Signal strength [0, 1]
    pub strength: f64,
    /// Confidence score [0, 1]
    pub confidence: f64,
    /// Estimated time horizon (in milliseconds)
    pub time_horizon_ms: u64,
    /// Scale at which signal was detected (MERA layer)
    pub detection_scale: usize,
    /// Raw signal value before normalization
    pub raw_value: f64,
}

/// Configuration for signal projector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectorConfig {
    /// Minimum signal strength to report
    pub min_strength: f64,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Time horizon scaling factor per MERA layer
    pub horizon_scale_factor: u64,
    /// Base time horizon in ms
    pub base_horizon_ms: u64,
}

impl Default for ProjectorConfig {
    fn default() -> Self {
        Self {
            min_strength: 0.1,
            min_confidence: 0.5,
            horizon_scale_factor: 2,
            base_horizon_ms: 1, // 1ms base
        }
    }
}

/// Macro signal projector using MERA output
pub struct MacroSignalProjector {
    /// MERA renormalizer
    mera: MeraRenormalizer,
    /// Projector configuration
    config: ProjectorConfig,
    /// Entropy baseline for comparison
    entropy_baseline: f64,
}

impl MacroSignalProjector {
    /// Create a new signal projector
    pub fn new(
        mera_config: MeraConfig,
        tensor_config: TensorConfig,
        projector_config: ProjectorConfig,
    ) -> Result<Self, SignalProjectionError> {
        if projector_config.min_strength < 0.0 || projector_config.min_strength > 1.0 {
            return Err(SignalProjectionError::InvalidThreshold(projector_config.min_strength));
        }
        if projector_config.min_confidence < 0.0 || projector_config.min_confidence > 1.0 {
            return Err(SignalProjectionError::InvalidThreshold(projector_config.min_confidence));
        }

        let mera = MeraRenormalizer::new(mera_config, tensor_config)
            .map_err(|e| SignalProjectionError::RenormalizationFailed(e.to_string()))?;

        Ok(Self {
            mera,
            config: projector_config,
            entropy_baseline: 0.0,
        })
    }

    /// Set entropy baseline from historical data
    pub fn set_entropy_baseline(&mut self, baseline: f64) {
        self.entropy_baseline = baseline;
    }

    /// Process tape data and extract macro signal
    pub fn extract_signal(&self, tape_data: &[f64]) -> Result<Option<MacroSignal>, SignalProjectionError> {
        // Apply MERA renormalization
        let renormalized = self.mera.renormalize(tape_data)
            .map_err(|e| SignalProjectionError::RenormalizationFailed(e.to_string()))?;

        // Compute scale entropies
        let entropies = self.mera.compute_scale_entropies(tape_data)
            .map_err(|e| SignalProjectionError::SignalExtractionFailed(e.to_string()))?;

        // Find the scale with maximum entropy deviation from baseline
        let mut max_deviation = 0.0;
        let mut best_scale = 0;
        let mut best_direction = 0i8;
        let mut best_raw_value = 0.0;

        for (scale, &entropy) in entropies.iter().enumerate() {
            let deviation = (entropy - self.entropy_baseline).abs();
            
            if deviation > max_deviation {
                max_deviation = deviation;
                best_scale = scale;
                
                // Determine direction from sign of deviation
                best_direction = if entropy > self.entropy_baseline { 1 } else { -1 };
                best_raw_value = entropy - self.entropy_baseline;
            }
        }

        // Check if signal is strong enough
        let normalized_strength = max_deviation / (self.entropy_baseline + max_deviation);
        
        if normalized_strength < self.config.min_strength {
            return Ok(None);
        }

        // Compute confidence based on entropy contrast
        let entropy_contrast = if self.entropy_baseline > 0.0 {
            max_deviation / self.entropy_baseline
        } else {
            max_deviation
        };

        let confidence = (entropy_contrast / (1.0 + entropy_contrast)).min(1.0);

        if confidence < self.config.min_confidence {
            return Ok(None);
        }

        // Compute time horizon based on detection scale
        let time_horizon = self.config.base_horizon_ms 
            * self.config.horizon_scale_factor.pow(best_scale as u32);

        Ok(Some(MacroSignal {
            direction: best_direction,
            strength: normalized_strength,
            confidence,
            time_horizon_ms: time_horizon,
            detection_scale: best_scale,
            raw_value: best_raw_value,
        }))
    }

    /// Batch process multiple tape windows
    pub fn batch_extract(
        &self,
        tape_windows: &[Vec<f64>],
    ) -> Result<Vec<Option<MacroSignal>>, SignalProjectionError> {
        tape_windows
            .iter()
            .map(|window| self.extract_signal(window))
            .collect()
    }

    /// Detect regime change by comparing consecutive signals
    pub fn detect_regime_change(
        &self,
        prev_signal: Option<&MacroSignal>,
        curr_signal: Option<&MacroSignal>,
    ) -> RegimeChangeSignal {
        match (prev_signal, curr_signal) {
            (None, None) => RegimeChangeSignal {
                change_detected: false,
                change_type: "none",
                confidence: 0.0,
            },
            (None, Some(_)) => RegimeChangeSignal {
                change_detected: true,
                change_type: "signal_emergence",
                confidence: curr_signal.map(|s| s.confidence).unwrap_or(0.0),
            },
            (Some(prev), None) => RegimeChangeSignal {
                change_detected: true,
                change_type: "signal_disappearance",
                confidence: prev.confidence,
            },
            (Some(prev), Some(curr)) => {
                let direction_changed = prev.direction != curr.direction;
                let strength_changed = (prev.strength - curr.strength).abs() > 0.3;
                let scale_changed = prev.detection_scale != curr.detection_scale;

                let change_detected = direction_changed || strength_changed || scale_changed;

                let change_type = if direction_changed && strength_changed {
                    "full_reversal"
                } else if direction_changed {
                    "direction_flip"
                } else if strength_changed {
                    "strength_shift"
                } else if scale_changed {
                    "scale_migration"
                } else {
                    "none"
                };

                let confidence = if change_detected {
                    (prev.confidence + curr.confidence) / 2.0
                } else {
                    0.0
                };

                RegimeChangeSignal {
                    change_detected,
                    change_type,
                    confidence,
                }
            }
        }
    }

    /// Filter HFT noise by keeping only deep-scale components
    pub fn filter_hft_noise(&self, tape_data: &[f64], min_scale: usize) -> Result<Vec<f64>, SignalProjectionError> {
        let entropies = self.mera.compute_scale_entropies(tape_data)
            .map_err(|e| SignalProjectionError::SignalExtractionFailed(e.to_string()))?;

        // Keep only scales >= min_scale (deep institutional signals)
        let filtered: Vec<f64> = entropies
            .into_iter()
            .enumerate()
            .filter(|(scale, _)| *scale >= min_scale)
            .map(|(_, e)| e)
            .collect();

        if filtered.is_empty() {
            return Err(SignalProjectionError::NoMacroSignal);
        }

        Ok(filtered)
    }
}

/// Signal indicating regime change detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeChangeSignal {
    /// Whether a regime change was detected
    pub change_detected: bool,
    /// Type of change ("none", "signal_emergence", "signal_disappearance", 
    ///                  "full_reversal", "direction_flip", "strength_shift", "scale_migration")
    pub change_type: &'static str,
    /// Confidence score [0, 1]
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projector_creation() {
        let mera_config = MeraConfig::default();
        let tensor_config = TensorConfig::default();
        let proj_config = ProjectorConfig::default();
        
        let projector = MacroSignalProjector::new(mera_config, tensor_config, proj_config);
        assert!(projector.is_ok());
    }

    #[test]
    fn test_signal_extraction() {
        let mera_config = MeraConfig::default();
        let tensor_config = TensorConfig::default();
        let proj_config = ProjectorConfig::default();
        let mut projector = MacroSignalProjector::new(mera_config, tensor_config, proj_config).unwrap();
        
        // Set a baseline
        projector.set_entropy_baseline(0.5);

        // Create synthetic tape data with some structure
        let tape_data: Vec<f64> = vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.025];
        
        let signal = projector.extract_signal(&tape_data);
        assert!(signal.is_ok());
        // Signal may be None if not strong enough, which is valid
    }

    #[test]
    fn test_invalid_threshold() {
        let mera_config = MeraConfig::default();
        let tensor_config = TensorConfig::default();
        let mut proj_config = ProjectorConfig::default();
        proj_config.min_strength = 1.5; // Invalid
        
        let projector = MacroSignalProjector::new(mera_config, tensor_config, proj_config);
        assert!(projector.is_err());
    }
}
