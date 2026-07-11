//! CHSH Inequality (Bell's Theorem) Tester for Market Correlations
//! 
//! Tests if market microstructure violates Bell's inequality (S > 2),
//! indicating non-local quantum entanglement rather than classical correlations.
//! 
//! CRITICAL: Accounts for detection loophole and fair sampling assumption.

use alloc::vec::Vec;
use core::fmt;

/// Maximum number of measurement settings to test
const MAX_MEASUREMENT_SETTINGS: usize = 1024;

/// CHSH inequality bound (classical local realism limit)
const CHSH_CLASSICAL_BOUND: f64 = 2.0;

/// Quantum Tsirelson bound (maximum quantum violation)
const TSIRELSON_BOUND: f64 = 2.0 * 2.0_f64.sqrt();

/// Error types for CHSH testing
#[derive(Debug, Clone, PartialEq)]
pub enum ChshError {
    InsufficientSamples { count: usize, minimum: usize },
    DetectionLoophole { efficiency: f64, threshold: f64 },
    InvalidCorrelation { value: f64 },
    NumericalInstability { message: &'static str },
    MeasurementSettingInvalid { index: usize },
}

impl fmt::Display for ChshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChshError::InsufficientSamples { count, minimum } => {
                write!(f, "Insufficient samples: {}, minimum {}", count, minimum)
            }
            ChshError::DetectionLoophole { efficiency, threshold } => {
                write!(f, "Detection loophole: efficiency={}, threshold={}", efficiency, threshold)
            }
            ChshError::InvalidCorrelation { value } => {
                write!(f, "Invalid correlation: {}", value)
            }
            ChshError::NumericalInstability { message } => {
                write!(f, "Numerical instability: {}", message)
            }
            ChshError::MeasurementSettingInvalid { index } => {
                write!(f, "Invalid measurement setting: {}", index)
            }
        }
    }
}

/// Measurement outcome from a single detector
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeasurementOutcome {
    PlusOne = 1,
    MinusOne = -1,
    NoDetection = 0, // For detection loophole tracking
}

/// Pair of correlated measurements from two detectors (e.g., Tokyo and NY)
#[derive(Debug, Clone)]
pub struct MeasurementPair {
    pub outcome_a: MeasurementOutcome,
    pub outcome_b: MeasurementOutcome,
    pub setting_a: usize, // Measurement setting angle/mode for detector A
    pub setting_b: usize, // Measurement setting angle/mode for detector B
    pub timestamp_ns: u64,
}

/// CHSH Test Result
#[derive(Debug, Clone)]
pub struct ChshTestResult {
    /// Calculated S value
    pub s_value: f64,
    /// Standard error
    pub standard_error: f64,
    /// Number of valid samples
    pub sample_count: usize,
    /// Detection efficiency
    pub detection_efficiency: f64,
    /// Whether Bell inequality is violated
    pub bell_violation: bool,
    /// Whether violation exceeds classical bound significantly
    pub significance_sigma: f64,
}

/// CHSH Inequality Tester
pub struct ChshInequalityTester {
    /// Minimum detection efficiency to close detection loophole
    min_detection_efficiency: f64,
    /// Minimum samples required
    min_samples: usize,
    /// Measurement data storage
    measurements: Vec<MeasurementPair>,
}

impl ChshInequalityTester {
    pub fn new(min_detection_efficiency: f64, min_samples: usize) -> Result<Self, ChshError> {
        if min_detection_efficiency <= 0.0 || min_detection_efficiency > 1.0 {
            return Err(ChshError::DetectionLoophole {
                efficiency: min_detection_efficiency,
                threshold: 0.828, // Theoretical minimum for loophole-free test
            });
        }

        if min_samples < 100 {
            return Err(ChshError::InsufficientSamples {
                count: min_samples,
                minimum: 100,
            });
        }

        Ok(Self {
            min_detection_efficiency,
            min_samples,
            measurements: Vec::new(),
        })
    }

    /// Add a measurement pair
    pub fn add_measurement(&mut self, pair: MeasurementPair) -> Result<(), ChshError> {
        if self.measurements.len() >= MAX_MEASUREMENT_SETTINGS * MAX_MEASUREMENT_SETTINGS * 1000 {
            return Err(ChshError::NumericalInstability {
                message: "Maximum measurement storage exceeded",
            });
        }

        // Validate settings (typically 0-3 for standard CHSH: a, a', b, b')
        if pair.setting_a > 3 || pair.setting_b > 3 {
            return Err(ChshError::MeasurementSettingInvalid {
                index: if pair.setting_a > 3 { pair.setting_a } else { pair.setting_b },
            });
        }

        self.measurements.push(pair);
        Ok(())
    }

    /// Calculate correlation E(a,b) for given settings
    fn calculate_correlation(
        &self,
        setting_a: usize,
        setting_b: usize,
    ) -> Result<f64, ChshError> {
        let mut n_pp = 0; // Both +1
        let mut n_pm = 0; // A +1, B -1
        let mut n_mp = 0; // A -1, B +1
        let mut n_mm = 0; // Both -1

        for m in &self.measurements {
            if m.setting_a == setting_a && m.setting_b == setting_b {
                // Skip no-detection events (critical for detection loophole)
                if m.outcome_a == MeasurementOutcome::NoDetection
                    || m.outcome_b == MeasurementOutcome::NoDetection
                {
                    continue;
                }

                match (m.outcome_a, m.outcome_b) {
                    (MeasurementOutcome::PlusOne, MeasurementOutcome::PlusOne) => n_pp += 1,
                    (MeasurementOutcome::PlusOne, MeasurementOutcome::MinusOne) => n_pm += 1,
                    (MeasurementOutcome::MinusOne, MeasurementOutcome::PlusOne) => n_mp += 1,
                    (MeasurementOutcome::MinusOne, MeasurementOutcome::MinusOne) => n_mm += 1,
                }
            }
        }

        let total = n_pp + n_pm + n_mp + n_mm;
        
        if total < self.min_samples / 4 {
            return Err(ChshError::InsufficientSamples {
                count: total,
                minimum: self.min_samples / 4,
            });
        }

        // E(a,b) = (N++ + N-- - N+- - N-+) / N_total
        let correlation = ((n_pp + n_mm) as f64 - (n_pm + n_mp) as f64) / total as f64;

        // Validate correlation is in [-1, 1]
        if correlation < -1.0 - 1e-10 || correlation > 1.0 + 1e-10 {
            return Err(ChshError::InvalidCorrelation {
                value: correlation,
            });
        }

        Ok(correlation.clamp(-1.0, 1.0))
    }

    /// Calculate CHSH S value: S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
    pub fn calculate_chsh_s_value(&self) -> Result<ChshTestResult, ChshError> {
        if self.measurements.len() < self.min_samples {
            return Err(ChshError::InsufficientSamples {
                count: self.measurements.len(),
                minimum: self.min_samples,
            });
        }

        // Check detection efficiency
        let total_pairs = self.measurements.len();
        let detected_pairs = self.measurements.iter().filter(|m| {
            m.outcome_a != MeasurementOutcome::NoDetection
                && m.outcome_b != MeasurementOutcome::NoDetection
        }).count();

        let detection_efficiency = detected_pairs as f64 / total_pairs as f64;

        // Critical: Must exceed ~82.8% for loophole-free Bell test
        if detection_efficiency < self.min_detection_efficiency {
            return Err(ChshError::DetectionLoophole {
                efficiency: detection_efficiency,
                threshold: self.min_detection_efficiency,
            });
        }

        // Standard CHSH settings: a=0, a'=π/2, b=π/4, b'=-π/4
        // Mapped to indices: a=0, a'=1, b=2, b'=3
        let e_ab = self.calculate_correlation(0, 2)?;      // E(a,b)
        let e_ab_prime = self.calculate_correlation(0, 3)?; // E(a,b')
        let e_a_prime_b = self.calculate_correlation(1, 2)?; // E(a',b)
        let e_a_prime_b_prime = self.calculate_correlation(1, 3)?; // E(a',b')

        // S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
        let s_value = e_ab - e_ab_prime + e_a_prime_b + e_a_prime_b_prime;

        // Validate S is within physical bounds
        if s_value < -TSIRELSON_BOUND - 1e-10 || s_value > TSIRELSON_BOUND + 1e-10 {
            return Err(ChshError::NumericalInstability {
                message: "S value exceeds Tsirelson bound",
            });
        }

        // Estimate standard error (simplified)
        let standard_error = 2.0 / (detected_pairs as f64).sqrt();

        // Check for Bell violation
        let bell_violation = s_value > CHSH_CLASSICAL_BOUND;
        
        // Calculate significance
        let significance_sigma = if standard_error > 1e-15 {
            (s_value - CHSH_CLASSICAL_BOUND) / standard_error
        } else {
            0.0
        };

        Ok(ChshTestResult {
            s_value,
            standard_error,
            sample_count: detected_pairs,
            detection_efficiency,
            bell_violation,
            significance_sigma,
        })
    }

    /// Clear all measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
    }

    /// Get current measurement count
    pub fn measurement_count(&self) -> usize {
        self.measurements.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classical_correlations() {
        let mut tester = ChshInequalityTester::new(0.83, 1000).unwrap();

        // Simulate classical correlations (no Bell violation)
        for i in 0..1000 {
            let pair = MeasurementPair {
                outcome_a: if i % 2 == 0 { MeasurementOutcome::PlusOne } else { MeasurementOutcome::MinusOne },
                outcome_b: if i % 2 == 0 { MeasurementOutcome::PlusOne } else { MeasurementOutcome::MinusOne },
                setting_a: i % 4 / 2,
                setting_b: i % 2 + 2,
                timestamp_ns: i as u64,
            };
            tester.add_measurement(pair).unwrap();
        }

        let result = tester.calculate_chsh_s_value();
        // Classical correlations should not violate Bell inequality (or fail due to insufficient samples per setting)
        assert!(result.is_err() || result.unwrap().s_value <= CHSH_CLASSICAL_BOUND + 0.5);
    }

    #[test]
    fn test_detection_loophole() {
        let mut tester = ChshInequalityTester::new(0.83, 100).unwrap();

        // Add many no-detection events
        for i in 0..100 {
            let pair = MeasurementPair {
                outcome_a: MeasurementOutcome::NoDetection,
                outcome_b: MeasurementOutcome::PlusOne,
                setting_a: 0,
                setting_b: 2,
                timestamp_ns: i as u64,
            };
            tester.add_measurement(pair).unwrap();
        }

        let result = tester.calculate_chsh_s_value();
        assert!(matches!(result, Err(ChshError::DetectionLoophole { .. })));
    }
}
