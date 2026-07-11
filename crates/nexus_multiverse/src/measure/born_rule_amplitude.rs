//! Born Rule Amplitude Calculator
//! 
//! Computes squared amplitudes (probabilities) from quantum state vectors
//! using the Born rule. Ensures numerical stability and proper normalization.

use alloc::vec::Vec;
use core::fmt;
use super::hilbert_space_mps::{ComplexAmplitude, MpsError};

/// Error types for Born rule calculations
#[derive(Debug, Clone, PartialEq)]
pub enum BornRuleError {
    NonNormalizedState { norm: f64 },
    NegativeProbability { value: f64 },
    ProbabilityExceedsOne { value: f64 },
    NumericalInstability { message: &'static str },
}

impl fmt::Display for BornRuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BornRuleError::NonNormalizedState { norm } => {
                write!(f, "Non-normalized state: norm={}", norm)
            }
            BornRuleError::NegativeProbability { value } => {
                write!(f, "Negative probability: {}", value)
            }
            BornRuleError::ProbabilityExceedsOne { value } => {
                write!(f, "Probability exceeds one: {}", value)
            }
            BornRuleError::NumericalInstability { message } => {
                write!(f, "Numerical instability: {}", message)
            }
        }
    }
}

/// Born rule calculator for quantum probabilities
pub struct BornRuleCalculator {
    /// Tolerance for normalization checks
    normalization_tolerance: f64,
}

impl BornRuleCalculator {
    pub const fn new() -> Self {
        Self {
            normalization_tolerance: 1e-12,
        }
    }

    /// Calculate probability from a complex amplitude using Born rule
    /// P = |ψ|² = ψ*ψ = re² + im²
    #[inline]
    pub fn calculate_probability(&self, amplitude: ComplexAmplitude) -> Result<f64, BornRuleError> {
        let probability = amplitude.magnitude_squared();

        // Check for numerical issues
        if probability.is_nan() {
            return Err(BornRuleError::NumericalInstability {
                message: "NaN in probability calculation",
            });
        }

        if probability.is_infinite() {
            return Err(BornRuleError::NumericalInstability {
                message: "Infinity in probability calculation",
            });
        }

        // Probability must be non-negative (guaranteed by magnitude_squared)
        if probability < 0.0 {
            return Err(BornRuleError::NegativeProbability {
                value: probability,
            });
        }

        Ok(probability)
    }

    /// Calculate probabilities for multiple amplitudes and verify normalization
    pub fn calculate_probabilities_batch(
        &self,
        amplitudes: &[ComplexAmplitude],
    ) -> Result<Vec<f64>, BornRuleError> {
        let mut probabilities = Vec::with_capacity(amplitudes.len());
        let mut total = 0.0_f64;

        for &amp in amplitudes {
            let prob = self.calculate_probability(amp)?;
            
            // Check running sum for overflow
            total = total.checked_add(prob).ok_or_else(|| {
                BornRuleError::NumericalInstability {
                    message: "Overflow in probability sum",
                }
            })?;

            probabilities.push(prob);
        }

        // Verify total probability is approximately 1.0
        let deviation = (total - 1.0).abs();
        if deviation > self.normalization_tolerance && total > self.normalization_tolerance {
            return Err(BornRuleError::NonNormalizedState { norm: total });
        }

        Ok(probabilities)
    }

    /// Normalize a vector of amplitudes to ensure total probability = 1.0
    pub fn normalize_amplitudes(
        &self,
        amplitudes: &mut [ComplexAmplitude],
    ) -> Result<(), BornRuleError> {
        // Calculate current norm
        let mut norm_squared = 0.0_f64;
        
        for &amp in amplitudes.iter() {
            norm_squared += amp.magnitude_squared();
        }

        if norm_squared < self.normalization_tolerance {
            return Err(BornRuleError::NumericalInstability {
                message: "Cannot normalize zero-norm state",
            });
        }

        let norm = norm_squared.sqrt();
        let scale_factor = 1.0 / norm;

        // Scale all amplitudes
        for amp in amplitudes.iter_mut() {
            *amp = ComplexAmplitude::new(
                amp.re * scale_factor,
                amp.im * scale_factor,
            );
        }

        // Verify normalization
        let mut new_norm_squared = 0.0_f64;
        for &amp in amplitudes.iter() {
            new_norm_squared += amp.magnitude_squared();
        }

        let new_norm = new_norm_squared.sqrt();
        let deviation = (new_norm - 1.0).abs();

        if deviation > self.normalization_tolerance * 10.0 {
            return Err(BornRuleError::NonNormalizedState { norm: new_norm });
        }

        Ok(())
    }

    /// Calculate interference term between two amplitudes
    /// Interference = 2 * Re(ψ₁* ψ₂)
    #[inline]
    pub fn calculate_interference(
        &self,
        amp1: ComplexAmplitude,
        amp2: ComplexAmplitude,
    ) -> f64 {
        let conjugate1 = amp1.conjugate();
        let product = conjugate1.mul(&amp2);
        2.0 * product.re
    }

    /// Calculate probability with interference effects
    /// P = |ψ₁ + ψ₂|² = |ψ₁|² + |ψ₂|² + 2*Re(ψ₁*ψ₂)
    pub fn calculate_probability_with_interference(
        &self,
        amp1: ComplexAmplitude,
        amp2: ComplexAmplitude,
    ) -> Result<f64, BornRuleError> {
        let sum = ComplexAmplitude::new(
            amp1.re + amp2.re,
            amp1.im + amp2.im,
        );

        self.calculate_probability(sum)
    }

    /// Verify probability axioms for a distribution
    pub fn verify_probability_axioms(&self, probabilities: &[f64]) -> Result<(), BornRuleError> {
        let mut total = 0.0_f64;

        for (i, &prob) in probabilities.iter().enumerate() {
            // Check non-negativity
            if prob < 0.0 {
                return Err(BornRuleError::NegativeProbability { value: prob });
            }

            // Check individual probabilities don't exceed 1
            if prob > 1.0 + self.normalization_tolerance {
                return Err(BornRuleError::ProbabilityExceedsOne { value: prob });
            }

            // Check for NaN or Inf
            if prob.is_nan() || prob.is_infinite() {
                return Err(BornRuleError::NumericalInstability {
                    message: "NaN or Inf in probability distribution",
                });
            }

            total += prob;
        }

        // Check normalization (sum ≈ 1.0)
        let deviation = (total - 1.0).abs();
        if deviation > self.normalization_tolerance && 
           total > self.normalization_tolerance &&
           probabilities.len() > 0 {
            return Err(BornRuleError::NonNormalizedState { norm: total });
        }

        Ok(())
    }
}

impl Default for BornRuleCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_amplitude_probability() {
        let calc = BornRuleCalculator::new();
        
        // |1⟩ state should have probability 1.0
        let amp = ComplexAmplitude::one();
        let prob = calc.calculate_probability(amp).unwrap();
        assert!((prob - 1.0).abs() < 1e-14);

        // |0⟩ state should have probability 0.0
        let amp = ComplexAmplitude::zero();
        let prob = calc.calculate_probability(amp).unwrap();
        assert!(prob.abs() < 1e-14);

        // Superposition (1/√2)(|0⟩ + |1⟩)
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let amp = ComplexAmplitude::new(inv_sqrt2, 0.0);
        let prob = calc.calculate_probability(amp).unwrap();
        assert!((prob - 0.5).abs() < 1e-14);
    }

    #[test]
    fn test_batch_normalization() {
        let calc = BornRuleCalculator::new();
        
        let mut amplitudes = vec![
            ComplexAmplitude::new(0.6, 0.0),
            ComplexAmplitude::new(0.8, 0.0),
        ];

        calc.normalize_amplitudes(&mut amplitudes).unwrap();

        // Verify normalized
        let probs = calc.calculate_probabilities_batch(&amplitudes).unwrap();
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < calc.normalization_tolerance);
    }

    #[test]
    fn test_interference() {
        let calc = BornRuleCalculator::new();
        
        let amp1 = ComplexAmplitude::new(0.5, 0.0);
        let amp2 = ComplexAmplitude::new(0.5, 0.0);

        // Without interference: P1 + P2 = 0.25 + 0.25 = 0.5
        // With constructive interference: |0.5 + 0.5|² = 1.0
        let prob_with_interference = calc
            .calculate_probability_with_interference(amp1, amp2)
            .unwrap();
        
        assert!((prob_with_interference - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_probability_axioms() {
        let calc = BornRuleCalculator::new();
        
        let valid_probs = vec![0.3, 0.5, 0.2];
        assert!(calc.verify_probability_axioms(&valid_probs).is_ok());

        let invalid_negative = vec![0.5, -0.1, 0.6];
        assert!(calc.verify_probability_axioms(&invalid_negative).is_err());

        let invalid_sum = vec![0.5, 0.6]; // Sum > 1
        assert!(calc.verify_probability_axioms(&invalid_sum).is_err());
    }
}
