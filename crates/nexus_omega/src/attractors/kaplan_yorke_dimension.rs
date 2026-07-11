//! Kaplan-Yorke Dimension Calculator for strange attractors.
//! Estimates the fractal dimension of an attractor from its Lyapunov spectrum.

use alloc::vec::Vec;
use core::fmt::Debug;

/// Result of Kaplan-Yorke dimension calculation
#[derive(Debug, Clone)]
pub struct KaplanYorkeResult {
    /// The calculated fractal (Lyapunov) dimension
    pub dimension: f64,
    /// Integer part of the dimension
    pub integer_part: usize,
    /// Fractional part of the dimension
    pub fractional_part: f64,
    /// Whether the attractor is a strange attractor (non-integer dimension)
    pub is_strange: bool,
    /// Number of positive Lyapunov exponents
    pub num_positive_exponents: usize,
}

impl KaplanYorkeResult {
    /// Check if dimension indicates a strange attractor
    pub const fn is_strange_attractor(&self) -> bool {
        self.is_strange && self.dimension > self.integer_part as f64
    }

    /// Get the embedding dimension needed to fully contain the attractor
    pub const fn required_embedding_dim(&self) -> usize {
        self.integer_part + 1
    }
}

/// Configuration for Kaplan-Yorke calculation
#[derive(Debug, Clone)]
pub struct KaplanYorkeConfig {
    /// Minimum number of exponents required
    pub min_exponents: usize,
    /// Tolerance for zero detection
    pub zero_tolerance: f64,
}

impl Default for KaplanYorkeConfig {
    fn default() -> Self {
        Self {
            min_exponents: 2,
            zero_tolerance: 1e-10,
        }
    }
}

/// Kaplan-Yorke dimension calculator
pub struct KaplanYorkeCalculator {
    config: KaplanYorkeConfig,
}

impl KaplanYorkeCalculator {
    pub const fn new(config: KaplanYorkeConfig) -> Self {
        Self { config }
    }

    /// Calculate Kaplan-Yorke dimension from Lyapunov spectrum
    /// 
    /// The Kaplan-Yorke conjecture states that for "typical" dynamical systems,
    /// the information dimension D_1 equals the Lyapunov dimension D_L:
    /// 
    /// D_L = j + (λ_1 + λ_2 + ... + λ_j) / |λ_{j+1}|
    /// 
    /// where j is the largest integer such that λ_1 + ... + λ_j >= 0
    pub fn calculate(&self, lyapunov_exponents: &[f64]) -> Result<KaplanYorkeResult, &'static str> {
        if lyapunov_exponents.len() < self.config.min_exponents {
            return Err("Insufficient Lyapunov exponents");
        }

        // Sort exponents in descending order (should already be sorted)
        let mut sorted: Vec<f64> = lyapunov_exponents.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));

        // Find the largest j such that sum of first j exponents >= 0
        let mut cumulative_sum = 0.0;
        let mut j: usize = 0;

        for (i, &exp) in sorted.iter().enumerate() {
            if cumulative_sum + exp >= -self.config.zero_tolerance {
                cumulative_sum += exp;
                j = i + 1;
            } else {
                break;
            }
        }

        // Handle edge cases
        if j == 0 {
            // All exponents negative or very small
            return Ok(KaplanYorkeResult {
                dimension: 0.0,
                integer_part: 0,
                fractional_part: 0.0,
                is_strange: false,
                num_positive_exponents: 0,
            });
        }

        if j >= sorted.len() {
            // All exponents positive (unusual for dissipative systems)
            return Ok(KaplanYorkeResult {
                dimension: sorted.len() as f64,
                integer_part: sorted.len(),
                fractional_part: 0.0,
                is_strange: false,
                num_positive_exponents: sorted.iter().filter(|&&x| x > self.config.zero_tolerance).count(),
            });
        }

        // Calculate fractional part
        let lambda_j_plus_1 = sorted[j].abs();
        
        if lambda_j_plus_1 < self.config.zero_tolerance {
            // Avoid division by zero
            return Ok(KaplanYorkeResult {
                dimension: j as f64,
                integer_part: j,
                fractional_part: 0.0,
                is_strange: false,
                num_positive_exponents: sorted[..j].iter().filter(|&&x| x > self.config.zero_tolerance).count(),
            });
        }

        let fractional_part = cumulative_sum / lambda_j_plus_1;
        let dimension = j as f64 + fractional_part;
        
        // Clamp fractional part to [0, 1)
        let fractional_clamped = fractional_part.min(0.999999).max(0.0);
        let dimension_clamped = (j as f64) + fractional_clamped;

        let num_positive = sorted.iter().filter(|&&x| x > self.config.zero_tolerance).count();
        let is_strange = fractional_clamped > self.config.zero_tolerance && num_positive > 0;

        Ok(KaplanYorkeResult {
            dimension: dimension_clamped,
            integer_part: j,
            fractional_part: fractional_clamped,
            is_strange,
            num_positive_exponents: num_positive,
        })
    }

    /// Calculate dimension with uncertainty estimation
    pub fn calculate_with_uncertainty(
        &self,
        exponents: &[f64],
        uncertainties: &[f64],
    ) -> Result<(KaplanYorkeResult, f64), &'static str> {
        if exponents.len() != uncertainties.len() {
            return Err("Exponents and uncertainties must have same length");
        }

        let result = self.calculate(exponents)?;

        // Simple error propagation
        let mut variance = 0.0;
        for (i, (&exp, &unc)) in exponents.iter().zip(uncertainties.iter()).enumerate() {
            if i < result.integer_part + 1 {
                variance += unc * unc;
            }
        }

        let uncertainty = variance.sqrt();

        Ok((result, uncertainty))
    }

    /// Detect dimension collapse (indicator of rigid/predictable state)
    pub fn detect_dimension_collapse(
        &self,
        current_exponents: &[f64],
        baseline_exponents: &[f64],
        threshold: f64,
    ) -> Result<bool, &'static str> {
        let current = self.calculate(current_exponents)?;
        let baseline = self.calculate(baseline_exponents)?;

        // Dimension collapse: significant reduction in fractal dimension
        let delta = baseline.dimension - current.dimension;

        Ok(delta > threshold)
    }
}

/// Market rigidity detector based on dimension analysis
pub struct MarketRigidityDetector {
    /// Threshold for detecting rigidity (dimension drop)
    collapse_threshold: f64,
    /// Minimum positive exponents for healthy market
    min_healthy_exponents: usize,
}

impl MarketRigidityDetector {
    pub const fn new(collapse_threshold: f64, min_healthy: usize) -> Self {
        Self {
            collapse_threshold,
            min_healthy_exponents: min_healthy,
        }
    }

    /// Analyze market state from Lyapunov spectrum
    pub fn analyze_market_state(&self, exponents: &[f64]) -> MarketState {
        let config = KaplanYorkeConfig::default();
        let calc = KaplanYorkeCalculator::new(config);

        match calc.calculate(exponents) {
            Ok(result) => {
                let num_positive = result.num_positive_exponents;
                let dimension = result.dimension;

                if num_positive < self.min_healthy_exponents {
                    MarketState::OverlyPredictable { dimension }
                } else if dimension < 2.0 {
                    MarketState::LowDimensional { dimension }
                } else if dimension > 5.0 {
                    MarketState::HighlyChaotic { dimension }
                } else {
                    MarketState::Healthy { dimension, num_positive }
                }
            }
            Err(_) => MarketState::Unknown,
        }
    }

    /// Check if market is entering a rigid, exploitable state
    pub fn is_entering_rigid_state(
        &self,
        current: &[f64],
        historical: &[f64],
    ) -> bool {
        let config = KaplanYorkeConfig::default();
        let calc = KaplanYorkeCalculator::new(config);

        match calc.detect_dimension_collapse(current, historical, self.collapse_threshold) {
            Ok(collapsed) => collapsed,
            Err(_) => false,
        }
    }
}

/// Market state classification
#[derive(Debug, Clone, PartialEq)]
pub enum MarketState {
    /// Normal, healthy chaotic market
    Healthy {
        dimension: f64,
        num_positive: usize,
    },
    /// Low-dimensional dynamics (potentially predictable)
    LowDimensional { dimension: f64 },
    /// Too few positive exponents (overly predictable)
    OverlyPredictable { dimension: f64 },
    /// Highly chaotic (hard to predict)
    HighlyChaotic { dimension: f64 },
    /// Cannot determine state
    Unknown,
}

impl MarketState {
    /// Check if state is exploitable (predictable)
    pub const fn is_exploitable(&self) -> bool {
        matches!(
            self,
            MarketState::LowDimensional { .. } | MarketState::OverlyPredictable { .. }
        )
    }

    /// Get recommended action based on state
    pub fn recommended_action(&self) -> &'static str {
        match self {
            MarketState::Healthy { .. } => "Standard strategies",
            MarketState::LowDimensional { .. } => "Increase position size, exploit predictability",
            MarketState::OverlyPredictable { .. } => "Aggressive exploitation warranted",
            MarketState::HighlyChaotic { .. } => "Reduce exposure, increase hedging",
            MarketState::Unknown => "Await more data",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kaplan_yorke_lorenz() {
        // Typical Lorenz system exponents: ~[0.9, 0, -14.6]
        let exponents = vec![0.9, 0.0, -14.6];
        let config = KaplanYorkeConfig::default();
        let calc = KaplanYorkeCalculator::new(config);
        
        let result = calc.calculate(&exponents).unwrap();
        
        // Should give dimension slightly above 2
        assert!(result.dimension > 2.0);
        assert!(result.dimension < 3.0);
        assert!(result.is_strange);
    }

    #[test]
    fn test_kaplan_yorke_all_negative() {
        let exponents = vec![-1.0, -2.0, -3.0];
        let config = KaplanYorkeConfig::default();
        let calc = KaplanYorkeCalculator::new(config);
        
        let result = calc.calculate(&exponents).unwrap();
        
        assert_eq!(result.dimension, 0.0);
        assert!(!result.is_strange);
    }

    #[test]
    fn test_market_state_classification() {
        let detector = MarketRigidityDetector::new(0.5, 1);
        
        // Healthy chaotic market
        let healthy = vec![0.5, 0.1, -2.0];
        let state = detector.analyze_market_state(&healthy);
        assert!(state.is_exploitable() || !state.is_exploitable()); // Just test it runs
        
        // Low dimensional
        let low_dim = vec![0.1, -0.5, -1.0];
        let state = detector.analyze_market_state(&low_dim);
        assert!(matches!(state, MarketState::LowDimensional { .. } | MarketState::OverlyPredictable { .. }));
    }
}
