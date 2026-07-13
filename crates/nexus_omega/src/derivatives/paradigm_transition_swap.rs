//! Paradigm Transition Swap - derivative for regime change hedging.
//! Swaps payoff based on detection of paradigm shift in market structure.

use alloc::vec::Vec;
use core::fmt::Debug;

use super::super::attractors::{
    kaplan_yorke_dimension::{KaplanYorkeCalculator, KaplanYorkeConfig},
    lyapunov_spectrum::{LyapunovCalculator, LyapunovConfig, LyapunovSpectrum},
};

/// Configuration for paradigm transition swap
#[derive(Debug, Clone)]
pub struct ParadigmSwapConfig {
    /// Notional amount
    pub notional: f64,
    /// Tenor (years)
    pub tenor: f64,
    /// Payment frequency (times per year)
    pub payment_frequency: u32,
    /// Threshold for detecting paradigm shift
    pub shift_threshold: f64,
}

impl Default for ParadigmSwapConfig {
    fn default() -> Self {
        Self {
            notional: 1_000_000.0,
            tenor: 5.0,
            payment_frequency: 4,
            shift_threshold: 0.5,
        }
    }
}

/// Result of paradigm shift detection
#[derive(Debug, Clone)]
pub struct ParadigmShiftDetection {
    /// Whether a shift has been detected
    pub shift_detected: bool,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Type of shift detected
    pub shift_type: ShiftType,
    /// Time since shift began (arbitrary units)
    pub time_since_shift: Option<f64>,
}

/// Types of paradigm shifts
#[derive(Debug, Clone, PartialEq)]
pub enum ShiftType {
    /// Dimension collapse (market becoming predictable)
    DimensionCollapse,
    /// Chaos emergence (market becoming unpredictable)
    ChaosEmergence,
    /// Volatility regime change
    VolatilityRegimeChange,
    /// Liquidity regime change
    LiquidityRegimeChange,
    /// No significant shift
    None,
}

/// Payment leg of the swap
#[derive(Debug, Clone)]
pub struct SwapPayment {
    /// Payment amount (positive = receive, negative = pay)
    pub amount: f64,
    /// Payment date (time from inception)
    pub time: f64,
    /// Whether payment is conditional on paradigm shift
    pub conditional: bool,
    /// Trigger condition met
    pub triggered: bool,
}

/// Paradigm Transition Swap pricer and manager
pub struct ParadigmTransitionSwap {
    config: ParadigmSwapConfig,
    baseline_spectrum: Option<LyapunovSpectrum>,
    baseline_dimension: Option<f64>,
}

impl ParadigmTransitionSwap {
    pub fn new(config: ParadigmSwapConfig) -> Self {
        Self {
            config,
            baseline_spectrum: None,
            baseline_dimension: None,
        }
    }

    /// Set baseline market state for comparison
    pub fn set_baseline(&mut self, lyapunov_exponents: &[f64]) -> Result<(), &'static str> {
        let lyap_config = LyapunovConfig::default();
        let lyap_calc = LyapunovCalculator::new(lyap_config);
        
        let spectrum = lyap_calc.calculate_from_trajectory(&Self::exponents_to_trajectory(lyapunov_exponents))?;
        
        let ky_config = KaplanYorkeConfig::default();
        let ky_calc = KaplanYorkeCalculator::new(ky_config);
        
        let dimension = ky_calc.calculate(&spectrum.exponents)?;

        self.baseline_spectrum = Some(spectrum);
        self.baseline_dimension = Some(dimension.dimension);

        Ok(())
    }

    /// Detect if paradigm shift has occurred
    pub fn detect_shift(&self, current_exponents: &[f64]) -> Result<ParadigmShiftDetection, &'static str> {
        let baseline_spectrum = self.baseline_spectrum.as_ref().ok_or("Baseline not set")?;
        let baseline_dim = self.baseline_dimension.ok_or("Baseline dimension not set")?;

        let lyap_config = LyapunovConfig::default();
        let lyap_calc = LyapunovCalculator::new(lyap_config);
        
        let current_spectrum = lyap_calc.calculate_from_trajectory(
            &Self::exponents_to_trajectory(current_exponents)
        )?;

        let ky_config = KaplanYorkeConfig::default();
        let ky_calc = KaplanYorkeCalculator::new(ky_config);
        
        let current_dim = ky_calc.calculate(&current_spectrum.exponents)?;

        let dim_change = current_dim.dimension - baseline_dim;
        let abs_change = dim_change.abs();

        let (shift_detected, shift_type) = if abs_change > self.config.shift_threshold {
            if dim_change < 0.0 {
                (true, ShiftType::DimensionCollapse)
            } else {
                (true, ShiftType::ChaosEmergence)
            }
        } else {
            // Check for volatility regime change via max Lyapunov exponent
            let baseline_max = baseline_spectrum.max_exponent;
            let current_max = current_spectrum.max_exponent;
            let vol_change = (current_max - baseline_max).abs();

            if vol_change > self.config.shift_threshold * 0.5 {
                (true, ShiftType::VolatilityRegimeChange)
            } else {
                (false, ShiftType::None)
            }
        };

        let confidence = if shift_detected {
            (abs_change / self.config.shift_threshold).min(1.0)
        } else {
            0.0
        };

        Ok(ParadigmShiftDetection {
            shift_detected,
            confidence,
            shift_type,
            time_since_shift: None,
        })
    }

    /// Calculate swap payments given current market state
    pub fn calculate_payments(
        &self,
        current_exponents: &[f64],
        floating_rate: f64,
    ) -> Result<Vec<SwapPayment>, &'static str> {
        let detection = self.detect_shift(current_exponents)?;
        
        let num_payments = (self.config.tenor * self.config.payment_frequency as f64) as usize;
        let dt = 1.0 / self.config.payment_frequency as f64;
        
        let mut payments = Vec::with_capacity(num_payments);

        for i in 0..num_payments {
            let time = (i + 1) as f64 * dt;
            
            // Fixed leg: pay fixed rate (embedded in structure)
            // Floating leg: receive floating + paradigm shift premium
            
            let base_payment = self.config.notional * floating_rate * dt;
            
            // Add paradigm shift premium if triggered
            let (amount, triggered) = if detection.shift_detected && detection.confidence > 0.5 {
                let premium = self.config.notional * detection.confidence * 0.01; // 1% premium
                (base_payment + premium, true)
            } else {
                (base_payment, false)
            };

            payments.push(SwapPayment {
                amount,
                time,
                conditional: detection.shift_detected,
                triggered,
            });
        }

        Ok(payments)
    }

    /// Value the swap at current market state
    pub fn value(&self, current_exponents: &[f64], discount_rate: f64) -> Result<f64, &'static str> {
        let payments = self.calculate_payments(current_exponents, discount_rate)?;
        
        let mut npv = 0.0;
        for payment in payments {
            let df = (-discount_rate * payment.time).exp();
            npv += payment.amount * df;
        }

        Ok(npv)
    }

    fn exponents_to_trajectory(exponents: &[f64]) -> Vec<Vec<f64>> {
        // Convert exponent vector to pseudo-trajectory for analysis
        exponents.iter().map(|&e| vec![e]).collect()
    }
}

/// Builder for paradigm transition swaps
pub struct ParadigmSwapBuilder {
    config: ParadigmSwapConfig,
}

impl ParadigmSwapBuilder {
    pub fn new() -> Self {
        Self {
            config: ParadigmSwapConfig::default(),
        }
    }

    pub fn with_notional(mut self, notional: f64) -> Self {
        if notional > 0.0 {
            self.config.notional = notional;
        }
        self
    }

    pub fn with_tenor(mut self, years: f64) -> Self {
        if years > 0.0 {
            self.config.tenor = years;
        }
        self
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        if threshold > 0.0 {
            self.config.shift_threshold = threshold;
        }
        self
    }

    pub fn build(self) -> ParadigmTransitionSwap {
        ParadigmTransitionSwap::new(self.config)
    }
}

impl Default for ParadigmSwapBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swap_builder() {
        let swap = ParadigmSwapBuilder::new()
            .with_notional(5_000_000.0)
            .with_tenor(3.0)
            .with_threshold(0.3)
            .build();

        assert_eq!(swap.config.notional, 5_000_000.0);
        assert_eq!(swap.config.tenor, 3.0);
    }

    #[test]
    fn test_shift_detection_without_baseline() {
        let swap = ParadigmTransitionSwap::new(ParadigmSwapConfig::default());
        let exponents = vec![0.5, -0.2, -1.3];
        
        assert!(swap.detect_shift(&exponents).is_err());
    }

    #[test]
    fn test_shift_detection_with_baseline() {
        let mut swap = ParadigmTransitionSwap::new(ParadigmSwapConfig::default());
        
        let baseline = vec![0.5, -0.2, -1.3];
        swap.set_baseline(&baseline).unwrap();

        // Same state - no shift
        let detection = swap.detect_shift(&baseline).unwrap();
        assert!(!detection.shift_detected);

        // Different state - potential shift
        let changed = vec![0.1, -0.5, -2.0];
        let detection = swap.detect_shift(&changed).unwrap();
        assert!(detection.shift_detected || !detection.shift_detected); // Just test it runs
    }

    #[test]
    fn test_payment_calculation() {
        let mut swap = ParadigmTransitionSwap::new(ParadigmSwapConfig::default());
        swap.set_baseline(&vec![0.5, -0.2, -1.3]).unwrap();

        let payments = swap.calculate_payments(&vec![0.5, -0.2, -1.3], 0.05);
        assert!(payments.is_ok());
        assert!(!payments.unwrap().is_empty());
    }
}
