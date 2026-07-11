//! Bio-Financial Arbitrage Engine
//! 
//! Cross-references genomic/epigenetic signals with mortality forecasts
//! to identify mispricing between biological reality and financial expectations.

use crate::genomics::elastic_net_prs::{PrsCalculator, RiskCategory};
use crate::epigenetics::horvath_clock_solver::{HorvathClockSolver, EpigeneticClockError};
use crate::mortality::lee_carter_kalman::{LeeCarterKalmanModel, MortalityModelError};
use crate::derivatives::affine_longevity_bond::{AffineLongevityBondPricer, LongevityDerivativeError};

/// Error types for bio-financial arbitrage
#[derive(Debug, Clone, PartialEq)]
pub enum BioFinArbError {
    SignalMismatch,
    DataUnavailable,
    ModelCalibrationFailure,
    ArbitrageComputationFailure,
}

impl core::fmt::Display for BioFinArbError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SignalMismatch => write!(f, "Signal mismatch"),
            Self::DataUnavailable => write!(f, "Data unavailable"),
            Self::ModelCalibrationFailure => write!(f, "Model calibration failure"),
            Self::ArbitrageComputationFailure => write!(f, "Arbitrage computation failure"),
        }
    }
}

/// Biological age signal from epigenetic data
#[derive(Debug, Clone)]
pub struct BiologicalAgeSignal {
    /// Chronological age
    pub chrono_age: f64,
    /// Biological age from epigenetic clock
    pub bio_age: f64,
    /// Age acceleration (bio - expected)
    pub age_acceleration: f64,
    /// Confidence in measurement (0-1)
    pub confidence: f64,
}

impl BiologicalAgeSignal {
    pub fn is_accelerated(&self, threshold: f64) -> bool {
        self.age_acceleration > threshold
    }
}

/// Mortality forecast signal from Lee-Carter model
#[derive(Debug, Clone)]
pub struct MortalityForecastSignal {
    /// Current kappa (mortality trend)
    pub kappa: f64,
    /// Forecasted mortality improvement rate
    pub improvement_rate: f64,
    /// Expected remaining life at age 65
    pub le_65: f64,
}

/// Trading signal for longevity derivatives
#[derive(Debug, Clone)]
pub struct LongevityArbSignal {
    /// Asset to trade (bond, swap, equity)
    pub asset_type: u8, // 1=bond, 2=swap, 3=equity
    /// Direction (true=long protection/short bond, false=opposite)
    pub long_protection: bool,
    /// Signal strength (0-1)
    pub strength: f64,
    /// Expected alpha (basis points)
    pub expected_alpha_bps: i32,
    /// Time horizon (days)
    pub horizon_days: u32,
    /// Rationale
    pub rationale: &'static str,
}

impl LongevityArbSignal {
    pub fn new(
        asset_type: u8,
        long_protection: bool,
        strength: f64,
        expected_alpha_bps: i32,
        horizon_days: u32,
        rationale: &'static str,
    ) -> Result<Self, BioFinArbError> {
        if asset_type == 0 || asset_type > 3 {
            return Err(BioFinArbError::SignalMismatch);
        }
        if strength < 0.0 || strength > 1.0 {
            return Err(BioFinArbError::SignalMismatch);
        }

        Ok(Self {
            asset_type,
            long_protection,
            strength,
            expected_alpha_bps,
            horizon_days,
            rationale,
        })
    }
}

/// Main bio-financial arbitrage engine
pub struct BioFinancialArbEngine {
    /// Horvath clock for biological age
    horvath_clock: HorvathClockSolver,
    /// Lee-Carter model for mortality trends
    mortality_model: LeeCarterKalmanModel,
    /// Longevity bond pricer
    bond_pricer: AffineLongevityBondPricer,
    /// Threshold for significant age acceleration
    acceleration_threshold: f64,
}

impl BioFinancialArbEngine {
    pub fn new() -> Self {
        Self {
            horvath_clock: HorvathClockSolver::new(),
            mortality_model: LeeCarterKalmanModel::new(),
            bond_pricer: AffineLongevityBondPricer::new(),
            acceleration_threshold: 2.0, // 2 years acceleration
        }
    }

    /// Initialize all models
    pub fn initialize(&mut self) -> Result<(), BioFinArbError> {
        self.horvath_clock.initialize()
            .map_err(|_| BioFinArbError::ModelCalibrationFailure)?;
        Ok(())
    }

    /// Process biological age signal and generate arbitrage signals
    pub fn process_bio_signal(
        &self,
        bio_signal: &BiologicalAgeSignal,
        market_signal: &MortalityForecastSignal,
    ) -> Result<Vec<LongevityArbSignal>, BioFinArbError> {
        let mut signals = Vec::new();

        // Check for significant age acceleration
        if bio_signal.is_accelerated(self.acceleration_threshold) && bio_signal.confidence > 0.8 {
            // Biological age accelerating faster than market expects
            
            // Signal 1: Short longevity bonds (people dying sooner)
            let bond_strength = (bio_signal.age_acceleration / 10.0).clamp(0.0, 1.0) * bio_signal.confidence;
            let bond_signal = LongevityArbSignal::new(
                1, // Bond
                true, // Long protection = short bond
                bond_strength,
                (bond_strength * 400.0) as i32,
                60,
                "Biological age acceleration exceeds actuarial assumptions",
            )?;
            signals.push(bond_signal);

            // Signal 2: Long mortality swaps
            let swap_strength = bond_strength * 0.9;
            let swap_signal = LongevityArbSignal::new(
                2, // Swap
                true, // Long protection
                swap_strength,
                (swap_strength * 350.0) as i32,
                90,
                "Mortality swap spread widening expected",
            )?;
            signals.push(swap_signal);

            // Signal 3: Short annuity insurers
            let insurer_strength = bond_strength * 0.7;
            let insurer_signal = LongevityArbSignal::new(
                3, // Equity
                false, // Short equity (liabilities increase)
                insurer_strength,
                -(insurer_strength * 250.0) as i32,
                120,
                "Annuity issuer liability mismatch",
            )?;
            signals.push(insurer_signal);
        } else if bio_signal.age_acceleration < -self.acceleration_threshold && bio_signal.confidence > 0.8 {
            // Biological age decelerating (longevity improving)
            
            // Signal 1: Long longevity bonds
            let decel = bio_signal.age_acceleration.abs();
            let bond_strength = (decel / 10.0).clamp(0.0, 1.0) * bio_signal.confidence;
            let bond_signal = LongevityArbSignal::new(
                1, // Bond
                false, // Long bond
                bond_strength,
                (bond_strength * 300.0) as i32,
                90,
                "Biological age deceleration - longevity improving",
            )?;
            signals.push(bond_signal);
        }

        // Cross-check with mortality forecast
        if market_signal.improvement_rate < -0.02 {
            // Mortality improvements slowing or reversing
            let reversal_strength = market_signal.improvement_rate.abs().clamp(0.0, 0.1) * 10.0;
            let reversal_signal = LongevityArbSignal::new(
                1,
                true,
                reversal_strength,
                (reversal_strength * 200.0) as i32,
                30,
                "Mortality improvement reversal detected",
            )?;
            signals.push(reversal_signal);
        }

        Ok(signals)
    }

    /// Compute mispricing between biological and actuarial ages
    pub fn compute_mispricing(
        &self,
        bio_age: f64,
        actuarial_age: f64,
    ) -> Result<f64, BioFinArbError> {
        let diff = bio_age - actuarial_age;
        
        if !diff.is_finite() {
            return Err(BioFinArbError::ArbitrageComputationFailure);
        }

        // Mispricing in basis points of notional
        let mispricing_bps = diff * 50.0; // 50 bps per year of difference
        
        Ok(mispricing_bps.clamp(-500.0, 500.0))
    }

    /// Aggregate signals across population cohort
    pub fn aggregate_cohort_signals(
        &self,
        bio_signals: &[BiologicalAgeSignal],
        market_signal: &MortalityForecastSignal,
    ) -> Result<Vec<LongevityArbSignal>, BioFinArbError> {
        if bio_signals.is_empty() {
            return Err(BioFinArbError::DataUnavailable);
        }

        // Compute cohort statistics
        let mean_acceleration: f64 = bio_signals.iter()
            .map(|s| s.age_acceleration * s.confidence)
            .sum::<f64>() / bio_signals.len() as f64;

        let n_accelerated = bio_signals.iter()
            .filter(|s| s.is_accelerated(self.acceleration_threshold))
            .count();

        let cohort_signal = BiologicalAgeSignal {
            chrono_age: bio_signals.iter().map(|s| s.chrono_age).sum::<f64>() / bio_signals.len() as f64,
            bio_age: bio_signals.iter().map(|s| s.bio_age).sum::<f64>() / bio_signals.len() as f64,
            age_acceleration: mean_acceleration,
            confidence: 0.9, // Higher confidence for cohort
        };

        let mut signals = self.process_bio_signal(&cohort_signal, market_signal)?;

        // Add cohort-specific signal if significant fraction accelerated
        let frac_accelerated = n_accelerated as f64 / bio_signals.len() as f64;
        if frac_accelerated > 0.3 {
            let cohort_strength = frac_accelerated * 0.8;
            let cohort_sig = LongevityArbSignal::new(
                2, // Swap
                true,
                cohort_strength,
                (cohort_strength * 500.0) as i32,
                180,
                "Significant cohort-wide age acceleration",
            )?;
            signals.push(cohort_sig);
        }

        Ok(signals)
    }

    /// Set acceleration threshold
    pub fn set_acceleration_threshold(&mut self, threshold: f64) {
        self.acceleration_threshold = threshold.max(0.5);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bio_signal_processing() {
        let mut engine = BioFinancialArbEngine::new();
        assert!(engine.initialize().is_ok());

        let bio_signal = BiologicalAgeSignal {
            chrono_age: 50.0,
            bio_age: 55.0,
            age_acceleration: 3.0,
            confidence: 0.9,
        };

        let market_signal = MortalityForecastSignal {
            kappa: -0.5,
            improvement_rate: 0.01,
            le_65: 20.0,
        };

        let signals = engine.process_bio_signal(&bio_signal, &market_signal);
        assert!(signals.is_ok());
        assert!(!signals.unwrap().is_empty());
    }

    #[test]
    fn test_mispricing_computation() {
        let engine = BioFinancialArbEngine::new();
        let mispricing = engine.compute_mispricing(55.0, 50.0).unwrap();
        assert!(mispricing > 0.0);
    }
}
