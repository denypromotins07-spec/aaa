//! Senolytic Trial Arbitrage Module
//! 
//! Detects positive senolytic drug trial results and generates trading signals
//! to buy pharma equities and short annuity/long-term care insurers.

use crate::epigenetics::horvath_clock_solver::{HorvathClockSolver, EpigeneticClockError};
use crate::mortality::lee_carter_kalman::{LeeCarterKalmanModel, MortalityModelError};

/// Error types for senolytic arbitrage
#[derive(Debug, Clone, PartialEq)]
pub enum SenolyticArbError {
    InvalidTrialData,
    StatisticalInsigificance,
    SignalGenerationFailure,
    MarketDataUnavailable,
}

impl core::fmt::Display for SenolyticArbError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidTrialData => write!(f, "Invalid clinical trial data"),
            Self::StatisticalInsigificance => write!(f, "Results not statistically significant"),
            Self::SignalGenerationFailure => write!(f, "Failed to generate trading signal"),
            Self::MarketDataUnavailable => write!(f, "Market data unavailable"),
        }
    }
}

/// Clinical trial result for senolytic drug
#[derive(Debug, Clone)]
pub struct SenolyticTrialResult {
    /// Drug identifier
    pub drug_id: u64,
    /// Sponsor company ID
    pub sponsor_id: u64,
    /// Phase (1, 2, 3)
    pub phase: u8,
    /// Primary endpoint met
    pub endpoint_met: bool,
    /// P-value for primary endpoint
    pub p_value: f64,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
    /// Change in epigenetic age acceleration (years)
    pub delta_age_acceleration: f64,
    /// Sample size
    pub n_subjects: usize,
}

impl SenolyticTrialResult {
    /// Check if result is statistically significant
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha && self.endpoint_met
    }

    /// Check if effect size is clinically meaningful
    pub fn is_clinically_meaningful(&self, min_effect: f64) -> bool {
        self.effect_size.abs() > min_effect
    }
}

/// Trading signal generated from trial results
#[derive(Debug, Clone)]
pub struct LongevityTradingSignal {
    /// Target asset (pharma stock or insurer)
    pub asset_id: u64,
    /// Signal direction (true=long, false=short)
    pub is_long: bool,
    /// Signal strength (0-1)
    pub strength: f64,
    /// Expected return (basis points)
    pub expected_return_bps: i32,
    /// Time horizon (days)
    pub horizon_days: u32,
}

impl LongevityTradingSignal {
    pub fn new(
        asset_id: u64,
        is_long: bool,
        strength: f64,
        expected_return_bps: i32,
        horizon_days: u32,
    ) -> Result<Self, SenolyticArbError> {
        if strength < 0.0 || strength > 1.0 {
            return Err(SenolyticArbError::SignalGenerationFailure);
        }

        Ok(Self {
            asset_id,
            is_long,
            strength,
            expected_return_bps,
            horizon_days,
        })
    }
}

/// Senolytic trial arbitrage engine
pub struct SenolyticTrialArb {
    /// Horvath clock for evaluating age acceleration changes
    horvath_clock: HorvathClockSolver,
    /// Lee-Carter model for mortality impact
    mortality_model: LeeCarterKalmanModel,
    /// Significance threshold
    alpha: f64,
    /// Minimum clinically meaningful effect size
    min_effect_size: f64,
}

impl SenolyticTrialArb {
    pub fn new() -> Self {
        Self {
            horvath_clock: HorvathClockSolver::new(),
            mortality_model: LeeCarterKalmanModel::new(),
            alpha: 0.05,
            min_effect_size: 0.5,
        }
    }

    /// Initialize models
    pub fn initialize(&mut self) -> Result<(), EpigeneticClockError> {
        self.horvath_clock.initialize()?;
        Ok(())
    }

    /// Process clinical trial result and generate trading signals
    pub fn process_trial_result(
        &self,
        result: &SenolyticTrialResult,
    ) -> Result<Vec<LongevityTradingSignal>, SenolyticArbError> {
        // Validate statistical significance
        if !result.is_significant(self.alpha) {
            return Err(SenolyticArbError::StatisticalInsigificance);
        }

        // Validate clinical meaningfulness
        if !result.is_clinically_meaningful(self.min_effect_size) {
            return Err(SenolyticArbError::StatisticalInsigificance);
        }

        let mut signals = Vec::new();

        // Signal 1: Long pharma sponsor equity
        let pharma_strength = self.compute_pharma_signal_strength(result);
        let pharma_signal = LongevityTradingSignal::new(
            result.sponsor_id,
            true, // Long
            pharma_strength,
            (pharma_strength * 500.0) as i32, // Up to 500 bps expected return
            30, // 30-day horizon
        )?;
        signals.push(pharma_signal);

        // Signal 2: Short annuity insurers
        let annuity_impact = self.compute_annuity_liability_impact(result);
        if annuity_impact > 0.01 {
            // If liability increase > 1%
            let short_strength = annuity_impact.clamp(0.0, 1.0);
            let annuity_signal = LongevityTradingSignal::new(
                result.sponsor_id + 1000, // Placeholder for insurer ID
                false, // Short
                short_strength,
                -(short_strength * 300.0) as i32, // Negative expected return
                90, // 90-day horizon
            )?;
            signals.push(annuity_signal);
        }

        // Signal 3: Long longevity ETFs
        let etf_strength = pharma_strength * 0.7;
        let etf_signal = LongevityTradingSignal::new(
            9999, // Placeholder ETF ID
            true, // Long
            etf_strength,
            (etf_strength * 200.0) as i32,
            60,
        )?;
        signals.push(etf_signal);

        Ok(signals)
    }

    /// Compute signal strength for pharma equity
    fn compute_pharma_signal_strength(&self, result: &SenolyticTrialResult) -> f64 {
        let mut strength = 0.0;

        // Phase weighting
        let phase_weight = match result.phase {
            1 => 0.3,
            2 => 0.5,
            3 => 1.0,
            _ => 0.1,
        };

        // Statistical significance component
        let stat_component = (1.0 - result.p_value / self.alpha).clamp(0.0, 1.0);

        // Effect size component
        let effect_component = (result.effect_size / 2.0).clamp(0.0, 1.0);

        // Age acceleration component
        let age_component = if result.delta_age_acceleration < 0.0 {
            // Negative delta = good (age deceleration)
            ((result.delta_age_acceleration.abs() / 5.0).clamp(0.0, 1.0))
        } else {
            0.0
        };

        strength = phase_weight * (0.4 * stat_component + 0.4 * effect_component + 0.2 * age_component);
        strength.clamp(0.0, 1.0)
    }

    /// Compute impact on annuity liabilities
    fn compute_annuity_liability_impact(&self, result: &SenolyticTrialResult) -> f64 {
        // Simplified calculation: years of life extension * discount factor
        let life_extension_years = result.delta_age_acceleration.abs();
        
        // Discount for probability of approval
        let approval_prob = match result.phase {
            1 => 0.1,
            2 => 0.3,
            3 => 0.7,
            _ => 0.05,
        };

        // Liability impact = life extension * approval prob * population factor
        let population_factor = result.n_subjects as f64 / 1000.0;
        
        life_extension_years * approval_prob * population_factor.clamp(0.1, 10.0) * 0.01
    }

    /// Aggregate signals across multiple trials
    pub fn aggregate_signals(
        &self,
        results: &[SenolyticTrialResult],
    ) -> Vec<LongevityTradingSignal> {
        let mut all_signals = Vec::new();

        for result in results {
            if let Ok(signals) = self.process_trial_result(result) {
                all_signals.extend(signals);
            }
        }

        // Aggregate by asset
        self.consolidate_signals(all_signals)
    }

    /// Consolidate duplicate signals for same asset
    fn consolidate_signals(&self, signals: Vec<LongevityTradingSignal>) -> Vec<LongevityTradingSignal> {
        use std::collections::BTreeMap;
        
        let mut by_asset: BTreeMap<u64, Vec<&LongevityTradingSignal>> = BTreeMap::new();
        
        for signal in &signals {
            by_asset.entry(signal.asset_id).or_default().push(signal);
        }

        let mut consolidated = Vec::new();
        
        for (_asset_id, asset_signals) in by_asset {
            if asset_signals.is_empty() {
                continue;
            }

            // Weighted average of signals
            let total_weight: f64 = asset_signals.iter().map(|s| s.strength).sum();
            let weighted_return: f64 = asset_signals.iter()
                .map(|s| s.expected_return_bps as f64 * s.strength)
                .sum::<f64>() / total_weight.max(1e-10);

            let avg_direction = if asset_signals[0].is_long { 1.0 } else { -1.0 };
            
            if let Ok(signal) = LongevityTradingSignal::new(
                asset_signals[0].asset_id,
                avg_direction > 0.0,
                total_weight.clamp(0.0, 1.0),
                (weighted_return * avg_direction.signum()) as i32,
                asset_signals.iter().map(|s| s.horizon_days).sum::<u32>() / asset_signals.len() as u32,
            ) {
                consolidated.push(signal);
            }
        }

        consolidated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trial_significance() {
        let result = SenolyticTrialResult {
            drug_id: 1,
            sponsor_id: 100,
            phase: 3,
            endpoint_met: true,
            p_value: 0.001,
            effect_size: 1.2,
            delta_age_acceleration: -2.5,
            n_subjects: 500,
        };

        assert!(result.is_significant(0.05));
        assert!(result.is_clinically_meaningful(0.5));
    }

    #[test]
    fn test_arb_engine_initialization() {
        let mut arb = SenolyticTrialArb::new();
        assert!(arb.initialize().is_ok());
    }
}
