//! Mortality Swap Pricer
//! 
//! Prices mortality swaps where one party pays a fixed rate
//! and receives payments linked to realized mortality rates.

use crate::derivatives::affine_longevity_bond::{AffineLongevityBondPricer, LongevityDerivativeError};

/// Error types for mortality swap pricing
#[derive(Debug, Clone, PartialEq)]
pub enum MortalitySwapError {
    InvalidNotional,
    InvalidTenor,
    InvalidFixedRate,
    NumericalInstability,
    CounterpartyRiskError,
}

impl core::fmt::Display for MortalitySwapError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidNotional => write!(f, "Invalid notional amount"),
            Self::InvalidTenor => write!(f, "Invalid tenor"),
            Self::InvalidFixedRate => write!(f, "Invalid fixed rate"),
            Self::NumericalInstability => write!(f, "Numerical instability"),
            Self::CounterpartyRiskError => write!(f, "Counterparty risk calculation error"),
        }
    }
}

/// Mortality swap terms
#[derive(Debug, Clone)]
pub struct MortalitySwapTerms {
    /// Notional amount (in currency units)
    pub notional: f64,
    /// Tenor in years
    pub tenor_years: u32,
    /// Fixed rate (annual, as decimal)
    pub fixed_rate: f64,
    /// Reference population (e.g., age 65 UK males)
    pub reference_population: u32,
    /// Payment frequency (1=annual, 2=semi-annual, 4=quarterly)
    pub payment_frequency: u32,
}

impl MortalitySwapTerms {
    pub fn validate(&self) -> Result<(), MortalitySwapError> {
        if self.notional <= 0.0 || self.notional > 1e12 {
            return Err(MortalitySwapError::InvalidNotional);
        }
        if self.tenor_years == 0 || self.tenor_years > 50 {
            return Err(MortalitySwapError::InvalidTenor);
        }
        if self.fixed_rate < 0.0 || self.fixed_rate > 1.0 {
            return Err(MortalitySwapError::InvalidFixedRate);
        }
        Ok(())
    }

    #[inline]
    pub fn n_payments(&self) -> u32 {
        self.tenor_years * self.payment_frequency
    }
}

/// Mortality swap valuation result
#[derive(Debug, Clone)]
pub struct MortalitySwapValuation {
    /// Present value of fixed leg
    pub fixed_leg_pv: f64,
    /// Present value of floating leg
    pub floating_leg_pv: f64,
    /// Net present value (floating - fixed)
    pub npv: f64,
    /// Par swap rate (fair value)
    pub par_rate: f64,
    /// DV01 (sensitivity to 1bp rate change)
    pub dv01: f64,
    /// Expected exposure at maturity
    pub expected_exposure: f64,
}

impl MortalitySwapValuation {
    pub const fn new() -> Self {
        Self {
            fixed_leg_pv: 0.0,
            floating_leg_pv: 0.0,
            npv: 0.0,
            par_rate: 0.0,
            dv01: 0.0,
            expected_exposure: 0.0,
        }
    }
}

/// Mortality swap pricer
pub struct MortalitySwapPricer {
    bond_pricer: AffineLongevityBondPricer,
    /// Hazard rate curve
    hazard_rates: Box<[f64]>,
    /// Discount curve (risk-free)
    discount_factors: Box<[f64]>,
    /// Counterparty spread
    counterparty_spread: f64,
}

impl MortalitySwapPricer {
    pub fn new() -> Self {
        Self {
            bond_pricer: AffineLongevityBondPricer::new(),
            hazard_rates: vec![0.01; 51].into_boxed_slice(), // 50 years of hazard rates
            discount_factors: vec![1.0; 51].into_boxed_slice(),
            counterparty_spread: 0.001, // 10 bps
        }
    }

    /// Initialize hazard rates from mortality model
    pub fn initialize_hazard_rates(&mut self, initial_lambda: f64, improvement_rate: f64) {
        let mut lambda = initial_lambda;
        for i in 0..self.hazard_rates.len() {
            self.hazard_rates[i] = lambda;
            // Apply improvement (lambda decreases over time)
            lambda *= (1.0 - improvement_rate);
            lambda = lambda.max(0.0001).min(0.5);
        }

        // Build discount factors
        let r = 0.03; // Risk-free rate
        for i in 0..self.discount_factors.len() {
            self.discount_factors[i] = (-r * i as f64).exp();
        }
    }

    /// Price mortality swap
    pub fn price_swap(&self, terms: &MortalitySwapTerms) -> Result<MortalitySwapValuation, MortalitySwapError> {
        terms.validate()?;

        let mut val = MortalitySwapValuation::new();

        let dt = 1.0 / terms.payment_frequency as f64;
        let n_payments = terms.n_payments() as usize;

        // Fixed leg PV
        let mut fixed_pv = 0.0;
        for i in 1..=n_payments {
            let t = i as f64 * dt;
            let df = self.get_discount_factor(t);
            let survival = self.get_survival_probability(t);
            
            if !df.is_finite() || !survival.is_finite() {
                return Err(MortalitySwapError::NumericalInstability);
            }

            fixed_pv += terms.fixed_rate * dt * df * survival;
        }
        val.fixed_leg_pv = fixed_pv * terms.notional;

        // Floating leg PV (expected mortality payments)
        let mut float_pv = 0.0;
        for i in 1..=n_payments {
            let t = i as f64 * dt;
            let t_prev = (i - 1) as f64 * dt;
            
            let df = self.get_discount_factor(t);
            let survival_t = self.get_survival_probability(t);
            let survival_prev = self.get_survival_probability(t_prev);
            
            // Expected mortality payment = probability of death in period
            let death_prob = survival_prev - survival_t;
            
            if !df.is_finite() || !death_prob.is_finite() {
                return Err(MortalitySwapError::NumericalInstability);
            }

            float_pv += death_prob * df;
        }
        val.floating_leg_pv = float_pv * terms.notional;

        // NPV (from floating receiver perspective)
        val.npv = val.floating_leg_pv - val.fixed_leg_pv;

        // Par swap rate
        let annuity = self.compute_annuity(terms.tenor_years as f64, terms.payment_frequency);
        if annuity > 1e-10 {
            val.par_rate = float_pv / annuity;
        } else {
            val.par_rate = terms.fixed_rate;
        }

        // DV01 (sensitivity to 1bp parallel shift in rates)
        val.dv01 = self.compute_dv01(terms)?;

        // Expected exposure
        val.expected_exposure = self.compute_expected_exposure(terms)?;

        Ok(val)
    }

    /// Get discount factor at time t
    #[inline]
    fn get_discount_factor(&self, t: f64) -> f64 {
        let idx = t.min(50.0) as usize;
        let frac = t - idx as f64;
        
        if idx >= self.discount_factors.len() - 1 {
            return self.discount_factors[self.discount_factors.len() - 1];
        }

        // Linear interpolation
        self.discount_factors[idx] * (1.0 - frac) + self.discount_factors[idx + 1] * frac
    }

    /// Get survival probability at time t
    #[inline]
    fn get_survival_probability(&self, t: f64) -> f64 {
        let idx = t.min(50.0) as usize;
        
        if idx >= self.hazard_rates.len() {
            return self.hazard_rates[self.hazard_rates.len() - 1].exp();
        }

        // Cumulative hazard
        let mut cumulative_hazard = 0.0;
        for i in 0..=idx {
            cumulative_hazard += self.hazard_rates[i];
        }

        (-cumulative_hazard).exp()
    }

    /// Compute annuity factor
    fn compute_annuity(&self, tenor: f64, frequency: u32) -> f64 {
        let dt = 1.0 / frequency as f64;
        let n_payments = (tenor * frequency as f64) as usize;
        
        let mut annuity = 0.0;
        for i in 1..=n_payments {
            let t = i as f64 * dt;
            let df = self.get_discount_factor(t);
            let survival = self.get_survival_probability(t);
            annuity += dt * df * survival;
        }
        annuity
    }

    /// Compute DV01
    fn compute_dv01(&self, terms: &MortalitySwapTerms) -> Result<f64, MortalitySwapError> {
        // Bump fixed rate by 1bp
        let mut bumped_terms = terms.clone();
        bumped_terms.fixed_rate += 0.0001;
        
        let base_val = self.price_swap(terms)?;
        let bumped_val = self.price_swap(&bumped_terms)?;
        
        let dv01 = (bumped_val.npv - base_val.npv).abs();
        
        if !dv01.is_finite() {
            return Err(MortalitySwapError::NumericalInstability);
        }
        
        Ok(dv01)
    }

    /// Compute expected exposure at maturity
    fn compute_expected_exposure(&self, terms: &MortalitySwapTerms) -> Result<f64, MortalitySwapError> {
        // Simplified: expected positive exposure at midpoint
        let mid_t = terms.tenor_years as f64 / 2.0;
        let survival = self.get_survival_probability(mid_t);
        
        // Exposure proportional to remaining payments
        let remaining_payments = (terms.tenor_years as f64 / 2.0 * terms.payment_frequency as f64) as usize;
        let annuity_remaining = self.compute_annuity(mid_t, terms.payment_frequency);
        
        let exposure = annuity_remaining * terms.notional * survival;
        
        if !exposure.is_finite() {
            return Err(MortalitySwapError::NumericalInstability);
        }
        
        Ok(exposure)
    }

    /// Set counterparty credit spread
    pub fn set_counterparty_spread(&mut self, spread: f64) -> Result<(), MortalitySwapError> {
        if spread < 0.0 || spread > 0.1 {
            return Err(MortalitySwapError::CounterpartyRiskError);
        }
        self.counterparty_spread = spread;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swap_terms_validation() {
        let valid_terms = MortalitySwapTerms {
            notional: 100_000_000.0,
            tenor_years: 10,
            fixed_rate: 0.02,
            reference_population: 1,
            payment_frequency: 1,
        };
        assert!(valid_terms.validate().is_ok());

        let invalid_terms = MortalitySwapTerms {
            notional: -100.0,
            tenor_years: 10,
            fixed_rate: 0.02,
            reference_population: 1,
            payment_frequency: 1,
        };
        assert!(invalid_terms.validate().is_err());
    }

    #[test]
    fn test_swap_pricing() {
        let mut pricer = MortalitySwapPricer::new();
        pricer.initialize_hazard_rates(0.01, 0.02);

        let terms = MortalitySwapTerms {
            notional: 100_000_000.0,
            tenor_years: 10,
            fixed_rate: 0.02,
            reference_population: 1,
            payment_frequency: 1,
        };

        let val = pricer.price_swap(&terms).unwrap();
        assert!(val.fixed_leg_pv > 0.0);
        assert!(val.floating_leg_pv > 0.0);
    }
}
