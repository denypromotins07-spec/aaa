//! Schrödinger's Cat Option Pricer
//! Prices exotic derivatives dependent on quantum measure of market states.

use alloc::vec::Vec;
use core::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum SchrodingerOptionError {
    InvalidStrike,
    InvalidMaturity,
    DecoherenceTooFast,
    NumericalInstability,
}

impl fmt::Display for SchrodingerOptionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchrodingerOptionError::InvalidStrike => write!(f, "Invalid strike"),
            SchrodingerOptionError::InvalidMaturity => write!(f, "Invalid maturity"),
            SchrodingerOptionError::DecoherenceTooFast => write!(f, "Decoherence too fast"),
            SchrodingerOptionError::NumericalInstability => write!(f, "Numerical instability"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SchrodingerCatOption {
    pub strike: f64,
    pub maturity_ns: u64,
    pub underlying_superposition: Vec<(f64, f64)>, // (state_value, amplitude)
    pub decoherence_rate: f64,
}

pub struct SchrodingerOptionPricer {
    risk_free_rate: f64,
}

impl SchrodingerOptionPricer {
    pub const fn new(risk_free_rate: f64) -> Self {
        Self { risk_free_rate }
    }

    pub fn price_cat_option(
        &self,
        option: &SchrodingerCatOption,
    ) -> Result<f64, SchrodingerOptionError> {
        if option.strike <= 0.0 {
            return Err(SchrodingerOptionError::InvalidStrike);
        }
        if option.maturity_ns == 0 {
            return Err(SchrodingerOptionError::InvalidMaturity);
        }

        let maturity_years = option.maturity_ns as f64 / 1e9 / 31536000.0;
        
        if maturity_years <= 0.0 {
            return Err(SchrodingerOptionError::InvalidMaturity);
        }

        // Calculate survival probability considering decoherence
        let decoherence_factor = (-option.decoherence_rate * maturity_years).exp();
        
        if decoherence_factor < 1e-10 {
            return Err(SchrodingerOptionError::DecoherenceTooFast);
        }

        // Calculate expected payoff from superposition
        let mut expected_payoff = 0.0;
        for (state_value, amplitude) in &option.underlying_superposition {
            let probability = amplitude * amplitude; // Born rule
            let payoff = (*state_value - option.strike).max(0.0);
            expected_payoff += probability * payoff * decoherence_factor;
        }

        // Discount to present value
        let discount_factor = (-self.risk_free_rate * maturity_years).exp();
        let price = expected_payoff * discount_factor;

        if price.is_nan() || price.is_infinite() {
            return Err(SchrodingerOptionError::NumericalInstability);
        }

        Ok(price.max(0.0))
    }
}
