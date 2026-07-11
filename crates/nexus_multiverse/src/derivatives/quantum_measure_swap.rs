//! Quantum Measure Swap Pricer
//! Derivatives where payout depends on quantum measure of market states.

use alloc::vec::Vec;
use core::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum QuantumSwapError {
    InvalidNotional,
    InvalidTenor,
    MeasureDrift,
    NumericalInstability,
}

impl fmt::Display for QuantumSwapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantumSwapError::InvalidNotional => write!(f, "Invalid notional"),
            QuantumSwapError::InvalidTenor => write!(f, "Invalid tenor"),
            QuantumSwapError::MeasureDrift => write!(f, "Measure drift detected"),
            QuantumSwapError::NumericalInstability => write!(f, "Numerical instability"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantumMeasureSwap {
    pub notional: f64,
    pub tenor_ns: u64,
    pub floating_leg_measure: f64,
    pub fixed_leg_rate: f64,
    pub decoherence_adjustment: f64,
}

pub struct QuantumSwapPricer {
    risk_free_rate: f64,
}

impl QuantumSwapPricer {
    pub const fn new(risk_free_rate: f64) -> Self {
        Self { risk_free_rate }
    }

    pub fn price_swap(
        &self,
        swap: &QuantumMeasureSwap,
    ) -> Result<f64, QuantumSwapError> {
        if swap.notional <= 0.0 {
            return Err(QuantumSwapError::InvalidNotional);
        }
        if swap.tenor_ns == 0 {
            return Err(QuantumSwapError::InvalidTenor);
        }

        let tenor_years = swap.tenor_ns as f64 / 1e9 / 31536000.0;
        
        if tenor_years <= 0.0 {
            return Err(QuantumSwapError::InvalidTenor);
        }

        // Verify measure conservation
        if (swap.floating_leg_measure - 1.0).abs() > 0.01 {
            return Err(QuantumSwapError::MeasureDrift);
        }

        // Calculate floating leg value with decoherence adjustment
        let decoherence_factor = (-swap.decoherence_adjustment * tenor_years).exp();
        let floating_value = swap.notional * swap.floating_leg_measure * decoherence_factor;

        // Calculate fixed leg present value
        let discount_factor = (-self.risk_free_rate * tenor_years).exp();
        let fixed_value = swap.notional * swap.fixed_leg_rate * discount_factor;

        // Net present value (receiver swap: receive floating, pay fixed)
        let npv = floating_value - fixed_value;

        if npv.is_nan() || npv.is_infinite() {
            return Err(QuantumSwapError::NumericalInstability);
        }

        Ok(npv)
    }

    /// Calculate fair fixed rate that makes swap NPV = 0
    pub fn calculate_par_rate(
        &self,
        notional: f64,
        tenor_ns: u64,
        floating_measure: f64,
        decoherence: f64,
    ) -> Result<f64, QuantumSwapError> {
        let swap = QuantumMeasureSwap {
            notional,
            tenor_ns,
            floating_leg_measure: floating_measure,
            fixed_leg_rate: 0.0, // Will be solved
            decoherence_adjustment: decoherence,
        };

        let tenor_years = tenor_ns as f64 / 1e9 / 31536000.0;
        let decoherence_factor = (-decoherence * tenor_years).exp();
        let discount_factor = (-self.risk_free_rate * tenor_years).exp();

        // Par rate: floating_value = fixed_value
        // notional * measure * decoh = notional * par_rate * discount
        let par_rate = (floating_measure * decoherence_factor) / discount_factor;

        if par_rate.is_nan() || par_rate.is_infinite() {
            return Err(QuantumSwapError::NumericalInstability);
        }

        Ok(par_rate)
    }
}
