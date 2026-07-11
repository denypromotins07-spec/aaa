//! Non-Local Hidden Variables Alpha Router
//! Exploits Bell inequality violations for statistical arbitrage.

use alloc::vec::Vec;
use core::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum NonLocalAlphaError {
    NoBellViolation,
    InsufficientSignificance,
    ExecutionFailed,
}

impl fmt::Display for NonLocalAlphaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NonLocalAlphaError::NoBellViolation => write!(f, "No Bell violation detected"),
            NonLocalAlphaError::InsufficientSignificance => write!(f, "Insufficient significance"),
            NonLocalAlphaError::ExecutionFailed => write!(f, "Execution failed"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NonLocalAlphaSignal {
    pub asset_long: usize,
    pub asset_short: usize,
    pub expected_alpha_bps: f64,
    pub confidence_sigma: f64,
    pub decay_half_life_ms: u64,
}

pub struct NonLocalAlphaRouter {
    min_significance: f64,
}

impl NonLocalAlphaRouter {
    pub const fn new(min_significance: f64) -> Self {
        Self { min_significance }
    }

    pub fn generate_alpha_signal(
        &self,
        s_value: f64,
        significance: f64,
        asset_a: usize,
        asset_b: usize,
    ) -> Result<NonLocalAlphaSignal, NonLocalAlphaError> {
        // Require Bell violation (S > 2) with statistical significance
        if s_value <= 2.0 {
            return Err(NonLocalAlphaError::NoBellViolation);
        }

        if significance < self.min_significance {
            return Err(NonLocalAlphaError::InsufficientSignificance);
        }

        // Alpha scales with violation magnitude
        let alpha_bps = (s_value - 2.0) * 100.0;
        
        // Higher significance = longer half-life
        let half_life = (significance * 10.0) as u64;

        Ok(NonLocalAlphaSignal {
            asset_long: asset_a,
            asset_short: asset_b,
            expected_alpha_bps: alpha_bps.clamp(0.0, 1000.0),
            confidence_sigma: significance,
            decay_half_life_ms: half_life,
        })
    }
}
