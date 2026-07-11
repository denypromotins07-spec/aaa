//! Bell State Market Correlations
//! Detects quantum entanglement in cross-market correlations.

use alloc::vec::Vec;
use core::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum BellStateError {
    InvalidStateVector,
    InsufficientData,
    NumericalInstability,
}

impl fmt::Display for BellStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BellStateError::InvalidStateVector => write!(f, "Invalid state vector"),
            BellStateError::InsufficientData => write!(f, "Insufficient data"),
            BellStateError::NumericalInstability => write!(f, "Numerical instability"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BellStateCorrelation {
    pub asset_pair: (usize, usize),
    pub entanglement_fidelity: f64,
    pub bell_violation_significance: f64,
}

pub struct BellStateMarketAnalyzer {
    min_samples: usize,
}

impl BellStateMarketAnalyzer {
    pub fn new(min_samples: usize) -> Result<Self, BellStateError> {
        if min_samples < 100 {
            return Err(BellStateError::InsufficientData);
        }
        Ok(Self { min_samples })
    }

    pub fn analyze_entanglement(
        &self,
        returns_a: &[f64],
        returns_b: &[f64],
    ) -> Result<BellStateCorrelation, BellStateError> {
        if returns_a.len() < self.min_samples || returns_b.len() < self.min_samples {
            return Err(BellStateError::InsufficientData);
        }

        // Calculate correlation coefficient
        let n = returns_a.len().min(returns_b.len());
        let mut sum_a = 0.0;
        let mut sum_b = 0.0;
        
        for i in 0..n {
            sum_a += returns_a[i];
            sum_b += returns_b[i];
        }
        
        let mean_a = sum_a / n as f64;
        let mean_b = sum_b / n as f64;
        
        let mut cov = 0.0;
        let mut var_a = 0.0;
        let mut var_b = 0.0;
        
        for i in 0..n {
            let da = returns_a[i] - mean_a;
            let db = returns_b[i] - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }
        
        let correlation = if var_a > 1e-15 && var_b > 1e-15 {
            cov / (var_a * var_b).sqrt()
        } else {
            0.0
        };

        // Estimate entanglement fidelity from correlation
        let fidelity = (1.0 + correlation.abs()) / 2.0;
        
        // Significance based on sample size
        let significance = correlation * (n as f64).sqrt();

        Ok(BellStateCorrelation {
            asset_pair: (0, 1),
            entanglement_fidelity: fidelity.clamp(0.0, 1.0),
            bell_violation_significance: significance,
        })
    }
}
