//! Elastic Net Regularized Polygenic Risk Score Calculator
//! 
//! Computes weighted sums of millions of SNP effects using L1/L2 elastic net
//! regularization for disease risk prediction without heap allocations in hot paths.

use core::slice;
use std::cmp::Ordering;

/// Maximum number of SNPs supported (typical GWAS arrays have 500K-2M SNPs)
pub const MAX_SNPS: usize = 2_097_152; // 2^21

/// Error types for PRS calculation
#[derive(Debug, Clone, PartialEq)]
pub enum PrsError {
    InvalidAlleleDosage,
    WeightMismatch,
    ConvergenceFailure,
    NumericalInstability,
    BufferOverflow,
}

impl core::fmt::Display for PrsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidAlleleDosage => write!(f, "Invalid allele dosage"),
            Self::WeightMismatch => write!(f, "Weight vector length mismatch"),
            Self::ConvergenceFailure => write!(f, "Failed to converge"),
            Self::NumericalInstability => write!(f, "Numerical instability detected"),
            Self::BufferOverflow => write!(f, "Buffer overflow"),
        }
    }
}

/// Allele dosage (0, 1, or 2 copies of effect allele)
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlleleDosage {
    HomRef = 0,
    Het = 1,
    HomAlt = 2,
}

impl AlleleDosage {
    #[inline]
    pub fn from_u8(v: u8) -> Result<Self, PrsError> {
        match v {
            0 => Ok(Self::HomRef),
            1 => Ok(Self::Het),
            2 => Ok(Self::HomAlt),
            _ => Err(PrsError::InvalidAlleleDosage),
        }
    }

    #[inline]
    pub fn as_f64(self) -> f64 {
        self as u8 as f64
    }
}

/// Pre-allocated PRS computation state
pub struct PrsState {
    /// SNP dosages (packed 2 bits per SNP)
    dosages: [u8; MAX_SNPS / 4 + 1],
    /// Effect weights (f64 array)
    weights: Box<[f64; MAX_SNPS]>,
    /// Number of active SNPs
    n_snps: usize,
    /// Mean for standardization
    mean: f64,
    /// Standard deviation for standardization
    std_dev: f64,
}

impl PrsState {
    pub fn new() -> Self {
        Self {
            dosages: [0u8; MAX_SNPS / 4 + 1],
            weights: Box::new([0.0; MAX_SNPS]),
            n_snps: 0,
            mean: 0.0,
            std_dev: 1.0,
        }
    }

    #[inline]
    pub fn set_dosage(&mut self, idx: usize, dosage: AlleleDosage) -> Result<(), PrsError> {
        if idx >= self.n_snps {
            return Err(PrsError::BufferOverflow);
        }
        let byte_idx = idx / 4;
        let bit_shift = ((idx % 4) * 2) as u32;
        let packed = dosage as u8;
        
        unsafe {
            let ptr = self.dosages.as_mut_ptr().add(byte_idx);
            let current = ptr.read_volatile();
            let cleared = current & !(0b11 << bit_shift);
            ptr.write_volatile(cleared | (packed << bit_shift));
        }
        Ok(())
    }

    #[inline]
    pub fn get_dosage(&self, idx: usize) -> Option<AlleleDosage> {
        if idx >= self.n_snps {
            return None;
        }
        let byte_idx = idx / 4;
        let bit_shift = ((idx % 4) * 2) as u32;
        let packed = unsafe { *self.dosages.get_unchecked(byte_idx) };
        let value = (packed >> bit_shift) & 0b11;
        AlleleDosage::from_u8(value).ok()
    }

    #[inline]
    pub fn set_weight(&mut self, idx: usize, weight: f64) -> Result<(), PrsError> {
        if idx >= self.n_snps {
            return Err(PrsError::BufferOverflow);
        }
        self.weights[idx] = weight;
        Ok(())
    }

    #[inline]
    pub fn set_n_snps(&mut self, n: usize) -> Result<(), PrsError> {
        if n > MAX_SNPS {
            return Err(PrsError::BufferOverflow);
        }
        self.n_snps = n;
        Ok(())
    }
}

/// Elastic Net regularizer for sparse PRS
pub struct ElasticNetRegularizer {
    /// L1 penalty (Lasso) - promotes sparsity
    alpha: f64,
    /// L2 penalty (Ridge) - promotes stability
    lambda: f64,
    /// Convergence threshold
    tolerance: f64,
    /// Maximum iterations
    max_iterations: usize,
}

impl ElasticNetRegularizer {
    pub const fn new(alpha: f64, lambda: f64) -> Self {
        Self {
            alpha,
            lambda,
            tolerance: 1e-6,
            max_iterations: 1000,
        }
    }

    /// Soft thresholding operator for L1 penalty
    #[inline]
    fn soft_threshold(x: f64, threshold: f64) -> f64 {
        if x > threshold {
            x - threshold
        } else if x < -threshold {
            x + threshold
        } else {
            0.0
        }
    }

    /// Coordinate descent optimization for elastic net
    pub fn fit(&self, state: &mut PrsState, y: &[f64]) -> Result<(), PrsError> {
        let n = state.n_snps;
        if y.len() != n {
            return Err(PrsError::WeightMismatch);
        }

        // Initialize weights to OLS estimates
        let mut weights = vec![0.0; n];
        let mut residuals = y.to_vec();

        // Precompute X'X diagonal (assumes standardized predictors)
        let xtx_diag = vec![1.0; n];

        for iter in 0..self.max_iterations {
            let mut max_change = 0.0;

            for j in 0..n {
                // Compute partial residual
                let mut partial_residual = 0.0;
                for i in 0..y.len() {
                    if let Some(dosage) = state.get_dosage(j) {
                        partial_residual += dosage.as_f64() * residuals[i];
                    }
                }

                // Add back current contribution
                let wj_old = weights[j];
                partial_residual += wj_old * xtx_diag[j];

                // Apply elastic net update
                let rho = partial_residual;
                let z = xtx_diag[j];
                
                let wj_new = if z > 1e-10 {
                    let l1_thresh = self.alpha * self.lambda;
                    let l2_factor = 1.0 + (1.0 - self.alpha) * self.lambda;
                    
                    Self::soft_threshold(rho, l1_thresh) / (z * l2_factor)
                } else {
                    0.0
                };

                weights[j] = wj_new;
                let change = (wj_new - wj_old).abs();
                if change > max_change {
                    max_change = change;
                }

                // Update residuals incrementally
                if let Some(dosage) = state.get_dosage(j) {
                    let delta = wj_new - wj_old;
                    for i in 0..y.len() {
                        residuals[i] -= delta * dosage.as_f64();
                    }
                }
            }

            if max_change < self.tolerance {
                // Converged - copy weights to state
                for j in 0..n {
                    state.set_weight(j, weights[j])?;
                }
                return Ok(());
            }
        }

        Err(PrsError::ConvergenceFailure)
    }
}

/// Polygenic Risk Score calculator
pub struct PrsCalculator {
    state: PrsState,
    regularizer: ElasticNetRegularizer,
}

impl PrsCalculator {
    pub fn new(alpha: f64, lambda: f64) -> Self {
        Self {
            state: PrsState::new(),
            regularizer: ElasticNetRegularizer::new(alpha, lambda),
        }
    }

    /// Compute raw polygenic risk score
    #[inline]
    pub fn compute_raw_prs(&self) -> Result<f64, PrsError> {
        let mut prs = 0.0;
        
        for j in 0..self.state.n_snps {
            if let Some(dosage) = self.state.get_dosage(j) {
                let weight = self.state.weights[j];
                prs += dosage.as_f64() * weight;
            }
        }

        if !prs.is_finite() {
            return Err(PrsError::NumericalInstability);
        }

        Ok(prs)
    }

    /// Compute standardized polygenic risk score (Z-score)
    #[inline]
    pub fn compute_standardized_prs(&self) -> Result<f64, PrsError> {
        let raw = self.compute_raw_prs()?;
        
        if self.state.std_dev < 1e-10 {
            return Err(PrsError::NumericalInstability);
        }

        Ok((raw - self.state.mean) / self.state.std_dev)
    }

    /// Fit model with elastic net regularization
    pub fn fit(&mut self, y: &[f64]) -> Result<(), PrsError> {
        self.regularizer.fit(&mut self.state, y)
    }

    /// Set mean and std for standardization
    pub fn set_standardization_params(&mut self, mean: f64, std_dev: f64) {
        self.state.mean = mean;
        self.state.std_dev = std_dev.max(1e-10);
    }

    pub fn state(&self) -> &PrsState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut PrsState {
        &mut self.state
    }
}

/// Batch PRS processor for multiple individuals
pub struct BatchPrsProcessor {
    calculators: Box<[PrsCalculator]>,
    results: Box<[f64]>,
}

impl BatchPrsProcessor {
    pub fn new(n_individuals: usize, alpha: f64, lambda: f64) -> Result<Self, PrsError> {
        if n_individuals == 0 || n_individuals > 100_000 {
            return Err(PrsError::BufferOverflow);
        }

        let mut calculators = Vec::with_capacity(n_individuals);
        for _ in 0..n_individuals {
            calculators.push(PrsCalculator::new(alpha, lambda));
        }

        Ok(Self {
            calculators: calculators.into_boxed_slice(),
            results: vec![0.0; n_individuals].into_boxed_slice(),
        })
    }

    pub fn compute_all(&mut self) -> Result<(), PrsError> {
        for (i, calc) in self.calculators.iter().enumerate() {
            self.results[i] = calc.compute_standardized_prs()?;
        }
        Ok(())
    }

    pub fn get_result(&self, idx: usize) -> Option<f64> {
        self.results.get(idx).copied()
    }
}

/// Disease risk category based on PRS percentile
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskCategory {
    VeryLow,      // < 10th percentile
    Low,          // 10-25th
    Average,      // 25-75th
    High,         // 75-90th
    VeryHigh,     // 90-95th
    Extreme,      // > 95th
}

impl RiskCategory {
    pub fn from_zscore(z: f64) -> Self {
        // Approximate percentiles from Z-scores
        if z < -1.28 {
            Self::VeryLow
        } else if z < -0.67 {
            Self::Low
        } else if z < 0.67 {
            Self::Average
        } else if z < 1.28 {
            Self::High
        } else if z < 1.64 {
            Self::VeryHigh
        } else {
            Self::Extreme
        }
    }

    pub fn relative_risk(self) -> f64 {
        // Approximate relative risks from epidemiological data
        match self {
            Self::VeryLow => 0.5,
            Self::Low => 0.75,
            Self::Average => 1.0,
            Self::High => 1.5,
            Self::VeryHigh => 2.0,
            Self::Extreme => 3.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allele_dosage() {
        assert_eq!(AlleleDosage::from_u8(0).unwrap(), AlleleDosage::HomRef);
        assert_eq!(AlleleDosage::from_u8(1).unwrap(), AlleleDosage::Het);
        assert_eq!(AlleleDosage::from_u8(2).unwrap(), AlleleDosage::HomAlt);
        assert!(AlleleDosage::from_u8(3).is_err());
    }

    #[test]
    fn test_soft_threshold() {
        assert_eq!(ElasticNetRegularizer::soft_threshold(0.5, 0.3), 0.2);
        assert_eq!(ElasticNetRegularizer::soft_threshold(-0.5, 0.3), -0.2);
        assert_eq!(ElasticNetRegularizer::soft_threshold(0.2, 0.3), 0.0);
    }

    #[test]
    fn test_risk_category() {
        assert_eq!(RiskCategory::from_zscore(-2.0), RiskCategory::VeryLow);
        assert_eq!(RiskCategory::from_zscore(0.0), RiskCategory::Average);
        assert_eq!(RiskCategory::from_zscore(2.0), RiskCategory::Extreme);
    }
}
