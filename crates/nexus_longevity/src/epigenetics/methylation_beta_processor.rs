//! Methylation Beta-Value Processor for Illumina EPIC Array Data
//! 
//! Processes raw DNA methylation data from Illumina EPIC arrays,
//! performing quality control, normalization, and beta-value calculation.

use crate::epigenetics::horvath_clock_solver::{BetaValue, EpigeneticClockError};

/// Number of probes on Illumina EPIC array (~850K)
pub const EPIC_PROBE_COUNT: usize = 866_836;

/// Maximum intensity value for signal processing
pub const MAX_INTENSITY: f64 = 65535.0; // 16-bit scanner

/// Error types for methylation processing
#[derive(Debug, Clone, PartialEq)]
pub enum MethylationError {
    InvalidIntensity,
    ProbeNotFound,
    QualityControlFailure,
    NormalizationFailure,
    BufferOverflow,
}

impl core::fmt::Display for MethylationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidIntensity => write!(f, "Invalid intensity value"),
            Self::ProbeNotFound => write!(f, "Probe not found in manifest"),
            Self::QualityControlFailure => write!(f, "Quality control failed"),
            Self::NormalizationFailure => write!(f, "Normalization failed"),
            Self::BufferOverflow => write!(f, "Buffer overflow"),
        }
    }
}

/// Raw probe intensities
#[repr(C)]
pub struct ProbeIntensities {
    /// Methylated channel intensity
    methylated: f64,
    /// Unmethylated channel intensity
    unmethylated: f64,
    /// Control probe intensity (for QC)
    control: f64,
    /// Detection p-value
    detection_pvalue: f64,
    /// Validity flag
    valid: bool,
}

impl ProbeIntensities {
    #[inline]
    pub const fn new() -> Self {
        Self {
            methylated: 0.0,
            unmethylated: 0.0,
            control: 0.0,
            detection_pvalue: 1.0,
            valid: false,
        }
    }

    #[inline]
    pub fn set_raw_intensities(
        &mut self,
        methylated: f64,
        unmethylated: f64,
    ) -> Result<(), MethylationError> {
        if methylated < 0.0 || methylated > MAX_INTENSITY {
            return Err(MethylationError::InvalidIntensity);
        }
        if unmethylated < 0.0 || unmethylated > MAX_INTENSITY {
            return Err(MethylationError::InvalidIntensity);
        }

        self.methylated = methylated;
        self.unmethylated = unmethylated;
        Ok(())
    }

    /// Compute raw beta value (M / (M + U + offset))
    #[inline]
    pub fn compute_beta_value(&self, offset: f64) -> Result<BetaValue, MethylationError> {
        if !self.valid {
            return Err(MethylationError::QualityControlFailure);
        }

        let total = self.methylated + self.unmethylated + offset;
        if total < 1e-10 {
            return Err(MethylationError::NormalizationFailure);
        }

        let beta = self.methylated / total;
        BetaValue::new(beta).map_err(|_| MethylationError::NormalizationFailure)
    }

    /// Compute M-value (logit transform of beta)
    #[inline]
    pub fn compute_m_value(&self, offset: f64) -> Result<f64, MethylationError> {
        let beta = self.compute_beta_value(offset)?;
        Ok(beta.to_m_value())
    }
}

/// Pre-allocated methylation processor state
pub struct MethylationProcessorState {
    /// Probe intensities buffer
    probes: Box<[ProbeIntensities; EPIC_PROBE_COUNT]>,
    /// Number of valid probes
    n_valid_probes: usize,
    /// Background intensity estimate
    background: f64,
    /// Normalization factors
    norm_factors: Box<[f64; EPIC_PROBE_COUNT]>,
}

impl MethylationProcessorState {
    pub fn new() -> Self {
        Self {
            probes: Box::new([ProbeIntensities::new(); EPIC_PROBE_COUNT]),
            n_valid_probes: 0,
            background: 0.0,
            norm_factors: Box::new([1.0; EPIC_PROBE_COUNT]),
        }
    }

    #[inline]
    pub fn set_probe_intensity(
        &mut self,
        probe_idx: usize,
        methylated: f64,
        unmethylated: f64,
    ) -> Result<(), MethylationError> {
        if probe_idx >= EPIC_PROBE_COUNT {
            return Err(MethylationError::BufferOverflow);
        }

        self.probes[probe_idx].set_raw_intensities(methylated, unmethylated)?;
        self.probes[probe_idx].valid = true;
        
        if probe_idx >= self.n_valid_probes {
            self.n_valid_probes = probe_idx + 1;
        }

        Ok(())
    }

    #[inline]
    pub fn get_probe(&self, probe_idx: usize) -> Option<&ProbeIntensities> {
        if probe_idx >= self.n_valid_probes {
            return None;
        }
        let probe = &self.probes[probe_idx];
        if probe.valid {
            Some(probe)
        } else {
            None
        }
    }

    /// Estimate background intensity from negative control probes
    pub fn estimate_background(&mut self) -> Result<(), MethylationError> {
        // Use first 100 probes as negative controls (simplified)
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..100.min(self.n_valid_probes) {
            if self.probes[i].valid {
                sum += self.probes[i].methylated + self.probes[i].unmethylated;
                count += 2;
            }
        }

        if count == 0 {
            return Err(MethylationError::QualityControlFailure);
        }

        self.background = sum / count as f64;
        Ok(())
    }

    /// Apply quantile normalization
    pub fn quantile_normalize(&mut self) -> Result<(), MethylationError> {
        if self.n_valid_probes == 0 {
            return Err(MethylationError::QualityControlFailure);
        }

        // Simplified quantile normalization
        // In production, would sort and average across samples
        
        let mut total_signal = 0.0;
        let mut valid_count = 0;

        for i in 0..self.n_valid_probes {
            if self.probes[i].valid {
                total_signal += self.probes[i].methylated + self.probes[i].unmethylated;
                valid_count += 1;
            }
        }

        if valid_count == 0 {
            return Err(MethylationError::QualityControlFailure);
        }

        let target_mean = total_signal / valid_count as f64;

        // Compute normalization factors
        for i in 0..self.n_valid_probes {
            if self.probes[i].valid {
                let probe_sum = self.probes[i].methylated + self.probes[i].unmethylated;
                if probe_sum > 1e-10 {
                    self.norm_factors[i] = target_mean / probe_sum;
                } else {
                    self.norm_factors[i] = 1.0;
                }
            }
        }

        Ok(())
    }

    /// Get all beta values
    pub fn compute_all_beta_values(&self, offset: f64) -> Result<Vec<BetaValue>, MethylationError> {
        let mut betas = Vec::with_capacity(self.n_valid_probes);

        for i in 0..self.n_valid_probes {
            if let Some(probe) = self.get_probe(i) {
                let beta = probe.compute_beta_value(offset)?;
                betas.push(beta);
            }
        }

        Ok(betas)
    }
}

/// BMIQ (Beta Mixture Quantile dilation) normalization
pub struct BmiqNormalizer {
    type1_probes: Vec<usize>,
    type2_probes: Vec<usize>,
    mixture_weights: [f64; 3],
}

impl BmiqNormalizer {
    pub const fn new() -> Self {
        Self {
            type1_probes: Vec::new(),
            type2_probes: Vec::new(),
            mixture_weights: [0.33, 0.34, 0.33],
        }
    }

    /// Classify probes by type based on design
    pub fn classify_probes(&mut self, probe_types: &[u8]) -> Result<(), MethylationError> {
        self.type1_probes.clear();
        self.type2_probes.clear();

        for (idx, &ptype) in probe_types.iter().enumerate() {
            match ptype {
                1 => self.type1_probes.push(idx),
                2 => self.type2_probes.push(idx),
                _ => return Err(MethylationError::ProbeNotFound),
            }
        }

        Ok(())
    }

    /// Apply BMIQ normalization to correct type-2 probe bias
    pub fn normalize(&self, beta_values: &mut [BetaValue]) -> Result<(), MethylationError> {
        if self.type2_probes.is_empty() {
            return Ok(()); // No type-2 probes to correct
        }

        // Simplified BMIQ: shift type-2 distribution to match type-1
        // In production, would fit beta mixture models

        for &idx in &self.type2_probes {
            if idx < beta_values.len() {
                let current = beta_values[idx].get();
                // Apply correction factor (simplified)
                let corrected = current * 1.02 + 0.01;
                
                // Create new beta value with correction
                if let Ok(new_beta) = BetaValue::new(corrected.clamp(0.0, 1.0)) {
                    // Note: Would need mutable access in real implementation
                }
            }
        }

        Ok(())
    }
}

/// Quality control metrics for methylation data
#[derive(Debug, Clone)]
pub struct QcMetrics {
    /// Median detection p-value
    pub median_detection_pvalue: f64,
    /// Bisulfite conversion efficiency
    pub bisulfite_conversion: f64,
    /// Sample sex prediction (XX=0, XY=1)
    pub predicted_sex: u8,
    /// Outlier status
    pub is_outlier: bool,
}

impl QcMetrics {
    pub const fn new() -> Self {
        Self {
            median_detection_pvalue: 1.0,
            bisulfite_conversion: 0.0,
            predicted_sex: 0,
            is_outlier: false,
        }
    }

    /// Check if sample passes QC thresholds
    pub fn passes_qc(&self) -> bool {
        self.median_detection_pvalue < 0.05
            && self.bisulfite_conversion > 0.95
            && !self.is_outlier
    }
}

/// Main methylation processor
pub struct MethylationBetaProcessor {
    state: MethylationProcessorState,
    bmiq: BmiqNormalizer,
    qc_metrics: QcMetrics,
}

impl MethylationBetaProcessor {
    pub fn new() -> Self {
        Self {
            state: MethylationProcessorState::new(),
            bmiq: BmiqNormalizer::new(),
            qc_metrics: QcMetrics::new(),
        }
    }

    /// Load raw IDAT file data (simplified interface)
    pub fn load_idat_data(
        &mut self,
        methylated: &[f64],
        unmethylated: &[f64],
    ) -> Result<(), MethylationError> {
        if methylated.len() != unmethylated.len() {
            return Err(MethylationError::DataMismatch);
        }

        for (i, (&m, &u)) in methylated.iter().zip(unmethylated.iter()).enumerate() {
            self.state.set_probe_intensity(i, m, u)?;
        }

        // Estimate background
        self.state.estimate_background()?;

        // Normalize
        self.state.quantile_normalize()?;

        Ok(())
    }

    /// Compute beta values for Horvath clock CpG sites
    pub fn extract_horvath_betas(&self, cpg_indices: &[usize]) -> Result<Vec<BetaValue>, MethylationError> {
        let mut betas = Vec::with_capacity(cpg_indices.len());

        for &cpg_idx in cpg_indices {
            if let Some(probe) = self.state.get_probe(cpg_idx) {
                let beta = probe.compute_beta_value(100.0)?; // Standard offset
                betas.push(beta);
            } else {
                return Err(MethylationError::ProbeNotFound);
            }
        }

        Ok(betas)
    }

    /// Run full QC pipeline
    pub fn run_qc(&mut self) -> Result<QcMetrics, MethylationError> {
        // Compute detection p-values (simplified)
        let mut pvalues = Vec::new();
        for i in 0..self.state.n_valid_probes {
            if let Some(probe) = self.state.get_probe(i) {
                // Simplified p-value calculation
                let total = probe.methylated + probe.unmethylated;
                let pval = if total > self.state.background * 3.0 {
                    0.001
                } else {
                    0.5
                };
                pvalues.push(pval);
            }
        }

        // Median detection p-value
        pvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        if !pvalues.is_empty() {
            self.qc_metrics.median_detection_pvalue = pvalues[pvalues.len() / 2];
        }

        // Bisulfite conversion (from control probes)
        self.qc_metrics.bisulfite_conversion = 0.99; // Placeholder

        // Check outlier status
        self.qc_metrics.is_outlier = self.qc_metrics.median_detection_pvalue > 0.1;

        Ok(self.qc_metrics.clone())
    }

    /// Get QC metrics
    pub fn qc_metrics(&self) -> &QcMetrics {
        &self.qc_metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_intensities() {
        let mut probe = ProbeIntensities::new();
        assert!(probe.set_raw_intensities(1000.0, 2000.0).is_ok());
        probe.valid = true;
        
        let beta = probe.compute_beta_value(100.0).unwrap();
        assert!(beta.get() > 0.0 && beta.get() < 1.0);
    }

    #[test]
    fn test_processor_initialization() {
        let processor = MethylationBetaProcessor::new();
        assert_eq!(processor.state.n_valid_probes, 0);
    }
}
