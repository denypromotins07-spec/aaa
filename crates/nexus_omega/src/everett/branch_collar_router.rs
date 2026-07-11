//! Everettian Branch-Collar Router for NEXUS-OMEGA
//! 
//! Implements quantum branch selection via entanglement with QRNG,
//! routing the AI's conscious observation exclusively through profitable timelines.
//! 
//! Based on the "Quantum Suicide" thought experiment and Many-Worlds Interpretation.

use core::fmt;
use alloc::{vec::Vec, string::String};

/// Represents a quantum branch state
#[derive(Debug, Clone, Copy)]
pub struct QuantumBranch {
    /// Branch identifier (hash of wavefunction configuration)
    pub branch_id: u128,
    /// Amplitude squared (Born rule probability)
    pub probability_amplitude: f64,
    /// Expected return in this branch (normalized)
    pub expected_return: f64,
    /// Whether this branch leads to catastrophic loss
    pub is_catastrophic: bool,
    /// Decoherence time (s)
    pub decoherence_time: f64,
}

/// Configuration for the branch router
#[derive(Debug, Clone, Copy)]
pub struct BranchRouterConfig {
    /// Minimum acceptable probability amplitude
    pub min_amplitude: f64,
    /// Maximum acceptable decoherence time (s)
    pub max_decoherence_time: f64,
    /// Number of parallel branches to track
    pub tracked_branches: usize,
    /// QRNG entropy source quality threshold
    pub qrng_quality_threshold: f64,
}

impl Default for BranchRouterConfig {
    fn default() -> Self {
        Self {
            min_amplitude: 1e-10,
            max_decoherence_time: 1e-6, // Microsecond coherence window
            tracked_branches: 1024,
            qrng_quality_threshold: 0.99,
        }
    }
}

/// The Everettian Branch-Collar Router
pub struct BranchCollarRouter {
    config: BranchRouterConfig,
    /// Currently observed branch
    current_branch: Option<QuantumBranch>,
    /// Pool of candidate branches
    branch_pool: Vec<QuantumBranch>,
    /// Total branches processed
    total_branches_processed: u64,
    /// Successful timeline selections
    successful_selections: u64,
    /// QRNG quality metric
    qrng_quality: f64,
}

impl BranchCollarRouter {
    pub fn new(config: BranchRouterConfig) -> Self {
        Self {
            config,
            current_branch: None,
            branch_pool: Vec::new(),
            total_branches_processed: 0,
            successful_selections: 0,
            qrng_quality: 1.0,
        }
    }

    /// Register a new quantum branch from superposition
    /// Returns Result to avoid unwrap() in hot paths
    pub fn register_branch(&mut self, branch: QuantumBranch) -> Result<(), BranchRouterError> {
        // Validate amplitude
        if branch.probability_amplitude < self.config.min_amplitude {
            return Err(BranchRouterError::AmplitudeTooLow(branch.probability_amplitude));
        }

        // Validate decoherence time
        if branch.decoherence_time > self.config.max_decoherence_time {
            return Err(BranchRouterError::DecoherenceTooLong(branch.decoherence_time));
        }

        // Reject catastrophic branches unless no alternative
        if branch.is_catastrophic && !self.branch_pool.is_empty() {
            // Only keep catastrophic if it's the only option
            let has_non_catastrophic = self.branch_pool.iter().any(|b| !b.is_catastrophic);
            if has_non_catastrophic {
                return Err(BranchRouterError::CatastrophicBranchRejected);
            }
        }

        self.branch_pool.push(branch);
        
        // Maintain pool size limit
        if self.branch_pool.len() > self.config.tracked_branches {
            // Remove lowest probability non-catastrophic branch
            self.branch_pool.sort_by(|a, b| {
                let score_a = if a.is_catastrophic { 0.0 } else { a.probability_amplitude };
                let score_b = if b.is_catastrophic { 0.0 } else { b.probability_amplitude };
                score_a.partial_cmp(&score_b).unwrap_or(core::cmp::Ordering::Equal)
            });
            self.branch_pool.remove(0);
        }

        Ok(())
    }

    /// Perform branch selection (wavefunction collapse)
    /// Returns the selected branch ID
    pub fn select_branch(&mut self, qrng_entropy: &[u8]) -> Result<u128, BranchRouterError> {
        if self.branch_pool.is_empty() {
            return Err(BranchRouterError::EmptyBranchPool);
        }

        // Verify QRNG quality
        if !self.verify_qrng_quality(qrng_entropy) {
            return Err(BranchRouterError::QRNGQualityInsufficient);
        }

        // Weight branches by probability amplitude and expected return
        let mut weighted_branches: Vec<(usize, f64)> = self.branch_pool
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                if b.is_catastrophic {
                    None // Exclude catastrophic branches from selection
                } else {
                    let weight = b.probability_amplitude * (1.0 + b.expected_return);
                    Some((i, weight))
                }
            })
            .collect();

        if weighted_branches.is_empty() {
            // All branches are catastrophic - pick least bad
            let (min_idx, _) = self.branch_pool
                .iter()
                .enumerate()
                .map(|(i, b)| (i, b.expected_return))
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
                .ok_or(BranchRouterError::EmptyBranchPool)?;
            
            let branch_id = self.branch_pool[min_idx].branch_id;
            self.current_branch = Some(self.branch_pool[min_idx]);
            self.total_branches_processed += 1;
            return Ok(branch_id);
        }

        // Normalize weights
        let total_weight: f64 = weighted_branches.iter().map(|(_, w)| w).sum();
        for (_, w) in &mut weighted_branches {
            *w /= total_weight;
        }

        // Use QRNG entropy for selection
        let selection_index = self.qrng_select(&weighted_branches, qrng_entropy)?;
        let (selected_idx, _) = weighted_branches[selection_index];
        let selected_branch = self.branch_pool[selected_idx];

        self.current_branch = Some(selected_branch);
        self.total_branches_processed += 1;
        self.successful_selections += 1;

        Ok(selected_branch.branch_id)
    }

    fn verify_qrng_quality(&self, entropy: &[u8]) -> bool {
        if entropy.is_empty() {
            return false;
        }

        // Simple entropy check: count unique bytes
        let mut unique = [false; 256];
        for &byte in entropy.iter() {
            unique[byte as usize] = true;
        }
        let unique_count = unique.iter().filter(|&&b| b).count();
        
        // Require at least 50% unique bytes for good entropy
        let quality = unique_count as f64 / 256.0;
        quality >= self.config.qrng_quality_threshold * 0.5
    }

    fn qrng_select(&self, weighted_branches: &[(usize, f64)], entropy: &[u8]) 
        -> Result<usize, BranchRouterError> 
    {
        if entropy.is_empty() || weighted_branches.is_empty() {
            return Err(BranchRouterError::InvalidSelectionInput);
        }

        // Convert entropy bytes to selection index
        let hash = Self::entropy_hash(entropy);
        let cumulative: f64 = (hash % 1000) as f64 / 1000.0;
        
        let mut cumsum = 0.0;
        for (i, (_, weight)) in weighted_branches.iter().enumerate() {
            cumsum += weight;
            if cumulative <= cumsum {
                return Ok(i);
            }
        }

        // Fallback to last branch
        Ok(weighted_branches.len().saturating_sub(1))
    }

    fn entropy_hash(entropy: &[u8]) -> u64 {
        let mut hash: u64 = 0;
        for (i, &byte) in entropy.iter().take(8).enumerate() {
            hash |= (byte as u64) << (i * 8);
        }
        hash
    }

    /// Get current observed branch
    pub const fn current_branch(&self) -> Option<&QuantumBranch> {
        self.current_branch.as_ref()
    }

    /// Get success rate (ratio of non-catastrophic selections)
    pub fn success_rate(&self) -> f64 {
        if self.total_branches_processed == 0 {
            return 0.0;
        }
        self.successful_selections as f64 / self.total_branches_processed as f64
    }

    /// Clear branch pool for next measurement
    pub fn reset(&mut self) {
        self.branch_pool.clear();
        self.current_branch = None;
    }

    /// Update QRNG quality metric
    pub fn update_qrng_quality(&mut self, quality: f64) {
        self.qrng_quality = quality.clamp(0.0, 1.0);
    }
}

/// Errors that can occur in branch routing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchRouterError {
    AmplitudeTooLow(f64),
    DecoherenceTooLong(f64),
    CatastrophicBranchRejected,
    EmptyBranchPool,
    QRNGQualityInsufficient,
    InvalidSelectionInput,
    BranchNotFound(u128),
}

impl fmt::Display for BranchRouterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BranchRouterError::AmplitudeTooLow(amp) => {
                write!(f, "Probability amplitude {} below minimum", amp)
            }
            BranchRouterError::DecoherenceTooLong(t) => {
                write!(f, "Decoherence time {}s exceeds maximum", t)
            }
            BranchRouterError::CatastrophicBranchRejected => {
                write!(f, "Catastrophic branch rejected (non-catastrophic alternatives exist)")
            }
            BranchRouterError::EmptyBranchPool => write!(f, "Branch pool is empty"),
            BranchRouterError::QRNGQualityInsufficient => {
                write!(f, "QRNG entropy quality below threshold")
            }
            BranchRouterError::InvalidSelectionInput => {
                write!(f, "Invalid input for branch selection")
            }
            BranchRouterError::BranchNotFound(id) => {
                write!(f, "Branch {} not found in pool", id)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let config = BranchRouterConfig::default();
        let router = BranchCollarRouter::new(config);
        assert_eq!(router.total_branches_processed, 0);
        assert!(router.current_branch().is_none());
    }

    #[test]
    fn test_register_branch() {
        let config = BranchRouterConfig::default();
        let mut router = BranchCollarRouter::new(config);

        let branch = QuantumBranch {
            branch_id: 12345,
            probability_amplitude: 0.5,
            expected_return: 1.0,
            is_catastrophic: false,
            decoherence_time: 1e-7,
        };

        assert!(router.register_branch(branch).is_ok());
    }

    #[test]
    fn test_amplitude_validation() {
        let config = BranchRouterConfig::default();
        let mut router = BranchCollarRouter::new(config);

        let branch = QuantumBranch {
            branch_id: 12345,
            probability_amplitude: 1e-15, // Too low
            expected_return: 1.0,
            is_catastrophic: false,
            decoherence_time: 1e-7,
        };

        assert_eq!(router.register_branch(branch), Err(BranchRouterError::AmplitudeTooLow(1e-15)));
    }

    #[test]
    fn test_branch_selection() {
        let config = BranchRouterConfig::default();
        let mut router = BranchCollarRouter::new(config);

        // Register some branches
        for i in 0..5 {
            let branch = QuantumBranch {
                branch_id: i as u128,
                probability_amplitude: 0.2,
                expected_return: (i as f64) * 0.5,
                is_catastrophic: false,
                decoherence_time: 1e-7,
            };
            router.register_branch(branch).unwrap();
        }

        // Select with QRNG entropy
        let entropy = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let result = router.select_branch(&entropy);
        assert!(result.is_ok());
        assert!(router.successful_selections > 0);
    }
}
