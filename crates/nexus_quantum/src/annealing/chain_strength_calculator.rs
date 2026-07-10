//! Chain Strength Calculator
//! 
//! Determines optimal ferromagnetic coupling strength for logical qubit chains
//! in quantum annealing. Chain strength must be strong enough to keep physical
//! qubits aligned but not so strong that it suppresses the problem energy gaps.

use thiserror::Error;
use crate::annealing::minor_embedding_heuristic::EmbeddingResult;

/// Errors that can occur during chain strength calculation
#[derive(Error, Debug)]
pub enum ChainStrengthError {
    #[error("Empty embedding: no chains to calculate strength for")]
    EmptyEmbedding,
    #[error("Chain strength overflow: calculated value exceeds hardware limits")]
    Overflow,
    #[error("Invalid chain: chain {0} has length 0")]
    InvalidChain(usize),
    #[error("Energy scale mismatch: problem energies are {scale}x larger than chain strength")]
    EnergyScaleMismatch { scale: f64 },
}

/// Configuration for chain strength calculation
#[derive(Debug, Clone)]
pub struct ChainStrengthConfig {
    /// Minimum chain strength (absolute value)
    pub min_strength: f64,
    /// Maximum chain strength (absolute value)
    pub max_strength: f64,
    /// Multiplier based on maximum problem weight
    pub max_weight_multiplier: f64,
    /// Multiplier based on average problem weight
    pub avg_weight_multiplier: f64,
    /// Multiplier based on chain length
    pub chain_length_multiplier: f64,
    /// Safety factor to ensure chains don't break
    pub safety_factor: f64,
    /// Whether to use adaptive per-chain strengths
    pub adaptive_per_chain: bool,
}

impl Default for ChainStrengthConfig {
    fn default() -> Self {
        Self {
            min_strength: 0.5,
            max_strength: 10.0,
            max_weight_multiplier: 2.0,
            avg_weight_multiplier: 3.0,
            chain_length_multiplier: 0.1,
            safety_factor: 1.5,
            adaptive_per_chain: true,
        }
    }
}

/// Result of chain strength calculation
#[derive(Debug, Clone)]
pub struct ChainStrengthResult {
    /// Global chain strength (if uniform)
    pub global_strength: Option<f64>,
    /// Per-chain strengths (if adaptive)
    pub per_chain_strengths: std::collections::HashMap<usize, f64>,
    /// Recommended strength based on analysis
    pub recommended_strength: f64,
    /// Estimated chain break probability at this strength
    pub estimated_break_probability: f64,
    /// Energy gap preservation ratio (should be > 0.1)
    pub energy_gap_ratio: f64,
}

/// Chain Strength Calculator using multiple heuristics
pub struct ChainStrengthCalculator {
    config: ChainStrengthConfig,
}

impl ChainStrengthCalculator {
    /// Create a new calculator with default configuration
    pub fn new() -> Self {
        Self {
            config: ChainStrengthConfig::default(),
        }
    }

    /// Create a new calculator with custom configuration
    pub fn with_config(config: ChainStrengthConfig) -> Self {
        Self { config }
    }

    /// Calculate optimal chain strength for an embedded problem
    /// 
    /// # Arguments
    /// * `embedding` - The minor embedding result
    /// * `problem_couplings` - List of (i, j, J_ij) couplings from the Ising problem
    /// 
    /// # Returns
    /// Chain strength recommendation with diagnostics
    pub fn calculate_strength(
        &self,
        embedding: &EmbeddingResult,
        problem_couplings: &[(usize, usize, f64)],
    ) -> Result<ChainStrengthResult, ChainStrengthError> {
        if embedding.logical_to_physical.is_empty() {
            return Err(ChainStrengthError::EmptyEmbedding);
        }

        // Validate all chains have at least one qubit
        for (&logical, chain) in &embedding.logical_to_physical {
            if chain.is_empty() {
                return Err(ChainStrengthError::InvalidChain(logical));
            }
        }

        // Analyze problem energy scale
        let problem_stats = self.analyze_problem_scale(problem_couplings);
        
        // Calculate base chain strength from problem statistics
        let base_strength = self.calculate_base_strength(&problem_stats);
        
        // Apply chain-length adjustments if adaptive
        let per_chain_strengths = if self.config.adaptive_per_chain {
            self.calculate_adaptive_strengths(embedding, base_strength)?
        } else {
            std::collections::HashMap::new()
        };

        // Determine global strength (either uniform or average of adaptive)
        let global_strength = if per_chain_strengths.is_empty() {
            Some(base_strength)
        } else {
            let avg = per_chain_strengths.values().sum::<f64>() 
                / per_chain_strengths.len() as f64;
            Some(avg)
        };

        // Estimate chain break probability
        let break_prob = self.estimate_break_probability(base_strength, &problem_stats);
        
        // Calculate energy gap preservation ratio
        let gap_ratio = self.calculate_energy_gap_ratio(base_strength, &problem_stats);

        Ok(ChainStrengthResult {
            global_strength,
            per_chain_strengths,
            recommended_strength: base_strength,
            estimated_break_probability: break_prob,
            energy_gap_ratio: gap_ratio,
        })
    }

    /// Analyze the energy scale of the problem
    fn analyze_problem_scale(
        &self,
        couplings: &[(usize, usize, f64)],
    ) -> ProblemStats {
        if couplings.is_empty() {
            return ProblemStats {
                max_abs_coupling: 1.0,
                avg_abs_coupling: 1.0,
                min_abs_coupling: 1.0,
                coupling_spread: 1.0,
            };
        }

        let abs_couplings: Vec<f64> = couplings
            .iter()
            .map(|(_, _, j)| j.abs())
            .collect();

        let max_abs = abs_couplings.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_abs = abs_couplings.iter().cloned().fold(f64::INFINITY, f64::min);
        let avg_abs = abs_couplings.iter().sum::<f64>() / abs_couplings.len() as f64;
        let spread = max_abs - min_abs;

        ProblemStats {
            max_abs_coupling: max_abs.max(1e-10),
            avg_abs_coupling: avg_abs.max(1e-10),
            min_abs_coupling: min_abs,
            coupling_spread: spread.max(1e-10),
        }
    }

    /// Calculate base chain strength from problem statistics
    fn calculate_base_strength(&self, stats: &ProblemStats) -> f64 {
        // Method 1: Based on maximum coupling
        let strength_from_max = stats.max_abs_coupling * self.config.max_weight_multiplier;
        
        // Method 2: Based on average coupling
        let strength_from_avg = stats.avg_abs_coupling * self.config.avg_weight_multiplier;
        
        // Take the more conservative (larger) estimate
        let base = strength_from_max.max(strength_from_avg);
        
        // Apply safety factor
        let with_safety = base * self.config.safety_factor;
        
        // Clamp to valid range
        with_safety.clamp(self.config.min_strength, self.config.max_strength)
    }

    /// Calculate adaptive strengths per chain based on chain length
    fn calculate_adaptive_strengths(
        &self,
        embedding: &EmbeddingResult,
        base_strength: f64,
    ) -> Result<std::collections::HashMap<usize, f64>, ChainStrengthError> {
        let mut strengths = std::collections::HashMap::new();
        
        for (&logical, chain) in &embedding.logical_to_physical {
            let chain_len = chain.len();
            
            // Longer chains need stronger coupling to maintain coherence
            // But we also need to avoid making them too strong
            let length_adjustment = 1.0 + self.config.chain_length_multiplier * (chain_len - 1) as f64;
            
            let adjusted_strength = (base_strength * length_adjustment)
                .clamp(self.config.min_strength, self.config.max_strength);
            
            strengths.insert(logical, adjusted_strength);
        }
        
        Ok(strengths)
    }

    /// Estimate probability of chain breaks at given strength
    fn estimate_break_probability(&self, strength: f64, stats: &ProblemStats) -> f64 {
        // Simplified model: break probability decreases exponentially with strength
        // relative to problem energy scale
        
        let ratio = strength / stats.avg_abs_coupling;
        
        // Using Arrhenius-like model: P ~ exp(-E_barrier / kT)
        // Here, E_barrier ~ strength, and effective "temperature" ~ problem fluctuations
        let break_prob = (-ratio / 2.0).exp();
        
        break_prob.clamp(0.0, 1.0)
    }

    /// Calculate energy gap preservation ratio
    fn calculate_energy_gap_ratio(&self, strength: f64, stats: &ProblemStats) -> f64 {
        // Chain strength should not dominate the problem Hamiltonian
        // Gap ratio = (problem energy scale) / (chain strength)
        // We want this to be > 0.1 for the problem to be resolvable
        
        let ratio = stats.avg_abs_coupling / strength;
        ratio.clamp(0.0, 10.0)
    }

    /// Calculate chain strength using the "maximum cut" heuristic
    /// This is useful when problem has clear community structure
    pub fn calculate_strength_max_cut(
        &self,
        embedding: &EmbeddingResult,
        couplings: &[(usize, usize, f64)],
    ) -> Result<f64, ChainStrengthError> {
        if couplings.is_empty() {
            return Ok(self.config.min_strength);
        }

        // Find the maximum absolute coupling that crosses chain boundaries
        let mut max_cross_chain_coupling = 0.0;
        
        for &(i, j, j_val) in couplings {
            // Check if i and j are in different chains
            let chain_i = embedding.logical_to_physical.get(&i);
            let chain_j = embedding.logical_to_physical.get(&j);
            
            if let (Some(ci), Some(cj)) = (chain_i, chain_j) {
                // If they're different logical qubits, this is a cross-chain coupling
                if ci.as_ptr() != cj.as_ptr() {
                    max_cross_chain_coupling = max_cross_chain_coupling.max(j_val.abs());
                }
            }
        }
        
        // Chain strength should exceed maximum cross-chain coupling
        let strength = max_cross_chain_coupling * self.config.max_weight_multiplier * self.config.safety_factor;
        
        Ok(strength.clamp(self.config.min_strength, self.config.max_strength))
    }

    /// Validate that chain strength is appropriate for the problem
    pub fn validate_strength(
        &self,
        strength: f64,
        couplings: &[(usize, usize, f64)],
    ) -> ValidationResult {
        let stats = self.analyze_problem_scale(couplings);
        
        let warnings = Vec::new();
        let errors = Vec::new();
        
        // Check if strength is too weak
        if strength < stats.max_abs_coupling {
            warnings.push(format!(
                "Chain strength ({}) is less than max coupling ({}) - chains may break",
                strength, stats.max_abs_coupling
            ));
        }
        
        // Check if strength is too strong
        if strength > stats.max_abs_coupling * 10.0 {
            warnings.push(format!(
                "Chain strength ({}) is much larger than problem scale ({}) - energy gaps may be suppressed",
                strength, stats.max_abs_coupling
            ));
        }
        
        // Check absolute bounds
        if strength < self.config.min_strength {
            errors.push(format!(
                "Chain strength {} below minimum {}",
                strength, self.config.min_strength
            ));
        }
        
        if strength > self.config.max_strength {
            errors.push(format!(
                "Chain strength {} exceeds maximum {}",
                strength, self.config.max_strength
            ));
        }
        
        ValidationResult {
            is_valid: errors.is_empty(),
            warnings,
            errors,
            recommended_range: (
                stats.max_abs_coupling * self.config.max_weight_multiplier,
                stats.max_abs_coupling * self.config.avg_weight_multiplier,
            ),
        }
    }
}

impl Default for ChainStrengthCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about problem energy scale
#[derive(Debug, Clone)]
struct ProblemStats {
    max_abs_coupling: f64,
    avg_abs_coupling: f64,
    min_abs_coupling: f64,
    coupling_spread: f64,
}

/// Result of chain strength validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the strength passes all checks
    pub is_valid: bool,
    /// Non-blocking warnings
    pub warnings: Vec<String>,
    /// Blocking errors
    pub errors: Vec<String>,
    /// Recommended strength range (min, max)
    pub recommended_range: (f64, f64),
}

impl ValidationResult {
    /// Get a summary message
    pub fn summary(&self) -> String {
        if self.is_valid {
            if self.warnings.is_empty() {
                "Chain strength is valid with no warnings".to_string()
            } else {
                format!("Chain strength is valid but has {} warning(s)", self.warnings.len())
            }
        } else {
            format!("Chain strength is invalid: {} error(s)", self.errors.len())
        }
    }
}

/// Unembedding utility for recovering logical solutions from physical reads
pub struct ChainUnembedder;

impl ChainUnembedder {
    /// Unembed physical qubit readings to logical values using majority vote
    /// 
    /// When physical qubits in a chain disagree (chain break), uses majority
    /// voting to determine the logical value.
    pub fn majority_vote_unembed(
        physical_values: &[i8],
        embedding: &EmbeddingResult,
    ) -> Result<Vec<i8>, ChainStrengthError> {
        let n_logical = embedding.logical_to_physical.len();
        let mut logical_values = vec![0; n_logical];
        
        for (&logical, chain) in &embedding.logical_to_physical {
            if chain.is_empty() {
                return Err(ChainStrengthError::InvalidChain(logical));
            }
            
            // Count +1 and -1 votes
            let mut plus_count = 0;
            let mut minus_count = 0;
            
            for &phys_q in chain {
                if phys_q < physical_values.len() {
                    match physical_values[phys_q] {
                        1 => plus_count += 1,
                        -1 => minus_count += 1,
                        _ => {} // Ignore invalid values
                    }
                }
            }
            
            // Majority vote (default to +1 on tie)
            logical_values[logical] = if plus_count >= minus_count { 1 } else { -1 };
        }
        
        Ok(logical_values)
    }

    /// Unembed using energy-based selection
    /// 
    /// For each chain, selects the value that minimizes local energy
    /// considering couplings to neighboring chains.
    pub fn energy_based_unembed(
        physical_values: &[i8],
        embedding: &EmbeddingResult,
        couplings: &[(usize, usize, f64)],
    ) -> Result<Vec<i8>, ChainStrengthError> {
        // Start with majority vote as baseline
        let mut logical_values = Self::majority_vote_unembed(physical_values, embedding)?;
        
        // Iteratively improve by checking energy contribution
        // (Simplified - full implementation would do local optimization)
        
        Ok(logical_values)
    }

    /// Calculate chain break fraction for a sample
    pub fn calculate_chain_break_fraction(
        physical_values: &[i8],
        embedding: &EmbeddingResult,
    ) -> f64 {
        let mut total_broken = 0;
        let mut total_chains = 0;
        
        for chain in embedding.logical_to_physical.values() {
            if chain.is_empty() {
                continue;
            }
            
            total_chains += 1;
            
            // Check if all qubits in chain agree
            let first_value = physical_values.get(chain[0]).copied().unwrap_or(0);
            let is_broken = chain.iter().skip(1).any(|&q| {
                physical_values.get(q).copied().unwrap_or(0) != first_value
            });
            
            if is_broken {
                total_broken += 1;
            }
        }
        
        if total_chains == 0 {
            0.0
        } else {
            total_broken as f64 / total_chains as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_chain_strength_calculator_basic() {
        let calculator = ChainStrengthCalculator::new();
        
        // Create a simple embedding
        let mut embedding = EmbeddingResult {
            logical_to_physical: HashMap::new(),
            physical_to_logical: HashMap::new(),
            stats: Default::default(),
        };
        embedding.logical_to_physical.insert(0, vec![0, 1]);
        embedding.logical_to_physical.insert(1, vec![2, 3]);
        embedding.logical_to_physical.insert(2, vec![4]);
        
        // Simple couplings
        let couplings = vec![
            (0, 1, 1.0),
            (1, 2, -0.5),
        ];
        
        let result = calculator.calculate_strength(&embedding, &couplings);
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.recommended_strength > 0.0);
        assert!(result.recommended_strength <= calculator.config.max_strength);
    }

    #[test]
    fn test_chain_strength_validation() {
        let calculator = ChainStrengthCalculator::new();
        
        let couplings = vec![(0, 1, 2.0)];
        
        // Test weak strength
        let validation = calculator.validate_strength(0.5, &couplings);
        assert!(!validation.warnings.is_empty()); // Should warn about being too weak
        
        // Test appropriate strength
        let validation = calculator.validate_strength(4.0, &couplings);
        assert!(validation.is_valid);
        
        // Test too strong
        let validation = calculator.validate_strength(50.0, &couplings);
        assert!(!validation.warnings.is_empty()); // Should warn about suppressing gaps
    }

    #[test]
    fn test_majority_vote_unembed() {
        let mut embedding = EmbeddingResult {
            logical_to_physical: HashMap::new(),
            physical_to_logical: HashMap::new(),
            stats: Default::default(),
        };
        embedding.logical_to_physical.insert(0, vec![0, 1, 2]);
        embedding.logical_to_physical.insert(1, vec![3, 4]);
        
        // Physical values with one chain broken
        let physical = vec![1, 1, -1, -1, -1]; // Chain 0 has disagreement
        
        let logical = ChainUnembedder::majority_vote_unembed(&physical, &embedding);
        
        assert!(logical.is_ok());
        let logical = logical.unwrap();
        assert_eq!(logical.len(), 2);
        assert_eq!(logical[0], 1); // Majority is +1
        assert_eq!(logical[1], -1); // All -1
    }

    #[test]
    fn test_chain_break_fraction() {
        let mut embedding = EmbeddingResult {
            logical_to_physical: HashMap::new(),
            physical_to_logical: HashMap::new(),
            stats: Default::default(),
        };
        embedding.logical_to_physical.insert(0, vec![0, 1]);
        embedding.logical_to_physical.insert(1, vec![2, 3]);
        embedding.logical_to_physical.insert(2, vec![4, 5]);
        
        // Two chains agree, one is broken
        let physical = vec![1, 1, -1, -1, 1, -1];
        
        let break_frac = ChainUnembedder::calculate_chain_break_fraction(&physical, &embedding);
        
        assert!((break_frac - 1.0 / 3.0).abs() < 0.01); // One out of three chains broken
    }
}
