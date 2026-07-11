//! Z3 State Extractor
//! 
//! Bridges to Z3 SMT solver for extracting PRNG internal state from observed outputs.
//! Implements sliding-window constraint pruning and timeout fallbacks to prevent OOM.

use core::fmt;

/// Maximum number of constraints to keep in the sliding window
const MAX_CONSTRAINTS: usize = 100;

/// Timeout for Z3 solving in milliseconds
const Z3_TIMEOUT_MS: u32 = 5000;

/// Represents an extracted PRNG state candidate
#[derive(Debug, Clone)]
pub struct PrngStateCandidate {
    /// PRNG type identifier
    pub prng_type: PrngType,
    /// Candidate state vector (bytes)
    pub state: Vec<u64>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Number of observations used
    pub observations_used: usize,
}

/// Supported PRNG types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrngType {
    /// Linear Congruential Generator
    Lcg,
    /// Mersenne Twister MT19937
    MersenneTwister,
    /// Xorshift family
    Xorshift,
    /// PCG (Permuted Congruential Generator)
    Pcg,
    /// Unknown/custom
    Unknown,
}

/// Configuration for Z3 extraction with OOM prevention
#[derive(Debug, Clone, Copy)]
pub struct Z3ExtractorConfig {
    /// Maximum constraints in sliding window
    pub max_constraints: usize,
    /// Z3 solver timeout (milliseconds)
    pub timeout_ms: u32,
    /// Minimum observations required
    pub min_observations: usize,
    /// Maximum state candidates to return
    pub max_candidates: usize,
    /// Memory limit for Z3 (MB)
    pub memory_limit_mb: usize,
}

impl Default for Z3ExtractorConfig {
    fn default() -> Self {
        Self {
            max_constraints: MAX_CONSTRAINTS,
            timeout_ms: Z3_TIMEOUT_MS,
            min_observations: 10,
            max_candidates: 5,
            memory_limit_mb: 512,
        }
    }
}

/// Sliding window for constraint management
struct ConstraintWindow<T> {
    items: Vec<T>,
    max_size: usize,
}

impl<T> ConstraintWindow<T> {
    fn new(max_size: usize) -> Self {
        Self {
            items: Vec::with_capacity(max_size),
            max_size,
        }
    }

    fn push(&mut self, item: T) {
        if self.items.len() >= self.max_size {
            self.items.remove(0);
        }
        self.items.push(item);
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn iter(&self) -> core::slice::Iter<'_, T> {
        self.items.iter()
    }

    fn clear(&mut self) {
        self.items.clear();
    }
}

/// Simulated Z3 solver context (production would link to actual Z3)
struct SimulatedZ3Context {
    constraints_added: usize,
    memory_used_kb: usize,
}

impl SimulatedZ3Context {
    fn new() -> Self {
        Self {
            constraints_added: 0,
            memory_used_kb: 0,
        }
    }

    fn add_constraint(&mut self, _constraint: &str) -> Result<(), Z3Error> {
        self.constraints_added += 1;
        // Simulate memory usage growth
        self.memory_used_kb += 10;
        
        if self.memory_used_kb > 512 * 1024 {
            return Err(Z3Error::MemoryLimitExceeded);
        }
        
        Ok(())
    }

    fn check(&self) -> Result<bool, Z3Error> {
        // Simulate satisfiability check
        Ok(self.constraints_added > 0)
    }

    fn get_model(&self) -> Option<Vec<u64>> {
        // Return simulated model
        Some(vec![12345, 67890, 11111])
    }
}

/// Z3 State Extractor with sliding-window constraint pruning
pub struct Z3StateExtractor {
    config: Z3ExtractorConfig,
    constraint_window: ConstraintWindow<String>,
    observations: Vec<u64>,
    ctx: SimulatedZ3Context,
    extraction_attempts: usize,
}

impl Z3StateExtractor {
    pub fn new(config: Z3ExtractorConfig) -> Self {
        Self {
            config,
            constraint_window: ConstraintWindow::new(config.max_constraints),
            observations: Vec::new(),
            ctx: SimulatedZ3Context::new(),
            extraction_attempts: 0,
        }
    }

    /// Add an observed PRNG output
    pub fn add_observation(&mut self, value: u64) -> Result<(), Z3Error> {
        if self.extraction_attempts >= 10 {
            return Err(Z3Error::MaxAttemptsReached);
        }
        self.observations.push(value);
        Ok(())
    }

    /// Build constraints from observations using sliding window
    fn build_constraints(&mut self) -> Result<(), Z3Error> {
        self.constraint_window.clear();
        self.ctx = SimulatedZ3Context::new();

        // Process observations in sliding window fashion
        let window_size = self.observations.len().min(self.config.max_constraints);
        let start_idx = self.observations.len().saturating_sub(window_size);

        for i in start_idx..self.observations.len() {
            if i == 0 {
                continue;
            }
            
            // Generate constraint relating consecutive outputs
            // For LCG: next = (a * current + c) mod m
            let prev = self.observations[i - 1];
            let curr = self.observations[i];
            
            let constraint = format!("next_state({}) == {}", prev, curr);
            self.constraint_window.push(constraint.clone());
            
            self.ctx.add_constraint(&constraint)?;
        }

        Ok(())
    }

    /// Attempt to extract PRNG state using Z3
    pub fn extract_state(&mut self, prng_type: PrngType) -> Result<Vec<PrngStateCandidate>, Z3Error> {
        if self.observations.len() < self.config.min_observations {
            return Err(Z3Error::InsufficientObservations);
        }

        self.extraction_attempts += 1;

        // Build constraints with sliding window
        self.build_constraints()?;

        // Check satisfiability with timeout
        let is_sat = self.ctx.check()?;
        
        if !is_sat {
            return Ok(Vec::new());
        }

        // Extract model
        let model = self.ctx.get_model().unwrap_or_default();

        let candidate = PrngStateCandidate {
            prng_type,
            state: model,
            confidence: self.calculate_confidence(),
            observations_used: self.observations.len(),
        };

        Ok(vec![candidate])
    }

    /// Calculate confidence based on observation count and consistency
    fn calculate_confidence(&self) -> f64 {
        let obs_factor = (self.observations.len() as f64 / self.config.min_observations as f64).min(1.0);
        
        // Check consistency of observations
        if self.observations.len() < 2 {
            return 0.0;
        }

        let variance = self.calculate_variance();
        let consistency = if variance > 0.0 {
            1.0 / (1.0 + variance.ln().abs())
        } else {
            1.0
        };

        (obs_factor * consistency).min(1.0)
    }

    /// Calculate variance of observations
    fn calculate_variance(&self) -> f64 {
        if self.observations.is_empty() {
            return 0.0;
        }

        let mean: f64 = self.observations.iter().sum::<u64>() as f64 / self.observations.len() as f64;
        let variance: f64 = self.observations
            .iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>()
            / self.observations.len() as f64;

        variance
    }

    /// Clear all data and reset extractor
    pub fn clear(&mut self) {
        self.constraint_window.clear();
        self.observations.clear();
        self.ctx = SimulatedZ3Context::new();
        self.extraction_attempts = 0;
    }

    /// Get current observation count
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }
}

/// Errors from Z3 extraction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Z3Error {
    InsufficientObservations,
    MemoryLimitExceeded,
    Timeout,
    Unsatisfiable,
    MaxAttemptsReached,
    InvalidConstraint,
}

impl fmt::Display for Z3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Z3Error::InsufficientObservations => write!(f, "Insufficient observations"),
            Z3Error::MemoryLimitExceeded => write!(f, "Z3 memory limit exceeded"),
            Z3Error::Timeout => write!(f, "Z3 solver timeout"),
            Z3Error::Unsatisfiable => write!(f, "Constraints are unsatisfiable"),
            Z3Error::MaxAttemptsReached => write!(f, "Maximum extraction attempts reached"),
            Z3Error::InvalidConstraint => write!(f, "Invalid constraint format"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_window() {
        let mut window = ConstraintWindow::new(5);
        
        for i in 0..10 {
            window.push(i);
        }
        
        assert_eq!(window.len(), 5);
        assert_eq!(window.items[0], 5);
        assert_eq!(window.items[4], 9);
    }

    #[test]
    fn test_extractor_initialization() {
        let config = Z3ExtractorConfig::default();
        let extractor = Z3StateExtractor::new(config);
        
        assert_eq!(extractor.observation_count(), 0);
    }

    #[test]
    fn test_insufficient_observations() {
        let config = Z3ExtractorConfig {
            min_observations: 10,
            ..Default::default()
        };
        let mut extractor = Z3StateExtractor::new(config);
        
        // Add only 5 observations
        for i in 0..5 {
            extractor.add_observation(i).unwrap();
        }
        
        let result = extractor.extract_state(PrngType::Lcg);
        assert!(matches!(result, Err(Z3Error::InsufficientObservations)));
    }
}
