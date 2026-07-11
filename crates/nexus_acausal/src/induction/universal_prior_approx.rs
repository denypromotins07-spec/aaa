//! Universal Prior Approximation via Solomonoff Induction
//! 
//! Combines Levin Search with Kolmogorov complexity to approximate
//! the uncomputable Universal Prior for strategy prediction.

use crate::induction::levin_search::{LevinSearch, LevinSearchResult};
use crate::induction::kolmogorov_complexity::{KolmogorovComplexity, KolmogorovEstimate};

/// Minimum data size for prior estimation
const MIN_DATA_SIZE: usize = 32;

/// Maximum number of hypotheses in the prior distribution
const MAX_HYPOTHESES: usize = 1024;

/// Universal Prior approximation result
#[derive(Debug, Clone)]
pub struct UniversalPriorResult {
    /// Estimated prior probability distribution
    pub probabilities: Vec<f64>,
    /// Best hypothesis ID
    pub best_hypothesis: Option<usize>,
    /// Total probability mass accounted for
    pub total_mass: f64,
    /// Confidence in approximation
    pub confidence: f64,
}

/// Single hypothesis in the prior distribution
#[derive(Debug, Clone)]
pub struct PriorHypothesis {
    /// Unique identifier
    pub id: usize,
    /// Program/strategy representation
    pub program: Vec<u8>,
    /// Kolmogorov complexity estimate
    pub complexity: KolmogorovEstimate,
    /// Prior probability (2^(-complexity))
    pub prior_probability: f64,
    /// Posterior probability after observing data
    pub posterior_probability: f64,
}

/// Universal Prior Approximator using Solomonoff Induction
pub struct UniversalPriorApproximator {
    levin_search: LevinSearch,
    complexity_estimator: KolmogorovComplexity,
    hypotheses: Vec<PriorHypothesis>,
    next_id: usize,
    /// Normalization constant for prior distribution
    normalization_constant: f64,
}

impl UniversalPriorApproximator {
    /// Create a new universal prior approximator
    pub fn new() -> Self {
        Self {
            levin_search: LevinSearch::new(),
            complexity_estimator: KolmogorovComplexity::new(),
            hypotheses: Vec::with_capacity(64),
            next_id: 0,
            normalization_constant: 1.0,
        }
    }
    
    /// Add a hypothesis to the prior distribution
    pub fn add_hypothesis(&mut self, program: &[u8]) -> Result<usize, &'static str> {
        if self.hypotheses.len() >= MAX_HYPOTHESES {
            return Err("Maximum hypotheses count reached");
        }
        
        if program.is_empty() {
            return Err("Program cannot be empty");
        }
        
        // Estimate Kolmogorov complexity
        let complexity = self.complexity_estimator.estimate(program)?;
        
        // Calculate prior probability: P(h) ∝ 2^(-K(h))
        let prior_prob = 2.0_f64.powi(-(complexity.normalized * 20.0) as i32);
        
        let id = self.next_id;
        self.next_id += 1;
        
        let hypothesis = PriorHypothesis {
            id,
            program: program.to_vec(),
            complexity,
            prior_probability: prior_prob,
            posterior_probability: prior_prob, // Initially same as prior
        };
        
        self.hypotheses.push(hypothesis);
        
        // Register with Levin Search
        let _ = self.levin_search.register_hypothesis(program, complexity.normalized as usize);
        
        // Update normalization constant
        self.update_normalization();
        
        Ok(id)
    }
    
    /// Update normalization constant for proper probability distribution
    fn update_normalization(&mut self) {
        let total: f64 = self.hypotheses.iter().map(|h| h.prior_probability).sum();
        self.normalization_constant = if total > 0.0 { total } else { 1.0 };
    }
    
    /// Compute posterior distribution given observed data
    pub fn compute_posterior(&mut self, observed_data: &[u8]) -> Result<UniversalPriorResult, &'static str> {
        if observed_data.len() < MIN_DATA_SIZE {
            return Err("Insufficient data for posterior computation");
        }
        
        if self.hypotheses.is_empty() {
            return Ok(UniversalPriorResult {
                probabilities: vec![],
                best_hypothesis: None,
                total_mass: 0.0,
                confidence: 0.0,
            });
        }
        
        // Run Levin Search to find best matching hypotheses
        let search_result = self.levin_search.search(observed_data, 0);
        
        // Update posterior probabilities based on likelihood
        let mut total_posterior = 0.0;
        let mut best_id = None;
        let mut best_posterior = 0.0;
        
        for hypothesis in &mut self.hypotheses {
            // Simplified likelihood: based on program-data match quality
            let likelihood = self.compute_likelihood(&hypothesis.program, observed_data);
            
            // Bayes' theorem: P(h|d) ∝ P(d|h) * P(h)
            let posterior = likelihood * hypothesis.prior_probability;
            hypothesis.posterior_probability = posterior;
            
            total_posterior += posterior;
            
            if posterior > best_posterior {
                best_posterior = posterior;
                best_id = Some(hypothesis.id);
            }
        }
        
        // Normalize posteriors
        if total_posterior > 0.0 {
            for hypothesis in &mut self.hypotheses {
                hypothesis.posterior_probability /= total_posterior;
            }
        }
        
        // Build result
        let probabilities: Vec<f64> = self.hypotheses.iter().map(|h| h.posterior_probability).collect();
        
        let confidence = if let Some(best_id) = best_id {
            self.hypotheses.iter()
                .find(|h| h.id == best_id)
                .map(|h| h.posterior_probability)
                .unwrap_or(0.0)
        } else {
            0.0
        };
        
        Ok(UniversalPriorResult {
            probabilities,
            best_hypothesis: best_id,
            total_mass: total_posterior.min(1.0),
            confidence: confidence.clamp(0.0, 1.0),
        })
    }
    
    /// Compute likelihood of data given hypothesis
    fn compute_likelihood(&self, program: &[u8], data: &[u8]) -> f64 {
        // Simplified likelihood based on pattern matching
        // In production, this would execute the program and compare output
        
        if program.is_empty() || data.is_empty() {
            return 0.0;
        }
        
        let min_len = program.len().min(data.len());
        let mut matches = 0usize;
        
        for i in 0..min_len {
            if program[i] == data[i] {
                matches += 1;
            }
        }
        
        matches as f64 / min_len as f64
    }
    
    /// Get the prior probability of a specific hypothesis
    pub fn get_prior_probability(&self, hypothesis_id: usize) -> Option<f64> {
        self.hypotheses.iter()
            .find(|h| h.id == hypothesis_id)
            .map(|h| h.prior_probability / self.normalization_constant)
    }
    
    /// Get the posterior probability of a specific hypothesis
    pub fn get_posterior_probability(&self, hypothesis_id: usize) -> Option<f64> {
        self.hypotheses.iter()
            .find(|h| h.id == hypothesis_id)
            .map(|h| h.posterior_probability)
    }
    
    /// Get all hypotheses sorted by posterior probability
    pub fn get_sorted_hypotheses(&self) -> Vec<&PriorHypothesis> {
        let mut sorted: Vec<&PriorHypothesis> = self.hypotheses.iter().collect();
        sorted.sort_by(|a, b| b.posterior_probability.partial_cmp(&a.posterior_probability).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }
    
    /// Clear all hypotheses and reset state
    pub fn clear(&mut self) {
        self.hypotheses.clear();
        self.levin_search.clear();
        self.next_id = 0;
        self.normalization_constant = 1.0;
    }
    
    /// Get number of registered hypotheses
    pub fn hypothesis_count(&self) -> usize {
        self.hypotheses.len()
    }
}

impl Default for UniversalPriorApproximator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_approximator_creation() {
        let approximator = UniversalPriorApproximator::new();
        assert_eq!(approximator.hypothesis_count(), 0);
    }
    
    #[test]
    fn test_add_hypothesis() {
        let mut approximator = UniversalPriorApproximator::new();
        let program = b"\x01\x02\x03\x04\x05\x06\x07\x08";
        
        let result = approximator.add_hypothesis(program);
        assert!(result.is_ok());
        assert_eq!(approximator.hypothesis_count(), 1);
    }
    
    #[test]
    fn test_empty_program_rejected() {
        let mut approximator = UniversalPriorApproximator::new();
        let result = approximator.add_hypothesis(&[]);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_max_hypotheses_limit() {
        let mut approximator = UniversalPriorApproximator::new();
        
        // Try to add more than MAX_HYPOTHESES
        for _ in 0..MAX_HYPOTHESES + 10 {
            let _ = approximator.add_hypothesis(b"test_program_data");
        }
        
        assert_eq!(approximator.hypothesis_count(), MAX_HYPOTHESES);
    }
    
    #[test]
    fn test_prior_probability_sum() {
        let mut approximator = UniversalPriorApproximator::new();
        
        // Add several hypotheses
        for i in 0..10 {
            let program: Vec<u8> = (0..20).map(|j| ((i + j) % 256) as u8).collect();
            let _ = approximator.add_hypothesis(&program);
        }
        
        // Sum of normalized priors should be close to 1.0
        let total: f64 = (0..10)
            .filter_map(|i| approximator.get_prior_probability(i))
            .sum();
        
        assert!(total > 0.9 && total <= 1.1); // Allow small floating point error
    }
    
    #[test]
    fn test_posterior_computation() {
        let mut approximator = UniversalPriorApproximator::new();
        
        // Add hypotheses
        let _ = approximator.add_hypothesis(b"aaaaaaaaaa");
        let _ = approximator.add_hypothesis(b"bbbbbbbbbb");
        let _ = approximator.add_hypothesis(b"cccccccccc");
        
        // Compute posterior with observed data
        let data = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let result = approximator.compute_posterior(data);
        
        assert!(result.is_ok());
        let posterior = result.unwrap();
        
        assert!(!posterior.probabilities.is_empty());
        assert!(posterior.confidence > 0.0);
    }
}
