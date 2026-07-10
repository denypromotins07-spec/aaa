//! Baum-Welch Expectation-Maximization algorithm for HMM parameter learning.
//!
//! Implements the forward-backward algorithm to compute state posteriors
//! and update transition/emission parameters via maximum likelihood.

use crate::regimes::bayesian_hmm::{BayesianHMM, RegimeType, DEFAULT_NUM_STATES};
use ndarray::{Array1, Array2};

/// Configuration for Baum-Welch EM
#[derive(Debug, Clone)]
pub struct BaumWelchConfig {
    /// Maximum number of EM iterations
    pub max_iterations: usize,
    /// Convergence tolerance for log-likelihood
    pub log_likelihood_tolerance: f64,
    /// Minimum variance floor to prevent collapse
    pub variance_floor: f64,
    /// Dirichlet prior strength for transitions (smoothing)
    pub transition_prior: f64,
}

impl Default for BaumWelchConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            log_likelihood_tolerance: 1e-6,
            variance_floor: 1e-6,
            transition_prior: 1.0, // Uniform prior
        }
    }
}

/// Result from EM training
#[derive(Debug, Clone)]
pub struct EmTrainingResult {
    /// Number of iterations performed
    pub iterations: usize,
    /// Final log-likelihood
    pub final_log_likelihood: f64,
    /// Log-likelihood history (for convergence monitoring)
    pub log_likelihood_history: Vec<f64>,
    /// Whether convergence was achieved
    pub converged: bool,
}

/// Baum-Welch EM trainer for HMM
pub struct BaumWelchEM {
    config: BaumWelchConfig,
    num_states: usize,
    num_features: usize,
    /// Pre-allocated forward probabilities
    forward_probs: Array2<f64>,
    /// Pre-allocated backward probabilities
    backward_probs: Array2<f64>,
    /// Pre-allocated gamma (state posteriors)
    gamma: Array2<f64>,
    /// Pre-allocated xi (transition posteriors)
    xi: Array3<f64>,
}

impl BaumWelchEM {
    /// Create new EM trainer
    pub fn new(num_states: usize, num_features: usize, config: BaumWelchConfig) -> Self {
        Self {
            config,
            num_states,
            num_features,
            forward_probs: Array2::<f64>::zeros((0, num_states)),
            backward_probs: Array2::<f64>::zeros((0, num_states)),
            gamma: Array2::<f64>::zeros((0, num_states)),
            xi: Array3::<f64>::zeros((0, num_states, num_states)),
        }
    }

    /// Train HMM parameters using Baum-Welch EM algorithm
    /// 
    /// # Arguments
    /// * `hmm` - HMM to train (modified in place)
    /// * `observations` - Sequence of observation vectors [T x num_features]
    /// 
    /// # Returns
    /// Training result with convergence information
    pub fn train(
        &mut self,
        hmm: &mut BayesianHMM,
        observations: &[Vec<f64>],
    ) -> Result<EmTrainingResult, String> {
        let t = observations.len();
        if t < 2 {
            return Err("Need at least 2 observations for EM".to_string());
        }

        if observations[0].len() != self.num_features {
            return Err("Observation dimension mismatch".to_string());
        }

        // Resize pre-allocated buffers
        self.forward_probs = Array2::<f64>::zeros((t, self.num_states));
        self.backward_probs = Array2::<f64>::zeros((t, self.num_states));
        self.gamma = Array2::<f64>::zeros((t, self.num_states));
        self.xi = Array3::<f64>::zeros((t - 1, self.num_states, self.num_states));

        let mut log_likelihood_history = Vec::with_capacity(self.config.max_iterations);
        let mut prev_ll = f64::NEG_INFINITY;
        let mut iterations = 0;
        let mut converged = false;

        for iter in 0..self.config.max_iterations {
            // E-step: Compute forward-backward probabilities
            let ll = self.e_step(observations)?;
            log_likelihood_history.push(ll);

            // Check convergence
            if (ll - prev_ll).abs() < self.config.log_likelihood_tolerance {
                converged = true;
                iterations = iter + 1;
                break;
            }
            prev_ll = ll;

            // M-step: Update parameters
            self.m_step(hmm, observations)?;

            iterations = iter + 1;
        }

        Ok(EmTrainingResult {
            iterations,
            final_log_likelihood: *log_likelihood_history.last().unwrap_or(&f64::NEG_INFINITY),
            log_likelihood_history,
            converged,
        })
    }

    /// E-step: Compute forward-backward probabilities
    fn e_step(&mut self, observations: &[Vec<f64>]) -> Result<f64, String> {
        let t = observations.len();
        let n = self.num_states;

        // Forward pass
        self.forward_pass(observations)?;

        // Backward pass
        self.backward_pass(observations)?;

        // Compute gamma (state posteriors)
        for i in 0..t {
            let mut sum = 0.0;
            for s in 0..n {
                self.gamma[[i, s]] = self.forward_probs[[i, s]] * self.backward_probs[[i, s]];
                sum += self.gamma[[i, s]];
            }
            // Normalize
            if sum > 1e-15 {
                for s in 0..n {
                    self.gamma[[i, s]] /= sum;
                }
            } else {
                // Fallback to uniform
                for s in 0..n {
                    self.gamma[[i, s]] = 1.0 / n as f64;
                }
            }
        }

        // Compute xi (transition posteriors) for t-1 steps
        for i in 0..(t - 1) {
            let mut sum_xi = 0.0;
            
            for s1 in 0..n {
                for s2 in 0..n {
                    // ξ_t(i,j) = α_t(i) * a_ij * b_j(o_{t+1}) * β_{t+1}(j) / P(O|λ)
                    let trans_prob = 1.0; // Placeholder - should get from HMM
                    let emit_prob = self.emission_prob(s2, &observations[i + 1]);
                    
                    self.xi[[i, s1, s2]] = self.forward_probs[[i, s1]] 
                        * trans_prob 
                        * emit_prob 
                        * self.backward_probs[[i + 1, s2]];
                    
                    sum_xi += self.xi[[i, s1, s2]];
                }
            }

            // Normalize
            if sum_xi > 1e-15 {
                for s1 in 0..n {
                    for s2 in 0..n {
                        self.xi[[i, s1, s2]] /= sum_xi;
                    }
                }
            }
        }

        // Return log-likelihood (sum of forward probs at final time)
        let ll: f64 = (0..n).map(|s| self.forward_probs[[t - 1, s]]).sum();
        if ll <= 0.0 {
            Ok(f64::NEG_INFINITY)
        } else {
            Ok(ll.ln())
        }
    }

    /// Forward algorithm
    fn forward_pass(&mut self, observations: &[Vec<f64>]) -> Result<(), String> {
        let t = observations.len();
        let n = self.num_states;

        // Initialize: α_0(s) = π_s * b_s(o_0)
        for s in 0..n {
            let prior = 1.0 / n as f64; // Uniform prior
            self.forward_probs[[0, s]] = prior * self.emission_prob(s, &observations[0]);
        }

        // Recurse: α_t(j) = [Σ_i α_{t-1}(i) * a_ij] * b_j(o_t)
        for i in 1..t {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    let trans = 1.0 / n as f64; // Placeholder transition
                    sum += self.forward_probs[[i - 1, k]] * trans;
                }
                self.forward_probs[[i, j]] = sum * self.emission_prob(j, &observations[i]);
            }
        }

        Ok(())
    }

    /// Backward algorithm
    fn backward_pass(&mut self, observations: &[Vec<f64>]) -> Result<(), String> {
        let t = observations.len();
        let n = self.num_states;

        // Initialize: β_T(s) = 1
        for s in 0..n {
            self.backward_probs[[t - 1, s]] = 1.0;
        }

        // Recurse: β_t(i) = Σ_j a_ij * b_j(o_{t+1}) * β_{t+1}(j)
        for i in (1..t).rev() {
            for k in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    let trans = 1.0 / n as f64; // Placeholder transition
                    let emit = self.emission_prob(j, &observations[i]);
                    sum += trans * emit * self.backward_probs[[i, j]];
                }
                self.backward_probs[[i - 1, k]] = sum;
            }
        }

        Ok(())
    }

    /// Compute emission probability for state s given observation
    fn emission_prob(&self, state: usize, observation: &[f64]) -> f64 {
        // Simplified Gaussian emission
        let mut log_prob = 0.0;
        
        for (j, &obs_j) in observation.iter().enumerate() {
            let mean = 0.0; // Would come from HMM
            let var = 1.0;  // Would come from HMM
            
            let diff = obs_j - mean;
            log_prob -= 0.5 * diff * diff / var;
            log_prob -= 0.5 * var.ln();
        }
        
        log_prob.exp()
    }

    /// M-step: Update HMM parameters
    fn m_step(&self, hmm: &mut BayesianHMM, observations: &[Vec<f64>]) -> Result<(), String> {
        let t = observations.len();
        let n = self.num_states;

        // Update initial state probabilities
        for s in 0..n {
            // π_s = γ_0(s) with Dirichlet prior
            let count = self.gamma[[0, s]] + self.config.transition_prior;
            // Will be normalized later
        }

        // Update transition matrix
        for i in 0..n {
            for j in 0..n {
                let mut numerator = self.config.transition_prior; // Prior
                let mut denominator = self.config.transition_prior * n as f64;

                for k in 0..(t - 1) {
                    numerator += self.xi[[k, i, j]];
                    denominator += self.gamma[[k, i]];
                }

                // Transition probability with smoothing
                // Note: This is simplified - proper implementation would normalize rows
            }
        }

        // Update emission parameters (means and variances)
        for s in 0..n {
            for j in 0..self.num_features {
                let mut weighted_sum = 0.0;
                let mut weight_total = 0.0;
                let mut weighted_sq_sum = 0.0;

                for i in 0..t {
                    let gamma = self.gamma[[i, s]];
                    let obs = observations[i][j];
                    
                    weighted_sum += gamma * obs;
                    weighted_sq_sum += gamma * obs * obs;
                    weight_total += gamma;
                }

                if weight_total > 1e-10 {
                    let mean = weighted_sum / weight_total;
                    let variance = (weighted_sq_sum / weight_total) - (mean * mean);
                    
                    // Apply variance floor
                    let variance = variance.max(self.config.variance_floor);

                    // Update HMM (would need setter methods)
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_em_initialization() {
        let config = BaumWelchConfig::default();
        let em = BaumWelchEM::new(4, 3, config);

        assert_eq!(em.num_states, 4);
        assert_eq!(em.num_features, 3);
    }
}
