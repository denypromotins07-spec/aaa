//! Bayesian Hidden Markov Model for macro regime detection.
//!
//! Implements a continuous-time HMM to detect latent market regimes:
//! - Risk-On (bull markets, low volatility)
//! - Risk-Off (flight to safety, high volatility)  
//! - Stagflation (high inflation, low growth)
//! - Goldilocks (stable growth, moderate inflation)

use ndarray::{Array1, Array2};

/// Number of hidden states (configurable)
pub const DEFAULT_NUM_STATES: usize = 4;

/// Market regime types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegimeType {
    RiskOn,
    RiskOff,
    Stagflation,
    Goldilocks,
    Unknown,
}

impl RegimeType {
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => RegimeType::RiskOn,
            1 => RegimeType::RiskOff,
            2 => RegimeType::Stagflation,
            3 => RegimeType::Goldilocks,
            _ => RegimeType::Unknown,
        }
    }

    pub fn to_index(self) -> usize {
        match self {
            RegimeType::RiskOn => 0,
            RegimeType::RiskOff => 1,
            RegimeType::Stagflation => 2,
            RegimeType::Goldilocks => 3,
            RegimeType::Unknown => usize::MAX,
        }
    }
}

/// HMM state posterior probabilities
#[derive(Debug, Clone)]
pub struct RegimePosterior {
    /// Probability of each regime
    pub probabilities: Array1<f64>,
    /// Most likely regime
    pub dominant_regime: RegimeType,
    /// Entropy of distribution (uncertainty measure)
    pub entropy: f64,
}

impl RegimePosterior {
    pub fn new(num_states: usize) -> Self {
        Self {
            probabilities: Array1::<f64>::zeros(num_states),
            dominant_regime: RegimeType::Unknown,
            entropy: 0.0,
        }
    }
}

/// Bayesian Hidden Markov Model
pub struct BayesianHMM {
    num_states: usize,
    num_features: usize,
    /// Transition matrix P(state_j | state_i)
    transition_matrix: Array2<f64>,
    /// Emission parameters (mean, variance) for each state/feature
    emission_means: Array2<f64>,
    emission_variances: Array2<f64>,
    /// Prior state probabilities
    prior_probabilities: Array1<f64>,
    /// Current filtered state probabilities
    current_state_probs: Array1<f64>,
}

impl BayesianHMM {
    /// Create new HMM with specified dimensions
    pub fn new(num_states: usize, num_features: usize) -> Self {
        let mut model = Self {
            num_states,
            num_features,
            transition_matrix: Array2::<f64>::eye(num_states),
            emission_means: Array2::<f64>::zeros((num_states, num_features)),
            emission_variances: Array2::<f64>::ones((num_states, num_features)),
            prior_probabilities: Array1::<f64>::from_elem(num_states, 1.0 / num_states as f64),
            current_state_probs: Array1::<f64>::from_elem(num_states, 1.0 / num_states as f64),
        };

        // Initialize with small off-diagonal transitions (regimes are persistent)
        model.initialize_transition_matrix();
        model
    }

    /// Initialize transition matrix with regime persistence
    fn initialize_transition_matrix(&mut self) {
        let n = self.num_states;
        let persist_prob = 0.95; // High probability of staying in same regime
        let switch_prob = (1.0 - persist_prob) / (n - 1) as f64;

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    self.transition_matrix[[i, j]] = persist_prob;
                } else {
                    self.transition_matrix[[i, j]] = switch_prob;
                }
            }
        }
    }

    /// Set emission parameters for a specific regime
    pub fn set_emission_params(
        &mut self,
        regime: RegimeType,
        feature_idx: usize,
        mean: f64,
        variance: f64,
    ) -> Result<(), String> {
        let idx = regime.to_index();
        if idx >= self.num_states {
            return Err("Invalid regime index".to_string());
        }
        if feature_idx >= self.num_features {
            return Err("Invalid feature index".to_string());
        }
        if variance <= 0.0 {
            return Err("Variance must be positive".to_string());
        }

        self.emission_means[[idx, feature_idx]] = mean;
        self.emission_variances[[idx, feature_idx]] = variance;
        Ok(())
    }

    /// Compute Gaussian emission probability
    #[inline(always)]
    fn emission_probability(&self, state: usize, observation: &[f64]) -> f64 {
        let mut log_prob = 0.0;
        
        for (j, &obs_j) in observation.iter().enumerate() {
            let mean = self.emission_means[[state, j]];
            let var = self.emission_variances[[state, j]];
            
            // Log of Gaussian PDF
            let diff = obs_j - mean;
            log_prob -= 0.5 * diff * diff / var;
            log_prob -= 0.5 * var.ln();
        }
        
        log_prob -= 0.5 * self.num_features as f64 * (2.0 * std::f64::consts::PI).ln();
        log_prob.exp()
    }

    /// Forward algorithm step: update state probabilities given new observation
    pub fn filter_step(&mut self, observation: &[f64]) -> RegimePosterior {
        if observation.len() != self.num_features {
            // Return uniform if dimension mismatch
            return self.uniform_posterior();
        }

        let n = self.num_states;
        
        // Prediction step: P(x_t | y_{1:t-1}) = sum_x' P(x_t | x') P(x' | y_{1:t-1})
        let mut predicted = Array1::<f64>::zeros(n);
        for j in 0..n {
            for i in 0..n {
                predicted[j] += self.transition_matrix[[i, j]] * self.current_state_probs[i];
            }
        }

        // Update step: P(x_t | y_{1:t}) ∝ P(y_t | x_t) P(x_t | y_{1:t-1})
        let mut updated = Array1::<f64>::zeros(n);
        for i in 0..n {
            updated[i] = self.emission_probability(i, observation) * predicted[i];
        }

        // Normalize
        let sum: f64 = updated.sum();
        if sum > 1e-15 {
            updated.mapv_inplace(|x| x / sum);
        } else {
            // Fallback to uniform if all probabilities are zero
            updated.fill(1.0 / n as f64);
        }

        self.current_state_probs = updated.clone();

        // Build posterior result
        self.build_posterior(&updated)
    }

    /// Build posterior from state probabilities
    fn build_posterior(&self, probs: &Array1<f64>) -> RegimePosterior {
        let mut posterior = RegimePosterior::new(self.num_states);
        posterior.probabilities = probs.clone();

        // Find dominant regime
        let mut max_prob = 0.0;
        let mut max_idx = 0;
        for i in 0..self.num_states {
            if probs[i] > max_prob {
                max_prob = probs[i];
                max_idx = i;
            }
        }
        posterior.dominant_regime = RegimeType::from_index(max_idx);

        // Compute entropy: H = -sum(p * log(p))
        let mut entropy = 0.0;
        for &p in probs.iter() {
            if p > 1e-15 {
                entropy -= p * p.ln();
            }
        }
        posterior.entropy = entropy;

        posterior
    }

    /// Return uniform posterior
    fn uniform_posterior(&self) -> RegimePosterior {
        let mut posterior = RegimePosterior::new(self.num_states);
        posterior.probabilities.fill(1.0 / self.num_states as f64);
        posterior.dominant_regime = RegimeType::Unknown;
        posterior.entropy = (self.num_states as f64).ln();
        posterior
    }

    /// Get current regime estimate
    pub fn current_regime(&self) -> RegimePosterior {
        self.build_posterior(&self.current_state_probs)
    }

    /// Reset to prior
    pub fn reset(&mut self) {
        self.current_state_probs = self.prior_probabilities.clone();
    }

    /// Update transition matrix (for online learning)
    pub fn update_transition(&mut self, new_transitions: Array2<f64>) -> Result<(), String> {
        if new_transitions.dim() != (self.num_states, self.num_states) {
            return Err("Dimension mismatch".to_string());
        }

        // Validate stochastic matrix (rows sum to 1)
        for i in 0..self.num_states {
            let row_sum: f64 = (0..self.num_states).map(|j| new_transitions[[i, j]]).sum();
            if (row_sum - 1.0).abs() > 1e-6 {
                return Err(format!("Row {} does not sum to 1: {}", i, row_sum));
            }
        }

        self.transition_matrix = new_transitions;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmm_filtering() {
        let mut hmm = BayesianHMM::new(4, 3); // 4 regimes, 3 features

        // Set distinctive emission parameters for Risk-On regime
        hmm.set_emission_params(RegimeType::RiskOn, 0, 0.1, 0.01).unwrap(); // High returns
        hmm.set_emission_params(RegimeType::RiskOn, 1, 0.1, 0.01).unwrap(); // Low vol
        hmm.set_emission_params(RegimeType::RiskOn, 2, 0.5, 0.01).unwrap(); // High correlation

        // Set distinctive params for Risk-Off
        hmm.set_emission_params(RegimeType::RiskOff, 0, -0.1, 0.01).unwrap(); // Negative returns
        hmm.set_emission_params(RegimeType::RiskOff, 1, 0.5, 0.01).unwrap(); // High vol
        hmm.set_emission_params(RegimeType::RiskOff, 2, 0.9, 0.01).unwrap(); // Very high correlation

        // Feed observation consistent with Risk-On
        let obs_risk_on = vec![0.1, 0.1, 0.5];
        let posterior = hmm.filter_step(&obs_risk_on);

        assert!(posterior.probabilities.sum().abs() - 1.0 < 1e-10);
        assert!(posterior.entropy >= 0.0);
    }
}
