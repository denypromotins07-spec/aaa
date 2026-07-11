//! Variational Free Energy Implementation for Active Inference
//! 
//! Implements Karl Friston's Free Energy Principle for biological neural networks.
//! Uses log-sum-exp tricks and probability clamping to prevent numerical overflow
//! during market shock events.

use core::f64::consts::E;

/// Maximum number of hidden states in the generative model
pub const MAX_HIDDEN_STATES: usize = 256;

/// Maximum number of sensory observations
pub const MAX_SENSORY_STATES: usize = 1024;

/// Maximum number of policies (action sequences)
pub const MAX_POLICIES: usize = 64;

/// Time horizon for policy evaluation
pub const POLICY_HORIZON: usize = 8;

/// Numerical stability constants
const LOG_ZERO: f64 = -1e10;
const PROB_MIN: f64 = 1e-10;
const PROB_MAX: f64 = 1.0 - 1e-10;

/// Error types for free energy computation
#[derive(Debug, Clone, Copy)]
pub enum FreeEnergyError {
    InvalidProbability,
    NormalizationFailed,
    MatrixDimensionMismatch,
    OverflowDetected,
    NotInitialized,
}

/// Generative model for active inference
/// Encodes the organism's beliefs about how sensations are generated
#[repr(C, align(64))]
pub struct GenerativeModel {
    /// A matrix: likelihood P(observation | hidden_state) [sensory x hidden]
    a_matrix: [[f64; MAX_HIDDEN_STATES]; MAX_SENSORY_STATES],
    /// B matrices: transition probabilities P(state_t+1 | state_t, action) [hidden x hidden x actions]
    b_matrices: [[[f64; MAX_HIDDEN_STATES]; MAX_HIDDEN_STATES]; 8], // Up to 8 actions
    /// D vector: prior preferences over hidden states
    d_prior: [f64; MAX_HIDDEN_STATES],
    /// Number of valid hidden states
    num_hidden_states: usize,
    /// Number of valid sensory states
    num_sensory_states: usize,
    /// Number of available actions
    num_actions: usize,
}

impl GenerativeModel {
    /// Create a new generative model with specified dimensions
    pub fn new(num_hidden: usize, num_sensory: usize, num_actions: usize) -> Result<Self, FreeEnergyError> {
        if num_hidden > MAX_HIDDEN_STATES || num_sensory > MAX_SENSORY_STATES || num_actions > 8 {
            return Err(FreeEnergyError::MatrixDimensionMismatch);
        }

        let mut model = Self {
            a_matrix: [[0.0; MAX_HIDDEN_STATES]; MAX_SENSORY_STATES],
            b_matrices: [[[0.0; MAX_HIDDEN_STATES]; MAX_HIDDEN_STATES]; 8],
            d_prior: [0.0; MAX_HIDDEN_STATES],
            num_hidden_states: num_hidden,
            num_sensory_states: num_sensory,
            num_actions: num_actions,
        };

        // Initialize with uniform priors
        model.initialize_uniform();
        Ok(model)
    }

    /// Initialize all distributions uniformly
    fn initialize_uniform(&mut self) {
        let h_inv = 1.0 / self.num_hidden_states as f64;
        let s_inv = 1.0 / self.num_sensory_states as f64;

        // Uniform A matrix
        for i in 0..self.num_sensory_states {
            for j in 0..self.num_hidden_states {
                self.a_matrix[i][j] = s_inv;
            }
        }

        // Uniform B matrices
        for action_idx in 0..self.num_actions {
            for i in 0..self.num_hidden_states {
                for j in 0..self.num_hidden_states {
                    self.b_matrices[action_idx][i][j] = h_inv;
                }
            }
        }

        // Uniform D prior
        for i in 0..self.num_hidden_states {
            self.d_prior[i] = h_inv;
        }
    }

    /// Set A matrix row (likelihood for a specific observation)
    pub fn set_likelihood(&mut self, obs_idx: usize, likelihoods: &[f64]) -> Result<(), FreeEnergyError> {
        if obs_idx >= self.num_sensory_states || likelihoods.len() != self.num_hidden_states {
            return Err(FreeEnergyError::MatrixDimensionMismatch);
        }

        // Normalize and clamp probabilities
        let sum: f64 = likelihoods.iter().sum();
        if sum < PROB_MIN {
            return Err(FreeEnergyError::InvalidProbability);
        }

        for (j, &lik) in likelihoods.iter().enumerate() {
            let normalized = (lik / sum).clamp(PROB_MIN, PROB_MAX);
            self.a_matrix[obs_idx][j] = normalized;
        }

        Ok(())
    }

    /// Set B matrix for a specific action
    pub fn set_transition(&mut self, action_idx: usize, transitions: &[[f64; MAX_HIDDEN_STATES]]) -> Result<(), FreeEnergyError> {
        if action_idx >= self.num_actions {
            return Err(FreeEnergyError::MatrixDimensionMismatch);
        }

        for i in 0..self.num_hidden_states {
            let row_sum: f64 = transitions[i][..self.num_hidden_states].iter().sum();
            if row_sum < PROB_MIN {
                return Err(FreeEnergyError::InvalidProbability);
            }

            for j in 0..self.num_hidden_states {
                let normalized = (transitions[i][j] / row_sum).clamp(PROB_MIN, PROB_MAX);
                self.b_matrices[action_idx][i][j] = normalized;
            }
        }

        Ok(())
    }

    /// Set prior preferences (D vector)
    pub fn set_prior(&mut self, preferences: &[f64]) -> Result<(), FreeEnergyError> {
        if preferences.len() != self.num_hidden_states {
            return Err(FreeEnergyError::MatrixDimensionMismatch);
        }

        let sum: f64 = preferences.iter().sum();
        if sum < PROB_MIN {
            return Err(FreeEnergyError::InvalidProbability);
        }

        for (i, &pref) in preferences.iter().enumerate() {
            self.d_prior[i] = (pref / sum).clamp(PROB_MIN, PROB_MAX);
        }

        Ok(())
    }
}

/// Belief state for variational inference
#[repr(C, align(64))]
pub struct BeliefState {
    /// Posterior over hidden states q(s)
    posterior: [f64; MAX_HIDDEN_STATES],
    /// Log posterior for numerical stability
    log_posterior: [f64; MAX_HIDDEN_STATES],
    /// Valid state count
    num_states: usize,
}

impl BeliefState {
    /// Create a new belief state
    pub fn new(num_states: usize) -> Self {
        let uniform = 1.0 / num_states as f64;
        let log_uniform = uniform.ln();

        Self {
            posterior: [uniform; MAX_HIDDEN_STATES],
            log_posterior: [log_uniform; MAX_HIDDEN_STATES],
            num_states,
        }
    }

    /// Get posterior probability for a state
    #[inline]
    pub fn get_probability(&self, state_idx: usize) -> f64 {
        if state_idx < self.num_states {
            self.posterior[state_idx]
        } else {
            0.0
        }
    }

    /// Get log posterior for a state
    #[inline]
    pub fn get_log_probability(&self, state_idx: usize) -> f64 {
        if state_idx < self.num_states {
            self.log_posterior[state_idx]
        } else {
            LOG_ZERO
        }
    }
}

/// Variational Free Energy calculator
pub struct VariationalFreeEnergy {
    /// Current belief state
    beliefs: BeliefState,
    /// Generative model
    model: GenerativeModel,
    /// Running free energy estimate
    running_fe: f64,
    /// Free energy history for convergence detection
    fe_history: [f64; 16],
    fe_history_idx: usize,
    /// Precision parameter (inverse temperature)
    precision: f64,
}

impl VariationalFreeEnergy {
    /// Create a new free energy calculator
    pub fn new(model: GenerativeModel) -> Self {
        let num_states = model.num_hidden_states;
        Self {
            beliefs: BeliefState::new(num_states),
            model,
            running_fe: 0.0,
            fe_history: [0.0; 16],
            fe_history_idx: 0,
            precision: 1.0,
        }
    }

    /// Compute variational free energy using log-sum-exp trick
    /// F = E_q[ln q(s) - ln P(o,s)] = Complexity - Accuracy
    pub fn compute_free_energy(&mut self, observation: usize) -> Result<f64, FreeEnergyError> {
        if observation >= self.model.num_sensory_states {
            return Err(FreeEnergyError::MatrixDimensionMismatch);
        }

        let mut log_joint_sum = LOG_ZERO;
        let mut entropy = 0.0;

        // Compute free energy components
        for s in 0..self.beliefs.num_states {
            let q_s = self.beliefs.get_probability(s);
            let ln_q_s = self.beliefs.get_log_probability(s);

            if q_s < PROB_MIN {
                continue;
            }

            // Likelihood term: ln P(o|s)
            let p_o_given_s = self.model.a_matrix[observation][s].max(PROB_MIN);
            let ln_p_o_given_s = p_o_given_s.ln();

            // Prior term: ln P(s)
            let p_s = self.model.d_prior[s].max(PROB_MIN);
            let ln_p_s = p_s.ln();

            // Joint log probability: ln P(o,s) = ln P(o|s) + ln P(s)
            let ln_joint = ln_p_o_given_s + ln_p_s;

            // Accumulate using log-sum-exp for numerical stability
            log_joint_sum = self.log_sum_exp(log_joint_sum, ln_joint + ln_q_s);

            // Entropy: -E_q[ln q(s)]
            entropy -= q_s * ln_q_s;
        }

        // Free energy = -ln P(o) - H[q] (negative log evidence minus entropy)
        // Using: F = E_q[ln q(s)] - E_q[ln P(o,s)]
        let expected_energy = -log_joint_sum;
        let free_energy = -entropy + expected_energy;

        // Clamp to prevent overflow
        let free_energy = free_energy.clamp(-1e6, 1e6);

        // Update running estimate
        self.running_fe = 0.95 * self.running_fe + 0.05 * free_energy;

        // Store in history
        self.fe_history[self.fe_history_idx] = free_energy;
        self.fe_history_idx = (self.fe_history_idx + 1) % 16;

        Ok(free_energy)
    }

    /// Numerically stable log-sum-exp
    #[inline]
    fn log_sum_exp(&self, a: f64, b: f64) -> f64 {
        if a <= LOG_ZERO {
            return b;
        }
        if b <= LOG_ZERO {
            return a;
        }
        
        let max_val = a.max(b);
        let min_val = a.min(b);
        
        max_val + ((min_val - max_val).exp())
    }

    /// Update beliefs using variational message passing
    pub fn update_beliefs(&mut self, observation: usize) -> Result<(), FreeEnergyError> {
        if observation >= self.model.num_sensory_states {
            return Err(FreeEnergyError::MatrixDimensionMismatch);
        }

        let mut new_log_posterior = [LOG_ZERO; MAX_HIDDEN_STATES];
        let mut max_log = LOG_ZERO;

        // Compute unnormalized log posterior
        for s in 0..self.beliefs.num_states {
            // Likelihood
            let ln_p_o_given_s = self.model.a_matrix[observation][s]
                .max(PROB_MIN)
                .ln();

            // Prior (from previous belief)
            let ln_prior = self.beliefs.get_log_probability(s);

            // Unnormalized log posterior
            new_log_posterior[s] = ln_p_o_given_s + ln_prior;
            max_log = max_log.max(new_log_posterior[s]);
        }

        // Normalize using log-sum-exp
        let log_norm = {
            let mut sum = LOG_ZERO;
            for s in 0..self.beliefs.num_states {
                sum = self.log_sum_exp(sum, new_log_posterior[s]);
            }
            sum
        };

        // Update belief state
        for s in 0..self.beliefs.num_states {
            self.beliefs.log_posterior[s] = new_log_posterior[s] - log_norm;
            self.beliefs.posterior[s] = self.beliefs.log_posterior[s].exp();
        }

        // Renormalize to ensure sum = 1 (numerical cleanup)
        let sum: f64 = self.beliefs.posterior[..self.beliefs.num_states].iter().sum();
        if sum > PROB_MIN {
            for s in 0..self.beliefs.num_states {
                self.beliefs.posterior[s] /= sum;
                self.beliefs.posterior[s] = self.beliefs.posterior[s].clamp(PROB_MIN, PROB_MAX);
            }
            // Recompute log posteriors
            for s in 0..self.beliefs.num_states {
                self.beliefs.log_posterior[s] = self.beliefs.posterior[s].ln();
            }
        }

        Ok(())
    }

    /// Get current free energy estimate
    #[inline]
    pub fn get_running_free_energy(&self) -> f64 {
        self.running_fe
    }

    /// Check for free energy divergence (indicates model failure)
    pub fn check_divergence(&self, threshold: f64) -> bool {
        // Check if recent FE values exceed threshold
        let recent_max = self.fe_history
            .iter()
            .take(self.fe_history_idx.max(1))
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        recent_max > threshold
    }

    /// Set precision parameter (controls confidence in predictions)
    pub fn set_precision(&mut self, precision: f64) {
        self.precision = precision.max(0.01).min(100.0);
    }

    /// Get precision parameter
    #[inline]
    pub fn get_precision(&self) -> f64 {
        self.precision
    }

    /// Get current belief distribution
    pub fn get_beliefs(&self) -> &[f64; MAX_HIDDEN_STATES] {
        &self.beliefs.posterior
    }
}

/// Policy evaluation for active inference
pub struct PolicyEvaluator {
    /// Expected free energy for each policy
    efe_values: [f64; MAX_POLICIES],
    /// Policy selection probabilities (softmax over EFE)
    policy_probs: [f64; MAX_POLICIES],
    /// Number of valid policies
    num_policies: usize,
    /// Precision for policy selection
    policy_precision: f64,
}

impl PolicyEvaluator {
    /// Create a new policy evaluator
    pub fn new(num_policies: usize) -> Self {
        Self {
            efe_values: [0.0; MAX_POLICIES],
            policy_probs: [0.0; MAX_POLICIES],
            num_policies: num_policies.min(MAX_POLICIES),
            policy_precision: 16.0, // Softmax temperature
        }
    }

    /// Compute expected free energy for a policy
    /// G(pi) = E_Q[ln Q(s|pi) - ln P(s,o|pi)]
    pub fn compute_efe(
        &self,
        policy_idx: usize,
        predicted_states: &[f64],
        preferred_states: &[f64],
    ) -> Result<f64, FreeEnergyError> {
        if policy_idx >= self.num_policies {
            return Err(FreeEnergyError::MatrixDimensionMismatch);
        }

        let mut efe = 0.0;
        let n = predicted_states.len().min(preferred_states.len());

        for i in 0..n {
            let q_s = predicted_states[i].max(PROB_MIN);
            let p_s = preferred_states[i].max(PROB_MIN);

            // Risk term: KL divergence from preferences
            efe += q_s * (q_s.ln() - p_s.ln());
        }

        // Ambiguity term would be added here based on A matrix entropy

        Ok(efe)
    }

    /// Evaluate all policies and compute selection probabilities
    pub fn evaluate_all_policies(
        &mut self,
        predictions: &[[f64; MAX_HIDDEN_STATES]],
        preferences: &[f64; MAX_HIDDEN_STATES],
    ) -> Result<(), FreeEnergyError> {
        let mut min_efe = f64::INFINITY;
        let mut max_efe = f64::NEG_INFINITY;

        // Compute EFE for each policy
        for pi in 0..self.num_policies {
            let efe = self.compute_efe(pi, &predictions[pi], preferences)?;
            self.efe_values[pi] = efe;
            min_efe = min_efe.min(efe);
            max_efe = max_efe.max(efe);
        }

        // Softmax over negative EFE (prefer lower EFE)
        let mut sum_exp = 0.0;
        for pi in 0..self.num_policies {
            // Normalize EFE to prevent overflow
            let norm_efe = (self.efe_values[pi] - min_efe) / (max_efe - min_efe + 1e-10);
            let exp_val = (-self.policy_precision * norm_efe).exp();
            self.policy_probs[pi] = exp_val;
            sum_exp += exp_val;
        }

        // Normalize probabilities
        if sum_exp > PROB_MIN {
            for pi in 0..self.num_policies {
                self.policy_probs[pi] /= sum_exp;
            }
        }

        Ok(())
    }

    /// Select best policy (argmax over policy probabilities)
    pub fn select_policy(&self) -> usize {
        let mut best_idx = 0;
        let mut best_prob = self.policy_probs[0];

        for pi in 1..self.num_policies {
            if self.policy_probs[pi] > best_prob {
                best_prob = self.policy_probs[pi];
                best_idx = pi;
            }
        }

        best_idx
    }

    /// Sample policy according to probabilities (for exploration)
    pub fn sample_policy(&self, random_value: f64) -> usize {
        let mut cumsum = 0.0;
        for pi in 0..self.num_policies {
            cumsum += self.policy_probs[pi];
            if random_value < cumsum {
                return pi;
            }
        }
        self.num_policies - 1
    }

    /// Set policy selection precision
    pub fn set_policy_precision(&mut self, precision: f64) {
        self.policy_precision = precision.max(0.1).min(100.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generative_model_initialization() {
        let model = GenerativeModel::new(4, 8, 2).unwrap();
        assert_eq!(model.num_hidden_states, 4);
        assert_eq!(model.num_sensory_states, 8);
        assert_eq!(model.num_actions, 2);
    }

    #[test]
    fn test_free_energy_computation() {
        let model = GenerativeModel::new(4, 8, 2).unwrap();
        let mut fe_calc = VariationalFreeEnergy::new(model);

        // Compute FE for a valid observation
        let fe = fe_calc.compute_free_energy(0);
        assert!(fe.is_ok());
        assert!(fe.unwrap().is_finite());
    }

    #[test]
    fn test_log_sum_exp_stability() {
        let fe_calc = VariationalFreeEnergy::new(
            GenerativeModel::new(4, 8, 2).unwrap()
        );

        // Test with extreme values
        let result = fe_calc.log_sum_exp(-1e9, -1e9);
        assert!(result.is_finite());
        assert!(result < -1e8);

        // Test with normal values
        let result = fe_calc.log_sum_exp(0.0, 0.0);
        assert!((result - 2.0f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_policy_evaluation() {
        let mut evaluator = PolicyEvaluator::new(4);
        let predictions = [[0.25; MAX_HIDDEN_STATES]; MAX_POLICIES];
        let preferences = [0.25; MAX_HIDDEN_STATES];

        let result = evaluator.evaluate_all_policies(&predictions, &preferences);
        assert!(result.is_ok());

        let selected = evaluator.select_policy();
        assert!(selected < 4);
    }
}
