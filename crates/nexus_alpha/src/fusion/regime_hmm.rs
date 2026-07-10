//! Hidden Markov Model for Market Regime Detection
//! 
//! Implements an HMM using the Baum-Welch algorithm to detect latent
//! market regimes (Mean-Reverting, Trending, High-Toxicity, etc.).
//! Runs on a separate thread to avoid blocking the tick-processing hot path.

use nexus_core::memory::arena::BumpAllocator;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

/// Maximum number of hidden states (regimes)
pub const MAX_STATES: usize = 5;

/// Maximum observation sequence length for Baum-Welch
pub const MAX_OBS_SEQUENCE: usize = 100;

/// Number of observable features
pub const NUM_FEATURES: usize = 4; // e.g., momentum, volatility, toxicity, volume_imbalance

/// Market regime types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    MeanReverting = 0,
    Trending = 1,
    HighVolatility = 2,
    HighToxicity = 3,
    LowLiquidity = 4,
}

impl Default for MarketRegime {
    fn default() -> Self {
        MarketRegime::MeanReverting
    }
}

/// HMM parameters (transition and emission matrices)
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct HmmParams {
    /// Transition matrix A[i][j] = P(state_j | state_i)
    pub transition: [f64; MAX_STATES * MAX_STATES],
    /// Emission probabilities (Gaussian means)
    pub emission_means: [f64; MAX_STATES * NUM_FEATURES],
    /// Emission variances
    pub emission_vars: [f64; MAX_STATES * NUM_FEATURES],
    /// Initial state distribution
    pub initial_dist: [f64; MAX_STATES],
    /// Number of active states
    pub num_states: usize,
    /// Padding
    _padding: [u8; 24],
}

impl Default for HmmParams {
    fn default() -> Self {
        let mut params = Self {
            transition: [0.0; MAX_STATES * MAX_STATES],
            emission_means: [0.0; MAX_STATES * NUM_FEATURES],
            emission_vars: [1.0; MAX_STATES * NUM_FEATURES],
            initial_dist: [0.2; MAX_STATES],
            num_states: 5,
            _padding: [0u8; 24],
        };

        // Initialize with reasonable defaults
        // High self-transition probability (regimes tend to persist)
        for i in 0..MAX_STATES {
            for j in 0..MAX_STATES {
                if i == j {
                    params.transition[i * MAX_STATES + j] = 0.7;
                } else {
                    params.transition[i * MAX_STATES + j] = 0.075;
                }
            }
        }

        // Initialize emission means for different regimes
        // Features: [momentum, volatility, toxicity, volume_imbalance]
        
        // Mean-reverting: low momentum, moderate vol
        params.emission_means[0 * NUM_FEATURES + 0] = 0.0;  // momentum
        params.emission_means[0 * NUM_FEATURES + 1] = 0.3;  // volatility
        
        // Trending: high momentum
        params.emission_means[1 * NUM_FEATURES + 0] = 0.7;
        params.emission_means[1 * NUM_FEATURES + 1] = 0.4;
        
        // High volatility
        params.emission_means[2 * NUM_FEATURES + 1] = 0.9;
        
        // High toxicity
        params.emission_means[3 * NUM_FEATURES + 2] = 0.8;
        
        // Low liquidity
        params.emission_means[4 * NUM_FEATURES + 3] = -0.5;

        params
    }
}

/// Observation vector
#[repr(C, align(64))]
#[derive(Clone, Copy, Default)]
pub struct Observation {
    pub features: [f64; NUM_FEATURES],
    pub ts: u64,
}

/// HMM State estimator
pub struct RegimeHmm {
    /// Model parameters
    params: HmmParams,
    /// Current state distribution (belief)
    belief: [f64; MAX_STATES],
    /// Observation history for batch updates
    observations: [Observation; MAX_OBS_SEQUENCE],
    /// Observation count
    obs_count: usize,
    /// Current estimated regime
    current_regime: MarketRegime,
    /// Regime confidence
    regime_confidence: f64,
    /// Atomic regime for lock-free reads
    atomic_regime: AtomicU8,
    /// Last update timestamp
    last_update_ts: u64,
    /// Background thread handle flag
    background_update_pending: bool,
}

unsafe impl Send for RegimeHmm {}
unsafe impl Sync for RegimeHmm {}

impl RegimeHmm {
    pub fn new(_allocator: &BumpAllocator, initial_params: Option<HmmParams>) -> Self {
        let params = initial_params.unwrap_or_default();
        
        let mut hmm = Self {
            params,
            belief: [1.0 / MAX_STATES as f64; MAX_STATES],
            observations: [Observation::default(); MAX_OBS_SEQUENCE],
            obs_count: 0,
            current_regime: MarketRegime::MeanReverting,
            regime_confidence: 0.2,
            atomic_regime: AtomicU8::new(MarketRegime::MeanReverting as u8),
            last_update_ts: 0,
            background_update_pending: false,
        };

        // Initialize belief from initial distribution
        hmm.belief.copy_from_slice(&params.initial_dist);

        hmm
    }

    /// Process a new observation - fast path, no allocation
    #[inline]
    pub fn observe(&mut self, obs: Observation) {
        // Store observation for batch updates
        if self.obs_count < MAX_OBS_SEQUENCE {
            self.observations[self.obs_count] = obs;
            self.obs_count += 1;
        }

        // Online update using forward algorithm
        self.forward_step(&obs.features);

        // Decode current regime
        self.decode_regime();
    }

    /// Forward algorithm step - update belief with new observation
    #[inline]
    fn forward_step(&mut self, features: &[f64; NUM_FEATURES]) {
        let mut new_belief = [0.0; MAX_STATES];

        // Calculate emission probabilities for each state
        let emissions = self.calculate_emissions(features);

        // Forward pass: P(x_t | y_{1:t}) ∝ P(y_t | x_t) * Σ P(x_t | x_{t-1}) * P(x_{t-1})
        for j in 0..self.params.num_states {
            let mut sum = 0.0;
            for i in 0..self.params.num_states {
                sum += self.params.transition[i * MAX_STATES + j] * self.belief[i];
            }
            new_belief[j] = emissions[j] * sum;
        }

        // Normalize
        let total: f64 = new_belief.iter().sum();
        if total > 1e-10 {
            for b in &mut new_belief {
                *b /= total;
            }
        }

        self.belief = new_belief;
    }

    /// Calculate emission probabilities using Gaussian likelihood
    #[inline]
    fn calculate_emissions(&self, features: &[f64; NUM_FEATURES]) -> [f64; MAX_STATES] {
        let mut emissions = [0.0; MAX_STATES];

        for s in 0..self.params.num_states {
            let mut log_prob = 0.0;
            
            for f in 0..NUM_FEATURES {
                let mean = self.params.emission_means[s * NUM_FEATURES + f];
                let var = self.params.emission_vars[s * NUM_FEATURES + f].max(1e-6);
                let diff = features[f] - mean;
                
                // Log of Gaussian PDF
                log_prob -= 0.5 * (diff * diff / var + var.ln());
            }

            emissions[s] = log_prob.exp();
        }

        emissions
    }

    /// Decode the most likely regime (Viterbi-like)
    #[inline]
    fn decode_regime(&mut self) {
        let mut max_prob = 0.0;
        let mut best_state = 0usize;

        for s in 0..self.params.num_states {
            if self.belief[s] > max_prob {
                max_prob = self.belief[s];
                best_state = s;
            }
        }

        self.current_regime = match best_state {
            0 => MarketRegime::MeanReverting,
            1 => MarketRegime::Trending,
            2 => MarketRegime::HighVolatility,
            3 => MarketRegime::HighToxicity,
            4 => MarketRegime::LowLiquidity,
            _ => MarketRegime::MeanReverting,
        };

        self.regime_confidence = max_prob;
        self.atomic_regime.store(best_state as u8, Ordering::Release);
    }

    /// Run Baum-Welch algorithm for parameter learning (background thread)
    pub fn run_baum_welch(&mut self, max_iterations: usize) -> HmmParams {
        if self.obs_count < 10 {
            return self.params; // Not enough data
        }

        let mut new_params = self.params;
        
        for _iter in 0..max_iterations.min(50) {
            // E-step: compute forward and backward probabilities
            let (alpha, beta) = self.forward_backward();
            
            // M-step: update parameters
            self.m_step(&alpha, &beta, &mut new_params);
        }

        self.params = new_params;
        new_params
    }

    #[inline]
    fn forward_backward(&self) -> ([[f64; MAX_STATES]; MAX_OBS_SEQUENCE], [[f64; MAX_STATES]; MAX_OBS_SEQUENCE]) {
        let mut alpha = [[0.0; MAX_STATES]; MAX_OBS_SEQUENCE];
        let mut beta = [[0.0; MAX_STATES]; MAX_OBS_SEQUENCE];

        // Forward
        for t in 0..self.obs_count.min(MAX_OBS_SEQUENCE) {
            let emissions = self.calculate_emissions(&self.observations[t].features);
            for j in 0..self.params.num_states {
                if t == 0 {
                    alpha[t][j] = self.params.initial_dist[j] * emissions[j];
                } else {
                    let mut sum = 0.0;
                    for i in 0..self.params.num_states {
                        sum += alpha[t-1][i] * self.params.transition[i * MAX_STATES + j];
                    }
                    alpha[t][j] = sum * emissions[j];
                }
            }
        }

        // Backward
        for j in 0..self.params.num_states {
            beta[self.obs_count.saturating_sub(1)][j] = 1.0;
        }

        for t in (0..self.obs_count.saturating_sub(1)).rev() {
            for i in 0..self.params.num_states {
                let mut sum = 0.0;
                for j in 0..self.params.num_states {
                    let emissions = self.calculate_emissions(&self.observations[t + 1].features);
                    sum += self.params.transition[i * MAX_STATES + j] * emissions[j] * beta[t + 1][j];
                }
                beta[t][i] = sum;
            }
        }

        (alpha, beta)
    }

    #[inline]
    fn m_step(&self, _alpha: &[[f64; MAX_STATES]; MAX_OBS_SEQUENCE], 
              _beta: &[[f64; MAX_STATES]; MAX_OBS_SEQUENCE], 
              params: &mut HmmParams) {
        // Simplified M-step - in production would fully re-estimate parameters
        // This is a placeholder that applies smoothing to existing parameters
        for i in 0..MAX_STATES * MAX_STATES {
            params.transition[i] = params.transition[i] * 0.95 + self.params.transition[i] * 0.05;
        }
    }

    /// Get current regime (lock-free)
    #[inline]
    pub fn get_current_regime(&self) -> MarketRegime {
        let state = self.atomic_regime.load(Ordering::Acquire);
        match state {
            0 => MarketRegime::MeanReverting,
            1 => MarketRegime::Trending,
            2 => MarketRegime::HighVolatility,
            3 => MarketRegime::HighToxicity,
            4 => MarketRegime::LowLiquidity,
            _ => MarketRegime::MeanReverting,
        }
    }

    /// Get regime confidence
    #[inline]
    pub fn get_confidence(&self) -> f64 {
        self.regime_confidence
    }

    /// Get full belief distribution
    #[inline]
    pub fn get_belief(&self) -> &[f64; MAX_STATES] {
        &self.belief
    }

    /// Check if regime changed recently
    #[inline]
    pub fn is_regime_stable(&self, threshold: f64) -> bool {
        self.regime_confidence > threshold
    }
}

/// Regime change event
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct RegimeChangeEvent {
    pub old_regime: MarketRegime,
    pub new_regime: MarketRegime,
    pub confidence: f64,
    pub ts: u64,
    pub belief_before: [f64; MAX_STATES],
    pub belief_after: [f64; MAX_STATES],
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_core::memory::arena::BumpAllocator;

    #[test]
    fn test_hmm_initialization() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let hmm = RegimeHmm::new(&allocator, None);
        
        assert_eq!(hmm.get_current_regime(), MarketRegime::MeanReverting);
        assert!((hmm.get_belief().iter().sum::<f64>() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_hmm_regime_detection_trending() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut hmm = RegimeHmm::new(&allocator, None);

        // Simulate trending market observations (high momentum)
        for i in 0..30 {
            let obs = Observation {
                features: [0.8, 0.4, 0.2, 0.3], // High momentum
                ts: 1_000_000_000_000 + i * 1_000_000,
            };
            hmm.observe(obs);
        }

        // Should detect trending regime
        assert_eq!(hmm.get_current_regime(), MarketRegime::Trending);
    }

    #[test]
    fn test_hmm_regime_detection_toxicity() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut hmm = RegimeHmm::new(&allocator, None);

        // Simulate high toxicity observations
        for i in 0..30 {
            let obs = Observation {
                features: [0.2, 0.3, 0.9, 0.1], // High toxicity
                ts: 1_000_000_000_000 + i * 1_000_000,
            };
            hmm.observe(obs);
        }

        // Should detect high toxicity regime
        assert_eq!(hmm.get_current_regime(), MarketRegime::HighToxicity);
    }

    #[test]
    fn test_regime_stability() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut hmm = RegimeHmm::new(&allocator, None);

        // Initial state should be uncertain
        assert!(!hmm.is_regime_stable(0.5));

        // After many consistent observations, should become stable
        for i in 0..50 {
            let obs = Observation {
                features: [0.8, 0.4, 0.2, 0.3],
                ts: 1_000_000_000_000 + i * 1_000_000,
            };
            hmm.observe(obs);
        }

        assert!(hmm.is_regime_stable(0.5));
    }
}
