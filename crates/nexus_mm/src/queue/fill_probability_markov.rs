//! Fill Probability Markov Chain Model.
//! Calculates probability of limit order fill using Poisson arrival and Markov states.
//! Zero-allocation, no unwrap/expect in hot paths.

use crate::queue::queue_position_tracker::QueuePosition;

/// Error types for fill probability calculations
#[derive(Debug, Clone, PartialEq)]
pub enum FillProbabilityError {
    InvalidParameters,
    NumericalInstability,
}

/// Markov state for queue position
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueState {
    /// At front of queue
    Front,
    /// Middle of queue
    Middle,
    /// Back of queue
    Back,
    /// Behind book (not displayed)
    BehindBook,
}

impl QueueState {
    pub fn from_rank(rank: u32, total_orders: u32) -> Self {
        if rank <= 2 {
            QueueState::Front
        } else if rank <= total_orders / 2 {
            QueueState::Middle
        } else if total_orders > 0 {
            QueueState::Back
        } else {
            QueueState::BehindBook
        }
    }
}

/// Transition probabilities between queue states
#[derive(Debug, Clone, Copy)]
pub struct TransitionMatrix {
    /// P(Front -> Filled)
    pub front_fill: f64,
    /// P(Front -> Middle) - new orders ahead
    pub front_to_middle: f64,
    /// P(Front -> Back) - many new orders
    pub front_to_back: f64,
    /// P(Middle -> Filled)
    pub middle_fill: f64,
    /// P(Middle -> Front) - orders ahead cancel
    pub middle_to_front: f64,
    /// P(Middle -> Back) - new orders ahead
    pub middle_to_back: f64,
    /// P(Back -> Filled)
    pub back_fill: f64,
    /// P(Back -> Middle) - orders ahead cancel
    pub back_to_middle: f64,
}

impl Default for TransitionMatrix {
    fn default() -> Self {
        Self {
            front_fill: 0.3,
            front_to_middle: 0.5,
            front_to_back: 0.2,
            middle_fill: 0.1,
            middle_to_front: 0.3,
            middle_to_back: 0.6,
            back_fill: 0.02,
            back_to_middle: 0.2,
        }
    }
}

/// Configuration for Markov model
#[derive(Debug, Clone)]
pub struct MarkovConfig {
    /// Base arrival rate of market orders (per second)
    pub arrival_rate: f64,
    /// Cancellation rate (per second)
    pub cancellation_rate: f64,
    /// Time horizon for probability calculation (seconds)
    pub time_horizon: f64,
    /// Number of steps for Markov chain
    pub num_steps: usize,
}

impl Default for MarkovConfig {
    fn default() -> Self {
        Self {
            arrival_rate: 100.0,
            cancellation_rate: 50.0,
            time_horizon: 1.0,
            num_steps: 100,
        }
    }
}

/// Fill Probability Calculator using Markov Chain
pub struct FillProbabilityMarkov {
    config: MarkovConfig,
    transition_matrix: TransitionMatrix,
    /// Pre-allocated state probability vector
    state_probs: [f64; 4], // Front, Middle, Back, Filled
}

impl FillProbabilityMarkov {
    pub fn new(config: MarkovConfig) -> Result<Self, FillProbabilityError> {
        if config.arrival_rate <= 0.0 || config.time_horizon <= 0.0 {
            return Err(FillProbabilityError::InvalidParameters);
        }
        
        Ok(Self {
            config,
            transition_matrix: TransitionMatrix::default(),
            state_probs: [0.0; 4],
        })
    }
    
    /// Calculate fill probability given current queue position
    #[inline(always)]
    pub fn calculate_fill_probability(&self, position: &QueuePosition) -> f64 {
        // Determine initial state based on position
        let initial_state = self.determine_initial_state(position);
        
        // Initialize state probabilities
        let mut probs = [0.0; 4];
        match initial_state {
            QueueState::Front => probs[0] = 1.0,
            QueueState::Middle => probs[1] = 1.0,
            QueueState::Back => probs[2] = 1.0,
            QueueState::BehindBook => {
                // Behind book has very low probability
                return 0.0;
            }
        }
        
        // Simulate Markov chain for num_steps
        let dt = self.config.time_horizon / self.config.num_steps as f64;
        
        for _ in 0..self.config.num_steps {
            probs = self.step(probs, dt);
        }
        
        // Return probability of being in Filled state
        probs[3].clamp(0.0, 1.0)
    }
    
    /// Determine initial Markov state from queue position
    #[inline(always)]
    fn determine_initial_state(&self, position: &QueuePosition) -> QueueState {
        if position.rank <= 2 {
            QueueState::Front
        } else if position.volume_ahead < position.total_volume / 2 {
            QueueState::Middle
        } else if position.total_volume > 0 {
            QueueState::Back
        } else {
            QueueState::BehindBook
        }
    }
    
    /// Single step of Markov chain evolution
    #[inline(always)]
    fn step(&self, probs: [f64; 4], dt: f64) -> [f64; 4] {
        let tm = self.transition_matrix;
        
        // Scale probabilities by time step
        let p_front_fill = (tm.front_fill * dt).min(1.0);
        let p_front_mid = (tm.front_to_middle * dt).min(1.0);
        let p_front_back = (tm.front_to_back * dt).min(1.0);
        
        let p_mid_fill = (tm.middle_fill * dt).min(1.0);
        let p_mid_front = (tm.middle_to_front * dt).min(1.0);
        let p_mid_back = (tm.middle_to_back * dt).min(1.0);
        
        let p_back_fill = (tm.back_fill * dt).min(1.0);
        let p_back_mid = (tm.back_to_middle * dt).min(1.0);
        
        let mut new_probs = [0.0; 4];
        
        // From Front state
        new_probs[0] += probs[0] * (1.0 - p_front_fill - p_front_mid - p_front_back);
        new_probs[1] += probs[0] * p_front_mid;
        new_probs[2] += probs[0] * p_front_back;
        new_probs[3] += probs[0] * p_front_fill;
        
        // From Middle state
        new_probs[0] += probs[1] * p_mid_front;
        new_probs[1] += probs[1] * (1.0 - p_mid_fill - p_mid_front - p_mid_back);
        new_probs[2] += probs[1] * p_mid_back;
        new_probs[3] += probs[1] * p_mid_fill;
        
        // From Back state
        new_probs[0] += probs[2] * 0.0; // Can't jump to front directly
        new_probs[1] += probs[2] * p_back_mid;
        new_probs[2] += probs[2] * (1.0 - p_back_fill - p_back_mid);
        new_probs[3] += probs[2] * p_back_fill;
        
        // Filled state is absorbing
        new_probs[3] += probs[3];
        
        // Normalize to handle numerical drift
        let sum: f64 = new_probs.iter().sum();
        if sum > 1e-15 {
            for p in &mut new_probs {
                *p /= sum;
            }
        }
        
        new_probs
    }
    
    /// Update transition matrix based on observed fill rates
    pub fn update_from_observations(
        &mut self,
        fills_at_front: u32,
        total_at_front: u32,
        fills_at_middle: u32,
        total_at_middle: u32,
    ) {
        if total_at_front > 0 {
            self.transition_matrix.front_fill = fills_at_front as f64 / total_at_front as f64;
        }
        if total_at_middle > 0 {
            self.transition_matrix.middle_fill = fills_at_middle as f64 / total_at_middle as f64;
        }
    }
    
    /// Get expected time to fill (in seconds)
    #[inline(always)]
    pub fn expected_time_to_fill(&self, position: &QueuePosition) -> Option<f64> {
        let prob = self.calculate_fill_probability(position);
        
        if prob < 1e-10 {
            return None;
        }
        
        // Approximate: time_horizon / probability gives expected time
        Some(self.config.time_horizon / prob)
    }
    
    /// Reset to default transition matrix
    pub fn reset(&mut self) {
        self.transition_matrix = TransitionMatrix::default();
    }
}

/// Poisson-based fill probability estimator (alternative/simpler model)
pub struct PoissonFillEstimator {
    /// Arrival rate lambda (orders per second)
    lambda: f64,
}

impl PoissonFillEstimator {
    pub fn new(arrival_rate: f64) -> Result<Self, FillProbabilityError> {
        if arrival_rate <= 0.0 {
            return Err(FillProbabilityError::InvalidParameters);
        }
        
        Ok(Self {
            lambda: arrival_rate,
        })
    }
    
    /// Calculate probability of at least `volume_ahead` orders arriving
    /// before our order fills
    #[inline(always)]
    pub fn probability_of_fill(
        &self,
        volume_ahead: u64,
        our_size: u64,
        time_horizon: f64,
    ) -> f64 {
        if volume_ahead == 0 {
            // No queue - high probability of immediate fill
            return 1.0 - (-self.lambda * time_horizon).exp();
        }
        
        // Poisson probability: P(N >= volume_ahead) where N ~ Poisson(lambda * t)
        let expected_arrivals = self.lambda * time_horizon;
        
        // Use complementary CDF approximation
        // P(N >= k) = 1 - P(N < k) = 1 - sum_{i=0}^{k-1} e^(-λt) * (λt)^i / i!
        
        // For large k, use normal approximation
        if volume_ahead > 20 {
            let mean = expected_arrivals;
            let variance = expected_arrivals;
            let z = (volume_ahead as f64 - mean) / variance.sqrt().max(1e-15);
            
            // Standard normal CDF approximation
            1.0 - self.normal_cdf(z)
        } else {
            // Direct calculation for small k
            let mut prob_less = 0.0;
            let mut term = (-expected_arrivals).exp();
            prob_less += term;
            
            for i in 1..volume_ahead {
                term *= expected_arrivals / i as f64;
                prob_less += term;
            }
            
            1.0 - prob_less
        }
    }
    
    /// Standard normal CDF approximation
    #[inline(always)]
    fn normal_cdf(&self, z: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let sign = if z < 0.0 { -1.0 } else { 1.0 };
        let z_abs = z.abs();
        
        let t = 1.0 / (1.0 + 0.2316419 * z_abs);
        let d = 0.3989423 * (-z_abs * z_abs / 2.0).exp();
        
        let prob = d * t * (0.3193815 
            + t * (-0.3565638 
            + t * (1.781478 
            + t * (-1.821256 
            + t * 1.330274))));
        
        0.5 * (1.0 + sign * (1.0 - 2.0 * prob))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_markov_fill_probability() {
        let config = MarkovConfig::default();
        let calc = FillProbabilityMarkov::new(config).unwrap();
        
        let position = QueuePosition::new(100, 1000, 100, 1, 0.0);
        let prob = calc.calculate_fill_probability(&position);
        
        assert!(prob >= 0.0);
        assert!(prob <= 1.0);
    }
    
    #[test]
    fn test_poisson_estimator() {
        let est = PoissonFillEstimator::new(100.0).unwrap();
        
        // No queue - should have decent probability
        let prob_no_queue = est.probability_of_fill(0, 100, 1.0);
        assert!(prob_no_queue > 0.5);
        
        // Large queue - lower probability
        let prob_large_queue = est.probability_of_fill(1000, 100, 1.0);
        assert!(prob_large_queue < prob_no_queue);
    }
}
