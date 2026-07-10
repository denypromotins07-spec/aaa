//! Bayesian Weight Updater for Swarm Particle Filter.
//! 
//! Updates particle weights based on Bayesian likelihood of observations,
//! enabling the swarm to converge on the true iceberg order characteristics.

use crate::particles::iceberg_state_hypothesis::{IcebergParticle, IcebergState, ParticleStorage};
use nexus_types::market::{Side, PriceLevel, TradeExecution};
use std::time::Instant;

/// Likelihood function types for different observation scenarios
#[derive(Debug, Clone, Copy)]
pub enum LikelihoodModel {
    /// Gaussian likelihood for size observations
    Gaussian { mean: f64, std_dev: f64 },
    /// Exponential likelihood for time-between-events
    Exponential { rate: f64 },
    /// Bernoulli likelihood for binary events (fill/no-fill)
    Bernoulli { probability: f64 },
    /// Custom likelihood based on market microstructure
    Microstructure {
        fill_probability: f64,
        replenishment_probability: f64,
        size_decay_factor: f64,
    },
}

impl LikelihoodModel {
    /// Calculate likelihood of an observation given a hypothesis state
    pub fn calculate(&self, state: &IcebergState, observation: &ParticleObservation) -> f64 {
        match self {
            LikelihoodModel::Gaussian { mean, std_dev } => {
                self.gaussian_likelihood(observation.observed_size as f64, *mean, *std_dev)
            }
            LikelihoodModel::Exponential { rate } => {
                self.exponential_likelihood(observation.time_delta_ns as f64 / 1e9, *rate)
            }
            LikelihoodModel::Bernoulli { probability } => {
                if observation.was_filled {
                    *probability
                } else {
                    1.0 - probability
                }
            }
            LikelihoodModel::Microstructure {
                fill_probability,
                replenishment_probability,
                size_decay_factor,
            } => self.microstructure_likelihood(state, observation, *fill_probability, *replenishment_probability, *size_decay_factor),
        }
    }

    /// Gaussian likelihood: P(x|μ,σ) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
    fn gaussian_likelihood(&self, x: f64, mean: f64, std_dev: f64) -> f64 {
        if std_dev <= 0.0 {
            return if (x - mean).abs() < 1e-10 { 1.0 } else { 0.0 };
        }

        let variance = std_dev * std_dev;
        let exponent = -((x - mean).powi(2)) / (2.0 * variance);
        let coefficient = 1.0 / (2.0 * std::f64::consts::PI * variance).sqrt();

        coefficient * exponent.exp()
    }

    /// Exponential likelihood: P(t|λ) = λ * exp(-λ*t)
    fn exponential_likelihood(&self, t: f64, rate: f64) -> f64 {
        if t < 0.0 || rate <= 0.0 {
            return 0.0;
        }
        rate * (-rate * t).exp()
    }

    /// Microstructure-aware likelihood combining multiple factors
    fn microstructure_likelihood(
        &self,
        state: &IcebergState,
        obs: &ParticleObservation,
        fill_prob: f64,
        replenish_prob: f64,
        size_decay: f64,
    ) -> f64 {
        let mut likelihood = 1.0;

        // Factor 1: Fill event likelihood
        if obs.was_filled {
            likelihood *= fill_prob;
        } else {
            likelihood *= 1.0 - fill_prob;
        }

        // Factor 2: Replenishment consistency
        if obs.replenishment_observed {
            // If we observed replenishment, check if state predicts it
            let expected_replenishment = state.executed_size > 0 && 
                state.visible_size < state.replenishment_size;
            if expected_replenishment {
                likelihood *= replenish_prob;
            } else {
                likelihood *= 0.1; // Unexpected replenishment is unlikely
            }
        }

        // Factor 3: Size consistency with decay
        if obs.observed_size > 0 {
            let expected_visible = state.visible_size as f64;
            let actual = obs.observed_size as f64;
            
            // Likelihood decreases as observed size deviates from expected
            let size_ratio = (actual / expected_visible.max(1.0)).min(1.0);
            likelihood *= size_decay.powf((1.0 - size_ratio).abs());
        }

        likelihood.clamp(0.0, 1.0)
    }
}

/// Observation data from market events
#[derive(Debug, Clone, Copy)]
pub struct ParticleObservation {
    /// Observed trade size
    pub observed_size: u64,
    /// Time delta since last observation (ns)
    pub time_delta_ns: u64,
    /// Whether a fill occurred
    pub was_filled: bool,
    /// Whether replenishment was detected
    pub replenishment_observed: bool,
    /// Price level of the observation
    pub price_level: PriceLevel,
    /// Side of the observation
    pub side: Side,
}

impl ParticleObservation {
    pub fn from_trade(execution: &TradeExecution, prev_time_ns: u64) -> Self {
        let current_time_ns = execution.timestamp_ns;
        let time_delta = current_time_ns.saturating_sub(prev_time_ns);

        Self {
            observed_size: execution.size,
            time_delta_ns: time_delta,
            was_filled: true,
            replenishment_observed: false, // Will be set by analysis
            price_level: execution.price_level,
            side: execution.side,
        }
    }
}

/// Bayesian weight updater with adaptive learning rate
pub struct BayesianWeightUpdater {
    default_model: LikelihoodModel,
    /// Adaptive learning rate for weight updates
    learning_rate: f64,
    /// Minimum weight floor to prevent particle death
    min_weight_floor: f64,
    /// Weight normalization method
    normalize_on_update: bool,
    /// Count of updates performed
    update_count: u64,
}

impl BayesianWeightUpdater {
    pub fn new(default_model: LikelihoodModel) -> Self {
        Self {
            default_model,
            learning_rate: 1.0,
            min_weight_floor: 1e-10,
            normalize_on_update: true,
            update_count: 0,
        }
    }

    /// Update all particle weights based on observation
    pub fn update_weights(
        &mut self,
        storage: &mut ParticleStorage,
        observation: &ParticleObservation,
    ) -> f64 {
        let mut total_weight = 0.0;

        // First pass: calculate new weights
        for particle in storage.active_particles_mut() {
            let likelihood = self.default_model.calculate(&particle.state, observation);
            
            // Bayesian update: posterior ∝ prior × likelihood
            // Using learning rate for tempered Bayesian updating
            let new_weight = particle.weight * likelihood.powf(self.learning_rate);
            
            // Apply minimum weight floor
            particle.weight = new_weight.max(self.min_weight_floor);
            
            total_weight += particle.weight;
        }

        // Second pass: normalize if enabled
        if self.normalize_on_update && total_weight > 0.0 {
            for particle in storage.active_particles_mut() {
                particle.weight /= total_weight;
            }
        }

        self.update_count += 1;
        total_weight
    }

    /// Update weights with custom model for specific particles
    pub fn update_weights_selective(
        &mut self,
        storage: &mut ParticleStorage,
        observation: &ParticleObservation,
        model: LikelihoodModel,
        particle_indices: &[usize],
    ) -> Result<(), WeightUpdateError> {
        let mut total_weight = 0.0;

        for &idx in particle_indices {
            if let Some(particle) = storage.get_mut(idx) {
                if !particle.active {
                    continue;
                }

                let likelihood = model.calculate(&particle.state, observation);
                let new_weight = particle.weight * likelihood.powf(self.learning_rate);
                particle.weight = new_weight.max(self.min_weight_floor);
                total_weight += particle.weight;
            }
        }

        if total_weight <= 0.0 {
            return Err(WeightUpdateError::ZeroTotalWeight);
        }

        // Normalize updated particles
        for &idx in particle_indices {
            if let Some(particle) = storage.get_mut(idx) {
                if particle.active {
                    particle.weight /= total_weight;
                }
            }
        }

        self.update_count += 1;
        Ok(())
    }

    /// Set adaptive learning rate based on filter convergence
    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate.clamp(0.01, 2.0);
    }

    /// Get current learning rate
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Enable/disable weight normalization
    pub fn set_normalize(&mut self, normalize: bool) {
        self.normalize_on_update = normalize;
    }

    /// Get update count
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Reset update count
    pub fn reset_count(&mut self) {
        self.update_count = 0;
    }
}

/// Multi-model Bayesian updater that combines multiple likelihood models
pub struct MultiModelBayesianUpdater {
    models: Vec<(LikelihoodModel, f64)>, // (model, weight)
    base_updater: BayesianWeightUpdater,
}

impl MultiModelBayesianUpdater {
    pub fn new(base_model: LikelihoodModel) -> Self {
        Self {
            models: vec![(base_model, 1.0)],
            base_updater: BayesianWeightUpdater::new(base_model),
        }
    }

    /// Add additional model with weight
    pub fn add_model(&mut self, model: LikelihoodModel, weight: f64) {
        self.models.push((model, weight));
    }

    /// Update weights using ensemble of models
    pub fn update_weights_ensemble(
        &mut self,
        storage: &mut ParticleStorage,
        observation: &ParticleObservation,
    ) -> f64 {
        let mut total_weight = 0.0;

        for particle in storage.active_particles_mut() {
            // Combine likelihoods from all models
            let combined_likelihood: f64 = self.models.iter()
                .map(|(model, weight)| {
                    let likelihood = model.calculate(&particle.state, observation);
                    likelihood.powf(*weight)
                })
                .product();

            let new_weight = particle.weight * combined_likelihood.powf(self.base_updater.learning_rate());
            particle.weight = new_weight.max(self.base_updater.min_weight_floor);
            total_weight += particle.weight;
        }

        // Normalize
        if total_weight > 0.0 {
            for particle in storage.active_particles_mut() {
                particle.weight /= total_weight;
            }
        }

        total_weight
    }

    /// Adapt model weights based on recent performance
    pub fn adapt_model_weights(&mut self, performance_scores: &[f64]) {
        if performance_scores.len() != self.models.len() {
            return;
        }

        // Softmax normalization of performance scores
        let max_score = performance_scores.iter().cloned().fold(f64::MIN, f64::max);
        let exp_scores: Vec<f64> = performance_scores.iter()
            .map(|s| (s - max_score).exp())
            .collect();
        
        let sum_exp: f64 = exp_scores.iter().sum();
        
        if sum_exp > 0.0 {
            for (i, (_, weight)) in self.models.iter_mut().enumerate() {
                *weight = exp_scores[i] / sum_exp;
            }
        }
    }
}

/// Errors for weight update operations
#[derive(Debug, thiserror::Error)]
pub enum WeightUpdateError {
    #[error("Zero total weight after update")]
    ZeroTotalWeight,
    #[error("Invalid particle index")]
    InvalidIndex,
    #[error("No active particles")]
    NoActiveParticles,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_likelihood() {
        let model = LikelihoodModel::Gaussian { mean: 100.0, std_dev: 10.0 };
        
        // Observation at mean should have highest likelihood
        let obs_at_mean = ParticleObservation {
            observed_size: 100,
            time_delta_ns: 1000,
            was_filled: true,
            replenishment_observed: false,
            price_level: PriceLevel::new(100.0),
            side: Side::Buy,
        };
        
        let likelihood_at_mean = model.calculate(&IcebergState::new(1000, 100, PriceLevel::new(100.0), 100), &obs_at_mean);
        
        // Observation far from mean should have lower likelihood
        let obs_far = ParticleObservation {
            observed_size: 500,
            time_delta_ns: 1000,
            was_filled: true,
            replenishment_observed: false,
            price_level: PriceLevel::new(100.0),
            side: Side::Buy,
        };
        
        let likelihood_far = model.calculate(&IcebergState::new(1000, 100, PriceLevel::new(100.0), 100), &obs_far);
        
        assert!(likelihood_at_mean > likelihood_far);
    }

    #[test]
    fn test_bayesian_updater_creation() {
        let model = LikelihoodModel::Gaussian { mean: 100.0, std_dev: 10.0 };
        let updater = BayesianWeightUpdater::new(model);
        
        assert_eq!(updater.learning_rate(), 1.0);
        assert_eq!(updater.update_count(), 0);
    }

    #[test]
    fn test_learning_rate_clamping() {
        let model = LikelihoodModel::Gaussian { mean: 100.0, std_dev: 10.0 };
        let mut updater = BayesianWeightUpdater::new(model);
        
        updater.set_learning_rate(5.0);
        assert!(updater.learning_rate() <= 2.0);
        
        updater.set_learning_rate(0.001);
        assert!(updater.learning_rate() >= 0.01);
    }

    #[test]
    fn test_exponential_likelihood() {
        let model = LikelihoodModel::Exponential { rate: 1.0 };
        
        // Shorter time should have higher likelihood for exponential
        let obs_short = ParticleObservation {
            observed_size: 100,
            time_delta_ns: 100_000_000, // 0.1 seconds
            was_filled: true,
            replenishment_observed: false,
            price_level: PriceLevel::new(100.0),
            side: Side::Buy,
        };
        
        let obs_long = ParticleObservation {
            observed_size: 100,
            time_delta_ns: 1_000_000_000, // 1 second
            was_filled: true,
            replenishment_observed: false,
            price_level: PriceLevel::new(100.0),
            side: Side::Buy,
        };
        
        let state = IcebergState::new(1000, 100, PriceLevel::new(100.0), 100);
        
        let likelihood_short = model.calculate(&state, &obs_short);
        let likelihood_long = model.calculate(&state, &obs_long);
        
        assert!(likelihood_short > likelihood_long);
    }
}
