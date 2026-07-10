//! Iceberg Order State Hypothesis for Swarm Particle Filter.
//! 
//! Each particle represents a hypothesis about the hidden size and
//! replenishment characteristics of an iceberg order.

use nexus_types::market::{VenueId, Side, PriceLevel};
use std::time::Instant;

/// Maximum number of particles supported per filter (fixed allocation)
pub const MAX_PARTICLES: usize = 4096;

/// Unique identifier for a particle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParticleId(pub u64);

impl ParticleId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// State representation of a potential iceberg order
#[derive(Debug, Clone, Copy)]
pub struct IcebergState {
    /// Total hidden size in shares
    pub total_size: u64,
    /// Currently visible (displayed) size
    pub visible_size: u64,
    /// Size already executed
    pub executed_size: u64,
    /// Replenishment size when visible portion is filled
    pub replenishment_size: u64,
    /// Price level of the iceberg
    pub price_level: PriceLevel,
    /// Estimated replenishment rate (shares per second)
    pub replenishment_rate: f64,
    /// Time since last replenishment observation (ns)
    pub time_since_replenishment_ns: u64,
    /// Confidence in this hypothesis [0, 1]
    pub confidence: f64,
}

impl IcebergState {
    /// Create a new hypothesis with default parameters
    pub fn new(
        total_size: u64,
        visible_size: u64,
        price_level: PriceLevel,
        replenishment_size: u64,
    ) -> Self {
        Self {
            total_size,
            visible_size,
            executed_size: 0,
            replenishment_size,
            price_level,
            replenishment_rate: 0.0,
            time_since_replenishment_ns: 0,
            confidence: 0.5, // Start with neutral confidence
        }
    }

    /// Update state after observing a fill
    pub fn on_fill(&mut self, fill_size: u64, timestamp_ns: u64) {
        self.executed_size += fill_size;
        
        // Check if replenishment occurred
        if self.visible_size < self.replenishment_size {
            // Visible portion was consumed, should have replenished
            self.time_since_replenishment_ns = timestamp_ns;
        }
    }

    /// Update replenishment rate estimate
    pub fn update_replenishment_rate(&mut self, observed_rate: f64) {
        // Exponential moving average
        let alpha = 0.2;
        self.replenishment_rate = self.replenishment_rate * (1.0 - alpha) + observed_rate * alpha;
    }

    /// Get remaining hidden size
    pub fn remaining_hidden_size(&self) -> u64 {
        self.total_size.saturating_sub(self.executed_size)
    }

    /// Check if hypothesis suggests iceberg is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.executed_size >= self.total_size
    }

    /// Get expected lifetime in milliseconds
    pub fn expected_lifetime_ms(&self) -> f64 {
        if self.replenishment_rate <= 0.0 {
            return f64::MAX;
        }
        self.remaining_hidden_size() as f64 / self.replenishment_rate * 1000.0
    }
}

/// Particle representing a single hypothesis in the swarm
#[derive(Debug, Clone)]
pub struct IcebergParticle {
    pub id: ParticleId,
    pub state: IcebergState,
    /// Weight (likelihood) of this particle given observations
    pub weight: f64,
    /// Cumulative weight for resampling
    pub cumulative_weight: f64,
    /// Number of times this particle has been resampled
    pub resample_count: u32,
    /// Last time this particle was updated
    pub last_update: Instant,
    /// Whether this particle is active (not killed during resampling)
    pub active: bool,
}

impl IcebergParticle {
    pub fn new(id: ParticleId, state: IcebergState, initial_weight: f64) -> Self {
        Self {
            id,
            state,
            weight: initial_weight,
            cumulative_weight: 0.0,
            resample_count: 0,
            last_update: Instant::now(),
            active: true,
        }
    }

    /// Normalize particle weight
    pub fn normalize_weight(&mut self, total_weight: f64) {
        if total_weight > 0.0 {
            self.weight = (self.weight / total_weight).clamp(0.0, 1.0);
        }
    }

    /// Reset particle for reuse (object pooling)
    pub fn reset(&mut self, new_state: IcebergState) {
        self.state = new_state;
        self.weight = 1.0;
        self.cumulative_weight = 0.0;
        self.resample_count = 0;
        self.last_update = Instant::now();
        self.active = true;
    }

    /// Kill this particle (mark as inactive)
    pub fn kill(&mut self) {
        self.active = false;
        self.weight = 0.0;
    }

    /// Clone from another particle (for resampling)
    pub fn clone_from(&mut self, other: &IcebergParticle) {
        self.state = other.state;
        self.weight = other.weight;
        self.resample_count = other.resample_count + 1;
        self.last_update = Instant::now();
        self.active = true;
    }
}

/// Pre-allocated particle storage for zero-allocation filtering
pub struct ParticleStorage {
    particles: Vec<IcebergParticle>,
    active_count: usize,
    next_id: u64,
}

impl ParticleStorage {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.min(MAX_PARTICLES);
        let mut particles = Vec::with_capacity(capacity);
        
        // Pre-initialize particles with default states
        for i in 0..capacity {
            let default_state = IcebergState::new(1000, 100, PriceLevel::new(100.0), 100);
            particles.push(IcebergParticle::new(ParticleId::new(i as u64), default_state, 1.0));
        }

        Self {
            particles,
            active_count: capacity,
            next_id: capacity as u64,
        }
    }

    /// Get mutable reference to a particle by index
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut IcebergParticle> {
        self.particles.get_mut(idx)
    }

    /// Get immutable reference to a particle by index
    pub fn get(&self, idx: usize) -> Option<&IcebergParticle> {
        self.particles.get(idx)
    }

    /// Get all active particles
    pub fn active_particles(&self) -> impl Iterator<Item = &IcebergParticle> {
        self.particles.iter().filter(|p| p.active)
    }

    /// Get mutable iterator over active particles
    pub fn active_particles_mut(&mut self) -> impl Iterator<Item = &mut IcebergParticle> {
        self.particles.iter_mut().filter(|p| p.active)
    }

    /// Get total capacity
    pub fn capacity(&self) -> usize {
        self.particles.capacity()
    }

    /// Get count of active particles
    pub fn active_count(&self) -> usize {
        self.particles.iter().filter(|p| p.active).count()
    }

    /// Initialize particles with diverse hypotheses
    pub fn initialize_diverse(
        &mut self,
        base_price: PriceLevel,
        side: Side,
        size_range: (u64, u64),
        replenishment_range: (u64, u64),
    ) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        self.active_count = 0;
        
        for (i, particle) in self.particles.iter_mut().enumerate() {
            let total_size = rng.gen_range(size_range.0..=size_range.1);
            let visible_size = total_size / 10; // Typical iceberg shows ~10%
            let replenishment_size = rng.gen_range(replenishment_range.0..=replenishment_range.1);

            let state = IcebergState::new(total_size, visible_size, base_price, replenishment_size);
            particle.reset(state);
            particle.id = ParticleId::new(i as u64);
            
            self.active_count += 1;
        }
    }

    /// Add jitter to particle states (for roughening during resampling)
    pub fn apply_roughening(&mut self, magnitude: f64) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for particle in self.particles.iter_mut().filter(|p| p.active) {
            // Add Gaussian noise to total_size estimate
            let noise = rng.gen::<f64>() * magnitude;
            let current = particle.state.total_size as f64;
            particle.state.total_size = ((current + noise * current).max(100.0) as u64)
                .max(particle.state.executed_size);

            // Add noise to replenishment size
            let rep_noise = rng.gen::<f64>() * magnitude * 0.5;
            let current_rep = particle.state.replenishment_size as f64;
            particle.state.replenishment_size = ((current_rep + rep_noise * current_rep).max(10.0) as u64);
        }
    }
}

/// Effective sample size calculator for adaptive resampling
pub struct EffectiveSampleSizeCalculator;

impl EffectiveSampleSizeCalculator {
    /// Calculate N_eff from particle weights
    /// N_eff = 1 / sum(w_i^2)
    /// Low N_eff indicates particle degeneracy
    pub fn calculate(weights: &[f64]) -> f64 {
        if weights.is_empty() {
            return 0.0;
        }

        let sum_squared: f64 = weights.iter().map(|w| w * w).sum();
        
        if sum_squared <= 0.0 {
            return 0.0;
        }

        1.0 / sum_squared
    }

    /// Check if resampling is needed based on N_eff threshold
    pub fn needs_resampling(weights: &[f64], threshold_ratio: f64) -> bool {
        let n_eff = Self::calculate(weights);
        let n_particles = weights.len() as f64;
        
        // Resample if N_eff drops below threshold ratio of total particles
        n_eff < n_particles * threshold_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iceberg_state_creation() {
        let state = IcebergState::new(10000, 1000, PriceLevel::new(100.0), 1000);
        
        assert_eq!(state.total_size, 10000);
        assert_eq!(state.visible_size, 1000);
        assert_eq!(state.executed_size, 0);
        assert!(!state.is_exhausted());
    }

    #[test]
    fn test_iceberg_state_fill_update() {
        let mut state = IcebergState::new(10000, 1000, PriceLevel::new(100.0), 1000);
        
        state.on_fill(500, 1000000);
        assert_eq!(state.executed_size, 500);
        assert_eq!(state.remaining_hidden_size(), 9500);
    }

    #[test]
    fn test_particle_storage_initialization() {
        let mut storage = ParticleStorage::new(100);
        
        assert_eq!(storage.capacity(), 100);
        assert_eq!(storage.active_count(), 100);
    }

    #[test]
    fn test_effective_sample_size() {
        // Uniform weights should give N_eff = N
        let uniform_weights = vec![0.1; 10];
        let n_eff = EffectiveSampleSizeCalculator::calculate(&uniform_weights);
        assert!(n_eff > 9.0 && n_eff <= 10.0);

        // Degenerate weights (one particle dominates) should give low N_eff
        let degenerate_weights = vec![0.99, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.003];
        let n_eff_degenerate = EffectiveSampleSizeCalculator::calculate(&degenerate_weights);
        assert!(n_eff_degenerate < 2.0);
    }

    #[test]
    fn test_needs_resampling() {
        let uniform = vec![0.1; 10];
        assert!(!EffectiveSampleSizeCalculator::needs_resampling(&uniform, 0.5));

        let degenerate = vec![0.99, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.003];
        assert!(EffectiveSampleSizeCalculator::needs_resampling(&degenerate, 0.5));
    }
}
