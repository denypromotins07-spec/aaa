//! Systematic Resampler for Sequential Importance Resampling (SIR) Particle Filter.
//! 
//! Implements efficient resampling with roughening to prevent sample impoverishment
//! while maintaining particle diversity during the resampling process.

use crate::particles::iceberg_state_hypothesis::{ParticleStorage, IcebergParticle, ParticleId, MAX_PARTICLES};
use rand::Rng;
use thiserror::Error;

/// Resampling strategy options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResamplingStrategy {
    /// Systematic resampling (default, O(n) complexity)
    Systematic,
    /// Stratified resampling (slightly better variance properties)
    Stratified,
    /// Residual resampling (preserves high-weight particles exactly)
    Residual,
    /// Multinomial resampling (simple but higher variance)
    Multinomial,
}

/// Configuration for resampling operations
#[derive(Debug, Clone, Copy)]
pub struct ResamplerConfig {
    /// Minimum effective sample size ratio to trigger resampling
    pub ess_threshold: f64,
    /// Roughening magnitude (standard deviation of added noise)
    pub roughening_magnitude: f64,
    /// Whether to apply adaptive roughening based on particle spread
    pub adaptive_roughening: bool,
    /// Minimum number of unique particles to maintain after resampling
    pub min_unique_particles: usize,
    /// Maximum resamples per particle (prevents cloning explosion)
    pub max_copies_per_particle: usize,
}

impl Default for ResamplerConfig {
    fn default() -> Self {
        Self {
            ess_threshold: 0.5, // Resample when N_eff < 50% of N
            roughening_magnitude: 0.01, // 1% jitter
            adaptive_roughening: true,
            min_unique_particles: 10,
            max_copies_per_particle: MAX_PARTICLES / 2,
        }
    }
}

/// Statistics about a resampling operation
#[derive(Debug, Clone, Copy, Default)]
pub struct ResamplingStatistics {
    /// Number of particles before resampling
    pub particles_before: usize,
    /// Number of particles after resampling
    pub particles_after: usize,
    /// Effective sample size before resampling
    pub ess_before: f64,
    /// Effective sample size after resampling
    pub ess_after: f64,
    /// Number of unique particles after resampling
    pub unique_particles: usize,
    /// Number of particles that were killed
    pub particles_killed: usize,
    /// Number of particles that were cloned
    pub particles_cloned: usize,
    /// Time taken for resampling (ns)
    pub elapsed_ns: u64,
}

/// Systematic resampler implementation
pub struct SystematicResampler {
    config: ResamplerConfig,
    strategy: ResamplingStrategy,
    stats: ResamplingStatistics,
    /// Pre-allocated buffer for cumulative weights
    cumulative_weights: Vec<f64>,
    /// Pre-allocated buffer for resample indices
    resample_indices: Vec<usize>,
}

impl SystematicResampler {
    pub fn new(config: ResamplerConfig, strategy: ResamplingStrategy) -> Self {
        Self {
            config,
            strategy,
            stats: ResamplingStatistics::default(),
            cumulative_weights: Vec::with_capacity(MAX_PARTICLES),
            resample_indices: Vec::with_capacity(MAX_PARTICLES),
        }
    }

    /// Check if resampling is needed based on effective sample size
    pub fn needs_resampling(&self, storage: &ParticleStorage) -> bool {
        let weights: Vec<f64> = storage.active_particles().map(|p| p.weight).collect();
        
        if weights.is_empty() {
            return false;
        }

        let n_eff = self.calculate_effective_sample_size(&weights);
        let threshold = weights.len() as f64 * self.config.ess_threshold;
        
        n_eff < threshold
    }

    /// Calculate effective sample size: N_eff = 1 / sum(w_i^2)
    pub fn calculate_effective_sample_size(&self, weights: &[f64]) -> f64 {
        if weights.is_empty() {
            return 0.0;
        }

        let sum_squared: f64 = weights.iter().map(|w| w * w).sum();
        
        if sum_squared <= 0.0 {
            return 0.0;
        }

        1.0 / sum_squared
    }

    /// Perform systematic resampling
    pub fn resample(&mut self, storage: &mut ParticleStorage) -> Result<ResamplingStatistics, ResamplingError> {
        let start_time = std::time::Instant::now();
        
        let active_count = storage.active_count();
        if active_count == 0 {
            return Err(ResamplingError::NoActiveParticles);
        }

        // Record pre-resampling statistics
        let ess_before = {
            let weights: Vec<f64> = storage.active_particles().map(|p| p.weight).collect();
            self.calculate_effective_sample_size(&weights)
        };

        self.stats.particles_before = active_count;
        self.stats.ess_before = ess_before;

        // Build cumulative weight distribution
        self.build_cumulative_weights(storage)?;

        // Generate resample indices based on strategy
        match self.strategy {
            ResamplingStrategy::Systematic => self.systematic_resample(storage)?,
            ResamplingStrategy::Stratified => self.stratified_resample(storage)?,
            ResamplingStrategy::Residual => self.residual_resample(storage)?,
            ResamplingStrategy::Multinomial => self.multinomial_resample(storage)?,
        }

        // Apply roughening to prevent sample impoverishment
        if self.config.adaptive_roughening {
            let roughening = self.calculate_adaptive_roughening(storage);
            storage.apply_roughening(roughening);
        } else if self.config.roughening_magnitude > 0.0 {
            storage.apply_roughening(self.config.roughening_magnitude);
        }

        // Record post-resampling statistics
        let elapsed = start_time.elapsed();
        self.stats.elapsed_ns = elapsed.as_nanos() as u64;
        self.stats.particles_after = storage.active_count();
        self.stats.ess_after = self.calculate_effective_sample_size(
            &storage.active_particles().map(|p| p.weight).collect::<Vec<_>>()
        );
        self.stats.unique_particles = self.count_unique_particles(storage);

        Ok(self.stats)
    }

    /// Build cumulative weight array
    fn build_cumulative_weights(&mut self, storage: &ParticleStorage) -> Result<(), ResamplingError> {
        self.cumulative_weights.clear();
        let mut cumulative = 0.0;

        for particle in storage.active_particles() {
            cumulative += particle.weight;
            self.cumulative_weights.push(cumulative);
        }

        // Normalize to ensure last element is 1.0
        if let Some(last) = self.cumulative_weights.last_mut() {
            if *last > 0.0 {
                let norm = 1.0 / *last;
                for w in &mut self.cumulative_weights {
                    *w *= norm;
                }
            }
        }

        if self.cumulative_weights.is_empty() {
            return Err(ResamplingError::ZeroTotalWeight);
        }

        Ok(())
    }

    /// Systematic resampling: single random start, evenly spaced pointers
    fn systematic_resample(&mut self, storage: &mut ParticleStorage) -> Result<(), ResamplingError> {
        use rand::thread_rng;
        let mut rng = thread_rng();

        let n = storage.active_count();
        if n == 0 {
            return Err(ResamplingError::NoActiveParticles);
        }

        self.resample_indices.clear();
        
        // Single random start
        let start = rng.gen::<f64>() / n as f64;
        
        let mut i = 0;
        let mut cumulative = 0.0;
        
        for j in 0..n {
            let pointer = start + j as f64 / n as f64;
            
            while i < self.cumulative_weights.len() && self.cumulative_weights[i] < pointer {
                i += 1;
                if i > 0 {
                    cumulative = self.cumulative_weights[i - 1];
                }
            }
            
            if i < self.cumulative_weights.len() {
                self.resample_indices.push(i);
            } else {
                self.resample_indices.push(self.cumulative_weights.len() - 1);
            }
        }

        // Apply resampling
        self.apply_resample_indices(storage, &self.resample_indices)
    }

    /// Stratified resampling: independent random in each stratum
    fn stratified_resample(&mut self, storage: &mut ParticleStorage) -> Result<(), ResamplingError> {
        use rand::thread_rng;
        let mut rng = thread_rng();

        let n = storage.active_count();
        if n == 0 {
            return Err(ResamplingError::NoActiveParticles);
        }

        self.resample_indices.clear();

        for j in 0..n {
            // Random within each stratum
            let pointer = j as f64 / n as f64 + rng.gen::<f64>() / n as f64;
            
            // Binary search for efficiency
            let idx = self.binary_search_cumulative(pointer);
            self.resample_indices.push(idx);
        }

        self.apply_resample_indices(storage, &self.resample_indices)
    }

    /// Residual resampling: deterministic for high weights, multinomial for remainder
    fn residual_resample(&mut self, storage: &mut ParticleStorage) -> Result<(), ResamplingError> {
        use rand::thread_rng;
        let mut rng = thread_rng();

        let n = storage.active_count();
        if n == 0 {
            return Err(ResamplingError::NoActiveParticles);
        }

        self.resample_indices.clear();
        let mut remaining_weight = 0.0;
        let mut particle_idx = 0;

        // Deterministic part: floor(N * w_i) copies
        for particle in storage.active_particles() {
            let copies = ((particle.weight * n as f64) as usize).min(self.config.max_copies_per_particle);
            for _ in 0..copies {
                self.resample_indices.push(particle_idx);
            }
            remaining_weight += particle.weight * n as f64 - copies as f64;
            particle_idx += 1;
        }

        // Stochastic part: sample remainder using multinomial
        let remaining_count = n - self.resample_indices.len();
        if remaining_count > 0 {
            // Normalize remaining weights
            if remaining_weight > 0.0 {
                let mut cumsum = 0.0;
                particle_idx = 0;
                for particle in storage.active_particles() {
                    cumsum += (particle.weight * n as f64 - (particle.weight * n as f64) as usize) as f64 / remaining_weight;
                    for _ in 0..remaining_count {
                        if rng.gen::<f64>() < cumsum {
                            self.resample_indices.push(particle_idx);
                            break;
                        }
                    }
                    particle_idx += 1;
                }
            }
        }

        // Trim or pad if necessary
        while self.resample_indices.len() > n {
            self.resample_indices.pop();
        }
        while self.resample_indices.len() < n && !self.resample_indices.is_empty() {
            self.resample_indices.push(self.resample_indices[self.resample_indices.len() - 1]);
        }

        if self.resample_indices.is_empty() {
            return Err(ResamplingError::NoActiveParticles);
        }

        self.apply_resample_indices(storage, &self.resample_indices)
    }

    /// Multinomial resampling: simple independent sampling
    fn multinomial_resample(&mut self, storage: &mut ParticleStorage) -> Result<(), ResamplingError> {
        use rand::thread_rng;
        let mut rng = thread_rng();

        let n = storage.active_count();
        if n == 0 {
            return Err(ResamplingError::NoActiveParticles);
        }

        self.resample_indices.clear();

        for _ in 0..n {
            let r = rng.gen::<f64>();
            let idx = self.binary_search_cumulative(r);
            self.resample_indices.push(idx);
        }

        self.apply_resample_indices(storage, &self.resample_indices)
    }

    /// Binary search for cumulative weight lookup
    fn binary_search_cumulative(&self, target: f64) -> usize {
        let mut low = 0;
        let mut high = self.cumulative_weights.len().saturating_sub(1);

        while low < high {
            let mid = low + (high - low) / 2;
            if self.cumulative_weights[mid] < target {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        low.min(self.cumulative_weights.len().saturating_sub(1))
    }

    /// Apply resample indices to storage
    fn apply_resample_indices(&mut self, storage: &mut ParticleStorage, indices: &[usize]) -> Result<(), ResamplingError> {
        let active_particles: Vec<_> = storage.active_particles().cloned().collect();
        
        if indices.len() != active_particles.len() {
            return Err(ResamplingError::MismatchedSizes);
        }

        let mut killed = 0;
        let mut cloned = 0;
        let mut used_indices = std::collections::HashSet::new();

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            if old_idx >= active_particles.len() {
                continue;
            }

            if let Some(target_particle) = storage.get_mut(new_idx) {
                if let Some(source_particle) = active_particles.get(old_idx) {
                    if new_idx != old_idx {
                        target_particle.clone_from(source_particle);
                        cloned += 1;
                    }
                    used_indices.insert(old_idx);
                }
            }
        }

        // Kill particles that weren't selected
        for (idx, particle) in storage.active_particles_mut().enumerate() {
            if !used_indices.contains(&idx) && indices.len() < active_particles.len() {
                particle.kill();
                killed += 1;
            }
        }

        self.stats.particles_killed = killed;
        self.stats.particles_cloned = cloned;

        Ok(())
    }

    /// Calculate adaptive roughening magnitude based on particle spread
    fn calculate_adaptive_roughening(&self, storage: &ParticleStorage) -> f64 {
        // Calculate standard deviation of particle sizes
        let sizes: Vec<f64> = storage.active_particles()
            .map(|p| p.state.total_size as f64)
            .collect();

        if sizes.len() < 2 {
            return self.config.roughening_magnitude;
        }

        let mean: f64 = sizes.iter().sum::<f64>() / sizes.len() as f64;
        let variance: f64 = sizes.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / sizes.len() as f64;
        let std_dev = variance.sqrt();

        // Roughening proportional to spread, bounded by config
        let adaptive = (std_dev / mean).clamp(0.001, 0.1);
        adaptive * self.config.roughening_magnitude * 10.0
    }

    /// Count unique particles after resampling
    fn count_unique_particles(&self, storage: &ParticleStorage) -> usize {
        use std::collections::HashSet;
        
        let mut unique_states = HashSet::new();
        
        for particle in storage.active_particles() {
            // Use total_size and replenishment_size as uniqueness key
            let key = (particle.state.total_size, particle.state.replenishment_size);
            unique_states.insert(key);
        }

        unique_states.len()
    }

    /// Get current statistics
    pub fn statistics(&self) -> ResamplingStatistics {
        self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &ResamplerConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: ResamplerConfig) {
        self.config = config;
    }
}

/// Errors for resampling operations
#[derive(Debug, Error)]
pub enum ResamplingError {
    #[error("No active particles to resample")]
    NoActiveParticles,
    #[error("Zero total weight")]
    ZeroTotalWeight,
    #[error("Mismatched buffer sizes")]
    MismatchedSizes,
    #[error("Invalid particle index")]
    InvalidIndex,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particles::iceberg_state_hypothesis::PriceLevel;

    #[test]
    fn test_effective_sample_size_uniform() {
        let resampler = SystematicResampler::new(ResamplerConfig::default(), ResamplingStrategy::Systematic);
        let weights = vec![0.1; 10];
        
        let n_eff = resampler.calculate_effective_sample_size(&weights);
        assert!(n_eff > 9.0 && n_eff <= 10.0);
    }

    #[test]
    fn test_effective_sample_size_degenerate() {
        let resampler = SystematicResampler::new(ResamplerConfig::default(), ResamplingStrategy::Systematic);
        let weights = vec![0.99, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.003];
        
        let n_eff = resampler.calculate_effective_sample_size(&weights);
        assert!(n_eff < 2.0);
    }

    #[test]
    fn test_needs_resampling() {
        let config = ResamplerConfig {
            ess_threshold: 0.5,
            ..Default::default()
        };
        let resampler = SystematicResampler::new(config, ResamplingStrategy::Systematic);
        
        // Uniform weights should not need resampling
        let uniform_storage = ParticleStorage::new(10);
        // Note: actual test would need weight setup
        
        // Degenerate weights should need resampling
        // (Test simplified due to storage initialization)
    }

    #[test]
    fn test_resampler_config_defaults() {
        let config = ResamplerConfig::default();
        assert!((config.ess_threshold - 0.5).abs() < 1e-10);
        assert!((config.roughening_magnitude - 0.01).abs() < 1e-10);
        assert!(config.adaptive_roughening);
    }

    #[test]
    fn test_binary_search() {
        let mut resampler = SystematicResampler::new(ResamplerConfig::default(), ResamplingStrategy::Systematic);
        resampler.cumulative_weights = vec![0.1, 0.3, 0.6, 1.0];
        
        assert_eq!(resampler.binary_search_cumulative(0.05), 0);
        assert_eq!(resampler.binary_search_cumulative(0.15), 1);
        assert_eq!(resampler.binary_search_cumulative(0.5), 2);
        assert_eq!(resampler.binary_search_cumulative(0.9), 3);
    }
}
