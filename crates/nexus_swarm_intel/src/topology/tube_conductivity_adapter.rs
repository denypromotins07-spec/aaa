//! Tube Conductivity Adapter for Physarum Network Optimization.
//! 
//! Implements the adaptive tube conductivity mechanism that allows the
//! slime mold network to dynamically rewire based on flow requirements.

use std::collections::HashMap;
use nexus_types::network::NodeId;
use crate::topology::physarum_ode_solver::{PhysarumEdge, MIN_CONDUCTIVITY};
use thiserror::Error;

/// Configuration for conductivity adaptation
#[derive(Debug, Clone, Copy)]
pub struct ConductivityConfig {
    /// Rate at which conductivity adapts to flux changes
    pub adaptation_speed: f64,
    /// Minimum conductivity threshold (prevents disconnection)
    pub min_conductivity: f64,
    /// Maximum conductivity (prevents numerical issues)
    pub max_conductivity: f64,
    /// Hysteresis factor to prevent oscillation
    pub hysteresis: f64,
    /// Whether to enable activity-dependent plasticity
    pub enable_plasticity: bool,
}

impl Default for ConductivityConfig {
    fn default() -> Self {
        Self {
            adaptation_speed: 0.1,
            min_conductivity: MIN_CONDUCTIVITY,
            max_conductivity: 1e6,
            hysteresis: 0.05,
            enable_plasticity: true,
        }
    }
}

/// Statistics about a tube's conductivity history
#[derive(Debug, Clone)]
pub struct TubeStatistics {
    /// Running average of flux magnitude
    pub avg_flux: f64,
    /// Flux variance
    pub flux_variance: f64,
    /// Number of observations
    pub observation_count: u64,
    /// Time since last significant flux
    pub idle_time_ns: u64,
    /// Peak historical flux
    pub peak_flux: f64,
}

impl Default for TubeStatistics {
    fn default() -> Self {
        Self {
            avg_flux: 0.0,
            flux_variance: 0.0,
            observation_count: 0,
            idle_time_ns: 0,
            peak_flux: 0.0,
        }
    }
}

impl TubeStatistics {
    /// Update statistics with new flux observation
    pub fn update(&mut self, flux: f64, current_time_ns: u64) {
        let n = self.observation_count as f64;
        let delta = flux - self.avg_flux;
        
        // Welford's online algorithm for mean and variance
        self.avg_flux += delta / (n + 1.0);
        self.flux_variance += delta * (flux - self.avg_flux);
        
        if self.observation_count > 0 {
            self.flux_variance /= self.observation_count as f64;
        }
        
        self.observation_count += 1;
        self.peak_flux = self.peak_flux.max(flux.abs());
        
        // Reset idle time on significant flux
        if flux.abs() > self.min_conductivity {
            self.idle_time_ns = 0;
        }
    }

    /// Get flux standard deviation
    pub fn flux_std_dev(&self) -> f64 {
        self.flux_variance.sqrt()
    }

    /// Check if tube is idle (low flux for extended period)
    pub fn is_idle(&self, threshold: f64, time_threshold_ns: u64) -> bool {
        self.avg_flux < threshold && self.idle_time_ns > time_threshold_ns
    }
}

/// Tube conductivity adapter with plasticity
pub struct TubeConductivityAdapter {
    config: ConductivityConfig,
    conductivities: HashMap<(NodeId, NodeId), f64>,
    statistics: HashMap<(NodeId, NodeId), TubeStatistics>,
    /// Previous conductivity for hysteresis check
    previous_conductivities: HashMap<(NodeId, NodeId), f64>,
}

impl TubeConductivityAdapter {
    pub fn new(config: ConductivityConfig) -> Self {
        Self {
            config,
            conductivities: HashMap::new(),
            statistics: HashMap::new(),
            previous_conductivities: HashMap::new(),
        }
    }

    /// Initialize conductivity for an edge
    pub fn initialize_edge(&mut self, from: NodeId, to: NodeId, initial_conductivity: f64) {
        let key = (from, to);
        let init = initial_conductivity.clamp(self.config.min_conductivity, self.config.max_conductivity);
        self.conductivities.insert(key, init);
        self.previous_conductivities.insert(key, init);
        self.statistics.entry(key).or_default();
    }

    /// Adapt conductivity based on observed flux
    pub fn adapt_conductivity(
        &mut self,
        from: NodeId,
        to: NodeId,
        flux: f64,
        current_time_ns: u64,
    ) -> Result<f64, ConductivityError> {
        let key = (from, to);

        let conductivity = self.conductivities.get_mut(&key)
            .ok_or(ConductivityError::EdgeNotFound(from, to))?;

        let stats = self.statistics.entry(key).or_default();
        stats.update(flux, current_time_ns);

        let old_conductivity = *conductivity;

        // Compute new conductivity based on flux
        let mut new_conductivity = if self.config.enable_plasticity {
            self.compute_plastic_conductivity(flux, stats)
        } else {
            self.compute_basic_conductivity(flux)
        };

        // Apply bounds
        new_conductivity = new_conductivity.clamp(
            self.config.min_conductivity,
            self.config.max_conductivity,
        );

        // Apply hysteresis to prevent oscillation
        let change = (new_conductivity - old_conductivity).abs();
        if change < self.config.hysteresis * old_conductivity.max(self.config.min_conductivity) {
            new_conductivity = old_conductivity;
        }

        *conductivity = new_conductivity;
        self.previous_conductivities.insert(key, old_conductivity);

        Ok(new_conductivity)
    }

    /// Compute conductivity using activity-dependent plasticity
    fn compute_plastic_conductivity(&self, flux: f64, stats: &TubeStatistics) -> f64 {
        let flux_magnitude = flux.abs();
        
        // Base adaptation: higher flux -> higher conductivity
        let base_adaptation = flux_magnitude * self.config.adaptation_speed;
        
        // Plasticity bonus: consistent high flux strengthens tube more
        let consistency_factor = if stats.observation_count > 10 {
            1.0 + (stats.avg_flux / stats.peak_flux.max(1.0)) * 0.5
        } else {
            1.0
        };

        // Variance penalty: highly variable flux reduces conductivity growth
        let variance_penalty = 1.0 - (stats.flux_std_dev() / stats.avg_flux.max(1.0)).min(0.5);

        base_adaptation * consistency_factor * variance_penalty
    }

    /// Compute basic conductivity without plasticity
    fn compute_basic_conductivity(&self, flux: f64) -> f64 {
        flux.abs() * self.config.adaptation_speed
    }

    /// Prune low-conductivity tubes (with safety check)
    pub fn prune_weak_tubes(&mut self, threshold: f64) -> Vec<(NodeId, NodeId)> {
        let mut pruned = Vec::new();
        let safe_threshold = threshold.max(self.config.min_conductivity * 10.0);

        for (&key, &conductivity) in &self.conductivities {
            if conductivity <= safe_threshold {
                // Only prune if truly weak (not just temporarily low flux)
                if let Some(stats) = self.statistics.get(&key) {
                    if stats.is_idle(safe_threshold, 1_000_000_000) { // 1 second idle
                        // Don't actually remove, just reduce to minimum
                        self.conductivities.insert(key, self.config.min_conductivity);
                        pruned.push(key);
                    }
                }
            }
        }

        pruned
    }

    /// Reinforce high-traffic tubes
    pub fn reinforce_strong_tubes(&mut self, percentile: f64) -> Vec<(NodeId, NodeId)> {
        let mut reinforced = Vec::new();

        // Collect all fluxes
        let mut fluxes: Vec<(f64, (NodeId, NodeId))> = self.statistics.iter()
            .filter_map(|(&key, stats)| {
                if stats.observation_count > 0 {
                    Some((stats.avg_flux, key))
                } else {
                    None
                }
            })
            .collect();

        if fluxes.is_empty() {
            return reinforced;
        }

        // Sort by flux
        fluxes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Find threshold for top percentile
        let threshold_idx = ((fluxes.len() as f64 * (1.0 - percentile)) as usize).min(fluxes.len() - 1);
        let threshold_flux = fluxes.get(threshold_idx).map(|(f, _)| *f).unwrap_or(0.0);

        // Reinforce tubes above threshold
        for (flux, key) in fluxes.iter().take(threshold_idx + 1) {
            if *flux >= threshold_flux {
                if let Some(conductivity) = self.conductivities.get_mut(key) {
                    let boost = 1.0 + self.config.adaptation_speed * 0.5;
                    *conductivity = (*conductivity * boost).min(self.config.max_conductivity);
                    reinforced.push(*key);
                }
            }
        }

        reinforced
    }

    /// Get current conductivity for an edge
    pub fn get_conductivity(&self, from: NodeId, to: NodeId) -> Option<f64> {
        self.conductivities.get(&(from, to)).copied()
    }

    /// Get statistics for an edge
    pub fn get_statistics(&self, from: NodeId, to: NodeId) -> Option<&TubeStatistics> {
        self.statistics.get(&(from, to))
    }

    /// Get all conductivities
    pub fn all_conductivities(&self) -> impl Iterator<Item = ((NodeId, NodeId), f64)> + '_ {
        self.conductivities.iter().map(|(&k, &v)| (k, v))
    }

    /// Reset statistics for an edge
    pub fn reset_statistics(&mut self, from: NodeId, to: NodeId) {
        if let Some(stats) = self.statistics.get_mut(&(from, to)) {
            *stats = TubeStatistics::default();
        }
    }

    /// Get configuration
    pub fn config(&self) -> &ConductivityConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: ConductivityConfig) {
        self.config = config;
    }
}

/// Errors for conductivity operations
#[derive(Debug, Error)]
pub enum ConductivityError {
    #[error("Edge not found: {0:?} -> {1:?}")]
    EdgeNotFound(NodeId, NodeId),
    #[error("Invalid conductivity value")]
    InvalidConductivity,
    #[error("Numerical overflow")]
    Overflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_initialization() {
        let config = ConductivityConfig::default();
        let mut adapter = TubeConductivityAdapter::new(config);

        adapter.initialize_edge(NodeId::new(0), NodeId::new(1), 0.5);

        let cond = adapter.get_conductivity(NodeId::new(0), NodeId::new(1));
        assert!(cond.is_some());
        assert!((cond.unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_conductivity_adaptation() {
        let config = ConductivityConfig::default();
        let mut adapter = TubeConductivityAdapter::new(config);

        adapter.initialize_edge(NodeId::new(0), NodeId::new(1), 0.5);

        // High flux should increase conductivity
        let new_cond = adapter.adapt_conductivity(NodeId::new(0), NodeId::new(1), 10.0, 0).unwrap();
        assert!(new_cond > 0.5);

        // Zero flux should decrease conductivity over time
        let prev = new_cond;
        for i in 1..10 {
            let _ = adapter.adapt_conductivity(NodeId::new(0), NodeId::new(1), 0.0, i * 100_000_000);
        }
        let final_cond = adapter.get_conductivity(NodeId::new(0), NodeId::new(1)).unwrap();
        assert!(final_cond < prev);
    }

    #[test]
    fn test_min_conductivity_enforcement() {
        let config = ConductivityConfig::default();
        let mut adapter = TubeConductivityAdapter::new(config);

        adapter.initialize_edge(NodeId::new(0), NodeId::new(1), 0.5);

        // Many iterations with zero flux
        for i in 0..1000 {
            let _ = adapter.adapt_conductivity(NodeId::new(0), NodeId::new(1), 0.0, i * 100_000_000);
        }

        let cond = adapter.get_conductivity(NodeId::new(0), NodeId::new(1)).unwrap();
        assert!(cond >= config.min_conductivity);
    }

    #[test]
    fn test_statistics_tracking() {
        let config = ConductivityConfig::default();
        let mut adapter = TubeConductivityAdapter::new(config);

        adapter.initialize_edge(NodeId::new(0), NodeId::new(1), 0.5);

        // Simulate varying flux
        for i in 0..100 {
            let flux = if i % 2 == 0 { 10.0 } else { 5.0 };
            let _ = adapter.adapt_conductivity(NodeId::new(0), NodeId::new(1), flux, i * 100_000_000);
        }

        let stats = adapter.get_statistics(NodeId::new(0), NodeId::new(1)).unwrap();
        assert_eq!(stats.observation_count, 100);
        assert!(stats.avg_flux > 0.0);
        assert!(stats.peak_flux >= 10.0);
    }

    #[test]
    fn test_reinforcement() {
        let config = ConductivityConfig::default();
        let mut adapter = TubeConductivityAdapter::new(config);

        // Create multiple edges with different traffic patterns
        adapter.initialize_edge(NodeId::new(0), NodeId::new(1), 0.5);
        adapter.initialize_edge(NodeId::new(0), NodeId::new(2), 0.5);
        adapter.initialize_edge(NodeId::new(0), NodeId::new(3), 0.5);

        // Simulate high traffic on edge 0->1
        for i in 0..50 {
            let _ = adapter.adapt_conductivity(NodeId::new(0), NodeId::new(1), 100.0, i * 100_000_000);
        }

        // Low traffic on others
        for i in 0..50 {
            let _ = adapter.adapt_conductivity(NodeId::new(0), NodeId::new(2), 1.0, i * 100_000_000);
            let _ = adapter.adapt_conductivity(NodeId::new(0), NodeId::new(3), 1.0, i * 100_000_000);
        }

        // Reinforce top 33%
        let reinforced = adapter.reinforce_strong_tubes(0.33);
        assert!(reinforced.contains(&(NodeId::new(0), NodeId::new(1))));
    }
}
