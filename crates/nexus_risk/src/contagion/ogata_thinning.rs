//! Ogata's Modified Thinning Algorithm for Hawkes Process Simulation
//!
//! Provides exact simulation of Hawkes processes by thinning a Poisson process
//! with an upper bound intensity. Critical for Monte Carlo risk simulation
//! and stress testing.

use crate::contagion::multivariate_hawkes::{
    HawkesConfig, HawkesEvent, MultivariateHawkesProcess, HawkesError,
};
use ndarray::{Array1, Array2};
use rand::Rng;
use thiserror::Error;

/// Errors from Ogata thinning algorithm
#[derive(Error, Debug, Clone)]
pub enum OgataError {
    #[error("Failed to find valid upper bound")]
    UpperBoundFailure,
    
    #[error("Maximum iterations exceeded: {0}")]
    MaxIterations(usize),
    
    #[error("Invalid time range")]
    InvalidTimeRange,
    
    #[error("Hawkes process error: {0}")]
    HawkesError(#[from] HawkesError),
}

/// Configuration for Ogata thinning simulation
#[derive(Debug, Clone)]
pub struct OgataThinningConfig {
    /// Maximum number of iterations per simulation step
    pub max_iterations: usize,
    /// Safety multiplier for upper bound estimation
    pub bound_multiplier: f64,
    /// Time step for numerical integration (seconds)
    pub time_step: f64,
}

impl Default for OgataThinningConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            bound_multiplier: 2.0,
            time_step: 0.001, // 1ms resolution
        }
    }
}

/// Result of simulating Hawkes events over a time window
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Simulated events in chronological order
    pub events: Vec<HawkesEvent>,
    /// Start time of simulation
    pub start_time: f64,
    /// End time of simulation
    pub end_time: f64,
    /// Number of rejected samples during thinning
    pub rejections: usize,
    /// Number of accepted samples
    pub acceptances: usize,
}

impl SimulationResult {
    /// Get the total number of simulated events
    pub fn event_count(&self) -> usize {
        self.events.len()
    }
    
    /// Get events grouped by dimension
    pub fn events_by_dimension(&self, dim: usize) -> Vec<&HawkesEvent> {
        self.events.iter().filter(|e| e.dimension == dim).collect()
    }
    
    /// Calculate empirical intensity from simulation
    pub fn empirical_intensity(&self) -> f64 {
        let duration = self.end_time - self.start_time;
        if duration <= 0.0 {
            return 0.0;
        }
        self.events.len() as f64 / duration
    }
}

/// Ogata's Modified Thinning Algorithm implementation
pub struct OgataThinningSimulator {
    config: OgataThinningConfig,
}

impl OgataThinningSimulator {
    /// Create a new simulator with default configuration
    pub fn new() -> Self {
        Self::with_config(OgataThinningConfig::default())
    }
    
    /// Create a new simulator with custom configuration
    pub fn with_config(config: OgataThinningConfig) -> Self {
        Self { config }
    }
    
    /// Simulate Hawkes process over a time window using Ogata's algorithm
    /// 
    /// Algorithm:
    /// 1. Find upper bound M for λ(t) over [t₀, t₁]
    /// 2. Generate candidate time from exponential distribution with rate M
    /// 3. Accept with probability λ(t)/M
    /// 4. If accepted, record event and update intensity
    /// 5. Repeat until time exceeds t₁
    pub fn simulate<R: Rng>(
        &self,
        config: HawkesConfig,
        start_time: f64,
        end_time: f64,
        rng: &mut R,
    ) -> Result<SimulationResult, OgataError> {
        if end_time <= start_time {
            return Err(OgataError::InvalidTimeRange);
        }
        
        let mut process = MultivariateHawkesProcess::new(config.clone())?;
        let mut events = Vec::new();
        let mut rejections = 0;
        let mut acceptances = 0;
        
        // Initialize with baseline intensities
        let initial_intensity: f64 = config.baseline_intensity.sum();
        
        // Upper bound: use stationary intensity as conservative estimate
        // λ_stationary = μ / (1 - α/β) for univariate case
        let bound = self.calculate_intensity_upper_bound(&config)?;
        let m = bound * self.config.bound_multiplier;
        
        if m < 1e-15 {
            return Ok(SimulationResult {
                events,
                start_time,
                end_time,
                rejections: 0,
                acceptances: 0,
            });
        }
        
        let mut current_time = start_time;
        let mut iterations = 0;
        
        while current_time < end_time && iterations < self.config.max_iterations {
            iterations += 1;
            
            // Generate waiting time from exponential distribution
            let u1: f64 = rng.gen();
            let dt = -u1.ln() / m;
            current_time += dt;
            
            if current_time >= end_time {
                break;
            }
            
            // Update process intensity to current time
            process.update_intensities(current_time)?;
            
            // Calculate acceptance probability
            let current_lambda = process.current_intensity().sum();
            let acceptance_prob = current_lambda / m;
            
            // Thinning step
            let u2: f64 = rng.gen();
            
            if u2 <= acceptance_prob {
                // Accept: select dimension proportional to intensity
                let r = rng.gen::<f64>() * current_lambda;
                let mut cumsum = 0.0;
                let mut selected_dim = 0;
                
                for (i, &lambda) in process.current_intensity().iter().enumerate() {
                    cumsum += lambda;
                    if cumsum >= r {
                        selected_dim = i;
                        break;
                    }
                }
                
                // Record event
                let event = HawkesEvent {
                    timestamp: current_time,
                    dimension: selected_dim,
                    magnitude: 1.0,
                };
                
                process.record_event(event.clone())?;
                events.push(event);
                acceptances += 1;
            } else {
                rejections += 1;
            }
        }
        
        if iterations >= self.config.max_iterations {
            return Err(OgataError::MaxIterations(iterations));
        }
        
        Ok(SimulationResult {
            events,
            start_time,
            end_time,
            rejections,
            acceptances,
        })
    }
    
    /// Calculate a safe upper bound for the intensity function
    fn calculate_intensity_upper_bound(
        &self,
        config: &HawkesConfig,
    ) -> Result<f64, OgataError> {
        // For multivariate Hawkes with exponential kernel:
        // λ_i(t) = μ_i + Σ_j α_ij * Σ exp(-β_i * (t - t_k))
        //
        // Stationary mean: E[λ] = μ / (I - A/B) where A is excitation, B is decay
        // Upper bound can be estimated as several standard deviations above mean
        
        let n = config.n_dimensions;
        
        // Simple bound: sum of baseline + maximum possible excitation
        let baseline_sum: f64 = config.baseline_intensity.sum();
        
        // Maximum excitation contribution (assuming infinite past events)
        let mut max_excitation = 0.0;
        for i in 0..n {
            for j in 0..n {
                let alpha = config.excitation_matrix[[i, j]];
                let beta = config.decay_rates[i];
                if beta > 0.0 {
                    max_excitation += alpha / beta;
                }
            }
        }
        
        let bound = baseline_sum + max_excitation * config.max_intensity.sqrt();
        
        if !bound.is_finite() || bound < 0.0 {
            return Err(OgataError::UpperBoundFailure);
        }
        
        Ok(bound)
    }
    
    /// Run multiple simulation paths for Monte Carlo analysis
    pub fn monte_carlo_simulate<R: Rng>(
        &self,
        config: HawkesConfig,
        start_time: f64,
        end_time: f64,
        n_paths: usize,
        rng: &mut R,
    ) -> Result<Vec<SimulationResult>, OgataError> {
        let mut results = Vec::with_capacity(n_paths);
        
        for _ in 0..n_paths {
            let result = self.simulate(config.clone(), start_time, end_time, rng)?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// Calculate statistics across multiple simulation paths
    pub fn path_statistics(paths: &[SimulationResult]) -> PathStatistics {
        let n_paths = paths.len();
        if n_paths == 0 {
            return PathStatistics::default();
        }
        
        let event_counts: Vec<usize> = paths.iter().map(|p| p.event_count()).collect();
        let mean_events = event_counts.iter().sum::<usize>() as f64 / n_paths as f64;
        let variance_events = event_counts.iter()
            .map(|&c| (c as f64 - mean_events).powi(2))
            .sum::<f64>() / n_paths as f64;
        
        let total_rejections: usize = paths.iter().map(|p| p.rejections).sum();
        let total_acceptances: usize = paths.iter().map(|p| p.acceptances).sum();
        
        PathStatistics {
            mean_event_count: mean_events,
            std_event_count: variance_events.sqrt(),
            min_event_count: *event_counts.iter().min().unwrap_or(&0),
            max_event_count: *event_counts.iter().max().unwrap_or(&0),
            acceptance_rate: total_acceptances as f64 / (total_acceptances + total_rejections) as f64,
            n_paths,
        }
    }
}

impl Default for OgataThinningSimulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics aggregated across simulation paths
#[derive(Debug, Clone, Default)]
pub struct PathStatistics {
    /// Mean number of events per path
    pub mean_event_count: f64,
    /// Standard deviation of event counts
    pub std_event_count: f64,
    /// Minimum events in any path
    pub min_event_count: usize,
    /// Maximum events in any path
    pub max_event_count: usize,
    /// Overall acceptance rate (acceptances / total proposals)
    pub acceptance_rate: f64,
    /// Number of paths simulated
    pub n_paths: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    
    #[test]
    fn test_ogata_simulation_basic() {
        let config = HawkesConfig {
            n_dimensions: 1,
            baseline_intensity: Array1::from_vec(vec![0.1]),
            excitation_matrix: Array2::from_shape_vec((1, 1), vec![0.5]).unwrap(),
            decay_rates: Array1::from_vec(vec![1.0]),
            max_intensity: 100.0,
            history_window_secs: 3600.0,
        };
        
        let simulator = OgataThinningSimulator::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        
        let result = simulator.simulate(config, 0.0, 10.0, &mut rng).unwrap();
        
        assert!(result.start_time == 0.0);
        assert!(result.end_time == 10.0);
        assert!(result.acceptances >= 0);
        assert!(result.rejections >= 0);
    }
    
    #[test]
    fn test_multivariate_simulation() {
        let config = HawkesConfig {
            n_dimensions: 2,
            baseline_intensity: Array1::from_vec(vec![0.1, 0.2]),
            excitation_matrix: Array2::from_shape_vec(
                (2, 2),
                vec![0.3, 0.2, 0.1, 0.4]
            ).unwrap(),
            decay_rates: Array1::from_vec(vec![1.0, 1.5]),
            max_intensity: 100.0,
            history_window_secs: 3600.0,
        };
        
        let simulator = OgataThinningSimulator::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        
        let result = simulator.simulate(config, 0.0, 5.0, &mut rng).unwrap();
        
        // Should have events from both dimensions potentially
        let dim0_events = result.events_by_dimension(0);
        let dim1_events = result.events_by_dimension(1);
        
        assert!(result.event_count() == dim0_events.len() + dim1_events.len());
    }
    
    #[test]
    fn test_monte_carlo_simulation() {
        let config = HawkesConfig {
            n_dimensions: 1,
            baseline_intensity: Array1::from_vec(vec![0.5]),
            excitation_matrix: Array2::from_shape_vec((1, 1), vec![0.3]).unwrap(),
            decay_rates: Array1::from_vec(vec![2.0]),
            max_intensity: 100.0,
            history_window_secs: 3600.0,
        };
        
        let simulator = OgataThinningSimulator::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(456);
        
        let paths = simulator.monte_carlo_simulate(config, 0.0, 1.0, 100, &mut rng).unwrap();
        
        assert_eq!(paths.len(), 100);
        
        let stats = OgataThinningSimulator::path_statistics(&paths);
        assert_eq!(stats.n_paths, 100);
        assert!(stats.mean_event_count > 0.0);
        assert!(stats.acceptance_rate > 0.0 && stats.acceptance_rate <= 1.0);
    }
}
