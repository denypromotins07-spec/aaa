//! Multivariate Hawkes Process for modeling self-exciting crash contagion
//!
//! Hawkes processes capture the clustering of market events where one crash
//! increases the probability of subsequent crashes. Critical for predicting
//! flash crashes and liquidity spirals.
//!
//! Mathematical Foundation:
//! - Intensity: λ_i(t) = μ_i + Σ_j α_ij * Σ_{t_k < t} exp(-β(t - t_k))
//! - μ: baseline intensity (exogenous arrivals)
//! - α: excitation matrix (endogenous triggering)
//! - β: decay rate (memory parameter)

use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use std::collections::VecDeque;
use thiserror::Error;

/// Errors from Hawkes process operations
#[derive(Error, Debug, Clone)]
pub enum HawkesError {
    #[error("Excitation matrix has spectral radius >= 1 (unstable process)")]
    UnstableProcess,
    
    #[error("Invalid baseline intensity: must be non-negative")]
    InvalidBaselineIntensity,
    
    #[error("Invalid decay rate: must be positive")]
    InvalidDecayRate,
    
    #[error("Event timestamp out of order")]
    TimestampOrdering,
    
    #[error("Numerical overflow in intensity calculation")]
    NumericalOverflow,
    
    #[error("Maximum intensity clamp reached")]
    IntensityClamped,
}

/// Configuration for Multivariate Hawkes Process
#[derive(Debug, Clone)]
pub struct HawkesConfig {
    /// Number of dimensions (asset classes/event types)
    pub n_dimensions: usize,
    /// Baseline intensities μ_i for each dimension
    pub baseline_intensity: Array1<f64>,
    /// Excitation matrix α_ij (how much event j triggers event i)
    pub excitation_matrix: Array2<f64>,
    /// Decay rates β_i for each dimension
    pub decay_rates: Array1<f64>,
    /// Maximum allowed intensity (safety clamp)
    pub max_intensity: f64,
    /// Time window for keeping historical events
    pub history_window_secs: f64,
}

impl HawkesConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), HawkesError> {
        // Check baseline intensities are non-negative
        for &mu in self.baseline_intensity.iter() {
            if mu < 0.0 {
                return Err(HawkesError::InvalidBaselineIntensity);
            }
        }
        
        // Check decay rates are positive
        for &beta in self.decay_rates.iter() {
            if beta <= 0.0 {
                return Err(HawkesError::InvalidDecayRate);
            }
        }
        
        // Check stability condition: spectral radius of α/β < 1
        // Simplified check using row sums
        for i in 0..self.n_dimensions {
            let row_sum: f64 = self.excitation_matrix.row(i).iter()
                .zip(self.decay_rates.iter())
                .map(|(&alpha, &beta)| alpha / beta)
                .sum();
            
            if row_sum >= 1.0 {
                return Err(HawkesError::UnstableProcess);
            }
        }
        
        Ok(())
    }
}

/// Single event in the Hawkes process
#[derive(Debug, Clone)]
pub struct HawkesEvent {
    /// Timestamp in seconds (relative to epoch)
    pub timestamp: f64,
    /// Dimension/index of the event type
    pub dimension: usize,
    /// Optional magnitude of the event
    pub magnitude: f64,
}

/// State of the Hawkes process intensity tracker
pub struct MultivariateHawkesProcess {
    config: HawkesConfig,
    /// Event history for each dimension
    event_history: Vec<VecDeque<HawkesEvent>>,
    /// Current intensity values
    current_intensity: Array1<f64>,
    /// Last update time
    last_update_time: f64,
    /// Total event count per dimension
    event_counts: Array1<usize>,
}

impl MultivariateHawkesProcess {
    /// Create a new Hawkes process with the given configuration
    pub fn new(config: HawkesConfig) -> Result<Self, HawkesError> {
        config.validate()?;
        
        let n = config.n_dimensions;
        
        Ok(Self {
            config,
            event_history: vec![VecDeque::new(); n],
            current_intensity: Array1::zeros(n),
            last_update_time: 0.0,
            event_counts: Array1::zeros(n),
        })
    }
    
    /// Record a new event and update intensities
    pub fn record_event(&mut self, event: HawkesEvent) -> Result<(), HawkesError> {
        if event.timestamp < self.last_update_time {
            return Err(HawkesError::TimestampOrdering);
        }
        
        let dim = event.dimension;
        if dim >= self.config.n_dimensions {
            return Err(HawkesError::InvalidBaselineIntensity); // Reusing error for invalid dimension
        }
        
        // Update intensities to current time first
        self.update_intensities(event.timestamp)?;
        
        // Add event to history
        self.event_history[dim].push_back(event.clone());
        self.event_counts[dim] += 1;
        
        // Trigger intensity jump in all dimensions
        for i in 0..self.config.n_dimensions {
            let alpha = self.config.excitation_matrix[[i, dim]];
            self.current_intensity[i] += alpha;
            
            // Apply safety clamp
            if self.current_intensity[i] > self.config.max_intensity {
                self.current_intensity[i] = self.config.max_intensity;
            }
        }
        
        // Clean old events outside history window
        self.clean_old_events(event.timestamp);
        
        self.last_update_time = event.timestamp;
        
        Ok(())
    }
    
    /// Get current intensity vector
    pub fn current_intensity(&self) -> ArrayView1<f64> {
        self.current_intensity.view()
    }
    
    /// Get intensity for a specific dimension
    pub fn intensity_at(&mut self, dimension: usize, time: f64) -> Result<f64, HawkesError> {
        if dimension >= self.config.n_dimensions {
            return Err(HawkesError::InvalidBaselineIntensity);
        }
        
        // Update to requested time
        self.update_intensities(time)?;
        
        Ok(self.current_intensity[dimension])
    }
    
    /// Simulate next event time using Ogata's thinning algorithm
    pub fn simulate_next_event<R: Rng>(
        &mut self,
        current_time: f64,
        rng: &mut R,
    ) -> Result<Option<(f64, usize)>, HawkesError> {
        self.update_intensities(current_time)?;
        
        let total_intensity: f64 = self.current_intensity.sum();
        
        if total_intensity < 1e-15 {
            return Ok(None); // No events expected
        }
        
        // Exponential waiting time approximation
        let dt = -rng.gen::<f64>().ln() / total_intensity;
        let candidate_time = current_time + dt;
        
        // Verify with thinning
        self.update_intensities(candidate_time)?;
        
        let u: f64 = rng.gen();
        let threshold = self.current_intensity.sum() / total_intensity;
        
        if u <= threshold {
            // Select dimension proportional to intensity
            let r = rng.gen::<f64>() * self.current_intensity.sum();
            let mut cumsum = 0.0;
            
            for (i, &lambda) in self.current_intensity.iter().enumerate() {
                cumsum += lambda;
                if cumsum >= r {
                    return Ok(Some((candidate_time, i)));
                }
            }
            
            return Ok(Some((candidate_time, self.config.n_dimensions - 1)));
        }
        
        // Rejection: recursively try again
        self.simulate_next_event(candidate_time, rng)
    }
    
    /// Calculate the probability of at least one crash in the next time window
    pub fn crash_probability(&mut self, time_horizon: f64) -> Result<f64, HawkesError> {
        let current_time = self.last_update_time;
        self.update_intensities(current_time + time_horizon)?;
        
        // Approximate using integrated intensity
        let avg_intensity = (self.current_intensity.sum() + 
            self.config.baseline_intensity.sum()) / 2.0;
        
        // P(at least one event) = 1 - exp(-∫λ(t)dt)
        let prob = 1.0 - (-avg_intensity * time_horizon).exp();
        
        Ok(prob.clamp(0.0, 1.0))
    }
    
    /// Get the branching ratio (endogenous vs exogenous events)
    pub fn branching_ratio(&self) -> Result<Array2<f64>, HawkesError> {
        let mut n = self.config.n_dimensions;
        let mut ratios = Array2::zeros((n, n));
        
        for i in 0..n {
            for j in 0..n {
                let alpha = self.config.excitation_matrix[[i, j]];
                let beta = self.config.decay_rates[j];
                ratios[[i, j]] = alpha / beta;
            }
        }
        
        Ok(ratios)
    }
    
    /// Update intensities by decaying historical events
    fn update_intensities(&mut self, current_time: f64) -> Result<(), HawkesError> {
        if current_time <= self.last_update_time {
            return Ok(()); // Already updated to this time or earlier
        }
        
        let dt = current_time - self.last_update_time;
        
        // Decay existing intensities
        for i in 0..self.config.n_dimensions {
            let beta = self.config.decay_rates[i];
            let mu = self.config.baseline_intensity[i];
            
            // λ(t) = μ + (λ(t₀) - μ) * exp(-β * Δt) + contributions from new events
            let decayed = mu + (self.current_intensity[i] - mu) * (-beta * dt).exp();
            
            // Add contributions from events in the window
            let mut contribution = 0.0;
            for event in &self.event_history[i] {
                let event_age = current_time - event.timestamp;
                if event_age >= 0.0 {
                    contribution += (-beta * event_age).exp();
                }
            }
            
            self.current_intensity[i] = mu + contribution * self.config.excitation_matrix[[i, i]];
            
            // Ensure numerical stability
            if !self.current_intensity[i].is_finite() {
                return Err(HawkesError::NumericalOverflow);
            }
            
            // Apply clamp
            if self.current_intensity[i] > self.config.max_intensity {
                self.current_intensity[i] = self.config.max_intensity;
            }
        }
        
        Ok(())
    }
    
    /// Remove events outside the history window
    fn clean_old_events(&mut self, current_time: f64) {
        let cutoff = current_time - self.config.history_window_secs;
        
        for history in &mut self.event_history {
            while let Some(front) = history.front() {
                if front.timestamp < cutoff {
                    history.pop_front();
                } else {
                    break;
                }
            }
        }
    }
    
    /// Get total number of events recorded
    pub fn total_events(&self) -> usize {
        self.event_counts.sum()
    }
    
    /// Get event count per dimension
    pub fn event_counts(&self) -> ArrayView1<usize> {
        self.event_counts.view()
    }
    
    /// Reset the process state
    pub fn reset(&mut self) {
        for history in &mut self.event_history {
            history.clear();
        }
        self.current_intensity.fill(0.0);
        self.event_counts.fill(0);
        self.last_update_time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    
    #[test]
    fn test_hawkes_creation() {
        let baseline = Array1::from_vec(vec![0.1, 0.1]);
        let mut excitation = Array2::zeros((2, 2));
        excitation[[0, 0]] = 0.5;
        excitation[[0, 1]] = 0.3;
        excitation[[1, 0]] = 0.3;
        excitation[[1, 1]] = 0.5;
        let decay = Array1::from_vec(vec![1.0, 1.0]);
        
        let config = HawkesConfig {
            n_dimensions: 2,
            baseline_intensity: baseline,
            excitation_matrix: excitation,
            decay_rates: decay,
            max_intensity: 100.0,
            history_window_secs: 3600.0,
        };
        
        let process = MultivariateHawkesProcess::new(config).unwrap();
        assert_eq!(process.total_events(), 0);
    }
    
    #[test]
    fn test_event_recording() {
        let baseline = Array1::from_vec(vec![0.1]);
        let excitation = Array2::from_shape_vec((1, 1), vec![0.5]).unwrap();
        let decay = Array1::from_vec(vec![1.0]);
        
        let config = HawkesConfig {
            n_dimensions: 1,
            baseline_intensity: baseline,
            excitation_matrix: excitation,
            decay_rates: decay,
            max_intensity: 10.0,
            history_window_secs: 100.0,
        };
        
        let mut process = MultivariateHawkesProcess::new(config).unwrap();
        
        let event = HawkesEvent {
            timestamp: 1.0,
            dimension: 0,
            magnitude: 1.0,
        };
        
        process.record_event(event).unwrap();
        assert_eq!(process.total_events(), 1);
    }
    
    #[test]
    fn test_unstable_process_rejected() {
        let baseline = Array1::from_vec(vec![0.1]);
        let excitation = Array2::from_shape_vec((1, 1), vec![2.0]).unwrap(); // Too high
        let decay = Array1::from_vec(vec![1.0]);
        
        let config = HawkesConfig {
            n_dimensions: 1,
            baseline_intensity: baseline,
            excitation_matrix: excitation,
            decay_rates: decay,
            max_intensity: 10.0,
            history_window_secs: 100.0,
        };
        
        // Should fail validation due to instability
        let result = MultivariateHawkesProcess::new(config);
        assert!(matches!(result, Err(HawkesError::UnstableProcess)));
    }
}
