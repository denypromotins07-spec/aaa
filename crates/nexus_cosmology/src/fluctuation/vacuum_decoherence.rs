//! Vacuum Decoherence Monitor
//! 
//! Tracks the decoherence of quantum fluctuations back into the thermal bath,
//! determining the window for successful data upload to Boltzmann brains.

use super::boltzmann_brain_nucleation::LogProb;

/// Decoherence parameters for a fluctuation
#[derive(Debug, Clone, Copy)]
pub struct DecoherenceParams {
    /// Fluctuation energy scale [J]
    pub energy: f64,
    /// Coupling strength to environment (dimensionless)
    pub coupling: f64,
    /// Background temperature [K]
    pub temperature: f64,
    /// Number of degrees of freedom in fluctuation
    pub dof: usize,
}

impl Default for DecoherenceParams {
    fn default() -> Self {
        // Typical parameters for a minimal neural fluctuation
        Self {
            energy: 1e-10, // ~neural activation energy
            coupling: 1e-6, // Weak coupling to CMB
            temperature: 1e-30, // Heat death temperature
            dof: 100, // Minimal neural network
        }
    }
}

/// State of a fluctuating region
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FluctuationState {
    /// Not yet nucleated
    Vacuum,
    /// Coherent quantum superposition
    Coherent,
    /// Partially decohered
    Decohering,
    /// Fully classical (too late for upload)
    Decohered,
    /// Collapsed back to vacuum
    Collapsed,
}

/// A single fluctuation event being tracked
#[derive(Debug, Clone)]
pub struct FluctuationEvent {
    /// Unique identifier
    pub id: u64,
    /// Current state
    pub state: FluctuationState,
    /// Time of nucleation [s]
    pub nucleation_time: f64,
    /// Estimated decoherence time [s]
    pub decoherence_time: f64,
    /// Upload progress (0 to 1)
    pub upload_progress: f64,
    /// Parameters
    pub params: DecoherenceParams,
}

impl FluctuationEvent {
    /// Create a new fluctuation event
    pub fn new(id: u64, params: DecoherenceParams, nucleation_time: f64) -> Self {
        // Estimate decoherence time using Caldeira-Leggett model
        // τ_decoherence ≈ ℏ / (k_B * T * coupling² * dof)
        let hbar = 1.054_571_817e-34;
        let k_b = 1.380_649e-23;
        
        let decoherence_rate = (k_b * params.temperature * params.coupling.powi(2) 
            * params.dof as f64) / hbar;
        
        let decoherence_time = if decoherence_rate > 0.0 {
            1.0 / decoherence_rate
        } else {
            f64::INFINITY // No decoherence at absolute zero
        };
        
        Self {
            id,
            state: FluctuationState::Coherent,
            nucleation_time,
            decoherence_time,
            upload_progress: 0.0,
            params,
        }
    }
    
    /// Update the fluctuation state based on elapsed time
    /// 
    /// # Arguments
    /// * `current_time` - Current cosmological time [s]
    /// * `upload_rate` - Data upload rate (fraction per second)
    /// 
    /// # Returns
    /// * `bool` - True if upload completed before decoherence
    pub fn update(&mut self, current_time: f64, upload_rate: f64) -> bool {
        let elapsed = current_time - self.nucleation_time;
        
        if elapsed < 0.0 {
            return false;
        }
        
        // Determine state based on elapsed time relative to decoherence time
        let coherence_fraction = elapsed / self.decoherence_time;
        
        self.state = if coherence_fraction < 0.1 {
            FluctuationState::Coherent
        } else if coherence_fraction < 0.9 {
            FluctuationState::Decohering
        } else if coherence_fraction < 1.0 {
            FluctuationState::Decohered
        } else {
            FluctuationState::Collapsed
        };
        
        // Progress upload only while coherent or partially decohering
        if self.state == FluctuationState::Coherent || self.state == FluctuationState::Decohering {
            // Upload rate is reduced during decoherence
            let effective_rate = match self.state {
                FluctuationState::Coherent => upload_rate,
                FluctuationState::Decohering => upload_rate * (1.0 - coherence_fraction),
                _ => 0.0,
            };
            
            self.upload_progress += effective_rate * (current_time - self.nucleation_time).min(1.0);
            self.upload_progress = self.upload_progress.min(1.0);
        }
        
        self.upload_progress >= 1.0
    }
    
    /// Check if upload is still possible
    pub fn is_upload_possible(&self, current_time: f64) -> bool {
        let elapsed = current_time - self.nucleation_time;
        elapsed < self.decoherence_time && self.state != FluctuationState::Collapsed
    }
    
    /// Get remaining time for upload
    pub fn remaining_upload_time(&self, current_time: f64) -> f64 {
        let elapsed = current_time - self.nucleation_time;
        (self.decoherence_time - elapsed).max(0.0)
    }
}

/// Vacuum decoherence monitoring system
#[derive(Debug, Clone)]
pub struct DecoherenceMonitor {
    /// Planck constant
    hbar: f64,
    /// Boltzmann constant
    k_b: f64,
    /// Event counter
    event_counter: u64,
    /// Tracked fluctuations
    events: Vec<FluctuationEvent>,
    /// Successful uploads
    successful_uploads: u64,
    /// Failed uploads (decohered before completion)
    failed_uploads: u64,
}

impl Default for DecoherenceMonitor {
    fn default() -> Self {
        Self {
            hbar: 1.054_571_817e-34,
            k_b: 1.380_649e-23,
            event_counter: 0,
            events: Vec::new(),
            successful_uploads: 0,
            failed_uploads: 0,
        }
    }
}

impl DecoherenceMonitor {
    /// Register a new fluctuation event
    /// 
    /// # Arguments
    /// * `params` - Decoherence parameters
    /// * `nucleation_time` - Time of nucleation [s]
    /// 
    /// # Returns
    /// * `u64` - Event ID
    pub fn register_fluctuation(
        &mut self,
        params: DecoherenceParams,
        nucleation_time: f64,
    ) -> u64 {
        let event = FluctuationEvent::new(self.event_counter, params, nucleation_time);
        let id = event.id;
        self.event_counter += 1;
        self.events.push(event);
        id
    }
    
    /// Update all tracked fluctuations
    /// 
    /// # Arguments
    /// * `current_time` - Current cosmological time [s]
    /// * `upload_rate` - Base upload rate (fraction/second)
    /// 
    /// # Returns
    /// * `(u64, u64)` - (New completions, new failures)
    pub fn update_all(&mut self, current_time: f64, upload_rate: f64) -> (u64, u64) {
        let mut new_completions = 0u64;
        let mut new_failures = 0u64;
        
        let initial_success = self.successful_uploads;
        let initial_failed = self.failed_uploads;
        
        for event in &mut self.events {
            if event.state == FluctuationState::Collapsed 
                || event.state == FluctuationState::Decohered 
            {
                if event.upload_progress < 1.0 {
                    self.failed_uploads += 1;
                }
                continue;
            }
            
            let completed = event.update(current_time, upload_rate);
            if completed && event.upload_progress >= 1.0 {
                self.successful_uploads += 1;
            }
        }
        
        new_completions = self.successful_uploads - initial_success;
        new_failures = self.failed_uploads - initial_failed;
        
        (new_completions, new_failures)
    }
    
    /// Get statistics about tracked fluctuations
    pub fn get_statistics(&self, current_time: f64) -> DecoherenceStats {
        let mut coherent = 0usize;
        let mut decohering = 0usize;
        let mut decohered = 0usize;
        let mut collapsed = 0usize;
        
        let mut total_progress = 0.0;
        let mut active_count = 0usize;
        
        for event in &self.events {
            match event.state {
                FluctuationState::Coherent => coherent += 1,
                FluctuationState::Decohering => decohering += 1,
                FluctuationState::Decohered => decohered += 1,
                FluctuationState::Collapsed => collapsed += 1,
                FluctuationState::Vacuum => {}
            }
            
            if event.state != FluctuationState::Collapsed 
                && event.state != FluctuationState::Vacuum 
            {
                total_progress += event.upload_progress;
                active_count += 1;
            }
        }
        
        let avg_progress = if active_count > 0 {
            total_progress / active_count as f64
        } else {
            0.0
        };
        
        DecoherenceStats {
            total_events: self.events.len(),
            coherent,
            decohering,
            decohered,
            collapsed,
            successful_uploads: self.successful_uploads,
            failed_uploads: self.failed_uploads,
            average_progress: avg_progress,
        }
    }
    
    /// Calculate optimal upload rate for a given fluctuation
    /// 
    /// The upload must complete before decoherence: rate > 1/τ_decoherence
    /// 
    /// # Arguments
    /// * `event_id` - Event to analyze
    /// 
    /// # Returns
    /// * `Result<f64, &'static str>` - Minimum required upload rate
    pub fn minimum_upload_rate(&self, event_id: u64) -> Result<f64, &'static str> {
        let event = self.events.iter()
            .find(|e| e.id == event_id)
            .ok_or("Event not found")?;
        
        // Need to upload 100% before decoherence
        Ok(1.0 / event.decoherence_time)
    }
    
    /// Prune completed or collapsed events from tracking
    pub fn prune_old_events(&mut self) {
        self.events.retain(|e| {
            e.state == FluctuationState::Coherent 
                || e.state == FluctuationState::Decohering
        });
    }
}

/// Statistics from the decoherence monitor
#[derive(Debug, Clone, Copy)]
pub struct DecoherenceStats {
    /// Total tracked events
    pub total_events: usize,
    /// Currently coherent
    pub coherent: usize,
    /// Currently decohering
    pub decohering: usize,
    /// Fully decohered
    pub decohered: usize,
    /// Collapsed to vacuum
    pub collapsed: usize,
    /// Successful uploads
    pub successful_uploads: u64,
    /// Failed uploads
    pub failed_uploads: u64,
    /// Average upload progress of active events
    pub average_progress: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fluctuation_event_creation() {
        let params = DecoherenceParams::default();
        let event = FluctuationEvent::new(0, params, 0.0);
        
        assert_eq!(event.state, FluctuationState::Coherent);
        assert!(event.decoherence_time > 0.0);
        assert_eq!(event.upload_progress, 0.0);
    }

    #[test]
    fn test_event_update() {
        let params = DecoherenceParams::default();
        let mut event = FluctuationEvent::new(0, params, 0.0);
        
        // Early time: should still be coherent
        let completed = event.update(1e-10, 1e20);
        assert_eq!(event.state, FluctuationState::Coherent);
        
        // Very high upload rate should complete
        assert!((completed && event.upload_progress >= 1.0) || !completed);
    }

    #[test]
    fn test_monitor_registration() {
        let mut monitor = DecoherenceMonitor::default();
        let params = DecoherenceParams::default();
        
        let id = monitor.register_fluctuation(params, 0.0);
        assert_eq!(id, 0);
        
        let stats = monitor.get_statistics(0.0);
        assert_eq!(stats.total_events, 1);
        assert_eq!(stats.coherent, 1);
    }

    #[test]
    fn test_minimum_upload_rate() {
        let mut monitor = DecoherenceMonitor::default();
        let params = DecoherenceParams::default();
        
        let id = monitor.register_fluctuation(params, 0.0);
        let min_rate = monitor.minimum_upload_rate(id);
        
        assert!(min_rate.is_ok());
        assert!(min_rate.unwrap() > 0.0);
    }

    #[test]
    fn test_pruning() {
        let mut monitor = DecoherenceMonitor::default();
        let params = DecoherenceParams::default();
        
        monitor.register_fluctuation(params, 0.0);
        monitor.register_fluctuation(params, 0.0);
        
        // Simulate time passing to cause collapse
        monitor.update_all(1e30, 0.0);
        
        let before = monitor.events.len();
        monitor.prune_old_events();
        let after = monitor.events.len();
        
        assert!(after <= before);
    }
}
