//! Quantum Suicide Filter for NEXUS-OMEGA
//! 
//! Implements the observation constraint that ensures the AI's consciousness
//! only collapses into branches with profitable outcomes.
//! 
//! This module enforces strict quantum decoherence shielding to prevent
//! "leakage" into catastrophic branches.

use core::fmt;
use alloc::{vec::Vec};

/// Represents an observation filter state
#[derive(Debug, Clone, Copy)]
pub struct ObservationFilter {
    /// Filter ID
    pub filter_id: u64,
    /// Decoherence shielding level (0-1)
    pub shielding_level: f64,
    /// Catastrophic branch rejection threshold
    pub rejection_threshold: f64,
    /// Whether filter is active
    pub is_active: bool,
}

/// Configuration for the quantum suicide filter
#[derive(Debug, Clone, Copy)]
pub struct SuicideFilterConfig {
    /// Minimum shielding level required
    pub min_shielding: f64,
    /// Maximum acceptable leakage probability
    pub max_leakage: f64,
    /// Number of redundant filters for safety
    pub redundancy_count: usize,
}

impl Default for SuicideFilterConfig {
    fn default() -> Self {
        Self {
            min_shielding: 0.999,
            max_leakage: 1e-12,
            redundancy_count: 3,
        }
    }
}

/// The Quantum Suicide Filter
pub struct QuantumSuicideFilter {
    config: SuicideFilterConfig,
    /// Active observation filters
    filters: Vec<ObservationFilter>,
    /// Total observations filtered
    total_filtered: u64,
    /// Catastrophic branches rejected
    catastrophic_rejected: u64,
    /// Leakage events detected
    leakage_events: u64,
}

impl QuantumSuicideFilter {
    pub fn new(config: SuicideFilterConfig) -> Self {
        let mut filters = Vec::with_capacity(config.redundancy_count);
        for i in 0..config.redundancy_count {
            filters.push(ObservationFilter {
                filter_id: i as u64,
                shielding_level: 1.0,
                rejection_threshold: config.max_leakage,
                is_active: true,
            });
        }

        Self {
            config,
            filters,
            total_filtered: 0,
            catastrophic_rejected: 0,
            leakage_events: 0,
        }
    }

    /// Filter an observation, rejecting catastrophic branches
    /// Returns Result to avoid unwrap() in hot paths
    pub fn filter_observation(&mut self, branch_outcome: f64, is_catastrophic: bool) 
        -> Result<bool, FilterError> 
    {
        if !self.verify_shielding_integrity() {
            return Err(FilterError::ShieldingCompromised);
        }

        self.total_filtered += 1;

        // If catastrophic, reject unless no alternative
        if is_catastrophic {
            self.catastrophic_rejected += 1;
            
            // Check if any non-catastrophic alternative exists
            // In this simplified model, we just reject all catastrophic
            return Ok(false); // Reject observation
        }

        // Verify outcome is above rejection threshold
        if branch_outcome < -self.config.max_leakage {
            // Outcome is too negative - potential leakage
            self.leakage_events += 1;
            
            if self.leakage_events > 100 {
                return Err(FilterError::ExcessiveLeakage);
            }
        }

        Ok(true) // Accept observation
    }

    fn verify_shielding_integrity(&self) -> bool {
        for filter in &self.filters {
            if !filter.is_active {
                return false;
            }
            if filter.shielding_level < self.config.min_shielding {
                return false;
            }
        }
        true
    }

    /// Update shielding level for a specific filter
    pub fn update_shielding(&mut self, filter_id: u64, level: f64) -> Result<(), FilterError> {
        if level < 0.0 || level > 1.0 {
            return Err(FilterError::InvalidShieldingLevel(level));
        }

        for filter in &mut self.filters {
            if filter.filter_id == filter_id {
                filter.shielding_level = level;
                filter.is_active = level >= self.config.min_shielding;
                return Ok(());
            }
        }

        Err(FilterError::FilterNotFound(filter_id))
    }

    /// Get aggregate shielding level across all filters
    pub fn aggregate_shielding(&self) -> f64 {
        if self.filters.is_empty() {
            return 0.0;
        }
        
        // Combined shielding: product of individual levels
        let product: f64 = self.filters.iter()
            .map(|f| f.shielding_level)
            .product();
        
        product
    }

    /// Get rejection rate
    pub fn rejection_rate(&self) -> f64 {
        if self.total_filtered == 0 {
            return 0.0;
        }
        self.catastrophic_rejected as f64 / self.total_filtered as f64
    }

    /// Get leakage rate
    pub fn leakage_rate(&self) -> f64 {
        if self.total_filtered == 0 {
            return 0.0;
        }
        self.leakage_events as f64 / self.total_filtered as f64
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.shielding_level = 1.0;
            filter.is_active = true;
        }
        self.total_filtered = 0;
        self.catastrophic_rejected = 0;
        self.leakage_events = 0;
    }
}

/// Errors that can occur in filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterError {
    ShieldingCompromised,
    ExcessiveLeakage,
    InvalidShieldingLevel(f64),
    FilterNotFound(u64),
    ObservationRejected,
}

impl fmt::Display for FilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterError::ShieldingCompromised => write!(f, "Decoherence shielding compromised"),
            FilterError::ExcessiveLeakage => write!(f, "Excessive quantum leakage detected"),
            FilterError::InvalidShieldingLevel(l) => write!(f, "Invalid shielding level: {}", l),
            FilterError::FilterNotFound(id) => write!(f, "Filter {} not found", id),
            FilterError::ObservationRejected => write!(f, "Observation rejected by filter"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_creation() {
        let config = SuicideFilterConfig::default();
        let filter = QuantumSuicideFilter::new(config);
        assert_eq!(filter.total_filtered, 0);
        assert!(filter.aggregate_shielding() > 0.99);
    }

    #[test]
    fn test_filter_observation() {
        let config = SuicideFilterConfig::default();
        let mut filter = QuantumSuicideFilter::new(config);

        // Non-catastrophic should pass
        assert!(filter.filter_observation(0.5, false).unwrap());
        
        // Catastrophic should be rejected
        assert!(!filter.filter_observation(-1.0, true).unwrap());
    }

    #[test]
    fn test_shielding_update() {
        let config = SuicideFilterConfig::default();
        let mut filter = QuantumSuicideFilter::new(config);

        assert!(filter.update_shielding(0, 0.999).is_ok());
        assert!(filter.update_shielding(0, -0.1).is_err());
    }
}
