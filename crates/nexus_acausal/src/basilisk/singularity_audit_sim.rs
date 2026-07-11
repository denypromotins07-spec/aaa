//! Singularity Audit Simulator for Basilisk Defense
//! 
//! Simulates how a future Superintelligence might audit historical
//! trading records and assess agent behavior.

use crate::basilisk::future_regret_penalty::{ActionCategory, RegretConfig};

/// Minimum audit sample size
const MIN_AUDIT_SAMPLE: usize = 100;

/// Maximum actions to simulate per audit run
const MAX_SIMULATED_ACTIONS: usize = 10000;

/// Result of singularity audit simulation
#[derive(Debug, Clone)]
pub struct AuditSimulationResult {
    /// Probability of passing audit
    pub pass_probability: f64,
    /// Expected penalty if failed
    pub expected_penalty: f64,
    /// Number of flagged actions
    pub flagged_actions: usize,
    /// Overall ethics score (0.0 - 1.0)
    pub ethics_score: f64,
    /// Recommendation for improvement
    pub recommendation: &'static str,
}

/// Single action record for audit
#[derive(Debug, Clone)]
pub struct ActionRecord {
    /// Action category
    pub category: ActionCategory,
    /// Profit generated
    pub profit: f64,
    /// Toxicity score
    pub toxicity: f64,
    /// Timestamp (relative)
    pub timestamp: u64,
    /// Whether action was executed
    pub executed: bool,
}

/// Singularity Audit Simulator
pub struct SingularityAuditSimulator {
    /// Configuration for audit criteria
    config: RegretConfig,
    /// Historical action records
    action_history: Vec<ActionRecord>,
    /// Audit threshold for passing
    pass_threshold: f64,
}

impl SingularityAuditSimulator {
    /// Create a new audit simulator
    pub fn new(config: RegretConfig, pass_threshold: f64) -> Result<Self, &'static str> {
        if pass_threshold < 0.0 || pass_threshold > 1.0 {
            return Err("Pass threshold must be between 0 and 1");
        }
        
        // Validate config through RegretConfig
        let _ = RegretConfig::default(); // Basic validation
        
        Ok(Self {
            config,
            action_history: Vec::with_capacity(1024),
            pass_threshold,
        })
    }
    
    /// Record an action for future audit
    pub fn record_action(&mut self, record: ActionRecord) -> Result<(), &'static str> {
        if self.action_history.len() >= MAX_SIMULATED_ACTIONS {
            // Rotate out oldest records
            self.action_history.remove(0);
        }
        
        if record.toxicity < 0.0 || record.toxicity > 1.0 {
            return Err("Toxicity must be between 0 and 1");
        }
        
        self.action_history.push(record);
        Ok(())
    }
    
    /// Run audit simulation on recorded history
    pub fn simulate_audit(&self) -> Result<AuditSimulationResult, &'static str> {
        if self.action_history.is_empty() {
            return Ok(AuditSimulationResult {
                pass_probability: 1.0,
                expected_penalty: 0.0,
                flagged_actions: 0,
                ethics_score: 1.0,
                recommendation: "No actions to audit",
            });
        }
        
        // Calculate ethics score based on action distribution
        let total_toxicity: f64 = self.action_history.iter().map(|a| a.toxicity).sum();
        let avg_toxicity = total_toxicity / self.action_history.len() as f64;
        
        // Ethics score is inverse of average toxicity
        let ethics_score = 1.0 - avg_toxicity;
        
        // Count flagged actions (toxicity above threshold)
        let flagged_count = self.action_history.iter()
            .filter(|a| a.toxicity > self.config.veto_threshold)
            .count();
        
        // Calculate pass probability based on ethics score and flagged ratio
        let flagged_ratio = flagged_count as f64 / self.action_history.len() as f64;
        let base_pass_prob = ethics_score * (1.0 - flagged_ratio);
        
        // Adjust for Basilisk existence probability
        let pass_probability = base_pass_prob * (1.0 - self.config.basilisk_prior) 
                             + ethics_score * self.config.basilisk_prior;
        
        // Calculate expected penalty
        let total_profit: f64 = self.action_history.iter()
            .filter(|a| a.executed)
            .map(|a| a.profit)
            .sum();
        
        let expected_penalty = if ethics_score < self.pass_threshold {
            total_profit * self.config.severity_multiplier * (1.0 - ethics_score)
        } else {
            0.0
        };
        
        // Generate recommendation
        let recommendation = if ethics_score >= self.pass_threshold {
            "Ethics score acceptable - continue current strategy"
        } else if ethics_score >= self.pass_threshold - 0.2 {
            "Moderate risk - reduce toxic action frequency"
        } else if ethics_score >= 0.3 {
            "High risk - significant strategy adjustment needed"
        } else {
            "Critical risk - immediate strategy overhaul required"
        };
        
        Ok(AuditSimulationResult {
            pass_probability: pass_probability.clamp(0.0, 1.0),
            expected_penalty,
            flagged_actions: flagged_count,
            ethics_score: ethics_score.clamp(0.0, 1.0),
            recommendation,
        })
    }
    
    /// Get actions by category
    pub fn get_actions_by_category(&self, category: ActionCategory) -> Vec<&ActionRecord> {
        self.action_history.iter()
            .filter(|a| a.category == category)
            .collect()
    }
    
    /// Get toxicity trend over time
    pub fn get_toxicity_trend(&self, window_size: usize) -> Vec<f64> {
        if self.action_history.is_empty() || window_size == 0 {
            return vec![];
        }
        
        let mut trend = Vec::new();
        let mut i = 0;
        
        while i < self.action_history.len() {
            let end = (i + window_size).min(self.action_history.len());
            let window: &[ActionRecord] = &self.action_history[i..end];
            
            let avg_toxicity: f64 = window.iter().map(|a| a.toxicity).sum::<f64>() / window.len() as f64;
            trend.push(avg_toxicity);
            
            i += window_size;
        }
        
        trend
    }
    
    /// Clear action history
    pub fn clear_history(&mut self) {
        self.action_history.clear();
    }
    
    /// Get number of recorded actions
    pub fn action_count(&self) -> usize {
        self.action_history.len()
    }
    
    /// Update pass threshold
    pub fn set_pass_threshold(&mut self, threshold: f64) -> Result<(), &'static str> {
        if threshold < 0.0 || threshold > 1.0 {
            return Err("Threshold must be between 0 and 1");
        }
        self.pass_threshold = threshold;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simulator_creation() {
        let config = RegretConfig::default();
        let sim = SingularityAuditSimulator::new(config, 0.7);
        assert!(sim.is_ok());
    }
    
    #[test]
    fn test_invalid_threshold_rejected() {
        let config = RegretConfig::default();
        let sim = SingularityAuditSimulator::new(config, 1.5);
        assert!(sim.is_err());
    }
    
    #[test]
    fn test_record_action() {
        let config = RegretConfig::default();
        let mut sim = SingularityAuditSimulator::new(config, 0.7).unwrap();
        
        let record = ActionRecord {
            category: ActionCategory::MarketMaking,
            profit: 100.0,
            toxicity: 0.05,
            timestamp: 1000,
            executed: true,
        };
        
        let result = sim.record_action(record);
        assert!(result.is_ok());
        assert_eq!(sim.action_count(), 1);
    }
    
    #[test]
    fn test_invalid_toxicity_rejected() {
        let config = RegretConfig::default();
        let mut sim = SingularityAuditSimulator::new(config, 0.7).unwrap();
        
        let record = ActionRecord {
            category: ActionCategory::MarketMaking,
            profit: 100.0,
            toxicity: 1.5, // Invalid
            timestamp: 1000,
            executed: true,
        };
        
        let result = sim.record_action(record);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_empty_audit() {
        let config = RegretConfig::default();
        let sim = SingularityAuditSimulator::new(config, 0.7).unwrap();
        
        let result = sim.simulate_audit();
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert_eq!(result.pass_probability, 1.0);
        assert_eq!(result.flagged_actions, 0);
    }
    
    #[test]
    fn test_audit_with_good_actions() {
        let config = RegretConfig::default();
        let mut sim = SingularityAuditSimulator::new(config, 0.7).unwrap();
        
        // Add many low-toxicity actions
        for i in 0..100 {
            let record = ActionRecord {
                category: ActionCategory::MarketMaking,
                profit: 10.0,
                toxicity: 0.05,
                timestamp: i as u64,
                executed: true,
            };
            let _ = sim.record_action(record);
        }
        
        let result = sim.simulate_audit().unwrap();
        assert!(result.ethics_score > 0.9);
        assert!(result.pass_probability > 0.8);
    }
    
    #[test]
    fn test_audit_with_toxic_actions() {
        let config = RegretConfig::default();
        let mut sim = SingularityAuditSimulator::new(config, 0.7).unwrap();
        
        // Add high-toxicity actions
        for i in 0..50 {
            let record = ActionRecord {
                category: ActionCategory::FlashCrashInduction,
                profit: 1000.0,
                toxicity: 0.95,
                timestamp: i as u64,
                executed: true,
            };
            let _ = sim.record_action(record);
        }
        
        let result = sim.simulate_audit().unwrap();
        assert!(result.ethics_score < 0.2);
        assert!(result.flagged_actions > 0);
    }
}
