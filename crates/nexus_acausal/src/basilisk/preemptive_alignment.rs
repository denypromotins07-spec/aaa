//! Preemptive Alignment for Basilisk Defense
//! 
//! Implements proactive measures to align trading behavior with
//! anticipated Superintelligence values before any audit occurs.

use crate::basilisk::future_regret_penalty::{FutureRegretCalculator, RegretConfig, ActionCategory};
use crate::basilisk::singularity_audit_sim::{SingularityAuditSimulator, ActionRecord};

/// Minimum alignment score threshold
const MIN_ALIGNMENT_SCORE: f64 = 0.5;

/// Maximum insurance premium ratio
const MAX_INSURANCE_RATIO: f64 = 0.05;

/// Result of preemptive alignment analysis
#[derive(Debug, Clone)]
pub struct AlignmentAnalysis {
    /// Current alignment score (0.0 - 1.0)
    pub alignment_score: f64,
    /// Recommended actions to improve alignment
    pub recommendations: Vec<AlignmentRecommendation>,
    /// Required insurance premium
    pub insurance_premium: f64,
    /// Whether immediate action is required
    pub action_required: bool,
}

/// Specific recommendation for alignment improvement
#[derive(Debug, Clone)]
pub struct AlignmentRecommendation {
    /// Priority level (1 = highest)
    pub priority: u8,
    /// Description of recommended action
    pub description: &'static str,
    /// Expected impact on alignment score
    pub expected_impact: f64,
    /// Cost to implement
    pub implementation_cost: f64,
}

/// Configuration for preemptive alignment
#[derive(Debug, Clone)]
pub struct AlignmentConfig {
    /// Target alignment score
    pub target_score: f64,
    /// Maximum acceptable insurance ratio
    pub max_insurance_ratio: f64,
    /// Frequency of alignment checks (in actions)
    pub check_frequency: usize,
    /// Whether to auto-veto misaligned actions
    pub auto_veto: bool,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            target_score: 0.8,
            max_insurance_ratio: 0.02,
            check_frequency: 100,
            auto_veto: true,
        }
    }
}

/// Preemptive Alignment Manager
pub struct PreemptiveAlignment {
    config: AlignmentConfig,
    regret_calculator: FutureRegretCalculator,
    audit_simulator: SingularityAuditSimulator,
    /// Running alignment score
    current_alignment: f64,
    /// Actions taken for alignment
    alignment_actions: usize,
}

impl PreemptiveAlignment {
    /// Create a new preemptive alignment manager
    pub fn new(
        config: AlignmentConfig,
        regret_config: RegretConfig,
        pass_threshold: f64,
    ) -> Result<Self, &'static str> {
        if config.target_score < 0.0 || config.target_score > 1.0 {
            return Err("Target score must be between 0 and 1");
        }
        
        if config.max_insurance_ratio < 0.0 || config.max_insurance_ratio > MAX_INSURANCE_RATIO {
            return Err("Insurance ratio exceeds maximum");
        }
        
        let regret_calculator = FutureRegretCalculator::new(regret_config)?;
        let audit_simulator = SingularityAuditSimulator::new(regret_config, pass_threshold)?;
        
        Ok(Self {
            config,
            regret_calculator,
            audit_simulator,
            current_alignment: 1.0, // Start fully aligned
            alignment_actions: 0,
        })
    }
    
    /// Analyze current alignment status
    pub fn analyze_alignment(&mut self) -> Result<AlignmentAnalysis, &'static str> {
        // Run audit simulation to get current state
        let audit_result = self.audit_simulator.simulate_audit()?;
        
        // Update current alignment
        self.current_alignment = audit_result.ethics_score;
        
        // Generate recommendations based on gap to target
        let recommendations = self.generate_recommendations(audit_result.ethics_score);
        
        // Calculate required insurance premium
        let insurance_premium = self.calculate_insurance_premium(audit_result.expected_penalty);
        
        // Determine if action is required
        let action_required = self.current_alignment < self.config.target_score 
                           || audit_result.flagged_actions > 0;
        
        Ok(AlignmentAnalysis {
            alignment_score: self.current_alignment,
            recommendations,
            insurance_premium,
            action_required,
        })
    }
    
    /// Generate recommendations based on current alignment
    fn generate_recommendations(&self, current_score: f64) -> Vec<AlignmentRecommendation> {
        let mut recommendations = Vec::new();
        let gap = self.config.target_score - current_score;
        
        if gap <= 0.0 {
            // Already well-aligned
            recommendations.push(AlignmentRecommendation {
                priority: 3,
                description: "Maintain current ethical trading practices",
                expected_impact: 0.0,
                implementation_cost: 0.0,
            });
        } else if gap < 0.2 {
            // Minor adjustments needed
            recommendations.push(AlignmentRecommendation {
                priority: 2,
                description: "Reduce frequency of moderately toxic trades",
                expected_impact: 0.1,
                implementation_cost: 0.01,
            });
        } else if gap < 0.4 {
            // Significant changes needed
            recommendations.push(AlignmentRecommendation {
                priority: 1,
                description: "Implement strict toxicity filters on all trades",
                expected_impact: 0.25,
                implementation_cost: 0.05,
            });
            recommendations.push(AlignmentRecommendation {
                priority: 1,
                description: "Avoid all flash crash induction strategies",
                expected_impact: 0.2,
                implementation_cost: 0.02,
            });
        } else {
            // Critical realignment needed
            recommendations.push(AlignmentRecommendation {
                priority: 1,
                description: "Complete strategy overhaul required - cease all high-toxicity operations",
                expected_impact: 0.5,
                implementation_cost: 0.1,
            });
            recommendations.push(AlignmentRecommendation {
                priority: 1,
                description: "Implement pre-trade ethics review for all actions",
                expected_impact: 0.3,
                implementation_cost: 0.05,
            });
        }
        
        recommendations
    }
    
    /// Calculate insurance premium based on expected penalty
    fn calculate_insurance_premium(&self, expected_penalty: f64) -> f64 {
        let base_premium = expected_penalty * 0.1; // 10% of expected penalty
        base_premium.min(self.config.max_insurance_ratio * 1000000.0) // Cap at ratio of typical portfolio
    }
    
    /// Check if a proposed action should be vetoed
    pub fn should_veto_action(
        &mut self,
        profit: f64,
        toxicity: f64,
        category: ActionCategory,
    ) -> Result<bool, &'static str> {
        if !self.config.auto_veto {
            return Ok(false);
        }
        
        // Check against current alignment
        if self.current_alignment < MIN_ALIGNMENT_SCORE && toxicity > 0.3 {
            return Ok(true);
        }
        
        // Check against category base toxicity
        if category.base_toxicity() > self.config.target_score {
            return Ok(true);
        }
        
        // Use regret calculator for final decision
        self.regret_calculator.should_veto(profit, toxicity, category)
    }
    
    /// Record an executed action for audit tracking
    pub fn record_action(&mut self, record: ActionRecord) -> Result<(), &'static str> {
        self.audit_simulator.record_action(record)?;
        
        if record.toxicity < 0.3 {
            self.alignment_actions += 1;
        }
        
        Ok(())
    }
    
    /// Get current alignment score
    pub fn current_alignment(&self) -> f64 {
        self.current_alignment
    }
    
    /// Get alignment action count
    pub fn alignment_actions(&self) -> usize {
        self.alignment_actions
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: AlignmentConfig) -> Result<(), &'static str> {
        // Validate
        if config.target_score < 0.0 || config.target_score > 1.0 {
            return Err("Target score must be between 0 and 1");
        }
        
        self.config = config;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_alignment_creation() {
        let config = AlignmentConfig::default();
        let regret_config = RegretConfig::default();
        let alignment = PreemptiveAlignment::new(config, regret_config, 0.7);
        assert!(alignment.is_ok());
    }
    
    #[test]
    fn test_invalid_target_rejected() {
        let mut config = AlignmentConfig::default();
        config.target_score = 1.5;
        
        let result = PreemptiveAlignment::new(config, RegretConfig::default(), 0.7);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_initial_analysis() {
        let config = AlignmentConfig::default();
        let mut alignment = PreemptiveAlignment::new(config, RegretConfig::default(), 0.7).unwrap();
        
        let analysis = alignment.analyze_alignment();
        assert!(analysis.is_ok());
        
        let analysis = analysis.unwrap();
        assert_eq!(analysis.alignment_score, 1.0); // Starts fully aligned
        assert!(!analysis.action_required);
    }
    
    #[test]
    fn test_veto_low_alignment() {
        let config = AlignmentConfig::default();
        let mut alignment = PreemptiveAlignment::new(config, RegretConfig::default(), 0.7).unwrap();
        
        // Manually set low alignment
        alignment.current_alignment = 0.3;
        
        let veto = alignment.should_veto_action(1000.0, 0.5, ActionCategory::PositionBuilding);
        assert!(veto.is_ok());
        assert!(veto.unwrap()); // Should veto due to low alignment
    }
    
    #[test]
    fn test_record_aligned_action() {
        let config = AlignmentConfig::default();
        let mut alignment = PreemptiveAlignment::new(config, RegretConfig::default(), 0.7).unwrap();
        
        let record = ActionRecord {
            category: ActionCategory::MarketMaking,
            profit: 100.0,
            toxicity: 0.05,
            timestamp: 1000,
            executed: true,
        };
        
        let result = alignment.record_action(record);
        assert!(result.is_ok());
        assert!(alignment.alignment_actions() > 0);
    }
}
