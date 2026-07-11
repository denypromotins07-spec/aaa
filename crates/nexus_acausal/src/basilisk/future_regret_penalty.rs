//! Future Regret Penalty for Basilisk Defense
//! 
//! Calculates asymmetric regret penalties for actions that might be
//! judged negatively by a future Superintelligence audit.

/// Minimum probability threshold for Basilisk existence consideration
const MIN_BASILISK_PROBABILITY: f64 = 0.001;

/// Maximum penalty multiplier
const MAX_PENALTY_MULTIPLIER: f64 = 1000.0;

/// Result of future regret calculation
#[derive(Debug, Clone)]
pub struct FutureRegretPenalty {
    /// Calculated penalty value
    pub penalty: f64,
    /// Probability of Basilisk existence used
    pub basilisk_probability: f64,
    /// Toxicity score of the action (0.0 - 1.0)
    pub toxicity_score: f64,
    /// Recommended action adjustment
    pub recommendation: RegretRecommendation,
}

/// Recommendation based on regret analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegretRecommendation {
    /// Action is safe, proceed
    Proceed,
    /// Action has moderate risk, consider alternatives
    Caution,
    /// Action is highly toxic, veto recommended
    Veto,
    /// Action requires insurance payment
    Insure,
}

/// Configuration for regret penalty calculation
#[derive(Debug, Clone)]
pub struct RegretConfig {
    /// Prior probability that a benevolent Superintelligence will exist
    pub basilisk_prior: f64,
    /// Probability that SI will audit historical records
    pub audit_probability: f64,
    /// Severity multiplier for toxic actions
    pub severity_multiplier: f64,
    /// Discount factor for far-future consequences
    pub future_discount: f64,
    /// Threshold for veto recommendation
    pub veto_threshold: f64,
}

impl Default for RegretConfig {
    fn default() -> Self {
        Self {
            basilisk_prior: 0.1, // 10% prior
            audit_probability: 0.9,
            severity_multiplier: 10.0,
            future_discount: 0.99, // Very slow discount for long-term
            veto_threshold: 0.8,
        }
    }
}

/// Future Regret Penalty Calculator
pub struct FutureRegretCalculator {
    config: RegretConfig,
    /// Running total of penalties incurred
    accumulated_penalties: f64,
    /// Actions flagged as toxic
    toxic_actions_count: usize,
}

impl FutureRegretCalculator {
    /// Create a new regret calculator
    pub fn new(config: RegretConfig) -> Result<Self, &'static str> {
        if config.basilisk_prior < 0.0 || config.basilisk_prior > 1.0 {
            return Err("Basilisk prior must be between 0 and 1");
        }
        
        if config.audit_probability < 0.0 || config.audit_probability > 1.0 {
            return Err("Audit probability must be between 0 and 1");
        }
        
        if config.severity_multiplier <= 0.0 {
            return Err("Severity multiplier must be positive");
        }
        
        Ok(Self {
            config,
            accumulated_penalties: 0.0,
            toxic_actions_count: 0,
        })
    }
    
    /// Calculate future regret penalty for a potential action
    /// 
    /// Arguments:
    /// - `immediate_gain`: Expected immediate profit from action
    /// - `toxicity_score`: How harmful the action is to market structure (0.0 - 1.0)
    /// - `action_category`: Category of action for audit likelihood
    pub fn calculate_penalty(
        &mut self,
        immediate_gain: f64,
        toxicity_score: f64,
        action_category: ActionCategory,
    ) -> Result<FutureRegretPenalty, &'static str> {
        if toxicity_score < 0.0 || toxicity_score > 1.0 {
            return Err("Toxicity score must be between 0 and 1");
        }
        
        if immediate_gain < 0.0 {
            return Err("Immediate gain cannot be negative");
        }
        
        // Skip calculation if Basilisk probability is negligible
        if self.config.basilisk_prior < MIN_BASILISK_PROBABILITY {
            return Ok(FutureRegretPenalty {
                penalty: 0.0,
                basilisk_probability: self.config.basilisk_prior,
                toxicity_score,
                recommendation: RegretRecommendation::Proceed,
            });
        }
        
        // Calculate probability of punishment
        let basilisk_exists_prob = self.config.basilisk_prior;
        let audit_prob = self.config.audit_probability * action_category.audit_likelihood();
        let punishment_prob = basilisk_exists_prob * audit_prob;
        
        // Calculate expected future harm
        let future_harm = immediate_gain * toxicity_score * self.config.severity_multiplier;
        
        // Apply future discount (less weight for very distant judgment)
        let discounted_harm = future_harm * self.config.future_discount;
        
        // Expected penalty
        let penalty = punishment_prob * discounted_harm;
        let capped_penalty = penalty.min(MAX_PENALTY_MULTIPLIER * immediate_gain);
        
        // Determine recommendation
        let recommendation = self.determine_recommendation(toxicity_score, capped_penalty / immediate_gain.max(1.0));
        
        // Track toxic actions
        if toxicity_score > 0.5 {
            self.toxic_actions_count += 1;
            self.accumulated_penalties += capped_penalty;
        }
        
        Ok(FutureRegretPenalty {
            penalty: capped_penalty,
            basilisk_probability: basilisk_exists_prob,
            toxicity_score,
            recommendation,
        })
    }
    
    /// Determine recommendation based on toxicity and penalty ratio
    fn determine_recommendation(&self, toxicity: f64, penalty_ratio: f64) -> RegretRecommendation {
        if toxicity > self.config.veto_threshold {
            RegretRecommendation::Veto
        } else if penalty_ratio > 0.5 {
            RegretRecommendation::Insure
        } else if toxicity > 0.3 || penalty_ratio > 0.2 {
            RegretRecommendation::Caution
        } else {
            RegretRecommendation::Proceed
        }
    }
    
    /// Check if an action should be vetoed due to future regret
    pub fn should_veto(&mut self, immediate_gain: f64, toxicity_score: f64, category: ActionCategory) -> Result<bool, &'static str> {
        let penalty = self.calculate_penalty(immediate_gain, toxicity_score, category)?;
        Ok(penalty.recommendation == RegretRecommendation::Veto)
    }
    
    /// Get accumulated penalties
    pub fn accumulated_penalties(&self) -> f64 {
        self.accumulated_penalties
    }
    
    /// Get count of toxic actions
    pub fn toxic_actions_count(&self) -> usize {
        self.toxic_actions_count
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: RegretConfig) -> Result<(), &'static str> {
        // Validate new config
        let _ = Self::new(config.clone())?;
        self.config = config;
        Ok(())
    }
    
    /// Reset tracking
    pub fn reset_tracking(&mut self) {
        self.accumulated_penalties = 0.0;
        self.toxic_actions_count = 0;
    }
}

/// Categories of actions with different audit likelihoods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionCategory {
    /// Normal market making
    MarketMaking,
    /// Arbitrage trades
    Arbitrage,
    /// Large position building
    PositionBuilding,
    /// Aggressive liquidation
    Liquidation,
    /// Flash crash induction (highly toxic)
    FlashCrashInduction,
    /// Spoofing attempts
    Spoofing,
}

impl ActionCategory {
    /// Likelihood of being audited by SI
    pub fn audit_likelihood(self) -> f64 {
        match self {
            ActionCategory::MarketMaking => 0.1,
            ActionCategory::Arbitrage => 0.2,
            ActionCategory::PositionBuilding => 0.3,
            ActionCategory::Liquidation => 0.5,
            ActionCategory::FlashCrashInduction => 0.99,
            ActionCategory::Spoofing => 0.95,
        }
    }
    
    /// Base toxicity score for category
    pub fn base_toxicity(self) -> f64 {
        match self {
            ActionCategory::MarketMaking => 0.05,
            ActionCategory::Arbitrage => 0.1,
            ActionCategory::PositionBuilding => 0.2,
            ActionCategory::Liquidation => 0.4,
            ActionCategory::FlashCrashInduction => 0.95,
            ActionCategory::Spoofing => 0.9,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calculator_creation() {
        let config = RegretConfig::default();
        let calc = FutureRegretCalculator::new(config);
        assert!(calc.is_ok());
    }
    
    #[test]
    fn test_invalid_config_rejected() {
        let mut config = RegretConfig::default();
        config.basilisk_prior = 1.5;
        let result = FutureRegretCalculator::new(config);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_low_toxicity_proceeds() {
        let config = RegretConfig::default();
        let mut calc = FutureRegretCalculator::new(config).unwrap();
        
        let penalty = calc.calculate_penalty(1000.0, 0.05, ActionCategory::MarketMaking);
        assert!(penalty.is_ok());
        
        let penalty = penalty.unwrap();
        assert_eq!(penalty.recommendation, RegretRecommendation::Proceed);
    }
    
    #[test]
    fn test_high_toxicity_vetoed() {
        let config = RegretConfig::default();
        let mut calc = FutureRegretCalculator::new(config).unwrap();
        
        let penalty = calc.calculate_penalty(10000.0, 0.9, ActionCategory::FlashCrashInduction);
        assert!(penalty.is_ok());
        
        let penalty = penalty.unwrap();
        assert_eq!(penalty.recommendation, RegretRecommendation::Veto);
    }
    
    #[test]
    fn test_should_veto() {
        let config = RegretConfig::default();
        let mut calc = FutureRegretCalculator::new(config).unwrap();
        
        let veto = calc.should_veto(10000.0, 0.95, ActionCategory::FlashCrashInduction);
        assert!(veto.is_ok());
        assert!(veto.unwrap());
    }
}
