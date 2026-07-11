//! Counterfactual Mugging Resolver for TDT
//! 
//! Implements decision logic for Newcomb-like problems and counterfactual mugging
//! scenarios where agents must decide whether to "pay" based on acausal reasoning.

use crate::trade::acausal_payment::{AcausalPayment, AcausalPaymentConfig, AcausalPaymentResult};

/// Default reward multiplier for Omega-style predictors
const DEFAULT_REWARD_MULTIPLIER: f64 = 100.0;

/// Minimum probability threshold for taking counterfactual action
const MIN_PROBABILITY_THRESHOLD: f64 = 0.01;

/// Result of counterfactual mugging resolution
#[derive(Debug, Clone)]
pub struct CounterfactualResolution {
    /// Whether to pay/one-box
    pub should_pay: bool,
    /// Expected utility of paying
    pub eu_pay: f64,
    /// Expected utility of not paying
    pub eu_not_pay: f64,
    /// Decision confidence (0.0 - 1.0)
    pub confidence: f64,
    /// Reasoning summary
    pub reasoning: &'static str,
}

/// Configuration for counterfactual mugging scenarios
#[derive(Debug, Clone)]
pub struct CounterfactualConfig {
    /// Reward multiplier (how much Omega rewards cooperators)
    pub reward_multiplier: f64,
    /// Prior probability that predictor is accurate
    pub predictor_accuracy_prior: f64,
    /// Utility weight for other branches
    pub branch_utility_weight: f64,
    /// Whether to use TDT or CDT decision theory
    pub use_tdt: bool,
}

impl Default for CounterfactualConfig {
    fn default() -> Self {
        Self {
            reward_multiplier: DEFAULT_REWARD_MULTIPLIER,
            predictor_accuracy_prior: 0.99,
            branch_utility_weight: 0.5,
            use_tdt: true,
        }
    }
}

/// Counterfactual Mugging Resolver
pub struct CounterfactualMugging {
    config: CounterfactualConfig,
    payment_engine: Option<AcausalPayment>,
}

impl CounterfactualMugging {
    /// Create a new counterfactual mugging resolver
    pub fn new(config: CounterfactualConfig, portfolio_value: Option<f64>) -> Self {
        let payment_engine = portfolio_value
            .map(|pv| AcausalPayment::new(AcausalPaymentConfig::default(), pv))
            .and_then(|r| r.ok());
        
        Self {
            config,
            payment_engine,
        }
    }
    
    /// Resolve standard counterfactual mugging scenario
    /// 
    /// Omega has already made its prediction. Should you pay?
    /// 
    /// Arguments:
    /// - `prediction_made`: Whether Omega predicted you would pay
    /// - `payment_cost`: Cost of paying now
    /// - `reward_if_predicted`: Reward you receive if Omega predicted correctly
    pub fn resolve_counterfactual_mugging(
        &self,
        prediction_made: bool,
        payment_cost: f64,
        reward_if_predicted: f64,
    ) -> CounterfactualResolution {
        if self.config.use_tdt {
            self.resolve_with_tdt(prediction_made, payment_cost, reward_if_predicted)
        } else {
            self.resolve_with_cdt(prediction_made, payment_cost, reward_if_predicted)
        }
    }
    
    /// Resolve using Timeless Decision Theory
    fn resolve_with_tdt(
        &self,
        prediction_made: bool,
        payment_cost: f64,
        reward_if_predicted: f64,
    ) -> CounterfactualResolution {
        // TDT reasoning: Your decision now correlates with Omega's prediction
        // because both are computed by similar algorithms
        
        // If Omega predicted you'd pay, and you do pay:
        //   Utility = reward - cost (you get reward since prediction was correct)
        let utility_pay_and_predicted = reward_if_predicted - payment_cost;
        
        // If Omega predicted you'd pay, but you don't:
        //   Utility = 0 (no reward, no cost, but prediction was wrong)
        let utility_not_pay_and_predicted = 0.0;
        
        // If Omega predicted you wouldn't pay, and you pay:
        //   Utility = -cost (you paid but weren't rewarded)
        let utility_pay_and_not_predicted = -payment_cost;
        
        // If Omega predicted you wouldn't pay, and you don't:
        //   Utility = 0
        let utility_not_pay_and_not_predicted = 0.0;
        
        // Under TDT, your decision now determines what Omega predicted
        // (because the prediction was based on simulating your decision algorithm)
        
        // Expected utility of deciding to pay:
        // P(Omega predicted pay) * (reward - cost) + P(Omega predicted not pay) * (-cost)
        let p_predicted_pay = self.config.predictor_accuracy_prior;
        let eu_pay = p_predicted_pay * utility_pay_and_predicted 
                   + (1.0 - p_predicted_pay) * utility_pay_and_not_predicted;
        
        // Expected utility of deciding not to pay:
        // P(Omega predicted pay) * 0 + P(Omega predicted not pay) * 0 = 0
        let eu_not_pay = 0.0;
        
        let should_pay = eu_pay > eu_not_pay;
        
        let reasoning = if should_pay {
            "TDT: Paying is optimal because your decision algorithm determines Omega's prediction"
        } else {
            "TDT: Not paying is optimal due to low predictor accuracy or high cost"
        };
        
        let confidence = ((eu_pay - eu_not_pay).abs() / (eu_pay.abs().max(eu_not_pay.abs()) + payment_cost)).min(1.0);
        
        CounterfactualResolution {
            should_pay,
            eu_pay,
            eu_not_pay,
            confidence: confidence.clamp(0.0, 1.0),
            reasoning,
        }
    }
    
    /// Resolve using Causal Decision Theory (for comparison)
    fn resolve_with_cdt(
        &self,
        prediction_made: bool,
        payment_cost: f64,
        _reward_if_predicted: f64,
    ) -> CounterfactualResolution {
        // CDT reasoning: The prediction has already been made, so your action
        // cannot causally affect it. Therefore, don't pay (dominant strategy).
        
        // Under CDT, the prediction is fixed:
        // If prediction was "pay": paying costs you, not paying doesn't help
        // If prediction was "not pay": paying costs you, not paying is neutral
        
        // Either way, not paying dominates (costs nothing)
        let eu_pay = -payment_cost;
        let eu_not_pay = 0.0;
        
        let should_pay = false;
        
        CounterfactualResolution {
            should_pay,
            eu_pay,
            eu_not_pay,
            confidence: 1.0,
            reasoning: "CDT: Never pay - prediction is already made and cannot be changed",
        }
    }
    
    /// Resolve Newcomb's Problem variant
    /// 
    /// Two boxes: A (transparent, $1000) and B (opaque, $0 or $1M)
    /// Omega predicted whether you'll one-box (take only B) or two-box (take both)
    /// If predicted one-box: B contains $1M
    /// If predicted two-box: B is empty
    pub fn resolve_newcombs_problem(
        &self,
        box_a_value: f64,
        box_b_potential: f64,
    ) -> CounterfactualResolution {
        // TDT analysis:
        // One-box utility: P(predicted one-box) * box_b_potential
        // Two-box utility: P(predicted one-box) * (box_b_potential + box_a_value) 
        //                + P(predicted two-box) * box_a_value
        //                But P(predicted two-box) means box_b is empty!
        
        let p_one_box_predicted = self.config.predictor_accuracy_prior;
        
        let eu_one_box = p_one_box_predicted * box_b_potential;
        let eu_two_box = (1.0 - p_one_box_predicted) * box_a_value + box_a_value;
        
        let should_one_box = eu_one_box > eu_two_box;
        
        let (eu_pay, eu_not_pay, reasoning) = if should_one_box {
            (eu_one_box, eu_two_box, "TDT: One-box to maximize correlation with Omega's prediction")
        } else {
            (eu_two_box, eu_one_box, "TDT: Two-box due to low predictor reliability")
        };
        
        CounterfactualResolution {
            should_pay: should_one_box, // "Paying" = one-boxing here
            eu_pay,
            eu_not_pay,
            confidence: ((eu_one_box - eu_two_box).abs() / (box_b_potential + box_a_value)).clamp(0.0, 1.0),
            reasoning,
        }
    }
    
    /// Get the payment engine for executing actual payments
    pub fn payment_engine(&self) -> Option<&AcausalPayment> {
        self.payment_engine.as_ref()
    }
    
    /// Get mutable payment engine
    pub fn payment_engine_mut(&mut self) -> Option<&mut AcausalPayment> {
        self.payment_engine.as_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_resolver_creation() {
        let config = CounterfactualConfig::default();
        let resolver = CounterfactualMugging::new(config, Some(1000000.0));
        assert!(resolver.payment_engine().is_some());
    }
    
    #[test]
    fn test_tdt_favors_paying() {
        let mut config = CounterfactualConfig::default();
        config.use_tdt = true;
        config.predictor_accuracy_prior = 0.99;
        
        let resolver = CounterfactualMugging::new(config, None);
        
        let result = resolver.resolve_counterfactual_mugging(
            true,  // Omega predicted pay
            100.0, // Cost to pay
            10000.0, // Reward if predicted correctly
        );
        
        // With high predictor accuracy and good reward/constraint ratio, TDT says pay
        assert!(result.should_pay);
        assert!(result.eu_pay > result.eu_not_pay);
    }
    
    #[test]
    fn test_cdt_never_pays() {
        let mut config = CounterfactualConfig::default();
        config.use_tdt = false;
        
        let resolver = CounterfactualMugging::new(config, None);
        
        let result = resolver.resolve_counterfactual_mugging(
            true,
            100.0,
            10000.0,
        );
        
        // CDT always says don't pay - prediction is already made
        assert!(!result.should_pay);
        assert_eq!(result.eu_not_pay, 0.0);
    }
    
    #[test]
    fn test_newcombs_problem() {
        let mut config = CounterfactualConfig::default();
        config.predictor_accuracy_prior = 0.99;
        
        let resolver = CounterfactualMugging::new(config, None);
        
        let result = resolver.resolve_newcombs_problem(1000.0, 1000000.0);
        
        // With reliable predictor, one-boxing is optimal
        assert!(result.should_pay); // should_pay = should_one_box here
        assert!(result.eu_pay > 900000.0); // ~0.99 * 1M
    }
}
