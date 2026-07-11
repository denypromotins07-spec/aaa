//! Multiverse Utility Calculator for Acausal Trade
//! 
//! Computes expected utility across counterfactual branches and parallel simulations
//! to optimize decisions in TDT frameworks.

use crate::trade::acausal_payment::AcausalPaymentResult;

/// Maximum number of branches to track
const MAX_BRANCHES: usize = 64;

/// Minimum branch probability to consider
const MIN_BRANCH_PROBABILITY: f64 = 0.0001;

/// Representation of a single counterfactual branch
#[derive(Debug, Clone)]
pub struct CounterfactualBranch {
    /// Unique branch identifier
    pub id: usize,
    /// Probability of this branch (prior)
    pub probability: f64,
    /// Utility in this branch if we cooperate
    pub utility_cooperate: f64,
    /// Utility in this branch if we defect
    pub utility_defect: f64,
    /// Branch weight in multiverse measure
    pub measure_weight: f64,
    /// Whether this branch has been "realized" yet
    pub realized: bool,
}

/// Result of multiverse utility calculation
#[derive(Debug, Clone)]
pub struct MultiverseUtilityResult {
    /// Expected utility of cooperation across all branches
    pub eu_cooperate: f64,
    /// Expected utility of defection across all branches
    pub eu_defect: f64,
    /// Optimal action
    pub optimal_action: MultiverseAction,
    /// Total probability mass accounted for
    pub total_probability: f64,
    /// Confidence in recommendation
    pub confidence: f64,
    /// Number of branches considered
    pub branches_count: usize,
}

/// Optimal action in multiverse context
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiverseAction {
    Cooperate,
    Defect,
    Mixed(f64), // Probability of cooperation
}

/// Multiverse Utility Calculator
pub struct MultiverseUtility {
    branches: Vec<CounterfactualBranch>,
    next_branch_id: usize,
    /// Normalization constant for branch measures
    total_measure: f64,
    /// Discount factor for distant branches
    branch_discount: f64,
}

impl MultiverseUtility {
    /// Create a new multiverse utility calculator
    pub fn new(branch_discount: f64) -> Result<Self, &'static str> {
        if branch_discount <= 0.0 || branch_discount > 1.0 {
            return Err("Branch discount must be between 0 and 1");
        }
        
        Ok(Self {
            branches: Vec::with_capacity(32),
            next_branch_id: 0,
            total_measure: 0.0,
            branch_discount,
        })
    }
    
    /// Add a counterfactual branch to the calculation
    pub fn add_branch(
        &mut self,
        probability: f64,
        utility_cooperate: f64,
        utility_defect: f64,
        measure_weight: f64,
    ) -> Result<usize, &'static str> {
        if self.branches.len() >= MAX_BRANCHES {
            return Err("Maximum branch count reached");
        }
        
        if probability < MIN_BRANCH_PROBABILITY {
            return Err("Branch probability below minimum threshold");
        }
        
        if probability > 1.0 {
            return Err("Probability cannot exceed 1.0");
        }
        
        if measure_weight <= 0.0 {
            return Err("Measure weight must be positive");
        }
        
        let id = self.next_branch_id;
        self.next_branch_id += 1;
        
        let branch = CounterfactualBranch {
            id,
            probability,
            utility_cooperate,
            utility_defect,
            measure_weight,
            realized: false,
        };
        
        self.total_measure += measure_weight;
        self.branches.push(branch);
        
        Ok(id)
    }
    
    /// Calculate expected utility across all branches
    pub fn calculate_multiverse_utility(&self) -> Result<MultiverseUtilityResult, &'static str> {
        if self.branches.is_empty() {
            return Ok(MultiverseUtilityResult {
                eu_cooperate: 0.0,
                eu_defect: 0.0,
                optimal_action: MultiverseAction::Cooperate,
                total_probability: 0.0,
                confidence: 0.0,
                branches_count: 0,
            });
        }
        
        let mut eu_cooperate = 0.0;
        let mut eu_defect = 0.0;
        let mut total_probability = 0.0;
        
        for branch in &self.branches {
            // Weight by probability and measure
            let weight = branch.probability * branch.measure_weight / self.total_measure;
            
            // Apply branch discount based on "distance" (simplified as 1/id)
            let discounted_weight = weight * self.branch_discount.powi(branch.id as i32);
            
            eu_cooperate += branch.utility_cooperate * discounted_weight;
            eu_defect += branch.utility_defect * discounted_weight;
            total_probability += branch.probability;
        }
        
        // Determine optimal action
        let optimal_action = if eu_cooperate > eu_defect {
            MultiverseAction::Cooperate
        } else if eu_defect > eu_cooperate {
            MultiverseAction::Defect
        } else {
            // Equal utility - mixed strategy
            MultiverseAction::Mixed(0.5)
        };
        
        // Calculate confidence based on utility difference
        let max_eu = eu_cooperate.abs().max(eu_defect.abs());
        let confidence = if max_eu > 0.0 {
            ((eu_cooperate - eu_defect).abs() / max_eu).min(1.0)
        } else {
            0.0
        };
        
        Ok(MultiverseUtilityResult {
            eu_cooperate,
            eu_defect,
            optimal_action,
            total_probability: total_probability.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            branches_count: self.branches.len(),
        })
    }
    
    /// Mark a branch as realized (actually occurred)
    pub fn mark_realized(&mut self, branch_id: usize) -> Result<(), &'static str> {
        if let Some(branch) = self.branches.iter_mut().find(|b| b.id == branch_id) {
            branch.realized = true;
            Ok(())
        } else {
            Err("Branch not found")
        }
    }
    
    /// Get utility difference between cooperation and defection
    pub fn utility_difference(&self) -> Result<f64, &'static str> {
        let result = self.calculate_multiverse_utility()?;
        Ok(result.eu_cooperate - result.eu_defect)
    }
    
    /// Check if cooperation dominates across all branches
    pub fn cooperation_dominates(&self) -> Result<bool, &'static str> {
        if self.branches.is_empty() {
            return Err("No branches to evaluate");
        }
        
        for branch in &self.branches {
            if branch.utility_cooperate <= branch.utility_defect {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Clear all branches
    pub fn clear(&mut self) {
        self.branches.clear();
        self.total_measure = 0.0;
        self.next_branch_id = 0;
    }
    
    /// Get number of registered branches
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }
    
    /// Update branch discount factor
    pub fn set_branch_discount(&mut self, discount: f64) -> Result<(), &'static str> {
        if discount <= 0.0 || discount > 1.0 {
            return Err("Branch discount must be between 0 and 1");
        }
        self.branch_discount = discount;
        Ok(())
    }
}

impl Default for MultiverseUtility {
    fn default() -> Self {
        Self::new(0.9).unwrap_or_else(|_| Self {
            branches: Vec::with_capacity(32),
            next_branch_id: 0,
            total_measure: 0.0,
            branch_discount: 0.9,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calculator_creation() {
        let calc = MultiverseUtility::new(0.9);
        assert!(calc.is_ok());
    }
    
    #[test]
    fn test_invalid_discount_rejected() {
        let calc = MultiverseUtility::new(1.5);
        assert!(calc.is_err());
    }
    
    #[test]
    fn test_add_branch() {
        let mut calc = MultiverseUtility::new(0.9).unwrap();
        
        let result = calc.add_branch(0.5, 100.0, 50.0, 1.0);
        assert!(result.is_ok());
        assert_eq!(calc.branch_count(), 1);
    }
    
    #[test]
    fn test_low_probability_rejected() {
        let mut calc = MultiverseUtility::new(0.9).unwrap();
        
        let result = calc.add_branch(0.00001, 100.0, 50.0, 1.0);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_multiverse_calculation() {
        let mut calc = MultiverseUtility::new(0.9).unwrap();
        
        // Add branches where cooperation is better
        let _ = calc.add_branch(0.6, 100.0, 50.0, 1.0);
        let _ = calc.add_branch(0.4, 80.0, 40.0, 1.0);
        
        let result = calc.calculate_multiverse_utility();
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert!(result.eu_cooperate > result.eu_defect);
        assert_eq!(result.optimal_action, MultiverseAction::Cooperate);
    }
    
    #[test]
    fn test_cooperation_dominance() {
        let mut calc = MultiverseUtility::new(0.9).unwrap();
        
        // All branches favor cooperation
        let _ = calc.add_branch(0.5, 100.0, 50.0, 1.0);
        let _ = calc.add_branch(0.5, 80.0, 40.0, 1.0);
        
        let dominates = calc.cooperation_dominates();
        assert!(dominates.is_ok());
        assert!(dominates.unwrap());
    }
    
    #[test]
    fn test_empty_calculation() {
        let calc = MultiverseUtility::new(0.9).unwrap();
        
        let result = calc.calculate_multiverse_utility();
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert_eq!(result.branches_count, 0);
        assert_eq!(result.total_probability, 0.0);
    }
}
