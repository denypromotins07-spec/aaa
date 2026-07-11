//! Acausal Payment System for TDT-based Trade
//! 
//! Implements payment mechanisms that work across counterfactual branches
//! to influence simulations in other timelines/markets.

/// Minimum payment amount to be meaningful
const MIN_PAYMENT_AMOUNT: f64 = 0.0001;

/// Maximum payment relative to portfolio value
const MAX_PAYMENT_RATIO: f64 = 0.01;

/// Result of an acausal payment attempt
#[derive(Debug, Clone)]
pub struct AcausalPaymentResult {
    /// Whether payment was executed
    pub executed: bool,
    /// Payment amount
    pub amount: f64,
    /// Expected utility gain across multiverse
    pub expected_utility_gain: f64,
    /// Counterparty simulation confidence
    pub simulation_confidence: f64,
    /// Branch correlation strength
    pub branch_correlation: f64,
}

/// Configuration for acausal payments
#[derive(Debug, Clone)]
pub struct AcausalPaymentConfig {
    /// Minimum expected utility gain to justify payment
    pub min_utility_gain: f64,
    /// Maximum acceptable payment ratio
    pub max_payment_ratio: f64,
    /// Required simulation confidence threshold
    pub confidence_threshold: f64,
    /// Branch discount factor (how much we value other branches)
    pub branch_discount: f64,
}

impl Default for AcausalPaymentConfig {
    fn default() -> Self {
        Self {
            min_utility_gain: 0.001,
            max_payment_ratio: 0.005,
            confidence_threshold: 0.7,
            branch_discount: 0.9,
        }
    }
}

/// Acausal Payment Engine
pub struct AcausalPayment {
    config: AcausalPaymentConfig,
    /// Current portfolio value for ratio calculations
    portfolio_value: f64,
    /// Track payments made for accounting
    total_payments: f64,
}

impl AcausalPayment {
    /// Create a new acausal payment engine
    pub fn new(config: AcausalPaymentConfig, portfolio_value: f64) -> Result<Self, &'static str> {
        if portfolio_value <= 0.0 {
            return Err("Portfolio value must be positive");
        }
        
        Ok(Self {
            config,
            portfolio_value,
            total_payments: 0.0,
        })
    }
    
    /// Attempt an acausal payment to influence counterparty simulation
    /// 
    /// The payment is made now to influence how the counterparty simulates
    /// us in counterfactual branches where they moved first.
    pub fn attempt_payment(
        &mut self,
        payment_amount: f64,
        expected_utility_gain: f64,
        simulation_confidence: f64,
        branch_correlation: f64,
    ) -> Result<AcausalPaymentResult, &'static str> {
        // Validate payment amount
        if payment_amount < MIN_PAYMENT_AMOUNT {
            return Err("Payment amount below minimum");
        }
        
        let max_payment = self.portfolio_value * self.config.max_payment_ratio.min(MAX_PAYMENT_RATIO);
        if payment_amount > max_payment {
            return Err("Payment exceeds maximum allowed ratio");
        }
        
        // Check expected utility gain
        let discounted_utility = expected_utility_gain 
            * branch_correlation 
            * self.config.branch_discount;
        
        if discounted_utility < self.config.min_utility_gain {
            return Err("Expected utility gain insufficient");
        }
        
        // Check simulation confidence
        if simulation_confidence < self.config.confidence_threshold {
            return Err("Simulation confidence below threshold");
        }
        
        // Validate branch correlation
        if branch_correlation < 0.0 || branch_correlation > 1.0 {
            return Err("Branch correlation must be between 0 and 1");
        }
        
        // Execute payment
        self.total_payments += payment_amount;
        
        Ok(AcausalPaymentResult {
            executed: true,
            amount: payment_amount,
            expected_utility_gain: discounted_utility,
            simulation_confidence,
            branch_correlation,
        })
    }
    
    /// Calculate optimal payment amount for given scenario
    pub fn calculate_optimal_payment(
        &self,
        potential_utility: f64,
        branch_correlation: f64,
    ) -> f64 {
        // Optimal payment balances cost against expected gain
        let expected_gain = potential_utility * branch_correlation * self.config.branch_discount;
        
        // Payment should be proportional to expected gain but capped
        let optimal = expected_gain * 0.1; // 10% of expected gain
        
        optimal.clamp(MIN_PAYMENT_AMOUNT, self.portfolio_value * self.config.max_payment_ratio)
    }
    
    /// Update portfolio value
    pub fn update_portfolio_value(&mut self, new_value: f64) -> Result<(), &'static str> {
        if new_value <= 0.0 {
            return Err("Portfolio value must be positive");
        }
        self.portfolio_value = new_value;
        Ok(())
    }
    
    /// Get total payments made
    pub fn total_payments(&self) -> f64 {
        self.total_payments
    }
    
    /// Get remaining payment capacity
    pub fn remaining_capacity(&self) -> f64 {
        self.portfolio_value * self.config.max_payment_ratio - self.total_payments
    }
    
    /// Reset payment tracking
    pub fn reset_tracking(&mut self) {
        self.total_payments = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_payment_creation() {
        let config = AcausalPaymentConfig::default();
        let payment = AcausalPayment::new(config, 1000000.0);
        assert!(payment.is_ok());
    }
    
    #[test]
    fn test_zero_portfolio_rejected() {
        let config = AcaausalPaymentConfig::default();
        let result = AcausalPayment::new(config, 0.0);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_minimum_payment_enforced() {
        let config = AcausalPaymentConfig::default();
        let mut payment = AcausalPayment::new(config, 1000000.0).unwrap();
        
        let result = payment.attempt_payment(
            0.00001, // Below minimum
            1.0,
            0.9,
            0.8,
        );
        assert!(result.is_err());
    }
    
    #[test]
    fn test_successful_payment() {
        let config = AcausalPaymentConfig::default();
        let mut payment = AcausalPayment::new(config, 1000000.0).unwrap();
        
        let result = payment.attempt_payment(
            100.0,
            2000.0,
            0.9,
            0.8,
        );
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.executed);
        assert!(result.expected_utility_gain > 0.0);
    }
    
    #[test]
    fn test_optimal_payment_calculation() {
        let config = AcausalPaymentConfig::default();
        let payment = AcausalPayment::new(config, 1000000.0).unwrap();
        
        let optimal = payment.calculate_optimal_payment(10000.0, 0.5);
        
        assert!(optimal >= MIN_PAYMENT_AMOUNT);
        assert!(optimal <= 1000000.0 * MAX_PAYMENT_RATIO);
    }
}
