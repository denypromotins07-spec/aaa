//! Convex Hedge Optimizer for crash protection allocation
//!
//! Dynamically calculates optimal allocation to "crash protection" instruments
//! (deep OTM puts, VIX futures, short high-beta assets) based on tail risk metrics.
//! Uses modified Kelly Criterion for power-law distributions.

use ndarray::{Array1, ArrayView1};
use thiserror::Error;

/// Errors from hedge optimization
#[derive(Error, Debug, Clone)]
pub enum HedgeError {
    #[error("Invalid hedge instrument configuration")]
    InvalidInstrument,
    
    #[error("Convexity parameter must be positive: got {0}")]
    InvalidConvexity(f64),
    
    #[error("Budget constraint violation: requested {requested}, available {available}")]
    BudgetViolation { requested: f64, available: f64 },
    
    #[error("Optimization failed to converge")]
    OptimizationFailure,
    
    #[error("Tail risk estimate invalid: {0}")]
    InvalidTailRisk(String),
}

/// Configuration for a convex hedge instrument
#[derive(Debug, Clone)]
pub struct HedgeInstrument {
    /// Unique identifier
    pub id: String,
    /// Instrument type
    pub instrument_type: HedgeType,
    /// Expected convexity (gamma) - how much value increases per unit market decline
    pub convexity: f64,
    /// Cost of carry (theta decay for options, roll cost for futures)
    pub carry_cost: f64,
    /// Maximum position size as fraction of portfolio
    pub max_position: f64,
    /// Liquidity score (0-1, higher = more liquid)
    pub liquidity_score: f64,
}

/// Types of hedge instruments
#[derive(Debug, Clone, Copy)]
pub enum HedgeType {
    /// Deep OTM put options
    PutOption,
    /// VIX futures
    VixFuture,
    /// Inverse ETF
    InverseEtf,
    /// Short position in high-beta asset
    ShortPosition,
    /// Tail risk swap
    TailSwap,
    /// Safe haven asset (gold, treasuries)
    SafeHaven,
}

impl HedgeInstrument {
    /// Validate instrument parameters
    pub fn validate(&self) -> Result<(), HedgeError> {
        if self.convexity <= 0.0 {
            return Err(HedgeError::InvalidConvexity(self.convexity));
        }
        
        if self.max_position <= 0.0 || self.max_position > 1.0 {
            return Err(HedgeError::InvalidInstrument);
        }
        
        if self.liquidity_score < 0.0 || self.liquidity_score > 1.0 {
            return Err(HedgeError::InvalidInstrument);
        }
        
        Ok(())
    }
    
    /// Calculate expected PnL for a given market move
    pub fn expected_pnl(&self, market_return: f64) -> f64 {
        match self.instrument_type {
            HedgeType::PutOption => {
                // Put payoff: max(0, K - S) adjusted for convexity
                let intrinsic = (-market_return * self.convexity).max(0.0);
                intrinsic - self.carry_cost
            }
            HedgeType::VixFuture => {
                // VIX typically rises when market falls
                (-market_return * self.convexity) - self.carry_cost
            }
            HedgeType::InverseEtf | HedgeType::ShortPosition => {
                // Linear inverse exposure
                -market_return * self.convexity - self.carry_cost
            }
            HedgeType::TailSwap => {
                // Binary payoff for extreme moves
                if market_return < -0.10 {
                    self.convexity - self.carry_cost
                } else {
                    -self.carry_cost
                }
            }
            HedgeType::SafeHaven => {
                // Modest positive correlation to stress
                (-market_return * self.convexity * 0.3) - self.carry_cost
            }
        }
    }
}

/// Result of hedge optimization
#[derive(Debug, Clone)]
pub struct HedgeAllocation {
    /// Allocation weight for each instrument
    pub weights: Array1<f64>,
    /// Total portfolio allocation to hedges
    total_hedge_weight: f64,
    /// Expected portfolio convexity
    portfolio_convexity: f64,
    /// Total carry cost
    total_carry_cost: f64,
    /// Estimated protection level (VaR reduction)
    var_reduction: f64,
}

impl HedgeAllocation {
    /// Get individual weight
    pub fn weight(&self, idx: usize) -> f64 {
        if idx < self.weights.len() {
            self.weights[idx]
        } else {
            0.0
        }
    }
    
    /// Check if allocation is fully invested
    pub fn is_fully_allocated(&self, tolerance: f64) -> bool {
        (self.total_hedge_weight - 1.0).abs() < tolerance
    }
}

/// Convex Hedge Optimizer using modified Kelly Criterion
pub struct ConvexHedgeOptimizer {
    /// Available hedge instruments
    instruments: Vec<HedgeInstrument>,
    /// Current tail risk estimate (probability of crash)
    crash_probability: f64,
    /// Estimated crash magnitude
    crash_magnitude: f64,
    /// Portfolio value
    portfolio_value: f64,
    /// Risk aversion parameter
    risk_aversion: f64,
}

impl ConvexHedgeOptimizer {
    /// Create a new optimizer with the given instruments
    pub fn new(instruments: Vec<HedgeInstrument>) -> Result<Self, HedgeError> {
        // Validate all instruments
        for inst in &instruments {
            inst.validate()?;
        }
        
        Ok(Self {
            instruments,
            crash_probability: 0.05,
            crash_magnitude: -0.20,
            portfolio_value: 1_000_000.0,
            risk_aversion: 0.5,
        })
    }
    
    /// Update tail risk estimates
    pub fn update_risk_estimates(
        &mut self,
        crash_prob: f64,
        crash_mag: f64,
    ) -> Result<(), HedgeError> {
        if crash_prob < 0.0 || crash_prob > 1.0 {
            return Err(HedgeError::InvalidTailRisk(
                format!("Invalid probability: {}", crash_prob)
            ));
        }
        
        if crash_mag >= 0.0 {
            return Err(HedgeError::InvalidTailRisk(
                "Crash magnitude must be negative".to_string()
            ));
        }
        
        self.crash_probability = crash_prob;
        self.crash_magnitude = crash_mag;
        
        Ok(())
    }
    
    /// Optimize hedge allocation using modified Kelly Criterion
    /// 
    /// For power-law distributions, standard Kelly (f* = p/a - q/b) is modified to:
    /// f*_modified = f* / (1 + variance_adjustment)
    /// where variance_adjustment accounts for infinite variance in fat tails
    pub fn optimize(&self) -> Result<HedgeAllocation, HedgeError> {
        let n = self.instruments.len();
        let mut weights = Array1::zeros(n);
        
        // Calculate optimal Kelly fraction for each instrument
        let mut total_kelly = 0.0;
        
        for (i, inst) in self.instruments.iter().enumerate() {
            // Expected payoff in crash scenario
            let crash_payoff = inst.expected_pnl(self.crash_magnitude);
            
            // Expected payoff in normal scenario (small positive drift)
            let normal_payoff = inst.expected_pnl(0.01);
            
            // Probability-weighted expected return
            let expected_return = self.crash_probability * crash_payoff
                + (1.0 - self.crash_probability) * normal_payoff;
            
            // Kelly fraction: f* = E[R] / (variance + E[R]^2)
            // Simplified for convex instruments
            let variance_estimate = crash_payoff.powi(2) * self.crash_probability
                + normal_payoff.powi(2) * (1.0 - self.crash_probability);
            
            let kelly_fraction = if variance_estimate > 1e-10 {
                expected_return / variance_estimate
            } else {
                0.0
            };
            
            // Apply power-law adjustment (reduce for fat tails)
            let adjusted_fraction = kelly_fraction / (1.0 + self.risk_aversion);
            
            // Clamp to instrument limits
            weights[i] = adjusted_fraction.clamp(0.0, inst.max_position);
            
            total_kelly += weights[i];
        }
        
        // Normalize if total exceeds budget
        let max_total = 0.30; // Maximum 30% in hedges
        if total_kelly > max_total {
            let scale = max_total / total_kelly;
            weights.mapv_inplace(|w| w * scale);
        }
        
        // Calculate portfolio-level metrics
        let total_hedge_weight = weights.sum();
        let portfolio_convexity: f64 = weights.iter()
            .zip(self.instruments.iter())
            .map(|(&w, inst)| w * inst.convexity)
            .sum();
        
        let total_carry_cost: f64 = weights.iter()
            .zip(self.instruments.iter())
            .map(|(&w, inst)| w * inst.carry_cost)
            .sum();
        
        // Estimate VaR reduction from hedging
        let var_reduction = self.estimate_var_reduction(&weights);
        
        Ok(HedgeAllocation {
            weights,
            total_hedge_weight,
            portfolio_convexity,
            total_carry_cost,
            var_reduction,
        })
    }
    
    /// Calculate marginal benefit of adding more hedge
    pub fn marginal_benefit(&self, instrument_idx: usize) -> Result<f64, HedgeError> {
        if instrument_idx >= self.instruments.len() {
            return Err(HedgeError::InvalidInstrument);
        }
        
        let inst = &self.instruments[instrument_idx];
        
        // Marginal benefit = d(Expected Utility) / d(weight)
        let crash_payoff = inst.expected_pnl(self.crash_magnitude);
        let normal_payoff = inst.expected_pnl(0.01);
        
        let marginal = self.crash_probability * crash_payoff
            + (1.0 - self.crash_probability) * normal_payoff;
        
        // Adjust for convexity benefit
        let convexity_bonus = inst.convexity * self.crash_probability * 0.1;
        
        Ok(marginal + convexity_bonus)
    }
    
    /// Estimate VaR reduction from the hedge portfolio
    fn estimate_var_reduction(&self, weights: &Array1<f64>) -> f64 {
        // Simplified estimation based on portfolio convexity
        let portfolio_convexity: f64 = weights.iter()
            .zip(self.instruments.iter())
            .map(|(&w, inst)| w * inst.convexity)
            .sum();
        
        // VaR reduction approximately proportional to convexity * crash prob
        let base_reduction = portfolio_convexity * self.crash_probability;
        
        // Cap at reasonable level
        base_reduction.min(0.50)
    }
    
    /// Get list of instruments
    pub fn instruments(&self) -> &[HedgeInstrument] {
        &self.instruments
    }
    
    /// Set risk aversion parameter
    pub fn set_risk_aversion(&mut self, ra: f64) {
        self.risk_aversion = ra.max(0.0).min(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hedge_instrument_validation() {
        let valid_inst = HedgeInstrument {
            id: "SPY_PUT".to_string(),
            instrument_type: HedgeType::PutOption,
            convexity: 5.0,
            carry_cost: 0.02,
            max_position: 0.10,
            liquidity_score: 0.9,
        };
        
        assert!(valid_inst.validate().is_ok());
        
        let invalid_inst = HedgeInstrument {
            id: "BAD".to_string(),
            instrument_type: HedgeType::PutOption,
            convexity: -1.0,
            carry_cost: 0.02,
            max_position: 0.10,
            liquidity_score: 0.9,
        };
        
        assert!(invalid_inst.validate().is_err());
    }
    
    #[test]
    fn test_put_option_payoff() {
        let put = HedgeInstrument {
            id: "PUT".to_string(),
            instrument_type: HedgeType::PutOption,
            convexity: 10.0,
            carry_cost: 0.05,
            max_position: 0.10,
            liquidity_score: 0.8,
        };
        
        // Market down 20%
        let pnl_crash = put.expected_pnl(-0.20);
        assert!(pnl_crash > 0.0);
        
        // Market up 5%
        let pnl_up = put.expected_pnl(0.05);
        assert!(pnl_up < 0.0); // Just the carry cost
    }
    
    #[test]
    fn test_optimizer_basic() {
        let instruments = vec![
            HedgeInstrument {
                id: "PUT".to_string(),
                instrument_type: HedgeType::PutOption,
                convexity: 8.0,
                carry_cost: 0.03,
                max_position: 0.15,
                liquidity_score: 0.9,
            },
            HedgeInstrument {
                id: "VIX".to_string(),
                instrument_type: HedgeType::VixFuture,
                convexity: 4.0,
                carry_cost: 0.05,
                max_position: 0.10,
                liquidity_score: 0.8,
            },
        ];
        
        let optimizer = ConvexHedgeOptimizer::new(instruments).unwrap();
        let allocation = optimizer.optimize().unwrap();
        
        assert!(allocation.total_hedge_weight > 0.0);
        assert!(allocation.total_hedge_weight <= 0.30);
        assert!(allocation.portfolio_convexity > 0.0);
    }
}
