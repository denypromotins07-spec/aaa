//! Market Impact Penalty Calculator for RL Reward Shaping
//! 
//! Dynamically scales transaction cost penalties based on estimated market impact
//! and current spread conditions to encourage stealthy execution.

use std::sync::atomic::{AtomicU64, Ordering};

/// Epsilon for numerical stability
const EPSILON: f64 = 1e-10;

/// Market impact model parameters
#[derive(Debug, Clone)]
pub struct MarketImpactModel {
    /// Linear impact coefficient (alpha)
    pub alpha: f64,
    /// Square-root impact coefficient (beta) - typical for real markets
    pub beta: f64,
    /// Temporary vs permanent impact ratio
    pub temporary_ratio: f64,
    /// Maximum impact cap (as fraction of price)
    pub max_impact: f64,
}

impl Default for MarketImpactModel {
    fn default() -> Self {
        Self {
            alpha: 0.001,    // Linear coefficient
            beta: 0.01,      // Square-root coefficient (dominant)
            temporary_ratio: 0.7, // 70% temporary, 30% permanent
            max_impact: 0.05,     // Cap at 5% of price
        }
    }
}

impl MarketImpactModel {
    /// Create a new market impact model
    pub const fn new(alpha: f64, beta: f64, temporary_ratio: f64, max_impact: f64) -> Self {
        Self {
            alpha,
            beta,
            temporary_ratio,
            max_impact,
        }
    }
    
    /// Estimate market impact for a given order size
    /// 
    /// Uses the Almgren-Chriss square-root impact model:
    /// impact = alpha * (size/volume) + beta * sqrt(size/volume)
    #[inline]
    pub fn estimate_impact(&self, order_size: f64, daily_volume: f64, price: f64) -> f64 {
        if order_size <= 0.0 || daily_volume <= EPSILON || price <= EPSILON {
            return 0.0;
        }
        
        let participation_rate = (order_size / daily_volume).min(1.0);
        let sqrt_participation = participation_rate.sqrt();
        
        // Square-root impact model
        let raw_impact = self.alpha * participation_rate + self.beta * sqrt_participation;
        
        // Apply cap
        raw_impact.min(self.max_impact)
    }
    
    /// Estimate temporary (reversible) impact
    #[inline]
    pub fn temporary_impact(&self, order_size: f64, daily_volume: f64, price: f64) -> f64 {
        self.estimate_impact(order_size, daily_volume, price) * self.temporary_ratio
    }
    
    /// Estimate permanent (irreversible) impact
    #[inline]
    pub fn permanent_impact(&self, order_size: f64, daily_volume: f64, price: f64) -> f64 {
        self.estimate_impact(order_size, daily_volume, price) * (1.0 - self.temporary_ratio)
    }
}

/// Spread-aware penalty calculator
pub struct SpreadPenalty {
    /// Base penalty multiplier
    base_multiplier: f64,
    /// Adaptive scaling factor for wide spreads
    spread_sensitivity: f64,
    /// Minimum spread threshold (below which no extra penalty)
    min_spread_bps: f64,
    /// Maximum penalty multiplier
    max_multiplier: f64,
}

impl SpreadPenalty {
    /// Create a new spread penalty calculator
    pub const fn new(
        base_multiplier: f64,
        spread_sensitivity: f64,
        min_spread_bps: f64,
        max_multiplier: f64,
    ) -> Self {
        Self {
            base_multiplier,
            spread_sensitivity,
            min_spread_bps,
            max_multiplier,
        }
    }
    
    /// Default configuration for crypto markets
    pub const fn crypto_default() -> Self {
        Self::new(1.0, 2.0, 1.0, 5.0)
    }
    
    /// Calculate spread-adjusted penalty multiplier
    #[inline]
    pub fn calculate_multiplier(&self, spread_bps: f64) -> f64 {
        if spread_bps <= self.min_spread_bps {
            return self.base_multiplier;
        }
        
        // Exponential scaling for wide spreads
        let excess_spread = spread_bps - self.min_spread_bps;
        let adaptive_factor = 1.0 + self.spread_sensitivity * (excess_spread / self.min_spread_bps);
        
        (self.base_multiplier * adaptive_factor).min(self.max_multiplier)
    }
    
    /// Calculate penalty in basis points
    #[inline]
    pub fn calculate_penalty_bps(&self, spread_bps: f64, notional: f64) -> f64 {
        let multiplier = self.calculate_multiplier(spread_bps);
        spread_bps * multiplier * notional * 1e-4
    }
}

/// Liquidity-adjusted transaction cost calculator
pub struct TransactionCostCalculator {
    /// Market impact model
    impact_model: MarketImpactModel,
    /// Spread penalty calculator
    spread_penalty: SpreadPenalty,
    /// Fixed fee rate (exchange fees)
    fee_rate: f64,
    /// Slippage estimation factor
    slippage_factor: f64,
}

impl TransactionCostCalculator {
    /// Create a new transaction cost calculator
    pub fn new(
        impact_model: MarketImpactModel,
        spread_penalty: SpreadPenalty,
        fee_rate: f64,
        slippage_factor: f64,
    ) -> Self {
        Self {
            impact_model,
            spread_penalty,
            fee_rate,
            slippage_factor,
        }
    }
    
    /// Default configuration for crypto perpetual futures
    pub fn crypto_perp_default() -> Self {
        Self::new(
            MarketImpactModel::default(),
            SpreadPenalty::crypto_default(),
            0.0004,   // 4 bps maker/taker average
            0.5,      // Conservative slippage estimate
        )
    }
    
    /// Calculate total transaction cost for an order
    /// 
    /// Returns cost as a fraction of notional value
    #[inline]
    pub fn calculate_total_cost(
        &self,
        order_size: f64,
        price: f64,
        daily_volume: f64,
        spread_bps: f64,
    ) -> f64 {
        let notional = order_size * price;
        
        if notional <= EPSILON {
            return 0.0;
        }
        
        // Market impact cost
        let impact_cost = self.impact_model.estimate_impact(order_size, daily_volume, price);
        
        // Spread cost (half-spread for crossing)
        let spread_multiplier = self.spread_penalty.calculate_multiplier(spread_bps);
        let spread_cost = spread_bps * 1e-4 * spread_multiplier * 0.5;
        
        // Fixed fees
        let fee_cost = self.fee_rate;
        
        // Slippage estimate
        let slippage_cost = self.slippage_factor * impact_cost;
        
        // Total cost as fraction of notional
        let total_fraction = impact_cost + spread_cost + fee_cost + slippage_cost;
        
        // Convert to absolute cost
        (total_fraction * notional).min(notional * 0.5) // Cap at 50% of notional
    }
    
    /// Calculate cost components separately
    #[inline]
    pub fn calculate_cost_breakdown(
        &self,
        order_size: f64,
        price: f64,
        daily_volume: f64,
        spread_bps: f64,
    ) -> CostBreakdown {
        let notional = order_size * price;
        
        if notional <= EPSILON {
            return CostBreakdown::zero();
        }
        
        let impact_cost = self.impact_model.estimate_impact(order_size, daily_volume, price);
        let spread_multiplier = self.spread_penalty.calculate_multiplier(spread_bps);
        let spread_cost = spread_bps * 1e-4 * spread_multiplier * 0.5;
        let fee_cost = self.fee_rate;
        let slippage_cost = self.slippage_factor * impact_cost;
        
        CostBreakdown {
            impact_cost: impact_cost * notional,
            spread_cost: spread_cost * notional,
            fee_cost: fee_cost * notional,
            slippage_cost: slippage_cost * notional,
            total: (impact_cost + spread_cost + fee_cost + slippage_cost) * notional,
        }
    }
}

/// Breakdown of transaction cost components
#[derive(Debug, Clone, Copy)]
pub struct CostBreakdown {
    pub impact_cost: f64,
    pub spread_cost: f64,
    pub fee_cost: f64,
    pub slippage_cost: f64,
    pub total: f64,
}

impl CostBreakdown {
    #[inline]
    pub const fn zero() -> Self {
        Self {
            impact_cost: 0.0,
            spread_cost: 0.0,
            fee_cost: 0.0,
            slippage_cost: 0.0,
            total: 0.0,
        }
    }
    
    /// Get cost as fraction of notional
    #[inline]
    pub fn as_fraction(&self, notional: f64) -> Self {
        if notional <= EPSILON {
            return *self;
        }
        Self {
            impact_cost: self.impact_cost / notional,
            spread_cost: self.spread_cost / notional,
            fee_cost: self.fee_cost / notional,
            slippage_cost: self.slippage_cost / notional,
            total: self.total / notional,
        }
    }
}

/// Dynamic penalty module for RL reward shaping
pub struct MarketImpactPenalty {
    /// Transaction cost calculator
    tc_calculator: TransactionCostCalculator,
    /// Cumulative penalty tracker
    cumulative_penalty: f64,
    /// Step counter
    step_count: AtomicU64,
    /// Penalty decay factor for historical weighting
    decay_factor: f64,
    /// Maximum single-step penalty
    max_step_penalty: f64,
}

impl MarketImpactPenalty {
    /// Create a new market impact penalty module
    pub fn new(tc_calculator: TransactionCostCalculator, decay_factor: f64) -> Self {
        Self {
            tc_calculator,
            cumulative_penalty: 0.0,
            step_count: AtomicU64::new(0),
            decay_factor: decay_factor.clamp(0.0, 1.0),
            max_step_penalty: 0.1, // 10% max penalty per step
        }
    }
    
    /// Default configuration
    pub fn crypto_default() -> Self {
        Self::new(
            TransactionCostCalculator::crypto_perp_default(),
            0.95,
        )
    }
    
    /// Calculate penalty for a potential action
    /// 
    /// Returns negative reward (penalty) scaled by estimated market impact
    #[inline]
    pub fn calculate_penalty(
        &mut self,
        order_size: f64,
        price: f64,
        daily_volume: f64,
        spread_bps: f64,
    ) -> f64 {
        let breakdown = self.tc_calculator.calculate_cost_breakdown(
            order_size,
            price,
            daily_volume,
            spread_bps,
        );
        
        // Apply decay to historical penalty
        self.cumulative_penalty *= self.decay_factor;
        self.cumulative_penalty += breakdown.total;
        
        // Increment step count
        self.step_count.fetch_add(1, Ordering::Relaxed);
        
        // Return normalized penalty (negative reward)
        let penalty = -breakdown.total / (price * order_size + EPSILON);
        penalty.max(-self.max_step_penalty) // Cap penalty
    }
    
    /// Get cumulative penalty over episode
    #[inline]
    pub fn cumulative_penalty(&self) -> f64 {
        self.cumulative_penalty
    }
    
    /// Get average penalty per step
    #[inline]
    pub fn average_penalty(&self) -> f64 {
        let count = self.step_count.load(Ordering::Acquire) as f64;
        if count < EPSILON {
            return 0.0;
        }
        self.cumulative_penalty / count
    }
    
    /// Reset state for new episode
    #[inline]
    pub fn reset(&mut self) {
        self.cumulative_penalty = 0.0;
        self.step_count.store(0, Ordering::Release);
    }
    
    /// Get step count
    #[inline]
    pub fn step_count(&self) -> u64 {
        self.step_count.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_market_impact_basic() {
        let model = MarketImpactModel::default();
        
        // Small order should have minimal impact
        let impact_small = model.estimate_impact(1.0, 1_000_000.0, 50_000.0);
        assert!(impact_small > 0.0);
        assert!(impact_small < 0.001);
        
        // Large order should have higher impact
        let impact_large = model.estimate_impact(10_000.0, 1_000_000.0, 50_000.0);
        assert!(impact_large > impact_small);
        
        // Impact should be capped
        let impact_huge = model.estimate_impact(1_000_000.0, 1_000_000.0, 50_000.0);
        assert!(impact_huge <= model.max_impact);
    }
    
    #[test]
    fn test_spread_penalty_scaling() {
        let penalty = SpreadPenalty::crypto_default();
        
        // Normal spread
        let mult_normal = penalty.calculate_multiplier(1.0);
        assert!((mult_normal - 1.0).abs() < 0.01);
        
        // Wide spread should increase multiplier
        let mult_wide = penalty.calculate_multiplier(10.0);
        assert!(mult_wide > mult_normal);
        
        // Very wide spread should hit max
        let mult_extreme = penalty.calculate_multiplier(100.0);
        assert!(mult_extreme <= penalty.max_multiplier);
    }
    
    #[test]
    fn test_transaction_cost_breakdown() {
        let calc = TransactionCostCalculator::crypto_perp_default();
        
        let breakdown = calc.calculate_cost_breakdown(
            100.0,   // order size
            50_000.0, // price
            1_000_000.0, // daily volume
            2.0,     // spread bps
        );
        
        // All components should be positive
        assert!(breakdown.impact_cost >= 0.0);
        assert!(breakdown.spread_cost >= 0.0);
        assert!(breakdown.fee_cost >= 0.0);
        assert!(breakdown.slippage_cost >= 0.0);
        assert!(breakdown.total > 0.0);
        
        // Total should equal sum of components
        let sum = breakdown.impact_cost + breakdown.spread_cost 
                + breakdown.fee_cost + breakdown.slippage_cost;
        assert!((breakdown.total - sum).abs() < 1e-10);
    }
    
    #[test]
    fn test_penalty_module() {
        let mut penalty = MarketImpactPenalty::crypto_default();
        
        let p1 = penalty.calculate_penalty(10.0, 50_000.0, 1_000_000.0, 2.0);
        let p2 = penalty.calculate_penalty(100.0, 50_000.0, 1_000_000.0, 2.0);
        
        // Larger order should have larger penalty
        assert!(p1.abs() < p2.abs());
        
        // Penalties should be negative (reducing reward)
        assert!(p1 <= 0.0);
        assert!(p2 <= 0.0);
        
        // Check cumulative tracking
        assert!(penalty.cumulative_penalty() > 0.0);
    }
    
    #[test]
    fn test_zero_order_handling() {
        let mut penalty = MarketImpactPenalty::crypto_default();
        
        let p = penalty.calculate_penalty(0.0, 50_000.0, 1_000_000.0, 2.0);
        assert!((p - 0.0).abs() < 1e-10);
    }
}
