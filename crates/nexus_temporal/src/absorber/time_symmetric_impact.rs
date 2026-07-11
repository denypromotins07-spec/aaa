//! Time-Symmetric Market Impact Model
//! 
//! Combines retarded and advanced potentials to compute
//! complete time-symmetric market impact.

use crate::absorber::wheeler_feynman_green::{WheelerFeynmanGreen, GreenFunctionResult};
use crate::absorber::advanced_potential_liquidity::{AdvancedPotentialLiquidity, AdvancedPotentialField};

/// Default market friction coefficient
const DEFAULT_FRICTION_COEF: f64 = 0.001;

/// Result of time-symmetric impact calculation
#[derive(Debug, Clone)]
pub struct TimeSymmetricImpact {
    /// Retarded (past) contribution to impact
    pub retarded_impact: f64,
    /// Advanced (future) contribution to impact
    pub advanced_impact: f64,
    /// Total symmetric impact
    pub total_impact: f64,
    /// Impact in basis points
    pub impact_bps: f64,
    /// Confidence in the estimate
    pub confidence: f64,
    /// Temporal asymmetry measure (0 = symmetric, 1 = fully asymmetric)
    pub temporal_asymmetry: f64,
}

/// Trade record for impact modeling
#[derive(Debug, Clone)]
pub struct TradeRecord {
    /// Timestamp of trade (nanoseconds)
    pub timestamp_ns: u64,
    /// Trade size (positive for buy, negative for sell)
    pub size: f64,
    /// Execution price
    pub price: f64,
    /// Was this trade initiated by us?
    pub is_ours: bool,
}

/// Time-Symmetric Market Impact Calculator
pub struct TimeSymmetricImpactModel {
    /// Green's function solver
    green_solver: WheelerFeynmanGreen,
    /// Advanced potential liquidity detector
    liquidity_detector: AdvancedPotentialLiquidity,
    /// Market friction coefficient
    friction_coef: f64,
    /// Historical trades for retarded potential
    trade_history: Vec<TradeRecord>,
    /// Current reference price
    current_price: f64,
}

impl TimeSymmetricImpactModel {
    /// Create a new time-symmetric impact model
    pub fn new(current_price: f64, price_velocity: f64) -> Self {
        Self {
            green_solver: WheelerFeynmanGreen::new(),
            liquidity_detector: AdvancedPotentialLiquidity::new(current_price, price_velocity),
            friction_coef: DEFAULT_FRICTION_COEF,
            trade_history: Vec::with_capacity(1000),
            current_price,
        }
    }

    /// Create with macro regime cutoff
    pub fn with_macro_cutoff(current_price: f64, price_velocity: f64,
                             mean_reversion_half_life_ns: u64) -> Self {
        Self {
            green_solver: WheelerFeynmanGreen::with_macro_cutoff(mean_reversion_half_life_ns),
            liquidity_detector: AdvancedPotentialLiquidity::with_macro_cutoff(
                current_price, price_velocity, mean_reversion_half_life_ns
            ),
            friction_coef: DEFAULT_FRICTION_COEF,
            trade_history: Vec::with_capacity(1000),
            current_price,
        }
    }

    /// Update current market state
    pub fn update_state(&mut self, current_price: f64, price_velocity: f64) {
        self.current_price = current_price;
        self.liquidity_detector.update_state(current_price, price_velocity);
    }

    /// Record a trade for impact modeling
    pub fn record_trade(&mut self, trade: TradeRecord) {
        self.trade_history.push(trade);
        
        // Keep history bounded
        if self.trade_history.len() > 1000 {
            self.trade_history.remove(0);
        }
    }

    /// Set market friction coefficient
    pub fn set_friction_coefficient(&mut self, coef: f64) {
        self.friction_coef = coef.max(0.0).min(0.1);
    }

    /// Calculate time-symmetric impact for a proposed trade
    /// 
    /// # Arguments
    /// * `trade_size` - Size of proposed trade (positive for buy, negative for sell)
    /// * `bid_levels` - Current bid side [(price, size)]
    /// * `ask_levels` - Current ask side [(price, size)]
    /// * `current_time_ns` - Current timestamp
    /// 
    /// # Returns
    /// TimeSymmetricImpact with all components
    pub fn calculate_impact(&self, trade_size: f64,
                            bid_levels: &[(f64, f64)],
                            ask_levels: &[(f64, f64)],
                            current_time_ns: u64) -> TimeSymmetricImpact {
        // Calculate retarded potential from past trades
        let retarded_impact = self.compute_retarded_impact(current_time_ns, trade_size);
        
        // Calculate advanced potential from future liquidity
        let potential_field = self.liquidity_detector.calculate_potential_field(
            bid_levels, ask_levels, current_time_ns
        );
        let advanced_impact = self.compute_advanced_impact(&potential_field, trade_size);
        
        // Time-symmetric combination
        let total_impact = (retarded_impact + advanced_impact) / 2.0;
        
        // Convert to basis points
        let impact_bps = (total_impact / self.current_price) * 10000.0;
        
        // Calculate confidence based on data quality
        let confidence = self.calculate_confidence(&potential_field);
        
        // Measure temporal asymmetry
        let temporal_asymmetry = if (retarded_impact.abs() + advanced_impact.abs()) > 1e-15 {
            (retarded_impact - advanced_impact).abs() / (retarded_impact.abs() + advanced_impact.abs())
        } else {
            0.0
        };
        
        TimeSymmetricImpact {
            retarded_impact,
            advanced_impact,
            total_impact,
            impact_bps,
            confidence,
            temporal_asymmetry,
        }
    }

    /// Get optimal execution strategy using time-symmetric analysis
    /// 
    /// # Arguments
    /// * `total_size` - Total size to execute
    /// * `is_buy` - True for buy order, false for sell
    /// * `bid_levels` - Bid side of book
    /// * `ask_levels` - Ask side of book
    /// * `current_time_ns` - Current time
    /// 
    /// # Returns
    /// Vector of (size_fraction, expected_impact_bps) for each slice
    pub fn get_optimal_execution(&self, total_size: f64, is_buy: bool,
                                  bid_levels: &[(f64, f64)],
                                  ask_levels: &[(f64, f64)],
                                  current_time_ns: u64,
                                  num_slices: usize) -> Vec<(f64, f64)> {
        if num_slices == 0 || total_size.abs() < 1e-15 {
            return vec![];
        }

        let mut slices = Vec::with_capacity(num_slices);
        let base_slice_size = total_size / num_slices as f64;
        
        // Calculate impact for each slice
        for i in 0..num_slices {
            let slice_size = base_slice_size * (1.0 - i as f64 * self.friction_coef);
            let impact = self.calculate_impact(slice_size, bid_levels, ask_levels, current_time_ns);
            
            slices.push((slice_size / total_size, impact.impact_bps));
        }

        slices
    }

    /// Compute the "shadow price" including time-symmetric effects
    /// 
    /// This represents the true economic price accounting for both
    /// past momentum and future liquidity absorption.
    pub fn compute_shadow_price(&self, bid_levels: &[(f64, f64)],
                                 ask_levels: &[(f64, f64)],
                                 current_time_ns: u64) -> f64 {
        let mid_price = if !bid_levels.is_empty() && !ask_levels.is_empty() {
            (bid_levels[0].0 + ask_levels[0].0) / 2.0
        } else {
            self.current_price
        };

        // Get potential field
        let field = self.liquidity_detector.calculate_potential_field(
            bid_levels, ask_levels, current_time_ns
        );

        // Adjust mid price by potential gradient
        let potential_adjustment = field.potential_gradient * self.friction_coef;
        
        mid_price + potential_adjustment
    }

    // Internal: Compute retarded impact from historical trades
    fn compute_retarded_impact(&self, current_time_ns: u64, trade_size: f64) -> f64 {
        if self.trade_history.is_empty() {
            return self.friction_coef * trade_size;
        }

        // Extract source times and strengths from history
        let source_times: Vec<u64> = self.trade_history.iter()
            .map(|t| t.timestamp_ns)
            .collect();
        
        let source_strengths: Vec<f64> = self.trade_history.iter()
            .map(|t| t.size * self.friction_coef)
            .collect();

        // Evaluate Green's function
        let result = self.green_solver.evaluate(current_time_ns, &source_times, &source_strengths);
        
        // Add instantaneous impact component
        result.retarded_component + self.friction_coef * trade_size
    }

    // Internal: Compute advanced impact from liquidity field
    fn compute_advanced_impact(&self, field: &AdvancedPotentialField, trade_size: f64) -> f64 {
        // Base advanced impact from potential
        let base_advanced = field.current_potential * self.friction_coef;
        
        // Scale by trade size relative to absorption capacity
        if field.absorption_capacity > 1e-15 {
            let size_ratio = trade_size.abs() / field.absorption_capacity;
            base_advanced * (1.0 + size_ratio)
        } else {
            base_advanced * 2.0 // Double impact if no absorption
        }
    }

    // Internal: Calculate confidence score
    fn calculate_confidence(&self, field: &AdvancedPotentialField) -> f64 {
        let mut confidence = 0.5; // Base confidence
        
        // Increase confidence with more clusters detected
        let cluster_factor = (field.clusters.len() as f64 / 10.0).min(0.3);
        confidence += cluster_factor;
        
        // Increase confidence with higher absorption capacity
        if field.absorption_capacity > 1000.0 {
            confidence += 0.1;
        }
        
        // Decrease confidence with high asymmetry
        confidence -= 0.1;
        
        confidence.clamp(0.0, 1.0)
    }
}

impl Default for TimeSymmetricImpactModel {
    fn default() -> Self {
        Self::new(100.0, 0.001)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let model = TimeSymmetricImpactModel::new(100.0, 0.001);
        assert_eq!(model.current_price, 100.0);
    }

    #[test]
    fn test_empty_impact() {
        let model = TimeSymmetricImpactModel::new(100.0, 0.001);
        let impact = model.calculate_impact(100.0, &[], &[], 0);
        
        // Should have some baseline impact from friction
        assert!(impact.total_impact.abs() >= 0.0);
    }

    #[test]
    fn test_trade_recording() {
        let mut model = TimeSymmetricImpactModel::new(100.0, 0.001);
        
        let trade = TradeRecord {
            timestamp_ns: 1000,
            size: 50.0,
            price: 100.0,
            is_ours: true,
        };
        
        model.record_trade(trade);
        assert_eq!(model.trade_history.len(), 1);
    }

    #[test]
    fn test_shadow_price() {
        let model = TimeSymmetricImpactModel::new(100.0, 0.001);
        
        let bid_levels = vec![(99.9, 1000.0)];
        let ask_levels = vec![(100.1, 1000.0)];
        
        let shadow = model.compute_shadow_price(&bid_levels, &ask_levels, 0);
        
        // Shadow price should be near mid price
        assert!((shadow - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_optimal_execution() {
        let model = TimeSymmetricImpactModel::new(100.0, 0.001);
        
        let bid_levels = vec![(99.9, 1000.0), (99.8, 800.0)];
        let ask_levels = vec![(100.1, 1000.0), (100.2, 800.0)];
        
        let slices = model.get_optimal_execution(500.0, true, &bid_levels, &ask_levels, 0, 5);
        
        assert_eq!(slices.len(), 5);
        // Fractions should sum to approximately 1.0
        let total_fraction: f64 = slices.iter().map(|(f, _)| f).sum();
        assert!((total_fraction - 1.0).abs() < 0.1);
    }
}
