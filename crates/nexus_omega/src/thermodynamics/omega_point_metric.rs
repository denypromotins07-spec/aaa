//! Omega Point Metric - measures approach to maximum computational efficiency.
//! Tracks when markets reach thermodynamic limits of alpha extraction.

use alloc::vec::Vec;
use core::fmt::Debug;

use super::landauer_limit_calculator::{
    ComputeEfficiency, EfficiencyStatus, LandauerCalculator, MarketEfficiencyTracker,
    StrategyEfficiency,
};

/// Configuration for Omega Point analysis
#[derive(Debug, Clone)]
pub struct OmegaPointConfig {
    /// Temperature for Landauer calculations (K)
    pub temperature: f64,
    /// Threshold for declaring Omega Point (orders of magnitude)
    pub omega_threshold: f64,
    /// Minimum data points for trend analysis
    pub min_data_points: usize,
}

impl Default for OmegaPointConfig {
    fn default() -> Self {
        Self {
            temperature: 300.0,
            omega_threshold: 3.0,
            min_data_points: 100,
        }
    }
}

/// Result of Omega Point analysis
#[derive(Debug, Clone)]
pub struct OmegaPointResult {
    /// Current efficiency ratio (log scale)
    pub current_efficiency_log: f64,
    /// Trend in efficiency (positive = improving)
    pub efficiency_trend: f64,
    /// Estimated time to Omega Point (in arbitrary units)
    pub time_to_omega: Option<f64>,
    /// Current market state
    pub market_state: OmegaMarketState,
    /// Recommended action
    pub recommendation: &'static str,
}

/// Market state relative to Omega Point
#[derive(Debug, Clone, PartialEq)]
pub enum OmegaMarketState {
    /// Far from limit, abundant alpha
    AlphaAbundant,
    /// Approaching limits, alpha depleting
    AlphaDepleting,
    /// Near Omega Point, minimal alpha
    NearOmega,
    /// At Omega Point, no extractable alpha
    AtOmega,
    /// Past Omega (theoretical impossibility)
    Anomaly,
}

/// Omega Point metric calculator
pub struct OmegaPointMetric {
    config: OmegaPointConfig,
    tracker: MarketEfficiencyTracker,
}

impl OmegaPointMetric {
    pub fn new(config: OmegaPointConfig) -> Self {
        let tracker = MarketEfficiencyTracker::new(config.temperature, config.omega_threshold);
        Self { config, tracker }
    }

    /// Analyze current market efficiency state
    pub fn analyze(&self, energy_per_trade_j: f64, bits_processed: u64) -> OmegaPointResult {
        let efficiency = self.tracker.analyze_strategy_efficiency(energy_per_trade_j, bits_processed);
        
        let market_state = match efficiency.status {
            EfficiencyStatus::Inefficient => OmegaMarketState::AlphaAbundant,
            EfficiencyStatus::ModeratelyEfficient => OmegaMarketState::AlphaDepleting,
            EfficiencyStatus::NearLimit => OmegaMarketState::NearOmega,
            EfficiencyStatus::AtOmegaPoint => OmegaMarketState::AtOmega,
        };

        let recommendation = match market_state {
            OmegaMarketState::AlphaAbundant => "EXPLOIT: High alpha available, increase position",
            OmegaMarketState::AlphaDepleting => "CAUTION: Alpha depleting, reduce exposure",
            OmegaMarketState::NearOmega => "EXIT: Nearly at thermodynamic limit",
            OmegaMarketState::AtOmega => "ABANDON: No further alpha possible",
            OmegaMarketState::Anomaly => "INVESTIGATE: Measurement error or new physics",
        };

        OmegaPointResult {
            current_efficiency_log: efficiency.efficiency_ratio.log10(),
            efficiency_trend: 0.0, // Would calculate from historical data
            time_to_omega: self.estimate_time_to_omega(efficiency.efficiency_ratio),
            market_state,
            recommendation,
        }
    }

    /// Estimate time until Omega Point based on efficiency trend
    fn estimate_time_to_omega(&self, current_ratio: f64) -> Option<f64> {
        if current_ratio.log10() < self.config.omega_threshold {
            return Some(0.0); // Already at Omega Point
        }
        
        // Simplified: would need historical trend data
        // Assuming exponential improvement in efficiency
        let improvement_rate = 0.1; // 10% per time unit
        let log_current = current_ratio.log10();
        let log_threshold = self.config.omega_threshold;
        
        if improvement_rate > 0.0 && log_current > log_threshold {
            Some((log_current - log_threshold) / improvement_rate)
        } else {
            None
        }
    }

    /// Analyze multiple strategies for portfolio-level Omega assessment
    pub fn analyze_portfolio(
        &self,
        strategies: &[(f64, u64)], // (energy_per_trade, bits)
    ) -> PortfolioOmegaResult {
        if strategies.is_empty() {
            return PortfolioOmegaResult {
                average_efficiency_log: f64::MAX,
                num_at_omega: 0,
                num_near_omega: 0,
                portfolio_recommendation: "NO_DATA",
            };
        }

        let mut total_log_efficiency = 0.0;
        let mut num_at_omega = 0usize;
        let mut num_near_omega = 0usize;

        for &(energy, bits) in strategies {
            let result = self.analyze(energy, bits);
            total_log_efficiency += result.current_efficiency_log;
            
            match result.market_state {
                OmegaMarketState::AtOmega => num_at_omega += 1,
                OmegaMarketState::NearOmega => num_near_omega += 1,
                _ => {}
            }
        }

        let avg_log = total_log_efficiency / strategies.len() as f64;
        
        let recommendation = if num_at_omega > strategies.len() / 2 {
            "REBALANCE: Majority of strategies at Omega Point"
        } else if num_near_omega + num_at_omega > strategies.len() * 2 / 3 {
            "DIVERSIFY: Seek new alpha sources"
        } else if avg_log < 5.0 {
            "MONITOR: Portfolio approaching efficiency limits"
        } else {
            "MAINTAIN: Adequate alpha across portfolio"
        };

        PortfolioOmegaResult {
            average_efficiency_log: avg_log,
            num_at_omega,
            num_near_omega,
            portfolio_recommendation: recommendation,
        }
    }
}

/// Portfolio-level Omega Point result
#[derive(Debug, Clone)]
pub struct PortfolioOmegaResult {
    pub average_efficiency_log: f64,
    pub num_at_omega: usize,
    pub num_near_omega: usize,
    pub portfolio_recommendation: &'static str,
}

/// Efficiency horizon tracker for strategy mutation decisions
pub struct EfficiencyHorizonTracker {
    omega_metric: OmegaPointMetric,
    mutation_threshold: f64,
}

impl EfficiencyHorizonTracker {
    pub fn new(config: OmegaPointConfig, mutation_threshold: f64) -> Self {
        Self {
            omega_metric: OmegaPointMetric::new(config),
            mutation_threshold,
        }
    }

    /// Determine if strategy should mutate to higher-dimensional alpha
    pub fn should_mutate(&self, energy_per_trade_j: f64, bits_processed: u64) -> MutationDecision {
        let result = self.omega_metric.analyze(energy_per_trade_j, bits_processed);
        
        let should_mutate = match result.market_state {
            OmegaMarketState::AtOmega => true,
            OmegaMarketState::NearOmega => result.current_efficiency_log < self.mutation_threshold,
            _ => false,
        };

        let urgency = match result.market_state {
            OmegaMarketState::AtOmega => MutationUrgency::Critical,
            OmegaMarketState::NearOmega => MutationUrgency::High,
            OmegaMarketState::AlphaDepleting => MutationUrgency::Medium,
            _ => MutationUrgency::Low,
        };

        MutationDecision {
            should_mutate,
            urgency,
            current_efficiency: result.current_efficiency_log,
            recommended_direction: self.get_mutation_direction(&result.market_state),
        }
    }

    fn get_mutation_direction(&self, state: &OmegaMarketState) -> &'static str {
        match state {
            OmegaMarketState::AtOmega => "Seek entirely new asset class or timeframe",
            OmegaMarketState::NearOmega => "Increase model complexity or reduce latency",
            OmegaMarketState::AlphaDepleting => "Optimize current strategy parameters",
            _ => "Maintain current approach",
        }
    }
}

/// Mutation decision result
#[derive(Debug, Clone)]
pub struct MutationDecision {
    pub should_mutate: bool,
    pub urgency: MutationUrgency,
    pub current_efficiency: f64,
    pub recommended_direction: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MutationUrgency {
    Critical,
    High,
    Medium,
    Low,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_omega_point_far_from_limit() {
        let config = OmegaPointConfig::default();
        let metric = OmegaPointMetric::new(config);
        
        // High energy usage = far from Omega Point
        let result = metric.analyze(1e-9, 1000);
        
        assert_eq!(result.market_state, OmegaMarketState::AlphaAbundant);
        assert!(result.time_to_omega.is_some());
    }

    #[test]
    fn test_omega_point_at_limit() {
        let config = OmegaPointConfig::default();
        let metric = OmegaPointMetric::new(config);
        
        let calc = LandauerCalculator::room_temperature();
        let landauer = calc.calculate_multi_bit(1000);
        
        // Energy very close to Landauer limit
        let result = metric.analyze(landauer.total_energy_fj.to_joules() * 10.0, 1000);
        
        assert_eq!(result.market_state, OmegaMarketState::AtOmega);
    }

    #[test]
    fn test_portfolio_analysis() {
        let config = OmegaPointConfig::default();
        let metric = OmegaPointMetric::new(config);
        
        let strategies = vec![
            (1e-9, 1000),  // Inefficient
            (1e-12, 1000), // Moderate
            (1e-15, 1000), // Efficient
        ];
        
        let portfolio = metric.analyze_portfolio(&strategies);
        
        assert!(portfolio.average_efficiency_log.is_finite());
    }
}
