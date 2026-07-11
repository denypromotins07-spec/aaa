//! Teleological Barbell Hedging Engine.
//! Constructs barbell portfolios: aggressive short-term alpha + tail-risk hedges against paradigm collapse.

use alloc::vec::Vec;
use core::fmt::Debug;

use super::super::attractors::kaplan_yorke_dimension::{KaplanYorkeCalculator, KaplanYorkeConfig};
use super::eschatological_option_pricer::{
    EschatologicalEvent, EschatologicalOptionBuilder, EschatologicalOptionPricer,
};
use super::paradigm_transition_swap::{ParadigmSwapBuilder, ParadigmTransitionSwap};

/// Configuration for teleological barbell strategy
#[derive(Debug, Clone)]
pub struct TeleologicalBarbellConfig {
    /// Total portfolio value
    pub total_value: f64,
    /// Allocation to aggressive alpha (0-1)
    pub alpha_allocation: f64,
    /// Allocation to tail hedges (0-1)
    pub hedge_allocation: f64,
    /// Target tail protection level (probability)
    pub tail_protection_target: f64,
    /// Rebalance threshold (deviation from target)
    pub rebalance_threshold: f64,
}

impl Default for TeleologicalBarbellConfig {
    fn default() -> Self {
        Self {
            total_value: 10_000_000.0,
            alpha_allocation: 0.8,
            hedge_allocation: 0.2,
            tail_protection_target: 0.95,
            rebalance_threshold: 0.05,
        }
    }
}

/// Components of the barbell portfolio
#[derive(Debug, Clone)]
pub struct BarbellPortfolio {
    /// Value in aggressive alpha strategies
    pub alpha_value: f64,
    /// Value in tail hedges
    pub hedge_value: f64,
    /// List of eschatological put options
    pub eschatological_puts: Vec<HedgePosition>,
    /// Paradigm transition swaps
    pub paradigm_swaps: Vec<SwapPosition>,
    /// Cash reserve
    pub cash: f64,
}

/// Hedge position details
#[derive(Debug, Clone)]
pub struct HedgePosition {
    /// Option type (put on specific asset/event)
    pub event_type: EschatologicalEvent,
    /// Notional exposure
    pub notional: f64,
    /// Strike level
    pub strike: f64,
    /// Time to expiry (years)
    pub time_to_expiry: f64,
    /// Current option value
    pub current_value: f64,
}

/// Swap position details
#[derive(Debug, Clone)]
pub struct SwapPosition {
    /// Notional amount
    pub notional: f64,
    /// Fixed rate paid
    pub fixed_rate: f64,
    /// Floating rate received
    pub floating_rate: f64,
    /// Remaining tenor (years)
    pub remaining_tenor: f64,
    /// Current MTM value
    pub mtm_value: f64,
}

/// Result of barbell analysis
#[derive(Debug, Clone)]
pub struct BarbellAnalysis {
    /// Current portfolio composition
    pub portfolio: BarbellPortfolio,
    /// Effective tail protection achieved
    pub tail_protection: f64,
    /// Cost of hedging (annualized)
    pub hedge_cost: f64,
    /// Whether rebalancing is needed
    pub needs_rebalance: bool,
    /// Recommended rebalancing trades
    pub rebalancing_trades: Vec<RebalanceTrade>,
}

/// Rebalancing trade recommendation
#[derive(Debug, Clone)]
pub struct RebalanceTrade {
    /// Action (buy/sell)
    pub action: &'static str,
    /// Component (alpha/hedge)
    pub component: &'static str,
    /// Amount to trade
    pub amount: f64,
    /// Reason for trade
    pub reason: &'static str,
}

/// Teleological Barbell Hedging Engine
pub struct TeleologicalBarbellEngine {
    config: TeleologicalBarbellConfig,
    option_pricer: EschatologicalOptionPricer,
    swap_manager: ParadigmTransitionSwap,
}

impl TeleologicalBarbellEngine {
    pub fn new(config: TeleologicalBarbellConfig) -> Self {
        let option_config = super::eschatological_option_pricer::EschatologicalOptionConfig {
            event_type: EschatologicalEvent::CurrencyCollapse,
            time_horizon: 10.0,
            event_probability: 0.1,
            risk_free_rate: 0.02,
        };
        
        let swap_config = super::paradigm_transition_swap::ParadigmSwapConfig {
            notional: config.total_value * config.hedge_allocation,
            tenor: 5.0,
            payment_frequency: 4,
            shift_threshold: 0.3,
        };

        Self {
            config,
            option_pricer: EschatologicalOptionPricer::new(option_config),
            swap_manager: ParadigmTransitionSwap::new(swap_config),
        }
    }

    /// Construct optimal barbell portfolio
    pub fn construct_portfolio(
        &self,
        alpha_strategies: &[AlphaStrategy],
        market_exponents: &[f64],
    ) -> Result<BarbellPortfolio, &'static str> {
        let alpha_value = self.config.total_value * self.config.alpha_allocation;
        let hedge_value = self.config.total_value * self.config.hedge_allocation;

        // Allocate hedges based on tail protection target
        let mut eschatological_puts = Vec::new();
        let mut paradigm_swaps = Vec::new();

        // Diversify across event types
        let events = [
            EschatologicalEvent::CurrencyCollapse,
            EschatologicalEvent::RegimeChange,
            EschatologicalEvent::AssetExtinction,
        ];

        let hedge_per_event = hedge_value / events.len() as f64;

        for event in &events {
            let option_config = super::eschatological_option_pricer::EschatologicalOptionConfig {
                event_type: event.clone(),
                time_horizon: 10.0,
                event_probability: 0.1,
                risk_free_rate: 0.02,
            };
            let pricer = EschatologicalOptionPricer::new(option_config);
            
            // Price OTM put for tail protection
            let price_result = pricer.price(150.0, 100.0)?;
            
            eschatological_puts.push(HedgePosition {
                event_type: event.clone(),
                notional: hedge_per_event,
                strike: 150.0,
                time_to_expiry: 10.0,
                current_value: price_result.put_price * hedge_per_event / 100.0,
            });
        }

        // Add paradigm swap if market shows signs of instability
        if let Ok(detection) = self.swap_manager.detect_shift(market_exponents) {
            if detection.shift_detected || detection.confidence > 0.3 {
                paradigm_swaps.push(SwapPosition {
                    notional: hedge_value * 0.3,
                    fixed_rate: 0.03,
                    floating_rate: 0.05,
                    remaining_tenor: 5.0,
                    mtm_value: 0.0, // Initial MTM
                });
            }
        }

        // Remaining cash
        let total_hedge_value: f64 = eschatological_puts.iter().map(|h| h.current_value).sum();
        let swap_value: f64 = paradigm_swaps.iter().map(|s| s.mtm_value).sum();
        let cash = hedge_value - total_hedge_value - swap_value;

        Ok(BarbellPortfolio {
            alpha_value,
            hedge_value,
            eschatological_puts,
            paradigm_swaps,
            cash,
        })
    }

    /// Analyze current barbell state
    pub fn analyze(&self, portfolio: &BarbellPortfolio, market_exponents: &[f64]) -> BarbellAnalysis {
        let total_value = portfolio.alpha_value 
            + portfolio.eschatological_puts.iter().map(|h| h.current_value).sum::<f64>()
            + portfolio.paradigm_swaps.iter().map(|s| s.mtm_value).sum::<f64>()
            + portfolio.cash;

        // Calculate effective tail protection
        let tail_protection = self.calculate_tail_protection(portfolio, market_exponents);

        // Calculate annualized hedge cost
        let hedge_cost = self.calculate_hedge_cost(portfolio);

        // Check if rebalancing needed
        let actual_alpha_ratio = portfolio.alpha_value / total_value;
        let target_alpha_ratio = self.config.alpha_allocation;
        let deviation = (actual_alpha_ratio - target_alpha_ratio).abs();

        let needs_rebalance = deviation > self.config.rebalance_threshold;

        // Generate rebalancing recommendations
        let rebalancing_trades = if needs_rebalance {
            self.generate_rebalance_trades(portfolio, total_value)
        } else {
            Vec::new()
        };

        BarbellAnalysis {
            portfolio: portfolio.clone(),
            tail_protection,
            hedge_cost,
            needs_rebalance,
            rebalancing_trades,
        }
    }

    fn calculate_tail_protection(&self, portfolio: &BarbellPortfolio, exponents: &[f64]) -> f64 {
        // Simplified: sum of hedge notionals weighted by event probability
        let mut protection = 0.0;
        
        for hedge in &portfolio.eschatological_puts {
            let ky_config = KaplanYorkeConfig::default();
            let ky_calc = KaplanYorkeCalculator::new(ky_config);
            
            // Higher protection when dimension collapse detected
            let dim_factor = match ky_calc.calculate(exponents) {
                Ok(result) => if result.dimension < 2.0 { 1.5 } else { 1.0 },
                Err(_) => 1.0,
            };
            
            protection += hedge.notional * 0.1 * dim_factor; // 10% base protection
        }

        for swap in &portfolio.paradigm_swaps {
            protection += swap.notional * 0.05; // 5% protection from swaps
        }

        protection / self.config.total_value
    }

    fn calculate_hedge_cost(&self, portfolio: &BarbellPortfolio) -> f64 {
        let mut annual_cost = 0.0;

        // Option decay (theta)
        for hedge in &portfolio.eschatological_puts {
            annual_cost += hedge.current_value * 0.1; // ~10% annual decay
        }

        // Swap carry
        for swap in &portfolio.paradigm_swaps {
            let net_rate = swap.floating_rate - swap.fixed_rate;
            annual_cost -= swap.notional * net_rate; // Negative = income
        }

        annual_cost
    }

    fn generate_rebalance_trades(&self, portfolio: &BarbellPortfolio, total_value: f64) -> Vec<RebalanceTrade> {
        let mut trades = Vec::new();

        let actual_alpha = portfolio.alpha_value / total_value;
        let target_alpha = self.config.alpha_allocation;

        if actual_alpha > target_alpha + self.config.rebalance_threshold {
            let excess = (actual_alpha - target_alpha) * total_value;
            trades.push(RebalanceTrade {
                action: "SELL",
                component: "ALPHA",
                amount: excess,
                reason: "Alpha allocation exceeds target",
            });
            trades.push(RebalanceTrade {
                action: "BUY",
                component: "HEDGE",
                amount: excess,
                reason: "Increase tail protection",
            });
        } else if actual_alpha < target_alpha - self.config.rebalance_threshold {
            let deficit = (target_alpha - actual_alpha) * total_value;
            trades.push(RebalanceTrade {
                action: "SELL",
                component: "HEDGE",
                amount: deficit,
                reason: "Reduce excess hedging",
            });
            trades.push(RebalanceTrade {
                action: "BUY",
                component: "ALPHA",
                amount: deficit,
                reason: "Increase alpha exposure",
            });
        }

        trades
    }
}

/// Alpha strategy definition
#[derive(Debug, Clone)]
pub struct AlphaStrategy {
    pub name: String,
    pub expected_return: f64,
    pub volatility: f64,
    pub capacity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barbell_construction() {
        let config = TeleologicalBarbellConfig::default();
        let engine = TeleologicalBarbellEngine::new(config);

        let strategies = vec![
            AlphaStrategy {
                name: "HFT Momentum".to_string(),
                expected_return: 0.15,
                volatility: 0.05,
                capacity: 5_000_000.0,
            },
        ];

        let exponents = vec![0.5, -0.2, -1.3];
        let portfolio = engine.construct_portfolio(&strategies, &exponents);

        assert!(portfolio.is_ok());
        let p = portfolio.unwrap();
        assert!(p.alpha_value > 0.0);
        assert!(p.hedge_value > 0.0);
    }

    #[test]
    fn test_portfolio_analysis() {
        let config = TeleologicalBarbellConfig::default();
        let engine = TeleologicalBarbellEngine::new(config);

        let portfolio = BarbellPortfolio {
            alpha_value: 8_000_000.0,
            hedge_value: 2_000_000.0,
            eschatological_puts: vec![],
            paradigm_swaps: vec![],
            cash: 100_000.0,
        };

        let exponents = vec![0.5, -0.2, -1.3];
        let analysis = engine.analyze(&portfolio, &exponents);

        assert!(analysis.tail_protection >= 0.0);
        assert!(analysis.hedge_cost.is_finite());
    }
}
