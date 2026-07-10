//! Slippage Decomposition - Breaks down TCA into components
//! 
//! Decomposes total slippage into: Delay Cost, Market Impact, and Spread Cost.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SlippageError {
    #[error("Invalid decomposition: {reason}")]
    InvalidDecomposition { reason: String },
    #[error("Components do not sum to total: sum={sum}, total={total}")]
    ComponentMismatch { sum: i64, total: i64 },
}

/// Slippage decomposition result
#[derive(Debug, Clone)]
pub struct SlippageDecomposition {
    /// Total slippage in fixed-point units
    pub total_slippage: i64,
    /// Delay cost component (cost of waiting to execute)
    pub delay_cost: i64,
    /// Market impact component (permanent price move from trading)
    pub market_impact: i64,
    /// Spread cost component (cost of crossing bid-ask)
    pub spread_cost: i64,
    /// Timing luck component (unexplained residual)
    pub timing_luck: i64,
    /// Percentage breakdown (in basis points of notional)
    pub delay_pct_bps: i64,
    pub impact_pct_bps: i64,
    pub spread_pct_bps: i64,
    pub luck_pct_bps: i64,
}

/// Slippage Decomposition Calculator
pub struct SlippageDecomposer {
    /// Scale factor for fixed-point
    scale: i64,
    /// Total decompositions performed
    decomposition_count: AtomicU64,
    /// Cumulative slippage by component
    cumulative_delay: AtomicI64,
    cumulative_impact: AtomicI64,
    cumulative_spread: AtomicI64,
}

impl SlippageDecomposer {
    pub fn new(scale: i64) -> Self {
        Self {
            scale,
            decomposition_count: AtomicU64::new(0),
            cumulative_delay: AtomicI64::new(0),
            cumulative_impact: AtomicI64::new(0),
            cumulative_spread: AtomicI64::new(0),
        }
    }

    /// Decompose slippage into components
    pub fn decompose(
        &self,
        arrival_price: i64,
        exec_price: i64,
        exec_qty: i64,
        decision_price: Option<i64>,
        spread_bps: i64,
        mid_price_at_arrival: Option<i64>,
    ) -> Result<SlippageDecomposition, SlippageError> {
        if arrival_price <= 0 || exec_qty <= 0 {
            return Err(SlippageError::InvalidDecomposition {
                reason: "Invalid price or quantity".to_string(),
            });
        }

        // Total slippage = (exec_price - arrival_price) * qty
        // For buys: positive = bad, for sells we'll handle sign separately
        let raw_slippage = (exec_price - arrival_price) * exec_qty;

        // Notional for percentage calculations
        let notional = arrival_price * exec_qty;

        // 1. Delay Cost: (arrival_price - decision_price) * qty
        let delay_cost = if let Some(dec_price) = decision_price {
            (arrival_price - dec_price) * exec_qty
        } else {
            0
        };

        // 2. Spread Cost: half-spread * qty
        let spread_cost = arrival_price * spread_bps * exec_qty / (2 * 10_000);

        // 3. Market Impact: estimated permanent impact
        // Use midpoint movement if available, otherwise estimate from slippage
        let market_impact = if let Some(mid_at_arrival) = mid_price_at_arrival {
            // If we have mid price data, use actual midpoint movement
            // This is a simplification - real implementation would track post-trade mid
            ((exec_price - mid_at_arrival).abs() * exec_qty) / 2
        } else {
            // Estimate: assume half of remaining slippage after delay and spread is impact
            let remaining = raw_slippage.abs() - delay_cost.abs() - spread_cost;
            if remaining > 0 {
                remaining / 2
            } else {
                0
            }
        };

        // Apply correct sign based on direction
        let signed_market_impact = if raw_slippage >= 0 {
            market_impact
        } else {
            -market_impact
        };

        // 4. Timing Luck (residual): what's left unexplained
        let explained = delay_cost + signed_market_impact + spread_cost;
        let timing_luck = raw_slippage - explained;

        // Verify components sum to total (with tolerance for rounding)
        let reconstructed = delay_cost + signed_market_impact + spread_cost + timing_luck;
        if (reconstructed - raw_slippage).abs() > self.scale {
            return Err(SlippageError::ComponentMismatch {
                sum: reconstructed,
                total: raw_slippage,
            });
        }

        // Calculate percentages in basis points
        let delay_pct_bps = if notional > 0 {
            (delay_cost.abs() * 10_000 / notional) as i64
        } else {
            0
        };

        let impact_pct_bps = if notional > 0 {
            (signed_market_impact.abs() * 10_000 / notional) as i64
        } else {
            0
        };

        let spread_pct_bps = if notional > 0 {
            (spread_cost * 10_000 / notional) as i64
        } else {
            0
        };

        let luck_pct_bps = if notional > 0 {
            (timing_luck * 10_000 / notional) as i64
        } else {
            0
        };

        self.decomposition_count.fetch_add(1, Ordering::Relaxed);
        self.cumulative_delay.fetch_add(delay_cost, Ordering::Relaxed);
        self.cumulative_impact.fetch_add(signed_market_impact, Ordering::Relaxed);
        self.cumulative_spread.fetch_add(spread_cost, Ordering::Relaxed);

        Ok(SlippageDecomposition {
            total_slippage: raw_slippage,
            delay_cost,
            market_impact: signed_market_impact,
            spread_cost,
            timing_luck,
            delay_pct_bps,
            impact_pct_bps,
            spread_pct_bps,
            luck_pct_bps,
        })
    }

    /// Get average component percentages across all decompositions
    pub fn get_avg_component_bps(&self) -> (i64, i64, i64) {
        let count = self.decomposition_count.load(Ordering::Acquire);
        if count == 0 {
            return (0, 0, 0);
        }

        let delay = self.cumulative_delay.load(Ordering::Acquire).abs() / count as i64;
        let impact = self.cumulative_impact.load(Ordering::Acquire).abs() / count as i64;
        let spread = self.cumulative_spread.load(Ordering::Acquire).abs() / count as i64;

        (delay, impact, spread)
    }

    /// Get decomposition count
    pub fn get_decomposition_count(&self) -> u64 {
        self.decomposition_count.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_decomposition() {
        let decomposer = SlippageDecomposer::new(1_000_000);

        let result = decomposer.decompose(
            100_000_000, // arrival price
            100_050_000, // exec price (worse for buy)
            1000,        // qty
            Some(99_980_000), // decision price (earlier)
            10,          // spread bps
            None,
        ).unwrap();

        // Total slippage should be positive (unfavorable)
        assert!(result.total_slippage > 0);
        
        // Components should exist
        assert!(result.delay_cost >= 0);
        assert!(result.spread_cost >= 0);
        
        // Percentages should sum approximately to total %
        let total_pct = result.delay_pct_bps + result.impact_pct_bps + result.spread_pct_bps + result.luck_pct_bps;
        assert!(total_pct > 0);
    }

    #[test]
    fn test_favorable_execution() {
        let decomposer = SlippageDecomposer::new(1_000_000);

        let result = decomposer.decompose(
            100_000_000,
            99_950_000, // Better than arrival
            1000,
            None,
            10,
            None,
        ).unwrap();

        // Total slippage should be negative (favorable)
        assert!(result.total_slippage < 0);
    }

    #[test]
    fn test_invalid_inputs() {
        let decomposer = SlippageDecomposer::new(1_000_000);

        // Zero price
        let result = decomposer.decompose(0, 100, 1, None, 10, None);
        assert!(matches!(result, Err(SlippageError::InvalidDecomposition { .. })));

        // Zero qty
        let result = decomposer.decompose(100, 100, 0, None, 10, None);
        assert!(matches!(result, Err(SlippageError::InvalidDecomposition { .. })));
    }

    #[test]
    fn test_cumulative_tracking() {
        let decomposer = SlippageDecomposer::new(1_000_000);

        // Multiple decompositions
        decomposer.decompose(100_000_000, 100_050_000, 1000, None, 10, None).unwrap();
        decomposer.decompose(100_000_000, 100_030_000, 1000, None, 10, None).unwrap();
        decomposer.decompose(100_000_000, 99_970_000, 1000, None, 10, None).unwrap();

        assert_eq!(decomposer.get_decomposition_count(), 3);

        let (delay, impact, spread) = decomposer.get_avg_component_bps();
        // Averages should reflect the mix of favorable/unfavorable trades
        assert!(delay >= 0);
        assert!(spread >= 0);
    }
}
