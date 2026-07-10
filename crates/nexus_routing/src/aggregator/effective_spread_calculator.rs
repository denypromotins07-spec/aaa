//! Effective Spread Calculator - Real-time true cost of liquidity
//! 
//! Factors in maker/taker fees, routing latency, and tail-risk metrics.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SpreadError {
    #[error("Invalid spread calculation: {reason}")]
    InvalidCalculation { reason: String },
    #[error("Missing venue data for venue {venue_id}")]
    MissingVenueData { venue_id: u32 },
}

/// Effective spread result with all cost components
#[derive(Debug, Clone)]
pub struct EffectiveSpreadResult {
    pub raw_spread: i64,           // Raw bid-ask spread (fixed-point)
    pub fee_adjusted_spread: i64,  // After maker/taker fees
    pub latency_cost: i64,         // Estimated slippage from routing latency
    pub tail_risk_premium: i64,    // Additional cost from Stage 11 tail risk
    pub total_effective_spread: i64, // Sum of all costs
}

/// Effective Spread Calculator
pub struct EffectiveSpreadCalculator {
    /// Fee data per venue: venue_id -> (maker_fee_bps, taker_fee_bps)
    venue_fees: dashmap::DashMap<u32, (i64, i64)>,
    /// Latency estimates per venue in microseconds
    venue_latencies: dashmap::DashMap<u32, u32>,
    /// Volatility estimate for slippage calculation (annualized bps)
    volatility_bps: AtomicU64,
    calculation_count: AtomicU64,
}

impl EffectiveSpreadCalculator {
    pub fn new() -> Self {
        Self {
            venue_fees: dashmap::DashMap::new(),
            venue_latencies: dashmap::DashMap::new(),
            volatility_bps: AtomicU64::new(100), // Default 1% annualized vol
            calculation_count: AtomicU64::new(0),
        }
    }

    /// Set fee structure for a venue
    pub fn set_venue_fees(&self, venue_id: u32, maker_bps: i64, taker_bps: i64) {
        self.venue_fees.insert(venue_id, (maker_bps, taker_bps));
    }

    /// Set latency estimate for a venue (microseconds)
    pub fn set_venue_latency(&self, venue_id: u32, latency_us: u32) {
        self.venue_latencies.insert(venue_id, latency_us);
    }

    /// Set global volatility estimate (annualized basis points)
    pub fn set_volatility(&self, vol_bps: u64) {
        self.volatility_bps.store(vol_bps, Ordering::Release);
    }

    /// Calculate effective spread for a taker order
    pub fn calc_taker_spread(
        &self,
        venue_id: u32,
        best_bid: i64,
        best_ask: i64,
        tail_risk_premium: i64,
    ) -> Result<EffectiveSpreadResult, SpreadError> {
        if best_bid >= best_ask || best_bid <= 0 || best_ask <= 0 {
            return Err(SpreadError::InvalidCalculation {
                reason: format!("Invalid quotes: bid={}, ask={}", best_bid, best_ask),
            });
        }

        let raw_spread = best_ask - best_bid;
        let mid_price = (best_bid + best_ask) / 2;

        // Get venue fees
        let (_, taker_fee_bps) = self.venue_fees.get(&venue_id)
            .map(|entry| *entry.value())
            .unwrap_or((0, 20)); // Default 20 bps taker fee

        // Fee cost = mid_price * taker_fee_bps / 10000
        let fee_cost = mid_price * taker_fee_bps / 10_000;

        // Fee-adjusted spread includes round-trip fees
        let fee_adjusted_spread = raw_spread + 2 * fee_cost;

        // Latency cost estimation based on volatility
        let latency_us = self.venue_latencies.get(&venue_id)
            .map(|entry| *entry.value())
            .unwrap_or(1000); // Default 1ms

        let vol_bps = self.volatility_bps.load(Ordering::Acquire) as f64;
        // Latency cost = vol * sqrt(latency_seconds) * mid_price
        // Simplified: vol_bps/10000 * sqrt(latency_us/1e6) * mid_price
        let latency_factor = (latency_us as f64 / 1_000_000.0).sqrt();
        let latency_cost = (vol_bps / 10_000.0 * latency_factor * mid_price as f64) as i64;

        // Total effective spread
        let total_effective_spread = fee_adjusted_spread
            .saturating_add(2 * latency_cost)  // Round-trip latency
            .saturating_add(2 * tail_risk_premium);

        self.calculation_count.fetch_add(1, Ordering::Relaxed);

        Ok(EffectiveSpreadResult {
            raw_spread,
            fee_adjusted_spread,
            latency_cost,
            tail_risk_premium,
            total_effective_spread,
        })
    }

    /// Calculate effective spread for a maker order (earning rebates)
    pub fn calc_maker_spread(
        &self,
        venue_id: u32,
        best_bid: i64,
        best_ask: i64,
    ) -> Result<EffectiveSpreadResult, SpreadError> {
        if best_bid >= best_ask || best_bid <= 0 || best_ask <= 0 {
            return Err(SpreadError::InvalidCalculation {
                reason: format!("Invalid quotes: bid={}, ask={}", best_bid, best_ask),
            });
        }

        let raw_spread = best_ask - best_bid;
        let mid_price = (best_bid + best_ask) / 2;

        // Get venue maker rebate
        let (maker_fee_bps, _) = self.venue_fees.get(&venue_id)
            .map(|entry| *entry.value())
            .unwrap_or((0, 20));

        // Maker earns rebate (negative fee)
        let rebate = mid_price * maker_fee_bps / 10_000;

        // Effective spread reduced by rebates
        let fee_adjusted_spread = raw_spread.saturating_sub(2 * rebate);

        Ok(EffectiveSpreadResult {
            raw_spread,
            fee_adjusted_spread,
            latency_cost: 0,
            tail_risk_premium: 0,
            total_effective_spread: fee_adjusted_spread,
        })
    }

    /// Compare venues and return best venue for execution
    pub fn compare_venues(
        &self,
        venues: &[(u32, i64, i64)], // (venue_id, best_bid, best_ask)
        tail_risk_premium: i64,
    ) -> Option<(u32, EffectiveSpreadResult)> {
        let mut best: Option<(u32, EffectiveSpreadResult)> = None;

        for &(venue_id, bid, ask) in venues {
            if let Ok(result) = self.calc_taker_spread(venue_id, bid, ask, tail_risk_premium) {
                if best.is_none() || result.total_effective_spread < best.as_ref().unwrap().1.total_effective_spread {
                    best = Some((venue_id, result));
                }
            }
        }

        best
    }

    /// Get calculation count
    pub fn get_calculation_count(&self) -> u64 {
        self.calculation_count.load(Ordering::Acquire)
    }
}

impl Default for EffectiveSpreadCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_spread_calculation() {
        let calc = EffectiveSpreadCalculator::new();
        calc.set_venue_fees(1, 5, 20); // 5 bps maker, 20 bps taker
        calc.set_venue_latency(1, 500); // 500us latency

        let result = calc.calc_taker_spread(1, 100_000_000, 100_000_100, 10)
            .unwrap();

        assert_eq!(result.raw_spread, 100);
        assert!(result.fee_adjusted_spread > result.raw_spread);
        assert!(result.latency_cost >= 0);
    }

    #[test]
    fn test_invalid_quotes() {
        let calc = EffectiveSpreadCalculator::new();
        
        // Crossed market
        let result = calc.calc_taker_spread(1, 100_000_100, 100_000_000, 0);
        assert!(matches!(result, Err(SpreadError::InvalidCalculation { .. })));
        
        // Negative prices
        let result = calc.calc_taker_spread(1, -100, 100, 0);
        assert!(matches!(result, Err(SpreadError::InvalidCalculation { .. })));
    }

    #[test]
    fn test_maker_rebate() {
        let calc = EffectiveSpreadCalculator::new();
        calc.set_venue_fees(1, 10, 20); // 10 bps maker rebate

        let result = calc.calc_maker_spread(1, 100_000_000, 100_000_100)
            .unwrap();

        // Maker spread should be less than raw spread due to rebates
        assert!(result.fee_adjusted_spread <= result.raw_spread);
    }

    #[test]
    fn test_venue_comparison() {
        let calc = EffectiveSpreadCalculator::new();
        calc.set_venue_fees(1, 5, 10);
        calc.set_venue_fees(2, 5, 30); // Higher taker fee

        let venues = vec![
            (1, 100_000_000, 100_000_100),
            (2, 100_000_000, 100_000_100),
        ];

        let best = calc.compare_venues(&venues, 0);
        assert!(best.is_some());
        assert_eq!(best.unwrap().0, 1); // Venue 1 has lower fees
    }
}
