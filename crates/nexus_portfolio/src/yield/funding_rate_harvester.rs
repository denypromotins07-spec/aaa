//! Funding Rate Harvester - Tracks and captures perpetual funding rate yield.
//! 
//! Uses scaled integer math to avoid precision loss in APY calculations.

use std::sync::atomic::{AtomicI128, AtomicBool, Ordering};

/// Scale factor for funding rates (1e12 for nanopercent precision)
const FUNDING_SCALE: i128 = 1_000_000_000_000;

/// Represents a funding rate update from an exchange
#[derive(Debug, Clone)]
pub struct FundingRateUpdate {
    pub exchange_id: u8,
    pub symbol: [u8; 16],
    /// Current funding rate (scaled by FUNDING_SCALE)
    pub funding_rate_scaled: i128,
    /// Predicted next funding rate (scaled)
    pub predicted_rate_scaled: i128,
    /// Time until next funding print (milliseconds)
    pub ms_until_funding: u64,
}

/// Annualized funding rate info
#[derive(Debug, Clone)]
pub struct AnnualizedFunding {
    /// APY in basis points (1 bp = 0.01%)
    pub apy_bps: i128,
    /// APY scaled by 1e18 for precise calculations
    pub apy_scaled: i128,
}

/// Thresholds
const MIN_APY_BPS: i128 = 1500; // 15% APY threshold for arb
const FUNDING_INTERVALS_PER_YEAR: i128 = 365 * 3; // 8-hour intervals

/// Lock-free funding rate tracker
pub struct FundingRateHarvester {
    /// Latest funding rate per exchange/symbol (simplified: single position for now)
    current_rate: AtomicI128,
    /// Flag indicating profitable arb opportunity
    arb_opportunity_active: AtomicBool,
    /// Accumulated funding yield (scaled)
    accumulated_yield: AtomicI128,
}

impl FundingRateHarvester {
    pub fn new() -> Self {
        Self {
            current_rate: AtomicI128::new(0),
            arb_opportunity_active: AtomicBool::new(false),
            accumulated_yield: AtomicI128::new(0),
        }
    }

    /// Process a funding rate update
    pub fn on_funding_update(&self, update: &FundingRateUpdate) -> Option<AnnualizedFunding> {
        self.current_rate.store(update.predicted_rate_scaled, Ordering::SeqCst);
        
        // Calculate annualized rate
        // APY = rate_per_interval * intervals_per_year * 100 (for percentage)
        // Using scaled math: (predicted_rate * FUNDING_INTERVALS_PER_YEAR * 1e18) / FUNDING_SCALE
        let rate = update.predicted_rate_scaled;
        
        // Check for overflow before multiplication
        let yearly_rate = rate.checked_mul(FUNDING_INTERVALS_PER_YEAR)?;
        let apy_scaled = yearly_rate.checked_mul(1_000_000_000_000_000_000i128 / FUNDING_SCALE)?;
        
        // Convert to basis points: apy_scaled / 1e14
        let apy_bps = apy_scaled / 100_000_000_000_000i128;
        
        let annualized = AnnualizedFunding {
            apy_bps,
            apy_scaled,
        };

        // Check if arb opportunity exists (> 15% APY)
        if apy_bps > MIN_APY_BPS {
            self.arb_opportunity_active.store(true, Ordering::Relaxed);
        } else {
            self.arb_opportunity_active.store(false, Ordering::Relaxed);
        }

        Some(annualized)
    }

    /// Record funding received
    pub fn record_funding_received(&self, amount_scaled: i128) {
        self.accumulated_yield.fetch_add(amount_scaled, Ordering::Relaxed);
    }

    /// Check if arb opportunity is active
    pub fn is_arb_opportunity_active(&self) -> bool {
        self.arb_opportunity_active.load(Ordering::Relaxed)
    }

    /// Get accumulated yield
    pub fn get_accumulated_yield(&self) -> i128 {
        self.accumulated_yield.load(Ordering::Relaxed)
    }

    /// Get current funding rate (scaled)
    pub fn get_current_rate(&self) -> i128 {
        self.current_rate.load(Ordering::Relaxed)
    }
}

impl Default for FundingRateHarvester {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_annualized_funding_calculation() {
        let harvester = FundingRateHarvester::new();
        
        // Simulate 0.01% per 8 hours = ~10.95% annually
        // 0.01% = 0.0001 = 100_000_000 (scaled by 1e12)
        let update = FundingRateUpdate {
            exchange_id: 1,
            symbol: [0; 16],
            funding_rate_scaled: 100_000_000,
            predicted_rate_scaled: 100_000_000,
            ms_until_funding: 1000,
        };
        
        let annualized = harvester.on_funding_update(&update).unwrap();
        
        // Expected: 0.0001 * 1095 * 100 = ~10.95% = 1095 bps
        assert!(annualized.apy_bps > 1000);
        assert!(!harvester.is_arb_opportunity_active()); // Below 15%
        
        // Now test high rate: 0.02% per 8h = ~21.9% annually
        let high_update = FundingRateUpdate {
            predicted_rate_scaled: 200_000_000,
            ..update
        };
        
        let annualized = harvester.on_funding_update(&high_update).unwrap();
        assert!(annualized.apy_bps > MIN_APY_BPS);
        assert!(harvester.is_arb_opportunity_active());
    }
}
