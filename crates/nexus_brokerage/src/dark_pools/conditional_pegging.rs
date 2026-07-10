//! Conditional Pegging Engine for Dark Pool Execution
//! 
//! Implements dynamic price calculation for pegged orders in dark pools,
//! ensuring strict price improvement over lit market NBBO.

use std::sync::atomic::{AtomicI64, AtomicBool, Ordering};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PeggingError {
    #[error("Invalid peg offset: {reason}")]
    InvalidOffset { reason: String },
    #[error("NBBO data stale: age_ms={age_ms} > max_age_ms={max_age_ms}")]
    StaleNBBO { age_ms: u64, max_age_ms: u64 },
    #[error("Crossed market: bid {bid} >= ask {ask}")]
    CrossedMarket { bid: i64, ask: i64 },
    #[error("Price improvement not achievable with current parameters")]
    ImprovementNotAchievable,
    #[error("Peg price outside acceptable range")]
    PriceOutOfRange,
}

/// National Best Bid and Offer snapshot
#[derive(Debug, Clone)]
pub struct NBBO {
    pub best_bid: i64,      // Fixed-point
    pub best_ask: i64,      // Fixed-point
    pub bid_size: i64,
    pub ask_size: i64,
    pub timestamp_ns: u64,  // Nanoseconds since epoch
}

impl NBBO {
    pub fn new(best_bid: i64, best_ask: i64, bid_size: i64, ask_size: i64) -> Self {
        Self {
            best_bid,
            best_ask,
            bid_size,
            ask_size,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
        }
    }

    /// Calculate midpoint
    pub fn midpoint(&self) -> i64 {
        self.best_bid.saturating_add(self.best_ask) / 2
    }

    /// Calculate spread
    pub fn spread(&self) -> i64 {
        self.best_ask.saturating_sub(self.best_bid)
    }

    /// Check if NBBO is valid (not crossed)
    pub fn is_valid(&self) -> bool {
        self.best_bid < self.best_ask && self.best_bid > 0 && self.best_ask > 0
    }

    /// Check if NBBO data is fresh enough
    pub fn is_fresh(&self, max_age_ms: u64) -> bool {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        let age_ns = now_ns.saturating_sub(self.timestamp_ns);
        let age_ms = age_ns / 1_000_000;
        
        age_ms <= max_age_ms
    }

    /// Get age in milliseconds
    pub fn age_ms(&self) -> u64 {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        let age_ns = now_ns.saturating_sub(self.timestamp_ns);
        age_ns / 1_000_000
    }
}

/// Peg order types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PegType {
    /// Peg to midpoint of NBBO
    Midpoint,
    /// Peg to primary side (bid for buys, ask for sells)
    Primary,
    /// Peg to opposite side (ask for buys, bid for sells)
    Opposite,
    /// Fixed price (no pegging)
    Fixed,
}

/// Side of the market
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Conditional peg order configuration
#[derive(Debug, Clone)]
pub struct PegOrderConfig {
    pub peg_type: PegType,
    pub side: OrderSide,
    pub quantity: i64,
    pub peg_offset: i64,        // Fixed-point offset from peg reference
    pub limit_price: Option<i64>, // Optional price limit
    pub min_improvement: i64,   // Minimum improvement over lit market
    pub max_age_ms: u64,        // Maximum acceptable NBBO age
}

/// Calculated peg price with validation
#[derive(Debug, Clone)]
pub struct PegPriceResult {
    pub peg_price: i64,
    pub reference_price: i64,
    pub improvement_over_lit: i64,
    pub meets_limit: bool,
    pub meets_improvement: bool,
}

/// Conditional Pegging Engine
pub struct ConditionalPegEngine {
    current_nbbo: dashmap::DashMap<u32, NBBO>, // asset_id -> NBBO
    default_max_age_ms: AtomicU64,
    enabled: AtomicBool,
}

impl ConditionalPegEngine {
    pub fn new() -> Self {
        Self {
            current_nbbo: dashmap::DashMap::new(),
            default_max_age_ms: AtomicU64::new(100), // 100ms default
            enabled: AtomicBool::new(true),
        }
    }

    /// Update NBBO for an asset
    pub fn update_nbbo(&self, asset_id: u32, nbbo: NBBO) -> Result<(), PeggingError> {
        if !nbbo.is_valid() {
            return Err(PeggingError::CrossedMarket {
                bid: nbbo.best_bid,
                ask: nbbo.best_ask,
            });
        }

        self.current_nbbo.insert(asset_id, nbbo);
        Ok(())
    }

    /// Get current NBBO for an asset
    pub fn get_nbbo(&self, asset_id: u32) -> Option<NBBO> {
        self.current_nbbo.get(&asset_id).map(|entry| entry.value().clone())
    }

    /// Calculate peg price for a buy order
    pub fn calculate_buy_peg_price(
        &self,
        asset_id: u32,
        config: &PegOrderConfig,
    ) -> Result<PegPriceResult, PeggingError> {
        let nbbo = self.get_nbbo(asset_id).ok_or_else(|| PeggingError::StaleNBBO {
            age_ms: u64::MAX,
            max_age_ms: config.max_age_ms,
        })?;

        // Check NBBO freshness
        if !nbbo.is_fresh(config.max_age_ms) {
            return Err(PeggingError::StaleNBBO {
                age_ms: nbbo.age_ms(),
                max_age_ms: config.max_age_ms,
            });
        }

        // Calculate reference price based on peg type
        let reference_price = match config.peg_type {
            PegType::Midpoint => nbbo.midpoint(),
            PegType::Primary => nbbo.best_bid,
            PegType::Opposite => nbbo.best_ask,
            PegType::Fixed => config.limit_price.unwrap_or(nbbo.midpoint()),
        };

        // Apply offset (negative for better price on buys)
        let mut peg_price = reference_price.saturating_add(config.peg_offset);

        // For buys: ensure peg price is below lit ask
        let improvement = nbbo.best_ask.saturating_sub(peg_price);
        let meets_improvement = improvement >= config.min_improvement;

        // Adjust price if improvement requirement not met
        if !meets_improvement {
            peg_price = nbbo.best_ask.saturating_sub(config.min_improvement);
            if peg_price <= nbbo.best_bid {
                return Err(PeggingError::ImprovementNotAchievable);
            }
        }

        // Check against limit price if provided
        let meets_limit = config.limit_price
            .map(|limit| peg_price <= limit)
            .unwrap_or(true);

        // Final validation: peg price must be between bid and ask
        if peg_price <= nbbo.best_bid || peg_price >= nbbo.best_ask {
            return Err(PeggingError::PriceOutOfRange);
        }

        Ok(PegPriceResult {
            peg_price,
            reference_price,
            improvement_over_lit: improvement,
            meets_limit,
            meets_improvement,
        })
    }

    /// Calculate peg price for a sell order
    pub fn calculate_sell_peg_price(
        &self,
        asset_id: u32,
        config: &PegOrderConfig,
    ) -> Result<PegPriceResult, PeggingError> {
        let nbbo = self.get_nbbo(asset_id).ok_or_else(|| PeggingError::StaleNBBO {
            age_ms: u64::MAX,
            max_age_ms: config.max_age_ms,
        })?;

        // Check NBBO freshness
        if !nbbo.is_fresh(config.max_age_ms) {
            return Err(PeggingError::StaleNBBO {
                age_ms: nbbo.age_ms(),
                max_age_ms: config.max_age_ms,
            });
        }

        // Calculate reference price based on peg type
        let reference_price = match config.peg_type {
            PegType::Midpoint => nbbo.midpoint(),
            PegType::Primary => nbbo.best_ask,
            PegType::Opposite => nbbo.best_bid,
            PegType::Fixed => config.limit_price.unwrap_or(nbbo.midpoint()),
        };

        // Apply offset (positive for better price on sells)
        let mut peg_price = reference_price.saturating_add(config.peg_offset);

        // For sells: ensure peg price is above lit bid
        let improvement = peg_price.saturating_sub(nbbo.best_bid);
        let meets_improvement = improvement >= config.min_improvement;

        // Adjust price if improvement requirement not met
        if !meets_improvement {
            peg_price = nbbo.best_bid.saturating_add(config.min_improvement);
            if peg_price >= nbbo.best_ask {
                return Err(PeggingError::ImprovementNotAchievable);
            }
        }

        // Check against limit price if provided
        let meets_limit = config.limit_price
            .map(|limit| peg_price >= limit)
            .unwrap_or(true);

        // Final validation: peg price must be between bid and ask
        if peg_price <= nbbo.best_bid || peg_price >= nbbo.best_ask {
            return Err(PeggingError::PriceOutOfRange);
        }

        Ok(PegPriceResult {
            peg_price,
            reference_price,
            improvement_over_lit: improvement,
            meets_limit,
            meets_improvement,
        })
    }

    /// Calculate peg price for either side
    pub fn calculate_peg_price(
        &self,
        asset_id: u32,
        config: &PegOrderConfig,
    ) -> Result<PegPriceResult, PeggingError> {
        match config.side {
            OrderSide::Buy => self.calculate_buy_peg_price(asset_id, config),
            OrderSide::Sell => self.calculate_sell_peg_price(asset_id, config),
        }
    }

    /// Enable/disable the peg engine
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    /// Check if engine is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Set default maximum NBBO age
    pub fn set_max_nbbo_age(&self, max_age_ms: u64) {
        self.default_max_age_ms.store(max_age_ms, Ordering::Release);
    }

    /// Get default maximum NBBO age
    pub fn get_max_nbbo_age(&self) -> u64 {
        self.default_max_age_ms.load(Ordering::Acquire)
    }
}

impl Default for ConditionalPegEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nbbo_midpoint() {
        let nbbo = NBBO::new(100_000_000, 100_000_100, 1000, 1000);
        assert_eq!(nbbo.midpoint(), 100_000_050);
        assert_eq!(nbbo.spread(), 100);
        assert!(nbbo.is_valid());
    }

    #[test]
    fn test_crossed_market_detection() {
        let nbbo = NBBO::new(100_000_100, 100_000_000, 1000, 1000); // Bid > Ask
        assert!(!nbbo.is_valid());
    }

    #[test]
    fn test_buy_peg_midpoint() {
        let engine = ConditionalPegEngine::new();
        let nbbo = NBBO::new(100_000_000, 100_000_100, 1000, 1000);
        engine.update_nbbo(1, nbbo).unwrap();

        let config = PegOrderConfig {
            peg_type: PegType::Midpoint,
            side: OrderSide::Buy,
            quantity: 1000,
            peg_offset: -10, // Want 10 units below midpoint
            limit_price: None,
            min_improvement: 5,
            max_age_ms: 100,
        };

        let result = engine.calculate_buy_peg_price(1, &config).unwrap();
        
        // Midpoint is 100_000_050, offset -10 = 100_000_040
        // Should be below ask (100_000_100) with at least 5 improvement
        assert!(result.peg_price < 100_000_100);
        assert!(result.improvement_over_lit >= 5);
    }

    #[test]
    fn test_sell_peg_midpoint() {
        let engine = ConditionalPegEngine::new();
        let nbbo = NBBO::new(100_000_000, 100_000_100, 1000, 1000);
        engine.update_nbbo(1, nbbo).unwrap();

        let config = PegOrderConfig {
            peg_type: PegType::Midpoint,
            side: OrderSide::Sell,
            quantity: 1000,
            peg_offset: 10, // Want 10 units above midpoint
            limit_price: None,
            min_improvement: 5,
            max_age_ms: 100,
        };

        let result = engine.calculate_sell_peg_price(1, &config).unwrap();
        
        // Midpoint is 100_000_050, offset +10 = 100_000_060
        // Should be above bid (100_000_000) with at least 5 improvement
        assert!(result.peg_price > 100_000_000);
        assert!(result.improvement_over_lit >= 5);
    }

    #[test]
    fn test_stale_nbbo_rejection() {
        let engine = ConditionalPegEngine::new();
        
        // Create NBBO with old timestamp
        let mut nbbo = NBBO::new(100_000_000, 100_000_100, 1000, 1000);
        nbbo.timestamp_ns = 0; // Very old
        
        engine.update_nbbo(1, nbbo).unwrap();

        let config = PegOrderConfig {
            peg_type: PegType::Midpoint,
            side: OrderSide::Buy,
            quantity: 1000,
            peg_offset: 0,
            limit_price: None,
            min_improvement: 5,
            max_age_ms: 100,
        };

        let result = engine.calculate_buy_peg_price(1, &config);
        assert!(matches!(result, Err(PeggingError::StaleNBBO { .. })));
    }

    #[test]
    fn test_improvement_not_achievable() {
        let engine = ConditionalPegEngine::new();
        
        // Very tight spread
        let nbbo = NBBO::new(100_000_000, 100_000_001, 1000, 1000);
        engine.update_nbbo(1, nbbo).unwrap();

        let config = PegOrderConfig {
            peg_type: PegType::Midpoint,
            side: OrderSide::Buy,
            quantity: 1000,
            peg_offset: 0,
            limit_price: None,
            min_improvement: 10, // Impossible with 1-unit spread
            max_age_ms: 100,
        };

        let result = engine.calculate_buy_peg_price(1, &config);
        assert!(matches!(result, Err(PeggingError::ImprovementNotAchievable)));
    }
}
