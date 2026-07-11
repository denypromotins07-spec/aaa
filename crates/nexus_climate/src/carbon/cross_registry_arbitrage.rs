//! Cross-Registry Carbon Arbitrage Engine
//! Tracks price discrepancies across EU ETS, California Cap-and-Trade, and VCM markets

use alloc::vec::Vec;
use core::fmt;

/// Error types for carbon arbitrage operations
#[derive(Debug, Clone, PartialEq)]
pub enum CarbonArbError {
    InvalidPrice,
    RegistryNotFound,
    VerificationFailed,
    InsufficientLiquidity,
    SettlementError,
}

impl fmt::Display for CarbonArbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrice => write!(f, "Invalid price value"),
            Self::RegistryNotFound => write!(f, "Carbon registry not found"),
            Self::VerificationFailed => write!(f, "Offset verification failed"),
            Self::InsufficientLiquidity => write!(f, "Insufficient market liquidity"),
            Self::SettlementError => write!(f, "Settlement execution error"),
        }
    }
}

/// Supported carbon registries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarbonRegistry {
    EU_ETS,              // European Union Emissions Trading System
    California_CAP,      // California Cap-and-Trade
    UK_ETS,              // UK Emissions Trading Scheme
    RGGI,                // Regional Greenhouse Gas Initiative
    Verra_VCS,           // Verified Carbon Standard (Voluntary)
    GoldStandard,        // Gold Standard (Voluntary)
    AmericanCarbon,      // American Carbon Registry
    ClimateActionReserve,// Climate Action Reserve
}

/// Carbon credit representation
#[derive(Debug, Clone)]
pub struct CarbonCredit {
    pub registry: CarbonRegistry,
    /// Credit ID in registry
    pub credit_id: u64,
    /// Vintage year
    pub vintage: u16,
    /// Project type (forestry, renewable, etc.)
    pub project_type: &'static str,
    /// Geographic location (lat, lon)
    pub location: (f64, f64),
    /// Volume in tonnes CO2e
    pub volume_tonnes: f64,
    /// Current price per tonne
    pub price_per_tonne: f64,
    /// Verification status
    pub verified: bool,
}

/// Market quote for a registry
#[derive(Debug, Clone)]
pub struct MarketQuote {
    pub registry: CarbonRegistry,
    pub bid: f64,
    pub ask: f64,
    pub volume_available: f64,
    pub timestamp_us: u64,
}

/// Arbitrage opportunity detection
#[derive(Debug, Clone)]
pub struct ArbitrageOpportunity {
    pub buy_registry: CarbonRegistry,
    pub sell_registry: CarbonRegistry,
    pub buy_price: f64,
    pub sell_price: f64,
    pub spread_bps: f64,
    pub max_volume: f64,
    pub expected_profit: f64,
    pub confidence: f64,
}

/// Cross-registry arbitrage engine
pub struct CrossRegistryArbitrageEngine {
    /// Current quotes per registry
    quotes: alloc::collections::BTreeMap<CarbonRegistry, MarketQuote>,
    /// Transaction costs per registry pair (basis points)
    transaction_costs: alloc::collections::BTreeMap<(CarbonRegistry, CarbonRegistry), f64>,
    /// Minimum spread threshold for execution (bps)
    min_spread_bps: f64,
    /// Position limits per registry
    position_limits: alloc::collections::BTreeMap<CarbonRegistry, f64>,
    /// Current positions
    positions: alloc::collections::BTreeMap<CarbonRegistry, f64>,
}

impl CrossRegistryArbitrageEngine {
    /// Create new arbitrage engine
    pub fn new(min_spread_bps: f64) -> Self {
        let mut transaction_costs = alloc::collections::BTreeMap::new();
        let mut position_limits = alloc::collections::BTreeMap::new();
        let mut positions = alloc::collections::BTreeMap::new();

        // Initialize default transaction costs
        let registries = [
            CarbonRegistry::EU_ETS,
            CarbonRegistry::California_CAP,
            CarbonRegistry::UK_ETS,
            CarbonRegistry::RGGI,
            CarbonRegistry::Verra_VCS,
            CarbonRegistry::GoldStandard,
            CarbonRegistry::AmericanCarbon,
            CarbonRegistry::ClimateActionReserve,
        ];

        for &reg in &registries {
            position_limits.insert(reg, 1_000_000.0); // 1M tonne limit
            positions.insert(reg, 0.0);

            for &reg2 in &registries {
                if reg != reg2 {
                    // Higher costs for voluntary <-> compliance arb
                    let cost = if Self::is_compliance(reg) && Self::is_compliance(reg2) {
                        5.0 // 5 bps for compliance-compliance
                    } else if !Self::is_compliance(reg) && !Self::is_compliance(reg2) {
                        10.0 // 10 bps for voluntary-voluntary
                    } else {
                        25.0 // 25 bps for cross-type (verification overhead)
                    };
                    transaction_costs.insert((reg, reg2), cost);
                }
            }
        }

        Self {
            quotes: alloc::collections::BTreeMap::new(),
            transaction_costs,
            min_spread_bps: min_spread_bps.max(1.0),
            position_limits,
            positions,
        }
    }

    fn is_compliance(reg: CarbonRegistry) -> bool {
        matches!(
            reg,
            CarbonRegistry::EU_ETS
                | CarbonRegistry::California_CAP
                | CarbonRegistry::UK_ETS
                | CarbonRegistry::RGGI
        )
    }

    /// Update market quote
    pub fn update_quote(&mut self, quote: MarketQuote) -> Result<(), CarbonArbError> {
        if quote.bid <= 0.0 || quote.ask <= 0.0 || quote.bid > quote.ask {
            return Err(CarbonArbError::InvalidPrice);
        }
        if quote.volume_available < 0.0 {
            return Err(CarbonArbError::InsufficientLiquidity);
        }

        self.quotes.insert(quote.registry, quote);
        Ok(())
    }

    /// Scan for arbitrage opportunities
    pub fn scan_opportunities(&self) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();

        let registries: Vec<_> = self.quotes.keys().copied().collect();

        for &buy_reg in &registries {
            for &sell_reg in &registries {
                if buy_reg == sell_reg {
                    continue;
                }

                if let (Some(buy_quote), Some(sell_quote)) =
                    (self.quotes.get(&buy_reg), self.quotes.get(&sell_reg))
                {
                    // Calculate gross spread
                    let gross_spread_bps = 
                        ((sell_quote.bid - buy_quote.ask) / buy_quote.ask) * 10000.0;

                    // Get transaction cost
                    let tx_cost = self.transaction_costs
                        .get(&(buy_reg, sell_reg))
                        .copied()
                        .unwrap_or(25.0);

                    // Net spread after costs
                    let net_spread_bps = gross_spread_bps - tx_cost;

                    if net_spread_bps >= self.min_spread_bps {
                        // Determine max executable volume
                        let available_buy = buy_quote.volume_available;
                        let available_sell = sell_quote.volume_available;
                        let position_headroom_buy = self.position_limits
                            .get(&buy_reg)
                            .copied()
                            .unwrap_or(0.0)
                            + self.positions.get(&buy_reg).copied().unwrap_or(0.0);
                        let position_headroom_sell = self.position_limits
                            .get(&sell_reg)
                            .copied()
                            .unwrap_or(0.0)
                            - self.positions.get(&sell_reg).copied().unwrap_or(0.0);

                        let max_volume = available_buy
                            .min(available_sell)
                            .min(position_headroom_buy)
                            .min(position_headroom_sell)
                            .max(0.0);

                        if max_volume > 0.0 {
                            let expected_profit = (net_spread_bps / 10000.0) 
                                * buy_quote.ask * max_volume;

                            // Confidence based on liquidity and spread stability
                            let liquidity_score = (available_buy.min(available_sell) / 1000.0).min(1.0);
                            let spread_score = (net_spread_bps / 100.0).min(1.0);
                            let confidence = (liquidity_score * 0.5 + spread_score * 0.5).clamp(0.0, 1.0);

                            opportunities.push(ArbitrageOpportunity {
                                buy_registry: buy_reg,
                                sell_registry: sell_reg,
                                buy_price: buy_quote.ask,
                                sell_price: sell_quote.bid,
                                spread_bps: net_spread_bps,
                                max_volume,
                                expected_profit,
                                confidence,
                            });
                        }
                    }
                }
            }
        }

        // Sort by expected profit descending
        opportunities.sort_by(|a, b| b.expected_profit.partial_cmp(&a.expected_profit).unwrap_or(core::cmp::Ordering::Equal));

        opportunities
    }

    /// Execute arbitrage (simulation mode)
    pub fn execute_arbitrage(&mut self, opp: &ArbitrageOpportunity) -> Result<f64, CarbonArbError> {
        if opp.max_volume <= 0.0 {
            return Err(CarbonArbError::InsufficientLiquidity);
        }

        // Check position limits
        let current_buy_pos = self.positions.get(&opp.buy_registry).copied().unwrap_or(0.0);
        let current_sell_pos = self.positions.get(&opp.sell_registry).copied().unwrap_or(0.0);

        let buy_limit = self.position_limits.get(&opp.buy_registry).copied().unwrap_or(0.0);
        let sell_limit = self.position_limits.get(&opp.sell_registry).copied().unwrap_or(0.0);

        if current_buy_pos + opp.max_volume > buy_limit {
            return Err(CarbonArbError::InsufficientLiquidity);
        }
        if current_sell_pos - opp.max_volume < -sell_limit {
            return Err(CarbonArbError::InsufficientLiquidity);
        }

        // Update positions
        *self.positions.entry(opp.buy_registry).or_insert(0.0) += opp.max_volume;
        *self.positions.entry(opp.sell_registry).or_insert(0.0) -= opp.max_volume;

        Ok(opp.expected_profit)
    }

    /// Get current position for a registry
    pub fn get_position(&self, registry: CarbonRegistry) -> f64 {
        self.positions.get(&registry).copied().unwrap_or(0.0)
    }

    /// Get total PnL
    pub fn total_pnl(&self) -> f64 {
        let mut pnl = 0.0;
        for (&reg, &pos) in &self.positions {
            if let Some(quote) = self.quotes.get(&reg) {
                pnl += pos * quote.bid;
            }
        }
        pnl
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arbitrage_detection() {
        let mut engine = CrossRegistryArbitrageEngine::new(10.0);

        // EU ETS: bid 80, ask 81
        let eu_quote = MarketQuote {
            registry: CarbonRegistry::EU_ETS,
            bid: 80.0,
            ask: 81.0,
            volume_available: 10000.0,
            timestamp_us: 1000,
        };

        // California: bid 85, ask 86 (higher price)
        let ca_quote = MarketQuote {
            registry: CarbonRegistry::California_CAP,
            bid: 85.0,
            ask: 86.0,
            volume_available: 10000.0,
            timestamp_us: 1000,
        };

        engine.update_quote(eu_quote).unwrap();
        engine.update_quote(ca_quote).unwrap();

        let opportunities = engine.scan_opportunities();
        
        // Should find opportunity to buy EU, sell California
        assert!(!opportunities.is_empty());
        let best = &opportunities[0];
        assert_eq!(best.buy_registry, CarbonRegistry::EU_ETS);
        assert_eq!(best.sell_registry, CarbonRegistry::California_CAP);
    }
}
