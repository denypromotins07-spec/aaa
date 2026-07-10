// STAGE 23: Computational Law, RegTech Compliance & Immutable Audit Trails
// Chapter 2: Computational Law & Pre-Trade Compliance
// File: crates/nexus_legal/src/compliance/mifid_tick_size.rs

//! MiFID II Tick Size Regime Compliance Module
//! Implements tick size validation, best execution proof, and transparency waivers.
//! Supports multiple tick size regimes based on liquidity and price bands.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dashmap::DashMap;

use crate::compliance::dag_rule_engine::{ComplianceState, ComplianceFlag, ComplianceResult};

/// Configuration for MiFID II compliance
#[derive(Debug, Clone)]
pub struct MifidConfig {
    /// Enable strict tick size enforcement
    pub enable_tick_size_check: bool,
    /// Enable best execution verification
    pub enable_best_exec_check: bool,
    /// Enable transparency waiver checks
    pub enable_transparency_check: bool,
    /// Reference data source (for tick size tables)
    pub reference_data_path: Option<String>,
}

impl Default for MifidConfig {
    fn default() -> Self {
        Self {
            enable_tick_size_check: true,
            enable_best_exec_check: true,
            enable_transparency_check: true,
            reference_data_path: None,
        }
    }
}

/// Tick size regime based on MiFID II RTS 11
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickSizeRegime {
    /// Standard regime - most liquid instruments
    Standard,
    /// Liquidity 1 - moderate liquidity
    Liquidity1,
    /// Liquidity 2 - lower liquidity
    Liquidity2,
    /// Custom regime for specific instruments
    Custom,
}

/// Price band for tick size determination
#[derive(Debug, Clone)]
pub struct PriceBand {
    pub min_price: u64,
    pub max_price: u64,
    pub tick_size: u64,
    pub regime: TickSizeRegime,
}

/// Tick size table for an instrument
#[derive(Debug, Clone)]
pub struct TickSizeTable {
    pub symbol: String,
    pub venue: String,
    pub regime: TickSizeRegime,
    pub bands: Vec<PriceBand>,
    /// Last update timestamp
    pub updated_at_ns: u64,
}

impl TickSizeTable {
    pub fn new(symbol: String, venue: String, regime: TickSizeRegime) -> Self {
        Self {
            symbol,
            venue,
            regime,
            bands: Vec::new(),
            updated_at_ns: 0,
        }
    }

    /// Add a price band to the table
    pub fn add_band(&mut self, min_price: u64, max_price: u64, tick_size: u64) {
        self.bands.push(PriceBand {
            min_price,
            max_price,
            tick_size,
            regime: self.regime,
        });
        // Keep bands sorted by min_price
        self.bands.sort_by_key(|b| b.min_price);
    }

    /// Get tick size for a given price
    pub fn get_tick_size(&self, price: u64) -> Option<u64> {
        self.bands
            .iter()
            .find(|b| price >= b.min_price && price <= b.max_price)
            .map(|b| b.tick_size)
    }

    /// Check if a price is compliant with tick size rules
    pub fn is_price_compliant(&self, price: u64) -> bool {
        if let Some(tick_size) = self.get_tick_size(price) {
            price % tick_size == 0
        } else {
            false // Price outside defined bands
        }
    }

    /// Round price to nearest valid tick
    pub fn round_to_tick(&self, price: u64) -> Option<u64> {
        self.get_tick_size(price).map(|tick_size| {
            (price / tick_size) * tick_size
        })
    }
}

/// Best execution venue comparison record
#[derive(Debug, Clone)]
pub struct VenueComparison {
    pub venue_id: String,
    pub price: u64,
    pub available_quantity: i64,
    pub timestamp_ns: u64,
    pub fees_bps: u32,
    pub latency_us: u32,
}

/// Best execution analysis result
#[derive(Debug, Clone)]
pub struct BestExecAnalysis {
    pub symbol: String,
    pub side: OrderSide,
    pub requested_quantity: i64,
    /// Best available price across venues
    pub best_price: u64,
    /// Venue offering best price
    pub best_venue: String,
    /// Price improvement vs reference (bps)
    pub price_improvement_bps: i32,
    /// Total cost including fees (in fixed point)
    pub total_cost: u64,
    /// Whether this execution meets best exec requirements
    pub is_compliant: bool,
    /// Analysis timestamp
    pub analyzed_at_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Transparency waiver types under MiFID II
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransparencyWaiver {
    /// Reference price waiver
    ReferencePrice,
    /// Large in scale waiver
    LargeInScale,
    /// Order management system waiver
    OmsWaiver,
    /// Negotiated trade waiver
    Negotiated,
    /// No waiver (full transparency required)
    None,
}

impl TransparencyWaiver {
    pub fn applies(&self, quantity: i64, threshold: i64) -> bool {
        match self {
            TransparencyWaiver::LargeInScale => quantity >= threshold,
            TransparencyWaiver::ReferencePrice => true, // Always applies if configured
            TransparencyWaiver::OmsWaiver => true,
            TransparencyWaiver::Negotiated => true,
            TransparencyWaiver::None => false,
        }
    }
}

/// MiFID II compliance engine
pub struct MifidEngine {
    config: MifidConfig,
    /// Tick size tables by symbol/venue
    tick_tables: DashMap<(String, String), TickSizeTable>,
    /// Recent venue prices for best exec analysis
    venue_prices: DashMap<String, Vec<VenueComparison>>,
    /// Waiver configurations by symbol
    waivers: DashMap<String, TransparencyWaiver>,
    /// Statistics
    total_tick_checks: AtomicU64,
    total_tick_violations: AtomicU64,
    total_best_exec_checks: AtomicU64,
}

impl MifidEngine {
    pub fn new() -> Self {
        Self::new_with_config(MifidConfig::default())
    }

    pub fn new_with_config(config: MifidConfig) -> Self {
        Self {
            config,
            tick_tables: DashMap::new(),
            venue_prices: DashMap::new(),
            waivers: DashMap::new(),
            total_tick_checks: AtomicU64::new(0),
            total_tick_violations: AtomicU64::new(0),
            total_best_exec_checks: AtomicU64::new(0),
        }
    }

    /// Register or update a tick size table
    pub fn register_tick_table(&self, table: TickSizeTable) {
        let key = (table.symbol.clone(), table.venue.clone());
        
        let current_time_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        
        let mut table = table;
        table.updated_at_ns = current_time_ns;
        
        self.tick_tables.insert(key, table);
    }

    /// Validate order price against tick size rules
    pub fn validate_tick_size(
        &self,
        symbol: &str,
        venue: &str,
        price: u64,
    ) -> ComplianceResult {
        let start = Instant::now();
        self.total_tick_checks.fetch_add(1, Ordering::Relaxed);

        if !self.config.enable_tick_size_check {
            return ComplianceResult::PASSED;
        }

        let key = (symbol.to_string(), venue.to_string());
        
        if let Some(table) = self.tick_tables.get(&key) {
            if table.is_price_compliant(price) {
                let elapsed = start.elapsed().as_nanos() as u64;
                ComplianceResult {
                    passed: true,
                    failed_rules: 0,
                    evaluation_time_ns: elapsed,
                }
            } else {
                self.total_tick_violations.fetch_add(1, Ordering::Relaxed);
                let elapsed = start.elapsed().as_nanos() as u64;
                ComplianceResult::failed(
                    ComplianceFlag::MifidTickSizeCompliant as u128,
                    elapsed,
                )
            }
        } else {
            // No tick table found - allow with warning (could be new instrument)
            let elapsed = start.elapsed().as_nanos() as u64;
            ComplianceResult {
                passed: true,
                failed_rules: 0,
                evaluation_time_ns: elapsed,
            }
        }
    }

    /// Record venue price for best execution analysis
    pub fn record_venue_price(&self, comparison: VenueComparison) {
        self.venue_prices
            .entry(comparison.symbol.clone())
            .or_insert_with(Vec::new)
            .push(comparison);

        // Keep only last 100 comparisons per symbol
        if let Some(mut prices) = self.venue_prices.get_mut(&comparison.symbol) {
            if prices.len() > 100 {
                prices.remove(0);
            }
        }
    }

    /// Analyze best execution for an order
    pub fn analyze_best_execution(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: i64,
        proposed_price: u64,
        proposed_venue: &str,
    ) -> BestExecAnalysis {
        self.total_best_exec_checks.fetch_add(1, Ordering::Relaxed);

        let current_time_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        let mut best_price = u64::MAX;
        let mut best_venue = String::new();

        if let Some(prices) = self.venue_prices.get(symbol) {
            // Filter recent prices (within last 1 second)
            let cutoff = current_time_ns.saturating_sub(1_000_000_000);
            
            for comp in prices.iter().filter(|c| c.timestamp_ns >= cutoff) {
                let is_better = match side {
                    OrderSide::Buy => comp.price < best_price,
                    OrderSide::Sell => comp.price > best_price && comp.price > 0,
                };

                if is_better && comp.available_quantity >= quantity {
                    best_price = comp.price;
                    best_venue = comp.venue_id.clone();
                }
            }
        }

        // Calculate price improvement
        let price_improvement_bps = if best_price > 0 {
            match side {
                OrderSide::Buy => {
                    ((best_price as i64 - proposed_price as i64) * 10_000 / best_price as i64) as i32
                }
                OrderSide::Sell => {
                    ((proposed_price as i64 - best_price as i64) * 10_000 / best_price as i64) as i32
                }
            }
        } else {
            0
        };

        // Determine compliance
        // For now, consider compliant if within 5 bps of best available
        let is_compliant = price_improvement_bps >= -5 || best_venue.is_empty();

        BestExecAnalysis {
            symbol: symbol.to_string(),
            side,
            requested_quantity: quantity,
            best_price: if best_price == u64::MAX { 0 } else { best_price },
            best_venue,
            price_improvement_bps,
            total_cost: proposed_price * quantity as u64,
            is_compliant,
            analyzed_at_ns: current_time_ns,
        }
    }

    /// Set transparency waiver for a symbol
    pub fn set_waiver(&self, symbol: &str, waiver: TransparencyWaiver) {
        self.waivers.insert(symbol.to_string(), waiver);
    }

    /// Check if transparency waiver applies
    pub fn check_waiver(&self, symbol: &str, quantity: i64, threshold: i64) -> TransparencyWaiver {
        self.waivers
            .get(symbol)
            .copied()
            .unwrap_or(TransparencyWaiver::None)
    }

    /// Get tick size for a price
    pub fn get_tick_size(&self, symbol: &str, venue: &str, price: u64) -> Option<u64> {
        let key = (symbol.to_string(), venue.to_string());
        self.tick_tables
            .get(&key)
            .and_then(|t| t.get_tick_size(price))
    }

    /// Round price to valid tick
    pub fn round_to_tick(&self, symbol: &str, venue: &str, price: u64) -> Option<u64> {
        let key = (symbol.to_string(), venue.to_string());
        self.tick_tables
            .get(&key)
            .and_then(|t| t.round_to_tick(price))
    }

    /// Get statistics
    pub fn get_stats(&self) -> MifidStats {
        MifidStats {
            total_tick_checks: self.total_tick_checks.load(Ordering::Relaxed),
            total_tick_violations: self.total_tick_violations.load(Ordering::Relaxed),
            total_best_exec_checks: self.total_best_exec_checks.load(Ordering::Relaxed),
            registered_tables: self.tick_tables.len(),
            active_waivers: self.waivers.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MifidStats {
    pub total_tick_checks: u64,
    pub total_tick_violations: u64,
    pub total_best_exec_checks: u64,
    pub registered_tables: usize,
    pub active_waivers: usize,
}

impl Default for MifidEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_size_table() {
        let mut table = TickSizeTable::new("AAPL".to_string(), "XNYS".to_string(), TickSizeRegime::Standard);
        
        // Add standard tick size bands
        table.add_band(0, 100, 1);      // $0-100: $0.01 ticks
        table.add_band(100, 1000, 5);   // $100-1000: $0.05 ticks
        table.add_band(1000, 10000, 10);// $1000-10000: $0.10 ticks

        // Test tick size lookup
        assert_eq!(table.get_tick_size(50), Some(1));
        assert_eq!(table.get_tick_size(500), Some(5));
        assert_eq!(table.get_tick_size(5000), Some(10));

        // Test price compliance
        assert!(table.is_price_compliant(50));
        assert!(table.is_price_compliant(55));
        assert!(!table.is_price_compliant(502)); // Not multiple of 5

        // Test rounding
        assert_eq!(table.round_to_tick(503), Some(500));
        assert_eq!(table.round_to_tick(507), Some(505));
    }

    #[test]
    fn test_validate_tick_size() {
        let engine = MifidEngine::new();
        
        let mut table = TickSizeTable::new("AAPL".to_string(), "XNYS".to_string(), TickSizeRegime::Standard);
        table.add_band(0, 10000, 1);
        engine.register_tick_table(table);

        let result = engine.validate_tick_size("AAPL", "XNYS", 150);
        assert!(result.passed);

        let result = engine.validate_tick_size("UNKNOWN", "XNYS", 150);
        assert!(result.passed); // Unknown symbols pass with warning
    }

    #[test]
    fn test_best_execution_analysis() {
        let engine = MifidEngine::new();

        // Record some venue prices
        let base_time = 1000000000u64;
        engine.record_venue_price(VenueComparison {
            venue_id: "VENUE1".to_string(),
            price: 10000,
            available_quantity: 1000,
            timestamp_ns: base_time,
            fees_bps: 10,
            latency_us: 100,
        });
        engine.record_venue_price(VenueComparison {
            venue_id: "VENUE2".to_string(),
            price: 9998,
            available_quantity: 500,
            timestamp_ns: base_time + 100,
            fees_bps: 15,
            latency_us: 150,
        });

        let analysis = engine.analyze_best_execution(
            "TEST",
            OrderSide::Buy,
            500,
            10001,
            "VENUE3",
        );

        assert_eq!(analysis.best_price, 9998);
        assert_eq!(analysis.best_venue, "VENUE2");
        assert!(analysis.price_improvement_bps < 0); // Worse than best
    }
}
