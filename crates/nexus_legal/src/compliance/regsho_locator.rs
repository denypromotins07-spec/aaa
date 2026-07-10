// STAGE 23: Computational Law, RegTech Compliance & Immutable Audit Trails
// Chapter 2: Computational Law & Pre-Trade Compliance
// File: crates/nexus_legal/src/compliance/regsho_locator.rs

//! Regulation SHO Compliance Module
//! Implements short-sale locate requirements and tick-test restrictions.
//! Ensures all short sales have valid locates before order submission.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use chrono::{DateTime, Utc, NaiveDate};

use crate::compliance::dag_rule_engine::{ComplianceState, ComplianceFlag, ComplianceResult};

/// Configuration for Reg SHO compliance
#[derive(Debug, Clone)]
pub struct RegShoConfig {
    /// Enable strict tick test enforcement
    pub enable_tick_test: bool,
    /// Default locate timeout (24 hours)
    pub locate_timeout_secs: u64,
    /// Auto-renew locates before expiry
    pub auto_renew_locates: bool,
    /// Threshold for close-out requirement (13 consecutive fails)
    pub closeout_threshold: u32,
}

impl Default for RegShoConfig {
    fn default() -> Self {
        Self {
            enable_tick_test: true,
            locate_timeout_secs: 24 * 60 * 60,
            auto_renew_locates: false,
            closeout_threshold: 13,
        }
    }
}

/// A short-sale locate record
#[derive(Debug, Clone)]
pub struct LocateRecord {
    /// Unique locate ID
    pub locate_id: u64,
    /// Symbol being located
    pub symbol: String,
    /// Quantity located
    pub quantity: i64,
    /// Remaining available quantity
    pub remaining_quantity: i64,
    /// Timestamp when locate was obtained
    pub obtained_at_ns: u64,
    /// Timestamp when locate expires
    pub expires_at_ns: u64,
    /// Source of the locate (broker/dealer ID)
    pub source_id: String,
    /// Whether this is a bona fide market making locate
    pub bona_fide_mm: bool,
    /// Number of times used
    pub usage_count: u64,
}

impl LocateRecord {
    pub fn is_valid(&self, current_time_ns: u64) -> bool {
        current_time_ns < self.expires_at_ns && self.remaining_quantity > 0
    }

    pub fn is_expired(&self, current_time_ns: u64) -> bool {
        current_time_ns >= self.expires_at_ns
    }

    pub fn can_fill(&self, quantity: i64) -> bool {
        self.remaining_quantity >= quantity
    }

    pub fn use_quantity(&mut self, quantity: i64) -> bool {
        if self.can_fill(quantity) {
            self.remaining_quantity -= quantity;
            self.usage_count += 1;
            true
        } else {
            false
        }
    }
}

/// Tick test state for a symbol
#[derive(Debug, Clone)]
pub struct TickTestState {
    pub symbol: String,
    /// Last trade price (fixed point)
    pub last_trade_price: u64,
    /// Previous trade price
    pub prev_trade_price: u64,
    /// Current bid price
    pub current_bid: u64,
    /// Current ask price
    pub current_ask: u64,
    /// Is uptick (price increased from previous)
    pub is_uptick: bool,
    /// Is zero uptick (price same but last move was up)
    pub is_zero_uptick: bool,
    /// Timestamp of last update
    pub last_update_ns: u64,
}

impl TickTestState {
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            last_trade_price: 0,
            prev_trade_price: 0,
            current_bid: 0,
            current_ask: 0,
            is_uptick: false,
            is_zero_uptick: false,
            last_update_ns: 0,
        }
    }

    /// Update tick state with new trade
    pub fn update_trade(&mut self, price: u64, timestamp_ns: u64) {
        self.prev_trade_price = self.last_trade_price;
        self.last_trade_price = price;
        self.last_update_ns = timestamp_ns;

        if price > self.prev_trade_price && self.prev_trade_price > 0 {
            self.is_uptick = true;
            self.is_zero_uptick = false;
        } else if price == self.prev_trade_price && self.prev_trade_price > 0 {
            // Zero tick - inherit direction from last price change
            self.is_zero_uptick = self.is_uptick;
        } else if price < self.prev_trade_price {
            self.is_uptick = false;
            self.is_zero_uptick = false;
        }
    }

    /// Update with new quote
    pub fn update_quote(&mut self, bid: u64, ask: u64, timestamp_ns: u64) {
        self.current_bid = bid;
        self.current_ask = ask;
        self.last_update_ns = timestamp_ns;
    }

    /// Check if short sale is allowed under tick test
    /// Rule: Short sales only allowed on upticks or zero upticks
    pub fn short_allowed(&self, price: u64) -> bool {
        // Price must be above last trade (uptick) or at last trade if zero uptick
        if price > self.last_trade_price {
            true
        } else if price == self.last_trade_price {
            self.is_zero_uptick || self.is_uptick
        } else {
            false
        }
    }
}

/// Failure to deliver (FTD) tracking for close-out requirements
#[derive(Debug, Clone)]
pub struct FtdRecord {
    pub symbol: String,
    pub quantity: i64,
    pub settlement_date: NaiveDate,
    pub days_outstanding: u32,
}

/// Reg SHO compliance engine
pub struct RegShoEngine {
    config: RegShoConfig,
    /// Locates by symbol
    locates: DashMap<String, Vec<LocateRecord>>,
    /// Tick test states by symbol
    tick_states: DashMap<String, TickTestState>,
    /// FTD records by symbol
    ftd_records: DashMap<String, Vec<FtdRecord>>,
    /// Symbols on threshold list (13+ consecutive fails)
    threshold_list: DashMap<String, u32>,
    /// Statistics
    total_locate_requests: AtomicU64,
    total_locate_failures: AtomicU64,
    total_tick_test_failures: AtomicU64,
}

impl RegShoEngine {
    pub fn new() -> Self {
        Self::new_with_config(RegShoConfig::default())
    }

    pub fn new_with_config(config: RegShoConfig) -> Self {
        Self {
            config,
            locates: DashMap::new(),
            tick_states: DashMap::new(),
            ftd_records: DashMap::new(),
            threshold_list: DashMap::new(),
            total_locate_requests: AtomicU64::new(0),
            total_locate_failures: AtomicU64::new(0),
            total_tick_test_failures: AtomicU64::new(0),
        }
    }

    /// Request a locate for a short sale
    pub fn request_locate(
        &self,
        symbol: &str,
        quantity: i64,
        source_id: &str,
        bona_fide_mm: bool,
    ) -> Result<LocateRecord, RegShoError> {
        self.total_locate_requests.fetch_add(1, Ordering::Relaxed);

        let current_time_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| RegShoError::TimeError)?
            .as_nanos() as u64;

        // Check if symbol is on threshold list (requires additional restrictions)
        let on_threshold = self.threshold_list.contains_key(symbol);

        // Generate unique locate ID
        let locate_id = current_time_ns ^ (quantity as u64);

        let expires_at_ns = current_time_ns + (self.config.locate_timeout_secs * 1_000_000_000);

        let record = LocateRecord {
            locate_id,
            symbol: symbol.to_string(),
            quantity,
            remaining_quantity: quantity,
            obtained_at_ns: current_time_ns,
            expires_at_ns,
            source_id: source_id.to_string(),
            bona_fide_mm,
            usage_count: 0,
        };

        // Store the locate
        self.locates
            .entry(symbol.to_string())
            .or_insert_with(Vec::new)
            .push(record.clone());

        Ok(record)
    }

    /// Validate a short sale order
    pub fn validate_short_sale(
        &self,
        symbol: &str,
        quantity: i64,
        price: u64,
    ) -> ComplianceResult {
        let start = Instant::now();
        let mut failed_flags: u128 = 0;

        // Check locate requirement
        if !self.has_valid_locate(symbol, quantity) {
            failed_flags |= ComplianceFlag::RegShoLocated as u128;
            self.total_locate_failures.fetch_add(1, Ordering::Relaxed);
        }

        // Check tick test if enabled
        if self.config.enable_tick_test {
            if let Some(tick_state) = self.tick_states.get(symbol) {
                if !tick_state.short_allowed(price) {
                    failed_flags |= ComplianceFlag::RegShoTickTestPassed as u128;
                    self.total_tick_test_failures.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        // Check if symbol is restricted
        if self.is_symbol_restricted(symbol) {
            failed_flags |= ComplianceFlag::RegShoNotShortRestricted as u128;
        }

        let elapsed = start.elapsed().as_nanos() as u64;

        if failed_flags != 0 {
            ComplianceResult::failed(failed_flags, elapsed)
        } else {
            let mut result = ComplianceResult::PASSED;
            result.evaluation_time_ns = elapsed;
            result
        }
    }

    /// Check if a valid locate exists for the given quantity
    pub fn has_valid_locate(&self, symbol: &str, quantity: i64) -> bool {
        let current_time_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        if let Some(locates) = self.locates.get(symbol) {
            let available: i64 = locates
                .iter()
                .filter(|l| l.is_valid(current_time_ns))
                .map(|l| l.remaining_quantity)
                .sum();
            
            available >= quantity
        } else {
            false
        }
    }

    /// Use a locate for an executed short sale
    pub fn use_locate(&self, symbol: &str, quantity: i64) -> bool {
        let current_time_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        if let Some(mut locates) = self.locates.get_mut(symbol) {
            // Sort by expiration (use soonest expiring first)
            locates.sort_by_key(|l| l.expires_at_ns);

            let mut remaining = quantity;
            for locate in locates.iter_mut() {
                if locate.is_valid(current_time_ns) {
                    let use_qty = remaining.min(locate.remaining_quantity);
                    if locate.use_quantity(use_qty) {
                        remaining -= use_qty;
                    }
                    if remaining <= 0 {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Update tick state for a symbol
    pub fn update_tick_state(&self, symbol: &str, price: u64, timestamp_ns: u64) {
        if let Some(mut state) = self.tick_states.get_mut(symbol) {
            state.update_trade(price, timestamp_ns);
        } else {
            let mut state = TickTestState::new(symbol.to_string());
            state.update_trade(price, timestamp_ns);
            self.tick_states.insert(symbol.to_string(), state);
        }
    }

    /// Update quote for tick test
    pub fn update_quote(&self, symbol: &str, bid: u64, ask: u64, timestamp_ns: u64) {
        if let Some(mut state) = self.tick_states.get_mut(symbol) {
            state.update_quote(bid, ask, timestamp_ns);
        }
    }

    /// Record a failure to deliver
    pub fn record_ftd(&self, symbol: &str, quantity: i64, settlement_date: NaiveDate) {
        let ftd = FtdRecord {
            symbol: symbol.to_string(),
            quantity,
            settlement_date,
            days_outstanding: 0,
        };

        self.ftd_records
            .entry(symbol.to_string())
            .or_insert_with(Vec::new)
            .push(ftd);

        // Update threshold list counter
        let count = self.threshold_list
            .entry(symbol.to_string())
            .or_insert(0);
        *count += 1;
    }

    /// Check if symbol is on threshold list
    pub fn is_symbol_restricted(&self, symbol: &str) -> bool {
        self.threshold_list
            .get(symbol)
            .map(|c| *c >= self.config.closeout_threshold)
            .unwrap_or(false)
    }

    /// Get current tick test state
    pub fn get_tick_state(&self, symbol: &str) -> Option<TickTestState> {
        self.tick_states.get(symbol).map(|s| s.clone())
    }

    /// Clean up expired locates
    pub fn cleanup_expired_locates(&self) -> usize {
        let current_time_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        let mut cleaned = 0;
        
        for mut entry in self.locates.iter_mut() {
            let before_len = entry.value().len();
            entry.value_mut().retain(|l| l.is_valid(current_time_ns));
            cleaned += before_len - entry.value().len();
        }

        cleaned
    }

    /// Get statistics
    pub fn get_stats(&self) -> RegShoStats {
        RegShoStats {
            total_locate_requests: self.total_locate_requests.load(Ordering::Relaxed),
            total_locate_failures: self.total_locate_failures.load(Ordering::Relaxed),
            total_tick_test_failures: self.total_tick_test_failures.load(Ordering::Relaxed),
            active_locates: self.locates.iter().map(|e| e.value().len()).sum(),
            symbols_on_threshold: self.threshold_list.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegShoStats {
    pub total_locate_requests: u64,
    pub total_locate_failures: u64,
    pub total_tick_test_failures: u64,
    pub active_locates: usize,
    pub symbols_on_threshold: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegShoError {
    NoValidLocate,
    TickTestFailed,
    SymbolRestricted,
    InsufficientQuantity,
    TimeError,
}

impl std::fmt::Display for RegShoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegShoError::NoValidLocate => write!(f, "No valid locate available"),
            RegShoError::TickTestFailed => write!(f, "Tick test failed"),
            RegShoError::SymbolRestricted => write!(f, "Symbol is restricted"),
            RegShoError::InsufficientQuantity => write!(f, "Insufficient locate quantity"),
            RegShoError::TimeError => write!(f, "System time error"),
        }
    }
}

impl std::error::Error for RegShoError {}

impl Default for RegShoEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_locate_request() {
        let engine = RegShoEngine::new();
        
        let locate = engine.request_locate("AAPL", 1000, "BROKER1", false);
        assert!(locate.is_ok());
        
        let locate = locate.unwrap();
        assert_eq!(locate.symbol, "AAPL");
        assert_eq!(locate.quantity, 1000);
        assert!(locate.remaining_quantity > 0);
    }

    #[test]
    fn test_tick_test_uptick() {
        let mut state = TickTestState::new("AAPL".to_string());
        
        // Simulate price sequence: 100 -> 101 (uptick)
        state.update_trade(100, 1000);
        state.update_trade(101, 2000);
        
        assert!(state.is_uptick);
        assert!(state.short_allowed(102)); // Above last trade
        assert!(state.short_allowed(101)); // At last trade with uptick
        assert!(!state.short_allowed(100)); // Below last trade
    }

    #[test]
    fn test_validate_short_sale() {
        let engine = RegShoEngine::new();
        
        // First get a locate
        let _ = engine.request_locate("AAPL", 1000, "BROKER1", false);
        
        // Now validate should pass locate check
        let result = engine.validate_short_sale("AAPL", 500, 150);
        // Should fail tick test since we haven't set up tick state
        assert!(!result.passed || result.failed_rules & (ComplianceFlag::RegShoTickTestPassed as u128) != 0);
    }
}
