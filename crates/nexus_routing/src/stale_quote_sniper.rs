//! Stale Quote Sniper for latency arbitrage.
//! Detects when lagging exchange quotes haven't updated to reflect price moves.

use nexus_oms::{FixedPoint, Side};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use parking_lot::RwLock;

const SCALE: i64 = 100_000_000;

/// Maximum number of tracked symbols
pub const MAX_SYMBOLS: usize = 256;

/// Symbol state tracking
#[derive(Debug, Clone, Copy)]
pub struct SymbolState {
    /// Leading exchange best bid
    pub leader_bid: FixedPoint,
    /// Leading exchange best ask
    pub leader_ask: FixedPoint,
    /// Leading exchange timestamp
    pub leader_timestamp_ns: u64,
    /// Lagging exchange best bid
    pub lagger_bid: FixedPoint,
    /// Lagging exchange best ask
    pub lagger_ask: FixedPoint,
    /// Lagging exchange timestamp
    pub lagger_timestamp_ns: u64,
}

impl SymbolState {
    #[inline]
    pub fn new() -> Self {
        Self {
            leader_bid: FixedPoint::from_raw(0),
            leader_ask: FixedPoint::from_raw(0),
            leader_timestamp_ns: 0,
            lagger_bid: FixedPoint::from_raw(0),
            lagger_ask: FixedPoint::from_raw(0),
            lagger_timestamp_ns: 0,
        }
    }

    /// Check if there's a stale quote opportunity
    #[inline]
    pub fn has_stale_quote(&self, max_age_ns: u64) -> Option<StaleQuoteOpportunity> {
        let now = self.leader_timestamp_ns; // Use leader timestamp as reference
        
        // Check if lagger quote is old enough
        let age = now.saturating_sub(self.lagger_timestamp_ns);
        if age < max_age_ns {
            return None;
        }

        // Check for cross-market arbitrage opportunities
        // Opportunity exists if leader bid > lagger ask (buy on lagger, sell on leader)
        if self.leader_bid > self.lagger_ask && !self.lagger_ask.is_zero() {
            return Some(StaleQuoteOpportunity {
                side: Side::Buy,
                buy_price: self.lagger_ask,
                sell_price: self.leader_bid,
                spread: self.leader_bid - self.lagger_ask,
                age_ns: age,
            });
        }

        // Or if leader ask < lagger bid (sell on lagger, buy on leader)
        if self.leader_ask < self.lagger_bid && !self.lagger_bid.is_zero() {
            return Some(StaleQuoteOpportunity {
                side: Side::Sell,
                buy_price: self.leader_ask,
                sell_price: self.lagger_bid,
                spread: self.lagger_bid - self.leader_ask,
                age_ns: age,
            });
        }

        None
    }
}

impl Default for SymbolState {
    fn default() -> Self {
        Self::new()
    }
}

/// Stale quote arbitrage opportunity
#[derive(Debug, Clone, Copy)]
pub struct StaleQuoteOpportunity {
    pub side: Side,
    pub buy_price: FixedPoint,
    pub sell_price: FixedPoint,
    pub spread: FixedPoint,
    pub age_ns: u64,
}

impl StaleQuoteOpportunity {
    /// Expected profit (scaled by 10^8)
    #[inline]
    pub fn expected_profit(&self) -> FixedPoint {
        self.spread
    }

    /// Risk score (higher is riskier, based on age)
    #[inline]
    pub fn risk_score(&self) -> FixedPoint {
        // Risk increases with age (quote might be truly stale)
        let age_factor = FixedPoint::from_raw((self.age_ns / 100_000) as i64).min(FixedPoint::from_int(100));
        age_factor * FixedPoint::from_fractional(1_000_000) // Scale to 0-1 range roughly
    }
}

/// Stale Quote Sniper
pub struct StaleQuoteSniper {
    /// Symbol states (indexed by symbol_id)
    symbols: RwLock<[SymbolState; MAX_SYMBOLS]>,
    /// Maximum acceptable quote age in nanoseconds
    max_quote_age_ns: AtomicU64,
    /// Minimum spread threshold for execution (scaled by 10^8)
    min_spread: FixedPoint,
    /// Enabled flag
    enabled: AtomicBool,
    /// Opportunities detected counter
    opportunities_detected: AtomicU64,
    /// Opportunities executed counter
    opportunities_executed: AtomicU64,
}

impl StaleQuoteSniper {
    #[inline]
    pub fn new(max_quote_age_ms: u64, min_spread_bps: u32) -> Self {
        let min_spread = FixedPoint::from_raw((min_spread_bps as i64 * SCALE) / 10_000);
        
        Self {
            symbols: RwLock::new(std::array::from_fn(|_| SymbolState::new())),
            max_quote_age_ns: AtomicU64::new(max_quote_age_ms * 1_000_000),
            min_spread,
            enabled: AtomicBool::new(true),
            opportunities_detected: AtomicU64::new(0),
            opportunities_executed: AtomicU64::new(0),
        }
    }

    /// Update leader quote
    #[inline]
    pub fn update_leader(&self, symbol_id: u32, bid: FixedPoint, ask: FixedPoint, timestamp_ns: u64) {
        if symbol_id >= MAX_SYMBOLS as u32 {
            return;
        }

        let mut symbols = self.symbols.write();
        let state = &mut symbols[symbol_id as usize];
        state.leader_bid = bid;
        state.leader_ask = ask;
        state.leader_timestamp_ns = timestamp_ns;
    }

    /// Update lagger quote
    #[inline]
    pub fn update_lagger(&self, symbol_id: u32, bid: FixedPoint, ask: FixedPoint, timestamp_ns: u64) {
        if symbol_id >= MAX_SYMBOLS as u32 {
            return;
        }

        let mut symbols = self.symbols.write();
        let state = &mut symbols[symbol_id as usize];
        state.lagger_bid = bid;
        state.lagger_ask = ask;
        state.lagger_timestamp_ns = timestamp_ns;
    }

    /// Scan for stale quote opportunities
    /// Returns vector of (symbol_id, opportunity)
    #[inline]
    pub fn scan_opportunities(&self) -> Vec<(u32, StaleQuoteOpportunity)> {
        if !self.enabled.load(Ordering::Relaxed) {
            return Vec::new();
        }

        let symbols = self.symbols.read();
        let max_age = self.max_quote_age_ns.load(Ordering::Relaxed);
        let mut results = Vec::new();

        for (idx, state) in symbols.iter().enumerate() {
            if let Some(opp) = state.has_stale_quote(max_age) {
                // Only include if spread meets minimum threshold
                if opp.spread >= self.min_spread {
                    results.push((idx as u32, opp));
                    self.opportunities_detected.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        // Sort by spread descending (best opportunities first)
        results.sort_by(|a, b| b.1.spread.cmp(&a.1.spread));
        results
    }

    /// Record an executed opportunity
    #[inline]
    pub fn record_execution(&self) {
        self.opportunities_executed.fetch_add(1, Ordering::Relaxed);
    }

    /// Enable/disable the sniper
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Get opportunities detected count
    #[inline]
    pub fn get_opportunities_detected(&self) -> u64 {
        self.opportunities_detected.load(Ordering::Relaxed)
    }

    /// Get opportunities executed count
    #[inline]
    pub fn get_opportunities_executed(&self) -> u64 {
        self.opportunities_executed.load(Ordering::Relaxed)
    }

    /// Set maximum quote age
    #[inline]
    pub fn set_max_quote_age_ms(&self, age_ms: u64) {
        self.max_quote_age_ns.store(age_ms * 1_000_000, Ordering::Relaxed);
    }

    /// Set minimum spread threshold
    #[inline]
    pub fn set_min_spread_bps(&self, bps: u32) {
        let spread = FixedPoint::from_raw((bps as i64 * SCALE) / 10_000);
        self.min_spread = spread;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stale_quote_detection() {
        let sniper = StaleQuoteSniper::new(5, 10); // 5ms max age, 10bps min spread

        // Set up leader quote at time T
        sniper.update_leader(
            0,
            FixedPoint::from_int(101), // leader bid
            FixedPoint::from_int(102), // leader ask
            1_000_000_000, // T = 1s
        );

        // Set up stale lagger quote at time T - 10ms
        sniper.update_lagger(
            0,
            FixedPoint::from_int(100), // lagger bid
            FixedPoint::from_int(100), // lagger ask (stale!)
            990_000_000, // T - 10ms
        );

        // Scan for opportunities
        let opps = sniper.scan_opportunities();
        
        assert_eq!(opps.len(), 1);
        let (symbol_id, opp) = &opps[0];
        assert_eq!(*symbol_id, 0);
        assert_eq!(opp.side, Side::Buy);
        assert_eq!(opp.buy_price.to_f64(), 100.0);
        assert_eq!(opp.sell_price.to_f64(), 101.0);
        assert_eq!(opp.spread.to_f64(), 1.0);
    }
}
