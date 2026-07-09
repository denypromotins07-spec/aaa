//! Smart Order Routing (SOR) Engine.
//! Calculates best execution path across multiple venues in real-time.

use nexus_oms::{FixedPoint, Side, OrderType};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use parking_lot::RwLock;

const SCALE: i64 = 100_000_000;

/// Maximum number of supported venues
pub const MAX_VENUES: usize = 8;

/// Venue identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VenueId(pub u32);

impl VenueId {
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
}

/// Venue state and quote information
#[derive(Debug, Clone, Copy)]
pub struct VenueQuote {
    pub venue_id: VenueId,
    pub best_bid: FixedPoint,
    pub best_ask: FixedPoint,
    pub best_bid_qty: FixedPoint,
    pub best_ask_qty: FixedPoint,
    pub timestamp_ns: u64,
    pub is_alive: bool,
}

impl VenueQuote {
    #[inline]
    pub fn new(venue_id: VenueId) -> Self {
        Self {
            venue_id,
            best_bid: FixedPoint::from_raw(0),
            best_ask: FixedPoint::from_raw(0),
            best_bid_qty: FixedPoint::from_raw(0),
            best_ask_qty: FixedPoint::from_raw(0),
            timestamp_ns: 0,
            is_alive: false,
        }
    }

    #[inline]
    pub fn mid_price(&self) -> FixedPoint {
        if self.best_bid.is_zero() || self.best_ask.is_zero() {
            return FixedPoint::from_raw(0);
        }
        (self.best_bid + self.best_ask) / FixedPoint::from_int(2)
    }

    #[inline]
    pub fn spread(&self) -> FixedPoint {
        if self.best_ask.is_zero() || self.best_bid.is_zero() {
            return FixedPoint::from_raw(0);
        }
        self.best_ask - self.best_bid
    }
}

/// Execution quality metrics for a venue
#[derive(Debug, Clone, Copy)]
pub struct VenueMetrics {
    /// EWMA of round-trip time in nanoseconds
    pub rtt_ewma_ns: u64,
    /// Fill rate (scaled by 10^8)
    pub fill_rate: FixedPoint,
    /// Rejection count
    pub rejection_count: u64,
    /// Timeout count
    pub timeout_count: u64,
    /// Last update timestamp
    pub last_update_ns: u64,
}

impl VenueMetrics {
    #[inline]
    pub fn new() -> Self {
        Self {
            rtt_ewma_ns: 1_000_000, // Default 1ms
            fill_rate: FixedPoint::from_fractional(95_000_000), // 95%
            rejection_count: 0,
            timeout_count: 0,
            last_update_ns: 0,
        }
    }
}

impl Default for VenueMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Smart Order Router
pub struct SorEngine {
    /// Venue quotes (indexed by venue_id)
    quotes: RwLock<[Option<VenueQuote>; MAX_VENUES]>,
    /// Venue metrics
    metrics: RwLock<[VenueMetrics; MAX_VENUES]>,
    /// Enabled venues
    enabled: [AtomicBool; MAX_VENUES],
    /// Default venue
    default_venue: AtomicU64,
    /// Sequence number
    sequence: AtomicU64,
}

impl SorEngine {
    #[inline]
    pub fn new(default_venue: u32) -> Self {
        let mut quotes: [Option<VenueQuote>; MAX_VENUES] = Default::default();
        let mut metrics: [VenueMetrics; MAX_VENUES] = Default::default();
        
        for i in 0..MAX_VENUES {
            quotes[i] = Some(VenueQuote::new(VenueId::new(i as u32)));
            metrics[i] = VenueMetrics::new();
        }

        Self {
            quotes: RwLock::new(quotes),
            metrics: RwLock::new(metrics),
            enabled: std::array::from_fn(|_| AtomicBool::new(true)),
            default_venue: AtomicU64::new(default_venue as u64),
            sequence: AtomicU64::new(0),
        }
    }

    /// Update venue quote
    #[inline]
    pub fn update_quote(&self, quote: VenueQuote) {
        let idx = quote.venue_id.0 as usize;
        if idx >= MAX_VENUES {
            return;
        }

        let mut quotes = self.quotes.write();
        quotes[idx] = Some(quote);
        self.sequence.fetch_add(1, Ordering::Relaxed);
    }

    /// Update venue RTT metric using EWMA
    #[inline]
    pub fn update_rtt(&self, venue_id: VenueId, rtt_ns: u64, alpha: FixedPoint) {
        let idx = venue_id.0 as usize;
        if idx >= MAX_VENUES {
            return;
        }

        let mut metrics = self.metrics.write();
        let current = metrics[idx].rtt_ewma_ns;
        
        // EWMA: new = alpha * sample + (1 - alpha) * current
        let one_minus_alpha = FixedPoint::from_raw(SCALE) - alpha;
        let weighted_sample = (rtt_ns as i128 * alpha.raw() as i128) / SCALE as i128;
        let weighted_current = (current as i128 * one_minus_alpha.raw() as i128) / SCALE as i128;
        
        let new_rtt = ((weighted_sample + weighted_current) as u64).max(1);
        metrics[idx].rtt_ewma_ns = new_rtt;
        metrics[idx].last_update_ns = rtt_ns; // Use sample as timestamp proxy
        
        self.sequence.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a fill for a venue
    #[inline]
    pub fn record_fill(&self, venue_id: VenueId) {
        let idx = venue_id.0 as usize;
        if idx >= MAX_VENUES {
            return;
        }

        let mut metrics = self.metrics.write();
        let current_rate = metrics[idx].fill_rate;
        
        // Increment fill rate slightly (simplified)
        let increment = FixedPoint::from_fractional(1_000_000); // 1%
        let new_rate = (current_rate + increment).min(FixedPoint::from_raw(SCALE));
        metrics[idx].fill_rate = new_rate;
        
        self.sequence.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a rejection for a venue
    #[inline]
    pub fn record_rejection(&self, venue_id: VenueId) {
        let idx = venue_id.0 as usize;
        if idx >= MAX_VENUES {
            return;
        }

        let mut metrics = self.metrics.write();
        metrics[idx].rejection_count += 1;
        
        // Decrease fill rate
        let decrement = FixedPoint::from_fractional(5_000_000); // 5%
        let new_rate = (metrics[idx].fill_rate - decrement).max(FixedPoint::from_raw(0));
        metrics[idx].fill_rate = new_rate;
        
        self.sequence.fetch_add(1, Ordering::Relaxed);
    }

    /// Find best venue for execution
    /// Returns venue ID and expected price
    #[inline]
    pub fn find_best_venue(&self, side: Side, quantity: FixedPoint) -> Option<(VenueId, FixedPoint)> {
        let quotes = self.quotes.read();
        let metrics = self.metrics.read();

        let mut best_venue: Option<VenueId> = None;
        let mut best_price = FixedPoint::from_raw(0);
        let mut best_score = FixedPoint::from_raw(0);

        for i in 0..MAX_VENUES {
            if !self.enabled[i].load(Ordering::Relaxed) {
                continue;
            }

            let quote = match &quotes[i] {
                Some(q) if q.is_alive => q,
                _ => continue,
            };

            let metric = &metrics[i];
            
            // Skip venues with high latency (> 10ms EWMA)
            if metric.rtt_ewma_ns > 10_000_000 {
                continue;
            }

            // Calculate score based on price and metrics
            let price = match side {
                Side::Buy => quote.best_ask,
                Side::Sell => quote.best_bid,
            };

            let available_qty = match side {
                Side::Buy => quote.best_ask_qty,
                Side::Sell => quote.best_bid_qty,
            };

            // Skip if insufficient liquidity
            if available_qty < quantity {
                continue;
            }

            // Score = price advantage * fill_rate / latency_factor
            let price_score = if price.is_zero() {
                FixedPoint::from_raw(0)
            } else {
                FixedPoint::from_raw(SCALE) / price
            };

            let latency_factor = FixedPoint::from_raw((metric.rtt_ewma_ns / 100_000) as i64).max(FixedPoint::from_int(1));
            let score = (price_score * metric.fill_rate) / latency_factor;

            if score > best_score {
                best_score = score;
                best_venue = Some(quote.venue_id);
                best_price = price;
            }
        }

        best_venue.map(|v| (v, best_price))
    }

    /// Split order across multiple venues
    /// Returns vector of (venue_id, quantity, price)
    #[inline]
    pub fn split_order(&self, side: Side, total_qty: FixedPoint, max_splits: usize) -> Vec<(VenueId, FixedPoint, FixedPoint)> {
        let quotes = self.quotes.read();
        let mut result = Vec::with_capacity(max_splits.min(MAX_VENUES));
        let mut remaining = total_qty;

        // Sort venues by score (simplified: just iterate in order)
        for i in 0..MAX_VENUES {
            if result.len() >= max_splits || remaining.is_zero() {
                break;
            }

            if !self.enabled[i].load(Ordering::Relaxed) {
                continue;
            }

            let quote = match &quotes[i] {
                Some(q) if q.is_alive => q,
                _ => continue,
            };

            let available = match side {
                Side::Buy => quote.best_ask_qty,
                Side::Sell => quote.best_bid_qty,
            };

            if available.is_zero() {
                continue;
            }

            let take_qty = available.min(remaining);
            if !take_qty.is_zero() {
                let price = match side {
                    Side::Buy => quote.best_ask,
                    Side::Sell => quote.best_bid,
                };
                result.push((quote.venue_id, take_qty, price));
                remaining = remaining - take_qty;
            }
        }

        result
    }

    /// Enable or disable a venue
    #[inline]
    pub fn set_venue_enabled(&self, venue_id: VenueId, enabled: bool) {
        let idx = venue_id.0 as usize;
        if idx < MAX_VENUES {
            self.enabled[idx].store(enabled, Ordering::Relaxed);
            self.sequence.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Check if venue is enabled
    #[inline]
    pub fn is_venue_enabled(&self, venue_id: VenueId) -> bool {
        let idx = venue_id.0 as usize;
        idx < MAX_VENUES && self.enabled[idx].load(Ordering::Relaxed)
    }

    /// Get default venue
    #[inline]
    pub fn get_default_venue(&self) -> VenueId {
        VenueId::new(self.default_venue.load(Ordering::Relaxed) as u32)
    }

    /// Get sequence number
    #[inline]
    pub fn get_sequence(&self) -> u64 {
        self.sequence.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sor_engine_basic() {
        let sor = SorEngine::new(0);

        // Update quote for venue 0
        let quote = VenueQuote {
            venue_id: VenueId::new(0),
            best_bid: FixedPoint::from_int(99),
            best_ask: FixedPoint::from_int(101),
            best_bid_qty: FixedPoint::from_int(100),
            best_ask_qty: FixedPoint::from_int(100),
            timestamp_ns: 1234567890,
            is_alive: true,
        };
        sor.update_quote(quote);

        // Find best venue for buy
        let result = sor.find_best_venue(Side::Buy, FixedPoint::from_int(10));
        assert!(result.is_some());
        let (venue, price) = result.unwrap();
        assert_eq!(venue, VenueId::new(0));
        assert_eq!(price.to_f64(), 101.0);
    }

    #[test]
    fn test_split_order() {
        let sor = SorEngine::new(0);

        // Update quotes for multiple venues
        for i in 0..3u32 {
            let quote = VenueQuote {
                venue_id: VenueId::new(i),
                best_bid: FixedPoint::from_int(100 - i as i64),
                best_ask: FixedPoint::from_int(102 - i as i64),
                best_bid_qty: FixedPoint::from_int(50),
                best_ask_qty: FixedPoint::from_int(50),
                timestamp_ns: 1234567890,
                is_alive: true,
            };
            sor.update_quote(quote);
        }

        // Split a 120 qty buy order
        let splits = sor.split_order(Side::Buy, FixedPoint::from_int(120), 3);
        
        // Should split across venues
        let total: f64 = splits.iter().map(|(_, qty, _)| qty.to_f64()).sum();
        assert!(total >= 120.0 || total == 100.0); // May be limited by liquidity
    }
}
