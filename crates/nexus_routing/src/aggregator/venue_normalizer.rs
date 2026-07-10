//! Venue Normalizer - Maps external exchange order books to unified internal format
//! 
//! Handles different tick sizes, lot sizes, and fee structures across venues.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VenueError {
    #[error("Invalid tick size for venue {venue}: {tick_size}")]
    InvalidTickSize { venue: String, tick_size: i64 },
    #[error("Invalid lot size for venue {venue}: {lot_size}")]
    InvalidLotSize { venue: String, lot_size: i64 },
    #[error("Price {price} not aligned to tick size {tick_size}")]
    PriceNotAligned { price: i64, tick_size: i64 },
    #[error("Quantity {qty} not aligned to lot size {lot_size}")]
    QuantityNotAligned { qty: i64, lot_size: i64 },
    #[error("Venue {venue} not found")]
    UnknownVenue { venue: String },
}

/// Venue configuration
#[derive(Debug, Clone)]
pub struct VenueConfig {
    pub venue_id: u32,
    pub name: String,
    pub tick_size: i64,        // Minimum price increment (fixed-point)
    pub lot_size: i64,         // Minimum quantity increment
    pub maker_fee_bps: i64,    // Maker fee in basis points
    pub taker_fee_bps: i64,    // Taker fee in basis points
    pub latency_ms: u32,       // Expected routing latency
}

impl VenueConfig {
    pub fn validate(&self) -> Result<(), VenueError> {
        if self.tick_size <= 0 {
            return Err(VenueError::InvalidTickSize {
                venue: self.name.clone(),
                tick_size: self.tick_size,
            });
        }
        if self.lot_size <= 0 {
            return Err(VenueError::InvalidLotSize {
                venue: self.name.clone(),
                lot_size: self.lot_size,
            });
        }
        Ok(())
    }

    /// Check if price is aligned to tick size
    pub fn is_price_aligned(&self, price: i64) -> bool {
        price % self.tick_size == 0
    }

    /// Round price to nearest tick
    pub fn round_price(&self, price: i64) -> i64 {
        let remainder = price % self.tick_size;
        if remainder >= self.tick_size / 2 {
            price + (self.tick_size - remainder)
        } else {
            price - remainder
        }
    }

    /// Check if quantity is aligned to lot size
    pub fn is_qty_aligned(&self, qty: i64) -> bool {
        qty % self.lot_size == 0
    }

    /// Round quantity to nearest lot
    pub fn round_qty(&self, qty: i64) -> i64 {
        let remainder = qty % self.lot_size;
        if remainder >= self.lot_size / 2 {
            qty + (self.lot_size - remainder)
        } else {
            qty - remainder
        }
    }

    /// Calculate taker fee for a given notional value
    pub fn calc_taker_fee(&self, notional: i64) -> i64 {
        notional * self.taker_fee_bps / 10_000
    }

    /// Calculate maker fee for a given notional value
    pub fn calc_maker_fee(&self, notional: i64) -> i64 {
        notional * self.maker_fee_bps / 10_000
    }
}

/// Normalized order book level
#[derive(Debug, Clone)]
pub struct NormalizedLevel {
    pub price: i64,      // Fixed-point, normalized to internal units
    pub qty: i64,        // Normalized quantity
    pub order_count: u32,
}

/// Normalized order book snapshot
#[derive(Debug, Clone)]
pub struct NormalizedOrderBook {
    pub asset_id: u32,
    pub bids: Vec<NormalizedLevel>,
    pub asks: Vec<NormalizedLevel>,
    pub timestamp_ns: u64,
    pub venue_id: u32,
}

impl NormalizedOrderBook {
    pub fn best_bid(&self) -> Option<&NormalizedLevel> {
        self.bids.first()
    }

    pub fn best_ask(&self) -> Option<&NormalizedLevel> {
        self.asks.first()
    }

    pub fn mid_price(&self) -> Option<i64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid.price + ask.price) / 2),
            _ => None,
        }
    }

    pub fn spread(&self) -> Option<i64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask.price - bid.price),
            _ => None,
        }
    }
}

/// Venue Normalizer - converts external venue books to normalized format
pub struct VenueNormalizer {
    venues: dashmap::DashMap<u32, VenueConfig>,
    enabled: AtomicBool,
    normalization_count: AtomicU64,
}

impl VenueNormalizer {
    pub fn new() -> Self {
        Self {
            venues: dashmap::DashMap::new(),
            enabled: AtomicBool::new(true),
            normalization_count: AtomicU64::new(0),
        }
    }

    /// Register a new venue
    pub fn register_venue(&self, config: VenueConfig) -> Result<(), VenueError> {
        config.validate()?;
        self.venues.insert(config.venue_id, config);
        Ok(())
    }

    /// Get venue configuration
    pub fn get_venue(&self, venue_id: u32) -> Option<VenueConfig> {
        self.venues.get(&venue_id).map(|entry| entry.value().clone())
    }

    /// Normalize raw venue order book to internal format
    pub fn normalize_book(
        &self,
        venue_id: u32,
        raw_bids: &[(i64, i64)], // (price, qty) tuples
        raw_asks: &[(i64, i64)],
        timestamp_ns: u64,
    ) -> Result<NormalizedOrderBook, VenueError> {
        let config = self.get_venue(venue_id)
            .ok_or_else(|| VenueError::UnknownVenue { 
                venue: format!("venue_{}", venue_id) 
            })?;

        // Normalize bids
        let mut normalized_bids: Vec<NormalizedLevel> = raw_bids
            .iter()
            .filter_map(|&(price, qty)| {
                let norm_price = config.round_price(price);
                let norm_qty = config.round_qty(qty);
                
                if norm_price > 0 && norm_qty > 0 {
                    Some(NormalizedLevel {
                        price: norm_price,
                        qty: norm_qty,
                        order_count: 1,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort bids descending
        normalized_bids.sort_by(|a, b| b.price.cmp(&a.price));

        // Normalize asks
        let mut normalized_asks: Vec<NormalizedLevel> = raw_asks
            .iter()
            .filter_map(|&(price, qty)| {
                let norm_price = config.round_price(price);
                let norm_qty = config.round_qty(qty);
                
                if norm_price > 0 && norm_qty > 0 {
                    Some(NormalizedLevel {
                        price: norm_price,
                        qty: norm_qty,
                        order_count: 1,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort asks ascending
        normalized_asks.sort_by(|a, b| a.price.cmp(&b.price));

        self.normalization_count.fetch_add(1, Ordering::Relaxed);

        Ok(NormalizedOrderBook {
            asset_id: 0, // Would be passed in real implementation
            bids: normalized_bids,
            asks: normalized_asks,
            timestamp_ns,
            venue_id,
        })
    }

    /// Convert internal price to venue-specific price
    pub fn to_venue_price(&self, venue_id: u32, internal_price: i64) -> Result<i64, VenueError> {
        let config = self.get_venue(venue_id)
            .ok_or_else(|| VenueError::UnknownVenue { 
                venue: format!("venue_{}", venue_id) 
            })?;
        
        Ok(config.round_price(internal_price))
    }

    /// Convert venue quantity to internal quantity
    pub fn from_venue_qty(&self, venue_id: u32, venue_qty: i64) -> Result<i64, VenueError> {
        let config = self.get_venue(venue_id)
            .ok_or_else(|| VenueError::UnknownVenue { 
                venue: format!("venue_{}", venue_id) 
            })?;
        
        Ok(venue_qty) // In real impl, would handle unit conversion
    }

    /// Enable/disable normalizer
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    /// Check if enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Get normalization count
    pub fn get_normalization_count(&self) -> u64 {
        self.normalization_count.load(Ordering::Acquire)
    }
}

impl Default for VenueNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_venue_validation() {
        let config = VenueConfig {
            venue_id: 1,
            name: "TEST".to_string(),
            tick_size: 100,
            lot_size: 1,
            maker_fee_bps: 10,
            taker_fee_bps: 20,
            latency_ms: 5,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_tick_size() {
        let config = VenueConfig {
            venue_id: 1,
            name: "TEST".to_string(),
            tick_size: 0,
            lot_size: 1,
            maker_fee_bps: 10,
            taker_fee_bps: 20,
            latency_ms: 5,
        };
        assert!(matches!(config.validate(), Err(VenueError::InvalidTickSize { .. })));
    }

    #[test]
    fn test_price_rounding() {
        let config = VenueConfig {
            venue_id: 1,
            name: "TEST".to_string(),
            tick_size: 100,
            lot_size: 1,
            maker_fee_bps: 10,
            taker_fee_bps: 20,
            latency_ms: 5,
        };

        // 10050 should round to 10000 (closer to lower tick)
        assert_eq!(config.round_price(10050), 10000);
        
        // 10060 should round to 10100 (closer to higher tick)
        assert_eq!(config.round_price(10060), 10100);
        
        // Already aligned
        assert_eq!(config.round_price(10000), 10000);
    }

    #[test]
    fn test_fee_calculation() {
        let config = VenueConfig {
            venue_id: 1,
            name: "TEST".to_string(),
            tick_size: 100,
            lot_size: 1,
            maker_fee_bps: 10,
            taker_fee_bps: 20,
            latency_ms: 5,
        };

        let notional = 1_000_000; // $1M in fixed-point
        
        // Taker fee: 20 bps = 0.2%
        assert_eq!(config.calc_taker_fee(notional), 2000);
        
        // Maker fee: 10 bps = 0.1%
        assert_eq!(config.calc_maker_fee(notional), 1000);
    }

    #[test]
    fn test_normalize_orderbook() {
        let normalizer = VenueNormalizer::new();
        
        let config = VenueConfig {
            venue_id: 1,
            name: "BINANCE".to_string(),
            tick_size: 100,
            lot_size: 1,
            maker_fee_bps: 10,
            taker_fee_bps: 20,
            latency_ms: 5,
        };
        normalizer.register_venue(config).unwrap();

        let raw_bids = vec![(100050, 100), (99980, 200)];
        let raw_asks = vec![(100120, 150), (100250, 100)];

        let book = normalizer.normalize_book(1, &raw_bids, &raw_asks, 1000).unwrap();

        // Bids should be sorted descending
        assert!(book.bids[0].price >= book.bids[1].price);
        
        // Asks should be sorted ascending
        assert!(book.asks[0].price <= book.asks[1].price);
        
        // Prices should be rounded to tick
        assert!(book.bids.iter().all(|l| l.price % 100 == 0));
        assert!(book.asks.iter().all(|l| l.price % 100 == 0));
    }
}
