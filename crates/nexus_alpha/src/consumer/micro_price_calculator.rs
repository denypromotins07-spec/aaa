//! Micro-Price Calculator for Accurate Fair Value Estimation
//! 
//! Instead of using the standard mid-price (bid + ask) / 2, this module
//! calculates the volume-weighted micro-price which provides a more accurate
//! reflection of true fair value based on live liquidity.
//! 
//! Formula: micro_price = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
//! 
//! ZERO-ALLOC: All calculations use stack-allocated f64 values.

use std::sync::atomic::{AtomicU64, Ordering};

/// Represents a single order book level
#[derive(Debug, Clone, Copy)]
pub struct OrderBookLevel {
    pub price: f64,
    pub size: f64,
}

/// Top N levels of the order book for micro-price calculation
#[derive(Debug, Clone, Copy)]
pub struct OrderBookSnapshot {
    pub bids: [OrderBookLevel; 5],
    pub asks: [OrderBookLevel; 5],
    pub bid_count: usize,
    pub ask_count: usize,
    pub timestamp_ns: u64,
}

impl Default for OrderBookSnapshot {
    fn default() -> Self {
        Self {
            bids: [OrderBookLevel { price: 0.0, size: 0.0 }; 5],
            asks: [OrderBookLevel { price: 0.0, size: 0.0 }; 5],
            bid_count: 0,
            ask_count: 0,
            timestamp_ns: 0,
        }
    }
}

/// Zero-allocation micro-price calculator
pub struct MicroPriceCalculator {
    /// Current micro-price
    current_micro_price: f64,
    /// Current bid-ask spread
    spread: f64,
    /// Total bid volume in top 5 levels
    total_bid_volume: f64,
    /// Total ask volume in top 5 levels
    total_ask_volume: f64,
    /// Last update timestamp
    last_update_ns: AtomicU64,
    /// Update counter
    update_count: AtomicU64,
}

impl MicroPriceCalculator {
    pub fn new() -> Self {
        Self {
            current_micro_price: 0.0,
            spread: 0.0,
            total_bid_volume: 0.0,
            total_ask_volume: 0.0,
            last_update_ns: AtomicU64::new(0),
            update_count: AtomicU64::new(0),
        }
    }

    /// Calculate micro-price from order book snapshot
    /// 
    /// Uses volume-weighted formula for accurate fair value
    pub fn update(&mut self, snapshot: &OrderBookSnapshot) -> Option<f64> {
        if snapshot.bid_count == 0 || snapshot.ask_count == 0 {
            return None;
        }

        // Sum volumes in top 5 levels (or however many are available)
        let mut total_bid_size = 0.0f64;
        let mut total_ask_size = 0.0f64;
        let mut weighted_bid_sum = 0.0f64;
        let mut weighted_ask_sum = 0.0f64;

        for i in 0..snapshot.bid_count.min(5) {
            let level = snapshot.bids[i];
            if level.size > 0.0 && level.price > 0.0 {
                total_bid_size += level.size;
                weighted_bid_sum += level.price * level.size;
            }
        }

        for i in 0..snapshot.ask_count.min(5) {
            let level = snapshot.asks[i];
            if level.size > 0.0 && level.price > 0.0 {
                total_ask_size += level.size;
                weighted_ask_sum += level.price * level.size;
            }
        }

        // Avoid division by zero
        let total_liquidity = total_bid_size + total_ask_size;
        if total_liquidity <= 0.0 {
            return None;
        }

        // ROOT CAUSE FIX: Use best bid/ask for micro-price calculation
        // The classic micro-price formula uses top-of-book
        let best_bid = snapshot.bids[0].price;
        let best_ask = snapshot.asks[0].price;
        let best_bid_size = snapshot.bids[0].size;
        let best_ask_size = snapshot.asks[0].size;

        // Classic micro-price: weighted by opposite side size
        let denominator = best_bid_size + best_ask_size;
        if denominator <= 0.0 {
            return None;
        }

        self.current_micro_price = (best_bid * best_ask_size + best_ask * best_bid_size) / denominator;
        self.spread = best_ask - best_bid;
        self.total_bid_volume = total_bid_size;
        self.total_ask_volume = total_ask_size;

        self.last_update_ns.store(snapshot.timestamp_ns, Ordering::Relaxed);
        self.update_count.fetch_add(1, Ordering::Relaxed);

        Some(self.current_micro_price)
    }

    /// Get current micro-price
    pub fn micro_price(&self) -> f64 {
        self.current_micro_price
    }

    /// Get current spread
    pub fn spread(&self) -> f64 {
        self.spread
    }

    /// Get spread in basis points
    pub fn spread_bps(&self) -> f64 {
        if self.current_micro_price <= 0.0 {
            return 0.0;
        }
        (self.spread / self.current_micro_price) * 10000.0
    }

    /// Get total bid volume
    pub fn total_bid_volume(&self) -> f64 {
        self.total_bid_volume
    }

    /// Get total ask volume
    pub fn total_ask_volume(&self) -> f64 {
        self.total_ask_volume
    }

    /// Get volume imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
    pub fn volume_imbalance(&self) -> f64 {
        let total = self.total_bid_volume + self.total_ask_volume;
        if total <= 0.0 {
            return 0.0;
        }
        (self.total_bid_volume - self.total_ask_volume) / total
    }

    /// Get update count
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }

    /// Get last update timestamp
    pub fn last_update_ns(&self) -> u64 {
        self.last_update_ns.load(Ordering::Relaxed)
    }
}

impl Default for MicroPriceCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_micro_price_basic() {
        let mut calc = MicroPriceCalculator::new();
        
        let snapshot = OrderBookSnapshot {
            bids: [
                OrderBookLevel { price: 99.0, size: 100.0 },
                OrderBookLevel { price: 98.0, size: 50.0 },
                OrderBookLevel { price: 97.0, size: 25.0 },
                OrderBookLevel { price: 96.0, size: 10.0 },
                OrderBookLevel { price: 95.0, size: 5.0 },
            ],
            asks: [
                OrderBookLevel { price: 101.0, size: 100.0 },
                OrderBookLevel { price: 102.0, size: 50.0 },
                OrderBookLevel { price: 103.0, size: 25.0 },
                OrderBookLevel { price: 104.0, size: 10.0 },
                OrderBookLevel { price: 105.0, size: 5.0 },
            ],
            bid_count: 5,
            ask_count: 5,
            timestamp_ns: 1000000,
        };

        let micro_price = calc.update(&snapshot).unwrap();
        
        // With equal sizes, micro-price should be exactly mid
        assert!((micro_price - 100.0).abs() < 1e-10);
        assert_eq!(calc.spread(), 2.0);
        assert_eq!(calc.spread_bps(), 200.0);
    }

    #[test]
    fn test_micro_price_imbalance() {
        let mut calc = MicroPriceCalculator::new();
        
        // More ask size pushes micro-price down toward bid
        let snapshot = OrderBookSnapshot {
            bids: [
                OrderBookLevel { price: 99.0, size: 100.0 },
                OrderBookLevel { price: 98.0, size: 50.0 },
                OrderBookLevel { price: 97.0, size: 25.0 },
                OrderBookLevel { price: 96.0, size: 10.0 },
                OrderBookLevel { price: 95.0, size: 5.0 },
            ],
            asks: [
                OrderBookLevel { price: 101.0, size: 1000.0 }, // Large ask wall
                OrderBookLevel { price: 102.0, size: 500.0 },
                OrderBookLevel { price: 103.0, size: 250.0 },
                OrderBookLevel { price: 104.0, size: 100.0 },
                OrderBookLevel { price: 105.0, size: 50.0 },
            ],
            bid_count: 5,
            ask_count: 5,
            timestamp_ns: 1000000,
        };

        let micro_price = calc.update(&snapshot).unwrap();
        
        // Large ask size should push micro-price closer to bid
        assert!(micro_price < 100.0);
        assert!(micro_price > 99.0);
        
        // Volume imbalance should be negative (more ask volume)
        assert!(calc.volume_imbalance() < 0.0);
    }

    #[test]
    fn test_micro_price_zero_liquidity() {
        let mut calc = MicroPriceCalculator::new();
        
        let snapshot = OrderBookSnapshot {
            bids: [OrderBookLevel { price: 0.0, size: 0.0 }; 5],
            asks: [OrderBookLevel { price: 0.0, size: 0.0 }; 5],
            bid_count: 0,
            ask_count: 0,
            timestamp_ns: 1000000,
        };

        assert!(calc.update(&snapshot).is_none());
    }
}
