//! Order Book Imbalance (OBI) Signal Calculator
//! 
//! Calculates the ratio of bid volume to ask volume in the top N levels
//! of the live order book. Extreme imbalances predict immediate 
//! micro-second price movements.
//! 
//! Formula: OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
//! Range: [-1.0, 1.0] where positive = bullish, negative = bearish
//! 
//! ZERO-ALLOC: All calculations use stack-allocated values.

use crate::consumer::micro_price_calculator::OrderBookSnapshot;
use std::sync::atomic::{AtomicU64, Ordering};

/// Configuration for OBI calculation
#[derive(Debug, Clone, Copy)]
pub struct ObiConfig {
    /// Number of top levels to consider (default: 5)
    pub levels: usize,
    /// Volume weighting exponent (default: 1.0 = linear)
    pub volume_exponent: f64,
    /// Price distance weighting (default: 1.0 = closer levels weighted more)
    pub price_weight: f64,
}

impl Default for ObiConfig {
    fn default() -> Self {
        Self {
            levels: 5,
            volume_exponent: 1.0,
            price_weight: 0.5, // Closer levels have more influence
        }
    }
}

/// Order Book Imbalance signal calculator
pub struct OrderBookImbalance {
    config: ObiConfig,
    current_obi: f64,
    total_bid_volume: f64,
    total_ask_volume: f64,
    last_update_ns: AtomicU64,
    update_count: AtomicU64,
}

impl OrderBookImbalance {
    pub fn new() -> Self {
        Self::with_config(ObiConfig::default())
    }

    pub fn with_config(config: ObiConfig) -> Self {
        Self {
            config,
            current_obi: 0.0,
            total_bid_volume: 0.0,
            total_ask_volume: 0.0,
            last_update_ns: AtomicU64::new(0),
            update_count: AtomicU64::new(0),
        }
    }

    /// Calculate OBI from order book snapshot
    /// 
    /// ROOT CAUSE FIX: Properly handles empty levels and zero volumes
    pub fn update(&mut self, snapshot: &OrderBookSnapshot) -> Option<f64> {
        let levels_to_use = self.config.levels.min(5);
        
        if snapshot.bid_count == 0 || snapshot.ask_count == 0 {
            return None;
        }

        let mut weighted_bid_volume = 0.0f64;
        let mut weighted_ask_volume = 0.0f64;
        let mut raw_bid_volume = 0.0f64;
        let mut raw_ask_volume = 0.0f64;

        // Process bid levels
        for i in 0..levels_to_use {
            if i >= snapshot.bid_count {
                break;
            }
            
            let level = snapshot.bids[i];
            if level.size <= 0.0 || level.price <= 0.0 {
                continue;
            }

            raw_bid_volume += level.size;

            // Apply volume weighting
            let volume_weight = level.size.powf(self.config.volume_exponent);
            
            // Apply price distance weighting (closer to mid = higher weight)
            let price_weight = if self.config.price_weight > 0.0 {
                1.0 / (1.0 + i as f64 * self.config.price_weight)
            } else {
                1.0
            };

            weighted_bid_volume += volume_weight * price_weight;
        }

        // Process ask levels
        for i in 0..levels_to_use {
            if i >= snapshot.ask_count {
                break;
            }
            
            let level = snapshot.asks[i];
            if level.size <= 0.0 || level.price <= 0.0 {
                continue;
            }

            raw_ask_volume += level.size;

            // Apply volume weighting
            let volume_weight = level.size.powf(self.config.volume_exponent);
            
            // Apply price distance weighting
            let price_weight = if self.config.price_weight > 0.0 {
                1.0 / (1.0 + i as f64 * self.config.price_weight)
            } else {
                1.0
            };

            weighted_ask_volume += volume_weight * price_weight;
        }

        // Avoid division by zero
        let total_volume = weighted_bid_volume + weighted_ask_volume;
        if total_volume <= 0.0 {
            return None;
        }

        // Calculate OBI: ranges from -1 (all ask) to +1 (all bid)
        self.current_obi = (weighted_bid_volume - weighted_ask_volume) / total_volume;
        self.total_bid_volume = raw_bid_volume;
        self.total_ask_volume = raw_ask_volume;

        self.last_update_ns.store(snapshot.timestamp_ns, Ordering::Relaxed);
        self.update_count.fetch_add(1, Ordering::Relaxed);

        Some(self.current_obi)
    }

    /// Get current OBI value
    pub fn obi(&self) -> f64 {
        self.current_obi
    }

    /// Get OBI as a signal strength [-1, 1]
    pub fn signal_strength(&self) -> f64 {
        self.current_obi.clamp(-1.0, 1.0)
    }

    /// Get buy signal intensity [0, 1]
    pub fn buy_signal(&self) -> f64 {
        if self.current_obi > 0.0 {
            self.current_obi
        } else {
            0.0
        }
    }

    /// Get sell signal intensity [0, 1]
    pub fn sell_signal(&self) -> f64 {
        if self.current_obi < 0.0 {
            -self.current_obi
        } else {
            0.0
        }
    }

    /// Get total bid volume
    pub fn total_bid_volume(&self) -> f64 {
        self.total_bid_volume
    }

    /// Get total ask volume
    pub fn total_ask_volume(&self) -> f64 {
        self.total_ask_volume
    }

    /// Get volume ratio (bid/ask)
    pub fn volume_ratio(&self) -> f64 {
        if self.total_ask_volume <= 0.0 {
            if self.total_bid_volume > 0.0 {
                f64::INFINITY
            } else {
                1.0
            }
        } else {
            self.total_bid_volume / self.total_ask_volume
        }
    }

    /// Check if OBI indicates extreme imbalance
    pub fn is_extreme(&self, threshold: f64) -> bool {
        self.current_obi.abs() > threshold.clamp(0.0, 1.0)
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

impl Default for OrderBookImbalance {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consumer::micro_price_calculator::OrderBookLevel;

    #[test]
    fn test_obi_balanced() {
        let mut obi = OrderBookImbalance::new();
        
        let snapshot = OrderBookSnapshot {
            bids: [
                OrderBookLevel { price: 99.0, size: 100.0 },
                OrderBookLevel { price: 98.0, size: 100.0 },
                OrderBookLevel { price: 97.0, size: 100.0 },
                OrderBookLevel { price: 96.0, size: 100.0 },
                OrderBookLevel { price: 95.0, size: 100.0 },
            ],
            asks: [
                OrderBookLevel { price: 101.0, size: 100.0 },
                OrderBookLevel { price: 102.0, size: 100.0 },
                OrderBookLevel { price: 103.0, size: 100.0 },
                OrderBookLevel { price: 104.0, size: 100.0 },
                OrderBookLevel { price: 105.0, size: 100.0 },
            ],
            bid_count: 5,
            ask_count: 5,
            timestamp_ns: 1000000,
        };

        let value = obi.update(&snapshot).unwrap();
        
        // Balanced book should give OBI near 0
        assert!(value.abs() < 0.01);
        assert_eq!(obi.volume_ratio(), 1.0);
    }

    #[test]
    fn test_obi_bid_heavy() {
        let mut obi = OrderBookImbalance::new();
        
        let snapshot = OrderBookSnapshot {
            bids: [
                OrderBookLevel { price: 99.0, size: 1000.0 }, // Heavy bid wall
                OrderBookLevel { price: 98.0, size: 500.0 },
                OrderBookLevel { price: 97.0, size: 250.0 },
                OrderBookLevel { price: 96.0, size: 100.0 },
                OrderBookLevel { price: 95.0, size: 50.0 },
            ],
            asks: [
                OrderBookLevel { price: 101.0, size: 100.0 },
                OrderBookLevel { price: 102.0, size: 100.0 },
                OrderBookLevel { price: 103.0, size: 100.0 },
                OrderBookLevel { price: 104.0, size: 100.0 },
                OrderBookLevel { price: 105.0, size: 100.0 },
            ],
            bid_count: 5,
            ask_count: 5,
            timestamp_ns: 1000000,
        };

        let value = obi.update(&snapshot).unwrap();
        
        // Heavy bids should give positive OBI
        assert!(value > 0.5);
        assert!(obi.buy_signal() > 0.5);
        assert_eq!(obi.sell_signal(), 0.0);
    }

    #[test]
    fn test_obi_ask_heavy() {
        let mut obi = OrderBookImbalance::new();
        
        let snapshot = OrderBookSnapshot {
            bids: [
                OrderBookLevel { price: 99.0, size: 100.0 },
                OrderBookLevel { price: 98.0, size: 100.0 },
                OrderBookLevel { price: 97.0, size: 100.0 },
                OrderBookLevel { price: 96.0, size: 100.0 },
                OrderBookLevel { price: 95.0, size: 100.0 },
            ],
            asks: [
                OrderBookLevel { price: 101.0, size: 1000.0 }, // Heavy ask wall
                OrderBookLevel { price: 102.0, size: 500.0 },
                OrderBookLevel { price: 103.0, size: 250.0 },
                OrderBookLevel { price: 104.0, size: 100.0 },
                OrderBookLevel { price: 105.0, size: 50.0 },
            ],
            bid_count: 5,
            ask_count: 5,
            timestamp_ns: 1000000,
        };

        let value = obi.update(&snapshot).unwrap();
        
        // Heavy asks should give negative OBI
        assert!(value < -0.5);
        assert!(obi.sell_signal() > 0.5);
        assert_eq!(obi.buy_signal(), 0.0);
    }

    #[test]
    fn test_obi_zero_volume() {
        let mut obi = OrderBookImbalance::new();
        
        let snapshot = OrderBookSnapshot {
            bids: [OrderBookLevel { price: 0.0, size: 0.0 }; 5],
            asks: [OrderBookLevel { price: 0.0, size: 0.0 }; 5],
            bid_count: 0,
            ask_count: 0,
            timestamp_ns: 1000000,
        };

        assert!(obi.update(&snapshot).is_none());
    }
}
