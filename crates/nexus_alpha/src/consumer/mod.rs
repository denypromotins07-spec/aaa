//! Consumer module for Alpha Generation
//! 
//! Provides zero-allocation consumers for reading from SPSC ring buffers
//! and calculating micro-price, rolling windows, and alpha signals.

pub mod spsc_reader;
pub mod micro_price_calculator;
pub mod rolling_window_buffer;

pub use spsc_reader::{SpscReader, SpscReaderConfig};
pub use micro_price_calculator::{MicroPriceCalculator, OrderBookLevel, OrderBookSnapshot};
pub use rolling_window_buffer::RollingWindowBuffer;

/// Main Alpha Consumer that ties together all components
use crate::consumer::*;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info};

/// Configuration for the Alpha Consumer
#[derive(Debug, Clone)]
pub struct AlphaConsumerConfig {
    /// Rolling window size for micro-price history
    pub micro_price_window_size: usize,
    /// Rolling window size for volume history
    pub volume_window_size: usize,
    /// Enable OBI calculation
    pub enable_obi: bool,
    /// Enable VPIN calculation
    pub enable_vpin: bool,
}

impl Default for AlphaConsumerConfig {
    fn default() -> Self {
        Self {
            micro_price_window_size: 100,
            volume_window_size: 1000,
            enable_obi: true,
            enable_vpin: true,
        }
    }
}

/// Main Alpha Consumer - orchestrates all signal calculations
pub struct AlphaConsumer {
    config: AlphaConsumerConfig,
    micro_price_calc: MicroPriceCalculator,
    micro_price_history: RollingWindowBuffer<100>,
    volume_history: RollingWindowBuffer<1000>,
    last_micro_price: f64,
    last_update_ns: AtomicU64,
    update_count: AtomicU64,
}

impl AlphaConsumer {
    pub fn new() -> Self {
        Self::with_config(AlphaConsumerConfig::default())
    }

    pub fn with_config(config: AlphaConsumerConfig) -> Self {
        // Note: In production, you'd use const generics properly
        // For now we use fixed sizes that match defaults
        Self {
            config,
            micro_price_calc: MicroPriceCalculator::new(),
            micro_price_history: RollingWindowBuffer::new(),
            volume_history: RollingWindowBuffer::new(),
            last_micro_price: 0.0,
            last_update_ns: AtomicU64::new(0),
            update_count: AtomicU64::new(0),
        }
    }

    /// Process an order book snapshot and return alpha signals
    pub fn process_orderbook(&mut self, snapshot: &OrderBookSnapshot) -> Option<AlphaSignals> {
        // Update micro-price
        let micro_price = self.micro_price_calc.update(snapshot)?;
        
        // Update rolling windows
        self.micro_price_history.push(micro_price);
        self.volume_history.push(self.micro_price_calc.total_bid_volume());
        
        self.last_micro_price = micro_price;
        self.last_update_ns.store(snapshot.timestamp_ns, Ordering::Relaxed);
        self.update_count.fetch_add(1, Ordering::Relaxed);

        Some(AlphaSignals {
            micro_price,
            spread: self.micro_price_calc.spread(),
            spread_bps: self.micro_price_calc.spread_bps(),
            volume_imbalance: self.micro_price_calc.volume_imbalance(),
            micro_price_mean: self.micro_price_history.mean(),
            micro_price_std: self.micro_price_history.std_dev(),
            timestamp_ns: snapshot.timestamp_ns,
        })
    }

    /// Get current micro-price
    pub fn micro_price(&self) -> f64 {
        self.last_micro_price
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

impl Default for AlphaConsumer {
    fn default() -> Self {
        Self::new()
    }
}

/// Alpha signals output structure
#[derive(Debug, Clone, Copy)]
pub struct AlphaSignals {
    pub micro_price: f64,
    pub spread: f64,
    pub spread_bps: f64,
    pub volume_imbalance: f64,
    pub micro_price_mean: Option<f64>,
    pub micro_price_std: Option<f64>,
    pub timestamp_ns: u64,
}

impl AlphaSignals {
    /// Check if signals are valid
    pub fn is_valid(&self) -> bool {
        self.micro_price > 0.0 && self.spread > 0.0
    }

    /// Get signal quality score [0.0, 1.0]
    pub fn quality_score(&self) -> f64 {
        let mut score = 1.0;
        
        // Penalize wide spreads
        if self.spread_bps > 100.0 {
            score -= 0.2;
        }
        
        // Penalize high volatility
        if let Some(std) = self.micro_price_std {
            if self.micro_price > 0.0 {
                let cv = std / self.micro_price; // Coefficient of variation
                if cv > 0.01 {
                    score -= cv * 10.0;
                }
            }
        }
        
        score.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_consumer_basic() {
        let mut consumer = AlphaConsumer::new();
        
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

        let signals = consumer.process_orderbook(&snapshot).unwrap();
        
        assert!(signals.is_valid());
        assert!((signals.micro_price - 100.0).abs() < 1e-10);
        assert_eq!(consumer.update_count(), 1);
    }
}
