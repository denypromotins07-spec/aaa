//! Volume Bucket Aggregator for VPIN Calculation
//! 
//! This module handles the aggregation of trades into volume buckets,
//! including proper handling of whale orders that span multiple buckets.
//! 
//! ROOT CAUSE FIX: Implements fractional bucket splitting to prevent
//! VPIN math corruption from massive market orders.

use std::sync::atomic::{AtomicU64, Ordering};

/// Trade classification: buyer-initiated or seller-initiated
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// A trade record with classification
#[derive(Debug, Clone, Copy)]
pub struct ClassifiedTrade {
    pub volume: f64,
    pub price: f64,
    pub side: TradeSide,
    pub timestamp_ns: u64,
}

/// Configuration for volume bucket aggregation
#[derive(Debug, Clone, Copy)]
pub struct BucketAggregatorConfig {
    /// Target volume per bucket
    pub bucket_size: f64,
    /// Minimum trades per bucket before finalizing
    pub min_trades_per_bucket: usize,
    /// Maximum trades per bucket (force finalize)
    pub max_trades_per_bucket: usize,
}

impl Default for BucketAggregatorConfig {
    fn default() -> Self {
        Self {
            bucket_size: 10.0,
            min_trades_per_bucket: 1,
            max_trades_per_bucket: 10000,
        }
    }
}

/// Volume bucket with trade tracking
#[derive(Debug, Clone)]
pub struct TrackedBucket {
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub target_volume: f64,
    pub trade_count: usize,
    pub min_trades: usize,
    pub max_trades: usize,
}

impl TrackedBucket {
    pub fn new(target_volume: f64, min_trades: usize, max_trades: usize) -> Self {
        Self {
            buy_volume: 0.0,
            sell_volume: 0.0,
            target_volume,
            trade_count: 0,
            min_trades,
            max_trades,
        }
    }

    pub fn total_volume(&self) -> f64 {
        self.buy_volume + self.sell_volume
    }

    pub fn is_ready_to_finalize(&self) -> bool {
        let volume_full = self.total_volume() >= self.target_volume;
        let trade_count_ok = self.trade_count >= self.min_trades;
        let trade_max_reached = self.trade_count >= self.max_trades;
        
        (volume_full && trade_count_ok) || trade_max_reached
    }

    pub fn add_trade(&mut self, trade: &ClassifiedTrade) -> f64 {
        let remaining_capacity = self.target_volume - self.total_volume();
        
        if remaining_capacity <= 0.0 && self.trade_count >= self.min_trades {
            return trade.volume;
        }

        let volume_to_add = trade.volume.min(remaining_capacity);
        let remaining = trade.volume - volume_to_add;

        match trade.side {
            TradeSide::Buy => self.buy_volume += volume_to_add,
            TradeSide::Sell => self.sell_volume += volume_to_add,
        }

        self.trade_count += 1;

        remaining
    }

    pub fn vpin(&self) -> f64 {
        let total = self.buy_volume + self.sell_volume;
        if total <= 0.0 {
            return 0.0;
        }
        (self.buy_volume - self.sell_volume).abs() / total
    }
}

/// Volume bucket aggregator with whale order handling
pub struct VolumeBucketAggregator {
    config: BucketAggregatorConfig,
    current_bucket: Option<TrackedBucket>,
    completed_vpin_sum: f64,
    completed_count: usize,
    last_finalized_vpin: f64,
    total_trades_processed: AtomicU64,
}

impl VolumeBucketAggregator {
    pub fn new() -> Self {
        Self::with_config(BucketAggregatorConfig::default())
    }

    pub fn with_config(config: BucketAggregatorConfig) -> Self {
        Self {
            config,
            current_bucket: Some(TrackedBucket::new(
                config.bucket_size,
                config.min_trades_per_bucket,
                config.max_trades_per_bucket,
            )),
            completed_vpin_sum: 0.0,
            completed_count: 0,
            last_finalized_vpin: 0.0,
            total_trades_processed: AtomicU64::new(0),
        }
    }

    /// Process a trade, returning VPIN if a bucket was finalized
    /// 
    /// ROOT CAUSE FIX: Splits whale orders across multiple buckets
    pub fn process_trade(&mut self, trade: ClassifiedTrade) -> Vec<f64> {
        self.total_trades_processed.fetch_add(1, Ordering::Relaxed);
        
        let mut finalized_vpins = Vec::new();
        let mut remaining_trade = trade;

        while remaining_trade.volume > 0.0 {
            let bucket = self.current_bucket.get_or_insert_with(|| {
                TrackedBucket::new(
                    self.config.bucket_size,
                    self.config.min_trades_per_bucket,
                    self.config.max_trades_per_bucket,
                )
            });

            let leftover = bucket.add_trade(&remaining_trade);
            
            if leftover > 0.0 {
                // Trade exceeded bucket capacity
                remaining_trade = ClassifiedTrade {
                    volume: leftover,
                    ..remaining_trade
                };
            } else {
                remaining_trade.volume = 0.0;
            }

            // Check if bucket should be finalized
            if bucket.is_ready_to_finalize() {
                let vpin = bucket.vpin();
                finalized_vpins.push(vpin);
                
                self.completed_vpin_sum += vpin;
                self.completed_count += 1;
                self.last_finalized_vpin = vpin;
                
                // Start new bucket
                self.current_bucket = Some(TrackedBucket::new(
                    self.config.bucket_size,
                    self.config.min_trades_per_bucket,
                    self.config.max_trades_per_bucket,
                ));
            }
        }

        finalized_vpins
    }

    /// Get rolling average VPIN
    pub fn rolling_vpin(&self) -> f64 {
        if self.completed_count == 0 {
            return 0.0;
        }
        self.completed_vpin_sum / self.completed_count as f64
    }

    /// Get last finalized VPIN
    pub fn last_vpin(&self) -> f64 {
        self.last_finalized_vpin
    }

    /// Get completed bucket count
    pub fn completed_count(&self) -> usize {
        self.completed_count
    }

    /// Get total trades processed
    pub fn total_trades(&self) -> u64 {
        self.total_trades_processed.load(Ordering::Relaxed)
    }

    /// Get current bucket progress
    pub fn current_bucket_progress(&self) -> f64 {
        if let Some(bucket) = &self.current_bucket {
            bucket.total_volume() / bucket.target_volume
        } else {
            0.0
        }
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.current_bucket = Some(TrackedBucket::new(
            self.config.bucket_size,
            self.config.min_trades_per_bucket,
            self.config.max_trades_per_bucket,
        ));
        self.completed_vpin_sum = 0.0;
        self.completed_count = 0;
        self.last_finalized_vpin = 0.0;
    }
}

impl Default for VolumeBucketAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregator_basic() {
        let mut agg = VolumeBucketAggregator::with_config(BucketAggregatorConfig {
            bucket_size: 100.0,
            min_trades_per_bucket: 1,
            max_trades_per_bucket: 1000,
        });

        let trade1 = ClassifiedTrade {
            volume: 50.0,
            price: 100.0,
            side: TradeSide::Buy,
            timestamp_ns: 1000,
        };

        let trade2 = ClassifiedTrade {
            volume: 50.0,
            price: 100.0,
            side: TradeSide::Sell,
            timestamp_ns: 2000,
        };

        // First trade shouldn't finalize bucket
        let vpins = agg.process_trade(trade1);
        assert!(vpins.is_empty());

        // Second trade should complete bucket
        let vpins = agg.process_trade(trade2);
        assert_eq!(vpins.len(), 1);
        assert_eq!(vpins[0], 0.0); // Balanced = 0 VPIN
    }

    #[test]
    fn test_whale_order_splitting() {
        let mut agg = VolumeBucketAggregator::with_config(BucketAggregatorConfig {
            bucket_size: 100.0,
            min_trades_per_bucket: 1,
            max_trades_per_bucket: 1000,
        });

        // Whale order: 350 units
        let whale_trade = ClassifiedTrade {
            volume: 350.0,
            price: 100.0,
            side: TradeSide::Buy,
            timestamp_ns: 1000,
        };

        let vpins = agg.process_trade(whale_trade);
        
        // Should have finalized 3 buckets
        assert_eq!(vpins.len(), 3);
        
        // Each bucket should have VPIN = 1.0 (all buys)
        for vpin in &vpins {
            assert!((vpin - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_min_trades_requirement() {
        let mut agg = VolumeBucketAggregator::with_config(BucketAggregatorConfig {
            bucket_size: 100.0,
            min_trades_per_bucket: 3,
            max_trades_per_bucket: 1000,
        });

        // Single large trade fills bucket but doesn't meet min trades
        let trade = ClassifiedTrade {
            volume: 100.0,
            price: 100.0,
            side: TradeSide::Buy,
            timestamp_ns: 1000,
        };

        let vpins = agg.process_trade(trade);
        assert!(vpins.is_empty()); // Not enough trades yet

        // Add more small trades
        for i in 0..2 {
            let small_trade = ClassifiedTrade {
                volume: 0.1,
                price: 100.0,
                side: TradeSide::Sell,
                timestamp_ns: 2000 + i,
            };
            let vpins = agg.process_trade(small_trade);
            if i == 1 {
                assert_eq!(vpins.len(), 1); // Now should finalize
            } else {
                assert!(vpins.is_empty());
            }
        }
    }
}
