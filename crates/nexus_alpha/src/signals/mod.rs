//! Signals module for Alpha Generation
//! 
//! Provides institutional-grade predictive signals including:
//! - Order Book Imbalance (OBI)
//! - VPIN (Volume-Synchronized Probability of Informed Trading)
//! - Volume Bucket Aggregation

pub mod order_book_imbalance;
pub mod vpin_toxicity;
pub mod volume_bucket_aggregator;

pub use order_book_imbalance::{OrderBookImbalance, ObiConfig};
pub use vpin_toxicity::{VpinToxicity, VpinConfig, VolumeBucket};
pub use volume_bucket_aggregator::{
    VolumeBucketAggregator, BucketAggregatorConfig, ClassifiedTrade, TradeSide,
};

/// Combined signal output structure
#[derive(Debug, Clone, Copy)]
pub struct CombinedSignals {
    pub obi: f64,
    pub vpin: f64,
    pub is_toxic: bool,
    pub spread_multiplier: f64,
    pub timestamp_ns: u64,
}

impl CombinedSignals {
    pub fn new(obi: f64, vpin: f64, is_toxic: bool, timestamp_ns: u64) -> Self {
        let spread_multiplier = if is_toxic {
            1.0 + (vpin * 2.0)
        } else {
            1.0
        };

        Self {
            obi,
            vpin,
            is_toxic,
            spread_multiplier,
            timestamp_ns,
        }
    }

    /// Get combined signal strength [-1, 1] accounting for toxicity
    pub fn adjusted_signal(&self) -> f64 {
        if self.is_toxic {
            // Reduce signal strength when market is toxic
            self.obi * (1.0 - self.vpin * 0.5)
        } else {
            self.obi
        }
    }

    /// Check if trading should proceed
    pub fn should_trade(&self, vpin_halt_threshold: f64) -> bool {
        !self.is_toxic || self.vpin < vpin_halt_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combined_signals_basic() {
        let signals = CombinedSignals::new(0.5, 0.3, false, 1000);
        
        assert_eq!(signals.obi, 0.5);
        assert_eq!(signals.vpin, 0.3);
        assert!(!signals.is_toxic);
        assert_eq!(signals.spread_multiplier, 1.0);
        assert!(signals.should_trade(0.8));
    }

    #[test]
    fn test_combined_signals_toxic() {
        let signals = CombinedSignals::new(0.8, 0.9, true, 1000);
        
        assert!(signals.is_toxic);
        assert!(signals.spread_multiplier > 2.0);
        
        // Signal should be reduced due to toxicity
        assert!(signals.adjusted_signal() < signals.obi);
        
        // Should not trade with high VPIN threshold
        assert!(!signals.should_trade(0.7));
    }
}
