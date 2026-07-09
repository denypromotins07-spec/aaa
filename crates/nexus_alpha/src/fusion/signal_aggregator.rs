//! Signal Aggregator - Combines all alpha signals into unified output
//! 
//! Aggregates signals from SMC, micro-structure, and other alpha sources
//! and feeds them into the Bayesian conviction scorer.

use nexus_core::memory::arena::BumpAllocator;
use crate::smc::order_blocks::OrderBlock;
use crate::smc::liquidity_voids::FairValueGap;
use crate::micro::hawkes_intensity::BurstSignal;
use crate::micro::vpin_toxicity::{VpinResult, ToxicityLevel};
use crate::micro::kalman_efficient_price::{DualKalmanResult, ArbSignal};
use crate::fusion::regime_hmm::MarketRegime;
use crate::fusion::bayesian_conviction::{BayesianConviction, AlphaSignal, ConvictionResult};

/// Maximum number of signals to aggregate
pub const MAX_SIGNALS: usize = 8;

/// Signal type identifiers
pub const SIGNAL_SMC_ORDER_BLOCK: u8 = 1;
pub const SIGNAL_SMC_FVG: u8 = 2;
pub const SIGNAL_HAWKES_BURST: u8 = 3;
pub const SIGNAL_VPIN_TOXICITY: u8 = 4;
pub const SIGNAL_KALMAN_SPREAD: u8 = 5;
pub const SIGNAL_MOMENTUM: u8 = 6;
pub const SIGNAL_MEAN_REVERSION: u8 = 7;
pub const SIGNAL_FLOW_IMBALANCE: u8 = 8;

/// Aggregated signal container
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct AggregatedSignals {
    /// Individual signal values
    pub signals: [AlphaSignal; MAX_SIGNALS],
    /// Number of active signals
    pub count: usize,
    /// Timestamp
    pub ts: u64,
}

impl Default for AggregatedSignals {
    fn default() -> Self {
        Self {
            signals: [AlphaSignal::default(); MAX_SIGNALS],
            count: 0,
            ts: 0,
        }
    }
}

/// Signal Aggregator
pub struct SignalAggregator {
    /// Bayesian conviction calculator
    conviction: BayesianConviction,
    /// Current aggregated signals
    aggregated: AggregatedSignals,
    /// Signal enable flags
    enabled_signals: [bool; MAX_SIGNALS],
}

unsafe impl Send for SignalAggregator {}
unsafe impl Sync for SignalAggregator {}

impl SignalAggregator {
    pub fn new(allocator: &BumpAllocator) -> Self {
        let mut aggregator = Self {
            conviction: BayesianConviction::new(allocator, MAX_SIGNALS),
            aggregated: AggregatedSignals::default(),
            enabled_signals: [true; MAX_SIGNALS],
        };
        
        // Initialize with default regime adjustments
        // SMC signals work better in trending regimes
        aggregator.conviction.set_regime_adjustment(0, MarketRegime::Trending, 0.2);
        aggregator.conviction.set_regime_adjustment(0, MarketRegime::MeanReverting, -0.1);
        
        // Mean reversion signals work better in mean-reverting regimes
        aggregator.conviction.set_regime_adjustment(6, MarketRegime::MeanReverting, 0.25);
        aggregator.conviction.set_regime_adjustment(6, MarketRegime::Trending, -0.15);
        
        aggregator
    }

    /// Update with order block signal
    #[inline]
    pub fn update_order_block(&mut self, block: &OrderBlock, current_price: i64, ts: u64) {
        if !self.enabled_signals[0] {
            return;
        }

        // Calculate signal from order block
        let signal_value = if block.is_bullish && !block.mitigated {
            // Bullish OB not yet mitigated - bullish signal if price approaching
            if current_price <= block.high_price && current_price > block.low_price {
                0.7
            } else {
                0.3
            }
        } else if !block.is_bullish && !block.mitigated {
            // Bearish OB not yet mitigated - bearish signal if price approaching
            if current_price >= block.low_price && current_price < block.high_price {
                -0.7
            } else {
                -0.3
            }
        } else {
            0.0
        };

        let confidence = block.strength as f64 / 100.0;
        
        self.aggregated.signals[0] = AlphaSignal {
            value: signal_value,
            confidence: confidence.clamp(0.3, 0.9),
            signal_type: SIGNAL_SMC_ORDER_BLOCK,
            ts,
            recent_accuracy: 0.6, // Will be updated by Bayesian learning
        };
        
        if self.aggregated.count == 0 {
            self.aggregated.count = 1;
        }
        self.aggregated.ts = ts;
    }

    /// Update with FVG signal
    #[inline]
    pub fn update_fvg(&mut self, fvg: &FairValueGap, current_price: i64, ts: u64) {
        if !self.enabled_signals[1] {
            return;
        }

        let signal_value = if fvg.is_bullish && !fvg.filled {
            if current_price <= fvg.high && current_price >= fvg.low {
                0.6
            } else {
                0.2
            }
        } else if !fvg.is_bullish && !fvg.filled {
            if current_price >= fvg.low && current_price <= fvg.high {
                -0.6
            } else {
                -0.2
            }
        } else {
            0.0
        };

        let confidence = fvg.strength as f64 / 100.0;
        
        self.aggregated.signals[1] = AlphaSignal {
            value: signal_value,
            confidence: confidence.clamp(0.3, 0.85),
            signal_type: SIGNAL_SMC_FVG,
            ts,
            recent_accuracy: 0.55,
        };
        
        if self.aggregated.count < 2 {
            self.aggregated.count = 2;
        }
    }

    /// Update with Hawkes burst signal
    #[inline]
    pub fn update_hawkes(&mut self, burst: &BurstSignal) {
        if !self.enabled_signals[2] {
            return;
        }

        let signal_value = if burst.is_burst {
            burst.direction.clamp(-1.0, 1.0) * 0.8
        } else {
            0.0
        };

        self.aggregated.signals[2] = AlphaSignal {
            value: signal_value,
            confidence: burst.confidence.clamp(0.2, 0.9),
            signal_type: SIGNAL_HAWKES_BURST,
            ts: burst.ts,
            recent_accuracy: 0.6,
        };
        
        if self.aggregated.count < 3 {
            self.aggregated.count = 3;
        }
    }

    /// Update with VPIN toxicity signal
    #[inline]
    pub fn update_vpin(&mut self, vpin_result: &VpinResult) {
        if !self.enabled_signals[3] {
            return;
        }

        // High toxicity suggests potential reversal
        let signal_value = match vpin_result.toxicity_level {
            ToxicityLevel::Extreme => -0.5, // Very high toxicity often precedes reversal
            ToxicityLevel::High => -0.3,
            ToxicityLevel::Moderate => 0.0,
            ToxicityLevel::Low => 0.1,
            ToxicityLevel::Minimal => 0.2,
        };

        let confidence = vpin_result.vpin;
        
        self.aggregated.signals[3] = AlphaSignal {
            value: signal_value,
            confidence: confidence.clamp(0.2, 0.8),
            signal_type: SIGNAL_VPIN_TOXICITY,
            ts: 0, // VPIN doesn't have specific timestamp
            recent_accuracy: 0.5,
        };
        
        if self.aggregated.count < 4 {
            self.aggregated.count = 4;
        }
    }

    /// Update with Kalman spread/arbitrage signal
    #[inline]
    pub fn update_kalman(&mut self, kalman_result: &DualKalmanResult) {
        if !self.enabled_signals[4] {
            return;
        }

        let signal_value = match kalman_result.arb_signal {
            ArbSignal::BuyExchange1SellExchange2 => 0.8,
            ArbSignal::SellExchange1BuyExchange2 => -0.8,
            ArbSignal::WeakBuyExchange1 => 0.4,
            ArbSignal::WeakSellExchange1 => -0.4,
            ArbSignal::None => 0.0,
        };

        let confidence = kalman_result.lead_lag_confidence;
        
        self.aggregated.signals[4] = AlphaSignal {
            value: signal_value,
            confidence: confidence.clamp(0.3, 0.9),
            signal_type: SIGNAL_KALMAN_SPREAD,
            ts: kalman_result.ts,
            recent_accuracy: 0.55,
        };
        
        if self.aggregated.count < 5 {
            self.aggregated.count = 5;
        }
    }

    /// Update momentum signal
    #[inline]
    pub fn update_momentum(&mut self, momentum: f64, confidence: f64, ts: u64) {
        if !self.enabled_signals[5] {
            return;
        }

        self.aggregated.signals[5] = AlphaSignal {
            value: momentum.clamp(-1.0, 1.0) * 0.7,
            confidence: confidence.clamp(0.2, 0.9),
            signal_type: SIGNAL_MOMENTUM,
            ts,
            recent_accuracy: 0.5,
        };
        
        if self.aggregated.count < 6 {
            self.aggregated.count = 6;
        }
    }

    /// Update mean reversion signal
    #[inline]
    pub fn update_mean_reversion(&mut self, zscore: f64, confidence: f64, ts: u64) {
        if !self.enabled_signals[6] {
            return;
        }

        // Mean reversion: bet against extreme moves
        let signal_value = -zscore.clamp(-3.0, 3.0) / 3.0 * 0.6;
        
        self.aggregated.signals[6] = AlphaSignal {
            value: signal_value,
            confidence: confidence.clamp(0.2, 0.85),
            signal_type: SIGNAL_MEAN_REVERSION,
            ts,
            recent_accuracy: 0.55,
        };
        
        if self.aggregated.count < 7 {
            self.aggregated.count = 7;
        }
    }

    /// Update order flow imbalance signal
    #[inline]
    pub fn update_flow_imbalance(&mut self, imbalance: f64, confidence: f64, ts: u64) {
        if !self.enabled_signals[7] {
            return;
        }

        self.aggregated.signals[7] = AlphaSignal {
            value: imbalance.clamp(-1.0, 1.0) * 0.5,
            confidence: confidence.clamp(0.2, 0.9),
            signal_type: SIGNAL_FLOW_IMBALANCE,
            ts,
            recent_accuracy: 0.5,
        };
        
        if self.aggregated.count < 8 {
            self.aggregated.count = 8;
        }
    }

    /// Compute final conviction score
    #[inline]
    pub fn compute_conviction(&mut self, regime: MarketRegime, ts: u64) -> ConvictionResult {
        let signals = &self.aggregated.signals[..self.aggregated.count];
        self.conviction.update(signals, regime, ts)
    }

    /// Get all current signals
    #[inline]
    pub fn get_signals(&self) -> &[AlphaSignal] {
        &self.aggregated.signals[..self.aggregated.count]
    }

    /// Enable/disable a specific signal type
    #[inline]
    pub fn set_signal_enabled(&mut self, signal_type: u8, enabled: bool) {
        let idx = (signal_type as usize).saturating_sub(1);
        if idx < MAX_SIGNALS {
            self.enabled_signals[idx] = enabled;
        }
    }

    /// Reset all signals
    #[inline]
    pub fn reset(&mut self) {
        self.aggregated = AggregatedSignals::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_core::memory::arena::BumpAllocator;
    use crate::smc::order_blocks::OrderBlock;

    #[test]
    fn test_signal_aggregator_initialization() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let aggregator = SignalAggregator::new(&allocator);
        
        assert_eq!(aggregator.aggregated.count, 0);
    }

    #[test]
    fn test_order_block_signal() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut aggregator = SignalAggregator::new(&allocator);
        
        let block = OrderBlock {
            is_bullish: true,
            mitigated: false,
            strength: 80,
            high_price: 100_0000_0000,
            low_price: 99_0000_0000,
            ..Default::default()
        };
        
        aggregator.update_order_block(&block, 99_5000_0000, 1000);
        
        assert_eq!(aggregator.aggregated.count, 1);
        assert!(aggregator.aggregated.signals[0].value > 0.0);
    }

    #[test]
    fn test_conviction_computation() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut aggregator = SignalAggregator::new(&allocator);
        
        // Add multiple bullish signals
        let block = OrderBlock {
            is_bullish: true,
            mitigated: false,
            strength: 80,
            high_price: 100_0000_0000,
            low_price: 99_0000_0000,
            ..Default::default()
        };
        aggregator.update_order_block(&block, 99_5000_0000, 1000);
        aggregator.update_momentum(0.6, 0.8, 1000);
        aggregator.update_flow_imbalance(0.5, 0.7, 1000);
        
        let result = aggregator.compute_conviction(MarketRegime::Trending, 1000);
        
        // Should have positive conviction with multiple bullish signals
        assert!(result.conviction > 0.3);
    }

    #[test]
    fn test_signal_disable() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut aggregator = SignalAggregator::new(&allocator);
        
        aggregator.set_signal_enabled(SIGNAL_SMC_ORDER_BLOCK, false);
        aggregator.update_order_block(&OrderBlock::default(), 100, 1000);
        
        // Signal should not be added when disabled
        assert_eq!(aggregator.aggregated.count, 0);
    }
}
