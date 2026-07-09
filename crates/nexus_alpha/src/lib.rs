//! NEXUS-OMEGA Stage 3: Advanced Alpha Generation, SMC Feature Extraction & Signal Fusion
//! 
//! This crate implements institutional-grade alpha extraction from zero-copy order books,
//! including Smart Money Concepts, Order Flow Toxicity, Lead-Lag dynamics, and Bayesian
//! signal fusion.

pub mod smc {
    //! Smart Money Concepts module
    
    pub mod order_blocks;
    pub mod liquidity_voids;
}

pub mod orderflow {
    //! Order flow analysis module
    
    pub mod volume_profile_simd;
}

pub mod micro {
    //! Micro-structure alpha module
    
    pub mod hawkes_intensity;
    pub mod vpin_toxicity;
    pub mod kalman_efficient_price;
}

pub mod fusion {
    //! Signal fusion and conviction scoring module
    
    pub mod regime_hmm;
    pub mod bayesian_conviction;
    pub mod signal_aggregator;
}

// Re-export main types for convenience
pub use smc::order_blocks::{OrderBlock, OrderBlockDetector};
pub use smc::liquidity_voids::{FairValueGap, FvgDetector, LiquidityVoid, LiquidityVoidDetector};
pub use orderflow::volume_profile_simd::{VolumeProfileSimd, TpoCalculator, MarketProfileStats};
pub use micro::hawkes_intensity::{HawkesIntensity, HawkesParams, BurstSignal};
pub use micro::vpin_toxicity::{VpinCalculator, VpinResult, ToxicityLevel};
pub use micro::kalman_efficient_price::{DualKalmanFilter, DualKalmanResult, ArbSignal};
pub use fusion::regime_hmm::{RegimeHmm, MarketRegime, Observation};
pub use fusion::bayesian_conviction::{BayesianConviction, ConvictionResult, AlphaSignal};
pub use fusion::signal_aggregator::{SignalAggregator, AggregatedSignals};

/// Nexus Alpha Engine - Main entry point
/// 
/// Combines all alpha sources into a unified engine for signal generation.
pub struct NexusAlphaEngine {
    /// Order block detector
    order_block_detector: OrderBlockDetector,
    /// FVG detector
    fvg_detector: FvgDetector,
    /// Liquidity void detector
    liquidity_detector: LiquidityVoidDetector,
    /// Volume profile calculator
    volume_profile: VolumeProfileSimd,
    /// Hawkes intensity calculator
    hawkes: HawkesIntensity,
    /// VPIN calculator
    vpin: VpinCalculator,
    /// Signal aggregator
    aggregator: SignalAggregator,
    /// Regime HMM
    regime_hmm: RegimeHmm,
}

impl NexusAlphaEngine {
    /// Create a new Nexus Alpha Engine
    pub fn new(allocator: &nexus_core::memory::arena::BumpAllocator) -> Self {
        let min_price = 50_0000_0000i64;
        let max_price = 150_0000_0000i64;
        
        Self {
            order_block_detector: OrderBlockDetector::new(allocator),
            fvg_detector: FvgDetector::new(allocator),
            liquidity_detector: LiquidityVoidDetector::new(allocator, min_price, max_price),
            volume_profile: VolumeProfileSimd::new(allocator, min_price, max_price),
            hawkes: HawkesIntensity::new(allocator, HawkesParams::default()),
            vpin: VpinCalculator::new(allocator, 10000),
            aggregator: SignalAggregator::new(allocator),
            regime_hmm: RegimeHmm::new(allocator, None),
        }
    }
    
    /// Process a quote tick
    #[inline]
    pub fn on_quote_tick(
        &mut self,
        ts: u64,
        bid_price: i64,
        ask_price: i64,
        bid_size: i64,
        ask_size: i64,
    ) -> Option<ConvictionResult> {
        let mid_price = (bid_price + ask_price) / 2;
        let total_size = bid_size + ask_size;
        let buy_ratio = bid_size as f64 / total_size as f64;
        
        // Update all alpha models
        self.order_block_detector.on_tick(ts, mid_price, total_size, buy_ratio > 0.5);
        self.volume_profile.on_tick(ts, mid_price, total_size, buy_ratio > 0.5);
        self.liquidity_detector.on_tick(ts, mid_price, total_size);
        
        // Get current price for signal evaluation
        let current_price = mid_price;
        
        // Update signals in aggregator
        if let Some(block) = self.order_block_detector.get_recent_blocks(1).next() {
            self.aggregator.update_order_block(block, current_price, ts);
        }
        
        if let Some(fvg) = self.fvg_detector.get_unfilled_fvgs().next() {
            self.aggregator.update_fvg(fvg, current_price, ts);
        }
        
        // Compute regime and conviction
        let regime = self.regime_hmm.get_current_regime();
        self.aggregator.compute_conviction(regime, ts).into()
    }
    
    /// Process a trade tick
    #[inline]
    pub fn on_trade_tick(
        &mut self,
        ts: u64,
        price: i64,
        size: i64,
        is_buy: bool,
    ) -> Option<ConvictionResult> {
        // Update Hawkes process
        self.hawkes.on_trade(ts, size as f64, is_buy);
        
        // Update VPIN
        self.vpin.on_trade(ts, price, size);
        
        // Update volume profile
        self.volume_profile.on_tick(ts, price, size, is_buy);
        
        // Get burst signal
        let burst = BurstSignal::from_hawkes(&self.hawkes, ts, 3.0);
        self.aggregator.update_hawkes(&burst);
        
        // Get VPIN result
        let vpin_result = self.vpin.get_vpin_with_stats();
        self.aggregator.update_vpin(&vpin_result);
        
        // Compute regime and conviction
        let regime = self.regime_hmm.get_current_regime();
        self.aggregator.compute_conviction(regime, ts).into()
    }
    
    /// Get current conviction score
    #[inline]
    pub fn get_conviction(&self) -> f64 {
        self.aggregator.get_signals().iter()
            .map(|s| s.value * s.confidence)
            .sum()
    }
    
    /// Get current market regime
    #[inline]
    pub fn get_regime(&self) -> MarketRegime {
        self.regime_hmm.get_current_regime()
    }
    
    /// Get volume profile POC
    #[inline]
    pub fn get_poc(&self) -> i64 {
        self.volume_profile.get_poc()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_core::memory::arena::BumpAllocator;

    #[test]
    fn test_nexus_engine_initialization() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let engine = NexusAlphaEngine::new(&allocator);
        
        assert_eq!(engine.get_conviction(), 0.0);
        assert_eq!(engine.get_regime(), MarketRegime::MeanReverting);
    }

    #[test]
    fn test_nexus_quote_processing() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut engine = NexusAlphaEngine::new(&allocator);
        
        let base_ts = 1_000_000_000_000u64;
        
        // Process several quote ticks
        for i in 0..10 {
            let ts = base_ts + i * 1_000_000;
            let bid = 100_0000_0000 + i * 1000;
            let ask = bid + 500;
            
            let _result = engine.on_quote_tick(ts, bid, ask, 100, 150);
        }
        
        // Engine should have processed data
        let poc = engine.get_poc();
        assert!(poc > 0);
    }

    #[test]
    fn test_nexus_trade_processing() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut engine = NexusAlphaEngine::new(&allocator);
        
        let base_ts = 1_000_000_000_000u64;
        
        // Process trade bursts
        for i in 0..20 {
            let ts = base_ts + i * 100_000;
            let price = 100_0000_0000 + (i % 5) * 1000;
            
            let _result = engine.on_trade_tick(ts, price, 50, i % 2 == 0);
        }
        
        // Hawkes should detect increased activity
        let regime = engine.get_regime();
        // Regime may have changed based on observations
        assert!(matches!(regime, MarketRegime::MeanReverting | MarketRegime::Trending | MarketRegime::HighVolatility));
    }
}
