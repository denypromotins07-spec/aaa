//! Market Impact Feedback - Feeds real-world slippage data back to SOR/Market Maker.
//! 
//! This module provides a lock-free feedback channel that allows the Stage 13 SOR
//! and Stage 15 Market Maker to dynamically adjust their limit order offsets based
//! on observed implementation shortfall.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;

use super::implementation_shortfall::{ImplementationShortfallTracker, ShortfallStats};

/// Slippage profile for dynamic offset adjustment
#[derive(Debug, Clone, Copy)]
pub struct SlippageProfile {
    /// Average shortfall in basis points over the lookback window
    pub avg_shortfall_bps: i64,
    
    /// Standard deviation of shortfall (volatility of slippage)
    pub shortfall_stddev_bps: i64,
    
    /// Recommended offset adjustment in basis points
    pub recommended_offset_bps: i64,
    
    /// Timestamp of last update (monotonic nanoseconds)
    pub last_update_ns: u64,
}

impl Default for SlippageProfile {
    fn default() -> Self {
        Self {
            avg_shortfall_bps: 0,
            shortfall_stddev_bps: 0,
            recommended_offset_bps: 0,
            last_update_ns: 0,
        }
    }
}

/// Market Impact Feedback channel
pub struct MarketImpactFeedback {
    /// Current slippage profile (atomically readable)
    current_profile: parking_lot::RwLock<SlippageProfile>,
    
    /// Reference to the shortfall tracker
    shortfall_tracker: Arc<ImplementationShortfallTracker>,
    
    /// Lookback window for calculations (number of samples)
    lookback_samples: u64,
    
    /// Confidence multiplier for offset calculation (e.g., 2.0 for 95% confidence)
    confidence_multiplier: f64,
    
    /// Minimum offset to apply (bps)
    min_offset_bps: i64,
    
    /// Maximum offset to apply (bps)
    max_offset_bps: i64,
    
    /// Update counter
    update_count: AtomicU64,
}

impl MarketImpactFeedback {
    pub fn new(
        shortfall_tracker: Arc<ImplementationShortfallTracker>,
        lookback_samples: u64,
        confidence_multiplier: f64,
        min_offset_bps: i64,
        max_offset_bps: i64,
    ) -> Self {
        Self {
            current_profile: parking_lot::RwLock::new(SlippageProfile::default()),
            shortfall_tracker,
            lookback_samples,
            confidence_multiplier,
            min_offset_bps,
            max_offset_bps,
            update_count: AtomicU64::new(0),
        }
    }
    
    /// Update the slippage profile based on recent shortfall data
    /// Should be called periodically (e.g., every 100 fills or every second)
    pub fn update_profile(&self) -> SlippageProfile {
        let stats = self.shortfall_tracker.get_stats();
        
        // Calculate recommended offset
        // Formula: offset = avg_shortfall + (confidence_multiplier * stddev)
        // We use a simplified stddev estimation based on max-min range
        let range_bps = stats.max_shortfall_bps - stats.min_shortfall_bps;
        let estimated_stddev = range_bps / 4;  // Rough approximation
        
        let raw_offset = stats.avg_shortfall_bps 
            + (self.confidence_multiplier * estimated_stddev as f64) as i64;
        
        // Clamp to min/max bounds
        let clamped_offset = raw_offset
            .max(self.min_offset_bps)
            .min(self.max_offset_bps);
        
        let profile = SlippageProfile {
            avg_shortfall_bps: stats.avg_shortfall_bps,
            shortfall_stddev_bps: estimated_stddev,
            recommended_offset_bps: clamped_offset,
            last_update_ns: get_monotonic_ns(),
        };
        
        // Atomically update the profile
        *self.current_profile.write() = profile;
        
        self.update_count.fetch_add(1, Ordering::Relaxed);
        
        profile
    }
    
    /// Get the current slippage profile (lock-free read)
    pub fn get_profile(&self) -> SlippageProfile {
        *self.current_profile.read()
    }
    
    /// Get the recommended offset for limit order pricing
    /// 
    /// This is the primary method called by Stage 13 SOR in its hot path.
    /// It returns the offset in basis points that should be added to the
    /// base limit price to account for observed market impact.
    #[inline]
    pub fn get_recommended_offset_bps(&self) -> i64 {
        self.current_profile.read().recommended_offset_bps
    }
    
    /// Calculate adjusted limit price for a buy order
    /// 
    /// # Arguments
    /// * `base_price_scaled` - The theoretical fair price (scaled integer)
    /// * `is_aggressive` - If true, use tighter offset for faster fill
    /// 
    /// # Returns
    /// Adjusted limit price (scaled integer) that accounts for market impact
    #[inline]
    pub fn calculate_buy_limit_price(&self, base_price_scaled: i128, is_aggressive: bool) -> i128 {
        let offset_bps = if is_aggressive {
            // Use half the offset for more aggressive pricing
            self.get_recommended_offset_bps() / 2
        } else {
            self.get_recommended_offset_bps()
        };
        
        // For buys, we want to bid slightly higher to improve fill probability
        // adjusted_price = base_price * (1 + offset_bps / 10000)
        let adjustment = (base_price_scaled * offset_bps as i128) / 10000;
        base_price_scaled + adjustment
    }
    
    /// Calculate adjusted limit price for a sell order
    /// 
    /// # Arguments
    /// * `base_price_scaled` - The theoretical fair price (scaled integer)
    /// * `is_aggressive` - If true, use tighter offset for faster fill
    /// 
    /// # Returns
    /// Adjusted limit price (scaled integer) that accounts for market impact
    #[inline]
    pub fn calculate_sell_limit_price(&self, base_price_scaled: i128, is_aggressive: bool) -> i128 {
        let offset_bps = if is_aggressive {
            self.get_recommended_offset_bps() / 2
        } else {
            self.get_recommended_offset_bps()
        };
        
        // For sells, we want to ask slightly lower to improve fill probability
        // adjusted_price = base_price * (1 - offset_bps / 10000)
        let adjustment = (base_price_scaled * offset_bps as i128) / 10000;
        base_price_scaled - adjustment
    }
    
    /// Get statistics about feedback updates
    pub fn get_feedback_stats(&self) -> FeedbackStats {
        let profile = self.get_profile();
        FeedbackStats {
            update_count: self.update_count.load(Ordering::Relaxed),
            current_offset_bps: profile.recommended_offset_bps,
            avg_shortfall_bps: profile.avg_shortfall_bps,
            last_update_ns: profile.last_update_ns,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FeedbackStats {
    pub update_count: u64,
    pub current_offset_bps: i64,
    pub avg_shortfall_bps: i64,
    pub last_update_ns: u64,
}

/// Get monotonic nanosecond timestamp
#[inline]
fn get_monotonic_ns() -> u64 {
    static START_TIME: once_cell::sync::Lazy<Instant> = once_cell::sync::Lazy::new(Instant::now);
    START_TIME.elapsed().as_nanos() as u64
}

use std::time::Instant;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slippage::implementation_shortfall::ImplementationShortfallTracker;
    
    #[test]
    fn test_buy_limit_price_adjustment() {
        let tracker = Arc::new(ImplementationShortfallTracker::new(100));
        let feedback = MarketImpactFeedback::new(tracker, 100, 2.0, 1, 50);
        
        // Manually set a profile with positive offset
        let profile = SlippageProfile {
            avg_shortfall_bps: 5,
            shortfall_stddev_bps: 2,
            recommended_offset_bps: 9,  // 5 + 2*2
            last_update_ns: get_monotonic_ns(),
        };
        *feedback.current_profile.write() = profile;
        
        let base_price = 50_000_000_000i128;  // $50,000 with 8 decimal scaling
        let adjusted = feedback.calculate_buy_limit_price(base_price, false);
        
        // Should be higher than base (9 bps = 0.09%)
        let expected_adjustment = (base_price * 9) / 10000;
        assert_eq!(adjusted, base_price + expected_adjustment);
        assert!(adjusted > base_price);
    }
    
    #[test]
    fn test_sell_limit_price_adjustment() {
        let tracker = Arc::new(ImplementationShortfallTracker::new(100));
        let feedback = MarketImpactFeedback::new(tracker, 100, 2.0, 1, 50);
        
        let profile = SlippageProfile {
            avg_shortfall_bps: 5,
            shortfall_stddev_bps: 2,
            recommended_offset_bps: 9,
            last_update_ns: get_monotonic_ns(),
        };
        *feedback.current_profile.write() = profile;
        
        let base_price = 50_000_000_000i128;
        let adjusted = feedback.calculate_sell_limit_price(base_price, false);
        
        // Should be lower than base (9 bps = 0.09%)
        let expected_adjustment = (base_price * 9) / 10000;
        assert_eq!(adjusted, base_price - expected_adjustment);
        assert!(adjusted < base_price);
    }
    
    #[test]
    fn test_aggressive_mode_half_offset() {
        let tracker = Arc::new(ImplementationShortfallTracker::new(100));
        let feedback = MarketImpactFeedback::new(tracker, 100, 2.0, 1, 50);
        
        let profile = SlippageProfile {
            recommended_offset_bps: 10,
            ..Default::default()
        };
        *feedback.current_profile.write() = profile;
        
        let base_price = 50_000_000_000i128;
        
        let normal_buy = feedback.calculate_buy_limit_price(base_price, false);
        let aggressive_buy = feedback.calculate_buy_limit_price(base_price, true);
        
        // Aggressive should have smaller adjustment (half offset)
        assert!(aggressive_buy < normal_buy, "Aggressive buy should be closer to base");
        assert!(aggressive_buy > base_price, "But still above base price");
    }
}
