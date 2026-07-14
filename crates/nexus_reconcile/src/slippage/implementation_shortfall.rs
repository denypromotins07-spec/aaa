//! Implementation Shortfall Tracker - Measures slippage between signal and fill.
//! 
//! CRITICAL: Captures the exact nanosecond the Alpha signal is emitted (Paper Price)
//! and compares it against the Actual Fill Price when the FILL report arrives.
//! This measures true implementation shortfall including latency decay.

use std::sync::atomic::{AtomicU64, AtomicI128, Ordering};
use std::time::Instant;
use std::sync::Arc;

use super::nanosecond_latency_tracker::NanosecondLatencyTracker;

/// Tracks a single signal-to-fill lifecycle
#[derive(Debug, Clone)]
pub struct SignalFillRecord {
    /// Unique signal ID
    pub signal_id: u64,
    
    /// Paper price at signal emission (scaled integer)
    pub paper_price_scaled: i128,
    
    /// Actual fill price (scaled integer, 0 if not filled)
    pub fill_price_scaled: i128,
    
    /// Signal emission time (monotonic nanoseconds since boot)
    pub signal_time_ns: u64,
    
    /// Fill reception time (monotonic nanoseconds since boot)
    pub fill_time_ns: u64,
    
    /// Execution quantity (scaled integer)
    pub qty_scaled: i128,
    
    /// Fee paid (scaled integer)
    pub fee_scaled: i128,
}

impl SignalFillRecord {
    /// Calculate implementation shortfall in basis points
    /// Positive = unfavorable (paid more than expected for buy, received less for sell)
    pub fn calculate_shortfall_bps(&self, is_buy: bool) -> i64 {
        if self.fill_price_scaled == 0 || self.paper_price_scaled == 0 {
            return 0;
        }
        
        let price_diff = if is_buy {
            self.fill_price_scaled - self.paper_price_scaled
        } else {
            self.paper_price_scaled - self.fill_price_scaled
        };
        
        // Convert to basis points (1 bp = 0.01%)
        // shortfall_bps = (price_diff / paper_price) * 10000
        let shortfall_bps = (price_diff * 10000) / self.paper_price_scaled;
        
        // Add fee impact (fees always increase shortfall)
        let fee_bps = if self.qty_scaled > 0 {
            (self.fee_scaled * 10000) / (self.fill_price_scaled * self.qty_scaled / 100_000_000)
        } else {
            0
        };
        
        shortfall_bps as i64 + fee_bps as i64
    }
    
    /// Get latency in nanoseconds
    pub fn latency_ns(&self) -> u64 {
        if self.fill_time_ns > self.signal_time_ns {
            self.fill_time_ns - self.signal_time_ns
        } else {
            0
        }
    }
}

/// Statistics about implementation shortfall
#[derive(Debug, Clone, Default)]
pub struct ShortfallStats {
    pub total_signals: u64,
    pub filled_signals: u64,
    pub avg_shortfall_bps: i64,
    pub max_shortfall_bps: i64,
    pub min_shortfall_bps: i64,
    pub avg_latency_us: u64,
    pub total_slippage_cost_scaled: i128,
}

/// Implementation Shortfall Tracker
pub struct ImplementationShortfallTracker {
    /// Latency tracker for signal-to-fill timing
    latency_tracker: NanosecondLatencyTracker,
    
    /// Total signals tracked
    signal_count: AtomicU64,
    
    /// Total fills received
    fill_count: AtomicU64,
    
    /// Sum of shortfall values (in bps * 100 for precision)
    sum_shortfall_bps_x100: AtomicI128,
    
    /// Maximum shortfall observed (in bps)
    max_shortfall_bps: AtomicI128,
    
    /// Minimum shortfall observed (in bps)
    min_shortfall_bps: AtomicI128,
    
    /// Total slippage cost in scaled units
    total_slippage_cost: AtomicI128,
    
    /// Rolling window size for average calculation
    rolling_window_size: u64,
}

impl ImplementationShortfallTracker {
    pub fn new(rolling_window_size: u64) -> Self {
        Self {
            latency_tracker: NanosecondLatencyTracker::new(),
            signal_count: AtomicU64::new(0),
            fill_count: AtomicU64::new(0),
            sum_shortfall_bps_x100: AtomicI128::new(0),
            max_shortfall_bps: AtomicI128::new(i128::MIN),
            min_shortfall_bps: AtomicI128::new(i128::MAX),
            total_slippage_cost: AtomicI128::new(0),
            rolling_window_size,
        }
    }
    
    /// Record a new alpha signal emission
    /// 
    /// # Arguments
    /// * `signal_id` - Unique identifier for this signal
    /// * `paper_price_scaled` - Micro-price at signal emission (scaled integer)
    /// * `qty_scaled` - Order quantity (scaled integer)
    /// * `is_buy` - True for buy orders, false for sells
    #[inline]
    pub fn record_signal(
        &self,
        signal_id: u64,
        paper_price_scaled: i128,
        qty_scaled: i128,
        _is_buy: bool,
    ) -> SignalContext {
        self.signal_count.fetch_add(1, Ordering::Relaxed);
        
        SignalContext {
            signal_id,
            paper_price_scaled,
            qty_scaled,
            signal_time_ns: get_monotonic_ns(),
            start_instant: self.latency_tracker.mark_start(),
        }
    }
    
    /// Record a fill execution report
    /// 
    /// # Arguments
    /// * `context` - The SignalContext from record_signal()
    /// * `fill_price_scaled` - Actual execution price (scaled integer)
    /// * `fee_scaled` - Commission/fee paid (scaled integer)
    /// * `is_buy` - True for buy orders, false for sells
    #[inline]
    pub fn record_fill(
        &self,
        context: SignalContext,
        fill_price_scaled: i128,
        fee_scaled: i128,
        is_buy: bool,
    ) -> SignalFillRecord {
        self.fill_count.fetch_add(1, Ordering::Relaxed);
        
        let fill_time_ns = get_monotonic_ns();
        let latency_ns = self.latency_tracker.mark_end(context.start_instant);
        
        let record = SignalFillRecord {
            signal_id: context.signal_id,
            paper_price_scaled: context.paper_price_scaled,
            fill_price_scaled,
            signal_time_ns: context.signal_time_ns,
            fill_time_ns,
            qty_scaled: context.qty_scaled,
            fee_scaled,
        };
        
        // Calculate shortfall
        let shortfall_bps = record.calculate_shortfall_bps(is_buy);
        
        // Update statistics atomically
        self.update_stats(shortfall_bps, latency_ns, &record);
        
        record
    }
    
    /// Update internal statistics
    fn update_stats(&self, shortfall_bps: i64, latency_ns: u64, record: &SignalFillRecord) {
        // Update shortfall stats
        let shortfall_x100 = shortfall_bps as i128 * 100;
        self.sum_shortfall_bps_x100.fetch_add(shortfall_x100, Ordering::Relaxed);
        
        // Update max
        let mut current_max = self.max_shortfall_bps.load(Ordering::Relaxed);
        while shortfall_bps as i128 > current_max {
            match self.max_shortfall_bps.compare_exchange_weak(
                current_max,
                shortfall_bps as i128,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_current) => current_max = new_current,
            }
        }
        
        // Update min
        let mut current_min = self.min_shortfall_bps.load(Ordering::Relaxed);
        while shortfall_bps as i128 < current_min {
            match self.min_shortfall_bps.compare_exchange_weak(
                current_min,
                shortfall_bps as i128,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_current) => current_min = new_current,
            }
        }
        
        // Calculate slippage cost
        let slippage_cost = (record.fill_price_scaled - record.paper_price_scaled) 
            * record.qty_scaled / 100_000_000;
        self.total_slippage_cost.fetch_add(slippage_cost, Ordering::Relaxed);
    }
    
    /// Get current statistics
    pub fn get_stats(&self) -> ShortfallStats {
        let signals = self.signal_count.load(Ordering::Relaxed);
        let fills = self.fill_count.load(Ordering::Relaxed);
        
        let avg_shortfall = if fills > 0 {
            (self.sum_shortfall_bps_x100.load(Ordering::Relaxed) / fills as i128) as i64 / 100
        } else {
            0
        };
        
        ShortfallStats {
            total_signals: signals,
            filled_signals: fills,
            avg_shortfall_bps: avg_shortfall,
            max_shortfall_bps: self.max_shortfall_bps.load(Ordering::Relaxed) as i64,
            min_shortfall_bps: self.min_shortfall_bps.load(Ordering::Relaxed) as i64,
            avg_latency_us: self.latency_tracker.avg_ns() / 1000,
            total_slippage_cost_scaled: self.total_slippage_cost.load(Ordering::Relaxed),
        }
    }
}

impl Default for ImplementationShortfallTracker {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Context object passed between record_signal and record_fill
#[derive(Debug, Clone)]
pub struct SignalContext {
    pub signal_id: u64,
    pub paper_price_scaled: i128,
    pub qty_scaled: i128,
    pub signal_time_ns: u64,
    pub start_instant: Instant,
}

/// Get monotonic nanosecond timestamp
#[inline]
fn get_monotonic_ns() -> u64 {
    static START_TIME: once_cell::sync::Lazy<Instant> = once_cell::sync::Lazy::new(Instant::now);
    START_TIME.elapsed().as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shortfall_calculation_buy() {
        let tracker = ImplementationShortfallTracker::new(100);
        
        // Buy signal at 50000 (scaled)
        let ctx = tracker.record_signal(1, 50_000_000_000, 100_000_000, true);
        
        // Fill at higher price (unfavorable for buy)
        let record = tracker.record_fill(ctx, 50_050_000_000, 100_000, true);
        
        // Shortfall should be positive (paid more than expected)
        let shortfall = record.calculate_shortfall_bps(true);
        assert!(shortfall > 0, "Buy shortfall should be positive when fill > paper");
    }
    
    #[test]
    fn test_shortfall_calculation_sell() {
        let tracker = ImplementationShortfallTracker::new(100);
        
        // Sell signal at 50000 (scaled)
        let ctx = tracker.record_signal(1, 50_000_000_000, 100_000_000, false);
        
        // Fill at lower price (unfavorable for sell)
        let record = tracker.record_fill(ctx, 49_950_000_000, 100_000, false);
        
        // Shortfall should be positive (received less than expected)
        let shortfall = record.calculate_shortfall_bps(false);
        assert!(shortfall > 0, "Sell shortfall should be positive when fill < paper");
    }
    
    #[test]
    fn test_statistics_tracking() {
        let tracker = ImplementationShortfallTracker::new(100);
        
        // Record multiple signals/fills
        for i in 0..10 {
            let ctx = tracker.record_signal(i, 50_000_000_000, 100_000_000, true);
            let fill_price = 50_000_000_000 + (i as i128 * 10_000_000);  // Increasing slippage
            tracker.record_fill(ctx, fill_price, 100_000, true);
        }
        
        let stats = tracker.get_stats();
        assert_eq!(stats.total_signals, 10);
        assert_eq!(stats.filled_signals, 10);
        assert!(stats.avg_shortfall_bps > 0);
    }
}
