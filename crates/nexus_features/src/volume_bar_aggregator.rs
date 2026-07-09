//! Chapter 3: Volume Bar and Time Bar Aggregator
//!
//! This module provides zero-allocation bar aggregation that emits
//! candles exactly when volume or time thresholds are breached.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use nexus_core::memory::cache_padder::CachePadded64;
use nexus_core::time::tsc_clock::MonotonicNanosClock;

/// Type of bar aggregation
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarType {
    /// Time-based bars (e.g., 1-minute)
    Time = 0,
    /// Volume-based bars (e.g., 1000 contracts)
    Volume = 1,
    /// Tick-based bars (e.g., every 100 trades)
    Tick = 2,
    /// Dollar-value bars (e.g., $1M notional)
    Dollar = 3,
}

impl BarType {
    #[inline]
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(BarType::Time),
            1 => Some(BarType::Volume),
            2 => Some(BarType::Tick),
            3 => Some(BarType::Dollar),
            _ => None,
        }
    }
}

/// OHLCV bar data
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OhlcvBar {
    /// Open price (nanodollars)
    pub open: CachePadded64<AtomicU64>,
    /// High price (nanodollars)
    pub high: CachePadded64<AtomicU64>,
    /// Low price (nanodollars)
    pub low: CachePadded64<AtomicU64>,
    /// Close price (nanodollars)
    pub close: CachePadded64<AtomicU64>,
    /// Volume (base units * 1e9)
    pub volume: CachePadded64<AtomicU64>,
    /// VWAP numerator (price * volume sum)
    pub vwap_num: CachePadded64<AtomicU64>,
    /// Number of ticks in bar
    pub tick_count: CachePadded64<AtomicU64>,
    /// Start timestamp (nanoseconds)
    pub start_time_ns: CachePadded64<AtomicU64>,
    /// End timestamp (nanoseconds)
    pub end_time_ns: CachePadded64<AtomicU64>,
    /// Whether bar is complete
    pub complete: CachePadded64<AtomicBool>,
}

use std::sync::atomic::AtomicBool;

// SAFETY: OhlcvBar uses atomic operations
unsafe impl Send for OhlcvBar {}
unsafe impl Sync for OhlcvBar {}

impl OhlcvBar {
    #[inline]
    pub const fn new() -> Self {
        Self {
            open: CachePadded64::new(AtomicU64::new(0)),
            high: CachePadded64::new(AtomicU64::new(0)),
            low: CachePadded64::new(AtomicU64::new(u64::MAX)),
            close: CachePadded64::new(AtomicU64::new(0)),
            volume: CachePadded64::new(AtomicU64::new(0)),
            vwap_num: CachePadded64::new(AtomicU64::new(0)),
            tick_count: CachePadded64::new(AtomicU64::new(0)),
            start_time_ns: CachePadded64::new(AtomicU64::new(0)),
            end_time_ns: CachePadded64::new(AtomicU64::new(0)),
            complete: CachePadded64::new(AtomicBool::new(false)),
        }
    }

    #[inline]
    pub fn reset(&self) {
        self.open.0.store(0, Ordering::Release);
        self.high.0.store(0, Ordering::Release);
        self.low.0.store(u64::MAX, Ordering::Release);
        self.close.0.store(0, Ordering::Release);
        self.volume.0.store(0, Ordering::Release);
        self.vwap_num.0.store(0, Ordering::Release);
        self.tick_count.0.store(0, Ordering::Release);
        self.complete.0.store(false, Ordering::Release);
    }

    #[inline]
    pub fn get_vwap(&self) -> u64 {
        let vol = self.volume.0.load(Ordering::Acquire);
        if vol == 0 {
            return 0;
        }
        
        let num = self.vwap_num.0.load(Ordering::Acquire);
        num / vol
    }
}

impl Default for OhlcvBar {
    fn default() -> Self {
        Self::new()
    }
}

/// Volume bar aggregator
#[repr(C)]
pub struct VolumeBarAggregator {
    /// Current incomplete bar
    current_bar: CachePadded64<OhlcvBar>,
    /// Volume threshold for bar completion
    volume_threshold: u64,
    /// Completed bar count
    completed_bars: CachePadded64<AtomicUsize>,
    /// Clock for timestamps
    clock: MonotonicNanosClock,
    /// Bar type
    bar_type: BarType,
}

// SAFETY: VolumeBarAggregator is single-threaded
unsafe impl Send for VolumeBarAggregator {}
unsafe impl Sync for VolumeBarAggregator {}

impl VolumeBarAggregator {
    /// Create a new volume bar aggregator
    #[inline]
    pub fn new(volume_threshold: u64) -> Self {
        let bar = OhlcvBar::new();
        bar.start_time_ns.0.store(MonotonicNanosClock::new().now_ns(), Ordering::Release);
        
        Self {
            current_bar: CachePadded64::new(bar),
            volume_threshold,
            completed_bars: CachePadded64::new(AtomicUsize::new(0)),
            clock: MonotonicNanosClock::new(),
            bar_type: BarType::Volume,
        }
    }

    /// Process a tick (price, volume)
    /// Returns true if bar is complete
    #[inline]
    pub fn process_tick(&self, price: u64, volume: u64) -> bool {
        let bar = &self.current_bar.0;
        
        // Initialize open if first tick
        let current_open = bar.open.0.load(Ordering::Acquire);
        if current_open == 0 {
            bar.open.0.store(price, Ordering::Release);
            bar.start_time_ns.0.store(self.clock.now_ns(), Ordering::Release);
        }
        
        // Update high
        let current_high = bar.high.0.load(Ordering::Acquire);
        if price > current_high {
            bar.high.0.store(price, Ordering::Release);
        }
        
        // Update low
        let current_low = bar.low.0.load(Ordering::Acquire);
        if price < current_low {
            bar.low.0.store(price, Ordering::Release);
        }
        
        // Update close
        bar.close.0.store(price, Ordering::Release);
        
        // Update volume
        let old_vol = bar.volume.0.fetch_add(volume, Ordering::AcqRel);
        let new_vol = old_vol + volume;
        
        // Update VWAP numerator
        let pv = price as u128 * volume as u128;
        let old_num = bar.vwap_num.0.load(Ordering::Acquire) as u128;
        bar.vwap_num.0.store((old_num + pv) as u64, Ordering::Release);
        
        // Update tick count
        bar.tick_count.0.fetch_add(1, Ordering::AcqRel);
        
        // Check if bar is complete
        if new_vol >= self.volume_threshold {
            bar.end_time_ns.0.store(self.clock.now_ns(), Ordering::Release);
            bar.complete.0.store(true, Ordering::Release);
            self.completed_bars.0.fetch_add(1, Ordering::AcqRel);
            return true;
        }
        
        false
    }

    /// Get current bar (may be incomplete)
    #[inline]
    pub fn current_bar(&self) -> &OhlcvBar {
        &self.current_bar.0
    }

    /// Take completed bar and start new one
    #[inline]
    pub fn take_bar(&self) -> Option<OhlcvBar> {
        let bar = &self.current_bar.0;
        
        if !bar.complete.0.load(Ordering::Acquire) {
            return None;
        }
        
        // Copy current bar
        let completed = OhlcvBar {
            open: CachePadded64::new(AtomicU64::new(bar.open.0.load(Ordering::Acquire))),
            high: CachePadded64::new(AtomicU64::new(bar.high.0.load(Ordering::Acquire))),
            low: CachePadded64::new(AtomicU64::new(bar.low.0.load(Ordering::Acquire))),
            close: CachePadded64::new(AtomicU64::new(bar.close.0.load(Ordering::Acquire))),
            volume: CachePadded64::new(AtomicU64::new(bar.volume.0.load(Ordering::Acquire))),
            vwap_num: CachePadded64::new(AtomicU64::new(bar.vwap_num.0.load(Ordering::Acquire))),
            tick_count: CachePadded64::new(AtomicU64::new(bar.tick_count.0.load(Ordering::Acquire))),
            start_time_ns: CachePadded64::new(AtomicU64::new(bar.start_time_ns.0.load(Ordering::Acquire))),
            end_time_ns: CachePadded64::new(AtomicU64::new(bar.end_time_ns.0.load(Ordering::Acquire))),
            complete: CachePadded64::new(AtomicBool::new(true)),
        };
        
        // Reset for next bar
        bar.reset();
        bar.start_time_ns.0.store(self.clock.now_ns(), Ordering::Release);
        
        Some(completed)
    }

    /// Get completed bar count
    #[inline]
    pub fn completed_count(&self) -> usize {
        self.completed_bars.0.load(Ordering::Relaxed)
    }

    /// Get bar type
    #[inline]
    pub fn bar_type(&self) -> BarType {
        self.bar_type
    }
}

/// Time bar aggregator
#[repr(C)]
pub struct TimeBarAggregator {
    /// Current incomplete bar
    current_bar: CachePadded64<OhlcvBar>,
    /// Time interval in nanoseconds
    interval_ns: u64,
    /// Next bar boundary timestamp
    next_boundary_ns: CachePadded64<AtomicU64>,
    /// Completed bar count
    completed_bars: CachePadded64<AtomicUsize>,
    /// Clock for timestamps
    clock: MonotonicNanosClock,
    /// Bar type
    bar_type: BarType,
}

// SAFETY: TimeBarAggregator is single-threaded
unsafe impl Send for TimeBarAggregator {}
unsafe impl Sync for TimeBarAggregator {}

impl TimeBarAggregator {
    /// Create a new time bar aggregator
    /// interval_ns: e.g., 60_000_000_000 for 1-minute bars
    #[inline]
    pub fn new(interval_ns: u64) -> Self {
        let now = MonotonicNanosClock::new().now_ns();
        let aligned_now = (now / interval_ns) * interval_ns;
        let next_boundary = aligned_now + interval_ns;
        
        let bar = OhlcvBar::new();
        bar.start_time_ns.0.store(aligned_now, Ordering::Release);
        
        Self {
            current_bar: CachePadded64::new(bar),
            interval_ns,
            next_boundary: CachePadded64::new(AtomicU64::new(next_boundary)),
            completed_bars: CachePadded64::new(AtomicUsize::new(0)),
            clock: MonotonicNanosClock::new(),
            bar_type: BarType::Time,
        }
    }

    /// Process a tick
    /// Returns true if bar is complete (time boundary crossed)
    #[inline]
    pub fn process_tick(&self, price: u64, volume: u64) -> bool {
        let now = self.clock.now_ns();
        let bar = &self.current_bar.0;
        let boundary = self.next_boundary_ns.0.load(Ordering::Acquire);
        
        // Check if we've crossed the time boundary
        if now >= boundary {
            bar.end_time_ns.0.store(boundary, Ordering::Release);
            bar.complete.0.store(true, Ordering::Release);
            self.completed_bars.0.fetch_add(1, Ordering::AcqRel);
            
            // Start new bar
            bar.reset();
            bar.start_time_ns.0.store(boundary, Ordering::Release);
            
            // Update next boundary
            self.next_boundary_ns.0.store(boundary + self.interval_ns, Ordering::Release);
        }
        
        // Update OHLCV
        let current_open = bar.open.0.load(Ordering::Acquire);
        if current_open == 0 {
            bar.open.0.store(price, Ordering::Release);
        }
        
        let current_high = bar.high.0.load(Ordering::Acquire);
        if price > current_high || current_high == 0 {
            bar.high.0.store(price, Ordering::Release);
        }
        
        let current_low = bar.low.0.load(Ordering::Acquire);
        if price < current_low || current_low == u64::MAX {
            bar.low.0.store(price, Ordering::Release);
        }
        
        bar.close.0.store(price, Ordering::Release);
        bar.volume.0.fetch_add(volume, Ordering::AcqRel);
        
        let pv = price as u128 * volume as u128;
        let old_num = bar.vwap_num.0.load(Ordering::Acquire) as u128;
        bar.vwap_num.0.store((old_num + pv) as u64, Ordering::Release);
        
        bar.tick_count.0.fetch_add(1, Ordering::AcqRel);
        bar.end_time_ns.0.store(now, Ordering::Release);
        
        now >= boundary
    }

    /// Take completed bar
    #[inline]
    pub fn take_bar(&self) -> Option<OhlcvBar> {
        let bar = &self.current_bar.0;
        
        if !bar.complete.0.load(Ordering::Acquire) {
            return None;
        }
        
        let completed = OhlcvBar {
            open: CachePadded64::new(AtomicU64::new(bar.open.0.load(Ordering::Acquire))),
            high: CachePadded64::new(AtomicU64::new(bar.high.0.load(Ordering::Acquire))),
            low: CachePadded64::new(AtomicU64::new(bar.low.0.load(Ordering::Acquire))),
            close: CachePadded64::new(AtomicU64::new(bar.close.0.load(Ordering::Acquire))),
            volume: CachePadded64::new(AtomicU64::new(bar.volume.0.load(Ordering::Acquire))),
            vwap_num: CachePadded64::new(AtomicU64::new(bar.vwap_num.0.load(Ordering::Acquire))),
            tick_count: CachePadded64::new(AtomicU64::new(bar.tick_count.0.load(Ordering::Acquire))),
            start_time_ns: CachePadded64::new(AtomicU64::new(bar.start_time_ns.0.load(Ordering::Acquire))),
            end_time_ns: CachePadded64::new(AtomicU64::new(bar.end_time_ns.0.load(Ordering::Acquire))),
            complete: CachePadded64::new(AtomicBool::new(true)),
        };
        
        bar.reset();
        bar.start_time_ns.0.store(self.next_boundary_ns.0.load(Ordering::Acquire), Ordering::Release);
        
        Some(completed)
    }

    /// Get completed bar count
    #[inline]
    pub fn completed_count(&self) -> usize {
        self.completed_bars.0.load(Ordering::Relaxed)
    }

    /// Get bar type
    #[inline]
    pub fn bar_type(&self) -> BarType {
        self.bar_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_bar_single_tick() {
        let agg = VolumeBarAggregator::new(1000);
        
        // Single tick completes bar
        assert!(agg.process_tick(100_000_000_000, 1000));
        assert_eq!(agg.completed_count(), 1);
        
        let bar = agg.current_bar();
        assert_eq!(bar.open.0.load(Ordering::Acquire), 100_000_000_000);
        assert_eq!(bar.high.0.load(Ordering::Acquire), 100_000_000_000);
        assert_eq!(bar.low.0.load(Ordering::Acquire), 100_000_000_000);
        assert_eq!(bar.close.0.load(Ordering::Acquire), 100_000_000_000);
        assert_eq!(bar.volume.0.load(Ordering::Acquire), 1000);
    }

    #[test]
    fn test_volume_bar_multiple_ticks() {
        let agg = VolumeBarAggregator::new(1000);
        
        // Multiple ticks to complete bar
        assert!(!agg.process_tick(100_000_000_000, 300));
        assert!(!agg.process_tick(101_000_000_000, 400));
        assert!(agg.process_tick(102_000_000_000, 300)); // Completes
        
        assert_eq!(agg.completed_count(), 1);
        
        let bar = agg.current_bar();
        assert_eq!(bar.open.0.load(Ordering::Acquire), 100_000_000_000);
        assert_eq!(bar.high.0.load(Ordering::Acquire), 102_000_000_000);
        assert_eq!(bar.low.0.load(Ordering::Acquire), 100_000_000_000);
        assert_eq!(bar.close.0.load(Ordering::Acquire), 102_000_000_000);
    }

    #[test]
    fn test_volume_bar_take() {
        let agg = VolumeBarAggregator::new(500);
        
        agg.process_tick(100_000_000_000, 500);
        
        let bar = agg.take_bar();
        assert!(bar.is_some());
        assert_eq!(agg.completed_count(), 1);
        
        // Continue with new bar
        assert!(!agg.process_tick(101_000_000_000, 200));
        assert_eq!(agg.completed_count(), 1);
    }

    #[test]
    fn test_time_bar_basic() {
        // 1-second interval
        let agg = TimeBarAggregator::new(1_000_000_000);
        
        assert_eq!(agg.bar_type(), BarType::Time);
        
        // Process some ticks
        agg.process_tick(100_000_000_000, 100);
        agg.process_tick(101_000_000_000, 200);
        
        // Bar may or may not be complete depending on timing
        let _count = agg.completed_count();
    }

    #[test]
    fn test_bar_type_conversion() {
        assert_eq!(BarType::from_u8(0), Some(BarType::Time));
        assert_eq!(BarType::from_u8(1), Some(BarType::Volume));
        assert_eq!(BarType::from_u8(2), Some(BarType::Tick));
        assert_eq!(BarType::from_u8(3), Some(BarType::Dollar));
        assert_eq!(BarType::from_u8(4), None);
    }
}
