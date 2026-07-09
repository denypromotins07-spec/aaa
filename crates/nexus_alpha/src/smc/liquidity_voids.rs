//! Smart Money Concepts: Liquidity Voids and Fair Value Gaps
//! 
//! Detects Fair Value Gaps (FVG) - imbalances between buy/sell pressure
//! that create "gaps" in price action, and Liquidity Voids - areas with
//! minimal trading activity that price tends to move through quickly.
//! Zero-allocation implementation using fixed-size buffers.

use nexus_core::memory::arena::BumpAllocator;
use crate::smc::order_blocks::CandleRingBuffer;

/// Maximum number of FVGs to track
pub const MAX_FVGS: usize = 128;

/// Maximum number of liquidity voids to track
pub const MAX_VOID_ZONES: usize = 64;

/// Cache-line padded Fair Value Gap entry
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct FairValueGap {
    /// Start timestamp
    pub start_ts: u64,
    /// End timestamp  
    pub end_ts: u64,
    /// Gap high price (scaled integer)
    pub high: i64,
    /// Gap low price (scaled integer)
    pub low: i64,
    /// Is this a bullish FVG (price gapped up)?
    pub is_bullish: bool,
    /// Has the gap been filled?
    pub filled: bool,
    /// Fill timestamp
    pub fill_ts: u64,
    /// Gap size in ticks
    pub gap_size: i64,
    /// Volume imbalance at creation
    pub volume_imbalance: f64,
    /// Strength score (0-100)
    pub strength: u8,
    /// Padding
    _padding: [u8; 5],
}

impl Default for FairValueGap {
    fn default() -> Self {
        Self {
            start_ts: 0,
            end_ts: 0,
            high: 0,
            low: 0,
            is_bullish: false,
            filled: false,
            fill_ts: 0,
            gap_size: 0,
            volume_imbalance: 0.0,
            strength: 0,
            _padding: [0u8; 5],
        }
    }
}

/// Liquidity Void Zone - area of minimal activity
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct LiquidityVoid {
    /// Zone high price
    pub high: i64,
    /// Zone low price
    pub low: i64,
    /// First detected timestamp
    pub detected_ts: u64,
    /// Last time price was in this zone
    pub last_visit_ts: u64,
    /// Number of times price visited
    pub visit_count: u32,
    /// Average time spent in zone (nanos)
    pub avg_duration: u64,
    /// Is this zone still valid?
    pub valid: bool,
    /// Padding
    _padding: [u8; 7],
}

impl Default for LiquidityVoid {
    fn default() -> Self {
        Self {
            high: 0,
            low: 0,
            detected_ts: 0,
            last_visit_ts: 0,
            visit_count: 0,
            avg_duration: 0,
            valid: true,
            _padding: [0u8; 7],
        }
    }
}

/// Fair Value Gap Detector
pub struct FvgDetector {
    /// Detected FVGs
    fvgs: [FairValueGap; MAX_FVGS],
    /// Write index
    write_idx: usize,
    /// Count of valid FVGs
    count: usize,
    /// Previous candle data for gap detection
    prev_candle_valid: bool,
    prev_high: i64,
    prev_low: i64,
    prev_close: i64,
    /// Price scale
    price_scale: i64,
}

impl FvgDetector {
    pub fn new(_allocator: &BumpAllocator) -> Self {
        Self {
            fvgs: [FairValueGap::default(); MAX_FVGS],
            write_idx: 0,
            count: 0,
            prev_candle_valid: false,
            prev_high: 0,
            prev_low: 0,
            prev_close: 0,
            price_scale: 100_000_000,
        }
    }

    /// Process a new candle and detect FVGs
    #[inline]
    pub fn on_candle(&mut self, ts: u64, high: i64, low: i64, close: i64, volume: i64, buy_vol: i64) {
        if !self.prev_candle_valid {
            self.prev_high = high;
            self.prev_low = low;
            self.prev_close = close;
            self.prev_candle_valid = true;
            return;
        }

        // Detect bullish FVG: current low > previous high
        if low > self.prev_high {
            let gap_size = low - self.prev_high;
            let vol_imbalance = if volume > 0 {
                (buy_vol as f64 - (volume - buy_vol) as f64) / volume as f64
            } else {
                0.0
            };

            let fvg = FairValueGap {
                start_ts: ts,
                end_ts: ts,
                high: low,
                low: self.prev_high,
                is_bullish: true,
                filled: false,
                fill_ts: 0,
                gap_size,
                volume_imbalance: vol_imbalance,
                strength: self.calculate_fvg_strength(gap_size, vol_imbalance),
                _padding: [0u8; 5],
            };

            self.store_fvg(fvg);
        }

        // Detect bearish FVG: current high < previous low
        if high < self.prev_low {
            let gap_size = self.prev_low - high;
            let sell_vol = volume - buy_vol;
            let vol_imbalance = if volume > 0 {
                (buy_vol as f64 - sell_vol as f64) / volume as f64
            } else {
                0.0
            };

            let fvg = FairValueGap {
                start_ts: ts,
                end_ts: ts,
                high: self.prev_low,
                low: high,
                is_bullish: false,
                filled: false,
                fill_ts: 0,
                gap_size,
                volume_imbalance: vol_imbalance,
                strength: self.calculate_fvg_strength(gap_size, vol_imbalance),
                _padding: [0u8; 5],
            };

            self.store_fvg(fvg);
        }

        // Check if existing FVGs are filled
        self.check_fvg_fills(close, ts);

        // Update previous candle
        self.prev_high = high;
        self.prev_low = low;
        self.prev_close = close;
    }

    #[inline]
    fn store_fvg(&mut self, fvg: FairValueGap) {
        self.fvgs[self.write_idx] = fvg;
        self.write_idx = (self.write_idx + 1) % MAX_FVGS;
        if self.count < MAX_FVGS {
            self.count += 1;
        }
    }

    #[inline]
    fn calculate_fvg_strength(&self, gap_size: i64, vol_imbalance: f64) -> u8 {
        // Normalize gap size (assuming typical gap is ~10-100 ticks)
        let size_component = ((gap_size as f64 / 50.0).min(1.0) * 50.0) as u8;
        
        // Volume imbalance component
        let vol_component = (vol_imbalance.abs() * 50.0) as u8;
        
        size_component.saturating_add(vol_component)
    }

    #[inline]
    fn check_fvg_fills(&mut self, current_price: i64, ts: u64) {
        for i in 0..self.count {
            let idx = (self.write_idx + i) % MAX_FVGS;
            let fvg = &mut self.fvgs[idx];
            
            if fvg.filled || !fvg.valid() {
                continue;
            }

            let filled = if fvg.is_bullish {
                // Bullish FVG fills when price drops back into the gap
                current_price <= fvg.high && current_price >= fvg.low
            } else {
                // Bearish FVG fills when price rises back into the gap
                current_price >= fvg.low && current_price <= fvg.high
            };

            if filled {
                fvg.filled = true;
                fvg.fill_ts = ts;
            }
        }
    }

    /// Get all unfilled FVGs
    #[inline]
    pub fn get_unfilled_fvgs(&self) -> impl Iterator<Item = &FairValueGap> {
        (0..self.count).filter_map(move |i| {
            let idx = (self.write_idx + i) % MAX_FVGS;
            let fvg = &self.fvgs[idx];
            if !fvg.filled && fvg.valid() {
                Some(fvg)
            } else {
                None
            }
        })
    }

    /// Get recent FVGs
    #[inline]
    pub fn get_recent_fvgs(&self, n: usize) -> impl Iterator<Item = &FairValueGap> {
        let count = n.min(self.count);
        (0..count).map(move |i| {
            let idx = (self.write_idx + MAX_FVGS - count + i) % MAX_FVGS;
            &self.fvgs[idx]
        })
    }
}

impl FairValueGap {
    #[inline]
    fn valid(&self) -> bool {
        // FVG is valid if not too old (e.g., within last 1000 candles worth of time)
        // Simplified: just check it's not filled
        !self.filled
    }
}

/// Liquidity Void Detector
pub struct LiquidityVoidDetector {
    /// Detected void zones
    voids: [LiquidityVoid; MAX_VOID_ZONES],
    /// Write index
    write_idx: usize,
    /// Count of valid voids
    count: usize,
    /// Price bins for detecting low-activity zones
    /// Using fixed-size array instead of HashMap
    price_bins: [VolumeBin; 4096],
    /// Bin configuration
    bin_size: i64,
    /// Minimum price tracked
    min_price: i64,
    /// Timestamp for duration tracking
    in_zone_start: Option<(i64, u64)>,
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Default)]
struct VolumeBin {
    total_volume: i64,
    tick_count: u32,
    last_update_ts: u64,
}

impl LiquidityVoidDetector {
    pub fn new(_allocator: &BumpAllocator, min_price: i64, max_price: i64) -> Self {
        let bin_size = (max_price - min_price) / 4096;
        
        Self {
            voids: [LiquidityVoid::default(); MAX_VOID_ZONES],
            write_idx: 0,
            count: 0,
            price_bins: [VolumeBin::default(); 4096],
            bin_size: bin_size.max(1),
            min_price,
            in_zone_start: None,
        }
    }

    /// Process a tick and update volume profile
    #[inline]
    pub fn on_tick(&mut self, ts: u64, price: i64, volume: i64) {
        // Update the appropriate bin
        if let Some(bin_idx) = self.price_to_bin(price) {
            let bin = &mut self.price_bins[bin_idx];
            bin.total_volume += volume;
            bin.tick_count += 1;
            bin.last_update_ts = ts;
        }

        // Periodically analyze for void zones (every N ticks in production)
        // For simplicity, we do lightweight analysis here
        self.detect_void_zones(ts);
    }

    #[inline]
    fn price_to_bin(&self, price: i64) -> Option<usize> {
        if price < self.min_price {
            return None;
        }
        let offset = price - self.min_price;
        let bin_idx = (offset / self.bin_size) as usize;
        if bin_idx >= 4096 {
            None
        } else {
            Some(bin_idx)
        }
    }

    #[inline]
    fn detect_void_zones(&mut self, ts: u64) {
        // Find consecutive bins with very low volume
        let threshold = self.calculate_volume_threshold();
        
        let mut void_start: Option<(usize, u64)> = None;
        
        for i in 0..4096 {
            let bin = &self.price_bins[i];
            let is_low_volume = bin.total_volume < threshold;
            
            if is_low_volume {
                if void_start.is_none() {
                    void_start = Some((i, ts));
                }
            } else {
                if let Some((start_idx, start_ts)) = void_start {
                    // End of potential void zone
                    if i - start_idx >= 3 {
                        // At least 3 consecutive low-volume bins
                        self.create_void_zone(start_idx, i - 1, start_ts, ts);
                    }
                    void_start = None;
                }
            }
        }
        
        // Handle case where void extends to end
        if let Some((start_idx, start_ts)) = void_start {
            if 4096 - start_idx >= 3 {
                self.create_void_zone(start_idx, 4095, start_ts, ts);
            }
        }
    }

    #[inline]
    fn calculate_volume_threshold(&self) -> i64 {
        // Calculate average volume per bin, use fraction as threshold
        let mut total_vol = 0i64;
        let mut active_bins = 0u32;
        
        for bin in &self.price_bins {
            if bin.tick_count > 0 {
                total_vol += bin.total_volume;
                active_bins += 1;
            }
        }
        
        if active_bins == 0 {
            return 1000; // Default threshold
        }
        
        let avg = total_vol / active_bins as i64;
        (avg / 10).max(100) // 10% of average, minimum 100
    }

    #[inline]
    fn create_void_zone(&mut self, start_bin: usize, end_bin: usize, start_ts: u64, end_ts: u64) {
        let low = self.min_price + (start_bin as i64 * self.bin_size);
        let high = self.min_price + ((end_bin + 1) as i64 * self.bin_size);
        
        // Check if we already have a similar void
        for i in 0..self.count {
            let idx = (self.write_idx + i) % MAX_VOID_ZONES;
            let existing = &self.voids[idx];
            
            // If overlapping void exists, skip
            if !(high < existing.low || low > existing.high) {
                return;
            }
        }
        
        let void = LiquidityVoid {
            high,
            low,
            detected_ts: start_ts,
            last_visit_ts: end_ts,
            visit_count: 0,
            avg_duration: 0,
            valid: true,
            _padding: [0u8; 7],
        };
        
        self.voids[self.write_idx] = void;
        self.write_idx = (self.write_idx + 1) % MAX_VOID_ZONES;
        if self.count < MAX_VOID_ZONES {
            self.count += 1;
        }
    }

    /// Check if price is currently in a void zone
    #[inline]
    pub fn check_void_visit(&mut self, ts: u64, price: i64) {
        for i in 0..self.count {
            let idx = (self.write_idx + i) % MAX_VOID_ZONES;
            let void = &mut self.voids[idx];
            
            if !void.valid {
                continue;
            }
            
            if price >= void.low && price <= void.high {
                // Currently in void zone
                if self.in_zone_start.is_none() {
                    self.in_zone_start = Some((idx as i64, ts));
                }
                void.last_visit_ts = ts;
                void.visit_count += 1;
            } else {
                // Left void zone
                if let Some((zone_idx, start_ts)) = self.in_zone_start {
                    if zone_idx as usize == idx {
                        let duration = ts - start_ts;
                        let void = &mut self.voids[idx];
                        // Update average duration
                        void.avg_duration = (void.avg_duration * void.visit_count as u64 + duration) 
                            / (void.visit_count as u64 + 1);
                    }
                }
                self.in_zone_start = None;
            }
        }
    }

    /// Get all valid void zones
    #[inline]
    pub fn get_voids(&self) -> impl Iterator<Item = &LiquidityVoid> {
        (0..self.count).filter_map(move |i| {
            let idx = (self.write_idx + i) % MAX_VOID_ZONES;
            let void = &self.voids[idx];
            if void.valid {
                Some(void)
            } else {
                None
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_core::memory::arena::BumpAllocator;

    #[test]
    fn test_fvg_detection_bullish() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut detector = FvgDetector::new(&allocator);

        // First candle
        detector.on_candle(1000, 100_0000_0000, 99_0000_0000, 99_5000_0000, 1000, 600);
        
        // Second candle gaps up
        detector.on_candle(2000, 102_0000_0000, 101_0000_0000, 101_5000_0000, 1000, 700);

        let unfilled: Vec<_> = detector.get_unfilled_fvgs().collect();
        assert_eq!(unfilled.len(), 1);
        assert!(unfilled[0].is_bullish);
        assert_eq!(unfilled[0].gap_size, 101_0000_0000 - 100_0000_0000);
    }

    #[test]
    fn test_fvg_fill_detection() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut detector = FvgDetector::new(&allocator);

        // Create bullish FVG
        detector.on_candle(1000, 100_0000_0000, 99_0000_0000, 99_5000_0000, 1000, 600);
        detector.on_candle(2000, 102_0000_0000, 101_0000_0000, 101_5000_0000, 1000, 700);
        
        // Price returns to fill the gap
        detector.on_candle(3000, 101_0000_0000, 100_2000_0000, 100_5000_0000, 1000, 500);

        let unfilled: Vec<_> = detector.get_unfilled_fvgs().collect();
        assert_eq!(unfilled.len(), 0); // Should be filled now
    }

    #[test]
    fn test_liquidity_void_detection() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let min_price = 90_0000_0000i64;
        let max_price = 110_0000_0000i64;
        let mut detector = LiquidityVoidDetector::new(&allocator, min_price, max_price);

        // Add volume only in certain price ranges
        for i in 0..100 {
            let ts = 1000 + i * 100;
            // High volume around 100
            detector.on_tick(ts, 100_0000_0000, 10000);
            // Low volume elsewhere
            detector.on_tick(ts + 50, 95_0000_0000, 10);
        }

        let voids: Vec<_> = detector.get_voids().collect();
        // Should detect void zones away from the high-volume area
        assert!(voids.len() > 0);
    }
}
