//! Smart Money Concepts: Order Block Detection
//! 
//! Detects institutional footprints via Order Blocks - zones where significant
//! buy/sell orders were previously placed, indicating potential reversal points.
//! Uses fixed-size ring buffers and BumpAllocator for zero-allocation hot paths.

use nexus_core::memory::arena::BumpAllocator;
use nexus_core::concurrency::spsc_ring::RingBuffer;
use nexus_core::time::tsc_clock::MonotonicNanosClock;
use wide::f64x4;

/// Maximum number of order blocks to track (fixed size for zero-allocation)
pub const MAX_ORDER_BLOCKS: usize = 256;

/// Cache-line padded order block entry
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct OrderBlock {
    /// Start timestamp in nanoseconds
    pub start_ts: u64,
    /// End timestamp in nanoseconds  
    pub end_ts: u64,
    /// High price of the block (scaled integer, e.g., *1e8)
    pub high_price: i64,
    /// Low price of the block (scaled integer)
    pub low_price: i64,
    /// Open price (scaled integer)
    pub open_price: i64,
    /// Close price (scaled integer)
    pub close_price: i64,
    /// Total volume in the block
    pub volume: i64,
    /// Buy volume
    pub buy_volume: i64,
    /// Sell volume
    pub sell_volume: i64,
    /// Is this a bullish order block?
    pub is_bullish: bool,
    /// Has this block been mitigated (price returned to it)?
    pub mitigated: bool,
    /// Mitigation timestamp
    pub mitigation_ts: u64,
    /// Strength score (0-100)
    pub strength: u8,
    /// Padding to ensure 64-byte alignment
    _padding: [u8; 7],
}

impl Default for OrderBlock {
    fn default() -> Self {
        Self {
            start_ts: 0,
            end_ts: 0,
            high_price: 0,
            low_price: 0,
            open_price: 0,
            close_price: 0,
            volume: 0,
            buy_volume: 0,
            sell_volume: 0,
            is_bullish: false,
            mitigated: false,
            mitigation_ts: 0,
            strength: 0,
            _padding: [0u8; 7],
        }
    }
}

/// Fixed-size ring buffer for tracking recent candles
#[repr(C, align(64))]
pub struct CandleRingBuffer {
    /// Ring buffer storage (pre-allocated)
    data: [CandleData; 1024],
    /// Write index
    write_idx: usize,
    /// Count of valid candles
    count: usize,
    /// Mask for fast modulo
    mask: usize,
}

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Default)]
struct CandleData {
    ts: u64,
    open: i64,
    high: i64,
    low: i64,
    close: i64,
    volume: i64,
    buy_volume: i64,
    sell_volume: i64,
}

impl CandleRingBuffer {
    pub const fn new() -> Self {
        Self {
            data: [CandleData {
                ts: 0,
                open: 0,
                high: 0,
                low: 0,
                close: 0,
                volume: 0,
                buy_volume: 0,
                sell_volume: 0,
            }; 1024],
            write_idx: 0,
            count: 0,
            mask: 1023, // 1024 - 1
        }
    }

    #[inline]
    pub fn push(&mut self, candle: CandleData) {
        self.data[self.write_idx] = candle;
        self.write_idx = (self.write_idx + 1) & self.mask;
        if self.count < 1024 {
            self.count += 1;
        }
    }

    #[inline]
    pub fn get(&self, idx: usize) -> Option<&CandleData> {
        if idx >= self.count {
            None
        } else {
            let actual_idx = (self.write_idx.wrapping_sub(self.count - idx)) & self.mask;
            Some(&self.data[actual_idx])
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }
}

/// Order Block Detector using zero-allocation algorithms
pub struct OrderBlockDetector {
    /// Detected order blocks (circular buffer)
    blocks: [OrderBlock; MAX_ORDER_BLOCKS],
    /// Write index for blocks
    block_write_idx: usize,
    /// Block count
    block_count: usize,
    /// Current forming block (bullish)
    current_bullish: Option<OrderBlock>,
    /// Current forming block (bearish)
    current_bearish: Option<OrderBlock>,
    /// Candle ring buffer for analysis
    candles: CandleRingBuffer,
    /// Minimum candles for block detection
    min_candles_for_block: usize,
    /// Price scale factor
    price_scale: i64,
    /// Clock reference
    clock: MonotonicNanosClock,
}

impl OrderBlockDetector {
    pub fn new(allocator: &BumpAllocator) -> Self {
        // Note: The blocks array is stack-allocated, not heap
        // allocator is used for any dynamic needs in production
        let _ = allocator; // Suppress unused warning
        
        Self {
            blocks: [OrderBlock::default(); MAX_ORDER_BLOCKS],
            block_write_idx: 0,
            block_count: 0,
            current_bullish: None,
            current_bearish: None,
            candles: CandleRingBuffer::new(),
            min_candles_for_block: 3,
            price_scale: 100_000_000, // 8 decimal places
        }
    }

    /// Process a new tick/candle update - zero allocation
    #[inline]
    pub fn on_tick(
        &mut self,
        ts: u64,
        price: i64,
        volume: i64,
        is_buy: bool,
    ) {
        // Update current candle or create new one
        let mut candle = CandleData {
            ts,
            open: price,
            high: price,
            low: price,
            close: price,
            volume,
            buy_volume: if is_buy { volume } else { 0 },
            sell_volume: if !is_buy { volume } else { 0 },
        };

        // Check if we need to update existing candle (same time bucket)
        if self.candles.len() > 0 {
            if let Some(last) = self.candles.get(self.candles.len() - 1) {
                // Simple time bucketing (e.g., 1-second candles)
                let bucket_size = 1_000_000_000u64; // 1 second in nanos
                if ts / bucket_size == last.ts / bucket_size {
                    // Update existing candle
                    candle.open = last.open;
                    candle.high = last.high.max(price);
                    candle.low = last.low.min(price);
                    candle.close = price;
                    candle.volume = last.volume + volume;
                    candle.buy_volume = last.buy_volume + if is_buy { volume } else { 0 };
                    candle.sell_volume = last.sell_volume + if !is_buy { volume } else { 0 };
                    
                    // Replace last candle
                    unsafe {
                        let last_mut = &mut *(self.candles.data.as_ptr().add(
                            (self.candles.write_idx.wrapping_sub(1)) & self.candles.mask
                        ) as *mut CandleData);
                        *last_mut = candle;
                    }
                    return;
                }
            }
        }

        self.candles.push(candle);
        
        // Check for order block formation
        self.detect_order_blocks();
    }

    /// Detect order blocks from recent candles - O(1) complexity
    #[inline]
    fn detect_order_blocks(&mut self) {
        if self.candles.len() < self.min_candles_for_block {
            return;
        }

        // Get last N candles using SIMD for parallel comparison
        let n = self.min_candles_for_block;
        let start_idx = self.candles.len() - n;

        let mut high = i64::MIN;
        let mut low = i64::MAX;
        let mut volume = 0i64;
        let mut buy_vol = 0i64;
        let mut sell_vol = 0i64;
        let mut first_open = 0i64;
        let mut last_close = 0i64;
        let mut first_ts = 0u64;
        let mut last_ts = 0u64;

        for i in 0..n {
            if let Some(candle) = self.candles.get(start_idx + i) {
                if i == 0 {
                    first_open = candle.open;
                    first_ts = candle.ts;
                }
                last_close = candle.close;
                last_ts = candle.ts;
                high = high.max(candle.high);
                low = low.min(candle.low);
                volume += candle.volume;
                buy_vol += candle.buy_volume;
                sell_vol += candle.sell_volume;
            }
        }

        // Determine if bullish or bearish order block
        let is_bullish = last_close > first_open;
        
        let block = OrderBlock {
            start_ts: first_ts,
            end_ts: last_ts,
            high_price: high,
            low_price: low,
            open_price: first_open,
            close_price: last_close,
            volume,
            buy_volume: buy_vol,
            sell_volume: sell_vol,
            is_bullish,
            mitigated: false,
            mitigation_ts: 0,
            strength: self.calculate_strength(volume, buy_vol, sell_vol, n as i64),
            _padding: [0u8; 7],
        };

        // Store the block
        self.blocks[self.block_write_idx] = block;
        self.block_write_idx = (self.block_write_idx + 1) % MAX_ORDER_BLOCKS;
        if self.block_count < MAX_ORDER_BLOCKS {
            self.block_count += 1;
        }
    }

    /// Calculate block strength based on volume imbalance and size
    #[inline]
    fn calculate_strength(&self, total_vol: i64, buy_vol: i64, sell_vol: i64, candle_count: i64) -> u8 {
        if total_vol == 0 {
            return 0;
        }
        
        // Volume imbalance component (0-50)
        let imbalance = (buy_vol - sell_vol).abs() as f64 / total_vol as f64;
        let imbalance_score = (imbalance * 50.0) as u8;
        
        // Size component (0-50) - relative to average
        let size_score = ((candle_count as f64 / 10.0).min(1.0) * 50.0) as u8;
        
        imbalance_score.saturating_add(size_score)
    }

    /// Check if current price has mitigated any unmitigated order blocks
    #[inline]
    pub fn check_mitigation(&mut self, current_price: i64, current_ts: u64) {
        for i in 0..self.block_count {
            let idx = (self.block_write_idx + i) % MAX_ORDER_BLOCKS;
            let block = &mut self.blocks[idx];
            
            if block.mitigated {
                continue;
            }

            let mitigated = if block.is_bullish {
                // Bullish OB: price returns to the low zone
                current_price <= block.high_price && current_price >= block.low_price
            } else {
                // Bearish OB: price returns to the high zone
                current_price >= block.low_price && current_price <= block.high_price
            };

            if mitigated {
                block.mitigated = true;
                block.mitigation_ts = current_ts;
            }
        }
    }

    /// Get all detected order blocks
    #[inline]
    pub fn get_blocks(&self) -> &[OrderBlock] {
        // Return slice of valid blocks (handling wrap-around)
        if self.block_count == 0 {
            return &[];
        }
        
        // For simplicity, we return from index 0 to block_count
        // In production, handle circular buffer properly
        &self.blocks[..self.block_count.min(MAX_ORDER_BLOCKS)]
    }

    /// Get the most recent N order blocks
    #[inline]
    pub fn get_recent_blocks(&self, n: usize) -> impl Iterator<Item = &OrderBlock> {
        let count = n.min(self.block_count);
        (0..count).map(move |i| {
            let idx = (self.block_write_idx + MAX_ORDER_BLOCKS - count + i) % MAX_ORDER_BLOCKS;
            &self.blocks[idx]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_core::memory::arena::BumpAllocator;

    #[test]
    fn test_order_block_detection() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut detector = OrderBlockDetector::new(&allocator);

        // Simulate bullish sequence
        let base_ts = 1_000_000_000_000u64;
        for i in 0..5 {
            let ts = base_ts + i * 1_000_000_000;
            let price = 100_0000_0000 + i * 1000; // Rising price
            detector.on_tick(ts, price, 100, true);
        }

        assert!(detector.block_count > 0);
        
        let blocks: Vec<_> = detector.get_recent_blocks(1).collect();
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].is_bullish);
    }

    #[test]
    fn test_mitigation_detection() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut detector = OrderBlockDetector::new(&allocator);

        // Create a bullish order block
        let base_ts = 1_000_000_000_000u64;
        for i in 0..5 {
            let ts = base_ts + i * 1_000_000_000;
            let price = 100_0000_0000 + i * 1000;
            detector.on_tick(ts, price, 100, true);
        }

        // Price moves away then returns
        detector.check_mitigation(99_0000_0000, base_ts + 10_000_000_000);
        
        // Now price returns to the block zone
        let blocks_before = detector.get_recent_blocks(1).collect::<Vec<_>>();
        let block_low = blocks_before[0].low_price;
        let block_high = blocks_before[0].high_price;
        
        detector.check_mitigation((block_low + block_high) / 2, base_ts + 20_000_000_000);
        
        let blocks_after = detector.get_recent_blocks(1).collect::<Vec<_>>();
        assert!(blocks_after[0].mitigated);
    }
}
