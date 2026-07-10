//! VPIN (Volume-Synchronized Probability of Informed Trading) Calculator
//! 
//! Implements the Lee-Ready algorithm for trade classification and
//! calculates VPIN to measure order flow toxicity. Zero-allocation
//! implementation using fixed-size buffers.

use nexus_core::memory::arena::BumpAllocator;

/// Maximum number of volume buckets for VPIN calculation
pub const MAX_VPIN_BUCKETS: usize = 100;

/// Volume bucket for VPIN calculation
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct VolumeBucket {
    /// Buy volume in this bucket
    buy_volume: i64,
    /// Sell volume in this bucket
    sell_volume: i64,
    /// Total volume
    total_volume: i64,
    /// Number of trades
    trade_count: u32,
    /// Padding
    _padding: [u8; 20],
}

impl Default for VolumeBucket {
    fn default() -> Self {
        Self {
            buy_volume: 0,
            sell_volume: 0,
            total_volume: 0,
            trade_count: 0,
            _padding: [0u8; 20],
        }
    }
}

/// Trade classification result from Lee-Ready algorithm
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct ClassifiedTrade {
    /// Timestamp
    pub ts: u64,
    /// Price (scaled integer)
    pub price: i64,
    /// Volume
    pub volume: i64,
    /// Is buy-initiated? (true = buy, false = sell)
    pub is_buy: bool,
    /// Classification confidence (0-1)
    pub confidence: f64,
    /// Padding
    _padding: [u8; 8],
}

impl Default for ClassifiedTrade {
    fn default() -> Self {
        Self {
            ts: 0,
            price: 0,
            volume: 0,
            is_buy: false,
            confidence: 0.5,
            _padding: [0u8; 8],
        }
    }
}

/// VPIN Calculator with Lee-Ready trade classification
pub struct VpinCalculator {
    /// Volume buckets (circular buffer)
    buckets: [VolumeBucket; MAX_VPIN_BUCKETS],
    /// Current bucket index
    current_bucket: usize,
    /// Target volume per bucket
    target_bucket_volume: i64,
    /// Current bucket volume
    current_bucket_volume: i64,
    /// Previous price for tick test
    prev_price: i64,
    /// Previous previous price
    prev_prev_price: i64,
    /// Price valid flag
    price_valid: bool,
    /// Sum of absolute buy-sell imbalances
    sum_abs_imbalance: f64,
    /// Total volume across all buckets
    total_volume: i64,
    /// Bucket count
    bucket_count: usize,
}

unsafe impl Send for VpinCalculator {}
unsafe impl Sync for VpinCalculator {}

impl VpinCalculator {
    pub fn new(_allocator: &BumpAllocator, target_bucket_volume: i64) -> Self {
        Self {
            buckets: [VolumeBucket::default(); MAX_VPIN_BUCKETS],
            current_bucket: 0,
            target_bucket_volume,
            current_bucket_volume: 0,
            prev_price: 0,
            prev_prev_price: 0,
            price_valid: false,
            sum_abs_imbalance: 0.0,
            total_volume: 0,
            bucket_count: 0,
        }
    }

    /// Process a new trade - classifies using Lee-Ready and updates VPIN
    #[inline]
    pub fn on_trade(&mut self, ts: u64, price: i64, volume: i64) -> ClassifiedTrade {
        // Classify the trade using Lee-Ready algorithm
        let classified = self.classify_trade(ts, price, volume);
        
        // Add to current bucket
        self.add_to_bucket(&classified);
        
        classified
    }

    /// Lee-Ready trade classification algorithm
    /// Uses tick rule and quote rule for classification
    #[inline]
    fn classify_trade(&mut self, ts: u64, price: i64, volume: i64) -> ClassifiedTrade {
        let mut is_buy = false;
        let mut confidence = 0.5;

        if self.price_valid {
            // Tick rule: compare to previous price
            if price > self.prev_price {
                is_buy = true;
                confidence = 0.7;
            } else if price < self.prev_price {
                is_buy = false;
                confidence = 0.7;
            } else {
                // Price unchanged - use quote rule if available
                // For simplicity, use uptick/downtick from two periods ago
                if self.prev_price > self.prev_prev_price {
                    is_buy = true;
                    confidence = 0.6;
                } else if self.prev_price < self.prev_prev_price {
                    is_buy = false;
                    confidence = 0.6;
                } else {
                    // No clear signal - assume 50/50
                    is_buy = true; // Default to buy
                    confidence = 0.5;
                }
            }
        } else {
            // First trade - no classification possible
            is_buy = true;
            confidence = 0.5;
        }

        // Update price history
        self.prev_prev_price = self.prev_price;
        self.prev_price = price;
        self.price_valid = true;

        ClassifiedTrade {
            ts,
            price,
            volume,
            is_buy,
            confidence,
            _padding: [0u8; 8],
        }
    }

    /// Add classified trade to current volume bucket
    #[inline]
    fn add_to_bucket(&mut self, trade: &ClassifiedTrade) {
        let bucket = &mut self.buckets[self.current_bucket];
        
        if trade.is_buy {
            bucket.buy_volume += trade.volume;
        } else {
            bucket.sell_volume += trade.volume;
        }
        
        bucket.total_volume += trade.volume;
        bucket.trade_count += 1;
        self.current_bucket_volume += trade.volume;
        self.total_volume += trade.volume;

        // Check if bucket is full
        if self.current_bucket_volume >= self.target_bucket_volume {
            // Calculate imbalance for this bucket
            let imbalance = (bucket.buy_volume - bucket.sell_volume).abs() as f64;
            let normalized = if bucket.total_volume > 0 {
                imbalance / bucket.total_volume as f64
            } else {
                0.0
            };
            
            self.sum_abs_imbalance += normalized;
            
            // Move to next bucket
            if self.bucket_count < MAX_VPIN_BUCKETS {
                self.bucket_count += 1;
            }
            self.current_bucket = (self.current_bucket + 1) % MAX_VPIN_BUCKETS;
            self.current_bucket_volume = 0;
            
            // Reset the new bucket
            self.buckets[self.current_bucket] = VolumeBucket::default();
        }
    }

    /// Calculate current VPIN value
    /// VPIN = (1/n) * Σ|V_buy - V_sell| / (V_buy + V_sell)
    #[inline]
    pub fn calculate_vpin(&self) -> f64 {
        if self.bucket_count == 0 {
            return 0.0;
        }

        // Use all filled buckets
        let n = self.bucket_count.min(MAX_VPIN_BUCKETS);
        
        if n == 0 {
            return 0.0;
        }

        self.sum_abs_imbalance / n as f64
    }

    /// Get VPIN with confidence bounds
    #[inline]
    pub fn get_vpin_with_stats(&self) -> VpinResult {
        let vpin = self.calculate_vpin();
        
        // Calculate standard deviation of imbalances
        let mut variance = 0.0;
        let n = self.bucket_count.min(MAX_VPIN_BUCKETS);
        
        if n > 1 {
            let mean = self.sum_abs_imbalance / n as f64;
            let mut sum_sq_diff = 0.0;
            
            for i in 0..n {
                let bucket = &self.buckets[i];
                if bucket.total_volume > 0 {
                    let imbalance = (bucket.buy_volume - bucket.sell_volume).abs() as f64 
                        / bucket.total_volume as f64;
                    let diff = imbalance - mean;
                    sum_sq_diff += diff * diff;
                }
            }
            
            variance = sum_sq_diff / (n - 1) as f64;
        }

        let std_dev = variance.sqrt();
        
        // Toxicity level based on VPIN
        let toxicity_level = if vpin > 0.8 {
            ToxicityLevel::Extreme
        } else if vpin > 0.6 {
            ToxicityLevel::High
        } else if vpin > 0.4 {
            ToxicityLevel::Moderate
        } else if vpin > 0.2 {
            ToxicityLevel::Low
        } else {
            ToxicityLevel::Minimal
        };

        VpinResult {
            vpin,
            std_dev,
            bucket_count: n,
            toxicity_level,
            total_volume: self.total_volume,
        }
    }

    /// Get current buy/sell volumes in active bucket
    #[inline]
    pub fn get_current_bucket_volumes(&self) -> (i64, i64) {
        let bucket = &self.buckets[self.current_bucket];
        (bucket.buy_volume, bucket.sell_volume)
    }

    /// Reset the calculator
    #[inline]
    pub fn reset(&mut self) {
        self.buckets = [VolumeBucket::default(); MAX_VPIN_BUCKETS];
        self.current_bucket = 0;
        self.current_bucket_volume = 0;
        self.prev_price = 0;
        self.prev_prev_price = 0;
        self.price_valid = false;
        self.sum_abs_imbalance = 0.0;
        self.total_volume = 0;
        self.bucket_count = 0;
    }

    /// Set target bucket volume
    #[inline]
    pub fn set_target_volume(&mut self, volume: i64) {
        self.target_bucket_volume = volume.max(1);
    }
}

/// VPIN calculation result
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct VpinResult {
    /// VPIN value (0-1)
    pub vpin: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Number of buckets used
    pub bucket_count: usize,
    /// Toxicity level
    pub toxicity_level: ToxicityLevel,
    /// Total volume processed
    pub total_volume: i64,
    /// Padding
    _padding: [u8; 24],
}

impl Default for VpinResult {
    fn default() -> Self {
        Self {
            vpin: 0.0,
            std_dev: 0.0,
            bucket_count: 0,
            toxicity_level: ToxicityLevel::Minimal,
            total_volume: 0,
            _padding: [0u8; 24],
        }
    }
}

/// Order flow toxicity level
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToxicityLevel {
    Minimal = 0,    // VPIN < 0.2
    Low = 1,        // VPIN 0.2-0.4
    Moderate = 2,   // VPIN 0.4-0.6
    High = 3,       // VPIN 0.6-0.8
    Extreme = 4,    // VPIN > 0.8
}

impl Default for ToxicityLevel {
    fn default() -> Self {
        ToxicityLevel::Minimal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_core::memory::arena::BumpAllocator;

    #[test]
    fn test_vpin_initial() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let calc = VpinCalculator::new(&allocator, 1000);
        
        assert_eq!(calc.calculate_vpin(), 0.0);
    }

    #[test]
    fn test_lee_ready_classification() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut calc = VpinCalculator::new(&allocator, 1000);

        // First trade - no classification
        let t1 = calc.on_trade(1000, 100_0000_0000, 100);
        assert!(!calc.price_valid);
        
        // Second trade - price up, should be buy
        let t2 = calc.on_trade(2000, 100_0100_0000, 100);
        assert!(t2.is_buy);
        assert!(t2.confidence > 0.5);
        
        // Third trade - price down, should be sell
        let t3 = calc.on_trade(3000, 100_0050_0000, 100);
        assert!(!t3.is_buy);
        assert!(t3.confidence > 0.5);
    }

    #[test]
    fn test_vpin_calculation_balanced() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut calc = VpinCalculator::new(&allocator, 500);

        // Alternating buys and sells - should give low VPIN
        for i in 0..10 {
            let price = 100_0000_0000 + (i as i64 * 10_0000);
            calc.on_trade(1000 + i * 100, price, 100);
        }

        let result = calc.get_vpin_with_stats();
        // With balanced flow, VPIN should be relatively low
        assert!(result.vpin < 0.5);
    }

    #[test]
    fn test_vpin_calculation_imbalanced() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut calc = VpinCalculator::new(&allocator, 200);

        // All buys - should give high VPIN
        for i in 0..20 {
            let price = 100_0000_0000 + (i as i64 * 10_0000);
            calc.on_trade(1000 + i * 100, price, 50);
        }

        let result = calc.get_vpin_with_stats();
        // With one-sided flow, VPIN should be high
        assert!(result.vpin > 0.5);
        assert_eq!(result.toxicity_level, ToxicityLevel::High);
    }

    #[test]
    fn test_toxicity_levels() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut calc = VpinCalculator::new(&allocator, 100);

        // Generate some trades
        for i in 0..50 {
            let price = 100_0000_0000 + (i as i64 * 10_0000);
            calc.on_trade(1000 + i * 100, price, 20);
        }

        let result = calc.get_vpin_with_stats();
        
        // Verify toxicity level matches VPIN value
        match result.toxicity_level {
            ToxicityLevel::Extreme => assert!(result.vpin > 0.8),
            ToxicityLevel::High => assert!(result.vpin > 0.6 && result.vpin <= 0.8),
            ToxicityLevel::Moderate => assert!(result.vpin > 0.4 && result.vpin <= 0.6),
            ToxicityLevel::Low => assert!(result.vpin > 0.2 && result.vpin <= 0.4),
            ToxicityLevel::Minimal => assert!(result.vpin <= 0.2),
        }
    }
}
