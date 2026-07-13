//! VPIN (Volume-Synchronized Probability of Informed Trading) Toxicity Calculator
//! 
//! VPIN is the ultimate institutional toxicity metric. It groups live trades
//! into "Volume Buckets" and calculates the absolute difference between
//! buyer-initiated and seller-initiated volume in each bucket.
//! 
//! High VPIN indicates informed institutional flow is toxic, signaling the bot
//! to widen spreads or halt trading.
//! 
//! Formula: VPIN = |V_buy - V_sell| / (V_buy + V_sell) per volume bucket
//! Range: [0.0, 1.0] where high values = toxic flow
//! 
//! ZERO-ALLOC: Uses fixed-size arrays for bucket storage.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// A single volume bucket for VPIN calculation
#[derive(Debug, Clone, Copy)]
pub struct VolumeBucket {
    /// Total buy volume in this bucket
    pub buy_volume: f64,
    /// Total sell volume in this bucket
    pub sell_volume: f64,
    /// Target volume for this bucket
    pub target_volume: f64,
    /// Whether this bucket is complete
    pub is_complete: bool,
}

impl VolumeBucket {
    pub fn new(target_volume: f64) -> Self {
        Self {
            buy_volume: 0.0,
            sell_volume: 0.0,
            target_volume,
            is_complete: false,
        }
    }

    /// Get total volume in bucket
    pub fn total_volume(&self) -> f64 {
        self.buy_volume + self.sell_volume
    }

    /// Check if bucket is full
    pub fn is_full(&self) -> bool {
        self.total_volume() >= self.target_volume
    }

    /// Calculate VPIN for this bucket
    pub fn vpin(&self) -> f64 {
        let total = self.buy_volume + self.sell_volume;
        if total <= 0.0 {
            return 0.0;
        }
        (self.buy_volume - self.sell_volume).abs() / total
    }

    /// Add a trade to the bucket
    /// Returns remaining volume that couldn't fit
    pub fn add_trade(&mut self, volume: f64, is_buy: bool) -> f64 {
        let remaining_capacity = self.target_volume - self.total_volume();
        
        if remaining_capacity <= 0.0 {
            // Bucket already full
            return volume;
        }

        let volume_to_add = volume.min(remaining_capacity);
        let remaining = volume - volume_to_add;

        if is_buy {
            self.buy_volume += volume_to_add;
        } else {
            self.sell_volume += volume_to_add;
        }

        if self.is_full() {
            self.is_complete = true;
        }

        remaining
    }
}

/// Configuration for VPIN calculation
#[derive(Debug, Clone, Copy)]
pub struct VpinConfig {
    /// Target volume per bucket (e.g., 10 BTC)
    pub bucket_volume: f64,
    /// Number of buckets to average over
    pub num_buckets: usize,
    /// VPIN threshold for toxicity warning
    pub toxicity_threshold: f64,
}

impl Default for VpinConfig {
    fn default() -> Self {
        Self {
            bucket_volume: 10.0, // 10 units per bucket
            num_buckets: 50,     // Average over 50 buckets
            toxicity_threshold: 0.7, // 70% imbalance = toxic
        }
    }
}

/// VPIN Toxicity calculator with fractional bucket splitting
pub struct VpinToxicity<const MAX_BUCKETS: usize = 100> {
    config: VpinConfig,
    /// Circular buffer of volume buckets (stack-allocated)
    buckets: [VolumeBucket; MAX_BUCKETS],
    /// Current write index
    write_idx: usize,
    /// Number of completed buckets
    completed_count: usize,
    /// Current active bucket
    current_bucket: VolumeBucket,
    /// Rolling VPIN average
    rolling_vpin: f64,
    /// Sum of VPIN values for rolling average
    vpin_sum: f64,
    /// Last update timestamp
    last_update_ns: AtomicU64,
    /// Trade count
    trade_count: AtomicU64,
    /// Toxicity alert flag
    is_toxic: AtomicBool,
}

impl<const MAX_BUCKETS: usize> VpinToxicity<MAX_BUCKETS> {
    pub fn new() -> Self {
        Self::with_config(VpinConfig::default())
    }

    pub fn with_config(config: VpinConfig) -> Self {
        assert!(config.num_buckets <= MAX_BUCKETS, "num_buckets must be <= MAX_BUCKETS");
        
        Self {
            config,
            buckets: std::array::from_fn(|_| VolumeBucket::new(config.bucket_volume)),
            write_idx: 0,
            completed_count: 0,
            current_bucket: VolumeBucket::new(config.bucket_volume),
            rolling_vpin: 0.0,
            vpin_sum: 0.0,
            last_update_ns: AtomicU64::new(0),
            trade_count: AtomicU64::new(0),
            is_toxic: AtomicBool::new(false),
        }
    }

    /// Process a trade and update VPIN
    /// 
    /// ROOT CAUSE FIX: Properly handles whale orders that exceed bucket size
    /// by splitting across multiple buckets
    pub fn process_trade(&mut self, volume: f64, price: f64, is_buy: bool, timestamp_ns: u64) -> Option<f64> {
        if volume <= 0.0 {
            return None;
        }

        self.trade_count.fetch_add(1, Ordering::Relaxed);

        // ROOT CAUSE FIX: Handle whale orders that exceed bucket size
        let mut remaining_volume = volume;
        let mut last_vpin = None;

        while remaining_volume > 0.0 {
            // Add trade to current bucket, get remaining volume
            let leftover = self.current_bucket.add_trade(remaining_volume, is_buy);
            remaining_volume = leftover;

            // If bucket is full, finalize it and start new one
            if self.current_bucket.is_full() {
                let vpin = self.finalize_bucket();
                last_vpin = Some(vpin);
                
                // Start new bucket
                self.current_bucket = VolumeBucket::new(self.config.bucket_volume);
            }
        }

        self.last_update_ns.store(timestamp_ns, Ordering::Relaxed);

        // Update toxicity flag
        self.is_toxic.store(
            self.rolling_vpin > self.config.toxicity_threshold,
            Ordering::Release,
        );

        last_vpin
    }

    /// Finalize current bucket and update rolling average
    fn finalize_bucket(&mut self) -> f64 {
        let vpin = self.current_bucket.vpin();
        
        // Store bucket in circular buffer
        let idx = self.write_idx % MAX_BUCKETS;
        
        // If we're overwriting an old bucket, subtract its VPIN from sum
        if self.completed_count >= MAX_BUCKETS {
            let old_vpin = self.buckets[idx].vpin();
            self.vpin_sum -= old_vpin;
        } else {
            self.completed_count += 1;
        }

        // Store new bucket and add its VPIN
        self.buckets[idx] = self.current_bucket;
        self.vpin_sum += vpin;
        
        // Update rolling average
        let effective_count = self.completed_count.min(MAX_BUCKETS);
        self.rolling_vpin = self.vpin_sum / effective_count as f64;

        self.write_idx += 1;

        vpin
    }

    /// Get current rolling VPIN
    pub fn rolling_vpin(&self) -> f64 {
        self.rolling_vpin
    }

    /// Get current bucket's partial VPIN
    pub fn current_bucket_vpin(&self) -> f64 {
        self.current_bucket.vpin()
    }

    /// Check if market is currently toxic
    pub fn is_toxic(&self) -> bool {
        self.is_toxic.load(Ordering::Acquire)
    }

    /// Get toxicity level [0.0, 1.0]
    pub fn toxicity_level(&self) -> f64 {
        self.rolling_vpin.clamp(0.0, 1.0)
    }

    /// Get recommended spread multiplier based on toxicity
    /// Higher VPIN = wider spreads needed
    pub fn spread_multiplier(&self) -> f64 {
        // Base multiplier of 1.0, increases with toxicity
        1.0 + (self.rolling_vpin * 2.0) // Max 3x spread at VPIN=1.0
    }

    /// Get whether trading should be halted
    pub fn should_halt_trading(&self, halt_threshold: f64) -> bool {
        self.rolling_vpin > halt_threshold
    }

    /// Get completed bucket count
    pub fn completed_buckets(&self) -> usize {
        self.completed_count
    }

    /// Get trade count
    pub fn trade_count(&self) -> u64 {
        self.trade_count.load(Ordering::Relaxed)
    }

    /// Get last update timestamp
    pub fn last_update_ns(&self) -> u64 {
        self.last_update_ns.load(Ordering::Relaxed)
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.buckets = std::array::from_fn(|_| VolumeBucket::new(self.config.bucket_volume));
        self.write_idx = 0;
        self.completed_count = 0;
        self.current_bucket = VolumeBucket::new(self.config.bucket_volume);
        self.rolling_vpin = 0.0;
        self.vpin_sum = 0.0;
        self.is_toxic.store(false, Ordering::Release);
    }
}

impl<const MAX_BUCKETS: usize> Default for VpinToxicity<MAX_BUCKETS> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vpin_basic() {
        let mut vpin = VpinToxicity::<10>::with_config(VpinConfig {
            bucket_volume: 100.0,
            num_buckets: 5,
            toxicity_threshold: 0.7,
        });

        // Add balanced buy/sell volume
        vpin.process_trade(50.0, 100.0, true, 1000);
        vpin.process_trade(50.0, 100.0, false, 2000);

        // Bucket should be full now (100 total)
        assert!(vpin.current_bucket.is_full());
        
        // VPIN should be 0 (perfectly balanced)
        assert_eq!(vpin.current_bucket_vpin(), 0.0);
    }

    #[test]
    fn test_vpin_imbalanced() {
        let mut vpin = VpinToxicity::<10>::with_config(VpinConfig {
            bucket_volume: 100.0,
            num_buckets: 5,
            toxicity_threshold: 0.7,
        });

        // Add only buy volume
        vpin.process_trade(100.0, 100.0, true, 1000);

        // Bucket should be full
        assert!(vpin.current_bucket.is_full());
        
        // VPIN should be 1.0 (all buys)
        assert!((vpin.current_bucket_vpin() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vpin_whale_order_splitting() {
        let mut vpin = VpinToxicity::<10>::with_config(VpinConfig {
            bucket_volume: 100.0,
            num_buckets: 5,
            toxicity_threshold: 0.7,
        });

        // Whale order: 350 units (should span 3.5 buckets)
        vpin.process_trade(350.0, 100.0, true, 1000);

        // Should have completed 3 full buckets
        assert_eq!(vpin.completed_buckets(), 3);
        
        // Current bucket should have 50 units remaining
        assert!((vpin.current_bucket.total_volume() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_vpin_toxicity_detection() {
        let mut vpin = VpinToxicity::<10>::with_config(VpinConfig {
            bucket_volume: 100.0,
            num_buckets: 5,
            toxicity_threshold: 0.7,
        });

        // Add several imbalanced buckets
        for _ in 0..5 {
            vpin.process_trade(100.0, 100.0, true, 1000);
        }

        // Should detect toxicity
        assert!(vpin.is_toxic());
        assert!(vpin.toxicity_level() > 0.7);
        
        // Spread should be widened
        assert!(vpin.spread_multiplier() > 2.0);
    }

    #[test]
    fn test_vpin_zero_volume() {
        let mut vpin = VpinToxicity::<10>::new();

        // Zero volume trade should be ignored
        assert!(vpin.process_trade(0.0, 100.0, true, 1000).is_none());
    }
}
