//! Easley-O'Hara VPIN (Volume-Synchronized Probability of Informed Trading) implementation.
//! 
//! Calculates VPIN using volume-synchronized buckets rather than time-based buckets,
//! providing a more robust measure of order flow toxicity.

use std::collections::VecDeque;
use parking_lot::RwLock;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VpinError {
    #[error("Invalid bucket size")]
    InvalidBucketSize,
    #[error("Insufficient data for VPIN calculation")]
    InsufficientData,
}

/// VPIN estimation result
#[derive(Debug, Clone)]
pub struct VpinEstimate {
    /// Current VPIN value (0-1)
    pub vpin: f64,
    /// Number of buckets used
    pub bucket_count: usize,
    /// Average buy volume per bucket
    pub avg_buy_volume: u64,
    /// Average sell volume per bucket
    pub avg_sell_volume: u64,
    /// Timestamp
    pub timestamp_ns: u64,
}

/// Volume bucket for VPIN calculation
struct VolumeBucket {
    buy_volume: u64,
    sell_volume: u64,
}

impl VolumeBucket {
    fn new() -> Self {
        Self {
            buy_volume: 0,
            sell_volume: 0,
        }
    }
}

/// Configuration for VPIN estimator
pub struct VpinConfig {
    /// Target volume per bucket (in base units)
    pub bucket_volume: u64,
    /// Number of buckets to use for rolling estimate
    pub num_buckets: usize,
    /// Minimum buckets required before valid estimate
    pub min_buckets: usize,
}

impl Default for VpinConfig {
    fn default() -> Self {
        Self {
            bucket_volume: 10000, // 10K units per bucket
            num_buckets: 50,
            min_buckets: 10,
        }
    }
}

/// Easley-O'Hara VPIN Estimator
pub struct EasleyOHaravPIN {
    config: VpinConfig,
    /// Rolling buffer of volume buckets
    buckets: RwLock<VecDeque<VolumeBucket>>,
    /// Current incomplete bucket
    current_bucket: RwLock<VolumeBucket>,
    /// Current bucket volume accumulator
    current_bucket_volume: RwLock<u64>,
}

impl EasleyOHaravPIN {
    /// Create a new VPIN estimator
    pub fn new(config: VpinConfig) -> Result<Self, VpinError> {
        if config.bucket_volume == 0 || config.num_buckets == 0 || config.min_buckets > config.num_buckets {
            return Err(VpinError::InvalidBucketSize);
        }
        
        Ok(Self {
            config,
            buckets: RwLock::new(VecDeque::with_capacity(config.num_buckets)),
            current_bucket: RwLock::new(VolumeBucket::new()),
            current_bucket_volume: RwLock::new(0),
        })
    }

    /// Record a trade with volume and direction
    #[inline]
    pub fn record_trade(&self, volume: u64, is_buy: bool, timestamp_ns: u64) {
        let mut current_vol = self.current_bucket_volume.write();
        let mut current_bucket = self.current_bucket.write();
        
        // Add volume to appropriate side
        if is_buy {
            current_bucket.buy_volume += volume;
        } else {
            current_bucket.sell_volume += volume;
        }
        
        *current_vol += volume;
        
        // Check if bucket is full
        if *current_vol >= self.config.bucket_volume {
            // Move current bucket to rolling buffer
            drop(current_vol);
            drop(current_bucket);
            
            self.finalize_bucket();
        }
    }

    /// Finalize current bucket and manage rolling window
    fn finalize_bucket(&self) {
        let mut current_bucket = self.current_bucket.write();
        let mut current_vol = self.current_bucket_volume.write();
        let mut buckets = self.buckets.write();
        
        // Push completed bucket
        let completed_bucket = VolumeBucket {
            buy_volume: current_bucket.buy_volume,
            sell_volume: current_bucket.sell_volume,
        };
        
        buckets.push_back(completed_bucket);
        
        // Maintain fixed window size
        while buckets.len() > self.config.num_buckets {
            buckets.pop_front();
        }
        
        // Reset current bucket
        *current_bucket = VolumeBucket::new();
        *current_vol = 0;
    }

    /// Calculate current VPIN estimate
    /// 
    /// VPIN = (1/n) * Σ|V_buy - V_sell| / (V_buy + V_sell)
    /// where the sum is over all buckets
    pub fn calculate_vpin(&self, timestamp_ns: u64) -> Result<VpinEstimate, VpinError> {
        let buckets = self.buckets.read();
        
        if buckets.len() < self.config.min_buckets {
            return Err(VpinError::InsufficientData);
        }
        
        let mut total_abs_imbalance = 0u128;
        let mut total_volume = 0u128;
        let mut total_buy = 0u64;
        let mut total_sell = 0u64;
        
        for bucket in buckets.iter() {
            let bucket_total = (bucket.buy_volume + bucket.sell_volume) as u128;
            let abs_imbalance = (bucket.buy_volume as i128 - bucket.sell_volume as i128).unsigned_abs() as u128;
            
            total_abs_imbalance += abs_imbalance;
            total_volume += bucket_total;
            total_buy += bucket.buy_volume;
            total_sell += bucket.sell_volume;
        }
        
        let vpin = if total_volume > 0 {
            (total_abs_imbalance as f64) / (total_volume as f64)
        } else {
            0.0
        };
        
        // Clamp to [0, 1]
        let vpin_clamped = vpin.min(1.0).max(0.0);
        
        Ok(VpinEstimate {
            vpin: vpin_clamped,
            bucket_count: buckets.len(),
            avg_buy_volume: total_buy / buckets.len() as u64,
            avg_sell_volume: total_sell / buckets.len() as u64,
            timestamp_ns,
        })
    }

    /// Get current VPIN value (returns None if insufficient data)
    pub fn get_vpin(&self) -> Option<f64> {
        self.calculate_vpin(0).ok().map(|e| e.vpin)
    }

    /// Check if market is toxic (VPIN above threshold)
    pub fn is_toxic(&self, threshold: f64) -> bool {
        if let Ok(estimate) = self.calculate_vpin(0) {
            estimate.vpin > threshold
        } else {
            false
        }
    }

    /// Get toxicity level classification
    pub fn get_toxicity_level(&self) -> ToxicityLevel {
        match self.get_vpin() {
            Some(vpin) if vpin > 0.7 => ToxicityLevel::Extreme,
            Some(vpin) if vpin > 0.5 => ToxicityLevel::High,
            Some(vpin) if vpin > 0.3 => ToxicityLevel::Moderate,
            Some(_) => ToxicityLevel::Low,
            None => ToxicityLevel::Unknown,
        }
    }

    /// Reset estimator state
    pub fn reset(&self) {
        self.buckets.write().clear();
        *self.current_bucket.write() = VolumeBucket::new();
        *self.current_bucket_volume.write() = 0;
    }
}

/// Toxicity level classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToxicityLevel {
    Unknown,
    Low,
    Moderate,
    High,
    Extreme,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vpin_balanced_flow() {
        let estimator = EasleyOHaravPIN::new(VpinConfig::default()).unwrap();
        
        // Balanced buy/sell flow should give low VPIN
        for i in 0..1000 {
            let volume = 100;
            let is_buy = i % 2 == 0;
            estimator.record_trade(volume, is_buy, i * 1_000_000);
        }
        
        let estimate = estimator.calculate_vpin(1_000_000_000).unwrap();
        assert!(estimate.vpin < 0.3); // Should be low for balanced flow
    }

    #[test]
    fn test_vpin_imbalanced_flow() {
        let estimator = EasleyOHaravPIN::new(VpinConfig::default()).unwrap();
        
        // Heavily imbalanced flow (mostly buys) should give high VPIN
        for i in 0..1000 {
            let volume = 100;
            let is_buy = i % 10 != 0; // 90% buys
            estimator.record_trade(volume, is_buy, i * 1_000_000);
        }
        
        let estimate = estimator.calculate_vpin(1_000_000_000).unwrap();
        assert!(estimate.vpin > 0.5); // Should be high for imbalanced flow
    }

    #[test]
    fn test_insufficient_data() {
        let estimator = EasleyOHaravPIN::new(VpinConfig::default()).unwrap();
        
        // Only a few trades, not enough buckets
        for i in 0..5 {
            estimator.record_trade(100, true, i * 1_000_000);
        }
        
        let result = estimator.calculate_vpin(5_000_000);
        assert!(result.is_err());
    }
}
