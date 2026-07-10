//! Lock-Free Bloom Filter for Deduplication
//! 
//! Uses atomic operations and multiple hash functions to provide
//! thread-safe membership testing without locks.

use std::sync::atomic::{AtomicU64, Ordering};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Configuration for the Bloom filter
#[derive(Debug, Clone)]
pub struct BloomConfig {
    pub num_bits: usize,
    pub num_hashes: usize,
}

impl Default for BloomConfig {
    fn default() -> Self {
        // Optimal for ~1M items with 1% false positive rate
        Self {
            num_bits: 10_000_000,
            num_hashes: 7,
        }
    }
}

/// Lock-free Bloom filter using atomic bit array
pub struct LockFreeBloomFilter {
    bits: Vec<AtomicU64>,
    config: BloomConfig,
    seeds: Vec<u64>,
}

impl LockFreeBloomFilter {
    /// Create a new lock-free bloom filter
    pub fn new(config: BloomConfig) -> Self {
        let num_words = (config.num_bits + 63) / 64;
        let bits: Vec<AtomicU64> = (0..num_words)
            .map(|_| AtomicU64::new(0))
            .collect();
        
        // Generate deterministic seeds for hash functions
        let seeds: Vec<u64> = (0..config.num_hashes)
            .map(|i| i as u64 * 0x9E3779B97F4A7C15)
            .collect();
        
        Self {
            bits,
            config,
            seeds,
        }
    }

    /// Compute multiple hash values for an item
    fn compute_hashes<T: Hash>(&self, item: &T) -> Vec<usize> {
        self.seeds.iter().map(|&seed| {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            item.hash(&mut hasher);
            let hash = hasher.finish();
            (hash as usize) % self.config.num_bits
        }).collect()
    }

    /// Insert an item into the filter (returns true if possibly new)
    pub fn insert<T: Hash>(&self, item: &T) -> bool {
        let hashes = self.compute_hashes(item);
        let mut was_new = true;
        
        for hash in hashes {
            let word_idx = hash / 64;
            let bit_idx = hash % 64;
            let mask = 1u64 << bit_idx;
            
            // Atomically set the bit
            let prev = self.bits[word_idx].fetch_or(mask, Ordering::Relaxed);
            
            // If bit was already set, item might be duplicate
            if prev & mask != 0 {
                was_new = false;
            }
        }
        
        was_new
    }

    /// Check if an item might be in the filter
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let hashes = self.compute_hashes(item);
        
        for hash in hashes {
            let word_idx = hash / 64;
            let bit_idx = hash % 64;
            let mask = 1u64 << bit_idx;
            
            if self.bits[word_idx].load(Ordering::Relaxed) & mask == 0 {
                return false;
            }
        }
        
        true
    }

    /// Clear all bits (not thread-safe, use with caution)
    pub fn clear(&self) {
        for word in &self.bits {
            word.store(0, Ordering::Relaxed);
        }
    }

    /// Get approximate fill ratio
    pub fn fill_ratio(&self) -> f64 {
        let set_bits: u64 = self.bits.iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones() as u64)
            .sum();
        
        set_bits as f64 / self.config.num_bits as f64
    }
}

/// Adaptive rate limiter using token bucket algorithm
pub struct AdaptiveRateLimiter {
    tokens: AtomicU64,
    max_tokens: u64,
    refill_rate: AtomicU64, // tokens per second
    last_refill: AtomicU64, // timestamp in microseconds
}

impl AdaptiveRateLimiter {
    /// Create a new adaptive rate limiter
    pub fn new(max_tokens: u64, initial_rate: u64) -> Self {
        Self {
            tokens: AtomicU64::new(max_tokens),
            max_tokens,
            refill_rate: AtomicU64::new(initial_rate),
            last_refill: AtomicU64::new(0),
        }
    }

    /// Try to acquire a token (returns true if successful)
    pub fn try_acquire(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        
        // Refill tokens based on elapsed time
        let last = self.last_refill.swap(now, Ordering::Relaxed);
        let elapsed = now.saturating_sub(last);
        
        if elapsed > 0 {
            let rate = self.refill_rate.load(Ordering::Relaxed);
            let refill = (elapsed * rate) / 1_000_000;
            
            self.tokens.fetch_min(self.max_tokens, Ordering::Relaxed);
            self.tokens.fetch_add(refill, Ordering::Relaxed);
        }
        
        // Try to consume a token
        loop {
            let current = self.tokens.load(Ordering::Relaxed);
            if current == 0 {
                return false;
            }
            
            if self.tokens.compare_exchange_weak(
                current,
                current - 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ).is_ok() {
                return true;
            }
        }
    }

    /// Adjust the rate based on conditions
    pub fn adjust_rate(&self, factor: f64) {
        let current = self.refill_rate.load(Ordering::Relaxed) as f64;
        let new_rate = (current * factor).clamp(1.0, 1_000_000.0) as u64;
        self.refill_rate.store(new_rate, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter() {
        let config = BloomConfig {
            num_bits: 1024,
            num_hashes: 3,
        };
        let filter = LockFreeBloomFilter::new(config);
        
        assert!(!filter.contains(&"hello"));
        assert!(filter.insert(&"hello"));
        assert!(filter.contains(&"hello"));
        assert!(!filter.contains(&"world"));
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = AdaptiveRateLimiter::new(10, 100);
        
        // Should allow initial tokens
        for _ in 0..10 {
            assert!(limiter.try_acquire());
        }
        
        // Should deny when exhausted
        assert!(!limiter.try_acquire());
    }
}
