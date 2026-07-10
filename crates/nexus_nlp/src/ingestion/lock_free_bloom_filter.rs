//! Lock-Free Bloom Filter for high-speed deduplication of news alerts
//!
//! This implementation uses atomic operations to ensure thread-safe access
//! without mutex contention, critical for handling thousands of news items
//! per second during market-moving events.

use std::sync::atomic::{AtomicU64, Ordering};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Number of hash functions to use
const NUM_HASH_FUNCTIONS: usize = 7;

/// Bloom filter size in bits (power of 2 for fast modulo)
const BLOOM_SIZE_BITS: usize = 1 << 20; // 1M bits = 128KB
const BLOOM_SIZE_MASK: u64 = (BLOOM_SIZE_BITS - 1) as u64;

/// A lock-free Bloom filter using atomic bit operations
pub struct LockFreeBloomFilter {
    /// Bit array stored as atomic u64 values
    bits: Box<[AtomicU64; BLOOM_SIZE_BITS / 64]>,
}

impl LockFreeBloomFilter {
    /// Create a new empty Bloom filter
    pub fn new() -> Self {
        // SAFETY: AtomicU64 has the same representation as u64 and can be zero-initialized
        let bits = unsafe {
            let mut bits: Box<[AtomicU64; BLOOM_SIZE_BITS / 64]> = 
                Box::new_uninit_slice(BLOOM_SIZE_BITS / 64);
            let slice = bits.as_mut_ptr() as *mut AtomicU64;
            std::ptr::write_bytes(slice, 0, BLOOM_SIZE_BITS / 64);
            bits.assume_init()
        };
        Self { bits }
    }

    /// Compute multiple hash values for an item using different seeds
    #[inline]
    fn compute_hashes<T: Hash>(&self, item: &T) -> [usize; NUM_HASH_FUNCTIONS] {
        let mut hashes = [0usize; NUM_HASH_FUNCTIONS];
        
        for i in 0..NUM_HASH_FUNCTIONS {
            let mut hasher = DefaultHasher::new();
            // Mix in the seed to get different hash values
            item.hash(&mut hasher);
            (i as u64).hash(&mut hasher);
            hashes[i] = (hasher.finish() & BLOOM_SIZE_MASK) as usize;
        }
        
        hashes
    }

    /// Insert an item into the Bloom filter
    /// Returns true if the item was newly inserted (probably), false if it was already present
    pub fn insert<T: Hash>(&self, item: &T) -> bool {
        let hashes = self.compute_hashes(item);
        let mut was_new = true;

        for hash in hashes {
            let word_idx = hash / 64;
            let bit_idx = hash % 64;
            let mask = 1u64 << bit_idx;

            // Atomically set the bit
            let old = self.bits[word_idx].fetch_or(mask, Ordering::Relaxed);
            
            // If the bit was already set, this item might have been seen before
            if old & mask != 0 {
                // Continue setting other bits but mark as not new
                was_new = false;
            }
        }

        was_new
    }

    /// Check if an item might be in the Bloom filter
    /// Returns true if possibly present, false if definitely not present
    #[inline]
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

    /// Clear all bits in the filter (use with caution in production)
    pub fn clear(&self) {
        for word in self.bits.iter() {
            word.store(0, Ordering::Relaxed);
        }
    }

    /// Estimate the fill ratio (for monitoring purposes)
    pub fn fill_ratio(&self) -> f64 {
        let mut set_bits: u64 = 0;
        for word in self.bits.iter() {
            set_bits += word.load(Ordering::Relaxed).count_ones() as u64;
        }
        set_bits as f64 / BLOOM_SIZE_BITS as f64
    }
}

impl Default for LockFreeBloomFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insert_contains() {
        let filter = LockFreeBloomFilter::new();
        
        assert!(!filter.contains(&"hello"));
        assert!(filter.insert(&"hello"));
        assert!(filter.contains(&"hello"));
        assert!(!filter.contains(&"world"));
    }

    #[test]
    fn test_false_positive_rate() {
        let filter = LockFreeBloomFilter::new();
        
        // Insert 10000 items
        for i in 0..10000 {
            filter.insert(&i);
        }
        
        // Check false positive rate on 10000 non-inserted items
        let mut false_positives = 0;
        for i in 10000..20000 {
            if filter.contains(&i) {
                false_positives += 1;
            }
        }
        
        let fp_rate = false_positives as f64 / 10000.0;
        // False positive rate should be reasonably low (< 5%)
        assert!(fp_rate < 0.05, "False positive rate too high: {}", fp_rate);
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;
        
        let filter = std::sync::Arc::new(LockFreeBloomFilter::new());
        let mut handles = vec![];
        
        // Spawn multiple threads inserting and checking
        for t in 0..10 {
            let filter_clone = filter.clone();
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    let val = t * 1000 + i;
                    filter_clone.insert(&val);
                    assert!(filter_clone.contains(&val));
                }
            }));
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
    }
}
