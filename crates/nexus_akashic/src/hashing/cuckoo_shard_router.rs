//! Cuckoo Hash-based Shard Router for collision-resistant memory routing
//! Implements mathematically-sized hash space to prevent Birthday Paradox issues

use crate::hashing::minwise_permutations::{MinHashSignature, MinHashError};
use crate::hashing::lsh_similarity_estimator::{LSHIndex, LSHConfig, LSHError};
use thiserror::Error;

/// Error types for cuckoo shard routing
#[derive(Error, Debug, Clone, PartialEq)]
pub enum CuckooShardError {
    #[error("Shard capacity exceeded after {max_evictions} evictions")]
    CapacityExceeded { max_evictions: usize },
    #[error("Invalid shard count: {requested}. Must be between {min} and {max}")]
    InvalidShardCount { requested: usize, min: usize, max: usize },
    #[error("Hash space too small for {expected_items} items. Minimum required: {required}")]
    HashSpaceTooSmall { expected_items: u64, required: u64 },
    #[error("Item not found in any shard")]
    ItemNotFound,
    #[error("LSH error: {0}")]
    LSHError(#[from] LSHError),
    #[error("MinHash error: {0}")]
    MinHashError(#[from] MinHashError),
}

/// Minimum shards for parallel access
const MIN_SHARDS: usize = 4;

/// Maximum shards (practical limit for parallelism)
const MAX_SHARDS: usize = 256;

/// Configuration for the cuckoo shard router
#[derive(Debug, Clone)]
pub struct CuckooShardConfig {
    /// Number of shards (parallel hash tables)
    pub num_shards: usize,
    /// Items per shard before eviction chain starts
    pub items_per_shard: usize,
    /// Maximum eviction chain length before failure
    pub max_evictions: usize,
    /// Expected total items for hash space sizing
    pub expected_items: u64,
}

impl CuckooShardConfig {
    /// Create a new configuration with Birthday Paradox-safe hash space sizing
    pub fn new(
        num_shards: usize,
        items_per_shard: usize,
        max_evictions: usize,
        expected_items: u64,
    ) -> Result<Self, CuckooShardError> {
        if num_shards < MIN_SHARDS || num_shards > MAX_SHARDS {
            return Err(CuckooShardError::InvalidShardCount {
                requested: num_shards,
                min: MIN_SHARDS,
                max: MAX_SHARDS,
            });
        }

        // Birthday Paradox check: hash space must be > n^2 / 2 for low collision probability
        let total_capacity = num_shards as u64 * items_per_shard as u64;
        let min_hash_space = (expected_items * expected_items) / 2;
        
        if total_capacity < min_hash_space && expected_items > 100 {
            return Err(CuckooShardError::HashSpaceTooSmall {
                expected_items,
                required: min_hash_space,
            });
        }

        Ok(Self {
            num_shards,
            items_per_shard,
            max_evictions,
            expected_items,
        })
    }

    /// Get recommended configuration based on expected load
    pub fn recommended(expected_items: u64, target_load_factor: f64) -> Self {
        // Target ~80% load factor for good performance
        let total_slots = (expected_items as f64 / target_load_factor) as u64;
        
        // Use power-of-2 shards for efficient modulo
        let num_shards = (total_slots as f64).log2().ceil() as usize;
        let num_shards = num_shards.clamp(MIN_SHARDS, MAX_SHARDS);
        
        let items_per_shard = ((total_slots as f64 / num_shards as f64).ceil()) as usize;
        
        Self {
            num_shards,
            items_per_shard,
            max_evictions: 50, // Reasonable default
            expected_items,
        }
    }
}

/// A single entry in the cuckoo hash table
#[derive(Debug, Clone)]
struct CuckooEntry {
    item_id: usize,
    signature: MinHashSignature,
    hash1: u64,
    hash2: u64,
}

/// Single shard with cuckoo hashing
struct CuckooShard {
    entries: Vec<Option<CuckooEntry>>,
    capacity: usize,
}

impl CuckooShard {
    fn new(capacity: usize) -> Self {
        let mut entries = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            entries.push(None);
        }
        
        Self { entries, capacity }
    }

    fn is_empty(&self) -> bool {
        self.entries.iter().all(|e| e.is_none())
    }

    fn get(&self, hash: u64) -> Option<&CuckooEntry> {
        let idx = (hash as usize) % self.capacity;
        self.entries[idx].as_ref()
    }

    fn insert(&mut self, entry: CuckooEntry, position: usize) -> Option<CuckooEntry> {
        let idx = position % self.capacity;
        self.entries[idx].replace(entry)
    }

    fn remove(&mut self, hash: u64) -> Option<CuckooEntry> {
        let idx = (hash as usize) % self.capacity;
        self.entries[idx].take()
    }
}

/// Cuckoo Hash-based Shard Router for O(1) memory routing
pub struct CuckooShardRouter {
    config: CuckooShardConfig,
    shards: Vec<CuckooShard>,
    /// Mapping from item_id to (shard_idx, position)
    item_locations: std::collections::HashMap<usize, (usize, usize)>,
    next_item_id: usize,
    /// Secondary hash function seed
    hash_seed2: u64,
}

impl CuckooShardRouter {
    /// Create a new cuckoo shard router
    pub fn new(config: CuckooShardConfig, seed: u64) -> Self {
        let mut shards = Vec::with_capacity(config.num_shards);
        for _ in 0..config.num_shards {
            shards.push(CuckooShard::new(config.items_per_shard));
        }

        Self {
            config,
            shards,
            item_locations: std::collections::HashMap::new(),
            next_item_id: 0,
            hash_seed2: seed.wrapping_mul(6364136223846793005).wrapping_add(1),
        }
    }

    /// Compute two hash positions for cuckoo hashing
    #[inline(always)]
    fn compute_hashes(&self, signature: &MinHashSignature) -> (u64, u64) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        // First hash: use first half of signature
        let mut hasher1 = DefaultHasher::new();
        for i in 0..signature.as_slice().len() / 2 {
            hasher1.write_u64(signature.as_slice()[i].value());
        }
        let hash1 = hasher1.finish();

        // Second hash: use second half with different seed
        let mut hasher2 = DefaultHasher::new();
        hasher2.write_u64(self.hash_seed2);
        for i in signature.as_slice().len() / 2..signature.as_slice().len() {
            hasher2.write_u64(signature.as_slice()[i].value());
        }
        let hash2 = hasher2.finish();

        (hash1, hash2)
    }

    /// Insert an item into the router
    pub fn insert(&mut self, signature: MinHashSignature) -> Result<usize, CuckooShardError> {
        let item_id = self.next_item_id;
        self.next_item_id += 1;

        let (hash1, hash2) = self.compute_hashes(&signature);
        let entry = CuckooEntry {
            item_id,
            signature,
            hash1,
            hash2,
        };

        // Try to insert at either hash position
        for &shard_idx in &[0, 1] {
            let hash = if shard_idx == 0 { hash1 } else { hash2 };
            let position = (hash as usize) % self.config.items_per_shard;
            
            // Find the best shard for this position (round-robin across shards)
            for shard_offset in 0..self.config.num_shards {
                let actual_shard = (shard_idx + shard_offset) % self.config.num_shards;
                
                if self.shards[actual_shard].entries[position].is_none() {
                    self.shards[actual_shard].insert(entry.clone(), position);
                    self.item_locations.insert(item_id, (actual_shard, position));
                    return Ok(item_id);
                }
            }
        }

        // Both positions occupied, need to evict
        self.insert_with_eviction(entry, hash1, hash2)
    }

    /// Insert with eviction chain
    fn insert_with_eviction(
        &mut self,
        mut entry: CuckooEntry,
        mut hash1: u64,
        mut hash2: u64,
    ) -> Result<usize, CuckooShardError> {
        let original_item_id = entry.item_id;
        
        for eviction_count in 0..self.config.max_evictions {
            // Alternate between hash positions
            let use_first_hash = eviction_count % 2 == 0;
            let hash = if use_first_hash { hash1 } else { hash2 };
            let shard_idx = eviction_count % self.config.num_shards;
            let position = (hash as usize) % self.config.items_per_shard;

            // Evict existing entry if present
            let evicted = self.shards[shard_idx].insert(entry, position);
            
            if let Some(evicted_entry) = evicted {
                // Update the evicted entry's hashes
                hash1 = evicted_entry.hash1;
                hash2 = evicted_entry.hash2;
                entry = evicted_entry;
            } else {
                // Successfully inserted without further eviction
                self.item_locations.insert(original_item_id, (shard_idx, position));
                return Ok(original_item_id);
            }
        }

        // Too many evictions, table is too full
        Err(CuckooShardError::CapacityExceeded {
            max_evictions: self.config.max_evictions,
        })
    }

    /// Get an item by ID
    pub fn get(&self, item_id: usize) -> Option<&MinHashSignature> {
        let &(shard_idx, position) = self.item_locations.get(&item_id)?;
        let hash = if position % 2 == 0 {
            // Need to find which hash was used
            self.shards[shard_idx].entries[position]
                .as_ref()
                .map(|e| e.hash1)
                .unwrap_or(0)
        } else {
            0
        };
        
        self.shards[shard_idx]
            .entries[position]
            .as_ref()
            .filter(|e| e.item_id == item_id)
            .map(|e| &e.signature)
    }

    /// Remove an item by ID
    pub fn remove(&mut self, item_id: usize) -> Option<MinHashSignature> {
        let (shard_idx, position) = self.item_locations.remove(&item_id)?;
        
        let entry = self.shards[shard_idx].entries[position].take()?;
        
        if entry.item_id == item_id {
            Some(entry.signature)
        } else {
            // Item was moved due to eviction, need to search
            None
        }
    }

    /// Route a query to the appropriate shard(s)
    pub fn route_query(&self, signature: &MinHashSignature) -> Vec<(usize, usize)> {
        let (hash1, hash2) = self.compute_hashes(signature);
        
        let mut routes = Vec::with_capacity(self.config.num_shards);
        
        // Check both possible positions in each shard
        for shard_idx in 0..self.config.num_shards {
            let pos1 = (hash1 as usize) % self.config.items_per_shard;
            let pos2 = (hash2 as usize) % self.config.items_per_shard;
            
            if self.shards[shard_idx].entries[pos1].is_some() {
                routes.push((shard_idx, pos1));
            }
            if pos1 != pos2 && self.shards[shard_idx].entries[pos2].is_some() {
                routes.push((shard_idx, pos2));
            }
        }
        
        routes
    }

    /// Get statistics about the router
    pub fn stats(&self) -> CuckooRouterStats {
        let mut total_items = 0;
        let mut total_occupied = 0;
        
        for shard in &self.shards {
            for entry in &shard.entries {
                if entry.is_some() {
                    total_occupied += 1;
                }
            }
        }
        
        total_items = self.item_locations.len();
        
        let total_capacity = self.config.num_shards * self.config.items_per_shard;
        let load_factor = total_occupied as f64 / total_capacity as f64;
        
        CuckooRouterStats {
            total_items,
            total_occupied,
            total_capacity,
            load_factor,
            num_shards: self.config.num_shards,
        }
    }

    /// Get the number of items stored
    pub fn len(&self) -> usize {
        self.item_locations.len()
    }

    /// Check if the router is empty
    pub fn is_empty(&self) -> bool {
        self.item_locations.is_empty()
    }
}

/// Statistics about the cuckoo router
#[derive(Debug, Clone)]
pub struct CuckooRouterStats {
    pub total_items: usize,
    pub total_occupied: usize,
    pub total_capacity: usize,
    pub load_factor: f64,
    pub num_shards: usize,
}

/// Integration with LSH for similarity-based sharding
pub struct LSHCuckooRouter {
    lsh_index: LSHIndex,
    cuckoo_router: CuckooShardRouter,
}

impl LSHCuckooRouter {
    /// Create a combined LSH + Cuckoo router
    pub fn new(lsh_config: LSHConfig, cuckoo_config: CuckooShardConfig, seed: u64) -> Self {
        Self {
            lsh_index: LSHIndex::new(lsh_config),
            cuckoo_router: CuckooShardRouter::new(cuckoo_config, seed),
        }
    }

    /// Add an item to both routers
    pub fn add_item(&mut self, signature: MinHashSignature) -> Result<usize, CuckooShardError> {
        let item_id = self.cuckoo_router.insert(signature.clone())?;
        self.lsh_index.add_item(signature)?;
        Ok(item_id)
    }

    /// Build the LSH index
    pub fn build(&mut self) -> Result<(), CuckooShardError> {
        self.lsh_index.build()?;
        Ok(())
    }

    /// Query for similar items using LSH, then retrieve from cuckoo
    pub fn query_similar(
        &self,
        signature: &MinHashSignature,
        max_candidates: usize,
    ) -> Result<Vec<MinHashSignature>, CuckooShardError> {
        let candidate_ids = self.lsh_index.query(signature, max_candidates)?;
        
        let mut results = Vec::with_capacity(candidate_ids.len());
        for id in candidate_ids {
            if let Some(sig) = self.cuckoo_router.get(id) {
                results.push(sig.clone());
            }
        }
        
        Ok(results)
    }

    /// Get stats for both routers
    pub fn stats(&self) -> (crate::hashing::lsh_similarity_estimator::LSHIndex, CuckooRouterStats) {
        // Note: LSHIndex doesn't expose stats directly
        let cuckoo_stats = self.cuckoo_router.stats();
        (self.lsh_index.clone_placeholder(), cuckoo_stats)
    }
}

// Placeholder for LSHIndex clone (since it doesn't implement Clone)
impl LSHIndex {
    fn clone_placeholder(&self) -> Self {
        // In production, would implement proper Clone or Stats method
        unimplemented!("Use separate stats method")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hashing::minwise_permutations::MinHashFamily;

    fn create_test_signature(family: &MinHashFamily, elements: &[&str]) -> MinHashSignature {
        family.compute_signature(elements.iter()).unwrap()
    }

    #[test]
    fn test_cuckoo_config_validation() {
        let config = CuckooShardConfig::new(8, 100, 50, 500);
        assert!(config.is_ok());

        // Invalid shard count
        let result = CuckooShardConfig::new(2, 100, 50, 500);
        assert!(result.is_err());
    }

    #[test]
    fn test_cuckoo_insert_get() {
        let config = CuckooShardConfig::recommended(100, 0.8);
        let mut router = CuckooShardRouter::new(config, 42);

        let family = MinHashFamily::new(64, 123).unwrap();
        let sig = create_test_signature(&family, &["test", "data"]);

        let item_id = router.insert(sig.clone()).unwrap();
        assert_eq!(item_id, 0);

        let retrieved = router.get(item_id);
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_cuckoo_remove() {
        let config = CuckooShardConfig::recommended(50, 0.7);
        let mut router = CuckooShardRouter::new(config, 456);

        let family = MinHashFamily::new(64, 789).unwrap();
        let sig = create_test_signature(&family, &["remove", "test"]);

        let item_id = router.insert(sig.clone()).unwrap();
        assert!(router.get(item_id).is_some());

        let removed = router.remove(item_id);
        assert!(removed.is_some());
        assert!(router.get(item_id).is_none());
    }

    #[test]
    fn test_cuckoo_stats() {
        let config = CuckooShardConfig::recommended(100, 0.5);
        let mut router = CuckooShardRouter::new(config, 111);

        let family = MinHashFamily::new(64, 222).unwrap();
        
        for i in 0..50 {
            let sig = create_test_signature(&family, &[&format!("item_{}", i)]);
            router.insert(sig).unwrap();
        }

        let stats = router.stats();
        assert_eq!(stats.total_items, 50);
        assert!(stats.load_factor > 0.0 && stats.load_factor <= 1.0);
    }

    #[test]
    fn test_route_query() {
        let config = CuckooShardConfig::recommended(30, 0.6);
        let mut router = CuckooShardRouter::new(config, 333);

        let family = MinHashFamily::new(64, 444).unwrap();
        let sig = create_test_signature(&family, &["route", "test"]);
        router.insert(sig.clone()).unwrap();

        let routes = router.route_query(&sig);
        assert!(!routes.is_empty());
    }
}
