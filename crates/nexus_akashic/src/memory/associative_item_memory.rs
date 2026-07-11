//! Associative Item Memory for Hyper-Dimensional Computing
//! Stores and retrieves items based on similarity in HDC space

use crate::hdc::bipolar_vector_generator::{BipolarVector, BipolarVectorError};
use thiserror::Error;

/// Error types for associative memory operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum AssociativeMemoryError {
    #[error("Memory capacity exceeded: {current} >= {max}")]
    CapacityExceeded { current: usize, max: usize },
    #[error("Item not found with sufficient similarity")]
    NotFound,
    #[error("Invalid similarity threshold: {value}. Must be between -1 and 1")]
    InvalidThreshold { value: f64 },
    #[error("HDC error: {0}")]
    HdcError(#[from] BipolarVectorError),
}

/// Configuration for associative memory
#[derive(Debug, Clone)]
pub struct AssociativeMemoryConfig {
    /// Maximum number of items to store
    pub max_items: usize,
    /// Minimum similarity threshold for retrieval
    pub similarity_threshold: f64,
    /// Number of nearest neighbors to return
    pub k_nearest: usize,
}

impl AssociativeMemoryConfig {
    pub fn new(max_items: usize, similarity_threshold: f64, k_nearest: usize) -> Result<Self, AssociativeMemoryError> {
        if similarity_threshold < -1.0 || similarity_threshold > 1.0 {
            return Err(AssociativeMemoryError::InvalidThreshold {
                value: similarity_threshold,
            });
        }

        Ok(Self {
            max_items,
            similarity_threshold,
            k_nearest,
        })
    }

    /// Get recommended configuration for typical use cases
    pub fn recommended() -> Self {
        Self {
            max_items: 10_000,
            similarity_threshold: 0.7,
            k_nearest: 5,
        }
    }
}

/// A stored item in associative memory
#[derive(Debug, Clone)]
pub struct MemoryItem {
    pub id: usize,
    pub vector: BipolarVector,
    pub metadata: Vec<u8>,
    pub access_count: u64,
    pub last_access_ns: u64,
}

/// Associative Item Memory for HDC-based storage and retrieval
pub struct AssociativeItemMemory {
    config: AssociativeMemoryConfig,
    items: Vec<MemoryItem>,
    next_id: usize,
    /// Cached statistics
    total_accesses: u64,
}

impl AssociativeItemMemory {
    /// Create a new associative memory with the given configuration
    pub fn new(config: AssociativeMemoryConfig) -> Self {
        Self {
            config,
            items: Vec::with_capacity(config.max_items),
            next_id: 0,
            total_accesses: 0,
        }
    }

    /// Store an item in memory
    pub fn store(&mut self, vector: BipolarVector, metadata: Vec<u8>) -> Result<usize, AssociativeMemoryError> {
        if self.items.len() >= self.config.max_items {
            return Err(AssociativeMemoryError::CapacityExceeded {
                current: self.items.len(),
                max: self.config.max_items,
            });
        }

        let id = self.next_id;
        self.next_id += 1;

        let item = MemoryItem {
            id,
            vector,
            metadata,
            access_count: 0,
            last_access_ns: 0,
        };

        self.items.push(item);
        Ok(id)
    }

    /// Retrieve the most similar item to a query vector
    pub fn retrieve(&mut self, query: &BipolarVector, timestamp_ns: u64) -> Result<&MemoryItem, AssociativeMemoryError> {
        self.retrieve_with_similarity(query, timestamp_ns)
            .and_then(|(item, sim)| {
                if sim >= self.config.similarity_threshold {
                    Ok(item)
                } else {
                    Err(AssociativeMemoryError::NotFound)
                }
            })
    }

    /// Retrieve with similarity score
    pub fn retrieve_with_similarity(
        &mut self,
        query: &BipolarVector,
        timestamp_ns: u64,
    ) -> Result<(&MemoryItem, f64), AssociativeMemoryError> {
        if self.items.is_empty() {
            return Err(AssociativeMemoryError::NotFound);
        }

        let mut best_idx = 0;
        let mut best_sim = f64::NEG_INFINITY;

        for (idx, item) in self.items.iter().enumerate() {
            let sim = query.cosine_similarity(&item.vector);
            if sim > best_sim {
                best_sim = sim;
                best_idx = idx;
            }
        }

        // Update access statistics
        self.items[best_idx].access_count += 1;
        self.items[best_idx].last_access_ns = timestamp_ns;
        self.total_accesses += 1;

        Ok((&self.items[best_idx], best_sim))
    }

    /// Retrieve k-nearest neighbors
    pub fn retrieve_k_nearest(
        &mut self,
        query: &BipolarVector,
        timestamp_ns: u64,
    ) -> Vec<(&MemoryItem, f64)> {
        if self.items.is_empty() {
            return Vec::new();
        }

        let mut candidates: Vec<(usize, f64)> = self
            .items
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                let sim = query.cosine_similarity(&item.vector);
                (idx, sim)
            })
            .collect();

        // Sort by similarity descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        let k = self.config.k_nearest.min(candidates.len());
        let mut results = Vec::with_capacity(k);

        for (idx, sim) in candidates.into_iter().take(k) {
            if sim >= self.config.similarity_threshold {
                self.items[idx].access_count += 1;
                self.items[idx].last_access_ns = timestamp_ns;
                results.push((&self.items[idx], sim));
            }
        }

        self.total_accesses += k as u64;
        results
    }

    /// Remove an item by ID
    pub fn remove(&mut self, id: usize) -> Option<MemoryItem> {
        if let Some(pos) = self.items.iter().position(|item| item.id == id) {
            Some(self.items.remove(pos))
        } else {
            None
        }
    }

    /// Clear all items from memory
    pub fn clear(&mut self) {
        self.items.clear();
        self.total_accesses = 0;
    }

    /// Get the number of stored items
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if memory is empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        let total_accesses = self.total_accesses;
        let avg_access_count = if self.items.is_empty() {
            0.0
        } else {
            self.items.iter().map(|i| i.access_count as f64).sum::<f64>() / self.items.len() as f64
        };

        MemoryStats {
            total_items: self.items.len(),
            max_items: self.config.max_items,
            total_accesses,
            avg_access_count,
            capacity_utilization: self.items.len() as f64 / self.config.max_items as f64,
        }
    }

    /// Find items by metadata prefix (for debugging/inspection)
    pub fn find_by_metadata_prefix(&self, prefix: &[u8]) -> Vec<&MemoryItem> {
        self.items
            .iter()
            .filter(|item| item.metadata.starts_with(prefix))
            .collect()
    }
}

/// Statistics about associative memory usage
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_items: usize,
    pub max_items: usize,
    pub total_accesses: u64,
    pub avg_access_count: f64,
    pub capacity_utilization: f64,
}

/// Continuous Item Memory for storing analog values
pub struct ContinuousItemMemory {
    items: std::collections::HashMap<u64, ContinuousMemoryItem>,
    next_key: u64,
}

#[derive(Debug, Clone)]
struct ContinuousMemoryItem {
    key: u64,
    value: f64,
    vector: BipolarVector,
    count: u64,
}

impl ContinuousItemMemory {
    pub fn new() -> Self {
        Self {
            items: std::collections::HashMap::new(),
            next_key: 0,
        }
    }

    /// Add or update a continuous value
    pub fn add(&mut self, value: f64, vector: BipolarVector) -> u64 {
        // Check if similar vector exists
        for item in self.items.values_mut() {
            let sim = vector.cosine_similarity(&item.vector);
            if sim > 0.95 {
                // Merge with existing item (running average)
                let new_count = item.count + 1;
                item.value = (item.value * item.count as f64 + value) / new_count as f64;
                item.count = new_count;
                return item.key;
            }
        }

        // Create new item
        let key = self.next_key;
        self.next_key += 1;

        self.items.insert(
            key,
            ContinuousMemoryItem {
                key,
                value,
                vector,
                count: 1,
            },
        );

        key
    }

    /// Query for similar vector and return associated value
    pub fn query(&self, vector: &BipolarVector) -> Option<(f64, f64)> {
        let mut best_item: Option<&ContinuousMemoryItem> = None;
        let mut best_sim = f64::NEG_INFINITY;

        for item in self.items.values() {
            let sim = vector.cosine_similarity(&item.vector);
            if sim > best_sim {
                best_sim = sim;
                best_item = Some(item);
            }
        }

        best_item.map(|item| (item.value, best_sim))
    }

    /// Get all stored values
    pub fn all_values(&self) -> Vec<(u64, f64, u64)> {
        self.items
            .values()
            .map(|item| (item.key, item.value, item.count))
            .collect()
    }
}

impl Default for ContinuousItemMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::bipolar_vector_generator::BipolarVectorGenerator;

    #[test]
    fn test_store_retrieve() {
        let config = AssociativeMemoryConfig::recommended();
        let mut memory = AssociativeItemMemory::new(config);

        let mut gen = BipolarVectorGenerator::new(42);
        let vector = gen.generate().unwrap();
        let metadata = vec![1, 2, 3, 4];

        let id = memory.store(vector.clone(), metadata.clone()).unwrap();
        assert_eq!(id, 0);

        let retrieved = memory.retrieve(&vector, 1000).unwrap();
        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.metadata, metadata);
    }

    #[test]
    fn test_capacity_limit() {
        let config = AssociativeMemoryConfig::new(5, 0.5, 3).unwrap();
        let mut memory = AssociativeItemMemory::new(config);

        let mut gen = BipolarVectorGenerator::new(123);

        for _ in 0..5 {
            let vector = gen.generate().unwrap();
            memory.store(vector, vec![]).unwrap();
        }

        // Next store should fail
        let vector = gen.generate().unwrap();
        let result = memory.store(vector, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_k_nearest_neighbors() {
        let config = AssociativeMemoryConfig::new(10, 0.3, 3).unwrap();
        let mut memory = AssociativeItemMemory::new(config);

        let mut gen = BipolarVectorGenerator::new(456);
        
        // Store several vectors
        let base = gen.generate().unwrap();
        memory.store(base.clone(), vec![1]).unwrap();
        
        for _ in 0..4 {
            let v = gen.generate().unwrap();
            memory.store(v, vec![2]).unwrap();
        }

        let neighbors = memory.retrieve_k_nearest(&base, 2000);
        
        // Should return at least the exact match
        assert!(!neighbors.is_empty());
        assert!(neighbors[0].1 > 0.9); // First should be very similar
    }

    #[test]
    fn test_similarity_threshold() {
        let config = AssociativeMemoryConfig::new(10, 0.9, 1).unwrap();
        let mut memory = AssociativeItemMemory::new(config);

        let mut gen = BipolarVectorGenerator::new(789);
        let v1 = gen.generate().unwrap();
        let v2 = gen.generate().unwrap(); // Different random vector

        memory.store(v1.clone(), vec![1]).unwrap();

        // v2 should not match v1 with high threshold
        let result = memory.retrieve(&v2, 3000);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_stats() {
        let config = AssociativeMemoryConfig::new(100, 0.5, 5).unwrap();
        let mut memory = AssociativeItemMemory::new(config);

        let mut gen = BipolarVectorGenerator::new(111);
        
        for _ in 0..10 {
            let v = gen.generate().unwrap();
            memory.store(v, vec![]).unwrap();
        }

        // Access some items
        let v = gen.generate().unwrap();
        let _ = memory.retrieve(&v, 4000);

        let stats = memory.stats();
        assert_eq!(stats.total_items, 10);
        assert_eq!(stats.max_items, 100);
        assert!(stats.capacity_utilization > 0.0);
    }

    #[test]
    fn test_continuous_memory() {
        let mut cmem = ContinuousItemMemory::new();
        let mut gen = BipolarVectorGenerator::new(222);

        let v1 = gen.generate().unwrap();
        let key1 = cmem.add(42.0, v1.clone());

        let (value, sim) = cmem.query(&v1).unwrap();
        assert_eq!(value, 42.0);
        assert!(sim > 0.95);

        // Add similar vector - should merge
        let v1_perturbed = v1.clone(); // In practice would be slightly different
        let key2 = cmem.add(58.0, v1_perturbed);

        // Same key since merged
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_remove_item() {
        let config = AssociativeMemoryConfig::new(10, 0.5, 1).unwrap();
        let mut memory = AssociativeItemMemory::new(config);

        let mut gen = BipolarVectorGenerator::new(333);
        let v = gen.generate().unwrap();
        let id = memory.store(v.clone(), vec![1, 2]).unwrap();

        assert!(memory.retrieve(&v, 5000).is_ok());

        let removed = memory.remove(id);
        assert!(removed.is_some());

        assert!(memory.retrieve(&v, 6000).is_err());
    }
}
