//! Locality-Sensitive Hashing (LSH) Similarity Estimator
//! Enables O(1) approximate nearest neighbor search in hyper-dimensional space

use crate::hashing::minwise_permutations::{MinHashFamily, MinHashSignature, MinHashError};
use thiserror::Error;

/// Error types for LSH operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum LSHError {
    #[error("Invalid number of hash tables: {requested}. Must be between {min} and {max}")]
    InvalidHashTableCount { requested: usize, min: usize, max: usize },
    #[error("Invalid band size: {requested}. Must divide signature size evenly")]
    InvalidBandSize { requested: usize, signature_size: usize },
    #[error("No candidates found for query")]
    NoCandidates,
    #[error("Index not built")]
    IndexNotBuilt,
}

/// Minimum number of hash tables for reliable LSH
const MIN_HASH_TABLES: usize = 4;

/// Maximum number of hash tables
const MAX_HASH_TABLES: usize = 64;

/// Configuration for LSH index
#[derive(Debug, Clone)]
pub struct LSHConfig {
    /// Number of hash tables (more tables = better recall, higher memory)
    pub num_tables: usize,
    /// Band size for each table (smaller = more selective)
    pub band_size: usize,
    /// Signature size from MinHash
    pub signature_size: usize,
    /// Similarity threshold for considering items as neighbors
    pub similarity_threshold: f64,
}

impl LSHConfig {
    /// Create a new LSH configuration with validation
    pub fn new(
        num_tables: usize,
        band_size: usize,
        signature_size: usize,
        similarity_threshold: f64,
    ) -> Result<Self, LSHError> {
        if num_tables < MIN_HASH_TABLES || num_tables > MAX_HASH_TABLES {
            return Err(LSHError::InvalidHashTableCount {
                requested: num_tables,
                min: MIN_HASH_TABLES,
                max: MAX_HASH_TABLES,
            });
        }

        if signature_size % band_size != 0 {
            return Err(LSHError::InvalidBandSize {
                requested: band_size,
                signature_size,
            });
        }

        Ok(Self {
            num_tables,
            band_size,
            signature_size,
            similarity_threshold,
        })
    }

    /// Get recommended parameters based on desired similarity threshold
    pub fn recommended(signature_size: usize, threshold: f64) -> Self {
        // Optimal band size based on threshold: r ≈ ln(2) / ln(1/threshold)
        let optimal_bands = if threshold > 0.9 {
            4
        } else if threshold > 0.7 {
            8
        } else if threshold > 0.5 {
            16
        } else {
            32
        };

        let num_tables = (signature_size / optimal_bands).min(MAX_HASH_TABLES);
        
        Self {
            num_tables,
            band_size: optimal_bands,
            signature_size,
            similarity_threshold: threshold,
        }
    }
}

/// A single LSH band hash
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LSHBandHash(u64);

impl LSHBandHash {
    pub fn from_signature_band(signature: &MinHashSignature, start: usize, end: usize) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        for i in start..end {
            hasher.write_u64(signature.as_slice()[i].value());
        }
        
        Self(hasher.finish())
    }

    #[inline(always)]
    pub fn value(&self) -> u64 {
        self.0
    }
}

/// LSH Index for fast similarity search
pub struct LSHIndex {
    config: LSHConfig,
    /// For each table, a map from band hash to item IDs
    tables: Vec<std::collections::HashMap<LSHBandHash, Vec<usize>>>,
    /// Stored signatures for lookup
    signatures: Vec<MinHashSignature>,
    /// Whether the index has been built
    is_built: bool,
}

impl LSHIndex {
    /// Create a new empty LSH index
    pub fn new(config: LSHConfig) -> Self {
        let mut tables = Vec::with_capacity(config.num_tables);
        for _ in 0..config.num_tables {
            tables.push(std::collections::HashMap::new());
        }

        Self {
            config,
            tables,
            signatures: Vec::new(),
            is_built: false,
        }
    }

    /// Add an item to the index
    pub fn add_item(&mut self, signature: MinHashSignature) -> Result<usize, LSHError> {
        let item_id = self.signatures.len();
        self.signatures.push(signature);

        // Compute band hashes for each table
        for table_idx in 0..self.config.num_tables {
            let band_start = (table_idx * self.config.band_size) % self.config.signature_size;
            let band_end = (band_start + self.config.band_size).min(self.config.signature_size);
            
            let band_hash = LSHBandHash::from_signature_band(
                &self.signatures[item_id],
                band_start,
                band_end,
            );

            self.tables[table_idx]
                .entry(band_hash)
                .or_insert_with(Vec::new)
                .push(item_id);
        }

        Ok(item_id)
    }

    /// Build the index (finalize internal structures)
    pub fn build(&mut self) -> Result<(), LSHError> {
        if self.signatures.is_empty() {
            return Ok(()); // Empty index is valid
        }
        self.is_built = true;
        Ok(())
    }

    /// Query for similar items
    pub fn query(&self, signature: &MinHashSignature, max_candidates: usize) -> Result<Vec<usize>, LSHError> {
        if !self.is_built {
            return Err(LSHError::IndexNotBuilt);
        }

        let mut candidate_counts = std::collections::HashMap::new();

        // Count how many bands match for each candidate
        for table_idx in 0..self.config.num_tables {
            let band_start = (table_idx * self.config.band_size) % self.config.signature_size;
            let band_end = (band_start + self.config.band_size).min(self.config.signature_size);
            
            let query_band_hash = LSHBandHash::from_signature_band(
                signature,
                band_start,
                band_end,
            );

            if let Some(candidates) = self.tables[table_idx].get(&query_band_hash) {
                for &candidate_id in candidates {
                    *candidate_counts.entry(candidate_id).or_insert(0u32) += 1;
                }
            }
        }

        // Sort candidates by number of matching bands (proxy for similarity)
        let mut candidates: Vec<(usize, u32)> = candidate_counts.into_iter().collect();
        candidates.sort_by(|a, b| b.1.cmp(&a.1));

        // Return top candidates
        let result: Vec<usize> = candidates
            .into_iter()
            .take(max_candidates)
            .map(|(id, _)| id)
            .collect();

        if result.is_empty() {
            Err(LSHError::NoCandidates)
        } else {
            Ok(result)
        }
    }

    /// Get the signature for an item
    pub fn get_signature(&self, item_id: usize) -> Option<&MinHashSignature> {
        self.signatures.get(item_id)
    }

    /// Get the number of items in the index
    pub fn len(&self) -> usize {
        self.signatures.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.signatures.is_empty()
    }

    /// Estimate similarity between query and candidate based on band matches
    pub fn estimate_similarity_from_bands(
        &self,
        query: &MinHashSignature,
        candidate_id: usize,
    ) -> Option<f64> {
        let candidate = self.get_signature(candidate_id)?;
        
        // Count matching bands across all tables
        let mut matching_bands = 0u32;
        
        for table_idx in 0..self.config.num_tables {
            let band_start = (table_idx * self.config.band_size) % self.config.signature_size;
            let band_end = (band_start + self.config.band_size).min(self.config.signature_size);
            
            let query_hash = LSHBandHash::from_signature_band(query, band_start, band_end);
            let candidate_hash = LSHBandHash::from_signature_band(candidate, band_start, band_end);
            
            if query_hash == candidate_hash {
                matching_bands += 1;
            }
        }

        // Convert band match ratio to estimated Jaccard similarity
        // P(match) = 1 - (1 - s^b)^r where s is similarity, b is band size, r is num tables
        // Simplified: estimate ≈ matching_bands / num_tables (rough approximation)
        Some(matching_bands as f64 / self.config.num_tables as f64)
    }
}

/// Batch LSH query processor for multiple simultaneous queries
pub struct LSHBatchQuery<'a> {
    index: &'a LSHIndex,
    queries: Vec<MinHashSignature>,
    results: Vec<Result<Vec<usize>, LSHError>>,
}

impl<'a> LSHBatchQuery<'a> {
    pub fn new(index: &'a LSHIndex) -> Self {
        Self {
            index,
            queries: Vec::new(),
            results: Vec::new(),
        }
    }

    pub fn add_query(&mut self, signature: MinHashSignature) {
        self.queries.push(signature);
    }

    pub fn execute(mut self, max_candidates: usize) -> Result<Vec<Vec<usize>>, LSHError> {
        self.results.reserve(self.queries.len());
        
        for query in &self.queries {
            match self.index.query(query, max_candidates) {
                Ok(candidates) => self.results.push(Ok(candidates)),
                Err(e) => self.results.push(Err(e)),
            }
        }

        let errors: Vec<_> = self.results.iter().filter_map(|r| r.as_ref().err()).collect();
        if errors.len() == self.results.len() {
            // All queries failed
            Err(LSHError::NoCandidates)
        } else {
            Ok(self.results
                .into_iter()
                .filter_map(|r| r.ok())
                .collect())
        }
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
    fn test_lsh_config_validation() {
        let config = LSHConfig::new(8, 16, 128, 0.7);
        assert!(config.is_ok());

        // Invalid: too few tables
        let result = LSHConfig::new(2, 16, 128, 0.7);
        assert!(result.is_err());

        // Invalid: band size doesn't divide signature size
        let result = LSHConfig::new(8, 15, 128, 0.7);
        assert!(result.is_err());
    }

    #[test]
    fn test_lsh_index_basic() {
        let family = MinHashFamily::new(64, 42).unwrap();
        let config = LSHConfig::recommended(64, 0.7);
        let mut index = LSHIndex::new(config);

        let sig1 = create_test_signature(&family, &["a", "b", "c"]);
        let sig2 = create_test_signature(&family, &["a", "b", "d"]);
        let sig3 = create_test_signature(&family, &["x", "y", "z"]);

        index.add_item(sig1).unwrap();
        index.add_item(sig2).unwrap();
        index.add_item(sig3).unwrap();
        index.build().unwrap();

        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_lsh_query_similar() {
        let family = MinHashFamily::new(64, 123).unwrap();
        let config = LSHConfig::recommended(64, 0.5);
        let mut index = LSHIndex::new(config);

        // Add items with some overlap
        let sig1 = create_test_signature(&family, &["cat", "dog", "bird", "fish"]);
        let sig2 = create_test_signature(&family, &["cat", "dog", "mouse", "rat"]);
        let sig3 = create_test_signature(&family, &["apple", "orange", "banana"]);

        index.add_item(sig1).unwrap();
        index.add_item(sig2).unwrap();
        index.add_item(sig3).unwrap();
        index.build().unwrap();

        // Query with sig1's signature
        let query = create_test_signature(&family, &["cat", "dog", "bird", "fish"]);
        let candidates = index.query(&query, 5).unwrap();

        // Should find at least one candidate
        assert!(!candidates.is_empty());
    }

    #[test]
    fn test_lsh_empty_index() {
        let config = LSHConfig::recommended(64, 0.7);
        let mut index = LSHIndex::new(config);
        index.build().unwrap();

        let family = MinHashFamily::new(64, 456).unwrap();
        let query = create_test_signature(&family, &["test"]);
        
        let result = index.query(&query, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_query() {
        let family = MinHashFamily::new(64, 789).unwrap();
        let config = LSHConfig::recommended(64, 0.5);
        let mut index = LSHIndex::new(config);

        for i in 0..10 {
            let sig = create_test_signature(&family, &[&format!("item_{}", i)]);
            index.add_item(sig).unwrap();
        }
        index.build().unwrap();

        let mut batch = LSHBatchQuery::new(&index);
        batch.add_query(create_test_signature(&family, &["query1"]));
        batch.add_query(create_test_signature(&family, &["query2"]));

        let results = batch.execute(3);
        assert!(results.is_ok());
    }
}
