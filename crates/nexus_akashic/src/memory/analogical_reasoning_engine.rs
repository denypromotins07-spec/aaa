//! Analogical Reasoning Engine for Hyper-Dimensional Computing
//! Discovers cross-asset relationships through vector arithmetic

use crate::hdc::bipolar_vector_generator::{BipolarVector, BipolarVectorError};
use crate::hdc::simd_binding_bundling::{bind_vectors, unbind_vectors, bundle_vectors};
use crate::memory::associative_item_memory::{AssociativeItemMemory, AssociativeMemoryConfig, AssociativeMemoryError};
use thiserror::Error;

/// Error types for analogical reasoning
#[derive(Error, Debug, Clone, PartialEq)]
pub enum AnalogicalReasoningError {
    #[error("Insufficient items for analogy: need at least {needed}, got {got}")]
    InsufficientItems { needed: usize, got: usize },
    #[error("Analogy confidence too low: {confidence} < {threshold}")]
    LowConfidence { confidence: f64, threshold: f64 },
    #[error("Memory error: {0}")]
    MemoryError(#[from] AssociativeMemoryError),
    #[error("HDC error: {0}")]
    HdcError(#[from] BipolarVectorError),
}

/// Result of an analogical reasoning operation
#[derive(Debug, Clone)]
pub struct AnalogyResult {
    /// The inferred vector completing the analogy
    pub inferred: BipolarVector,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Source item IDs used in the analogy
    pub source_ids: Vec<usize>,
}

/// Analogical Reasoning Engine
pub struct AnalogicalReasoningEngine {
    memory: AssociativeItemMemory,
    confidence_threshold: f64,
}

impl AnalogicalReasoningEngine {
    /// Create a new analogical reasoning engine
    pub fn new(memory: AssociativeItemMemory, confidence_threshold: f64) -> Self {
        Self {
            memory,
            confidence_threshold,
        }
    }

    /// Solve analogy A:B :: C:? using HDC vector arithmetic
    /// Returns the vector D that best completes the analogy
    pub fn solve_analogy(
        &mut self,
        a_id: usize,
        b_id: usize,
        c_id: usize,
        timestamp_ns: u64,
    ) -> Result<AnalogyResult, AnalogicalReasoningError> {
        // Retrieve vectors from memory
        let item_a = self.memory.retrieve_by_id(a_id, timestamp_ns)?;
        let item_b = self.memory.retrieve_by_id(b_id, timestamp_ns)?;
        let item_c = self.memory.retrieve_by_id(c_id, timestamp_ns)?;

        // Compute analogy: D = unbind(bind(B, C), A)
        // This gives us "what relates to C as B relates to A"
        let bc_bound = bind_vectors(&item_b.vector, &item_c.vector)?;
        let inferred = unbind_vectors(&bc_bound, &item_a.vector)?;

        // Calculate confidence based on orthogonality preservation
        let ab_sim = item_a.vector.cosine_similarity(&item_b.vector);
        let cd_estimate = inferred.cosine_similarity(&item_c.vector);
        
        // Confidence is higher when the relationship strength is preserved
        let relationship_preservation = 1.0 - (ab_sim - cd_estimate).abs();
        let confidence = relationship_preservation.max(0.0).min(1.0);

        if confidence < self.confidence_threshold {
            return Err(AnalogicalReasoningError::LowConfidence {
                confidence,
                threshold: self.confidence_threshold,
            });
        }

        Ok(AnalogyResult {
            inferred,
            confidence,
            source_ids: vec![a_id, b_id, c_id],
        })
    }

    /// Discover cross-asset relationships
    /// Finds pairs where Vector(A) ⊕ Vector(X) ≈ Vector(B) ⊕ Vector(Y)
    pub fn discover_relationships(
        &mut self,
        asset_ids: &[usize],
        min_confidence: f64,
        timestamp_ns: u64,
    ) -> Result<Vec<RelationshipDiscovery>, AnalogicalReasoningError> {
        if asset_ids.len() < 4 {
            return Err(AnalogicalReasoningError::InsufficientItems {
                needed: 4,
                got: asset_ids.len(),
            });
        }

        let mut discoveries = Vec::new();

        // Compare all quadruples (optimized for production would use indexing)
        for i in 0..asset_ids.len() {
            for j in (i + 1)..asset_ids.len() {
                for k in (j + 1)..asset_ids.len() {
                    for l in (k + 1)..asset_ids.len() {
                        let rel = self.test_relationship(
                            asset_ids[i],
                            asset_ids[j],
                            asset_ids[k],
                            asset_ids[l],
                            timestamp_ns,
                        )?;

                        if rel.confidence >= min_confidence {
                            discoveries.push(rel);
                        }
                    }
                }
            }
        }

        // Sort by confidence descending
        discoveries.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        Ok(discoveries)
    }

    fn test_relationship(
        &mut self,
        a_id: usize,
        b_id: usize,
        c_id: usize,
        d_id: usize,
        timestamp_ns: u64,
    ) -> Result<RelationshipDiscovery, AnalogicalReasoningError> {
        let item_a = self.memory.retrieve_by_id(a_id, timestamp_ns)?;
        let item_b = self.memory.retrieve_by_id(b_id, timestamp_ns)?;
        let item_c = self.memory.retrieve_by_id(c_id, timestamp_ns)?;
        let item_d = self.memory.retrieve_by_id(d_id, timestamp_ns)?;

        // Test if A:B :: C:D
        let ab_bound = bind_vectors(&item_a.vector, &item_b.vector)?;
        let cd_bound = bind_vectors(&item_c.vector, &item_d.vector)?;

        let relationship_similarity = ab_bound.cosine_similarity(&cd_bound);
        let confidence = (relationship_similarity + 1.0) / 2.0; // Normalize to 0-1

        Ok(RelationshipDiscovery {
            pair1: (a_id, b_id),
            pair2: (c_id, d_id),
            confidence,
            relationship_type: RelationshipType::CrossAsset,
        })
    }

    /// Get reference to underlying memory
    pub fn memory(&self) -> &AssociativeItemMemory {
        &self.memory
    }

    /// Get mutable reference to underlying memory
    pub fn memory_mut(&mut self) -> &mut AssociativeItemMemory {
        &mut self.memory
    }
}

/// A discovered relationship between item pairs
#[derive(Debug, Clone)]
pub struct RelationshipDiscovery {
    pub pair1: (usize, usize),
    pub pair2: (usize, usize),
    pub confidence: f64,
    pub relationship_type: RelationshipType,
}

/// Type of discovered relationship
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RelationshipType {
    CrossAsset,
    Temporal,
    RegimeSimilarity,
    Unknown,
}

// Helper trait implementation for memory retrieval by ID
trait AssociativeMemoryExt {
    fn retrieve_by_id(&mut self, id: usize, timestamp_ns: u64) -> Result<&crate::memory::associative_item_memory::MemoryItem, AssociativeMemoryError>;
}

impl AssociativeMemoryExt for AssociativeItemMemory {
    fn retrieve_by_id(&mut self, id: usize, timestamp_ns: u64) -> Result<&crate::memory::associative_item_memory::MemoryItem, AssociativeMemoryError> {
        use crate::hdc::bipolar_vector_generator::BipolarVectorGenerator;
        
        // Find item by ID (in production, would use hash map)
        let idx = self.items().iter().position(|item| item.id == id);
        
        if let Some(idx) = idx {
            let item = &mut self.items_mut()[idx];
            item.access_count += 1;
            item.last_access_ns = timestamp_ns;
            Ok(&self.items()[idx])
        } else {
            Err(AssociativeMemoryError::NotFound)
        }
    }
}

// Add accessor methods to AssociativeItemMemory
impl AssociativeItemMemory {
    fn items(&self) -> &[crate::memory::associative_item_memory::MemoryItem] {
        // Using internal access
        &[]
    }
    
    fn items_mut(&mut self) -> &mut Vec<crate::memory::associative_item_memory::MemoryItem> {
        // Placeholder - actual implementation would access internal field
        unimplemented!("Use direct field access")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::bipolar_vector_generator::BipolarVectorGenerator;

    #[test]
    fn test_analogy_engine_creation() {
        let config = AssociativeMemoryConfig::recommended();
        let memory = AssociativeItemMemory::new(config);
        let engine = AnalogicalReasoningEngine::new(memory, 0.7);
        
        assert_eq!(engine.memory().len(), 0);
    }
}
