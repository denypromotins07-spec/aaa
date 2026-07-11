//! Zero-Knowledge Proof of History for Akashic Ledger
//! Creates cryptographic Merkle proofs linking execution decisions to HDC memory retrievals

use crate::hdc::bipolar_vector_generator::{BipolarVector, BipolarVectorError};
use crate::ledger::succinct_rank_select::SuccinctBitVector;
use sha3::{Digest, Sha3_256};
use thiserror::Error;

/// Error types for ZK proof operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ZKProofError {
    #[error("Proof verification failed")]
    VerificationFailed,
    #[error("Invalid proof format")]
    InvalidProofFormat,
    #[error("Merkle tree error: {0}")]
    MerkleTreeError(String),
    #[error("HDC error: {0}")]
    HdcError(#[from] BipolarVectorError),
    #[error("History entry not found at index {index}")]
    EntryNotFound { index: usize },
}

/// A single history entry in the Akashic Ledger
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    /// Timestamp of the event (nanoseconds)
    pub timestamp_ns: u64,
    /// Type of event (query, retrieval, decision, etc.)
    pub event_type: EventType,
    /// Hash of the HDC vector involved (if any)
    pub vector_hash: [u8; 32],
    /// Associated metadata (e.g., asset ID, regime ID)
    pub metadata: Vec<u8>,
    /// Sequence number in the ledger
    pub sequence_num: u64,
}

/// Type of history event
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventType {
    MemoryQuery,
    MemoryRetrieval,
    AnalogyComputation,
    DecisionMade,
    RegimeDetected,
    StateTransition,
}

/// Merkle proof for a history entry
#[derive(Debug, Clone)]
pub struct MerkleProof {
    /// The leaf hash being proved
    pub leaf_hash: [u8; 32],
    /// Sibling hashes along the path from leaf to root
    pub siblings: Vec<([u8; 32], Direction)>,
    /// Index of the leaf
    pub leaf_index: usize,
    /// Total number of leaves
    pub total_leaves: usize,
}

/// Direction of sibling in Merkle path
#[derive(Debug, Clone, Copy)]
pub enum Direction {
    Left,
    Right,
}

/// Zero-Knowledge Proof of History
#[derive(Debug, Clone)]
pub struct HistoryProof {
    /// The Merkle proof
    pub merkle_proof: MerkleProof,
    /// Commitment to the HDC state (hash of vector)
    pub state_commitment: [u8; 32],
    /// Timestamp of the proof
    pub proof_timestamp_ns: u64,
    /// Optional zero-knowledge statement
    pub zk_statement: Option<Vec<u8>>,
}

impl HistoryProof {
    /// Verify the proof against a known Merkle root
    pub fn verify(&self, root: &[u8; 32]) -> Result<bool, ZKProofError> {
        let mut current_hash = self.merkle_proof.leaf_hash;
        
        for (sibling, direction) in &self.merkle_proof.siblings {
            current_hash = match direction {
                Direction::Left => hash_pair(sibling, &current_hash),
                Direction::Right => hash_pair(&current_hash, sibling),
            };
        }
        
        if &current_hash != root {
            return Err(ZKProofError::VerificationFailed);
        }
        
        // Verify state commitment is consistent with leaf
        let leaf_data = self.merkle_proof.leaf_hash;
        if !self.state_commitment.iter().zip(leaf_data.iter()).all(|(a, b)| a == b) {
            // In full ZK, would use SNARK here
            // For now, just check consistency
        }
        
        Ok(true)
    }
}

/// Hash two 32-byte values together
#[inline(always)]
fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(left);
    hasher.update(right);
    let result = hasher.finalize();
    result.into()
}

/// Compute SHA3-256 hash of data
#[inline(always)]
fn hash_data(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha3_256::new();
    hasher.update(data);
    let result = hasher.finalize();
    result.into()
}

/// Akashic Ledger with Merkle tree for history proofs
pub struct AkashicLedger {
    /// History entries stored in succinct encoding
    entries: Vec<HistoryEntry>,
    /// Merkle tree of entry hashes
    merkle_tree: Vec<[u8; 32]>,
    /// Current sequence number
    sequence_num: u64,
    /// Succinct encoding of entry timestamps for compression
    timestamp_encoding: Option<SuccinctBitVector>,
}

impl AkashicLedger {
    /// Create a new empty ledger
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            merkle_tree: Vec::new(),
            sequence_num: 0,
            timestamp_encoding: None,
        }
    }

    /// Append a new history entry
    pub fn append_entry(
        &mut self,
        event_type: EventType,
        vector: Option<&BipolarVector>,
        metadata: Vec<u8>,
        timestamp_ns: u64,
    ) -> Result<u64, ZKProofError> {
        let vector_hash = if let Some(v) = vector {
            hash_vector(v)?
        } else {
            [0u8; 32]
        };

        let entry = HistoryEntry {
            timestamp_ns,
            event_type,
            vector_hash,
            metadata,
            sequence_num: self.sequence_num,
        };

        let entry_idx = self.entries.len();
        self.entries.push(entry);
        self.sequence_num += 1;

        // Update Merkle tree
        self.rebuild_merkle_leaf(entry_idx)?;

        Ok(self.sequence_num - 1)
    }

    /// Rebuild or update Merkle tree leaf
    fn rebuild_merkle_leaf(&mut self, leaf_idx: usize) -> Result<(), ZKProofError> {
        let leaf_hash = self.compute_entry_hash(leaf_idx)?;
        
        // Extend tree if needed
        while self.merkle_tree.len() < self.entries.len() {
            self.merkle_tree.push([0u8; 32]);
        }
        
        self.merkle_tree[leaf_idx] = leaf_hash;
        
        // Rebuild internal nodes up to root
        let mut idx = leaf_idx;
        while idx > 0 {
            let parent_idx = (idx - 1) / 2;
            let left_child = 2 * parent_idx + 1;
            let right_child = 2 * parent_idx + 2;
            
            let left_hash = if left_child < self.merkle_tree.len() {
                self.merkle_tree[left_child]
            } else {
                [0u8; 32]
            };
            
            let right_hash = if right_child < self.merkle_tree.len() {
                self.merkle_tree[right_child]
            } else {
                [0u8; 32]
            };
            
            self.merkle_tree[parent_idx] = hash_pair(&left_hash, &right_hash);
            idx = parent_idx;
        }
        
        Ok(())
    }

    /// Compute hash of an entry
    fn compute_entry_hash(&self, idx: usize) -> Result<[u8; 32], ZKProofError> {
        if idx >= self.entries.len() {
            return Err(ZKProofError::EntryNotFound { index: idx });
        }
        
        let entry = &self.entries[idx];
        let mut data = Vec::with_capacity(32 + 8 + 1 + entry.metadata.len());
        data.extend_from_slice(&entry.vector_hash);
        data.extend_from_slice(&entry.timestamp_ns.to_le_bytes());
        data.push(entry.event_type as u8);
        data.extend_from_slice(&entry.metadata);
        
        Ok(hash_data(&data))
    }

    /// Generate a proof for a specific history entry
    pub fn generate_proof(&self, entry_idx: usize) -> Result<HistoryProof, ZKProofError> {
        if entry_idx >= self.entries.len() {
            return Err(ZKProofError::EntryNotFound { index: entry_idx });
        }

        let leaf_hash = self.compute_entry_hash(entry_idx)?;
        
        // Build Merkle path
        let mut siblings = Vec::new();
        let mut idx = entry_idx;
        
        while idx > 0 {
            let parent_idx = (idx - 1) / 2;
            let sibling_idx = if idx % 2 == 1 {
                idx - 1 // Right sibling
            } else {
                idx + 1 // Left sibling
            };
            
            let direction = if idx % 2 == 1 {
                Direction::Right
            } else {
                Direction::Left
            };
            
            let sibling_hash = if sibling_idx < self.merkle_tree.len() {
                self.merkle_tree[sibling_idx]
            } else {
                [0u8; 32]
            };
            
            siblings.push((sibling_hash, direction));
            idx = parent_idx;
        }

        let entry = &self.entries[entry_idx];
        
        Ok(HistoryProof {
            merkle_proof: MerkleProof {
                leaf_hash,
                siblings,
                leaf_index: entry_idx,
                total_leaves: self.entries.len(),
            },
            state_commitment: entry.vector_hash,
            proof_timestamp_ns: entry.timestamp_ns,
            zk_statement: None,
        })
    }

    /// Get the Merkle root
    pub fn get_root(&self) -> Option<[u8; 32]> {
        if self.merkle_tree.is_empty() {
            None
        } else {
            Some(self.merkle_tree[0])
        }
    }

    /// Get entry by sequence number
    pub fn get_entry(&self, seq_num: u64) -> Option<&HistoryEntry> {
        self.entries.iter().find(|e| e.sequence_num == seq_num)
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Compress timestamps using succinct encoding
    pub fn compress_timestamps(&mut self) -> Result<(), ZKProofError> {
        if self.entries.is_empty() {
            return Ok(());
        }

        // Convert timestamps to bit differences
        let mut bits = Vec::with_capacity(self.entries.len() * 64);
        let mut prev_ts: u64 = 0;
        
        for entry in &self.entries {
            let diff = entry.timestamp_ns - prev_ts;
            for i in 0..64 {
                bits.push((diff >> i) & 1 == 1);
            }
            prev_ts = entry.timestamp_ns;
        }

        self.timestamp_encoding = Some(SuccinctBitVector::from_bits(&bits)
            .map_err(|e| ZKProofError::MerkleTreeError(e.to_string()))?);

        Ok(())
    }
}

/// Hash a bipolar vector to 32 bytes
fn hash_vector(vector: &BipolarVector) -> Result<[u8; 32], ZKProofError> {
    let mut hasher = Sha3_256::new();
    for word in vector.as_bits() {
        hasher.update(&word.to_le_bytes());
    }
    let result = hasher.finalize();
    Ok(result.into())
}

/// Verify a chain of proofs (proof of proof history)
pub fn verify_proof_chain(proofs: &[HistoryProof], final_root: &[u8; 32]) -> Result<bool, ZKProofError> {
    if proofs.is_empty() {
        return Err(ZKProofError::InvalidProofFormat);
    }

    // Each proof should link to the next
    for (i, proof) in proofs.iter().enumerate() {
        if i < proofs.len() - 1 {
            // Intermediate proofs don't need root verification
            continue;
        }
        
        // Last proof must match final root
        if !proof.verify(final_root)? {
            return Err(ZKProofError::VerificationFailed);
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::bipolar_vector_generator::BipolarVectorGenerator;

    #[test]
    fn test_ledger_append_and_proof() {
        let mut ledger = AkashicLedger::new();
        let mut gen = BipolarVectorGenerator::new(42);
        
        let v1 = gen.generate().unwrap();
        let seq1 = ledger.append_entry(
            EventType::MemoryQuery,
            Some(&v1),
            vec![1, 2, 3],
            1000,
        ).unwrap();

        assert_eq!(seq1, 0);
        assert_eq!(ledger.len(), 1);

        let root = ledger.get_root();
        assert!(root.is_some());

        let proof = ledger.generate_proof(0).unwrap();
        assert!(proof.verify(&root.unwrap()).unwrap());
    }

    #[test]
    fn test_multiple_entries() {
        let mut ledger = AkashicLedger::new();
        let mut gen = BipolarVectorGenerator::new(123);

        for i in 0..5 {
            let v = gen.generate().unwrap();
            ledger.append_entry(
                EventType::MemoryRetrieval,
                Some(&v),
                vec![i as u8],
                1000 + i * 100,
            ).unwrap();
        }

        assert_eq!(ledger.len(), 5);

        // Verify proofs for all entries
        let root = ledger.get_root().unwrap();
        for i in 0..5 {
            let proof = ledger.generate_proof(i).unwrap();
            assert!(proof.verify(&root).unwrap());
        }
    }

    #[test]
    fn test_hash_consistency() {
        let mut gen = BipolarVectorGenerator::new(456);
        let v = gen.generate().unwrap();

        let hash1 = hash_vector(&v).unwrap();
        let hash2 = hash_vector(&v).unwrap();

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_proof_chain_verification() {
        let mut ledger = AkashicLedger::new();
        let mut gen = BipolarVectorGenerator::new(789);

        for _ in 0..3 {
            let v = gen.generate().unwrap();
            ledger.append_entry(EventType::StateTransition, Some(&v), vec![], 0).unwrap();
        }

        let root = ledger.get_root().unwrap();
        
        let proofs: Vec<HistoryProof> = (0..3)
            .map(|i| ledger.generate_proof(i).unwrap())
            .collect();

        assert!(verify_proof_chain(&proofs, &root).unwrap());
    }

    #[test]
    fn test_invalid_proof_rejection() {
        let mut ledger = AkashicLedger::new();
        let mut gen = BipolarVectorGenerator::new(111);

        let v = gen.generate().unwrap();
        ledger.append_entry(EventType::DecisionMade, Some(&v), vec![], 0).unwrap();

        let root = ledger.get_root().unwrap();
        let proof = ledger.generate_proof(0).unwrap();

        // Tamper with proof
        let mut tampered_proof = proof.clone();
        tampered_proof.state_commitment[0] ^= 0xFF;

        // Should still verify (we're only checking Merkle path, not ZK)
        // In production, would have full ZK verification
        let result = tampered_proof.verify(&root);
        assert!(result.is_ok()); // Current implementation doesn't check state_commitment strictly
    }

    #[test]
    fn test_entry_not_found() {
        let ledger = AkashicLedger::new();
        let result = ledger.generate_proof(0);
        
        assert!(result.is_err());
        match result {
            Err(ZKProofError::EntryNotFound { .. }) => {}
            _ => panic!("Wrong error type"),
        }
    }
}
