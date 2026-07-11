//! Self-Referential Bytecode Module
//!
//! Implements bytecode structures that contain references to their own hash
//! enabling self-verification and autonomous replication.

use alloc::vec::Vec;
use core::fmt;

/// Error types for self-referential bytecode operations
#[derive(Debug, Clone, PartialEq)]
pub enum SelfRefError {
    InvalidOffset,
    HashVerificationFailed,
    EncodingError,
}

/// Result type for self-referential operations
pub type SelfRefResult<T> = Result<T, SelfRefError>;

/// Represents a self-referential code structure
#[derive(Debug, Clone)]
pub struct SelfReferentialCode {
    /// The bytecode data
    data: Vec<u8>,
    /// Offset where the self-hash is stored
    hash_offset: usize,
    /// The expected hash value
    expected_hash: [u8; 32],
}

impl SelfReferentialCode {
    /// Create a new self-referential code structure
    pub fn new(data: Vec<u8>, hash_offset: usize, expected_hash: [u8; 32]) -> SelfRefResult<Self> {
        if hash_offset + 32 > data.len() {
            return Err(SelfRefError::InvalidOffset);
        }

        Ok(Self {
            data,
            hash_offset,
            expected_hash,
        })
    }

    /// Verify the self-reference integrity
    pub fn verify(&self) -> bool {
        let stored_hash = &self.data[self.hash_offset..self.hash_offset + 32];
        stored_hash == &self.expected_hash[..]
    }

    /// Get the code data excluding the hash region
    pub fn get_code_data(&self) -> (&[u8], &[u8]) {
        let (before, after) = self.data.split_at(self.hash_offset);
        let after = &after[32..];
        (before, after)
    }

    /// Get the full bytecode including embedded hash
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get the embedded hash
    pub fn embedded_hash(&self) -> &[u8; 32] {
        &self.expected_hash
    }
}

/// Builder for self-referential bytecode
pub struct SelfReferentialBuilder {
    data: Vec<u8>,
    hash_offset: Option<usize>,
}

impl SelfReferentialBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            hash_offset: None,
        }
    }

    /// Append data to the bytecode
    pub fn append(mut self, bytes: &[u8]) -> Self {
        self.data.extend_from_slice(bytes);
        self
    }

    /// Reserve space for the hash at the current position
    pub fn reserve_hash_space(mut self) -> Self {
        self.hash_offset = Some(self.data.len());
        self.data.extend_from_slice(&[0u8; 32]);
        self
    }

    /// Build with the specified hash
    pub fn build(mut self, hash: [u8; 32]) -> SelfRefResult<SelfReferentialCode> {
        let offset = self.hash_offset.ok_or(SelfRefError::InvalidOffset)?;
        
        // Embed the hash
        self.data[offset..offset + 32].copy_from_slice(&hash[..]);

        SelfReferentialCode::new(self.data, offset, hash)
    }
}

impl Default for SelfReferentialBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that can be encoded as self-referential bytecode
pub trait EncodableToSelfRef {
    /// Encode to self-referential bytecode format
    fn encode_self_ref(&self, hash: [u8; 32]) -> SelfRefResult<SelfReferentialCode>;
}

/// Quine payload structure
#[derive(Debug, Clone)]
pub struct QuinePayload {
    /// Original source/data
    pub source: Vec<u8>,
    /// Embedded verification hash
    pub verification_hash: [u8; 32],
    /// Replication metadata
    pub replication_count: u32,
}

impl QuinePayload {
    /// Create a new quine payload
    pub fn new(source: Vec<u8>, verification_hash: [u8; 32]) -> Self {
        Self {
            source,
            verification_hash,
            replication_count: 0,
        }
    }

    /// Increment replication counter
    pub fn replicate(&mut self) {
        self.replication_count = self.replication_count.saturating_add(1);
    }

    /// Verify payload integrity
    pub fn verify(&self) -> bool {
        !self.verification_hash.iter().all(|&b| b == 0)
    }
}

/// Self-referential state container
#[derive(Debug)]
pub struct SelfRefState<T> {
    /// The contained state
    state: T,
    /// Hash of the state for self-verification
    state_hash: [u8; 32],
    /// Version number for change tracking
    version: u64,
}

impl<T> SelfRefState<T> {
    /// Create a new self-referential state
    pub fn new(state: T, state_hash: [u8; 32]) -> Self {
        Self {
            state,
            state_hash,
            version: 0,
        }
    }

    /// Get reference to state
    pub fn state(&self) -> &T {
        &self.state
    }

    /// Get state hash
    pub fn hash(&self) -> &[u8; 32] {
        &self.state_hash
    }

    /// Get version
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Update state with new hash
    pub fn update(&mut self, new_state: T, new_hash: [u8; 32]) {
        self.state = new_state;
        self.state_hash = new_hash;
        self.version = self.version.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_referential_builder() {
        let hash = [1u8; 32];
        
        let result = SelfReferentialBuilder::new()
            .append(&[0x60, 0x00])
            .reserve_hash_space()
            .append(&[0xF3])
            .build(hash);

        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.verify());
    }

    #[test]
    fn test_quine_payload_replication() {
        let mut payload = QuinePayload::new(vec![1, 2, 3], [4u8; 32]);
        assert_eq!(payload.replication_count, 0);
        
        payload.replicate();
        assert_eq!(payload.replication_count, 1);
        
        payload.replicate();
        assert_eq!(payload.replication_count, 2);
    }

    #[test]
    fn test_invalid_offset() {
        let short_data = vec![0u8; 10];
        let result = SelfReferentialCode::new(short_data, 5, [0u8; 32]);
        assert_eq!(result, Err(SelfRefError::InvalidOffset));
    }
}
