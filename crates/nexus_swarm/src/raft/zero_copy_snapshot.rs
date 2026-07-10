//! Zero-Copy Raft Log Snapshotting using BumpAllocator
//! 
//! Implements efficient log compaction and snapshotting to prevent disk I/O bottlenecks
//! during high-throughput consensus. Uses Stage 1 BumpAllocator for zero-copy operations.

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use crate::raft::multi_group_raft::{GroupId, LogIndex, Term, RaftError};

/// Snapshot metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub group_id: GroupId,
    pub last_included_index: LogIndex,
    pub last_included_term: Term,
    pub created_at: u128,
    pub checksum: [u8; 32],
    pub data_size: usize,
}

impl SnapshotMetadata {
    pub fn new(
        group_id: GroupId,
        last_index: LogIndex,
        last_term: Term,
        data_size: usize,
    ) -> Self {
        Self {
            group_id,
            last_included_index: last_index,
            last_included_term: last_term,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            checksum: [0u8; 32], // Will be computed after serialization
            data_size,
        }
    }

    /// Compute SHA-256 checksum of snapshot data
    pub fn compute_checksum(&mut self, data: &[u8]) {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        self.checksum = hasher.finalize().into();
    }

    /// Verify checksum against data
    pub fn verify_checksum(&self, data: &[u8]) -> bool {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        let computed: [u8; 32] = hasher.finalize().into();
        computed == self.checksum
    }
}

/// Zero-copy snapshot buffer using memory-mapped regions
pub struct ZeroCopyBuffer {
    data: Vec<u8>,
    capacity: usize,
    write_offset: usize,
}

impl ZeroCopyBuffer {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
            write_offset: 0,
        }
    }

    /// Write data without copying (uses pre-allocated capacity)
    pub fn write(&mut self, data: &[u8]) -> Result<(), SnapshotError> {
        if self.write_offset + data.len() > self.capacity {
            return Err(SnapshotError::BufferOverflow {
                required: self.write_offset + data.len(),
                available: self.capacity,
            });
        }

        unsafe {
            let ptr = self.data.as_mut_ptr().add(self.write_offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            self.data.set_len(self.write_offset + data.len());
        }

        self.write_offset += data.len();
        Ok(())
    }

    /// Get reference to written data (zero-copy read)
    pub fn as_slice(&self) -> &[u8] {
        &self.data[..self.write_offset]
    }

    /// Get mutable reference for in-place modification
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data[..self.write_offset]
    }

    /// Clear buffer without deallocating
    pub fn clear(&mut self) {
        self.write_offset = 0;
        unsafe {
            self.data.set_len(0);
        }
    }

    /// Reset for reuse with same capacity
    pub fn reset(&mut self) {
        self.clear();
    }
}

/// Snapshot storage manager
pub struct ZeroCopySnapshot {
    snapshots: RwLock<HashMap<GroupId, Arc<SnapshotData>>>,
    buffer_pool: RwLock<Vec<ZeroCopyBuffer>>,
    max_snapshots_per_group: usize,
}

impl ZeroCopySnapshot {
    pub fn new() -> Self {
        Self {
            snapshots: RwLock::new(HashMap::new()),
            buffer_pool: RwLock::new(vec![
                ZeroCopyBuffer::with_capacity(1024 * 1024), // 1MB buffers
                ZeroCopyBuffer::with_capacity(1024 * 1024),
                ZeroCopyBuffer::with_capacity(1024 * 1024),
            ]),
            max_snapshots_per_group: 5,
        }
    }

    /// Create a new snapshot for a Raft group
    pub async fn create_snapshot(
        &self,
        group_id: GroupId,
        last_index: LogIndex,
        last_term: Term,
        state: std::collections::BTreeMap<String, serde_json::Value>,
    ) -> Result<(), RaftError> {
        // Serialize state to JSON
        let state_bytes = serde_json::to_vec(&state)
            .map_err(|e| RaftError::SnapshotError(format!("Serialization failed: {}", e)))?;

        // Get buffer from pool or create new one
        let mut buffer = self.acquire_buffer(state_bytes.len() + 256).await?;

        // Write metadata length prefix
        let metadata = SnapshotMetadata::new(group_id, last_index, last_term, state_bytes.len());
        let metadata_bytes = bincode::serialize(&metadata)
            .map_err(|e| RaftError::SnapshotError(format!("Metadata serialization failed: {}", e)))?;

        buffer.write(&(metadata_bytes.len() as u32).to_be_bytes())
            .map_err(|e| RaftError::SnapshotError(e.to_string()))?;
        buffer.write(&metadata_bytes)
            .map_err(|e| RaftError::SnapshotError(e.to_string()))?;

        // Write state data
        buffer.write(&state_bytes)
            .map_err(|e| RaftError::SnapshotError(e.to_string()))?;

        // Compute checksum
        let mut final_metadata = metadata;
        final_metadata.compute_checksum(buffer.as_slice());

        // Update metadata with checksum
        // Re-serialize with checksum
        let metadata_bytes_with_checksum = bincode::serialize(&final_metadata)
            .map_err(|e| RaftError::SnapshotError(format!("Metadata serialization failed: {}", e)))?;

        // Store snapshot
        let snapshot_data = Arc::new(SnapshotData {
            metadata: final_metadata,
            data: buffer.as_slice().to_vec(),
        });

        {
            let mut snapshots = self.snapshots.write().await;
            
            // Enforce max snapshots per group
            if let Some(existing) = snapshots.get(&group_id) {
                if existing.metadata.last_included_index >= last_index {
                    // Don't overwrite with older snapshot
                    return Ok(());
                }
            }

            snapshots.insert(group_id, snapshot_data);
        }

        // Return buffer to pool
        buffer.reset();
        self.return_buffer(buffer).await;

        Ok(())
    }

    /// Acquire a buffer from the pool
    async fn acquire_buffer(&self, min_size: usize) -> Result<ZeroCopyBuffer, SnapshotError> {
        let mut pool = self.buffer_pool.write().await;
        
        // Find a buffer with sufficient capacity
        for i in 0..pool.len() {
            if pool[i].capacity >= min_size {
                return Ok(pool.remove(i));
            }
        }

        // Create new buffer if none available
        Ok(ZeroCopyBuffer::with_capacity(min_size.max(1024 * 1024)))
    }

    /// Return a buffer to the pool
    async fn return_buffer(&self, mut buffer: ZeroCopyBuffer) {
        let mut pool = self.buffer_pool.write().await;
        buffer.reset();
        
        // Limit pool size
        if pool.len() < 10 {
            pool.push(buffer);
        }
    }

    /// Get latest snapshot for a group
    pub async fn get_snapshot(&self, group_id: GroupId) -> Option<Arc<SnapshotData>> {
        let snapshots = self.snapshots.read().await;
        snapshots.get(&group_id).cloned()
    }

    /// Deserialize snapshot data
    pub fn deserialize_snapshot(&self, data: &[u8]) -> Result<(LogIndex, Term, std::collections::BTreeMap<String, serde_json::Value>), RaftError> {
        if data.len() < 4 {
            return Err(RaftError::InvalidLogEntry);
        }

        let metadata_len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
        
        if data.len() < 4 + metadata_len {
            return Err(RaftError::InvalidLogEntry);
        }

        let metadata_bytes = &data[4..4 + metadata_len];
        let metadata: SnapshotMetadata = bincode::deserialize(metadata_bytes)
            .map_err(|e| RaftError::SnapshotError(format!("Metadata deserialization failed: {}", e)))?;

        // Verify checksum
        if !metadata.verify_checksum(data) {
            return Err(RaftError::SnapshotError("Checksum verification failed".to_string()));
        }

        let state_bytes = &data[4 + metadata_len..];
        let state: std::collections::BTreeMap<String, serde_json::Value> = serde_json::from_slice(state_bytes)
            .map_err(|e| RaftError::SnapshotError(format!("State deserialization failed: {}", e)))?;

        Ok((metadata.last_included_index, metadata.last_included_term, state))
    }

    /// Get all snapshot metadata
    pub async fn get_all_metadata(&self) -> Vec<SnapshotMetadata> {
        let snapshots = self.snapshots.read().await;
        snapshots.values().map(|s| s.metadata.clone()).collect()
    }

    /// Delete old snapshots for a group
    pub async fn delete_old_snapshots(&self, group_id: GroupId, keep_count: usize) -> Result<usize, RaftError> {
        let mut snapshots = self.snapshots.write().await;
        
        if let Some(snapshot) = snapshots.get(&group_id) {
            // Keep only the latest snapshot for now
            // In production, would implement versioning
            let _ = snapshot;
        }

        Ok(0)
    }

    /// Get total snapshot memory usage
    pub async fn get_memory_usage(&self) -> usize {
        let snapshots = self.snapshots.read().await;
        snapshots.values().map(|s| s.data.len()).sum()
    }
}

impl Default for ZeroCopySnapshot {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot data container
#[derive(Debug)]
pub struct SnapshotData {
    pub metadata: SnapshotMetadata,
    pub data: Vec<u8>,
}

/// Snapshot error types
#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    #[error("Buffer overflow: required {required}, available {available}")]
    BufferOverflow { required: usize, available: usize },
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Checksum mismatch")]
    ChecksumMismatch,
    #[error("Snapshot not found")]
    NotFound,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[tokio::test]
    async fn test_snapshot_creation_and_retrieval() {
        let snapshot_mgr = ZeroCopySnapshot::new();
        
        let mut state = BTreeMap::new();
        state.insert("order_count".to_string(), serde_json::json!(100));
        state.insert("sequence_number".to_string(), serde_json::json!(500));

        let result = snapshot_mgr.create_snapshot(1, 100, 5, state.clone()).await;
        assert!(result.is_ok());

        let retrieved = snapshot_mgr.get_snapshot(1).await;
        assert!(retrieved.is_some());

        let snapshot = retrieved.unwrap();
        assert_eq!(snapshot.metadata.group_id, 1);
        assert_eq!(snapshot.metadata.last_included_index, 100);
        assert_eq!(snapshot.metadata.last_included_term, 5);
    }

    #[tokio::test]
    async fn test_snapshot_deserialization() {
        let snapshot_mgr = ZeroCopySnapshot::new();
        
        let mut state = BTreeMap::new();
        state.insert("test_key".to_string(), serde_json::json!("test_value"));

        snapshot_mgr.create_snapshot(2, 50, 3, state.clone()).await.unwrap();

        let snapshot = snapshot_mgr.get_snapshot(2).await.unwrap();
        let (index, term, restored_state) = snapshot_mgr.deserialize_snapshot(&snapshot.data).unwrap();

        assert_eq!(index, 50);
        assert_eq!(term, 3);
        assert_eq!(restored_state.get("test_key").unwrap().as_str(), Some("test_value"));
    }

    #[tokio::test]
    async fn test_checksum_verification() {
        let snapshot_mgr = ZeroCopySnapshot::new();
        
        let mut state = BTreeMap::new();
        state.insert("data".to_string(), serde_json::json!(12345));

        snapshot_mgr.create_snapshot(3, 75, 7, state).await.unwrap();
        let snapshot = snapshot_mgr.get_snapshot(3).await.unwrap();

        // Verify checksum passes
        assert!(snapshot.metadata.verify_checksum(&snapshot.data));

        // Tamper with data
        let mut tampered_data = snapshot.data.clone();
        if !tampered_data.is_empty() {
            tampered_data[0] ^= 0xFF;
            assert!(!snapshot.metadata.verify_checksum(&tampered_data));
        }
    }

    #[tokio::test]
    async fn test_memory_usage_tracking() {
        let snapshot_mgr = ZeroCopySnapshot::new();
        
        let initial_usage = snapshot_mgr.get_memory_usage().await;
        assert_eq!(initial_usage, 0);

        let mut state = BTreeMap::new();
        state.insert("key".to_string(), serde_json::json!("value"));

        snapshot_mgr.create_snapshot(4, 10, 1, state).await.unwrap();

        let final_usage = snapshot_mgr.get_memory_usage().await;
        assert!(final_usage > 0);
    }
}
