//! Distributed State Synchronization for Swarm Resurrection
//! 
//! Syncs checkpoint data across distributed file systems (JuiceFS, Ceph)
//! to enable node resurrection from any location in the swarm.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use serde::{Serialize, Deserialize};

/// Unique checkpoint identifier
pub type CheckpointId = String;
/// Unique node identifier
pub type NodeId = u64;

/// Distributed storage backend
#[derive(Debug, Clone)]
pub enum StorageBackend {
    /// Local filesystem (for testing)
    Local(PathBuf),
    /// JuiceFS distributed filesystem
    JuiceFS { mount_point: PathBuf, bucket: String },
    /// Ceph object storage
    Ceph { cluster_id: String, pool: String },
    /// IPFS for decentralized storage
    Ipfs { gateway: String },
}

/// Configuration for distributed state sync
#[derive(Debug, Clone)]
pub struct StateSyncConfig {
    pub node_id: NodeId,
    pub storage_backend: StorageBackend,
    pub replication_factor: usize,
    pub sync_interval: Duration,
    pub max_concurrent_transfers: usize,
    pub compression_enabled: bool,
    pub encryption_enabled: bool,
}

impl Default for StateSyncConfig {
    fn default() -> Self {
        Self {
            node_id: 0,
            storage_backend: StorageBackend::Local(PathBuf::from("/tmp/nexus_state")),
            replication_factor: 3,
            sync_interval: Duration::from_secs(60),
            max_concurrent_transfers: 10,
            compression_enabled: true,
            encryption_enabled: false,
        }
    }
}

/// State metadata for synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateMetadata {
    pub checkpoint_id: CheckpointId,
    pub source_node: NodeId,
    pub created_at: u128,
    pub size_bytes: u64,
    pub checksum: [u8; 32],
    pub version: u64,
    pub is_compressed: bool,
    pub replicas: Vec<NodeId>,
}

impl StateMetadata {
    pub fn new(checkpoint_id: CheckpointId, source_node: NodeId, size_bytes: u64) -> Self {
        Self {
            checkpoint_id,
            source_node,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            size_bytes,
            checksum: [0u8; 32],
            version: 1,
            is_compressed: false,
            replicas: Vec::new(),
        }
    }

    pub fn compute_checksum(&mut self, data: &[u8]) {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        self.checksum = hasher.finalize().into();
    }

    pub fn verify_checksum(&self, data: &[u8]) -> bool {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        let computed: [u8; 32] = hasher.finalize().into();
        computed == self.checksum
    }
}

/// Synchronization status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Verified,
}

/// State transfer record
#[derive(Debug, Clone)]
pub struct StateTransfer {
    pub checkpoint_id: CheckpointId,
    pub source_node: NodeId,
    pub target_nodes: Vec<NodeId>,
    pub status: SyncStatus,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
    pub bytes_transferred: u64,
    pub error: Option<String>,
}

/// Distributed State Synchronizer
pub struct DistributedStateSync {
    config: StateSyncConfig,
    local_states: RwLock<HashMap<CheckpointId, Vec<u8>>>,
    metadata_cache: RwLock<HashMap<CheckpointId, StateMetadata>>,
    pending_transfers: RwLock<HashMap<CheckpointId, StateTransfer>>,
    peer_nodes: RwLock<HashMap<NodeId, String>>, // NodeId -> Address
    event_tx: mpsc::Sender<StateSyncEvent>,
}

/// Events emitted by state sync
#[derive(Debug, Clone)]
pub enum StateSyncEvent {
    TransferStarted(CheckpointId, NodeId),
    TransferCompleted(CheckpointId, NodeId),
    TransferFailed(CheckpointId, String),
    StateVerified(CheckpointId),
    ReplicationComplete(CheckpointId, usize),
}

impl DistributedStateSync {
    pub fn new(config: StateSyncConfig) -> Self {
        let (event_tx, _) = mpsc::channel(100);

        Self {
            config,
            local_states: RwLock::new(HashMap::new()),
            metadata_cache: RwLock::new(HashMap::new()),
            pending_transfers: RwLock::new(HashMap::new()),
            peer_nodes: RwLock::new(HashMap::new()),
            event_tx,
        }
    }

    /// Initialize the state synchronizer
    pub async fn initialize(&self) -> Result<(), StateSyncError> {
        // Create storage directory if using local backend
        if let StorageBackend::Local(path) = &self.config.storage_backend {
            tokio::fs::create_dir_all(path).await?;
        }

        Ok(())
    }

    /// Register a peer node for replication
    pub async fn register_peer(&self, node_id: NodeId, address: String) {
        let mut peers = self.peer_nodes.write().await;
        peers.insert(node_id, address);
    }

    /// Store state locally
    pub async fn store_local(&self, checkpoint_id: CheckpointId, data: Vec<u8>) -> Result<(), StateSyncError> {
        let size_bytes = data.len() as u64;
        
        // Create metadata
        let mut metadata = StateMetadata::new(
            checkpoint_id.clone(),
            self.config.node_id,
            size_bytes,
        );
        metadata.compute_checksum(&data);
        metadata.is_compressed = self.config.compression_enabled;

        // Compress if enabled
        let stored_data = if self.config.compression_enabled {
            // In production, use actual compression
            data.clone()
        } else {
            data.clone()
        };

        // Store locally
        {
            let mut states = self.local_states.write().await;
            states.insert(checkpoint_id.clone(), stored_data);
        }

        {
            let mut metadata_cache = self.metadata_cache.write().await;
            metadata_cache.insert(checkpoint_id.clone(), metadata);
        }

        Ok(())
    }

    /// Initiate replication to peer nodes
    pub async fn replicate_to_peers(&self, checkpoint_id: &str) -> Result<usize, StateSyncError> {
        let metadata = {
            let cache = self.metadata_cache.read().await;
            cache.get(checkpoint_id)
                .cloned()
                .ok_or_else(|| StateSyncError::CheckpointNotFound(checkpoint_id.to_string()))?
        };

        let data = {
            let states = self.local_states.read().await;
            states.get(checkpoint_id)
                .cloned()
                .ok_or_else(|| StateSyncError::CheckpointNotFound(checkpoint_id.to_string()))?
        };

        // Get peer nodes for replication
        let peers = {
            let peer_nodes = self.peer_nodes.read().await;
            peer_nodes.keys()
                .take(self.config.replication_factor)
                .copied()
                .collect::<Vec<_>>()
        };

        if peers.is_empty() {
            return Ok(0);
        }

        // Create transfer record
        let transfer = StateTransfer {
            checkpoint_id: checkpoint_id.to_string(),
            source_node: self.config.node_id,
            target_nodes: peers.clone(),
            status: SyncStatus::InProgress,
            started_at: Some(Instant::now()),
            completed_at: None,
            bytes_transferred: 0,
            error: None,
        };

        {
            let mut transfers = self.pending_transfers.write().await;
            transfers.insert(checkpoint_id.to_string(), transfer);
        }

        // Emit event
        let _ = self.event_tx.send(StateSyncEvent::TransferStarted(
            checkpoint_id.to_string(),
            self.config.node_id,
        )).await;

        // Simulate replication (in production, would send to peers)
        let mut successful_replicas = 0;
        for peer_id in &peers {
            // Simulate network transfer
            tokio::time::sleep(Duration::from_millis(10)).await;
            
            // Update metadata with replica
            if let Some(meta) = self.metadata_cache.read().await.get(checkpoint_id) {
                let mut updated_meta = meta.clone();
                updated_meta.replicas.push(*peer_id);
                
                let mut cache = self.metadata_cache.write().await;
                cache.insert(checkpoint_id.to_string(), updated_meta);
            }

            successful_replicas += 1;
        }

        // Mark transfer complete
        {
            let mut transfers = self.pending_transfers.write().await;
            if let Some(transfer) = transfers.get_mut(checkpoint_id) {
                transfer.status = SyncStatus::Completed;
                transfer.completed_at = Some(Instant::now());
                transfer.bytes_transferred = data.len() as u64 * successful_replicas as u64;
            }
        }

        // Emit completion event
        let _ = self.event_tx.send(StateSyncEvent::ReplicationComplete(
            checkpoint_id.to_string(),
            successful_replicas,
        )).await;

        Ok(successful_replicas)
    }

    /// Fetch state from a remote node
    pub async fn fetch_remote(&self, checkpoint_id: &str, source_node: NodeId) -> Result<Vec<u8>, StateSyncError> {
        // In production, would fetch from remote node via network
        // For now, simulate by checking if we have it locally
        
        let states = self.local_states.read().await;
        if let Some(data) = states.get(checkpoint_id) {
            return Ok(data.clone());
        }

        Err(StateSyncError::CheckpointNotFound(checkpoint_id.to_string()))
    }

    /// Verify state integrity
    pub async fn verify_state(&self, checkpoint_id: &str) -> Result<bool, StateSyncError> {
        let metadata = {
            let cache = self.metadata_cache.read().await;
            cache.get(checkpoint_id)
                .cloned()
                .ok_or_else(|| StateSyncError::CheckpointNotFound(checkpoint_id.to_string()))?
        };

        let data = {
            let states = self.local_states.read().await;
            states.get(checkpoint_id)
                .cloned()
                .ok_or_else(|| StateSyncError::CheckpointNotFound(checkpoint_id.to_string()))?
        };

        if metadata.verify_checksum(&data) {
            Ok(true)
        } else {
            Err(StateSyncError::ChecksumMismatch)
        }
    }

    /// Get metadata for a checkpoint
    pub async fn get_metadata(&self, checkpoint_id: &str) -> Option<StateMetadata> {
        let cache = self.metadata_cache.read().await;
        cache.get(checkpoint_id).cloned()
    }

    /// List all available checkpoints
    pub async fn list_checkpoints(&self) -> Vec<CheckpointId> {
        let cache = self.metadata_cache.read().await;
        cache.keys().cloned().collect()
    }

    /// Delete a checkpoint
    pub async fn delete_checkpoint(&self, checkpoint_id: &str) -> Result<(), StateSyncError> {
        {
            let mut states = self.local_states.write().await;
            states.remove(checkpoint_id);
        }

        {
            let mut cache = self.metadata_cache.write().await;
            cache.remove(checkpoint_id);
        }

        {
            let mut transfers = self.pending_transfers.write().await;
            transfers.remove(checkpoint_id);
        }

        Ok(())
    }

    /// Get transfer status
    pub async fn get_transfer_status(&self, checkpoint_id: &str) -> Option<StateTransfer> {
        let transfers = self.pending_transfers.read().await;
        transfers.get(checkpoint_id).cloned()
    }

    /// Cleanup old checkpoints
    pub async fn cleanup_old_checkpoints(&self, max_age: Duration) -> Result<usize, StateSyncError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        let mut to_delete = Vec::new();

        {
            let cache = self.metadata_cache.read().await;
            for (id, meta) in cache.iter() {
                let age_nanos = now.saturating_sub(meta.created_at);
                if age_nanos > max_age.as_nanos() {
                    to_delete.push(id.clone());
                }
            }
        }

        let count = to_delete.len();
        for id in to_delete {
            self.delete_checkpoint(&id).await?;
        }

        Ok(count)
    }
}

/// State sync error types
#[derive(Debug, thiserror::Error)]
pub enum StateSyncError {
    #[error("Checkpoint not found: {0}")]
    CheckpointNotFound(String),
    #[error("Checksum mismatch")]
    ChecksumMismatch,
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Storage backend error: {0}")]
    BackendError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_local_storage() {
        let config = StateSyncConfig::default();
        let sync = DistributedStateSync::new(config);
        sync.initialize().await.unwrap();

        let data = vec![1u8, 2, 3, 4, 5];
        sync.store_local("test-checkpoint".to_string(), data.clone()).await.unwrap();

        let metadata = sync.get_metadata("test-checkpoint").await;
        assert!(metadata.is_some());
        assert_eq!(metadata.unwrap().size_bytes, 5);
    }

    #[tokio::test]
    async fn test_replication() {
        let config = StateSyncConfig::default();
        let sync = DistributedStateSync::new(config);
        sync.initialize().await.unwrap();

        // Register peers
        sync.register_peer(1, "192.168.1.1:8080".to_string()).await;
        sync.register_peer(2, "192.168.1.2:8080".to_string()).await;

        // Store and replicate
        let data = vec![1u8, 2, 3, 4, 5];
        sync.store_local("repl-test".to_string(), data).await.unwrap();
        
        let replicas = sync.replicate_to_peers("repl-test").await.unwrap();
        assert!(replicas >= 1);
    }

    #[tokio::test]
    async fn test_verification() {
        let config = StateSyncConfig::default();
        let sync = DistributedStateSync::new(config);
        sync.initialize().await.unwrap();

        let data = vec![1u8, 2, 3, 4, 5];
        sync.store_local("verify-test".to_string(), data.clone()).await.unwrap();

        let verified = sync.verify_state("verify-test").await;
        assert!(verified.is_ok());
        assert!(verified.unwrap());
    }
}
