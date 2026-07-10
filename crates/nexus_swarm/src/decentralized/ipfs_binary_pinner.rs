//! IPFS Binary Pinner - Decentralized Binary Distribution
//! 
//! Packages NEXUS-OMEGA binaries and FPGA bitstreams into IPFS CIDs
//! for uncensorable deployment across decentralized compute networks.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use serde::{Serialize, Deserialize};

/// Unique content identifier
pub type Cid = String;
/// Unique node identifier
pub type NodeId = u64;

/// Binary artifact type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactType {
    /// Compiled Rust binary
    RustBinary { target: String, profile: String },
    /// FPGA bitstream
    FpgaBitstream { vendor: String, family: String },
    /// Configuration file
    ConfigFile { name: String },
    /// Checkpoint data
    Checkpoint { checkpoint_id: String },
    /// Smart contract bytecode
    ContractBytecode { chain: String },
}

/// Artifact metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    pub name: String,
    pub version: String,
    pub artifact_type: ArtifactType,
    pub size_bytes: u64,
    pub checksum: [u8; 32],
    pub created_at: u128,
    pub pinned: bool,
    pub pin_count: usize,
    pub gateway_urls: Vec<String>,
}

impl ArtifactMetadata {
    pub fn new(name: String, version: String, artifact_type: ArtifactType, size_bytes: u64) -> Self {
        Self {
            name,
            version,
            artifact_type,
            size_bytes,
            checksum: [0u8; 32],
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            pinned: false,
            pin_count: 0,
            gateway_urls: Vec::new(),
        }
    }

    pub fn compute_checksum(&mut self, data: &[u8]) {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        self.checksum = hasher.finalize().into();
    }
}

/// Pinning status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinStatus {
    Pending,
    Pinning,
    Pinned,
    Failed,
    Unpinned,
}

/// Pinning request
#[derive(Debug, Clone)]
pub struct PinRequest {
    pub cid: Cid,
    pub artifact_name: String,
    priority: u8,
    status: PinStatus,
    created_at: Instant,
    gateways: Vec<String>,
}

/// IPFS configuration
#[derive(Debug, Clone)]
pub struct IpfsConfig {
    /// IPFS daemon API endpoint
    pub api_endpoint: String,
    /// IPFS gateways for retrieval
    pub gateways: Vec<String>,
    /// Default replication factor
    pub replication_factor: usize,
    /// Pin timeout
    pub pin_timeout: Duration,
    /// Whether to use IPNS for mutable references
    pub use_ipns: bool,
}

impl Default for IpfsConfig {
    fn default() -> Self {
        Self {
            api_endpoint: "http://127.0.0.1:5001".to_string(),
            gateways: vec![
                "https://ipfs.io".to_string(),
                "https://gateway.pinata.cloud".to_string(),
                "https://cloudflare-ipfs.com".to_string(),
            ],
            replication_factor: 3,
            pin_timeout: Duration::from_secs(300),
            use_ipns: true,
        }
    }
}

/// IPFS Binary Pinner
pub struct IpfsBinaryPinner {
    config: IpfsConfig,
    artifacts: RwLock<HashMap<Cid, ArtifactMetadata>>,
    pin_requests: RwLock<HashMap<Cid, PinRequest>>,
    local_cache: RwLock<HashMap<Cid, Vec<u8>>>,
    event_tx: mpsc::Sender<PinnerEvent>,
}

/// Events emitted by pinner
#[derive(Debug, Clone)]
pub enum PinnerEvent {
    ArtifactUploaded(Cid, String),
    PinStarted(Cid),
    PinCompleted(Cid),
    PinFailed(Cid, String),
    ArtifactRetrieved(Cid),
}

impl IpfsBinaryPinner {
    pub fn new(config: IpfsConfig) -> Self {
        let (event_tx, _) = mpsc::channel(100);

        Self {
            config,
            artifacts: RwLock::new(HashMap::new()),
            pin_requests: RwLock::new(HashMap::new()),
            local_cache: RwLock::new(HashMap::new()),
            event_tx,
        }
    }

    /// Initialize the pinner
    pub async fn initialize(&self) -> Result<(), PinnerError> {
        // Verify IPFS daemon is accessible (simplified check)
        // In production, would make actual API call
        
        Ok(())
    }

    /// Add an artifact to IPFS
    pub async fn add_artifact(
        &self,
        name: String,
        version: String,
        artifact_type: ArtifactType,
        data: Vec<u8>,
    ) -> Result<Cid, PinnerError> {
        let size_bytes = data.len() as u64;
        
        // Create metadata
        let mut metadata = ArtifactMetadata::new(name.clone(), version, artifact_type, size_bytes);
        metadata.compute_checksum(&data);

        // Generate CID (in production, this would be actual IPFS CID)
        let cid = self.generate_cid(&data).await;

        // Store in local cache
        {
            let mut cache = self.local_cache.write().await;
            cache.insert(cid.clone(), data);
        }

        // Store metadata
        {
            let mut artifacts = self.artifacts.write().await;
            artifacts.insert(cid.clone(), metadata);
        }

        // Emit event
        let _ = self.event_tx.send(PinnerEvent::ArtifactUploaded(
            cid.clone(),
            name,
        )).await;

        Ok(cid)
    }

    /// Generate CID from data (simplified - in production use actual IPFS hashing)
    async fn generate_cid(&self, data: &[u8]) -> Cid {
        use sha2::{Sha256, Digest};
        let hash = Sha256::digest(data);
        
        // Format as IPFS CID v1
        format!("bafybeig{}", hex::encode(&hash[..8]))
    }

    /// Pin an artifact to IPFS nodes
    pub async fn pin_artifact(&self, cid: &str) -> Result<(), PinnerError> {
        let metadata = {
            let artifacts = self.artifacts.read().await;
            artifacts.get(cid)
                .cloned()
                .ok_or_else(|| PinnerError::ArtifactNotFound(cid.to_string()))?
        };

        // Create pin request
        let request = PinRequest {
            cid: cid.to_string(),
            artifact_name: metadata.name.clone(),
            priority: 5,
            status: PinStatus::Pinning,
            created_at: Instant::now(),
            gateways: self.config.gateways.clone(),
        };

        {
            let mut requests = self.pin_requests.write().await;
            requests.insert(cid.to_string(), request);
        }

        // Emit pin started event
        let _ = self.event_tx.send(PinnerEvent::PinStarted(cid.to_string())).await;

        // Simulate pinning to multiple gateways
        // In production, would call IPFS pin API
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Update metadata with pin count
        {
            let mut artifacts = self.artifacts.write().await;
            if let Some(meta) = artifacts.get_mut(cid) {
                meta.pinned = true;
                meta.pin_count = self.config.replication_factor;
                meta.gateway_urls = self.config.gateways
                    .iter()
                    .take(self.config.replication_factor)
                    .map(|g| format!("{}/ipfs/{}", g, cid))
                    .collect();
            }
        }

        // Update pin request status
        {
            let mut requests = self.pin_requests.write().await;
            if let Some(req) = requests.get_mut(cid) {
                req.status = PinStatus::Pinned;
            }
        }

        // Emit completion event
        let _ = self.event_tx.send(PinnerEvent::PinCompleted(cid.to_string())).await;

        Ok(())
    }

    /// Retrieve artifact data
    pub async fn retrieve_artifact(&self, cid: &str) -> Result<Vec<u8>, PinnerError> {
        // Check local cache first
        {
            let cache = self.local_cache.read().await;
            if let Some(data) = cache.get(cid) {
                return Ok(data.clone());
            }
        }

        // In production, would fetch from IPFS gateway
        // For now, return error if not in cache
        Err(PinnerError::ArtifactNotFound(cid.to_string()))
    }

    /// Get artifact metadata
    pub async fn get_metadata(&self, cid: &str) -> Option<ArtifactMetadata> {
        let artifacts = self.artifacts.read().await;
        artifacts.get(cid).cloned()
    }

    /// List all artifacts
    pub async fn list_artifacts(&self) -> Vec<(Cid, ArtifactMetadata)> {
        let artifacts = self.artifacts.read().await;
        artifacts.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    /// Unpin an artifact
    pub async fn unpin_artifact(&self, cid: &str) -> Result<(), PinnerError> {
        {
            let mut artifacts = self.artifacts.write().await;
            if let Some(meta) = artifacts.get_mut(cid) {
                meta.pinned = false;
                meta.pin_count = 0;
                meta.gateway_urls.clear();
            }
        }

        {
            let mut requests = self.pin_requests.write().await;
            if let Some(req) = requests.get_mut(cid) {
                req.status = PinStatus::Unpinned;
            }
        }

        Ok(())
    }

    /// Get pin status
    pub async fn get_pin_status(&self, cid: &str) -> Option<PinStatus> {
        let requests = self.pin_requests.read().await;
        requests.get(cid).map(|r| r.status)
    }

    /// Cleanup local cache
    pub async fn cleanup_cache(&self, max_age: Duration) -> Result<usize, PinnerError> {
        let now = Instant::now();
        let mut to_remove = Vec::new();

        {
            let requests = self.pin_requests.read().await;
            for (cid, req) in requests.iter() {
                if now.duration_since(req.created_at) > max_age {
                    to_remove.push(cid.clone());
                }
            }
        }

        let count = to_remove.len();
        for cid in to_remove {
            let mut cache = self.local_cache.write().await;
            cache.remove(&cid);
        }

        Ok(count)
    }

    /// Export artifact info for smart contract deployment
    pub async fn export_deployment_info(&self, cid: &str) -> Result<DeploymentInfo, PinnerError> {
        let metadata = self.get_metadata(cid).await
            .ok_or_else(|| PinnerError::ArtifactNotFound(cid.to_string()))?;

        Ok(DeploymentInfo {
            cid,
            name: metadata.name,
            version: metadata.version,
            size_bytes: metadata.size_bytes,
            gateway_urls: metadata.gateway_urls,
            checksum: hex::encode(&metadata.checksum),
        })
    }
}

/// Deployment information for smart contracts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentInfo {
    pub cid: String,
    pub name: String,
    pub version: String,
    pub size_bytes: u64,
    pub gateway_urls: Vec<String>,
    pub checksum: String,
}

/// Pinner error types
#[derive(Debug, thiserror::Error)]
pub enum PinnerError {
    #[error("Artifact not found: {0}")]
    ArtifactNotFound(String),
    #[error("IPFS API error: {0}")]
    IpfsApiError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Checksum mismatch")]
    ChecksumMismatch,
    #[error("Pin timeout")]
    PinTimeout,
}

// Helper for hex encoding
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add_artifact() {
        let config = IpfsConfig::default();
        let pinner = IpfsBinaryPinner::new(config);
        pinner.initialize().await.unwrap();

        let data = vec![1u8, 2, 3, 4, 5];
        let cid = pinner.add_artifact(
            "test-binary".to_string(),
            "1.0.0".to_string(),
            ArtifactType::RustBinary {
                target: "x86_64-unknown-linux-gnu".to_string(),
                profile: "release".to_string(),
            },
            data,
        ).await.unwrap();

        assert!(cid.starts_with("bafybeig"));

        let metadata = pinner.get_metadata(&cid).await;
        assert!(metadata.is_some());
        assert_eq!(metadata.unwrap().size_bytes, 5);
    }

    #[tokio::test]
    async fn test_pin_artifact() {
        let config = IpfsConfig::default();
        let pinner = IpfsBinaryPinner::new(config);
        pinner.initialize().await.unwrap();

        let data = vec![1u8, 2, 3];
        let cid = pinner.add_artifact(
            "pin-test".to_string(),
            "1.0.0".to_string(),
            ArtifactType::ConfigFile { name: "test".to_string() },
            data,
        ).await.unwrap();

        pinner.pin_artifact(&cid).await.unwrap();

        let metadata = pinner.get_metadata(&cid).await.unwrap();
        assert!(metadata.pinned);
        assert!(metadata.pin_count > 0);
    }
}
