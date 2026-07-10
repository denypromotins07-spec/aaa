//! Akash Bidding Engine - Decentralized Compute Resource Acquisition
//! 
//! Bids for compute resources on Akash Network using USDC,
//! automatically deploying NEXUS-OMEGA nodes to sovereign data centers.

use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use serde::{Serialize, Deserialize};

/// Unique bid identifier
pub type BidId = String;
/// Unique deployment identifier  
pub type DeploymentId = String;
/// Unique node identifier
pub type NodeId = u64;

/// Compute resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// CPU cores required
    pub cpu_cores: u32,
    /// Memory in GB
    pub memory_gb: u32,
    /// Storage in GB
    pub storage_gb: u32,
    /// GPU required
    pub gpu_required: bool,
    /// GPU model (if required)
    pub gpu_model: Option<String>,
    /// Bandwidth in Mbps
    pub bandwidth_mbps: u32,
    /// Geographic region preference
    pub preferred_regions: Vec<String>,
    /// Anti-affinity (avoid same provider)
    pub anti_affinity: bool,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: 8,
            memory_gb: 32,
            storage_gb: 500,
            gpu_required: false,
            gpu_model: None,
            bandwidth_mbps: 1000,
            preferred_regions: vec![],
            anti_affinity: true,
        }
    }
}

/// Provider offer from Akash network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderOffer {
    pub provider_id: String,
    pub provider_name: String,
    pub region: String,
    pub price_per_block: u128, // In micro-USDC
    pub available_resources: ResourceRequirements,
    pub reputation_score: f64,
    pub uptime_percentage: f64,
    pub lease_duration_blocks: u64,
}

/// Bid status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BidStatus {
    Open,
    Active,
    Outbid,
    Accepted,
    Rejected,
    Closed,
}

/// Bid information
#[derive(Debug, Clone)]
pub struct BidInfo {
    pub bid_id: BidId,
    pub deployment_id: DeploymentId,
    pub max_price: u128,
    pub current_price: u128,
    pub status: BidStatus,
    pub created_at: Instant,
    pub expires_at: Option<Instant>,
    pub provider_id: Option<String>,
    pub resources: ResourceRequirements,
}

/// Lease state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseState {
    Pending,
    Active,
    Expiring,
    Expired,
    Terminated,
}

/// Active lease
#[derive(Debug, Clone)]
pub struct ActiveLease {
    pub lease_id: String,
    pub deployment_id: DeploymentId,
    pub provider_id: String,
    pub price_per_block: u128,
    pub start_block: u64,
    pub end_block: u64,
    pub state: LeaseState,
    pub node_id: Option<NodeId>,
    pub ip_addresses: Vec<String>,
}

/// Configuration for bidding engine
#[derive(Debug, Clone)]
pub struct BiddingConfig {
    /// Maximum price per block (micro-USDC)
    pub max_price_per_block: u128,
    /// Minimum reputation score for providers
    pub min_reputation_score: f64,
    /// Minimum uptime percentage
    pub min_uptime_percentage: f64,
    /// Auto-renew leases before expiry (in blocks)
    pub auto_renew_threshold: u64,
    /// USDC contract address
    pub usdc_contract: String,
    /// Akash RPC endpoint
    pub akash_rpc_endpoint: String,
    /// Wallet address for payments
    pub wallet_address: String,
}

impl Default for BiddingConfig {
    fn default() -> Self {
        Self {
            max_price_per_block: 1000000, // 1 USDC per block max
            min_reputation_score: 0.8,
            min_uptime_percentage: 95.0,
            auto_renew_threshold: 100,
            usdc_contract: "0x...".to_string(),
            akash_rpc_endpoint: "https://rpc.akash.network".to_string(),
            wallet_address: "akash1...".to_string(),
        }
    }
}

/// Akash Bidding Engine
pub struct AkashBiddingEngine {
    config: BiddingConfig,
    active_bids: RwLock<HashMap<BidId, BidInfo>>,
    active_leases: RwLock<HashMap<String, ActiveLease>>,
    provider_cache: RwLock<HashMap<String, ProviderOffer>>,
    event_tx: mpsc::Sender<BiddingEvent>,
}

/// Events emitted by bidding engine
#[derive(Debug, Clone)]
pub enum BiddingEvent {
    BidCreated(BidId, DeploymentId),
    BidWon(BidId, String),
    BidLost(BidId),
    LeaseCreated(String, DeploymentId),
    LeaseExpiring(String, u64),
    LeaseTerminated(String),
    ProviderSelected(String, u128),
    InsufficientFunds(DeploymentId),
}

impl AkashBiddingEngine {
    pub fn new(config: BiddingConfig) -> Self {
        let (event_tx, _) = mpsc::channel(100);

        Self {
            config,
            active_bids: RwLock::new(HashMap::new()),
            active_leases: RwLock::new(HashMap::new()),
            provider_cache: RwLock::new(HashMap::new()),
            event_tx,
        }
    }

    /// Initialize the bidding engine
    pub async fn initialize(&self) -> Result<(), BiddingError> {
        // Verify wallet has sufficient USDC balance
        // In production, would query USDC contract
        
        Ok(())
    }

    /// Create a new bid for compute resources
    pub async fn create_bid(
        &self,
        deployment_id: DeploymentId,
        resources: ResourceRequirements,
        max_price: u128,
    ) -> Result<BidId, BiddingError> {
        let bid_id = format!("bid_{}_{}", deployment_id, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis());

        let bid = BidInfo {
            bid_id: bid_id.clone(),
            deployment_id,
            max_price,
            current_price: 0,
            status: BidStatus::Open,
            created_at: Instant::now(),
            expires_at: Some(Instant::now() + Duration::from_secs(300)), // 5 min default
            provider_id: None,
            resources,
        };

        {
            let mut bids = self.active_bids.write().await;
            bids.insert(bid_id.clone(), bid);
        }

        // Emit event
        let _ = self.event_tx.send(BiddingEvent::BidCreated(
            bid_id.clone(),
            bid.deployment_id.clone(),
        )).await;

        Ok(bid_id)
    }

    /// Receive provider offers and select best one
    pub async fn evaluate_offers(&self, bid_id: &str, offers: Vec<ProviderOffer>) -> Result<Option<String>, BiddingError> {
        let mut bids = self.active_bids.write().await;
        let bid = bids.get_mut(bid_id)
            .ok_or_else(|| BiddingError::BidNotFound(bid_id.to_string()))?;

        if bid.status != BidStatus::Open {
            return Ok(None);
        }

        // Filter offers by requirements
        let qualified_offers: Vec<_> = offers.into_iter()
            .filter(|offer| {
                offer.reputation_score >= self.config.min_reputation_score
                    && offer.uptime_percentage >= self.config.min_uptime_percentage
                    && offer.price_per_block <= bid.max_price
                    && self.resources_satisfied(&bid.resources, &offer.available_resources)
            })
            .collect();

        if qualified_offers.is_empty() {
            return Ok(None);
        }

        // Select best offer (lowest price with good reputation)
        let best_offer = qualified_offers.into_iter()
            .min_by(|a, b| {
                // Score = price * (1/reputation)
                let score_a = a.price_per_block as f64 / a.reputation_score;
                let score_b = b.price_per_block as f64 / b.reputation_score;
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        // Cache provider info
        {
            let mut cache = self.provider_cache.write().await;
            cache.insert(best_offer.provider_id.clone(), best_offer.clone());
        }

        // Update bid status
        bid.current_price = best_offer.price_per_block;
        bid.provider_id = Some(best_offer.provider_id.clone());
        bid.status = BidStatus::Accepted;

        // Emit events
        let _ = self.event_tx.send(BiddingEvent::ProviderSelected(
            best_offer.provider_id.clone(),
            best_offer.price_per_block,
        )).await;

        let _ = self.event_tx.send(BiddingEvent::BidWon(
            bid_id.to_string(),
            best_offer.provider_id.clone(),
        )).await;

        Ok(Some(best_offer.provider_id))
    }

    /// Check if resources are satisfied
    fn resources_satisfied(&self, required: &ResourceRequirements, available: &ResourceRequirements) -> bool {
        available.cpu_cores >= required.cpu_cores
            && available.memory_gb >= required.memory_gb
            && available.storage_gb >= required.storage_gb
            && (!required.gpu_required || available.gpu_required)
            && available.bandwidth_mbps >= required.bandwidth_mbps
    }

    /// Create a lease after winning a bid
    pub async fn create_lease(
        &self,
        bid_id: &str,
        provider_id: &str,
        lease_duration_blocks: u64,
    ) -> Result<String, BiddingError> {
        let bid = {
            let bids = self.active_bids.read().await;
            bids.get(bid_id)
                .cloned()
                .ok_or_else(|| BiddingError::BidNotFound(bid_id.to_string()))?
        };

        // In production, would interact with Akash smart contracts
        // For now, simulate lease creation
        
        let lease_id = format!("lease_{}_{}", bid.deployment_id, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis());

        let current_block = 1000000u64; // Simulated current block
        
        let lease = ActiveLease {
            lease_id: lease_id.clone(),
            deployment_id: bid.deployment_id.clone(),
            provider_id: provider_id.to_string(),
            price_per_block: bid.current_price,
            start_block: current_block,
            end_block: current_block + lease_duration_blocks,
            state: LeaseState::Active,
            node_id: None,
            ip_addresses: vec![],
        };

        {
            let mut leases = self.active_leases.write().await;
            leases.insert(lease_id.clone(), lease);
        }

        // Emit event
        let _ = self.event_tx.send(BiddingEvent::LeaseCreated(
            lease_id.clone(),
            bid.deployment_id,
        )).await;

        Ok(lease_id)
    }

    /// Get active leases
    pub async fn get_active_leases(&self) -> Vec<ActiveLease> {
        let leases = self.active_leases.read().await;
        leases.values()
            .filter(|l| l.state == LeaseState::Active)
            .cloned()
            .collect()
    }

    /// Check for expiring leases
    pub async fn check_expiring_leases(&self, current_block: u64) -> Result<Vec<String>, BiddingError> {
        let mut expiring = Vec::new();

        {
            let mut leases = self.active_leases.write().await;
            for (lease_id, lease) in leases.iter_mut() {
                if lease.state == LeaseState::Active {
                    let blocks_remaining = lease.end_block.saturating_sub(current_block);
                    if blocks_remaining <= self.config.auto_renew_threshold {
                        lease.state = LeaseState::Expiring;
                        expiring.push(lease_id.clone());

                        // Emit event
                        let _ = self.event_tx.send(BiddingEvent::LeaseExpiring(
                            lease_id.clone(),
                            blocks_remaining,
                        )).await;
                    }
                }
            }
        }

        Ok(expiring)
    }

    /// Renew an expiring lease
    pub async fn renew_lease(&self, lease_id: &str, additional_blocks: u64) -> Result<(), BiddingError> {
        let mut leases = self.active_leases.write().await;
        let lease = leases.get_mut(lease_id)
            .ok_or_else(|| BiddingError::LeaseNotFound(lease_id.to_string()))?;

        if lease.state != LeaseState::Expiring && lease.state != LeaseState::Active {
            return Err(BiddingError::InvalidLeaseState);
        }

        // In production, would negotiate renewal with provider
        lease.end_block += additional_blocks;
        lease.state = LeaseState::Active;

        Ok(())
    }

    /// Terminate a lease
    pub async fn terminate_lease(&self, lease_id: &str) -> Result<(), BiddingError> {
        let mut leases = self.active_leases.write().await;
        let lease = leases.get_mut(lease_id)
            .ok_or_else(|| BiddingError::LeaseNotFound(lease_id.to_string()))?;

        lease.state = LeaseState::Terminated;

        // Emit event
        let _ = self.event_tx.send(BiddingEvent::LeaseTerminated(lease_id.to_string())).await;

        Ok(())
    }

    /// Get total cost of active leases
    pub async fn get_total_cost(&self) -> u128 {
        let leases = self.active_leases.read().await;
        leases.values()
            .filter(|l| l.state == LeaseState::Active)
            .map(|l| l.price_per_block)
            .sum()
    }

    /// Get provider statistics
    pub async fn get_provider_stats(&self, provider_id: &str) -> Option<ProviderStats> {
        let cache = self.provider_cache.read().await;
        cache.get(provider_id).map(|p| ProviderStats {
            provider_id: p.provider_id.clone(),
            provider_name: p.provider_name.clone(),
            total_leases: 0,
            avg_price: p.price_per_block,
            reputation: p.reputation_score,
        })
    }
}

/// Provider statistics
#[derive(Debug, Clone)]
pub struct ProviderStats {
    pub provider_id: String,
    pub provider_name: String,
    pub total_leases: usize,
    pub avg_price: u128,
    pub reputation: f64,
}

/// Bidding error types
#[derive(Debug, thiserror::Error)]
pub enum BiddingError {
    #[error("Bid not found: {0}")]
    BidNotFound(String),
    #[error("Lease not found: {0}")]
    LeaseNotFound(String),
    #[error("Invalid lease state")]
    InvalidLeaseState,
    #[error("Insufficient funds")]
    InsufficientFunds,
    #[error("No qualified providers")]
    NoQualifiedProviders,
    #[error("Smart contract error: {0}")]
    ContractError(String),
    #[error("RPC error: {0}")]
    RpcError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_bid() {
        let config = BiddingConfig::default();
        let engine = AkashBiddingEngine::new(config);
        engine.initialize().await.unwrap();

        let resources = ResourceRequirements::default();
        let bid_id = engine.create_bid("deploy-1".to_string(), resources, 500000).await.unwrap();

        assert!(bid_id.starts_with("bid_deploy-1_"));

        let bids = engine.active_bids.read().await;
        assert!(bids.contains_key(&bid_id));
    }

    #[tokio::test]
    async fn test_evaluate_offers() {
        let config = BiddingConfig::default();
        let engine = AkashBiddingEngine::new(config);
        engine.initialize().await.unwrap();

        let resources = ResourceRequirements::default();
        let bid_id = engine.create_bid("deploy-2".to_string(), resources, 500000).await.unwrap();

        let offers = vec![
            ProviderOffer {
                provider_id: "provider-1".to_string(),
                provider_name: "Test Provider 1".to_string(),
                region: "us-west".to_string(),
                price_per_block: 300000,
                available_resources: ResourceRequirements::default(),
                reputation_score: 0.95,
                uptime_percentage: 99.0,
                lease_duration_blocks: 10000,
            },
            ProviderOffer {
                provider_id: "provider-2".to_string(),
                provider_name: "Test Provider 2".to_string(),
                region: "eu-central".to_string(),
                price_per_block: 250000,
                available_resources: ResourceRequirements::default(),
                reputation_score: 0.85,
                uptime_percentage: 97.0,
                lease_duration_blocks: 10000,
            },
        ];

        let winner = engine.evaluate_offers(&bid_id, offers).await.unwrap();
        assert!(winner.is_some());
        
        // Should select provider-2 (lower effective cost despite lower reputation)
        assert_eq!(winner.unwrap(), "provider-2");
    }

    #[tokio::test]
    async fn test_lease_lifecycle() {
        let config = BiddingConfig::default();
        let engine = AkashBiddingEngine::new(config);
        engine.initialize().await.unwrap();

        let resources = ResourceRequirements::default();
        let bid_id = engine.create_bid("deploy-3".to_string(), resources, 500000).await.unwrap();

        let offer = vec![ProviderOffer {
            provider_id: "provider-1".to_string(),
            provider_name: "Test".to_string(),
            region: "us".to_string(),
            price_per_block: 300000,
            available_resources: ResourceRequirements::default(),
            reputation_score: 0.9,
            uptime_percentage: 98.0,
            lease_duration_blocks: 10000,
        }];

        engine.evaluate_offers(&bid_id, offer).await.unwrap();
        
        let lease_id = engine.create_lease(&bid_id, "provider-1", 10000).await.unwrap();
        
        let leases = engine.get_active_leases().await;
        assert_eq!(leases.len(), 1);
        assert_eq!(leases[0].lease_id, lease_id);
    }
}
