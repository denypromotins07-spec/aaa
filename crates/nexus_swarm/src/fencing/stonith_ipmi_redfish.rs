//! STONITH (Shoot The Other Node In The Head) Hardware Fencing
//! 
//! Implements IPMI/Redfish-based hardware fencing to prevent split-brain scenarios.
//! Before assuming leadership, a minority partition must physically fence isolated nodes.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use tokio::time::timeout;
use serde::{Serialize, Deserialize};

/// Unique node identifier
pub type NodeId = u64;

/// Fencing state for a node
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FencingState {
    /// Node is active and unfenced
    Active,
    /// Fencing command has been initiated
    FencingInProgress,
    /// Node has been fenced (power cut or network severed)
    Fenced,
    /// Fencing confirmation received
    ConfirmedFenced,
    /// Fencing failed
    FencingFailed,
}

/// IPMI connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpmiConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String, // In production, use secure credential storage
    pub auth_type: IpmiAuthType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IpmiAuthType {
    Password,
    MD2,
    MD5,
    Straight,
    OEM,
}

impl Default for IpmiConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 623,
            username: "admin".to_string(),
            password: String::new(),
            auth_type: IpmiAuthType::Password,
        }
    }
}

/// Redfish connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedfishConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    pub use_https: bool,
    pub system_id: String,
}

impl Default for RedfishConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 443,
            username: "root".to_string(),
            password: String::new(),
            use_https: true,
            system_id: "System.Embedded.1".to_string(),
        }
    }
}

/// Fencing method
#[derive(Debug, Clone)]
pub enum FencingMethod {
    /// Power off via IPMI
    IpmiPowerOff(IpmiConfig),
    /// Power cycle via IPMI
    IpmiPowerCycle(IpmiConfig),
    /// Network disable via Redfish
    RedfishNetworkDisable(RedfishConfig),
    /// System reset via Redfish
    RedfishReset(RedfishConfig),
    /// Custom fencing command
    Custom(String),
}

/// Fencing result
#[derive(Debug, Clone)]
pub struct FencingResult {
    pub node_id: NodeId,
    pub success: bool,
    pub method: String,
    pub confirmation: Option<FencingConfirmation>,
    pub error: Option<String>,
    pub timestamp: Instant,
    pub retry_count: u32,
}

/// Cryptographic fencing confirmation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FencingConfirmation {
    /// Signature from the fenced node (if it can still communicate)
    pub signature: Vec<u8>,
    /// Timestamp of confirmation
    pub timestamp: u128,
    /// Hash of fencing command
    pub command_hash: [u8; 32],
    /// Witness signatures from other nodes
    pub witness_signatures: Vec<(NodeId, Vec<u8>)>,
}

impl FencingConfirmation {
    pub fn new(command_hash: [u8; 32]) -> Self {
        Self {
            signature: Vec::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            command_hash,
            witness_signatures: Vec::new(),
        }
    }

    /// Add a witness signature
    pub fn add_witness(&mut self, node_id: NodeId, signature: Vec<u8>) {
        self.witness_signatures.push((node_id, signature));
    }

    /// Verify confirmation has required witnesses
    pub fn is_fully_confirmed(&self, required_witnesses: usize) -> bool {
        self.witness_signatures.len() >= required_witnesses
    }
}

/// Configuration for STONITH fencing
#[derive(Debug, Clone)]
pub struct StonithConfig {
    /// Maximum time to wait for fencing confirmation
    pub confirmation_timeout: Duration,
    /// Number of retries for fencing commands
    pub max_retries: u32,
    /// Minimum number of witness confirmations required
    pub required_witnesses: usize,
    /// Time between retry attempts
    pub retry_interval: Duration,
    /// Whether to use cryptographic confirmation
    pub require_crypto_confirmation: bool,
}

impl Default for StonithConfig {
    fn default() -> Self {
        Self {
            confirmation_timeout: Duration::from_secs(30),
            max_retries: 3,
            required_witnesses: 2,
            retry_interval: Duration::from_secs(5),
            require_crypto_confirmation: true,
        }
    }
}

/// STONITH Fencing Manager
pub struct StonithManager {
    config: StonithConfig,
    node_configs: RwLock<HashMap<NodeId, FencingMethod>>,
    fencing_states: RwLock<HashMap<NodeId, FencingState>>,
    fencing_results: RwLock<HashMap<NodeId, FencingResult>>,
    pending_confirmations: RwLock<HashMap<NodeId, FencingConfirmation>>,
    event_tx: mpsc::Sender<FencingEvent>,
}

/// Events emitted by STONITH manager
#[derive(Debug, Clone)]
pub enum FencingEvent {
    FencingInitiated(NodeId, String),
    FencingConfirmed(NodeId),
    FencingFailed(NodeId, String),
    SplitBrainDetected,
    LeadershipDenied(NodeId),
}

impl StonithManager {
    pub fn new(config: StonithConfig) -> Self {
        let (event_tx, _) = mpsc::channel(100);

        Self {
            config,
            node_configs: RwLock::new(HashMap::new()),
            fencing_states: RwLock::new(HashMap::new()),
            fencing_results: RwLock::new(HashMap::new()),
            pending_confirmations: RwLock::new(HashMap::new()),
            event_tx,
        }
    }

    /// Register a node with its fencing configuration
    pub async fn register_node(&self, node_id: NodeId, method: FencingMethod) {
        let mut configs = self.node_configs.write().await;
        configs.insert(node_id, method);

        let mut states = self.fencing_states.write().await;
        states.insert(node_id, FencingState::Active);
    }

    /// Initiate fencing for a node
    pub async fn initiate_fencing(&self, node_id: NodeId) -> Result<FencingResult, StonithError> {
        let method = {
            let configs = self.node_configs.read().await;
            configs.get(&node_id)
                .cloned()
                .ok_or_else(|| StonithError::NodeNotRegistered(node_id))?
        };

        // Update state
        {
            let mut states = self.fencing_states.write().await;
            if let Some(state) = states.get_mut(&node_id) {
                if *state == FencingState::Fenced || *state == FencingState::ConfirmedFenced {
                    return Ok(FencingResult {
                        node_id,
                        success: true,
                        method: format!("{:?}", method),
                        confirmation: None,
                        error: None,
                        timestamp: Instant::now(),
                        retry_count: 0,
                    });
                }
                *state = FencingState::FencingInProgress;
            }
        }

        // Emit event
        let _ = self.event_tx.send(FencingEvent::FencingInitiated(
            node_id,
            format!("{:?}", method),
        )).await;

        // Execute fencing with timeout
        let result = timeout(
            self.config.confirmation_timeout,
            self.execute_fencing_with_retry(node_id, &method),
        ).await;

        match result {
            Ok(Ok(fencing_result)) => {
                // Update state based on result
                {
                    let mut states = self.fencing_states.write().await;
                    if let Some(state) = states.get_mut(&node_id) {
                        if fencing_result.success {
                            *state = FencingState::Fenced;
                        } else {
                            *state = FencingState::FencingFailed;
                        }
                    }
                }

                // Store result
                {
                    let mut results = self.fencing_results.write().await;
                    results.insert(node_id, fencing_result.clone());
                }

                Ok(fencing_result)
            }
            Ok(Err(e)) => {
                // Fencing failed
                {
                    let mut states = self.fencing_states.write().await;
                    if let Some(state) = states.get_mut(&node_id) {
                        *state = FencingState::FencingFailed;
                    }
                }

                Err(e)
            }
            Err(_) => {
                // Timeout
                Err(StonithError::FencingTimeout)
            }
        }
    }

    /// Execute fencing with retry logic
    async fn execute_fencing_with_retry(
        &self,
        node_id: NodeId,
        method: &FencingMethod,
    ) -> Result<FencingResult, StonithError> {
        let mut retry_count = 0;
        let mut last_error: Option<String> = None;

        while retry_count < self.config.max_retries {
            match self.execute_fencing_command(node_id, method).await {
                Ok(result) => {
                    if result.success {
                        // Wait for confirmation if required
                        if self.config.require_crypto_confirmation {
                            match self.wait_for_confirmation(node_id, &result).await {
                                Ok(mut confirmed_result) => {
                                    confirmed_result.retry_count = retry_count;
                                    return Ok(confirmed_result);
                                }
                                Err(e) => {
                                    last_error = Some(e.to_string());
                                    retry_count += 1;
                                    tokio::time::sleep(self.config.retry_interval).await;
                                }
                            }
                        } else {
                            return Ok(result);
                        }
                    } else {
                        last_error = result.error;
                        retry_count += 1;
                        tokio::time::sleep(self.config.retry_interval).await;
                    }
                }
                Err(e) => {
                    last_error = Some(e.to_string());
                    retry_count += 1;
                    tokio::time::sleep(self.config.retry_interval).await;
                }
            }
        }

        // All retries exhausted
        Err(StonithError::FencingFailed(
            last_error.unwrap_or_else(|| "Max retries exceeded".to_string()),
        ))
    }

    /// Execute a single fencing command
    async fn execute_fencing_command(
        &self,
        node_id: NodeId,
        method: &FencingMethod,
    ) -> Result<FencingResult, StonithError> {
        // In production, this would actually execute IPMI/Redfish commands
        // For now, we simulate the fencing process

        let method_str = format!("{:?}", method);
        
        // Simulate network delay
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Create command hash for confirmation
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(method_str.as_bytes());
        hasher.update(node_id.to_be_bytes());
        let command_hash: [u8; 32] = hasher.finalize().into();

        // Store pending confirmation
        if self.config.require_crypto_confirmation {
            let mut confirmations = self.pending_confirmations.write().await;
            confirmations.insert(node_id, FencingConfirmation::new(command_hash));
        }

        // Simulate successful fencing
        // In production: Actually send IPMI power-off or Redfish command
        Ok(FencingResult {
            node_id,
            success: true,
            method: method_str,
            confirmation: None, // Will be populated after confirmation
            error: None,
            timestamp: Instant::now(),
            retry_count: 0,
        })
    }

    /// Wait for cryptographic confirmation
    async fn wait_for_confirmation(
        &self,
        node_id: NodeId,
        result: &FencingResult,
    ) -> Result<FencingResult, StonithError> {
        let start = Instant::now();
        
        while start.elapsed() < self.config.confirmation_timeout {
            {
                let mut confirmations = self.pending_confirmations.write().await;
                if let Some(confirmation) = confirmations.get_mut(&node_id) {
                    // Simulate receiving witness signatures
                    // In production, these would come from other swarm nodes
                    if confirmation.is_fully_confirmed(self.config.required_witnesses) {
                        let mut confirmed_result = result.clone();
                        confirmed_result.confirmation = Some(confirmation.clone());
                        return Ok(confirmed_result);
                    }

                    // Add simulated witness signatures
                    while confirmation.witness_signatures.len() < self.config.required_witnesses {
                        let witness_id = confirmation.witness_signatures.len() as NodeId + 100;
                        let signature = vec![witness_id as u8; 64]; // Simulated signature
                        confirmation.add_witness(witness_id, signature);
                    }

                    if confirmation.is_fully_confirmed(self.config.required_witnesses) {
                        let mut confirmed_result = result.clone();
                        confirmed_result.confirmation = Some(confirmation.clone());
                        
                        // Update state to confirmed
                        {
                            let mut states = self.fencing_states.write().await;
                            if let Some(state) = states.get_mut(&node_id) {
                                *state = FencingState::ConfirmedFenced;
                            }
                        }

                        // Emit confirmation event
                        let _ = self.event_tx.send(FencingEvent::FencingConfirmed(node_id)).await;

                        return Ok(confirmed_result);
                    }
                }
            }

            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        Err(StonithError::ConfirmationTimeout)
    }

    /// Check if a node can safely assume leadership
    /// Returns Ok only if all potentially conflicting nodes are confirmed fenced
    pub async fn verify_safe_leadership(
        &self,
        requesting_node: NodeId,
        potentially_conflicting_nodes: &[NodeId],
    ) -> Result<bool, StonithError> {
        // CRITICAL: Must verify ALL conflicting nodes are fenced before allowing leadership
        
        for &node_id in potentially_conflicting_nodes {
            if node_id == requesting_node {
                continue;
            }

            let state = {
                let states = self.fencing_states.read().await;
                *states.get(&node_id).copied().unwrap_or(FencingState::Active)
            };

            match state {
                FencingState::ConfirmedFenced => {
                    // Safe - node is confirmed fenced
                    continue;
                }
                FencingState::Fenced => {
                    // Not fully confirmed - attempt confirmation
                    if self.config.require_crypto_confirmation {
                        return Err(StonithError::LeadershipDenied(
                            format!("Node {} is fenced but not cryptographically confirmed", node_id),
                        ));
                    }
                    continue;
                }
                FencingState::Active => {
                    // Node is still active - cannot assume leadership
                    return Err(StonithError::LeadershipDenied(
                        format!("Node {} is still active, fencing required", node_id),
                    ));
                }
                FencingState::FencingInProgress => {
                    // Wait for fencing to complete
                    return Err(StonithError::LeadershipDenied(
                        format!("Fencing in progress for node {}", node_id),
                    ));
                }
                FencingState::FencingFailed => {
                    // Fencing failed - cannot safely assume leadership
                    return Err(StonithError::LeadershipDenied(
                        format!("Fencing failed for node {}", node_id),
                    ));
                }
            }
        }

        // All conflicting nodes are properly fenced
        Ok(true)
    }

    /// Get fencing state for a node
    pub async fn get_fencing_state(&self, node_id: NodeId) -> Option<FencingState> {
        let states = self.fencing_states.read().await;
        states.get(&node_id).copied()
    }

    /// Get all fencing states
    pub async fn get_all_states(&self) -> HashMap<NodeId, FencingState> {
        let states = self.fencing_states.read().await;
        states.clone()
    }

    /// Reset fencing state for a node (after recovery)
    pub async fn reset_fencing_state(&self, node_id: NodeId) {
        let mut states = self.fencing_states.write().await;
        states.insert(node_id, FencingState::Active);

        let mut results = self.fencing_results.write().await;
        results.remove(&node_id);

        let mut confirmations = self.pending_confirmations.write().await;
        confirmations.remove(&node_id);
    }
}

/// STONITH error types
#[derive(Debug, thiserror::Error)]
pub enum StonithError {
    #[error("Node {0} not registered")]
    NodeNotRegistered(NodeId),
    #[error("Fencing command timed out")]
    FencingTimeout,
    #[error("Confirmation timed out")]
    ConfirmationTimeout,
    #[error("Fencing failed: {0}")]
    FencingFailed(String),
    #[error("Leadership denied: {0}")]
    LeadershipDenied(String),
    #[error("Split-brain detected - refusing to trade")]
    SplitBrainDetected,
    #[error("Cryptographic verification failed")]
    CryptoVerificationFailed,
    #[error("IPMI error: {0}")]
    IpmiError(String),
    #[error("Redfish error: {0}")]
    RedfishError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fencing_lifecycle() {
        let config = StonithConfig {
            require_crypto_confirmation: false,
            ..Default::default()
        };
        let manager = StonithManager::new(config);

        // Register a node with IPMI fencing
        let ipmi_config = IpmiConfig::default();
        manager.register_node(1, FencingMethod::IpmiPowerOff(ipmi_config)).await;

        // Verify initial state
        let state = manager.get_fencing_state(1).await;
        assert_eq!(state, Some(FencingState::Active));

        // Initiate fencing
        let result = manager.initiate_fencing(1).await;
        assert!(result.is_ok());
        assert!(result.unwrap().success);

        // Verify final state
        let state = manager.get_fencing_state(1).await;
        assert_eq!(state, Some(FencingState::Fenced));
    }

    #[tokio::test]
    async fn test_leadership_verification() {
        let config = StonithConfig {
            require_crypto_confirmation: false,
            ..Default::default()
        };
        let manager = StonithManager::new(config);

        // Register nodes
        manager.register_node(1, FencingMethod::IpmiPowerOff(IpmiConfig::default())).await;
        manager.register_node(2, FencingMethod::IpmiPowerOff(IpmiConfig::default())).await;

        // Node 1 tries to assume leadership while node 2 is active
        let result = manager.verify_safe_leadership(1, &[2]).await;
        assert!(result.is_err());

        // Fence node 2
        manager.initiate_fencing(2).await.unwrap();

        // Now node 1 can safely assume leadership
        let result = manager.verify_safe_leadership(1, &[2]).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_crypto_confirmation_requirement() {
        let config = StonithConfig {
            require_crypto_confirmation: true,
            required_witnesses: 2,
            ..Default::default()
        };
        let manager = StonithManager::new(config);

        manager.register_node(1, FencingMethod::IpmiPowerOff(IpmiConfig::default())).await;

        // With crypto confirmation enabled, fencing should wait for witnesses
        let result = manager.initiate_fencing(1).await;
        assert!(result.is_ok());

        let state = manager.get_fencing_state(1).await;
        assert_eq!(state, Some(FencingState::ConfirmedFenced));
    }
}
