//! Hash Time-Locked Contract (HTLC) Atomic Swap Implementation
//! Ensures atomic execution across chains - if one leg fails, the other refunds

use thiserror::Error;
use alloc::vec::Vec;
use sha2::{Sha256, Digest};

#[derive(Error, Debug)]
pub enum HtlcError {
    #[error("Invalid secret length")]
    InvalidSecretLength,
    #[error("Hash mismatch")]
    HashMismatch,
    #[error("Timeout expired")]
    TimeoutExpired,
    #[error("Timeout too short")]
    TimeoutTooShort,
    #[error("Chain not supported")]
    ChainNotSupported,
}

pub type Result<T> = core::result::Result<T, HtlcError>;

/// Supported blockchain networks for cross-chain swaps
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chain {
    Ethereum,
    Solana,
    Arbitrum,
    Optimism,
    Polygon,
}

impl Chain {
    /// Get typical block time in seconds for timeout calculations
    pub const fn block_time_seconds(self) -> u64 {
        match self {
            Chain::Ethereum => 12,
            Chain::Solana => 1, // ~400ms but use conservative estimate
            Chain::Arbitrum => 1,
            Chain::Optimism => 2,
            Chain::Polygon => 2,
        }
    }
}

/// HTLC State Machine states
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HtlcState {
    Initiated,
    SecretRevealed,
    Claimed,
    Refunded,
    Expired,
}

/// HTLC Contract Parameters
#[derive(Clone, Debug)]
pub struct HtlcParams {
    /// Hash of the secret (SHA256)
    pub hash_lock: [u8; 32],
    /// Unix timestamp when swap expires
    pub time_lock: u64,
    /// Source chain
    pub source_chain: Chain,
    /// Destination chain
    pub dest_chain: Chain,
    /// Amount on source chain (in smallest unit)
    pub source_amount: u64,
    /// Expected amount on destination chain
    pub dest_amount: u64,
    /// Source asset address/token
    pub source_asset: [u8; 32],
    /// Destination asset address/token
    pub dest_asset: [u8; 32],
}

/// Complete HTLC Atomic Swap state
#[derive(Clone, Debug)]
pub struct HtlcAtomicSwap {
    /// Unique swap identifier
    pub swap_id: [u8; 32],
    /// The secret preimage (known only to initiator until claimed)
    secret: Option<[u8; 32]>,
    /// Current state of the swap
    pub state: HtlcState,
    /// Contract parameters
    pub params: HtlcParams,
    /// Transaction hash on source chain
    pub source_tx_hash: Option<[u8; 32]>,
    /// Transaction hash on destination chain
    pub dest_tx_hash: Option<[u8; 32]>,
    /// Claim transaction hash
    pub claim_tx_hash: Option<[u8; 32]>,
}

impl HtlcAtomicSwap {
    /// Create a new HTLC swap with a random secret
    pub fn new(
        source_chain: Chain,
        dest_chain: Chain,
        source_amount: u64,
        dest_amount: u64,
        source_asset: [u8; 32],
        dest_asset: [u8; 32],
        timeout_seconds: u64,
    ) -> Result<Self> {
        // Generate random secret (in production, use CSPRNG)
        let secret = Self::generate_secret();
        let hash_lock = Self::hash_secret(&secret);
        
        // Validate timeout is reasonable
        let min_timeout = source_chain.block_time_seconds()
            .max(dest_chain.block_time_seconds()) * 10;
        
        if timeout_seconds < min_timeout {
            return Err(HtlcError::TimeoutTooShort);
        }

        let time_lock = current_timestamp() + timeout_seconds;
        let swap_id = generate_swap_id(&hash_lock, source_chain, dest_chain);

        Ok(Self {
            swap_id,
            secret: Some(secret),
            state: HtlcState::Initiated,
            params: HtlcParams {
                hash_lock,
                time_lock,
                source_chain,
                dest_chain,
                source_amount,
                dest_amount,
                source_asset,
                dest_asset,
            },
            source_tx_hash: None,
            dest_tx_hash: None,
            claim_tx_hash: None,
        })
    }

    /// Create from existing parameters (for tracking on-chain swaps)
    pub fn from_params(params: HtlcParams, swap_id: [u8; 32]) -> Self {
        Self {
            swap_id,
            secret: None,
            state: HtlcState::Initiated,
            params,
            source_tx_hash: None,
            dest_tx_hash: None,
            claim_tx_hash: None,
        }
    }

    /// Get the hash lock
    pub const fn hash_lock(&self) -> &[u8; 32] {
        &self.params.hash_lock
    }

    /// Get the secret (if known)
    pub fn secret(&self) -> Option<&[u8; 32]> {
        self.secret.as_ref()
    }

    /// Set the secret (when learned from counterparty's claim)
    pub fn set_secret(&mut self, secret: [u8; 32]) -> Result<()> {
        if Self::hash_secret(&secret) != self.params.hash_lock {
            return Err(HtlcError::HashMismatch);
        }
        self.secret = Some(secret);
        self.state = HtlcState::SecretRevealed;
        Ok(())
    }

    /// Verify a claimed secret
    pub fn verify_secret(&self, secret: &[u8; 32]) -> bool {
        Self::hash_secret(secret) == self.params.hash_lock
    }

    /// Check if swap has expired
    pub fn is_expired(&self) -> bool {
        current_timestamp() >= self.params.time_lock
    }

    /// Get remaining time in seconds
    pub fn remaining_time(&self) -> u64 {
        let now = current_timestamp();
        if now >= self.params.time_lock {
            0
        } else {
            self.params.time_lock - now
        }
    }

    /// Update state based on on-chain events
    pub fn on_claim(&mut self, secret: [u8; 32], tx_hash: [u8; 32]) -> Result<()> {
        if !self.verify_secret(&secret) {
            return Err(HtlcError::HashMismatch);
        }
        
        self.secret = Some(secret);
        self.state = HtlcState::Claimed;
        self.claim_tx_hash = Some(tx_hash);
        Ok(())
    }

    /// Mark as refunded
    pub fn on_refund(&mut self, tx_hash: [u8; 32]) -> Result<()> {
        if !self.is_expired() && self.state != HtlcState::Expired {
            // Can only refund after expiry unless counterparty agrees
            return Err(HtlcError::TimeoutExpired);
        }
        
        self.state = HtlcState::Refunded;
        self.source_tx_hash = Some(tx_hash);
        Ok(())
    }

    /// Record source chain transaction
    pub fn record_source_tx(&mut self, tx_hash: [u8; 32]) {
        self.source_tx_hash = Some(tx_hash);
    }

    /// Record destination chain transaction
    pub fn record_dest_tx(&mut self, tx_hash: [u8; 32]) {
        self.dest_tx_hash = Some(tx_hash);
    }

    /// Generate a random secret
    fn generate_secret() -> [u8; 32] {
        // In production, use proper CSPRNG
        // For now, use a placeholder that would be replaced
        let mut secret = [0u8; 32];
        for i in 0..32 {
            secret[i] = (i as u8).wrapping_add(current_timestamp() as u8);
        }
        secret
    }

    /// Hash a secret using SHA256
    fn hash_secret(secret: &[u8; 32]) -> [u8; 32] {
        Sha256::digest(secret).into()
    }

    /// Check if it's safe to initiate the second leg
    pub fn safe_to_initiate_second_leg(&self) -> bool {
        // Must have sufficient time remaining for the second chain to confirm
        let remaining_blocks = self.remaining_time() / self.params.dest_chain.block_time_seconds();
        remaining_blocks >= 6 // Wait for 6 confirmations worth of buffer
    }

    /// Calculate expected profit/loss
    pub fn expected_pnl(&self, current_rate: f64) -> f64 {
        let source_value = self.params.source_amount as f64;
        let dest_value = self.params.dest_amount as f64 * current_rate;
        dest_value - source_value
    }
}

/// Generate unique swap ID
fn generate_swap_id(hash_lock: &[u8; 32], source: Chain, dest: Chain) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(hash_lock);
    hasher.update(&(source as u8));
    hasher.update(&(dest as u8));
    hasher.finalize().into()
}

/// Get current Unix timestamp
fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_htlc() {
        let swap = HtlcAtomicSwap::new(
            Chain::Ethereum,
            Chain::Solana,
            1_000_000_000, // 1 ETH in wei
            100_000_000,   // SOL in lamports
            [1u8; 32],
            [2u8; 32],
            3600, // 1 hour timeout
        );
        
        assert!(swap.is_ok());
        let swap = swap.unwrap();
        assert_eq!(swap.state, HtlcState::Initiated);
        assert!(swap.secret().is_some());
    }

    #[test]
    fn test_timeout_too_short() {
        let result = HtlcAtomicSwap::new(
            Chain::Ethereum,
            Chain::Solana,
            1_000_000_000,
            100_000_000,
            [1u8; 32],
            [2u8; 32],
            5, // Too short
        );
        
        assert!(matches!(result, Err(HtlcError::TimeoutTooShort)));
    }

    #[test]
    fn test_secret_verification() {
        let mut swap = HtlcAtomicSwap::new(
            Chain::Ethereum,
            Chain::Arbitrum,
            1_000_000_000,
            1_000_000_000,
            [1u8; 32],
            [1u8; 32],
            3600,
        ).unwrap();
        
        let secret = swap.secret().copied().unwrap();
        assert!(swap.verify_secret(&secret));
        
        let wrong_secret = [0u8; 32];
        assert!(!swap.verify_secret(&wrong_secret));
    }

    #[test]
    fn test_secret_setting() {
        let mut swap = HtlcAtomicSwap::new(
            Chain::Ethereum,
            Chain::Arbitrum,
            1_000_000_000,
            1_000_000_000,
            [1u8; 32],
            [1u8; 32],
            3600,
        ).unwrap();
        
        // Remove secret to simulate counterparty scenario
        swap.secret = None;
        
        // Try to set wrong secret
        let result = swap.set_secret([0u8; 32]);
        assert!(matches!(result, Err(HtlcError::HashMismatch)));
    }
}
