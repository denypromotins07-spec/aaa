//! Flashbots Bundle Signer
//! 
//! Cryptographically packages transactions for direct submission to block builders.
//! Bypasses public mempool to prevent adversarial front-running.

use thiserror::Error;
use alloc::vec::Vec;
use sha2::{Sha256, Digest};
use secp256k1::{Secp256k1, SecretKey, PublicKey, Message};

#[derive(Error, Debug)]
pub enum BundleSignerError {
    #[error("Invalid private key")]
    InvalidPrivateKey,
    #[error("Bundle validation failed: {0}")]
    ValidationFailed(&'static str),
    #[error("Signature generation failed")]
    SignatureFailed,
    #[error("Recovery id calculation failed")]
    RecoveryIdFailed,
    #[error("Bundle too large: {size} > {max}"),
    BundleTooLarge { size: usize, max: usize },
    #[error("Invalid target block number")]
    InvalidBlockNumber,
}

pub type Result<T> = core::result::Result<T, BundleSignerError>;

/// Maximum bundle size (transactions)
const MAX_BUNDLE_SIZE: usize = 16;

/// Flashbots bundle format for submission
#[derive(Clone, Debug)]
pub struct FlashbotsBundle {
    /// Bundle ID (hash of contents)
    pub bundle_id: [u8; 32],
    /// Encoded transactions (RLP serialized)
    pub transactions: Vec<Vec<u8>>,
    /// Target block number (or None for next block)
    pub target_block: Option<u64>,
    /// Minimum timestamp for execution
    pub min_timestamp: Option<u64>,
    /// Maximum timestamp for execution  
    pub max_timestamp: Option<u64>,
    /// Reverting transaction hashes allowed
    pub reverting_tx_hashes: Vec<[u8; 32]>,
    /// Signature over bundle
    pub signature: [u8; 65],
    /// Signer address (recovered from signature)
    pub signer: [u8; 20],
}

/// Bundle configuration options
#[derive(Clone, Debug)]
pub struct BundleConfig {
    pub target_block: Option<u64>,
    pub min_timestamp: Option<u64>,
    pub max_timestamp: Option<u64>,
    pub allow_reverting: bool,
}

impl Default for BundleConfig {
    fn default() -> Self {
        Self {
            target_block: None,
            min_timestamp: None,
            max_timestamp: None,
            allow_reverting: false,
        }
    }
}

/// Flashbots Bundle Signer
/// 
/// Signs and packages transaction bundles for private submission.
pub struct FlashbotsBundleSigner {
    secp: Secp256k1,
    private_key: SecretKey,
    public_key: PublicKey,
    address: [u8; 20],
}

impl FlashbotsBundleSigner {
    /// Create a new signer with the given private key
    pub fn new(private_key_bytes: [u8; 32]) -> Result<Self> {
        let secp = Secp256k1::new();
        
        let private_key = SecretKey::from_slice(&private_key_bytes)
            .map_err(|_| BundleSignerError::InvalidPrivateKey)?;
        
        let public_key = PublicKey::from_secret_key(&secp, &private_key);
        
        // Derive Ethereum address from public key
        let address = derive_address(&public_key);
        
        Ok(Self {
            secp,
            private_key,
            public_key,
            address,
        })
    }

    /// Get the signer's address
    pub const fn address(&self) -> &[u8; 20] {
        &self.address
    }

    /// Create and sign a bundle
    pub fn create_bundle(
        &self,
        transactions: Vec<Vec<u8>>,
        config: BundleConfig,
    ) -> Result<FlashbotsBundle> {
        // Validate bundle size
        if transactions.is_empty() {
            return Err(BundleSignerError::ValidationFailed("Bundle cannot be empty"));
        }
        if transactions.len() > MAX_BUNDLE_SIZE {
            return Err(BundleSignerError::BundleTooLarge {
                size: transactions.len(),
                max: MAX_BUNDLE_SIZE,
            });
        }

        // Validate target block if specified
        if let Some(block) = config.target_block {
            if block == 0 {
                return Err(BundleSignerError::InvalidBlockNumber);
            }
        }

        // Calculate bundle ID (hash of all transactions)
        let bundle_id = self.calculate_bundle_id(&transactions);

        // Collect reverting tx hashes if allowed
        let reverting_hashes = if config.allow_reverting {
            transactions.iter()
                .enumerate()
                .filter(|(i, _)| *i < transactions.len() - 1) // Allow all but last to revert
                .map(|(_, tx)| hash_transaction(tx))
                .collect()
        } else {
            Vec::new()
        };

        // Create message to sign
        let message = self.create_signing_message(&bundle_id, &config);
        let message_hash = Sha256::digest(&message);

        // Sign the message
        let signature = self.secp.sign_ecdsa_recoverable(
            &Message::from_digest_slice(&message_hash)
                .map_err(|_| BundleSignerError::SignatureFailed)?,
            &self.private_key,
        );

        // Convert to compact format with recovery id
        let (recovery_id, sig_bytes) = signature.serialize_compact();
        
        let mut full_signature = [0u8; 65];
        full_signature[0..64].copy_from_slice(&sig_bytes);
        full_signature[64] = recovery_id.to_i32() as u8 + 27;

        Ok(FlashbotsBundle {
            bundle_id,
            transactions,
            target_block: config.target_block,
            min_timestamp: config.min_timestamp,
            max_timestamp: config.max_timestamp,
            reverting_tx_hashes: reverting_hashes,
            signature: full_signature,
            signer: self.address,
        })
    }

    /// Calculate bundle ID as hash of concatenated transactions
    fn calculate_bundle_id(&self, transactions: &[Vec<u8>]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for tx in transactions {
            hasher.update(tx);
        }
        hasher.finalize().into()
    }

    /// Create the signing message per Flashbots spec
    fn create_signing_message(&self, bundle_id: &[u8; 32], config: &BundleConfig) -> Vec<u8> {
        let mut message = Vec::with_capacity(64);
        
        // Prefix with bundle ID
        message.extend_from_slice(bundle_id);
        
        // Add target block if specified
        if let Some(block) = config.target_block {
            message.extend_from_slice(&block.to_be_bytes());
        }
        
        // Add timestamps if specified
        if let Some(min_ts) = config.min_timestamp {
            message.extend_from_slice(&min_ts.to_be_bytes());
        }
        if let Some(max_ts) = config.max_timestamp {
            message.extend_from_slice(&max_ts.to_be_bytes());
        }
        
        message
    }

    /// Verify a bundle signature
    pub fn verify_bundle(&self, bundle: &FlashbotsBundle) -> Result<bool> {
        // Recreate signing message
        let config = BundleConfig {
            target_block: bundle.target_block,
            min_timestamp: bundle.min_timestamp,
            max_timestamp: bundle.max_timestamp,
            allow_reverting: !bundle.reverting_tx_hashes.is_empty(),
        };
        
        let message = self.create_signing_message(&bundle.bundle_id, &config);
        let message_hash = Sha256::digest(&message);

        // Extract signature components
        if bundle.signature.len() != 65 {
            return Ok(false);
        }

        let r = bundle.signature[0..32].to_vec();
        let s = bundle.signature[32..64].to_vec();
        let v = bundle.signature[64];

        // Recover public key and verify
        let recovered = secp256k1::ecdsa_recover(
            &bundle.signature[0..64],
            v.checked_sub(27).unwrap_or(0) as i32,
            &message_hash,
        );

        match recovered {
            Ok(recovered_key) => {
                let recovered_address = derive_address(&recovered_key);
                Ok(recovered_address == bundle.signer)
            }
            Err(_) => Ok(false),
        }
    }

    /// Encode bundle for JSON-RPC submission
    pub fn encode_for_submission(&self, bundle: &FlashbotsBundle) -> serde_json::Value {
        use serde_json::json;
        
        let txs_hex: Vec<String> = bundle.transactions.iter()
            .map(|tx| format!("0x{}", hex_encode(tx)))
            .collect();
        
        json!({
            "txs": txs_hex,
            "blockNumber": format!("0x{:x}", bundle.target_block.unwrap_or(0)),
            "minTimestamp": bundle.min_timestamp,
            "maxTimestamp": bundle.max_timestamp,
            "revertingTxHashes": bundle.reverting_tx_hashes.iter()
                .map(|h| format!("0x{}", hex_encode(h)))
                .collect::<Vec<_>>(),
        })
    }
}

/// Derive Ethereum address from public key (Keccak256 hash of uncompressed pubkey)
fn derive_address(public_key: &PublicKey) -> [u8; 20] {
    use sha3::{Keccak256, Digest};
    
    let serialized = public_key.serialize_uncompressed();
    // Skip first byte (0x04 prefix) and hash the rest
    let hash = Keccak256::digest(&serialized[1..]);
    
    // Take last 20 bytes as address
    let mut address = [0u8; 20];
    address.copy_from_slice(&hash[12..32]);
    address
}

/// Hash a transaction for identification
fn hash_transaction(tx: &[u8]) -> [u8; 32] {
    Sha256::digest(tx).into()
}

/// Simple hex encoder
fn hex_encode(data: &[u8]) -> String {
    const HEX_CHARS: &[u8] = b"0123456789abcdef";
    let mut result = String::with_capacity(data.len() * 2);
    
    for &byte in data {
        result.push(HEX_CHARS[(byte >> 4) as usize] as char);
        result.push(HEX_CHARS[(byte & 0x0F) as usize] as char);
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signer_creation() {
        let key_bytes = [0x42u8; 32];
        let signer = FlashbotsBundleSigner::new(key_bytes);
        assert!(signer.is_ok());
    }

    #[test]
    fn test_invalid_key() {
        // All zeros is not a valid secp256k1 private key
        let key_bytes = [0u8; 32];
        let signer = FlashbotsBundleSigner::new(key_bytes);
        assert!(matches!(signer, Err(BundleSignerError::InvalidPrivateKey)));
    }

    #[test]
    fn test_bundle_creation() {
        let key_bytes = [0x42u8; 32];
        let signer = FlashbotsBundleSigner::new(key_bytes).unwrap();
        
        let txs = vec![
            vec![0x01, 0x02, 0x03],
            vec![0x04, 0x05, 0x06],
        ];
        
        let config = BundleConfig {
            target_block: Some(18_000_000),
            ..Default::default()
        };
        
        let bundle = signer.create_bundle(txs, config);
        assert!(bundle.is_ok());
        
        let bundle = bundle.unwrap();
        assert_eq!(bundle.transactions.len(), 2);
        assert_eq!(bundle.target_block, Some(18_000_000));
        assert_eq!(bundle.signer, signer.address);
    }

    #[test]
    fn test_bundle_size_limit() {
        let key_bytes = [0x42u8; 32];
        let signer = FlashbotsBundleSigner::new(key_bytes).unwrap();
        
        let txs: Vec<Vec<u8>> = (0..MAX_BUNDLE_SIZE + 1)
            .map(|i| vec![i as u8])
            .collect();
        
        let result = signer.create_bundle(txs, BundleConfig::default());
        assert!(matches!(result, Err(BundleSignerError::BundleTooLarge { .. })));
    }

    #[test]
    fn test_signature_verification() {
        let key_bytes = [0x42u8; 32];
        let signer = FlashbotsBundleSigner::new(key_bytes).unwrap();
        
        let txs = vec![vec![0x01, 0x02]];
        let bundle = signer.create_bundle(txs, BundleConfig::default()).unwrap();
        
        let verified = signer.verify_bundle(&bundle);
        assert!(verified.is_ok());
        assert!(verified.unwrap());
    }
}
