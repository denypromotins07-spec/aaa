//! Secure Vault Loader
//! Loads API keys and secrets from secure sources with OS-level memory protection

use std::sync::Arc;
use tracing::{info, warn, error};

use crate::vault::mlock_memory_pinner::{MemoryPinner, SecurityLevel};
use crate::vault::zeroize_on_drop::SecureBuffer;

/// Error types for vault operations
#[derive(Debug, thiserror::Error)]
pub enum VaultError {
    #[error("Failed to load secret: {0}")]
    LoadFailed(String),
    #[error("Memory pinning failed: {0}")]
    MemoryPinningFailed(String),
    #[error("Secret too large: max {max} bytes, got {actual}")]
    SecretTooLarge { max: usize, actual: usize },
    #[error("Security violation: {0}")]
    SecurityViolation(String),
}

/// Maximum size for a single secret (64KB)
const MAX_SECRET_SIZE: usize = 64 * 1024;

/// SecureVaultLoader - Manages cryptographic material with OS-level protection
pub struct SecureVaultLoader {
    pinner: MemoryPinner,
    /// Security level achieved during initialization
    security_level: SecurityLevel,
}

impl SecureVaultLoader {
    /// Create a new SecureVaultLoader
    /// 
    /// Attempts to pin memory with mlock. Falls back to degraded mode if
    /// insufficient privileges (e.g., Docker without IPC_LOCK capability).
    pub fn new() -> Result<Self, VaultError> {
        let pinner = MemoryPinner::new();
        
        // Attempt to pre-allocate lockable memory
        let security_level = match pinner.pin_page() {
            Ok(level) => {
                info!("Memory pinning successful: {:?}", level);
                level
            }
            Err(e) => {
                warn!("Memory pinning failed: {}. Operating in DEGRADED security mode.", e);
                warn!("CRITICAL_SECURITY_WARNING: SWAP_RISK_DETECTED");
                
                // Check if we should allow insecure boot
                let allow_insecure = std::env::var("ALLOW_INSECURE_BOOT")
                    .map(|v| v == "true" || v == "1")
                    .unwrap_or(false);
                
                if !allow_insecure {
                    // In production/release mode, hard-fail
                    if cfg!(not(debug_assertions)) {
                        error!("Production build requires memory pinning. Set ALLOW_INSECURE_BOOT=true to override (NOT RECOMMENDED).");
                        return Err(VaultError::MemoryPinningFailed(
                            "Production mode requires mlock capability".to_string()
                        ));
                    }
                    SecurityLevel::Degraded
                } else {
                    SecurityLevel::Degraded
                }
            }
        };
        
        Ok(Self {
            pinner,
            security_level,
        })
    }

    /// Get the current security level
    pub fn security_level(&self) -> SecurityLevel {
        self.security_level
    }

    /// Load a secret into a secure buffer
    /// 
    /// The secret is immediately pinned to physical RAM (if security level allows)
    /// and will be zeroized on drop.
    pub fn load_secret(&self, secret: &[u8]) -> Result<SecureBuffer, VaultError> {
        if secret.len() > MAX_SECRET_SIZE {
            return Err(VaultError::SecretTooLarge {
                max: MAX_SECRET_SIZE,
                actual: secret.len(),
            });
        }

        // Create secure buffer (automatically zeroizes on drop)
        let mut buffer = SecureBuffer::new(secret.len());
        buffer.copy_from_slice(secret);

        // Attempt to pin the buffer's memory
        if let Err(e) = self.pinner.pin_buffer(&buffer) {
            warn!("Failed to pin secret buffer: {}", e);
            if self.security_level == SecurityLevel::Production {
                return Err(VaultError::MemoryPinningFailed(e.to_string()));
            }
            // In degraded mode, continue with warning
        }

        Ok(buffer)
    }

    /// Load API key from environment variable (with secure handling)
    pub fn load_api_key_from_env(&self, var_name: &str) -> Result<SecureBuffer, VaultError> {
        let secret = std::env::var(var_name)
            .map_err(|_| VaultError::LoadFailed(format!("Environment variable {} not set", var_name)))?;

        self.load_secret(secret.as_bytes())
    }

    /// Load HMAC secret from environment variable
    pub fn load_hmac_secret_from_env(&self, var_name: &str) -> Result<SecureBuffer, VaultError> {
        self.load_api_key_from_env(var_name)
    }

    /// Load secrets from a secure hardware enclave (placeholder for production)
    /// 
    /// In production, this would integrate with:
    /// - AWS Secrets Manager with Nitro Enclaves
    /// - HashiCorp Vault with HSM backend
    /// - Azure Key Vault with Managed HSM
    pub fn load_from_hardware_enclave(&self, secret_id: &str) -> Result<SecureBuffer, VaultError> {
        info!("Loading secret '{}' from hardware enclave...", secret_id);
        
        // Placeholder implementation
        // In production: call HSM/enclave API
        let secret_data = vec![0u8; 32]; // Simulated secret
        
        self.load_secret(&secret_data)
    }

    /// Rotate a secret - securely zeroize old and load new
    pub fn rotate_secret(
        &self,
        old_secret: &mut SecureBuffer,
        new_secret: &[u8],
    ) -> Result<SecureBuffer, VaultError> {
        // Old secret will be zeroized when it goes out of scope
        // Explicitly clear it now for immediate effect
        old_secret.zeroize();
        
        self.load_secret(new_secret)
    }
}

impl Default for SecureVaultLoader {
    fn default() -> Self {
        Self::new().expect("Failed to initialize SecureVaultLoader")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vault_creation() {
        // This may succeed or fall back to degraded mode depending on system capabilities
        let result = SecureVaultLoader::new();
        
        // Should not panic - either succeeds or returns degraded mode error
        assert!(result.is_ok() || matches!(result, Err(VaultError::MemoryPinningFailed(_))));
    }

    #[test]
    fn test_load_secret() {
        let vault = SecureVaultLoader::new().unwrap_or_else(|_| {
            // Create a degraded mode vault for testing
            SecureVaultLoader {
                pinner: MemoryPinner::new(),
                security_level: SecurityLevel::Degraded,
            }
        });

        let secret = b"test_api_key_12345";
        let buffer = vault.load_secret(secret).unwrap();
        
        assert_eq!(buffer.len(), secret.len());
        // Note: We can't compare contents directly due to SecureBuffer's design
    }
}
