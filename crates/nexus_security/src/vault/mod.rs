//! Security Vault Module - Secure secret management with OS-level protection

pub mod secure_vault_loader;
pub mod mlock_memory_pinner;
pub mod zeroize_on_drop;

pub use secure_vault_loader::{SecureVaultLoader, VaultError};
pub use mlock_memory_pinner::{MemoryPinner, SecurityLevel};
pub use zeroize_on_drop::{SecureBuffer, SecretKey, HMACTokenPair};
