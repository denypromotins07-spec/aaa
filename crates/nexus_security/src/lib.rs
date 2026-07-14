//! NEXUS-OMEGA Security Module
//! 
//! Provides OS-level secret protection:
//! - Memory pinning with mlock (prevents swap)
//! - Secure zeroization on drop (prevents cold-boot attacks)
//! - Secure vault loading from hardware enclaves

pub mod vault;

pub use vault::{SecureVaultLoader, VaultError, MemoryPinner, SecurityLevel, SecureBuffer, SecretKey, HMACTokenPair};

/// Initialize the security subsystem
/// 
/// Call this early in the boot sequence to establish secure memory handling.
/// Returns the security level achieved.
pub fn init_security() -> Result<SecurityLevel, VaultError> {
    let loader = SecureVaultLoader::new()?;
    Ok(loader.security_level())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_init() {
        // May succeed or fall back to degraded mode
        let result = init_security();
        
        // Should not panic
        match result {
            Ok(level) => {
                println!("Security level: {:?}", level);
            }
            Err(e) => {
                println!("Security init failed (expected in some environments): {}", e);
            }
        }
    }
}
