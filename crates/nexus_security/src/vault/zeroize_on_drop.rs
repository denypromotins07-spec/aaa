//! Secure Zeroing on Drop
//! Uses the zeroize crate to cryptographically erase secrets from memory

use zeroize::{Zeroize, ZeroizeOnDrop};
use std::sync::atomic::{AtomicBool, Ordering};

/// SecureBuffer - A buffer that automatically zeroes its contents on drop
/// 
/// This prevents cold-boot attacks and core-dump leaks by ensuring
/// cryptographic material is erased before memory is freed.
#[derive(ZeroizeOnDrop)]
pub struct SecureBuffer {
    /// The actual data storage
    data: Vec<u8>,
    /// Flag to prevent double-zeroize (defensive)
    zeroized: AtomicBool,
    /// Whether this buffer has been pinned (for audit)
    pinned: bool,
}

impl SecureBuffer {
    /// Create a new SecureBuffer with the given capacity
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
            zeroized: AtomicBool::new(false),
            pinned: false,
        }
    }

    /// Create a new SecureBuffer initialized with data
    pub fn from_slice(data: &[u8]) -> Self {
        let mut buffer = Self::new(data.len());
        buffer.copy_from_slice(data);
        buffer
    }

    /// Get the length of the buffer
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Copy data into the buffer
    pub fn copy_from_slice(&mut self, src: &[u8]) {
        assert_eq!(self.data.len(), src.len(), "Length mismatch");
        self.data.copy_from_slice(src);
        self.zeroized.store(false, Ordering::Relaxed);
    }

    /// Get immutable reference to data
    pub fn as_ref(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable reference to data
    pub fn as_mut(&mut self) -> &mut [u8] {
        self.zeroized.store(false, Ordering::Relaxed);
        &mut self.data
    }

    /// Explicitly zeroize the buffer before drop
    pub fn zeroize(&mut self) {
        if self.zeroized.load(Ordering::Relaxed) {
            return; // Already zeroized
        }

        // Use zeroize trait for secure erasure
        self.data.zeroize();
        self.zeroized.store(true, Ordering::Relaxed);
    }

    /// Mark buffer as pinned (for audit tracking)
    pub fn mark_pinned(&mut self) {
        self.pinned = true;
    }

    /// Check if buffer is pinned
    pub fn is_pinned(&self) -> bool {
        self.pinned
    }

    /// Check if buffer has been zeroized
    pub fn is_zeroized(&self) -> bool {
        self.zeroized.load(Ordering::Relaxed)
    }

    /// Fill with random data (useful for testing)
    #[cfg(test)]
    pub fn fill_random(&mut self) {
        use rand::RngCore;
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut self.data);
        self.zeroized.store(false, Ordering::Relaxed);
    }
}

impl Clone for SecureBuffer {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            zeroized: AtomicBool::new(false),
            pinned: self.pinned,
        }
    }
}

impl AsRef<[u8]> for SecureBuffer {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

impl AsMut<[u8]> for SecureBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

impl Drop for SecureBuffer {
    fn drop(&mut self) {
        // Ensure zeroization happens even if not explicitly called
        if !self.zeroized.load(Ordering::Relaxed) {
            self.data.zeroize();
        }
    }
}

/// SecretKey - A specialized SecureBuffer for cryptographic keys
/// 
/// Provides additional type safety and can be used where a specific
/// key type is needed rather than a generic buffer.
#[derive(ZeroizeOnDrop)]
pub struct SecretKey {
    inner: SecureBuffer,
}

impl SecretKey {
    /// Create a new SecretKey
    pub fn new(key_data: &[u8]) -> Self {
        Self {
            inner: SecureBuffer::from_slice(key_data),
        }
    }

    /// Get reference to underlying buffer
    pub fn as_buffer(&self) -> &SecureBuffer {
        &self.inner
    }

    /// Get key bytes
    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_ref()
    }

    /// Get key length
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if key is empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Drop for SecretKey {
    fn drop(&mut self) {
        // Inner SecureBuffer will be zeroized by its own Drop
    }
}

/// HMACKeys - Pair of API key and HMAC secret
pub struct HMACTokenPair {
    pub api_key: SecureBuffer,
    pub hmac_secret: SecureBuffer,
}

impl HMACTokenPair {
    pub fn new(api_key: &[u8], hmac_secret: &[u8]) -> Self {
        Self {
            api_key: SecureBuffer::from_slice(api_key),
            hmac_secret: SecureBuffer::from_slice(hmac_secret),
        }
    }

    /// Zeroize both keys
    pub fn zeroize_all(&mut self) {
        self.api_key.zeroize();
        self.hmac_secret.zeroize();
    }
}

impl Drop for HMACTokenPair {
    fn drop(&mut self) {
        self.zeroize_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_buffer_zeroize_on_drop() {
        let data = b"secret_api_key_12345";
        let mut buffer = SecureBuffer::from_slice(data);
        
        // Verify data is present
        assert_eq!(buffer.as_ref(), data);
        assert!(!buffer.is_zeroized());
        
        // Explicitly zeroize
        buffer.zeroize();
        assert!(buffer.is_zeroized());
        
        // After zeroize, data should be all zeros
        assert!(buffer.as_ref().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_secure_buffer_clone() {
        let data = b"test_secret";
        let buffer = SecureBuffer::from_slice(data);
        let cloned = buffer.clone();
        
        assert_eq!(buffer.as_ref(), cloned.as_ref());
    }

    #[test]
    fn test_secret_key() {
        let key_data = b"hmac_secret_key_32bytes_long!!";
        let key = SecretKey::new(key_data);
        
        assert_eq!(key.as_bytes(), key_data);
        assert_eq!(key.len(), key_data.len());
    }

    #[test]
    fn test_hmac_token_pair() {
        let api_key = b"api_key_123";
        let hmac_secret = b"hmac_secret_456";
        
        let pair = HMACTokenPair::new(api_key, hmac_secret);
        
        assert_eq!(pair.api_key.as_ref(), api_key);
        assert_eq!(pair.hmac_secret.as_ref(), hmac_secret);
    }
}
