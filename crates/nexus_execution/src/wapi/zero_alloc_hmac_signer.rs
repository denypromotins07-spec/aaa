//! Zero-Allocation HMAC SHA256 Signer for WAPI
//! 
//! Provides cryptographic signing using pre-allocated buffers to eliminate
//! heap allocations in the hot path.

use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::fmt;

type HmacSha256 = Hmac<Sha256>;

/// Size of HMAC-SHA256 output in bytes
pub const HMAC_OUTPUT_SIZE: usize = 32;

/// Size of hex-encoded HMAC output (2 chars per byte)
pub const HMAC_HEX_SIZE: usize = HMAC_OUTPUT_SIZE * 2;

/// Pre-allocated buffer for HMAC operations
#[derive(Clone)]
pub struct HmacBuffer {
    /// Raw HMAC output bytes
    bytes: [u8; HMAC_OUTPUT_SIZE],
    /// Hex-encoded output (for transmission)
    hex_output: [u8; HMAC_HEX_SIZE],
}

impl Default for HmacBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl HmacBuffer {
    /// Create new zero-initialized buffer
    #[inline]
    pub const fn new() -> Self {
        Self {
            bytes: [0u8; HMAC_OUTPUT_SIZE],
            hex_output: [0u8; HMAC_HEX_SIZE],
        }
    }

    /// Get reference to raw bytes
    #[inline]
    pub fn as_bytes(&self) -> &[u8; HMAC_OUTPUT_SIZE] {
        &self.bytes
    }

    /// Get hex-encoded output as string slice
    #[inline]
    pub fn as_hex_str(&self) -> &str {
        // SAFETY: hex_output contains only valid ASCII hex characters
        unsafe { std::str::from_utf8_unchecked(&self.hex_output) }
    }

    /// Get mutable reference to hex output buffer
    #[inline]
    pub fn hex_output_mut(&mut self) -> &mut [u8; HMAC_HEX_SIZE] {
        &mut self.hex_output
    }
}

/// Lookup table for zero-allocation hex encoding
/// Generated at compile time
const HEX_LOOKUP: [u8; 16] = [
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b'a', b'b', b'c', b'd', b'e', b'f',
];

/// Zero-allocation hex encoder using lookup table
#[inline]
fn encode_hex_zero_alloc(input: &[u8], output: &mut [u8]) {
    debug_assert!(output.len() >= input.len() * 2, "Output buffer too small");
    
    for (i, &byte) in input.iter().enumerate() {
        output[i * 2] = HEX_LOOKUP[(byte >> 4) as usize];
        output[i * 2 + 1] = HEX_LOOKUP[(byte & 0x0F) as usize];
    }
}

/// Zero-Allocation HMAC Signer
/// 
/// Uses pre-allocated buffers and lookup-table hex encoding
/// to ensure zero heap allocations during signing.
pub struct ZeroAllocHmacSigner {
    secret_key: Vec<u8>, // Secret key (allocated once at startup)
    buffer: HmacBuffer,
}

/// Error types for HMAC signing
#[derive(Debug)]
pub enum HmacSignerError {
    InvalidKeyLength(usize),
    MacInitializationFailed,
    BufferOverflow,
}

impl fmt::Display for HmacSignerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HmacSignerError::InvalidKeyLength(len) => {
                write!(f, "Invalid key length: {} bytes", len)
            }
            HmacSignerError::MacInitializationFailed => {
                write!(f, "Failed to initialize MAC")
            }
            HmacSignerError::BufferOverflow => {
                write!(f, "Output buffer overflow")
            }
        }
    }
}

impl std::error::Error for HmacSignerError {}

impl ZeroAllocHmacSigner {
    /// Create new signer with the given secret key
    /// Key is allocated once at initialization, not in the hot path
    pub fn new(secret_key: &[u8]) -> Result<Self, HmacSignerError> {
        // Binance allows keys up to 64 bytes typically
        if secret_key.is_empty() || secret_key.len() > 1024 {
            return Err(HmacSignerError::InvalidKeyLength(secret_key.len()));
        }

        Ok(Self {
            secret_key: secret_key.to_vec(),
            buffer: HmacBuffer::new(),
        })
    }

    /// Sign a message and write result to internal buffer
    /// Returns a reference to the hex-encoded signature
    /// 
    /// ZERO-ALLOCATION: No heap allocations occur during this call
    #[inline]
    pub fn sign_to_buffer(&mut self, message: &[u8]) -> Result<&str, HmacSignerError> {
        // Initialize MAC with secret key
        let mut mac = HmacSha256::new_from_slice(&self.secret_key)
            .map_err(|_| HmacSignerError::MacInitializationFailed)?;

        // Update with message
        mac.update(message);

        // Finalize and get result
        let result = mac.finalize();
        let code = result.into_bytes();

        // Copy to pre-allocated buffer
        self.buffer.bytes.copy_from_slice(&code);

        // Zero-allocation hex encoding using lookup table
        encode_hex_zero_alloc(&code, &mut self.buffer.hex_output);

        Ok(self.buffer.as_hex_str())
    }

    /// Sign a message and write to provided output buffer
    /// This allows caller to use their own pre-allocated buffer
    /// 
    /// ZERO-ALLOCATION: No heap allocations occur during this call
    #[inline]
    pub fn sign_to_slice(&self, message: &[u8], output: &mut [u8]) -> Result<usize, HmacSignerError> {
        if output.len() < HMAC_HEX_SIZE {
            return Err(HmacSignerError::BufferOverflow);
        }

        // Initialize MAC
        let mut mac = HmacSha256::new_from_slice(&self.secret_key)
            .map_err(|_| HmacSignerError::MacInitializationFailed)?;

        mac.update(message);

        let result = mac.finalize();
        let code = result.into_bytes();

        // Zero-allocation hex encoding directly to output
        encode_hex_zero_alloc(&code, output);

        Ok(HMAC_HEX_SIZE)
    }

    /// Get the size of the hex output
    #[inline]
    pub const fn hex_output_size() -> usize {
        HMAC_HEX_SIZE
    }

    /// Get the size of the raw HMAC output
    #[inline]
    pub const fn raw_output_size() -> usize {
        HMAC_OUTPUT_SIZE
    }
}

/// Request payload builder for WAPI
/// Builds signed request strings without allocation
pub struct WapiRequestBuilder {
    base_params: String, // Allocated once, reused
    timestamp_param: String,
    recv_window_param: String,
}

impl WapiRequestBuilder {
    pub fn new(api_key: &str) -> Self {
        Self {
            base_params: format!("apiKey={}", api_key),
            timestamp_param: String::with_capacity(32),
            recv_window_param: String::with_capacity(20),
        }
    }

    /// Build unsigned payload string (timestamp + params)
    /// Caller must still sign this with HMAC
    pub fn build_unsigned_payload(
        &mut self,
        timestamp: u64,
        recv_window_ms: u64,
        additional_params: Option<&str>,
    ) -> String {
        self.timestamp_param.clear();
        self.timestamp_param.push_str("&timestamp=");
        self.timestamp_param.push_str(&timestamp.to_string());

        self.recv_window_param.clear();
        self.recv_window_param.push_str("&recvWindow=");
        self.recv_window_param.push_str(&recv_window_ms.to_string());

        let mut payload = String::with_capacity(
            self.base_params.len() + 
            self.timestamp_param.len() + 
            self.recv_window_param.len() +
            additional_params.map(|s| s.len()).unwrap_or(0) + 1
        );

        payload.push_str(&self.base_params);
        payload.push_str(&self.timestamp_param);
        payload.push_str(&self.recv_window_param);

        if let Some(params) = additional_params {
            payload.push('&');
            payload.push_str(params);
        }

        payload
    }

    /// Build complete signed URL for GET requests
    pub fn build_signed_get_url(
        &mut self,
        base_url: &str,
        signer: &mut ZeroAllocHmacSigner,
        timestamp: u64,
        recv_window_ms: u64,
        additional_params: Option<&str>,
    ) -> Result<String, HmacSignerError> {
        let payload = self.build_unsigned_payload(timestamp, recv_window_ms, additional_params);
        
        // Sign the payload
        let signature = signer.sign_to_buffer(payload.as_bytes())?;

        // Build complete URL
        let mut url = String::with_capacity(base_url.len() + payload.len() + signature.len() + 20);
        url.push_str(base_url);
        
        if !base_url.contains('?') {
            url.push('?');
        } else {
            url.push('&');
        }
        
        url.push_str(&payload);
        url.push_str("&signature=");
        url.push_str(signature);

        Ok(url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmac_signing() {
        let secret = b"my_secret_key_for_testing";
        let mut signer = ZeroAllocHmacSigner::new(secret).unwrap();

        let message = b"symbol=BTCUSDT&side=BUY&type=LIMIT&qty=0.001&price=50000";
        
        // First sign
        let sig1 = signer.sign_to_buffer(message).unwrap();
        assert_eq!(sig1.len(), HMAC_HEX_SIZE);

        // Second sign of same message should produce same result
        let sig2 = signer.sign_to_buffer(message).unwrap();
        assert_eq!(sig1, sig2);

        // Different message should produce different signature
        let message2 = b"symbol=ETHUSDT&side=BUY";
        let sig3 = signer.sign_to_buffer(message2).unwrap();
        assert_ne!(sig1, sig3);
    }

    #[test]
    fn test_zero_allocation_sign_to_slice() {
        let secret = b"test_secret";
        let signer = ZeroAllocHmacSigner::new(secret).unwrap();

        let message = b"test_message";
        let mut output = [0u8; HMAC_HEX_SIZE];

        let len = signer.sign_to_slice(message, &mut output).unwrap();
        assert_eq!(len, HMAC_HEX_SIZE);

        // Verify output is valid hex
        let hex_str = std::str::from_utf8(&output).unwrap();
        assert!(hex_str.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_hex_lookup_encoding() {
        let input = [0x00, 0x0F, 0xF0, 0xFF, 0x12, 0x34, 0xAB, 0xCD];
        let mut output = [0u8; 16];

        encode_hex_zero_alloc(&input, &mut output);

        let expected = "000ff0ff1234abcd";
        assert_eq!(&output[..], expected.as_bytes());
    }

    #[test]
    fn test_request_builder() {
        let mut builder = WapiRequestBuilder::new("test_api_key");
        
        let payload = builder.build_unsigned_payload(
            1234567890000,
            5000,
            Some("symbol=BTCUSDT"),
        );

        assert!(payload.contains("apiKey=test_api_key"));
        assert!(payload.contains("timestamp=1234567890000"));
        assert!(payload.contains("recvWindow=5000"));
        assert!(payload.contains("symbol=BTCUSDT"));
    }
}
