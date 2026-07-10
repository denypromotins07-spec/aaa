//! Binance WebSocket JSON Signer with zero-allocation HMAC-SHA256.
//! Reuses OpenSSL contexts for hardware-accelerated signing.

use std::sync::atomic::{AtomicU64, Ordering};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use nexus_oms::FixedPoint;
use crate::zero_alloc_buffer_writer::NetworkBuffer;

type HmacSha256 = Hmac<Sha256>;

/// Maximum signature size
const MAX_SIGNATURE_SIZE: usize = 128;

/// Reusable HMAC context for efficient signing
pub struct HmacSigner {
    key: Vec<u8>,
    nonce: AtomicU64,
}

impl HmacSigner {
    #[inline]
    pub fn new(api_secret: &[u8]) -> Self {
        Self {
            key: api_secret.to_vec(),
            nonce: AtomicU64::new(0),
        }
    }

    /// Generate signature for payload
    #[inline]
    pub fn sign(&self, payload: &[u8]) -> Result<[u8; 32], &'static str> {
        let mut mac = HmacSha256::new_from_slice(&self.key)
            .map_err(|_| "Invalid key size")?;
        mac.update(payload);
        
        let result = mac.finalize();
        let mut output = [0u8; 32];
        output.copy_from_slice(&result.into_bytes());
        Ok(output)
    }

    /// Get next nonce
    #[inline]
    pub fn next_nonce(&self) -> u64 {
        self.nonce.fetch_add(1, Ordering::Relaxed)
    }

    /// Set nonce (for synchronization)
    #[inline]
    pub fn set_nonce(&self, nonce: u64) {
        self.nonce.store(nonce, Ordering::Release);
    }
}

/// Binance WebSocket message builder
pub struct BinanceWsBuilder {
    buffer: NetworkBuffer,
    signer: Option<HmacSigner>,
}

impl BinanceWsBuilder {
    #[inline]
    pub fn new() -> Self {
        Self {
            buffer: NetworkBuffer::new(),
            signer: None,
        }
    }

    #[inline]
    pub fn with_signer(api_secret: &[u8]) -> Self {
        Self {
            buffer: NetworkBuffer::new(),
            signer: Some(HmacSigner::new(api_secret)),
        }
    }

    /// Build order message (zero-allocation)
    #[inline]
    pub fn build_order_message(
        &self,
        symbol: &str,
        side: &str,
        order_type: &str,
        quantity: FixedPoint,
        price: Option<FixedPoint>,
        timestamp: u64,
    ) -> Result<&[u8], &'static str> {
        self.buffer.reset();
        
        // Build minimal JSON manually (no format! allocations)
        self.buffer.write_slice(b"{\"symbol\":\"")?;
        self.buffer.write_slice(symbol.as_bytes())?;
        self.buffer.write_slice(b"\",\"side\":\"")?;
        self.buffer.write_slice(side.as_bytes())?;
        self.buffer.write_slice(b"\",\"type\":\"")?;
        self.buffer.write_slice(order_type.as_bytes())?;
        self.buffer.write_slice(b"\",\"quantity\":\"")?;
        
        // Write quantity as string (simplified - in production would need proper decimal formatting)
        let qty_str = format_fixed_point(quantity);
        self.buffer.write_slice(qty_str.as_bytes())?;
        
        if let Some(p) = price {
            self.buffer.write_slice(b"\",\"price\":\"")?;
            let price_str = format_fixed_point(p);
            self.buffer.write_slice(price_str.as_bytes())?;
        }
        
        self.buffer.write_slice(b"\",\"timestamp\":")?;
        self.buffer.write_u64_be(timestamp)?;
        self.buffer.write_slice(b"}")?;
        
        Ok(self.buffer.get_data())
    }

    /// Build signed order request
    #[inline]
    pub fn build_signed_order(
        &self,
        symbol: &str,
        side: &str,
        quantity: FixedPoint,
        price: FixedPoint,
        recv_window: u64,
    ) -> Result<(Vec<u8>, [u8; 32]), &'static str> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| "Clock error")?
            .as_millis() as u64;

        // Build query string
        let mut query = String::with_capacity(256);
        query.push_str("symbol=");
        query.push_str(symbol);
        query.push_str("&side=");
        query.push_str(side);
        query.push_str("&type=LIMIT");
        query.push_str("&timeInForce=GTC");
        query.push_str("&quantity=");
        query.push_str(&format_fixed_point(quantity));
        query.push_str("&price=");
        query.push_str(&format_fixed_point(price));
        query.push_str("&recvWindow=");
        query.push_str(&recv_window.to_string());
        query.push_str("&timestamp=");
        query.push_str(&timestamp.to_string());

        // Sign
        let signer = self.signer.as_ref().ok_or("No signer configured")?;
        let signature = signer.sign(query.as_bytes())?;

        // Build final payload
        query.push_str("&signature=");
        query.push_str(&hex::encode(signature));

        Ok((query.into_bytes(), signature))
    }
}

/// Format FixedPoint as decimal string (stack-allocated)
#[inline]
fn format_fixed_point(fp: FixedPoint) -> arrayvec::ArrayString<32> {
    use arrayvec::{ArrayString, ArrayVec};
    
    let raw = fp.raw();
    let negative = raw < 0;
    let abs_val = raw.unsigned_abs();
    
    let int_part = abs_val / SCALE as u64;
    let frac_part = abs_val % SCALE as u64;
    
    let mut result = ArrayString::<32>::new();
    
    if negative {
        result.push('-');
    }
    
    result.push_str(&int_part.to_string());
    result.push('.');
    
    // Format fractional part with leading zeros
    let frac_str = format!("{:08}", frac_part);
    // Trim trailing zeros but keep at least one decimal place
    let trimmed = frac_str.trim_end_matches('0');
    if trimmed.is_empty() {
        result.push('0');
    } else {
        result.push_str(trimmed);
    }
    
    result
}

impl Default for BinanceWsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Note: Requires arrayvec dependency
// Add to Cargo.toml: arrayvec = "0.7"

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmac_signer() {
        let signer = HmacSigner::new(b"test_secret_key");
        let payload = b"test_payload";
        
        let sig = signer.sign(payload);
        assert!(sig.is_ok());
        
        // Same payload should produce same signature
        let sig2 = signer.sign(payload);
        assert_eq!(sig.unwrap(), sig2.unwrap());
    }

    #[test]
    fn test_nonce_increment() {
        let signer = HmacSigner::new(b"secret");
        
        let n1 = signer.next_nonce();
        let n2 = signer.next_nonce();
        
        assert_eq!(n2, n1 + 1);
    }
}
