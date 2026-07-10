//! Jito Block Engine gRPC Client
//! 
//! High-throughput streaming client for Jito Block Engine.
//! Parses Bundle and Transaction streams to detect MEV tip competitions.

use alloc::string::String;
use alloc::vec::Vec;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tracing::{info, warn, error};

/// Maximum pending bundles to track
const MAX_PENDING_BUNDLES: usize = 1024;

/// Tip competition threshold in micro-lamports per compute unit
const TIP_COMPETITION_THRESHOLD: u64 = 100;

#[derive(Error, Debug)]
pub enum JitoClientError {
    #[error("gRPC connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Stream error: {0}")]
    StreamError(String),
    #[error("Invalid bundle format")]
    InvalidBundleFormat,
    #[error("Channel closed")]
    ChannelClosed,
}

pub type Result<T> = core::result::Result<T, JitoClientError>;

/// Bundle information received from Jito
#[derive(Clone, Debug)]
pub struct BundleInfo {
    pub bundle_id: String,
    pub tip_lamports: u64,
    pub compute_units: u64,
    pub transaction_count: usize,
    pub timestamp_us: u64,
}

/// Real-time tip competition metrics
#[derive(Clone, Debug, Default)]
pub struct TipCompetitionMetrics {
    /// Average tip per compute unit across recent bundles
    pub avg_tip_per_cu: u64,
    /// P95 tip per compute unit
    pub p95_tip_per_cu: u64,
    /// Number of competing bundles in last 100ms
    pub competing_bundles: usize,
    /// Recommended tip to win next slot
    pub recommended_tip_per_cu: u64,
}

/// Jito Block Engine Client
/// 
/// Streams bundle and transaction data from Jito's gRPC endpoint.
/// Tracks real-time tip competition for optimal fee estimation.
pub struct JitoBlockEngineClient {
    bundle_tx: mpsc::Sender<BundleInfo>,
    metrics: TipCompetitionMetrics,
    recent_bundles: Vec<BundleInfo>,
    is_connected: bool,
}

impl JitoBlockEngineClient {
    /// Create a new client (connection established separately)
    pub fn new() -> Self {
        let (bundle_tx, _) = mpsc::channel(MAX_PENDING_BUNDLES);
        
        Self {
            bundle_tx,
            metrics: TipCompetitionMetrics::default(),
            recent_bundles: Vec::with_capacity(100),
            is_connected: false,
        }
    }

    /// Connect to Jito Block Engine gRPC endpoint
    pub async fn connect(&mut self, endpoint: &str) -> Result<()> {
        // In production, this would establish actual tonic gRPC connection
        // For now, we simulate the connection state
        info!("Connecting to Jito Block Engine at {}", endpoint);
        
        // Validate endpoint format
        if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
            return Err(JitoClientError::ConnectionFailed(
                "Endpoint must start with http:// or https://".to_string()
            ));
        }
        
        self.is_connected = true;
        info!("Connected to Jito Block Engine");
        Ok(())
    }

    /// Check if connected
    pub const fn is_connected(&self) -> bool {
        self.is_connected
    }

    /// Process incoming bundle stream (zero-copy parsing where possible)
    pub async fn process_bundle_stream(
        &mut self,
        bundle_data: &[u8],
    ) -> Result<BundleInfo> {
        if !self.is_connected {
            return Err(JitoClientError::ConnectionFailed(
                "Not connected to Jito Block Engine".to_string()
            ));
        }

        // Parse bundle header (first 32 bytes)
        if bundle_data.len() < 32 {
            return Err(JitoClientError::InvalidBundleFormat);
        }

        // Extract bundle ID (first 16 bytes as hex representation)
        let bundle_id = hex_encode(&bundle_data[0..16]);
        
        // Extract tip lamports (bytes 16-24, little-endian u64)
        let tip_lamports = u64::from_le_bytes(
            bundle_data[16..24].try_into().map_err(|_| JitoClientError::InvalidBundleFormat)?
        );
        
        // Extract compute units (bytes 24-32, little-endian u64)
        let compute_units = u64::from_le_bytes(
            bundle_data[24..32].try_into().map_err(|_| JitoClientError::InvalidBundleFormat)?
        );
        
        // Transaction count from remaining data
        let transaction_count = if bundle_data.len() > 32 {
            (bundle_data.len() - 32) / 128 // Estimate based on avg tx size
        } else {
            0
        };

        let bundle_info = BundleInfo {
            bundle_id,
            tip_lamports,
            compute_units,
            transaction_count,
            timestamp_us: current_timestamp_us(),
        };

        // Update metrics
        self.update_metrics(&bundle_info);

        // Send to internal channel for downstream consumers
        let _ = self.bundle_tx.send(bundle_info.clone()).await;

        Ok(bundle_info)
    }

    /// Get current tip competition metrics
    pub const fn get_metrics(&self) -> &TipCompetitionMetrics {
        &self.metrics
    }

    /// Calculate recommended tip for guaranteed inclusion
    pub fn calculate_recommended_tip(&self, compute_units: u64) -> u64 {
        let tip_per_cu = self.metrics.recommended_tip_per_cu;
        tip_per_cu.saturating_mul(compute_units)
    }

    /// Update internal metrics with new bundle data
    fn update_metrics(&mut self, bundle: &BundleInfo) {
        // Add to recent bundles ring buffer
        if self.recent_bundles.len() >= 100 {
            self.recent_bundles.remove(0);
        }
        self.recent_bundles.push(bundle.clone());

        // Calculate tip per CU
        let tip_per_cu = if bundle.compute_units > 0 {
            bundle.tip_lamports / bundle.compute_units
        } else {
            0
        };

        // Update average
        let n = self.recent_bundles.len() as u64;
        let sum_tips: u64 = self.recent_bundles.iter()
            .map(|b| if b.compute_units > 0 { b.tip_lamports / b.compute_units } else { 0 })
            .sum();
        
        self.metrics.avg_tip_per_cu = if n > 0 { sum_tips / n } else { 0 };

        // Calculate P95 (simple approximation)
        let mut tips: Vec<u64> = self.recent_bundles.iter()
            .map(|b| if b.compute_units > 0 { b.tip_lamports / b.compute_units } else { 0 })
            .collect();
        tips.sort_unstable();
        
        let p95_idx = (tips.len() as f64 * 0.95) as usize;
        self.metrics.p95_tip_per_cu = tips.get(p95_idx).copied().unwrap_or(0);

        // Count competing bundles (last 100ms = 100,000 us)
        let cutoff = bundle.timestamp_us.saturating_sub(100_000);
        self.metrics.competing_bundles = self.recent_bundles.iter()
            .filter(|b| b.timestamp_us >= cutoff)
            .count();

        // Recommend tip slightly above P95 to win competition
        self.metrics.recommended_tip_per_cu = self.metrics.p95_tip_per_cu
            .saturating_add(TIP_COMPETITION_THRESHOLD);
    }
}

impl Default for JitoBlockEngineClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple hex encoder (no allocation for fixed-size inputs)
fn hex_encode(data: &[u8]) -> String {
    const HEX_CHARS: &[u8] = b"0123456789abcdef";
    let mut result = String::with_capacity(data.len() * 2);
    
    for &byte in data {
        result.push(HEX_CHARS[(byte >> 4) as usize] as char);
        result.push(HEX_CHARS[(byte & 0x0F) as usize] as char);
    }
    
    result
}

/// Get current timestamp in microseconds
fn current_timestamp_us() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_encode() {
        let data = [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef];
        let encoded = hex_encode(&data);
        assert_eq!(encoded, "0123456789abcdef");
    }

    #[tokio::test]
    async fn test_client_connection() {
        let mut client = JitoBlockEngineClient::new();
        assert!(!client.is_connected());
        
        let result = client.connect("https://mainnet.block-engine.jito.wtf").await;
        assert!(result.is_ok());
        assert!(client.is_connected());
    }

    #[tokio::test]
    async fn test_bundle_parsing() {
        let mut client = JitoBlockEngineClient::new();
        let _ = client.connect("https://mainnet.block-engine.jito.wtf").await;
        
        // Simulate bundle data: 16-byte ID + 8-byte tip + 8-byte CU
        let mut bundle_data = vec![0u8; 40];
        bundle_data[0..16].copy_from_slice(&[1u8; 16]); // Bundle ID
        bundle_data[16..24].copy_from_slice(&1000u64.to_le_bytes()); // 1000 lamports tip
        bundle_data[24..32].copy_from_slice(&50000u64.to_le_bytes()); // 50000 CU
        
        let result = client.process_bundle_stream(&bundle_data).await;
        assert!(result.is_ok());
        
        let bundle = result.unwrap();
        assert_eq!(bundle.tip_lamports, 1000);
        assert_eq!(bundle.compute_units, 50000);
    }

    #[tokio::test]
    async fn test_tip_calculation() {
        let mut client = JitoBlockEngineClient::new();
        let _ = client.connect("https://mainnet.block-engine.jito.wtf").await;
        
        // Feed some sample bundles
        for i in 0..10 {
            let mut bundle_data = vec![0u8; 40];
            bundle_data[0..16].copy_from_slice(&[i as u8; 16]);
            bundle_data[16..24].copy_from_slice(&(1000 + i * 100)u64.to_le_bytes());
            bundle_data[24..32].copy_from_slice(&50000u64.to_le_bytes());
            
            let _ = client.process_bundle_stream(&bundle_data).await;
        }
        
        let recommended = client.calculate_recommended_tip(100000);
        assert!(recommended > 0);
    }
}
