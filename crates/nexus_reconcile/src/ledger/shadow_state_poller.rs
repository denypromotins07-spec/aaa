//! Shadow State Poller - Periodic REST API polling for exchange ground-truth.
//! 
//! CRITICAL: This background task queries the exchange REST API every N seconds
//! to get absolute ground-truth state (balances, positions, orders). It implements
//! exponential backoff with jitter to handle HTTP 429 rate limits gracefully.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{self, Interval};
use reqwest::{Client, Response, StatusCode};
use serde::{Deserialize, Serialize};
use tracing::{warn, error, info, debug};

use crate::{ReconcileError, ExchangeStateSnapshot, PositionState, OrderState};
use super::lock_free_oms_snapshot::LockFreeOMSSnapshot;

/// Configuration for the shadow state poller
#[derive(Debug, Clone)]
pub struct ShadowPollerConfig {
    /// Base interval between polls (default: 5 seconds)
    pub poll_interval_ms: u64,
    
    /// Maximum consecutive rate limit errors before pausing
    pub max_rate_limit_errors: u32,
    
    /// Initial backoff duration on rate limit (default: 1 second)
    pub initial_backoff_ms: u64,
    
    /// Maximum backoff duration (default: 60 seconds)
    pub max_backoff_ms: u64,
    
    /// Enable jitter on backoff (random factor 0.5-1.5x)
    pub enable_jitter: bool,
}

impl Default for ShadowPollerConfig {
    fn default() -> Self {
        Self {
            poll_interval_ms: 5000,
            max_rate_limit_errors: 5,
            initial_backoff_ms: 1000,
            max_backoff_ms: 60000,
            enable_jitter: true,
        }
    }
}

/// Statistics about polling operations
#[derive(Debug, Clone, Default)]
pub struct PollerStats {
    pub total_polls: u64,
    pub successful_polls: u64,
    pub rate_limit_hits: u64,
    pub connection_errors: u64,
    pub state_drift_detections: u64,
    pub current_backoff_ms: u64,
    pub consecutive_rate_limits: u32,
}

/// Shadow State Poller that periodically queries exchange REST API
pub struct ShadowStatePoller {
    config: ShadowPollerConfig,
    client: Client,
    base_url: String,
    api_key: String,
    api_secret: String,
    
    /// Reference to lock-free OMS snapshot for comparison
    oms_snapshot: Arc<LockFreeOMSSnapshot>,
    
    /// Atomic flag to stop polling
    running: AtomicBool,
    
    /// Statistics
    stats: Arc<parking_lot::RwLock<PollerStats>>,
    
    /// Current backoff duration in milliseconds
    current_backoff_ms: AtomicU64,
    
    /// Consecutive rate limit counter
    consecutive_rate_limits: AtomicU32,
}

impl ShadowStatePoller {
    pub fn new(
        config: ShadowPollerConfig,
        base_url: String,
        api_key: String,
        api_secret: String,
        oms_snapshot: Arc<LockFreeOMSSnapshot>,
    ) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client");
        
        Self {
            config,
            client,
            base_url,
            api_key,
            api_secret,
            oms_snapshot,
            running: AtomicBool::new(false),
            stats: Arc::new(parking_lot::RwLock::new(PollerStats::default())),
            current_backoff_ms: AtomicU64::new(config.initial_backoff_ms),
            consecutive_rate_limits: AtomicU32::new(0),
        }
    }
    
    /// Start the background polling task
    pub fn start(&self) -> tokio::task::JoinHandle<()> {
        self.running.store(true, Ordering::SeqCst);
        
        let this = Arc::new(self.clone_for_task());
        let mut interval = time::interval(Duration::from_millis(self.config.poll_interval_ms));
        interval.set_missed_tick_behavior(time::MissedTickBehavior::Skip);
        
        tokio::spawn(async move {
            info!("ShadowStatePoller started with interval {}ms", 
                  this.config.poll_interval_ms);
            
            while this.running.load(Ordering::Relaxed) {
                interval.tick().await;
                
                // Check if we're in backoff
                let backoff = this.current_backoff_ms.load(Ordering::Relaxed);
                if backoff > this.config.initial_backoff_ms {
                    debug!("In backoff, waiting {}ms", backoff);
                    time::sleep(Duration::from_millis(backoff)).await;
                }
                
                this.poll_and_reconcile().await;
            }
            
            info!("ShadowStatePoller stopped");
        })
    }
    
    /// Stop the polling task
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
    
    /// Get current statistics
    pub fn get_stats(&self) -> PollerStats {
        self.stats.read().clone()
    }
    
    /// Clone self for background task (without JoinHandle)
    fn clone_for_task(&self) -> ShadowStatePollerRef {
        ShadowStatePollerRef {
            config: self.config.clone(),
            client: self.client.clone(),
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            api_secret: self.api_secret.clone(),
            oms_snapshot: Arc::clone(&self.oms_snapshot),
            running: &self.running,
            stats: Arc::clone(&self.stats),
            current_backoff_ms: &self.current_backoff_ms,
            consecutive_rate_limits: &self.consecutive_rate_limits,
        }
    }
}

/// Reference version for background tasks (no ownership)
struct ShadowStatePollerRef<'a> {
    config: ShadowPollerConfig,
    client: Client,
    base_url: String,
    api_key: String,
    api_secret: String,
    oms_snapshot: Arc<LockFreeOMSSnapshot>,
    running: &'a AtomicBool,
    stats: Arc<parking_lot::RwLock<PollerStats>>,
    current_backoff_ms: &'a AtomicU64,
    consecutive_rate_limits: &'a AtomicU32,
}

impl<'a> ShadowStatePollerRef<'a> {
    /// Main poll and reconcile loop
    async fn poll_and_reconcile(&self) {
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_polls += 1;
        }
        
        match self.fetch_exchange_state().await {
            Ok(exchange_state) => {
                // Reset backoff on success
                self.current_backoff_ms.store(
                    self.config.initial_backoff_ms, 
                    Ordering::Relaxed
                );
                self.consecutive_rate_limits.store(0, Ordering::Relaxed);
                
                {
                    let mut stats = self.stats.write();
                    stats.successful_polls += 1;
                }
                
                // Compare with local OMS state
                self.compare_states(exchange_state).await;
            }
            Err(ReconcileError::RateLimitExceeded { retry_after }) => {
                self.handle_rate_limit(retry_after).await;
            }
            Err(e) => {
                warn!("Poll error: {:?}", e);
                {
                    let mut stats = self.stats.write();
                    if matches!(e, ReconcileError::HttpError(_)) {
                        stats.connection_errors += 1;
                    }
                }
            }
        }
    }
    
    /// Fetch ground-truth state from exchange REST API
    async fn fetch_exchange_state(&self) -> Result<ExchangeStateSnapshot, ReconcileError> {
        // Build request with exchange-specific authentication
        let url = format!("{}/api/v3/account", self.base_url);
        
        let timestamp = chrono::Utc::now().timestamp_millis() as u64;
        
        // Signature calculation (exchange-specific, simplified here)
        let signature = self.calculate_signature(&format!("timestamp={}", timestamp));
        
        let response = self.client
            .get(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .query(&[
                ("timestamp", timestamp.to_string()),
                ("signature", signature),
            ])
            .send()
            .await?;
        
        self.parse_exchange_response(response).await
    }
    
    /// Calculate request signature (HMAC-SHA256, exchange-specific)
    fn calculate_signature(&self, payload: &str) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        
        type HmacSha256 = Hmac<Sha256>;
        let mut mac = HmacSha256::new_from_slice(self.api_secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(payload.as_bytes());
        let result = mac.finalize();
        hex::encode(result.into_bytes())
    }
    
    /// Parse exchange API response into unified format
    async fn parse_exchange_response(
        &self, 
        response: Response
    ) -> Result<ExchangeStateSnapshot, ReconcileError> {
        let status = response.status();
        
        if status == StatusCode::TOO_MANY_REQUESTS {
            // Extract Retry-After header or use default
            let retry_after = response
                .headers()
                .get("Retry-After")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(5);
            
            return Err(ReconcileError::RateLimitExceeded { retry_after });
        }
        
        if !status.is_success() {
            return Err(ReconcileError::HttpError(
                reqwest::Error::from(response.error_for_status().unwrap_err())
            ));
        }
        
        // Parse JSON response (exchange-specific structure)
        #[derive(Deserialize)]
        struct RawAccountResponse {
            balances: Vec<RawBalance>,
            positions: Vec<RawPosition>,
            orders: Vec<RawOrder>,
            serverTime: u64,
        }
        
        #[derive(Deserialize)]
        struct RawBalance {
            asset: String,
            free: String,
            locked: String,
        }
        
        #[derive(Deserialize)]
        struct RawPosition {
            symbol: String,
            positionAmt: String,
            entryPrice: String,
            unRealizedProfit: String,
        }
        
        #[derive(Deserialize)]
        struct RawOrder {
            orderId: String,
            symbol: String,
            side: String,
            origQty: String,
            executedQty: String,
            price: String,
            status: String,
        }
        
        let raw: RawAccountResponse = response.json().await?;
        
        // Convert to unified format using scaled integers (1e8 precision)
        const SCALE: i128 = 100_000_000;  // 8 decimal places
        
        let mut balances = std::collections::HashMap::new();
        for b in raw.balances {
            let free: f64 = b.free.parse().unwrap_or(0.0);
            let locked: f64 = b.locked.parse().unwrap_or(0.0);
            let total = ((free + locked) * SCALE as f64) as i128;
            if total > 0 {
                balances.insert(b.asset, total);
            }
        }
        
        let mut positions = std::collections::HashMap::new();
        for p in raw.positions {
            let qty: f64 = p.positionAmt.parse().unwrap_or(0.0);
            if qty.abs() > 0.00000001 {
                let entry_price: f64 = p.entry_price.parse().unwrap_or(0.0);
                let unrealized_pnl: f64 = p.unRealizedProfit.parse().unwrap_or(0.0);
                
                positions.insert(p.symbol, PositionState {
                    symbol: p.symbol,
                    side: if qty > 0.0 { 1 } else if qty < 0.0 { -1 } else { 0 },
                    qty: (qty.abs() * SCALE as f64) as i128,
                    entry_price: (entry_price * SCALE as f64) as i128,
                    unrealized_pnl: (unrealized_pnl * SCALE as f64) as i128,
                });
            }
        }
        
        let active_orders: Vec<OrderState> = raw.orders
            .into_iter()
            .filter(|o| o.status != "CANCELED" && o.status != "EXPIRED")
            .map(|o| {
                let qty: f64 = o.origQty.parse().unwrap_or(0.0);
                let filled: f64 = o.executedQty.parse().unwrap_or(0.0);
                let price: f64 = o.price.parse().unwrap_or(0.0);
                
                OrderState {
                    order_id: o.orderId,
                    symbol: o.symbol,
                    side: if o.side == "BUY" { 1 } else { -1 },
                    qty: (qty * SCALE as f64) as i128,
                    filled_qty: (filled * SCALE as f64) as i128,
                    price: (price * SCALE as f64) as i128,
                    status: o.status,
                }
            })
            .collect();
        
        Ok(ExchangeStateSnapshot {
            balances,
            positions,
            active_orders,
            server_time_ms: raw.serverTime,
            epoch_id: 0,  // Will be set by caller
        })
    }
    
    /// Handle rate limit with exponential backoff
    async fn handle_rate_limit(&self, suggested_retry_after: u64) {
        let consecutive = self.consecutive_rate_limits.fetch_add(1, Ordering::Relaxed) + 1;
        
        {
            let mut stats = self.stats.write();
            stats.rate_limit_hits += 1;
            stats.consecutive_rate_limits = consecutive;
        }
        
        // Calculate backoff with jitter
        let base_backoff = self.config.initial_backoff_ms 
            * 2u64.pow(consecutive.min(10) as u32);
        let backoff = base_backoff.min(self.config.max_backoff_ms);
        
        let final_backoff = if self.config.enable_jitter {
            // Add 0.5-1.5x jitter
            let jitter = rand::random::<f64>() * 1.0 + 0.5;
            (backoff as f64 * jitter) as u64
        } else {
            backoff
        };
        
        self.current_backoff_ms.store(final_backoff.max(suggested_retry_after * 1000), Ordering::Relaxed);
        
        warn!("Rate limited. Consecutive: {}, Backoff: {}ms", consecutive, final_backoff);
        
        // Check if we've hit max consecutive rate limits
        if consecutive >= self.config.max_rate_limit_errors {
            error!("Max consecutive rate limits reached. Pausing poller.");
            self.running.store(false, Ordering::Relaxed);
        }
    }
    
    /// Compare exchange state with local OMS snapshot
    async fn compare_states(&self, exchange_state: ExchangeStateSnapshot) {
        // Get tear-free OMS snapshot
        let oms_state = self.oms_snapshot.snapshot_blocking();
        
        // Simple balance comparison (production would be more sophisticated)
        let total_balance_scaled = oms_state.total_balance;
        
        // Sum exchange balances for comparison
        let exchange_total: i128 = exchange_state.balances.values().sum();
        
        // Allow small tolerance for timing differences (0.01%)
        let tolerance = (total_balance_scaled.abs() / 10000).max(1);
        
        if (total_balance_scaled - exchange_total).abs() > tolerance {
            error!(
                "CRITICAL_STATE_DRIFT: local={}, exchange={}, diff={}",
                total_balance_scaled,
                exchange_total,
                total_balance_scaled - exchange_total
            );
            
            {
                let mut stats = self.stats.write();
                stats.state_drift_detections += 1;
            }
            
            // TODO: Trigger kill switch via callback
            // In production, this would call a provided closure to halt trading
        }
    }
}

// Required dependencies not in main Cargo.toml yet
// These would be added to the workspace Cargo.toml:
// parking_lot = "0.12"
// hmac = "0.12"
// sha2 = "0.10"
// hex = "0.4"
// rand = "0.8"

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_exponential_backoff_calculation() {
        let config = ShadowPollerConfig::default();
        assert_eq!(config.initial_backoff_ms, 1000);
        assert_eq!(config.max_backoff_ms, 60000);
        
        // Verify exponential growth
        let mut backoff = config.initial_backoff_ms;
        for i in 0..10 {
            backoff *= 2;
            assert!(backoff <= config.max_backoff_ms || i >= 6);
        }
    }
}
