//! Chapter 2: REST API Snapshot Fetcher with Rate Limit Handling
//!
//! This module implements the REST API client for fetching order book snapshots
//! from Binance to heal sequence gaps. It includes strict rate limit handling
//! to prevent 429 errors from causing infinite retry loops.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tracing::{debug, error, info, warn};

/// Configuration for REST snapshot fetcher
#[derive(Debug, Clone)]
pub struct RestSnapshotConfig {
    /// Base URL for Binance REST API
    pub base_url: String,
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
    /// Minimum interval between requests (rate limit protection) in ms
    pub min_request_interval_ms: u64,
    /// Maximum retries on transient failure
    pub max_retries: u32,
    /// Initial retry delay in ms
    pub initial_retry_delay_ms: u64,
    /// Weight limit per second (Binance-specific)
    pub weight_limit_per_second: u64,
}

impl Default for RestSnapshotConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.binance.com".to_string(),
            request_timeout_secs: 5,
            min_request_interval_ms: 200, // 5 requests/second minimum spacing
            max_retries: 3,
            initial_retry_delay_ms: 100,
            weight_limit_per_second: 1200, // Binance default weight limit
        }
    }
}

/// Order book snapshot data
#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    pub last_update_id: u64,
    pub bids: Vec<(u64, u64)>, // (price, quantity)
    pub asks: Vec<(u64, u64)>,
    pub timestamp_ns: u64,
}

/// Rate limiter state
#[derive(Debug)]
struct RateLimiterState {
    /// Timestamp of last request in nanoseconds
    last_request_ns: u64,
    /// Current weight used in current second
    current_weight: u64,
    /// Window start timestamp
    window_start_ns: u64,
}

/// Statistics for REST operations
#[derive(Debug, Clone, Default)]
pub struct RestFetcherStats {
    pub total_requests: u64,
    pub successful_snapshots: u64,
    pub failed_requests: u64,
    pub rate_limited_requests: u64,
    pub retries_exhausted: u64,
    pub avg_latency_ms: f64,
    pub last_snapshot_last_id: u64,
}

/// REST Snapshot Fetcher with rate limit protection
pub struct RestSnapshotFetcher {
    config: RestSnapshotConfig,
    rate_limiter: Arc<RwLock<RateLimiterState>>,
    stats: Arc<RwLock<RestFetcherStats>>,
    consecutive_429s: Arc<AtomicUsize>,
    circuit_breaker_open: Arc<AtomicBool>,
    circuit_breaker_until_ns: Arc<AtomicU64>,
}

// SAFETY: Uses RwLock and atomics for thread safety
unsafe impl Send for RestSnapshotFetcher {}
unsafe impl Sync for RestSnapshotFetcher {}

impl RestSnapshotFetcher {
    pub fn new(config: RestSnapshotConfig) -> Self {
        Self {
            config,
            rate_limiter: Arc::new(RwLock::new(RateLimiterState {
                last_request_ns: 0,
                current_weight: 0,
                window_start_ns: 0,
            })),
            stats: Arc::new(RwLock::new(RestFetcherStats::default())),
            consecutive_429s: Arc::new(AtomicUsize::new(0)),
            circuit_breaker_open: Arc::new(AtomicBool::new(false)),
            circuit_breaker_until_ns: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Check if circuit breaker is open (too many 429s)
    fn is_circuit_breaker_open(&self) -> bool {
        if !self.circuit_breaker_open.load(Ordering::Acquire) {
            return false;
        }

        let now = current_time_ns();
        let until = self.circuit_breaker_until_ns.load(Ordering::Acquire);

        if now >= until {
            // Circuit breaker timeout expired
            self.circuit_breaker_open.store(false, Ordering::Release);
            info!("Circuit breaker closed, resuming REST requests");
            false
        } else {
            true
        }
    }

    /// Open circuit breaker due to excessive 429s
    fn open_circuit_breaker(&self) {
        let cooldown_ns = Duration::from_secs(60).as_nanos() as u64;
        self.circuit_breaker_until_ns.store(
            current_time_ns() + cooldown_ns,
            Ordering::Release,
        );
        self.circuit_breaker_open.store(true, Ordering::Release);
        error!("Circuit breaker OPEN - too many 429 responses. Cooldown: 60s");
    }

    /// Wait for rate limit compliance
    async fn wait_for_rate_limit(&self) {
        loop {
            let should_wait = {
                let mut guard = self.rate_limiter.write();
                let now = current_time_ns();

                // Reset window if needed (1 second window)
                if now - guard.window_start_ns >= Duration::from_secs(1).as_nanos() as u64 {
                    guard.window_start_ns = now;
                    guard.current_weight = 0;
                }

                // Check minimum interval
                if guard.last_request_ns > 0 {
                    let elapsed_ms = (now - guard.last_request_ns) / 1_000_000;
                    if elapsed_ms < self.config.min_request_interval_ms {
                        return Duration::from_millis(
                            self.config.min_request_interval_ms - elapsed_ms,
                        );
                    }
                }

                // Check weight limit
                if guard.current_weight >= self.config.weight_limit_per_second {
                    let wait_until = guard.window_start_ns + Duration::from_secs(1).as_nanos() as u64;
                    return Duration::from_nanos(wait_until - now);
                }

                // Update state for this request
                guard.last_request_ns = now;
                guard.current_weight += 1; // Simplified: each request = 1 weight
                None
            };

            match should_wait {
                Some(delay) => tokio::time::sleep(delay).await,
                None => break,
            }
        }
    }

    /// Fetch order book snapshot from Binance REST API
    pub async fn fetch_snapshot(&self, symbol: &str, depth: u16) -> Result<OrderBookSnapshot, Box<dyn std::error::Error + Send + Sync>> {
        // Check circuit breaker
        if self.is_circuit_breaker_open() {
            return Err("Circuit breaker open - too many rate limit violations".into());
        }

        // Wait for rate limit compliance
        self.wait_for_rate_limit().await;

        let mut retry_count = 0;
        let mut last_error: Option<Box<dyn std::error::Error + Send + Sync>> = None;

        while retry_count < self.config.max_retries {
            let start = std::time::Instant::now();

            match self.do_fetch_snapshot(symbol, depth).await {
                Ok(snapshot) => {
                    let latency_ms = start.elapsed().as_millis() as f64;
                    
                    // Update stats
                    {
                        let mut stats = self.stats.write();
                        stats.total_requests += 1;
                        stats.successful_snapshots += 1;
                        stats.avg_latency_ms = (stats.avg_latency_ms * (stats.successful_snapshots - 1) as f64 + latency_ms) 
                            / stats.successful_snapshots as f64;
                        stats.last_snapshot_last_id = snapshot.last_update_id;
                    }

                    // Reset consecutive 429 counter on success
                    self.consecutive_429s.store(0, Ordering::Relaxed);

                    info!(
                        "Snapshot fetched: {} depth={}, last_update_id={}, latency={:.2}ms",
                        symbol, depth, snapshot.last_update_id, latency_ms
                    );

                    return Ok(snapshot);
                }
                Err(e) => {
                    let error_str = e.to_string();
                    last_error = Some(e);

                    // Check if it's a 429 rate limit error
                    if error_str.contains("429") || error_str.contains("rate limit") {
                        let count = self.consecutive_429s.fetch_add(1, Ordering::Relaxed) + 1;
                        
                        {
                            let mut stats = self.stats.write();
                            stats.rate_limited_requests += 1;
                        }

                        warn!("Rate limited (429) - count: {}", count);

                        if count >= 5 {
                            self.open_circuit_breaker();
                            return Err("Circuit breaker triggered due to excessive 429s".into());
                        }

                        // Exponential backoff for rate limit
                        let backoff_ms = self.config.initial_retry_delay_ms * (2u64.pow(retry_count as u32));
                        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    } else {
                        // Non-rate-limit error, use standard retry
                        retry_count += 1;
                        
                        if retry_count < self.config.max_retries {
                            let backoff_ms = self.config.initial_retry_delay_ms * (2u64.pow(retry_count - 1));
                            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                        }
                    }
                }
            }
        }

        // All retries exhausted
        {
            let mut stats = self.stats.write();
            stats.total_requests += 1;
            stats.failed_requests += 1;
            stats.retries_exhausted += 1;
        }

        error!("Failed to fetch snapshot after {} retries", self.config.max_retries);
        Err(last_error.unwrap_or_else(|| "Unknown error".into()))
    }

    /// Perform the actual HTTP request
    async fn do_fetch_snapshot(
        &self,
        symbol: &str,
        depth: u16,
    ) -> Result<OrderBookSnapshot, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!(
            "{}/api/v3/depth?symbol={}&limit={}",
            self.config.base_url,
            symbol.to_uppercase(),
            depth
        );

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(self.config.request_timeout_secs))
            .build()?;

        let response = client.get(&url).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("HTTP {}: {}", status, body).into());
        }

        // Parse JSON response
        let json: serde_json::Value = response.json().await?;

        let last_update_id = json["lastUpdateId"]
            .as_u64()
            .ok_or("Missing lastUpdateId in response")?;

        let parse_price_level = |v: &serde_json::Value| -> Option<(u64, u64)> {
            if let Some(arr) = v.as_array() {
                if arr.len() >= 2 {
                    let price = arr[0].as_str()?.parse::<u64>().ok()?;
                    let qty = arr[1].as_str()?.parse::<u64>().ok()?;
                    return Some((price, qty));
                }
            }
            None
        };

        let bids = json["bids"]
            .as_array()
            .map(|arr| arr.iter().filter_map(parse_price_level).collect())
            .unwrap_or_default();

        let asks = json["asks"]
            .as_array()
            .map(|arr| arr.iter().filter_map(parse_price_level).collect())
            .unwrap_or_default();

        Ok(OrderBookSnapshot {
            last_update_id,
            bids,
            asks,
            timestamp_ns: current_time_ns(),
        })
    }

    /// Get statistics snapshot
    pub fn get_stats(&self) -> RestFetcherStats {
        self.stats.read().clone()
    }

    /// Reset consecutive 429 counter (call after successful heal)
    pub fn reset_429_counter(&self) {
        self.consecutive_429s.store(0, Ordering::Relaxed);
    }
}

/// Get current time in nanoseconds
fn current_time_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

use std::sync::atomic::AtomicBool;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_state() {
        let config = RestSnapshotConfig::default();
        let fetcher = RestSnapshotFetcher::new(config);

        // Should not be circuit breaker open initially
        assert!(!fetcher.is_circuit_breaker_open());
    }

    #[test]
    fn test_consecutive_429_tracking() {
        let config = RestSnapshotConfig::default();
        let fetcher = RestSnapshotFetcher::new(config);

        assert_eq!(fetcher.consecutive_429s.load(Ordering::Relaxed), 0);

        fetcher.consecutive_429s.fetch_add(1, Ordering::Relaxed);
        fetcher.consecutive_429s.fetch_add(1, Ordering::Relaxed);

        assert_eq!(fetcher.consecutive_429s.load(Ordering::Relaxed), 2);

        fetcher.reset_429_counter();
        assert_eq!(fetcher.consecutive_429s.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_stats_initialization() {
        let config = RestSnapshotConfig::default();
        let fetcher = RestSnapshotFetcher::new(config);

        let stats = fetcher.get_stats();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.successful_snapshots, 0);
        assert_eq!(stats.failed_requests, 0);
    }
}
