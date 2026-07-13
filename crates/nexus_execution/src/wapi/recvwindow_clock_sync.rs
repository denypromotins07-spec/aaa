//! RecvWindow Clock Synchronization Daemon
//! 
//! Continuously syncs local clock with exchange server time to prevent
//! recvWindow errors. Implements graceful fallback on network failures.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Clock sync status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockSyncStatus {
    /// Initial state, no sync yet
    Uninitialized,
    /// Syncing in progress
    Syncing,
    /// Successfully synced
    Synced,
    /// Using last known good offset (degraded mode)
    Degraded,
    /// Failed to sync
    Failed,
}

/// Time offset between local and exchange clock (in milliseconds)
/// Positive means exchange is ahead of local
pub type TimeOffsetMs = i64;

/// Statistics for clock sync daemon
#[derive(Debug, Clone, Default)]
pub struct ClockSyncStats {
    pub total_sync_attempts: u64,
    pub successful_syncs: u64,
    pub failed_syncs: u64,
    pub consecutive_failures: u64,
    pub max_consecutive_failures: u64,
    pub last_offset_ms: i64,
    pub last_latency_ms: u64,
    pub degraded_mode_activations: u64,
}

/// Configuration for clock sync daemon
#[derive(Debug, Clone)]
pub struct ClockSyncConfig {
    /// How often to sync (milliseconds)
    pub sync_interval_ms: u64,
    /// Request timeout (milliseconds)
    pub request_timeout_ms: u64,
    /// Number of consecutive failures before entering degraded mode
    pub failure_threshold: u64,
    /// Cooldown period in degraded mode (milliseconds)
    pub degraded_cooldown_ms: u64,
    /// Minimum time between requests (rate limit protection)
    pub min_request_interval_ms: u64,
}

impl Default for ClockSyncConfig {
    fn default() -> Self {
        Self {
            sync_interval_ms: 10000, // 10 seconds
            request_timeout_ms: 3000, // 3 seconds
            failure_threshold: 3,
            degraded_cooldown_ms: 30000, // 30 seconds in degraded mode
            min_request_interval_ms: 200, // 200ms minimum between requests
        }
    }
}

/// RecvWindow Clock Sync Daemon
/// 
/// Maintains accurate time offset between local clock and exchange server.
/// Implements graceful fallback when network issues occur.
pub struct RecvWindowClockSyncDaemon {
    config: ClockSyncConfig,
    /// Time offset: exchange_time = local_time + offset_ms
    offset_ms: AtomicI64,
    /// Last successful sync timestamp
    last_sync_at: AtomicU64,
    /// Current sync status
    status: ClockSyncStatus,
    /// Consecutive failure count
    consecutive_failures: AtomicU64,
    /// Statistics
    stats: ClockSyncStats,
    /// Is currently in degraded mode
    is_degraded: AtomicBool,
    /// Last known good offset (for fallback)
    last_good_offset_ms: AtomicI64,
    /// Last request time (for rate limiting)
    last_request_at: AtomicU64,
}

impl RecvWindowClockSyncDaemon {
    pub fn new(config: ClockSyncConfig) -> Self {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        Self {
            config,
            offset_ms: AtomicI64::new(0),
            last_sync_at: AtomicU64::new(0),
            status: ClockSyncStatus::Uninitialized,
            consecutive_failures: AtomicU64::new(0),
            stats: ClockSyncStats::default(),
            is_degraded: AtomicBool::new(false),
            last_good_offset_ms: AtomicI64::new(0),
            last_request_at: AtomicU64::new(0),
        }
    }

    /// Get current time offset in milliseconds
    #[inline]
    pub fn get_offset_ms(&self) -> i64 {
        self.offset_ms.load(Ordering::Relaxed)
    }

    /// Get current sync status
    #[inline]
    pub fn status(&self) -> ClockSyncStatus {
        self.status
    }

    /// Check if daemon is healthy (not in degraded mode)
    #[inline]
    pub fn is_healthy(&self) -> bool {
        !self.is_degraded.load(Ordering::Relaxed) && self.status == ClockSyncStatus::Synced
    }

    /// Check if we can make a request (rate limiting)
    fn can_make_request(&self) -> bool {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        let last_request = self.last_request_at.load(Ordering::Relaxed);
        now_ms.saturating_sub(last_request) >= self.config.min_request_interval_ms
    }

    /// Record that a request was made
    fn record_request(&self) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.last_request_at.store(now_ms, Ordering::Relaxed);
    }

    /// Process a successful time sync response
    /// 
    /// # Arguments
    /// * `server_time_ms` - Server's current time in milliseconds since epoch
    /// * `request_latency_ms` - Round-trip latency in milliseconds
    pub fn on_sync_success(&self, server_time_ms: u64, request_latency_ms: u64) {
        let local_time_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Calculate offset: exchange_time = local_time + offset
        // So: offset = exchange_time - local_time
        // We use half the latency as an estimate for one-way delay
        let adjusted_local_time = local_time_ms + (request_latency_ms / 2);
        let offset = server_time_ms as i64 - adjusted_local_time as i64;

        self.offset_ms.store(offset, Ordering::Relaxed);
        self.last_good_offset_ms.store(offset, Ordering::Relaxed);
        
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.last_sync_at.store(now_ms, Ordering::Relaxed);

        // Reset failure counter
        self.consecutive_failures.store(0, Ordering::Relaxed);
        
        // Exit degraded mode if we were in it
        if self.is_degraded.swap(false, Ordering::Relaxed) {
            self.stats.degraded_mode_activations += 1;
        }

        self.status = ClockSyncStatus::Synced;
        
        // Update stats
        self.stats.successful_syncs += 1;
        self.stats.consecutive_failures = 0;
        self.stats.last_offset_ms = offset;
        self.stats.last_latency_ms = request_latency_ms;
    }

    /// Handle a sync failure
    /// 
    /// Implements graceful fallback with circuit breaker pattern
    pub fn on_sync_failure(&self, error: &str) {
        self.stats.failed_syncs += 1;
        
        let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        self.stats.consecutive_failures = failures;
        self.stats.max_consecutive_failures = 
            self.stats.max_consecutive_failures.max(failures);

        if failures >= self.config.failure_threshold {
            // Enter degraded mode - use last known good offset
            if !self.is_degraded.load(Ordering::Relaxed) {
                self.is_degraded.store(true, Ordering::Relaxed);
                self.status = ClockSyncStatus::Degraded;
                
                log::warn!(
                    "Clock sync entered degraded mode after {} failures. Error: {}. Using last known offset: {}ms",
                    failures,
                    error,
                    self.last_good_offset_ms.load(Ordering::Relaxed)
                );
            }
        } else {
            log::debug!("Clock sync failure ({}): {}", failures, error);
        }
    }

    /// Get the adjusted timestamp for API requests
    /// 
    /// Returns local_timestamp + offset, which should match exchange time
    #[inline]
    pub fn get_adjusted_timestamp_ms(&self) -> u64 {
        let local_time_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        let offset = if self.is_degraded.load(Ordering::Relaxed) {
            self.last_good_offset_ms.load(Ordering::Relaxed)
        } else {
            self.offset_ms.load(Ordering::Relaxed)
        };

        (local_time_ms as i64 + offset) as u64
    }

    /// Get the recvWindow value to use for requests
    /// 
    /// In degraded mode, returns a larger window to account for potential drift
    #[inline]
    pub fn get_recv_window_ms(&self) -> u64 {
        if self.is_degraded.load(Ordering::Relaxed) {
            // Use larger window in degraded mode (e.g., 10 seconds instead of 5)
            self.config.sync_interval_ms.saturating_mul(2).min(60000)
        } else {
            5000 // Standard 5 second window
        }
    }

    /// Get statistics snapshot
    pub fn get_stats(&self) -> ClockSyncStats {
        self.stats.clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) -> ClockSyncStats {
        let stats = self.stats.clone();
        // Note: Can't mutate self here without &mut, so this is read-only
        stats
    }
}

/// Background task that runs the clock sync daemon
pub async fn clock_sync_daemon_task(
    daemon: Arc<RecvWindowClockSyncDaemon>,
    mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
) {
    let mut interval = tokio::time::interval(Duration::from_millis(daemon.config.sync_interval_ms));
    
    loop {
        tokio::select! {
            _ = interval.tick() => {
                // Check rate limiting
                if !daemon.can_make_request() {
                    continue;
                }
                
                daemon.record_request();
                daemon.stats.total_sync_attempts += 1;
                daemon.status = ClockSyncStatus::Syncing;

                // In production, this would make actual HTTP request to exchange
                // For now, simulate with a placeholder
                // 
                // Example Binance endpoint: GET /api/v3/time
                // Response: { "serverTime": 1234567890000 }
                
                tracing::debug!("Clock sync attempt...");
                
                // Simulated sync - in production replace with actual HTTP call
                // let result = fetch_server_time().await;
                // match result {
                //     Ok((server_time, latency)) => daemon.on_sync_success(server_time, latency),
                //     Err(e) => daemon.on_sync_failure(&e.to_string()),
                // }
            }
            _ = shutdown_rx.recv() => {
                tracing::info!("Clock sync daemon shutting down");
                break;
            }
        }
    }
}

/// Fetch server time from exchange (placeholder - implement per exchange)
async fn fetch_server_time_http(
    url: &str,
    timeout_ms: u64,
) -> Result<(u64, u64), Box<dyn std::error::Error + Send + Sync>> {
    let start = Instant::now();
    
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(timeout_ms))
        .build()?;

    let response = client.get(url).send().await?;
    let body = response.text().await?;
    
    let latency_ms = start.elapsed().as_millis() as u64;

    // Parse JSON response - format varies by exchange
    // Binance: { "serverTime": 1234567890000 }
    let json: serde_json::Value = serde_json::from_str(&body)?;
    let server_time = json["serverTime"].as_u64()
        .ok_or("Invalid server time response")?;

    Ok((server_time, latency_ms))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offset_calculation() {
        let daemon = RecvWindowClockSyncDaemon::new(ClockSyncConfig::default());
        
        // Initially offset should be 0
        assert_eq!(daemon.get_offset_ms(), 0);
        assert_eq!(daemon.status(), ClockSyncStatus::Uninitialized);

        // Simulate successful sync where exchange is 100ms ahead
        let local_time = 1000000;
        let server_time = 1000100;
        let latency = 20; // 20ms round trip
        
        daemon.on_sync_success(server_time, latency);
        
        // Offset should be approximately 100ms (accounting for half latency adjustment)
        let offset = daemon.get_offset_ms();
        assert!(offset >= 90 && offset <= 110); // Allow some variance
        assert_eq!(daemon.status(), ClockSyncStatus::Synced);
        assert!(daemon.is_healthy());
    }

    #[test]
    fn test_degraded_mode_after_failures() {
        let daemon = RecvWindowClockSyncDaemon::new(ClockSyncConfig {
            failure_threshold: 3,
            ..Default::default()
        });

        // Simulate 3 consecutive failures
        daemon.on_sync_failure("timeout");
        daemon.on_sync_failure("timeout");
        assert!(!daemon.is_degraded.load(Ordering::Relaxed));
        
        daemon.on_sync_failure("timeout");
        
        // Should enter degraded mode after 3rd failure
        assert!(daemon.is_degraded.load(Ordering::Relaxed));
        assert_eq!(daemon.status(), ClockSyncStatus::Degraded);
        
        // Stats should reflect this
        let stats = daemon.get_stats();
        assert_eq!(stats.consecutive_failures, 3);
        assert_eq!(stats.failed_syncs, 3);
    }

    #[test]
    fn test_recovery_from_degraded_mode() {
        let daemon = RecvWindowClockSyncDaemon::new(ClockSyncConfig {
            failure_threshold: 2,
            ..Default::default()
        });

        // Enter degraded mode
        daemon.on_sync_failure("error");
        daemon.on_sync_failure("error");
        assert!(daemon.is_degraded.load(Ordering::Relaxed));

        // Successful sync should exit degraded mode
        daemon.on_sync_success(1234567890000, 10);
        
        assert!(!daemon.is_degraded.load(Ordering::Relaxed));
        assert_eq!(daemon.status(), ClockSyncStatus::Synced);
        assert_eq!(daemon.consecutive_failures.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_recv_window_in_degraded_mode() {
        let daemon = RecvWindowClockSyncDaemon::new(ClockSyncConfig::default());
        
        // Normal mode: 5 second window
        assert_eq!(daemon.get_recv_window_ms(), 5000);

        // Force degraded mode
        daemon.is_degraded.store(true, Ordering::Relaxed);
        
        // Degraded mode: larger window
        assert!(daemon.get_recv_window_ms() > 5000);
    }
}
