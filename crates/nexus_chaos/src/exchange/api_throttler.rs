// STAGE 25: CHAPTER 1 - API THROTTLER
// Simulates exchange rate limiting and API bans
// Tests Stage 22 Swarm Raft consensus under throttling conditions

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use serde::{Deserialize, Serialize};

/// Throttling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThrottleConfig {
    pub requests_per_second: u32,
    pub burst_size: u32,
    pub ban_duration_ms: u64,
    pub http_429_rate: f64, // Probability of returning 429
}

/// API response status
#[derive(Debug, Clone, PartialEq)]
pub enum ApiStatus {
    Success,
    RateLimited(u32), // Retry-after in seconds
    Banned(Duration),
    ConnectionRefused,
    Timeout,
}

/// Throttle state tracking
pub struct ThrottleState {
    pub request_count: AtomicUsize,
    pub last_request_time: AtomicU64,
    pub is_banned: AtomicBool,
    pub ban_until: AtomicU64,
    pub total_429s: AtomicU64,
}

impl Default for ThrottleState {
    fn default() -> Self {
        Self {
            request_count: AtomicUsize::new(0),
            last_request_time: AtomicU64::new(0),
            is_banned: AtomicBool::new(false),
            ban_until: AtomicU64::new(0),
            total_429s: AtomicU64::new(0),
        }
    }
}

/// Exchange API Throttler
/// Simulates rate limiting, bans, and connection issues
pub struct ApiThrottler {
    state: std::sync::Arc<ThrottleState>,
    config: ThrottleConfig,
    chaos_mode_flag: AtomicBool, // CRITICAL: Distinguishes test from real throttling
    rng_seed: u64,
}

impl ApiThrottler {
    pub fn new(config: ThrottleConfig, rng_seed: u64) -> Self {
        Self {
            state: std::sync::Arc::new(ThrottleState::default()),
            config,
            chaos_mode_flag: AtomicBool::new(false),
            rng_seed,
        }
    }

    /// Enable chaos mode (test scenario)
    pub fn enable_chaos_mode(&self) {
        self.chaos_mode_flag.store(true, Ordering::SeqCst);
    }

    /// Disable chaos mode
    pub fn disable_chaos_mode(&self) {
        self.chaos_mode_flag.store(false, Ordering::SeqCst);
    }

    /// Check if operating in chaos mode
    pub fn is_chaos_mode(&self) -> bool {
        self.chaos_mode_flag.load(Ordering::SeqCst)
    }

    /// Check if currently banned
    pub fn is_banned(&self) -> bool {
        if !self.state.is_banned.load(Ordering::Relaxed) {
            return false;
        }

        let now = Instant::now().as_millis() as u64;
        let ban_until = self.state.ban_until.load(Ordering::Relaxed);

        if now >= ban_until {
            // Ban expired
            self.state.is_banned.store(false, Ordering::Relaxed);
            false
        } else {
            true
        }
    }

    /// Simulate API request with throttling logic
    /// Returns ApiStatus indicating success or failure mode
    pub fn simulate_request(&self, request_id: u64) -> ApiStatus {
        // Check ban first
        if self.is_banned() {
            let remaining_ms = self.state.ban_until.load(Ordering::Relaxed) 
                - Instant::now().as_millis() as u64;
            return ApiStatus::Banned(Duration::from_millis(remaining_ms));
        }

        let now = Instant::now();
        let now_ms = now.as_millis() as u64;

        // Update request timing
        self.state.last_request_time.store(now_ms, Ordering::Relaxed);
        self.state.request_count.fetch_add(1, Ordering::Relaxed);

        // Deterministic RNG for reproducibility
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rng_seed + request_id);

        // Check rate limit (token bucket simulation)
        let should_429 = rng.gen::<f64>() < self.config.http_429_rate;
        
        if should_429 {
            self.state.total_429s.fetch_add(1, Ordering::Relaxed);
            let retry_after = 30; // Standard 30 second backoff
            return ApiStatus::RateLimited(retry_after);
        }

        // Check burst limit
        let current_count = self.state.request_count.load(Ordering::Relaxed);
        if current_count > self.config.burst_size as usize {
            // Trigger temporary ban for excessive requests
            let ban_duration = Duration::from_millis(self.config.ban_duration_ms);
            self.state.ban_until.store(
                now_ms + self.config.ban_duration_ms,
                Ordering::Relaxed,
            );
            self.state.is_banned.store(true, Ordering::Relaxed);
            return ApiStatus::Banned(ban_duration);
        }

        ApiStatus::Success
    }

    /// Get throttle statistics
    pub fn get_stats(&self) -> ThrottleStats {
        ThrottleStats {
            request_count: self.state.request_count.load(Ordering::Relaxed),
            total_429s: self.state.total_429s.load(Ordering::Relaxed),
            is_banned: self.state.is_banned.load(Ordering::Relaxed),
            chaos_mode: self.chaos_mode_flag.load(Ordering::SeqCst),
        }
    }

    /// Reset throttle state (for test cleanup)
    pub fn reset(&self) {
        self.state.request_count.store(0, Ordering::Relaxed);
        self.state.total_429s.store(0, Ordering::Relaxed);
        self.state.is_banned.store(false, Ordering::Relaxed);
        self.state.ban_until.store(0, Ordering::Relaxed);
    }
}

/// Throttle statistics
#[derive(Debug, Clone)]
pub struct ThrottleStats {
    pub request_count: usize,
    pub total_429s: u64,
    pub is_banned: bool,
    pub chaos_mode: bool,
}

/// Builder for throttle configurations
pub struct ThrottleConfigBuilder {
    requests_per_second: u32,
    burst_size: u32,
    ban_duration_ms: u64,
    http_429_rate: f64,
}

impl ThrottleConfigBuilder {
    pub fn new() -> Self {
        Self {
            requests_per_second: 100,
            burst_size: 200,
            ban_duration_ms: 60000,
            http_429_rate: 0.05,
        }
    }

    pub fn requests_per_second(mut self, rps: u32) -> Self {
        self.requests_per_second = rps;
        self
    }

    pub fn burst_size(mut self, size: u32) -> Self {
        self.burst_size = size;
        self
    }

    pub fn ban_duration(mut self, ms: u64) -> Self {
        self.ban_duration_ms = ms;
        self
    }

    pub fn http_429_rate(mut self, rate: f64) -> Self {
        self.http_429_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn build(self) -> ThrottleConfig {
        ThrottleConfig {
            requests_per_second: self.requests_per_second,
            burst_size: self.burst_size,
            ban_duration_ms: self.ban_duration_ms,
            http_429_rate: self.http_429_rate,
        }
    }
}

impl Default for ThrottleConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_throttle_429_generation() {
        let config = ThrottleConfigBuilder::new()
            .http_429_rate(1.0) // 100% 429 rate
            .build();

        let throttler = ApiThrottler::new(config, 42);
        throttler.enable_chaos_mode();

        // All requests should be rate limited
        assert!(matches!(throttler.simulate_request(1), ApiStatus::RateLimited(_)));
        assert!(matches!(throttler.simulate_request(2), ApiStatus::RateLimited(_)));
        assert!(matches!(throttler.simulate_request(3), ApiStatus::RateLimited(_)));

        let stats = throttler.get_stats();
        assert_eq!(stats.total_429s, 3);
    }

    #[test]
    fn test_ban_expiration() {
        let config = ThrottleConfigBuilder::new()
            .ban_duration_ms(100) // 100ms ban
            .burst_size(1) // Trigger ban after 1 request
            .build();

        let throttler = ApiThrottler::new(config, 42);
        
        // First request succeeds
        assert!(matches!(throttler.simulate_request(1), ApiStatus::Success));
        
        // Second request triggers ban
        assert!(matches!(throttler.simulate_request(2), ApiStatus::Banned(_)));
        assert!(throttler.is_banned());

        // Wait for ban to expire
        std::thread::sleep(Duration::from_millis(150));
        assert!(!throttler.is_banned());
    }
}
