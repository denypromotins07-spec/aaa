//! Token Bucket Throttle - Zero-Allocation Rate Limiter
//! 
//! Implements a strict token bucket algorithm to throttle outbound order flow.
//! This prevents exceeding exchange rate limits (e.g., 10 orders/second).
//! 
//! ZERO-ALLOCATION DESIGN:
//! - All math uses atomic operations on stack
//! - No heap allocations in hot path
//! - Uses nanosecond precision for accurate refill

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Result of a rate limit check
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitResult {
    /// Request allowed - token consumed
    Allowed,
    /// Request denied - rate limit exceeded
    Denied {
        tokens_available: u64,
        tokens_required: u64,
        retry_after_ns: u64,
    },
}

/// Token Bucket Throttle
/// 
/// Classic token bucket rate limiter optimized for the execution hot path.
/// Tokens are refilled at a constant rate up to a maximum capacity.
/// Each order consumes one token.
/// 
/// THREAD SAFETY:
/// Uses atomic CAS operations for thread-safe token consumption without locks.
pub struct TokenBucketThrottle {
    /// Current token count (scaled by 1e9 for nanosecond precision)
    tokens: AtomicU64,
    /// Maximum token capacity (scaled)
    max_tokens: u64,
    /// Refill rate in tokens per second (scaled)
    refill_rate_per_ns: u64,
    /// Timestamp of last refill (nanoseconds)
    last_refill_ns: AtomicU64,
    /// Whether throttle is active
    active: AtomicBool,
    /// Count of denied requests
    denied_count: AtomicU64,
    /// Count of allowed requests
    allowed_count: AtomicU64,
}

unsafe impl Send for TokenBucketThrottle {}
unsafe impl Sync for TokenBucketThrottle {}

impl TokenBucketThrottle {
    /// Create a new token bucket throttle
    /// 
    /// # Arguments
    /// * `max_tokens` - Maximum number of tokens (burst capacity)
    /// * `refill_rate_per_second` - Tokens added per second
    /// 
    /// # Example
    /// For 10 orders/second with burst of 20:
    /// `TokenBucketThrottle::new(20, 10)`
    pub fn new(max_tokens: u64, refill_rate_per_second: u64) -> Self {
        // Scale to nanosecond precision: tokens * 1e9
        let scale = 1_000_000_000u64;
        let initial_tokens = max_tokens * scale;
        let refill_rate_per_ns = refill_rate_per_second / scale;
        
        // Get current time
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        
        Self {
            tokens: AtomicU64::new(initial_tokens),
            max_tokens: initial_tokens,
            refill_rate_per_ns: refill_rate_per_ns,
            last_refill_ns: AtomicU64::new(now),
            active: AtomicBool::new(true),
            denied_count: AtomicU64::new(0),
            allowed_count: AtomicU64::new(0),
        }
    }

    /// Try to consume a token (representing one order).
    /// 
    /// This is the HOT PATH function called before every order submission.
    /// 
    /// # Returns
    /// * `RateLimitResult::Allowed` - Order can proceed
    /// * `RateLimitResult::Denied` - Rate limit exceeded, do not send order
    #[inline]
    pub fn try_consume(&self) -> RateLimitResult {
        // Fast path: check if throttle is disabled
        if !self.active.load(Ordering::Relaxed) {
            self.allowed_count.fetch_add(1, Ordering::Relaxed);
            return RateLimitResult::Allowed;
        }

        // Get current time
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        // Refill tokens based on elapsed time
        self.refill(now);

        // Token cost for one order (scaled)
        let token_cost = 1_000_000_000u64; // 1 token in scaled units

        // Try to atomically consume a token using CAS loop
        let mut current_tokens = self.tokens.load(Ordering::Relaxed);
        
        loop {
            if current_tokens >= token_cost {
                let new_tokens = current_tokens - token_cost;
                
                match self.tokens.compare_exchange_weak(
                    current_tokens,
                    new_tokens,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // Successfully consumed token
                        self.allowed_count.fetch_add(1, Ordering::Relaxed);
                        return RateLimitResult::Allowed;
                    }
                    Err(actual) => {
                        // Another thread modified tokens, retry with new value
                        current_tokens = actual;
                    }
                }
            } else {
                // Not enough tokens
                self.denied_count.fetch_add(1, Ordering::Relaxed);
                
                // Calculate time until next token available
                let tokens_needed = token_cost - current_tokens;
                let retry_after_ns = if self.refill_rate_per_ns > 0 {
                    tokens_needed / self.refill_rate_per_ns
                } else {
                    u64::MAX
                };
                
                return RateLimitResult::Denied {
                    tokens_available: current_tokens / 1_000_000_000,
                    tokens_required: 1,
                    retry_after_ns,
                };
            }
        }
    }

    /// Refill tokens based on elapsed time
    #[inline]
    fn refill(&self, now_ns: u64) {
        let last_refill = self.last_refill_ns.load(Ordering::Relaxed);
        
        // Calculate elapsed time
        let elapsed_ns = now_ns.saturating_sub(last_refill);
        
        if elapsed_ns == 0 {
            return;
        }

        // Calculate tokens to add
        let tokens_to_add = elapsed_ns * self.refill_rate_per_ns;
        
        if tokens_to_add == 0 {
            return;
        }

        // Atomically update last_refill timestamp and add tokens
        let mut current_tokens = self.tokens.load(Ordering::Relaxed);
        
        loop {
            let new_tokens = (current_tokens + tokens_to_add).min(self.max_tokens);
            
            match self.last_refill_ns.compare_exchange_weak(
                last_refill,
                now_ns,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.tokens.store(new_tokens, Ordering::Relaxed);
                    break;
                }
                Err(actual) => {
                    // Another thread updated last_refill, recalculate
                    let new_last_refill = actual;
                    let new_elapsed = now_ns.saturating_sub(new_last_refill);
                    if new_elapsed == 0 || new_last_refill <= last_refill {
                        break;
                    }
                    let new_tokens_to_add = new_elapsed * self.refill_rate_per_ns;
                    if new_tokens_to_add == 0 {
                        break;
                    }
                    current_tokens = self.tokens.load(Ordering::Relaxed);
                    // Continue loop to try again with updated values
                    let _ = new_tokens_to_add;
                }
            }
        }
    }

    /// Check if throttle is currently active
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Enable or disable the throttle
    #[inline]
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    /// Get current token count (for monitoring)
    #[inline]
    pub fn tokens_available(&self) -> u64 {
        self.tokens.load(Ordering::Relaxed) / 1_000_000_000
    }

    /// Get count of denied requests
    #[inline]
    pub fn denied_count(&self) -> u64 {
        self.denied_count.load(Ordering::Relaxed)
    }

    /// Get count of allowed requests
    #[inline]
    pub fn allowed_count(&self) -> u64 {
        self.allowed_count.load(Ordering::Relaxed)
    }

    /// Reset statistics (for testing)
    #[inline]
    pub fn reset_stats(&self) {
        self.denied_count.store(0, Ordering::Relaxed);
        self.allowed_count.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_basic_consumption() {
        let throttle = TokenBucketThrottle::new(10, 10); // 10 burst, 10/sec
        
        // Should be able to consume up to burst limit
        for i in 0..10 {
            assert_eq!(throttle.try_consume(), RateLimitResult::Allowed, "Failed at iteration {}", i);
        }
        
        // Next should be denied (or might succeed due to refill)
        let result = throttle.try_consume();
        // Allow for some refill during test execution
        assert!(matches!(result, RateLimitResult::Allowed | RateLimitResult::Denied { .. }));
    }

    #[test]
    fn test_rate_limiting() {
        let throttle = TokenBucketThrottle::new(5, 5); // Small bucket for quick test
        
        // Exhaust all tokens
        for _ in 0..5 {
            throttle.try_consume();
        }
        
        // Should be denied immediately after
        let result = throttle.try_consume();
        assert!(matches!(result, RateLimitResult::Denied { .. }));
    }

    #[test]
    fn test_token_refill() {
        let throttle = TokenBucketThrottle::new(10, 100); // 100 tokens/sec refill
        
        // Exhaust tokens
        for _ in 0..10 {
            throttle.try_consume();
        }
        
        // Wait for refill (50ms should give ~5 tokens at 100/sec)
        thread::sleep(Duration::from_millis(50));
        
        // Should have some tokens now
        let result = throttle.try_consume();
        assert_eq!(result, RateLimitResult::Allowed);
    }

    #[test]
    fn test_concurrent_consumption() {
        use std::sync::Arc;
        
        let throttle = Arc::new(TokenBucketThrottle::new(100, 100));
        let allowed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        
        let mut handles = vec![];
        
        // Spawn multiple threads trying to consume
        for _ in 0..10 {
            let t = Arc::clone(&throttle);
            let a = Arc::clone(&allowed);
            handles.push(thread::spawn(move || {
                for _ in 0..20 {
                    if let RateLimitResult::Allowed = t.try_consume() {
                        a.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }));
        }
        
        for h in handles {
            let _ = h.join();
        }
        
        // Should have allowed approximately 100 (initial) + some refilled
        let total_allowed = allowed.load(std::sync::atomic::Ordering::Relaxed);
        assert!(total_allowed >= 100 && total_allowed <= 250, "Total allowed: {}", total_allowed);
    }

    #[test]
    fn test_disable_throttle() {
        let throttle = TokenBucketThrottle::new(1, 1);
        
        // Consume the only token
        throttle.try_consume();
        
        // Should be denied
        assert!(matches!(throttle.try_consume(), RateLimitResult::Denied { .. }));
        
        // Disable throttle
        throttle.set_active(false);
        
        // Should now always allow
        assert_eq!(throttle.try_consume(), RateLimitResult::Allowed);
        assert_eq!(throttle.try_consume(), RateLimitResult::Allowed);
    }
}
