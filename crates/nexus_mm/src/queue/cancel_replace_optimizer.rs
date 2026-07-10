//! Cancel-Replace Optimizer with Flicker Detection and Rate Limiting.
//! Prevents exchange bans from excessive message rates.
//! Zero-allocation, no unwrap/expect in hot paths.

use crate::queue::queue_position_tracker::QueuePosition;
use crate::queue::fill_probability_markov::FillProbabilityMarkov;

/// Error types for cancel-replace operations
#[derive(Debug, Clone, PartialEq)]
pub enum CancelReplaceError {
    RateLimitExceeded,
    InvalidThreshold,
    FlickerDetected,
}

/// Action to take
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelReplaceAction {
    Hold,      // Keep current order
    Cancel,    // Cancel only
    Replace,   // Cancel and replace at new price
    NewOrder,  // Place new order
}

/// Decision result
#[derive(Debug, Clone, Copy)]
pub struct CancelReplaceDecision {
    pub action: CancelReplaceAction,
    pub new_price_ticks: Option<i64>,
    pub reason: &'static str,
    pub cooldown_remaining_ms: u64,
}

impl CancelReplaceDecision {
    pub const fn hold(reason: &'static str) -> Self {
        Self {
            action: CancelReplaceAction::Hold,
            new_price_ticks: None,
            reason,
            cooldown_remaining_ms: 0,
        }
    }
    
    pub const fn cancel(reason: &'static str) -> Self {
        Self {
            action: CancelReplaceAction::Cancel,
            new_price_ticks: None,
            reason,
            cooldown_remaining_ms: 0,
        }
    }
    
    pub const fn replace(price_ticks: i64, reason: &'static str) -> Self {
        Self {
            action: CancelReplaceAction::Replace,
            new_price_ticks: Some(price_ticks),
            reason,
            cooldown_remaining_ms: 0,
        }
    }
    
    pub const fn with_cooldown(mut self, ms: u64) -> Self {
        self.cooldown_remaining_ms = ms;
        self
    }
}

/// Configuration for optimizer
#[derive(Debug, Clone)]
pub struct CancelReplaceConfig {
    /// Minimum fill probability threshold (below this, consider cancel)
    pub min_fill_probability: f64,
    /// Maximum messages per second allowed by exchange
    pub max_messages_per_second: u32,
    /// Cooldown after cancel (milliseconds) to prevent flickering
    pub cancel_cooldown_ms: u64,
    /// Price change threshold (in ticks) to justify replace
    pub min_price_change_ticks: i64,
    /// Maximum consecutive cancels before forced pause
    pub max_consecutive_cancels: u32,
    /// Time window for rate limiting (milliseconds)
    pub rate_limit_window_ms: u64,
}

impl Default for CancelReplaceConfig {
    fn default() -> Self {
        Self {
            min_fill_probability: 0.1,
            max_messages_per_second: 100,
            cancel_cooldown_ms: 50,
            min_price_change_ticks: 1,
            max_consecutive_cancels: 5,
            rate_limit_window_ms: 1000,
        }
    }
}

/// Rate limiter using sliding window counter
pub struct SlidingWindowRateLimiter {
    /// Window size in milliseconds
    window_ms: u64,
    /// Max events per window
    max_events: u32,
    /// Event timestamps (circular buffer)
    timestamps: Vec<u64>,
    /// Head of circular buffer
    head: usize,
    /// Count of events in current window
    count: u32,
}

impl SlidingWindowRateLimiter {
    pub fn new(window_ms: u64, max_events: u32) -> Self {
        Self {
            window_ms,
            max_events,
            timestamps: vec![0; max_events as usize],
            head: 0,
            count: 0,
        }
    }
    
    #[inline(always)]
    pub fn try_record(&mut self, timestamp_ms: u64) -> bool {
        let window_start = timestamp_ms.saturating_sub(self.window_ms);
        
        // Remove expired events
        while self.count > 0 {
            let oldest_idx = (self.head + self.timestamps.len() - self.count as usize) % self.timestamps.len();
            if self.timestamps[oldest_idx] < window_start {
                self.count -= 1;
            } else {
                break;
            }
        }
        
        // Check if we can add new event
        if self.count >= self.max_events {
            return false;
        }
        
        // Record new event
        self.timestamps[self.head] = timestamp_ms;
        self.head = (self.head + 1) % self.timestamps.len();
        self.count += 1;
        
        true
    }
    
    #[inline(always)]
    pub fn get_count(&self) -> u32 {
        self.count
    }
    
    #[inline(always)]
    pub fn remaining_capacity(&self) -> u32 {
        self.max_events.saturating_sub(self.count)
    }
    
    pub fn reset(&mut self) {
        self.count = 0;
        self.head = 0;
    }
}

/// Cancel-Replace Optimizer
pub struct CancelReplaceOptimizer {
    config: CancelReplaceConfig,
    /// Fill probability calculator
    fill_calc: FillProbabilityMarkov,
    /// Rate limiter
    rate_limiter: SlidingWindowRateLimiter,
    /// Last cancel timestamp (ms)
    last_cancel_ms: u64,
    /// Consecutive cancel count
    consecutive_cancels: u32,
    /// Current order price (ticks)
    current_price_ticks: i64,
    /// Order exists flag
    has_order: bool,
}

impl CancelReplaceOptimizer {
    pub fn new(config: CancelReplaceConfig) -> Result<Self, CancelReplaceError> {
        if config.min_fill_probability < 0.0 || config.min_fill_probability > 1.0 {
            return Err(CancelReplaceError::InvalidThreshold);
        }
        
        let fill_calc = FillProbabilityMarkov::new(
            crate::queue::fill_probability_markov::MarkovConfig::default()
        ).map_err(|_| CancelReplaceError::InvalidThreshold)?;
        
        Ok(Self {
            config,
            fill_calc,
            rate_limiter: SlidingWindowRateLimiter::new(
                config.rate_limit_window_ms,
                config.max_messages_per_second,
            ),
            last_cancel_ms: 0,
            consecutive_cancels: 0,
            current_price_ticks: 0,
            has_order: false,
        })
    }
    
    /// Evaluate whether to cancel/replace order
    #[inline(always)]
    pub fn evaluate(
        &mut self,
        position: &QueuePosition,
        optimal_price_ticks: i64,
        current_time_ms: u64,
    ) -> CancelReplaceDecision {
        // Check rate limit first
        if !self.check_rate_limit(current_time_ms) {
            return CancelReplaceDecision::hold("Rate limit exceeded")
                .with_cooldown(self.config.rate_limit_window_ms);
        }
        
        // Check cooldown after recent cancel
        let time_since_cancel = current_time_ms.saturating_sub(self.last_cancel_ms);
        if time_since_cancel < self.config.cancel_cooldown_ms {
            return CancelReplaceDecision::hold("In cancel cooldown")
                .with_cooldown(self.config.cancel_cooldown_ms - time_since_cancel);
        }
        
        // Check consecutive cancel limit
        if self.consecutive_cancels >= self.config.max_consecutive_cancels {
            return CancelReplaceDecision::hold("Max consecutive cancels reached")
                .with_cooldown(1000); // 1 second forced pause
        }
        
        // If no order, consider placing new one
        if !self.has_order {
            self.current_price_ticks = optimal_price_ticks;
            self.has_order = true;
            return CancelReplaceDecision::new_order("New order placement");
        }
        
        // Calculate fill probability
        let fill_prob = self.fill_calc.calculate_fill_probability(position);
        
        // Check if fill probability is too low
        if fill_prob < self.config.min_fill_probability {
            return self.handle_low_probability(optimal_price_ticks, current_time_ms);
        }
        
        // Check if price needs updating
        let price_diff = (optimal_price_ticks - self.current_price_ticks).abs();
        if price_diff >= self.config.min_price_change_ticks {
            // Price moved enough to justify replace
            if self.rate_limiter.try_record(current_time_ms) {
                self.current_price_ticks = optimal_price_ticks;
                self.consecutive_cancels = 0;
                return CancelReplaceDecision::replace(
                    optimal_price_ticks,
                    "Price update",
                );
            } else {
                return CancelReplaceDecision::hold("Rate limited")
                    .with_cooldown(self.config.rate_limit_window_ms);
            }
        }
        
        CancelReplaceDecision::hold("No action needed")
    }
    
    /// Handle low fill probability scenario
    #[inline(always)]
    fn handle_low_probability(
        &mut self,
        optimal_price_ticks: i64,
        current_time_ms: u64,
    ) -> CancelReplaceDecision {
        // Consider moving to better price or canceling
        
        let price_diff = (optimal_price_ticks - self.current_price_ticks).abs();
        
        if price_diff >= self.config.min_price_change_ticks 
            && self.rate_limiter.try_record(current_time_ms) 
        {
            // Move to new price
            self.current_price_ticks = optimal_price_ticks;
            self.consecutive_cancels = 0;
            self.last_cancel_ms = current_time_ms;
            
            CancelReplaceDecision::replace(optimal_price_ticks, "Low fill probability - repricing")
        } else {
            // Just cancel
            if self.rate_limiter.try_record(current_time_ms) {
                self.consecutive_cancels += 1;
                self.last_cancel_ms = current_time_ms;
                self.has_order = false;
                
                CancelReplaceDecision::cancel("Low fill probability")
            } else {
                CancelReplaceDecision::hold("Rate limited")
                    .with_cooldown(self.config.rate_limit_window_ms)
            }
        }
    }
    
    /// Check rate limit status
    #[inline(always)]
    fn check_rate_limit(&mut self, current_time_ms: u64) -> bool {
        self.rate_limiter.get_count() < self.config.max_messages_per_second
    }
    
    /// Notify of order fill
    #[inline(always)]
    pub fn notify_fill(&mut self) {
        self.has_order = false;
        self.consecutive_cancels = 0;
    }
    
    /// Notify of order cancel
    #[inline(always)]
    pub fn notify_cancel(&mut self) {
        self.has_order = false;
    }
    
    /// Get current rate limit status
    #[inline(always)]
    pub fn rate_limit_status(&self) -> (u32, u32) {
        (self.rate_limiter.get_count(), self.config.max_messages_per_second)
    }
    
    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.rate_limiter.reset();
        self.consecutive_cancels = 0;
        self.has_order = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_hold() {
        let config = CancelReplaceConfig::default();
        let mut optimizer = CancelReplaceOptimizer::new(config).unwrap();
        
        let position = QueuePosition::new(100, 1000, 100, 1, 0.5);
        
        // Good fill probability should hold
        let decision = optimizer.evaluate(&position, 100, 1000);
        assert_eq!(decision.action, CancelReplaceAction::Hold);
    }
    
    #[test]
    fn test_rate_limiting() {
        let config = CancelReplaceConfig {
            max_messages_per_second: 5,
            ..Default::default()
        };
        let mut optimizer = CancelReplaceOptimizer::new(config).unwrap();
        
        // Exhaust rate limit
        for i in 0..5 {
            let position = QueuePosition::new(0, 100, 100, 1, 0.9);
            optimizer.evaluate(&position, 100 + i as i64, 1000 + i);
        }
        
        // Should be rate limited now
        let position = QueuePosition::new(0, 100, 100, 1, 0.9);
        let decision = optimizer.evaluate(&position, 150, 1005);
        assert!(decision.cooldown_remaining_ms > 0);
    }
}
