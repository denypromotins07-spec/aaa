//! Chapter 1: Reconnection State Machine
//!
//! This module provides a sophisticated state machine for managing
//! WebSocket reconnection logic, including subscription management
//! and sequence tracking across reconnects.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tracing::{debug, info, warn};

/// Reconnection phase
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconnectPhase {
    /// Initial connection attempt
    InitialConnect,
    /// Active and receiving data
    Active,
    /// Connection lost, preparing to reconnect
    Disconnected,
    /// Waiting before reconnect attempt
    BackoffWait,
    /// Performing reconnection handshake
    Reconnecting,
    /// Re-subscribing to channels after reconnect
    Resubscribing,
    /// Validating sequence continuity after reconnect
    SequenceValidation,
    /// Fatal error, manual intervention required
    FatalError,
}

/// Subscription channel definition
#[derive(Debug, Clone)]
pub struct SubscriptionChannel {
    pub name: String,
    pub params: Option<String>,
    pub priority: u8, // Lower = higher priority for resubscription
}

/// State machine for reconnection management
pub struct ReconnectStateMachine {
    current_phase: Arc<std::sync::Mutex<ReconnectPhase>>,
    subscriptions: Vec<SubscriptionChannel>,
    last_sequence_id: Arc<AtomicU64>,
    reconnect_count: Arc<AtomicUsize>,
    max_reconnect_attempts: usize,
    sequence_gap_tolerance: u64,
}

impl ReconnectStateMachine {
    pub fn new(max_reconnect_attempts: usize, sequence_gap_tolerance: u64) -> Self {
        Self {
            current_phase: Arc::new(std::sync::Mutex::new(ReconnectPhase::InitialConnect)),
            subscriptions: Vec::new(),
            last_sequence_id: Arc::new(AtomicU64::new(0)),
            reconnect_count: Arc::new(AtomicUsize::new(0)),
            max_reconnect_attempts,
            sequence_gap_tolerance,
        }
    }

    /// Add a subscription channel
    pub fn add_subscription(&mut self, channel: SubscriptionChannel) {
        self.subscriptions.push(channel);
    }

    /// Get current phase
    pub fn get_phase(&self) -> ReconnectPhase {
        *self.current_phase.lock().unwrap()
    }

    /// Set current phase
    pub fn set_phase(&self, phase: ReconnectPhase) {
        let mut guard = self.current_phase.lock().unwrap();
        debug!("State transition: {:?} -> {:?}", *guard, phase);
        *guard = phase;
    }

    /// Record the last seen sequence ID
    pub fn record_sequence(&self, sequence_id: u64) {
        self.last_sequence_id.store(sequence_id, Ordering::Release);
    }

    /// Get last recorded sequence ID
    pub fn get_last_sequence(&self) -> u64 {
        self.last_sequence_id.load(Ordering::Acquire)
    }

    /// Increment reconnect counter
    pub fn increment_reconnect(&self) -> usize {
        let count = self.reconnect_count.fetch_add(1, Ordering::Relaxed) + 1;
        info!("Reconnect attempt #{}", count);
        count
    }

    /// Check if we've exceeded max reconnect attempts
    pub fn should_abort(&self) -> bool {
        self.reconnect_count.load(Ordering::Relaxed) >= self.max_reconnect_attempts
    }

    /// Reset reconnect counter (called on successful sustained connection)
    pub fn reset_reconnect_count(&self) {
        self.reconnect_count.store(0, Ordering::Relaxed);
        info!("Reconnect counter reset after stable connection");
    }

    /// Validate sequence continuity after reconnect
    /// Returns true if sequence is valid or within tolerance
    pub fn validate_sequence_continuity(&self, new_sequence: u64) -> bool {
        let last = self.get_last_sequence();
        
        if new_sequence == 0 {
            // Fresh start, always valid
            return true;
        }
        
        if last == 0 {
            // No previous sequence recorded
            return true;
        }
        
        // Check if new sequence is ahead (normal case after gap)
        if new_sequence > last {
            let gap = new_sequence - last;
            if gap <= self.sequence_gap_tolerance {
                debug!("Sequence gap {} within tolerance {}", gap, self.sequence_gap_tolerance);
                true
            } else {
                warn!("Large sequence gap detected: {} -> {} (gap={})", last, new_sequence, gap);
                false
            }
        } else if new_sequence == last {
            // Duplicate, acceptable
            true
        } else {
            // Sequence went backwards - definite problem
            warn!("Sequence regression detected: {} -> {}", last, new_sequence);
            false
        }
    }

    /// Get subscriptions in priority order for resubscription
    pub fn get_prioritized_subscriptions(&self) -> Vec<&SubscriptionChannel> {
        let mut subs: Vec<&SubscriptionChannel> = self.subscriptions.iter().collect();
        subs.sort_by_key(|s| s.priority);
        subs
    }

    /// Calculate backoff delay based on reconnect count
    pub fn calculate_backoff(&self, base_delay_ms: u64, max_delay_ms: u64) -> Duration {
        let count = self.reconnect_count.load(Ordering::Relaxed) as u64;
        
        // Exponential backoff: base * 2^count
        let delay = base_delay_ms.saturating_mul(1u64.saturating_shl(count as u32));
        let delay = delay.min(max_delay_ms);
        
        // Add jitter (±10%)
        let jitter = (delay as f64 * 0.1 * rand_jitter()) as u64;
        let delay_with_jitter = delay.saturating_add(jitter).min(max_delay_ms);
        
        Duration::from_millis(delay_with_jitter)
    }

    /// Transition to fatal error state
    pub fn enter_fatal_error(&self, reason: &str) {
        warn!("Entering FATAL_ERROR state: {}", reason);
        self.set_phase(ReconnectPhase::FatalError);
    }

    /// Check if state machine is in terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(self.get_phase(), ReconnectPhase::FatalError)
    }

    /// Get statistics
    pub fn get_stats(&self) -> ReconnectStats {
        ReconnectStats {
            reconnect_count: self.reconnect_count.load(Ordering::Relaxed),
            last_sequence: self.get_last_sequence(),
            phase: self.get_phase(),
            max_attempts_exceeded: self.should_abort(),
        }
    }
}

/// Statistics snapshot
#[derive(Debug, Clone)]
pub struct ReconnectStats {
    pub reconnect_count: usize,
    pub last_sequence: u64,
    pub phase: ReconnectPhase,
    pub max_attempts_exceeded: bool,
}

/// Simple jitter generator
fn rand_jitter() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as f64;
    ((nanos / 1_000_000_000.0) - 0.5) * 2.0
}

// SAFETY: State machine uses internal synchronization
unsafe impl Send for ReconnectStateMachine {}
unsafe impl Sync for ReconnectStateMachine {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_transitions() {
        let sm = ReconnectStateMachine::new(5, 100);
        
        assert_eq!(sm.get_phase(), ReconnectPhase::InitialConnect);
        
        sm.set_phase(ReconnectPhase::Active);
        assert_eq!(sm.get_phase(), ReconnectPhase::Active);
        
        sm.set_phase(ReconnectPhase::Disconnected);
        assert_eq!(sm.get_phase(), ReconnectPhase::Disconnected);
    }

    #[test]
    fn test_sequence_validation() {
        let sm = ReconnectStateMachine::new(5, 100);
        
        // Initial sequence
        sm.record_sequence(100);
        assert_eq!(sm.get_last_sequence(), 100);
        
        // Normal progression
        assert!(sm.validate_sequence_continuity(101));
        sm.record_sequence(101);
        
        // Gap within tolerance
        assert!(sm.validate_sequence_continuity(150));
        
        // Gap too large
        assert!(!sm.validate_sequence_continuity(300));
    }

    #[test]
    fn test_backoff_calculation() {
        let sm = ReconnectStateMachine::new(5, 30000);
        
        let first = sm.calculate_backoff(100, 30000);
        sm.increment_reconnect();
        
        let second = sm.calculate_backoff(100, 30000);
        
        // Second should be roughly double first (allowing for jitter)
        assert!(second.as_millis() >= first.as_millis());
    }

    #[test]
    fn test_max_reconnect_check() {
        let sm = ReconnectStateMachine::new(3, 100);
        
        assert!(!sm.should_abort());
        
        sm.increment_reconnect();
        sm.increment_reconnect();
        sm.increment_reconnect();
        
        assert!(sm.should_abort());
    }

    #[test]
    fn test_subscription_priority() {
        let mut sm = ReconnectStateMachine::new(5, 100);
        
        sm.add_subscription(SubscriptionChannel {
            name: "trade".to_string(),
            params: None,
            priority: 2,
        });
        
        sm.add_subscription(SubscriptionChannel {
            name: "depth".to_string(),
            params: None,
            priority: 1,
        });
        
        let sorted = sm.get_prioritized_subscriptions();
        assert_eq!(sorted[0].name, "depth");
        assert_eq!(sorted[1].name, "trade");
    }
}
