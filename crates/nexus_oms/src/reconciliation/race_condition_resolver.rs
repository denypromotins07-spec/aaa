//! Race Condition Resolver for Cancel/Replace scenarios
//! 
//! Mathematically resolves conflicts when cancel requests and fill reports
//! arrive in conflicting order due to network latency.

use crate::state_machine::order_state::{OrderId, OrderState, OrderStateMachine};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Nanosecond-precision timestamp for ordering events
pub type TimestampNs = u64;

/// Pending cancel request with metadata
#[derive(Debug, Clone)]
pub struct PendingCancel {
    pub order_id: OrderId,
    /// Local timestamp when cancel was requested (nanoseconds since epoch)
    pub request_timestamp_ns: TimestampNs,
    /// Exchange timestamp when cancel was received (if known)
    pub exchange_timestamp_ns: Option<TimestampNs>,
    /// Sequence number of cancel request
    pub sequence_num: u64,
    /// Registered at (local time for timeout tracking)
    pub registered_at: Instant,
}

/// Fill event that may conflict with pending cancels
#[derive(Debug, Clone)]
pub struct FillEvent {
    pub order_id: OrderId,
    /// Exchange timestamp when fill occurred (authoritative)
    pub transact_time_ns: TimestampNs,
    /// Local timestamp when fill was received
    pub received_at_ns: TimestampNs,
    pub fill_quantity: i64,
    pub fill_price: i64,
    pub trade_id: u64,
}

/// Result of race condition resolution
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RaceConditionResolution {
    /// No conflict detected
    NoConflict,
    /// Fill is authoritative, cancel is phantom
    FillAuthoritative {
        reason: &'static str,
        fill_transact_time_ns: TimestampNs,
        cancel_request_time_ns: TimestampNs,
    },
    /// Cancel is authoritative, fill should not have occurred
    CancelAuthoritative {
        reason: &'static str,
        cancel_exchange_time_ns: TimestampNs,
        fill_transact_time_ns: TimestampNs,
    },
    /// Unable to determine - use default behavior (fill wins)
    Indeterminate {
        reason: &'static str,
    },
}

/// Race condition resolver using mathematical timestamp comparison
pub struct RaceConditionResolver {
    /// Pending cancel requests awaiting confirmation
    pending_cancels: VecDeque<PendingCancel>,
    /// Recent fill events for conflict detection
    recent_fills: VecDeque<FillEvent>,
    /// Maximum history to retain (prevents unbounded growth)
    max_history_size: usize,
    /// Timeout for pending cancels
    cancel_timeout: Duration,
    /// Statistics
    stats: ResolverStats,
}

#[derive(Debug, Clone, Default)]
pub struct ResolverStats {
    pub total_conflicts_detected: u64,
    pub fills_won: u64,
    pub cancels_won: u64,
    pub indeterminate_resolutions: u64,
    pub timeouts: u64,
}

impl RaceConditionResolver {
    pub fn new(max_history_size: usize, cancel_timeout_ms: u64) -> Self {
        Self {
            pending_cancels: VecDeque::with_capacity(max_history_size / 4),
            recent_fills: VecDeque::with_capacity(max_history_size / 4),
            max_history_size,
            cancel_timeout: Duration::from_millis(cancel_timeout_ms),
            stats: ResolverStats::default(),
        }
    }

    /// Register a new cancel request
    pub fn register_cancel(&mut self, cancel: PendingCancel) {
        self.pending_cancels.push_back(cancel);
        self.maintain_history();
        self.cleanup_expired();
    }

    /// Register a fill event and check for conflicts
    pub fn register_fill(&mut self, fill: FillEvent) -> RaceConditionResolution {
        // Check if there's a pending cancel for this order
        let resolution = if let Some(pending_cancel) = self.find_pending_cancel(fill.order_id) {
            self.resolve_conflict(pending_cancel, &fill)
        } else {
            RaceConditionResolution::NoConflict
        };

        // Record the fill
        self.recent_fills.push_back(fill);
        self.maintain_history();

        if resolution != RaceConditionResolution::NoConflict {
            self.stats.total_conflicts_detected += 1;
            match resolution {
                RaceConditionResolution::FillAuthoritative { .. } => self.stats.fills_won += 1,
                RaceConditionResolution::CancelAuthoritative { .. } => self.stats.cancels_won += 1,
                RaceConditionResolution::Indeterminate { .. } => self.stats.indeterminate_resolutions += 1,
                _ => {}
            }
        }

        resolution
    }

    /// Find pending cancel for an order
    fn find_pending_cancel(&self, order_id: OrderId) -> Option<&PendingCancel> {
        self.pending_cancels.iter().find(|pc| pc.order_id == order_id)
    }

    /// Resolve conflict between cancel and fill using mathematical rules
    fn resolve_conflict(&self, cancel: &PendingCancel, fill: &FillEvent) -> RaceConditionResolution {
        // Rule 1: If fill's transact_time (exchange time) is before cancel's request time,
        //         the fill occurred before the cancel was even sent -> Fill wins
        if fill.transact_time_ns < cancel.request_timestamp_ns {
            return RaceConditionResolution::FillAuthoritative {
                reason: "Fill transact time precedes cancel request",
                fill_transact_time_ns: fill.transact_time_ns,
                cancel_request_time_ns: cancel.request_timestamp_ns,
            };
        }

        // Rule 2: If cancel has exchange timestamp and it's before fill's transact time,
        //         the cancel was processed first -> Cancel wins (rare, indicates late fill)
        if let Some(cancel_exchange_time) = cancel.exchange_timestamp_ns {
            if cancel_exchange_time < fill.transact_time_ns {
                return RaceConditionResolution::CancelAuthoritative {
                    reason: "Cancel exchange time precedes fill transact time",
                    cancel_exchange_time_ns: cancel_exchange_time,
                    fill_transact_time_ns: fill.transact_time_ns,
                };
            }
        }

        // Rule 3: If fill's transact time equals or exceeds cancel request time,
        //         but we don't have cancel exchange time, assume fill occurred
        //         before exchange processed the cancel -> Fill wins
        // This is the common case: we send cancel, but matching engine already matched
        if fill.transact_time_ns >= cancel.request_timestamp_ns {
            return RaceConditionResolution::FillAuthoritative {
                reason: "Fill occurred during cancel propagation window",
                fill_transact_time_ns: fill.transact_time_ns,
                cancel_request_time_ns: cancel.request_timestamp_ns,
            };
        }

        // Fallback: Indeterminate (should not reach here with above rules)
        RaceConditionResolution::Indeterminate {
            reason: "Unable to determine ordering",
        }
    }

    /// Get the authoritative action when a conflict is detected
    pub fn get_authoritative_action(
        &self,
        resolution: &RaceConditionResolution,
    ) -> AuthoritativeAction {
        match resolution {
            RaceConditionResolution::NoConflict => AuthoritativeAction::None,
            RaceConditionResolution::FillAuthoritative { .. } => {
                AuthoritativeAction::AcceptFill
            }
            RaceConditionResolution::CancelAuthoritative { .. } => {
                AuthoritativeAction::RejectFill
            }
            RaceConditionResolution::Indeterminate { .. } => {
                // Default: accept fill (safer than rejecting valid fills)
                AuthoritativeAction::AcceptFill
            }
        }
    }

    /// Clean up expired pending cancels
    fn cleanup_expired(&mut self) {
        let now = Instant::now();
        let initial_len = self.pending_cancels.len();
        
        self.pending_cancels.retain(|pc| {
            now.duration_since(pc.registered_at) < self.cancel_timeout
        });

        let removed = initial_len - self.pending_cancels.len();
        self.stats.timeouts += removed as u64;
    }

    /// Maintain bounded history
    fn maintain_history(&mut self) {
        while self.pending_cancels.len() > self.max_history_size / 2 {
            self.pending_cancels.pop_front();
        }
        while self.recent_fills.len() > self.max_history_size / 2 {
            self.recent_fills.pop_front();
        }
    }

    /// Remove a pending cancel after resolution
    pub fn remove_pending_cancel(&mut self, order_id: OrderId) {
        self.pending_cancels.retain(|pc| pc.order_id != order_id);
    }

    /// Get resolver statistics
    pub fn get_stats(&self) -> ResolverStats {
        self.stats.clone()
    }

    /// Reset statistics (for periodic reporting)
    pub fn reset_stats(&mut self) -> ResolverStats {
        let stats = self.stats.clone();
        self.stats = ResolverStats::default();
        stats
    }
}

/// Action to take based on race condition resolution
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthoritativeAction {
    /// No action needed
    None,
    /// Accept the fill and update position
    AcceptFill,
    /// Reject the fill (cancel was processed first)
    RejectFill,
    /// Wait for more information
    WaitForConfirmation,
}

/// Helper for creating pending cancels with current timestamp
impl PendingCancel {
    pub fn new(order_id: OrderId, sequence_num: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as TimestampNs;

        Self {
            order_id,
            request_timestamp_ns: now,
            exchange_timestamp_ns: None,
            sequence_num,
            registered_at: Instant::now(),
        }
    }

    pub fn with_exchange_time(mut self, exchange_time_ns: TimestampNs) -> Self {
        self.exchange_timestamp_ns = Some(exchange_time_ns);
        self
    }
}

/// Helper for creating fill events
impl FillEvent {
    pub fn new(
        order_id: OrderId,
        transact_time_ns: TimestampNs,
        fill_quantity: i64,
        fill_price: i64,
        trade_id: u64,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as TimestampNs;

        Self {
            order_id,
            transact_time_ns,
            received_at_ns: now,
            fill_quantity,
            fill_price,
            trade_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_before_cancel_request() {
        let mut resolver = RaceConditionResolver::new(100, 5000);

        // Create cancel request at time T
        let cancel = PendingCancel::new(OrderId(1), 1);
        let cancel_time = cancel.request_timestamp_ns;

        // Create fill that occurred BEFORE cancel was requested
        let fill = FillEvent::new(
            OrderId(1),
            cancel_time - 1_000_000, // 1ms before cancel request
            1000,
            50000_000,
            999,
        );

        resolver.register_cancel(cancel);
        let resolution = resolver.register_fill(fill);

        assert_eq!(resolution, RaceConditionResolution::FillAuthoritative {
            reason: "Fill transact time precedes cancel request",
            fill_transact_time_ns: fill.transact_time_ns,
            cancel_request_time_ns: cancel_time,
        });

        let action = resolver.get_authoritative_action(&resolution);
        assert_eq!(action, AuthoritativeAction::AcceptFill);
    }

    #[test]
    fn test_fill_during_cancel_propagation() {
        let mut resolver = RaceConditionResolver::new(100, 5000);

        // Create cancel request
        let mut cancel = PendingCancel::new(OrderId(1), 1);
        let cancel_time = cancel.request_timestamp_ns;

        // Create fill that occurred AFTER cancel request but BEFORE cancel ack
        // This simulates the race condition: we sent cancel, but exchange matched first
        let fill = FillEvent::new(
            OrderId(1),
            cancel_time + 500_000, // 0.5ms after cancel request
            1000,
            50000_000,
            999,
        );

        resolver.register_cancel(cancel);
        let resolution = resolver.register_fill(fill);

        // Fill should win because it occurred during the propagation window
        assert!(matches!(resolution, RaceConditionResolution::FillAuthoritative { .. }));
        
        let action = resolver.get_authoritative_action(&resolution);
        assert_eq!(action, AuthoritativeAction::AcceptFill);
    }

    #[test]
    fn test_no_conflict_when_no_pending_cancel() {
        let mut resolver = RaceConditionResolver::new(100, 5000);

        let fill = FillEvent::new(
            OrderId(1),
            1234567890000,
            1000,
            50000_000,
            999,
        );

        let resolution = resolver.register_fill(fill);
        assert_eq!(resolution, RaceConditionResolution::NoConflict);
    }

    #[test]
    fn test_statistics_tracking() {
        let mut resolver = RaceConditionResolver::new(100, 5000);

        // Register cancel
        let cancel = PendingCancel::new(OrderId(1), 1);
        let cancel_time = cancel.request_timestamp_ns;
        resolver.register_cancel(cancel);

        // Register conflicting fill
        let fill = FillEvent::new(
            OrderId(1),
            cancel_time + 500_000,
            1000,
            50000_000,
            999,
        );
        resolver.register_fill(fill);

        let stats = resolver.get_stats();
        assert_eq!(stats.total_conflicts_detected, 1);
        assert_eq!(stats.fills_won, 1);
        assert_eq!(stats.cancels_won, 0);
    }
}
