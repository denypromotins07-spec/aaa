//! Execution Report Handler for OMS Reconciliation
//! 
//! Atomically processes execution reports from the exchange and updates order state.

use crate::state_machine::order_state::{
    Order, OrderId, OrderState, OrderStateMachine, ExecutionId, Side,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Execution report types received from exchange
#[derive(Debug, Clone)]
pub enum ExecutionReportType {
    /// New order acknowledged
    NewOrderAck,
    /// Order canceled by exchange
    CancelAck,
    /// Order rejected by exchange
    Reject,
    /// Partial fill
    Trade {
        fill_quantity: i64,
        fill_price: i64,
        trade_id: u64,
        is_buyer_maker: bool, // true if we were the seller (maker)
    },
    /// Order status update (no fill)
    StatusUpdate,
}

/// Execution report message from exchange
#[derive(Debug, Clone)]
pub struct ExecutionReport {
    pub report_type: ExecutionReportType,
    pub order_id: OrderId,
    pub exchange_order_id: u64,
    pub symbol: [u8; 12],
    pub side: Side,
    pub timestamp_ns: u64,
    pub sequence_num: u64,
}

impl ExecutionReport {
    pub fn new_trade(
        order_id: OrderId,
        exchange_order_id: u64,
        symbol: [u8; 12],
        side: Side,
        fill_quantity: i64,
        fill_price: i64,
        trade_id: u64,
        is_buyer_maker: bool,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        Self {
            report_type: ExecutionReportType::Trade {
                fill_quantity,
                fill_price,
                trade_id,
                is_buyer_maker,
            },
            order_id,
            exchange_order_id,
            symbol,
            side,
            timestamp_ns: now,
            sequence_num: 0,
        }
    }

    pub fn new_reject(order_id: OrderId, exchange_order_id: u64, symbol: [u8; 12], side: Side) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        Self {
            report_type: ExecutionReportType::Reject,
            order_id,
            exchange_order_id,
            symbol,
            side,
            timestamp_ns: now,
            sequence_num: 0,
        }
    }
}

/// Result of processing an execution report
#[derive(Debug)]
pub enum ExecutionReportResult {
    /// Report processed successfully
    Success {
        order_id: OrderId,
        new_state: OrderState,
        fill_quantity: Option<i64>,
    },
    /// Order not found in OMS
    OrderNotFound(OrderId),
    /// Invalid state transition
    InvalidTransition {
        order_id: OrderId,
        from: OrderState,
        to: OrderState,
    },
    /// Duplicate report (already processed)
    Duplicate {
        order_id: OrderId,
        sequence_num: u64,
    },
    /// Report rejected due to race condition resolution
    RaceConditionResolved {
        order_id: OrderId,
        reason: &'static str,
    },
}

/// Tracks processed sequence numbers to detect duplicates
struct SequenceTracker {
    last_sequence: u64,
    seen_sequences: Vec<u64>, // Circular buffer for recent sequences
    max_history: usize,
}

impl SequenceTracker {
    fn new(max_history: usize) -> Self {
        Self {
            last_sequence: 0,
            seen_sequences: Vec::with_capacity(max_history),
            max_history,
        }
    }

    /// Check if sequence is duplicate
    fn is_duplicate(&self, seq: u64) -> bool {
        if seq <= self.last_sequence && self.last_sequence - seq < self.max_history as u64 {
            return self.seen_sequences.contains(&seq);
        }
        false
    }

    /// Record a new sequence
    fn record(&mut self, seq: u64) {
        if seq > self.last_sequence {
            self.last_sequence = seq;
            if self.seen_sequences.len() >= self.max_history {
                self.seen_sequences.remove(0);
            }
            self.seen_sequences.push(seq);
        }
    }
}

/// Execution Report Reconciler - processes reports and updates OMS state
pub struct ExecutionReportReconciler {
    sequence_tracker: SequenceTracker,
    stats: ReconciliationStats,
    /// Pending cancel requests (for race condition detection)
    pending_cancels: Vec<(OrderId, Instant)>,
    /// Maximum age for pending cancels before timeout
    cancel_timeout_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct ReconciliationStats {
    pub total_reports_processed: u64,
    pub fills_processed: u64,
    pub cancels_processed: u64,
    pub rejects_processed: u64,
    pub duplicates_detected: u64,
    pub race_conditions_resolved: u64,
    pub errors: u64,
}

impl ExecutionReportReconciler {
    pub fn new(cancel_timeout_ms: u64) -> Self {
        Self {
            sequence_tracker: SequenceTracker::new(1000),
            stats: ReconciliationStats::default(),
            pending_cancels: Vec::new(),
            cancel_timeout_ms,
        }
    }

    /// Register a cancel request (for race condition detection)
    pub fn register_cancel_request(&mut self, order_id: OrderId) {
        self.pending_cancels.push((order_id, Instant::now()));
        self.cleanup_expired_cancels();
    }

    /// Clean up expired cancel registrations
    fn cleanup_expired_cancels(&mut self) {
        let now = Instant::now();
        let timeout = Duration::from_millis(self.cancel_timeout_ms);
        self.pending_cancels.retain(|(_, registered_at)| now.duration_since(*registered_at) < timeout);
    }

    /// Check if a cancel is pending for an order
    fn has_pending_cancel(&self, order_id: OrderId) -> bool {
        self.pending_cancels.iter().any(|&(oid, _)| oid == order_id)
    }

    /// Remove pending cancel registration
    fn remove_pending_cancel(&mut self, order_id: OrderId) {
        self.pending_cancels.retain(|&(oid, _)| oid != order_id);
    }

    /// Process an execution report and update OMS state
    pub fn process_report(
        &mut self,
        report: ExecutionReport,
        oms: &mut OrderStateMachine,
    ) -> ExecutionReportResult {
        self.stats.total_reports_processed += 1;

        // Check for duplicate
        if self.sequence_tracker.is_duplicate(report.sequence_num) {
            self.stats.duplicates_detected += 1;
            return ExecutionReportResult::Duplicate {
                order_id: report.order_id,
                sequence_num: report.sequence_num,
            };
        }

        // Handle based on report type
        let result = match report.report_type {
            ExecutionReportType::NewOrderAck => {
                self.handle_new_order_ack(report, oms)
            }
            ExecutionReportType::CancelAck => {
                self.handle_cancel_ack(report, oms)
            }
            ExecutionReportType::Reject => {
                self.handle_reject(report, oms)
            }
            ExecutionReportType::Trade { fill_quantity, fill_price, trade_id, is_buyer_maker } => {
                self.handle_trade(report, fill_quantity, fill_price, trade_id, is_buyer_maker, oms)
            }
            ExecutionReportType::StatusUpdate => {
                self.handle_status_update(report, oms)
            }
        };

        // Record sequence if successful
        if !matches!(result, ExecutionReportResult::Duplicate { .. }) {
            self.sequence_tracker.record(report.sequence_num);
        }

        result
    }

    fn handle_new_order_ack(
        &mut self,
        report: ExecutionReport,
        oms: &mut OrderStateMachine,
    ) -> ExecutionReportResult {
        match oms.transition(report.order_id, OrderState::Open) {
            crate::state_machine::order_state::TransitionResult::Success(new_state) => {
                self.stats.fills_processed += 1; // Counting acks as successful events
                ExecutionReportResult::Success {
                    order_id: report.order_id,
                    new_state,
                    fill_quantity: None,
                }
            }
            crate::state_machine::order_state::TransitionResult::InvalidTransition { from, to } => {
                self.stats.errors += 1;
                ExecutionReportResult::InvalidTransition {
                    order_id: report.order_id,
                    from,
                    to,
                }
            }
            crate::state_machine::order_state::TransitionResult::TerminalStateReached(state) => {
                ExecutionReportResult::Success {
                    order_id: report.order_id,
                    new_state: state,
                    fill_quantity: None,
                }
            }
            crate::state_machine::order_state::TransitionResult::OrderNotFound => {
                self.stats.errors += 1;
                ExecutionReportResult::OrderNotFound(report.order_id)
            }
        }
    }

    fn handle_cancel_ack(
        &mut self,
        report: ExecutionReport,
        oms: &mut OrderStateMachine,
    ) -> ExecutionReportResult {
        // Remove from pending cancels
        self.remove_pending_cancel(report.order_id);

        match oms.transition(report.order_id, OrderState::Canceled) {
            crate::state_machine::order_state::TransitionResult::Success(new_state) |
            crate::state_machine::order_state::TransitionResult::TerminalStateReached(new_state) => {
                self.stats.cancels_processed += 1;
                ExecutionReportResult::Success {
                    order_id: report.order_id,
                    new_state,
                    fill_quantity: None,
                }
            }
            crate::state_machine::order_state::TransitionResult::InvalidTransition { from, to } => {
                // This can happen if order was already filled before cancel ack
                self.stats.race_conditions_resolved += 1;
                ExecutionReportResult::RaceConditionResolved {
                    order_id: report.order_id,
                    reason: "Order already in terminal state",
                }
            }
            crate::state_machine::order_state::TransitionResult::OrderNotFound => {
                // Order already purged, ignore
                ExecutionReportResult::RaceConditionResolved {
                    order_id: report.order_id,
                    reason: "Order already purged from OMS",
                }
            }
        }
    }

    fn handle_reject(
        &mut self,
        report: ExecutionReport,
        oms: &mut OrderStateMachine,
    ) -> ExecutionReportResult {
        match oms.transition(report.order_id, OrderState::Rejected) {
            crate::state_machine::order_state::TransitionResult::Success(new_state) |
            crate::state_machine::order_state::TransitionResult::TerminalStateReached(new_state) => {
                self.stats.rejects_processed += 1;
                ExecutionReportResult::Success {
                    order_id: report.order_id,
                    new_state,
                    fill_quantity: None,
                }
            }
            crate::state_machine::order_state::TransitionResult::InvalidTransition { from, to } => {
                self.stats.errors += 1;
                ExecutionReportResult::InvalidTransition {
                    order_id: report.order_id,
                    from,
                    to,
                }
            }
            crate::state_machine::order_state::TransitionResult::OrderNotFound => {
                self.stats.errors += 1;
                ExecutionReportResult::OrderNotFound(report.order_id)
            }
        }
    }

    fn handle_trade(
        &mut self,
        report: ExecutionReport,
        fill_quantity: i64,
        fill_price: i64,
        trade_id: u64,
        _is_buyer_maker: bool,
        oms: &mut OrderStateMachine,
    ) -> ExecutionReportResult {
        // Check for race condition: fill received after cancel request
        if self.has_pending_cancel(report.order_id) {
            // Fill arrived after we requested cancel - this is normal
            // The fill takes precedence, cancel will be resolved as phantom
            self.remove_pending_cancel(report.order_id);
            self.stats.race_conditions_resolved += 1;
        }

        match oms.apply_fill(report.order_id, fill_quantity, fill_price, ExecutionId(trade_id)) {
            Ok(()) => {
                self.stats.fills_processed += 1;
                
                // Get the new state
                let order = oms.get_order(report.order_id).unwrap();
                let new_state = order.state;

                ExecutionReportResult::Success {
                    order_id: report.order_id,
                    new_state,
                    fill_quantity: Some(fill_quantity),
                }
            }
            Err(e) => {
                self.stats.errors += 1;
                // Try to get current state for error reporting
                if let Some(order) = oms.get_order(report.order_id) {
                    ExecutionReportResult::InvalidTransition {
                        order_id: report.order_id,
                        from: order.state,
                        to: OrderState::Filled, // Approximate
                    }
                } else {
                    ExecutionReportResult::OrderNotFound(report.order_id)
                }
            }
        }
    }

    fn handle_status_update(
        &mut self,
        _report: ExecutionReport,
        _oms: &mut OrderStateMachine,
    ) -> ExecutionReportResult {
        // Status updates are informational only, no state change needed
        ExecutionReportResult::Success {
            order_id: _report.order_id,
            new_state: OrderState::Open, // Assume unchanged
            fill_quantity: None,
        }
    }

    /// Get reconciliation statistics
    pub fn get_stats(&self) -> ReconciliationStats {
        self.stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_machine::order_state::{Order, OrderType, TimeInForce};

    #[test]
    fn test_trade_execution_report() {
        let mut oms = OrderStateMachine::with_capacity(100);
        let mut reconciler = ExecutionReportReconciler::new(1000);

        // Create and add order
        let order = Order::new(
            oms.generate_order_id(),
            "BTCUSDT",
            Side::Buy,
            OrderType::Limit,
            TimeInForce::Gtc,
            50000_000,
            100_000,
            12345,
        );
        let order_id = order.id;
        oms.add_order(order).unwrap();
        oms.transition(order_id, OrderState::Open).unwrap();

        // Create trade report
        let mut symbol_buf = [0u8; 12];
        symbol_buf[..7].copy_from_slice(b"BTCUSDT");
        
        let report = ExecutionReport::new_trade(
            order_id,
            99999,
            symbol_buf,
            Side::Buy,
            50_000,
            50000_000,
            1,
            false,
        );

        // Process report
        let result = reconciler.process_report(report, &mut oms);
        
        assert!(matches!(
            result,
            ExecutionReportResult::Success { fill_quantity: Some(50_000), .. }
        ));

        // Verify order state
        let order = oms.get_order(order_id).unwrap();
        assert_eq!(order.state, OrderState::PartiallyFilled);
        assert_eq!(order.filled_quantity, 50_000);
    }

    #[test]
    fn test_race_condition_fill_after_cancel_request() {
        let mut oms = OrderStateMachine::with_capacity(100);
        let mut reconciler = ExecutionReportReconciler::new(1000);

        // Create and add order
        let order = Order::new(
            oms.generate_order_id(),
            "BTCUSDT",
            Side::Buy,
            OrderType::Limit,
            TimeInForce::Gtc,
            50000_000,
            100_000,
            12345,
        );
        let order_id = order.id;
        oms.add_order(order).unwrap();
        oms.transition(order_id, OrderState::Open).unwrap();

        // Register cancel request
        reconciler.register_cancel_request(order_id);

        // Simulate fill arriving before cancel ack
        let mut symbol_buf = [0u8; 12];
        symbol_buf[..7].copy_from_slice(b"BTCUSDT");
        
        let report = ExecutionReport::new_trade(
            order_id,
            99999,
            symbol_buf,
            Side::Buy,
            100_000, // Full fill
            50000_000,
            1,
            false,
        );

        // Process fill - should succeed and resolve race condition
        let result = reconciler.process_report(report, &mut oms);
        
        assert!(matches!(result, ExecutionReportResult::Success { .. }));
        
        // Verify race condition was tracked
        let stats = reconciler.get_stats();
        assert!(stats.race_conditions_resolved >= 1);

        // Order should be filled, not canceled
        let order = oms.get_order(order_id).unwrap();
        assert_eq!(order.state, OrderState::Filled);
    }
}
