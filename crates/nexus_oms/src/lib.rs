//! NEXUS-OMEGA Order Management System (OMS)
//! 
//! Zero-allocation order state machine with race condition resolution
//! for high-frequency trading on cryptocurrency exchanges.

pub mod state_machine;
pub mod reconciliation;

pub use state_machine::{
    OrderId, ExecutionId, Side, OrderType, TimeInForce, OrderState,
    Order, TransitionResult, OrderStateMachine, OmsStats,
};

pub use reconciliation::{
    ExecutionReport, ExecutionReportType, ExecutionReportResult,
    ExecutionReportReconciler, ReconciliationStats,
    RaceConditionResolver, RaceConditionResolution, AuthoritativeAction,
    PendingCancel, FillEvent, TimestampNs, ResolverStats,
};

/// Live Order Management System - main orchestrator
pub struct LiveOrderManagementSystem {
    pub state_machine: OrderStateMachine,
    pub reconciler: ExecutionReportReconciler,
    pub race_resolver: RaceConditionResolver,
}

impl LiveOrderManagementSystem {
    /// Create new OMS with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            state_machine: OrderStateMachine::with_capacity(capacity),
            reconciler: ExecutionReportReconciler::new(1000), // 1 second cancel timeout
            race_resolver: RaceConditionResolver::new(1000, 5000), // 5 second timeout
        }
    }

    /// Add a new order to the OMS
    pub fn add_order(&mut self, order: Order) -> Result<OrderId, &'static str> {
        self.state_machine.add_order(order)
    }

    /// Register a cancel request (for race condition detection)
    pub fn register_cancel_request(&mut self, order_id: OrderId) {
        self.reconciler.register_cancel_request(order_id);
        
        let cancel = PendingCancel::new(order_id, 0);
        self.race_resolver.register_cancel(cancel);
    }

    /// Process an execution report from the exchange
    pub fn process_execution_report(&mut self, report: ExecutionReport) -> ExecutionReportResult {
        // Check for race conditions first
        if let ExecutionReportType::Trade { fill_quantity, fill_price, trade_id, .. } = report.report_type {
            let fill_event = FillEvent::new(
                report.order_id,
                report.timestamp_ns,
                fill_quantity,
                fill_price,
                trade_id,
            );
            
            let resolution = self.race_resolver.register_fill(fill_event);
            
            // If cancel was authoritative, we might want to reject the fill
            // but typically we accept fills as they represent actual trades
            if matches!(resolution, RaceConditionResolution::CancelAuthoritative { .. }) {
                log::warn!("Fill received after cancel processed - possible late fill");
            }
        }

        // Process through reconciler
        self.reconciler.process_report(report, &mut self.state_machine)
    }

    /// Get order by ID
    pub fn get_order(&self, order_id: OrderId) -> Option<&Order> {
        self.state_machine.get_order(order_id)
    }

    /// Get active order count
    pub fn active_count(&self) -> usize {
        self.state_machine.active_count()
    }

    /// Purge terminal orders (call periodically)
    pub fn purge_terminal_orders(&mut self) -> usize {
        self.state_machine.purge_terminal_orders()
    }

    /// Get combined statistics
    pub fn get_stats(&self) -> OmsCombinedStats {
        OmsCombinedStats {
            oms: self.state_machine.get_stats(),
            reconciliation: self.reconciler.get_stats(),
            race_resolution: self.race_resolver.get_stats(),
        }
    }
}

/// Combined OMS statistics
#[derive(Debug, Clone)]
pub struct OmsCombinedStats {
    pub oms: OmsStats,
    pub reconciliation: ReconciliationStats,
    pub race_resolution: ResolverStats,
}

// Re-export types needed for execution reports
use crate::state_machine::order_state::ExecutionReportType as InternalExecutionReportType;
pub type ExecutionReportType = InternalExecutionReportType;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oms_lifecycle() {
        let mut oms = LiveOrderManagementSystem::with_capacity(100);

        // Create order
        let order = Order::new(
            oms.state_machine.generate_order_id(),
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

        // Verify order exists
        assert!(oms.get_order(order_id).is_some());
        assert_eq!(oms.active_count(), 1);
    }
}
