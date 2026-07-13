//! Zero-allocation Order State Machine for NEXUS-OMEGA OMS
//! 
//! Tracks order lifecycle states with atomic transitions and automatic memory cleanup.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Unique order identifier (zero-allocation, stack-friendly)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OrderId(pub u64);

/// Execution report identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionId(pub u64);

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Limit,
    Market,
    StopLimit,
}

/// Time in force
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    Gtc, // Good till cancel
    Fok, // Fill or kill
    Ioc, // Immediate or cancel
}

/// Order state - strictly defined states for the state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderState {
    /// Order sent but not yet acknowledged by exchange
    PendingNew,
    /// Order acknowledged and resting in book
    Open,
    /// Partial fill received
    PartiallyFilled,
    /// Fully filled
    Filled,
    /// Cancelled by user or system
    Canceled,
    /// Rejected by exchange
    Rejected,
}

impl OrderState {
    /// Check if state is terminal (no further transitions allowed)
    #[inline]
    pub fn is_terminal(self) -> bool {
        matches!(self, OrderState::Filled | OrderState::Canceled | OrderState::Rejected)
    }
}

/// Order information stored in zero-allocation format where possible
#[derive(Debug, Clone)]
pub struct Order {
    pub id: OrderId,
    pub symbol: [u8; 12], // Fixed-size buffer for symbol (e.g., "BTCUSDT")
    pub side: Side,
    pub order_type: OrderType,
    pub time_in_force: TimeInForce,
    pub price: i64, // Price in smallest quote unit (avoid f64)
    pub quantity: i64, // Quantity in smallest base unit
    pub filled_quantity: i64,
    pub remaining_quantity: i64,
    pub average_fill_price: i64,
    pub state: OrderState,
    pub created_at_ns: u64, // Nanosecond timestamp
    pub last_updated_ns: u64,
    pub client_order_id: u64,
    pub exchange_order_id: Option<u64>,
    /// Retention time for terminal states before purge
    pub purge_after_ms: u64,
}

impl Order {
    pub fn new(
        id: OrderId,
        symbol: &str,
        side: Side,
        order_type: OrderType,
        time_in_force: TimeInForce,
        price: i64,
        quantity: i64,
        client_order_id: u64,
    ) -> Self {
        let mut symbol_buf = [0u8; 12];
        let bytes = symbol.as_bytes();
        let len = bytes.len().min(12);
        symbol_buf[..len].copy_from_slice(&bytes[..len]);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        Self {
            id,
            symbol: symbol_buf,
            side,
            order_type,
            time_in_force,
            price,
            quantity,
            filled_quantity: 0,
            remaining_quantity: quantity,
            average_fill_price: 0,
            state: OrderState::PendingNew,
            created_at_ns: now,
            last_updated_ns: now,
            client_order_id,
            exchange_order_id: None,
            purge_after_ms: 5000, // Default 5 second retention
        }
    }

    /// Get symbol as string slice
    #[inline]
    pub fn symbol_str(&self) -> &str {
        let end = self.symbol.iter().position(|&b| b == 0).unwrap_or(12);
        std::str::from_utf8(&self.symbol[..end]).unwrap_or("")
    }

    /// Check if order can be canceled
    #[inline]
    pub fn is_cancelable(&self) -> bool {
        matches!(self.state, OrderState::Open | OrderState::PartiallyFilled)
    }

    /// Calculate fill percentage (basis points)
    #[inline]
    pub fn fill_percentage_bps(&self) -> u32 {
        if self.quantity == 0 {
            return 0;
        }
        ((self.filled_quantity * 10000) / self.quantity) as u32
    }
}

/// Result of a state transition attempt
#[derive(Debug)]
pub enum TransitionResult {
    Success(OrderState),
    InvalidTransition { from: OrderState, to: OrderState },
    TerminalStateReached(OrderState),
    OrderNotFound,
}

/// Pre-allocated order state tracker with automatic cleanup
pub struct OrderStateMachine {
    /// Active orders (includes pending, open, partially filled)
    active_orders: HashMap<OrderId, Order>,
    /// Orders pending purge (terminal states awaiting removal)
    purge_queue: Vec<(OrderId, Instant)>,
    /// Monotonic counter for internal IDs
    id_counter: AtomicU64,
    /// Statistics
    total_orders_created: u64,
    total_orders_filled: u64,
    total_orders_rejected: u64,
}

impl OrderStateMachine {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            active_orders: HashMap::with_capacity(capacity),
            purge_queue: Vec::with_capacity(capacity / 4),
            id_counter: AtomicU64::new(1),
            total_orders_created: 0,
            total_orders_filled: 0,
            total_orders_rejected: 0,
        }
    }

    /// Generate unique order ID (thread-safe, monotonic)
    #[inline]
    pub fn generate_order_id(&self) -> OrderId {
        OrderId(self.id_counter.fetch_add(1, Ordering::Relaxed))
    }

    /// Add new order to state machine
    pub fn add_order(&mut self, order: Order) -> Result<OrderId, &'static str> {
        if self.active_orders.contains_key(&order.id) {
            return Err("Order ID already exists");
        }
        
        let id = order.id;
        self.active_orders.insert(id, order);
        self.total_orders_created += 1;
        Ok(id)
    }

    /// Attempt to transition an order to a new state
    pub fn transition(
        &mut self,
        order_id: OrderId,
        new_state: OrderState,
    ) -> TransitionResult {
        let order = match self.active_orders.get_mut(&order_id) {
            Some(o) => o,
            None => return TransitionResult::OrderNotFound,
        };

        let old_state = order.state;

        // Validate transition
        if !Self::is_valid_transition(old_state, new_state) {
            return TransitionResult::InvalidTransition {
                from: old_state,
                to: new_state,
            };
        }

        // Apply transition
        order.state = new_state;
        order.last_updated_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Handle terminal states - schedule for purge
        if new_state.is_terminal() {
            if new_state == OrderState::Filled {
                self.total_orders_filled += 1;
            } else if new_state == OrderState::Rejected {
                self.total_orders_rejected += 1;
            }

            // Schedule purge after retention period
            let purge_time = Instant::now() + Duration::from_millis(order.purge_after_ms);
            self.purge_queue.push((order_id, purge_time));

            return TransitionResult::TerminalStateReached(new_state);
        }

        TransitionResult::Success(new_state)
    }

    /// Validate state transitions according to exchange rules
    #[inline]
    fn is_valid_transition(from: OrderState, to: OrderState) -> bool {
        match from {
            OrderState::PendingNew => {
                matches!(to, OrderState::Open | OrderState::PartiallyFilled | OrderState::Filled | OrderState::Canceled | OrderState::Rejected)
            }
            OrderState::Open => {
                matches!(to, OrderState::PartiallyFilled | OrderState::Filled | OrderState::Canceled)
            }
            OrderState::PartiallyFilled => {
                matches!(to, OrderState::Filled | OrderState::Canceled)
            }
            // Terminal states cannot transition
            OrderState::Filled | OrderState::Canceled | OrderState::Rejected => false,
        }
    }

    /// Update order with fill information
    pub fn apply_fill(
        &mut self,
        order_id: OrderId,
        fill_quantity: i64,
        fill_price: i64,
        execution_id: ExecutionId,
    ) -> Result<(), &'static str> {
        let order = self.active_orders.get_mut(&order_id).ok_or("Order not found")?;

        if order.state == OrderState::Rejected || order.state == OrderState::Canceled {
            return Err("Cannot fill rejected or canceled order");
        }

        // Validate fill quantity
        if fill_quantity <= 0 || fill_quantity > order.remaining_quantity {
            return Err("Invalid fill quantity");
        }

        // Update fill statistics
        let prev_filled_value = order.filled_quantity * order.average_fill_price;
        order.filled_quantity += fill_quantity;
        order.remaining_quantity -= fill_quantity;

        // Update average fill price
        let new_filled_value = prev_filled_value + (fill_quantity * fill_price);
        order.average_fill_price = if order.filled_quantity > 0 {
            new_filled_value / order.filled_quantity
        } else {
            0
        };

        // Transition state based on completion
        let new_state = if order.remaining_quantity == 0 {
            OrderState::Filled
        } else if order.state == OrderState::PendingNew || order.state == OrderState::Open {
            OrderState::PartiallyFilled
        } else {
            order.state // Already PartiallyFilled
        };

        if new_state != order.state {
            order.state = new_state;
            order.last_updated_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;

            if new_state.is_terminal() {
                let purge_time = Instant::now() + Duration::from_millis(order.purge_after_ms);
                self.purge_queue.push((order_id, purge_time));
            }
        }

        Ok(())
    }

    /// Purge terminal orders that have exceeded their retention period
    /// Call this periodically (e.g., every 100ms) to prevent memory leaks
    pub fn purge_terminal_orders(&mut self) -> usize {
        let now = Instant::now();
        let mut purged_count = 0;

        // Find orders ready for purge
        let mut to_remove = Vec::with_capacity(self.purge_queue.len());
        for (i, &(order_id, purge_time)) in self.purge_queue.iter().enumerate() {
            if now >= purge_time {
                to_remove.push(i);
            }
        }

        // Remove in reverse order to maintain indices
        for &idx in to_remove.iter().rev() {
            let (order_id, _) = self.purge_queue.swap_remove(idx);
            if self.active_orders.remove(&order_id).is_some() {
                purged_count += 1;
            }
        }

        purged_count
    }

    /// Get order by ID (immutable reference)
    #[inline]
    pub fn get_order(&self, order_id: OrderId) -> Option<&Order> {
        self.active_orders.get(&order_id)
    }

    /// Get order by ID (mutable reference)
    #[inline]
    pub fn get_order_mut(&mut self, order_id: OrderId) -> Option<&mut Order> {
        self.active_orders.get_mut(&order_id)
    }

    /// Get count of active orders
    #[inline]
    pub fn active_count(&self) -> usize {
        self.active_orders.len()
    }

    /// Get statistics
    pub fn get_stats(&self) -> OmsStats {
        OmsStats {
            total_created: self.total_orders_created,
            total_filled: self.total_orders_filled,
            total_rejected: self.total_orders_rejected,
            active_count: self.active_orders.len(),
            purge_queue_length: self.purge_queue.len(),
        }
    }
}

/// OMS statistics snapshot
#[derive(Debug, Clone)]
pub struct OmsStats {
    pub total_created: u64,
    pub total_filled: u64,
    pub total_rejected: u64,
    pub active_count: usize,
    pub purge_queue_length: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_lifecycle() {
        let mut sm = OrderStateMachine::with_capacity(100);
        
        let order = Order::new(
            sm.generate_order_id(),
            "BTCUSDT",
            Side::Buy,
            OrderType::Limit,
            TimeInForce::Gtc,
            50000_000, // $50,000.00
            100_000,   // 0.001 BTC
            12345,
        );

        let order_id = order.id;
        sm.add_order(order).unwrap();

        // Transition: PendingNew -> Open
        assert!(matches!(
            sm.transition(order_id, OrderState::Open),
            TransitionResult::Success(OrderState::Open)
        ));

        // Apply partial fill
        sm.apply_fill(order_id, 50_000, 50000_000, ExecutionId(1)).unwrap();
        let order = sm.get_order(order_id).unwrap();
        assert_eq!(order.state, OrderState::PartiallyFilled);
        assert_eq!(order.filled_quantity, 50_000);

        // Apply final fill
        sm.apply_fill(order_id, 50_000, 50000_000, ExecutionId(2)).unwrap();
        let order = sm.get_order(order_id).unwrap();
        assert_eq!(order.state, OrderState::Filled);
        assert_eq!(order.remaining_quantity, 0);

        // Verify scheduled for purge
        assert_eq!(sm.purge_queue.len(), 1);
    }

    #[test]
    fn test_memory_cleanup() {
        let mut sm = OrderStateMachine::with_capacity(100);
        
        // Create and fill multiple orders
        for i in 0..10 {
            let mut order = Order::new(
                sm.generate_order_id(),
                "BTCUSDT",
                Side::Buy,
                OrderType::Limit,
                TimeInForce::Gtc,
                50000_000,
                100_000,
                1000 + i,
            );
            order.purge_after_ms = 10; // Very short retention for testing
            let order_id = order.id;
            sm.add_order(order).unwrap();
            sm.transition(order_id, OrderState::Open).unwrap();
            sm.apply_fill(order_id, 100_000, 50000_000, ExecutionId(i)).unwrap();
        }

        assert_eq!(sm.active_count(), 10);
        assert_eq!(sm.purge_queue.len(), 10);

        // Wait for retention period
        std::thread::sleep(Duration::from_millis(20));

        // Purge should remove all
        let purged = sm.purge_terminal_orders();
        assert_eq!(purged, 10);
        assert_eq!(sm.active_count(), 0);
    }

    #[test]
    fn test_invalid_transitions() {
        let mut sm = OrderStateMachine::with_capacity(10);
        
        let order = Order::new(
            sm.generate_order_id(),
            "BTCUSDT",
            Side::Buy,
            OrderType::Limit,
            TimeInForce::Gtc,
            50000_000,
            100_000,
            12345,
        );
        let order_id = order.id;
        sm.add_order(order).unwrap();

        // Try invalid: PendingNew -> Canceled (valid) then Canceled -> Open (invalid)
        sm.transition(order_id, OrderState::Canceled).unwrap();
        
        let result = sm.transition(order_id, OrderState::Open);
        assert!(matches!(
            result,
            TransitionResult::InvalidTransition { .. }
        ));
    }
}
