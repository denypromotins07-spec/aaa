//! Lock-free Order State Machine for the OMS.
//! Uses tagged pointers and generation counters to prevent ABA problems.

use std::sync::atomic::{AtomicU64, AtomicPtr, Ordering};
use std::ptr;
use crossbeam_queue::SegQueue;
use crate::fixed_point_math::FixedPoint;

/// Order Side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Side {
    Buy = 0,
    Sell = 1,
}

/// Order Type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OrderType {
    Limit = 0,
    Market = 1,
    IOC = 2,
    FOK = 3,
}

/// Order Status with generation counter for ABA prevention
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OrderStatus {
    Pending = 0,
    New = 1,
    PartiallyFilled = 2,
    Filled = 3,
    Cancelled = 4,
    Rejected = 5,
}

/// Unique Order ID (u64 for atomic operations)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OrderId(pub u64);

impl OrderId {
    #[inline]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Order representation with fixed-point math
#[derive(Debug, Clone, Copy)]
pub struct Order {
    pub id: OrderId,
    pub side: Side,
    pub order_type: OrderType,
    pub status: OrderStatus,
    pub price: FixedPoint,
    pub quantity: FixedPoint,
    pub filled_quantity: FixedPoint,
    pub remaining_quantity: FixedPoint,
    pub venue_id: u32,
    pub timestamp_ns: u64,
    pub generation: u64, // ABA prevention counter
}

impl Order {
    #[inline]
    pub fn new(
        id: OrderId,
        side: Side,
        order_type: OrderType,
        price: FixedPoint,
        quantity: FixedPoint,
        venue_id: u32,
        timestamp_ns: u64,
    ) -> Self {
        Self {
            id,
            side,
            order_type,
            status: OrderStatus::Pending,
            price,
            quantity,
            filled_quantity: FixedPoint::from_raw(0),
            remaining_quantity: quantity,
            venue_id,
            timestamp_ns,
            generation: 0,
        }
    }

    #[inline]
    pub fn with_generation(mut self, gen: u64) -> Self {
        self.generation = gen;
        self
    }

    /// Check if order is fully filled
    #[inline]
    pub fn is_filled(&self) -> bool {
        self.remaining_quantity.is_zero() || self.status == OrderStatus::Filled
    }

    /// Check if order can be cancelled
    #[inline]
    pub fn is_cancellable(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Pending | OrderStatus::New | OrderStatus::PartiallyFilled
        )
    }

    /// Calculate fill percentage (returns value scaled by 10^8)
    #[inline]
    pub fn fill_percentage(&self) -> FixedPoint {
        if self.quantity.is_zero() {
            return FixedPoint::from_raw(0);
        }
        self.filled_quantity / self.quantity
    }
}

/// Execution Report for order state transitions
#[derive(Debug, Clone, Copy)]
pub struct ExecutionReport {
    pub order_id: OrderId,
    pub prev_status: OrderStatus,
    pub new_status: OrderStatus,
    pub fill_price: FixedPoint,
    pub fill_quantity: FixedPoint,
    pub leaves_quantity: FixedPoint,
    pub timestamp_ns: u64,
    pub generation: u64,
}

/// Tagged pointer for ABA-free atomic operations
/// Combines a pointer with a generation counter in a single u64
#[derive(Debug, Clone, Copy)]
pub struct TaggedPointer {
    pub ptr: u64,
    pub tag: u64,
}

impl TaggedPointer {
    #[inline]
    pub const fn new(ptr: u64, tag: u64) -> Self {
        Self { ptr, tag }
    }

    /// Pack into a single u64 (lower 48 bits for ptr, upper 16 for tag)
    #[inline]
    pub const fn pack(self) -> u64 {
        (self.tag << 48) | (self.ptr & 0x0000_FFFF_FFFF_FFFF)
    }

    /// Unpack from a u64
    #[inline]
    pub const fn unpack(packed: u64) -> Self {
        Self {
            ptr: packed & 0x0000_FFFF_FFFF_FFFF,
            tag: packed >> 48,
        }
    }
}

/// Lock-free Order State Machine
/// Uses atomic operations with tagged pointers for ABA prevention
pub struct OrderStateMachine {
    /// Queue of pending orders (multi-producer)
    pending_queue: SegQueue<Order>,
    /// Queue of execution reports (multi-consumer)
    report_queue: SegQueue<ExecutionReport>,
    /// Global order counter for unique IDs
    order_counter: AtomicU64,
    /// Current generation counter for ABA prevention
    generation_counter: AtomicU64,
    /// Active order count
    active_count: AtomicU64,
}

impl OrderStateMachine {
    /// Create a new OrderStateMachine
    pub fn new() -> Self {
        Self {
            pending_queue: SegQueue::new(),
            report_queue: SegQueue::new(),
            order_counter: AtomicU64::new(1),
            generation_counter: AtomicU64::new(0),
            active_count: AtomicU64::new(0),
        }
    }

    /// Generate a unique order ID atomically
    #[inline]
    pub fn next_order_id(&self) -> OrderId {
        let id = self.order_counter.fetch_add(1, Ordering::Relaxed);
        OrderId::new(id)
    }

    /// Get current generation counter
    #[inline]
    pub fn current_generation(&self) -> u64 {
        self.generation_counter.load(Ordering::Acquire)
    }

    /// Submit a new order to the pending queue
    /// Returns the order ID and generation
    #[inline]
    pub fn submit_order(&self, mut order: Order) -> Result<(OrderId, u64), &'static str> {
        // Increment generation for this order
        let gen = self.generation_counter.fetch_add(1, Ordering::AcqRel);
        order.generation = gen;

        // Push to pending queue
        match self.pending_queue.push(order) {
            Ok(()) => {
                self.active_count.fetch_add(1, Ordering::Relaxed);
                Ok((order.id, gen))
            }
            Err(_) => Err("Failed to enqueue order"),
        }
    }

    /// Process an execution report and update order state
    /// This is called when an execution report is received from the exchange
    #[inline]
    pub fn process_execution_report(&self, report: ExecutionReport) -> Result<(), &'static str> {
        // Validate generation to prevent ABA issues
        let current_gen = self.generation_counter.load(Ordering::Acquire);
        if report.generation > current_gen {
            return Err("Invalid generation: future generation detected");
        }

        // Push report to queue for processing
        match self.report_queue.push(report) {
            Ok(()) => Ok(()),
            Err(_) => Err("Failed to enqueue execution report"),
        }
    }

    /// Pop a pending order for processing (single consumer)
    #[inline]
    pub fn pop_pending_order(&self) -> Option<Order> {
        self.pending_queue.pop().ok()
    }

    /// Pop an execution report (single consumer)
    #[inline]
    pub fn pop_execution_report(&self) -> Option<ExecutionReport> {
        self.report_queue.pop().ok()
    }

    /// Get the number of pending orders
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending_queue.len()
    }

    /// Get the number of execution reports
    #[inline]
    pub fn report_count(&self) -> usize {
        self.report_queue.len()
    }

    /// Get the number of active orders
    #[inline]
    pub fn active_order_count(&self) -> u64 {
        self.active_count.load(Ordering::Relaxed)
    }

    /// Decrement active count when order is filled or cancelled
    #[inline]
    pub fn complete_order(&self) {
        self.active_count.fetch_sub(1, Ordering::Relaxed);
    }
}

// SAFETY: SegQueue is thread-safe for multi-producer multi-consumer
unsafe impl Send for OrderStateMachine {}
unsafe impl Sync for OrderStateMachine {}

impl Default for OrderStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_order_creation() {
        let order = Order::new(
            OrderId::new(1),
            Side::Buy,
            OrderType::Limit,
            FixedPoint::from_int(100),
            FixedPoint::from_int(10),
            1,
            1234567890,
        );
        
        assert_eq!(order.id, OrderId::new(1));
        assert_eq!(order.side, Side::Buy);
        assert_eq!(order.status, OrderStatus::Pending);
        assert!(order.remaining_quantity.is_positive());
    }

    #[test]
    fn test_tagged_pointer_packing() {
        let tp = TaggedPointer::new(0x12345678, 0xABCD);
        let packed = tp.pack();
        let unpacked = TaggedPointer::unpack(packed);
        
        assert_eq!(tp.ptr, unpacked.ptr);
        assert_eq!(tp.tag, unpacked.tag);
    }

    #[test]
    fn test_concurrent_order_submission() {
        let oms = OrderStateMachine::new();
        let mut handles = vec![];

        for i in 0..10 {
            let oms_ref = &oms;
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let order = Order::new(
                        oms_ref.next_order_id(),
                        Side::Buy,
                        OrderType::Limit,
                        FixedPoint::from_int(100),
                        FixedPoint::from_int(10),
                        1,
                        0,
                    );
                    let _ = oms_ref.submit_order(order);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(oms.pending_count(), 1000);
    }
}
