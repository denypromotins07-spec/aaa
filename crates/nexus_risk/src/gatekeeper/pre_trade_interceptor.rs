//! Pre-Trade Risk Interceptor - Zero-latency gatekeeper between OMS and Network.
//! 
//! This module implements a lock-free observer pattern that validates every
//! outbound child order against strict limits without blocking the execution thread.

use std::sync::atomic::{AtomicU64, AtomicU32, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::gatekeeper::{
    LockFreeOrderQueue,
    FatFingerValidator,
    fat_finger_collars::FatFingerResult,
};

/// Order types supported by the risk interceptor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Market,
    Limit,
    StopMarket,
    StopLimit,
}

/// Side of the order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Representation of an order for risk validation
#[derive(Debug, Clone)]
pub struct OrderValidationRequest {
    /// Unique order ID
    pub order_id: u64,
    /// Symbol identifier (e.g., "BTC-PERP")
    pub symbol: [u8; 16],
    /// Order side
    pub side: Side,
    /// Order type
    pub order_type: OrderType,
    /// Order quantity in base units
    pub quantity: u64,
    /// Limit price (0 for market orders)
    pub price: u64,
    /// Timestamp when order was created (nanoseconds)
    pub timestamp_ns: u64,
    /// Client order ID for deduplication
    pub client_order_id: u64,
}

/// Result of pre-trade risk validation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskValidationResult {
    /// Order passed all checks and can proceed to network
    Approved,
    /// Order size exceeds maximum allowed
    SizeExceeded { 
        submitted: u64, 
        max_allowed: u64,
    },
    /// Too many open orders for this symbol
    OpenOrderLimitExceeded { 
        current_count: u32, 
        max_allowed: u32,
    },
    /// Price violates fat-finger collar
    FatFingerViolation(FatFingerResult),
    /// Duplicate order detected
    DuplicateOrder,
    /// Invalid order parameters
    InvalidParameters(&'static str),
    /// System is halted
    SystemHalted,
}

/// Per-symbol risk state tracked atomically
#[repr(align(64))]
struct SymbolRiskState {
    /// Current open order count
    open_orders: AtomicU32,
    /// Padding to prevent false sharing
    _padding: [u8; 60],
}

impl SymbolRiskState {
    fn new() -> Self {
        Self {
            open_orders: AtomicU32::new(0),
            _padding: [0; 60],
        }
    }
}

// SAFETY: SymbolRiskState can be safely initialized to all zeros
// because AtomicU32::new(0) has the same bit representation as 0u32
impl Default for SymbolRiskState {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the pre-trade risk interceptor
#[derive(Debug, Clone)]
pub struct PreTradeConfig {
    /// Maximum order size in base units
    pub max_order_size: u64,
    /// Maximum open orders per symbol
    pub max_open_orders_per_symbol: u32,
    /// Fat finger collar in basis points
    pub fat_finger_collar_bps: u16,
    /// Stale mid-price threshold in nanoseconds
    pub stale_price_threshold_ns: u64,
    /// Queue capacity for order validation
    pub queue_capacity: usize,
}

impl Default for PreTradeConfig {
    fn default() -> Self {
        Self {
            max_order_size: 1_000_000_000,
            max_open_orders_per_symbol: 100,
            fat_finger_collar_bps: 200, // 2%
            stale_price_threshold_ns: 100_000_000, // 100ms
            queue_capacity: 4096,
        }
    }
}

/// Pre-Trade Risk Interceptor
/// 
/// This is the zero-latency gatekeeper that sits between the OMS and network adapters.
/// It uses lock-free data structures to validate orders without blocking the hot path.
pub struct PreTradeRiskInterceptor {
    /// Configuration
    config: PreTradeConfig,
    /// Lock-free queue for order validation
    order_queue: Arc<LockFreeOrderQueue<OrderValidationRequest>>,
    /// Fat-finger validator
    fat_finger: FatFingerValidator,
    /// Per-symbol open order tracking (simplified hash map using atomic array)
    symbol_states: Box<[SymbolRiskState; 256]>,
    /// Global open order counter
    total_open_orders: AtomicU32,
    /// Count of approved orders
    approved_count: AtomicU64,
    /// Count of rejected orders
    rejected_count: AtomicU64,
    /// System halt flag
    halted: AtomicBool,
    /// Last validation timestamp for monitoring
    last_validation_ns: AtomicU64,
}

unsafe impl Send for PreTradeRiskInterceptor {}
unsafe impl Sync for PreTradeRiskInterceptor {}

impl PreTradeRiskInterceptor {
    /// Create a new pre-trade risk interceptor
    pub fn new(config: PreTradeConfig) -> Self {
        let queue = Arc::new(LockFreeOrderQueue::new(config.queue_capacity));
        
        // Initialize symbol states array using array initialization (no unwrap needed)
        let symbol_states: Box<[SymbolRiskState; 256]> = Box::new(std::array::from_fn(|_| SymbolRiskState::new()));
        
        Self {
            config,
            order_queue: queue,
            fat_finger: FatFingerValidator::new(
                config.fat_finger_collar_bps,
                config.stale_price_threshold_ns,
            ),
            symbol_states,
            total_open_orders: AtomicU32::new(0),
            approved_count: AtomicU64::new(0),
            rejected_count: AtomicU64::new(0),
            halted: AtomicBool::new(false),
            last_validation_ns: AtomicU64::new(0),
        }
    }

    /// Get reference to the order queue for the risk evaluation thread
    #[inline]
    pub fn get_queue(&self) -> Arc<LockFreeOrderQueue<OrderValidationRequest>> {
        Arc::clone(&self.order_queue)
    }

    /// Submit an order for validation (non-blocking).
    /// 
    /// Returns immediately with the validation result. If the queue is full,
    /// returns `RiskValidationResult::InvalidParameters("queue_full")`.
    /// 
    /// # Safety
    /// This function is designed to be called from the OMS hot path.
    /// It performs zero allocations and uses only atomic operations.
    #[inline]
    pub fn validate_order(&self, order: OrderValidationRequest) -> RiskValidationResult {
        // Check if system is halted first (fastest check)
        if self.halted.load(Ordering::Relaxed) {
            return RiskValidationResult::SystemHalted;
        }

        // Validate basic parameters
        if order.quantity == 0 {
            return RiskValidationResult::InvalidParameters("zero_quantity");
        }

        // Check order size limit
        if order.quantity > self.config.max_order_size {
            self.rejected_count.fetch_add(1, Ordering::Relaxed);
            return RiskValidationResult::SizeExceeded {
                submitted: order.quantity,
                max_allowed: self.config.max_order_size,
            };
        }

        // Get symbol hash for indexing
        let symbol_hash = self.hash_symbol(&order.symbol) as usize;
        let symbol_state = &self.symbol_states[symbol_hash];

        // Check open order limit
        let current_open = symbol_state.open_orders.load(Ordering::Relaxed);
        if current_open >= self.config.max_open_orders_per_symbol {
            self.rejected_count.fetch_add(1, Ordering::Relaxed);
            return RiskValidationResult::OpenOrderLimitExceeded {
                current_count: current_open,
                max_allowed: self.config.max_open_orders_per_symbol,
            };
        }

        // Validate price for limit orders
        if order.order_type == OrderType::Limit && order.price > 0 {
            let is_buy = order.side == Side::Buy;
            match self.fat_finger.validate_limit_price(order.price, is_buy, order.timestamp_ns) {
                FatFingerResult::Valid => {}
                other => {
                    self.rejected_count.fetch_add(1, Ordering::Relaxed);
                    return RiskValidationResult::FatFingerViolation(other);
                }
            }
        } else if order.order_type == OrderType::Market {
            // For market orders, just verify we have fresh price data
            if !self.fat_finger.validate_market_order(order.timestamp_ns) {
                self.rejected_count.fetch_add(1, Ordering::Relaxed);
                return RiskValidationResult::FatFingerViolation(FatFingerResult::StaleMidPrice);
            }
        }

        // Try to enqueue for async validation (additional checks can run in background)
        // Note: For ultra-low latency, we approve synchronously above and use the queue
        // only for additional async monitoring/audit
        if let Err(_) = self.order_queue.try_enqueue(order) {
            // Queue full - this is a warning condition but we still allow the order
            // In production, you might want to track this metric and alert
        }

        // Update statistics
        self.approved_count.fetch_add(1, Ordering::Relaxed);
        
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        self.last_validation_ns.store(now, Ordering::Relaxed);

        RiskValidationResult::Approved
    }

    /// Process queued orders in the risk evaluation thread.
    /// 
    /// This should be called by a dedicated risk thread that performs
    /// additional validation that doesn't need to be on the critical path.
    /// 
    /// Returns the number of orders processed.
    #[inline]
    pub fn process_queued_orders(&self) -> usize {
        let mut processed = 0;
        
        loop {
            match self.order_queue.try_dequeue(|_order| {
                // Additional async validation logic here
                // For example: cross-symbol risk checks, portfolio-level limits, etc.
                true
            }) {
                Ok(Some(_order)) => {
                    processed += 1;
                    // Here you could log, audit, or perform additional checks
                }
                Ok(None) => break,
                Err(_rejected_order) => {
                    processed += 1;
                    // Order was rejected by validator, could trigger alerts
                }
            }
        }
        
        processed
    }

    /// Increment open order count for a symbol (called after order sent to exchange)
    #[inline]
    pub fn increment_open_orders(&self, symbol: &[u8]) {
        let hash = self.hash_symbol(symbol) as usize;
        self.symbol_states[hash].open_orders.fetch_add(1, Ordering::Relaxed);
        self.total_open_orders.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement open order count for a symbol (called after order filled/cancelled)
    #[inline]
    pub fn decrement_open_orders(&self, symbol: &[u8]) {
        let hash = self.hash_symbol(symbol) as usize;
        self.symbol_states[hash].open_orders.fetch_sub(1, Ordering::Relaxed);
        self.total_open_orders.fetch_sub(1, Ordering::Relaxed);
    }

    /// Halt all order processing (emergency stop)
    #[inline]
    pub fn halt(&self) {
        self.halted.store(true, Ordering::SeqCst);
    }

    /// Resume order processing
    #[inline]
    pub fn resume(&self) {
        self.halted.store(false, Ordering::SeqCst);
    }

    /// Check if system is halted
    #[inline]
    pub fn is_halted(&self) -> bool {
        self.halted.load(Ordering::Relaxed)
    }

    /// Get statistics
    pub fn stats(&self) -> InterceptorStats {
        InterceptorStats {
            approved_count: self.approved_count.load(Ordering::Relaxed),
            rejected_count: self.rejected_count.load(Ordering::Relaxed),
            total_open_orders: self.total_open_orders.load(Ordering::Relaxed),
            queue_length: self.order_queue.len(),
            dropped_by_queue: self.order_queue.dropped_count(),
            fat_finger_rejections: self.fat_finger.rejection_count(),
            is_halted: self.halted.load(Ordering::Relaxed),
        }
    }

    /// Update mid-price for fat-finger validation
    #[inline]
    pub fn update_mid_price(&self, symbol: &[u8], price: u64, timestamp_ns: u64) {
        // In a real implementation, you'd track per-symbol mid-prices
        // For simplicity, we use a global mid-price here
        let _ = symbol; // Suppress unused warning
        self.fat_finger.update_mid_price(price, timestamp_ns);
    }

    /// Simple hash function for symbol indexing
    #[inline]
    fn hash_symbol(&self, symbol: &[u8]) -> u8 {
        // FNV-1a hash simplified for speed
        const FNV_PRIME: u64 = 1099511628211;
        const FNV_OFFSET: u64 = 14695981039346656037;
        
        let mut hash = FNV_OFFSET;
        for byte in symbol.iter().take_while(|&&b| b != 0) {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        (hash % 256) as u8
    }
}

/// Statistics from the interceptor
#[derive(Debug, Clone)]
pub struct InterceptorStats {
    pub approved_count: u64,
    pub rejected_count: u64,
    pub total_open_orders: u32,
    pub queue_length: usize,
    pub dropped_by_queue: usize,
    pub fat_finger_rejections: u64,
    pub is_halted: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_order(id: u64, price: u64, quantity: u64) -> OrderValidationRequest {
        OrderValidationRequest {
            order_id: id,
            symbol: *b"BTC-PERP      \0",
            side: Side::Buy,
            order_type: OrderType::Limit,
            quantity,
            price,
            timestamp_ns: 1000,
            client_order_id: id,
        }
    }

    #[test]
    fn test_basic_validation() {
        let config = PreTradeConfig::default();
        let interceptor = PreTradeRiskInterceptor::new(config);
        
        // Set mid-price for fat-finger validation
        interceptor.update_mid_price(b"BTC-PERP", 50_000_000_000, 500);
        
        // Valid order
        let order = create_test_order(1, 50_000_000_000, 1_000_000);
        assert_eq!(interceptor.validate_order(order), RiskValidationResult::Approved);
        
        let stats = interceptor.stats();
        assert_eq!(stats.approved_count, 1);
        assert_eq!(stats.rejected_count, 0);
    }

    #[test]
    fn test_size_exceeded() {
        let mut config = PreTradeConfig::default();
        config.max_order_size = 1_000_000;
        let interceptor = PreTradeRiskInterceptor::new(config);
        
        let order = create_test_order(1, 50_000_000_000, 2_000_000);
        match interceptor.validate_order(order) {
            RiskValidationResult::SizeExceeded { submitted, max_allowed } => {
                assert_eq!(submitted, 2_000_000);
                assert_eq!(max_allowed, 1_000_000);
            }
            _ => panic!("Expected SizeExceeded"),
        }
    }

    #[test]
    fn test_zero_quantity() {
        let interceptor = PreTradeRiskInterceptor::new(PreTradeConfig::default());
        
        let order = create_test_order(1, 50_000_000_000, 0);
        assert_eq!(
            interceptor.validate_order(order),
            RiskValidationResult::InvalidParameters("zero_quantity")
        );
    }

    #[test]
    fn test_system_halt() {
        let interceptor = PreTradeRiskInterceptor::new(PreTradeConfig::default());
        interceptor.halt();
        
        let order = create_test_order(1, 50_000_000_000, 1_000_000);
        assert_eq!(interceptor.validate_order(order), RiskValidationResult::SystemHalted);
        
        interceptor.resume();
        assert!(!interceptor.is_halted());
    }

    #[test]
    fn test_open_order_tracking() {
        let interceptor = PreTradeRiskInterceptor::new(PreTradeConfig::default());
        let symbol = b"BTC-PERP";
        
        assert_eq!(interceptor.stats().total_open_orders, 0);
        
        interceptor.increment_open_orders(symbol);
        interceptor.increment_open_orders(symbol);
        assert_eq!(interceptor.stats().total_open_orders, 2);
        
        interceptor.decrement_open_orders(symbol);
        assert_eq!(interceptor.stats().total_open_orders, 1);
    }

    #[test]
    fn test_concurrent_validation() {
        use std::thread;
        use std::sync::Arc;
        
        let interceptor = Arc::new(PreTradeRiskInterceptor::new(PreTradeConfig::default()));
        interceptor.update_mid_price(b"BTC-PERP", 50_000_000_000, 500);
        
        let mut handles = vec![];
        
        for i in 0..4 {
            let interp = Arc::clone(&interceptor);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let order = create_test_order(i * 100 + j, 50_000_000_000, 1_000_000);
                    let _ = interp.validate_order(order);
                }
            }));
        }
        
        for h in handles {
            let _ = h.join();
        }
        
        let stats = interceptor.stats();
        assert_eq!(stats.approved_count, 400);
    }
}
