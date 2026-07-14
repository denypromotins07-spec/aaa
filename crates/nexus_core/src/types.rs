//! NEXUS-OMEGA Core Types
//!
//! This module defines unified financial primitives shared across all crates.
//! These types are designed for zero-copy operations and cross-crate compatibility.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Unique order identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderId(pub u64);

/// Execution report identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExecutionId(pub u64);

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Side {
    Buy = 0,
    Sell = 1,
}

impl Default for Side {
    fn default() -> Self {
        Side::Buy
    }
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum OrderType {
    Limit = 0,
    Market = 1,
    StopLimit = 2,
}

impl Default for OrderType {
    fn default() -> Self {
        OrderType::Limit
    }
}

/// Time in force
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum TimeInForce {
    Gtc = 0, // Good till cancel
    Fok = 1, // Fill or kill
    Ioc = 2, // Immediate or cancel
}

impl Default for TimeInForce {
    fn default() -> Self {
        TimeInForce::Gtc
    }
}

/// Order state - strictly defined states for the state machine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum OrderState {
    PendingNew = 0,
    Open = 1,
    PartiallyFilled = 2,
    Filled = 3,
    Canceled = 4,
    Rejected = 5,
}

impl Default for OrderState {
    fn default() -> Self {
        OrderState::PendingNew
    }
}

impl OrderState {
    /// Check if state is terminal (no further transitions allowed)
    #[inline]
    pub fn is_terminal(self) -> bool {
        matches!(self, OrderState::Filled | OrderState::Canceled | OrderState::Rejected)
    }
}

/// Normalized order book delta for internal routing
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C, align(64))]
pub struct OrderBookDelta {
    pub timestamp_ns: u64,
    pub symbol: [u8; 16], // Fixed-size symbol buffer (zero-copy)
    pub price: u64,       // Nanodollars
    pub quantity: u64,    // Base units * 1e9
    pub side: u8,         // 0=Bid, 1=Ask
    pub delta_type: u8,   // 0=Add, 1=Modify, 2=Cancel, 3=Trade
    pub sequence_id: u64,
    _padding: [u8; 22],
}

impl Default for OrderBookDelta {
    fn default() -> Self {
        Self {
            timestamp_ns: 0,
            symbol: [0u8; 16],
            price: 0,
            quantity: 0,
            side: 0,
            delta_type: 0,
            sequence_id: 0,
            _padding: [0u8; 22],
        }
    }
}

/// Normalized trade event
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C, align(64))]
pub struct Trade {
    pub timestamp_ns: u64,
    pub symbol: [u8; 16],
    pub price: u64,
    pub quantity: u64,
    pub aggressor_side: u8, // 0=Buyer, 1=Seller
    pub trade_id: u64,
    _padding: [u8; 30],
}

impl Default for Trade {
    fn default() -> Self {
        Self {
            timestamp_ns: 0,
            symbol: [0u8; 16],
            price: 0,
            quantity: 0,
            aggressor_side: 0,
            trade_id: 0,
            _padding: [0u8; 30],
        }
    }
}

/// Alpha signal from strategy agents
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C, align(64))]
pub struct AlphaSignal {
    /// Signal value (-1.0 to +1.0)
    pub value: f64,
    /// Signal confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Signal type identifier
    pub signal_type: u8,
    /// Timestamp
    pub ts: u64,
    /// Recent accuracy (for Bayesian update)
    pub recent_accuracy: f64,
    /// Strategy agent ID that generated this signal
    pub agent_id: u32,
    _padding: [u8; 19],
}

impl Default for AlphaSignal {
    fn default() -> Self {
        Self {
            value: 0.0,
            confidence: 0.5,
            signal_type: 0,
            ts: 0,
            recent_accuracy: 0.5,
            agent_id: 0,
            _padding: [0u8; 19],
        }
    }
}

/// Execution report from exchange
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionReport {
    pub order_id: OrderId,
    pub execution_id: Option<ExecutionId>,
    pub report_type: ExecutionReportType,
    pub timestamp_ns: u64,
    pub exchange_timestamp_ns: Option<u64>,
}

/// Execution report type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExecutionReportType {
    New { order_state: OrderState },
    Trade {
        fill_quantity: u64,
        fill_price: u64,
        trade_id: u64,
        last_shares: u64,
    },
    Canceled { reason: String },
    Rejected { reason: String },
    Status { order_state: OrderState },
}

/// Order information stored in zero-allocation format where possible
#[derive(Debug, Clone, Serialize, Deserialize)]
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
            .unwrap_or_default()
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
        ((self.filled_quantity * 10000) / self.quantity.abs()) as u32
    }
}

/// Market tick - unified market data primitive
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C, align(64))]
pub struct MarketTick {
    pub timestamp_ns: u64,
    pub symbol: [u8; 16],
    pub bid_price: u64,
    pub ask_price: u64,
    pub bid_size: u64,
    pub ask_size: u64,
    pub last_price: u64,
    pub last_size: u64,
    pub sequence_id: u64,
    _padding: [u8; 16],
}

impl Default for MarketTick {
    fn default() -> Self {
        Self {
            timestamp_ns: 0,
            symbol: [0u8; 16],
            bid_price: 0,
            ask_price: 0,
            bid_size: 0,
            ask_size: 0,
            last_price: 0,
            last_size: 0,
            sequence_id: 0,
            _padding: [0u8; 16],
        }
    }
}

/// Thread-safe wrapper for alpha signals used in channel passing
pub type SharedAlphaSignal = Arc<AlphaSignal>;

/// Result type for trading operations
pub type TradingResult<T> = Result<T, TradingError>;

/// Trading error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum TradingError {
    #[error("Order rejected: {0}")]
    OrderRejected(String),
    #[error("Insufficient balance: {0}")]
    InsufficientBalance(String),
    #[error("Market data error: {0}")]
    MarketDataError(String),
    #[error("Connection error: {0}")]
    ConnectionError(String),
    #[error("Risk limit exceeded: {0}")]
    RiskLimitExceeded(String),
    #[error("Invalid state transition: {0}")]
    InvalidStateTransition(String),
    #[error("Timeout: {0}")]
    Timeout(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_creation() {
        let order = Order::new(
            OrderId(1),
            "BTCUSDT",
            Side::Buy,
            OrderType::Limit,
            TimeInForce::Gtc,
            50000_00000,
            100_000000000,
            12345,
        );
        assert_eq!(order.symbol_str(), "BTCUSDT");
        assert_eq!(order.side, Side::Buy);
    }

    #[test]
    fn test_alpha_signal_default() {
        let signal = AlphaSignal::default();
        assert_eq!(signal.value, 0.0);
        assert_eq!(signal.confidence, 0.5);
    }

    #[test]
    fn test_order_book_delta_alignment() {
        // Verify 64-byte alignment for cache efficiency
        assert_eq!(std::mem::align_of::<OrderBookDelta>(), 64);
    }
}
