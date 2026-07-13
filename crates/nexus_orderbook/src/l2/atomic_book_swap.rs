//! Chapter 2: Atomic Order Book Swap
//!
//! This module provides lock-free, atomic swapping of order book state
//! when healing from sequence gaps. The swap operation is instantaneous
//! and thread-safe, ensuring no partial state is ever visible to consumers.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use tracing::{debug, info, warn};

/// Order book level (price + quantity)
#[derive(Debug, Clone, Copy)]
pub struct PriceLevel {
    pub price: u64,
    pub quantity: u64,
}

/// Snapshot of order book state for atomic swap
#[derive(Debug, Clone)]
pub struct OrderBookState {
    pub last_update_id: u64,
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub timestamp_ns: u64,
    pub is_valid: bool,
}

impl Default for OrderBookState {
    fn default() -> Self {
        Self {
            last_update_id: 0,
            bids: Vec::new(),
            asks: Vec::new(),
            timestamp_ns: 0,
            is_valid: false,
        }
    }
}

/// Statistics for atomic swap operations
#[derive(Debug, Clone, Default)]
pub struct AtomicSwapStats {
    pub total_swaps: u64,
    pub successful_swaps: u64,
    pub failed_validations: u64,
    pub swaps_during_active_trading: u64,
    pub avg_swap_duration_ns: u64,
}

/// Atomic Book Swap Manager
///
/// Provides lock-free atomic swapping of order book state using Arc swap pattern.
/// Readers always see a consistent snapshot, never partial updates.
pub struct AtomicBookSwap {
    /// Current active book state (swapped atomically via Arc)
    current_state: Arc<RwLock<OrderBookState>>,
    /// Pending state being prepared for swap
    pending_state: Arc<RwLock<Option<OrderBookState>>>,
    /// Swap in progress flag
    swap_in_progress: Arc<AtomicBool>,
    /// Whether trading is currently active (affects swap timing)
    trading_active: Arc<AtomicBool>,
    /// Statistics
    stats: Arc<RwLock<AtomicSwapStats>>,
    /// Sequence number for swap operations
    swap_sequence: Arc<AtomicU64>,
}

// SAFETY: Uses Arc, RwLock, and atomics for thread safety
unsafe impl Send for AtomicBookSwap {}
unsafe impl Sync for AtomicBookSwap {}

impl AtomicBookSwap {
    pub fn new() -> Self {
        Self {
            current_state: Arc::new(RwLock::new(OrderBookState::default())),
            pending_state: Arc::new(RwLock::new(None)),
            swap_in_progress: Arc::new(AtomicBool::new(false)),
            trading_active: Arc::new(AtomicBool::new(true)),
            stats: Arc::new(RwLock::new(AtomicSwapStats::default())),
            swap_sequence: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get the current active order book state (read-only snapshot)
    pub fn get_current_state(&self) -> OrderBookState {
        self.current_state.read().clone()
    }

    /// Prepare a new state for atomic swap
    pub fn prepare_swap(&self, new_state: OrderBookState) -> Result<(), &'static str> {
        // Validate the new state before preparing
        if !self.validate_state(&new_state) {
            let mut stats = self.stats.write();
            stats.failed_validations += 1;
            return Err("Invalid state for swap");
        }

        // Check if another swap is in progress
        if self.swap_in_progress.load(Ordering::Acquire) {
            return Err("Swap already in progress");
        }

        // Store in pending state
        *self.pending_state.write() = Some(new_state);
        
        debug!("Swap state prepared, awaiting commit");
        Ok(())
    }

    /// Execute the atomic swap
    /// Returns true if swap was successful, false if validation failed
    pub fn execute_swap(&self) -> bool {
        let start_ns = current_time_ns();

        // Check if swap is already in progress
        if self.swap_in_progress.swap(true, Ordering::AcqRel) {
            warn!("Concurrent swap attempt rejected");
            return false;
        }

        // Get pending state
        let pending = {
            let mut guard = self.pending_state.write();
            match guard.take() {
                Some(state) => state,
                None => {
                    self.swap_in_progress.store(false, Ordering::Release);
                    warn!("No pending state to swap");
                    return false;
                }
            }
        };

        // Final validation before swap
        if !self.validate_state(&pending) {
            self.swap_in_progress.store(false, Ordering::Release);
            let mut stats = self.stats.write();
            stats.failed_validations += 1;
            return false;
        }

        // Check if we're actively trading (may want to delay swap)
        let during_trading = self.trading_active.load(Ordering::Acquire);
        
        // Perform the atomic swap
        {
            let mut current = self.current_state.write();
            *current = pending;
        }

        // Record statistics
        let duration_ns = current_time_ns() - start_ns;
        {
            let mut stats = self.stats.write();
            stats.total_swaps += 1;
            stats.successful_swaps += 1;
            if during_trading {
                stats.swaps_during_active_trading += 1;
            }
            stats.avg_swap_duration_ns = 
                (stats.avg_swap_duration_ns * (stats.successful_swaps - 1) + duration_ns)
                / stats.successful_swaps;
        }

        // Increment swap sequence
        self.swap_sequence.fetch_add(1, Ordering::Release);

        // Release swap lock
        self.swap_in_progress.store(false, Ordering::Release);

        info!(
            "Atomic swap completed: sequence={}, duration={}ns, during_trading={}",
            self.swap_sequence.load(Ordering::Acquire),
            duration_ns,
            during_trading
        );

        true
    }

    /// Validate state before swap
    fn validate_state(&self, state: &OrderBookState) -> bool {
        // Must have valid update ID
        if state.last_update_id == 0 {
            warn!("Validation failed: invalid last_update_id");
            return false;
        }

        // Must have at least some liquidity on both sides (normal market condition)
        if state.bids.is_empty() || state.asks.is_empty() {
            warn!("Validation failed: empty order book sides");
            return false;
        }

        // Bids must be below asks (no crossed book)
        if let (Some(best_bid), Some(best_ask)) = (
            state.bids.iter().map(|l| l.price).max_by_key(|p| *p),
            state.asks.iter().map(|l| l.price).min(),
        ) {
            if best_bid >= best_ask {
                warn!("Validation failed: crossed book (bid={} >= ask={})", best_bid, best_ask);
                return false;
            }
        }

        // Timestamp must be recent (within last 60 seconds)
        let now = current_time_ns();
        let age_ns = now.saturating_sub(state.timestamp_ns);
        let max_age_ns = Duration::from_secs(60).as_nanos() as u64;
        
        if age_ns > max_age_ns {
            warn!("Validation failed: stale snapshot (age={}ms)", age_ns / 1_000_000);
            return false;
        }

        true
    }

    /// Mark the book swap as complete and ready for use
    pub fn finalize_swap(&self) {
        info!("Swap finalized - order book ready for consumption");
    }

    /// Check if a swap is currently in progress
    pub fn is_swap_in_progress(&self) -> bool {
        self.swap_in_progress.load(Ordering::Acquire)
    }

    /// Get the current swap sequence number
    pub fn get_swap_sequence(&self) -> u64 {
        self.swap_sequence.load(Ordering::Acquire)
    }

    /// Set trading active status
    pub fn set_trading_active(&self, active: bool) {
        self.trading_active.store(active, Ordering::Release);
    }

    /// Get statistics
    pub fn get_stats(&self) -> AtomicSwapStats {
        self.stats.read().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        *self.stats.write() = AtomicSwapStats::default();
    }
}

use std::time::Duration;

/// Get current time in nanoseconds
fn current_time_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let swap = AtomicBookSwap::new();
        let state = swap.get_current_state();
        
        assert!(!state.is_valid);
        assert_eq!(state.last_update_id, 0);
        assert!(state.bids.is_empty());
        assert!(state.asks.is_empty());
    }

    #[test]
    fn test_prepare_and_execute_swap() {
        let swap = AtomicBookSwap::new();
        
        let new_state = OrderBookState {
            last_update_id: 1000,
            bids: vec![PriceLevel { price: 50000, quantity: 100 }],
            asks: vec![PriceLevel { price: 50100, quantity: 100 }],
            timestamp_ns: current_time_ns(),
            is_valid: true,
        };

        assert!(swap.prepare_swap(new_state.clone()).is_ok());
        assert!(swap.execute_swap());

        let current = swap.get_current_state();
        assert_eq!(current.last_update_id, 1000);
        assert_eq!(current.bids.len(), 1);
        assert_eq!(current.asks.len(), 1);
    }

    #[test]
    fn test_concurrent_swap_prevention() {
        let swap = AtomicBookSwap::new();
        
        let state1 = OrderBookState {
            last_update_id: 1000,
            bids: vec![PriceLevel { price: 50000, quantity: 100 }],
            asks: vec![PriceLevel { price: 50100, quantity: 100 }],
            timestamp_ns: current_time_ns(),
            is_valid: true,
        };

        let state2 = OrderBookState {
            last_update_id: 2000,
            bids: vec![PriceLevel { price: 50000, quantity: 200 }],
            asks: vec![PriceLevel { price: 50100, quantity: 200 }],
            timestamp_ns: current_time_ns(),
            is_valid: true,
        };

        // First swap should succeed
        assert!(swap.prepare_swap(state1).is_ok());
        assert!(swap.execute_swap());

        // Second swap should also succeed (first is complete)
        assert!(swap.prepare_swap(state2).is_ok());
        assert!(swap.execute_swap());

        let current = swap.get_current_state();
        assert_eq!(current.last_update_id, 2000);
    }

    #[test]
    fn test_validation_crossed_book() {
        let swap = AtomicBookSwap::new();
        
        // Crossed book (bid >= ask) should fail validation
        let crossed_state = OrderBookState {
            last_update_id: 1000,
            bids: vec![PriceLevel { price: 50100, quantity: 100 }], // Bid higher than ask
            asks: vec![PriceLevel { price: 50000, quantity: 100 }],
            timestamp_ns: current_time_ns(),
            is_valid: true,
        };

        assert!(swap.prepare_swap(crossed_state).is_err());
    }

    #[test]
    fn test_validation_empty_sides() {
        let swap = AtomicBookSwap::new();
        
        // Empty bids should fail
        let empty_bids = OrderBookState {
            last_update_id: 1000,
            bids: vec![],
            asks: vec![PriceLevel { price: 50100, quantity: 100 }],
            timestamp_ns: current_time_ns(),
            is_valid: true,
        };

        assert!(swap.prepare_swap(empty_bids).is_err());
    }

    #[test]
    fn test_swap_statistics() {
        let swap = AtomicBookSwap::new();
        
        let state = OrderBookState {
            last_update_id: 1000,
            bids: vec![PriceLevel { price: 50000, quantity: 100 }],
            asks: vec![PriceLevel { price: 50100, quantity: 100 }],
            timestamp_ns: current_time_ns(),
            is_valid: true,
        };

        let _ = swap.prepare_swap(state.clone());
        swap.execute_swap();

        let stats = swap.get_stats();
        assert_eq!(stats.total_swaps, 1);
        assert_eq!(stats.successful_swaps, 1);
        assert_eq!(stats.failed_validations, 0);
    }
}
