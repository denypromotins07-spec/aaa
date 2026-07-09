//! Iceberg Order State Machine for hiding true order size.
//! Detects fleeting liquidity and dynamically adjusts clip size.

use nexus_oms::{FixedPoint, Side, OrderType};
use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};

/// Iceberg order state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcebergState {
    /// Waiting to submit first child order
    Pending,
    /// Active, submitting child orders
    Active,
    /// Paused, waiting for liquidity
    Paused,
    /// Fully filled or cancelled
    Completed,
}

/// Liquidity detection result
#[derive(Debug, Clone, Copy)]
pub struct LiquiditySnapshot {
    pub best_bid_qty: FixedPoint,
    pub best_ask_qty: FixedPoint,
    pub bid_depth_5: FixedPoint,
    pub ask_depth_5: FixedPoint,
    pub timestamp_ns: u64,
}

/// Iceberg order configuration
pub struct IcebergConfig {
    /// Total parent order quantity
    pub total_quantity: FixedPoint,
    /// Initial clip size (visible portion)
    pub initial_clip: FixedPoint,
    /// Minimum clip size
    pub min_clip: FixedPoint,
    /// Maximum clip size
    pub max_clip: FixedPoint,
    /// Target participation rate (scaled by 10^8, e.g., 0.10 = 10%)
    pub target_participation: FixedPoint,
    /// Aggressiveness: how quickly to increase clip when liquidity is abundant
    pub aggressiveness: u8, // 0-100
}

/// Iceberg execution state machine
pub struct IcebergState {
    /// Current state
    state: IcebergState,
    /// Configuration
    config: IcebergConfig,
    /// Remaining parent quantity
    remaining_parent: FixedPoint,
    /// Filled parent quantity
    filled_parent: FixedPoint,
    /// Current clip size
    current_clip: FixedPoint,
    /// Number of child orders submitted
    child_orders_submitted: AtomicU64,
    /// Number of child orders filled
    child_orders_filled: AtomicU64,
    /// Last liquidity check timestamp
    last_liquidity_check_ns: AtomicU64,
    /// Consecutive partial fills (for detecting adverse selection)
    consecutive_partial_fills: AtomicU64,
    /// Sequence number
    sequence: AtomicU64,
}

impl IcebergState {
    /// Create a new iceberg state machine
    pub fn new(config: IcebergConfig) -> Self {
        let total_qty = config.total_quantity;
        Self {
            state: IcebergState::Pending,
            config,
            remaining_parent: total_qty,
            filled_parent: FixedPoint::from_raw(0),
            current_clip: config.initial_clip,
            child_orders_submitted: AtomicU64::new(0),
            child_orders_filled: AtomicU64::new(0),
            last_liquidity_check_ns: AtomicU64::new(0),
            consecutive_partial_fills: AtomicU64::new(0),
            sequence: AtomicU64::new(0),
        }
    }

    /// Get current state
    #[inline]
    pub fn get_state(&self) -> IcebergState {
        self.state
    }

    /// Get remaining parent quantity
    #[inline]
    pub fn get_remaining_parent(&self) -> FixedPoint {
        self.remaining_parent
    }

    /// Get filled parent quantity
    #[inline]
    pub fn get_filled_parent(&self) -> FixedPoint {
        self.filled_parent
    }

    /// Get current clip size
    #[inline]
    pub fn get_current_clip(&self) -> FixedPoint {
        self.current_clip
    }

    /// Initialize the iceberg order
    #[inline]
    pub fn initialize(&mut self) -> Result<(), &'static str> {
        if self.state != IcebergState::Pending {
            return Err("Iceberg order not in pending state");
        }
        
        if self.config.total_quantity.is_zero() {
            return Err("Total quantity cannot be zero");
        }
        
        if self.config.initial_clip.is_zero() || 
           self.config.initial_clip > self.config.total_quantity {
            return Err("Invalid initial clip size");
        }
        
        self.state = IcebergState::Active;
        self.sequence.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Calculate optimal clip size based on current liquidity
    /// Uses adaptive algorithm to minimize market impact
    #[inline]
    pub fn calculate_optimal_clip(&self, liquidity: &LiquiditySnapshot, side: Side) -> FixedPoint {
        let available_liquidity = match side {
            Side::Buy => liquidity.best_ask_qty,
            Side::Sell => liquidity.best_bid_qty,
        };

        // Base clip: percentage of available liquidity
        let base_clip = available_liquidity * self.config.target_participation;

        // Adjust based on current fill rate
        let fill_rate_adjustment = {
            let submitted = self.child_orders_submitted.load(Ordering::Relaxed);
            let filled = self.child_orders_filled.load(Ordering::Relaxed);
            
            if submitted == 0 {
                FixedPoint::from_raw(SCALE) // 1.0
            } else {
                let rate = FixedPoint::from_raw((filled as i64 * SCALE) / submitted as i64);
                // If fill rate is low, reduce clip size
                if rate < FixedPoint::from_fractional(50_000_000) {
                    FixedPoint::from_fractional(75_000_000) // 0.75x
                } else if rate > FixedPoint::from_fractional(90_000_000) {
                    FixedPoint::from_fractional(125_000_000) // 1.25x
                } else {
                    FixedPoint::from_raw(SCALE)
                }
            }
        };

        let adjusted_clip = base_clip * fill_rate_adjustment;

        // Clamp to min/max bounds
        adjusted_clip
            .max(self.config.min_clip)
            .min(self.config.max_clip)
            .min(self.remaining_parent)
    }

    /// Update clip size based on liquidity detection
    #[inline]
    pub fn update_clip_from_liquidity(&mut self, liquidity: &LiquiditySnapshot, side: Side) {
        let new_clip = self.calculate_optimal_clip(liquidity, side);
        
        // Smooth transition: don't change clip too drastically
        let max_change = self.current_clip * FixedPoint::from_fractional(25_000_000); // 25%
        let diff = new_clip - self.current_clip;
        
        let adjusted_clip = if diff.abs() > max_change {
            if diff.is_positive() {
                self.current_clip + max_change
            } else {
                self.current_clip - max_change
            }
        } else {
            new_clip
        };

        self.current_clip = adjusted_clip.clamp(self.config.min_clip, self.config.max_clip);
        self.last_liquidity_check_ns.store(liquidity.timestamp_ns, Ordering::Release);
        self.sequence.fetch_add(1, Ordering::Relaxed);
    }

    /// Called when a child order is submitted
    #[inline]
    pub fn on_child_submitted(&self) -> u64 {
        self.child_orders_submitted.fetch_add(1, Ordering::Relaxed);
        self.sequence.fetch_add(1, Ordering::Relaxed)
    }

    /// Called when a child order is filled
    #[inline]
    pub fn on_child_filled(&self, fill_qty: FixedPoint, is_partial: bool) -> u64 {
        self.child_orders_filled.fetch_add(1, Ordering::Relaxed);
        
        if is_partial {
            self.consecutive_partial_fills.fetch_add(1, Ordering::Relaxed);
        } else {
            self.consecutive_partial_fills.store(0, Ordering::Relaxed);
        }
        
        self.sequence.fetch_add(1, Ordering::Relaxed)
    }

    /// Update parent order state after child fill
    #[inline]
    pub fn update_parent_on_fill(&mut self, fill_qty: FixedPoint) -> Result<(), &'static str> {
        if fill_qty.is_zero() {
            return Err("Fill quantity cannot be zero");
        }

        if fill_qty > self.remaining_parent {
            return Err("Fill quantity exceeds remaining parent");
        }

        self.remaining_parent = self.remaining_parent - fill_qty;
        self.filled_parent = self.filled_parent + fill_qty;

        // Check if parent is fully filled
        if self.remaining_parent.is_zero() {
            self.state = IcebergState::Completed;
        }

        self.sequence.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Check if we should pause due to adverse selection
    #[inline]
    pub fn should_pause(&self) -> bool {
        let consecutive = self.consecutive_partial_fills.load(Ordering::Relaxed);
        // Pause after 3+ consecutive partial fills
        consecutive >= 3
    }

    /// Pause the iceberg order
    #[inline]
    pub fn pause(&mut self) {
        if self.state == IcebergState::Active {
            self.state = IcebergState::Paused;
            self.sequence.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Resume the iceberg order
    #[inline]
    pub fn resume(&mut self) -> Result<(), &'static str> {
        if self.state != IcebergState::Paused {
            return Err("Iceberg order not paused");
        }
        self.state = IcebergState::Active;
        self.consecutive_partial_fills.store(0, Ordering::Relaxed);
        self.sequence.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Cancel the iceberg order
    #[inline]
    pub fn cancel(&mut self) {
        self.state = IcebergState::Completed;
        self.sequence.fetch_add(1, Ordering::Relaxed);
    }

    /// Get next child order quantity
    #[inline]
    pub fn get_next_child_qty(&self) -> FixedPoint {
        self.current_clip.min(self.remaining_parent)
    }

    /// Check if order is complete
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.state == IcebergState::Completed || self.remaining_parent.is_zero()
    }

    /// Get sequence number
    #[inline]
    pub fn get_sequence(&self) -> u64 {
        self.sequence.load(Ordering::Relaxed)
    }
}

const SCALE: i64 = 100_000_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iceberg_initialization() {
        let config = IcebergConfig {
            total_quantity: FixedPoint::from_int(1000),
            initial_clip: FixedPoint::from_int(100),
            min_clip: FixedPoint::from_int(50),
            max_clip: FixedPoint::from_int(200),
            target_participation: FixedPoint::from_fractional(10_000_000), // 10%
            aggressiveness: 50,
        };

        let mut iceberg = IcebergState::new(config);
        assert_eq!(iceberg.get_state(), IcebergState::Pending);
        
        iceberg.initialize().unwrap();
        assert_eq!(iceberg.get_state(), IcebergState::Active);
        assert_eq!(iceberg.get_current_clip().to_f64(), 100.0);
    }

    #[test]
    fn test_clip_adjustment() {
        let config = IcebergConfig {
            total_quantity: FixedPoint::from_int(1000),
            initial_clip: FixedPoint::from_int(100),
            min_clip: FixedPoint::from_int(50),
            max_clip: FixedPoint::from_int(200),
            target_participation: FixedPoint::from_fractional(10_000_000),
            aggressiveness: 50,
        };

        let iceberg = IcebergState::new(config);
        let liquidity = LiquiditySnapshot {
            best_bid_qty: FixedPoint::from_int(500),
            best_ask_qty: FixedPoint::from_int(600),
            bid_depth_5: FixedPoint::from_int(2000),
            ask_depth_5: FixedPoint::from_int(2500),
            timestamp_ns: 1234567890,
        };

        let optimal = iceberg.calculate_optimal_clip(&liquidity, Side::Buy);
        // 10% of 600 = 60, clamped to min 50
        assert!(optimal.to_f64() >= 50.0 && optimal.to_f64() <= 200.0);
    }
}
