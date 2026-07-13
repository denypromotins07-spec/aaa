//! Iceberg Sniper Execution Algorithm
//! 
//! Slices large meta-orders into micro-lots to minimize market impact.
//! Uses VPIN toxicity metric to switch between passive and aggressive execution.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Iceberg order configuration
#[derive(Debug, Clone)]
pub struct IcebergConfig {
    /// Minimum slice size (base units)
    pub min_slice_size: i64,
    /// Maximum slice size (base units)
    pub max_slice_size: i64,
    /// Target percentage of visible depth to use
    pub visible_depth_pct: u32, // basis points (e.g., 1000 = 10%)
    /// VPIN threshold for switching to passive-only mode
    pub vpin_passive_threshold: u32, // basis points (e.g., 7000 = 0.7)
    /// Maximum time to wait for a fill before repricing
    pub max_wait_time_ms: u64,
    /// Price improvement in ticks when repricing
    pub price_improvement_ticks: i64,
}

impl Default for IcebergConfig {
    fn default() -> Self {
        Self {
            min_slice_size: 1000,      // 0.001 BTC equivalent
            max_slice_size: 100000,    // 0.1 BTC equivalent
            visible_depth_pct: 1000,   // 10% of visible depth
            vpin_passive_threshold: 7000, // 0.7 VPIN
            max_wait_time_ms: 5000,    // 5 seconds
            price_improvement_ticks: 1, // 1 tick improvement
        }
    }
}

/// State of an iceberg order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcebergState {
    /// Waiting to start execution
    Pending,
    /// Actively executing slices
    Executing,
    /// Paused due to high VPIN toxicity
    PausedToxicity,
    /// Paused waiting for fill
    PausedWaiting,
    /// Completed all slices
    Completed,
    /// Cancelled by user
    Cancelled,
}

/// Single slice of an iceberg order
#[derive(Debug, Clone)]
pub struct IcebergSlice {
    pub slice_id: u64,
    pub parent_order_id: u64,
    pub quantity: i64,
    pub price: i64,
    pub side: OrderSide,
    pub state: SliceState,
    pub created_at: Instant,
    pub filled_quantity: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceState {
    Pending,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Iceberg Sniper execution engine
pub struct IcebergSniper {
    config: IcebergConfig,
    state: IcebergState,
    /// Total remaining quantity to execute
    remaining_quantity: i64,
    /// Total filled quantity
    filled_quantity: i64,
    /// Number of slices executed
    slices_executed: u64,
    /// Current slice being executed
    current_slice: Option<IcebergSlice>,
    /// Last VPIN value
    last_vpin: u32,
    /// Is currently in passive-only mode
    is_passive_only: AtomicBool,
    /// Statistics
    stats: IcebergStats,
}

#[derive(Debug, Clone, Default)]
pub struct IcebergStats {
    pub total_slices_created: u64,
    pub total_slices_filled: u64,
    pub total_slices_cancelled: u64,
    pub average_fill_time_ms: u64,
    pub toxicity_pauses: u64,
    pub queue_jumps: u64,
}

impl IcebergSniper {
    pub fn new(config: IcebergConfig, total_quantity: i64) -> Self {
        Self {
            config,
            state: IcebergState::Pending,
            remaining_quantity: total_quantity,
            filled_quantity: 0,
            slices_executed: 0,
            current_slice: None,
            last_vpin: 0,
            is_passive_only: AtomicBool::new(false),
            stats: IcebergStats::default(),
        }
    }

    /// Update VPIN toxicity metric
    /// Automatically switches to passive-only mode if toxicity is high
    pub fn update_vpin(&mut self, vpin_bps: u32) {
        self.last_vpin = vpin_bps;
        
        let should_be_passive = vpin_bps >= self.config.vpin_passive_threshold;
        let was_passive = self.is_passive_only.load(Ordering::Relaxed);
        
        if should_be_passive && !was_passive {
            self.is_passive_only.store(true, Ordering::Relaxed);
            self.state = IcebergState::PausedToxicity;
            self.stats.toxicity_pauses += 1;
            
            log::info!(
                "Iceberg sniper paused: VPIN {:.2}% exceeds threshold {:.2}%",
                vpin_bps as f64 / 100.0,
                self.config.vpin_passive_threshold as f64 / 100.0
            );
        } else if !should_be_passive && was_passive {
            self.is_passive_only.store(false, Ordering::Relaxed);
            if self.state == IcebergState::PausedToxicity {
                self.state = IcebergState::Executing;
                log::info!("Iceberg sniper resumed: VPIN toxicity normalized");
            }
        }
    }

    /// Calculate optimal slice size based on order book depth
    /// 
    /// # Arguments
    /// * `visible_depth` - Visible liquidity at best bid/ask (in base units)
    /// * `is_buy` - True for buy orders, false for sell
    pub fn calculate_slice_size(&self, visible_depth: i64) -> i64 {
        if visible_depth <= 0 {
            return self.config.min_slice_size;
        }

        // Target percentage of visible depth
        let target = (visible_depth as u64 * self.config.visible_depth_pct as u64 / 10000) as i64;
        
        // Clamp to min/max bounds
        target.clamp(self.config.min_slice_size, self.config.max_slice_size)
            .min(self.remaining_quantity)
    }

    /// Create next slice for execution
    pub fn create_next_slice(
        &mut self,
        slice_id: u64,
        parent_order_id: u64,
        price: i64,
        side: OrderSide,
        visible_depth: i64,
    ) -> Option<&IcebergSlice> {
        if self.remaining_quantity <= 0 {
            self.state = IcebergState::Completed;
            return None;
        }

        if self.state == IcebergState::PausedToxicity {
            return None; // Don't create slices while paused
        }

        let quantity = self.calculate_slice_size(visible_depth);
        if quantity <= 0 {
            return None;
        }

        let slice = IcebergSlice {
            slice_id,
            parent_order_id,
            quantity,
            price,
            side,
            state: SliceState::Pending,
            created_at: Instant::now(),
            filled_quantity: 0,
        };

        self.current_slice = Some(slice);
        self.state = IcebergState::Executing;
        self.stats.total_slices_created += 1;

        self.current_slice.as_ref()
    }

    /// Process a fill for the current slice
    pub fn on_slice_fill(&mut self, fill_quantity: i64, fill_price: i64) -> bool {
        if let Some(ref mut slice) = self.current_slice {
            slice.filled_quantity += fill_quantity;
            slice.state = if slice.filled_quantity >= slice.quantity {
                SliceState::Filled
            } else {
                SliceState::PartiallyFilled
            };

            self.filled_quantity += fill_quantity;
            self.remaining_quantity -= fill_quantity;
            self.stats.total_slices_filled += 1;

            if self.remaining_quantity <= 0 {
                self.state = IcebergState::Completed;
                return true; // All done
            }

            // Clear current slice if fully filled
            if slice.state == SliceState::Filled {
                self.slices_executed += 1;
                self.current_slice = None;
            }
        }
        false
    }

    /// Check if current slice needs repricing (timeout exceeded)
    pub fn check_slice_timeout(&self) -> bool {
        if let Some(ref slice) = self.current_slice {
            if slice.state == SliceState::Submitted || slice.state == SliceState::PartiallyFilled {
                return slice.created_at.elapsed() > Duration::from_millis(self.config.max_wait_time_ms);
            }
        }
        false
    }

    /// Get recommended price improvement for repricing
    pub fn get_price_improvement(&self, current_price: i64, side: OrderSide) -> i64 {
        let improvement = self.config.price_improvement_ticks;
        match side {
            OrderSide::Buy => current_price + improvement, // Bid higher
            OrderSide::Sell => current_price - improvement, // Ask lower
        }
    }

    /// Get current state
    pub fn state(&self) -> IcebergState {
        self.state
    }

    /// Get remaining quantity
    pub fn remaining_quantity(&self) -> i64 {
        self.remaining_quantity
    }

    /// Get filled quantity
    pub fn filled_quantity(&self) -> i64 {
        self.filled_quantity
    }

    /// Get execution progress (basis points)
    pub fn progress_bps(&self, total_quantity: i64) -> u32 {
        if total_quantity <= 0 {
            return 0;
        }
        ((self.filled_quantity * 10000) / total_quantity) as u32
    }

    /// Check if execution is complete
    pub fn is_complete(&self) -> bool {
        self.state == IcebergState::Completed || self.state == IcebergState::Cancelled
    }

    /// Cancel remaining execution
    pub fn cancel(&mut self) {
        self.state = IcebergState::Cancelled;
        if let Some(ref mut slice) = self.current_slice {
            slice.state = SliceState::Cancelled;
            self.stats.total_slices_cancelled += 1;
        }
    }

    /// Get statistics
    pub fn get_stats(&self) -> IcebergStats {
        self.stats.clone()
    }

    /// Check if currently in passive-only mode
    pub fn is_passive_only(&self) -> bool {
        self.is_passive_only.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_size_calculation() {
        let config = IcebergConfig {
            min_slice_size: 1000,
            max_slice_size: 100000,
            visible_depth_pct: 1000, // 10%
            ..Default::default()
        };

        let sniper = IcebergSniper::new(config.clone(), 500000);

        // 10% of 50000 = 5000
        assert_eq!(sniper.calculate_slice_size(50000), 5000);

        // Should clamp to min
        assert_eq!(sniper.calculate_slice_size(5000), 1000);

        // Should clamp to max
        assert_eq!(sniper.calculate_slice_size(2000000), 100000);
    }

    #[test]
    fn test_vpin_toxicity_switching() {
        let mut sniper = IcebergSniper::new(IcebergConfig::default(), 100000);

        // Initially not passive
        assert!(!sniper.is_passive_only());
        assert_ne!(sniper.state(), IcebergState::PausedToxicity);

        // High VPIN should trigger passive mode
        sniper.update_vpin(8000); // 0.8 > 0.7 threshold
        assert!(sniper.is_passive_only());
        assert_eq!(sniper.state(), IcebergState::PausedToxicity);

        // Low VPIN should resume
        sniper.update_vpin(3000); // 0.3 < 0.7 threshold
        assert!(!sniper.is_passive_only());
        assert_eq!(sniper.state(), IcebergState::Executing);
    }

    #[test]
    fn test_iceberg_execution_lifecycle() {
        let mut sniper = IcebergSniper::new(
            IcebergConfig {
                min_slice_size: 1000,
                max_slice_size: 10000,
                visible_depth_pct: 1000,
                ..Default::default()
            },
            30000, // Total 30k to execute
        );

        assert_eq!(sniper.state(), IcebergState::Pending);
        assert_eq!(sniper.remaining_quantity(), 30000);

        // Create first slice (10% of 50k depth = 5k, but clamped to max 10k)
        let slice = sniper.create_next_slice(1, 100, 50000, OrderSide::Buy, 50000).unwrap();
        assert_eq!(slice.quantity, 10000);

        // Fill the slice
        sniper.on_slice_fill(10000, 50000);
        assert_eq!(sniper.filled_quantity(), 10000);
        assert_eq!(sniper.remaining_quantity(), 20000);

        // Create and fill second slice
        sniper.create_next_slice(2, 100, 50000, OrderSide::Buy, 50000);
        sniper.on_slice_fill(10000, 50000);

        // Create and fill third slice (remaining 10k)
        sniper.create_next_slice(3, 100, 50000, OrderSide::Buy, 50000);
        sniper.on_slice_fill(10000, 50000);

        assert!(sniper.is_complete());
        assert_eq!(sniper.state(), IcebergState::Completed);
    }
}
