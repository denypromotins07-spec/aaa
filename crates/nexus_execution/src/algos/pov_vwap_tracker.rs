//! POV (Percentage of Volume) and VWAP tracking algorithms.
//! Uses EWMA for real-time market participation rate calculation.

use nexus_oms::FixedPoint;
use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};

const SCALE: i64 = 100_000_000;

/// EWMA (Exponentially Weighted Moving Average) calculator
/// Used for smoothing volume and participation rate estimates
pub struct EwmaCalculator {
    /// Current EWMA value (scaled by 10^8)
    value: AtomicI64,
    /// Alpha parameter (scaled by 10^8, e.g., 0.1 = 10% weight to new sample)
    alpha: FixedPoint,
    /// Initialized flag
    initialized: AtomicU64,
}

impl EwmaCalculator {
    #[inline]
    pub fn new(alpha: FixedPoint, initial_value: FixedPoint) -> Self {
        Self {
            value: AtomicI64::new(initial_value.raw()),
            alpha,
            initialized: AtomicU64::new(1),
        }
    }

    #[inline]
    pub fn update(&self, sample: FixedPoint) -> FixedPoint {
        let current_raw = self.value.load(Ordering::Acquire);
        
        // EWMA formula: new = alpha * sample + (1 - alpha) * current
        let one_minus_alpha = FixedPoint::from_raw(SCALE) - self.alpha;
        let weighted_sample = sample * self.alpha;
        let weighted_current = FixedPoint::from_raw(current_raw) * one_minus_alpha;
        
        let new_value = weighted_sample + weighted_current;
        self.value.store(new_value.raw(), Ordering::Release);
        
        new_value
    }

    #[inline]
    pub fn get_current(&self) -> FixedPoint {
        FixedPoint::from_raw(self.value.load(Ordering::Acquire))
    }

    #[inline]
    pub fn reset(&self, value: FixedPoint) {
        self.value.store(value.raw(), Ordering::Release);
        self.initialized.store(1, Ordering::Release);
    }
}

/// POV execution configuration
pub struct PovConfig {
    /// Target participation rate (e.g., 0.05 = 5% of market volume)
    pub target_participation: FixedPoint,
    /// Minimum order size
    pub min_order_size: FixedPoint,
    /// Maximum order size
    pub max_order_size: FixedPoint,
    /// EWMA alpha for volume estimation
    pub volume_alpha: FixedPoint,
    /// Maximum deviation from target participation before adjustment
    pub tolerance: FixedPoint,
}

/// POV execution state machine
pub struct PovTracker {
    config: PovConfig,
    /// Total quantity to execute
    total_qty: FixedPoint,
    /// Remaining quantity
    remaining_qty: FixedPoint,
    /// Executed quantity
    executed_qty: FixedPoint,
    /// Market volume observed (EWMA)
    market_volume_ewma: EwmaCalculator,
    /// Our volume (EWMA)
    our_volume_ewma: EwmaCalculator,
    /// Current participation rate
    current_participation: FixedPoint,
    /// Last adjustment timestamp
    last_adjustment_ns: AtomicU64,
    /// Sequence number
    sequence: AtomicU64,
}

impl PovTracker {
    #[inline]
    pub fn new(config: PovConfig, total_qty: FixedPoint) -> Self {
        let initial_volume = FixedPoint::from_int(1000); // Initial estimate
        
        Self {
            config,
            total_qty,
            remaining_qty: total_qty,
            executed_qty: FixedPoint::from_raw(0),
            market_volume_ewma: EwmaCalculator::new(config.volume_alpha, initial_volume),
            our_volume_ewma: EwmaCalculator::new(config.volume_alpha, FixedPoint::from_raw(0)),
            current_participation: FixedPoint::from_raw(0),
            last_adjustment_ns: AtomicU64::new(0),
            sequence: AtomicU64::new(0),
        }
    }

    /// Update tracker with new market volume observation
    #[inline]
    pub fn on_market_volume(&mut self, market_vol: FixedPoint, timestamp_ns: u64) {
        self.market_volume_ewma.update(market_vol);
        self.recalculate_participation();
        self.last_adjustment_ns.store(timestamp_ns, Ordering::Release);
        self.sequence.fetch_add(1, Ordering::Relaxed);
    }

    /// Update tracker with our fill
    #[inline]
    pub fn on_our_fill(&mut self, fill_qty: FixedPoint, timestamp_ns: u64) {
        if fill_qty > self.remaining_qty {
            return; // Invalid fill
        }
        
        self.executed_qty = self.executed_qty + fill_qty;
        self.remaining_qty = self.remaining_qty - fill_qty;
        self.our_volume_ewma.update(fill_qty);
        self.recalculate_participation();
        self.last_adjustment_ns.store(timestamp_ns, Ordering::Release);
        self.sequence.fetch_add(1, Ordering::Relaxed);
    }

    /// Recalculate current participation rate
    #[inline]
    fn recalculate_participation(&mut self) {
        let market_vol = self.market_volume_ewma.get_current();
        let our_vol = self.our_volume_ewma.get_current();
        
        if market_vol.is_zero() {
            self.current_participation = FixedPoint::from_raw(0);
        } else {
            self.current_participation = our_vol / market_vol;
        }
    }

    /// Calculate next order size based on POV algorithm
    #[inline]
    pub fn calculate_next_order_size(&self, current_market_vol: FixedPoint) -> FixedPoint {
        // Target: we should execute target_participation% of current market volume
        let target_size = current_market_vol * self.config.target_participation;
        
        // Clamp to min/max bounds
        let clamped = target_size
            .max(self.config.min_order_size)
            .min(self.config.max_order_size)
            .min(self.remaining_qty);
        
        clamped
    }

    /// Check if we're within tolerance of target participation
    #[inline]
    pub fn is_within_tolerance(&self) -> bool {
        let diff = if self.current_participation > self.config.target_participation {
            self.current_participation - self.config.target_participation
        } else {
            self.config.target_participation - self.current_participation
        };
        
        diff <= self.config.tolerance
    }

    /// Get recommended action: increase, decrease, or maintain pace
    #[inline]
    pub fn get_pace_recommendation(&self) -> PovPace {
        if self.current_participation < self.config.target_participation - self.config.tolerance {
            PovPace::Increase
        } else if self.current_participation > self.config.target_participation + self.config.tolerance {
            PovPace::Decrease
        } else {
            PovPace::Maintain
        }
    }

    /// Get remaining quantity
    #[inline]
    pub fn get_remaining_qty(&self) -> FixedPoint {
        self.remaining_qty
    }

    /// Get executed quantity
    #[inline]
    pub fn get_executed_qty(&self) -> FixedPoint {
        self.executed_qty
    }

    /// Get completion percentage
    #[inline]
    pub fn get_completion_pct(&self) -> FixedPoint {
        if self.total_qty.is_zero() {
            return FixedPoint::from_raw(0);
        }
        self.executed_qty / self.total_qty
    }

    /// Check if complete
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.remaining_qty.is_zero()
    }

    /// Get sequence number
    #[inline]
    pub fn get_sequence(&self) -> u64 {
        self.sequence.load(Ordering::Relaxed)
    }
}

/// POV pace recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PovPace {
    Increase,
    Decrease,
    Maintain,
}

/// VWAP tracker configuration
pub struct VwapConfig {
    /// Target VWAP price (optional, for comparison)
    pub target_vwap: Option<FixedPoint>,
    /// Maximum acceptable VWAP slippage (scaled by 10^8)
    pub max_slippage: FixedPoint,
    /// EWMA alpha for volume weighting
    pub volume_alpha: FixedPoint,
}

/// VWAP execution state machine
pub struct VwapTracker {
    config: VwapConfig,
    /// Total quantity to execute
    total_qty: FixedPoint,
    /// Remaining quantity
    remaining_qty: FixedPoint,
    /// Cumulative notional value (price * quantity)
    cumulative_notional: FixedPoint,
    /// Cumulative quantity
    cumulative_qty: FixedPoint,
    /// Current VWAP
    current_vwap: FixedPoint,
    /// Market VWAP (if available)
    market_vwap: Option<FixedPoint>,
    /// Sequence number
    sequence: AtomicU64,
}

impl VwapTracker {
    #[inline]
    pub fn new(config: VwapConfig, total_qty: FixedPoint) -> Self {
        Self {
            config,
            total_qty,
            remaining_qty: total_qty,
            cumulative_notional: FixedPoint::from_raw(0),
            cumulative_qty: FixedPoint::from_raw(0),
            current_vwap: FixedPoint::from_raw(0),
            market_vwap: None,
            sequence: AtomicU64::new(0),
        }
    }

    /// Update with a fill
    #[inline]
    pub fn on_fill(&mut self, fill_qty: FixedPoint, fill_price: FixedPoint) -> Result<(), &'static str> {
        if fill_qty.is_zero() {
            return Err("Fill quantity cannot be zero");
        }
        
        if fill_qty > self.remaining_qty {
            return Err("Fill quantity exceeds remaining");
        }

        // Update cumulative values
        let fill_notional = fill_qty * fill_price;
        self.cumulative_notional = self.cumulative_notional + fill_notional;
        self.cumulative_qty = self.cumulative_qty + fill_qty;
        self.remaining_qty = self.remaining_qty - fill_qty;

        // Recalculate VWAP
        if !self.cumulative_qty.is_zero() {
            self.current_vwap = self.cumulative_notional / self.cumulative_qty;
        }

        self.sequence.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Update market VWAP reference
    #[inline]
    pub fn update_market_vwap(&mut self, vwap: FixedPoint) {
        self.market_vwap = Some(vwap);
        self.sequence.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current VWAP
    #[inline]
    pub fn get_vwap(&self) -> FixedPoint {
        self.current_vwap
    }

    /// Get slippage vs target (if set)
    #[inline]
    pub fn get_slippage(&self) -> Option<FixedPoint> {
        self.config.target_vwap.map(|target| {
            if self.current_vwap.is_zero() {
                return FixedPoint::from_raw(0);
            }
            
            let diff = self.current_vwap - target;
            let abs_diff = diff.abs();
            abs_diff / target
        })
    }

    /// Check if slippage is within bounds
    #[inline]
    pub fn is_within_slippage_bounds(&self) -> bool {
        match self.get_slippage() {
            Some(slippage) => slippage <= self.config.max_slippage,
            None => true, // No target set
        }
    }

    /// Get remaining quantity
    #[inline]
    pub fn get_remaining_qty(&self) -> FixedPoint {
        self.remaining_qty
    }

    /// Check if complete
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.remaining_qty.is_zero()
    }

    /// Get sequence number
    #[inline]
    pub fn get_sequence(&self) -> u64 {
        self.sequence.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewma_calculator() {
        let ewma = EwmaCalculator::new(
            FixedPoint::from_fractional(20_000_000), // 0.2 alpha
            FixedPoint::from_int(100),
        );
        
        assert_eq!(ewma.get_current().to_f64(), 100.0);
        
        let new_val = ewma.update(FixedPoint::from_int(200));
        // 0.2 * 200 + 0.8 * 100 = 40 + 80 = 120
        assert!((new_val.to_f64() - 120.0).abs() < 0.01);
    }

    #[test]
    fn test_pov_tracker() {
        let config = PovConfig {
            target_participation: FixedPoint::from_fractional(5_000_000), // 5%
            min_order_size: FixedPoint::from_int(1),
            max_order_size: FixedPoint::from_int(100),
            volume_alpha: FixedPoint::from_fractional(10_000_000),
            tolerance: FixedPoint::from_fractional(1_000_000),
        };
        
        let mut tracker = PovTracker::new(config, FixedPoint::from_int(1000));
        
        // Simulate market volume
        tracker.on_market_volume(FixedPoint::from_int(10000), 1234567890);
        
        let next_order = tracker.calculate_next_order_size(FixedPoint::from_int(10000));
        // 5% of 10000 = 500, clamped to max 100
        assert_eq!(next_order.to_f64(), 100.0);
    }

    #[test]
    fn test_vwap_tracker() {
        let config = VwapConfig {
            target_vwap: Some(FixedPoint::from_int(100)),
            max_slippage: FixedPoint::from_fractional(1_000_000), // 1%
            volume_alpha: FixedPoint::from_fractional(10_000_000),
        };
        
        let mut tracker = VwapTracker::new(config, FixedPoint::from_int(100));
        
        // Fill 50 @ 99
        tracker.on_fill(FixedPoint::from_int(50), FixedPoint::from_int(99)).unwrap();
        // Fill 50 @ 101
        tracker.on_fill(FixedPoint::from_int(50), FixedPoint::from_int(101)).unwrap();
        
        // VWAP should be exactly 100
        assert_eq!(tracker.get_vwap().to_f64(), 100.0);
        assert!(tracker.is_within_slippage_bounds());
    }
}
