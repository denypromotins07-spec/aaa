// NEXUS-OMEGA Stage 34: Currency Peg Fluid Dynamics
// Chapter 3: Speculative Pressure Gradient Calculator
// File: crates/nexus_macro_physics/src/fluid/speculative_pressure_gradient.rs

//! Speculative Pressure Gradient Calculator
//!
//! Computes hydrodynamic pressure gradients from order book data
//! to detect imminent currency peg breaks.

#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]

use core::fmt;
use alloc::vec::Vec;

/// Error types for pressure gradient operations
#[derive(Debug, Clone, PartialEq)]
pub enum PressureGradientError {
    InsufficientData,
    InvalidGradient,
    NumericalOverflow,
}

impl fmt::Display for PressureGradientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientData => write!(f, "Insufficient data"),
            Self::InvalidGradient => write!(f, "Invalid gradient computation"),
            Self::NumericalOverflow => write!(f, "Numerical overflow"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PressureGradientError {}

/// Order book snapshot for pressure calculation
#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    /// Bid volumes at each price level
    pub bids: Vec<(f64, f64)>, // (price, volume)
    /// Ask volumes at each price level
    pub asks: Vec<(f64, f64)>, // (price, volume)
    /// Timestamp
    pub timestamp: u64,
}

impl OrderBookSnapshot {
    #[must_use]
    pub fn new(bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>, timestamp: u64) -> Self {
        Self { bids, asks, timestamp }
    }

    /// Compute bid-ask spread
    #[must_use]
    pub fn spread(&self) -> Option<f64> {
        let best_bid = self.bids.iter().map(|(p, _)| *p).fold(f64::NEG_INFINITY, f64::max);
        let best_ask = self.asks.iter().map(|(p, _)| *p).fold(f64::INFINITY, f64::min);
        
        if best_bid.is_finite() && best_ask.is_finite() && best_ask > best_bid {
            Some(best_ask - best_bid)
        } else {
            None
        }
    }

    /// Compute total bid volume
    #[must_use]
    pub fn total_bid_volume(&self) -> f64 {
        self.bids.iter().map(|(_, v)| *v).sum()
    }

    /// Compute total ask volume
    #[must_use]
    pub fn total_ask_volume(&self) -> f64 {
        self.asks.iter().map(|(_, v)| *v).sum()
    }

    /// Compute order imbalance (positive = buying pressure)
    #[must_use]
    pub fn order_imbalance(&self) -> f64 {
        let bid_vol = self.total_bid_volume();
        let ask_vol = self.total_ask_volume();
        
        let total = bid_vol + ask_vol;
        if total > 0.0 {
            (bid_vol - ask_vol) / total
        } else {
            0.0
        }
    }
}

/// Pressure gradient state
#[derive(Debug, Clone)]
pub struct PressureGradientState {
    /// Current pressure value
    pub pressure: f64,
    /// Pressure gradient (spatial derivative)
    pub gradient: f64,
    /// Pressure time derivative
    pub time_derivative: f64,
    /// Critical threshold for peg break
    pub critical_threshold: f64,
    /// Distance to critical threshold
    pub margin_to_failure: f64,
}

impl PressureGradientState {
    #[must_use]
    pub fn is_critical(&self) -> bool {
        self.pressure >= self.critical_threshold || self.margin_to_failure <= 0.0
    }

    #[must_use]
    pub fn warning_level(&self) -> WarningLevel {
        let ratio = self.pressure / self.critical_threshold;
        
        if ratio >= 1.0 {
            WarningLevel::Critical
        } else if ratio >= 0.8 {
            WarningLevel::High
        } else if ratio >= 0.5 {
            WarningLevel::Moderate
        } else {
            WarningLevel::Low
        }
    }
}

/// Warning level for speculative pressure
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WarningLevel {
    Low,
    Moderate,
    High,
    Critical,
}

/// Speculative Pressure Gradient Calculator
pub struct SpeculativePressureCalculator {
    /// Historical snapshots for trend analysis
    history: Vec<OrderBookSnapshot>,
    /// Maximum history size
    max_history: usize,
    /// Baseline pressure (calibrated to normal conditions)
    baseline_pressure: f64,
    /// Critical pressure threshold
    critical_threshold: f64,
}

impl SpeculativePressureCalculator {
    /// Create a new pressure calculator
    #[must_use]
    pub fn new(critical_threshold: f64) -> Self {
        Self {
            history: Vec::new(),
            max_history: 100,
            baseline_pressure: 0.0,
            critical_threshold,
        }
    }

    /// Set maximum history size
    pub fn with_max_history(mut self, size: usize) -> Self {
        self.max_history = size;
        self
    }

    /// Add an order book snapshot
    pub fn add_snapshot(&mut self, snapshot: OrderBookSnapshot) {
        self.history.push(snapshot);
        
        // Trim history if needed
        while self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Compute current pressure gradient state
    #[must_use]
    pub fn compute_pressure_state(&self) -> Option<PressureGradientState> {
        if self.history.len() < 2 {
            return None;
        }

        let latest = self.history.last()?;
        let previous = self.history.get(self.history.len().saturating_sub(1))?;

        // Compute pressure from order imbalance and spread
        let imbalance = latest.order_imbalance();
        let spread = latest.spread().unwrap_or(0.0);
        
        // Pressure formula: combines imbalance, spread widening, and volume
        let volume_ratio = latest.total_ask_volume() / (latest.total_bid_volume() + 1.0);
        let pressure = (1.0 - imbalance) * 0.5 + spread * 0.3 + (volume_ratio - 1.0).abs() * 0.2;
        
        // Normalize pressure
        let normalized_pressure = pressure.clamp(0.0, 1.0) * self.critical_threshold;

        // Compute time derivative
        let prev_imbalance = previous.order_imbalance();
        let time_derivative = (imbalance - prev_imbalance) / 
            ((latest.timestamp.saturating_sub(previous.timestamp)) as f64 + 1.0);

        // Compute gradient (spatial derivative approximation using volume distribution)
        let gradient = self.compute_spatial_gradient(latest);

        // Margin to failure
        let margin_to_failure = self.critical_threshold - normalized_pressure;

        Some(PressureGradientState {
            pressure: normalized_pressure,
            gradient,
            time_derivative,
            critical_threshold: self.critical_threshold,
            margin_to_failure,
        })
    }

    /// Compute spatial gradient from volume distribution
    fn compute_spatial_gradient(&self, snapshot: &OrderBookSnapshot) -> f64 {
        if snapshot.bids.is_empty() || snapshot.asks.is_empty() {
            return 0.0;
        }

        // Compute weighted average prices
        let mut bid_value = 0.0;
        let mut bid_volume = 0.0;
        for &(price, vol) in &snapshot.bids {
            bid_value += price * vol;
            bid_volume += vol;
        }
        let avg_bid = if bid_volume > 0.0 { bid_value / bid_volume } else { 0.0 };

        let mut ask_value = 0.0;
        let mut ask_volume = 0.0;
        for &(price, vol) in &snapshot.asks {
            ask_value += price * vol;
            ask_volume += vol;
        }
        let avg_ask = if ask_volume > 0.0 { ask_value / ask_volume } else { 0.0 };

        // Gradient is proportional to bid-ask separation
        if avg_bid > 0.0 {
            (avg_ask - avg_bid) / avg_bid
        } else {
            0.0
        }
    }

    /// Estimate time to peg break based on current pressure trend
    #[must_use]
    pub fn estimate_time_to_break(&self) -> Option<f64> {
        let state = self.compute_pressure_state()?;
        
        if state.time_derivative <= 0.0 {
            return Some(f64::INFINITY); // Pressure decreasing, no break expected
        }

        let remaining_pressure = state.margin_to_failure;
        if remaining_pressure <= 0.0 {
            return Some(0.0); // Already broken
        }

        // Time = distance / rate
        Some(remaining_pressure / state.time_derivative.abs())
    }

    /// Calibrate baseline pressure from historical data
    pub fn calibrate_baseline(&mut self) -> Result<(), PressureGradientError> {
        if self.history.len() < 10 {
            return Err(PressureGradientError::InsufficientData);
        }

        let mut pressures = Vec::with_capacity(self.history.len());
        
        for snapshot in &self.history {
            let imbalance = snapshot.order_imbalance();
            let pressure = (1.0 - imbalance) * 0.5;
            pressures.push(pressure);
        }

        pressures.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        
        // Use median of lower quartile as baseline
        let q1_idx = pressures.len() / 4;
        self.baseline_pressure = pressures[q1_idx];

        Ok(())
    }

    /// Get baseline pressure
    #[must_use]
    pub const fn baseline_pressure(&self) -> f64 {
        self.baseline_pressure
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_book_imbalance() {
        let snapshot = OrderBookSnapshot::new(
            vec![(1.0, 100.0), (0.99, 200.0)],
            vec![(1.01, 50.0), (1.02, 100.0)],
            1000,
        );

        let imbalance = snapshot.order_imbalance();
        assert!(imbalance > 0.0); // More bids than asks
    }

    #[test]
    fn test_pressure_calculator() {
        let mut calc = SpeculativePressureCalculator::new(1.0);
        
        let snapshot1 = OrderBookSnapshot::new(
            vec![(1.0, 100.0)],
            vec![(1.01, 100.0)],
            1000,
        );
        let snapshot2 = OrderBookSnapshot::new(
            vec![(1.0, 50.0)],
            vec![(1.01, 200.0)],
            1001,
        );

        calc.add_snapshot(snapshot1);
        calc.add_snapshot(snapshot2);

        let state = calc.compute_pressure_state();
        assert!(state.is_some());
    }

    #[test]
    fn test_warning_levels() {
        let state_low = PressureGradientState {
            pressure: 0.3,
            gradient: 0.1,
            time_derivative: 0.01,
            critical_threshold: 1.0,
            margin_to_failure: 0.7,
        };
        assert_eq!(state_low.warning_level(), WarningLevel::Low);

        let state_critical = PressureGradientState {
            pressure: 1.2,
            gradient: 0.5,
            time_derivative: 0.1,
            critical_threshold: 1.0,
            margin_to_failure: -0.2,
        };
        assert_eq!(state_critical.warning_level(), WarningLevel::Critical);
    }
}
