//! Fixed-point arithmetic for PnL calculations.
//! 
//! CRITICAL: Uses strictly i128 scaled integers to avoid floating-point rounding errors
//! that could trigger false liquidations on highly leveraged positions.

use std::ops::{Add, Sub, Mul, Div};

/// Scale factor: 1e18 (wei-like precision)
pub const SCALE: i128 = 1_000_000_000_000_000_000;

/// Represents a fixed-point number with 1e18 precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FixedPoint(pub i128);

impl FixedPoint {
    pub fn new(value: i128) -> Self {
        Self(value * SCALE)
    }

    pub fn from_scaled(scaled: i128) -> Self {
        Self(scaled)
    }

    pub fn to_scaled(&self) -> i128 {
        self.0
    }

    pub fn to_f64_approx(&self) -> f64 {
        self.0 as f64 / SCALE as f64
    }

    /// Multiply two fixed-point numbers without overflow if possible.
    pub fn checked_mul(&self, other: &FixedPoint) -> Option<FixedPoint> {
        // (a * SCALE) * (b * SCALE) / SCALE = a * b * SCALE
        let product = self.0.checked_mul(other.0)?;
        Some(FixedPoint(product / SCALE))
    }

    /// Divide two fixed-point numbers.
    pub fn checked_div(&self, other: &FixedPoint) -> Option<FixedPoint> {
        if other.0 == 0 {
            return None;
        }
        // (a * SCALE) / (b * SCALE) * SCALE = (a / b) * SCALE
        let quotient = self.0.checked_mul(SCALE)?;
        Some(FixedPoint(quotient / other.0))
    }

    pub fn checked_add(&self, other: &FixedPoint) -> Option<FixedPoint> {
        self.0.checked_add(other.0).map(FixedPoint)
    }

    pub fn checked_sub(&self, other: &FixedPoint) -> Option<FixedPoint> {
        self.0.checked_sub(other.0).map(FixedPoint)
    }

    pub fn abs(&self) -> FixedPoint {
        FixedPoint(self.0.abs())
    }

    pub fn is_negative(&self) -> bool {
        self.0 < 0
    }

    pub fn zero() -> Self {
        Self(0)
    }
}

/// Calculate unrealized PnL for a position using fixed-point math.
/// 
/// Formula: (mark_price - entry_price) * position_size * direction
/// Where direction: 1 for long, -1 for short
pub fn calculate_unrealized_pnl(
    entry_price: FixedPoint,
    mark_price: FixedPoint,
    position_size: i128, // In base units (e.g., satoshis)
    is_long: bool,
) -> Result<i128, &'static str> {
    let price_diff = if is_long {
        mark_price.checked_sub(&entry_price)
    } else {
        entry_price.checked_sub(&mark_price)
    }.ok_or("Price difference overflow")?;

    // PnL = price_diff * size / SCALE
    // position_size is already in base units, price is scaled
    let pnl_scaled = price_diff.0.checked_mul(position_size)
        .ok_or("PnL calculation overflow")?;

    // Result is in quote currency scaled by 1e18
    Ok(pnl_scaled)
}

/// Calculate maintenance margin requirement.
/// 
/// Formula: notional_value * maintenance_margin_rate
pub fn calculate_maintenance_margin(
    notional_value: FixedPoint,
    mm_rate: FixedPoint, // e.g., 0.005 for 0.5%
) -> Result<FixedPoint, &'static str> {
    notional_value.checked_mul(&mm_rate)
        .ok_or("Maintenance margin calculation overflow")
}

/// Calculate margin ratio: equity / maintenance_margin
/// Returns scaled value where 1.0 = 1e18
pub fn calculate_margin_ratio(
    equity: FixedPoint,
    maintenance_margin: FixedPoint,
) -> Result<FixedPoint, &'static str> {
    if maintenance_margin.0 == 0 {
        return Ok(FixedPoint(i128::MAX)); // Infinite ratio if no MM required
    }
    equity.checked_div(&maintenance_margin)
        .ok_or("Margin ratio calculation overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_float_drift_on_large_position() {
        // Simulate a $1M position at 50x leverage
        // Entry: 50000.0, Mark: 50001.0 (tiny $1 move)
        let entry = FixedPoint::new(50000);
        let mark = FixedPoint::new(50001);
        let size = 20_000_000_000; // 20 BTC in satoshis (approx $1M)

        let pnl = calculate_unrealized_pnl(entry, mark, size, true).unwrap();
        
        // Expected: $20,000 PnL = 20000 * 1e18
        let expected = 20_000 * SCALE;
        assert_eq!(pnl, expected);
        
        // Verify no floating point drift occurred
        // With f64, we might get 20000.000000000004 or similar
        // With i128, it's exactly 20000000000000000000000
    }

    #[test]
    fn test_margin_ratio_calculation() {
        let equity = FixedPoint::new(10000); // $10k
        let mm = FixedPoint::new(5000); // $5k maintenance
        
        let ratio = calculate_margin_ratio(equity, mm).unwrap();
        assert_eq!(ratio.to_scaled(), 2 * SCALE); // 2.0x
    }
}
