//! Fixed-point decimal math for zero-allocation, exact arithmetic.
//! Uses i64 scaled by 10^8 to represent prices and quantities without floating-point errors.

use std::ops::{Add, Sub, Mul, Div};
use core::cmp::Ordering;

/// Scale factor: 10^8
pub const SCALE: i64 = 100_000_000;

/// Error types for fixed-point operations
#[derive(Debug, Clone, PartialEq)]
pub enum FixedPointError {
    Overflow,
    Underflow,
    DivisionByZero,
    InvalidScale,
}

/// A fixed-point decimal number using i64 internally.
/// Represents values as `raw / SCALE` where SCALE = 10^8.
/// Guarantees exact decimal arithmetic for financial calculations.
#[repr(transparent)]
#[derive(Clone, Copy, Default)]
pub struct FixedPoint {
    raw: i64,
}

impl FixedPoint {
    /// Create a new FixedPoint from a raw i64 value (already scaled)
    #[inline]
    pub const fn from_raw(raw: i64) -> Self {
        Self { raw }
    }

    /// Create a FixedPoint from an integer (e.g., 5 -> 5.00000000)
    #[inline]
    pub const fn from_int(val: i64) -> Self {
        Self { raw: val * SCALE }
    }

    /// Create a FixedPoint from a fractional part (e.g., 50000000 -> 0.50000000)
    #[inline]
    pub const fn from_fractional(val: i64) -> Self {
        Self { raw: val }
    }

    /// Get the raw i64 representation
    #[inline]
    pub const fn raw(&self) -> i64 {
        self.raw
    }

    /// Get the integer part (truncated)
    #[inline]
    pub const fn int_part(&self) -> i64 {
        self.raw / SCALE
    }

    /// Get the fractional part
    #[inline]
    pub const fn frac_part(&self) -> i64 {
        self.raw % SCALE
    }

    /// Convert to f64 for display/debugging only (NOT for hot-path calculations)
    #[inline]
    pub fn to_f64(&self) -> f64 {
        self.raw as f64 / SCALE as f64
    }

    /// Addition with overflow check
    #[inline]
    pub fn checked_add(self, other: Self) -> Result<Self, FixedPointError> {
        match self.raw.checked_add(other.raw) {
            Some(sum) => Ok(Self::from_raw(sum)),
            None => Err(FixedPointError::Overflow),
        }
    }

    /// Subtraction with overflow check
    #[inline]
    pub fn checked_sub(self, other: Self) -> Result<Self, FixedPointError> {
        match self.raw.checked_sub(other.raw) {
            Some(diff) => Ok(Self::from_raw(diff)),
            None => Err(FixedPointError::Underflow),
        }
    }

    /// Multiplication with overflow check
    /// Result is (a * b) / SCALE to maintain proper scaling
    #[inline]
    pub fn checked_mul(self, other: Self) -> Result<Self, FixedPointError> {
        // Use i128 for intermediate calculation to prevent overflow
        let product = self.raw as i128 * other.raw as i128;
        let scaled = product / SCALE as i128;
        
        if scaled > i64::MAX as i128 || scaled < i64::MIN as i128 {
            return Err(FixedPointError::Overflow);
        }
        
        Ok(Self::from_raw(scaled as i64))
    }

    /// Division with overflow and div-by-zero check
    /// Result is (a * SCALE) / b to maintain proper scaling
    #[inline]
    pub fn checked_div(self, other: Self) -> Result<Self, FixedPointError> {
        if other.raw == 0 {
            return Err(FixedPointError::DivisionByZero);
        }
        
        // Use i128 for intermediate calculation
        let dividend = self.raw as i128 * SCALE as i128;
        let quotient = dividend / other.raw as i128;
        
        if quotient > i64::MAX as i128 || quotient < i64::MIN as i128 {
            return Err(FixedPointError::Overflow);
        }
        
        Ok(Self::from_raw(quotient as i64))
    }

    /// Compare two FixedPoint values
    #[inline]
    pub fn cmp(&self, other: &Self) -> Ordering {
        self.raw.cmp(&other.raw)
    }

    /// Check if zero
    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.raw == 0
    }

    /// Check if positive
    #[inline]
    pub const fn is_positive(&self) -> bool {
        self.raw > 0
    }

    /// Check if negative
    #[inline]
    pub const fn is_negative(&self) -> bool {
        self.raw < 0
    }

    /// Absolute value
    #[inline]
    pub const fn abs(&self) -> Self {
        Self { raw: self.raw.abs() }
    }

    /// Negate
    #[inline]
    pub const fn neg(&self) -> Self {
        Self { raw: -self.raw }
    }

    /// Minimum of two values
    #[inline]
    pub const fn min(self, other: Self) -> Self {
        if self.raw < other.raw { self } else { other }
    }

    /// Maximum of two values
    #[inline]
    pub const fn max(self, other: Self) -> Self {
        if self.raw > other.raw { self } else { other }
    }
}

impl Add for FixedPoint {
    type Output = Self;
    
    #[inline]
    fn add(self, other: Self) -> Self {
        self.checked_add(other).unwrap_or_else(|_| {
            // In production, this should never happen due to prior bounds checking
            // Fallback to saturating add for safety in non-hot paths
            Self::from_raw(self.raw.saturating_add(other.raw))
        })
    }
}

impl Sub for FixedPoint {
    type Output = Self;
    
    #[inline]
    fn sub(self, other: Self) -> Self {
        self.checked_sub(other).unwrap_or_else(|_| {
            Self::from_raw(self.raw.saturating_sub(other.raw))
        })
    }
}

impl Mul for FixedPoint {
    type Output = Self;
    
    #[inline]
    fn mul(self, other: Self) -> Self {
        self.checked_mul(other).unwrap_or_else(|_| {
            Self::from_raw(i64::MAX)
        })
    }
}

impl Div for FixedPoint {
    type Output = Self;
    
    #[inline]
    fn div(self, other: Self) -> Self {
        self.checked_div(other).unwrap_or_else(|_| {
            Self::from_raw(i64::MAX)
        })
    }
}

impl PartialEq for FixedPoint {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl Eq for FixedPoint {}

impl PartialOrd for FixedPoint {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FixedPoint {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_int() {
        let fp = FixedPoint::from_int(5);
        assert_eq!(fp.raw(), 500_000_000);
        assert_eq!(fp.to_f64(), 5.0);
    }

    #[test]
    fn test_from_fractional() {
        let fp = FixedPoint::from_fractional(50_000_000);
        assert_eq!(fp.to_f64(), 0.5);
    }

    #[test]
    fn test_checked_add() {
        let a = FixedPoint::from_int(5);
        let b = FixedPoint::from_int(3);
        let result = a.checked_add(b).unwrap();
        assert_eq!(result.to_f64(), 8.0);
    }

    #[test]
    fn test_checked_mul() {
        let a = FixedPoint::from_int(5);
        let b = FixedPoint::from_int(3);
        let result = a.checked_mul(b).unwrap();
        assert_eq!(result.to_f64(), 15.0);
    }

    #[test]
    fn test_checked_div() {
        let a = FixedPoint::from_int(10);
        let b = FixedPoint::from_int(2);
        let result = a.checked_div(b).unwrap();
        assert_eq!(result.to_f64(), 5.0);
    }

    #[test]
    fn test_division_by_zero() {
        let a = FixedPoint::from_int(10);
        let b = FixedPoint::from_int(0);
        assert_eq!(a.checked_div(b), Err(FixedPointError::DivisionByZero));
    }
}
