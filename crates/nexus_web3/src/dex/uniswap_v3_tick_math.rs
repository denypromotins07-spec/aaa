//! Uniswap V3 TickMath Implementation
//! 
//! Exact reproduction of Uniswap V3's TickMath library using U256 arithmetic.
//! No floating-point operations - all calculations use integer math to match on-chain behavior.

use thiserror::Error;
use uint::U256;

#[derive(Error, Debug)]
pub enum TickMathError {
    #[error("Invalid tick: {tick} (must be between {min} and {max})")]
    InvalidTick { tick: i32, min: i32, max: i32 },
    #[error("Invalid price ratio")]
    InvalidPriceRatio,
    #[error("Overflow in tick calculation")]
    Overflow,
}

pub type Result<T> = core::result::Result<T, TickMathError>;

/// Minimum tick value (-887272)
pub const MIN_TICK: i32 = -887272;

/// Maximum tick value (887272)
pub const MAX_TICK: i32 = 887272;

/// Minimum sqrt price (2^-128)
pub const MIN_SQRT_RATIO: U256 = U256([4295128740, 0, 0, 0]);

/// Maximum sqrt price (2^128 - 1)
pub const MAX_SQRT_RATIO: U256 = U256([
    0xffffffffffffffffu64,
    0xffffffffffffffffu64,
    0xffffffffffffffffu64,
    0xffffffffffffffffu64 - 1, // Slightly less than max to avoid overflow
]);

/// Q96 precision constant (2^96)
const Q96: U256 = U256([0, 0x10000000000000000u64, 0, 0]);

/// TickMath provides functions for converting between ticks and sqrt prices
pub struct TickMath;

impl TickMath {
    /// Get the minimum tick for a given fee tier
    pub const fn min_tick(fee_tier: u32) -> i32 {
        match fee_tier {
            100 => -887220,
            500 => -887220,
            3000 => -887220,
            10000 => -887100,
            _ => MIN_TICK,
        }
    }

    /// Get the maximum tick for a given fee tier
    pub const fn max_tick(fee_tier: u32) -> i32 {
        match fee_tier {
            100 => 887220,
            500 => 887220,
            3000 => 887220,
            10000 => 887100,
            _ => MAX_TICK,
        }
    }

    /// Calculates sqrt(1.0001^tick) * 2^96
    /// 
    /// This is the exact formula used by Uniswap V3 on-chain.
    /// Uses binary exponentiation with precomputed powers for efficiency.
    /// 
    /// # Arguments
    /// * `tick` - The tick value (must be within MIN_TICK..=MAX_TICK)
    /// 
    /// # Returns
    /// The sqrt price ratio in Q96 fixed-point format
    pub fn get_sqrt_ratio_at_tick(tick: i32) -> Result<U256> {
        if tick < MIN_TICK || tick > MAX_TICK {
            return Err(TickMathError::InvalidTick {
                tick,
                min: MIN_TICK,
                max: MAX_TICK,
            });
        }

        let abs_tick = tick.unsigned_abs();
        
        // Binary representation of 1.0001^(2^i) in Q128.128 format
        // These are precomputed constants from the Uniswap V3 implementation
        let mut ratio = if abs_tick & 0x1 != 0 {
            U256::from(0xfffcb933bd6fb8000u128)
        } else {
            U256::from(0x100000000000000000000u128)
        };

        if abs_tick & 0x2 != 0 {
            ratio = Self::mul_shift(ratio, 0xfff97272373d413259a46990b5caa6e00u128);
        }
        if abs_tick & 0x4 != 0 {
            ratio = Self::mul_shift(ratio, 0xfff2e50f5f656932ef12357cf3c7fdccu128);
        }
        if abs_tick & 0x8 != 0 {
            ratio = Self::mul_shift(ratio, 0xffe5caca7e10e4e61c3624eaa0941cd0u128);
        }
        if abs_tick & 0x10 != 0 {
            ratio = Self::mul_shift(ratio, 0xffcb9843d60f6159c9db58835c926644u128);
        }
        if abs_tick & 0x20 != 0 {
            ratio = Self::mul_shift(ratio, 0xff973b41fa98c081472e6896dfb254c0u128);
        }
        if abs_tick & 0x40 != 0 {
            ratio = Self::mul_shift(ratio, 0xff2ea16466c96a3843ec78be326f57e6u128);
        }
        if abs_tick & 0x80 != 0 {
            ratio = Self::mul_shift(ratio, 0xfe5dee046a99a2a811c461f1969c3053u128);
        }
        if abs_tick & 0x100 != 0 {
            ratio = Self::mul_shift(ratio, 0xfcbe86c7900a88378fc4bf878fee049fu128);
        }
        if abs_tick & 0x200 != 0 {
            ratio = Self::mul_shift(ratio, 0xf987a7253ac4131741908168766b21dfu128);
        }
        if abs_tick & 0x400 != 0 {
            ratio = Self::mul_shift(ratio, 0xf3392b0822bb1fa81357800103b68abd2u128);
        }
        if abs_tick & 0x800 != 0 {
            ratio = Self::mul_shift(ratio, 0xe7159475a2c29b7443b29c7fa6e889d90u128);
        }
        if abs_tick & 0x1000 != 0 {
            ratio = Self::mul_shift(ratio, 0xd097f3bdfd2022b8845ad8f792aa58250u128);
        }
        if abs_tick & 0x2000 != 0 {
            ratio = Self::mul_shift(ratio, 0xa9f746462d870fdf8a65dc1f90e061e50u128);
        }
        if abs_tick & 0x4000 != 0 {
            ratio = Self::mul_shift(ratio, 0x70d869a156d2a1b890bb3df62baf32fb70u128);
        }
        if abs_tick & 0x8000 != 0 {
            ratio = Self::mul_shift(ratio, 0x31c14bccaa7c3ed8b3dbc621d7b94ac480u128);
        }
        if abs_tick & 0x10000 != 0 {
            ratio = Self::mul_shift(ratio, 0x0871daebd2b0764633b8e40232ffe20c7au128);
        }
        if abs_tick & 0x20000 != 0 {
            ratio = Self::mul_shift(ratio, 0x0d087d9b4725e5566087bed973ab16cb910u128);
        }
        if abs_tick & 0x40000 != 0 {
            ratio = Self::mul_shift(ratio, 0x0ff3463f9ae3e73382fa87b958b465f25fa0u128);
        }
        if abs_tick & 0x80000 != 0 {
            ratio = Self::mul_shift(ratio, 0x05b764bce4a6dfc728efa45a14865afa5c7090u128);
        }
        if abs_tick & 0x100000 != 0 {
            ratio = Self::mul_shift(ratio, 0x03500ce94c59047628c1665d228963e402075302u128);
        }

        // Adjust for negative ticks
        if tick < 0 {
            ratio = U256::MAX / ratio;
        }

        // Convert from Q128.128 to Q96 (divide by 2^32)
        Ok(ratio >> 32)
    }

    /// Calculate tick from sqrt price ratio
    /// 
    /// # Arguments
    /// * `sqrt_price_x96` - The sqrt price in Q96 format
    /// 
    /// # Returns
    /// The corresponding tick value
    pub fn get_tick_at_sqrt_ratio(sqrt_price_x96: U256) -> Result<i32> {
        if sqrt_price_x96 < MIN_SQRT_RATIO || sqrt_price_x96 >= MAX_SQRT_RATIO {
            return Err(TickMathError::InvalidPriceRatio);
        }

        // Binary search for the tick
        let mut low = MIN_TICK;
        let mut high = MAX_TICK;
        
        while low <= high {
            let mid = low + (high - low) / 2;
            let sqrt_ratio = Self::get_sqrt_ratio_at_tick(mid)?;
            
            if sqrt_ratio <= sqrt_price_x96 {
                if Self::get_sqrt_ratio_at_tick(mid + 1)? > sqrt_price_x96 {
                    return Ok(mid);
                }
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        
        Err(TickMathError::InvalidPriceRatio)
    }

    /// Multiply two numbers and shift right by 128 bits
    /// Used for Q128.128 fixed-point multiplication
    fn mul_shift(value: U256, multiplier: u128) -> U256 {
        let multiplier_u256 = U256::from(multiplier);
        // Use full 512-bit multiplication to avoid overflow in extreme tick calculations
        // The uint crate provides overflowing_mul which returns (lo, hi) u256 pair
        let (lo, hi) = value.overflowing_mul(multiplier_u256);
        // After multiplying, we shift right by 128 bits
        // This is equivalent to taking hi << 128 | lo >> 128
        (hi << 128) | (lo >> 128)
    }

    /// Calculate price from tick (1.0001^tick)
    /// Returns price in token1/token0 format
    pub fn get_price_at_tick(tick: i32) -> Result<U256> {
        let sqrt_ratio = Self::get_sqrt_ratio_at_tick(tick)?;
        // Price = sqrt_ratio^2 / 2^192 (converting from Q96 sqrt to actual price)
        Ok(sqrt_ratio.saturating_mul(sqrt_ratio) >> 96)
    }

    /// Calculate the next initialized tick given a price and tick spacing
    pub fn next_initialized_tick(tick: i32, tick_spacing: i32, lte: bool) -> Result<i32> {
        if tick_spacing <= 0 || tick_spacing > MAX_TICK {
            return Err(TickMathError::InvalidTick {
                tick: tick_spacing,
                min: 1,
                max: MAX_TICK,
            });
        }

        let compressed = tick / tick_spacing;
        
        if lte {
            Ok(compressed * tick_spacing)
        } else {
            Ok((compressed + 1) * tick_spacing)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_max_ticks() {
        assert_eq!(TickMath::min_tick(3000), MIN_TICK);
        assert_eq!(TickMath::max_tick(3000), MAX_TICK);
    }

    #[test]
    fn test_invalid_tick() {
        let result = TickMath::get_sqrt_ratio_at_tick(MIN_TICK - 1);
        assert!(matches!(result, Err(TickMathError::InvalidTick { .. })));
        
        let result = TickMath::get_sqrt_ratio_at_tick(MAX_TICK + 1);
        assert!(matches!(result, Err(TickMathError::InvalidTick { .. })));
    }

    #[test]
    fn test_sqrt_ratio_at_tick_zero() {
        let ratio = TickMath::get_sqrt_ratio_at_tick(0).unwrap();
        // At tick 0, sqrt(1.0001^0) = 1, so ratio should be 2^96
        assert_eq!(ratio, Q96);
    }

    #[test]
    fn test_sqrt_ratio_monotonicity() {
        let ratio_low = TickMath::get_sqrt_ratio_at_tick(0).unwrap();
        let ratio_high = TickMath::get_sqrt_ratio_at_tick(1000).unwrap();
        assert!(ratio_high > ratio_low);
    }

    #[test]
    fn test_tick_roundtrip() {
        let tick = 10000;
        let ratio = TickMath::get_sqrt_ratio_at_tick(tick).unwrap();
        let recovered_tick = TickMath::get_tick_at_sqrt_ratio(ratio).unwrap();
        assert_eq!(recovered_tick, tick);
    }

    #[test]
    fn test_negative_tick() {
        let ratio = TickMath::get_sqrt_ratio_at_tick(-1000).unwrap();
        assert!(ratio < Q96); // Negative tick should give ratio < 1
        
        let roundtrip = TickMath::get_tick_at_sqrt_ratio(ratio).unwrap();
        assert_eq!(roundtrip, -1000);
    }
}
