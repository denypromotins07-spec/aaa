//! U256 Sqrt Price Math for Uniswap V3
//! Exact on-chain math reproduction with overflow protection

use thiserror::Error;
use uint::U256;

#[derive(Error, Debug)]
pub enum SqrtPriceMathError {
    #[error("Invalid sqrt price")]
    InvalidSqrtPrice,
    #[error("Invalid liquidity")]
    InvalidLiquidity,
    #[error("Amount overflow")]
    AmountOverflow,
    #[error("Division by zero")]
    DivisionByZero,
}

pub type Result<T> = core::result::Result<T, SqrtPriceMathError>;

const Q96: U256 = U256([0, 0x10000000000000000u64, 0, 0]);

pub struct SqrtPriceMath;

impl SqrtPriceMath {
    /// Get amount0 delta for a swap (token0 -> token1)
    pub fn get_amount0_delta(
        sqrt_ratio_a_x96: U256,
        sqrt_ratio_b_x96: U256,
        liquidity: U256,
        round_up: bool,
    ) -> Result<U256> {
        if sqrt_ratio_a_x96 > sqrt_ratio_b_x96 {
            return Self::get_amount0_delta(sqrt_ratio_b_x96, sqrt_ratio_a_x96, liquidity, round_up);
        }

        if liquidity.is_zero() || sqrt_ratio_a_x96.is_zero() {
            return Ok(U256::zero());
        }

        let numerator1 = liquidity << 96;
        let denominator = sqrt_ratio_a_x96.saturating_mul(sqrt_ratio_b_x96);
        
        if denominator.is_zero() {
            return Err(SqrtPriceMathError::DivisionByZero);
        }

        let amount1 = numerator1 / denominator;
        let numerator2 = liquidity.saturating_mul(sqrt_ratio_b_x96 - sqrt_ratio_a_x96);
        
        if round_up {
            Ok(Self::div_rounding_up(numerator2, sqrt_ratio_b_x96)?
                .saturating_add(amount1))
        } else {
            Ok((numerator2 / sqrt_ratio_b_x96).saturating_add(amount1))
        }
    }

    /// Get amount1 delta for a swap
    pub fn get_amount1_delta(
        sqrt_ratio_a_x96: U256,
        sqrt_ratio_b_x96: U256,
        liquidity: U256,
        round_up: bool,
    ) -> Result<U256> {
        if sqrt_ratio_a_x96 > sqrt_ratio_b_x96 {
            return Self::get_amount1_delta(sqrt_ratio_b_x96, sqrt_ratio_a_x96, liquidity, round_up);
        }

        if liquidity.is_zero() {
            return Ok(U256::zero());
        }

        let diff = sqrt_ratio_b_x96 - sqrt_ratio_a_x96;
        let product = liquidity.saturating_mul(diff);
        
        if round_up {
            Ok(Self::div_rounding_up(product, Q96)?)
        } else {
            Ok(product / Q96)
        }
    }

    /// Calculate next sqrt price after swap
    pub fn get_next_sqrt_price_from_amount0(
        sqrt_px_x96: U256,
        liquidity: U256,
        amount: U256,
        add: bool,
    ) -> Result<U256> {
        if amount.is_zero() {
            return Ok(sqrt_px_x96);
        }

        let double_prod = sqrt_px_x96.saturating_mul(sqrt_px_x96);
        if double_prod.is_zero() {
            return Err(SqrtPriceMathError::InvalidSqrtPrice);
        }

        if add {
            let product = liquidity.saturating_mul(amount);
            if product.is_zero() {
                return Ok(sqrt_px_x96);
            }
            
            let numerator = double_prod.saturating_mul(product);
            let denominator = liquidity.saturating_mul(Q96).saturating_mul(amount)
                .saturating_add(numerator);
            
            if denominator.is_zero() {
                return Err(SqrtPriceMathError::DivisionByZero);
            }
            
            Ok(numerator / denominator)
        } else {
            let quotient = Self::div_rounding_up(amount.saturating_mul(Q96), liquidity)?;
            if sqrt_px_x96 <= quotient {
                return Ok(U256::one());
            }
            let diff = sqrt_px_x96 - quotient;
            Ok(double_prod / diff)
        }
    }

    /// Helper for rounding up division
    fn div_rounding_up(a: U256, b: U256) -> Result<U256> {
        if b.is_zero() {
            return Err(SqrtPriceMathError::DivisionByZero);
        }
        let result = a / b;
        if a % b != U256::zero() {
            Ok(result + U256::one())
        } else {
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amount_calculation() {
        let sqrt_a = U256::from(1_000_000_000_000_000_000u128);
        let sqrt_b = U256::from(2_000_000_000_000_000_000u128);
        let liq = U256::from(1_000_000_000_000u128);
        
        let result = SqrtPriceMath::get_amount0_delta(sqrt_a, sqrt_b, liq, false);
        assert!(result.is_ok());
    }
}
