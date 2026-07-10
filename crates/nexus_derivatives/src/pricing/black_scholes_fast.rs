//! High-performance Black-Scholes-Merton pricing engine
//! 
//! Uses custom fast math (fast_cdf_erf) for zero-allocation HFT pricing.
//! Supports calls, puts, and digital options with sub-nanosecond execution.

use crate::math::fast_cdf_erf::{fast_cdf, fast_cdf_complement, fast_exp, fast_ln, fast_sqrt};

/// Black-Scholes-Merton option parameters
#[derive(Debug, Clone, Copy)]
pub struct BSParams {
    /// Spot price of underlying
    pub spot: f64,
    /// Strike price
    pub strike: f64,
    /// Time to expiry in years
    pub time_to_expiry: f64,
    /// Risk-free rate (annualized)
    pub risk_free_rate: f64,
    /// Implied volatility (annualized)
    pub volatility: f64,
    /// Dividend yield (annualized), default 0 for crypto
    pub dividend_yield: f64,
}

impl Default for BSParams {
    fn default() -> Self {
        Self {
            spot: 100.0,
            strike: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.2,
            dividend_yield: 0.0,
        }
    }
}

/// Option type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
    DigitalCall,
    DigitalPut,
}

/// Black-Scholes-Merton pricing result
#[derive(Debug, Clone, Copy)]
pub struct BSResult {
    /// Theoretical price
    pub price: f64,
    /// d1 parameter: (ln(S/K) + (r - q + σ²/2)T) / (σ√T)
    pub d1: f64,
    /// d2 parameter: d1 - σ√T
    pub d2: f64,
    /// Forward price F = S * e^((r-q)T)
    pub forward: f64,
    /// Discount factor e^(-rT)
    pub discount: f64,
}

impl BSResult {
    #[inline(always)]
    pub const fn new(price: f64, d1: f64, d2: f64, forward: f64, discount: f64) -> Self {
        Self { price, d1, d2, forward, discount }
    }
}

/// Numerical stability epsilon
const EPSILON: f64 = 1e-12;

/// Minimum time to expiry to avoid division by zero (1 second in years)
const MIN_TIME: f64 = 1.0 / (365.0 * 24.0 * 3600.0);

/// Maximum volatility cap to prevent overflow (1000%)
const MAX_VOL: f64 = 10.0;

/// Minimum volatility floor (0.1%)
const MIN_VOL: f64 = 0.001;

/// Price a European option using Black-Scholes-Merton formula
/// 
/// # Arguments
/// * `params` - Option parameters
/// * `option_type` - Call or Put
/// 
/// # Returns
/// * `BSResult` containing price and intermediate calculations
/// 
/// # Safety
/// - Handles zero time to expiry (returns intrinsic value)
/// - Clamps volatility to prevent overflow
/// - No heap allocations, all stack-based
#[inline(always)]
pub fn bs_price(params: &BSParams, option_type: OptionType) -> BSResult {
    // Clamp inputs for numerical stability
    let time = params.time_to_expiry.max(MIN_TIME);
    let vol = params.volatility.clamp(MIN_VOL, MAX_VOL);
    
    // Precompute common terms
    let sqrt_time = fast_sqrt(time);
    let vol_sqrt_time = vol * sqrt_time;
    
    // Forward price: F = S * e^((r-q)T)
    let drift = params.risk_free_rate - params.dividend_yield;
    let forward = params.spot * fast_exp(drift * time);
    
    // Discount factor
    let discount = fast_exp(-params.risk_free_rate * time);
    
    // Handle ATM-forward case specially for better precision
    let ln_moneyness = if (forward - params.strike).abs() < EPSILON * forward {
        // ATM: use Taylor expansion for ln(F/K) ≈ (F-K)/K
        (forward - params.strike) / params.strike
    } else {
        fast_ln(forward / params.strike)
    };
    
    // d1 = (ln(F/K) + 0.5*σ²T) / (σ√T)
    // d2 = d1 - σ√T
    let d1 = (ln_moneyness + 0.5 * vol * vol * time) / vol_sqrt_time;
    let d2 = d1 - vol_sqrt_time;
    
    // Calculate price based on option type
    let price = match option_type {
        OptionType::Call => {
            // C = e^(-rT) * [F * Φ(d1) - K * Φ(d2)]
            let nd1 = fast_cdf(d1);
            let nd2 = fast_cdf(d2);
            discount * (forward * nd1 - params.strike * nd2)
        }
        OptionType::Put => {
            // P = e^(-rT) * [K * Φ(-d2) - F * Φ(-d1)]
            let nmd1 = fast_cdf_complement(d1);
            let nmd2 = fast_cdf_complement(d2);
            discount * (params.strike * nmd2 - forward * nmd1)
        }
        OptionType::DigitalCall => {
            // Digital Call: e^(-rT) * Φ(d2)
            discount * fast_cdf(d2)
        }
        OptionType::DigitalPut => {
            // Digital Put: e^(-rT) * Φ(-d2)
            discount * fast_cdf_complement(d2)
        }
    };
    
    // Ensure non-negative price (arbitrage bound)
    let price = price.max(0.0);
    
    BSResult::new(price, d1, d2, forward, discount)
}

/// Price a straddle (call + put at same strike)
#[inline(always)]
pub fn bs_straddle(params: &BSParams) -> f64 {
    let call_result = bs_price(params, OptionType::Call);
    let put_result = bs_price(params, OptionType::Put);
    call_result.price + put_result.price
}

/// Price a strangle (OTM call + OTM put)
#[inline(always)]
pub fn bs_strangle(params: &BSParams, otm_call_strike: f64, otm_put_strike: f64) -> f64 {
    let mut call_params = *params;
    call_params.strike = otm_call_strike;
    let call_price = bs_price(&call_params, OptionType::Call).price;
    
    let mut put_params = *params;
    put_params.strike = otm_put_strike;
    let put_price = bs_price(&put_params, OptionType::Put).price;
    
    call_price + put_price
}

/// Calculate intrinsic value (no time value)
#[inline(always)]
pub fn intrinsic_value(spot: f64, strike: f64, option_type: OptionType) -> f64 {
    match option_type {
        OptionType::Call => (spot - strike).max(0.0),
        OptionType::Put => (strike - spot).max(0.0),
        OptionType::DigitalCall => if spot > strike { 1.0 } else { 0.0 },
        OptionType::DigitalPut => if spot < strike { 1.0 } else { 0.0 },
    }
}

/// Calculate time value = option price - intrinsic value
#[inline(always)]
pub fn time_value(params: &BSParams, option_type: OptionType) -> f64 {
    let price = bs_price(params, option_type).price;
    let intrinsic = intrinsic_value(params.spot, params.strike, option_type);
    (price - intrinsic).max(0.0)
}

/// Batch price multiple options (SIMD-ready structure)
/// 
/// # Arguments
/// * `params` - Slice of option parameters
/// * `option_types` - Slice of option types (must match params length)
/// * `output` - Pre-allocated output buffer
/// 
/// # Returns
/// * Number of options priced
/// 
/// # Safety
/// - Zero allocations
/// - Bounds checked
pub fn bs_batch_price(params: &[BSParams], option_types: &[OptionType], output: &mut [f64]) -> usize {
    let len = params.len().min(option_types.len()).min(output.len());
    
    for i in 0..len {
        output[i] = bs_price(&params[i], option_types[i]).price;
    }
    
    len
}

/// Verify put-call parity: C - P = S*e^(-qT) - K*e^(-rT)
/// Returns the parity error (should be ~0 for arbitrage-free prices)
#[inline(always)]
pub fn verify_put_call_parity(params: &BSParams) -> f64 {
    let call_price = bs_price(params, OptionType::Call).price;
    let put_price = bs_price(params, OptionType::Put).price;
    
    // Left side: C - P
    let left = call_price - put_price;
    
    // Right side: S*e^(-qT) - K*e^(-rT)
    let disc_div = fast_exp(-params.dividend_yield * params.time_to_expiry);
    let disc_rf = fast_exp(-params.risk_free_rate * params.time_to_expiry);
    let right = params.spot * disc_div - params.strike * disc_rf;
    
    (left - right).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bs_call_price() {
        let params = BSParams {
            spot: 100.0,
            strike: 100.0,
            time_to_expiry: 0.25, // 3 months
            risk_free_rate: 0.05,
            volatility: 0.2,
            dividend_yield: 0.0,
        };
        
        let result = bs_price(&params, OptionType::Call);
        
        // Expected ~3.97 for ATM call
        assert!(result.price > 3.5 && result.price < 4.5, "Call price out of range: {}", result.price);
        assert!(result.d1 > 0.0 && result.d1 < 1.0, "d1 out of range: {}", result.d1);
    }
    
    #[test]
    fn test_bs_put_price() {
        let params = BSParams {
            spot: 100.0,
            strike: 100.0,
            time_to_expiry: 0.25,
            risk_free_rate: 0.05,
            volatility: 0.2,
            dividend_yield: 0.0,
        };
        
        let result = bs_price(&params, OptionType::Put);
        
        // Expected ~3.48 for ATM put
        assert!(result.price > 3.0 && result.price < 4.0, "Put price out of range: {}", result.price);
    }
    
    #[test]
    fn test_put_call_parity() {
        let params = BSParams {
            spot: 100.0,
            strike: 100.0,
            time_to_expiry: 0.5,
            risk_free_rate: 0.03,
            volatility: 0.25,
            dividend_yield: 0.01,
        };
        
        let parity_error = verify_put_call_parity(&params);
        assert!(parity_error < 1e-6, "Put-call parity violation: {}", parity_error);
    }
    
    #[test]
    fn test_zero_time_expiry() {
        let params = BSParams {
            spot: 105.0,
            strike: 100.0,
            time_to_expiry: 0.0,
            risk_free_rate: 0.05,
            volatility: 0.2,
            dividend_yield: 0.0,
        };
        
        let call = bs_price(&params, OptionType::Call);
        let put = bs_price(&params, OptionType::Put);
        
        // At expiry, call should be intrinsic (5.0), put should be 0
        assert!((call.price - 5.0).abs() < 0.01, "ITM call at expiry: {}", call.price);
        assert!(put.price < 0.01, "OTM put at expiry: {}", put.price);
    }
    
    #[test]
    fn test_deep_otm_no_nan() {
        let params = BSParams {
            spot: 100.0,
            strike: 500.0, // Deep OTM call
            time_to_expiry: 0.1,
            risk_free_rate: 0.05,
            volatility: 0.8,
            dividend_yield: 0.0,
        };
        
        let result = bs_price(&params, OptionType::Call);
        assert!(result.price.is_finite(), "Deep OTM call produced NaN");
        assert!(result.price >= 0.0, "Negative option price");
    }
    
    #[test]
    fn test_batch_pricing() {
        let params = vec![
            BSParams { spot: 100.0, strike: 100.0, time_to_expiry: 0.25, risk_free_rate: 0.05, volatility: 0.2, dividend_yield: 0.0 },
            BSParams { spot: 100.0, strike: 110.0, time_to_expiry: 0.25, risk_free_rate: 0.05, volatility: 0.2, dividend_yield: 0.0 },
            BSParams { spot: 100.0, strike: 90.0, time_to_expiry: 0.25, risk_free_rate: 0.05, volatility: 0.2, dividend_yield: 0.0 },
        ];
        let option_types = vec![OptionType::Call, OptionType::Call, OptionType::Put];
        let mut output = [0.0; 10];
        
        let count = bs_batch_price(&params, &option_types, &mut output);
        
        assert_eq!(count, 3);
        assert!(output[0] > 0.0);
        assert!(output[1] < output[0]); // OTM call cheaper
        assert!(output[2] > 0.0);
    }
}
