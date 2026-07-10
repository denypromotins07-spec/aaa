//! NEXUS-OMEGA Stage 6: High-Performance Derivatives Pricing Engine
//! 
//! This crate provides zero-allocation, nanosecond-level pricing for:
//! - European options (Black-Scholes-Merton with fast math)
//! - Stochastic volatility models (SABR, Heston)
//! - Volatility surface construction and arbitrage enforcement
//! - Real-time Greeks calculation with SIMD aggregation
//! - Volatility arbitrage strategies (VRP, dispersion, term structure)
//! 
//! # Features
//! 
//! - **Zero allocations in hot paths**: All pricing uses stack-allocated buffers
//! - **Custom polynomial approximations**: Fast erf, CDF, exp without std::f64 calls
//! - **Arbitrage-free surfaces**: Calendar spread and butterfly arbitrage detection/correction
//! - **SIMD-ready**: Portfolio Greeks aggregation optimized for vector instructions
//! 
//! # Example
//! 
//! ```rust
//! use nexus_derivatives::pricing::{BSParams, OptionType, bs_price};
//! use nexus_derivatives::greeks::calculate_greeks;
//! 
//! let params = BSParams {
//!     spot: 100.0,
//!     strike: 100.0,
//!     time_to_expiry: 0.25,
//!     risk_free_rate: 0.05,
//!     volatility: 0.2,
//!     dividend_yield: 0.0,
//! };
//! 
//! let result = bs_price(&params, OptionType::Call);
//! let greeks = calculate_greeks(&params, OptionType::Call);
//! ```

#![no_std]
#![cfg_attr(target_feature = "simd", feature(portable_simd))]
#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

extern crate alloc;

pub mod math;
pub mod pricing;
pub mod surface;
pub mod greeks;
pub mod strategies;

// Re-export commonly used items at crate root
pub use math::{fast_erf, fast_cdf, fast_exp, fast_sqrt};
pub use pricing::{BSParams, OptionType, bs_price, SABRParams, HestonParams};
pub use surface::{VolatilitySurface, VolPoint, SVIParams};
pub use greeks::{FullGreeks, FirstOrderGreeks, SecondOrderGreeks};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Numerical constants
pub mod constants {
    /// Euler-Mascheroni constant
    pub const GAMMA_EULER: f64 = 0.5772156649015328606065120900824024310421;
    
    /// sqrt(2*pi)
    pub const SQRT_2_PI: f64 = 2.5066282746310005024157652848110452530069;
    
    /// 1/sqrt(2*pi)
    pub const INV_SQRT_2_PI: f64 = 0.3989422804014326779399460599343818684758;
    
    /// sqrt(2)
    pub const SQRT_2: f64 = 1.4142135623730950488016887242096980785696;
    
    /// ln(2)
    pub const LN_2: f64 = 0.6931471805599453094172321214581765680755;
    
    /// Trading days per year
    pub const TRADING_DAYS: f64 = 252.0;
    
    /// Calendar days per year
    pub const CALENDAR_DAYS: f64 = 365.0;
}

/// Error types for derivatives calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivativesError {
    /// Invalid parameter (negative time, negative vol, etc.)
    InvalidParameter,
    /// Numerical overflow/underflow
    NumericalOverflow,
    /// Arbitrage violation detected
    ArbitrageViolation,
    /// Calibration failed
    CalibrationFailed,
    /// Buffer full (zero-allocation mode)
    BufferFull,
}

impl core::fmt::Display for DerivativesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidParameter => write!(f, "Invalid parameter"),
            Self::NumericalOverflow => write!(f, "Numerical overflow"),
            Self::ArbitrageViolation => write!(f, "Arbitrage violation detected"),
            Self::CalibrationFailed => write!(f, "Calibration failed"),
            Self::BufferFull => write!(f, "Buffer full"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::{BSParams, OptionType, bs_price};
    
    #[test]
    fn test_basic_pricing() {
        let params = BSParams {
            spot: 100.0,
            strike: 100.0,
            time_to_expiry: 0.25,
            risk_free_rate: 0.05,
            volatility: 0.2,
            dividend_yield: 0.0,
        };
        
        let call = bs_price(&params, OptionType::Call);
        let put = bs_price(&params, OptionType::Put);
        
        assert!(call.price > 0.0);
        assert!(put.price > 0.0);
        assert!(call.price.is_finite());
        assert!(put.price.is_finite());
    }
    
    #[test]
    fn test_no_nan_extreme_inputs() {
        let extreme_params = [
            BSParams { spot: 1000.0, strike: 10.0, time_to_expiry: 0.001, risk_free_rate: 0.5, volatility: 2.0, dividend_yield: 0.0 },
            BSParams { spot: 10.0, strike: 1000.0, time_to_expiry: 5.0, risk_free_rate: 0.01, volatility: 0.01, dividend_yield: 0.0 },
        ];
        
        for params in &extreme_params {
            let call = bs_price(params, OptionType::Call);
            let put = bs_price(params, OptionType::Put);
            
            assert!(call.price.is_finite(), "Call price NaN for {:?}", params);
            assert!(put.price.is_finite(), "Put price NaN for {:?}", params);
            assert!(call.price >= 0.0, "Negative call price");
            assert!(put.price >= 0.0, "Negative put price");
        }
    }
}
