//! Pricing module for derivatives
//! 
//! Contains Black-Scholes, SABR, and Heston pricing engines.

pub mod black_scholes_fast;
pub mod sabr_hybrid;

pub use black_scholes_fast::{
    BSParams,
    BSResult,
    OptionType,
    bs_price,
    bs_straddle,
    bs_strangle,
    intrinsic_value,
    time_value,
    bs_batch_price,
    verify_put_call_parity,
};

pub use sabr_hybrid::{
    SABRParams,
    HestonParams,
    sabr_implied_vol,
    sabr_price,
    heston_char_func,
    heston_price_lewis,
    heston_price_approx,
    heston_to_sabr,
};
