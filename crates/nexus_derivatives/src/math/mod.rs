//! Math module for high-performance derivatives pricing
//! 
//! Contains fast approximations for transcendental functions
//! used throughout the pricing engine.

pub mod fast_cdf_erf;

pub use fast_cdf_erf::{
    fast_erf,
    fast_erfc,
    fast_cdf,
    fast_cdf_complement,
    fast_exp,
    fast_ln,
    fast_sqrt,
    fast_inv_sqrt,
};
