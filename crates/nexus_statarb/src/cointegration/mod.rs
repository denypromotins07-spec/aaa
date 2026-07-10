//! Cointegration module - Dynamic hedge ratio estimation

mod kalman_hedge_ratio;
mod rls_sherman_morrison;

pub use kalman_hedge_ratio::*;
pub use rls_sherman_morrison::*;
