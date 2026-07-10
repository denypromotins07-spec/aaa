//! Surface module for volatility modeling
//! 
//! Contains volatility surface construction, SVI parameterization,
//! and arbitrage enforcement.

pub mod vol_surface_builder;
pub mod svi_parameterization;
pub mod arbitrage_enforcer;

pub use vol_surface_builder::{
    VolPoint,
    ExpiryBucket,
    VolatilitySurface,
};

pub use svi_parameterization::{
    SVIParams,
    calibrate_svi,
    apply_svi_smoothing,
    check_svi_no_arbitrage,
};

pub use arbitrage_enforcer::{
    ArbitrageReport,
    check_all_arbitrage,
    enforce_no_arbitrage,
    compute_risk_neutral_density,
};
