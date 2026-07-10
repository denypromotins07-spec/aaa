//! Strategies module for volatility arbitrage
//! 
//! Contains VRP harvesting, dispersion trading, and term structure arb.

pub mod vol_risk_premium;
pub mod dispersion_trader;
pub mod term_structure_arb;

pub use vol_risk_premium::{
    VolatilityRiskPremium,
    VrpSignal,
};

pub use dispersion_trader::{
    DispersionTrader,
    DispersionSignal,
    IndexVolBasket,
};

pub use term_structure_arb::{
    TermStructureArb,
    TermStructureSignal,
    ContangoBackwardation,
};
