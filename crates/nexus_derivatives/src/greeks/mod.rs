//! Greeks module for risk sensitivities
//! 
//! Contains analytical Greeks, SIMD aggregation, and finite difference methods.

pub mod analytical_greeks;
pub mod simd_portfolio_aggregator;
pub mod finite_difference_bump;

pub use analytical_greeks::{
    FirstOrderGreeks,
    SecondOrderGreeks,
    FullGreeks,
    calculate_greeks,
    calculate_delta,
    calculate_gamma,
    batch_calculate_greeks,
};

pub use simd_portfolio_aggregator::{
    PortfolioGreeksAggregator,
    PortfolioRiskSummary,
};

pub use finite_difference_bump::{
    FiniteDifferenceEngine,
    BumpConfig,
};
