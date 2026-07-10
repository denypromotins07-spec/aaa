//! Nexus StatArb - High-Frequency Statistical Arbitrage Engine
//! 
//! Stage 7 of 50: Dynamic Cointegration, OU Spread Modeling & Atomic Execution

pub mod cointegration;
pub mod spread;
pub mod universe;
pub mod factors;
pub mod math;
pub mod execution;

pub use cointegration::*;
pub use spread::*;
pub use universe::*;
pub use factors::*;
pub use math::*;
pub use execution::*;
