//! Nexus Portfolio - Live Portfolio Margin, Funding Rate Harvesting & Cross-Exchange Arbitrage
//! 
//! This crate provides:
//! - Real-time cross-margin tracking with fixed-point arithmetic
//! - Perpetual funding rate harvesting and basis arbitrage
//! - Cross-exchange statistical arbitrage with atomic legging resolution

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![forbid(unsafe_code)]

pub mod margin;
pub mod yield;
pub mod arb;

pub use margin::*;
pub use yield::*;
pub use arb::*;
