//! NEXUS-OMEGA Stage 38: Consciousness Economics, Qualia Valuation & Subjective Experience Derivatives
//!
//! This crate implements zero-allocation Rust solvers for:
//! - High-density EEG/fNIRS neural telemetry processing
//! - Integrated Information Theory (Φ) approximations for consciousness density
//! - Neuro-economic utility functions and stochastic hedonic treadmill models
//! - Attention Yield Curves and cognitive load arbitrage

#![no_std]
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![warn(missing_docs)]
#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "std")]
extern crate std;

pub mod bci;
pub mod consciousness;
pub mod economics;
pub mod derivatives;
pub mod alpha;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Maximum supported channels for BCI processing
pub const MAX_BCI_CHANNELS: usize = 256;

/// Maximum supported nodes for consciousness network analysis
pub const MAX_CONSCIOUSNESS_NODES: usize = 1024;
