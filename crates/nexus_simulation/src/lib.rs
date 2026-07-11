//! Nexus Simulation - Stage 42 of 50
//! 
//! Simulation Theory Validation, Base-Reality Arbitrage & Glitch Exploitation
//! 
//! Financial exchanges are discrete computational simulations running on finite silicon.
//! This crate implements zero-allocation Rust solvers to reverse-engineer exchange
//! matching engine logic, crack PRNGs, exploit LOD artifacts, and execute base-reality
//! arbitrage by trading hardware/software flaws of the exchange itself.
//! 
//! # Chapters
//! 
//! - **Chapter 1**: Discrete Spacetime & Matching Engine Automata
//! - **Chapter 2**: Level-of-Detail (LOD) & Volatility Surface Rendering Limits  
//! - **Chapter 3**: PRNG State Extraction & Queue Tie-Breaking Prediction
//! - **Chapter 4**: Ontological Glitches & Base-Reality Arbitrage
//! 
//! # Safety Notes
//! 
//! Several modules contain "Academic Sandbox" flags that default to safe/disabled
//! modes in production to comply with exchange Terms of Service regarding
//! disruptive messaging. Always review configuration before deployment.

#![no_std]

#[cfg(test)]
extern crate alloc;

pub mod spacetime;
pub mod rendering;
pub mod prng;
pub mod glitches;

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Crate name
pub const NAME: &str = "nexus_simulation";
