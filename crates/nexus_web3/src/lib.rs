//! NEXUS-OMEGA Stage 14: Web3 MEV, Jito Bundles & Cross-Chain Arbitrage
//! 
//! This crate provides zero-allocation blockchain interaction primitives:
//! - Solana transaction building with Jito bundle orchestration
//! - EVM mempool sniping with RLP decoding
//! - Uniswap V3 concentrated liquidity simulation with U256 math
//! - Cross-chain atomic arbitrage with HTLCs

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![no_std]

extern crate alloc;

pub mod solana;
pub mod evm;
pub mod mev;
pub mod dex;
pub mod cross_chain;

/// Re-export U256 for external use
pub use uint::U256;
