//! Chapter 1: Digital Von Neumann Probes & Cryptographic Quines
//!
//! Implements self-replicating smart contracts and cross-chain atomic spawning.

pub mod evm_quine_generator;
pub mod self_referential_bytecode;
pub mod cross_chain_atomic_spawner;

pub use evm_quine_generator::EvmQuineGenerator;
pub use self_referential_bytecode::SelfReferentialBytecode;
pub use cross_chain_atomic_spawner::CrossChainAtomicSpawner;
