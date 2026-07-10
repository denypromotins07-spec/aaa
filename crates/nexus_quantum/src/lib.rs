//! NEXUS-OMEGA Stage 21: Quantum Computing Infrastructure, QAOA Portfolio Optimization & Annealing
//!
//! This crate implements quantum-inspired and quantum-ready portfolio optimization:
//! - Chapter 1: QUBO Formulation & Ising Hamiltonian Translation
//! - Chapter 2: QAOA Ansatz & Barren Plateau Mitigation (Python bindings)
//! - Chapter 3: Quantum Annealing & Minor-Embedding Heuristics
//! - Chapter 4: Asynchronous Quantum Oracle & Classical Fallback

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![forbid(unsafe_code)]

pub mod qubo;
pub mod annealing;
pub mod bridge;

/// Re-export commonly used types for Stage 21
pub use qubo::portfolio_hamiltonian::{PortfolioQuboBuilder, QuboMatrix, QuboConfig};
pub use qubo::adaptive_penalty_scaler::{AdaptivePenaltyScaler, PenaltyScalingResult};
pub use qubo::ising_mapper::{IsingHamiltonian, IsingMapper, IsingCoefficients};
pub use annealing::dwave_hybrid_bridge::{DWaveHybridBridge, DWaveConfig, DWaveSample};
pub use annealing::minor_embedding_heuristic::{MinorEmbedder, EmbeddingGraph, PegasusTopology};
pub use annealing::chain_strength_calculator::{ChainStrengthCalculator, ChainStrengthConfig};
pub use bridge::async_quantum_oracle::{AsyncQuantumOracle, OracleConfig, OracleResponse};
pub use bridge::classical_simulated_annealing::{ClassicalSimulatedAnnealer, AnnealingConfig, AnnealingState};
pub use bridge::energy_gap_validator::{EnergyGapValidator, GapValidationResult};
