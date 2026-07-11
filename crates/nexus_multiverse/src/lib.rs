//! NEXUS-OMEGA Stage 39: Interdimensional Mathematics, Multiverse Portfolio Theory & Quantum Measure Derivatives
//! 
//! This crate implements quantum measure theory, Many-Worlds Interpretation portfolio optimization,
//! Bell's Theorem market correlation testing, and decoherence-aware derivative pricing.

#![no_std]

extern crate alloc;

pub mod measure {
    pub mod hilbert_space_mps;
    pub mod everettian_branching;
    pub mod born_rule_amplitude;
}

pub mod portfolio {
    pub mod multiverse_efficient_frontier;
    pub mod feynman_path_integral;
    pub mod measure_weighted_utility;
}

pub mod entanglement {
    pub mod chsh_inequality_alpha;
    pub mod bell_state_market_correlations;
    pub mod non_local_hidden_variables;
}

pub mod derivatives {
    pub mod lindblad_decoherence_solver;
    pub mod schrodingers_cat_option;
    pub mod quantum_measure_swap;
}

// Re-export main types
pub use measure::hilbert_space_mps::{MatrixProductState, ComplexAmplitude, MpsError};
pub use measure::everettian_branching::{EverettianBranchingEngine, EverettianBranch, EverettianError};
pub use measure::born_rule_amplitude::{BornRuleCalculator, BornRuleError};

pub use portfolio::multiverse_efficient_frontier::{MultiversePortfolioOptimizer, MultiverseEfficientFrontier, MultiversePortfolioError};
pub use portfolio::feynman_path_integral::{FeynmanPathIntegralOptimizer, FeynmanPath, PathIntegralError};
pub use portfolio::measure_weighted_utility::{MeasureWeightedUtilityCalculator, BranchOutcome, MeasureWeightedUtilityError};

pub use entanglement::chsh_inequality_alpha::{ChshInequalityTester, ChshTestResult, ChshError};
pub use entanglement::bell_state_market_correlations::{BellStateMarketAnalyzer, BellStateCorrelation, BellStateError};
pub use entanglement::non_local_hidden_variables::{NonLocalAlphaRouter, NonLocalAlphaSignal, NonLocalAlphaError};

pub use derivatives::lindblad_decoherence_solver::{LindbladDecoherenceSolver, LindbladError};
pub use derivatives::schrodingers_cat_option::{SchrodingerOptionPricer, SchrodingerCatOption, SchrodingerOptionError};
pub use derivatives::quantum_measure_swap::{QuantumSwapPricer, QuantumMeasureSwap, QuantumSwapError};
