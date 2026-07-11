//! NEXUS-OMEGA Stage 44: Acausal Trade, Superintelligence Game Theory & Basilisk Mitigation
//! 
//! This crate implements advanced decision theory frameworks for trading in environments
//! saturated with autonomous AI agents. It covers:
//! 
//! - Timeless Decision Theory (TDT) for acausal cooperation
//! - Resource-bounded Solomonoff Induction via Levin Search
//! - Counterfactual Mugging resolution
//! - Basilisk Defense through future regret minimization
//! 
//! # Safety Guarantees
//! 
//! - No infinite recursion in mutual simulation (strict depth limits)
//! - No halting problem issues (instruction counting, WASM sandboxing)
//! - No logical paradoxes (weaker PA fragments for proofs)
//! - No unwrap()/expect() in hot paths

#![no_std]
#![warn(missing_docs)]
#![warn(unused_qualifications)]
#![forbid(clippy::unwrap_used)]
#![forbid(clippy::expect_used)]

extern crate alloc;

pub mod tdt {
    //! Timeless Decision Theory modules
    
    pub mod logical_handshake;
    pub mod source_code_mirror;
    pub mod modal_logic_prover;
}

pub mod induction {
    //! Solomonoff Induction and Universal Prior approximation
    
    pub mod levin_search;
    pub mod kolmogorov_complexity;
    pub mod universal_prior_approx;
}

pub mod trade {
    //! Acausal trade mechanisms
    
    pub mod acausal_payment;
    pub mod counterfactual_mugging;
    pub mod multiverse_utility;
}

pub mod basilisk {
    //! Basilisk Defense and Future-Regret Minimization
    
    pub mod future_regret_penalty;
    pub mod singularity_audit_sim;
    pub mod preemptive_alignment;
}

// Re-export main types for convenience
pub use tdt::logical_handshake::{LogicalHandshake, HandshakeConfig, HandshakeResult};
pub use tdt::source_code_mirror::{SourceCodeMirror, MirrorDepth, DecisionPattern};
pub use tdt::modal_logic_prover::{ModalProver, ProofResult, ModalFormula};

pub use induction::levin_search::{LevinSearch, LevinSearchResult};
pub use induction::kolmogorov_complexity::{KolmogorovComplexity, KolmogorovEstimate};
pub use induction::universal_prior_approx::{UniversalPriorApproximator, PriorHypothesis};

pub use trade::acausal_payment::{AcausalPayment, AcausalPaymentConfig, AcausalPaymentResult};
pub use trade::counterfactual_mugging::{CounterfactualMugging, CounterfactualConfig, CounterfactualResolution};
pub use trade::multiverse_utility::{MultiverseUtility, MultiverseUtilityResult, MultiverseAction};

pub use basilisk::future_regret_penalty::{FutureRegretCalculator, RegretConfig, ActionCategory, RegretRecommendation};
pub use basilisk::singularity_audit_sim::{SingularityAuditSimulator, ActionRecord, AuditSimulationResult};
pub use basilisk::preemptive_alignment::{PreemptiveAlignment, AlignmentConfig, AlignmentAnalysis};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Stage number
pub const STAGE: u32 = 44;
