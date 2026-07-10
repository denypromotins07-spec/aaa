//! Macro regime detection module.

pub mod bayesian_hmm;
pub mod baum_welch_em;
pub mod regime_conditional_router;

pub use bayesian_hmm::*;
pub use baum_welch_em::*;
pub use regime_conditional_router::*;
