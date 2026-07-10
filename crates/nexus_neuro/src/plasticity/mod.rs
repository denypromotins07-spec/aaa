//! Plasticity module for STDP and synaptic learning
pub mod stdp_learning_rule;
pub mod eligibility_traces;
pub mod atomic_weight_updater;

pub use stdp_learning_rule::*;
pub use eligibility_traces::*;
pub use atomic_weight_updater::*;
