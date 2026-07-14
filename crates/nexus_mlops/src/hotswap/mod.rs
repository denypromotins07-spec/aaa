//! Hot-swap module for atomic model weight updates

pub mod atomic_weight_swapper;
pub mod rcu_epoch_reclamation;

pub use atomic_weight_swapper::{AtomicWeightSwapper, ModelWeights, ModelMetadata, SwapResult};
pub use rcu_epoch_reclamation::{EpochReclaimer, EpochGuard, EpochStats, HazardPointer};
