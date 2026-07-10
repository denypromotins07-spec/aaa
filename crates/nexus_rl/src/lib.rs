//! NEXUS RL Module - High-Performance Reinforcement Learning
//! 
//! This crate provides zero-copy RL environment interfaces, reward shaping,
//! and lock-free experience replay for distributed training.

pub mod env;
pub mod actions;
pub mod rewards;
pub mod replay;

// Re-export main types
pub use env::zero_copy_env::{ZeroCopyEnv, StateObservation};
pub use env::shared_memory_mapper::{SharedMemoryMap, SHARED_MEMORY_SIZE};
pub use actions::hybrid_action_space::{HybridAction, ActionSpaceConfig, TradeSide, OrderType};
pub use rewards::differential_sharpe::{DifferentialSharpeRatio, RewardShaper};
pub use replay::lock_free_per::{LockFreeReplayBuffer, Experience};
pub use replay::gae_calculator::{GAECalculator, TrajectoryStep};
