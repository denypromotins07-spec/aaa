//! Chapter 3: Global Kill Switch & Portfolio Flatten FSM

pub mod global_atomic_halt;
pub mod flatten_fsm;
pub mod twap_liquidation_router;

// Re-export key types
pub use global_atomic_halt::GlobalKillSwitch;
pub use flatten_fsm::{FlattenFSM, FlattenState};
pub use twap_liquidation_router::TWAPLiquidationRouter;
