//! Nexus Execution Library - Algorithmic Execution State Machines

pub mod algos;

pub use algos::iceberg_state::{IcebergState, IcebergConfig, LiquiditySnapshot};
pub use algos::pov_vwap_tracker::{
    PovTracker, PovConfig, PovPace, VwapTracker, VwapConfig, EwmaCalculator,
};
pub use algos::child_order_generator::{
    ChildOrderGenerator, ChildOrderRequest, AlgoType,
};
