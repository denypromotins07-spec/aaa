//! Execution Algorithms Module
//! 
//! Iceberg Sniper and Queue Position Tracker for smart order execution.

pub mod iceberg_sniper;
pub mod queue_position_tracker;

pub use iceberg_sniper::{
    IcebergSniper,
    IcebergConfig,
    IcebergState,
    IcebergSlice,
    SliceState,
    OrderSide,
    IcebergStats,
};

pub use queue_position_tracker::{
    QueuePositionTracker,
    QueueTrackerConfig,
    QueuePriority,
    QueueAction,
    IcebergDetection,
    OrderBookLevel,
    QueueTrackerStats,
};
