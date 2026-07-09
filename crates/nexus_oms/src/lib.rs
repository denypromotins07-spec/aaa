//! Nexus OMS Library - Lock-free Order Management System

pub mod fixed_point_math;
pub mod order_state_machine;
pub mod lock_free_position_tracker;

pub use fixed_point_math::{FixedPoint, FixedPointError, SCALE};
pub use order_state_machine::{
    OrderStateMachine, Order, OrderId, Side, OrderType, OrderStatus,
    ExecutionReport, TaggedPointer,
};
pub use lock_free_position_tracker::{
    LockFreePositionTracker, Position,
};
