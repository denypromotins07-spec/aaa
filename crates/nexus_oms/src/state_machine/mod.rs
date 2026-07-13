//! OMS State Machine Module
//! 
//! Tracks order lifecycle states with atomic transitions.

pub mod order_state;

pub use order_state::{
    OrderId,
    ExecutionId,
    Side,
    OrderType,
    TimeInForce,
    OrderState,
    Order,
    TransitionResult,
    OrderStateMachine,
    OmsStats,
};
