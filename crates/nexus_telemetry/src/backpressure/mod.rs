//! Backpressure Module - Drop Policies and Slow Client Handling
//!
//! This module provides backpressure handling mechanisms:
//! - Bounded MPSC queue with drop policies
//! - Slow client guillotine for connection termination

pub mod bounded_mpsc_queue;
pub mod slow_client_guillotine;
pub mod drop_policy;

pub use bounded_mpsc_queue::{
    BoundedMpscQueue, QueueProducer, QueueConsumer, BoundedQueueConfig,
    DropPolicy as QueueDropPolicy, split_queue,
};
pub use slow_client_guillotine::{
    SlowClientGuillotine, GuillotineConfig, ClientStats, DisconnectReason,
};
pub use drop_policy::DropPolicy;
