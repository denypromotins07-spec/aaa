//! Events module for NEXUS-OMEGA
//!
//! Provides event bus and message passing infrastructure.

pub mod message_bus;

pub use message_bus::{MessageBus, MessageBusError, EventHeader};
