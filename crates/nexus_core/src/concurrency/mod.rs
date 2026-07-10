//! Concurrency primitives for NEXUS-OMEGA
//!
//! Provides lock-free data structures optimized for high-frequency trading.

pub mod spsc_ring;

pub use spsc_ring::{SPSCRingBuffer, RingBufferError};
