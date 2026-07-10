//! NEXUS-OMEGA Core Library
//!
//! This crate provides the core functionality for the NEXUS-OMEGA
//! high-frequency trading system, including:
//!
//! - Zero-allocation memory arenas
//! - Lock-free concurrent data structures
//! - High-performance event bus
//! - Microsecond-precision timing

pub mod memory;
pub mod concurrency;
pub mod events;
pub mod time;

// Re-export commonly used types
pub use memory::arena::{BumpAllocator, CachePadded64, ArenaError, CACHE_LINE_SIZE};
pub use memory::cache_padder::{CachePadded64 as CachePadded, CachePadding};
pub use concurrency::spsc_ring::{SPSCRingBuffer, RingBufferError};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
