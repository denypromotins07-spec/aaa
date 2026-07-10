//! Memory management module for NEXUS-OMEGA
//!
//! Provides zero-allocation memory arenas and cache-line padding utilities.

pub mod arena;
pub mod cache_padder;

pub use arena::{BumpAllocator, ArenaError, CACHE_LINE_SIZE};
pub use cache_padder::{CachePadded64, CachePadded128, CachePadding};
