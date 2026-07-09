//! Chapter 2: L2/L3 Order Book Reconstruction Engine
//! 
//! This module provides zero-allocation limit order book reconstruction
//! from exchange delta events. It uses custom cache-aligned data structures
//! instead of standard HashMap/BTreeMap to guarantee O(1) lookups and
//! zero heap allocations during hot-path updates.

pub mod price_ladder;
pub mod l3_slab_allocator;
pub mod lob_reconstructor;

pub use price_ladder::{PriceLadder, PriceLevel, Side};
pub use l3_slab_allocator::{L3SlabAllocator, OrderRecord};
pub use lob_reconstructor::{LobReconstructor, OrderBookDelta, DeltaType};
