//! Chapter 2: L2 Order Book Module
//!
//! This module provides sequence healing, REST snapshot fetching,
//! and atomic book swap functionality for maintaining order book integrity.

pub mod sequence_healer;
pub mod rest_snapshot_fetcher;
pub mod atomic_book_swap;

pub use sequence_healer::{
    OrderBookSequenceHealer, SequenceCheckResult, SequenceHealerConfig, SequenceHealerStats,
};
pub use rest_snapshot_fetcher::{
    OrderBookSnapshot, RestFetcherStats, RestSnapshotConfig, RestSnapshotFetcher,
};
pub use atomic_book_swap::{
    AtomicBookSwap, AtomicSwapStats, OrderBookState, PriceLevel,
};
