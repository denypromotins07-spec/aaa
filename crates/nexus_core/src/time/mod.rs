//! Time module for NEXUS-OMEGA
//!
//! Provides high-precision timing utilities.

pub mod tsc_clock;

pub use tsc_clock::{MonotonicNanosClock, ClockError, init_global_clock, global_now_nanos};
