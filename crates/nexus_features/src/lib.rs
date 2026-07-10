//! Chapter 3: SIMD-Accelerated Feature Engineering
//! 
//! This module provides real-time feature extraction using AVX2/AVX-512
//! instructions via Rust's std::simd for processing 8-16 ticks simultaneously.

pub mod simd_rolling_windows;
pub mod micro_price_calculator;
pub mod volume_bar_aggregator;

pub use simd_rolling_windows::{SimdRollingWindow, SimdVwapCalculator};
pub use micro_price_calculator::{MicroPriceCalculator, OrderBookImbalance};
pub use volume_bar_aggregator::{VolumeBarAggregator, TimeBarAggregator, BarType};
