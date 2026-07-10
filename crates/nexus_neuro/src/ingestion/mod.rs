//! Ingestion module for AER event-camera data and spike rasterization
pub mod aer_zero_copy_parser;
pub mod spike_rasterizer;
pub mod temporal_binning;

pub use aer_zero_copy_parser::*;
pub use spike_rasterizer::*;
pub use temporal_binning::*;
