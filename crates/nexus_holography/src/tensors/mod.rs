//! Chapter 3: MERA Tensor Renormalization & Multi-Scale Holography
//! 
//! Implements Multi-scale Entanglement Renormalization Ansatz (MERA)
//! for coarse-graining HFT noise and extracting macro institutional signals.

pub mod mera_renormalization;
pub mod disentangler_isometry;
pub mod macro_signal_projector;

pub use mera_renormalization::*;
pub use disentangler_isometry::*;
pub use macro_signal_projector::*;
