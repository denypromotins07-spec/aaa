//! NEXUS-OMEGA Stage 18: Advanced Time-Series Mathematics
//! 
//! This crate provides advanced time-series analysis tools including:
//! - Fractional Differentiation (FracDiff) for memory preservation
//! - Spectral Analysis (FFT, Wavelets) for signal decomposition  
//! - Non-linear Trend Filtering (Hamilton, Whittaker-Henderson)
//! - Information Theory metrics (Entropy, Transfer Entropy)

#![no_std]
#![allow(dead_code)]
#![allow(unused_variables)]

extern crate alloc;

pub mod fracdiff {
    pub mod fixed_window_filter;
    pub mod simd_weight_convolution;
    pub mod streaming_adf_test;
    
    pub use fixed_window_filter::*;
    pub use simd_weight_convolution::*;
    pub use streaming_adf_test::*;
}

pub mod spectral {
    pub mod zero_alloc_fft;
    pub mod wavelet_decomposition;
    pub mod bandpass_alpha_filter;
    
    pub use zero_alloc_fft::*;
    pub use wavelet_decomposition::*;
    pub use bandpass_alpha_filter::*;
}

pub mod trends {
    pub mod hamilton_filter;
    pub mod whittaker_smoother;
    pub mod sparse_banded_solver;
    
    pub use hamilton_filter::*;
    pub use whittaker_smoother::*;
    pub use sparse_banded_solver::*;
}

pub mod info_theory {
    pub mod shannon_entropy;
    pub mod transfer_entropy_ksg;
    pub mod kd_tree_arena;
    
    pub use shannon_entropy::*;
    pub use transfer_entropy_ksg::*;
    pub use kd_tree_arena::*;
}

/// Re-export all modules for convenience
pub mod prelude {
    pub use crate::fracdiff::{FixedWindowFracDiff, ExpandingWindowFracDiff, FracDiffWeights, StreamingAdfTest};
    pub use crate::spectral::{ZeroAllocFft, WaveletFilters, ZeroAllocDwt, BandpassAlphaFilter};
    pub use crate::trends::{HamiltonFilter, HamiltonFilterParams, WhittakerSmoother, WhittakerParams};
    pub use crate::info_theory::{ShannonEntropy, TransferEntropyKsg, KdTreeArena};
}
