//! Bandpass Alpha Filter - Module exports
//! Re-exports spectral analysis components for alpha extraction.

pub use super::zero_alloc_fft::{Complex, TwiddleFactors, ZeroAllocFft, SimdFft};
pub use super::wavelet_decomposition::{WaveletFilters, ZeroAllocDwt, BandpassAlphaFilter};

/// Combined spectral analysis engine
pub struct SpectralAlphaEngine {
    fft_size: usize,
    wavelet_levels: usize,
}

impl SpectralAlphaEngine {
    pub fn new(fft_size: usize, wavelet_levels: usize) -> Self {
        Self {
            fft_size,
            wavelet_levels,
        }
    }
}
