//! Integration tests for Spectral Analysis module

use nexus_timeseries::prelude::*;

#[test]
fn test_fft_round_trip() {
    let n = 128;
    let mut fft = ZeroAllocFft::new(n).unwrap();
    
    // Generate test signal
    let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
    
    // Forward FFT
    let freq = fft.fft(&input).unwrap().to_vec();
    
    // Inverse FFT
    let reconstructed = fft.ifft(&freq).unwrap();
    
    // Check reconstruction accuracy
    for i in 0..n {
        assert!((input[i] - reconstructed[i]).abs() < 1e-9);
    }
}

#[test]
fn test_wavelet_decomposition_reconstruction() {
    let filters = WaveletFilters::daubechies(4).unwrap();
    let mut dwt = ZeroAllocDwt::new(filters, 64, 3);
    
    let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.2).sin()).collect();
    
    let coeffs = dwt.decompose(&signal, 2).unwrap();
    let reconstructed = dwt.reconstruct(&coeffs).unwrap();
    
    assert_eq!(reconstructed.len(), signal.len());
}

#[test]
fn test_bandpass_alpha_extraction() {
    let n = 256;
    let mut filter = BandpassAlphaFilter::new(n, 2, 4).unwrap();
    
    // Multi-frequency signal
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            (i as f64 * 0.05).sin() +      // Low freq
            (i as f64 * 0.3).sin() * 0.5 +  // Mid freq (target)
            (i as f64 * 2.0).sin() * 0.2    // High freq
        })
        .collect();
    
    let alpha = filter.extract_alpha(&signal).unwrap();
    assert_eq!(alpha.len(), n);
    
    let snr = filter.snr(&signal).unwrap();
    assert!(snr.is_finite());
}
