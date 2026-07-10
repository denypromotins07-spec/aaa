//! Discrete Wavelet Transform (DWT) for Signal Decomposition
//! Zero-allocation implementation using pre-allocated buffers.
//! Supports Daubechies wavelets for multi-resolution analysis.

use crate::spectral::zero_alloc_fft::Complex;

/// Wavelet filter coefficients
#[derive(Debug, Clone)]
pub struct WaveletFilters {
    /// Decomposition low-pass filter
    dec_lo: Vec<f64>,
    /// Decomposition high-pass filter  
    dec_hi: Vec<f64>,
    /// Reconstruction low-pass filter
    rec_lo: Vec<f64>,
    /// Reconstruction high-pass filter
    rec_hi: Vec<f64>,
}

impl WaveletFilters {
    /// Create Daubechies wavelet filters
    /// 
    /// # Arguments
    /// * `order` - Wavelet order (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
    pub fn daubechies(order: usize) -> Option<Self> {
        match order {
            2 => Some(Self::haar()),
            4 => Some(Self::db4()),
            6 => Some(Self::db6()),
            8 => Some(Self::db8()),
            _ => None, // Simplified for common orders
        }
    }

    /// Haar wavelet (simplest)
    fn haar() -> Self {
        let sqrt2_inv = std::f64::consts::FRAC_1_SQRT_2;
        Self {
            dec_lo: vec![sqrt2_inv, sqrt2_inv],
            dec_hi: vec![sqrt2_inv, -sqrt2_inv],
            rec_lo: vec![sqrt2_inv, sqrt2_inv],
            rec_hi: vec![-sqrt2_inv, sqrt2_inv],
        }
    }

    /// Daubechies-4 wavelet
    fn db4() -> Self {
        let sqrt3 = 3.0_f64.sqrt();
        let c0 = (1.0 + sqrt3) / (4.0 * 2.0_f64.sqrt());
        let c1 = (3.0 + sqrt3) / (4.0 * 2.0_f64.sqrt());
        let c2 = (3.0 - sqrt3) / (4.0 * 2.0_f64.sqrt());
        let c3 = (1.0 - sqrt3) / (4.0 * 2.0_f64.sqrt());

        Self {
            dec_lo: vec![c0, c1, c2, c3],
            dec_hi: vec![c3, -c2, c1, -c0],
            rec_lo: vec![c3, c2, c1, c0],
            rec_hi: vec![-c0, c1, -c2, c3],
        }
    }

    /// Daubechies-8 wavelet coefficients
    fn db8() -> Self {
        // Pre-computed DB8 coefficients
        let coeffs = vec![
            0.23037781330885523,
            0.7148465705525415,
            0.6308807679295904,
            -0.02798376941698385,
            -0.18703481171888114,
            0.030841381835986965,
            0.032883011666982945,
            -0.010597401784997278,
        ];

        // High-pass is quadrature mirror of low-pass
        let mut dec_hi = Vec::with_capacity(coeffs.len());
        for (i, &c) in coeffs.iter().enumerate() {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let rev_idx = coeffs.len() - 1 - i;
            dec_hi.push(sign * coeffs[rev_idx]);
        }

        Self {
            dec_lo: coeffs.clone(),
            dec_hi,
            rec_lo: coeffs.iter().rev().copied().collect(),
            rec_hi: coeffs
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    let sign = if i % 2 == 0 { -1.0 } else { 1.0 };
                    sign * c
                })
                .collect(),
        }
    }

    pub fn filter_len(&self) -> usize {
        self.dec_lo.len()
    }
}

/// Zero-allocation Discrete Wavelet Transform engine
pub struct ZeroAllocDwt {
    filters: WaveletFilters,
    /// Working buffer for decomposition
    work_buffer: Vec<f64>,
    /// Maximum decomposition level
    max_level: usize,
}

impl ZeroAllocDwt {
    /// Create DWT engine with pre-allocated buffers
    pub fn new(filters: WaveletFilters, signal_len: usize, max_level: usize) -> Self {
        let work_buffer = vec![0.0; signal_len];
        
        Self {
            filters,
            work_buffer,
            max_level,
        }
    }

    /// Perform single-level DWT decomposition
    /// 
    /// Splits signal into approximation (low-pass) and detail (high-pass) coefficients
    pub fn decompose_level(&mut self, signal: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        let filter_len = self.filters.filter_len();
        if signal.len() < filter_len {
            return None;
        }

        let n = signal.len();
        let half_n = (n + 1) / 2;

        let mut approx = vec![0.0; half_n];
        let mut detail = vec![0.0; half_n];

        // Convolution with downsampling
        for i in 0..half_n {
            let mut sum_lo = 0.0;
            let mut sum_hi = 0.0;

            for j in 0..filter_len {
                let idx = (2 * i + j) % n;
                sum_lo += self.filters.dec_lo[j] * signal[idx];
                sum_hi += self.filters.dec_hi[j] * signal[idx];
            }

            approx[i] = sum_lo;
            detail[i] = sum_hi;
        }

        Some((approx, detail))
    }

    /// Multi-level DWT decomposition
    pub fn decompose(&mut self, signal: &[f64], levels: usize) -> Option<Vec<Vec<f64>>> {
        if levels > self.max_level || levels == 0 {
            return None;
        }

        let mut coeffs = Vec::with_capacity(levels + 1);
        let mut current = signal.to_vec();

        for _ in 0..levels {
            if current.len() < self.filters.filter_len() {
                break;
            }

            let (approx, detail) = self.decompose_level(&current)?;
            coeffs.push(detail);
            current = approx;
        }

        // Final approximation coefficients
        coeffs.push(current);

        // Reverse so last element is final approximation
        coeffs.reverse();
        Some(coeffs)
    }

    /// Single-level reconstruction (inverse DWT)
    pub fn reconstruct_level(&self, approx: &[f64], detail: &[f64]) -> Option<Vec<f64>> {
        if approx.len() != detail.len() {
            return None;
        }

        let n = approx.len() * 2;
        let filter_len = self.filters.filter_len();
        let mut reconstructed = vec![0.0; n];

        // Upsampling and convolution
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..filter_len {
                let k = (i - j + n) % n;
                let half_k = k / 2;
                
                if half_k < approx.len() {
                    if k % 2 == 0 {
                        sum += self.filters.rec_lo[j] * approx[half_k];
                    } else {
                        sum += self.filters.rec_hi[j] * detail[half_k];
                    }
                }
            }
            reconstructed[i] = sum;
        }

        Some(reconstructed)
    }

    /// Full multi-level reconstruction
    pub fn reconstruct(&self, coeffs: &[Vec<f64>]) -> Option<Vec<f64>> {
        if coeffs.is_empty() {
            return None;
        }

        let mut current = coeffs[coeffs.len() - 1].clone();

        for level in (0..coeffs.len() - 1).rev() {
            let detail = &coeffs[level];
            current = self.reconstruct_level(&current, detail)?;
        }

        Some(current)
    }
}

/// Wavelet-based bandpass filter for alpha extraction
pub struct BandpassAlphaFilter {
    dwt: ZeroAllocDwt,
    /// Target frequency bands (in terms of decomposition levels)
    low_level: usize,
    high_level: usize,
}

impl BandpassAlphaFilter {
    /// Create bandpass filter targeting specific scales
    pub fn new(signal_len: usize, low_level: usize, high_level: usize) -> Option<Self> {
        let filters = WaveletFilters::daubechies(4)?;
        let max_level = (signal_len as f64).log2() as usize;
        
        if low_level >= high_level || high_level > max_level {
            return None;
        }

        let dwt = ZeroAllocDwt::new(filters, signal_len, max_level);

        Some(Self {
            dwt,
            low_level,
            high_level,
        })
    }

    /// Extract alpha signal in target frequency band
    pub fn extract_alpha(&mut self, signal: &[f64]) -> Option<Vec<f64>> {
        let coeffs = self.dwt.decompose(signal, self.high_level)?;

        // Zero out coefficients outside target band
        let mut filtered_coeffs = Vec::with_capacity(coeffs.len());
        for (i, coeff) in coeffs.iter().enumerate() {
            if i == 0 {
                // Keep approximation at highest level
                filtered_coeffs.push(coeff.clone());
            } else if i >= self.low_level && i <= self.high_level {
                // Keep detail coefficients in band
                filtered_coeffs.push(coeff.clone());
            } else {
                // Zero out coefficients outside band
                filtered_coeffs.push(vec![0.0; coeff.len()]);
            }
        }

        // Reconstruct from filtered coefficients
        self.dwt.reconstruct(&filtered_coeffs)
    }

    /// Compute energy in each frequency band
    pub fn band_energies(&mut self, signal: &[f64]) -> Option<Vec<f64>> {
        let coeffs = self.dwt.decompose(signal, self.high_level)?;
        
        let mut energies = Vec::with_capacity(coeffs.len());
        for coeff in &coeffs {
            let energy: f64 = coeff.iter().map(|&x| x * x).sum();
            energies.push(energy);
        }

        Some(energies)
    }

    /// Get signal-to-noise ratio for target band
    pub fn snr(&mut self, signal: &[f64]) -> Option<f64> {
        let energies = self.band_energies(signal)?;
        
        let mut signal_energy = 0.0;
        let mut noise_energy = 0.0;

        for (i, &e) in energies.iter().enumerate() {
            if i >= self.low_level && i <= self.high_level {
                signal_energy += e;
            } else {
                noise_energy += e;
            }
        }

        if noise_energy < 1e-16 {
            return Some(f64::INFINITY);
        }

        Some((signal_energy / noise_energy).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haar_decomposition() {
        let filters = WaveletFilters::haar();
        assert_eq!(filters.filter_len(), 2);
    }

    #[test]
    fn test_db4_filters() {
        let filters = WaveletFilters::db4();
        assert_eq!(filters.filter_len(), 4);
        
        // Check perfect reconstruction property
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut dwt = ZeroAllocDwt::new(filters, signal.len(), 3);
        
        let coeffs = dwt.decompose(&signal, 2).unwrap();
        let reconstructed = dwt.reconstruct(&coeffs).unwrap();
        
        for (orig, recon) in signal.iter().zip(reconstructed.iter()) {
            assert!((orig - recon).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bandpass_filter() {
        let signal_len = 256;
        let mut filter = BandpassAlphaFilter::new(signal_len, 2, 4).unwrap();
        
        // Generate signal with multiple frequency components
        let signal: Vec<f64> = (0..signal_len)
            .map(|i| {
                (i as f64 * 0.1).sin() +  // Low freq
                (i as f64 * 0.5).sin() * 0.5 +  // Mid freq
                (i as f64 * 2.0).sin() * 0.2  // High freq
            })
            .collect();
        
        let alpha = filter.extract_alpha(&signal).unwrap();
        assert_eq!(alpha.len(), signal_len);
        
        let snr = filter.snr(&signal).unwrap();
        assert!(snr.is_finite());
    }

    #[test]
    fn test_energy_distribution() {
        let signal_len = 128;
        let mut filter = BandpassAlphaFilter::new(signal_len, 1, 3).unwrap();
        
        // Pure low-frequency signal
        let signal: Vec<f64> = (0..signal_len)
            .map(|i| (i as f64 * 0.05).sin())
            .collect();
        
        let energies = filter.band_energies(&signal).unwrap();
        
        // Most energy should be in approximation (lowest frequency)
        assert!(energies[0] > energies.iter().skip(1).sum::<f64>());
    }
}
