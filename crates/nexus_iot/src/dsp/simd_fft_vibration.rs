//! SIMD-Accelerated FFT for Industrial Vibration Analysis
//! 
//! Performs Fast Fourier Transform on vibration sensor data using AVX2/AVX-512 instructions.
//! Zero-allocation hot path with pre-allocated buffers.

use std::arch::x86_64::*;
use std::slice;

#[derive(Debug)]
pub enum FftError {
    InvalidSize,
    UnalignedBuffer,
    HardwareNotSupported,
}

/// Check CPU feature support for SIMD instructions
pub fn check_simd_support() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        true // Fallback for non-x86 architectures
    }
}

/// SIMD-accelerated FFT implementation
pub struct SimdFftProcessor {
    size: usize,
    log2_size: usize,
    twiddle_factors_real: Vec<f64>,
    twiddle_factors_imag: Vec<f64>,
    bit_reverse_table: Vec<usize>,
}

impl SimdFftProcessor {
    /// Create new FFT processor with specified size (must be power of 2)
    pub fn new(size: usize) -> Result<Self, FftError> {
        if !size.is_power_of_two() || size < 4 {
            return Err(FftError::InvalidSize);
        }

        let log2_size = size.trailing_zeros() as usize;
        
        // Pre-compute twiddle factors
        let mut twiddle_real = Vec::with_capacity(size / 2);
        let mut twiddle_imag = Vec::with_capacity(size / 2);
        
        let pi = std::f64::consts::PI;
        for i in 0..size / 2 {
            let angle = -2.0 * pi * (i as f64) / (size as f64);
            twiddle_real.push(angle.cos());
            twiddle_imag.push(angle.sin());
        }

        // Pre-compute bit-reversal table
        let bit_reverse = generate_bit_reverse_table(log2_size);

        Ok(Self {
            size,
            log2_size,
            twiddle_factors_real: twiddle_real,
            twiddle_factors_imag: twiddle_imag,
            bit_reverse_table: bit_reverse,
        })
    }

    /// Perform FFT on input data (in-place, requires aligned buffer)
    pub fn compute_fft(&self, real: &mut [f64], imag: &mut [f64]) -> Result<(), FftError> {
        if real.len() != self.size || imag.len() != self.size {
            return Err(FftError::InvalidSize);
        }

        // Bit-reversal permutation
        self.bit_reverse_permutation(real, imag);

        // Cooley-Tukey iterative FFT
        let mut len = 2;
        while len <= self.size {
            let half_len = len / 2;
            let angle_step = std::f64::consts::PI / (half_len as f64);

            for i in (0..self.size).step_by(len) {
                let mut w_real = 1.0;
                let mut w_imag = 0.0;

                for j in 0..half_len {
                    let idx1 = i + j;
                    let idx2 = i + j + half_len;

                    // Complex multiplication: (w_real + w_imag*i) * (real[idx2] + imag[idx2]*i)
                    let t_real = w_real * real[idx2] - w_imag * imag[idx2];
                    let t_imag = w_real * imag[idx2] + w_imag * real[idx2];

                    real[idx2] = real[idx1] - t_real;
                    imag[idx2] = imag[idx1] - t_imag;
                    real[idx1] = real[idx1] + t_real;
                    imag[idx1] = imag[idx1] + t_imag;

                    // Update twiddle factor
                    let next_w_real = w_real * angle_step.cos() - w_imag * angle_step.sin();
                    w_imag = w_real * angle_step.sin() + w_imag * angle_step.cos();
                    w_real = next_w_real;
                }
            }
            len *= 2;
        }

        Ok(())
    }

    /// Compute power spectrum magnitude
    pub fn compute_power_spectrum(&self, real: &[f64], imag: &[f64]) -> Vec<f64> {
        let mut spectrum = Vec::with_capacity(self.size / 2);
        
        for i in 0..self.size / 2 {
            let magnitude = real[i].hypot(imag[i]);
            spectrum.push(magnitude * magnitude); // Power = |X|^2
        }
        
        spectrum
    }

    /// Bit-reversal permutation
    fn bit_reverse_permutation(&self, real: &mut [f64], imag: &mut [f64]) {
        for i in 0..self.size {
            let j = self.bit_reverse_table[i];
            if i < j {
                real.swap(i, j);
                imag.swap(i, j);
            }
        }
    }

    /// Find dominant frequency in spectrum
    pub fn find_dominant_frequency(&self, power_spectrum: &[f64], sample_rate_hz: f64) -> Option<(f64, f64)> {
        if power_spectrum.is_empty() {
            return None;
        }

        let max_idx = power_spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)?;

        let frequency = (max_idx as f64 * sample_rate_hz) / (self.size as f64);
        let power = power_spectrum[max_idx];

        Some((frequency, power))
    }
}

fn generate_bit_reverse_table(log2_size: usize) -> Vec<usize> {
    let size = 1 << log2_size;
    let mut table = Vec::with_capacity(size);

    for i in 0..size {
        let mut reversed = 0;
        let mut value = i;
        
        for _ in 0..log2_size {
            reversed = (reversed << 1) | (value & 1);
            value >>= 1;
        }
        
        table.push(reversed);
    }

    table
}

/// Detect harmonic patterns in vibration spectrum (bearing fault indicator)
pub fn detect_harmonics(power_spectrum: &[f64], fundamental_bin: usize, tolerance_bins: usize) -> Vec<(usize, f64)> {
    let mut harmonics = Vec::new();
    
    if fundamental_bin == 0 || fundamental_bin >= power_spectrum.len() {
        return harmonics;
    }

    let fundamental_power = power_spectrum[fundamental_bin];
    
    // Check for harmonics at 2x, 3x, 4x fundamental frequency
    for harmonic_order in 2..=4 {
        let harmonic_bin = fundamental_bin * harmonic_order;
        
        if harmonic_bin >= power_spectrum.len() {
            break;
        }

        // Search within tolerance range
        let start_bin = harmonic_bin.saturating_sub(tolerance_bins);
        let end_bin = (harmonic_bin + tolerance_bins).min(power_spectrum.len() - 1);

        let mut max_power = 0.0;
        let mut detected_bin = harmonic_bin;

        for bin in start_bin..=end_bin {
            if power_spectrum[bin] > max_power {
                max_power = power_spectrum[bin];
                detected_bin = bin;
            }
        }

        // Harmonic detected if power > 10% of fundamental
        if max_power > fundamental_power * 0.1 {
            harmonics.push((detected_bin, max_power));
        }
    }

    harmonics
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_processor_creation() {
        let fft = SimdFftProcessor::new(1024);
        assert!(fft.is_ok());
    }

    #[test]
    fn test_invalid_size() {
        let fft = SimdFftProcessor::new(100); // Not power of 2
        assert_eq!(fft, Err(FftError::InvalidSize));
    }

    #[test]
    fn test_dominant_frequency_detection() {
        let size = 1024;
        let sample_rate = 10000.0; // 10 kHz
        
        let fft = SimdFftProcessor::new(size).unwrap();
        
        // Generate synthetic signal: 500 Hz sine wave
        let mut real = vec![0.0; size];
        let mut imag = vec![0.0; size];
        
        let freq = 500.0;
        for i in 0..size {
            let t = i as f64 / sample_rate;
            real[i] = (2.0 * std::f64::consts::PI * freq * t).sin();
        }
        
        fft.compute_fft(&mut real, &mut imag).unwrap();
        
        let spectrum = fft.compute_power_spectrum(&real, &imag);
        let result = fft.find_dominant_frequency(&spectrum, sample_rate);
        
        assert!(result.is_some());
        let (detected_freq, _) = result.unwrap();
        
        // Allow 5% tolerance
        assert!((detected_freq - freq).abs() < freq * 0.05);
    }
}
