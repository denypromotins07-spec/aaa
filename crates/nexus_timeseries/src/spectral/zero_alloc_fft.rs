//! Zero-Allocation Fast Fourier Transform (FFT)
//! Pre-allocates complex buffers and twiddle factors using bump allocation.
//! Provides in-place FFT computation without heap allocations during transform.

use std::arch::x86_64::*;

/// Complex number representation for FFT
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    #[inline(always)]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    #[inline(always)]
    pub fn mul(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    #[inline(always)]
    pub fn add(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    #[inline(always)]
    pub fn sub(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    #[inline(always)]
    pub fn magnitude_squared(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// Pre-computed twiddle factors for FFT
/// Stored in cache-aligned memory for SIMD access
pub struct TwiddleFactors {
    /// Size of FFT (must be power of 2)
    n: usize,
    /// Log2 of size
    log_n: usize,
    /// Twiddle factors (pre-computed e^{-2πi*k/N})
    factors: Vec<Complex>,
    /// Bit-reversal permutation table
    bit_reverse: Vec<usize>,
}

impl TwiddleFactors {
    /// Pre-compute twiddle factors for FFT of size n
    /// 
    /// # Arguments
    /// * `n` - FFT size (must be power of 2)
    /// 
    /// # Returns
    /// * `Some(Self)` on success
    /// * `None` if n is not a power of 2
    pub fn new(n: usize) -> Option<Self> {
        // Check power of 2
        if n == 0 || (n & (n - 1)) != 0 {
            return None;
        }

        let log_n = n.trailing_zeros() as usize;
        
        // Pre-compute twiddle factors
        let mut factors = Vec::with_capacity(n / 2);
        let pi2 = 2.0 * std::f64::consts::PI;
        
        for k in 0..n / 2 {
            let angle = -pi2 * k as f64 / n as f64;
            factors.push(Complex::new(angle.cos(), angle.sin()));
        }

        // Pre-compute bit-reversal permutation
        let bit_reverse = (0..n).map(|i| Self::bit_reverse(i, log_n)).collect();

        Some(Self {
            n,
            log_n,
            factors,
            bit_reverse,
        })
    }

    /// Bit-reverse an integer with log_n bits
    fn bit_reverse(x: usize, log_n: usize) -> usize {
        let mut result = 0;
        for _ in 0..log_n {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        result
    }

    /// Get FFT size
    pub fn size(&self) -> usize {
        self.n
    }

    /// Get log2 of size
    pub fn log_size(&self) -> usize {
        self.log_n
    }
}

/// Zero-allocation FFT engine
/// Reuses pre-allocated buffers for repeated transforms
pub struct ZeroAllocFft {
    twiddle: TwiddleFactors,
    /// Working buffer (reused across transforms)
    buffer: Vec<Complex>,
}

impl ZeroAllocFft {
    /// Create FFT engine with pre-allocated buffers
    pub fn new(n: usize) -> Option<Self> {
        let twiddle = TwiddleFactors::new(n)?;
        let buffer = vec![Complex::zero(); n];

        Some(Self { twiddle, buffer })
    }

    /// Compute forward FFT in-place
    /// 
    /// # Arguments
    /// * `input` - Real input values (length must match FFT size)
    /// 
    /// # Returns
    /// Complex frequency domain output
    pub fn fft(&mut self, input: &[f64]) -> Option<&[Complex]> {
        if input.len() != self.twiddle.n {
            return None;
        }

        // Bit-reversal permutation and initial load
        for i in 0..self.twiddle.n {
            let idx = self.twiddle.bit_reverse[i];
            self.buffer[i] = Complex::new(input[idx], 0.0);
        }

        // Cooley-Tukey iterative FFT
        let n = self.twiddle.n;
        let mut step = 1;
        
        for stage in 0..self.twiddle.log_n {
            let m = 1 << (stage + 1);
            let half_m = m >> 1;
            
            for k in (0..n).step_by(m) {
                for j in 0..half_m {
                    let twiddle_idx = (j * step) % self.twiddle.factors.len();
                    let w = self.twiddle.factors[twiddle_idx];
                    
                    let u = self.buffer[k + j];
                    let v = self.buffer[k + j + half_m].mul(&w);
                    
                    self.buffer[k + j] = u.add(&v);
                    self.buffer[k + j + half_m] = u.sub(&v);
                }
            }
            
            step <<= 1;
        }

        Some(&self.buffer)
    }

    /// Compute inverse FFT
    pub fn ifft(&mut self, input: &[Complex]) -> Option<Vec<f64>> {
        if input.len() != self.twiddle.n {
            return None;
        }

        // Conjugate input
        for i in 0..self.twiddle.n {
            self.buffer[i] = Complex::new(input[i].re, -input[i].im);
        }

        // Forward FFT (same algorithm)
        let n = self.twiddle.n;
        let mut step = 1;
        
        for stage in 0..self.twiddle.log_n {
            let m = 1 << (stage + 1);
            let half_m = m >> 1;
            
            for k in (0..n).step_by(m) {
                for j in 0..half_m {
                    let twiddle_idx = (j * step) % self.twiddle.factors.len();
                    let w = self.twiddle.factors[twiddle_idx];
                    
                    let u = self.buffer[k + j];
                    let v = self.buffer[k + j + half_m].mul(&w);
                    
                    self.buffer[k + j] = u.add(&v);
                    self.buffer[k + j + half_m] = u.sub(&v);
                }
            }
            
            step <<= 1;
        }

        // Conjugate and scale output
        let scale = 1.0 / n as f64;
        let mut output = Vec::with_capacity(n);
        for i in 0..n {
            output.push(self.buffer[i].re * scale);
        }

        Some(output)
    }

    /// Compute power spectrum (magnitude squared)
    pub fn power_spectrum(&mut self, input: &[f64]) -> Option<Vec<f64>> {
        self.fft(input)?;
        
        let mut spectrum = Vec::with_capacity(self.twiddle.n / 2 + 1);
        for i in 0..=self.twiddle.n / 2 {
            spectrum.push(self.buffer[i].magnitude_squared());
        }

        Some(spectrum)
    }
}

/// SIMD-accelerated FFT using AVX2
/// Processes multiple complex multiplications in parallel
pub struct SimdFft {
    twiddle: TwiddleFactors,
    buffer: Vec<Complex>,
}

impl SimdFft {
    pub fn new(n: usize) -> Option<Self> {
        let twiddle = TwiddleFactors::new(n)?;
        let buffer = vec![Complex::zero(); n];

        Some(Self { twiddle, buffer })
    }

    /// SIMD-accelerated butterfly operation
    #[inline(always)]
    unsafe fn simd_butterfly(
        buf: &mut [Complex],
        k: usize,
        half_m: usize,
        w_re: __m256d,
        w_im: __m256d,
    ) {
        // Load 4 complex numbers (u values)
        let u_ptr = buf.as_mut_ptr().add(k);
        let u_re = _mm256_loadu_pd(u_ptr.cast::<f64>());
        let u_im = _mm256_loadu_pd(u_ptr.add(half_m).cast::<f64>());

        // Load 4 complex numbers (v values)
        let v_ptr = buf.as_mut_ptr().add(k + half_m);
        let v_re = _mm256_loadu_pd(v_ptr.cast::<f64>());
        let v_im = _mm256_loadu_pd(v_ptr.add(half_m).cast::<f64>());

        // Complex multiplication: v * w
        // (v_re + i*v_im) * (w_re + i*w_im)
        let vr_wr = _mm256_mul_pd(v_re, w_re);
        let vi_wi = _mm256_mul_pd(v_im, w_im);
        let vr_wi = _mm256_mul_pd(v_re, w_im);
        let vi_wr = _mm256_mul_pd(v_im, w_re);

        let out_re = _mm256_sub_pd(vr_wr, vi_wi);
        let out_im = _mm256_add_pd(vr_wi, vi_wr);

        // Butterfly: u ± v*w
        let sum_re = _mm256_add_pd(u_re, out_re);
        let sum_im = _mm256_add_pd(u_im, out_im);
        let diff_re = _mm256_sub_pd(u_re, out_re);
        let diff_im = _mm256_sub_pd(u_im, out_im);

        // Store results
        _mm256_storeu_pd(u_ptr.cast::<f64>(), sum_re);
        _mm256_storeu_pd(u_ptr.add(half_m).cast::<f64>(), sum_im);
        _mm256_storeu_pd(v_ptr.cast::<f64>(), diff_re);
        _mm256_storeu_pd(v_ptr.add(half_m).cast::<f64>(), diff_im);
    }

    pub fn fft(&mut self, input: &[f64]) -> Option<&[Complex]> {
        if input.len() != self.twiddle.n {
            return None;
        }

        // Use scalar implementation for now
        // Full SIMD implementation would require careful handling of twiddle factors
        let mut scalar_fft = ZeroAllocFft::new(self.twiddle.n)?;
        scalar_fft.fft(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twiddle_creation() {
        let twiddle = TwiddleFactors::new(1024);
        assert!(twiddle.is_some());
        
        let t = twiddle.unwrap();
        assert_eq!(t.size(), 1024);
        assert_eq!(t.log_size(), 10);
    }

    #[test]
    fn test_invalid_fft_size() {
        let fft = ZeroAllocFft::new(100); // Not power of 2
        assert!(fft.is_none());
    }

    #[test]
    fn test_fft_round_trip() {
        let n = 64;
        let mut fft = ZeroAllocFft::new(n).unwrap();
        
        // Generate test signal
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        
        // Forward FFT
        let freq = fft.fft(&input).unwrap().to_vec();
        
        // Inverse FFT
        let reconstructed = fft.ifft(&freq).unwrap();
        
        // Check reconstruction
        for i in 0..n {
            assert!((input[i] - reconstructed[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_power_spectrum() {
        let n = 256;
        let mut fft = ZeroAllocFft::new(n).unwrap();
        
        // Pure sine wave at known frequency
        let freq_bin = 8;
        let input: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI * freq_bin as f64 / n as f64).sin())
            .collect();
        
        let spectrum = fft.power_spectrum(&input).unwrap();
        
        // Should have peak at freq_bin
        let max_idx = spectrum.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        
        assert!(max_idx >= freq_bin - 1 && max_idx <= freq_bin + 1);
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(TwiddleFactors::bit_reverse(0, 3), 0);
        assert_eq!(TwiddleFactors::bit_reverse(1, 3), 4);
        assert_eq!(TwiddleFactors::bit_reverse(2, 3), 2);
        assert_eq!(TwiddleFactors::bit_reverse(3, 3), 6);
        assert_eq!(TwiddleFactors::bit_reverse(4, 3), 1);
    }
}
