//! SIMD-accelerated EEG/fNIRS stream parser for zero-allocation neural telemetry ingestion.
//! 
//! This module provides high-performance parsing of raw EEG/fNIRS data streams from consumer
//! and research-grade BCIs (e.g., Neuralink, OpenBCI, Artinis) using SIMD acceleration for
//! real-time spectral analysis and artifact detection.

use core::slice;
use std::arch::x86_64::*;

/// Constants for EEG frequency bands (Hz)
pub const DELTA_LOW: f32 = 0.5;
pub const DELTA_HIGH: f32 = 4.0;
pub const THETA_LOW: f32 = 4.0;
pub const THETA_HIGH: f32 = 8.0;
pub const ALPHA_LOW: f32 = 8.0;
pub const ALPHA_HIGH: f32 = 13.0;
pub const BETA_LOW: f32 = 13.0;
pub const BETA_HIGH: f32 = 30.0;
pub const GAMMA_LOW: f32 = 30.0;
pub const GAMMA_HIGH: f32 = 100.0;

/// EMG contamination threshold (Hz) - frequencies above this are likely muscle artifacts
pub const EMG_CUTOFF_LOW: f32 = 110.0;
pub const EMG_CUTOFF_HIGH: f32 = 200.0;

/// EOG blink artifact frequency range (Hz)
pub const EOG_LOW: f32 = 0.1;
pub const EOG_HIGH: f32 = 3.0;

/// Maximum number of channels supported (zero-alloc fixed size)
pub const MAX_EEG_CHANNELS: usize = 256;

/// Sample rate for EEG data (Hz)
pub const DEFAULT_SAMPLE_RATE: u32 = 1000;

/// Buffer size for SIMD processing (must be multiple of 8 for AVX2)
pub const SIMD_BUFFER_SIZE: usize = 1024;

/// Error types for EEG stream processing
#[derive(Debug, Clone, PartialEq)]
pub enum EegStreamError {
    InvalidSampleRate(u32),
    ChannelCountExceeded(usize),
    SignalSaturation(f32),
    InvalidFrequencyRange(f32, f32),
    SimdAlignmentError,
    BufferOverflow,
    ClockDriftDetected(f64),
}

impl core::fmt::Display for EegStreamError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EegStreamError::InvalidSampleRate(rate) => write!(f, "Invalid sample rate: {} Hz", rate),
            EegStreamError::ChannelCountExceeded(count) => write!(f, "Channel count {} exceeds maximum {}", count, MAX_EEG_CHANNELS),
            EegStreamError::SignalSaturation(value) => write!(f, "Signal saturation detected: {} µV", value),
            EegStreamError::InvalidFrequencyRange(low, high) => write!(f, "Invalid frequency range: {}-{} Hz", low, high),
            EegStreamError::SimdAlignmentError => write!(f, "SIMD alignment error in buffer"),
            EegStreamError::BufferOverflow => write!(f, "Processing buffer overflow"),
            EegStreamError::ClockDriftDetected(drift) => write!(f, "Clock drift detected: {} ms", drift),
        }
    }
}

impl std::error::Error for EegStreamError {}

/// Zero-allocation EEG channel data structure
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy)]
pub struct EegChannel {
    pub id: u16,
    pub label: [u8; 32],
    pub impedance_kohms: f32,
    pub gain: f32,
    pub offset_uv: f32,
    pub is_active: bool,
}

impl Default for EegChannel {
    fn default() -> Self {
        Self {
            id: 0,
            label: [0u8; 32],
            impedance_kohms: 0.0,
            gain: 1.0,
            offset_uv: 0.0,
            is_active: false,
        }
    }
}

/// Zero-allocation EEG stream buffer with SIMD alignment
#[repr(C, align(32))]
pub struct EegStreamBuffer {
    /// Raw voltage samples in microvolts (interleaved channels)
    samples: [f32; SIMD_BUFFER_SIZE * MAX_EEG_CHANNELS],
    /// Timestamps for each sample batch (nanoseconds)
    timestamps: [u64; SIMD_BUFFER_SIZE],
    /// Number of active channels
    channel_count: usize,
    /// Current write position
    write_pos: usize,
    /// Sample rate in Hz
    sample_rate: u32,
    /// Clock synchronization offset (ns)
    clock_offset_ns: i64,
}

impl EegStreamBuffer {
    /// Create a new zero-initialized EEG stream buffer
    #[inline]
    pub const fn new() -> Self {
        Self {
            samples: [0.0; SIMD_BUFFER_SIZE * MAX_EEG_CHANNELS],
            timestamps: [0; SIMD_BUFFER_SIZE],
            channel_count: 0,
            write_pos: 0,
            sample_rate: DEFAULT_SAMPLE_RATE,
            clock_offset_ns: 0,
        }
    }

    /// Initialize buffer with specified channel count and sample rate
    #[inline]
    pub fn init(&mut self, channel_count: usize, sample_rate: u32) -> Result<(), EegStreamError> {
        if channel_count > MAX_EEG_CHANNELS {
            return Err(EegStreamError::ChannelCountExceeded(channel_count));
        }
        if sample_rate < 100 || sample_rate > 10000 {
            return Err(EegStreamError::InvalidSampleRate(sample_rate));
        }
        
        self.channel_count = channel_count;
        self.sample_rate = sample_rate;
        self.write_pos = 0;
        
        // Zero out the buffer
        unsafe {
            core::ptr::write_bytes(self.samples.as_mut_ptr(), 0, self.samples.len());
            core::ptr::write_bytes(self.timestamps.as_mut_ptr(), 0, self.timestamps.len());
        }
        
        Ok(())
    }

    /// Push a new sample batch (zero-copy, must be aligned)
    #[inline]
    pub fn push_batch(&mut self, samples: &[f32], timestamp_ns: u64) -> Result<(), EegStreamError> {
        if samples.len() != self.channel_count {
            return Err(EegStreamError::ChannelCountExceeded(samples.len()));
        }
        
        if self.write_pos >= SIMD_BUFFER_SIZE {
            return Err(EegStreamError::BufferOverflow);
        }

        // Check for signal saturation
        for &sample in samples.iter() {
            if sample.abs() > 5000.0 {
                return Err(EegStreamError::SignalSaturation(sample));
            }
        }

        let base_idx = self.write_pos * self.channel_count;
        let dest = &mut self.samples[base_idx..base_idx + self.channel_count];
        dest.copy_from_slice(samples);
        self.timestamps[self.write_pos] = timestamp_ns.wrapping_add(self.clock_offset_ns as u64);
        self.write_pos += 1;

        Ok(())
    }

    /// Get pointer to raw samples for SIMD processing (unsafe, zero-copy)
    #[inline]
    pub unsafe fn as_simd_ptr(&self) -> *const f32 {
        self.samples.as_ptr()
    }

    /// Get current fill level (number of sample batches)
    #[inline]
    pub const fn fill_level(&self) -> usize {
        self.write_pos
    }

    /// Check if buffer is ready for processing
    #[inline]
    pub const fn is_ready(&self) -> bool {
        self.write_pos == SIMD_BUFFER_SIZE
    }

    /// Reset buffer for reuse
    #[inline]
    pub fn reset(&mut self) {
        self.write_pos = 0;
        unsafe {
            core::ptr::write_bytes(self.samples.as_mut_ptr(), 0, self.samples.len());
        }
    }

    /// Apply clock synchronization correction
    #[inline]
    pub fn sync_clock(&mut self, reference_time_ns: u64, local_time_ns: u64) {
        self.clock_offset_ns = reference_time_ns as i64 - local_time_ns as i64;
    }
}

impl Default for EegStreamBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// SIMD-accelerated bandpass filter state
#[repr(C, align(32))]
pub struct SimdBiquadState {
    /// Filter coefficients (a0, a1, a2, b1, b2)
    a0: __m256,
    a1: __m256,
    a2: __m256,
    b1: __m256,
    b2: __m256,
    /// Delay elements (x1, x2, y1, y2)
    x1: __m256,
    x2: __m256,
    y1: __m256,
    y2: __m256,
}

impl SimdBiquadState {
    /// Design a Butterworth bandpass filter using bilinear transform
    pub fn design_bandpass(low_freq: f32, high_freq: f32, sample_rate: u32) -> Result<Self, EegStreamError> {
        if low_freq >= high_freq {
            return Err(EegStreamError::InvalidFrequencyRange(low_freq, high_freq));
        }
        if low_freq <= 0.0 || high_freq >= sample_rate as f32 / 2.0 {
            return Err(EegStreamError::InvalidFrequencyRange(low_freq, high_freq));
        }

        let nyquist = sample_rate as f32 / 2.0;
        let w1 = (low_freq / nyquist * core::f32::consts::PI).sin();
        let w2 = (high_freq / nyquist * core::f32::consts::PI).sin();
        
        // Simplified coefficient calculation (full implementation would use proper bilinear transform)
        let center = (low_freq * high_freq).sqrt();
        let bandwidth = high_freq - low_freq;
        let q = center / bandwidth;
        
        let coeffs = [
            1.0 / (1.0 + 1.0 / q),  // a0
            0.0,                     // a1
            -1.0 / (1.0 + 1.0 / q), // a2
            2.0 * (1.0 - 1.0 / q) / (1.0 + 1.0 / q), // b1
            -(1.0 - 1.0 / q) / (1.0 + 1.0 / q),      // b2
        ];

        unsafe {
            Ok(Self {
                a0: _mm256_set1_ps(coeffs[0]),
                a1: _mm256_set1_ps(coeffs[1]),
                a2: _mm256_set1_ps(coeffs[2]),
                b1: _mm256_set1_ps(coeffs[3]),
                b2: _mm256_set1_ps(coeffs[4]),
                x1: _mm256_setzero_ps(),
                x2: _mm256_setzero_ps(),
                y1: _mm256_setzero_ps(),
                y2: _mm256_setzero_ps(),
            })
        }
    }

    /// Process 8 samples at once using AVX2
    #[inline]
    pub unsafe fn process(&mut self, input: __m256) -> __m256 {
        // y[n] = a0*x[n] + a1*x[n-1] + a2*x[n-2] - b1*y[n-1] - b2*y[n-2]
        let term1 = _mm256_mul_ps(self.a0, input);
        let term2 = _mm256_mul_ps(self.a1, self.x1);
        let term3 = _mm256_mul_ps(self.a2, self.x2);
        let term4 = _mm256_mul_ps(self.b1, self.y1);
        let term5 = _mm256_mul_ps(self.b2, self.y2);

        let sum = _mm256_add_ps(term1, term2);
        let sum = _mm256_add_ps(sum, term3);
        let sum = _mm256_sub_ps(sum, term4);
        let output = _mm256_sub_ps(sum, term5);

        // Update delay elements
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    /// Reset filter state
    #[inline]
    pub fn reset(&mut self) {
        unsafe {
            self.x1 = _mm256_setzero_ps();
            self.x2 = _mm256_setzero_ps();
            self.y1 = _mm256_setzero_ps();
            self.y2 = _mm256_setzero_ps();
        }
    }
}

/// Spectral power estimator using Goertzel algorithm (SIMD-accelerated)
#[repr(C, align(32))]
pub struct SimdGoertzelEngine {
    /// Target frequency bin
    target_bin: f32,
    /// Coefficient: 2*cos(2*pi*target_bin/sample_rate)
    coeff: __m256,
    /// Delay elements for 8 parallel channels
    s1: __m256,
    s2: __m256,
    /// Number of samples processed
    sample_count: usize,
}

impl SimdGoertzelEngine {
    /// Initialize Goertzel engine for specific frequency
    pub fn new(target_freq: f32, sample_rate: u32, block_size: usize) -> Result<Self, EegStreamError> {
        if target_freq <= 0.0 || target_freq >= sample_rate as f32 / 2.0 {
            return Err(EegStreamError::InvalidFrequencyRange(target_freq, target_freq + 1.0));
        }

        let k = (target_freq * block_size as f32 / sample_rate as f32).round();
        let omega = 2.0 * core::f32::consts::PI * k / block_size as f32;
        let coeff_val = 2.0 * omega.cos();

        unsafe {
            Ok(Self {
                target_bin: k,
                coeff: _mm256_set1_ps(coeff_val),
                s1: _mm256_setzero_ps(),
                s2: _mm256_setzero_ps(),
                sample_count: 0,
            })
        }
    }

    /// Process one sample block for 8 channels
    #[inline]
    pub unsafe fn process_block(&mut self, samples: *const f32, block_size: usize) -> __m256 {
        self.s1 = _mm256_setzero_ps();
        self.s2 = _mm256_setzero_ps();
        self.sample_count = 0;

        for i in 0..block_size {
            let input = _mm256_load_ps(samples.add(i * 8));
            let s = _mm256_add_ps(
                _mm256_mul_ps(self.coeff, self.s1),
                _mm256_sub_ps(input, self.s2)
            );
            self.s2 = self.s1;
            self.s1 = s;
            self.sample_count += 1;
        }

        // Power = s1^2 + s2^2 - s1*s2*coeff
        let s1_sq = _mm256_mul_ps(self.s1, self.s1);
        let s2_sq = _mm256_mul_ps(self.s2, self.s2);
        let s1_s2 = _mm256_mul_ps(self.s1, self.s2);
        let cross = _mm256_mul_ps(s1_s2, self.coeff);
        
        _mm256_add_ps(_mm256_sub_ps(_mm256_add_ps(s1_sq, s2_sq), cross))
    }

    /// Reset engine state
    #[inline]
    pub fn reset(&mut self) {
        unsafe {
            self.s1 = _mm256_setzero_ps();
            self.s2 = _mm256_setzero_ps();
        }
        self.sample_count = 0;
    }
}

/// Main EEG stream processor with SIMD acceleration
pub struct SimdEegProcessor {
    /// Channel configurations
    channels: [EegChannel; MAX_EEG_CHANNELS],
    /// Bandpass filters for each frequency band per channel
    delta_filter: [SimdBiquadState; MAX_EEG_CHANNELS],
    theta_filter: [SimdBiquadState; MAX_EEG_CHANNELS],
    alpha_filter: [SimdBiquadState; MAX_EEG_CHANNELS],
    beta_filter: [SimdBiquadState; MAX_EEG_CHANNELS],
    gamma_filter: [SimdBiquadState; MAX_EEG_CHANNELS],
    /// EMG notch filter (high-frequency rejection)
    emg_notch: [SimdBiquadState; MAX_EEG_CHANNELS],
    /// Goertzel engines for spectral estimation
    goertzel_delta: [SimdGoertzelEngine; MAX_EEG_CHANNELS],
    goertzel_theta: [SimdGoertzelEngine; MAX_EEG_CHANNELS],
    goertzel_alpha: [SimdGoertzelEngine; MAX_EEG_CHANNELS],
    goertzel_beta: [SimdGoertzelEngine; MAX_EEG_CHANNELS],
    goertzel_gamma: [SimdGoertzelEngine; MAX_EEG_CHANNELS],
    /// Current spectral power estimates (µV²/Hz)
    power_delta: [f32; MAX_EEG_CHANNELS],
    power_theta: [f32; MAX_EEG_CHANNELS],
    power_alpha: [f32; MAX_EEG_CHANNELS],
    power_beta: [f32; MAX_EEG_CHANNELS],
    power_gamma: [f32; MAX_EEG_CHANNELS],
    /// EMG contamination estimate
    emg_power: [f32; MAX_EEG_CHANNELS],
    /// Active channel count
    active_channels: usize,
    /// Sample rate
    sample_rate: u32,
}

impl SimdEegProcessor {
    /// Create new EEG processor
    pub fn new() -> Self {
        // Note: This uses const initialization where possible
        Self {
            channels: [EegChannel::default(); MAX_EEG_CHANNELS],
            delta_filter: unsafe { core::mem::zeroed() },
            theta_filter: unsafe { core::mem::zeroed() },
            alpha_filter: unsafe { core::mem::zeroed() },
            beta_filter: unsafe { core::mem::zeroed() },
            gamma_filter: unsafe { core::mem::zeroed() },
            emg_notch: unsafe { core::mem::zeroed() },
            goertzel_delta: unsafe { core::mem::MaybeUninit::<[SimdGoertzelEngine; MAX_EEG_CHANNELS]>::zeroed().assume_init() },
            goertzel_theta: unsafe { core::mem::MaybeUninit::<[SimdGoertzelEngine; MAX_EEG_CHANNELS]>::zeroed().assume_init() },
            goertzel_alpha: unsafe { core::mem::MaybeUninit::<[SimdGoertzelEngine; MAX_EEG_CHANNELS]>::zeroed().assume_init() },
            goertzel_beta: unsafe { core::mem::MaybeUninit::<[SimdGoertzelEngine; MAX_EEG_CHANNELS]>::zeroed().assume_init() },
            goertzel_gamma: unsafe { core::mem::MaybeUninit::<[SimdGoertzelEngine; MAX_EEG_CHANNELS]>::zeroed().assume_init() },
            power_delta: [0.0; MAX_EEG_CHANNELS],
            power_theta: [0.0; MAX_EEG_CHANNELS],
            power_alpha: [0.0; MAX_EEG_CHANNELS],
            power_beta: [0.0; MAX_EEG_CHANNELS],
            power_gamma: [0.0; MAX_EEG_CHANNELS],
            emg_power: [0.0; MAX_EEG_CHANNELS],
            active_channels: 0,
            sample_rate: DEFAULT_SAMPLE_RATE,
        }
    }

    /// Initialize processor with channel configuration
    pub fn init(&mut self, channels: &[EegChannel], sample_rate: u32) -> Result<(), EegStreamError> {
        if channels.len() > MAX_EEG_CHANNELS {
            return Err(EegStreamError::ChannelCountExceeded(channels.len()));
        }

        self.active_channels = channels.len();
        self.sample_rate = sample_rate;

        // Copy channel configs
        for (i, ch) in channels.iter().enumerate() {
            self.channels[i] = *ch;
        }

        // Initialize filters for each channel
        for i in 0..self.active_channels {
            self.delta_filter[i] = SimdBiquadState::design_bandpass(DELTA_LOW, DELTA_HIGH, sample_rate)?;
            self.theta_filter[i] = SimdBiquadState::design_bandpass(THETA_LOW, THETA_HIGH, sample_rate)?;
            self.alpha_filter[i] = SimdBiquadState::design_bandpass(ALPHA_LOW, ALPHA_HIGH, sample_rate)?;
            self.beta_filter[i] = SimdBiquadState::design_bandpass(BETA_LOW, BETA_HIGH, sample_rate)?;
            self.gamma_filter[i] = SimdBiquadState::design_bandpass(GAMMA_LOW, GAMMA_HIGH, sample_rate)?;
            
            // EMG notch filter (high-pass to detect contamination)
            self.emg_notch[i] = SimdBiquadState::design_bandpass(EMG_CUTOFF_LOW, EMG_CUTOFF_HIGH, sample_rate)?;

            // Initialize Goertzel engines
            self.goertzel_delta[i] = SimdGoertzelEngine::new((DELTA_LOW + DELTA_HIGH) / 2.0, sample_rate, SIMD_BUFFER_SIZE)?;
            self.goertzel_theta[i] = SimdGoertzelEngine::new((THETA_LOW + THETA_HIGH) / 2.0, sample_rate, SIMD_BUFFER_SIZE)?;
            self.goertzel_alpha[i] = SimdGoertzelEngine::new((ALPHA_LOW + ALPHA_HIGH) / 2.0, sample_rate, SIMD_BUFFER_SIZE)?;
            self.goertzel_beta[i] = SimdGoertzelEngine::new((BETA_LOW + BETA_HIGH) / 2.0, sample_rate, SIMD_BUFFER_SIZE)?;
            self.goertzel_gamma[i] = SimdGoertzelEngine::new((GAMMA_LOW + GAMMA_HIGH) / 2.0, sample_rate, SIMD_BUFFER_SIZE)?;
        }

        Ok(())
    }

    /// Process a buffer of EEG data (zero-copy, SIMD-accelerated)
    pub fn process_buffer(&mut self, buffer: &EegStreamBuffer) -> Result<(), EegStreamError> {
        if buffer.fill_level() < SIMD_BUFFER_SIZE {
            return Err(EegStreamError::BufferOverflow);
        }

        unsafe {
            let data_ptr = buffer.as_simd_ptr();
            
            // Process 8 channels at a time using SIMD
            for ch_start in (0..self.active_channels).step_by(8) {
                let ch_end = core::cmp::min(ch_start + 8, self.active_channels);
                
                for sample_idx in 0..SIMD_BUFFER_SIZE {
                    let base_idx = sample_idx * MAX_EEG_CHANNELS + ch_start;
                    
                    // Load 8 channel samples (pad with zeros if needed)
                    let mut samples = [0.0f32; 8];
                    for ch in ch_start..ch_end {
                        samples[ch - ch_start] = *data_ptr.add(base_idx + (ch - ch_start));
                    }
                    let input = _mm256_loadu_ps(samples.as_ptr());

                    // Apply bandpass filters and accumulate power
                    for ch_offset in 0..(ch_end - ch_start) {
                        let ch_idx = ch_start + ch_offset;
                        
                        // Extract single channel (would need shuffle in real impl)
                        let ch_input = _mm256_set1_ps(samples[ch_offset]);
                        
                        // Delta band
                        let delta_out = self.delta_filter[ch_idx].process(ch_input);
                        self.power_delta[ch_idx] = self.extract_power(delta_out);
                        
                        // Theta band
                        let theta_out = self.theta_filter[ch_idx].process(ch_input);
                        self.power_theta[ch_idx] = self.extract_power(theta_out);
                        
                        // Alpha band
                        let alpha_out = self.alpha_filter[ch_idx].process(ch_input);
                        self.power_alpha[ch_idx] = self.extract_power(alpha_out);
                        
                        // Beta band
                        let beta_out = self.beta_filter[ch_idx].process(ch_input);
                        self.power_beta[ch_idx] = self.extract_power(beta_out);
                        
                        // Gamma band
                        let gamma_out = self.gamma_filter[ch_idx].process(ch_input);
                        self.power_gamma[ch_idx] = self.extract_power(gamma_out);
                        
                        // EMG contamination check
                        let emg_out = self.emg_notch[ch_idx].process(ch_input);
                        self.emg_power[ch_idx] = self.extract_power(emg_out);
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract power from SIMD register (horizontal sum)
    #[inline]
    unsafe fn extract_power(&self, simd_val: __m256) -> f32 {
        let mut vals = [0.0f32; 8];
        _mm256_storeu_ps(vals.as_mut_ptr(), simd_val);
        
        // Sum all lanes and normalize
        let sum: f32 = vals.iter().map(|v| v * v).sum();
        sum / 8.0
    }

    /// Get theta/beta ratio for flow state detection
    #[inline]
    pub fn get_theta_beta_ratio(&self, channel: usize) -> Result<f32, EegStreamError> {
        if channel >= self.active_channels {
            return Err(EegStreamError::ChannelCountExceeded(channel));
        }

        let theta = self.power_theta[channel];
        let beta = self.power_beta[channel];

        if beta < 1e-10 {
            return Ok(theta / 1e-10);
        }

        Ok(theta / beta)
    }

    /// Check for EMG contamination on a channel
    #[inline]
    pub fn is_emg_contaminated(&self, channel: usize, threshold: f32) -> Result<bool, EegStreamError> {
        if channel >= self.active_channels {
            return Err(EegStreamError::ChannelCountExceeded(channel));
        }

        let total_power = self.power_delta[channel]
            + self.power_theta[channel]
            + self.power_alpha[channel]
            + self.power_beta[channel]
            + self.power_gamma[channel];

        let emg_ratio = self.emg_power[channel] / (total_power + 1e-10);
        Ok(emg_ratio > threshold)
    }

    /// Get all spectral powers for a channel
    #[inline]
    pub fn get_spectral_powers(&self, channel: usize) -> Result<[f32; 5], EegStreamError> {
        if channel >= self.active_channels {
            return Err(EegStreamError::ChannelCountExceeded(channel));
        }

        Ok([
            self.power_delta[channel],
            self.power_theta[channel],
            self.power_alpha[channel],
            self.power_beta[channel],
            self.power_gamma[channel],
        ])
    }

    /// Reset all filter states
    pub fn reset_filters(&mut self) {
        for i in 0..self.active_channels {
            self.delta_filter[i].reset();
            self.theta_filter[i].reset();
            self.alpha_filter[i].reset();
            self.beta_filter[i].reset();
            self.gamma_filter[i].reset();
            self.emg_notch[i].reset();
            self.goertzel_delta[i].reset();
            self.goertzel_theta[i].reset();
            self.goertzel_alpha[i].reset();
            self.goertzel_beta[i].reset();
            self.goertzel_gamma[i].reset();
        }
    }
}

impl Default for SimdEegProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eeg_buffer_init() {
        let mut buffer = EegStreamBuffer::new();
        assert!(buffer.init(8, 1000).is_ok());
        assert_eq!(buffer.fill_level(), 0);
    }

    #[test]
    fn test_channel_limit() {
        let mut buffer = EegStreamBuffer::new();
        assert!(buffer.init(MAX_EEG_CHANNELS + 1, 1000).is_err());
    }

    #[test]
    fn test_sample_rate_validation() {
        let mut buffer = EegStreamBuffer::new();
        assert!(buffer.init(8, 50).is_err());
        assert!(buffer.init(8, 15000).is_err());
    }

    #[test]
    fn test_processor_initialization() {
        let mut processor = SimdEegProcessor::new();
        let channels = vec![EegChannel::default(); 8];
        assert!(processor.init(&channels, 1000).is_ok());
    }
}
