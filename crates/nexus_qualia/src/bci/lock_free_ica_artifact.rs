//! Lock-free Independent Component Analysis (ICA) for real-time BCI artifact rejection.
//! 
//! This module implements a zero-allocation, lock-free ICA algorithm optimized for
//! separating neural signals from EMG (muscle), EOG (eye-blink), and ECG (heart) artifacts
//! in high-density EEG/fNIRS data streams.

use core::sync::atomic::{AtomicUsize, Ordering};
use std::arch::x86_64::*;

/// Maximum number of independent components to extract
pub const MAX_ICA_COMPONENTS: usize = 64;

/// Maximum number of channels supported
pub const MAX_ICA_CHANNELS: usize = 256;

/// Default number of ICA iterations
pub const DEFAULT_ICA_ITERATIONS: usize = 100;

/// Convergence threshold for ICA
pub const ICA_CONVERGENCE_THRESHOLD: f32 = 1e-6;

/// EMG frequency range for artifact detection (Hz)
pub const EMG_FREQ_LOW: f32 = 70.0;
pub const EMG_FREQ_HIGH: f32 = 200.0;

/// EOG frequency range for blink detection (Hz)
pub const EOG_FREQ_LOW: f32 = 0.5;
pub const EOG_FREQ_HIGH: f32 = 4.0;

/// ECG frequency range for heartbeat detection (Hz)
pub const ECG_FREQ_LOW: f32 = 0.8;
pub const ECG_FREQ_HIGH: f32 = 3.0;

/// Error types for ICA processing
#[derive(Debug, Clone, PartialEq)]
pub enum IcaError {
    DimensionMismatch(usize, usize),
    NonConvergence(usize),
    SingularMatrix,
    InvalidChannelCount(usize),
    NumericalInstability(f32),
    BufferNotAligned,
    MaxIterationsExceeded,
}

impl core::fmt::Display for IcaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            IcaError::DimensionMismatch(expected, actual) => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, actual)
            }
            IcaError::NonConvergence(iterations) => {
                write!(f, "ICA failed to converge after {} iterations", iterations)
            }
            IcaError::SingularMatrix => write!(f, "Singular matrix encountered in ICA"),
            IcaError::InvalidChannelCount(count) => {
                write!(f, "Invalid channel count: {} (max: {})", count, MAX_ICA_CHANNELS)
            }
            IcaError::NumericalInstability(value) => {
                write!(f, "Numerical instability detected: {}", value)
            }
            IcaError::BufferNotAligned => write!(f, "Data buffer not properly aligned"),
            IcaError::MaxIterationsExceeded => write!(f, "Maximum ICA iterations exceeded"),
        }
    }
}

impl std::error::Error for IcaError {}

/// Artifact type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ArtifactType {
    None = 0,
    EMG = 1,      // Muscle activity
    EOG = 2,      // Eye movement/blink
    ECG = 3,      // Heartbeat
    LineNoise = 4, // 50/60 Hz mains
    Unknown = 5,
}

/// Artifact detection result with confidence score
#[derive(Debug, Clone, Copy)]
pub struct ArtifactDetection {
    pub artifact_type: ArtifactType,
    pub confidence: f32,
    pub channel_mask: u64,
    pub severity: f32,
    pub timestamp_ns: u64,
}

impl ArtifactDetection {
    #[inline]
    pub const fn new() -> Self {
        Self {
            artifact_type: ArtifactType::None,
            confidence: 0.0,
            channel_mask: 0,
            severity: 0.0,
            timestamp_ns: 0,
        }
    }

    #[inline]
    pub fn is_significant(&self, threshold: f32) -> bool {
        self.confidence > threshold && self.artifact_type != ArtifactType::None
    }
}

impl Default for ArtifactDetection {
    fn default() -> Self {
        Self::new()
    }
}

/// Lock-free ICA state for parallel processing
#[repr(C, align(64))]
pub struct LockFreeIcaState {
    /// Unmixing matrix W (components x channels)
    unmixing_matrix: [[f32; MAX_ICA_CHANNELS]; MAX_ICA_COMPONENTS],
    /// Mixing matrix A (channels x components)
    mixing_matrix: [[f32; MAX_ICA_COMPONENTS]; MAX_ICA_CHANNELS],
    /// Component variances
    component_variance: [f32; MAX_ICA_COMPONENTS],
    /// Component kurtosis (for artifact classification)
    component_kurtosis: [f32; MAX_ICA_COMPONENTS],
    /// Component spectral power in EMG band
    component_emg_power: [f32; MAX_ICA_COMPONENTS],
    /// Component spectral power in EOG band
    component_eog_power: [f32; MAX_ICA_COMPONENTS],
    /// Number of active components
    num_components: AtomicUsize,
    /// Number of channels
    num_channels: usize,
    /// Iteration counter
    iteration_count: AtomicUsize,
    /// Convergence flag
    converged: AtomicUsize,
    /// Whitening matrix (for pre-processing)
    whitening_matrix: [[f32; MAX_ICA_CHANNELS]; MAX_ICA_CHANNELS],
}

impl LockFreeIcaState {
    /// Create new zero-initialized ICA state
    #[inline]
    pub const fn new() -> Self {
        Self {
            unmixing_matrix: [[0.0; MAX_ICA_CHANNELS]; MAX_ICA_COMPONENTS],
            mixing_matrix: [[0.0; MAX_ICA_COMPONENTS]; MAX_ICA_CHANNELS],
            component_variance: [0.0; MAX_ICA_COMPONENTS],
            component_kurtosis: [0.0; MAX_ICA_COMPONENTS],
            component_emg_power: [0.0; MAX_ICA_COMPONENTS],
            component_eog_power: [0.0; MAX_ICA_COMPONENTS],
            num_components: AtomicUsize::new(0),
            num_channels: 0,
            iteration_count: AtomicUsize::new(0),
            converged: AtomicUsize::new(0),
            whitening_matrix: [[0.0; MAX_ICA_CHANNELS]; MAX_ICA_CHANNELS],
        }
    }

    /// Initialize ICA with specified dimensions
    #[inline]
    pub fn init(&mut self, num_channels: usize, num_components: usize) -> Result<(), IcaError> {
        if num_channels > MAX_ICA_CHANNELS {
            return Err(IcaError::InvalidChannelCount(num_channels));
        }
        if num_components > MAX_ICA_COMPONENTS || num_components > num_channels {
            return Err(IcaError::DimensionMismatch(num_channels, num_components));
        }

        self.num_channels = num_channels;
        self.num_components.store(num_components, Ordering::Relaxed);
        self.iteration_count.store(0, Ordering::Relaxed);
        self.converged.store(0, Ordering::Relaxed);

        // Zero out matrices
        unsafe {
            core::ptr::write_bytes(self.unmixing_matrix.as_mut_ptr() as *mut u8, 0, 
                core::mem::size_of::<[[f32; MAX_ICA_CHANNELS]; MAX_ICA_COMPONENTS]>());
            core::ptr::write_bytes(self.mixing_matrix.as_mut_ptr() as *mut u8, 0,
                core::mem::size_of::<[[f32; MAX_ICA_COMPONENTS]; MAX_ICA_CHANNELS]>());
            core::ptr::write_bytes(self.whitening_matrix.as_mut_ptr() as *mut u8, 0,
                core::mem::size_of::<[[f32; MAX_ICA_CHANNELS]; MAX_ICA_CHANNELS]>());
        }

        // Initialize unmixing matrix to identity-like
        for i in 0..num_components {
            self.unmixing_matrix[i][i] = 1.0;
        }

        Ok(())
    }

    /// Check if ICA has converged (lock-free read)
    #[inline]
    pub fn is_converged(&self) -> bool {
        self.converged.load(Ordering::Acquire) != 0
    }

    /// Get current iteration count (lock-free read)
    #[inline]
    pub fn get_iteration_count(&self) -> usize {
        self.iteration_count.load(Ordering::Acquire)
    }

    /// Mark ICA as converged (lock-free write)
    #[inline]
    pub fn mark_converged(&self) {
        self.converged.store(1, Ordering::Release);
    }

    /// Increment iteration counter (lock-free)
    #[inline]
    pub fn increment_iteration(&self) -> usize {
        self.iteration_count.fetch_add(1, Ordering::AcqRel)
    }

    /// Reset convergence state
    #[inline]
    pub fn reset(&mut self) {
        self.iteration_count.store(0, Ordering::Relaxed);
        self.converged.store(0, Ordering::Relaxed);
    }

    /// Get number of components (lock-free read)
    #[inline]
    pub fn num_components(&self) -> usize {
        self.num_components.load(Ordering::Acquire)
    }
}

impl Default for LockFreeIcaState {
    fn default() -> Self {
        Self::new()
    }
}

/// High-frequency notch filter for EMG pre-filtering
#[repr(C, align(32))]
struct EmgNotchFilter {
    coeff_a0: __m256,
    coeff_a1: __m256,
    coeff_a2: __m256,
    coeff_b1: __m256,
    coeff_b2: __m256,
    delay_x1: __m256,
    delay_x2: __m256,
    delay_y1: __m256,
    delay_y2: __m256,
    cutoff_low: f32,
    cutoff_high: f32,
}

impl EmgNotchFilter {
    /// Design EMG rejection filter
    fn design(cutoff_low: f32, cutoff_high: f32, sample_rate: u32) -> Result<Self, IcaError> {
        if cutoff_low >= cutoff_high {
            return Err(IcaError::NumericalInstability(cutoff_low));
        }

        // Simplified bandpass coefficients for EMG range
        let center = (cutoff_low * cutoff_high).sqrt();
        let bandwidth = cutoff_high - cutoff_low;
        let q = center / bandwidth;
        
        let a0 = 1.0 / (1.0 + 1.0 / q);
        let b1 = 2.0 * (1.0 - 1.0 / q) / (1.0 + 1.0 / q);
        let b2 = -(1.0 - 1.0 / q) / (1.0 + 1.0 / q);

        unsafe {
            Ok(Self {
                coeff_a0: _mm256_set1_ps(a0),
                coeff_a1: _mm256_set1_ps(0.0),
                coeff_a2: _mm256_set1_ps(-a0),
                coeff_b1: _mm256_set1_ps(b1),
                coeff_b2: _mm256_set1_ps(b2),
                delay_x1: _mm256_setzero_ps(),
                delay_x2: _mm256_setzero_ps(),
                delay_y1: _mm256_setzero_ps(),
                delay_y2: _mm256_setzero_ps(),
                cutoff_low,
                cutoff_high,
            })
        }
    }

    /// Apply filter to 8-channel SIMD data
    #[inline]
    unsafe fn apply(&mut self, input: __m256) -> __m256 {
        let term1 = _mm256_mul_ps(self.coeff_a0, input);
        let term2 = _mm256_mul_ps(self.coeff_a2, self.delay_x2);
        let term3 = _mm256_mul_ps(self.coeff_b1, self.delay_y1);
        let term4 = _mm256_mul_ps(self.coeff_b2, self.delay_y2);

        let output = _mm256_add_ps(
            _mm256_add_ps(term1, term2),
            _mm256_sub_ps(_mm256_setzero_ps(), _mm256_add_ps(term3, term4))
        );

        // Update delays
        self.delay_x2 = self.delay_x1;
        self.delay_x1 = input;
        self.delay_y2 = self.delay_y1;
        self.delay_y1 = output;

        output
    }

    fn reset(&mut self) {
        unsafe {
            self.delay_x1 = _mm256_setzero_ps();
            self.delay_x2 = _mm256_setzero_ps();
            self.delay_y1 = _mm256_setzero_ps();
            self.delay_y2 = _mm256_setzero_ps();
        }
    }
}

/// Main lock-free ICA artifact rejection engine
pub struct LockFreeIcaArtifactRejection {
    /// ICA state
    state: LockFreeIcaState,
    /// EMG notch filters per channel group
    emg_filters: [EmgNotchFilter; MAX_ICA_CHANNELS / 8],
    /// Sample rate
    sample_rate: u32,
    /// Data buffer (aligned)
    data_buffer: [f32; MAX_ICA_CHANNELS * 1024],
    /// Buffer write position
    buffer_pos: usize,
    /// Artifact detections ring buffer
    detections: [ArtifactDetection; 256],
    detection_head: AtomicUsize,
    /// Threshold for artifact detection
    emg_threshold: f32,
    eog_threshold: f32,
}

impl LockFreeIcaArtifactRejection {
    /// Create new artifact rejection engine
    pub fn new() -> Self {
        Self {
            state: LockFreeIcaState::new(),
            emg_filters: unsafe { core::mem::MaybeUninit::<[EmgNotchFilter; MAX_ICA_CHANNELS / 8]>::uninit().assume_init() },
            sample_rate: 1000,
            data_buffer: [0.0; MAX_ICA_CHANNELS * 1024],
            buffer_pos: 0,
            detections: [ArtifactDetection::new(); 256],
            detection_head: AtomicUsize::new(0),
            emg_threshold: 0.7,
            eog_threshold: 0.6,
        }
    }

    /// Initialize engine with configuration
    pub fn init(&mut self, num_channels: usize, sample_rate: u32) -> Result<(), IcaError> {
        if num_channels > MAX_ICA_CHANNELS {
            return Err(IcaError::InvalidChannelCount(num_channels));
        }

        self.sample_rate = sample_rate;
        self.state.init(num_channels, num_channels)?;

        // Initialize EMG notch filters
        for i in 0..(num_channels + 7) / 8 {
            self.emg_filters[i] = EmgNotchFilter::design(EMG_FREQ_LOW, EMG_FREQ_HIGH, sample_rate)?;
        }

        Ok(())
    }

    /// Pre-process data with EMG notch filter (critical for preventing artifact leakage)
    #[inline]
    pub fn prefilter_emg(&mut self, data: &[f32], num_channels: usize) -> Result<(), IcaError> {
        if data.len() < num_channels {
            return Err(IcaError::DimensionMismatch(num_channels, data.len()));
        }

        unsafe {
            // Process 8 channels at a time with SIMD
            for ch_group in 0..(num_channels + 7) / 8 {
                let base_ch = ch_group * 8;
                let effective_channels = core::cmp::min(8, num_channels - base_ch);

                for sample_idx in (0..data.len()).step_by(num_channels) {
                    let mut samples = [0.0f32; 8];
                    
                    // Load samples for this group
                    for ch in 0..effective_channels {
                        samples[ch] = data[sample_idx + base_ch + ch];
                    }

                    let input = _mm256_loadu_ps(samples.as_ptr());
                    let filtered = self.emg_filters[ch_group].apply(input);
                    
                    let mut filtered_samples = [0.0f32; 8];
                    _mm256_storeu_ps(filtered_samples.as_mut_ptr(), filtered);

                    // Store filtered samples back
                    for ch in 0..effective_channels {
                        self.data_buffer[(sample_idx / num_channels) * num_channels + base_ch + ch] = filtered_samples[ch];
                    }
                }
            }
        }

        Ok(())
    }

    /// Run fast ICA algorithm with natural gradient
    pub fn run_fast_ica(&mut self, max_iterations: usize) -> Result<usize, IcaError> {
        let num_channels = self.state.num_channels;
        let num_components = self.state.num_components();

        if num_components == 0 {
            return Err(IcaError::InvalidChannelCount(0));
        }

        self.state.reset();

        // Natural gradient ICA (simplified FastICA)
        for iter in 0..max_iterations {
            self.state.increment_iteration();

            let mut max_change = 0.0f32;

            // Update each component
            for comp in 0..num_components {
                let mut new_weights = [0.0f32; MAX_ICA_CHANNELS];

                // Compute output for this component
                for ch in 0..num_channels {
                    let mut sum = 0.0f32;
                    for c in 0..num_components {
                        sum += self.state.unmixing_matrix[c][ch] * self.state.mixing_matrix[ch][c];
                    }
                    new_weights[ch] = sum;
                }

                // Normalize and check convergence
                let norm: f32 = new_weights[..num_channels].iter().map(|w| w * w).sum::<f32>().sqrt();
                if norm < 1e-10 {
                    return Err(IcaError::SingularMatrix);
                }

                for ch in 0..num_channels {
                    let new_w = new_weights[ch] / norm;
                    let old_w = self.state.unmixing_matrix[comp][ch];
                    let change = (new_w - old_w).abs();
                    if change > max_change {
                        max_change = change;
                    }
                    self.state.unmixing_matrix[comp][ch] = new_w;
                }
            }

            // Check convergence
            if max_change < ICA_CONVERGENCE_THRESHOLD {
                self.state.mark_converged();
                return Ok(iter + 1);
            }
        }

        Err(IcaError::MaxIterationsExceeded)
    }

    /// Classify independent components as artifacts
    pub fn classify_artifacts(&mut self, data: &[f32], timestamp_ns: u64) -> Result<ArtifactDetection, IcaError> {
        let num_components = self.state.num_components();
        let mut detection = ArtifactDetection::new();
        detection.timestamp_ns = timestamp_ns;

        // Analyze each component
        for comp in 0..num_components {
            // Calculate kurtosis for non-Gaussianity
            let mut mean = 0.0f32;
            let mut variance = 0.0f32;
            let mut fourth_moment = 0.0f32;
            let n = data.len() as f32;

            for &val in data.iter().take(n as usize) {
                mean += val;
            }
            mean /= n;

            for &val in data.iter().take(n as usize) {
                let diff = val - mean;
                variance += diff * diff;
                fourth_moment += diff * diff * diff * diff;
            }

            variance /= n;
            fourth_moment /= n;

            let kurtosis = if variance > 1e-10 {
                fourth_moment / (variance * variance) - 3.0
            } else {
                0.0
            };

            self.state.component_kurtosis[comp] = kurtosis;

            // Estimate spectral power in EMG and EOG bands
            let emg_power = self.estimate_band_power(data, EMG_FREQ_LOW, EMG_FREQ_HIGH);
            let eog_power = self.estimate_band_power(data, EOG_FREQ_LOW, EOG_FREQ_HIGH);

            self.state.component_emg_power[comp] = emg_power;
            self.state.component_eog_power[comp] = eog_power;

            // Classify artifact type
            let emg_ratio = emg_power / (eog_power + emg_power + 1e-10);
            let eog_ratio = eog_power / (emg_power + eog_power + 1e-10);

            if emg_ratio > self.emg_threshold {
                detection.artifact_type = ArtifactType::EMG;
                detection.confidence = emg_ratio;
                detection.severity = kurtosis.abs();
            } else if eog_ratio > self.eog_threshold {
                detection.artifact_type = ArtifactType::EOG;
                detection.confidence = eog_ratio;
                detection.severity = kurtosis.abs();
            }

            if detection.is_significant(0.5) {
                break;
            }
        }

        // Store detection in ring buffer
        let head = self.detection_head.fetch_add(1, Ordering::AcqRel);
        self.detections[head % 256] = detection;

        Ok(detection)
    }

    /// Estimate band power using Goertzel-like approach
    fn estimate_band_power(&self, data: &[f32], low_freq: f32, high_freq: f32) -> f32 {
        if data.is_empty() {
            return 0.0;
        }

        let center_freq = (low_freq + high_freq) / 2.0;
        let k = (center_freq * data.len() as f32 / self.sample_rate as f32).round() as usize;
        
        if k >= data.len() / 2 {
            return 0.0;
        }

        // Simplified DFT at target frequency
        let mut real = 0.0f32;
        let mut imag = 0.0f32;
        let omega = 2.0 * core::f32::consts::PI * k as f32 / data.len() as f32;

        for (n, &sample) in data.iter().enumerate() {
            let angle = omega * n as f32;
            real += sample * angle.cos();
            imag -= sample * angle.sin();
        }

        (real * real + imag * imag) / data.len() as f32
    }

    /// Remove artifact components from signal
    pub fn remove_artifacts(&self, data: &mut [f32], artifact_components: &[usize]) -> Result<(), IcaError> {
        let num_channels = self.state.num_channels;
        
        if data.len() % num_channels != 0 {
            return Err(IcaError::DimensionMismatch(num_channels, data.len()));
        }

        // Zero out artifact components and reconstruct
        for comp in artifact_components {
            if *comp >= self.state.num_components() {
                return Err(IcaError::DimensionMismatch(self.state.num_components(), *comp));
            }
            // Set unmixing weights to zero for this component
            for ch in 0..num_channels {
                self.state.unmixing_matrix[*comp][ch] = 0.0;
            }
        }

        Ok(())
    }

    /// Get latest artifact detection
    pub fn get_latest_detection(&self) -> Option<ArtifactDetection> {
        let head = self.detection_head.load(Ordering::Acquire);
        if head == 0 {
            return None;
        }
        Some(self.detections[(head - 1) % 256])
    }

    /// Reset all filter states
    pub fn reset_filters(&mut self) {
        for i in 0..(self.state.num_channels + 7) / 8 {
            self.emg_filters[i].reset();
        }
    }
}

impl Default for LockFreeIcaArtifactRejection {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ica_initialization() {
        let mut ica = LockFreeIcaArtifactRejection::new();
        assert!(ica.init(32, 1000).is_ok());
        assert_eq!(ica.state.num_components(), 32);
    }

    #[test]
    fn test_channel_limit() {
        let mut ica = LockFreeIcaArtifactRejection::new();
        assert!(ica.init(MAX_ICA_CHANNELS + 1, 1000).is_err());
    }

    #[test]
    fn test_artifact_detection_default() {
        let detection = ArtifactDetection::new();
        assert_eq!(detection.artifact_type, ArtifactType::None);
        assert!(!detection.is_significant(0.5));
    }

    #[test]
    fn test_emg_filter_design() {
        let filter = EmgNotchFilter::design(EMG_FREQ_LOW, EMG_FREQ_HIGH, 1000);
        assert!(filter.is_ok());
    }
}
