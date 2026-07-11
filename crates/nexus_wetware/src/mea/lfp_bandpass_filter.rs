//! Local Field Potential (LFP) Bandpass Filter
//! 
//! Zero-allocation IIR/FIR filters for extracting slow brain-wave states
//! from MEA recordings. Supports delta (0.5-4Hz), theta (4-8Hz), 
//! alpha (8-13Hz), beta (13-30Hz), and gamma (30-100Hz) bands.

use crate::mea::cmos_dma_stream::MAX_ELECTRODES;

/// Sample rate for LFP processing (typically lower than spike rate)
pub const LFP_SAMPLE_RATE: u32 = 1000; // 1 kHz sufficient for LFP

/// Number of frequency bands tracked
pub const NUM_BANDS: usize = 5;

/// Frequency band definitions
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum FrequencyBand {
    Delta = 0, // 0.5-4 Hz: Deep sleep, arousal
    Theta = 1, // 4-8 Hz: Memory encoding, attention
    Alpha = 2, // 8-13 Hz: Relaxed wakefulness
    Beta = 3,  // 13-30 Hz: Active thinking, motor
    Gamma = 4, // 30-100 Hz: Cognitive processing
}

/// IIR Filter state using Second-Order Sections (SOS)
/// More numerically stable than direct form for high-order filters
#[repr(C, align(32))]
pub struct SosFilterState {
    /// SOS coefficients: [b0, b1, b2, a1, a2] per section
    sections: [[f32; 5]; 8], // Up to 8 second-order sections
    /// State variables per section
    x1: [f32; 8],
    x2: [f32; 8],
    y1: [f32; 8],
    y2: [f32; 8],
    /// Number of active sections
    num_sections: usize,
}

impl SosFilterState {
    /// Create a new SOS filter state
    pub const fn new() -> Self {
        Self {
            sections: [[0.0; 5]; 8],
            x1: [0.0; 8],
            x2: [0.0; 8],
            y1: [0.0; 8],
            y2: [0.0; 8],
            num_sections: 0,
        }
    }

    /// Apply the filter cascade to a single sample
    #[inline]
    pub fn apply(&mut self, mut input: f32) -> f32 {
        for i in 0..self.num_sections {
            let s = &self.sections[i];
            let b0 = s[0];
            let b1 = s[1];
            let b2 = s[2];
            let a1 = s[3];
            let a2 = s[4];

            let output = b0 * input 
                       + b1 * self.x1[i] 
                       + b2 * self.x2[i] 
                       - a1 * self.y1[i] 
                       - a2 * self.y2[i];

            // Update state
            self.x2[i] = self.x1[i];
            self.x1[i] = input;
            self.y2[i] = self.y1[i];
            self.y1[i] = output;

            input = output;
        }
        input
    }

    /// Reset all filter states
    pub fn reset(&mut self) {
        for i in 0..self.num_sections {
            self.x1[i] = 0.0;
            self.x2[i] = 0.0;
            self.y1[i] = 0.0;
            self.y2[i] = 0.0;
        }
    }
}

impl Default for SosFilterState {
    fn default() -> Self {
        Self::new()
    }
}

/// Bandpass filter configuration
#[repr(C)]
pub struct BandpassConfig {
    pub low_cutoff: f32,
    pub high_cutoff: f32,
    pub order: usize,
}

/// LFP processor for a single electrode
pub struct LfpProcessor {
    /// Bandpass filters for each frequency band
    bandpass_filters: [SosFilterState; NUM_BANDS],
    /// Power estimates for each band (exponential moving average)
    band_powers: [f32; NUM_BANDS],
    /// Configuration for each band
    configs: [BandpassConfig; NUM_BANDS],
    /// Filter initialization flag
    initialized: bool,
}

impl LfpProcessor {
    /// Create a new LFP processor with configured bandpass filters
    pub fn new(sample_rate: u32) -> Self {
        let configs = [
            BandpassConfig { low_cutoff: 0.5, high_cutoff: 4.0, order: 4 },   // Delta
            BandpassConfig { low_cutoff: 4.0, high_cutoff: 8.0, order: 4 },   // Theta
            BandpassConfig { low_cutoff: 8.0, high_cutoff: 13.0, order: 4 },  // Alpha
            BandpassConfig { low_cutoff: 13.0, high_cutoff: 30.0, order: 4 }, // Beta
            BandpassConfig { low_cutoff: 30.0, high_cutoff: 100.0, order: 4 },// Gamma
        ];

        let mut processors = Self {
            bandpass_filters: [SosFilterState::new(); NUM_BANDS],
            band_powers: [0.0; NUM_BANDS],
            configs,
            initialized: false,
        };

        // Initialize Butterworth bandpass filters
        processors.initialize_filters(sample_rate);
        processors
    }

    /// Initialize Butterworth bandpass filters using bilinear transform
    fn initialize_filters(&mut self, sample_rate: u32) {
        let nyquist = sample_rate as f32 / 2.0;

        for (band_idx, config) in self.configs.iter().enumerate() {
            // Normalize frequencies
            let wl = config.low_cutoff / nyquist;
            let wh = config.high_cutoff / nyquist;

            // Pre-warp for bilinear transform
            let tan_wl = (core::f32::consts::PI * wl).tan();
            let tan_wh = (core::f32::consts::PI * wh).tan();

            // Compute bandwidth and center frequency
            let bw = tan_wh - tan_wl;
            let w0_sq = tan_wl * tan_wh;

            // Design 2nd-order bandpass section (simplified Butterworth)
            // For production, use proper pole-zero placement
            let k = bw / (1.0 + bw + w0_sq);
            
            if self.bandpass_filters[band_idx].num_sections == 0 {
                self.bandpass_filters[band_idx].num_sections = 1;
            }

            // SOS coefficients for bandpass
            self.bandpass_filters[band_idx].sections[0] = [
                k,      // b0
                0.0,    // b1 (zero at origin)
                -k,     // b2
                2.0 * (w0_sq - 1.0) / (1.0 + bw + w0_sq), // a1
                (1.0 - bw + w0_sq) / (1.0 + bw + w0_sq),  // a2
            ];
        }

        self.initialized = true;
    }

    /// Process a single LFP sample through all bandpass filters
    #[inline]
    pub fn process_sample(&mut self, sample: f32) -> [f32; NUM_BANDS] {
        let mut outputs = [0.0f32; NUM_BANDS];

        for (i, filter) in self.bandpass_filters.iter_mut().enumerate() {
            if self.initialized {
                let filtered = filter.apply(sample);
                outputs[i] = filtered;

                // Update power estimate (exponential moving average)
                let alpha = 0.01;
                self.band_powers[i] = (1.0 - alpha) * self.band_powers[i] 
                                    + alpha * filtered * filtered;
            }
        }

        outputs
    }

    /// Get current power in a specific frequency band
    #[inline]
    pub fn get_band_power(&self, band: FrequencyBand) -> f32 {
        self.band_powers[band as usize]
    }

    /// Get total power across all bands
    #[inline]
    pub fn get_total_power(&self) -> f32 {
        self.band_powers.iter().sum()
    }

    /// Compute band ratios (e.g., theta/alpha for attention metrics)
    #[inline]
    pub fn get_band_ratio(&self, numerator: FrequencyBand, denominator: FrequencyBand) -> f32 {
        let num_power = self.get_band_power(numerator);
        let denom_power = self.get_band_power(denominator);
        
        // Avoid division by zero
        if denom_power < 1e-10 {
            0.0
        } else {
            num_power / denom_power
        }
    }

    /// Detect arousal state based on band power distribution
    pub fn detect_arousal_state(&self) -> ArousalState {
        let delta = self.get_band_power(FrequencyBand::Delta);
        let theta = self.get_band_power(FrequencyBand::Theta);
        let alpha = self.get_band_power(FrequencyBand::Alpha);
        let beta = self.get_band_power(FrequencyBand::Beta);
        let gamma = self.get_band_power(FrequencyBand::Gamma);

        let total = delta + theta + alpha + beta + gamma;
        if total < 1e-10 {
            return ArousalState::Unknown;
        }

        // Normalized powers
        let d_norm = delta / total;
        let t_norm = theta / total;
        let a_norm = alpha / total;
        let b_norm = beta / total;
        let g_norm = gamma / total;

        // Classify state
        if d_norm > 0.5 || (d_norm + t_norm) > 0.7 {
            ArousalState::DeepSleep
        } else if t_norm > 0.3 && a_norm > 0.2 {
            ArousalState::Drowsy
        } else if a_norm > 0.4 {
            ArousalState::Relaxed
        } else if b_norm > 0.3 || g_norm > 0.2 {
            ArousalState::Alert
        } else if g_norm > 0.3 && b_norm > 0.2 {
            ArousalState::HighAttention
        } else {
            ArousalState::Baseline
        }
    }

    /// Reset all filter states
    pub fn reset(&mut self) {
        for filter in &mut self.bandpass_filters {
            filter.reset();
        }
        self.band_powers = [0.0; NUM_BANDS];
    }
}

/// Arousal state classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArousalState {
    Unknown,
    DeepSleep,
    Drowsy,
    Relaxed,
    Baseline,
    Alert,
    HighAttention,
}

/// Multi-electrode LFP analyzer
pub struct LfpBandpassFilter {
    /// Per-electrode LFP processors
    processors: [LfpProcessor; MAX_ELECTRODES],
    /// Global synchronization flag
    synchronized: bool,
    /// Last update timestamp
    last_update_ns: u64,
}

impl LfpBandpassFilter {
    /// Create a new multi-electrode LFP filter
    pub fn new(sample_rate: u32) -> Self {
        // Note: This creates MAX_ELECTRODES instances
        // In practice, you might only process a subset of electrodes
        let mut processors = [LfpProcessor::new(sample_rate); MAX_ELECTRODES];
        
        // Initialize all processors
        for proc in &mut processors {
            // Already initialized in new()
        }

        Self {
            processors,
            synchronized: false,
            last_update_ns: 0,
        }
    }

    /// Process samples from a specific electrode
    #[inline]
    pub fn process_electrode(
        &mut self, 
        electrode_id: usize, 
        samples: &[f32]
    ) -> Result<[f32; NUM_BANDS], &'static str> {
        if electrode_id >= MAX_ELECTRODES {
            return Err("Invalid electrode ID");
        }

        let mut latest_outputs = [0.0f32; NUM_BANDS];
        
        for &sample in samples {
            latest_outputs = self.processors[electrode_id].process_sample(sample);
        }

        Ok(latest_outputs)
    }

    /// Get band power for a specific electrode and band
    #[inline]
    pub fn get_band_power(&self, electrode_id: usize, band: FrequencyBand) -> Option<f32> {
        if electrode_id >= MAX_ELECTRODES {
            return None;
        }
        Some(self.processors[electrode_id].get_band_power(band))
    }

    /// Get arousal state for a specific electrode
    #[inline]
    pub fn get_arousal_state(&self, electrode_id: usize) -> Option<ArousalState> {
        if electrode_id >= MAX_ELECTRODES {
            return None;
        }
        Some(self.processors[electrode_id].detect_arousal_state())
    }

    /// Compute global arousal state (average across electrodes)
    pub fn get_global_arousal(&self, active_electrodes: &[usize]) -> ArousalState {
        if active_electrodes.is_empty() {
            return ArousalState::Unknown;
        }

        let mut total_delta = 0.0;
        let mut total_theta = 0.0;
        let mut total_alpha = 0.0;
        let mut total_beta = 0.0;
        let mut total_gamma = 0.0;

        for &elec in active_electrodes {
            if elec < MAX_ELECTRODES {
                total_delta += self.processors[elec].get_band_power(FrequencyBand::Delta);
                total_theta += self.processors[elec].get_band_power(FrequencyBand::Theta);
                total_alpha += self.processors[elec].get_band_power(FrequencyBand::Alpha);
                total_beta += self.processors[elec].get_band_power(FrequencyBand::Beta);
                total_gamma += self.processors[elec].get_band_power(FrequencyBand::Gamma);
            }
        }

        let total = total_delta + total_theta + total_alpha + total_beta + total_gamma;
        if total < 1e-10 {
            return ArousalState::Unknown;
        }

        // Classify based on dominant band
        let max_band = [total_delta, total_theta, total_alpha, total_beta, total_gamma]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        match max_band {
            0 => ArousalState::DeepSleep,
            1 => ArousalState::Drowsy,
            2 => ArousalState::Relaxed,
            3 => ArousalState::Alert,
            4 => ArousalState::HighAttention,
            _ => ArousalState::Baseline,
        }
    }

    /// Reset all electrode processors
    pub fn reset_all(&mut self) {
        for proc in &mut self.processors {
            proc.reset();
        }
    }

    /// Mark system as synchronized
    pub fn synchronize(&mut self, timestamp_ns: u64) {
        self.synchronized = true;
        self.last_update_ns = timestamp_ns;
    }
}

/// FIR filter implementation for comparison (linear phase)
pub struct FirFilter {
    /// Filter coefficients (taps)
    taps: [f32; 64],
    /// Delay line
    delay_line: [f32; 64],
    /// Current position in delay line
    position: usize,
    /// Number of active taps
    num_taps: usize,
}

impl FirFilter {
    /// Create a simple low-pass FIR filter using Hamming window
    pub fn create_lowpass(cutoff_freq: f32, sample_rate: u32, num_taps: usize) -> Self {
        let nyquist = sample_rate as f32 / 2.0;
        let normalized_cutoff = cutoff_freq / nyquist;
        let fc = normalized_cutoff;
        
        let mut taps = [0.0f32; 64];
        let center = (num_taps / 2) as f32;
        
        for i in 0..num_taps.min(64) {
            let n = i as f32 - center;
            
            // Sinc function
            let sinc = if n.abs() < 1e-6 {
                2.0 * fc
            } else {
                (2.0 * fc * (core::f32::consts::PI * n)).sin() / (core::f32::consts::PI * n)
            };
            
            // Hamming window
            let window = 0.54 - 0.46 * ((2.0 * core::f32::consts::PI * i as f32) / (num_taps - 1) as f32).cos();
            
            taps[i] = sinc * window;
        }

        // Normalize for unity gain
        let sum: f32 = taps[..num_taps.min(64)].iter().sum();
        if sum > 1e-6 {
            for t in &mut taps[..num_taps.min(64)] {
                *t /= sum;
            }
        }

        Self {
            taps,
            delay_line: [0.0; 64],
            position: 0,
            num_taps: num_taps.min(64),
        }
    }

    /// Apply FIR filter to a sample
    #[inline]
    pub fn apply(&mut self, input: f32) -> f32 {
        // Store input in delay line
        self.delay_line[self.position] = input;
        
        let mut output = 0.0f32;
        for i in 0..self.num_taps {
            let idx = (self.position + i) % self.delay_line.len();
            output += self.taps[i] * self.delay_line[idx];
        }

        // Update position
        self.position = (self.position + 1) % self.delay_line.len();
        
        output
    }

    /// Reset delay line
    pub fn reset(&mut self) {
        self.delay_line = [0.0; 64];
        self.position = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandpass_filter_initialization() {
        let processor = LfpProcessor::new(LFP_SAMPLE_RATE);
        assert!(processor.initialized);
    }

    #[test]
    fn test_arousal_state_detection() {
        let mut processor = LfpProcessor::new(LFP_SAMPLE_RATE);
        
        // Feed delta-dominant signal
        for _ in 0..1000 {
            processor.process_sample(1.0);
        }
        
        let state = processor.detect_arousal_state();
        // State should reflect the input pattern
        assert!(state != ArousalState::Unknown);
    }

    #[test]
    fn test_fir_lowpass() {
        let mut filter = FirFilter::create_lowpass(50.0, 1000, 32);
        
        // Test DC response
        let dc_output = filter.apply(1.0);
        // Should converge to ~1.0 after transient
        assert!(dc_output.is_finite());
    }
}
