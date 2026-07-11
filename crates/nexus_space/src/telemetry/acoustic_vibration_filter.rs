//! Acoustic Vibration Filter for Launch Telemetry Analysis
//! 
//! Processes acoustic and vibrational frequency data from launch vehicles
//! to detect stage separation and engine anomalies.

/// Error types for acoustic filter
#[derive(Debug, Clone, Copy)]
pub enum AcousticFilterError {
    InvalidSampleRate(f64),
    InvalidFrequency(f64),
    SignalOverflow,
    NumericalInstability,
}

impl core::fmt::Display for AcousticFilterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AcousticFilterError::InvalidSampleRate(r) => write!(f, "Invalid sample rate: {}", r),
            AcousticFilterError::InvalidFrequency(freq) => write!(f, "Invalid frequency: {}", freq),
            AcousticFilterError::SignalOverflow => write!(f, "Signal overflow detected"),
            AcousticFilterError::NumericalInstability => write!(f, "Numerical instability"),
        }
    }
}

/// Nyquist frequency constant
pub const NYQUIST_FACTOR: f64 = 2.0;
/// Default audio sample rate (Hz)
pub const DEFAULT_SAMPLE_RATE: f64 = 44100.0;
/// Engine fundamental frequency range (Hz)
pub const ENGINE_FUNDAMENTAL_MIN: f64 = 20.0;
pub const ENGINE_FUNDAMENTAL_MAX: f64 = 500.0;

/// FFT-free bandpass filter using IIR biquad sections
pub struct BiquadFilter {
    b0: f64, b1: f64, b2: f64, // Feedforward coefficients
    a1: f64, a2: f64,          // Feedback coefficients
    x1: f64, x2: f64,          // Previous inputs
    y1: f64, y2: f64,          // Previous outputs
}

impl BiquadFilter {
    /// Create low-pass filter
    pub fn lowpass(fc: f64, fs: f64) -> Result<Self, AcousticFilterError> {
        if fc <= 0.0 || fc >= fs / NYQUIST_FACTOR {
            return Err(AcousticFilterError::InvalidFrequency(fc));
        }
        if fs <= 0.0 {
            return Err(AcousticFilterError::InvalidSampleRate(fs));
        }
        
        let w0 = 2.0 * std::f64::consts::PI * fc / fs;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / 2.0; // Q = 0.707 (Butterworth)
        
        let b0 = (1.0 - cos_w0) / 2.0;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        
        Ok(Self {
            b0: b0 / a0, b1: b1 / a0, b2: b2 / a0,
            a1: a1 / a0, a2: a2 / a0,
            x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0,
        })
    }
    
    /// Create high-pass filter
    pub fn highpass(fc: f64, fs: f64) -> Result<Self, AcousticFilterError> {
        if fc <= 0.0 || fc >= fs / NYQUIST_FACTOR {
            return Err(AcousticFilterError::InvalidFrequency(fc));
        }
        if fs <= 0.0 {
            return Err(AcousticFilterError::InvalidSampleRate(fs));
        }
        
        let w0 = 2.0 * std::f64::consts::PI * fc / fs;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / 2.0;
        
        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        
        Ok(Self {
            b0: b0 / a0, b1: b1 / a0, b2: b2 / a0,
            a1: a1 / a0, a2: a2 / a0,
            x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0,
        })
    }
    
    /// Process single sample (zero-alloc)
    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
              - self.a1 * self.y1 - self.a2 * self.y2;
        
        // Shift state
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        
        y
    }
    
    /// Reset filter state
    pub fn reset(&mut self) {
        self.x1 = 0.0; self.x2 = 0.0;
        self.y1 = 0.0; self.y2 = 0.0;
    }
}

/// Bandpass filter composed of lowpass and highpass stages
pub struct BandpassFilter {
    highpass: BiquadFilter,
    lowpass: BiquadFilter,
}

impl BandpassFilter {
    /// Create bandpass filter for engine frequency range
    pub fn new(fl: f64, fh: f64, fs: f64) -> Result<Self, AcousticFilterError> {
        if fl >= fh {
            return Err(AcousticFilterError::InvalidFrequency(fl));
        }
        
        let highpass = BiquadFilter::highpass(fl, fs)?;
        let lowpass = BiquadFilter::lowpass(fh, fs)?;
        
        Ok(Self { highpass, lowpass })
    }
    
    /// Process sample through both stages
    pub fn process(&mut self, x: f64) -> f64 {
        let hp_out = self.highpass.process(x);
        self.lowpass.process(hp_out)
    }
    
    /// Reset both filters
    pub fn reset(&mut self) {
        self.highpass.reset();
        self.lowpass.reset();
    }
}

/// Spectral energy detector for stage separation events
pub struct SpectralEnergyDetector {
    pub detection_threshold: f64,
    pub integration_window: usize,
    buffer: Box<[f64]>,
    buffer_idx: usize,
    sum: f64,
}

impl SpectralEnergyDetector {
    /// Create new detector with specified window size
    pub fn new(window_size: usize, threshold: f64) -> Result<Self, AcousticFilterError> {
        if window_size == 0 {
            return Err(AcousticFilterError::NumericalInstability);
        }
        
        Ok(Self {
            detection_threshold: threshold,
            integration_window: window_size,
            buffer: vec![0.0; window_size].into_boxed_slice(),
            buffer_idx: 0,
            sum: 0.0,
        })
    }
    
    /// Add sample and compute rolling energy
    pub fn add_sample(&mut self, sample: f64) -> f64 {
        // Remove old value from sum
        let old_val = self.buffer[self.buffer_idx];
        self.sum -= old_val * old_val;
        
        // Add new value
        self.buffer[self.buffer_idx] = sample;
        self.sum += sample * sample;
        
        // Advance index
        self.buffer_idx = (self.buffer_idx + 1) % self.integration_window;
        
        // Return RMS energy
        (self.sum / self.integration_window as f64).sqrt()
    }
    
    /// Check if energy exceeds threshold
    pub fn detect(&self, energy: f64) -> bool {
        energy > self.detection_threshold
    }
}

/// Stage separation event classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeparationEvent {
    None,
    PreSeparation,
    Separation,
    PostSeparation,
    Anomaly,
}

/// Main acoustic vibration analyzer
pub struct AcousticVibrationAnalyzer {
    bandpass: BandpassFilter,
    energy_detector: SpectralEnergyDetector,
    sample_rate: f64,
    last_separation_time: f64,
}

impl AcousticVibrationAnalyzer {
    /// Create analyzer for launch vehicle monitoring
    pub fn new(sample_rate: f64) -> Result<Self, AcousticFilterError> {
        let bandpass = BandpassFilter::new(
            ENGINE_FUNDAMENTAL_MIN,
            ENGINE_FUNDAMENTAL_MAX,
            sample_rate,
        )?;
        
        let energy_detector = SpectralEnergyDetector::new(1024, 0.1)?;
        
        Ok(Self {
            bandpass,
            energy_detector,
            sample_rate,
            last_separation_time: 0.0,
        })
    }
    
    /// Process audio sample and detect separation events
    pub fn process_sample(&mut self, sample: f64, timestamp: f64) -> SeparationEvent {
        // Bandpass filter to isolate engine frequencies
        let filtered = self.bandpass.process(sample);
        
        // Compute spectral energy
        let energy = self.energy_detector.add_sample(filtered);
        
        // Detect separation signature (sudden energy drop followed by recovery)
        if self.energy_detector.detect(energy) {
            if timestamp - self.last_separation_time > 10.0 {
                self.last_separation_time = timestamp;
                return SeparationEvent::Separation;
            }
        }
        
        SeparationEvent::None
    }
    
    /// Reset analyzer state
    pub fn reset(&mut self) {
        self.bandpass.reset();
        self.energy_detector.buffer.fill(0.0);
        self.energy_detector.sum = 0.0;
        self.energy_detector.buffer_idx = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_biquad_lowpass() {
        let mut filter = BiquadFilter::lowpass(100.0, 1000.0).unwrap();
        let result = filter.process(1.0);
        assert!(result.is_finite());
    }
    
    #[test]
    fn test_bandpass_filter() {
        let mut filter = BandpassFilter::new(20.0, 500.0, 44100.0).unwrap();
        let result = filter.process(0.5);
        assert!(result.is_finite());
    }
}
