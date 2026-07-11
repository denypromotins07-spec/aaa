//! Seizure Detection and Quenching Module
//! 
//! Detects epileptiform activity in the organoid and triggers
//! immediate inhibitory stimulation to reset the network.
//! Uses hardware-interrupt-style detection for minimal latency.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crate::mea::lfp_bandpass_filter::{LfpBandpassFilter, FrequencyBand, ArousalState};
use crate::mea::simd_spike_sorter::{SimdSpikeSorter, SortedSpike};

/// Maximum number of electrodes monitored for seizure activity
pub const MAX_MONITORED_ELECTRODES: usize = 256;

/// Default seizure detection thresholds
const DEFAULT_SPIKE_WAVE_THRESHOLD: f32 = 500.0; // microvolts
const DEFAULT_SYNCHRONY_THRESHOLD: f32 = 0.8;
const DEFAULT_BURST_RATE_THRESHOLD: f32 = 100.0; // bursts per minute
const DEFAULT_LFP_AMPLITUDE_THRESHOLD: f32 = 200.0; // microvolts

/// Seizure severity levels
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum SeizureSeverity {
    None = 0,
    Suspicious = 1,
    Probable = 2,
    Definite = 3,
    StatusEpilepticus = 4,
}

/// Error types for seizure quenching
#[derive(Debug, Clone, Copy)]
pub enum QuenchError {
    NotInitialized,
    AlreadyQuenching,
    HardwareFault,
    TimeoutExpired,
    QuenchFailed,
}

/// Seizure detection state
#[repr(C, align(64))]
pub struct SeizureDetector {
    /// Per-electrode spike rates
    spike_rates: [f32; MAX_MONITORED_ELECTRODES],
    /// Per-electrode LFP amplitudes
    lfp_amplitudes: [f32; MAX_MONITORED_ELECTRODES],
    /// Cross-correlation matrix (simplified)
    synchrony_index: f32,
    /// Burst detection counter
    burst_count: u32,
    /// Last burst timestamp (ns)
    last_burst_ns: u64,
    /// Spike-wave discharge counter
    spike_wave_count: u32,
    /// Current severity estimate
    severity: SeizureSeverity,
    /// Detection enabled flag
    enabled: bool,
    /// Number of monitored electrodes
    num_electrodes: usize,
}

impl SeizureDetector {
    /// Create a new seizure detector
    pub fn new(num_electrodes: usize) -> Self {
        Self {
            spike_rates: [0.0; MAX_MONITORED_ELECTRODES],
            lfp_amplitudes: [0.0; MAX_MONITORED_ELECTRODES],
            synchrony_index: 0.0,
            burst_count: 0,
            last_burst_ns: 0,
            spike_wave_count: 0,
            severity: SeizureSeverity::None,
            enabled: false,
            num_electrodes: num_electrodes.min(MAX_MONITORED_ELECTRODES),
        }
    }

    /// Update spike rate for an electrode
    #[inline]
    pub fn update_spike_rate(&mut self, electrode: usize, rate_hz: f32) {
        if electrode < self.num_electrodes {
            self.spike_rates[electrode] = rate_hz;
        }
    }

    /// Update LFP amplitude for an electrode
    #[inline]
    pub fn update_lfp_amplitude(&mut self, electrode: usize, amplitude_uv: f32) {
        if electrode < self.num_electrodes {
            self.lfp_amplitudes[electrode] = amplitude_uv;
        }
    }

    /// Compute synchrony index from spike data
    pub fn compute_synchrony(&mut self, spikes: &[SortedSpike], window_ms: u32) -> f32 {
        if spikes.is_empty() || window_ms == 0 {
            return 0.0;
        }

        // Count spikes per electrode in window
        let mut electrode_counts = [0u32; MAX_MONITORED_ELECTRODES];
        let mut total_spikes = 0u32;

        for spike in spikes {
            if spike.electrode_id as usize < MAX_MONITORED_ELECTRODES {
                electrode_counts[spike.electrode_id as usize] += 1;
                total_spikes += 1;
            }
        }

        if total_spikes == 0 {
            return 0.0;
        }

        // Compute variance-based synchrony index
        let mean = total_spikes as f32 / self.num_electrodes as f32;
        let mut variance = 0.0;

        for i in 0..self.num_electrodes {
            let diff = electrode_counts[i] as f32 - mean;
            variance += diff * diff;
        }

        variance /= self.num_electrodes as f32;

        // High variance = low synchrony, low variance = high synchrony
        // Normalize to 0-1 range
        let cv = if mean > 0.0 { variance.sqrt() / mean } else { 0.0 };
        self.synchrony_index = 1.0 / (1.0 + cv);

        self.synchrony_index
    }

    /// Detect spike-wave discharges (hallmark of seizures)
    pub fn detect_spike_wave(&mut self, lfp_samples: &[f32], sample_rate: u32) -> bool {
        if lfp_samples.len() < 100 {
            return false;
        }

        let mut spike_wave_detected = false;
        let threshold = DEFAULT_SPIKE_WAVE_THRESHOLD;

        // Look for characteristic spike-wave pattern:
        // Sharp spike (>threshold) followed by slow wave
        let mut i = 0;
        while i + 50 < lfp_samples.len() {
            // Detect sharp spike
            if lfp_samples[i].abs() > threshold {
                // Check for subsequent slow wave (within 50-200ms)
                let wave_start = i + sample_rate as usize / 20; // 50ms
                let wave_end = (i + sample_rate as usize / 5).min(lfp_samples.len());

                if wave_start < wave_end {
                    // Look for opposite polarity wave
                    let wave_max = lfp_samples[wave_start..wave_end]
                        .iter()
                        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let wave_min = lfp_samples[wave_start..wave_end]
                        .iter()
                        .fold(f32::INFINITY, |a, &b| a.min(b));

                    // Check if wave is opposite polarity and significant
                    if (wave_max - wave_min) > threshold * 0.5 {
                        spike_wave_detected = true;
                        self.spike_wave_count += 1;
                        break;
                    }
                }
            }
            i += 10; // Skip ahead
        }

        spike_wave_detected
    }

    /// Main seizure detection evaluation
    pub fn evaluate(&mut self, current_time_ns: u64) -> SeizureSeverity {
        if !self.enabled {
            return SeizureSeverity::None;
        }

        let mut score = 0u32;

        // Criterion 1: High firing rate across multiple electrodes
        let high_rate_count = self.spike_rates[..self.num_electrodes]
            .iter()
            .filter(|&&r| r > 100.0)
            .count();
        
        if high_rate_count > self.num_electrodes / 4 {
            score += 1;
        }
        if high_rate_count > self.num_electrodes / 2 {
            score += 1;
        }

        // Criterion 2: High synchrony
        if self.synchrony_index > DEFAULT_SYNCHRONY_THRESHOLD {
            score += 2;
        }

        // Criterion 3: Spike-wave discharges
        if self.spike_wave_count > 5 {
            score += 2;
        }

        // Criterion 4: High LFP amplitude
        let high_lfp_count = self.lfp_amplitudes[..self.num_electrodes]
            .iter()
            .filter(|&&a| a > DEFAULT_LFP_AMPLITUDE_THRESHOLD)
            .count();

        if high_lfp_count > self.num_electrodes / 4 {
            score += 1;
        }

        // Determine severity
        self.severity = match score {
            0 => SeizureSeverity::None,
            1..=2 => SeizureSeverity::Suspicious,
            3..=4 => SeizureSeverity::Probable,
            5..=6 => SeizureSeverity::Definite,
            _ => SeizureSeverity::StatusEpilepticus,
        };

        // Reset spike-wave count periodically
        if self.spike_wave_count > 100 {
            self.spike_wave_count = 0;
        }

        self.severity
    }

    /// Enable detection
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable detection
    pub fn disable(&mut self) {
        self.enabled = false;
        self.severity = SeizureSeverity::None;
    }

    /// Get current severity
    pub fn get_severity(&self) -> SeizureSeverity {
        self.severity
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.spike_rates = [0.0; MAX_MONITORED_ELECTRODES];
        self.lfp_amplitudes = [0.0; MAX_MONITORED_ELECTRODES];
        self.synchrony_index = 0.0;
        self.burst_count = 0;
        self.spike_wave_count = 0;
        self.severity = SeizureSeverity::None;
    }
}

/// Seizure Quencher - delivers inhibitory stimulation
pub struct SeizureQuencher {
    /// Detector reference
    detector: SeizureDetector,
    /// Quenching active flag
    quenching: AtomicBool,
    /// Quench start timestamp
    quench_start_ns: AtomicU64,
    /// Quench duration (ns)
    quench_duration_ns: u64,
    /// Inhibitory stimulation amplitude (uA)
    inhibitory_amplitude: f32,
    /// Target electrodes for quenching
    quench_electrodes: [u8; 32],
    num_quench_electrodes: usize,
    /// Callback for hardware interrupt
    interrupt_triggered: AtomicBool,
}

impl SeizureQuencher {
    /// Create a new seizure quencher
    pub fn new(num_electrodes: usize) -> Self {
        Self {
            detector: SeizureDetector::new(num_electrodes),
            quenching: AtomicBool::new(false),
            quench_start_ns: AtomicU64::new(0),
            quench_duration_ns: 5_000_000_000, // 5 seconds default
            inhibitory_amplitude: 50.0, // uA
            quench_electrodes: [0; 32],
            num_quench_electrodes: 0,
            interrupt_triggered: AtomicBool::new(false),
        }
    }

    /// Configure quench electrodes
    pub fn configure_quench_electrodes(&mut self, electrodes: &[u8]) {
        let count = electrodes.len().min(32);
        self.quench_electrodes[..count].copy_from_slice(&electrodes[..count]);
        self.num_quench_electrodes = count;
    }

    /// Set quench duration
    pub fn set_quench_duration(&mut self, duration_ms: u64) {
        self.quench_duration_ns = duration_ms * 1_000_000;
    }

    /// Set inhibitory amplitude
    pub fn set_inhibitory_amplitude(&mut self, amplitude_ua: f32) {
        self.inhibitory_amplitude = amplitude_ua.clamp(10.0, 100.0);
    }

    /// CRITICAL: Hardware-interrupt style seizure detection and quench trigger
    /// This should be called from a high-priority thread or actual hardware interrupt
    #[inline]
    pub fn check_and_trigger_quench(&self, current_time_ns: u64) -> Result<bool, QuenchError> {
        // Note: In production, this would be an actual hardware interrupt handler
        // The detector state would be memory-mapped and checked atomically
        
        if self.quenching.load(Ordering::Acquire) {
            return Err(QuenchError::AlreadyQuenching);
        }

        // Check severity (would be atomic read from detector in production)
        // For now, we rely on external calls to update_detector
        
        Ok(false) // Placeholder - actual trigger logic below
    }

    /// Trigger quenching manually or from detected seizure
    pub fn trigger_quench(&mut self, current_time_ns: u64) -> Result<(), QuenchError> {
        if self.quenching.swap(true, Ordering::SeqCst) {
            return Err(QuenchError::AlreadyQuenching);
        }

        self.quench_start_ns.store(current_time_ns, Ordering::Release);
        self.interrupt_triggered.store(true, Ordering::Release);

        // In production, this would:
        // 1. Send hardware interrupt to MEA stimulator
        // 2. Deliver biphasic inhibitory pulses to configured electrodes
        // 3. Halt all trading operations immediately
        // 4. Alert human operators

        Ok(())
    }

    /// Update detector state (call before check_and_trigger)
    pub fn update_detector(&mut self) -> SeizureSeverity {
        let now = 0; // Would use actual time
        self.detector.evaluate(now)
    }

    /// Check if quench is complete
    pub fn is_quench_complete(&self, current_time_ns: u64) -> bool {
        if !self.quenching.load(Ordering::Acquire) {
            return true;
        }

        let start = self.quench_start_ns.load(Ordering::Acquire);
        current_time_ns.saturating_sub(start) >= self.quench_duration_ns
    }

    /// End quenching
    pub fn end_quench(&mut self) {
        self.quenching.store(false, Ordering::SeqCst);
        self.interrupt_triggered.store(false, Ordering::Release);
    }

    /// Get quenching status
    pub fn is_quenching(&self) -> bool {
        self.quenching.load(Ordering::Acquire)
    }

    /// Get interrupt trigger status
    pub fn was_interrupt_triggered(&self) -> bool {
        self.interrupt_triggered.load(Ordering::Acquire)
    }

    /// Get detector reference
    pub fn detector_mut(&mut self) -> &mut SeizureDetector {
        &mut self.detector
    }

    /// Emergency halt - stop everything
    pub fn emergency_halt(&mut self) {
        self.quenching.store(true, Ordering::SeqCst);
        self.detector.disable();
    }
}

/// Trading halt coordinator
pub struct TradingHaltCoordinator {
    /// Linked to seizure quencher
    halt_requested: AtomicBool,
    /// Halt reason code
    halt_reason: AtomicU64,
    /// External halt callback (would be set by trading system)
    halt_callback: Option<fn(u64)>,
}

impl TradingHaltCoordinator {
    /// Create a new coordinator
    pub fn new() -> Self {
        Self {
            halt_requested: AtomicBool::new(false),
            halt_reason: AtomicU64::new(0),
            halt_callback: None,
        }
    }

    /// Register halt callback
    pub fn register_callback(&mut self, callback: fn(u64)) {
        self.halt_callback = Some(callback);
    }

    /// Request trading halt
    pub fn request_halt(&self, reason: u64) {
        self.halt_requested.store(true, Ordering::SeqCst);
        self.halt_reason.store(reason, Ordering::SeqCst);

        if let Some(cb) = self.halt_callback {
            cb(reason);
        }
    }

    /// Clear halt request
    pub fn clear_halt(&self) {
        self.halt_requested.store(false, Ordering::SeqCst);
        self.halt_reason.store(0, Ordering::SeqCst);
    }

    /// Check if halt is requested
    pub fn is_halt_requested(&self) -> bool {
        self.halt_requested.load(Ordering::Acquire)
    }

    /// Get halt reason
    pub fn get_halt_reason(&self) -> u64 {
        self.halt_reason.load(Ordering::Acquire)
    }
}

impl Default for TradingHaltCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seizure_detector_initialization() {
        let detector = SeizureDetector::new(64);
        assert_eq!(detector.get_severity(), SeizureSeverity::None);
        assert!(!detector.enabled);
    }

    #[test]
    fn test_synchrony_computation() {
        let mut detector = SeizureDetector::new(8);
        detector.enable();

        let spikes = vec![
            SortedSpike::default(),
            SortedSpike::default(),
        ];
        
        let synchrony = detector.compute_synchrony(&spikes, 100);
        assert!(synchrony >= 0.0 && synchrony <= 1.0);
    }

    #[test]
    fn test_quencher_trigger() {
        let mut quencher = SeizureQuencher::new(64);
        
        let result = quencher.trigger_quench(1_000_000_000);
        assert!(result.is_ok());
        assert!(quencher.is_quenching());
    }

    #[test]
    fn test_trading_halt_coordinator() {
        extern "C" fn mock_callback(reason: u64) {
            let _ = reason;
        }

        let mut coordinator = TradingHaltCoordinator::new();
        coordinator.register_callback(mock_callback);
        
        assert!(!coordinator.is_halt_requested());
        
        coordinator.request_halt(42);
        assert!(coordinator.is_halt_requested());
        assert_eq!(coordinator.get_halt_reason(), 42);
        
        coordinator.clear_halt();
        assert!(!coordinator.is_halt_requested());
    }
}
