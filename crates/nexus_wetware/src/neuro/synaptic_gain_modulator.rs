//! Synaptic Gain Modulator for Neuromodulation Effects
//! 
//! Implements biochemical state modulation through electrical stimulation
//! baseline adjustment and microfluidic coordination.

use crate::neuro::microfluidic_pump_controller::{MicrofluidicPumpController, BiochemicalAgent, PumpError};

/// Maximum number of stimulation channels
pub const MAX_STIM_CHANNELS: usize = 64;

/// Default stimulation parameters
const DEFAULT_PULSE_WIDTH_US: u32 = 200;
const DEFAULT_MAX_AMPLITUDE_UA: f32 = 100.0;
const DEFAULT_MIN_AMPLITUDE_UA: f32 = 0.1;

/// Stimulation waveform types
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum WaveformType {
    Monophasic = 0,
    Biphasic = 1,
    Triphasic = 2,
    Sinusoidal = 3,
    Noise = 4,
}

/// Network state classifications
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum NetworkState {
    Baseline = 0,
    Excited = 1,
    Inhibited = 2,
    Synchronous = 3,
    Asynchronous = 4,
    Seizure = 5,
    Silent = 6,
}

/// Error types for gain modulation
#[derive(Debug, Clone, Copy)]
pub enum GainError {
    InvalidChannel,
    AmplitudeOutOfRange,
    FrequencyOutOfRange,
    StateTransitionFailed,
    HardwareFault,
}

/// Stimulation pulse configuration
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct PulseConfig {
    /// Waveform type
    pub waveform: WaveformType,
    /// Amplitude in microamps
    pub amplitude_ua: f32,
    /// Pulse width in microseconds
    pub pulse_width_us: u32,
    /// Frequency in Hz
    pub frequency_hz: f32,
    /// Inter-pulse interval (computed from frequency)
    pub inter_pulse_ms: f32,
}

impl Default for PulseConfig {
    fn default() -> Self {
        Self {
            waveform: WaveformType::Biphasic,
            amplitude_ua: 10.0,
            pulse_width_us: DEFAULT_PULSE_WIDTH_US,
            frequency_hz: 50.0,
            inter_pulse_ms: 20.0,
        }
    }
}

impl PulseConfig {
    /// Validate pulse configuration
    pub fn validate(&self) -> Result<(), GainError> {
        if self.amplitude_ua < DEFAULT_MIN_AMPLITUDE_UA || self.amplitude_ua > DEFAULT_MAX_AMPLITUDE_UA {
            return Err(GainError::AmplitudeOutOfRange);
        }
        if self.frequency_hz < 0.1 || self.frequency_hz > 500.0 {
            return Err(GainError::FrequencyOutOfRange);
        }
        Ok(())
    }

    /// Compute inter-pulse interval from frequency
    pub fn compute_inter_pulse(&mut self) {
        if self.frequency_hz > 0.0 {
            self.inter_pulse_ms = 1000.0 / self.frequency_hz;
        }
    }
}

/// Channel state for synaptic gain control
#[repr(C, align(32))]
pub struct ChannelState {
    /// Current pulse configuration
    pub config: PulseConfig,
    /// Baseline amplitude (for modulation)
    pub baseline_amplitude: f32,
    /// Current gain multiplier
    pub gain_multiplier: f32,
    /// Target network state
    pub target_state: NetworkState,
    /// Current estimated state
    pub current_state: NetworkState,
    /// Spike rate estimate (Hz)
    pub spike_rate: f32,
    /// Last stimulation timestamp (ns)
    pub last_stim_ns: u64,
}

impl Default for ChannelState {
    fn default() -> Self {
        Self {
            config: PulseConfig::default(),
            baseline_amplitude: 10.0,
            gain_multiplier: 1.0,
            target_state: NetworkState::Baseline,
            current_state: NetworkState::Baseline,
            spike_rate: 0.0,
            last_stim_ns: 0,
        }
    }
}

/// Synaptic Gain Modulator
pub struct SynapticGainModulator {
    /// Per-channel states
    channels: [ChannelState; MAX_STIM_CHANNELS],
    /// Number of active channels
    num_channels: usize,
    /// Global neuromodulator levels
    dopamine_level: f32,
    serotonin_level: f32,
    cortisol_level: f32,
    /// Reference to pump controller for biochemical delivery
    pump_controller: Option<MicrofluidicPumpController>,
    /// System enabled flag
    enabled: bool,
}

impl SynapticGainModulator {
    /// Create a new gain modulator
    pub fn new(num_channels: usize) -> Self {
        Self {
            channels: [ChannelState::default(); MAX_STIM_CHANNELS],
            num_channels: num_channels.min(MAX_STIM_CHANNELS),
            dopamine_level: 0.5,
            serotonin_level: 0.5,
            cortisol_level: 0.0,
            pump_controller: None,
            enabled: false,
        }
    }

    /// Attach a pump controller for biochemical coordination
    pub fn attach_pump_controller(&mut self, controller: MicrofluidicPumpController) {
        self.pump_controller = Some(controller);
    }

    /// Configure a stimulation channel
    pub fn configure_channel(
        &mut self,
        channel: usize,
        amplitude_ua: f32,
        frequency_hz: f32,
        waveform: WaveformType,
    ) -> Result<(), GainError> {
        if channel >= self.num_channels {
            return Err(GainError::InvalidChannel);
        }

        let state = &mut self.channels[channel];
        state.config.amplitude_ua = amplitude_ua.clamp(DEFAULT_MIN_AMPLITUDE_UA, DEFAULT_MAX_AMPLITUDE_UA);
        state.config.frequency_hz = frequency_hz.clamp(0.1, 500.0);
        state.config.waveform = waveform;
        state.config.compute_inter_pulse();
        state.baseline_amplitude = state.config.amplitude_ua;

        state.config.validate()
    }

    /// Set gain multiplier for a channel
    pub fn set_gain(&mut self, channel: usize, gain: f32) -> Result<(), GainError> {
        if channel >= self.num_channels {
            return Err(GainError::InvalidChannel);
        }

        let clamped = gain.max(0.0).min(5.0);
        self.channels[channel].gain_multiplier = clamped;
        
        // Apply gain to amplitude
        let state = &mut self.channels[channel];
        state.config.amplitude_ua = state.baseline_amplitude * clamped;

        Ok(())
    }

    /// Update neuromodulator levels
    pub fn set_neuromodulators(
        &mut self,
        dopamine: Option<f32>,
        serotonin: Option<f32>,
        cortisol: Option<f32>,
    ) {
        if let Some(d) = dopamine {
            self.dopamine_level = d.clamp(0.0, 1.0);
        }
        if let Some(s) = serotonin {
            self.serotonin_level = s.clamp(0.0, 1.0);
        }
        if let Some(c) = cortisol {
            self.cortisol_level = c.clamp(0.0, 1.0);
        }

        // Apply global gain adjustments based on neuromodulators
        self.apply_neuromodulator_gains();
    }

    /// Apply neuromodulator-based gain adjustments
    fn apply_neuromodulator_gains(&mut self) {
        for i in 0..self.num_channels {
            let state = &mut self.channels[i];
            
            // Dopamine: increases excitability (higher gain)
            let dopaminergic_factor = 0.8 + self.dopamine_level * 0.4;
            
            // Serotonin: stabilizes (reduces variance, moderate gain)
            let serotonergic_factor = 1.0 - self.serotonin_level * 0.2;
            
            // Cortisol: inhibitory (lower gain, risk-averse)
            let cortisol_factor = 1.0 - self.cortisol_level * 0.6;

            let total_factor = dopaminergic_factor * serotonergic_factor * cortisol_factor;
            state.gain_multiplier = total_factor.clamp(0.1, 2.0);
            state.config.amplitude_ua = state.baseline_amplitude * state.gain_multiplier;
        }
    }

    /// Trigger cortisol analogue protocol (black swan response)
    pub fn trigger_cortisol_protocol(&mut self, intensity: f32) -> Result<(), GainError> {
        let clamped = intensity.clamp(0.0, 1.0);
        
        // Set cortisol level
        self.cortisol_level = clamped;
        
        // Reduce dopamine (stress response)
        self.dopamine_level = (1.0 - clamped * 0.5).max(0.2);
        
        // Apply gains
        self.apply_neuromodulator_gains();

        // Coordinate with pump controller if available
        if let Some(ref mut pump) = self.pump_controller {
            // Deliver cortisol analogue via channel 3
            let _ = pump.deliver_bolus(3, 500.0 * clamped, 50.0);
        }

        // Shift all channels to inhibited state
        for i in 0..self.num_channels {
            self.channels[i].target_state = NetworkState::Inhibited;
            // Reduce stimulation frequency during stress
            self.channels[i].config.frequency_hz = 
                self.channels[i].config.frequency_hz * 0.5;
        }

        Ok(())
    }

    /// Update spike rate estimate for a channel
    pub fn update_spike_rate(&mut self, channel: usize, rate_hz: f32) -> Result<(), GainError> {
        if channel >= self.num_channels {
            return Err(GainError::InvalidChannel);
        }

        self.channels[channel].spike_rate = rate_hz.max(0.0);
        
        // Update state estimation based on spike rate
        self.update_channel_state(channel);
        
        Ok(())
    }

    /// Update channel state based on activity
    fn update_channel_state(&mut self, channel: usize) {
        let state = &self.channels[channel];
        let rate = state.spike_rate;
        
        let new_state = if rate > 200.0 {
            NetworkState::Excited
        } else if rate < 1.0 {
            NetworkState::Silent
        } else if rate > 100.0 && rate < 150.0 {
            // Check for synchrony (would need cross-channel correlation)
            NetworkState::Synchronous
        } else {
            NetworkState::Baseline
        };

        self.channels[channel].current_state = new_state;
    }

    /// Get current network state estimate
    pub fn get_network_state(&self) -> NetworkState {
        // Return dominant state across channels
        let mut state_counts = [0usize; 7];
        
        for channel in &self.channels[..self.num_channels] {
            state_counts[channel.current_state as usize] += 1;
        }

        let max_count = state_counts.iter().max().copied().unwrap_or(0);
        for (i, &count) in state_counts.iter().enumerate() {
            if count == max_count {
                return unsafe { core::mem::transmute(i as u8) };
            }
        }

        NetworkState::Baseline
    }

    /// Enable stimulation
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable stimulation
    pub fn disable(&mut self) {
        self.enabled = false;
        // Zero all amplitudes when disabled
        for i in 0..self.num_channels {
            self.channels[i].config.amplitude_ua = 0.0;
        }
    }

    /// Get current stimulation amplitude for a channel
    pub fn get_amplitude(&self, channel: usize) -> Option<f32> {
        if channel < self.num_channels {
            Some(self.channels[channel].config.amplitude_ua)
        } else {
            None
        }
    }

    /// Get current neuromodulator state
    pub fn get_neuromodulator_state(&self) -> (f32, f32, f32) {
        (self.dopamine_level, self.serotonin_level, self.cortisol_level)
    }
}

/// State machine for biochemical state transitions
pub struct BiochemicalStateMachine {
    /// Current state
    current_state: NetworkState,
    /// State transition history
    history: [NetworkState; 32],
    history_idx: usize,
    /// Dwell time in current state (ms)
    dwell_time_ms: u64,
    /// Minimum dwell time before transition (ms)
    min_dwell_time_ms: u64,
    /// State transition callbacks
    transition_pending: bool,
}

impl BiochemicalStateMachine {
    /// Create a new state machine
    pub fn new(initial_state: NetworkState) -> Self {
        Self {
            current_state: initial_state,
            history: [NetworkState::Baseline; 32],
            history_idx: 0,
            dwell_time_ms: 0,
            min_dwell_time_ms: 100,
            transition_pending: false,
        }
    }

    /// Attempt state transition
    pub fn transition_to(&mut self, target: NetworkState, elapsed_ms: u64) -> bool {
        self.dwell_time_ms += elapsed_ms;

        if self.transition_pending {
            return false; // Wait for confirmation
        }

        if self.dwell_time_ms < self.min_dwell_time_ms {
            return false; // Must dwell minimum time
        }

        // Validate transition
        if !self.is_valid_transition(self.current_state, target) {
            return false;
        }

        // Record old state
        self.history[self.history_idx] = self.current_state;
        self.history_idx = (self.history_idx + 1) % 32;

        // Transition
        self.current_state = target;
        self.dwell_time_ms = 0;

        true
    }

    /// Check if transition is valid
    fn is_valid_transition(&self, from: NetworkState, to: NetworkState) -> bool {
        match (from, to) {
            // Can always go to baseline
            (_, NetworkState::Baseline) => true,
            // Seizure can only go to inhibited (quenching)
            (NetworkState::Seizure, NetworkState::Inhibited) => true,
            (NetworkState::Seizure, _) => false,
            // Excited can go to synchronous or inhibited
            (NetworkState::Excited, NetworkState::Synchronous) => true,
            (NetworkState::Excited, NetworkState::Inhibited) => true,
            // Default: allow most transitions
            _ => true,
        }
    }

    /// Get current state
    pub fn get_state(&self) -> NetworkState {
        self.current_state
    }

    /// Get dwell time in current state
    pub fn get_dwell_time(&self) -> u64 {
        self.dwell_time_ms
    }

    /// Get state history
    pub fn get_history(&self) -> &[NetworkState] {
        let end = self.history_idx.min(32);
        &self.history[..end]
    }

    /// Reset state machine
    pub fn reset(&mut self, state: NetworkState) {
        self.current_state = state;
        self.dwell_time_ms = 0;
        self.history = [NetworkState::Baseline; 32];
        self.history_idx = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pulse_config_validation() {
        let mut config = PulseConfig::default();
        assert!(config.validate().is_ok());

        config.amplitude_ua = 0.0;
        assert!(matches!(config.validate(), Err(GainError::AmplitudeOutOfRange)));

        config.amplitude_ua = 200.0;
        assert!(matches!(config.validate(), Err(GainError::AmplitudeOutOfRange)));
    }

    #[test]
    fn test_gain_modulation() {
        let mut modulator = SynapticGainModulator::new(8);
        modulator.configure_channel(0, 10.0, 50.0, WaveformType::Biphasic).unwrap();

        let baseline = modulator.get_amplitude(0).unwrap();
        assert!((baseline - 10.0).abs() < 1e-6);

        modulator.set_gain(0, 0.5).unwrap();
        let reduced = modulator.get_amplitude(0).unwrap();
        assert!((reduced - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_cortisol_protocol() {
        let mut modulator = SynapticGainModulator::new(8);
        modulator.configure_channel(0, 10.0, 50.0, WaveformType::Biphasic).unwrap();

        let (_, _, cortisol_before) = modulator.get_neuromodulator_state();
        assert!(cortisol_before < 0.1);

        modulator.trigger_cortisol_protocol(1.0).unwrap();

        let (_, _, cortisol_after) = modulator.get_neuromodulator_state();
        assert!(cortisol_after > 0.9);
    }

    #[test]
    fn test_state_machine_transitions() {
        let mut sm = BiochemicalStateMachine::new(NetworkState::Baseline);
        
        // Should allow baseline -> excited
        let result = sm.transition_to(NetworkState::Excited, 200);
        assert!(result);
        assert_eq!(sm.get_state(), NetworkState::Excited);

        // Seizure -> baseline should fail (must go through inhibited)
        sm.current_state = NetworkState::Seizure;
        let result = sm.transition_to(NetworkState::Baseline, 200);
        assert!(!result);
    }
}
