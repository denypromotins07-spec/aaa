//! Bio-Silicon Bridge Translator
//! 
//! Translates between biological neural signals (spikes, LFP) and
//! silicon trading system commands. Handles bidirectional conversion
//! with proper safety interlocks.

use crate::mea::simd_spike_sorter::{SortedSpike, SimdSpikeSorter};
use crate::mea::lfp_bandpass_filter::{LfpBandpassFilter, FrequencyBand, ArousalState};
use crate::containment::seizure_quencher::{SeizureQuencher, SeizureSeverity, TradingHaltCoordinator};
use crate::containment::iit_phi_calculator::{IitPhiCalculator, ContainmentAction};

/// Maximum number of spike events in a batch
pub const MAX_SPIKE_BATCH: usize = 4096;

/// Maximum number of electrode clusters
pub const MAX_CLUSTERS: usize = 256;

/// Trading action types from organoid output
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum TradingAction {
    /// No action
    None = 0,
    /// Buy signal
    Buy = 1,
    /// Sell signal
    Sell = 2,
    /// Hold/maintain position
    Hold = 3,
    /// Reduce exposure
    ReduceRisk = 4,
    /// Emergency liquidation
    Liquidate = 5,
}

/// Translated order from organoid activity
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct OrganoidOrder {
    /// Action type
    pub action: TradingAction,
    /// Confidence level (0-1)
    pub confidence: f32,
    /// Derived from cluster ID
    pub cluster_id: u16,
    /// Timestamp (ns)
    pub timestamp_ns: u64,
    /// Spike rate that triggered this
    pub trigger_rate: f32,
    /// LFP state at time of trigger
    pub lfp_state: u8,
}

impl Default for OrganoidOrder {
    fn default() -> Self {
        Self {
            action: TradingAction::None,
            confidence: 0.0,
            cluster_id: 0xFFFF,
            timestamp_ns: 0,
            trigger_rate: 0.0,
            lfp_state: 0,
        }
    }
}

/// Error types for bridge translation
#[derive(Debug, Clone, Copy)]
pub enum BridgeError {
    TranslationFailed,
    SafetyInterlockActive,
    InvalidClusterMapping,
    SignalTooWeak,
    ContainmentActive,
    NotInitialized,
}

/// Cluster-to-action mapping
#[repr(C)]
pub struct ClusterMapping {
    /// Cluster ID
    pub cluster_id: u16,
    /// Mapped trading action
    pub action: TradingAction,
    /// Minimum spike rate to trigger
    pub min_rate_hz: f32,
    /// Weight/priority of this cluster
    pub weight: f32,
    /// Enabled flag
    pub enabled: bool,
}

impl Default for ClusterMapping {
    fn default() -> Self {
        Self {
            cluster_id: 0xFFFF,
            action: TradingAction::None,
            min_rate_hz: 10.0,
            weight: 1.0,
            enabled: false,
        }
    }
}

/// Bio-Silicon Bridge Translator
pub struct BioSiliconTranslator {
    /// Cluster mappings
    cluster_mappings: [ClusterMapping; MAX_CLUSTERS],
    /// Number of configured clusters
    num_clusters: usize,
    /// Current arousal state from LFP
    current_arousal: ArousalState,
    /// Reference to seizure quencher
    seizure_quencher: Option<SeizureQuencher>,
    /// Reference to IIT calculator
    iit_calculator: Option<IitPhiCalculator>,
    /// Trading halt coordinator
    halt_coordinator: TradingHaltCoordinator,
    /// Safety interlock active
    safety_interlock: bool,
    /// Containment level
    containment_level: ContainmentAction,
    /// Order queue (output buffer)
    order_queue: [OrganoidOrder; 64],
    order_write_idx: usize,
    order_read_idx: usize,
}

impl BioSiliconTranslator {
    /// Create a new bridge translator
    pub fn new(num_clusters: usize) -> Self {
        Self {
            cluster_mappings: [ClusterMapping::default(); MAX_CLUSTERS],
            num_clusters: num_clusters.min(MAX_CLUSTERS),
            current_arousal: ArousalState::Unknown,
            seizure_quencher: None,
            iit_calculator: None,
            halt_coordinator: TradingHaltCoordinator::new(),
            safety_interlock: true, // Start locked
            containment_level: ContainmentAction::None,
            order_queue: [OrganoidOrder::default(); 64],
            order_write_idx: 0,
            order_read_idx: 0,
        }
    }

    /// Configure a cluster mapping
    pub fn configure_cluster(
        &mut self,
        cluster_id: u16,
        action: TradingAction,
        min_rate_hz: f32,
        weight: f32,
    ) -> Result<(), BridgeError> {
        if cluster_id as usize >= self.num_clusters {
            return Err(BridgeError::InvalidClusterMapping);
        }

        let mapping = &mut self.cluster_mappings[cluster_id as usize];
        mapping.cluster_id = cluster_id;
        mapping.action = action;
        mapping.min_rate_hz = min_rate_hz.max(0.1);
        mapping.weight = weight.max(0.0).min(10.0);
        mapping.enabled = true;

        Ok(())
    }

    /// Attach seizure quencher for safety monitoring
    pub fn attach_seizure_quencher(&mut self, quencher: SeizureQuencher) {
        self.seizure_quencher = Some(quencher);
    }

    /// Attach IIT calculator for containment monitoring
    pub fn attach_iit_calculator(&mut self, calculator: IitPhiCalculator) {
        self.iit_calculator = Some(calculator);
    }

    /// Translate spike data to trading orders
    pub fn translate_spikes(
        &mut self,
        spikes: &[SortedSpike],
        timestamp_ns: u64,
    ) -> Result<Option<OrganoidOrder>, BridgeError> {
        // Check safety interlocks first
        if !self.check_safety_interlocks()? {
            return Err(BridgeError::SafetyInterlockActive);
        }

        if spikes.is_empty() {
            return Ok(None);
        }

        // Compute cluster firing rates
        let mut cluster_rates = [0.0f32; MAX_CLUSTERS];
        let mut cluster_counts = [0u32; MAX_CLUSTERS];

        // Simple clustering by electrode ID (in production, use actual cluster assignments)
        for spike in spikes {
            let cluster = (spike.electrode_id as usize) % self.num_clusters;
            cluster_counts[cluster] += 1;
        }

        // Convert counts to rates (assuming 1 second window)
        for i in 0..self.num_clusters {
            cluster_rates[i] = cluster_counts[i] as f32;
        }

        // Find most active enabled cluster
        let mut best_cluster = None;
        let mut best_weighted_rate = 0.0;

        for i in 0..self.num_clusters {
            let mapping = &self.cluster_mappings[i];
            if !mapping.enabled {
                continue;
            }

            if cluster_rates[i] >= mapping.min_rate_hz {
                let weighted_rate = cluster_rates[i] * mapping.weight;
                if weighted_rate > best_weighted_rate {
                    best_weighted_rate = weighted_rate;
                    best_cluster = Some(i);
                }
            }
        }

        // Generate order if cluster found
        if let Some(cluster_idx) = best_cluster {
            let mapping = &self.cluster_mappings[cluster_idx];
            
            // Compute confidence based on rate and arousal state
            let base_confidence = (cluster_rates[cluster_idx] / mapping.min_rate_hz).min(1.0);
            let arousal_modifier = self.get_arousal_modifier();
            let confidence = (base_confidence * arousal_modifier).min(1.0);

            let mut order = OrganoidOrder::default();
            order.action = mapping.action;
            order.confidence = confidence;
            order.cluster_id = mapping.cluster_id;
            order.timestamp_ns = timestamp_ns;
            order.trigger_rate = cluster_rates[cluster_idx];
            order.lfp_state = self.current_arousal as u8;

            // Queue the order
            self.queue_order(order)?;

            return Ok(Some(order));
        }

        Ok(None)
    }

    /// Get arousal state modifier for confidence
    fn get_arousal_modifier(&self) -> f32 {
        match self.current_arousal {
            ArousalState::HighAttention => 1.2,
            ArousalState::Alert => 1.0,
            ArousalState::Baseline => 0.9,
            ArousalState::Relaxed => 0.7,
            ArousalState::Drowsy => 0.5,
            ArousalState::DeepSleep => 0.2,
            ArousalState::Unknown => 0.5,
        }
    }

    /// Queue an order for output
    fn queue_order(&mut self, order: OrganoidOrder) -> Result<(), BridgeError> {
        let next_write = (self.order_write_idx + 1) % self.order_queue.len();
        
        if next_write == self.order_read_idx {
            // Queue full - could implement overflow handling
            return Err(BridgeError::TranslationFailed);
        }

        self.order_queue[self.order_write_idx] = order;
        self.order_write_idx = next_write;
        
        Ok(())
    }

    /// Read pending orders
    pub fn read_orders(&mut self, dest: &mut [OrganoidOrder]) -> usize {
        let mut count = 0;
        
        while count < dest.len() && self.order_read_idx != self.order_write_idx {
            dest[count] = self.order_queue[self.order_read_idx];
            self.order_read_idx = (self.order_read_idx + 1) % self.order_queue.len();
            count += 1;
        }
        
        count
    }

    /// Check all safety interlocks
    fn check_safety_interlocks(&self) -> Result<bool, BridgeError> {
        if self.safety_interlock {
            return Ok(false);
        }

        if self.containment_level == ContainmentAction::FullHalt {
            return Ok(false);
        }

        // Check seizure status
        if let Some(ref quencher) = self.seizure_quencher {
            if quencher.is_quenching() {
                return Ok(false);
            }
        }

        // Check halt coordinator
        if self.halt_coordinator.is_halt_requested() {
            return Ok(false);
        }

        Ok(true)
    }

    /// Update arousal state from LFP
    pub fn update_arousal_state(&mut self, state: ArousalState) {
        self.current_arousal = state;
    }

    /// Update containment level
    pub fn update_containment_level(&mut self, level: ContainmentAction) {
        self.containment_level = level;
        
        if level == ContainmentAction::FullHalt {
            self.safety_interlock = true;
            self.halt_coordinator.request_halt(0x4949); // "II" for IIT
        }
    }

    /// Enable the bridge (release safety interlock)
    pub fn enable_bridge(&mut self) -> Result<(), BridgeError> {
        // Can only enable if no containment issues
        if self.containment_level == ContainmentAction::FullHalt {
            return Err(BridgeError::ContainmentActive);
        }

        self.safety_interlock = false;
        Ok(())
    }

    /// Disable the bridge (engage safety interlock)
    pub fn disable_bridge(&mut self) {
        self.safety_interlock = true;
    }

    /// Trigger emergency halt
    pub fn emergency_halt(&mut self, reason: u64) {
        self.safety_interlock = true;
        self.containment_level = ContainmentAction::FullHalt;
        self.halt_coordinator.request_halt(reason);
    }

    /// Get current bridge status
    pub fn get_status(&self) -> BridgeStatus {
        BridgeStatus {
            safety_interlock: self.safety_interlock,
            containment_level: self.containment_level,
            arousal_state: self.current_arousal,
            orders_pending: if self.order_write_idx >= self.order_read_idx {
                self.order_write_idx - self.order_read_idx
            } else {
                self.order_queue.len() - self.order_read_idx + self.order_write_idx
            },
        }
    }
}

/// Bridge status information
#[derive(Debug, Clone, Copy)]
pub struct BridgeStatus {
    pub safety_interlock: bool,
    pub containment_level: ContainmentAction,
    pub arousal_state: ArousalState,
    pub orders_pending: usize,
}

/// Silicon-to-Bio stimulation translator
pub struct StimulationTranslator {
    /// Current stimulation pattern
    stimulation_pattern: [f32; 64],
    /// Pattern length
    pattern_length: usize,
    /// Enabled channels
    enabled_channels: u64,
}

impl StimulationTranslator {
    /// Create a new stimulation translator
    pub fn new() -> Self {
        Self {
            stimulation_pattern: [0.0; 64],
            pattern_length: 0,
            enabled_channels: 0,
        }
    }

    /// Translate trading signal to stimulation pattern
    pub fn translate_to_stimulation(
        &mut self,
        market_signal: f32,
        volatility: f32,
    ) -> &[f32] {
        // Normalize inputs
        let norm_signal = market_signal.clamp(-1.0, 1.0);
        let norm_vol = volatility.clamp(0.0, 1.0);

        // Generate biphasic pulse pattern
        // Positive phase encodes signal direction
        // Amplitude encodes confidence
        // Frequency modulated by volatility
        
        let amplitude = norm_signal.abs() * 50.0; // 0-50 uA
        let polarity = if norm_signal >= 0.0 { 1.0 } else { -1.0 };
        
        // Create simple biphasic pulse
        self.stimulation_pattern[0] = amplitude * polarity;
        self.stimulation_pattern[1] = -amplitude * polarity; // Return phase
        self.stimulation_pattern[2] = 0.0; // Inter-pulse gap
        
        self.pattern_length = 3;
        self.enabled_channels = 0xFF; // Enable first 8 channels

        &self.stimulation_pattern[..self.pattern_length]
    }

    /// Enable a stimulation channel
    pub fn enable_channel(&mut self, channel: u8) {
        if channel < 64 {
            self.enabled_channels |= 1u64 << channel;
        }
    }

    /// Disable a stimulation channel
    pub fn disable_channel(&mut self, channel: u8) {
        if channel < 64 {
            self.enabled_channels &= !(1u64 << channel);
        }
    }

    /// Get enabled channels mask
    pub fn get_enabled_channels(&self) -> u64 {
        self.enabled_channels
    }
}

impl Default for StimulationTranslator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_mapping() {
        let mut translator = BioSiliconTranslator::new(8);
        
        translator.configure_cluster(0, TradingAction::Buy, 10.0, 1.0).unwrap();
        
        assert!(translator.cluster_mappings[0].enabled);
        assert_eq!(translator.cluster_mappings[0].action, TradingAction::Buy);
    }

    #[test]
    fn test_safety_interlock() {
        let mut translator = BioSiliconTranslator::new(8);
        
        // Should start locked
        assert!(translator.safety_interlock);
        
        translator.enable_bridge().unwrap();
        assert!(!translator.safety_interlock);
        
        translator.disable_bridge();
        assert!(translator.safety_interlock);
    }

    #[test]
    fn test_stimulation_translation() {
        let mut stim = StimulationTranslator::new();
        
        let pattern = stim.translate_to_stimulation(0.5, 0.2);
        assert_eq!(pattern.len(), 3);
        assert!(pattern[0] > 0.0); // Positive phase
        assert!(pattern[1] < 0.0); // Return phase
    }

    #[test]
    fn test_order_queue() {
        let mut translator = BioSiliconTranslator::new(8);
        translator.enable_bridge().unwrap();
        
        // Create mock spike data
        let spikes = vec![SortedSpike::default()];
        
        // Should fail due to no configured mappings
        let result = translator.translate_spikes(&spikes, 1_000_000_000);
        assert!(result.is_ok()); // Returns Ok(None) when no cluster triggers
    }
}
