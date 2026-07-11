//! Flow State Decoder using Theta/Beta ratio analysis for cognitive engagement quantification.
//!
//! This module implements real-time calculation of the Theta/Beta power ratio,
//! a well-established neurophysiological marker for attentional states, cognitive load,
//! and flow state detection. Used to quantify user engagement with digital platforms.

use super::{simd_eeg_stream::SimdEegProcessor, lock_free_ica_artifact::{LockFreeIcaArtifactRejection, ArtifactType}};
use std::arch::x86_64::*;

/// Flow state classification thresholds
pub const FLOW_STATE_THRESHOLD_HIGH: f32 = 2.5;  // High theta/beta = deep flow
pub const FLOW_STATE_THRESHOLD_LOW: f32 = 0.8;   // Low theta/beta = high alertness/fatigue
pub const FATIGUE_THRESHOLD: f32 = 3.0;          // Very high theta/beta = neural fatigue
pub const ENGAGEMENT_OPTIMAL_MIN: f32 = 1.0;
pub const ENGAGEMENT_OPTIMAL_MAX: f32 = 2.2;

/// Cognitive state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CognitiveState {
    DeepFlow = 0,       // Optimal engagement, peak performance
    ModerateFlow = 1,   // Good engagement
    Neutral = 2,        // Baseline state
    HighAlertness = 3,  // Beta-dominant, focused but potentially stressed
    NeuralFatigue = 4,  // Theta-dominant, exhaustion
    Drowsy = 5,         // Delta/Theta dominant, near sleep
}

/// Flow state metrics for a single channel or region
#[derive(Debug, Clone, Copy)]
pub struct FlowMetrics {
    /// Theta/Beta ratio
    pub theta_beta_ratio: f32,
    /// Alpha/Theta ratio (relaxation indicator)
    pub alpha_theta_ratio: f32,
    /// Gamma/Beta ratio (cognitive processing)
    pub gamma_beta_ratio: f32,
    /// Overall engagement score (0-1)
    pub engagement_score: f32,
    /// Fatigue index (0-1)
    pub fatigue_index: f32,
    /// Cognitive load estimate (0-1)
    pub cognitive_load: f32,
    /// Detected cognitive state
    pub state: CognitiveState,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Channel ID
    pub channel_id: usize,
}

impl FlowMetrics {
    #[inline]
    pub const fn new() -> Self {
        Self {
            theta_beta_ratio: 0.0,
            alpha_theta_ratio: 0.0,
            gamma_beta_ratio: 0.0,
            engagement_score: 0.0,
            fatigue_index: 0.0,
            cognitive_load: 0.0,
            state: CognitiveState::Neutral,
            timestamp_ns: 0,
            channel_id: 0,
        }
    }

    /// Calculate composite engagement score from spectral ratios
    #[inline]
    pub fn calculate_engagement(&mut self) {
        // Engagement is optimal when theta/beta is in the "flow zone"
        let tb = self.theta_beta_ratio.clamp(0.0, 5.0);
        
        // Gaussian-like engagement function centered on optimal ratio
        let optimal_tb = (ENGAGEMENT_OPTIMAL_MIN + ENGAGEMENT_OPTIMAL_MAX) / 2.0;
        let spread = (ENGAGEMENT_OPTIMAL_MAX - ENGAGEMENT_OPTIMAL_MIN) / 2.0;
        
        let tb_component = ((tb - optimal_tb) / (spread + 1e-10)).exp().recip();
        
        // Alpha component (moderate alpha indicates relaxed focus)
        let at = self.alpha_theta_ratio.clamp(0.0, 3.0);
        let alpha_component = if at > 0.5 && at < 2.0 { 1.0 } else { 0.5 };
        
        // Gamma component (high gamma indicates active processing)
        let gb = self.gamma_beta_ratio.clamp(0.0, 2.0);
        let gamma_component = gb.min(1.0);
        
        self.engagement_score = (tb_component * 0.5 + alpha_component * 0.3 + gamma_component * 0.2).clamp(0.0, 1.0);
    }

    /// Calculate fatigue index from spectral features
    #[inline]
    pub fn calculate_fatigue(&mut self) {
        // Fatigue increases with high theta/beta ratio
        let tb_fatigue = (self.theta_beta_ratio / FATIGUE_THRESHOLD).clamp(0.0, 1.0);
        
        // Also consider absolute theta power (not implemented here, would need raw powers)
        self.fatigue_index = tb_fatigue;
    }

    /// Classify cognitive state based on ratios
    #[inline]
    pub fn classify_state(&mut self) {
        self.state = if self.theta_beta_ratio >= FATIGUE_THRESHOLD {
            CognitiveState::NeuralFatigue
        } else if self.theta_beta_ratio >= FLOW_STATE_THRESHOLD_HIGH {
            CognitiveState::DeepFlow
        } else if self.theta_beta_ratio >= ENGAGEMENT_OPTIMAL_MAX {
            CognitiveState::ModerateFlow
        } else if self.theta_beta_ratio >= ENGAGEMENT_OPTIMAL_MIN {
            CognitiveState::Neutral
        } else if self.theta_beta_ratio >= FLOW_STATE_THRESHOLD_LOW {
            CognitiveState::HighAlertness
        } else {
            CognitiveState::Drowsy
        };
    }

    /// Update all derived metrics
    #[inline]
    pub fn update(&mut self) {
        self.calculate_engagement();
        self.calculate_fatigue();
        self.classify_state();
    }
}

impl Default for FlowMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated flow state across multiple channels/regions
#[derive(Debug, Clone)]
pub struct AggregateFlowState {
    /// Global average theta/beta ratio
    pub global_theta_beta: f32,
    /// Regional flow metrics (frontal, parietal, occipital, temporal)
    pub frontal_metrics: FlowMetrics,
    pub parietal_metrics: FlowMetrics,
    pub occipital_metrics: FlowMetrics,
    pub temporal_metrics: FlowMetrics,
    /// Overall population engagement (for multi-user scenarios)
    pub population_engagement: f32,
    /// Population fatigue level
    pub population_fatigue: f32,
    /// Number of users/channels contributing
    pub sample_count: usize,
    /// Timestamp
    pub timestamp_ns: u64,
}

impl AggregateFlowState {
    #[inline]
    pub const fn new() -> Self {
        Self {
            global_theta_beta: 0.0,
            frontal_metrics: FlowMetrics::new(),
            parietal_metrics: FlowMetrics::new(),
            occipital_metrics: FlowMetrics::new(),
            temporal_metrics: FlowMetrics::new(),
            population_engagement: 0.0,
            population_fatigue: 0.0,
            sample_count: 0,
            timestamp_ns: 0,
        }
    }

    /// Aggregate metrics from multiple channels
    pub fn aggregate(&mut self, channel_metrics: &[FlowMetrics], channel_regions: &[usize]) {
        if channel_metrics.is_empty() {
            return;
        }

        let mut sum_tb = 0.0f32;
        let mut sum_engagement = 0.0f32;
        let mut sum_fatigue = 0.0f32;
        let mut regional_counts = [0usize; 4]; // 0=frontal, 1=parietal, 2=occipital, 3=temporal

        for (i, metrics) in channel_metrics.iter().enumerate() {
            sum_tb += metrics.theta_beta_ratio;
            sum_engagement += metrics.engagement_score;
            sum_fatigue += metrics.fatigue_index;

            // Assign to regional bucket
            let region = channel_regions.get(i).copied().unwrap_or(0);
            match region {
                0 => {
                    self.frontal_metrics = *metrics;
                    regional_counts[0] += 1;
                }
                1 => {
                    self.parietal_metrics = *metrics;
                    regional_counts[1] += 1;
                }
                2 => {
                    self.occipital_metrics = *metrics;
                    regional_counts[2] += 1;
                }
                3 => {
                    self.temporal_metrics = *metrics;
                    regional_counts[3] += 1;
                }
                _ => {}
            }
        }

        let n = channel_metrics.len() as f32;
        self.global_theta_beta = sum_tb / n;
        self.population_engagement = sum_engagement / n;
        self.population_fatigue = sum_fatigue / n;
        self.sample_count = channel_metrics.len();
    }

    /// Determine overall cognitive state from aggregated data
    pub fn determine_dominant_state(&self) -> CognitiveState {
        // Weight frontal regions more heavily for cognitive state
        let frontal_weight = 0.4;
        let parietal_weight = 0.25;
        let temporal_weight = 0.2;
        let occipital_weight = 0.15;

        let weighted_tb = 
            self.frontal_metrics.theta_beta_ratio * frontal_weight +
            self.parietal_metrics.theta_beta_ratio * parietal_weight +
            self.temporal_metrics.theta_beta_ratio * temporal_weight +
            self.occipital_metrics.theta_beta_ratio * occipital_weight;

        if weighted_tb >= FATIGUE_THRESHOLD {
            CognitiveState::NeuralFatigue
        } else if weighted_tb >= FLOW_STATE_THRESHOLD_HIGH {
            CognitiveState::DeepFlow
        } else if weighted_tb >= ENGAGEMENT_OPTIMAL_MAX {
            CognitiveState::ModerateFlow
        } else if weighted_tb >= ENGAGEMENT_OPTIMAL_MIN {
            CognitiveState::Neutral
        } else if weighted_tb >= FLOW_STATE_THRESHOLD_LOW {
            CognitiveState::HighAlertness
        } else {
            CognitiveState::Drowsy
        }
    }
}

impl Default for AggregateFlowState {
    fn default() -> Self {
        Self::new()
    }
}

/// Main Flow State Decoder engine
pub struct FlowStateDecoder {
    /// EEG processor reference
    eeg_processor: SimdEegProcessor,
    /// ICA artifact rejection
    ica_engine: LockFreeIcaArtifactRejection,
    /// Per-channel flow metrics
    channel_metrics: [FlowMetrics; 256],
    /// Aggregate state
    aggregate_state: AggregateFlowState,
    /// Active channel count
    active_channels: usize,
    /// Moving average window for smoothing
    ma_window_size: usize,
    /// Moving average buffers
    tb_ma_buffers: [[f32; 64]; 256],
    tb_ma_indices: [usize; 256],
}

impl FlowStateDecoder {
    /// Create new flow state decoder
    pub fn new() -> Self {
        Self {
            eeg_processor: SimdEegProcessor::new(),
            ica_engine: LockFreeIcaArtifactRejection::new(),
            channel_metrics: [FlowMetrics::new(); 256],
            aggregate_state: AggregateFlowState::new(),
            active_channels: 0,
            ma_window_size: 32,
            tb_ma_buffers: [[0.0; 64]; 256],
            tb_ma_indices: [0; 256],
        }
    }

    /// Initialize decoder with configuration
    pub fn init(&mut self, num_channels: usize, sample_rate: u32) -> Result<(), &'static str> {
        if num_channels > 256 {
            return Err("Channel count exceeds maximum");
        }

        // Initialize EEG processor
        use super::simd_eeg_stream::EegChannel;
        let channels = vec![EegChannel::default(); num_channels];
        self.eeg_processor.init(&channels, sample_rate)
            .map_err(|_| "Failed to initialize EEG processor")?;

        // Initialize ICA engine
        self.ica_engine.init(num_channels, sample_rate)
            .map_err(|_| "Failed to initialize ICA engine")?;

        self.active_channels = num_channels;
        Ok(())
    }

    /// Process EEG data and compute flow metrics
    pub fn process_and_decode(&mut self, raw_data: &[f32], timestamp_ns: u64) -> Result<&AggregateFlowState, &'static str> {
        if raw_data.len() < self.active_channels {
            return Err("Insufficient data");
        }

        // Step 1: Pre-filter EMG artifacts (critical!)
        self.ica_engine.prefilter_emg(raw_data, self.active_channels)
            .map_err(|_| "EMG prefiltering failed")?;

        // Step 2: Run ICA to separate components
        let filtered_data: Vec<f32> = self.ica_engine.data_buffer[..raw_data.len()].to_vec();
        self.ica_engine.run_fast_ica(50).ok();

        // Step 3: Classify and remove artifact components
        if let Ok(detection) = self.ica_engine.classify_artifacts(&filtered_data, timestamp_ns) {
            if detection.is_significant(0.5) {
                // Artifacts detected - could remove components here
            }
        }

        // Step 4: Process through EEG processor for spectral powers
        use super::simd_eeg_stream::EegStreamBuffer;
        let mut buffer = EegStreamBuffer::new();
        if buffer.init(self.active_channels, 1000).is_ok() {
            // Push data in batches
            for batch_start in (0..filtered_data.len()).step_by(self.active_channels) {
                let batch_end = core::cmp::min(batch_start + self.active_channels, filtered_data.len());
                if batch_end - batch_start == self.active_channels {
                    let _ = buffer.push_batch(&filtered_data[batch_start..batch_end], timestamp_ns);
                }
            }

            if buffer.is_ready() {
                let _ = self.eeg_processor.process_buffer(&buffer);
            }
        }

        // Step 5: Calculate flow metrics per channel
        let mut channel_metrics_slice = Vec::new();
        for ch in 0..self.active_channels {
            if let Ok(powers) = self.eeg_processor.get_spectral_powers(ch) {
                let theta = powers[1];
                let beta = powers[3];
                let alpha = powers[2];
                let gamma = powers[4];

                // Check for EMG contamination
                if let Ok(true) = self.eeg_processor.is_emg_contaminated(ch, 0.3) {
                    // Skip contaminated channels
                    continue;
                }

                let tb_ratio = if beta > 1e-10 { theta / beta } else { 0.0 };
                let at_ratio = if theta > 1e-10 { alpha / theta } else { 0.0 };
                let gb_ratio = if beta > 1e-10 { gamma / beta } else { 0.0 };

                // Apply moving average smoothing
                let smoothed_tb = self.apply_moving_average(ch, tb_ratio);

                let mut metrics = FlowMetrics {
                    theta_beta_ratio: smoothed_tb,
                    alpha_theta_ratio: at_ratio,
                    gamma_beta_ratio: gb_ratio,
                    engagement_score: 0.0,
                    fatigue_index: 0.0,
                    cognitive_load: 0.0,
                    state: CognitiveState::Neutral,
                    timestamp_ns,
                    channel_id: ch,
                };

                metrics.update();
                self.channel_metrics[ch] = metrics;
                channel_metrics_slice.push(metrics);
            }
        }

        // Step 6: Aggregate metrics
        let regions = vec![0; channel_metrics_slice.len()]; // Simplified: all frontal
        self.aggregate_state.aggregate(&channel_metrics_slice, &regions);
        self.aggregate_state.timestamp_ns = timestamp_ns;

        Ok(&self.aggregate_state)
    }

    /// Apply moving average smoothing to theta/beta ratio
    fn apply_moving_average(&mut self, channel: usize, value: f32) -> f32 {
        let idx = self.tb_ma_indices[channel];
        self.tb_ma_buffers[channel][idx] = value;
        self.tb_ma_indices[channel] = (idx + 1) % self.ma_window_size.min(64);

        let mut sum = 0.0f32;
        let mut count = 0usize;
        for i in 0..self.ma_window_size.min(64) {
            if self.tb_ma_buffers[channel][i] > 0.0 {
                sum += self.tb_ma_buffers[channel][i];
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f32
        } else {
            value
        }
    }

    /// Get current flow state classification
    pub fn get_flow_state(&self) -> CognitiveState {
        self.aggregate_state.determine_dominant_state()
    }

    /// Get engagement score for trading signals
    pub fn get_engagement_signal(&self) -> f32 {
        self.aggregate_state.population_engagement
    }

    /// Get fatigue level for churn prediction
    pub fn get_fatigue_signal(&self) -> f32 {
        self.aggregate_state.population_fatigue
    }

    /// Reset all state
    pub fn reset(&mut self) {
        for i in 0..self.active_channels {
            self.tb_ma_buffers[i].fill(0.0);
            self.tb_ma_indices[i] = 0;
            self.channel_metrics[i] = FlowMetrics::new();
        }
        self.aggregate_state = AggregateFlowState::new();
        self.eeg_processor.reset_filters();
        self.ica_engine.reset_filters();
    }
}

impl Default for FlowStateDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_metrics_default() {
        let metrics = FlowMetrics::new();
        assert_eq!(metrics.theta_beta_ratio, 0.0);
        assert_eq!(metrics.state, CognitiveState::Neutral);
    }

    #[test]
    fn test_cognitive_state_classification() {
        let mut metrics = FlowMetrics::new();
        metrics.theta_beta_ratio = 3.5;
        metrics.classify_state();
        assert_eq!(metrics.state, CognitiveState::NeuralFatigue);

        metrics.theta_beta_ratio = 2.0;
        metrics.classify_state();
        assert_eq!(metrics.state, CognitiveState::ModerateFlow);
    }

    #[test]
    fn test_aggregate_state_default() {
        let state = AggregateFlowState::new();
        assert_eq!(state.sample_count, 0);
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = FlowStateDecoder::new();
        assert_eq!(decoder.active_channels, 0);
    }
}
