//! Markov Blanket Mapper for Active Inference
//! 
//! Defines the sensory and active states of the organoid, mapping
//! order book imbalances to sensory inputs and spike rates to motor outputs.

use crate::inference::variational_free_energy::{GenerativeModel, FreeEnergyError};

/// Maximum number of sensory channels (order book features)
pub const MAX_SENSORY_CHANNELS: usize = 128;

/// Maximum number of active channels (motor outputs / electrodes)
pub const MAX_ACTIVE_CHANNELS: usize = 64;

/// Maximum number of hidden states in the Markov blanket
pub const MAX_HIDDEN_STATES: usize = 256;

/// Sensory modality types
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum SensoryModality {
    OrderBookImbalance = 0,
    TradeFlow = 1,
    PriceVelocity = 2,
    Volatility = 3,
    SpreadWidth = 4,
    DepthImbalance = 5,
    Custom = 7,
}

/// Motor modality types
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum MotorModality {
    SpikeRate = 0,
    BurstPattern = 1,
    LfpPower = 2,
    Synchrony = 3,
    Custom = 7,
}

/// Error types for Markov blanket operations
#[derive(Debug, Clone, Copy)]
pub enum MarkovBlanketError {
    InvalidChannelIndex,
    ModalityMismatch,
    MappingNotConfigured,
    BufferOverflow,
    InitializationFailed,
}

/// Sensory input buffer (zero-copy, fixed size)
#[repr(C, align(64))]
pub struct SensoryBuffer {
    /// Current sensory values
    values: [f32; MAX_SENSORY_CHANNELS],
    /// Previous values for temporal derivatives
    previous: [f32; MAX_SENSORY_CHANNELS],
    /// Valid channel count
    num_channels: usize,
    /// Precision weights per channel
    precision: [f32; MAX_SENSORY_CHANNELS],
}

impl SensoryBuffer {
    /// Create a new sensory buffer
    pub fn new(num_channels: usize) -> Self {
        Self {
            values: [0.0; MAX_SENSORY_CHANNELS],
            previous: [0.0; MAX_SENSORY_CHANNELS],
            num_channels: num_channels.min(MAX_SENSORY_CHANNELS),
            precision: [1.0; MAX_SENSORY_CHANNELS],
        }
    }

    /// Update a sensory value with precision weighting
    #[inline]
    pub fn update(&mut self, channel: usize, value: f32, precision: f32) -> Result<(), MarkovBlanketError> {
        if channel >= self.num_channels {
            return Err(MarkovBlanketError::InvalidChannelIndex);
        }

        self.previous[channel] = self.values[channel];
        self.values[channel] = value;
        self.precision[channel] = precision.max(0.0).min(100.0);

        Ok(())
    }

    /// Get current value for a channel
    #[inline]
    pub fn get(&self, channel: usize) -> Option<f32> {
        if channel < self.num_channels {
            Some(self.values[channel])
        } else {
            None
        }
    }

    /// Get prediction error (value - prediction)
    #[inline]
    pub fn prediction_error(&self, channel: usize, prediction: f32) -> Option<f32> {
        self.get(channel).map(|v| v - prediction)
    }

    /// Get precision-weighted prediction error
    #[inline]
    pub fn weighted_prediction_error(&self, channel: usize, prediction: f32) -> Option<f32> {
        self.prediction_error(channel, prediction)
            .map(|e| e * self.precision[channel])
    }

    /// Compute temporal derivative
    #[inline]
    pub fn temporal_derivative(&self, channel: usize) -> Option<f32> {
        if channel < self.num_channels {
            Some(self.values[channel] - self.previous[channel])
        } else {
            None
        }
    }

    /// Get all sensory values as slice
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.values[..self.num_channels]
    }

    /// Set precision for a channel (attention modulation)
    #[inline]
    pub fn set_precision(&mut self, channel: usize, precision: f32) -> Result<(), MarkovBlanketError> {
        if channel >= self.num_channels {
            return Err(MarkovBlanketError::InvalidChannelIndex);
        }
        self.precision[channel] = precision.max(0.0).min(100.0);
        Ok(())
    }

    /// Get precision for a channel
    #[inline]
    pub fn get_precision(&self, channel: usize) -> Option<f32> {
        if channel < self.num_channels {
            Some(self.precision[channel])
        } else {
            None
        }
    }
}

/// Active output buffer (motor commands to organoid)
#[repr(C, align(64))]
pub struct ActiveBuffer {
    /// Current motor output values
    values: [f32; MAX_ACTIVE_CHANNELS],
    /// Target values from policy selection
    targets: [f32; MAX_ACTIVE_CHANNELS],
    /// Valid channel count
    num_channels: usize,
    /// Gain factors per channel
    gains: [f32; MAX_ACTIVE_CHANNELS],
}

impl ActiveBuffer {
    /// Create a new active buffer
    pub fn new(num_channels: usize) -> Self {
        Self {
            values: [0.0; MAX_ACTIVE_CHANNELS],
            targets: [0.0; MAX_ACTIVE_CHANNELS],
            num_channels: num_channels.min(MAX_ACTIVE_CHANNELS),
            gains: [1.0; MAX_ACTIVE_CHANNELS],
        }
    }

    /// Set target value for an active channel
    #[inline]
    pub fn set_target(&mut self, channel: usize, target: f32) -> Result<(), MarkovBlanketError> {
        if channel >= self.num_channels {
            return Err(MarkovBlanketError::InvalidChannelIndex);
        }
        self.targets[channel] = target;
        Ok(())
    }

    /// Update actual output value
    #[inline]
    pub fn set_value(&mut self, channel: usize, value: f32) -> Result<(), MarkovBlanketError> {
        if channel >= self.num_channels {
            return Err(MarkovBlanketError::InvalidChannelIndex);
        }
        self.values[channel] = value;
        Ok(())
    }

    /// Get current value
    #[inline]
    pub fn get(&self, channel: usize) -> Option<f32> {
        if channel < self.num_channels {
            Some(self.values[channel])
        } else {
            None
        }
    }

    /// Get target value
    #[inline]
    pub fn get_target(&self, channel: usize) -> Option<f32> {
        if channel < self.num_channels {
            Some(self.targets[channel])
        } else {
            None
        }
    }

    /// Compute prediction (expected outcome given action)
    #[inline]
    pub fn compute_prediction(&self, channel: usize) -> Option<f32> {
        if channel < self.num_channels {
            Some(self.values[channel] * self.gains[channel])
        } else {
            None
        }
    }

    /// Set gain for a channel
    #[inline]
    pub fn set_gain(&mut self, channel: usize, gain: f32) -> Result<(), MarkovBlanketError> {
        if channel >= self.num_channels {
            return Err(MarkovBlanketError::InvalidChannelIndex);
        }
        self.gains[channel] = gain.max(0.0).min(10.0);
        Ok(())
    }
}

/// Markov Blanket definition separating internal/external states
#[repr(C, align(64))]
pub struct MarkovBlanket {
    /// Sensory states (boundary inputs)
    sensory: SensoryBuffer,
    /// Active states (boundary outputs)
    active: ActiveBuffer,
    /// Hidden state mappings
    hidden_mappings: [usize; MAX_HIDDEN_STATES],
    /// Number of hidden states
    num_hidden: usize,
    /// Sensory modalities per channel
    sensory_modalities: [SensoryModality; MAX_SENSORY_CHANNELS],
    /// Motor modalities per channel
    motor_modalities: [MotorModality; MAX_ACTIVE_CHANNELS],
}

impl MarkovBlanket {
    /// Create a new Markov blanket
    pub fn new(num_sensory: usize, num_active: usize, num_hidden: usize) -> Self {
        Self {
            sensory: SensoryBuffer::new(num_sensory),
            active: ActiveBuffer::new(num_active),
            hidden_mappings: [0; MAX_HIDDEN_STATES],
            num_hidden: num_hidden.min(MAX_HIDDEN_STATES),
            sensory_modalities: [SensoryModality::Custom; MAX_SENSORY_CHANNELS],
            motor_modalities: [MotorModality::Custom; MAX_ACTIVE_CHANNELS],
        }
    }

    /// Configure sensory channel with modality
    pub fn configure_sensory(
        &mut self,
        channel: usize,
        modality: SensoryModality,
        default_precision: f32,
    ) -> Result<(), MarkovBlanketError> {
        if channel >= self.sensory.num_channels {
            return Err(MarkovBlanketError::InvalidChannelIndex);
        }
        
        self.sensory_modalities[channel] = modality;
        self.sensory.set_precision(channel, default_precision)?;
        Ok(())
    }

    /// Configure motor channel with modality
    pub fn configure_motor(
        &mut self,
        channel: usize,
        modality: MotorModality,
        default_gain: f32,
    ) -> Result<(), MarkovBlanketError> {
        if channel >= self.active.num_channels {
            return Err(MarkovBlanketError::InvalidChannelIndex);
        }
        
        self.motor_modalities[channel] = modality;
        self.active.set_gain(channel, default_gain)?;
        Ok(())
    }

    /// Map order book imbalance to sensory input
    pub fn map_order_book_imbalance(
        &mut self,
        channel: usize,
        bid_volume: f64,
        ask_volume: f64,
    ) -> Result<(), MarkovBlanketError> {
        // Compute normalized imbalance: (bid - ask) / (bid + ask)
        let sum = bid_volume + ask_volume;
        let imbalance = if sum > 0.0 {
            ((bid_volume - ask_volume) / sum) as f32
        } else {
            0.0
        };

        self.sensory.update(channel, imbalance, 1.0)?;
        self.sensory_modalities[channel] = SensoryModality::OrderBookImbalance;
        Ok(())
    }

    /// Map spike rate to motor output
    pub fn map_spike_rate(
        &mut self,
        channel: usize,
        spike_count: u32,
        time_window_ms: u32,
    ) -> Result<f32, MarkovBlanketError> {
        if channel >= self.active.num_channels {
            return Err(MarkovBlanketError::InvalidChannelIndex);
        }

        // Compute firing rate in Hz
        let rate = if time_window_ms > 0 {
            (spike_count as f32 * 1000.0) / time_window_ms as f32
        } else {
            0.0
        };

        self.active.set_value(channel, rate)?;
        self.motor_modalities[channel] = MotorModality::SpikeRate;
        Ok(rate)
    }

    /// Get sensory prediction error for a channel
    pub fn get_sensory_prediction_error(
        &self,
        channel: usize,
        predicted_value: f32,
    ) -> Option<(f32, f32)> {
        let error = self.sensory.prediction_error(channel, predicted_value)?;
        let weighted_error = self.sensory.weighted_prediction_error(channel, predicted_value)?;
        Some((error, weighted_error))
    }

    /// Set precision based on market volatility (attention modulation)
    pub fn modulate_precision_for_volatility(
        &mut self,
        volatility: f32,
        base_precision: f32,
    ) {
        // Higher volatility -> higher precision on sensory inputs
        // This implements "paying attention" during uncertain times
        let precision_factor = 1.0 + volatility.min(10.0) * 0.5;
        
        for channel in 0..self.sensory.num_channels {
            let new_precision = base_precision * precision_factor;
            let _ = self.sensory.set_precision(channel, new_precision);
        }
    }

    /// Convert Markov blanket to generative model configuration
    pub fn to_generative_model(&self) -> Result<GenerativeModel, FreeEnergyError> {
        GenerativeModel::new(
            self.num_hidden,
            self.sensory.num_channels,
            self.active.num_channels,
        )
    }

    /// Get reference to sensory buffer
    #[inline]
    pub fn sensory(&self) -> &SensoryBuffer {
        &self.sensory
    }

    /// Get mutable reference to sensory buffer
    #[inline]
    pub fn sensory_mut(&mut self) -> &mut SensoryBuffer {
        &mut self.sensory
    }

    /// Get reference to active buffer
    #[inline]
    pub fn active(&self) -> &ActiveBuffer {
        &self.active
    }

    /// Get mutable reference to active buffer
    #[inline]
    pub fn active_mut(&mut self) -> &mut ActiveBuffer {
        &mut self.active
    }
}

/// Cluster mapper for grouping electrodes into functional regions
pub struct ElectrodeClusterMapper {
    /// Cluster assignments per electrode
    cluster_ids: [u8; 256],
    /// Number of electrodes per cluster
    cluster_sizes: [usize; 32],
    /// Mean spike rate per cluster
    cluster_rates: [f32; 32],
    /// Total clusters
    num_clusters: usize,
}

impl ElectrodeClusterMapper {
    /// Create a new cluster mapper
    pub fn new(num_clusters: usize) -> Self {
        Self {
            cluster_ids: [0; 256],
            cluster_sizes: [0; 32],
            cluster_rates: [0.0; 32],
            num_clusters: num_clusters.min(32),
        }
    }

    /// Assign electrode to cluster
    pub fn assign_electrode(&mut self, electrode_id: u8, cluster_id: u8) -> Result<(), MarkovBlanketError> {
        if cluster_id as usize >= self.num_clusters {
            return Err(MarkovBlanketError::InvalidChannelIndex);
        }

        // Decrement old cluster size if reassigning
        let old_cluster = self.cluster_ids[electrode_id as usize] as usize;
        if old_cluster < 32 && self.cluster_sizes[old_cluster] > 0 {
            self.cluster_sizes[old_cluster] -= 1;
        }

        // Assign to new cluster
        self.cluster_ids[electrode_id as usize] = cluster_id;
        self.cluster_sizes[cluster_id as usize] += 1;

        Ok(())
    }

    /// Update cluster mean firing rate
    pub fn update_cluster_rate(
        &mut self,
        cluster_id: usize,
        electrode_rates: &[f32],
    ) -> Result<(), MarkovBlanketError> {
        if cluster_id >= self.num_clusters {
            return Err(MarkovBlanketError::InvalidChannelIndex);
        }

        let mut sum = 0.0;
        let mut count = 0;

        for (elec_id, &cluster) in self.cluster_ids.iter().enumerate() {
            if cluster as usize == cluster_id && elec_id < electrode_rates.len() {
                sum += electrode_rates[elec_id];
                count += 1;
            }
        }

        self.cluster_rates[cluster_id] = if count > 0 {
            sum / count as f32
        } else {
            0.0
        };

        Ok(())
    }

    /// Get cluster firing rate
    #[inline]
    pub fn get_cluster_rate(&self, cluster_id: usize) -> Option<f32> {
        if cluster_id < self.num_clusters {
            Some(self.cluster_rates[cluster_id])
        } else {
            None
        }
    }

    /// Get cluster size
    #[inline]
    pub fn get_cluster_size(&self, cluster_id: usize) -> Option<usize> {
        if cluster_id < self.num_clusters {
            Some(self.cluster_sizes[cluster_id])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensory_buffer_update() {
        let mut buffer = SensoryBuffer::new(8);
        
        buffer.update(0, 0.5, 1.0).unwrap();
        assert_eq!(buffer.get(0), Some(0.5));
        
        // Test precision weighting
        let error = buffer.weighted_prediction_error(0, 0.3).unwrap();
        assert!((error - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_markov_blanket_order_book_mapping() {
        let mut blanket = MarkovBlanket::new(8, 4, 16);
        
        blanket.configure_sensory(0, SensoryModality::OrderBookImbalance, 1.0).unwrap();
        
        let result = blanket.map_order_book_imbalance(0, 100.0, 50.0);
        assert!(result.is_ok());
        
        // Imbalance should be (100-50)/(100+50) = 0.333...
        let value = blanket.sensory().get(0).unwrap();
        assert!((value - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_spike_rate_mapping() {
        let mut blanket = MarkovBlanket::new(4, 8, 16);
        
        blanket.configure_motor(0, MotorModality::SpikeRate, 1.0).unwrap();
        
        let rate = blanket.map_spike_rate(0, 100, 1000).unwrap();
        assert!((rate - 100.0).abs() < 1e-6); // 100 spikes in 1000ms = 100 Hz
    }

    #[test]
    fn test_volatility_precision_modulation() {
        let mut blanket = MarkovBlanket::new(8, 4, 16);
        
        let base_precision = 1.0;
        blanket.modulate_precision_for_volatility(0.0, base_precision);
        
        let precision_low = blanket.sensory().get_precision(0).unwrap();
        assert!((precision_low - 1.0).abs() < 1e-6);
        
        blanket.modulate_precision_for_volatility(10.0, base_precision);
        let precision_high = blanket.sensory().get_precision(0).unwrap();
        assert!(precision_high > precision_low);
    }
}
