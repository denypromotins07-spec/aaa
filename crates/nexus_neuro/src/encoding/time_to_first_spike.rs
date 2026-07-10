//! Time-to-First-Spike (TTFS) Encoder
//! 
//! Translates continuous values (e.g., L2/L3 order book imbalances) into
//! precise spike latencies. Higher conviction/stronger input results in
//! earlier spikes, enabling efficient temporal coding for SNNs.

use std::sync::atomic::{AtomicU64, Ordering};

/// Default minimum latency in microseconds
pub const DEFAULT_MIN_LATENCY_US: u64 = 100;

/// Default maximum latency in microseconds
pub const DEFAULT_MAX_LATENCY_US: u64 = 10_000;

/// Encoded spike latency result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpikeLatency {
    /// Latency in microseconds from stimulus onset to spike
    pub latency_us: u64,
    /// Original input value (for reference)
    pub input_value: f64,
    /// Whether this represents a valid spike (false = no spike)
    pub valid: bool,
}

impl SpikeLatency {
    #[inline]
    pub fn new(latency_us: u64, input_value: f64, valid: bool) -> Self {
        Self {
            latency_us,
            input_value,
            valid,
        }
    }

    /// Create an invalid (no-spike) latency
    #[inline]
    pub fn no_spike() -> Self {
        Self {
            latency_us: u64::MAX,
            input_value: 0.0,
            valid: false,
        }
    }
}

/// Time-to-First-Spike encoder configuration
#[derive(Debug, Clone, Copy)]
pub struct TtfsConfig {
    /// Minimum latency (strongest input)
    pub min_latency_us: u64,
    /// Maximum latency (weakest input above threshold)
    pub max_latency_us: u64,
    /// Threshold below which no spike is generated
    pub threshold: f64,
    /// Saturation point (inputs above this get min latency)
    pub saturation: f64,
    /// Encoding curve: 'linear', 'log', or 'exp'
    pub encoding_curve: EncodingCurve,
}

impl Default for TtfsConfig {
    #[inline]
    fn default() -> Self {
        Self {
            min_latency_us: DEFAULT_MIN_LATENCY_US,
            max_latency_us: DEFAULT_MAX_LATENCY_US,
            threshold: 0.01,
            saturation: 1.0,
            encoding_curve: EncodingCurve::Linear,
        }
    }
}

/// Encoding curve types for TTFS mapping
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingCurve {
    /// Linear mapping: latency = max - (input * (max-min) / saturation)
    Linear,
    /// Logarithmic: more resolution for small inputs
    Logarithmic,
    /// Exponential: more resolution for large inputs
    Exponential,
    /// Sigmoidal: balanced resolution with soft thresholds
    Sigmoidal,
}

/// Time-to-First-Spike Encoder
pub struct TtfsEncoder {
    config: TtfsConfig,
    /// Base timestamp for latency calculations
    base_timestamp_us: AtomicU64,
    /// Total encodings performed
    encoding_count: AtomicU64,
    /// Spikes suppressed (below threshold)
    suppressed_count: AtomicU64,
}

impl TtfsEncoder {
    /// Create a new TTFS encoder with default configuration
    #[inline]
    pub fn new() -> Self {
        Self::with_config(TtfsConfig::default())
    }

    /// Create a new TTFS encoder with custom configuration
    #[inline]
    pub fn with_config(config: TtfsConfig) -> Self {
        Self {
            config,
            base_timestamp_us: AtomicU64::new(0),
            encoding_count: AtomicU64::new(0),
            suppressed_count: AtomicU64::new(0),
        }
    }

    /// Set the base timestamp for latency calculations
    #[inline]
    pub fn set_base_timestamp(&self, timestamp_us: u64) {
        self.base_timestamp_us.store(timestamp_us, Ordering::Release);
    }

    /// Encode a single input value to spike latency
    #[inline]
    pub fn encode(&self, input_value: f64) -> SpikeLatency {
        self.encoding_count.fetch_add(1, Ordering::Relaxed);

        // Check threshold
        if input_value.abs() < self.config.threshold {
            self.suppressed_count.fetch_add(1, Ordering::Relaxed);
            return SpikeLatency::no_spike();
        }

        // Use absolute value for latency calculation
        let abs_input = input_value.abs();

        // Calculate latency based on encoding curve
        let latency_us = match self.config.encoding_curve {
            EncodingCurve::Linear => self.encode_linear(abs_input),
            EncodingCurve::Logarithmic => self.encode_logarithmic(abs_input),
            EncodingCurve::Exponential => self.encode_exponential(abs_input),
            EncodingCurve::Sigmoidal => self.encode_sigmoidal(abs_input),
        };

        SpikeLatency::new(latency_us, input_value, true)
    }

    /// Encode multiple input values (batch processing)
    #[inline]
    pub fn encode_batch(&self, inputs: &[f64]) -> Vec<SpikeLatency> {
        inputs.iter().map(|&v| self.encode(v)).collect()
    }

    /// Linear encoding: latency decreases linearly with input strength
    #[inline]
    fn encode_linear(&self, input: f64) -> u64 {
        let normalized = (input / self.config.saturation).clamp(0.0, 1.0);
        let range = self.config.max_latency_us.saturating_sub(self.config.min_latency_us);
        self.config.max_latency_us - ((normalized * range as f64) as u64)
    }

    /// Logarithmic encoding: finer resolution for weak inputs
    #[inline]
    fn encode_logarithmic(&self, input: f64) -> u64 {
        let normalized = (input / self.config.saturation).clamp(0.001, 1.0);
        let log_input = normalized.ln() / self.config.saturation.ln();
        let range = self.config.max_latency_us.saturating_sub(self.config.min_latency_us);
        self.config.max_latency_us - ((log_input * range as f64) as u64)
    }

    /// Exponential encoding: finer resolution for strong inputs
    #[inline]
    fn encode_exponential(&self, input: f64) -> u64 {
        let normalized = (input / self.config.saturation).clamp(0.0, 1.0);
        let exp_input = (normalized * 3.0).exp() - 1.0;
        let exp_max = 3.0f64.exp() - 1.0;
        let scaled = exp_input / exp_max;
        let range = self.config.max_latency_us.saturating_sub(self.config.min_latency_us);
        self.config.max_latency_us - ((scaled * range as f64) as u64)
    }

    /// Sigmoidal encoding: balanced with soft saturation
    #[inline]
    fn encode_sigmoidal(&self, input: f64) -> u64 {
        let normalized = (input / self.config.saturation).clamp(-2.0, 2.0);
        // Sigmoid: 1 / (1 + exp(-x))
        let sigmoid = 1.0 / (1.0 + (-normalized * 3.0).exp());
        let range = self.config.max_latency_us.saturating_sub(self.config.min_latency_us);
        self.config.max_latency_us - ((sigmoid * range as f64) as u64)
    }

    /// Encode order book imbalance to spike latency
    /// Imbalance ranges from -1.0 (all asks) to +1.0 (all bids)
    #[inline]
    pub fn encode_orderbook_imbalance(&self, imbalance: f64) -> SpikeLatency {
        // Convert imbalance to conviction (absolute value)
        let conviction = imbalance.abs();
        
        // Encode based on conviction strength
        let latency = self.encode(conviction);
        
        // For negative imbalance, add polarity marker to latency
        if imbalance < 0.0 && latency.valid {
            // Mark as negative by adding offset (decoder should handle this)
            SpikeLatency::new(
                latency.latency_us,
                imbalance,
                latency.valid,
            )
        } else {
            latency
        }
    }

    /// Encode L2/L3 microstructure features
    /// Combines multiple features into a single spike latency
    #[inline]
    pub fn encode_microstructure(
        &self,
        imbalance: f64,
        spread_bps: f64,
        volume_imbalance: f64,
    ) -> SpikeLatency {
        // Weighted combination of features
        let weighted_conviction = 
            imbalance.abs() * 0.5 +
            (spread_bps / 100.0).clamp(0.0, 1.0) * 0.3 +
            volume_imbalance.abs() * 0.2;

        self.encode(weighted_conviction)
    }

    /// Get statistics
    #[inline]
    pub fn stats(&self) -> TtfsStats {
        TtfsStats {
            encoding_count: self.encoding_count.load(Ordering::Relaxed),
            suppressed_count: self.suppressed_count.load(Ordering::Relaxed),
            base_timestamp_us: self.base_timestamp_us.load(Ordering::Acquire),
        }
    }

    /// Reset encoder state
    #[inline]
    pub fn reset(&self) {
        self.encoding_count.store(0, Ordering::Relaxed);
        self.suppressed_count.store(0, Ordering::Relaxed);
        self.base_timestamp_us.store(0, Ordering::Release);
    }
}

impl Default for TtfsEncoder {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// TTFS encoder statistics
#[derive(Debug, Clone, Copy)]
pub struct TtfsStats {
    pub encoding_count: u64,
    pub suppressed_count: u64,
    pub base_timestamp_us: u64,
}

/// Multi-channel TTFS encoder for parallel feature encoding
pub struct MultiChannelTtfsEncoder {
    /// Per-channel encoders
    channels: Vec<TtfsEncoder>,
    /// Channel count
    channel_count: usize,
}

impl MultiChannelTtfsEncoder {
    #[inline]
    pub fn new(channel_count: usize, config: TtfsConfig) -> Self {
        let channels = (0..channel_count)
            .map(|_| TtfsEncoder::with_config(config))
            .collect();

        Self {
            channels,
            channel_count,
        }
    }

    /// Encode a vector of features (one per channel)
    #[inline]
    pub fn encode_vector(&self, features: &[f64]) -> Vec<SpikeLatency> {
        features
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                if i < self.channels.len() {
                    self.channels[i].encode(f)
                } else {
                    SpikeLatency::no_spike()
                }
            })
            .collect()
    }

    /// Get number of channels
    #[inline]
    pub fn channel_count(&self) -> usize {
        self.channel_count
    }

    /// Set base timestamp for all channels
    #[inline]
    pub fn sync_timestamps(&self, timestamp_us: u64) {
        for channel in &self.channels {
            channel.set_base_timestamp(timestamp_us);
        }
    }
}

/// Rank-based TTFS encoder for relative ordering
pub struct RankOrderEncoder {
    config: TtfsConfig,
    /// Number of input channels
    n_channels: usize,
}

impl RankOrderEncoder {
    #[inline]
    pub fn new(n_channels: usize, config: TtfsConfig) -> Self {
        Self {
            config,
            n_channels,
        }
    }

    /// Encode based on rank order of inputs
    /// Earlier spikes for higher-ranked (larger) inputs
    #[inline]
    pub fn encode_rank_order(&self, inputs: &[f64]) -> Vec<SpikeLatency> {
        if inputs.is_empty() {
            return vec![];
        }

        // Create indexed values for sorting
        let mut indexed: Vec<(usize, f64)> = inputs.iter().copied().enumerate().collect();
        
        // Sort by value descending (highest first = earliest spike)
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut latencies = vec![SpikeLatency::no_spike(); self.n_channels];
        let range = self.config.max_latency_us.saturating_sub(self.config.min_latency_us);

        for (rank, (original_idx, value)) in indexed.iter().enumerate() {
            if value.abs() >= self.config.threshold {
                // Earlier rank = earlier spike
                let latency = self.config.min_latency_us 
                    + ((rank as u64 * range) / (self.n_channels as u64).max(1));
                latencies[*original_idx] = SpikeLatency::new(latency, *value, true);
            }
        }

        latencies
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ttfs_encoder_linear() {
        let config = TtfsConfig {
            encoding_curve: EncodingCurve::Linear,
            ..Default::default()
        };
        let encoder = TtfsEncoder::with_config(config);

        // Strong input should produce short latency
        let strong = encoder.encode(0.9);
        assert!(strong.valid);
        assert!(strong.latency_us < 5000);

        // Weak input should produce longer latency
        let weak = encoder.encode(0.1);
        assert!(weak.valid);
        assert!(weak.latency_us > strong.latency_us);
    }

    #[test]
    fn test_ttfs_encoder_threshold() {
        let encoder = TtfsEncoder::new();
        
        // Below threshold should not spike
        let below = encoder.encode(0.005);
        assert!(!below.valid);

        // Above threshold should spike
        let above = encoder.encode(0.02);
        assert!(above.valid);
    }

    #[test]
    fn test_ttfs_encoder_curves() {
        let mut config = TtfsConfig::default();
        let encoder = TtfsEncoder::with_config(config);

        config.encoding_curve = EncodingCurve::Logarithmic;
        let log_encoder = TtfsEncoder::with_config(config);

        config.encoding_curve = EncodingCurve::Exponential;
        let exp_encoder = TtfsEncoder::with_config(config);

        // All should produce valid spikes for same input
        let input = 0.5;
        assert!(encoder.encode(input).valid);
        assert!(log_encoder.encode(input).valid);
        assert!(exp_encoder.encode(input).valid);
    }

    #[test]
    fn test_orderbook_imbalance_encoding() {
        let encoder = TtfsEncoder::new();

        // Strong bid imbalance
        let bid = encoder.encode_orderbook_imbalance(0.8);
        assert!(bid.valid);

        // Strong ask imbalance
        let ask = encoder.encode_orderbook_imbalance(-0.8);
        assert!(ask.valid);

        // Balanced book (should be suppressed)
        let balanced = encoder.encode_orderbook_imbalance(0.0);
        assert!(!balanced.valid);
    }

    #[test]
    fn test_multi_channel_encoder() {
        let encoder = MultiChannelTtfsEncoder::new(4, TtfsConfig::default());
        
        let features = [0.1, 0.5, 0.9, 0.01];
        let latencies = encoder.encode_vector(&features);

        assert_eq!(latencies.len(), 4);
        assert!(latencies[0].valid);
        assert!(latencies[1].valid);
        assert!(latencies[2].valid);
        // Last one might be below threshold
    }

    #[test]
    fn test_rank_order_encoding() {
        let encoder = RankOrderEncoder::new(4, TtfsConfig::default());
        
        let inputs = [0.3, 0.9, 0.1, 0.6];
        let latencies = encoder.encode_rank_order(&inputs);

        // Highest input (0.9 at index 1) should have shortest latency
        assert_eq!(latencies[1].latency_us, latencies.iter().map(|l| l.latency_us).min().unwrap());
    }
}
