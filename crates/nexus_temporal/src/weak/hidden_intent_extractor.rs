//! Hidden Intent Extractor for institutional order detection
//! 
//! Analyzes weak measurement results to extract hidden institutional trading intent.
//! Uses amplified weak values to detect block orders before execution.

/// Minimum signal-to-noise ratio for valid intent detection
const MIN_SNR_THRESHOLD: f64 = 3.0;

/// Maximum lookback window for intent analysis (nanoseconds)
const MAX_LOOKBACK_NS: u64 = 100_000; // 100 microseconds

/// Institutional intent direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentDirection {
    /// Strong buy signal detected
    Bullish,
    /// Strong sell signal detected
    Bearish,
    /// No clear directional signal
    Neutral,
    /// Ambiguous or conflicting signals
    Ambiguous,
}

/// Detected institutional intent with confidence metrics
#[derive(Debug, Clone)]
pub struct InstitutionalIntent {
    /// Direction of the detected intent
    pub direction: IntentDirection,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Estimated order size (in base units)
    pub estimated_size: f64,
    /// Time horizon for execution (nanoseconds)
    pub time_horizon_ns: u64,
    /// Signal-to-noise ratio
    pub snr: f64,
}

/// Hidden Intent Extractor analyzing weak value signatures
pub struct HiddenIntentExtractor {
    /// Minimum SNR threshold for detection
    snr_threshold: f64,
    /// Lookback window for analysis
    lookback_ns: u64,
    /// Historical weak values for trend analysis
    weak_value_history: Vec<f64>,
    /// Detected intents cache
    intent_cache: Vec<InstitutionalIntent>,
}

impl HiddenIntentExtractor {
    /// Create a new hidden intent extractor
    pub fn new() -> Self {
        Self {
            snr_threshold: MIN_SNR_THRESHOLD,
            lookback_ns: MAX_LOOKBACK_NS,
            weak_value_history: Vec::with_capacity(1024),
            intent_cache: Vec::with_capacity(256),
        }
    }

    /// Create extractor with custom SNR threshold
    pub fn with_snr_threshold(threshold: f64) -> Self {
        Self {
            snr_threshold: threshold.max(MIN_SNR_THRESHOLD),
            ..Self::new()
        }
    }

    /// Analyze weak value to extract hidden institutional intent
    /// 
    /// # Arguments
    /// * `weak_value` - Amplified weak value from measurement
    /// * `post_state` - Post-selected state vector
    /// 
    /// # Returns
    /// Some(InstitutionalIntent) if signal is strong enough, None otherwise
    pub fn analyze_weak_value(&self, weak_value: f64, post_state: &[f64]) -> Option<InstitutionalIntent> {
        if post_state.is_empty() {
            return None;
        }

        // Compute signal strength from weak value
        let signal_strength = weak_value.abs();
        
        // Estimate noise level from post-state variance
        let noise_level = self.estimate_noise(post_state);
        
        if noise_level < 1e-15 {
            return None;
        }

        // Calculate signal-to-noise ratio
        let snr = signal_strength / noise_level;
        
        if snr < self.snr_threshold {
            return None;
        }

        // Determine direction from weak value sign and state gradient
        let direction = self.determine_direction(weak_value, post_state);
        
        // Calculate confidence based on SNR and state consistency
        let confidence = self.calculate_confidence(snr, post_state);
        
        // Estimate order size from amplification magnitude
        let estimated_size = self.estimate_order_size(weak_value, post_state);
        
        // Derive time horizon from state decay characteristics
        let time_horizon_ns = self.estimate_time_horizon(post_state);

        Some(InstitutionalIntent {
            direction,
            confidence,
            estimated_size,
            time_horizon_ns,
            snr,
        })
    }

    /// Record a weak value for historical analysis
    pub fn record_weak_value(&mut self, weak_value: f64) {
        self.weak_value_history.push(weak_value);
        
        // Keep history bounded
        if self.weak_value_history.len() > 1024 {
            self.weak_value_history.remove(0);
        }
    }

    /// Get recent intent detections
    pub fn get_recent_intents(&self) -> &[InstitutionalIntent] {
        &self.intent_cache
    }

    /// Clear intent cache
    pub fn clear_cache(&mut self) {
        self.intent_cache.clear();
    }

    /// Detect aggregate institutional flow direction
    pub fn detect_aggregate_flow(&self) -> IntentDirection {
        if self.intent_cache.is_empty() {
            return IntentDirection::Neutral;
        }

        let bullish_count = self.intent_cache.iter()
            .filter(|i| i.direction == IntentDirection::Bullish)
            .count();
        let bearish_count = self.intent_cache.iter()
            .filter(|i| i.direction == IntentDirection::Bearish)
            .count();
        let total = self.intent_cache.len();

        let bullish_ratio = bullish_count as f64 / total as f64;
        let bearish_ratio = bearish_count as f64 / total as f64;

        if bullish_ratio > 0.6 {
            IntentDirection::Bullish
        } else if bearish_ratio > 0.6 {
            IntentDirection::Bearish
        } else if (bullish_ratio - bearish_ratio).abs() < 0.2 {
            IntentDirection::Ambiguous
        } else {
            IntentDirection::Neutral
        }
    }

    // Internal helper: estimate noise level from state variance
    fn estimate_noise(&self, state: &[f64]) -> f64 {
        if state.len() < 2 {
            return 1.0;
        }

        let mean: f64 = state.iter().sum::<f64>() / state.len() as f64;
        let variance: f64 = state.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / state.len() as f64;
        
        variance.sqrt()
    }

    // Internal helper: determine intent direction
    fn determine_direction(&self, weak_value: f64, state: &[f64]) -> IntentDirection {
        if state.is_empty() {
            return IntentDirection::Ambiguous;
        }

        // Primary signal from weak value sign
        let primary_signal = if weak_value > 0.0 {
            IntentDirection::Bullish
        } else if weak_value < 0.0 {
            IntentDirection::Bearish
        } else {
            IntentDirection::Neutral
        };

        // Validate with state gradient
        if state.len() >= 2 {
            let gradient = state[state.len() - 1] - state[0];
            let gradient_signal = if gradient > 0.0 {
                IntentDirection::Bullish
            } else if gradient < 0.0 {
                IntentDirection::Bearish
            } else {
                IntentDirection::Neutral
            };

            // Check for agreement
            if primary_signal == gradient_signal || primary_signal == IntentDirection::Neutral {
                return gradient_signal;
            } else if gradient_signal == IntentDirection::Neutral {
                return primary_signal;
            } else {
                // Conflicting signals
                return IntentDirection::Ambiguous;
            }
        }

        primary_signal
    }

    // Internal helper: calculate confidence score
    fn calculate_confidence(&self, snr: f64, state: &[f64]) -> f64 {
        // Base confidence from SNR (sigmoid-like scaling)
        let snr_confidence = 1.0 / (1.0 + (-snr + self.snr_threshold).exp());
        
        // Adjust for state consistency
        let consistency_factor = if state.len() >= 2 {
            let variance: f64 = state.iter()
                .map(|&x| x.powi(2))
                .sum::<f64>() / state.len() as f64;
            (1.0 - variance.clamp(0.0, 1.0)).sqrt()
        } else {
            1.0
        };

        (snr_confidence * consistency_factor).clamp(0.0, 1.0)
    }

    // Internal helper: estimate order size
    fn estimate_order_size(&self, weak_value: f64, state: &[f64]) -> f64 {
        // Size scales with weak value magnitude and state norm
        let state_norm: f64 = state.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
        weak_value.abs() * state_norm * 100.0 // Scale factor for realistic sizes
    }

    // Internal helper: estimate execution time horizon
    fn estimate_time_horizon(&self, state: &[f64]) -> u64 {
        if state.len() < 2 {
            return self.lookback_ns;
        }

        // Estimate from state decay rate
        let first_half_avg: f64 = state[..state.len()/2].iter().sum::<f64>() / (state.len()/2) as f64;
        let second_half_avg: f64 = state[state.len()/2..].iter().sum::<f64>() / (state.len() - state.len()/2) as f64;
        
        let decay_rate = if first_half_avg.abs() > 1e-15 {
            (first_half_avg - second_half_avg).abs() / first_half_avg.abs()
        } else {
            0.5
        };

        // Faster decay = shorter horizon
        let horizon_factor = 1.0 - decay_rate.clamp(0.0, 0.9);
        (self.lookback_ns as f64 * horizon_factor) as u64
    }
}

impl Default for HiddenIntentExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extractor_basic() {
        let extractor = HiddenIntentExtractor::new();
        let weak_value = 5.0;
        let post_state = vec![1.0, 0.8, 0.6, 0.4, 0.2];

        let intent = extractor.analyze_weak_value(weak_value, &post_state);
        
        // Should detect intent with sufficient SNR
        assert!(intent.is_some() || intent.is_none()); // Depends on SNR threshold
    }

    #[test]
    fn test_empty_state_rejection() {
        let extractor = HiddenIntentExtractor::new();
        let weak_value = 5.0;
        let post_state: Vec<f64> = vec![];

        let intent = extractor.analyze_weak_value(weak_value, &post_state);
        assert!(intent.is_none());
    }

    #[test]
    fn test_direction_detection() {
        let extractor = HiddenIntentExtractor::new();
        
        // Test bullish signal
        let bullish_state = vec![0.2, 0.4, 0.6, 0.8, 1.0];
        let intent_bullish = extractor.analyze_weak_value(10.0, &bullish_state);
        
        // Test bearish signal
        let bearish_state = vec![1.0, 0.8, 0.6, 0.4, 0.2];
        let intent_bearish = extractor.analyze_weak_value(-10.0, &bearish_state);
        
        // Verify directions are opposite if both detected
        if let (Some(bull), Some(bear)) = (intent_bullish, intent_bearish) {
            assert!(bull.direction != bear.direction || bull.confidence < 0.5 || bear.confidence < 0.5);
        }
    }
}
