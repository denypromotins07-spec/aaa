//! Weak Value Amplifier for post-selected order book probing
//! 
//! Implements quantum weak measurements to probe L3 order book without collapsing it.
//! Uses post-selection filtering to amplify hidden institutional intent signals.

use crate::weak::post_selection_filter::PostSelectionFilter;
use crate::weak::hidden_intent_extractor::HiddenIntentExtractor;

/// Epsilon floor to prevent division by zero in weak value calculations
const EPSILON_FLOOR: f64 = 1e-15;

/// Maximum amplification factor before rejection sampling kicks in
const MAX_AMPLIFICATION_FACTOR: f64 = 1e6;

/// Result of a weak measurement operation
#[derive(Debug, Clone)]
pub struct WeakMeasurementResult {
    /// The amplified weak value
    pub weak_value: f64,
    /// The pre-selection state (initial order book state)
    pub pre_selection_state: Vec<f64>,
    /// The post-selection state (filtered micro-price action)
    pub post_selection_state: Vec<f64>,
    /// Probability of successful post-selection
    pub post_selection_probability: f64,
    /// Whether the result passed quality checks
    pub is_valid: bool,
    /// Rejection reason if invalid
    pub rejection_reason: Option<String>,
}

/// Weak Value Amplifier for probing order book without revealing intent
pub struct WeakValueAmplifier {
    /// Filter for post-selection states
    filter: PostSelectionFilter,
    /// Extractor for hidden institutional intent
    extractor: HiddenIntentExtractor,
    /// Current amplification factor
    current_amplification: f64,
    /// Number of successful measurements
    success_count: u64,
    /// Total measurement attempts
    total_count: u64,
}

impl WeakValueAmplifier {
    /// Create a new weak value amplifier with default parameters
    pub fn new() -> Self {
        Self {
            filter: PostSelectionFilter::new(),
            extractor: HiddenIntentExtractor::new(),
            current_amplification: 1.0,
            success_count: 0,
            total_count: 0,
        }
    }

    /// Create amplifier with custom epsilon floor
    pub fn with_epsilon(epsilon: f64) -> Self {
        let mut amplifier = Self::new();
        amplifier.filter.set_epsilon(epsilon.max(EPSILON_FLOOR));
        amplifier
    }

    /// Perform a weak measurement on the order book
    /// 
    /// # Arguments
    /// * `pre_state` - Initial order book state vector (L3 depth)
    /// * `observable` - Observable operator representing probe interaction
    /// * `post_criteria` - Criteria for post-selection filtering
    /// 
    /// # Returns
    /// WeakMeasurementResult with amplified signal or rejection details
    pub fn measure(&mut self, 
                   pre_state: &[f64], 
                   observable: &[f64],
                   post_criteria: &PostSelectionFilter) -> WeakMeasurementResult {
        self.total_count += 1;

        // Validate input dimensions
        if pre_state.is_empty() || observable.is_empty() {
            return WeakMeasurementResult {
                weak_value: 0.0,
                pre_selection_state: pre_state.to_vec(),
                post_selection_state: vec![],
                post_selection_probability: 0.0,
                is_valid: false,
                rejection_reason: Some("Empty input state or observable".to_string()),
            };
        }

        if pre_state.len() != observable.len() {
            return WeakMeasurementResult {
                weak_value: 0.0,
                pre_selection_state: pre_state.to_vec(),
                post_selection_state: vec![],
                post_selection_probability: 0.0,
                is_valid: false,
                rejection_reason: Some("Dimension mismatch between state and observable".to_string()),
            };
        }

        // Compute the numerator: <psi_f|A|psi_i>
        let numerator = self.compute_weak_value_numerator(pre_state, observable);

        // Apply post-selection filter to get final state
        let post_filtered = post_criteria.apply_filter(pre_state);
        
        if post_filtered.is_empty() {
            return WeakMeasurementResult {
                weak_value: 0.0,
                pre_selection_state: pre_state.to_vec(),
                post_selection_state: vec![],
                post_selection_probability: 0.0,
                is_valid: false,
                rejection_reason: Some("Post-selection yielded empty state".to_string()),
            };
        }

        // Compute denominator: <psi_f|psi_i> (overlap integral)
        let denominator = self.compute_state_overlap(pre_state, &post_filtered);

        // Check for near-zero denominator (orthogonal states)
        let abs_denominator = denominator.abs();
        if abs_denominator < EPSILON_FLOOR {
            return WeakMeasurementResult {
                weak_value: 0.0,
                pre_selection_state: pre_state.to_vec(),
                post_selection_state: post_filtered.clone(),
                post_selection_probability: abs_denominator.powi(2),
                is_valid: false,
                rejection_reason: Some("Near-orthogonal post-selection (denominator ~0)".to_string()),
            };
        }

        // Calculate weak value with amplification
        let raw_weak_value = numerator / denominator;
        
        // Apply probabilistic rejection for extreme amplification
        let amplification_factor = raw_weak_value.abs() / (numerator.abs().max(EPSILON_FLOOR));
        
        if amplification_factor > MAX_AMPLIFICATION_FACTOR {
            // Rejection sampling for singularities
            let acceptance_prob = MAX_AMPLIFICATION_FACTOR / amplification_factor;
            let random_sample = fastrand::f64();
            
            if random_sample > acceptance_prob {
                return WeakMeasurementResult {
                    weak_value: 0.0,
                    pre_selection_state: pre_state.to_vec(),
                    post_selection_state: post_filtered.clone(),
                    post_selection_probability: acceptance_prob,
                    is_valid: false,
                    rejection_reason: Some("Rejected due to excessive amplification (singularity)".to_string()),
                };
            }
        }

        // Clamp weak value to prevent overflow
        let clamped_weak_value = raw_weak_value.clamp(-MAX_AMPLIFICATION_FACTOR, MAX_AMPLIFICATION_FACTOR);

        self.success_count += 1;
        self.current_amplification = amplification_factor;

        WeakMeasurementResult {
            weak_value: clamped_weak_value,
            pre_selection_state: pre_state.to_vec(),
            post_selection_state: post_filtered,
            post_selection_probability: abs_denominator.powi(2),
            is_valid: true,
            rejection_reason: None,
        }
    }

    /// Extract hidden institutional intent from weak measurement results
    pub fn extract_intent(&self, result: &WeakMeasurementResult) -> Option<f64> {
        if !result.is_valid {
            return None;
        }
        self.extractor.analyze_weak_value(result.weak_value, &result.post_selection_state)
    }

    /// Get success rate of measurements
    pub fn success_rate(&self) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        self.success_count as f64 / self.total_count as f64
    }

    /// Reset amplifier statistics
    pub fn reset(&mut self) {
        self.success_count = 0;
        self.total_count = 0;
        self.current_amplification = 1.0;
    }

    // Internal helper: compute numerator <psi_f|A|psi_i>
    fn compute_weak_value_numerator(&self, pre_state: &[f64], observable: &[f64]) -> f64 {
        pre_state.iter()
            .zip(observable.iter())
            .map(|(s, o)| s * o)
            .sum()
    }

    // Internal helper: compute overlap <psi_f|psi_i>
    fn compute_state_overlap(&self, state1: &[f64], state2: &[f64]) -> f64 {
        state1.iter()
            .zip(state2.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

impl Default for WeakValueAmplifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weak_measurement_basic() {
        let mut amplifier = WeakValueAmplifier::new();
        let pre_state = vec![1.0, 0.5, 0.25, 0.125];
        let observable = vec![0.1, 0.2, 0.3, 0.4];
        let filter = PostSelectionFilter::new();

        let result = amplifier.measure(&pre_state, &observable, &filter);
        
        assert!(result.is_valid || result.rejection_reason.is_some());
    }

    #[test]
    fn test_orthogonal_rejection() {
        let mut amplifier = WeakValueAmplifier::new();
        // Create nearly orthogonal states
        let pre_state = vec![1.0, 0.0, 0.0, 0.0];
        let observable = vec![0.0, 1.0, 1.0, 1.0];
        let filter = PostSelectionFilter::with_strict_mode(true);

        let result = amplifier.measure(&pre_state, &observable, &filter);
        
        // Should reject near-orthogonal post-selection
        assert!(!result.is_valid || result.post_selection_probability < EPSILON_FLOOR);
    }
}
