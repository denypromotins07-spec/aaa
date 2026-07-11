//! Post-Selection Filter for weak measurement validation
//! 
//! Implements strict filtering criteria for post-selected quantum states.
//! Used to validate micro-price action after weak probe interactions.

/// Epsilon threshold for probability comparisons
const PROBABILITY_EPSILON: f64 = 1e-12;

/// Minimum acceptable post-selection probability
const MIN_POST_SELECTION_PROB: f64 = 1e-8;

/// Filter criteria for post-selection states
#[derive(Debug, Clone)]
pub struct PostSelectionFilter {
    /// Epsilon floor for numerical stability
    epsilon: f64,
    /// Strict mode rejects near-orthogonal states more aggressively
    strict_mode: bool,
    /// Minimum probability threshold for valid post-selection
    min_probability: f64,
    /// Maximum allowed state vector norm deviation
    max_norm_deviation: f64,
    /// Temporal window for post-selection (nanoseconds)
    temporal_window_ns: u64,
}

impl PostSelectionFilter {
    /// Create a new post-selection filter with default parameters
    pub fn new() -> Self {
        Self {
            epsilon: PROBABILITY_EPSILON,
            strict_mode: false,
            min_probability: MIN_POST_SELECTION_PROB,
            max_norm_deviation: 0.1,
            temporal_window_ns: 1000, // 1 microsecond
        }
    }

    /// Create filter with custom epsilon
    pub fn with_epsilon(epsilon: f64) -> Self {
        Self {
            epsilon: epsilon.max(PROBABILITY_EPSILON),
            ..Self::new()
        }
    }

    /// Create filter in strict mode
    pub fn with_strict_mode(strict: bool) -> Self {
        Self {
            strict_mode: strict,
            min_probability: if strict { 1e-6 } else { MIN_POST_SELECTION_PROB },
            ..Self::new()
        }
    }

    /// Set custom epsilon floor
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon = epsilon.max(PROBABILITY_EPSILON);
    }

    /// Enable or disable strict mode
    pub fn set_strict_mode(&mut self, strict: bool) {
        self.strict_mode = strict;
        self.min_probability = if strict { 1e-6 } else { MIN_POST_SELECTION_PROB };
    }

    /// Apply post-selection filter to a state vector
    /// 
    /// # Arguments
    /// * `state` - Raw post-measurement state vector
    /// 
    /// # Returns
    /// Filtered state vector (empty if filtering fails)
    pub fn apply_filter(&self, state: &[f64]) -> Vec<f64> {
        if state.is_empty() {
            return vec![];
        }

        // Normalize the state
        let norm = self.compute_norm(state);
        if norm < self.epsilon {
            return vec![];
        }

        let normalized: Vec<f64> = state.iter().map(|&x| x / norm).collect();

        // Check probability distribution validity
        let total_probability: f64 = normalized.iter().map(|&x| x.powi(2)).sum();
        
        if total_probability.abs() < self.min_probability {
            return vec![];
        }

        // Validate norm consistency
        let norm_deviation = (total_probability - 1.0).abs();
        if norm_deviation > self.max_norm_deviation {
            return vec![];
        }

        // Apply temporal smoothing if in strict mode
        if self.strict_mode {
            self.apply_temporal_smoothing(&normalized)
        } else {
            normalized
        }
    }

    /// Validate a post-selection result
    /// 
    /// # Arguments
    /// * `pre_state` - Initial state before measurement
    /// * `post_state` - State after post-selection
    /// 
    /// # Returns
    /// true if post-selection is valid, false otherwise
    pub fn validate_post_selection(&self, pre_state: &[f64], post_state: &[f64]) -> bool {
        if pre_state.is_empty() || post_state.is_empty() {
            return false;
        }

        if pre_state.len() != post_state.len() {
            return false;
        }

        // Compute overlap
        let overlap = self.compute_overlap(pre_state, post_state);
        
        if overlap.abs() < self.epsilon {
            return false;
        }

        // Check post-state normalization
        let post_norm = self.compute_norm(post_state);
        if post_norm.abs() < self.epsilon {
            return false;
        }

        // Verify probability conservation (within tolerance)
        let pre_prob: f64 = pre_state.iter().map(|&x| x.powi(2)).sum();
        let post_prob: f64 = post_state.iter().map(|&x| x.powi(2)).sum();
        
        let prob_ratio = if pre_prob > self.epsilon {
            post_prob / pre_prob
        } else {
            0.0
        };

        // Probability should be conserved (ratio ~ 1.0) or reduced (measurement loss)
        prob_ratio <= 1.0 + self.max_norm_deviation && prob_ratio >= 0.0
    }

    /// Get the temporal window in nanoseconds
    pub fn temporal_window(&self) -> u64 {
        self.temporal_window_ns
    }

    /// Set temporal window for post-selection
    pub fn set_temporal_window(&mut self, window_ns: u64) {
        self.temporal_window_ns = window_ns;
    }

    // Internal helper: compute L2 norm of state vector
    fn compute_norm(&self, state: &[f64]) -> f64 {
        state.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt()
    }

    // Internal helper: compute overlap between two states
    fn compute_overlap(&self, state1: &[f64], state2: &[f64]) -> f64 {
        state1.iter().zip(state2.iter()).map(|(a, b)| a * b).sum()
    }

    // Internal helper: apply temporal smoothing in strict mode
    fn apply_temporal_smoothing(&self, state: &[f64]) -> Vec<f64> {
        if state.len() < 3 {
            return state.to_vec();
        }

        // Simple moving average smoothing
        let mut smoothed = Vec::with_capacity(state.len());
        
        // First element
        smoothed.push((state[0] * 2.0 + state[1]) / 3.0);
        
        // Middle elements
        for i in 1..state.len() - 1 {
            let avg = (state[i - 1] + state[i] + state[i + 1]) / 3.0;
            smoothed.push(avg);
        }
        
        // Last element
        let last_idx = state.len() - 1;
        smoothed.push((state[last_idx - 1] + state[last_idx] * 2.0) / 3.0);

        smoothed
    }
}

impl Default for PostSelectionFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_basic() {
        let filter = PostSelectionFilter::new();
        let state = vec![1.0, 0.5, 0.25, 0.125];
        
        let filtered = filter.apply_filter(&state);
        assert!(!filtered.is_empty());
    }

    #[test]
    fn test_filter_rejects_empty() {
        let filter = PostSelectionFilter::new();
        let state: Vec<f64> = vec![];
        
        let filtered = filter.apply_filter(&state);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_validation_valid() {
        let filter = PostSelectionFilter::new();
        let pre = vec![1.0, 0.5, 0.25];
        let post = vec![0.9, 0.45, 0.225];
        
        assert!(filter.validate_post_selection(&pre, &post));
    }

    #[test]
    fn test_orthogonal_rejection() {
        let filter = PostSelectionFilter::with_strict_mode(true);
        let pre = vec![1.0, 0.0, 0.0];
        let post = vec![0.0, 1.0, 0.0];
        
        assert!(!filter.validate_post_selection(&pre, &post));
    }
}
