//! Bayesian Conviction Scoring and Signal Fusion
//! 
//! Implements Bayesian updating to dynamically adjust alpha signal weights
//! based on recent predictive accuracy in the current market regime.
//! Produces a fused ConvictionScore (-1.0 to +1.0).

use nexus_core::memory::arena::BumpAllocator;
use crate::fusion::regime_hmm::{MarketRegime, MAX_STATES};

/// Maximum number of alpha signals that can be fused
pub const MAX_ALPHA_SIGNALS: usize = 16;

/// Individual alpha signal input
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct AlphaSignal {
    /// Signal value (-1.0 to +1.0)
    pub value: f64,
    /// Signal confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Signal type identifier
    pub signal_type: u8,
    /// Timestamp
    pub ts: u64,
    /// Recent accuracy (for Bayesian update)
    pub recent_accuracy: f64,
    /// Padding
    _padding: [u8; 23],
}

impl Default for AlphaSignal {
    fn default() -> Self {
        Self {
            value: 0.0,
            confidence: 0.5,
            signal_type: 0,
            ts: 0,
            recent_accuracy: 0.5,
            _padding: [0u8; 23],
        }
    }
}

/// Bayesian prior for a signal
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct BayesianPrior {
    /// Prior mean (expected signal accuracy)
    pub mean: f64,
    /// Prior variance (uncertainty about accuracy)
    pub variance: f64,
    /// Number of observations
    pub n_obs: u32,
    /// Sum of squared errors
    pub sse: f64,
}

impl Default for BayesianPrior {
    fn default() -> Self {
        Self {
            mean: 0.5,
            variance: 0.25, // High initial uncertainty
            n_obs: 0,
            sse: 0.0,
        }
    }
}

/// Signal weight with regime-specific adjustments
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct SignalWeight {
    /// Base weight
    pub base: f64,
    /// Regime-specific adjustments
    pub regime_adjustments: [f64; MAX_STATES],
    /// Current effective weight
    pub effective: f64,
    /// Last updated timestamp
    pub last_update: u64,
}

impl Default for SignalWeight {
    fn default() -> Self {
        Self {
            base: 1.0 / MAX_ALPHA_SIGNALS as f64,
            regime_adjustments: [0.0; MAX_STATES],
            effective: 1.0 / MAX_ALPHA_SIGNALS as f64,
            last_update: 0,
        }
    }
}

/// Bayesian Conviction Calculator
pub struct BayesianConviction {
    /// Signal weights
    weights: [SignalWeight; MAX_ALPHA_SIGNALS],
    /// Bayesian priors for each signal
    priors: [BayesianPrior; MAX_ALPHA_SIGNALS],
    /// Number of active signals
    num_signals: usize,
    /// Current conviction score
    conviction: f64,
    /// Conviction variance
    conviction_var: f64,
    /// Current regime
    current_regime: MarketRegime,
    /// Global shrinkage factor (prevents overfitting)
    shrinkage: f64,
    /// Minimum weight floor
    min_weight: f64,
}

unsafe impl Send for BayesianConviction {}
unsafe impl Sync for BayesianConviction {}

impl BayesianConviction {
    pub fn new(_allocator: &BumpAllocator, num_signals: usize) -> Self {
        let num_signals = num_signals.min(MAX_ALPHA_SIGNALS);
        let base_weight = 1.0 / num_signals as f64;
        
        let mut weights = [SignalWeight::default(); MAX_ALPHA_SIGNALS];
        let mut priors = [BayesianPrior::default(); MAX_ALPHA_SIGNALS];
        
        for i in 0..num_signals {
            weights[i].base = base_weight;
            weights[i].effective = base_weight;
            priors[i].mean = 0.5;
        }
        
        Self {
            weights,
            priors,
            num_signals,
            conviction: 0.0,
            conviction_var: 1.0,
            current_regime: MarketRegime::MeanReverting,
            shrinkage: 0.1, // 10% shrinkage toward equal weights
            min_weight: 0.01, // No weight below 1%
        }
    }

    /// Update signal and compute new conviction - zero allocation
    #[inline]
    pub fn update(&mut self, signals: &[AlphaSignal], regime: MarketRegime, ts: u64) -> ConvictionResult {
        self.current_regime = regime;
        
        // Update weights based on Bayesian posterior
        self.update_weights(signals, ts);
        
        // Compute weighted conviction
        let (conviction, variance) = self.compute_conviction(signals);
        
        self.conviction = conviction;
        self.conviction_var = variance;
        
        ConvictionResult {
            conviction,
            conviction_std: variance.sqrt(),
            effective_leverage: self.calculate_effective_leverage(),
            regime: regime as u8,
            num_signals: self.num_signals as u8,
            ts,
        }
    }

    /// Bayesian weight update based on signal accuracy
    #[inline]
    fn update_weights(&mut self, signals: &[AlphaSignal], ts: u64) {
        for i in 0..self.num_signals.min(signals.len()) {
            let signal = &signals[i];
            let prior = &mut self.priors[i];
            
            // Update posterior using conjugate Normal-Normal update
            // Posterior mean = (prior_mean / prior_var + obs_mean / obs_var) / (1/prior_var + 1/obs_var)
            
            let obs_accuracy = signal.recent_accuracy.max(0.0).min(1.0);
            let obs_variance = 0.25; // Assume observation noise
            
            // Precision-weighted update
            let prior_precision = 1.0 / prior.variance.max(1e-6);
            let obs_precision = 1.0 / obs_variance;
            
            let total_precision = prior_precision + obs_precision;
            
            // Updated mean
            prior.mean = (prior.mean * prior_precision + obs_accuracy * obs_precision) / total_precision;
            
            // Updated variance (decreases with more observations)
            prior.variance = 1.0 / total_precision;
            
            // Increment observation count
            prior.n_obs += 1;
            
            // Update SSE for potential empirical Bayes
            let error = obs_accuracy - prior.mean;
            prior.sse += error * error;
            
            // Update weight based on posterior mean accuracy
            let regime_idx = self.current_regime as usize;
            let accuracy_factor = prior.mean.powi(2); // Higher accuracy = higher weight
            
            // Apply regime-specific adjustment
            let regime_adj = self.weights[i].regime_adjustments[regime_idx];
            let raw_weight = accuracy_factor * (1.0 + regime_adj);
            
            // Apply shrinkage toward equal weights
            let equal_weight = 1.0 / self.num_signals as f64;
            self.weights[i].effective = (1.0 - self.shrinkage) * raw_weight + self.shrinkage * equal_weight;
            
            // Enforce minimum weight
            self.weights[i].effective = self.weights[i].effective.max(self.min_weight);
            
            self.weights[i].last_update = ts;
        }
        
        // Normalize weights
        self.normalize_weights();
    }

    /// Normalize weights to sum to 1
    #[inline]
    fn normalize_weights(&mut self) {
        let sum: f64 = self.weights.iter().take(self.num_signals).map(|w| w.effective).sum();
        
        if sum > 1e-10 {
            for i in 0..self.num_signals {
                self.weights[i].effective /= sum;
            }
        }
    }

    /// Compute weighted conviction score
    #[inline]
    fn compute_conviction(&self, signals: &[AlphaSignal]) -> (f64, f64) {
        let mut weighted_sum = 0.0;
        let mut weighted_var = 0.0;
        
        for i in 0..self.num_signals.min(signals.len()) {
            let signal = &signals[i];
            let weight = self.weights[i].effective;
            
            weighted_sum += weight * signal.value * signal.confidence;
            
            // Variance contribution
            let signal_var = (1.0 - signal.confidence).powi(2) * 0.25; // Max variance when confidence is low
            weighted_var += weight.powi(2) * signal_var;
        }
        
        // Clamp conviction to [-1, 1]
        let conviction = weighted_sum.clamp(-1.0, 1.0);
        
        (conviction, weighted_var)
    }

    /// Calculate effective leverage based on conviction certainty
    #[inline]
    fn calculate_effective_leverage(&self) -> f64 {
        // Higher conviction with lower variance = higher leverage
        let conviction_strength = self.conviction.abs();
        let uncertainty = self.conviction_var.sqrt();
        
        // Leverage scales with conviction and inverse uncertainty
        let base_leverage = conviction_strength / (uncertainty + 0.1);
        
        // Cap leverage
        base_leverage.min(5.0)
    }

    /// Set regime-specific weight adjustment for a signal
    #[inline]
    pub fn set_regime_adjustment(&mut self, signal_idx: usize, regime: MarketRegime, adjustment: f64) {
        if signal_idx < MAX_ALPHA_SIGNALS {
            let regime_idx = regime as usize;
            self.weights[signal_idx].regime_adjustments[regime_idx] = adjustment.clamp(-0.5, 0.5);
        }
    }

    /// Get current conviction
    #[inline]
    pub fn get_conviction(&self) -> f64 {
        self.conviction
    }

    /// Get conviction with z-score
    #[inline]
    pub fn get_conviction_zscore(&self) -> f64 {
        if self.conviction_var > 1e-10 {
            self.conviction / self.conviction_var.sqrt()
        } else {
            self.conviction.signum() * 10.0 // High z-score if very certain
        }
    }

    /// Get signal weights
    #[inline]
    pub fn get_weights(&self) -> &[SignalWeight] {
        &self.weights[..self.num_signals]
    }

    /// Reset all priors
    #[inline]
    pub fn reset_priors(&mut self) {
        for prior in &mut self.priors {
            *prior = BayesianPrior::default();
        }
    }

    /// Adjust shrinkage factor
    #[inline]
    pub fn set_shrinkage(&mut self, shrinkage: f64) {
        self.shrinkage = shrinkage.clamp(0.0, 0.5);
    }
}

/// Conviction result
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct ConvictionResult {
    /// Fused conviction score (-1 to +1)
    pub conviction: f64,
    /// Standard deviation of conviction
    pub conviction_std: f64,
    /// Effective leverage multiplier
    pub effective_leverage: f64,
    /// Current regime
    pub regime: u8,
    /// Number of signals used
    pub num_signals: u8,
    /// Timestamp
    pub ts: u64,
    /// Padding
    _padding: [u8; 38],
}

impl Default for ConvictionResult {
    fn default() -> Self {
        Self {
            conviction: 0.0,
            conviction_std: 0.0,
            effective_leverage: 0.0,
            regime: 0,
            num_signals: 0,
            ts: 0,
            _padding: [0u8; 38],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_core::memory::arena::BumpAllocator;

    #[test]
    fn test_bayesian_initialization() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let conv = BayesianConviction::new(&allocator, 4);
        
        assert_eq!(conv.get_conviction(), 0.0);
        assert_eq!(conv.num_signals, 4);
    }

    #[test]
    fn test_conviction_from_uniform_signals() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut conv = BayesianConviction::new(&allocator, 3);
        
        let signals = vec![
            AlphaSignal { value: 0.8, confidence: 0.9, signal_type: 1, ts: 1000, recent_accuracy: 0.7 },
            AlphaSignal { value: 0.7, confidence: 0.8, signal_type: 2, ts: 1000, recent_accuracy: 0.6 },
            AlphaSignal { value: 0.9, confidence: 0.85, signal_type: 3, ts: 1000, recent_accuracy: 0.65 },
        ];
        
        let result = conv.update(&signals, MarketRegime::Trending, 1000);
        
        // Should have positive conviction
        assert!(result.conviction > 0.5);
        assert!(result.conviction <= 1.0);
    }

    #[test]
    fn test_conviction_conflicting_signals() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut conv = BayesianConviction::new(&allocator, 2);
        
        let signals = vec![
            AlphaSignal { value: 1.0, confidence: 0.9, signal_type: 1, ts: 1000, recent_accuracy: 0.5 },
            AlphaSignal { value: -1.0, confidence: 0.9, signal_type: 2, ts: 1000, recent_accuracy: 0.5 },
        ];
        
        let result = conv.update(&signals, MarketRegime::MeanReverting, 1000);
        
        // Conflicting signals should give low conviction
        assert!(result.conviction.abs() < 0.3);
    }

    #[test]
    fn test_bayesian_weight_adaptation() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut conv = BayesianConviction::new(&allocator, 2);
        
        // Signal 1 consistently accurate
        // Signal 2 consistently wrong
        for i in 0..10 {
            let signals = vec![
                AlphaSignal { value: 0.8, confidence: 0.9, signal_type: 1, ts: i as u64 * 100, recent_accuracy: 0.9 },
                AlphaSignal { value: 0.5, confidence: 0.5, signal_type: 2, ts: i as u64 * 100, recent_accuracy: 0.2 },
            ];
            conv.update(&signals, MarketRegime::Trending, i as u64 * 100);
        }
        
        // Weight should shift toward signal 1
        let weights = conv.get_weights();
        assert!(weights[0].effective > weights[1].effective);
    }

    #[test]
    fn test_conviction_zscore() {
        let allocator = BumpAllocator::new(1024 * 1024);
        let mut conv = BayesianConviction::new(&allocator, 3);
        
        let signals = vec![
            AlphaSignal { value: 0.9, confidence: 0.95, signal_type: 1, ts: 1000, recent_accuracy: 0.8 },
            AlphaSignal { value: 0.85, confidence: 0.9, signal_type: 2, ts: 1000, recent_accuracy: 0.75 },
            AlphaSignal { value: 0.88, confidence: 0.92, signal_type: 3, ts: 1000, recent_accuracy: 0.78 },
        ];
        
        conv.update(&signals, MarketRegime::Trending, 1000);
        
        // High agreement should give high z-score
        assert!(conv.get_conviction_zscore() > 2.0);
    }
}
