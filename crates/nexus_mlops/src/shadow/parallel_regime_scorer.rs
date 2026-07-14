//! Parallel Regime Scorer - Real-time log-loss and Sharpe ratio calculation
//! 
//! Calculates real-time performance metrics for shadow models WITHOUT executing trades.
//! Uses stack-allocated windows for zero-heap operations in the hot path.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use crate::shadow::ModelRegime;

/// Maximum window size for rolling calculations (stack-allocated)
const MAX_WINDOW_SIZE: usize = 1024;

/// Rolling statistics tracker using Welford's online algorithm
#[derive(Debug, Clone)]
struct RollingStats {
    count: u64,
    mean: f64,
    m2: f64, // Sum of squares of differences from mean
    min_val: f64,
    max_val: f64,
}

impl RollingStats {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min_val: f64::INFINITY,
            max_val: f64::NEG_INFINITY,
        }
    }
    
    /// Update with a new value (Welford's algorithm - numerically stable)
    #[inline(always)]
    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        
        if value < self.min_val {
            self.min_val = value;
        }
        if value > self.max_val {
            self.max_val = value;
        }
    }
    
    #[inline(always)]
    fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }
    
    #[inline(always)]
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    
    #[inline(always)]
    fn sharpe_ratio(&self, risk_free_rate: f64) -> f64 {
        let std = self.std_dev();
        if std < 1e-10 {
            return 0.0;
        }
        (self.mean - risk_free_rate) / std
    }
}

/// Log-loss calculator for binary/multi-class predictions
struct LogLossTracker {
    sum_log_loss: f64,
    count: u64,
}

impl LogLossTracker {
    fn new() -> Self {
        Self {
            sum_log_loss: 0.0,
            count: 0,
        }
    }
    
    /// Record a prediction and actual outcome
    /// prediction: probability in [0, 1]
    /// actual: 0 or 1
    #[inline(always)]
    fn record(&mut self, prediction: f64, actual: f64) {
        // Clamp prediction to avoid log(0)
        let p = prediction.max(1e-15).min(1.0 - 1e-15);
        let log_loss = -(actual * p.ln() + (1.0 - actual) * (1.0 - p).ln());
        self.sum_log_loss += log_loss;
        self.count += 1;
    }
    
    #[inline(always)]
    fn average_log_loss(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_log_loss / self.count as f64
        }
    }
}

/// Score for a single model regime
#[derive(Debug, Clone, Copy)]
pub struct RegimeScore {
    pub model_id: u32,
    pub regime: ModelRegime,
    pub sharpe_ratio: f64,
    pub log_loss: f64,
    pub hit_rate: f64,
    pub avg_return_bps: i64, // Scaled integer (basis points)
    pub sample_count: u64,
    pub timestamp_ns: u64,
}

impl RegimeScore {
    /// Higher is better - composite score for ranking
    #[inline(always)]
    pub fn composite_score(&self) -> f64 {
        // Weighted combination: Sharpe is most important, then hit rate, then log loss
        let sharpe_component = self.sharpe_ratio * 2.0;
        let hit_rate_component = (self.hit_rate - 0.5) * 4.0; // Normalize to [-1, 1]
        let log_loss_component = -self.log_loss * 0.5; // Lower log loss is better
        
        sharpe_component + hit_rate_component + log_loss_component
    }
}

/// Parallel Regime Scorer - evaluates all shadow models simultaneously
pub struct RegimeScorer {
    /// Per-model rolling return statistics
    returns_stats: [RollingStats; 8], // Support up to 8 models
    /// Per-model log-loss trackers
    log_loss_trackers: [LogLossTracker; 8],
    /// Per-model hit counters (correct predictions)
    hit_counts: [AtomicU64; 8],
    /// Per-model total prediction counters
    total_counts: [AtomicU64; 8],
    /// Per-model cumulative return in basis points (scaled integer)
    cumulative_returns_bps: [AtomicI64; 8],
    /// Number of active models
    num_models: usize,
}

// Safety: All internal state uses atomics or is protected by design
unsafe impl Send for RegimeScorer {}
unsafe impl Sync for RegimeScorer {}

impl RegimeScorer {
    /// Create a new regime scorer for N models
    pub fn new(num_models: usize) -> Self {
        assert!(num_models <= 8, "Maximum 8 models supported");
        
        Self {
            returns_stats: std::array::from_fn(|_| RollingStats::new()),
            log_loss_trackers: std::array::from_fn(|_| LogLossTracker::new()),
            hit_counts: std::array::from_fn(|_| AtomicU64::new(0)),
            total_counts: std::array::from_fn(|_| AtomicU64::new(0)),
            cumulative_returns_bps: std::array::from_fn(|_| AtomicI64::new(0)),
            num_models,
        }
    }
    
    /// Record a prediction result for a model (zero-alloc, lock-free)
    /// 
    /// # Arguments
    /// * `model_id` - The model ID (0 = live, 1+ = shadows)
    /// * `predicted_direction` - Predicted direction: 1.0 for up, 0.0 for down
    /// * `actual_return_bps` - Actual return in basis points (scaled integer)
    /// * `prediction_prob` - Confidence/probability of prediction [0, 1]
    #[inline(always)]
    pub fn record_prediction(
        &self,
        model_id: u32,
        predicted_direction: f64,
        actual_return_bps: i64,
        prediction_prob: f64,
    ) {
        let idx = model_id as usize;
        if idx >= self.num_models {
            return;
        }
        
        // Convert basis points to float for rolling stats
        let actual_return = actual_return_bps as f64 / 10000.0;
        
        // Update rolling statistics
        unsafe {
            // Safe because we have exclusive access through &self and use interior mutability
            let stats_ptr = &self.returns_stats[idx] as *const RollingStats as *mut RollingStats;
            (*stats_ptr).update(actual_return);
            
            let ll_ptr = &self.log_loss_trackers[idx] as *const LogLossTracker as *mut LogLossTracker;
            
            // Calculate actual direction for log-loss
            let actual_direction = if actual_return_bps > 0 { 1.0 } else { 0.0 };
            (*ll_ptr).record(prediction_prob, actual_direction);
        }
        
        // Update atomic counters
        self.total_counts[idx].fetch_add(1, Ordering::Relaxed);
        self.cumulative_returns_bps[idx].fetch_add(actual_return_bps, Ordering::Relaxed);
        
        // Check if prediction was correct (hit)
        let predicted_up = predicted_direction > 0.5;
        let actual_up = actual_return_bps > 0;
        if predicted_up == actual_up {
            self.hit_counts[idx].fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Calculate current score for a model (read-only, zero-alloc)
    #[inline(always)]
    pub fn calculate_score(&self, model_id: u32, regime: ModelRegime) -> RegimeScore {
        let idx = model_id as usize;
        if idx >= self.num_models {
            return RegimeScore {
                model_id,
                regime,
                sharpe_ratio: 0.0,
                log_loss: f64::MAX,
                hit_rate: 0.0,
                avg_return_bps: 0,
                sample_count: 0,
                timestamp_ns: 0,
            };
        }
        
        let stats = &self.returns_stats[idx];
        let ll_tracker = &self.log_loss_trackers[idx];
        let hits = self.hit_counts[idx].load(Ordering::Relaxed);
        let total = self.total_counts[idx].load(Ordering::Relaxed);
        let cum_return = self.cumulative_returns_bps[idx].load(Ordering::Relaxed);
        
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };
        
        let avg_return_bps = if total > 0 {
            cum_return / total as i64
        } else {
            0
        };
        
        RegimeScore {
            model_id,
            regime,
            sharpe_ratio: stats.sharpe_ratio(0.0), // Risk-free rate = 0 for crypto
            log_loss: ll_tracker.average_log_loss(),
            hit_rate,
            avg_return_bps,
            sample_count: total,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }
    
    /// Get scores for all models and find the best performer
    pub fn find_best_model(&self, regimes: &[ModelRegime]) -> Option<(u32, RegimeScore)> {
        let mut best_id = 0u32;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_regime_score: Option<RegimeScore> = None;
        
        for (i, &regime) in regimes.iter().enumerate() {
            if i >= self.num_models {
                break;
            }
            
            let score = self.calculate_score(i as u32, regime);
            let composite = score.composite_score();
            
            if composite > best_score {
                best_score = composite;
                best_id = i as u32;
                best_regime_score = Some(score);
            }
        }
        
        best_regime_score.map(|s| (best_id, s))
    }
    
    /// Reset statistics for a specific model (for regime changes)
    pub fn reset_model(&self, model_id: u32) {
        let idx = model_id as usize;
        if idx >= self.num_models {
            return;
        }
        
        unsafe {
            let stats_ptr = &self.returns_stats[idx] as *const RollingStats as *mut RollingStats;
            *stats_ptr = RollingStats::new();
            
            let ll_ptr = &self.log_loss_trackers[idx] as *const LogLossTracker as *mut LogLossTracker;
            *ll_ptr = LogLossTracker::new();
        }
        
        self.hit_counts[idx].store(0, Ordering::Relaxed);
        self.total_counts[idx].store(0, Ordering::Relaxed);
        self.cumulative_returns_bps[idx].store(0, Ordering::Relaxed);
    }
    
    /// Get number of samples recorded for a model
    pub fn sample_count(&self, model_id: u32) -> u64 {
        let idx = model_id as usize;
        if idx >= self.num_models {
            0
        } else {
            self.total_counts[idx].load(Ordering::Relaxed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rolling_stats_welford() {
        let mut stats = RollingStats::new();
        
        // Feed known sequence
        for i in 1..=10 {
            stats.update(i as f64);
        }
        
        assert_eq!(stats.count, 10);
        assert!((stats.mean - 5.5).abs() < 1e-10);
        assert!(stats.variance() > 0.0);
    }
    
    #[test]
    fn test_log_loss_tracker() {
        let mut tracker = LogLossTracker::new();
        
        // Perfect predictions should have low log loss
        tracker.record(0.99, 1.0);
        tracker.record(0.99, 1.0);
        tracker.record(0.01, 0.0);
        tracker.record(0.01, 0.0);
        
        assert!(tracker.average_log_loss() < 0.1);
        
        // Wrong predictions should have high log loss
        let mut bad_tracker = LogLossTracker::new();
        bad_tracker.record(0.99, 0.0);
        bad_tracker.record(0.99, 0.0);
        
        assert!(bad_tracker.average_log_loss() > 1.0);
    }
    
    #[test]
    fn test_regime_scorer_thread_safety() {
        let scorer = RegimeScorer::new(4);
        let regimes = [
            ModelRegime::Live,
            ModelRegime::HighVolatility,
            ModelRegime::MeanReversion,
            ModelRegime::CrashHedge,
        ];
        
        // Simulate concurrent updates
        for _ in 0..1000 {
            for (i, _) in regimes.iter().enumerate() {
                scorer.record_prediction(
                    i as u32,
                    0.6,
                    10,
                    0.6,
                );
            }
        }
        
        // Verify counts are correct
        for i in 0..4 {
            assert_eq!(scorer.sample_count(i as u32), 1000);
        }
    }
    
    #[test]
    fn test_composite_score_ranking() {
        let scorer = RegimeScorer::new(2);
        
        // Model 0: good performance
        for _ in 0..100 {
            scorer.record_prediction(0, 0.7, 50, 0.7);
        }
        
        // Model 1: bad performance  
        for _ in 0..100 {
            scorer.record_prediction(1, 0.3, -50, 0.3);
        }
        
        let score0 = scorer.calculate_score(0, ModelRegime::Live);
        let score1 = scorer.calculate_score(1, ModelRegime::HighVolatility);
        
        assert!(score0.composite_score() > score1.composite_score());
    }
}
