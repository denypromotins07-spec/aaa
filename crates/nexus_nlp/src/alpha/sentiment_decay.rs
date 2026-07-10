//! Signal Decay & Alpha Fusion Module
//!
//! This module translates NLP sentiment scores into Stage 3 ConvictionScores
//! and applies exponential time-decay functions so alpha signals fade
//! mathematically as the market absorbs the news.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, debug};

/// Half-life of signal decay in milliseconds
const DEFAULT_HALF_LIFE_MS: u64 = 300_000; // 5 minutes

/// Minimum signal threshold before being considered zero
const MIN_SIGNAL_THRESHOLD: f64 = 0.01;

/// Conviction score type (compatible with Stage 3)
pub type ConvictionScore = f64;

/// Time-decayed sentiment signal
#[derive(Debug, Clone)]
pub struct DecayedSignal {
    /// Original sentiment score (-1.0 to 1.0)
    pub original_score: f64,
    /// Current decayed score
    pub current_score: f64,
    /// Time since signal creation (milliseconds)
    pub age_ms: u64,
    /// Half-life used for decay (milliseconds)
    pub half_life_ms: u64,
    /// Source of the signal
    pub source: SignalSource,
    /// Associated ticker/symbol
    pub symbol: Option<String>,
    /// Creation timestamp (nanoseconds)
    pub created_ns: u128,
    /// Last updated timestamp (nanoseconds)
    pub updated_ns: u128,
}

impl DecayedSignal {
    /// Create a new decayed signal
    pub fn new(score: f64, source: SignalSource, symbol: Option<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        
        Self {
            original_score: score,
            current_score: score,
            age_ms: 0,
            half_life_ms: DEFAULT_HALF_LIFE_MS,
            source,
            symbol,
            created_ns: now,
            updated_ns: now,
        }
    }

    /// Update the signal's decayed value based on elapsed time
    pub fn update_decay(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        
        self.age_ms = ((now - self.created_ns) / 1_000_000) as u64;
        self.current_score = self.calculate_decayed_value(self.age_ms);
        self.updated_ns = now;
    }

    /// Calculate decayed value for a given age
    fn calculate_decayed_value(&self, age_ms: u64) -> f64 {
        // Exponential decay: score * e^(-ln(2) * age / half_life)
        let decay_factor = (-std::f64::consts::LN_2 * age_ms as f64 / self.half_life_ms as f64).exp();
        let decayed = self.original_score * decay_factor;
        
        // Apply minimum threshold
        if decayed.abs() < MIN_SIGNAL_THRESHOLD {
            0.0
        } else {
            decayed
        }
    }

    /// Check if signal has decayed to negligible levels
    pub fn is_expired(&self) -> bool {
        self.current_score.abs() < MIN_SIGNAL_THRESHOLD
    }

    /// Convert to conviction score for Stage 3 fusion
    pub fn to_conviction(&self) -> ConvictionScore {
        // Map [-1, 1] to [0, 1] conviction scale
        ((self.current_score + 1.0) / 2.0).clamp(0.0, 1.0)
    }
}

/// Source of the alpha signal
#[derive(Debug, Clone, PartialEq)]
pub enum SignalSource {
    /// Hawkish/Dovish central bank analysis
    HawkishDovish { entity: String },
    /// General sentiment analysis
    Sentiment { model: String },
    /// Event propagation result
    EventPropagation { event_type: String },
    /// Named entity recognition
    EntityRecognition { entity_type: String },
    /// Custom/unknown source
    Custom(String),
}

/// Configuration for signal decay
#[derive(Debug, Clone)]
pub struct DecayConfig {
    /// Default half-life in milliseconds
    pub default_half_life_ms: u64,
    /// Half-life for Fed-related signals (shorter - markets react fast)
    pub fed_half_life_ms: u64,
    /// Half-life for earnings signals
    pub earnings_half_life_ms: u64,
    /// Half-life for general news
    pub general_half_life_ms: u64,
    /// Enable adaptive half-life based on volatility
    pub enable_adaptive_half_life: bool,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            default_half_life_ms: DEFAULT_HALF_LIFE_MS,
            fed_half_life_ms: 60_000,      // 1 minute for Fed signals
            earnings_half_life_ms: 180_000, // 3 minutes for earnings
            general_half_life_ms: 600_000,  // 10 minutes for general news
            enable_adaptive_half_life: true,
        }
    }
}

/// Signal decay manager
pub struct SentimentDecayManager {
    /// Active signals
    signals: dashmap::DashMap<String, DecayedSignal>,
    /// Configuration
    config: DecayConfig,
    /// Statistics
    total_signals: AtomicU64,
    expired_signals: AtomicU64,
}

impl SentimentDecayManager {
    /// Create a new decay manager
    pub fn new(config: DecayConfig) -> Self {
        Self {
            signals: dashmap::DashMap::new(),
            config,
            total_signals: AtomicU64::new(0),
            expired_signals: AtomicU64::new(0),
        }
    }

    /// Add a new signal to track
    pub fn add_signal(&self, key: String, score: f64, source: SignalSource, symbol: Option<String>) {
        let mut signal = DecayedSignal::new(score, source.clone(), symbol);
        
        // Set appropriate half-life based on source
        signal.half_life_ms = match &source {
            SignalSource::HawkishDovish { .. } => self.config.fed_half_life_ms,
            SignalSource::Sentiment { model } if model.contains("earnings") => {
                self.config.earnings_half_life_ms
            }
            _ => self.config.default_half_life_ms,
        };
        
        self.signals.insert(key, signal);
        self.total_signals.fetch_add(1, Ordering::Relaxed);
        
        debug!("Added signal {}: score={}, source={:?}", key, score, source);
    }

    /// Get the current decayed value for a signal
    pub fn get_signal(&self, key: &str) -> Option<DecayedSignal> {
        if let Some(mut entry) = self.signals.get_mut(key) {
            entry.update_decay();
            
            if entry.is_expired() {
                self.expired_signals.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            
            Some(entry.value().clone())
        } else {
            None
        }
    }

    /// Get all active signals with their decayed values
    pub fn get_all_active_signals(&self) -> Vec<(String, DecayedSignal)> {
        let mut results = Vec::new();
        
        for mut entry in self.signals.iter_mut() {
            entry.update_decay();
            
            if !entry.is_expired() {
                results.push((entry.key().clone(), entry.value().clone()));
            } else {
                self.expired_signals.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        // Remove expired signals
        self.signals.retain(|_, signal| !signal.is_expired());
        
        results
    }

    /// Get the conviction score for a specific symbol
    pub fn get_conviction_for_symbol(&self, symbol: &str) -> Option<ConvictionScore> {
        let mut total_conviction = 0.0;
        let mut count = 0;
        
        for mut entry in self.signals.iter_mut() {
            if let Some(sym) = &entry.symbol {
                if sym == symbol {
                    entry.update_decay();
                    if !entry.is_expired() {
                        total_conviction += entry.to_conviction();
                        count += 1;
                    }
                }
            }
        }
        
        if count > 0 {
            Some(total_conviction / count as f64)
        } else {
            None
        }
    }

    /// Aggregate signals for a symbol into a single directional score
    pub fn aggregate_signal(&self, symbol: &str) -> Option<f64> {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        
        for mut entry in self.signals.iter_mut() {
            if let Some(sym) = &entry.symbol {
                if sym == symbol {
                    entry.update_decay();
                    if !entry.is_expired() {
                        // Weight by recency (less decayed = more weight)
                        let weight = entry.current_score.abs();
                        weighted_sum += entry.current_score * weight;
                        total_weight += weight;
                    }
                }
            }
        }
        
        if total_weight > 0.0 {
            Some(weighted_sum / total_weight)
        } else {
            None
        }
    }

    /// Get statistics
    pub fn get_stats(&self) -> DecayStats {
        let total = self.total_signals.load(Ordering::Relaxed);
        let expired = self.expired_signals.load(Ordering::Relaxed);
        let active = self.signals.len() as u64;
        
        DecayStats {
            total_signals_added: total,
            total_signals_expired: expired,
            currently_active: active,
            expiration_rate: if total > 0 { expired as f64 / total as f64 } else { 0.0 },
        }
    }

    /// Clear all signals
    pub fn clear(&self) {
        self.signals.clear();
    }

    /// Prune expired signals
    pub fn prune_expired(&self) -> usize {
        let before = self.signals.len();
        self.signals.retain(|_, signal| !signal.is_expired());
        before - self.signals.len()
    }
}

/// Statistics for the decay manager
#[derive(Debug, Clone)]
pub struct DecayStats {
    pub total_signals_added: u64,
    pub total_signals_expired: u64,
    pub currently_active: usize,
    pub expiration_rate: f64,
}

/// Alpha fusion engine that combines multiple signal sources
pub struct AlphaFusionEngine {
    /// Decay manager for signal tracking
    decay_manager: Arc<SentimentDecayManager>,
    /// Weights for different signal sources
    source_weights: std::collections::HashMap<SignalSource, f64>,
}

impl AlphaFusionEngine {
    /// Create a new fusion engine
    pub fn new(decay_manager: Arc<SentimentDecayManager>) -> Self {
        let mut source_weights = std::collections::HashMap::new();
        
        // Default weights
        source_weights.insert(SignalSource::HawkishDovish { entity: String::new() }, 1.5);
        source_weights.insert(SignalSource::EventPropagation { event_type: String::new() }, 1.2);
        source_weights.insert(SignalSource::Sentiment { model: String::new() }, 1.0);
        source_weights.insert(SignalSource::EntityRecognition { entity_type: String::new() }, 0.8);
        
        Self {
            decay_manager,
            source_weights,
        }
    }

    /// Set weight for a signal source
    pub fn set_source_weight(&mut self, source: SignalSource, weight: f64) {
        self.source_weights.insert(source, weight.clamp(0.0, 2.0));
    }

    /// Compute fused alpha signal for a symbol
    pub fn compute_fused_alpha(&self, symbol: &str) -> Option<FusedAlphaSignal> {
        let signals = self.decay_manager.get_all_active_signals();
        
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        let mut signal_count = 0;
        
        for (key, signal) in signals {
            if let Some(sig_symbol) = &signal.symbol {
                if sig_symbol == symbol {
                    // Get weight for this source type
                    let weight = self.source_weights
                        .get(&signal.source)
                        .copied()
                        .unwrap_or(1.0);
                    
                    weighted_sum += signal.current_score * weight;
                    total_weight += weight;
                    signal_count += 1;
                }
            }
        }
        
        if signal_count > 0 && total_weight > 0.0 {
            Some(FusedAlphaSignal {
                symbol: symbol.to_string(),
                fused_score: weighted_sum / total_weight,
                signal_count,
                conviction: ((weighted_sum / total_weight + 1.0) / 2.0).clamp(0.0, 1.0),
                timestamp_ns: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
            })
        } else {
            None
        }
    }
}

/// Fused alpha signal combining multiple sources
#[derive(Debug, Clone)]
pub struct FusedAlphaSignal {
    /// Target symbol
    pub symbol: String,
    /// Fused score (-1.0 to 1.0)
    pub fused_score: f64,
    /// Number of signals contributing
    pub signal_count: usize,
    /// Conviction score (0.0 to 1.0)
    pub conviction: f64,
    /// Timestamp
    pub timestamp_ns: u128,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_decay() {
        let mut signal = DecayedSignal::new(1.0, SignalSource::Custom("test".to_string()), Some("AAPL".to_string()));
        signal.half_life_ms = 1000; // 1 second for testing
        
        // After one half-life, should be ~0.5
        signal.age_ms = 1000;
        signal.current_score = signal.calculate_decayed_value(1000);
        assert!((signal.current_score - 0.5).abs() < 0.01);
        
        // After two half-lives, should be ~0.25
        signal.age_ms = 2000;
        signal.current_score = signal.calculate_decayed_value(2000);
        assert!((signal.current_score - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_decay_manager() {
        let manager = SentimentDecayManager::new(DecayConfig::default());
        
        manager.add_signal(
            "sig1".to_string(),
            0.8,
            SignalSource::HawkishDovish { entity: "Fed".to_string() },
            Some("SPY".to_string()),
        );
        
        let signal = manager.get_signal("sig1");
        assert!(signal.is_some());
        assert_eq!(signal.unwrap().original_score, 0.8);
    }

    #[test]
    fn test_conviction_conversion() {
        let signal = DecayedSignal::new(1.0, SignalSource::Custom("test".to_string()), None);
        assert_eq!(signal.to_conviction(), 1.0);
        
        let signal = DecayedSignal::new(-1.0, SignalSource::Custom("test".to_string()), None);
        assert_eq!(signal.to_conviction(), 0.0);
        
        let signal = DecayedSignal::new(0.0, SignalSource::Custom("test".to_string()), None);
        assert_eq!(signal.to_conviction(), 0.5);
    }

    #[test]
    fn test_signal_aggregation() {
        let manager = SentimentDecayManager::new(DecayConfig::default());
        
        manager.add_signal("s1".to_string(), 0.6, SignalSource::Sentiment { model: "v1".to_string() }, Some("AAPL".to_string()));
        manager.add_signal("s2".to_string(), 0.8, SignalSource::Sentiment { model: "v2".to_string() }, Some("AAPL".to_string()));
        manager.add_signal("s3".to_string(), -0.4, SignalSource::Sentiment { model: "v3".to_string() }, Some("AAPL".to_string()));
        
        let aggregated = manager.aggregate_signal("AAPL");
        assert!(aggregated.is_some());
        // Should be positive since 2 positive signals outweigh 1 negative
        assert!(aggregated.unwrap() > 0.0);
    }
}
