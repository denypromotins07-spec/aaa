//! Sentiment Decay Module
//! 
//! Translates NLP sentiment scores into Stage 3 ConvictionScores with
//! exponential time-decay so alpha signals fade as the market absorbs news.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Conviction score for Stage 3 signal fusion
#[derive(Debug, Clone)]
pub struct ConvictionScore {
    pub value: f32,      // -1.0 to +1.0 (bearish to bullish)
    pub confidence: f32, // 0.0 to 1.0
    pub decay_rate: f32, // per-second decay factor
    pub created_at: u64, // timestamp in microseconds
    pub source: SignalSource,
}

/// Source of the signal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalSource {
    NewsArticle,
    SocialMedia,
    CentralBankStatement,
    EarningsReport,
    EconomicData,
    AnalystUpgrade,
    RegulatoryFiling,
}

impl SignalSource {
    /// Get default half-life for each source type
    pub fn default_half_life(&self) -> Duration {
        match self {
            SignalSource::NewsArticle => Duration::from_secs(300), // 5 minutes
            SignalSource::SocialMedia => Duration::from_secs(60),  // 1 minute
            SignalSource::CentralBankStatement => Duration::from_secs(1800), // 30 minutes
            SignalSource::EarningsReport => Duration::from_secs(600), // 10 minutes
            SignalSource::EconomicData => Duration::from_secs(120), // 2 minutes
            SignalSource::AnalystUpgrade => Duration::from_secs(900), // 15 minutes
            SignalSource::RegulatoryFiling => Duration::from_secs(600), // 10 minutes
        }
    }
}

/// Sentiment decay calculator
pub struct SentimentDecayCalculator {
    /// Base decay constant (lambda)
    decay_lambda: f32,
    sequence_counter: AtomicU64,
    total_signals: AtomicUsize,
}

impl SentimentDecayCalculator {
    /// Create a new decay calculator
    pub fn new() -> Self {
        Self {
            decay_lambda: 0.001, // Default decay rate
            sequence_counter: AtomicU64::new(0),
            total_signals: AtomicUsize::new(0),
        }
    }

    /// Create a conviction score from raw sentiment
    pub fn create_conviction(
        &self,
        sentiment_score: f32,
        sentiment_confidence: f32,
        source: SignalSource,
    ) -> ConvictionScore {
        let half_life = source.default_half_life();
        let decay_rate = Self::calculate_decay_rate(half_life);
        
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        self.total_signals.fetch_add(1, Ordering::Relaxed);

        ConvictionScore {
            value: sentiment_score.clamp(-1.0, 1.0),
            confidence: sentiment_confidence.clamp(0.0, 1.0),
            decay_rate,
            created_at: now,
            source,
        }
    }

    /// Calculate decay rate from half-life
    fn calculate_decay_rate(half_life: Duration) -> f32 {
        let half_life_secs = half_life.as_secs_f32();
        // lambda = ln(2) / half_life
        std::f32::consts::LN_2 / half_life_secs
    }

    /// Apply time decay to a conviction score
    pub fn apply_decay(&self, score: &ConvictionScore, elapsed_secs: f32) -> ConvictionScore {
        // Exponential decay: value * e^(-lambda * t)
        let decay_factor = (-self.decay_lambda * elapsed_secs).exp();
        
        // Also decay confidence more slowly
        let confidence_decay = (-self.decay_lambda * elapsed_secs * 0.5).exp();

        ConvictionScore {
            value: score.value * decay_factor,
            confidence: score.confidence * confidence_decay,
            decay_rate: score.decay_rate,
            created_at: score.created_at,
            source: score.source,
        }
    }

    /// Get current value of a conviction score after decay
    pub fn current_value(&self, score: &ConvictionScore) -> f32 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        
        let elapsed_micros = now.saturating_sub(score.created_at);
        let elapsed_secs = elapsed_micros as f32 / 1_000_000.0;
        
        // Use score's own decay rate
        let decay_factor = (-score.decay_rate * elapsed_secs).exp();
        score.value * decay_factor
    }

    /// Get next sequence ID
    pub fn next_sequence_id(&self) -> u64 {
        self.sequence_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Get total signals processed
    pub fn total_signals(&self) -> usize {
        self.total_signals.load(Ordering::Relaxed)
    }

    /// Adjust global decay lambda
    pub fn set_decay_lambda(&mut self, lambda: f32) {
        self.decay_lambda = lambda.max(0.0001).min(1.0);
    }
}

impl Default for SentimentDecayCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Rolling window tracker for decaying signals
pub struct RollingSignalWindow {
    signals: Vec<(ConvictionScore, Instant)>,
    max_age: Duration,
    capacity: usize,
}

impl RollingSignalWindow {
    /// Create a new rolling window
    pub fn new(max_age: Duration, capacity: usize) -> Self {
        Self {
            signals: Vec::with_capacity(capacity),
            max_age,
            capacity,
        }
    }

    /// Add a signal to the window
    pub fn add(&mut self, score: ConvictionScore) {
        let now = Instant::now();
        
        // Remove expired signals
        self.prune(now);
        
        // Add new signal
        if self.signals.len() < self.capacity {
            self.signals.push((score, now));
        } else if !self.signals.is_empty() {
            // Replace oldest
            self.signals.remove(0);
            self.signals.push((score, now));
        }
    }

    /// Get aggregated conviction from all signals in window
    pub fn aggregate(&self, calculator: &SentimentDecayCalculator) -> f32 {
        if self.signals.is_empty() {
            return 0.0;
        }

        let now = Instant::now();
        let mut total_value: f32 = 0.0;
        let mut total_weight: f32 = 0.0;

        for (score, added_at) in &self.signals {
            let elapsed = now.duration_since(*added_at).as_secs_f32();
            
            // Skip if expired
            if elapsed > self.max_age.as_secs_f32() {
                continue;
            }

            // Weight by confidence and recency
            let recency_weight = 1.0 - (elapsed / self.max_age.as_secs_f32());
            let weight = score.confidence * recency_weight;
            
            let decayed_value = calculator.current_value(score);
            total_value += decayed_value * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            total_value / total_weight
        } else {
            0.0
        }
    }

    /// Prune expired signals
    fn prune(&mut self, now: Instant) {
        let cutoff = now - self.max_age;
        self.signals.retain(|(_, added_at)| *added_at >= cutoff);
    }

    /// Get number of active signals
    pub fn len(&self) -> usize {
        self.signals.len()
    }

    /// Check if window is empty
    pub fn is_empty(&self) -> bool {
        self.signals.is_empty()
    }

    /// Clear all signals
    pub fn clear(&mut self) {
        self.signals.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conviction_creation() {
        let calculator = SentimentDecayCalculator::new();
        
        let score = calculator.create_conviction(
            0.8,
            0.9,
            SignalSource::NewsArticle,
        );
        
        assert!((score.value - 0.8).abs() < 0.01);
        assert!((score.confidence - 0.9).abs() < 0.01);
        assert!(score.decay_rate > 0.0);
    }

    #[test]
    fn test_decay_application() {
        let calculator = SentimentDecayCalculator::new();
        
        let score = calculator.create_conviction(
            1.0,
            1.0,
            SignalSource::SocialMedia,
        );
        
        // After one half-life, value should be ~0.5
        let half_life = SignalSource::SocialMedia.default_half_life();
        let decayed = calculator.apply_decay(&score, half_life.as_secs_f32());
        
        assert!(decayed.value.abs() < 0.6);
    }

    #[test]
    fn test_rolling_window() {
        let calculator = SentimentDecayCalculator::new();
        let mut window = RollingSignalWindow::new(Duration::from_secs(60), 10);
        
        // Add bullish signals
        for i in 0..3 {
            window.add(calculator.create_conviction(
                0.8,
                0.9,
                SignalSource::NewsArticle,
            ));
        }
        
        // Add bearish signal
        window.add(calculator.create_conviction(
            -0.6,
            0.8,
            SignalSource::NewsArticle,
        ));
        
        let aggregated = window.aggregate(&calculator);
        
        // Should be positive (more bullish signals)
        assert!(aggregated > 0.0);
    }

    #[test]
    fn test_source_half_lives() {
        assert!(SignalSource::SocialMedia.default_half_life() 
            < SignalSource::CentralBankStatement.default_half_life());
    }
}
