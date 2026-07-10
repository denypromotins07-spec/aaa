// STAGE 25: CHAPTER 3 - DATA FEED POISONER
// Intercepts Stage 9 NLP news feed and Stage 2 market data
// Tests Stage 24 Metacognitive Super-Ego's ontological drift detection

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Poison attack types
#[derive(Debug, Clone, PartialEq)]
pub enum PoisonType {
    SentimentFlip,      // Invert sentiment scores
    PriceCorruption,    // Add noise to prices
    VolumeSpike,        // Artificial volume injection
    TimestampDrift,     // Skew timestamps
    SymbolMismatch,     // Swap symbol identifiers
}

/// Poisoning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoisonConfig {
    pub poison_rate: f64,           // Probability of poisoning each message
    pub poison_types: Vec<PoisonType>,
    pub noise_magnitude: f64,       // Standard deviation of added noise
    pub target_feeds: Vec<String>,
}

/// Data message representation
#[derive(Debug, Clone)]
pub struct DataMessage {
    pub feed_id: String,
    pub symbol: String,
    pub timestamp_ns: u64,
    pub price: Option<u64>,
    pub volume: Option<u64>,
    pub sentiment_score: Option<f64>,
    pub payload: HashMap<String, String>,
}

/// Poisoning state
pub struct PoisonState {
    pub messages_poisoned: AtomicU64,
    pub total_messages: AtomicU64,
    pub active_attacks: AtomicU64,
}

impl Default for PoisonState {
    fn default() -> Self {
        Self {
            messages_poisoned: AtomicU64::new(0),
            total_messages: AtomicU64::new(0),
            active_attacks: AtomicU64::new(0),
        }
    }
}

/// Adversarial Data Feed Poisoner
pub struct DataFeedPoisoner {
    state: std::sync::Arc<PoisonState>,
    config: PoisonConfig,
    chaos_mode_flag: AtomicBool,
    rng_seed: u64,
}

impl DataFeedPoisoner {
    pub fn new(config: PoisonConfig, rng_seed: u64) -> Self {
        Self {
            state: std::sync::Arc::new(PoisonState::default()),
            config,
            chaos_mode_flag: AtomicBool::new(false),
            rng_seed,
        }
    }

    /// Activate chaos mode
    pub fn activate_chaos_mode(&self) {
        self.chaos_mode_flag.store(true, Ordering::SeqCst);
    }

    /// Deactivate chaos mode
    pub fn deactivate_chaos_mode(&self) {
        self.chaos_mode_flag.store(false, Ordering::SeqCst);
    }

    /// Check if chaos mode is active
    pub fn is_chaos_mode_active(&self) -> bool {
        self.chaos_mode_flag.load(Ordering::SeqCst)
    }

    /// Intercept and potentially poison a data message
    pub fn intercept_message(&self, mut msg: DataMessage) -> Result<DataMessage, PoisonError> {
        if !self.chaos_mode_flag.load(Ordering::SeqCst) {
            return Ok(msg);
        }

        self.state.total_messages.fetch_add(1, Ordering::Relaxed);

        let mut rng = rand::rngs::StdRng::seed_from_u64(
            self.rng_seed + msg.timestamp_ns
        );

        // Check if this message should be poisoned
        if rng.gen::<f64>() >= self.config.poison_rate {
            return Ok(msg);
        }

        // Apply random poison type from configured list
        if let Some(poison_type) = self.config.poison_types.get(
            rng.gen_range(0..self.config.poison_types.len())
        ) {
            match poison_type {
                PoisonType::SentimentFlip => {
                    if let Some(score) = msg.sentiment_score {
                        msg.sentiment_score = Some(-score);
                        self.state.messages_poisoned.fetch_add(1, Ordering::Relaxed);
                    }
                }
                PoisonType::PriceCorruption => {
                    if let Some(price) = msg.price {
                        let noise = (rng.gen::<f64>() * self.config.noise_magnitude) as i64;
                        let corrupted = (price as i64 + noise).max(0) as u64;
                        msg.price = Some(corrupted);
                        self.state.messages_poisoned.fetch_add(1, Ordering::Relaxed);
                    }
                }
                PoisonType::VolumeSpike => {
                    if let Some(volume) = msg.volume {
                        let multiplier = 1.0 + rng.gen::<f64>() * self.config.noise_magnitude * 10.0;
                        msg.volume = Some((volume as f64 * multiplier) as u64);
                        self.state.messages_poisoned.fetch_add(1, Ordering::Relaxed);
                    }
                }
                PoisonType::TimestampDrift => {
                    let drift_ns = (rng.gen::<f64>() * 1_000_000_000.0) as u64; // Up to 1 second
                    msg.timestamp_ns = msg.timestamp_ns.wrapping_add(drift_ns);
                    self.state.messages_poisoned.fetch_add(1, Ordering::Relaxed);
                }
                PoisonType::SymbolMismatch => {
                    // Swap last character of symbol
                    if msg.symbol.len() > 1 {
                        let mut chars: Vec<char> = msg.symbol.chars().collect();
                        chars.swap(chars.len() - 1, chars.len() - 2);
                        msg.symbol = chars.into_iter().collect();
                        self.state.messages_poisoned.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        Ok(msg)
    }

    /// Get poisoning statistics
    pub fn get_stats(&self) -> PoisonStats {
        PoisonStats {
            messages_poisoned: self.state.messages_poisoned.load(Ordering::Relaxed),
            total_messages: self.state.total_messages.load(Ordering::Relaxed),
            poison_rate_actual: {
                let total = self.state.total_messages.load(Ordering::Relaxed);
                let poisoned = self.state.messages_poisoned.load(Ordering::Relaxed);
                if total > 0 {
                    poisoned as f64 / total as f64
                } else {
                    0.0
                }
            },
            chaos_mode: self.chaos_mode_flag.load(Ordering::SeqCst),
        }
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.state.messages_poisoned.store(0, Ordering::Relaxed);
        self.state.total_messages.store(0, Ordering::Relaxed);
    }
}

/// Poisoning statistics
#[derive(Debug, Clone)]
pub struct PoisonStats {
    pub messages_poisoned: u64,
    pub total_messages: u64,
    pub poison_rate_actual: f64,
    pub chaos_mode: bool,
}

/// Poison errors
#[derive(Debug, Clone, PartialEq)]
pub enum PoisonError {
    ChaosModeNotActive,
    InvalidMessage,
    ConfigurationError,
}

impl std::fmt::Display for PoisonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoisonError::ChaosModeNotActive => write!(f, "Chaos mode not active"),
            PoisonError::InvalidMessage => write!(f, "Invalid message"),
            PoisonError::ConfigurationError => write!(f, "Configuration error"),
        }
    }
}

impl std::error::Error for PoisonError {}

/// Builder for poison configurations
pub struct PoisonConfigBuilder {
    poison_rate: f64,
    poison_types: Vec<PoisonType>,
    noise_magnitude: f64,
    target_feeds: Vec<String>,
}

impl PoisonConfigBuilder {
    pub fn new() -> Self {
        Self {
            poison_rate: 0.1,
            poison_types: vec![PoisonType::SentimentFlip],
            noise_magnitude: 0.05,
            target_feeds: vec!["nlp_news".to_string(), "market_data".to_string()],
        }
    }

    pub fn poison_rate(mut self, rate: f64) -> Self {
        self.poison_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn poison_type(mut self, poison_type: PoisonType) -> Self {
        self.poison_types.push(poison_type);
        self
    }

    pub fn noise_magnitude(mut self, mag: f64) -> Self {
        self.noise_magnitude = mag;
        self
    }

    pub fn target_feed(mut self, feed: &str) -> Self {
        self.target_feeds.push(feed.to_string());
        self
    }

    pub fn build(self) -> PoisonConfig {
        PoisonConfig {
            poison_rate: self.poison_rate,
            poison_types: self.poison_types,
            noise_magnitude: self.noise_magnitude,
            target_feeds: self.target_feeds,
        }
    }
}

impl Default for PoisonConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poison_without_chaos_mode() {
        let config = PoisonConfigBuilder::new()
            .poison_rate(1.0)
            .build();

        let poisoner = DataFeedPoisoner::new(config, 42);

        let msg = DataMessage {
            feed_id: "nlp_news".to_string(),
            symbol: "BTC".to_string(),
            timestamp_ns: 1000,
            price: Some(50000_000_000),
            volume: None,
            sentiment_score: Some(0.8),
            payload: HashMap::new(),
        };

        let result = poisoner.intercept_message(msg.clone());
        assert!(result.is_ok());
        
        // Without chaos mode, message should be unchanged
        let returned = result.unwrap();
        assert_eq!(returned.sentiment_score, msg.sentiment_score);
    }

    #[test]
    fn test_sentiment_flip() {
        let config = PoisonConfigBuilder::new()
            .poison_rate(1.0)
            .poison_type(PoisonType::SentimentFlip)
            .build();

        let poisoner = DataFeedPoisoner::new(config, 42);
        poisoner.activate_chaos_mode();

        let msg = DataMessage {
            feed_id: "nlp_news".to_string(),
            symbol: "BTC".to_string(),
            timestamp_ns: 1000,
            price: None,
            volume: None,
            sentiment_score: Some(0.8),
            payload: HashMap::new(),
        };

        let result = poisoner.intercept_message(msg).unwrap();
        assert_eq!(result.sentiment_score, Some(-0.8));

        let stats = poisoner.get_stats();
        assert_eq!(stats.messages_poisoned, 1);
    }
}
