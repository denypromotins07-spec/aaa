//! Hawkish/Dovish Sentence-Level Attention Scorer for Central Bank Communications
//!
//! This module implements a specialized classifier tuned for Federal Reserve FOMC
//! statements, detecting shifts in tone word-by-word as text is released.

use std::sync::Arc;
use std::time::Instant;
use tracing::{info, debug};

/// Sentiment polarity for central bank communications
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FedTone {
    Hawkish,    // Tightening bias (rate hikes)
    Dovish,     // Easing bias (rate cuts)
    Neutral,    // No clear direction
}

/// Word-level sentiment contribution
#[derive(Debug, Clone)]
pub struct WordContribution {
    /// The word/token
    pub word: String,
    /// Position in text
    pub position: usize,
    /// Contribution to hawkish score (-1.0 to 1.0)
    pub hawkish_score: f64,
    /// Confidence in classification
    pub confidence: f64,
}

/// Result of hawkish/dovish analysis
#[derive(Debug, Clone)]
pub struct HawkishDovishResult {
    /// Overall tone classification
    pub tone: FedTone,
    /// Hawkish score (-1.0 = very dovish, 1.0 = very hawkish)
    pub hawkish_score: f64,
    /// Confidence in classification (0.0 to 1.0)
    pub confidence: f64,
    /// Word-level contributions for explainability
    pub word_contributions: Vec<WordContribution>,
    /// Change from previous statement (if available)
    pub delta: Option<f64>,
    /// Processing time (microseconds)
    pub processing_time_us: u64,
}

/// Configuration for the scorer
#[derive(Debug, Clone)]
pub struct ScorerConfig {
    /// Threshold for hawkish classification
    pub hawkish_threshold: f64,
    /// Threshold for dovish classification
    pub dovish_threshold: f64,
    /// Enable word-level attention
    pub enable_word_attention: bool,
    /// Minimum confidence for classification
    pub min_confidence: f64,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            hawkish_threshold: 0.3,
            dovish_threshold: -0.3,
            enable_word_attention: true,
            min_confidence: 0.5,
        }
    }
}

/// Pre-defined hawkish/dovish keyword lexicon
pub mod fed_lexicon {
    use std::collections::HashMap;

    /// Hawkish keywords and their scores
    pub fn hawkish_keywords() -> HashMap<&'static str, f64> {
        [
            ("inflation", 0.7),
            ("tightening", 0.9),
            ("hike", 0.8),
            ("increase", 0.5),
            ("restrictive", 0.8),
            ("aggressive", 0.7),
            ("persistent", 0.6),
            ("overshoot", 0.7),
            ("withdrawal", 0.6),
            ("contraction", 0.7),
            ("brake", 0.6),
            ("cool", 0.4),
            ("moderate", 0.3),
            ("contain", 0.5),
            ("combat", 0.6),
            ("anchor", 0.4),
            ("expectations", 0.3),
            ("wages", 0.4),
            ("employment", 0.3),
            ("overheating", 0.8),
            ("bubble", 0.6),
            ("exuberance", 0.7),
        ].into_iter().collect()
    }

    /// Dovish keywords and their scores
    pub fn dovish_keywords() -> HashMap<&'static str, f64> {
        [
            ("support", 0.6),
            ("accommodative", 0.8),
            ("stimulus", 0.7),
            ("easing", 0.8),
            ("cut", 0.7),
            ("lower", 0.5),
            ("patient", 0.5),
            ("transitory", 0.4),
            ("temporary", 0.3),
            ("flexible", 0.4),
            ("average", 0.3),
            ("targeting", 0.2),
            ("maximum", 0.3),
            ("employment", 0.4),
            ("recovery", 0.5),
            ("supportive", 0.6),
            ("headwinds", 0.4),
            ("risks", 0.3),
            ("uncertainty", 0.3),
            ("challenge", 0.3),
            ("weakness", 0.5),
            ("soft", 0.4),
        ].into_iter().collect()
    }

    /// Negation words that flip sentiment
    pub fn negations() -> Vec<&'static str> {
        vec!["not", "no", "never", "neither", "none", "unlikely", "disagree"]
    }

    /// Intensifiers that amplify sentiment
    pub fn intensifiers() -> HashMap<&'static str, f64> {
        [
            ("very", 1.3),
            ("extremely", 1.5),
            ("highly", 1.4),
            ("strongly", 1.4),
            ("significantly", 1.3),
            ("substantially", 1.3),
            ("considerably", 1.2),
            ("remarkably", 1.4),
            ("particularly", 1.2),
            ("especially", 1.3),
        ].into_iter().collect()
    }
}

/// Hawkish/Dovish sentence-level attention scorer
pub struct HawkishDovishScorer {
    /// Configuration
    config: ScorerConfig,
    /// Previous statement score for delta calculation
    previous_score: Arc<std::sync::atomic::AtomicI64>,
}

impl HawkishDovishScorer {
    /// Create a new scorer
    pub fn new(config: ScorerConfig) -> Self {
        Self {
            config,
            previous_score: Arc::new(std::sync::atomic::AtomicI64::new(0)),
        }
    }

    /// Analyze text for hawkish/dovish tone
    pub fn analyze(&self, text: &str) -> HawkishDovishResult {
        let start = Instant::now();

        let mut raw_score = 0.0;
        let mut total_weight = 0.0;
        let mut word_contributions = Vec::new();

        // Tokenize and analyze word by word
        let tokens: Vec<&str> = text.split_whitespace().collect();
        
        for (position, token) in tokens.iter().enumerate() {
            let token_lower = token.to_lowercase();
            let clean_token = clean_word(&token_lower);

            // Check for negations in preceding context
            let is_negated = check_negation(&tokens, position);
            
            // Check for intensifiers
            let intensifier = check_intensifier(&tokens, position);

            // Score the word
            let mut word_score = 0.0;
            let mut confidence = 0.0;

            if let Some(&score) = fed_lexicon::hawkish_keywords().get(clean_token.as_str()) {
                word_score = if is_negated { -score } else { score };
                confidence = 0.8;
            } else if let Some(&score) = fed_lexicon::dovish_keywords().get(clean_token.as_str()) {
                word_score = if is_negated { score } else { -score };
                confidence = 0.8;
            }

            // Apply intensifier
            if word_score != 0.0 {
                word_score *= intensifier;
            }

            if word_score != 0.0 {
                raw_score += word_score;
                total_weight += 1.0;

                if self.config.enable_word_attention {
                    word_contributions.push(WordContribution {
                        word: token.to_string(),
                        position,
                        hawkish_score: word_score,
                        confidence,
                    });
                }
            }
        }

        // Normalize score to [-1, 1]
        let normalized_score = if total_weight > 0.0 {
            (raw_score / total_weight).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Calculate confidence based on number of signal words found
        let confidence = (total_weight / (text.split_whitespace().count() as f64 * 0.1)).min(1.0);

        // Classify tone
        let tone = if normalized_score >= self.config.hawkish_threshold {
            FedTone::Hawkish
        } else if normalized_score <= self.config.dovish_threshold {
            FedTone::Dovish
        } else {
            FedTone::Neutral
        };

        // Calculate delta from previous statement
        let prev = self.previous_score.load(std::sync::atomic::Ordering::Relaxed);
        let delta = Some(normalized_score - (prev as f64 / 1000.0));

        // Store current score for future delta calculations
        self.previous_score.store(
            (normalized_score * 1000.0) as i64,
            std::sync::atomic::Ordering::Relaxed,
        );

        let processing_time_us = start.elapsed().as_micros() as u64;

        info!(
            "Hawkish/Dovish analysis: score={:.3}, tone={:?}, confidence={:.2}",
            normalized_score, tone, confidence
        );

        HawkishDovishResult {
            tone,
            hawkish_score: normalized_score,
            confidence,
            word_contributions,
            delta,
            processing_time_us,
        }
    }

    /// Analyze streaming text incrementally (for real-time FOMC releases)
    pub fn analyze_streaming(&self, text_chunk: &str, cumulative_score: &mut f64, word_count: &mut usize) -> HawkishDovishResult {
        let chunk_result = self.analyze(text_chunk);
        
        *cumulative_score += chunk_result.hawkish_score * chunk_result.word_contributions.len() as f64;
        *word_count += text_chunk.split_whitespace().count();

        let avg_score = *cumulative_score / (*word_count as f64).max(1.0);

        HawkishDovishResult {
            tone: if avg_score >= self.config.hawkish_threshold {
                FedTone::Hawkish
            } else if avg_score <= self.config.dovish_threshold {
                FedTone::Dovish
            } else {
                FedTone::Neutral
            },
            hawkish_score: avg_score,
            confidence: chunk_result.confidence,
            word_contributions: chunk_result.word_contributions,
            delta: chunk_result.delta,
            processing_time_us: chunk_result.processing_time_us,
        }
    }

    /// Get the top N most influential words from an analysis
    pub fn get_top_influential_words(result: &HawkishDovishResult, n: usize) -> Vec<&WordContribution> {
        let mut sorted: Vec<&WordContribution> = result.word_contributions.iter().collect();
        sorted.sort_by(|a, b| b.hawkish_score.abs().partial_cmp(&a.hawkish_score.abs()).unwrap());
        sorted.into_iter().take(n).collect()
    }
}

/// Clean a word by removing punctuation
fn clean_word(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

/// Check if there's a negation word before the given position
fn check_negation(tokens: &[&str], position: usize) -> bool {
    let search_range = (position.saturating_sub(3))..position;
    for i in search_range {
        let clean = clean_word(tokens[i]);
        if fed_lexicon::negations().iter().any(|&n| n == clean) {
            return true;
        }
    }
    false
}

/// Check for intensifier before the given position and return multiplier
fn check_intensifier(tokens: &[&str], position: usize) -> f64 {
    if position == 0 {
        return 1.0;
    }
    
    let prev_clean = clean_word(tokens[position - 1]);
    fed_lexicon::intensifiers()
        .get(prev_clean.as_str())
        .copied()
        .unwrap_or(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hawkish_text() {
        let scorer = HawkishDovishScorer::new(ScorerConfig::default());
        
        let text = "The Committee remains committed to combating persistent inflation \
                    and will continue aggressive tightening measures to contain \
                    price pressures and anchor expectations.";
        
        let result = scorer.analyze(text);
        
        assert_eq!(result.tone, FedTone::Hawkish);
        assert!(result.hawkish_score > 0.3);
    }

    #[test]
    fn test_dovish_text() {
        let scorer = HawkishDovishScorer::new(ScorerConfig::default());
        
        let text = "The Fed will maintain its supportive and accommodative stance \
                    to support the economic recovery, viewing current inflation \
                    pressures as transitory.";
        
        let result = scorer.analyze(text);
        
        assert_eq!(result.tone, FedTone::Dovish);
        assert!(result.hawkish_score < -0.3);
    }

    #[test]
    fn test_neutral_text() {
        let scorer = HawkishDovishScorer::new(ScorerConfig::default());
        
        let text = "The Committee reviewed economic data and discussed various \
                    policy options for the upcoming meeting.";
        
        let result = scorer.analyze(text);
        
        assert_eq!(result.tone, FedTone::Neutral);
    }

    #[test]
    fn test_negation_handling() {
        let scorer = HawkishDovishScorer::new(ScorerConfig::default());
        
        // Negated hawkish should be less hawkish
        let text1 = "Inflation is persistent";
        let text2 = "Inflation is not persistent";
        
        let result1 = scorer.analyze(text1);
        let result2 = scorer.analyze(text2);
        
        assert!(result1.hawkish_score > result2.hawkish_score);
    }

    #[test]
    fn test_word_contributions() {
        let scorer = HawkishDovishScorer::new(ScorerConfig {
            enable_word_attention: true,
            ..Default::default()
        });
        
        let text = "Aggressive tightening to combat persistent inflation";
        let result = scorer.analyze(text);
        
        assert!(!result.word_contributions.is_empty());
        
        let top_words = HawkishDovishScorer::get_top_influential_words(&result, 3);
        assert!(!top_words.is_empty());
    }
}
