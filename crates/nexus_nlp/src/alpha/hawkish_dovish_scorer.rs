//! Hawkish/Dovish Scorer for Central Bank Communications
//! 
//! Specialized classifier tuned for Federal Reserve FOMC statements,
//! detecting shifts in tone word-by-word as text is released.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Sentiment score ranging from -1.0 (dovish) to +1.0 (hawkish)
#[derive(Debug, Clone)]
pub struct HawkishDovishScore {
    pub score: f32,
    pub confidence: f32,
    pub word_count: usize,
    pub key_phrases_detected: Vec<String>,
}

impl HawkishDovishScore {
    /// Create a neutral score
    pub fn neutral() -> Self {
        Self {
            score: 0.0,
            confidence: 0.0,
            word_count: 0,
            key_phrases_detected: Vec::new(),
        }
    }
}

/// Lexicon of hawkish and dovish terms with weights
pub struct FedLexicon {
    hawkish_terms: HashMap<&'static str, f32>,
    dovish_terms: HashMap<&'static str, f32>,
    intensifiers: HashMap<&'static str, f32>,
    negations: Vec<&'static str>,
}

impl FedLexicon {
    /// Create the Federal Reserve sentiment lexicon
    pub fn new() -> Self {
        let mut hawkish_terms = HashMap::new();
        let mut dovish_terms = HashMap::new();
        let mut intensifiers = HashMap::new();
        let negations = vec!["not", "no", "never", "neither", "without"];

        // Hawkish terms (indicating tightening/inflation concerns)
        hawkish_terms.insert("hawkish", 0.8);
        hawkish_terms.insert("tightening", 0.7);
        hawkish_terms.insert("hike", 0.75);
        hawkish_terms.insert("increase", 0.5);
        hawkish_terms.insert("raise", 0.6);
        hawkish_terms.insert("inflation", 0.4);
        hawkish_terms.insert("overheating", 0.7);
        hawkish_terms.insert("restrictive", 0.6);
        hawkish_terms.insert("withdrawal", 0.5);
        hawkish_terms.insert("normalization", 0.5);
        hawkish_terms.insert("vigilant", 0.4);
        hawkish_terms.insert("concerned", 0.3);
        hawkish_terms.insert("upside", 0.3); // upside risks
        hawkish_terms.insert("persistent", 0.4); // persistent inflation
        hawkish_terms.insert("entrenched", 0.6);
        hawkish_terms.insert("accelerating", 0.5);
        hawkish_terms.insert("overrun", 0.4);
        hawkish_terms.insert("above-target", 0.6);
        hawkish_terms.insert("further", 0.3); // further increases
        hawkish_terms.insert("additional", 0.3); // additional tightening

        // Dovish terms (indicating easing/growth concerns)
        dovish_terms.insert("dovish", 0.8);
        dovish_terms.insert("easing", 0.7);
        dovish_terms.insert("cut", 0.75);
        dovish_terms.insert("reduce", 0.5);
        dovish_terms.insert("lower", 0.6);
        dovish_terms.insert("stimulus", 0.6);
        dovish_terms.insert("accommodative", 0.7);
        dovish_terms.insert("supportive", 0.5);
        dovish_terms.insert("patient", 0.4);
        dovish_terms.insert("transitory", 0.5);
        dovish_terms.insert("temporary", 0.4);
        dovish_terms.insert("subdued", 0.4);
        dovish_terms.insert("weakness", 0.5);
        dovish_terms.insert("downside", 0.3); // downside risks
        dovish_terms.insert("slowing", 0.4);
        dovish_terms.insert("decelerating", 0.4);
        dovish_terms.insert("below-target", 0.6);
        dovish_terms.insert("undershoot", 0.5);
        dovish_terms.insert("headwinds", 0.4);
        dovish_terms.insert("fragile", 0.5);
        dovish_terms.insert("uncertainty", 0.3);

        // Intensifiers that amplify nearby sentiment
        intensifiers.insert("very", 1.3);
        intensifiers.insert("highly", 1.25);
        intensifiers.insert("extremely", 1.4);
        intensifiers.insert("significantly", 1.3);
        intensifiers.insert("substantially", 1.3);
        intensifiers.insert("considerably", 1.2);
        intensifiers.insert("markedly", 1.2);
        intensifiers.insert("strongly", 1.3);
        intensifiers.insert("firmly", 1.2);
        intensifiers.insert("clearly", 1.1);
        intensifiers.insert("definitely", 1.2);
        intensifiers.insert("certainly", 1.15);
        intensifiers.insert("particularly", 1.2);
        intensifiers.insert("especially", 1.25);

        Self {
            hawkish_terms,
            dovish_terms,
            intensifiers,
            negations,
        }
    }

    /// Score a single word
    pub fn score_word(&self, word: &str) -> f32 {
        let word_lower = word.to_lowercase();
        
        self.hawkish_terms.get(word_lower.as_str()).copied().unwrap_or(0.0)
            - self.dovish_terms.get(word_lower.as_str()).copied().unwrap_or(0.0)
    }

    /// Check if word is an intensifier
    pub fn get_intensifier(&self, word: &str) -> Option<f32> {
        let word_lower = word.to_lowercase();
        self.intensifiers.get(word_lower.as_str()).copied()
    }

    /// Check if word is a negation
    pub fn is_negation(&self, word: &str) -> bool {
        let word_lower = word.to_lowercase();
        self.negations.contains(&word_lower.as_str())
    }
}

impl Default for FedLexicon {
    fn default() -> Self {
        Self::new()
    }
}

/// Real-time hawkish/dovish scorer using streaming analysis
pub struct HawkishDovishScorer {
    lexicon: FedLexicon,
    sequence_id: AtomicU64,
    /// Rolling window for context-aware scoring
    context_window_size: usize,
}

impl HawkishDovishScorer {
    /// Create a new scorer
    pub fn new() -> Self {
        Self {
            lexicon: FedLexicon::new(),
            sequence_id: AtomicU64::new(0),
            context_window_size: 5, // Look back 5 words for context
        }
    }

    /// Score a complete text
    pub fn score(&self, text: &str) -> HawkishDovishScore {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut total_score: f32 = 0.0;
        let mut total_weight: f32 = 0.0;
        let mut key_phrases = Vec::new();
        let mut intensifier_stack: Vec<(f32, usize)> = Vec::new(); // (multiplier, position)

        for (i, word) in words.iter().enumerate() {
            // Clean word
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if clean_word.is_empty() {
                continue;
            }

            // Check for intensifier
            if let Some(mult) = self.lexicon.get_intensifier(clean_word) {
                intensifier_stack.push((mult, i));
            }

            // Pop old intensifiers outside window
            intensifier_stack.retain(|(_, pos)| i - pos <= self.context_window_size);

            // Calculate current intensifier multiplier
            let current_multiplier: f32 = intensifier_stack.iter()
                .map(|(m, _)| m)
                .product::<f32>();

            // Score the word
            let base_score = self.lexicon.score_word(clean_word);
            
            if base_score != 0.0 {
                // Check for negation in preceding words
                let is_negated = (i.saturating_sub(self.context_window_size)..i)
                    .any(|j| {
                        j < words.len() && 
                        self.lexicon.is_negation(words[j].trim_matches(|c: char| !c.is_alphanumeric()))
                    });

                let final_score = if is_negated {
                    -base_score * current_multiplier // Negate the sentiment
                } else {
                    base_score * current_multiplier
                };

                total_score += final_score.abs();
                total_weight += 1.0;

                if final_score.abs() > 0.5 {
                    key_phrases.push(clean_word.to_string());
                }
            }
        }

        let raw_score = if total_weight > 0.0 {
            total_score / total_weight
        } else {
            0.0
        };

        // Normalize to [-1, 1] range
        let normalized_score = raw_score.tanh();

        HawkishDovishScore {
            score: normalized_score,
            confidence: (total_weight / 10.0).min(1.0), // Confidence grows with word count
            word_count: words.len(),
            key_phrases_detected: key_phrases,
        }
    }

    /// Score streaming text incrementally (for real-time FOMC releases)
    pub fn score_incremental(&self, previous: &HawkishDovishScore, new_text: &str) -> HawkishDovishScore {
        let new_score = self.score(new_text);
        
        // Weighted average based on word counts
        let total_words = previous.word_count + new_score.word_count;
        
        if total_words == 0 {
            return HawkishDovishScore::neutral();
        }

        let prev_weight = previous.word_count as f32 / total_words as f32;
        let new_weight = new_score.word_count as f32 / total_words as f32;

        let combined_score = previous.score * prev_weight + new_score.score * new_weight;
        let combined_confidence = (previous.confidence * prev_weight + new_score.confidence * new_weight)
            .max(previous.confidence.max(new_score.confidence));

        let mut combined_phrases = previous.key_phrases_detected.clone();
        combined_phrases.extend(new_score.key_phrases_detected);

        HawkishDovishScore {
            score: combined_score,
            confidence: combined_confidence,
            word_count: total_words,
            key_phrases_detected: combined_phrases,
        }
    }

    /// Get next sequence ID for ordering
    pub fn next_sequence_id(&self) -> u64 {
        self.sequence_id.fetch_add(1, Ordering::Relaxed)
    }
}

impl Default for HawkishDovishScorer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hawkish_text() {
        let scorer = HawkishDovishScorer::new();
        let text = "The Committee is concerned about persistent inflation and may need to implement further rate hikes.";
        
        let score = scorer.score(text);
        assert!(score.score > 0.0, "Hawkish text should have positive score");
        assert!(!score.key_phrases_detected.is_empty());
    }

    #[test]
    fn test_dovish_text() {
        let scorer = HawkishDovishScorer::new();
        let text = "Inflation remains transitory and the economy needs continued accommodative support.";
        
        let score = scorer.score(text);
        assert!(score.score < 0.0, "Dovish text should have negative score");
    }

    #[test]
    fn test_neutral_text() {
        let scorer = HawkishDovishScorer::new();
        let text = "The meeting concluded at 2pm with no major announcements.";
        
        let score = scorer.score(text);
        assert!(score.score.abs() < 0.1, "Neutral text should have near-zero score");
    }

    #[test]
    fn test_incremental_scoring() {
        let scorer = HawkishDovishScorer::new();
        
        let initial = scorer.score("The Fed met today.");
        let updated = scorer.score_incremental(&initial, "Inflation is a major concern requiring action.");
        
        assert!(updated.word_count > initial.word_count);
        assert!(updated.score > initial.score, "Adding hawkish text should increase score");
    }

    #[test]
    fn test_negation_handling() {
        let scorer = HawkishDovishScorer::new();
        
        let hawkish = scorer.score("inflation is high");
        let negated = scorer.score("inflation is not high");
        
        // The negated version should have lower or opposite score
        assert!(negated.score < hawkish.score);
    }
}
