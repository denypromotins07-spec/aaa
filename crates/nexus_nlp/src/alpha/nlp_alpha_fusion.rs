//! NLP Alpha Fusion Module
//! 
//! Integrates NLP sentiment signals into the Stage 3 Signal Fusion engine
//! and Stage 8 Shared Memory RL Environment via zero-copy pointers.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use crate::alpha::sentiment_decay::{ConvictionScore, SentimentDecayCalculator, SignalSource};
use crate::alpha::hawkish_dovish_scorer::HawkishDovishScore;

/// Fused alpha signal combining multiple NLP sources
#[derive(Debug, Clone)]
pub struct FusedAlphaSignal {
    pub ticker: String,
    pub conviction: f32,
    pub confidence: f32,
    pub nlp_score: f32,
    pub decay_adjusted: bool,
    pub timestamp_us: u64,
    pub sources: Vec<SignalSource>,
}

impl FusedAlphaSignal {
    /// Create a neutral signal
    pub fn neutral(ticker: &str) -> Self {
        Self {
            ticker: ticker.to_string(),
            conviction: 0.0,
            confidence: 0.0,
            nlp_score: 0.0,
            decay_adjusted: false,
            timestamp_us: 0,
            sources: Vec::new(),
        }
    }
}

/// NLP Alpha Fusion Engine that combines signals from multiple NLP sources
pub struct NlpAlphaFusionEngine {
    decay_calculator: SentimentDecayCalculator,
    /// Minimum conviction threshold for signal generation
    min_conviction_threshold: f32,
    /// Maximum signals per ticker to prevent spam
    max_signals_per_ticker: usize,
    /// Global sequence counter for ordering
    sequence_counter: AtomicU64,
    /// Enabled flag for runtime control
    enabled: AtomicBool,
}

impl NlpAlphaFusionEngine {
    /// Create a new fusion engine
    pub fn new() -> Self {
        Self {
            decay_calculator: SentimentDecayCalculator::new(),
            min_conviction_threshold: 0.15,
            max_signals_per_ticker: 10,
            sequence_counter: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
        }
    }

    /// Fuse a hawkish/dovish score into an alpha signal
    pub fn fuse_hawkish_dovish(
        &self,
        ticker: &str,
        hd_score: &HawkishDovishScore,
    ) -> Option<FusedAlphaSignal> {
        if !self.enabled.load(Ordering::Relaxed) {
            return None;
        }

        // Convert hawkish/dovish to bullish/bearish conviction
        // Hawkish for USD = Bullish for USD
        let conviction_value = hd_score.score;
        
        let conviction = self.decay_calculator.create_conviction(
            conviction_value,
            hd_score.confidence,
            SignalSource::CentralBankStatement,
        );

        self.create_fused_signal(
            ticker,
            conviction,
            vec![SignalSource::CentralBankStatement],
        )
    }

    /// Fuse a general sentiment score into an alpha signal
    pub fn fuse_sentiment(
        &self,
        ticker: &str,
        sentiment_score: f32,
        confidence: f32,
        source: SignalSource,
    ) -> Option<FusedAlphaSignal> {
        if !self.enabled.load(Ordering::Relaxed) {
            return None;
        }

        let conviction = self.decay_calculator.create_conviction(
            sentiment_score,
            confidence,
            source,
        );

        self.create_fused_signal(ticker, conviction, vec![source])
    }

    /// Fuse multiple signals for the same ticker
    pub fn fuse_multiple(
        &self,
        ticker: &str,
        convictions: Vec<(f32, f32, SignalSource)>, // (score, confidence, source)
    ) -> Option<FusedAlphaSignal> {
        if !self.enabled.load(Ordering::Relaxed) || convictions.is_empty() {
            return None;
        }

        // Weighted average of convictions
        let mut total_value: f32 = 0.0;
        let mut total_confidence: f32 = 0.0;
        let mut sources = Vec::new();

        for (score, confidence, source) in convictions {
            let weight = confidence;
            total_value += score * weight;
            total_confidence += weight;
            sources.push(source);
        }

        if total_confidence > 0.0 {
            let avg_score = total_value / total_confidence;
            let avg_confidence = total_confidence / sources.len() as f32;

            let conviction = self.decay_calculator.create_conviction(
                avg_score,
                avg_confidence.min(1.0),
                sources[0], // Use first source for half-life
            );

            self.create_fused_signal(ticker, conviction, sources)
        } else {
            None
        }
    }

    /// Create a fused signal from a conviction score
    fn create_fused_signal(
        &self,
        ticker: &str,
        conviction: ConvictionScore,
        sources: Vec<SignalSource>,
    ) -> Option<FusedAlphaSignal> {
        // Apply decay based on time since creation
        let current_value = self.decay_calculator.current_value(&conviction);

        // Check threshold
        if current_value.abs() < self.min_conviction_threshold {
            return None;
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        Some(FusedAlphaSignal {
            ticker: ticker.to_string(),
            conviction: current_value,
            confidence: conviction.confidence,
            nlp_score: conviction.value,
            decay_adjusted: true,
            timestamp_us: now,
            sources,
        })
    }

    /// Get next sequence ID
    pub fn next_sequence_id(&self) -> u64 {
        self.sequence_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Enable/disable the fusion engine
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if engine is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Set minimum conviction threshold
    pub fn set_min_conviction_threshold(&mut self, threshold: f32) {
        self.min_conviction_threshold = threshold.clamp(0.0, 1.0);
    }
}

impl Default for NlpAlphaFusionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-copy shared memory writer for Stage 3/Stage 8 integration
pub struct SharedMemoryWriter {
    /// Pointer to shared memory region (simulated)
    memory_region: Arc<[u8]>,
    write_offset: AtomicUsize,
    is_mapped: AtomicBool,
}

use std::sync::atomic::AtomicUsize;

impl SharedMemoryWriter {
    /// Create a new shared memory writer
    pub fn new(size: usize) -> Self {
        let mut region = vec![0u8; size];
        
        // Write header magic bytes
        region[0..4].copy_from_slice(b"NLP\0");
        
        Self {
            memory_region: region.into_boxed_slice().into(),
            write_offset: AtomicUsize::new(64), // Reserve header space
            is_mapped: AtomicBool::new(false),
        }
    }

    /// Map the shared memory (simulate mapping)
    pub fn map(&self) -> Result<(), &'static str> {
        if self.is_mapped.swap(true, Ordering::AcqRel) {
            return Err("Already mapped");
        }
        Ok(())
    }

    /// Unmap the shared memory
    pub fn unmap(&self) {
        self.is_mapped.store(false, Ordering::Release);
    }

    /// Write a signal to shared memory (zero-copy simulation)
    pub fn write_signal(&self, signal: &FusedAlphaSignal) -> Result<usize, &'static str> {
        if !self.is_mapped.load(Ordering::Acquire) {
            return Err("Not mapped");
        }

        // Serialize signal to bytes
        let serialized = self.serialize_signal(signal);
        
        let offset = self.write_offset.fetch_add(serialized.len(), Ordering::AcqRel);
        
        if offset + serialized.len() > self.memory_region.len() {
            return Err("Out of memory");
        }

        // Zero-copy write (in real implementation, this would be direct memory access)
        // For simulation, we use regular slice copy
        unsafe {
            let ptr = self.memory_region.as_ptr().add(offset) as *mut u8;
            std::ptr::copy_nonoverlapping(serialized.as_ptr(), ptr, serialized.len());
        }

        Ok(offset)
    }

    /// Serialize signal to bytes
    fn serialize_signal(&self, signal: &FusedAlphaSignal) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(128);
        
        // Write ticker length and data
        let ticker_bytes = signal.ticker.as_bytes();
        bytes.push(ticker_bytes.len() as u8);
        bytes.extend_from_slice(ticker_bytes);
        
        // Write conviction
        bytes.extend_from_slice(&signal.conviction.to_le_bytes());
        
        // Write confidence
        bytes.extend_from_slice(&signal.confidence.to_le_bytes());
        
        // Write NLP score
        bytes.extend_from_slice(&signal.nlp_score.to_le_bytes());
        
        // Write flags
        let flags = if signal.decay_adjusted { 1u8 } else { 0u8 };
        bytes.push(flags);
        
        // Write timestamp
        bytes.extend_from_slice(&signal.timestamp_us.to_le_bytes());
        
        bytes
    }

    /// Read a signal from shared memory (zero-copy simulation)
    pub fn read_signal(&self, offset: usize) -> Option<FusedAlphaSignal> {
        if offset >= self.memory_region.len() {
            return None;
        }

        unsafe {
            let ptr = self.memory_region.as_ptr().add(offset) as *const u8;
            
            // Read ticker length
            let ticker_len = *ptr as usize;
            
            if offset + 1 + ticker_len + 24 > self.memory_region.len() {
                return None;
            }
            
            // Read ticker
            let ticker_bytes = std::slice::from_raw_parts(ptr.add(1), ticker_len);
            let ticker = String::from_utf8_lossy(ticker_bytes).to_string();
            
            let base = offset + 1 + ticker_len;
            
            // Read conviction
            let conv_bytes = std::slice::from_raw_parts(self.memory_region.as_ptr().add(base), 4);
            let conviction = f32::from_le_bytes(conv_bytes.try_into().unwrap_or([0; 4]));
            
            // Read confidence
            let conf_bytes = std::slice::from_raw_parts(self.memory_region.as_ptr().add(base + 4), 4);
            let confidence = f32::from_le_bytes(conf_bytes.try_into().unwrap_or([0; 4]));
            
            // Read NLP score
            let nlp_bytes = std::slice::from_raw_parts(self.memory_region.as_ptr().add(base + 8), 4);
            let nlp_score = f32::from_le_bytes(nlp_bytes.try_into().unwrap_or([0; 4]));
            
            // Read flags
            let flags = *self.memory_region.as_ptr().add(base + 12);
            let decay_adjusted = (flags & 1) != 0;
            
            // Read timestamp
            let ts_bytes = std::slice::from_raw_parts(self.memory_region.as_ptr().add(base + 13), 8);
            let timestamp_us = u64::from_le_bytes(ts_bytes.try_into().unwrap_or([0; 8]));
            
            Some(FusedAlphaSignal {
                ticker,
                conviction,
                confidence,
                nlp_score,
                decay_adjusted,
                timestamp_us,
                sources: Vec::new(),
            })
        }
    }

    /// Get current write offset
    pub fn current_offset(&self) -> usize {
        self.write_offset.load(Ordering::Relaxed)
    }

    /// Reset the writer
    pub fn reset(&self) {
        self.write_offset.store(64, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_engine() {
        let engine = NlpAlphaFusionEngine::new();
        
        let hd_score = HawkishDovishScore {
            score: 0.7,
            confidence: 0.9,
            word_count: 50,
            key_phrases_detected: vec!["inflation".to_string()],
        };
        
        let signal = engine.fuse_hawkish_dovish("USD", &hd_score);
        
        assert!(signal.is_some());
        let s = signal.unwrap();
        assert_eq!(s.ticker, "USD");
        assert!(s.conviction > 0.0);
    }

    #[test]
    fn test_shared_memory_writer() {
        let writer = SharedMemoryWriter::new(4096);
        writer.map().unwrap();
        
        let signal = FusedAlphaSignal {
            ticker: "AAPL".to_string(),
            conviction: 0.5,
            confidence: 0.8,
            nlp_score: 0.6,
            decay_adjusted: true,
            timestamp_us: 1234567890,
            sources: vec![SignalSource::NewsArticle],
        };
        
        let offset = writer.write_signal(&signal).unwrap();
        assert!(offset >= 64);
        
        // Read back
        let read_signal = writer.read_signal(offset);
        assert!(read_signal.is_some());
        let rs = read_signal.unwrap();
        assert_eq!(rs.ticker, "AAPL");
        assert!((rs.conviction - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_threshold_filtering() {
        let mut engine = NlpAlphaFusionEngine::new();
        engine.set_min_conviction_threshold(0.5);
        
        // Low conviction should be filtered
        let signal = engine.fuse_sentiment("TEST", 0.2, 0.5, SignalSource::NewsArticle);
        assert!(signal.is_none());
        
        // High conviction should pass
        let signal = engine.fuse_sentiment("TEST", 0.8, 0.9, SignalSource::NewsArticle);
        assert!(signal.is_some());
    }
}
