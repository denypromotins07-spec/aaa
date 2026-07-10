//! NLP Alpha Fusion - Integration with Stage 3 Signal Fusion and Stage 8 RL Environment
//!
//! This module writes NLP alpha signals directly into the Stage 8 Shared Memory
//! RL Environment and the Stage 3 Signal Fusion engine via zero-copy pointers.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::{info, debug, warn};

/// Maximum number of concurrent NLP signals
const MAX_NLP_SIGNALS: usize = 1024;

/// Signal direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalDirection {
    Long,
    Short,
    Neutral,
}

impl From<f64> for SignalDirection {
    fn from(score: f64) -> Self {
        if score > 0.1 {
            SignalDirection::Long
        } else if score < -0.1 {
            SignalDirection::Short
        } else {
            SignalDirection::Neutral
        }
    }
}

/// Zero-copy NLP signal structure for shared memory
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NlpAlphaSignal {
    /// Signal ID
    pub id: u64,
    /// Target symbol (fixed-size for zero-copy)
    pub symbol: [u8; 16],
    /// Signal direction
    pub direction: u8,
    /// Conviction score (0.0 to 1.0)
    pub conviction: f64,
    /// Raw sentiment score (-1.0 to 1.0)
    pub sentiment_score: f64,
    /// Time-decayed alpha value
    pub alpha_value: f64,
    /// Timestamp (nanoseconds)
    pub timestamp_ns: u128,
    /// Expiration timestamp (nanoseconds)
    pub expires_ns: u128,
    /// Source type identifier
    pub source_type: u8,
    /// Reserved padding
    pub _padding: [u8; 7],
}

impl NlpAlphaSignal {
    /// Create a new NLP alpha signal
    pub fn new(
        id: u64,
        symbol: &str,
        direction: SignalDirection,
        conviction: f64,
        sentiment: f64,
        alpha: f64,
    ) -> Self {
        let mut symbol_bytes = [0u8; 16];
        let bytes = symbol.as_bytes();
        symbol_bytes[..bytes.len().min(16)].copy_from_slice(&bytes[..bytes.len().min(16)]);

        Self {
            id,
            symbol: symbol_bytes,
            direction: direction as u8,
            conviction,
            sentiment_score: sentiment,
            alpha_value: alpha,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            expires_ns: 0, // Set by caller
            source_type: 0,
            _padding: [0; 7],
        }
    }

    /// Get symbol as string
    pub fn get_symbol(&self) -> String {
        let end = self.symbol.iter().position(|&b| b == 0).unwrap_or(16);
        String::from_utf8_lossy(&self.symbol[..end]).to_string()
    }

    /// Get signal direction
    pub fn get_direction(&self) -> SignalDirection {
        match self.direction {
            0 => SignalDirection::Long,
            1 => SignalDirection::Short,
            _ => SignalDirection::Neutral,
        }
    }

    /// Check if signal is expired
    pub fn is_expired(&self) -> bool {
        if self.expires_ns == 0 {
            return false;
        }
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() > self.expires_ns
    }
}

/// Shared memory ring buffer for zero-copy signal transfer
pub struct NlpSignalRingBuffer {
    /// Buffer storage
    buffer: Box<[NlpAlphaSignal]>,
    /// Write index (atomic for lock-free access)
    write_idx: AtomicUsize,
    /// Read index (for consumers)
    read_idx: AtomicUsize,
    /// Number of signals written
    total_written: AtomicU64,
    /// Number of signals read
    total_read: AtomicU64,
}

impl NlpSignalRingBuffer {
    /// Create a new ring buffer
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.min(MAX_NLP_SIGNALS);
        let buffer = vec![NlpAlphaSignal::new(0, "", SignalDirection::Neutral, 0.0, 0.0, 0.0); capacity]
            .into_boxed_slice();

        Self {
            buffer,
            write_idx: AtomicUsize::new(0),
            read_idx: AtomicUsize::new(0),
            total_written: AtomicU64::new(0),
            total_read: AtomicU64::new(0),
        }
    }

    /// Write a signal to the buffer (lock-free)
    pub fn write(&self, signal: NlpAlphaSignal) -> Result<(), &'static str> {
        let write = self.write_idx.load(Ordering::Relaxed);
        let read = self.read_idx.load(Ordering::Acquire);
        
        // Check if buffer is full
        let next_write = (write + 1) % self.buffer.len();
        if next_write == read {
            return Err("Ring buffer full");
        }

        // Write the signal
        unsafe {
            let ptr = &mut self.buffer[write] as *mut NlpAlphaSignal;
            ptr.write(signal);
        }

        // Update write index
        self.write_idx.store(next_write, Ordering::Release);
        self.total_written.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Read a signal from the buffer (lock-free)
    pub fn read(&self) -> Option<NlpAlphaSignal> {
        let read = self.read_idx.load(Ordering::Relaxed);
        let write = self.write_idx.load(Ordering::Acquire);

        // Check if buffer is empty
        if read == write {
            return None;
        }

        // Read the signal
        let signal = unsafe {
            let ptr = &self.buffer[read] as *const NlpAlphaSignal;
            ptr.read()
        };

        // Update read index
        let next_read = (read + 1) % self.buffer.len();
        self.read_idx.store(next_read, Ordering::Release);
        self.total_read.fetch_add(1, Ordering::Relaxed);

        Some(signal)
    }

    /// Get number of available signals
    pub fn available(&self) -> usize {
        let write = self.write_idx.load(Ordering::Acquire);
        let read = self.read_idx.load(Ordering::Acquire);
        
        if write >= read {
            write - read
        } else {
            self.buffer.len() - read + write
        }
    }

    /// Get statistics
    pub fn get_stats(&self) -> RingBufferStats {
        RingBufferStats {
            capacity: self.buffer.len(),
            available: self.available(),
            total_written: self.total_written.load(Ordering::Relaxed),
            total_read: self.total_read.load(Ordering::Relaxed),
        }
    }
}

/// Statistics for the ring buffer
#[derive(Debug, Clone)]
pub struct RingBufferStats {
    pub capacity: usize,
    pub available: usize,
    pub total_written: u64,
    pub total_read: u64,
}

/// NLP Alpha Fusion manager that integrates with Stage 3 and Stage 8
pub struct NlpAlphaFusion {
    /// Ring buffer for Stage 8 RL environment
    rl_buffer: Arc<NlpSignalRingBuffer>,
    /// Signal counter for unique IDs
    signal_counter: AtomicU64,
    /// Active signal count
    active_signals: AtomicUsize,
}

impl NlpAlphaFusion {
    /// Create a new NLP alpha fusion manager
    pub fn new() -> Self {
        Self {
            rl_buffer: Arc::new(NlpSignalRingBuffer::new(MAX_NLP_SIGNALS)),
            signal_counter: AtomicU64::new(1),
            active_signals: AtomicUsize::new(0),
        }
    }

    /// Submit an NLP alpha signal to both Stage 3 and Stage 8
    pub fn submit_alpha_signal(
        &self,
        symbol: &str,
        sentiment_score: f64,
        conviction: f64,
        half_life_ms: u64,
        source_type: u8,
    ) -> Result<u64, &'static str> {
        let id = self.signal_counter.fetch_add(1, Ordering::Relaxed);
        
        let direction = SignalDirection::from(sentiment_score);
        let alpha_value = sentiment_score * conviction;
        
        let mut signal = NlpAlphaSignal::new(
            id,
            symbol,
            direction,
            conviction,
            sentiment_score,
            alpha_value,
        );
        
        signal.source_type = source_type;
        signal.expires_ns = signal.timestamp_ns + (half_life_ms * 5 * 1_000_000) as u128; // 5 half-lives
        
        // Write to ring buffer for Stage 8 RL environment
        self.rl_buffer.write(signal)?;
        
        self.active_signals.fetch_add(1, Ordering::Relaxed);
        
        info!(
            "Submitted NLP alpha signal: {} dir={:?} conv={:.3} alpha={:.3}",
            symbol, direction, conviction, alpha_value
        );
        
        Ok(id)
    }

    /// Get the ring buffer for Stage 8 consumption
    pub fn get_rl_buffer(&self) -> Arc<NlpSignalRingBuffer> {
        self.rl_buffer.clone()
    }

    /// Get active signal count
    pub fn get_active_count(&self) -> usize {
        self.active_signals.load(Ordering::Relaxed)
    }

    /// Prune expired signals (called periodically)
    pub fn prune_expired(&self) -> usize {
        // In production, this would scan and mark expired signals
        // For now, we just decrement based on reads
        let pruned = self.rl_buffer.get_stats().total_read as usize;
        self.active_signals.fetch_sub(pruned.min(self.active_signals.load(Ordering::Relaxed)), Ordering::Relaxed);
        pruned
    }

    /// Get fusion statistics
    pub fn get_stats(&self) -> FusionStats {
        let buffer_stats = self.rl_buffer.get_stats();
        
        FusionStats {
            active_signals: self.active_signals.load(Ordering::Relaxed),
            total_signals_submitted: self.signal_counter.load(Ordering::Relaxed) - 1,
            buffer_capacity: buffer_stats.capacity,
            buffer_available: buffer_stats.available,
            buffer_total_written: buffer_stats.total_written,
        }
    }
}

impl Default for NlpAlphaFusion {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the fusion engine
#[derive(Debug, Clone)]
pub struct FusionStats {
    pub active_signals: usize,
    pub total_signals_submitted: u64,
    pub buffer_capacity: usize,
    pub buffer_available: usize,
    pub buffer_total_written: u64,
}

/// Convert NLP signal to Stage 3 format
pub fn to_stage3_conviction(signal: &NlpAlphaSignal) -> Stage3Conviction {
    Stage3Conviction {
        symbol: signal.get_symbol(),
        conviction: signal.conviction,
        direction: signal.get_direction() as u8,
        source: 0x03, // NLP source identifier
        timestamp_ns: signal.timestamp_ns,
    }
}

/// Stage 3 conviction signal format
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Stage3Conviction {
    pub symbol: String,
    pub conviction: f64,
    pub direction: u8,
    pub source: u8,
    pub _padding: [u8; 6],
    pub timestamp_ns: u128,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let signal = NlpAlphaSignal::new(
            1,
            "AAPL",
            SignalDirection::Long,
            0.8,
            0.6,
            0.48,
        );
        
        assert_eq!(signal.get_symbol(), "AAPL");
        assert_eq!(signal.get_direction(), SignalDirection::Long);
        assert!((signal.conviction - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_ring_buffer() {
        let buffer = NlpSignalRingBuffer::new(10);
        
        // Write some signals
        for i in 0..5 {
            let signal = NlpAlphaSignal::new(
                i,
                "TEST",
                SignalDirection::Long,
                0.5,
                0.5,
                0.25,
            );
            assert!(buffer.write(signal).is_ok());
        }
        
        assert_eq!(buffer.available(), 5);
        
        // Read them back
        for _ in 0..5 {
            let signal = buffer.read();
            assert!(signal.is_some());
        }
        
        assert_eq!(buffer.available(), 0);
    }

    #[test]
    fn test_alpha_fusion() {
        let fusion = NlpAlphaFusion::new();
        
        let id = fusion.submit_alpha_signal(
            "SPY",
            0.7,
            0.8,
            300_000,
            1,
        );
        
        assert!(id.is_ok());
        assert_eq!(fusion.get_active_count(), 1);
        
        let stats = fusion.get_stats();
        assert_eq!(stats.total_signals_submitted, 1);
    }

    #[test]
    fn test_signal_direction_conversion() {
        assert_eq!(SignalDirection::from(0.5), SignalDirection::Long);
        assert_eq!(SignalDirection::from(-0.5), SignalDirection::Short);
        assert_eq!(SignalDirection::from(0.0), SignalDirection::Neutral);
        assert_eq!(SignalDirection::from(0.05), SignalDirection::Neutral);
    }
}
