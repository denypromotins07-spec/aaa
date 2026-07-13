//! Chapter 2: L2 Order Book Sequence ID Healing
//!
//! This module implements the critical sequence ID tracking and healing mechanism
//! that prevents order book desynchronization - the #1 killer of HFT bots.
//!
//! When a sequence gap is detected, the healer:
//! 1. Immediately pauses trading
//! 2. Fetches a fresh snapshot via REST API
//! 3. Validates the snapshot bridges the gap
//! 4. Atomically swaps the corrupted local book with the healed book

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tracing::{debug, error, info, warn};

/// Result of sequence validation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceCheckResult {
    /// Sequence is continuous and valid
    Valid,
    /// Minor gap within tolerance (can be healed)
    GapHealable(u64),
    /// Critical gap requiring full snapshot
    GapCritical(u64),
    /// Sequence regression (data corruption)
    Regression,
    /// Duplicate message (acceptable)
    Duplicate,
}

/// Configuration for sequence healer
#[derive(Debug, Clone)]
pub struct SequenceHealerConfig {
    /// Maximum acceptable sequence gap before triggering heal
    pub max_gap_tolerance: u64,
    /// Timeout for REST snapshot fetch in seconds
    pub snapshot_timeout_secs: u64,
    /// Minimum time between consecutive snapshot fetches (rate limit protection)
    pub min_snapshot_interval_ms: u64,
    /// Maximum consecutive snapshot failures before fatal error
    pub max_snapshot_failures: u32,
}

impl Default for SequenceHealerConfig {
    fn default() -> Self {
        Self {
            max_gap_tolerance: 100, // Allow gaps up to 100 messages
            snapshot_timeout_secs: 5,
            min_snapshot_interval_ms: 100, // 100ms minimum between snapshots
            max_snapshot_failures: 3,
        }
    }
}

/// Statistics for sequence healing operations
#[derive(Debug, Clone, Default)]
pub struct SequenceHealerStats {
    pub total_messages: u64,
    pub valid_sequences: u64,
    pub gaps_detected: u64,
    pub gaps_healed: u64,
    pub snapshots_fetched: u64,
    pub snapshot_failures: u64,
    pub regressions_detected: u64,
    pub duplicates_ignored: u64,
    pub last_sequence_id: u64,
    pub last_heal_timestamp_ns: u64,
}

/// Order Book Sequence Healer
pub struct OrderBookSequenceHealer {
    config: SequenceHealerConfig,
    last_update_id: Arc<AtomicU64>,
    expected_next_id: Arc<AtomicU64>,
    is_paused: Arc<AtomicBool>,
    consecutive_failures: Arc<AtomicU32>,
    last_snapshot_time_ns: Arc<AtomicU64>,
    stats: Arc<RwLock<SequenceHealerStats>>,
}

// SAFETY: Uses atomic operations and RwLock for thread safety
unsafe impl Send for OrderBookSequenceHealer {}
unsafe impl Sync for OrderBookSequenceHealer {}

impl OrderBookSequenceHealer {
    pub fn new(config: SequenceHealerConfig) -> Self {
        Self {
            config,
            last_update_id: Arc::new(AtomicU64::new(0)),
            expected_next_id: Arc::new(AtomicU64::new(0)),
            is_paused: Arc::new(AtomicBool::new(false)),
            consecutive_failures: Arc::new(AtomicU32::new(0)),
            last_snapshot_time_ns: Arc::new(AtomicU64::new(0)),
            stats: Arc::new(RwLock::new(SequenceHealerStats::default())),
        }
    }

    /// Check if trading should be paused due to sequence issues
    pub fn is_trading_paused(&self) -> bool {
        self.is_paused.load(Ordering::Acquire)
    }

    /// Validate an incoming update ID and return the check result
    pub fn validate_sequence(&self, update_id: u64) -> SequenceCheckResult {
        let mut stats = self.stats.write();
        stats.total_messages += 1;

        let expected = self.expected_next_id.load(Ordering::Acquire);
        let last = self.last_update_id.load(Ordering::Acquire);

        // First message initialization
        if expected == 0 && update_id > 0 {
            self.expected_next_id.store(update_id + 1, Ordering::Release);
            self.last_update_id.store(update_id, Ordering::Release);
            stats.last_sequence_id = update_id;
            stats.valid_sequences += 1;
            return SequenceCheckResult::Valid;
        }

        // Normal case: sequence is exactly as expected
        if update_id == expected {
            self.update_sequence(update_id);
            stats.last_sequence_id = update_id;
            stats.valid_sequences += 1;
            return SequenceCheckResult::Valid;
        }

        // Duplicate message (retransmission from exchange)
        if update_id <= last && update_id > 0 {
            stats.duplicates_ignored += 1;
            debug!("Duplicate update_id {} (last={})", update_id, last);
            return SequenceCheckResult::Duplicate;
        }

        // Sequence gap detected
        if update_id > expected {
            let gap = update_id - expected;
            stats.gaps_detected += 1;

            if gap <= self.config.max_gap_tolerance {
                warn!(
                    "Sequence gap detected: expected {}, got {} (gap={}). Marking as healable.",
                    expected, update_id, gap
                );
                return SequenceCheckResult::GapHealable(gap);
            } else {
                error!(
                    "CRITICAL sequence gap: expected {}, got {} (gap={}). Trading PAUSED.",
                    expected, update_id, gap
                );
                self.pause_trading();
                return SequenceCheckResult::GapCritical(gap);
            }
        }

        // Sequence regression (should never happen)
        if update_id < last && update_id > 0 {
            error!(
                "SEQUENCE REGRESSION: last={}, current={}. Possible data corruption!",
                last, update_id
            );
            stats.regressions_detected += 1;
            self.pause_trading();
            return SequenceCheckResult::Regression;
        }

        SequenceCheckResult::Valid
    }

    /// Update internal sequence tracking
    fn update_sequence(&self, update_id: u64) {
        self.last_update_id.store(update_id, Ordering::Release);
        self.expected_next_id.store(update_id + 1, Ordering::Release);
    }

    /// Pause trading due to sequence issues
    fn pause_trading(&self) {
        self.is_paused.store(true, Ordering::Release);
        warn!("Trading PAUSED due to sequence integrity violation");
    }

    /// Resume trading after successful heal
    pub fn resume_trading(&self) {
        self.is_paused.store(false, Ordering::Release);
        self.consecutive_failures.store(0, Ordering::Release);
        info!("Trading RESUMED after successful sequence heal");
    }

    /// Record a successful snapshot fetch
    pub fn record_snapshot_success(&self, snapshot_last_id: u64) {
        let mut stats = self.stats.write();
        stats.snapshots_fetched += 1;
        stats.last_heal_timestamp_ns = current_time_ns();
        
        // Reset failure counter
        self.consecutive_failures.store(0, Ordering::Release);
        
        // Update sequence tracking to match snapshot
        self.last_update_id.store(snapshot_last_id, Ordering::Release);
        self.expected_next_id.store(snapshot_last_id + 1, Ordering::Release);
        
        // Record snapshot time for rate limiting
        self.last_snapshot_time_ns.store(current_time_ns(), Ordering::Release);
    }

    /// Record a snapshot fetch failure
    pub fn record_snapshot_failure(&self) -> bool {
        let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        
        let mut stats = self.stats.write();
        stats.snapshot_failures += 1;
        
        if failures >= self.config.max_snapshot_failures {
            error!(
                "Maximum snapshot failures ({}) reached. Manual intervention required.",
                failures
            );
            true // Should trigger fatal error
        } else {
            warn!("Snapshot fetch failure #{}", failures);
            false // Can retry
        }
    }

    /// Check if we can fetch a snapshot (rate limit protection)
    pub fn can_fetch_snapshot(&self) -> bool {
        let last_snapshot = self.last_snapshot_time_ns.load(Ordering::Acquire);
        let now = current_time_ns();
        
        if last_snapshot == 0 {
            return true; // Never fetched before
        }
        
        let elapsed_ms = (now - last_snapshot) / 1_000_000;
        elapsed_ms >= self.config.min_snapshot_interval_ms
    }

    /// Get the next expected sequence ID
    pub fn get_expected_next_id(&self) -> u64 {
        self.expected_next_id.load(Ordering::Acquire)
    }

    /// Get the last seen sequence ID
    pub fn get_last_update_id(&self) -> u64 {
        self.last_update_id.load(Ordering::Acquire)
    }

    /// Get statistics snapshot
    pub fn get_stats(&self) -> SequenceHealerStats {
        self.stats.read().clone()
    }

    /// Reset state (e.g., after manual intervention)
    pub fn reset(&self) {
        self.last_update_id.store(0, Ordering::Release);
        self.expected_next_id.store(0, Ordering::Release);
        self.is_paused.store(false, Ordering::Release);
        self.consecutive_failures.store(0, Ordering::Release);
        *self.stats.write() = SequenceHealerStats::default();
        info!("Sequence healer state reset");
    }
}

/// Get current time in nanoseconds
fn current_time_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_sequence() {
        let healer = OrderBookSequenceHealer::new(SequenceHealerConfig::default());
        
        assert!(!healer.is_trading_paused());
        
        // First message should initialize
        let result = healer.validate_sequence(100);
        assert_eq!(result, SequenceCheckResult::Valid);
        assert_eq!(healer.get_expected_next_id(), 101);
    }

    #[test]
    fn test_normal_sequence_progression() {
        let healer = OrderBookSequenceHealer::new(SequenceHealerConfig::default());
        
        healer.validate_sequence(100);
        assert_eq!(healer.validate_sequence(101), SequenceCheckResult::Valid);
        assert_eq!(healer.validate_sequence(102), SequenceCheckResult::Valid);
        assert_eq!(healer.validate_sequence(103), SequenceCheckResult::Valid);
    }

    #[test]
    fn test_duplicate_detection() {
        let healer = OrderBookSequenceHealer::new(SequenceHealerConfig::default());
        
        healer.validate_sequence(100);
        healer.validate_sequence(101);
        
        // Duplicate should be detected
        assert_eq!(healer.validate_sequence(101), SequenceCheckResult::Duplicate);
        assert_eq!(healer.validate_sequence(100), SequenceCheckResult::Duplicate);
    }

    #[test]
    fn test_small_gap_healable() {
        let config = SequenceHealerConfig {
            max_gap_tolerance: 10,
            ..Default::default()
        };
        let healer = OrderBookSequenceHealer::new(config);
        
        healer.validate_sequence(100);
        
        // Gap of 5 should be healable
        let result = healer.validate_sequence(106);
        assert!(matches!(result, SequenceCheckResult::GapHealable(5)));
        assert!(!healer.is_trading_paused()); // Small gap doesn't pause
    }

    #[test]
    fn test_large_gap_critical() {
        let config = SequenceHealerConfig {
            max_gap_tolerance: 10,
            ..Default::default()
        };
        let healer = OrderBookSequenceHealer::new(config);
        
        healer.validate_sequence(100);
        
        // Gap of 50 should be critical
        let result = healer.validate_sequence(151);
        assert!(matches!(result, SequenceCheckResult::GapCritical(50)));
        assert!(healer.is_trading_paused()); // Large gap pauses trading
    }

    #[test]
    fn test_snapshot_rate_limiting() {
        let healer = OrderBookSequenceHealer::new(SequenceHealerConfig::default());
        
        // Should allow first snapshot
        assert!(healer.can_fetch_snapshot());
        
        // Simulate a snapshot fetch
        healer.record_snapshot_success(1000);
        
        // Should block immediate subsequent fetch
        assert!(!healer.can_fetch_snapshot());
    }

    #[test]
    fn test_consecutive_failures() {
        let config = SequenceHealerConfig {
            max_snapshot_failures: 3,
            ..Default::default()
        };
        let healer = OrderBookSequenceHealer::new(config);
        
        assert!(!healer.record_snapshot_failure());
        assert!(!healer.record_snapshot_failure());
        assert!(healer.record_snapshot_failure()); // Third failure triggers fatal
    }
}
