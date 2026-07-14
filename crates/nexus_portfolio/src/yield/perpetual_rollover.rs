//! Perpetual Rollover Manager - Decides whether to hold through funding prints.
//! 
//! CRITICAL: Uses exchange server time, NOT local system clock, to avoid timezone drift.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Funding print times (UTC): 00:00, 08:00, 16:00
const FUNDING_INTERVAL_MS: u64 = 8 * 60 * 60 * 1000; // 8 hours in ms

/// Decision for rollover
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RolloverDecision {
    HoldThroughPrint,      // Collect funding
    CloseBeforePrint,      // Avoid negative funding/volatility
    Neutral,               // No strong signal
}

/// State of the rollover manager
pub struct PerpetualRolloverManager {
    /// Exchange server timestamp (ms since epoch)
    exchange_server_time_ms: AtomicU64,
    /// Time until next funding print (ms)
    ms_until_funding: AtomicU64,
    /// Flag indicating we're in the danger zone (< 30 seconds to print)
    in_danger_zone: AtomicBool,
    /// Current predicted funding rate (scaled by 1e12)
    predicted_rate_scaled: i128,
}

impl PerpetualRolloverManager {
    pub fn new() -> Self {
        Self {
            exchange_server_time_ms: AtomicU64::new(0),
            ms_until_funding: AtomicU64::new(0),
            in_danger_zone: AtomicBool::new(false),
            predicted_rate_scaled: 0,
        }
    }

    /// Update with exchange server time (CRITICAL: must come from exchange API, not local clock)
    pub fn update_exchange_time(&self, server_time_ms: u64) {
        self.exchange_server_time_ms.store(server_time_ms, Ordering::SeqCst);
        self.calculate_time_to_funding(server_time_ms);
    }

    /// Calculate time until next funding print
    fn calculate_time_to_funding(&self, server_time_ms: u64) {
        let ms_since_epoch_start = server_time_ms % FUNDING_INTERVAL_MS;
        let ms_until_next = FUNDING_INTERVAL_MS - ms_since_epoch_start;
        
        self.ms_until_funding.store(ms_until_next, Ordering::Relaxed);
        
        // Danger zone: < 30 seconds to funding print
        if ms_until_next < 30_000 {
            self.in_danger_zone.store(true, Ordering::Relaxed);
        } else {
            self.in_danger_zone.store(false, Ordering::Relaxed);
        }
    }

    /// Set predicted funding rate
    pub fn set_predicted_rate(&mut self, rate_scaled: i128) {
        self.predicted_rate_scaled = rate_scaled;
    }

    /// Get decision: hold or close before print
    /// 
    /// Logic:
    /// - If predicted rate > 0 (we receive funding): HOLD
    /// - If predicted rate < 0 (we pay funding) AND |rate| > threshold: CLOSE
    /// - If in danger zone (< 30s) and rate is negative/volatile: CLOSE
    pub fn get_rollover_decision(&self, volatility_threshold_scaled: i128) -> RolloverDecision {
        let rate = self.predicted_rate_scaled;
        
        if rate > 0 {
            // We receive funding - hold through print
            RolloverDecision::HoldThroughPrint
        } else if rate < -volatility_threshold_scaled {
            // We pay significant funding - close before print
            RolloverDecision::CloseBeforePrint
        } else if self.in_danger_zone.load(Ordering::Relaxed) {
            // In danger zone with uncertain/negative rate - close to avoid volatility
            RolloverDecision::CloseBeforePrint
        } else {
            RolloverDecision::Neutral
        }
    }

    /// Get milliseconds until next funding print
    pub fn get_ms_until_funding(&self) -> u64 {
        self.ms_until_funding.load(Ordering::Relaxed)
    }

    /// Check if in danger zone
    pub fn is_in_danger_zone(&self) -> bool {
        self.in_danger_zone.load(Ordering::Relaxed)
    }

    /// Get current exchange server time
    pub fn get_exchange_server_time(&self) -> u64 {
        self.exchange_server_time_ms.load(Ordering::Relaxed)
    }
}

impl Default for PerpetualRolloverManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rollover_decision_positive_rate() {
        let mut manager = PerpetualRolloverManager::new();
        
        // Simulate exchange time: 7 hours 59 minutes into interval
        // (1 minute until funding)
        let server_time = 7 * 60 * 60 * 1000 + 59 * 60 * 1000;
        manager.update_exchange_time(server_time);
        
        // Positive funding rate (we receive)
        manager.set_predicted_rate(100_000_000); // 0.01%
        
        let decision = manager.get_rollover_decision(50_000_000);
        assert_eq!(decision, RolloverDecision::HoldThroughPrint);
    }

    #[test]
    fn test_rollover_decision_negative_rate() {
        let mut manager = PerpetualRolloverManager::new();
        
        let server_time = 7 * 60 * 60 * 1000 + 59 * 60 * 1000;
        manager.update_exchange_time(server_time);
        
        // Negative funding rate (we pay)
        manager.set_predicted_rate(-200_000_000); // -0.02%
        
        let decision = manager.get_rollover_decision(50_000_000);
        assert_eq!(decision, RolloverDecision::CloseBeforePrint);
    }

    #[test]
    fn test_danger_zone_detection() {
        let manager = PerpetualRolloverManager::new();
        
        // 25 seconds until funding
        let server_time = FUNDING_INTERVAL_MS - 25_000;
        manager.update_exchange_time(server_time);
        
        assert!(manager.is_in_danger_zone());
        assert_eq!(manager.get_ms_until_funding(), 25_000);
    }
}
