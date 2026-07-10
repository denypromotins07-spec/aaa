//! Venue Latency Modeler.
//! Tracks EWMA of round-trip times and penalizes degraded venues.

use std::sync::atomic::{AtomicU64, Ordering};
use nexus_oms::FixedPoint;

const SCALE: i64 = 100_000_000;

/// Latency state for a single venue
pub struct VenueLatencyModel {
    /// EWMA of RTT in nanoseconds
    rtt_ewma_ns: AtomicU64,
    /// Variance of RTT (for confidence scoring)
    variance_ns: AtomicU64,
    /// Sample count
    sample_count: AtomicU64,
    /// Degraded threshold (nanoseconds)
    degraded_threshold_ns: AtomicU64,
    /// Penalty score (scaled by 10^8, lower is worse)
    penalty_score: AtomicU64,
}

impl VenueLatencyModel {
    #[inline]
    pub fn new(degraded_threshold_ms: u64) -> Self {
        Self {
            rtt_ewma_ns: AtomicU64::new(1_000_000), // Default 1ms
            variance_ns: AtomicU64::new(0),
            sample_count: AtomicU64::new(0),
            degraded_threshold_ns: AtomicU64::new(degraded_threshold_ms * 1_000_000),
            penalty_score: AtomicU64::new(SCALE as u64), // Full score initially
        }
    }

    /// Update RTT measurement using EWMA
    #[inline]
    pub fn update_rtt(&self, rtt_ns: u64, alpha: FixedPoint) {
        let current_ewma = self.rtt_ewma_ns.load(Ordering::Acquire);
        let count = self.sample_count.fetch_add(1, Ordering::Relaxed);

        // EWMA calculation
        let one_minus_alpha = FixedPoint::from_raw(SCALE) - alpha;
        let weighted_sample = ((rtt_ns as i128 * alpha.raw() as i128) / SCALE as i128) as u64;
        let weighted_current = ((current_ewma as i128 * one_minus_alpha.raw() as i128) / SCALE as i128) as u64;
        
        let new_ewma = weighted_sample.saturating_add(weighted_current).max(1);
        self.rtt_ewma_ns.store(new_ewma, Ordering::Release);

        // Update variance (simplified Welford's algorithm)
        if count > 0 {
            let diff = rtt_ns as i128 - current_ewma as i128;
            let variance_delta = (diff.abs() as u64 / 2).min(100_000_000); // Cap variance delta
            let current_var = self.variance_ns.load(Ordering::Acquire);
            let new_var = current_var.saturating_add(variance_delta / (count + 1));
            self.variance_ns.store(new_var, Ordering::Release);
        }

        // Update penalty score based on latency
        let threshold = self.degraded_threshold_ns.load(Ordering::Relaxed);
        let penalty = if new_ewma >= threshold {
            // Exponential decay as latency approaches threshold
            let ratio = (threshold as i128 / new_ewma.max(1) as i128) as f64;
            ((ratio * ratio * SCALE as f64) as u64).min(SCALE as u64)
        } else {
            SCALE as u64
        };
        
        self.penalty_score.store(penalty, Ordering::Release);
    }

    /// Get current EWMA RTT
    #[inline]
    pub fn get_rtt_ewma(&self) -> u64 {
        self.rtt_ewma_ns.load(Ordering::Acquire)
    }

    /// Get penalty score (scaled by 10^8)
    #[inline]
    pub fn get_penalty_score(&self) -> FixedPoint {
        FixedPoint::from_raw(self.penalty_score.load(Ordering::Acquire) as i64)
    }

    /// Check if venue is degraded
    #[inline]
    pub fn is_degraded(&self) -> bool {
        let rtt = self.rtt_ewma_ns.load(Ordering::Acquire);
        let threshold = self.degraded_threshold_ns.load(Ordering::Relaxed);
        rtt >= threshold
    }

    /// Get confidence score (based on variance)
    #[inline]
    pub fn get_confidence(&self) -> FixedPoint {
        let variance = self.variance_ns.load(Ordering::Acquire);
        let ewma = self.rtt_ewma_ns.load(Ordering::Acquire);
        
        if ewma == 0 {
            return FixedPoint::from_raw(SCALE);
        }

        // Coefficient of variation
        let cv = (variance as f64 / ewma as f64).min(1.0);
        let confidence = ((1.0 - cv) * SCALE as f64) as i64;
        FixedPoint::from_raw(confidence.max(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_model() {
        let model = VenueLatencyModel::new(10); // 10ms threshold

        // Initial state
        assert_eq!(model.get_rtt_ewma(), 1_000_000);
        assert!(!model.is_degraded());

        // Update with high latency
        model.update_rtt(15_000_000, FixedPoint::from_fractional(20_000_000));
        
        // Should be degraded now
        assert!(model.is_degraded());
        assert!(model.get_penalty_score().to_f64() < 1.0);
    }
}
