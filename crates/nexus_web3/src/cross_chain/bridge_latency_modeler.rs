//! Bridge Latency Modeler
//! Stochastic model for cross-chain bridge confirmation times

use thiserror::Error;
use alloc::vec::Vec;

#[derive(Error, Debug)]
pub enum BridgeModelError {
    #[error("Insufficient data points")]
    InsufficientData,
    #[error("Invalid latency value")]
    InvalidLatency,
}

pub type Result<T> = core::result::Result<T, BridgeModelError>;

/// Bridge type enumeration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BridgeType {
    Wormhole,
    LayerZero,
    Axelar,
    Synapse,
    Multichain,
    Native,
}

/// Latency distribution statistics
#[derive(Clone, Debug)]
pub struct LatencyStats {
    /// Mean latency in milliseconds
    pub mean_ms: f64,
    /// Standard deviation in milliseconds
    pub std_dev_ms: f64,
    /// P50 (median) latency
    pub p50_ms: f64,
    /// P95 latency
    pub p95_ms: f64,
    /// P99 latency
    pub p99_ms: f64,
    /// Minimum observed latency
    pub min_ms: f64,
    /// Maximum observed latency
    pub max_ms: f64,
    /// Sample count
    pub sample_count: usize,
}

/// Bridge latency observation
#[derive(Clone, Debug)]
pub struct LatencyObservation {
    pub timestamp_ms: u64,
    pub latency_ms: u64,
    pub success: bool,
    pub bridge_type: BridgeType,
    pub source_chain: u8,
    pub dest_chain: u8,
}

/// Stochastic model for bridge latency prediction
pub struct BridgeLatencyModeler {
    /// Historical observations per bridge route
    observations: Vec<LatencyObservation>,
    /// Maximum observations to keep (ring buffer behavior)
    max_observations: usize,
    /// Current route statistics
    stats: Option<LatencyStats>,
}

impl BridgeLatencyModeler {
    /// Create a new modeler with default settings
    pub fn new() -> Self {
        Self {
            observations: Vec::with_capacity(1000),
            max_observations: 10000,
            stats: None,
        }
    }

    /// Create with custom max observations
    pub fn with_capacity(max_obs: usize) -> Self {
        Self {
            observations: Vec::with_capacity(max_obs.min(1000)),
            max_observations: max_obs,
            stats: None,
        }
    }

    /// Record a new latency observation
    pub fn record_observation(&mut self, obs: LatencyObservation) {
        if obs.latency_ms == 0 {
            return; // Skip invalid observations
        }

        // Ring buffer behavior
        if self.observations.len() >= self.max_observations {
            self.observations.remove(0);
        }
        
        self.observations.push(obs);
        
        // Update statistics periodically
        if self.observations.len() % 10 == 0 || self.stats.is_none() {
            self.update_stats();
        }
    }

    /// Record batch of observations
    pub fn record_batch(&mut self, observations: &[LatencyObservation]) {
        for obs in observations {
            self.record_observation(obs.clone());
        }
    }

    /// Get current statistics
    pub const fn stats(&self) -> Option<&LatencyStats> {
        self.stats.as_ref()
    }

    /// Predict latency for given confidence level
    /// 
    /// # Arguments
    /// * `confidence` - Confidence level (0.0 to 1.0), e.g., 0.95 for 95% confidence
    /// 
    /// # Returns
    /// Predicted latency in milliseconds that should be exceeded only (1-confidence)% of the time
    pub fn predict_latency(&self, confidence: f64) -> Result<f64> {
        let stats = self.stats.as_ref()
            .ok_or(BridgeModelError::InsufficientData)?;
        
        if stats.sample_count < 10 {
            return Err(BridgeModelError::InsufficientData);
        }

        // Use percentile based on confidence
        let predicted = match confidence {
            c if c <= 0.5 => stats.p50_ms,
            c if c <= 0.95 => stats.p95_ms,
            _ => stats.p99_ms,
        };

        Ok(predicted)
    }

    /// Calculate probability that bridge will complete within timeout
    /// 
    /// # Arguments
    /// * `timeout_ms` - Timeout in milliseconds
    /// 
    /// # Returns
    /// Probability (0.0 to 1.0) that bridge confirms within timeout
    pub fn probability_within_timeout(&self, timeout_ms: u64) -> f64 {
        let stats = match &self.stats {
            Some(s) => s,
            None => return 0.0,
        };

        if stats.sample_count == 0 {
            return 0.0;
        }

        // Count observations within timeout
        let within_count = self.observations.iter()
            .filter(|o| o.success && o.latency_ms <= timeout_ms)
            .count() as f64;

        within_count / stats.sample_count as f64
    }

    /// Calculate expected slippage due to bridge delay
    /// 
    /// # Arguments
    /// * `volatility_per_ms` - Asset volatility per millisecond (as decimal)
    /// * `timeout_ms` - Expected bridge time
    /// 
    /// # Returns
    /// Expected price movement (as decimal) during bridge time
    pub fn expected_slippage(&self, volatility_per_ms: f64, timeout_ms: u64) -> f64 {
        let predicted = self.predict_latency(0.95).unwrap_or(0.0);
        let total_time = (predicted as u64).max(timeout_ms);
        
        // Simple model: slippage scales with sqrt(time) * volatility
        (total_time as f64).sqrt() * volatility_per_ms
    }

    /// Penalize arbitrage signal based on bridge risk
    /// 
    /// # Arguments
    /// * `base_profit_bps` - Expected profit in basis points without bridge risk
    /// * `timeout_ms` - Bridge timeout
    /// 
    /// # Returns
    /// Risk-adjusted profit in basis points (may be negative)
    pub fn risk_adjusted_profit(&self, base_profit_bps: i32, timeout_ms: u64) -> i32 {
        let success_prob = self.probability_within_timeout(timeout_ms);
        let failure_penalty = base_profit_bps.abs() * 2; // Penalty for failed arb
        
        let adjusted = (base_profit_bps as f64 * success_prob) 
            - (failure_penalty as f64 * (1.0 - success_prob));
        
        adjusted as i32
    }

    /// Check if bridge is currently healthy
    pub fn is_healthy(&self) -> bool {
        let stats = match &self.stats {
            Some(s) => s,
            None => return false,
        };

        // Consider unhealthy if:
        // - Not enough samples
        // - P95 latency > 5 minutes
        // - Success rate < 90%
        if stats.sample_count < 30 {
            return false;
        }

        if stats.p95_ms > 300_000.0 { // 5 minutes
            return false;
        }

        let recent_success_rate = self.observations.iter()
            .rev()
            .take(100)
            .filter(|o| o.success)
            .count() as f64 / 100.0;

        recent_success_rate >= 0.9
    }

    /// Get recommended timeout for given confidence
    pub fn recommended_timeout(&self, confidence: f64) -> Result<u64> {
        let predicted = self.predict_latency(confidence)?;
        // Add 20% buffer for safety
        Ok((predicted * 1.2) as u64)
    }

    /// Update internal statistics from observations
    fn update_stats(&mut self) {
        if self.observations.is_empty() {
            self.stats = None;
            return;
        }

        let latencies: Vec<f64> = self.observations.iter()
            .filter(|o| o.success)
            .map(|o| o.latency_ms as f64)
            .collect();

        if latencies.is_empty() {
            self.stats = None;
            return;
        }

        let n = latencies.len();
        let sum: f64 = latencies.iter().sum();
        let mean = sum / n as f64;

        // Calculate standard deviation
        let variance: f64 = latencies.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        // Sort for percentiles
        let mut sorted = latencies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p50_idx = (n as f64 * 0.50) as usize;
        let p95_idx = (n as f64 * 0.95) as usize;
        let p99_idx = (n as f64 * 0.99) as usize;

        self.stats = Some(LatencyStats {
            mean_ms: mean,
            std_dev_ms: std_dev,
            p50_ms: *sorted.get(p50_idx).unwrap_or(&mean),
            p95_ms: *sorted.get(p95_idx.min(n - 1)).unwrap_or(&mean),
            p99_ms: *sorted.get(p99_idx.min(n - 1)).unwrap_or(&mean),
            min_ms: *sorted.first().unwrap_or(&mean),
            max_ms: *sorted.last().unwrap_or(&mean),
            sample_count: n,
        });
    }
}

impl Default for BridgeLatencyModeler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modeler_creation() {
        let modeler = BridgeLatencyModeler::new();
        assert!(modeler.stats().is_none());
    }

    #[test]
    fn test_record_observations() {
        let mut modeler = BridgeLatencyModeler::new();
        
        for i in 0..50 {
            modeler.record_observation(LatencyObservation {
                timestamp_ms: i * 1000,
                latency_ms: 1000 + (i % 100) as u64,
                success: true,
                bridge_type: BridgeType::Wormhole,
                source_chain: 1,
                dest_chain: 2,
            });
        }

        assert!(modeler.stats().is_some());
        let stats = modeler.stats().unwrap();
        assert!(stats.mean_ms > 0.0);
    }

    #[test]
    fn test_prediction() {
        let mut modeler = BridgeLatencyModeler::new();
        
        // Add observations with known distribution
        for i in 0..100 {
            modeler.record_observation(LatencyObservation {
                timestamp_ms: i * 1000,
                latency_ms: 1000 + i as u64,
                success: true,
                bridge_type: BridgeType::LayerZero,
                source_chain: 1,
                dest_chain: 2,
            });
        }

        let p50 = modeler.predict_latency(0.50).unwrap();
        let p95 = modeler.predict_latency(0.95).unwrap();
        
        assert!(p95 > p50, "P95 should be greater than P50");
    }

    #[test]
    fn test_insufficient_data() {
        let modeler = BridgeLatencyModeler::new();
        let result = modeler.predict_latency(0.95);
        assert!(matches!(result, Err(BridgeModelError::InsufficientData)));
    }

    #[test]
    fn test_health_check() {
        let mut modeler = BridgeLatencyModeler::new();
        
        // Not healthy with no data
        assert!(!modeler.is_healthy());
        
        // Add some data
        for i in 0..50 {
            modeler.record_observation(LatencyObservation {
                timestamp_ms: i * 1000,
                latency_ms: 2000,
                success: true,
                bridge_type: BridgeType::Wormhole,
                source_chain: 1,
                dest_chain: 2,
            });
        }
        
        assert!(modeler.is_healthy());
    }
}
