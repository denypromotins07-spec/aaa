//! LOD Fidelity Scanner
//! 
//! Detects when exchanges reduce update fidelity for illiquid instruments
//! to save compute resources. Uses Poisson-distributed probing to avoid rate limits.

use core::fmt;
use core::time::Duration;

/// Represents the fidelity level of an instrument's data stream
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FidelityLevel {
    /// Full real-time updates (high liquidity)
    High,
    /// Reduced update frequency (medium liquidity)
    Medium,
    /// Significantly delayed/stale updates (low liquidity)
    Low,
    /// Critical - data may be minutes stale
    Critical,
}

/// Measurement result for a single instrument
#[derive(Debug, Clone)]
pub struct FidelityMeasurement {
    /// Instrument identifier (e.g., option strike/expiration)
    pub instrument_id: u64,
    /// Detected fidelity level
    pub fidelity: FidelityLevel,
    /// Average update latency in microseconds
    pub avg_latency_us: f64,
    /// Update frequency (updates per second)
    pub update_frequency_hz: f64,
    /// Time since last update in microseconds
    pub staleness_us: u64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
}

/// Configuration for LOD scanning with anti-ban measures
#[derive(Debug, Clone, Copy)]
pub struct LodScannerConfig {
    /// Mean time between probes (microseconds) - Poisson distribution parameter
    pub probe_interval_mean_us: u64,
    /// Maximum probes per minute to avoid rate limiting
    pub max_probes_per_minute: u32,
    /// Minimum observations per instrument for statistical significance
    pub min_observations: usize,
    /// Latency threshold for "High" fidelity (microseconds)
    pub high_fidelity_threshold_us: u64,
    /// Latency threshold for "Medium" fidelity (microseconds)
    pub medium_fidelity_threshold_us: u64,
    /// Latency threshold for "Low" fidelity (microseconds)
    pub low_fidelity_threshold_us: u64,
}

impl Default for LodScannerConfig {
    fn default() -> Self {
        Self {
            probe_interval_mean_us: 50_000, // 50ms average - stays under radar
            max_probes_per_minute: 100,     // Conservative limit
            min_observations: 50,
            high_fidelity_threshold_us: 100,
            medium_fidelity_threshold_us: 1_000,
            low_fidelity_threshold_us: 10_000,
        }
    }
}

/// Poisson random number generator for probe timing
struct PoissonRng {
    seed: u64,
    lambda: f64,
}

impl PoissonRng {
    fn new(seed: u64, lambda: f64) -> Self {
        Self { seed, lambda }
    }

    /// Generate next Poisson-distributed value
    fn next(&mut self) -> u64 {
        // LCG-based Poisson approximation
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (self.seed as f64) / (u64::MAX as f64);
        
        // Inverse transform sampling for Poisson
        let l = (-self.lambda).exp();
        let mut k = 0u64;
        let mut p = 1.0;
        
        loop {
            k += 1;
            self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u_next = (self.seed as f64) / (u64::MAX as f64);
            p *= u_next;
            
            if p <= l {
                break;
            }
            
            // Safety limit to prevent infinite loops
            if k > 1000 {
                break;
            }
        }
        
        k.saturating_sub(1)
    }
}

/// LOD Fidelity Scanner with rate-limit avoidance
pub struct LodFidelityScanner {
    config: LodScannerConfig,
    rng: PoissonRng,
    measurements: std::collections::HashMap<u64, Vec<FidelityMeasurement>>,
    probe_count_this_minute: u32,
    minute_start_timestamp_us: u64,
}

impl LodFidelityScanner {
    pub fn new(config: LodScannerConfig) -> Self {
        Self {
            config,
            rng: PoissonRng::new(42, config.probe_interval_mean_us as f64),
            measurements: std::collections::HashMap::new(),
            probe_count_this_minute: 0,
            minute_start_timestamp_us: 0,
        }
    }

    /// Get next probe interval using Poisson distribution
    #[inline]
    pub fn next_probe_interval_us(&mut self) -> u64 {
        let poisson_value = self.rng.next();
        // Scale by mean interval
        (poisson_value as f64 * self.config.probe_interval_mean_us as f64 / self.config.probe_interval_mean_us as f64) as u64
            .max(1000) // Minimum 1ms to avoid flooding
    }

    /// Check if we can send another probe without hitting rate limits
    pub fn can_probe(&mut self, current_timestamp_us: u64) -> bool {
        // Reset counter if new minute
        if current_timestamp_us >= self.minute_start_timestamp_us + 60_000_000 {
            self.minute_start_timestamp_us = current_timestamp_us;
            self.probe_count_this_minute = 0;
        }

        self.probe_count_this_minute < self.config.max_probes_per_minute
    }

    /// Record a probe measurement
    pub fn record_measurement(
        &mut self,
        instrument_id: u64,
        latency_us: f64,
        timestamp_us: u64,
    ) -> Result<(), LodScanError> {
        if !self.can_probe(timestamp_us) {
            return Err(LodScanError::RateLimitApproached);
        }

        self.probe_count_this_minute += 1;

        let entry = self.measurements.entry(instrument_id).or_default();
        entry.push(FidelityMeasurement {
            instrument_id,
            fidelity: FidelityLevel::High, // Will be updated after analysis
            avg_latency_us: latency_us,
            update_frequency_hz: 0.0,
            staleness_us: 0,
            confidence: 0.0,
        });

        Ok(())
    }

    /// Analyze collected measurements to determine fidelity levels
    pub fn analyze_fidelity(&mut self) -> Result<Vec<FidelityMeasurement>, LodScanError> {
        let mut results = Vec::new();

        for (instrument_id, measurements) in &mut self.measurements {
            if measurements.len() < self.config.min_observations {
                continue;
            }

            // Calculate statistics
            let total_latency: f64 = measurements.iter().map(|m| m.avg_latency_us).sum();
            let avg_latency = total_latency / measurements.len() as f64;

            // Determine fidelity level based on latency thresholds
            let fidelity = if avg_latency < self.config.high_fidelity_threshold_us as f64 {
                FidelityLevel::High
            } else if avg_latency < self.config.medium_fidelity_threshold_us as f64 {
                FidelityLevel::Medium
            } else if avg_latency < self.config.low_fidelity_threshold_us as f64 {
                FidelityLevel::Low
            } else {
                FidelityLevel::Critical
            };

            // Calculate update frequency from measurement timestamps
            let update_freq = if measurements.len() > 1 {
                let first_time = measurements.first().map(|m| m.staleness_us).unwrap_or(0);
                let last_time = measurements.last().map(|m| m.staleness_us).unwrap_or(0);
                let duration_us = last_time.saturating_sub(first_time);
                if duration_us > 0 {
                    (measurements.len() as f64 - 1.0) / (duration_us as f64 / 1_000_000.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Calculate confidence based on sample size and variance
            let variance: f64 = measurements
                .iter()
                .map(|m| (m.avg_latency_us - avg_latency).powi(2))
                .sum::<f64>()
                / measurements.len() as f64;
            
            let std_dev = variance.sqrt();
            let cv = if avg_latency > 0.0 { std_dev / avg_latency } else { 1.0 };
            let confidence = (1.0 - cv).max(0.0).min(1.0) * 
                (measurements.len() as f64 / self.config.min_observations as f64).min(1.0);

            let result = FidelityMeasurement {
                instrument_id: *instrument_id,
                fidelity,
                avg_latency_us: avg_latency,
                update_frequency_hz: update_freq,
                staleness_us: measurements.last().map(|m| m.staleness_us).unwrap_or(0),
                confidence,
            };

            results.push(result);
        }

        Ok(results)
    }

    /// Find instruments with degraded fidelity (exploitation targets)
    pub fn find_degraded_instruments(&mut self) -> Result<Vec<FidelityMeasurement>, LodScanError> {
        let all_measurements = self.analyze_fidelity()?;
        
        Ok(all_measurements
            .into_iter()
            .filter(|m| matches!(m.fidelity, FidelityLevel::Low | FidelityLevel::Critical))
            .collect())
    }

    /// Clear all measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
        self.probe_count_this_minute = 0;
    }
}

/// Errors from LOD scanning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LodScanError {
    RateLimitApproached,
    InsufficientObservations,
    InvalidLatencyData,
}

impl fmt::Display for LodScanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LodScanError::RateLimitApproached => {
                write!(f, "Rate limit approached - reduce probe frequency")
            }
            LodScanError::InsufficientObservations => {
                write!(f, "Insufficient observations for analysis")
            }
            LodScanError::InvalidLatencyData => write!(f, "Invalid latency data"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_rng() {
        let mut rng = PoissonRng::new(42, 50000.0);
        let val1 = rng.next();
        let val2 = rng.next();
        // Values should be different (random)
        assert_ne!(val1, val2);
    }

    #[test]
    fn test_rate_limiting() {
        let config = LodScannerConfig {
            max_probes_per_minute: 5,
            ..Default::default()
        };
        let mut scanner = LodFidelityScanner::new(config);

        let ts = 1_000_000u64;
        for i in 0..5 {
            assert!(scanner.can_probe(ts + i * 1000));
        }
        assert!(!scanner.can_probe(ts + 5000));
    }
}
