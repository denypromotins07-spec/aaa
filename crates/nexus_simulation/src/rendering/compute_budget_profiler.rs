//! Compute Budget Profiler
//! 
//! Measures latency variance across the option chain to detect when the exchange's
//! matching engine has deprioritized specific strikes (LOD dropping).

use core::fmt;

/// Profile result for a single instrument
#[derive(Debug, Clone)]
pub struct InstrumentProfile {
    /// Instrument identifier
    pub instrument_id: u64,
    /// Strike price (for options)
    pub strike: f64,
    /// Expiration timestamp
    pub expiration_ts: u64,
    /// Average response latency (microseconds)
    pub avg_latency_us: f64,
    /// Latency standard deviation (microseconds)
    pub latency_std_us: f64,
    /// 99th percentile latency (microseconds)
    pub p99_latency_us: f64,
    /// Number of samples collected
    pub sample_count: usize,
    /// Is this instrument likely deprioritized?
    pub is_deprioritized: bool,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
}

/// Configuration for compute budget profiling
#[derive(Debug, Clone, Copy)]
pub struct ComputeBudgetConfig {
    /// Minimum samples required per instrument
    pub min_samples: usize,
    /// Latency threshold above which an instrument is considered deprioritized
    pub deprioritize_latency_threshold_us: f64,
    /// Coefficient of variation threshold for detecting inconsistent updates
    pub cv_threshold: f64,
    /// Maximum instruments to profile simultaneously (memory limit)
    pub max_instruments: usize,
}

impl Default for ComputeBudgetConfig {
    fn default() -> Self {
        Self {
            min_samples: 100,
            deprioritize_latency_threshold_us: 5_000.0, // 5ms
            cv_threshold: 0.5,
            max_instruments: 10_000,
        }
    }
}

/// Running statistics calculator using Welford's algorithm
struct RunningStats {
    count: usize,
    mean: f64,
    m2: f64,
    min_val: f64,
    max_val: f64,
    values_for_percentile: Vec<f64>,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min_val: f64::MAX,
            max_val: f64::MIN,
            values_for_percentile: Vec::new(),
        }
    }

    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        
        self.min_val = self.min_val.min(value);
        self.max_val = self.max_val.max(value);
        
        // Store values for percentile calculation (with limit)
        if self.values_for_percentile.len() < 1000 {
            self.values_for_percentile.push(value);
        }
    }

    fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn percentile(&mut self, p: f64) -> f64 {
        if self.values_for_percentile.is_empty() {
            return self.mean;
        }
        
        self.values_for_percentile.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        
        let idx = ((p / 100.0) * (self.values_for_percentile.len() - 1) as f64) as usize;
        self.values_for_percentile.get(idx).copied().unwrap_or(self.mean)
    }
}

/// Compute Budget Profiler for detecting exchange resource allocation
pub struct ComputeBudgetProfiler {
    config: ComputeBudgetConfig,
    stats: std::collections::HashMap<u64, RunningStats>,
    instrument_metadata: std::collections::HashMap<u64, (f64, u64)>, // (strike, expiration)
}

impl ComputeBudgetProfiler {
    pub fn new(config: ComputeBudgetConfig) -> Self {
        Self {
            config,
            stats: std::collections::HashMap::new(),
            instrument_metadata: std::collections::HashMap::new(),
        }
    }

    /// Record a latency measurement for an instrument
    pub fn record_latency(
        &mut self,
        instrument_id: u64,
        strike: f64,
        expiration_ts: u64,
        latency_us: f64,
    ) -> Result<(), ComputeBudgetError> {
        if self.stats.len() >= self.config.max_instruments && !self.stats.contains_key(&instrument_id) {
            return Err(ComputeBudgetError::MaxInstrumentsReached);
        }

        // Store metadata
        self.instrument_metadata
            .entry(instrument_id)
            .or_insert((strike, expiration_ts));

        // Update running statistics
        self.stats
            .entry(instrument_id)
            .or_insert_with(RunningStats::new)
            .update(latency_us);

        Ok(())
    }

    /// Analyze all instruments and identify deprioritized ones
    pub fn analyze(&mut self) -> Result<Vec<InstrumentProfile>, ComputeBudgetError> {
        let mut profiles = Vec::new();

        for (&instrument_id, stats) in &mut self.stats {
            if stats.count < self.config.min_samples {
                continue;
            }

            let (strike, expiration_ts) = self.instrument_metadata
                .get(&instrument_id)
                .copied()
                .unwrap_or((0.0, 0));

            let avg_latency = stats.mean;
            let std_dev = stats.std_dev();
            let p99 = stats.percentile(99.0);
            let cv = if avg_latency > 0.0 { std_dev / avg_latency } else { 0.0 };

            // Determine if instrument is deprioritized
            let is_deprioritized = avg_latency > self.config.deprioritize_latency_threshold_us
                || cv > self.config.cv_threshold;

            // Calculate confidence based on sample size and consistency
            let sample_confidence = (stats.count as f64 / self.config.min_samples as f64).min(1.0);
            let consistency_confidence = if is_deprioritized {
                if cv > self.config.cv_threshold {
                    0.8
                } else {
                    1.0
                }
            } else {
                1.0 - cv
            };
            let confidence = sample_confidence * consistency_confidence;

            profiles.push(InstrumentProfile {
                instrument_id,
                strike,
                expiration_ts,
                avg_latency_us: avg_latency,
                latency_std_us: std_dev,
                p99_latency_us: p99,
                sample_count: stats.count,
                is_deprioritized,
                confidence,
            });
        }

        // Sort by deprioritization status and confidence
        profiles.sort_by(|a, b| {
            b.is_deprioritized
                .cmp(&a.is_deprioritized)
                .then_with(|| b.confidence.partial_cmp(&a.confidence).unwrap_or(core::cmp::Ordering::Equal))
        });

        Ok(profiles)
    }

    /// Get only deprioritized instruments (exploitation targets)
    pub fn get_deprioritized(&mut self) -> Result<Vec<InstrumentProfile>, ComputeBudgetError> {
        let all_profiles = self.analyze()?;
        Ok(all_profiles.into_iter().filter(|p| p.is_deprioritized).collect())
    }

    /// Clear all collected data
    pub fn clear(&mut self) {
        self.stats.clear();
        self.instrument_metadata.clear();
    }

    /// Get statistics for a specific instrument
    pub fn get_instrument_stats(&self, instrument_id: u64) -> Option<&RunningStats> {
        self.stats.get(&instrument_id)
    }
}

/// Errors from compute budget profiling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBudgetError {
    MaxInstrumentsReached,
    InvalidLatencyValue,
    InsufficientData,
}

impl fmt::Display for ComputeBudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeBudgetError::MaxInstrumentsReached => {
                write!(f, "Maximum instrument count reached")
            }
            ComputeBudgetError::InvalidLatencyValue => write!(f, "Invalid latency value"),
            ComputeBudgetError::InsufficientData => write!(f, "Insufficient data for analysis"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_running_stats() {
        let mut stats = RunningStats::new();
        
        for i in 1..=10 {
            stats.update(i as f64);
        }
        
        assert_eq!(stats.count, 10);
        assert!((stats.mean - 5.5).abs() < 0.001);
        assert!(stats.std_dev() > 0.0);
    }

    #[test]
    fn test_deprioritization_detection() {
        let config = ComputeBudgetConfig::default();
        let mut profiler = ComputeBudgetProfiler::new(config);

        // Record high-latency measurements for one instrument
        for _ in 0..100 {
            profiler.record_latency(1, 100.0, 1000000, 10_000.0).unwrap();
        }

        // Record low-latency measurements for another
        for _ in 0..100 {
            profiler.record_latency(2, 100.0, 1000000, 100.0).unwrap();
        }

        let deprioritized = profiler.get_deprioritized().unwrap();
        
        assert_eq!(deprioritized.len(), 1);
        assert_eq!(deprioritized[0].instrument_id, 1);
    }
}
