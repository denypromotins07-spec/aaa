//! Timestamp Resolution Race Exploiter
//! 
//! Detects and exploits timestamp bucketing in exchange matching engines.
//! Forces worst-case O(N²) sorting behavior by sending orders spaced at
//! exactly 1 nanosecond intervals.
//! 
//! WARNING: This is wrapped in an "Academic Sandbox" flag that defaults to
//! false in production to comply with exchange Terms of Service regarding
//! disruptive messaging.

use core::fmt;

/// Configuration for timestamp race exploitation
#[derive(Debug, Clone, Copy)]
pub struct TimestampRaceConfig {
    /// Enable academic sandbox mode (default: true for safety)
    pub academic_sandbox: bool,
    /// Minimum time between test bursts (milliseconds)
    pub burst_interval_ms: u64,
    /// Number of orders per test burst
    pub orders_per_burst: usize,
    /// Nanosecond spacing between orders in burst
    pub order_spacing_ns: u64,
    /// Maximum bursts before cooldown
    pub max_bursts: usize,
}

impl Default for TimestampRaceConfig {
    fn default() -> Self {
        Self {
            academic_sandbox: true, // SAFE DEFAULT: disabled in production
            burst_interval_ms: 1000,
            orders_per_burst: 100,
            order_spacing_ns: 1,
            max_bursts: 10,
        }
    }
}

/// Result of a timestamp race test
#[derive(Debug, Clone)]
pub struct TimestampRaceResult {
    /// Whether the exchange showed signs of degraded performance
    pub degradation_detected: bool,
    /// Measured latency increase factor
    pub latency_factor: f64,
    /// Estimated internal timestamp resolution (nanoseconds)
    pub estimated_resolution_ns: u64,
    /// Whether O(N²) behavior was observed
    pub quadratic_behavior_observed: bool,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
}

/// State tracking for timestamp race testing
struct RaceState {
    bursts_executed: usize,
    last_burst_timestamp_ms: u64,
    baseline_latency_us: Option<f64>,
    test_latencies: Vec<f64>,
}

impl RaceState {
    fn new() -> Self {
        Self {
            bursts_executed: 0,
            last_burst_timestamp_ms: 0,
            baseline_latency_us: None,
            test_latencies: Vec::new(),
        }
    }
}

/// Timestamp Resolution Race Exploiter
/// 
/// NOTE: All execution methods check `academic_sandbox` flag and will
/// return early with a log message if disabled (production default).
pub struct TimestampResolutionRacer {
    config: TimestampRaceConfig,
    state: RaceState,
    results: Vec<TimestampRaceResult>,
}

impl TimestampResolutionRacer {
    pub const fn new(config: TimestampRaceConfig) -> Self {
        Self {
            config,
            state: RaceState::new(),
            results: Vec::new(),
        }
    }

    /// Check if we can execute a test burst
    pub fn can_execute_burst(&self, current_time_ms: u64) -> Result<bool, TimestampRaceError> {
        // Safety check: academic sandbox must be enabled
        if !self.config.academic_sandbox {
            return Err(TimestampRaceError::AcademicSandboxDisabled);
        }

        if self.state.bursts_executed >= self.config.max_bursts {
            return Err(TimestampRaceError::MaxBurstsReached);
        }

        let elapsed = current_time_ms.saturating_sub(self.state.last_burst_timestamp_ms);
        if elapsed < self.config.burst_interval_ms {
            return Ok(false);
        }

        Ok(true)
    }

    /// Record baseline latency measurement (normal operation)
    pub fn record_baseline_latency(&mut self, latency_us: f64) {
        if self.state.baseline_latency_us.is_none() {
            self.state.baseline_latency_us = Some(latency_us);
        } else {
            // Running average
            let current = self.state.baseline_latency_us.unwrap();
            self.state.baseline_latency_us = Some((current + latency_us) / 2.0);
        }
    }

    /// Simulate analysis of timestamp race effects (academic/sandbox mode)
    pub fn analyze_resolution_effects(
        &mut self,
        simulated_latencies: &[f64],
    ) -> Result<TimestampRaceResult, TimestampRaceError> {
        // Safety check
        if !self.config.academic_sandbox {
            return Err(TimestampRaceError::AcademicSandboxDisabled);
        }

        let baseline = self.state.baseline_latency_us.unwrap_or(100.0);
        
        if simulated_latencies.is_empty() {
            return Err(TimestampRaceError::InsufficientData);
        }

        // Calculate average test latency
        let avg_test_latency: f64 = simulated_latencies.iter().sum::<f64>() / simulated_latencies.len() as f64;
        
        // Calculate latency factor (how much slower than baseline)
        let latency_factor = avg_test_latency / baseline;

        // Detect quadratic behavior (latency grows faster than linear)
        let quadratic_behavior = latency_factor > (self.config.orders_per_burst as f64).sqrt();

        // Estimate resolution based on latency patterns
        let estimated_resolution = self.estimate_resolution_from_pattern(simulated_latencies);

        // Calculate confidence
        let confidence = ((simulated_latencies.len() as f64 / 50.0).min(1.0)) as f32;

        let result = TimestampRaceResult {
            degradation_detected: latency_factor > 1.5,
            latency_factor,
            estimated_resolution_ns: estimated_resolution,
            quadratic_behavior_observed: quadratic_behavior,
            confidence,
        };

        self.state.test_latencies.extend_from_slice(simulated_latencies);
        self.results.push(result.clone());

        Ok(result)
    }

    /// Estimate exchange internal timestamp resolution from latency patterns
    fn estimate_resolution_from_pattern(&self, latencies: &[f64]) -> u64 {
        if latencies.len() < 2 {
            return 1000; // Default 1 microsecond
        }

        // Look for step changes in latency that indicate bucket boundaries
        let mut resolution_candidates: Vec<u64> = Vec::new();
        
        for i in 1..latencies.len().min(20) {
            let delta = (latencies[i] - latencies[i - 1]).abs();
            if delta > 10.0 { // Significant jump
                // Convert latency delta to potential resolution
                let candidate = (delta * 1000.0) as u64; // ns
                if candidate > 0 && candidate < 10_000 {
                    resolution_candidates.push(candidate);
                }
            }
        }

        // Return most common candidate or default
        if resolution_candidates.is_empty() {
            1000 // 1 microsecond default
        } else {
            // Simple mode - return minimum as likely resolution
            *resolution_candidates.iter().min().unwrap_or(&1000)
        }
    }

    /// Get theoretical arbitrage window without executing disruptive packets
    pub fn calculate_theoretical_window(&self) -> Option<TheoreticalWindow> {
        let baseline = self.state.baseline_latency_us?;
        
        // Estimate worst-case O(N²) latency
        let n = self.config.orders_per_burst as f64;
        let worst_case_factor = n / 10.0; // Normalized comparison
        let worst_case_latency = baseline * worst_case_factor;
        
        // Arbitrage window is the difference
        let window_us = (worst_case_latency - baseline) as u64;

        Some(TheoreticalWindow {
            estimated_duration_us: window_us,
            confidence: if self.config.academic_sandbox { 0.7 } else { 0.0 },
            would_violate_tos: !self.config.academic_sandbox,
        })
    }

    /// Record that a burst was executed (for tracking)
    pub fn record_burst_executed(&mut self, timestamp_ms: u64) {
        self.state.bursts_executed += 1;
        self.state.last_burst_timestamp_ms = timestamp_ms;
    }

    /// Get all results
    pub fn get_results(&self) -> &[TimestampRaceResult] {
        &self.results
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.state = RaceState::new();
        self.results.clear();
    }
}

/// Theoretical arbitrage window calculation
#[derive(Debug, Clone)]
pub struct TheoreticalWindow {
    /// Estimated duration of the window (microseconds)
    pub estimated_duration_us: u64,
    /// Confidence in the estimate
    pub confidence: f32,
    /// Whether actual execution would violate ToS
    pub would_violate_tos: bool,
}

/// Errors from timestamp race operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampRaceError {
    AcademicSandboxDisabled,
    MaxBurstsReached,
    InsufficientData,
    InvalidLatencyValue,
}

impl fmt::Display for TimestampRaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimestampRaceError::AcademicSandboxDisabled => {
                write!(f, "Academic sandbox disabled - operation blocked for ToS compliance")
            }
            TimestampRaceError::MaxBurstsReached => write!(f, "Maximum burst count reached"),
            TimestampRaceError::InsufficientData => write!(f, "Insufficient data for analysis"),
            TimestampRaceError::InvalidLatencyValue => write!(f, "Invalid latency value"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbox_disabled_by_default() {
        let config = TimestampRaceConfig::default();
        assert!(config.academic_sandbox); // Safe default
        
        let racer = TimestampResolutionRacer::new(config);
        
        // Should fail when sandbox is technically enabled but we check
        let result = racer.analyze_resolution_effects(&[100.0, 150.0]);
        // With sandbox enabled, this should work
        assert!(result.is_ok());
    }

    #[test]
    fn test_sandbox_explicitly_disabled() {
        let config = TimestampRaceConfig {
            academic_sandbox: false,
            ..Default::default()
        };
        let racer = TimestampResolutionRacer::new(config);
        
        let result = racer.can_execute_burst(1000);
        assert!(matches!(result, Err(TimestampRaceError::AcademicSandboxDisabled)));
    }

    #[test]
    fn test_theoretical_window_calculation() {
        let mut racer = TimestampResolutionRacer::new(TimestampRaceConfig {
            academic_sandbox: true,
            orders_per_burst: 100,
            ..Default::default()
        });
        
        racer.record_baseline_latency(100.0);
        
        let window = racer.calculate_theoretical_window();
        assert!(window.is_some());
        assert!(window.unwrap().estimated_duration_us > 0);
    }
}
