//! Planck Tick Resolution Mapper
//! 
//! Deduces the exchange's internal clock-cycle resolution and queue-position algorithm
//! by sending micro-probes and measuring nanosecond round-trip latency.

use core::fmt;
use core::time::Duration;

/// Represents a single probe measurement
#[derive(Debug, Clone, Copy)]
pub struct ProbeMeasurement {
    /// Send timestamp in nanoseconds
    pub send_time_ns: u64,
    /// Receive timestamp in nanoseconds
    pub recv_time_ns: u64,
    /// Order size (1 lot for minimal impact)
    pub order_size: i64,
    /// Price level index
    pub price_level: i64,
}

impl ProbeMeasurement {
    #[inline]
    pub fn round_trip_ns(&self) -> Option<u64> {
        self.recv_time_ns.checked_sub(self.send_time_ns)
    }
}

/// Queue algorithm detection result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueAlgorithm {
    Fifo,
    ProRata,
    TimeProRata,
    MakerTakerPriority,
    Unknown,
}

/// Configuration for Planck tick resolution probing
#[derive(Debug, Clone, Copy)]
pub struct PlanckConfig {
    /// Minimum number of probes required for statistical significance
    pub min_probes: usize,
    /// Maximum probes before timeout
    pub max_probes: usize,
    /// Latency threshold for distinguishing clock cycles (nanoseconds)
    pub clock_cycle_threshold_ns: u64,
    /// Minimum time between probes to avoid rate limiting (microseconds)
    pub probe_interval_us: u64,
}

impl Default for PlanckConfig {
    fn default() -> Self {
        Self {
            min_probes: 100,
            max_probes: 10_000,
            clock_cycle_threshold_ns: 50, // 50ns typical FPGA granularity
            probe_interval_us: 100, // 100us between probes
        }
    }
}

/// Statistics about detected clock cycles
#[derive(Debug, Clone)]
pub struct ClockCycleStats {
    /// Detected fundamental clock cycle in nanoseconds
    pub cycle_ns: u64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Number of samples used
    pub sample_count: usize,
    /// Histogram of latency buckets
    pub latency_histogram: Vec<(u64, u32)>,
}

/// Planck Tick Resolution analyzer
pub struct PlanckTickResolver {
    config: PlanckConfig,
    measurements: Vec<ProbeMeasurement>,
    detected_algorithm: Option<QueueAlgorithm>,
    clock_stats: Option<ClockCycleStats>,
}

impl PlanckTickResolver {
    pub const fn new(config: PlanckConfig) -> Self {
        Self {
            config,
            measurements: Vec::new(),
            detected_algorithm: None,
            clock_stats: None,
        }
    }

    /// Add a probe measurement
    #[inline]
    pub fn add_measurement(&mut self, m: ProbeMeasurement) -> Result<(), PlanckError> {
        if self.measurements.len() >= self.config.max_probes {
            return Err(PlanckError::MaxProbesReached);
        }
        self.measurements.push(m);
        Ok(())
    }

    /// Analyze collected measurements to detect clock cycle
    pub fn analyze_clock_cycle(&mut self) -> Result<ClockCycleStats, PlanckError> {
        if self.measurements.len() < self.config.min_probes {
            return Err(PlanckError::InsufficientProbes);
        }

        // Calculate round-trip times
        let mut rtts: Vec<u64> = Vec::with_capacity(self.measurements.len());
        for &m in &self.measurements {
            if let Some(rtt) = m.round_trip_ns() {
                rtts.push(rtt);
            }
        }

        if rtts.len() < self.config.min_probes {
            return Err(PlanckError::InsufficientValidMeasurements);
        }

        // Build histogram with bucket size = clock_cycle_threshold_ns
        let bucket_size = self.config.clock_cycle_threshold_ns;
        let mut histogram: Vec<(u64, u32)> = Vec::new();
        
        // Find min and max RTT
        let min_rtt = rtts.iter().min().copied().unwrap_or(0);
        let max_rtt = rtts.iter().max().copied().unwrap_or(0);
        
        if min_rtt == 0 || max_rtt == 0 {
            return Err(PlanckError::InvalidLatencyData);
        }

        // Create buckets aligned to potential clock boundaries
        let num_buckets = ((max_rtt - min_rtt) / bucket_size + 1) as usize;
        let mut bucket_counts = vec![0u32; num_buckets];
        
        for &rtt in &rtts {
            let bucket_idx = ((rtt - min_rtt) / bucket_size) as usize;
            if bucket_idx < bucket_counts.len() {
                bucket_counts[bucket_idx] += 1;
            }
        }

        // Find peaks in histogram (potential clock cycle multiples)
        let mut peaks: Vec<u64> = Vec::new();
        for (i, &count) in bucket_counts.iter().enumerate() {
            if count > 0 {
                let bucket_center = min_rtt + (i as u64 * bucket_size);
                peaks.push(bucket_center);
                histogram.push((bucket_center, count));
            }
        }

        // Detect fundamental cycle using GCD of peak differences
        if peaks.len() < 2 {
            return Err(PlanckError::NoClearPattern);
        }

        let mut cycle_candidates: Vec<u64> = Vec::new();
        for i in 1..peaks.len() {
            let diff = peaks[i].saturating_sub(peaks[i - 1]);
            if diff > 0 {
                cycle_candidates.push(diff);
            }
        }

        // Use GCD to find fundamental cycle
        let detected_cycle = if let Some(first) = cycle_candidates.first() {
            let mut gcd_result = *first;
            for &candidate in cycle_candidates.iter().skip(1) {
                gcd_result = Self::gcd(gcd_result, candidate);
                if gcd_result == 1 {
                    break;
                }
            }
            gcd_result
        } else {
            bucket_size
        };

        // Calculate confidence based on how well peaks align with detected cycle
        let mut aligned_count = 0usize;
        for &peak in &peaks {
            let remainder = peak % detected_cycle;
            if remainder < bucket_size / 4 || remainder > (3 * bucket_size / 4) {
                aligned_count += 1;
            }
        }
        let confidence = aligned_count as f64 / peaks.len() as f64;

        let stats = ClockCycleStats {
            cycle_ns: detected_cycle,
            confidence,
            sample_count: rtts.len(),
            latency_histogram: histogram,
        };

        self.clock_stats = Some(stats.clone());
        Ok(stats)
    }

    /// Detect queue algorithm based on fill patterns
    pub fn detect_queue_algorithm(&mut self) -> Result<QueueAlgorithm, PlanckError> {
        if self.measurements.len() < self.config.min_probes {
            return Err(PlanckError::InsufficientProbes);
        }

        // Analyze fill timing patterns to infer queue algorithm
        // This is a simplified heuristic; production would use more sophisticated analysis
        
        let mut fifo_score = 0.0;
        let mut pro_rata_score = 0.0;
        
        // Group measurements by price level
        let mut by_price: std::collections::BTreeMap<i64, Vec<&ProbeMeasurement>> = 
            std::collections::BTreeMap::new();
        
        for m in &self.measurements {
            by_price.entry(m.price_level).or_default().push(m);
        }

        for (_, measurements) in by_price.iter() {
            if measurements.len() < 2 {
                continue;
            }

            // Check if fills are strictly ordered by time (FIFO characteristic)
            let mut time_ordered = true;
            for i in 1..measurements.len() {
                if measurements[i].send_time_ns < measurements[i - 1].send_time_ns {
                    time_ordered = false;
                    break;
                }
            }

            if time_ordered {
                fifo_score += 1.0;
            } else {
                // Check for proportional allocation (Pro-Rata characteristic)
                let total_size: i64 = measurements.iter().map(|m| m.order_size).sum();
                if total_size > 0 {
                    let mut proportional = true;
                    let expected_ratio = measurements[0].order_size as f64 / total_size as f64;
                    
                    for m in measurements.iter().skip(1) {
                        let actual_ratio = m.order_size as f64 / total_size as f64;
                        if (actual_ratio - expected_ratio).abs() > 0.1 {
                            proportional = false;
                            break;
                        }
                    }
                    
                    if proportional {
                        pro_rata_score += 1.0;
                    }
                }
            }
        }

        let algorithm = if fifo_score > pro_rata_score * 1.5 {
            QueueAlgorithm::Fifo
        } else if pro_rata_score > fifo_score * 1.5 {
            QueueAlgorithm::ProRata
        } else if fifo_score > 0.0 && pro_rata_score > 0.0 {
            QueueAlgorithm::TimeProRata
        } else {
            QueueAlgorithm::Unknown
        };

        self.detected_algorithm = Some(algorithm);
        Ok(algorithm)
    }

    /// Get detected algorithm
    pub fn get_detected_algorithm(&self) -> Option<QueueAlgorithm> {
        self.detected_algorithm
    }

    /// Get clock stats
    pub fn get_clock_stats(&self) -> Option<&ClockCycleStats> {
        self.clock_stats.as_ref()
    }

    /// Clear all measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
        self.detected_algorithm = None;
        self.clock_stats = None;
    }

    /// Greatest Common Divisor helper
    const fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }
}

/// Errors from Planck tick resolution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanckError {
    InsufficientProbes,
    MaxProbesReached,
    InsufficientValidMeasurements,
    InvalidLatencyData,
    NoClearPattern,
}

impl fmt::Display for PlanckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanckError::InsufficientProbes => write!(f, "Insufficient probe measurements"),
            PlanckError::MaxProbesReached => write!(f, "Maximum probe count reached"),
            PlanckError::InsufficientValidMeasurements => {
                write!(f, "Insufficient valid measurements after filtering")
            }
            PlanckError::InvalidLatencyData => write!(f, "Invalid latency data detected"),
            PlanckError::NoClearPattern => write!(f, "No clear clock cycle pattern detected"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_measurement() {
        let m = ProbeMeasurement {
            send_time_ns: 1000,
            recv_time_ns: 1500,
            order_size: 1,
            price_level: 0,
        };
        assert_eq!(m.round_trip_ns(), Some(500));
    }

    #[test]
    fn test_gcd() {
        assert_eq!(PlanckTickResolver::gcd(48, 18), 6);
        assert_eq!(PlanckTickResolver::gcd(100, 25), 25);
    }
}
