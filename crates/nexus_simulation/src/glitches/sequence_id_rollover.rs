//! Sequence ID Rollover Detector
//! 
//! Detects when exchange sequence IDs approach MAX_INT and may experience
//! integer overflow wrap-arounds causing microsecond lock-contention.

use core::fmt;

/// Represents a detected rollover event
#[derive(Debug, Clone)]
pub struct RolloverEvent {
    /// Sequence ID where rollover occurred
    pub rollover_point: u64,
    /// Bit width of the counter (32 or 64)
    pub counter_bits: u8,
    /// Timestamp of detection (nanoseconds)
    pub detection_timestamp_ns: u64,
    /// Estimated duration of lock contention (microseconds)
    pub estimated_contention_us: u64,
    /// Severity level (1-10)
    pub severity: u8,
}

/// Configuration for rollover detection
#[derive(Debug, Clone, Copy)]
pub struct RolloverConfig {
    /// Warning threshold before rollover (number of IDs)
    pub warning_threshold: u64,
    /// Maximum sequence ID for 32-bit counters
    pub max_32bit: u64,
    /// Maximum sequence ID for 64-bit counters
    pub max_64bit: u64,
}

impl Default for RolloverConfig {
    fn default() -> Self {
        Self {
            warning_threshold: 1000,
            max_32bit: u32::MAX as u64,
            max_64bit: u64::MAX,
        }
    }
}

/// State tracking for sequence ID monitoring
struct RolloverState {
    last_seen_id: Option<u64>,
    rollover_count: usize,
    ids_since_last_rollover: u64,
    detected_bit_width: Option<u8>,
}

impl RolloverState {
    fn new() -> Self {
        Self {
            last_seen_id: None,
            rollover_count: 0,
            ids_since_last_rollover: 0,
            detected_bit_width: None,
        }
    }
}

/// Sequence ID Rollover Detector
pub struct SequenceIdRolloverDetector {
    config: RolloverConfig,
    state: RolloverState,
    events: Vec<RolloverEvent>,
}

impl SequenceIdRolloverDetector {
    pub const fn new(config: RolloverConfig) -> Self {
        Self {
            config,
            state: RolloverState::new(),
            events: Vec::new(),
        }
    }

    /// Process a new sequence ID observation
    pub fn observe_sequence_id(
        &mut self,
        sequence_id: u64,
        timestamp_ns: u64,
    ) -> Result<Option<RolloverEvent>, RolloverError> {
        // Auto-detect bit width on first observation
        if self.state.detected_bit_width.is_none() {
            self.state.detected_bit_width = Some(self.infer_bit_width(sequence_id));
        }

        let mut event = None;

        if let Some(last_id) = self.state.last_seen_id {
            // Check for rollover
            if sequence_id < last_id {
                // Rollover detected
                self.state.rollover_count += 1;
                
                let bit_width = self.state.detected_bit_width.unwrap_or(32);
                let max_id = if bit_width == 32 {
                    self.config.max_32bit
                } else {
                    self.config.max_64bit
                };

                // Estimate contention duration based on rollover proximity
                let distance_to_max = max_id.saturating_sub(last_id);
                let contention_us = if distance_to_max < self.config.warning_threshold {
                    // Very close to max - higher contention expected
                    (self.config.warning_threshold / distance_to_max.max(1)) as u64 * 10
                } else {
                    10 // Baseline contention
                };

                // Calculate severity
                let severity = ((self.config.warning_threshold as f64 
                    / distance_to_max.max(1) as f64) * 10.0) as u8 .min(10);

                event = Some(RolloverEvent {
                    rollover_point: last_id,
                    counter_bits: bit_width,
                    detection_timestamp_ns: timestamp_ns,
                    estimated_contention_us: contention_us.min(1000),
                    severity,
                });

                self.state.ids_since_last_rollover = 0;
                
                if let Some(ref e) = event {
                    self.events.push(e.clone());
                }
            } else {
                self.state.ids_since_last_rollover += 1;
            }

            // Check for approaching rollover (warning zone)
            let bit_width = self.state.detected_bit_width.unwrap_or(32);
            let max_id = if bit_width == 32 {
                self.config.max_32bit
            } else {
                self.config.max_64bit
            };

            if sequence_id > max_id.saturating_sub(self.config.warning_threshold) {
                // In warning zone - create a pre-rollover event
                let remaining = max_id.saturating_sub(sequence_id);
                let severity = ((self.config.warning_threshold as f64 
                    / remaining.max(1) as f64) * 5.0) as u8 .min(5);

                event = Some(RolloverEvent {
                    rollover_point: sequence_id,
                    counter_bits: bit_width,
                    detection_timestamp_ns: timestamp_ns,
                    estimated_contention_us: 0, // Not yet rolled over
                    severity,
                });
            }
        }

        self.state.last_seen_id = Some(sequence_id);
        Ok(event)
    }

    /// Infer bit width from observed sequence ID
    fn infer_bit_width(&self, id: u64) -> u8 {
        if id > u32::MAX as u64 {
            64
        } else if id > u16::MAX as u64 {
            32
        } else if id > u8::MAX as u64 {
            16
        } else {
            8
        }
    }

    /// Get all detected rollover events
    pub fn get_events(&self) -> &[RolloverEvent] {
        &self.events
    }

    /// Get the number of rollovers detected
    pub fn rollover_count(&self) -> usize {
        self.state.rollover_count
    }

    /// Check if we're currently in a warning zone
    pub fn is_in_warning_zone(&self) -> bool {
        if let (Some(last_id), Some(bit_width)) = (self.state.last_seen_id, self.state.detected_bit_width) {
            let max_id = if bit_width == 32 {
                self.config.max_32bit
            } else {
                self.config.max_64bit
            };
            last_id > max_id.saturating_sub(self.config.warning_threshold)
        } else {
            false
        }
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.state = RolloverState::new();
        self.events.clear();
    }
}

/// Errors from rollover detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RolloverError {
    InvalidSequenceId,
    TimestampRegression,
}

impl fmt::Display for RolloverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RolloverError::InvalidSequenceId => write!(f, "Invalid sequence ID"),
            RolloverError::TimestampRegression => write!(f, "Timestamp went backwards"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rollover_detection_32bit() {
        let config = RolloverConfig::default();
        let mut detector = SequenceIdRolloverDetector::new(config);

        // Observe IDs approaching 32-bit max
        let near_max = u32::MAX as u64 - 100;
        detector.observe_sequence_id(near_max, 1_000_000).unwrap();
        
        // Rollover to 0
        let event = detector.observe_sequence_id(0, 1_000_100).unwrap();
        
        assert!(event.is_some());
        assert_eq!(event.unwrap().counter_bits, 32);
    }

    #[test]
    fn test_warning_zone_detection() {
        let config = RolloverConfig {
            warning_threshold: 1000,
            ..Default::default()
        };
        let mut detector = SequenceIdRolloverDetector::new(config);

        // Enter warning zone
        let near_max = u32::MAX as u64 - 500;
        detector.observe_sequence_id(near_max, 1_000_000).unwrap();
        
        assert!(detector.is_in_warning_zone());
    }

    #[test]
    fn test_bit_width_inference() {
        let detector = SequenceIdRolloverDetector::new(RolloverConfig::default());
        
        assert_eq!(detector.infer_bit_width(100), 8);
        assert_eq!(detector.infer_bit_width(100_000), 32);
        assert_eq!(detector.infer_bit_width(u64::MAX), 64);
    }
}
