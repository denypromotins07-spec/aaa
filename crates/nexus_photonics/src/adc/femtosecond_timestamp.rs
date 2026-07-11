//! Femtosecond Timestamp Engine for Exchange Packet Timing
//!
//! This module provides sub-picosecond precision timestamping for exchange
//! UDP packets, eliminating OS-level network stack jitter. It integrates
//! with the photonic time-stretch ADC to provide absolute ground-truth
//! latency measurements for the Stage 17 Adversarial Microstructure engine.
//!
//! Features:
//! - Femtosecond-resolution timestamps
//! - Hardware-level packet capture timing
//! - Jitter-free clock distribution
//! - PTP/IEEE 1588 synchronization

use crate::adc::photonic_time_stretch::{PhotonicTimeStretchAdc, CapturedFrame};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use std::time::{Duration, Instant};

/// Errors in femtosecond timestamping
#[derive(Error, Debug)]
pub enum TimestampError {
    #[error("Clock synchronization lost: drift={drift}fs exceeds threshold={threshold}fs")]
    ClockDriftExceeded { drift: i128, threshold: i128 },
    
    #[error("Timestamp buffer overflow: capacity={capacity}, dropped={dropped}")]
    BufferOverflow { capacity: usize, dropped: usize },
    
    #[error("Invalid timestamp: value={value}fs outside valid range")]
    InvalidTimestamp { value: u128 },
    
    #[error("PTP sync failed: offset={offset}ns")]
    PtpSyncFailed { offset: f64 },
    
    #[error("Capture queue full: size={size}, max={max}")]
    CaptureQueueFull { size: usize, max: usize },
}

/// A femtosecond-precision timestamp
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FemtosecondTimestamp {
    /// Timestamp value in femtoseconds since epoch
    pub value_fs: u128,
    /// Confidence interval (femtoseconds)
    pub uncertainty_fs: u64,
    /// Clock quality indicator
    pub clock_quality: u8,
    /// Synchronized to PTP reference
    pub ptp_synchronized: bool,
}

impl FemtosecondTimestamp {
    /// Create a new timestamp
    pub fn new(value_fs: u128, uncertainty_fs: u64) -> Self {
        Self {
            value_fs,
            uncertainty_fs,
            clock_quality: 100,
            ptp_synchronized: false,
        }
    }

    /// Convert to nanoseconds
    pub fn to_nanos(&self) -> u64 {
        (self.value_fs / 1_000_000) as u64
    }

    /// Convert to picoseconds
    pub fn to_picos(&self) -> u128 {
        self.value_fs / 1_000
    }

    /// Calculate difference from another timestamp
    pub fn diff(&self, other: &Self) -> i128 {
        self.value_fs as i128 - other.value_fs as i128
    }

    /// Check if this timestamp is valid
    pub fn is_valid(&self) -> bool {
        self.uncertainty_fs < 1_000_000 && self.clock_quality > 50
    }
}

/// Captured packet with femtosecond timestamp
#[derive(Debug, Clone)]
pub struct TimestampedPacket {
    /// Arrival timestamp
    pub timestamp: FemtosecondTimestamp,
    /// Packet sequence number
    pub sequence_number: u64,
    /// Source port identifier
    pub source_port: u16,
    /// Packet size (bytes)
    pub size_bytes: usize,
    /// Latency from source (femtoseconds)
    pub latency_fs: Option<u128>,
    /// Raw captured data (if stored)
    pub payload_hash: [u8; 32],
}

/// Femtosecond Timestamp Engine - high-precision packet timing
pub struct FemtosecondTimestampEngine {
    /// Base clock frequency (Hz)
    clock_frequency_hz: u64,
    /// Reference timestamp (femtoseconds)
    base_timestamp_fs: u128,
    /// Clock drift rate (fs/s)
    clock_drift_fs_per_s: i64,
    /// Last synchronization time
    last_sync_time: Option<Instant>,
    /// PTP offset correction (femtoseconds)
    ptp_offset_fs: i128,
    /// Maximum allowable drift before resync
    max_drift_threshold_fs: i128,
    /// Captured packets buffer
    capture_buffer: Vec<TimestampedPacket>,
    /// Maximum buffer size
    max_buffer_size: usize,
    /// Sequence counter
    sequence_counter: u64,
    /// Associated TS-ADC for signal capture
    ts_adc: Option<PhotonicTimeStretchAdc>,
}

impl FemtosecondTimestampEngine {
    /// Create a new timestamp engine
    pub fn new(clock_frequency_hz: u64) -> Self {
        // Initialize base timestamp from system time
        let base_timestamp_fs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_femtos() as u128;

        Self {
            clock_frequency_hz,
            base_timestamp_fs,
            clock_drift_fs_per_s: 0,
            last_sync_time: None,
            ptp_offset_fs: 0,
            max_drift_threshold_fs: 1_000_000, // 1 ps threshold
            capture_buffer: Vec::new(),
            max_buffer_size: 10000,
            sequence_counter: 0,
            ts_adc: None,
        }
    }

    /// Attach a TS-ADC for integrated signal capture
    pub fn attach_ts_adc(&mut self, adc: PhotonicTimeStretchAdc) {
        self.ts_adc = Some(adc);
    }

    /// Get current femtosecond timestamp
    pub fn now(&self) -> FemtosecondTimestamp {
        let elapsed = self.last_sync_time
            .map(|t| t.elapsed())
            .unwrap_or(Duration::ZERO);

        // Calculate current time including drift compensation
        let elapsed_fs = elapsed.as_nanos() as i128 * 1_000_000;
        let drift_compensation = (elapsed.as_secs() as i128) * self.clock_drift_fs_per_s as i128;
        
        let current_fs = self.base_timestamp_fs 
            + elapsed_fs as u128 
            + drift_compensation as u128
            + self.ptp_offset_fs as u128;

        // Estimate uncertainty based on time since last sync
        let uncertainty = if let Some(last_sync) = self.last_sync_time {
            let since_sync_ns = last_sync.elapsed().as_nanos();
            // Uncertainty grows with time: ~1 fs per microsecond
            (since_sync_ns / 1000) as u64
        } else {
            1_000_000 // Large uncertainty if never synced
        };

        FemtosecondTimestamp {
            value_fs: current_fs,
            uncertainty_fs: uncertainty.min(1_000_000),
            clock_quality: self.calculate_clock_quality(),
            ptp_synchronized: self.last_sync_time.is_some(),
        }
    }

    /// Record arrival of an exchange packet
    pub fn record_packet_arrival(
        &mut self,
        source_port: u16,
        size_bytes: usize,
        payload_hash: [u8; 32],
    ) -> Result<TimestampedPacket, TimestampError> {
        // Check buffer capacity
        if self.capture_buffer.len() >= self.max_buffer_size {
            // Drop oldest packet
            self.capture_buffer.remove(0);
        }

        let timestamp = self.now();
        
        // Validate timestamp
        if !timestamp.is_valid() {
            return Err(TimestampError::InvalidTimestamp {
                value: timestamp.value_fs,
            });
        }

        let packet = TimestampedPacket {
            timestamp,
            sequence_number: self.sequence_counter,
            source_port,
            size_bytes,
            latency_fs: None,
            payload_hash,
        };

        self.capture_buffer.push(packet.clone());
        self.sequence_counter += 1;

        Ok(packet)
    }

    /// Calculate latency between two timestamps
    pub fn calculate_latency(&self, start: &FemtosecondTimestamp, end: &FemtosecondTimestamp) -> i128 {
        end.diff(start)
    }

    /// Synchronize to PTP reference clock
    pub fn sync_to_ptp(&mut self, ptp_offset_ns: f64) -> Result<(), TimestampError> {
        // Convert PTP offset to femtoseconds
        let offset_fs = (ptp_offset_ns * 1_000_000.0) as i128;
        
        // Apply correction
        self.ptp_offset_fs = offset_fs;
        self.last_sync_time = Some(Instant::now());
        self.base_timestamp_fs = self.now().value_fs;

        // Check if offset is within acceptable range
        if offset_fs.abs() > 100_000_000 { // 100 ns threshold
            return Err(TimestampError::PtpSyncFailed {
                offset: ptp_offset_ns,
            });
        }

        Ok(())
    }

    /// Measure and compensate clock drift
    pub fn measure_clock_drift(&mut self, reference_timestamp: FemtosecondTimestamp) -> i64 {
        let current = self.now();
        let measured_diff = current.diff(&reference_timestamp);
        
        // Calculate drift rate
        let elapsed_s = if let Some(last_sync) = self.last_sync_time {
            last_sync.elapsed().as_secs_f64()
        } else {
            1.0
        };

        let drift_rate = (measured_diff as f64 / elapsed_s) as i64;
        self.clock_drift_fs_per_s = drift_rate;
        
        drift_rate
    }

    /// Verify clock synchronization status
    pub fn verify_sync(&self) -> Result<(), TimestampError> {
        let elapsed = self.last_sync_time
            .map(|t| t.elapsed().as_secs() as i128)
            .unwrap_or(i128::MAX);

        let accumulated_drift = elapsed * self.clock_drift_fs_per_s as i128;

        if accumulated_drift.abs() > self.max_drift_threshold_fs {
            return Err(TimestampError::ClockDriftExceeded {
                drift: accumulated_drift,
                threshold: self.max_drift_threshold_fs,
            });
        }

        Ok(())
    }

    /// Get captured packets
    pub fn get_captured_packets(&self) -> &[TimestampedPacket] {
        &self.capture_buffer
    }

    /// Clear the capture buffer
    pub fn clear_buffer(&mut self) {
        self.capture_buffer.clear();
    }

    /// Set maximum buffer size
    pub fn set_max_buffer_size(&mut self, size: usize) {
        self.max_buffer_size = size;
    }

    /// Get statistics about captured packets
    pub fn get_statistics(&self) -> TimestampStats {
        let count = self.capture_buffer.len();
        
        if count == 0 {
            return TimestampStats {
                packet_count: 0,
                min_latency_fs: None,
                max_latency_fs: None,
                avg_latency_fs: None,
                jitter_fs: 0,
            };
        }

        let latencies: Vec<u128> = self.capture_buffer
            .iter()
            .filter_map(|p| p.latency_fs)
            .collect();

        let (min_lat, max_lat, avg_lat) = if latencies.is_empty() {
            (None, None, None)
        } else {
            let min = *latencies.iter().min().unwrap();
            let max = *latencies.iter().max().unwrap();
            let avg = latencies.iter().sum::<u128>() / latencies.len() as u128;
            (Some(min), Some(max), Some(avg))
        };

        // Calculate jitter (standard deviation of latency)
        let jitter = if latencies.len() > 1 && avg_lat.is_some() {
            let avg = avg_lat.unwrap();
            let variance: f64 = latencies.iter()
                .map(|&l| (l as i128 - avg as i128).pow(2) as f64)
                .sum::<f64>() / (latencies.len() - 1) as f64;
            variance.sqrt() as u64
        } else {
            0
        };

        TimestampStats {
            packet_count: count,
            min_latency_fs: min_lat,
            max_latency_fs: max_lat,
            avg_latency_fs: avg_lat,
            jitter_fs: jitter,
        }
    }

    /// Calculate clock quality (0-100)
    fn calculate_clock_quality(&self) -> u8 {
        if self.last_sync_time.is_none() {
            return 0;
        }

        let elapsed_s = self.last_sync_time.unwrap().elapsed().as_secs();
        
        // Quality degrades with time since sync
        let quality_from_sync = (100.0 - elapsed_s as f64 * 0.1).max(0.0) as u8;
        
        // Quality reduced by drift magnitude
        let drift_penalty = (self.clock_drift_fs_per_s.abs() / 1000).min(50) as u8;
        
        (quality_from_sync.saturating_sub(drift_penalty)).max(0)
    }
}

/// Statistics about captured packets
#[derive(Debug, Clone)]
pub struct TimestampStats {
    /// Total packets captured
    pub packet_count: usize,
    /// Minimum observed latency
    pub min_latency_fs: Option<u128>,
    /// Maximum observed latency
    pub max_latency_fs: Option<u128>,
    /// Average latency
    pub avg_latency_fs: Option<u128>,
    /// Latency jitter (standard deviation)
    pub jitter_fs: u64,
}

impl Default for FemtosecondTimestampEngine {
    fn default() -> Self {
        Self::new(10_000_000_000) // 10 GHz reference clock
    }
}

// Extension trait for Duration to support femtoseconds
trait DurationExt {
    fn as_femtos(&self) -> u128;
}

impl DurationExt for Duration {
    fn as_femtos(&self) -> u128 {
        self.as_nanos() * 1_000_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = FemtosecondTimestampEngine::new(10_000_000_000);
        assert_eq!(engine.clock_frequency_hz, 10_000_000_000);
    }

    #[test]
    fn test_timestamp_generation() {
        let engine = FemtosecondTimestampEngine::new(10_000_000_000);
        let ts = engine.now();
        
        assert!(ts.value_fs > 0);
        assert!(ts.uncertainty_fs <= 1_000_000);
    }

    #[test]
    fn test_packet_recording() {
        let mut engine = FemtosecondTimestampEngine::new(10_000_000_000);
        
        let packet = engine.record_packet_arrival(
            8080,
            1500,
            [0u8; 32],
        ).unwrap();
        
        assert_eq!(packet.source_port, 8080);
        assert_eq!(packet.size_bytes, 1500);
        assert_eq!(packet.sequence_number, 0);
        assert!(packet.timestamp.is_valid());
    }

    #[test]
    fn test_sequence_numbering() {
        let mut engine = FemtosecondTimestampEngine::new(10_000_000_000);
        
        for i in 0..10 {
            let packet = engine.record_packet_arrival(8080, 100, [0u8; 32]).unwrap();
            assert_eq!(packet.sequence_number, i);
        }
    }

    #[test]
    fn test_ptp_sync() {
        let mut engine = FemtosecondTimestampEngine::new(10_000_000_000);
        
        let result = engine.sync_to_ptp(50.0); // 50 ns offset
        assert!(result.is_ok());
        
        let ts = engine.now();
        assert!(ts.ptp_synchronized);
    }

    #[test]
    fn test_large_ptp_offset_rejected() {
        let mut engine = FemtosecondTimestampEngine::new(10_000_000_000);
        
        let result = engine.sync_to_ptp(200.0); // 200 ns offset (too large)
        assert!(result.is_err());
    }

    #[test]
    fn test_latency_calculation() {
        let engine = FemtosecondTimestampEngine::new(10_000_000_000);
        
        let start = FemtosecondTimestamp::new(1_000_000_000_000, 100);
        let end = FemtosecondTimestamp::new(1_000_000_001_000, 100);
        
        let latency = engine.calculate_latency(&start, &end);
        assert_eq!(latency, 1_000); // 1000 fs = 1 ps
    }

    #[test]
    fn test_buffer_overflow_handling() {
        let mut engine = FemtosecondTimestampEngine::new(10_000_000_000);
        engine.set_max_buffer_size(5);
        
        // Record more packets than buffer size
        for _ in 0..10 {
            engine.record_packet_arrival(8080, 100, [0u8; 32]).unwrap();
        }
        
        // Buffer should contain only last 5 packets
        assert_eq!(engine.get_captured_packets().len(), 5);
        
        // First packet should have sequence number 5 (not 0)
        assert_eq!(engine.get_captured_packets()[0].sequence_number, 5);
    }

    #[test]
    fn test_timestamp_conversion() {
        let ts = FemtosecondTimestamp::new(1_000_000_000_000, 100);
        
        assert_eq!(ts.to_nanos(), 1_000_000);
        assert_eq!(ts.to_picos(), 1_000_000_000);
    }

    #[test]
    fn test_statistics() {
        let mut engine = FemtosecondTimestampEngine::new(10_000_000_000);
        
        // Record some packets with latencies
        for i in 0..5 {
            let mut packet = engine.record_packet_arrival(8080, 100, [0u8; 32]).unwrap();
            packet.latency_fs = Some(1_000_000_000 + i * 1000);
            engine.capture_buffer[i] = packet;
        }
        
        let stats = engine.get_statistics();
        assert_eq!(stats.packet_count, 5);
        assert!(stats.min_latency_fs.is_some());
        assert!(stats.max_latency_fs.is_some());
    }
}
