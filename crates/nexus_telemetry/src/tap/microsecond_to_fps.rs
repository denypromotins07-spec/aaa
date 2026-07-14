//! Microsecond to FPS Conversion Utilities
//!
//! This module provides utilities for converting microsecond-precision trading
//! events into fixed FPS frames suitable for UI rendering.

use std::time::{Duration, Instant};

/// Converts a duration in microseconds to the equivalent number of frames at target FPS
#[inline]
pub const fn micros_to_frames(micros: u64, fps: u32) -> u64 {
    let frame_duration_micros = 1_000_000 / fps as u64;
    micros / frame_duration_micros
}

/// Converts a duration in nanoseconds to the equivalent number of frames at target FPS
#[inline]
pub const fn nanos_to_frames(nanos: u64, fps: u32) -> u64 {
    let frame_duration_nanos = 1_000_000_000 / fps as u64;
    nanos / frame_duration_nanos
}

/// Returns the frame duration in nanoseconds for a given FPS
#[inline]
pub const fn fps_to_frame_duration_ns(fps: u32) -> u64 {
    1_000_000_000 / fps as u64
}

/// Returns the frame duration in microseconds for a given FPS
#[inline]
pub const fn fps_to_frame_duration_us(fps: u32) -> u64 {
    1_000_000 / fps as u32
}

/// Standard 60fps frame duration in nanoseconds (16,666,667 ns)
pub const FRAME_60FPS_NS: u64 = 16_666_667;

/// Standard 60fps frame duration in microseconds (16,667 us)
pub const FRAME_60FPS_US: u64 = 16_667;

/// Standard 120fps frame duration in nanoseconds (8,333,333 ns)
pub const FRAME_120FPS_NS: u64 = 8_333_333;

/// Standard 144fps frame duration in nanoseconds (6,944,444 ns)
pub const FRAME_144FPS_NS: u64 = 6_944_444;

/// Helper for tracking frame timing
pub struct FrameTimer {
    /// Target frame duration in nanoseconds
    frame_duration_ns: u64,
    /// Last frame timestamp
    last_frame: Instant,
    /// Frame counter
    frame_count: u64,
}

impl FrameTimer {
    /// Create a new frame timer for the given FPS
    pub fn new(fps: u32) -> Self {
        Self {
            frame_duration_ns: fps_to_frame_duration_ns(fps),
            last_frame: Instant::now(),
            frame_count: 0,
        }
    }

    /// Check if it's time for a new frame
    #[inline]
    pub fn should_render(&self) -> bool {
        self.last_frame.elapsed().as_nanos() as u64 >= self.frame_duration_ns
    }

    /// Mark that a frame has been rendered
    #[inline]
    pub fn mark_frame(&mut self) {
        self.last_frame = Instant::now();
        self.frame_count += 1;
    }

    /// Get the current frame count
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get elapsed time since last frame in nanoseconds
    pub fn elapsed_since_last_frame_ns(&self) -> u64 {
        self.last_frame.elapsed().as_nanos() as u64
    }

    /// Wait until next frame (blocking)
    pub fn wait_for_next_frame(&mut self) {
        while !self.should_render() {
            std::hint::spin_loop();
        }
        self.mark_frame();
    }
}

/// Rate limiter for telemetry emission
pub struct TelemetryRateLimiter {
    /// Minimum interval between emissions in nanoseconds
    min_interval_ns: u64,
    /// Last emission timestamp
    last_emission: Instant,
    /// Count of dropped emissions due to rate limiting
    dropped: u64,
}

impl TelemetryRateLimiter {
    /// Create a rate limiter for the given FPS
    pub fn from_fps(fps: u32) -> Self {
        Self {
            min_interval_ns: fps_to_frame_duration_ns(fps),
            last_emission: Instant::now(),
            dropped: 0,
        }
    }

    /// Check if emission is allowed
    #[inline]
    pub fn can_emit(&self) -> bool {
        self.last_emission.elapsed().as_nanos() as u64 >= self.min_interval_ns
    }

    /// Try to emit - returns true if emission was allowed
    #[inline]
    pub fn try_emit(&mut self) -> bool {
        if self.can_emit() {
            self.last_emission = Instant::now();
            true
        } else {
            self.dropped += 1;
            false
        }
    }

    /// Get count of dropped emissions
    pub fn dropped_count(&self) -> u64 {
        self.dropped
    }

    /// Reset the rate limiter
    pub fn reset(&mut self) {
        self.last_emission = Instant::now();
        self.dropped = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_micros_to_frames() {
        assert_eq!(micros_to_frames(16_667, 60), 1);
        assert_eq!(micros_to_frames(33_334, 60), 2);
        assert_eq!(micros_to_frames(100_000, 60), 6);
    }

    #[test]
    fn test_nanos_to_frames() {
        assert_eq!(nanos_to_frames(16_666_667, 60), 1);
        assert_eq!(nanos_to_frames(33_333_334, 60), 2);
    }

    #[test]
    fn test_fps_constants() {
        assert_eq!(FRAME_60FPS_NS, 16_666_667);
        assert_eq!(FRAME_120FPS_NS, 8_333_333);
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = TelemetryRateLimiter::from_fps(60);
        
        // First emission should succeed
        assert!(limiter.try_emit());
        
        // Immediate second emission should fail (rate limited)
        assert!(!limiter.try_emit());
        assert_eq!(limiter.dropped_count(), 1);
    }
}
