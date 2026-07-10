//! Temporal Binning Module
//! 
//! Re-exports and additional temporal binning utilities for spike processing.

pub use crate::ingestion::spike_rasterizer::{
    BinningResult,
    DEFAULT_BIN_WIDTH_US,
    MAX_SPIKES_PER_BIN,
    RasterizerStats,
    SpikeEvent,
    SpikeRasterizer,
    SpikeRingBuffer,
    SpikeTrain,
    TemporalBinner,
    TemporalBinningConfig,
};

/// Advanced multi-scale temporal binning for hierarchical spike processing
pub struct MultiScaleBinner {
    /// Fine binning (microsecond resolution)
    fine_bin_width_us: u64,
    /// Coarse binning (millisecond resolution)
    coarse_bin_width_us: u64,
    /// Fine spike accumulator
    fine_spikes: Vec<SpikeEvent>,
    /// Coarse spike accumulator  
    coarse_spikes: Vec<SpikeEvent>,
}

impl MultiScaleBinner {
    #[inline]
    pub fn new(fine_bin_width_us: u64, coarse_bin_width_us: u64) -> Self {
        Self {
            fine_bin_width_us,
            coarse_bin_width_us,
            fine_spikes: Vec::with_capacity(1024),
            coarse_spikes: Vec::with_capacity(256),
        }
    }

    /// Process event at multiple temporal scales
    #[inline]
    pub fn process_multi_scale(&mut self, event: &SpikeEvent) -> MultiScaleResult {
        // Add to fine scale
        self.fine_spikes.push(*event);
        
        // Add to coarse scale (downsampled)
        if event.timestamp_us % self.coarse_bin_width_us < self.fine_bin_width_us {
            self.coarse_spikes.push(*event);
        }

        MultiScaleResult {
            fine_count: self.fine_spikes.len(),
            coarse_count: self.coarse_spikes.len(),
        }
    }

    /// Drain fine-scale spikes
    #[inline]
    pub fn drain_fine(&mut self) -> impl Iterator<Item = SpikeEvent> + '_ {
        self.fine_spikes.drain(..)
    }

    /// Drain coarse-scale spikes
    #[inline]
    pub fn drain_coarse(&mut self) -> impl Iterator<Item = SpikeEvent> + '_ {
        self.coarse_spikes.drain(..)
    }

    /// Clear all accumulators
    #[inline]
    pub fn clear(&mut self) {
        self.fine_spikes.clear();
        self.coarse_spikes.clear();
    }
}

/// Result of multi-scale binning
#[derive(Debug, Clone, Copy)]
pub struct MultiScaleResult {
    pub fine_count: usize,
    pub coarse_count: usize,
}

/// Adaptive temporal binning that adjusts bin width based on event rate
pub struct AdaptiveBinner {
    /// Current bin width in microseconds
    current_bin_width_us: u64,
    /// Minimum allowed bin width
    min_bin_width_us: u64,
    /// Maximum allowed bin width
    max_bin_width_us: u64,
    /// Target events per bin
    target_events_per_bin: usize,
    /// Events in current bin
    current_event_count: usize,
    /// Last adjustment timestamp
    last_adjustment_ts: u64,
}

impl AdaptiveBinner {
    #[inline]
    pub fn new(
        initial_bin_width_us: u64,
        min_bin_width_us: u64,
        max_bin_width_us: u64,
        target_events_per_bin: usize,
    ) -> Self {
        Self {
            current_bin_width_us: initial_bin_width_us,
            min_bin_width_us,
            max_bin_width_us,
            target_events_per_bin,
            current_event_count: 0,
            last_adjustment_ts: 0,
        }
    }

    /// Record an event and potentially adjust bin width
    #[inline]
    pub fn record_event(&mut self, timestamp_us: u64) {
        self.current_event_count += 1;

        // Adjust bin width every 100 events or 1ms
        let should_adjust = self.current_event_count >= self.target_events_per_bin
            || timestamp_us - self.last_adjustment_ts >= 1000;

        if should_adjust {
            self.adjust_bin_width();
            self.current_event_count = 0;
            self.last_adjustment_ts = timestamp_us;
        }
    }

    /// Adjust bin width based on observed event rate
    #[inline]
    fn adjust_bin_width(&mut self) {
        if self.current_event_count > self.target_events_per_bin * 2 {
            // Too many events - increase bin width
            self.current_bin_width_us = (self.current_bin_width_us * 3 / 2)
                .min(self.max_bin_width_us);
        } else if self.current_event_count < self.target_events_per_bin / 4 {
            // Too few events - decrease bin width for better resolution
            self.current_bin_width_us = (self.current_bin_width_us * 2 / 3)
                .max(self.min_bin_width_us);
        }
    }

    /// Get current bin width
    #[inline]
    pub fn current_bin_width(&self) -> u64 {
        self.current_bin_width_us
    }

    /// Reset adaptive state
    #[inline]
    pub fn reset(&mut self) {
        self.current_event_count = 0;
        self.last_adjustment_ts = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_binner_increase() {
        let mut binner = AdaptiveBinner::new(100, 50, 500, 10);
        
        // Simulate high event rate
        for _ in 0..25 {
            binner.record_event(1000);
        }
        
        // Bin width should have increased
        assert!(binner.current_bin_width() > 100);
    }

    #[test]
    fn test_adaptive_binner_decrease() {
        let mut binner = AdaptiveBinner::new(100, 50, 500, 10);
        
        // Simulate low event rate
        for _ in 0..2 {
            binner.record_event(1000);
        }
        
        // Force adjustment
        binner.record_event(2000);
        
        // Bin width should have decreased or stayed at minimum
        assert!(binner.current_bin_width() <= 100);
    }
}
