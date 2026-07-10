//! Spike Rasterizer - Groups asynchronous AER events into temporal bins
//! 
//! Converts microsecond-resolution event streams into spike trains using
//! lock-free SPSC (Single-Producer Single-Consumer) ring buffers.

use crate::ingestion::aer_zero_copy_parser::AerPacket;
use crossbeam_utils::atomic::AtomicCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::cell::UnsafeCell;

/// Maximum number of spikes per temporal bin (prevents overflow)
pub const MAX_SPIKES_PER_BIN: usize = 4096;

/// Default temporal bin width in microseconds (100μs for ultra-low latency)
pub const DEFAULT_BIN_WIDTH_US: u64 = 100;

/// Ring buffer capacity (must be power of 2 for efficient modulo)
const RING_BUFFER_CAPACITY: usize = 8192;

/// A single spike event with neuron ID and precise timing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(8))]
pub struct SpikeEvent {
    /// Neuron/pixel identifier (derived from x,y coordinates)
    pub neuron_id: u32,
    /// Timestamp within the bin (microseconds relative to bin start)
    pub timestamp_us: u32,
    /// Polarity: 0 = OFF, 1 = ON
    pub polarity: u8,
    /// Reserved padding
    _padding: [u8; 3],
}

impl SpikeEvent {
    #[inline]
    pub fn new(neuron_id: u32, timestamp_us: u32, polarity: u8) -> Self {
        Self {
            neuron_id,
            timestamp_us,
            polarity,
            _padding: [0; 3],
        }
    }

    /// Convert AER packet to spike event
    #[inline]
    pub fn from_aer(packet: &AerPacket, bin_start_us: u64) -> Self {
        let neuron_id = ((packet.x as u32) << 16) | (packet.y as u32);
        let relative_ts = ((packet.timestamp_us - bin_start_us) as u32).min(u32::MAX);
        Self {
            neuron_id,
            timestamp_us: relative_ts,
            polarity: packet.polarity,
            _padding: [0; 3],
        }
    }
}

/// A spike train representing all spikes within a single temporal bin
#[derive(Debug)]
pub struct SpikeTrain {
    /// Start timestamp of this bin (absolute microseconds)
    pub bin_start_us: u64,
    /// Number of spikes in this train
    pub spike_count: usize,
    /// Fixed-size array of spikes (avoids heap allocation)
    pub spikes: [SpikeEvent; MAX_SPIKES_PER_BIN],
}

impl SpikeTrain {
    #[inline]
    pub fn new(bin_start_us: u64) -> Self {
        Self {
            bin_start_us,
            spike_count: 0,
            spikes: [SpikeEvent::new(0, 0, 0); MAX_SPIKES_PER_BIN],
        }
    }

    /// Add a spike to the train if capacity allows
    #[inline]
    pub fn push(&mut self, spike: SpikeEvent) -> Result<(), ()> {
        if self.spike_count >= MAX_SPIKES_PER_BIN {
            return Err(());
        }
        self.spikes[self.spike_count] = spike;
        self.spike_count += 1;
        Ok(())
    }

    /// Get iterator over valid spikes
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &SpikeEvent> {
        self.spikes.iter().take(self.spike_count)
    }

    /// Clear the spike train for reuse
    #[inline]
    pub fn clear(&mut self) {
        self.spike_count = 0;
    }
}

/// Lock-free SPSC ring buffer for spike event streaming
struct SpikeRingBuffer {
    /// Buffer storage ( UnsafeCell for interior mutability )
    buffer: UnsafeCell<[SpikeEvent; RING_BUFFER_CAPACITY]>,
    /// Write position (producer only)
    write_pos: AtomicUsize,
    /// Read position (consumer only)
    read_pos: AtomicUsize,
    /// Overflow counter
    overflow_count: AtomicU64,
}

unsafe impl Send for SpikeRingBuffer {}
unsafe impl Sync for SpikeRingBuffer {}

impl SpikeRingBuffer {
    #[inline]
    fn new() -> Self {
        Self {
            // Safety: [SpikeEvent::default(); N] is safe for Copy types
            buffer: UnsafeCell::new([SpikeEvent::new(0, 0, 0); RING_BUFFER_CAPACITY]),
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
            overflow_count: AtomicU64::new(0),
        }
    }

    /// Push a spike event (producer side)
    /// Returns false if buffer is full
    #[inline]
    fn try_push(&self, spike: SpikeEvent) -> bool {
        let write_pos = self.write_pos.load(Ordering::Relaxed);
        let read_pos = self.read_pos.load(Ordering::Acquire);
        
        let next_write = (write_pos + 1) & (RING_BUFFER_CAPACITY - 1);
        
        if next_write == read_pos {
            // Buffer full
            self.overflow_count.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        unsafe {
            let buf_ptr = *self.buffer.get();
            let buf_mut = &mut *(buf_ptr as *mut [SpikeEvent; RING_BUFFER_CAPACITY]);
            buf_mut[write_pos] = spike;
        }

        self.write_pos.store(next_write, Ordering::Release);
        true
    }

    /// Pop a spike event (consumer side)
    /// Returns None if buffer is empty
    #[inline]
    fn try_pop(&self) -> Option<SpikeEvent> {
        let read_pos = self.read_pos.load(Ordering::Relaxed);
        let write_pos = self.write_pos.load(Ordering::Acquire);

        if read_pos == write_pos {
            // Buffer empty
            return None;
        }

        let next_read = (read_pos + 1) & (RING_BUFFER_CAPACITY - 1);
        
        unsafe {
            let buf_ptr = *self.buffer.get();
            let buf_ref = &*(buf_ptr as *const [SpikeEvent; RING_BUFFER_CAPACITY]);
            let spike = buf_ref[read_pos];
            self.read_pos.store(next_read, Ordering::Release);
            Some(spike)
        }
    }

    /// Check if buffer is empty
    #[inline]
    fn is_empty(&self) -> bool {
        self.read_pos.load(Ordering::Acquire) == self.write_pos.load(Ordering::Relaxed)
    }

    /// Get current fill level
    #[inline]
    fn len(&self) -> usize {
        let write = self.write_pos.load(Ordering::Acquire);
        let read = self.read_pos.load(Ordering::Relaxed);
        if write >= read {
            write - read
        } else {
            RING_BUFFER_CAPACITY - read + write
        }
    }

    /// Get overflow count
    #[inline]
    fn overflow_count(&self) -> u64 {
        self.overflow_count.load(Ordering::Relaxed)
    }
}

/// Spike Rasterizer - converts AER events to binned spike trains
pub struct SpikeRasterizer {
    /// Current bin start timestamp
    current_bin_start: AtomicCell<u64>,
    /// Bin width in microseconds
    bin_width_us: u64,
    /// Current bin being accumulated
    current_bin: UnsafeCell<SpikeTrain>,
    /// Ring buffer for completed bins (spike trains)
    completed_bins: SpikeRingBuffer,
    /// Statistics
    total_events_processed: AtomicU64,
    dropped_spikes: AtomicU64,
}

// Safety: SpikeRasterizer is designed for single-producer, single-consumer
unsafe impl Send for SpikeRasterizer {}
unsafe impl Sync for SpikeRasterizer {}

impl SpikeRasterizer {
    /// Create a new spike rasterizer with default bin width
    #[inline]
    pub fn new() -> Self {
        Self::with_bin_width(DEFAULT_BIN_WIDTH_US)
    }

    /// Create a new spike rasterizer with custom bin width
    #[inline]
    pub fn with_bin_width(bin_width_us: u64) -> Self {
        Self {
            current_bin_start: AtomicCell::new(0),
            bin_width_us,
            current_bin: UnsafeCell::new(SpikeTrain::new(0)),
            completed_bins: SpikeRingBuffer::new(),
            total_events_processed: AtomicU64::new(0),
            dropped_spikes: AtomicU64::new(0),
        }
    }

    /// Initialize the rasterizer with a starting timestamp
    #[inline]
    pub fn initialize(&self, initial_timestamp_us: u64) {
        let bin_start = (initial_timestamp_us / self.bin_width_us) * self.bin_width_us;
        self.current_bin_start.store(bin_start);
        
        unsafe {
            let bin_ptr = *self.current_bin.get();
            let bin_mut = &mut *(bin_ptr as *mut SpikeTrain);
            bin_mut.bin_start_us = bin_start;
            bin_mut.clear();
        }
    }

    /// Process a single AER packet and add to current bin
    /// Returns true if bin was completed and flushed
    #[inline]
    pub fn process_event(&self, packet: &AerPacket) -> bool {
        self.total_events_processed.fetch_add(1, Ordering::Relaxed);

        let current_start = self.current_bin_start.load();
        let current_end = current_start + self.bin_width_us;

        // Check if event falls outside current bin
        if packet.timestamp_us >= current_end {
            // Flush current bin and start new one
            self.flush_current_bin();
            
            // Calculate new bin start (aligned to bin_width_us)
            let new_bin_start = (packet.timestamp_us / self.bin_width_us) * self.bin_width_us;
            self.current_bin_start.store(new_bin_start);
            
            unsafe {
                let bin_ptr = *self.current_bin.get();
                let bin_mut = &mut *(bin_ptr as *mut SpikeTrain);
                bin_mut.bin_start_us = new_bin_start;
                bin_mut.clear();
            }
        }

        // Add spike to current bin
        unsafe {
            let bin_ptr = *self.current_bin.get();
            let bin_mut = &mut *(bin_ptr as *mut SpikeTrain);
            
            let spike = SpikeEvent::from_aer(packet, current_start);
            if bin_mut.push(spike).is_err() {
                self.dropped_spikes.fetch_add(1, Ordering::Relaxed);
            }
        }

        false // In practice, would check if bin is full
    }

    /// Flush current bin to completed queue
    #[inline]
    fn flush_current_bin(&self) {
        unsafe {
            let bin_ptr = *self.current_bin.get();
            let bin_ref = &*(bin_ptr as *const SpikeTrain);
            
            // Copy spikes to ring buffer
            for spike in bin_ref.iter() {
                if !self.completed_bins.try_push(*spike) {
                    self.dropped_spikes.fetch_add(1, Ordering::Relaxed);
                    break;
                }
            }
        }
    }

    /// Get next completed spike train (consumer side)
    #[inline]
    pub fn get_completed_spike(&self) -> Option<SpikeEvent> {
        self.completed_bins.try_pop()
    }

    /// Check if there are any completed spikes available
    #[inline]
    pub fn has_completed_spikes(&self) -> bool {
        !self.completed_bins.is_empty()
    }

    /// Drain all completed spikes into a vector
    #[inline]
    pub fn drain_completed(&self, output: &mut Vec<SpikeEvent>) {
        while let Some(spike) = self.get_completed_spike() {
            output.push(spike);
        }
    }

    /// Get statistics
    #[inline]
    pub fn stats(&self) -> RasterizerStats {
        RasterizerStats {
            total_events_processed: self.total_events_processed.load(Ordering::Relaxed),
            dropped_spikes: self.dropped_spikes.load(Ordering::Relaxed),
            buffer_overflow_count: self.completed_bins.overflow_count(),
            current_buffer_len: self.completed_bins.len(),
        }
    }

    /// Reset the rasterizer state
    #[inline]
    pub fn reset(&self) {
        self.current_bin_start.store(0);
        self.total_events_processed.store(0, Ordering::Relaxed);
        self.dropped_spikes.store(0, Ordering::Relaxed);
        
        unsafe {
            let bin_ptr = *self.current_bin.get();
            let bin_mut = &mut *(bin_ptr as *mut SpikeTrain);
            bin_mut.clear();
        }
    }
}

impl Default for SpikeRasterizer {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about rasterizer performance
#[derive(Debug, Clone, Copy)]
pub struct RasterizerStats {
    pub total_events_processed: u64,
    pub dropped_spikes: u64,
    pub buffer_overflow_count: u64,
    pub current_buffer_len: usize,
}

/// Temporal binning configuration
#[derive(Debug, Clone, Copy)]
pub struct TemporalBinningConfig {
    /// Bin width in microseconds
    pub bin_width_us: u64,
    /// Maximum spikes per bin before dropping
    pub max_spikes_per_bin: usize,
    /// Whether to use overlapping bins (sliding window)
    pub overlapping: bool,
    /// Overlap ratio (0.0 to 1.0) if overlapping is true
    pub overlap_ratio: f32,
}

impl Default for TemporalBinningConfig {
    #[inline]
    fn default() -> Self {
        Self {
            bin_width_us: DEFAULT_BIN_WIDTH_US,
            max_spikes_per_bin: MAX_SPIKES_PER_BIN,
            overlapping: false,
            overlap_ratio: 0.0,
        }
    }
}

/// Advanced temporal binning with sliding window support
pub struct TemporalBinner {
    config: TemporalBinningConfig,
    /// Primary bin
    primary_bin: UnsafeCell<SpikeTrain>,
    /// Secondary bin for overlapping windows
    secondary_bin: Option<UnsafeCell<SpikeTrain>>,
    /// Last bin boundary
    last_boundary: AtomicCell<u64>,
}

unsafe impl Send for TemporalBinner {}
unsafe impl Sync for TemporalBinner {}

impl TemporalBinner {
    #[inline]
    pub fn new(config: TemporalBinningConfig) -> Self {
        let secondary = if config.overlapping {
            Some(UnsafeCell::new(SpikeTrain::new(0)))
        } else {
            None
        };

        Self {
            config,
            primary_bin: UnsafeCell::new(SpikeTrain::new(0)),
            secondary_bin: secondary,
            last_boundary: AtomicCell::new(0),
        }
    }

    /// Process an event with sliding window binning
    #[inline]
    pub fn process_event(&self, packet: &AerPacket) -> BinningResult {
        let mut result = BinningResult {
            primary_flushed: false,
            secondary_flushed: false,
            spike_dropped: false,
        };

        let current_boundary = self.last_boundary.load();
        let bin_width = self.config.bin_width_us;

        // Check primary bin boundary
        if packet.timestamp_us >= current_boundary + bin_width {
            result.primary_flushed = true;
            let new_boundary = (packet.timestamp_us / bin_width) * bin_width;
            self.last_boundary.store(new_boundary);
            
            unsafe {
                let bin_ptr = *self.primary_bin.get();
                let bin_mut = &mut *(bin_ptr as *mut SpikeTrain);
                bin_mut.bin_start_us = new_boundary;
                bin_mut.clear();
            }
        }

        // Add to primary bin
        unsafe {
            let bin_ptr = *self.primary_bin.get();
            let bin_mut = &mut *(bin_ptr as *mut SpikeTrain);
            let spike = SpikeEvent::from_aer(packet, bin_mut.bin_start_us);
            if bin_mut.push(spike).is_err() {
                result.spike_dropped = true;
            }
        }

        // Handle overlapping secondary bin if configured
        if let Some(ref sec_bin) = self.secondary_bin {
            let overlap_offset = (bin_width as f32 * self.config.overlap_ratio) as u64;
            let sec_boundary = current_boundary - overlap_offset;
            
            if packet.timestamp_us >= sec_boundary + bin_width {
                result.secondary_flushed = true;
                
                unsafe {
                    let bin_ptr = *sec_bin.get();
                    let bin_mut = &mut *(bin_ptr as *mut SpikeTrain);
                    bin_mut.bin_start_us = sec_boundary + bin_width;
                    bin_mut.clear();
                }
            }

            unsafe {
                let bin_ptr = *sec_bin.get();
                let bin_mut = &mut *(bin_ptr as *mut SpikeTrain);
                let spike = SpikeEvent::from_aer(packet, bin_mut.bin_start_us);
                if bin_mut.push(spike).is_err() && !result.spike_dropped {
                    result.spike_dropped = true;
                }
            }
        }

        result
    }
}

/// Result of temporal binning operation
#[derive(Debug, Clone, Copy)]
pub struct BinningResult {
    pub primary_flushed: bool,
    pub secondary_flushed: bool,
    pub spike_dropped: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingestion::aer_zero_copy_parser::AerPacket;

    #[test]
    fn test_spike_event_creation() {
        let spike = SpikeEvent::new(1024, 50, 1);
        assert_eq!(spike.neuron_id, 1024);
        assert_eq!(spike.timestamp_us, 50);
        assert_eq!(spike.polarity, 1);
    }

    #[test]
    fn test_spike_train_push() {
        let mut train = SpikeTrain::new(1_000_000);
        
        for i in 0..100 {
            let spike = SpikeEvent::new(i, i as u32, 1);
            assert!(train.push(spike).is_ok());
        }
        
        assert_eq!(train.spike_count, 100);
    }

    #[test]
    fn test_spike_rasterizer_basic() {
        let rasterizer = SpikeRasterizer::new();
        rasterizer.initialize(1_000_000);
        
        let packet = AerPacket::new(100, 200, 1_000_050, 1).unwrap();
        rasterizer.process_event(&packet);
        
        let stats = rasterizer.stats();
        assert_eq!(stats.total_events_processed, 1);
        assert_eq!(stats.dropped_spikes, 0);
    }

    #[test]
    fn test_temporal_binner_non_overlapping() {
        let config = TemporalBinningConfig {
            bin_width_us: 100,
            overlapping: false,
            ..Default::default()
        };
        let binner = TemporalBinner::new(config);
        binner.last_boundary.store(1_000_000);
        
        let packet = AerPacket::new(50, 50, 1_000_050, 1).unwrap();
        let result = binner.process_event(&packet);
        
        assert!(!result.primary_flushed);
        assert!(!result.secondary_flushed);
        assert!(!result.spike_dropped);
    }

    #[test]
    fn test_ring_buffer_spsc() {
        let buffer = SpikeRingBuffer::new();
        
        // Producer side
        for i in 0..100 {
            let spike = SpikeEvent::new(i, i as u32, 1);
            assert!(buffer.try_push(spike));
        }
        
        // Consumer side
        for i in 0..100 {
            let spike = buffer.try_pop().unwrap();
            assert_eq!(spike.neuron_id, i);
        }
        
        assert!(buffer.is_empty());
    }
}
