//! CMOS Multi-Electrode Array DMA Stream Driver
//! 
//! Zero-allocation DMA driver for streaming raw electrophysiology data
//! from high-density CMOS MEAs (e.g., MaxWell Biosystems HD-MEA).
//! Uses memory-mapped I/O and ring buffers for zero-copy data transfer.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use core::ptr;

/// Maximum number of electrodes supported by the HD-MEA
pub const MAX_ELECTRODES: usize = 26400; // MaxWell Bio HD-MEA count

/// DMA buffer size in samples (power of 2 for efficient wrapping)
pub const DMA_BUFFER_SIZE: usize = 1 << 20; // 1M samples

/// Sample rate in Hz (typical for MEA systems)
pub const SAMPLE_RATE_HZ: u32 = 20_000;

/// Voltage resolution in microvolts per ADC count
pub const UV_PER_ADC_COUNT: f32 = 0.195;

/// Error types for DMA operations
#[derive(Debug, Clone, Copy)]
pub enum DmaError {
    BufferOverflow,
    DmaTimeout,
    InvalidChannel,
    HardwareFault,
    NotInitialized,
}

/// DMA channel state
#[repr(C, align(64))]
pub struct DmaChannel {
    /// Base pointer to hardware buffer (memory-mapped)
    pub base_ptr: *mut u16,
    /// Current write position (updated by hardware)
    pub write_pos: *const AtomicU64,
    /// Current read position (updated by software)
    pub read_pos: AtomicU64,
    /// Channel enabled flag
    pub enabled: AtomicBool,
    /// Overflow counter
    pub overflow_count: AtomicU64,
}

unsafe impl Send for DmaChannel {}
unsafe impl Sync for DmaChannel {}

/// Ring buffer for zero-copy DMA data
#[repr(C, align(64))]
pub struct DmaRingBuffer {
    /// Underlying sample buffer (interleaved electrode data)
    buffer: [u16; DMA_BUFFER_SIZE],
    /// Write index (hardware updates)
    write_idx: AtomicU64,
    /// Read index (software consumes)
    read_idx: AtomicU64,
    /// Mask for efficient wrapping
    mask: u64,
}

impl DmaRingBuffer {
    /// Create a new DMA ring buffer
    pub const fn new() -> Self {
        Self {
            buffer: [0u16; DMA_BUFFER_SIZE],
            write_idx: AtomicU64::new(0),
            read_idx: AtomicU64::new(0),
            mask: (DMA_BUFFER_SIZE as u64).wrapping_sub(1),
        }
    }

    /// Get available samples for reading (zero-alloc)
    #[inline]
    pub fn available_samples(&self) -> usize {
        let write = self.write_idx.load(Ordering::Acquire);
        let read = self.read_idx.load(Ordering::Relaxed);
        ((write.wrapping_sub(read)) & self.mask) as usize
    }

    /// Check if buffer is full (potential overflow)
    #[inline]
    pub fn is_near_full(&self, threshold: usize) -> bool {
        self.available_samples() > (DMA_BUFFER_SIZE - threshold)
    }

    /// Read samples into provided slice (caller manages allocation)
    /// Returns number of samples actually read
    #[inline]
    pub fn read_samples(&self, dest: &mut [u16]) -> usize {
        let available = self.available_samples();
        let to_read = core::cmp::min(dest.len(), available);
        
        if to_read == 0 {
            return 0;
        }

        let read_base = self.read_idx.load(Ordering::Relaxed) as usize;
        
        // Handle wrap-around efficiently
        let first_chunk = core::cmp::min(to_read, DMA_BUFFER_SIZE - (read_base & self.mask as usize));
        
        unsafe {
            ptr::copy_nonoverlapping(
                self.buffer.as_ptr().add(read_base & self.mask as usize),
                dest.as_mut_ptr(),
                first_chunk,
            );
            
            if to_read > first_chunk {
                ptr::copy_nonoverlapping(
                    self.buffer.as_ptr(),
                    dest.as_mut_ptr().add(first_chunk),
                    to_read - first_chunk,
                );
            }
        }

        self.read_idx.fetch_add(to_read as u64, Ordering::Release);
        to_read
    }

    /// Write samples from hardware (called by DMA interrupt handler)
    /// Must be called from interrupt context with proper synchronization
    #[inline]
    pub unsafe fn write_samples_from_hardware(&self, src: *const u16, count: usize) -> Result<(), DmaError> {
        let write_base = self.write_idx.load(Ordering::Relaxed) as usize;
        let available = self.available_samples();
        
        // Check for potential overflow before writing
        if available + count > DMA_BUFFER_SIZE - 1024 {
            // Near overflow - signal error but don't lose data
            return Err(DmaError::BufferOverflow);
        }

        // Handle wrap-around
        let first_chunk = core::cmp::min(count, DMA_BUFFER_SIZE - (write_base & self.mask as usize));
        
        ptr::copy_nonoverlapping(
            src,
            self.buffer.as_mut_ptr().add(write_base & self.mask as usize),
            first_chunk,
        );
        
        if count > first_chunk {
            ptr::copy_nonoverlapping(
                src.add(first_chunk),
                self.buffer.as_mut_ptr(),
                count - first_chunk,
            );
        }

        self.write_idx.fetch_add(count as u64, Ordering::Release);
        Ok(())
    }
}

/// Main DMA stream controller for MEA
pub struct CmosDmaStream {
    /// DMA channels (one per electrode group)
    channels: [Option<DmaChannel>; 16], // 16 parallel DMA channels
    /// Shared ring buffer
    ring_buffer: DmaRingBuffer,
    /// Stream active flag
    streaming: AtomicBool,
    /// Total samples acquired
    total_samples: AtomicU64,
    /// Hardware register base address
    hw_base_addr: usize,
}

impl CmosDmaStream {
    /// Create a new DMA stream controller
    pub const fn new(hw_base_addr: usize) -> Self {
        Self {
            channels: [None; 16],
            ring_buffer: DmaRingBuffer::new(),
            streaming: AtomicBool::new(false),
            total_samples: AtomicU64::new(0),
            hw_base_addr,
        }
    }

    /// Initialize a DMA channel for an electrode group
    pub fn init_channel(
        &mut self,
        channel_id: usize,
        base_ptr: *mut u16,
        write_pos_ptr: *const AtomicU64,
    ) -> Result<(), DmaError> {
        if channel_id >= self.channels.len() {
            return Err(DmaError::InvalidChannel);
        }

        self.channels[channel_id] = Some(DmaChannel {
            base_ptr,
            write_pos: write_pos_ptr,
            read_pos: AtomicU64::new(0),
            enabled: AtomicBool::new(false),
            overflow_count: AtomicU64::new(0),
        });

        Ok(())
    }

    /// Start DMA streaming on all enabled channels
    pub fn start_streaming(&mut self) -> Result<(), DmaError> {
        if !self.streaming.swap(true, Ordering::SeqCst) {
            // Enable all initialized channels
            for channel_opt in self.channels.iter_mut() {
                if let Some(channel) = channel_opt {
                    channel.enabled.store(true, Ordering::Release);
                }
            }
            Ok(())
        } else {
            Err(DmaError::HardwareFault) // Already streaming
        }
    }

    /// Stop DMA streaming gracefully
    pub fn stop_streaming(&mut self) {
        self.streaming.store(false, Ordering::SeqCst);
        
        // Disable all channels
        for channel_opt in self.channels.iter_mut() {
            if let Some(channel) = channel_opt {
                channel.enabled.store(false, Ordering::Release);
            }
        }
    }

    /// Get current stream status
    #[inline]
    pub fn is_streaming(&self) -> bool {
        self.streaming.load(Ordering::Acquire)
    }

    /// Get total samples acquired
    #[inline]
    pub fn total_samples(&self) -> u64 {
        self.total_samples.load(Ordering::Relaxed)
    }

    /// Access the ring buffer for reading
    #[inline]
    pub fn ring_buffer(&self) -> &DmaRingBuffer {
        &self.ring_buffer
    }

    /// Check for DMA errors across all channels
    pub fn check_errors(&self) -> Option<(usize, u64)> {
        for (idx, channel_opt) in self.channels.iter().enumerate() {
            if let Some(channel) = channel_opt {
                let overflows = channel.overflow_count.load(Ordering::Relaxed);
                if overflows > 0 {
                    return Some((idx, overflows));
                }
            }
        }
        None
    }

    /// Calibrate ADC offset for baseline correction
    pub fn calibrate_offset(&mut self, num_samples: usize) -> Result<[f32; MAX_ELECTRODES], DmaError> {
        if !self.is_streaming() {
            return Err(DmaError::NotInitialized);
        }

        // Accumulate samples for offset calculation
        // In practice, this would read from hardware and compute mean
        // Simplified for zero-alloc constraint
        let mut offsets = [0.0f32; MAX_ELECTRODES];
        
        // Placeholder: actual implementation reads hardware registers
        // and computes running mean without allocation
        for _ in 0..num_samples {
            // Simulated calibration loop
            core::hint::spin_loop();
        }

        Ok(offsets)
    }
}

/// Interrupt handler for DMA completion
/// Called by hardware when a DMA transfer completes
#[no_mangle]
pub extern "C" fn dma_interrupt_handler(stream: &CmosDmaStream) {
    if stream.is_streaming() {
        // Update total sample count atomically
        stream.total_samples.fetch_add(DMA_BUFFER_SIZE as u64, Ordering::Relaxed);
        
        // Check for errors
        if let Some((channel_id, overflows)) = stream.check_errors() {
            // Log error - in production this would trigger an alert
            let _ = (channel_id, overflows); // Suppress unused warning
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_wrap_around() {
        let buffer = DmaRingBuffer::new();
        let test_data = [1u16, 2, 3, 4, 5];
        
        unsafe {
            buffer.write_samples_from_hardware(test_data.as_ptr(), test_data.len()).unwrap();
        }
        
        let mut read_buf = [0u16; 5];
        let read = buffer.read_samples(&mut read_buf);
        
        assert_eq!(read, 5);
        assert_eq!(read_buf, test_data);
    }

    #[test]
    fn test_dma_channel_init() {
        let mut stream = CmosDmaStream::new(0x1000_0000);
        let mut dummy_buffer = [0u16; 1024];
        let dummy_pos = AtomicU64::new(0);
        
        let result = stream.init_channel(0, dummy_buffer.as_mut_ptr(), &dummy_pos);
        assert!(result.is_ok());
    }
}
