//! Angular Multiplexing for Holographic Storage
//! 
//! Implements zero-allocation streaming of tick data into staging buffers
//! for batched holographic writing with angular multiplexing.

use crate::holographic::bragg_grating_simulator::{BraggGratingSimulator, GratingParameters};
use crate::holographic::volumetric_page_encoder::VolumetricPageEncoder;
use thiserror::Error;

/// Maximum tick data buffer size (pre-allocated)
pub const MAX_TICK_BUFFER_SIZE: usize = 1_048_576; // 1M ticks

/// Tick data representation
#[derive(Debug, Clone, Copy)]
pub struct TickData {
    pub timestamp_ns: u64,
    pub price: f64,
    pub size: f64,
    pub bid_ask_flag: u8, // 0 = bid, 1 = ask
}

#[derive(Error, Debug)]
pub enum MultiplexerError {
    #[error("Buffer overflow")]
    BufferOverflow,
    #[error("Invalid tick data")]
    InvalidTickData,
    #[error("Holographic write failed")]
    HolographicWriteFailed,
    #[error("No available pages")]
    NoAvailablePages,
}

/// Zero-allocation ring buffer for tick data staging
pub struct TickStagingBuffer {
    data: Box<[TickData]>,
    head: usize,
    tail: usize,
    count: usize,
    capacity: usize,
}

impl TickStagingBuffer {
    /// Create a pre-allocated staging buffer
    pub fn new(capacity: usize) -> Result<Self, MultiplexerError> {
        if capacity > MAX_TICK_BUFFER_SIZE {
            return Err(MultiplexerError::BufferOverflow);
        }
        let data = vec![TickData { timestamp_ns: 0, price: 0.0, size: 0.0, bid_ask_flag: 0 }; capacity]
            .into_boxed_slice();
        Ok(Self { data, head: 0, tail: 0, count: 0, capacity })
    }

    #[inline]
    pub fn push(&mut self, tick: TickData) -> Result<(), MultiplexerError> {
        if self.count >= self.capacity {
            return Err(MultiplexerError::BufferOverflow);
        }
        self.data[self.tail] = tick;
        self.tail = (self.tail + 1) % self.capacity;
        self.count += 1;
        Ok(())
    }

    #[inline]
    pub fn pop(&mut self) -> Option<TickData> {
        if self.count == 0 {
            return None;
        }
        let tick = self.data[self.head];
        self.head = (self.head + 1) % self.capacity;
        self.count -= 1;
        Some(tick)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.count >= self.capacity
    }

    /// Get all ticks as a slice for batch processing
    pub fn as_slice(&self) -> &[TickData] {
        if self.count == 0 {
            return &[];
        }
        
        // Handle wrap-around by returning contiguous segment
        if self.head <= self.tail {
            &self.data[self.head..self.tail]
        } else {
            // Return from head to end (caller must handle remaining from 0 to tail)
            &self.data[self.head..]
        }
    }

    /// Clear buffer without deallocation
    pub fn clear(&mut self) {
        self.head = 0;
        self.tail = 0;
        self.count = 0;
    }
}

/// Holographic Multiplexer for angular multiplexed storage
pub struct HolographicMultiplexer {
    staging_buffer: TickStagingBuffer,
    bragg_simulator: BraggGratingSimulator,
    current_page_encoder: VolumetricPageEncoder,
    base_wavelength_nm: f64,
    ticks_per_page: usize,
    page_counter: u64,
}

impl HolographicMultiplexer {
    /// Create a new multiplexer with specified parameters
    pub fn new(
        crystal_thickness_um: f64,
        max_pages: usize,
        base_wavelength_nm: f64,
        ticks_per_page: usize,
    ) -> Result<Self, MultiplexerError> {
        let staging_buffer = TickStagingBuffer::new(MAX_TICK_BUFFER_SIZE)?;
        let bragg_simulator = BraggGratingSimulator::new(crystal_thickness_um, max_pages);
        
        // Initialize first page encoder
        let mut current_page_encoder = VolumetricPageEncoder::new(0, 64, 64, 16)
            .map_err(|_| MultiplexerError::HolographicWriteFailed)?;

        Ok(Self {
            staging_buffer,
            bragg_simulator,
            current_page_encoder,
            base_wavelength_nm,
            ticks_per_page,
            page_counter: 0,
        })
    }

    /// Stream live tick data into the staging buffer
    #[inline]
    pub fn stream_tick(&mut self, tick: TickData) -> Result<(), MultiplexerError> {
        self.staging_buffer.push(tick)
    }

    /// Check if enough ticks are buffered for a holographic write
    #[inline]
    pub fn is_ready_for_write(&self) -> bool {
        self.staging_buffer.len() >= self.ticks_per_page
    }

    /// Convert tick data to intensity matrix for holographic encoding
    fn ticks_to_matrix(&self, ticks: &[TickData], width: usize, height: usize) -> Vec<f32> {
        let mut matrix = vec![0.0f32; width * height];
        
        for (i, tick) in ticks.iter().take(width * height).enumerate() {
            // Encode price fractional part and size into intensity
            let price_component = (tick.price.fract() * 1000.0).abs() as f32 / 1000.0;
            let size_component = tick.size.clamp(0.0, 1000.0) as f32 / 1000.0;
            let bid_ask_component = if tick.bid_ask_flag == 0 { 0.0 } else { 0.5 };
            
            matrix[i] = (price_component + size_component + bid_ask_component) / 2.0;
        }
        
        matrix
    }

    /// Write buffered ticks to holographic storage using angular multiplexing
    pub fn write_batch(&mut self) -> Result<GratingParameters, MultiplexerError> {
        if self.staging_buffer.len() < self.ticks_per_page {
            return Err(MultiplexerError::NoAvailablePages);
        }

        // Allocate new page with unique angle
        let page_params = self.bragg_simulator.allocate_page(self.base_wavelength_nm)
            .map_err(|_| MultiplexerError::HolographicWriteFailed)?;

        // Get ticks for this page
        let mut ticks_buffer = [TickData { timestamp_ns: 0, price: 0.0, size: 0.0, bid_ask_flag: 0 }; 4096];
        let ticks_to_write = self.ticks_per_page.min(4096);
        
        let mut idx = 0;
        while idx < ticks_to_write {
            if let Some(tick) = self.staging_buffer.pop() {
                ticks_buffer[idx] = tick;
                idx += 1;
            } else {
                break;
            }
        }

        // Convert to matrix and encode
        let matrix = self.ticks_to_matrix(&ticks_buffer[..idx], 64, 64);
        
        self.current_page_encoder = VolumetricPageEncoder::new(page_params.page_id, 64, 64, 16)
            .map_err(|_| MultiplexerError::HolographicWriteFailed)?;
        
        self.current_page_encoder.encode_matrix(&matrix, 64, 64, 16)
            .map_err(|_| MultiplexerError::HolographicWriteFailed)?;

        self.page_counter += 1;

        Ok(page_params)
    }

    /// Read data from a specific page by angle
    pub fn read_page(&self, page_id: u64, angle_tolerance_rad: f64) -> Result<Vec<TickData>, MultiplexerError> {
        let angle = self.bragg_simulator.find_page_by_angle(
            // Find angle for this page ID
            0.0, // Placeholder - would need reverse lookup
            angle_tolerance_rad,
        ).ok_or(MultiplexerError::NoAvailablePages)?;

        // Simulate read operation
        let efficiency = self.bragg_simulator.simulate_read(page_id, angle, self.base_wavelength_nm)
            .map_err(|_| MultiplexerError::HolographicWriteFailed)?;

        if efficiency < 0.01 {
            return Err(MultiplexerError::HolographicWriteFailed);
        }

        // Return placeholder ticks (actual implementation would decode voxels)
        Ok(Vec::new())
    }

    /// Get current buffer fill level
    #[inline]
    pub fn buffer_fill_ratio(&self) -> f64 {
        self.staging_buffer.len() as f64 / self.staging_buffer.capacity as f64
    }

    /// Get total pages written
    #[inline]
    pub fn pages_written(&self) -> u64 {
        self.page_counter
    }

    /// Clear staging buffer (e.g., after successful write)
    pub fn clear_staging(&mut self) {
        self.staging_buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_staging_buffer_basic() {
        let mut buffer = TickStagingBuffer::new(100).unwrap();
        assert!(buffer.is_empty());
        
        let tick = TickData { timestamp_ns: 1000, price: 100.5, size: 10.0, bid_ask_flag: 0 };
        buffer.push(tick).unwrap();
        assert_eq!(buffer.len(), 1);
        
        let popped = buffer.pop().unwrap();
        assert_eq!(popped.price, 100.5);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_multiplexer_creation() {
        let mux = HolographicMultiplexer::new(1000.0, 100, 532.0, 1000).unwrap();
        assert_eq!(mux.pages_written(), 0);
    }

    #[test]
    fn test_stream_and_write() {
        let mut mux = HolographicMultiplexer::new(1000.0, 100, 532.0, 10).unwrap();
        
        // Stream some ticks
        for i in 0..10 {
            let tick = TickData {
                timestamp_ns: i * 1000,
                price: 100.0 + i as f64 * 0.01,
                size: 10.0,
                bid_ask_flag: i % 2,
            };
            mux.stream_tick(tick).unwrap();
        }
        
        assert!(mux.is_ready_for_write());
        
        // Write batch
        let result = mux.write_batch();
        assert!(result.is_ok());
        assert_eq!(mux.pages_written(), 1);
    }
}
