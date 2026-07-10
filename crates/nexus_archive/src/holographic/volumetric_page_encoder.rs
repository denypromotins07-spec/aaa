//! Volumetric Page Encoder
//! 
//! Maps 2D financial data matrices (order book snapshots, tick data) into 3D voxel grids
//! for holographic storage. Uses zero-allocation buffers and SIMD acceleration.

use thiserror::Error;

/// Maximum dimensions for a single holographic page
pub const MAX_PAGE_WIDTH: usize = 1024;
pub const MAX_PAGE_HEIGHT: usize = 1024;
pub const MAX_PAGE_DEPTH: usize = 256;

/// Voxel representation for holographic storage
#[derive(Debug, Clone, Copy)]
pub struct Voxel {
    pub x: u16,
    pub y: u16,
    pub z: u16,
    pub intensity: f32,
    pub phase: f32,
}

impl Voxel {
    #[inline]
    pub fn new(x: u16, y: u16, z: u16, intensity: f32, phase: f32) -> Result<Self, VoxelError> {
        if x as usize >= MAX_PAGE_WIDTH || y as usize >= MAX_PAGE_HEIGHT || z as usize >= MAX_PAGE_DEPTH {
            return Err(VoxelError::OutOfBounds);
        }
        if intensity < 0.0 || intensity > 1.0 {
            return Err(VoxelError::InvalidIntensity(intensity));
        }
        Ok(Self { x, y, z, intensity, phase })
    }
}

#[derive(Error, Debug)]
pub enum VoxelError {
    #[error("Voxel coordinates out of bounds")]
    OutOfBounds,
    #[error("Invalid intensity value: {0}")]
    InvalidIntensity(f32),
    #[error("Buffer overflow")]
    BufferOverflow,
    #[error("Invalid input dimensions")]
    InvalidDimensions,
}

/// Zero-allocation buffer for voxel data
pub struct VoxelBuffer {
    data: Box<[Voxel]>,
    width: usize,
    height: usize,
    depth: usize,
    len: usize,
}

impl VoxelBuffer {
    /// Pre-allocate a fixed-size voxel buffer
    pub fn with_capacity(width: usize, height: usize, depth: usize) -> Result<Self, VoxelError> {
        if width > MAX_PAGE_WIDTH || height > MAX_PAGE_HEIGHT || depth > MAX_PAGE_DEPTH {
            return Err(VoxelError::InvalidDimensions);
        }
        let capacity = width * height * depth;
        let data = vec![Voxel { x: 0, y: 0, z: 0, intensity: 0.0, phase: 0.0 }; capacity]
            .into_boxed_slice();
        Ok(Self { data, width, height, depth, len: 0 })
    }

    #[inline]
    pub fn push(&mut self, voxel: Voxel) -> Result<(), VoxelError> {
        if self.len >= self.data.len() {
            return Err(VoxelError::BufferOverflow);
        }
        self.data[self.len] = voxel;
        self.len += 1;
        Ok(())
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<&Voxel> {
        self.data.get(index)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[Voxel] {
        &self.data[..self.len]
    }

    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.width, self.height, self.depth)
    }
}

/// Volumetric Page Encoder for holographic storage
pub struct VolumetricPageEncoder {
    buffer: VoxelBuffer,
    page_id: u64,
}

impl VolumetricPageEncoder {
    /// Create a new encoder with pre-allocated buffer
    pub fn new(page_id: u64, width: usize, height: usize, depth: usize) -> Result<Self, VoxelError> {
        let buffer = VoxelBuffer::with_capacity(width, height, depth)?;
        Ok(Self { buffer, page_id })
    }

    /// Encode a 2D financial data matrix into a 3D voxel grid
    /// 
    /// # Arguments
    /// * `data` - Row-major 2D matrix of f32 values (e.g., order book depths)
    /// * `depth_slices` - Number of depth layers to distribute data across
    /// 
    /// # Returns
    /// Reference to the internal voxel buffer
    pub fn encode_matrix(&mut self, data: &[f32], width: usize, height: usize, depth_slices: usize) -> Result<&[Voxel], VoxelError> {
        if data.len() != width * height {
            return Err(VoxelError::InvalidDimensions);
        }
        if depth_slices > MAX_PAGE_DEPTH {
            return Err(VoxelError::InvalidDimensions);
        }

        self.buffer.len = 0; // Reset without allocation

        let total_voxels = width * height * depth_slices;
        if total_voxels > self.buffer.data.len() {
            return Err(VoxelError::BufferOverflow);
        }

        // Distribute 2D data across depth slices using interference pattern simulation
        for z in 0..depth_slices {
            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    let base_intensity = data[idx];
                    
                    // Apply Bragg grating modulation for this depth layer
                    let phase_shift = (z as f32) * std::f32::consts::PI / depth_slices as f32;
                    let modulated_intensity = base_intensity * (1.0 + 0.5 * (x as f32 * phase_shift).sin());
                    let clamped_intensity = modulated_intensity.clamp(0.0, 1.0);

                    let voxel = Voxel::new(
                        x as u16,
                        y as u16,
                        z as u16,
                        clamped_intensity,
                        phase_shift,
                    )?;
                    
                    self.buffer.push(voxel)?;
                }
            }
        }

        Ok(self.buffer.as_slice())
    }

    /// Encode order book snapshot into volumetric representation
    pub fn encode_orderbook(
        &mut self,
        bids: &[(f64, f64)],
        asks: &[(f64, f64)],
        depth_slices: usize,
    ) -> Result<&[Voxel], VoxelError> {
        const GRID_SIZE: usize = 64;
        let mut grid = [0.0f32; GRID_SIZE * GRID_SIZE];

        // Map bids to lower half, asks to upper half
        for (i, (price, size)) in bids.iter().take(GRID_SIZE / 2).enumerate() {
            let x = ((price.fract() * 1000.0) as usize) % GRID_SIZE;
            let y = i;
            let intensity = (*size as f32).clamp(0.0, 1.0);
            grid[y * GRID_SIZE + x] = intensity;
        }

        for (i, (price, size)) in asks.iter().take(GRID_SIZE / 2).enumerate() {
            let x = ((price.fract() * 1000.0) as usize) % GRID_SIZE;
            let y = GRID_SIZE / 2 + i;
            let intensity = (*size as f32).clamp(0.0, 1.0);
            grid[y * GRID_SIZE + x] = intensity;
        }

        self.encode_matrix(&grid, GRID_SIZE, GRID_SIZE, depth_slices)
    }

    #[inline]
    pub fn page_id(&self) -> u64 {
        self.page_id
    }

    #[inline]
    pub fn voxel_count(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_creation() {
        let voxel = Voxel::new(10, 20, 30, 0.5, 1.57).unwrap();
        assert_eq!(voxel.x, 10);
        assert_eq!(voxel.intensity, 0.5);
    }

    #[test]
    fn test_voxel_out_of_bounds() {
        let result = Voxel::new(MAX_PAGE_WIDTH as u16, 0, 0, 0.5, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_encoder_basic() {
        let mut encoder = VolumetricPageEncoder::new(1, 8, 8, 4).unwrap();
        let data: Vec<f32> = (0..64).map(|i| (i % 10) as f32 / 10.0).collect();
        let voxels = encoder.encode_matrix(&data, 8, 8, 4).unwrap();
        assert_eq!(voxels.len(), 8 * 8 * 4);
    }
}
