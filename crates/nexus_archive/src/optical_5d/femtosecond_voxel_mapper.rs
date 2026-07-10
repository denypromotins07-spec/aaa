//! Femtosecond Voxel Mapper for 5D Optical Storage
//! 
//! Calculates laser pulse parameters for writing data into fused silica glass.
//! Each voxel encodes data in 3 spatial dimensions + orientation + retardance.

use thiserror::Error;

/// Minimum voxel spacing to prevent interference (micrometers)
pub const MIN_VOXEL_SPACING_UM: f64 = 1.0;

/// Maximum retardance value (normalized)
pub const MAX_RETARDANCE: f64 = 1.0;

/// Minimum retardance to ensure detectability
pub const MIN_RETARDANCE: f64 = 0.01;

/// Wavelength of femtosecond laser (nm) - typical Ti:Sapphire
pub const LASER_WAVELENGTH_NM: f64 = 800.0;

/// Pulse duration (femtoseconds)
pub const PULSE_DURATION_FS: f64 = 150.0;

#[derive(Error, Debug)]
pub enum VoxelMapperError {
    #[error("Invalid coordinate: out of bounds")]
    CoordinateOutOfBounds,
    #[error("Invalid orientation angle: {0} radians")]
    InvalidOrientation(f64),
    #[error("Invalid retardance: {0}")]
    InvalidRetardance(f64),
    #[error("Pulse energy out of range: {0} μJ")]
    PulseEnergyOutOfRange(f64),
    #[error("Voxel collision detected")]
    VoxelCollision,
    #[error("Buffer overflow")]
    BufferOverflow,
}

/// 5D voxel representation
#[derive(Debug, Clone, Copy)]
pub struct Voxel5D {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub orientation_rad: f64, // 0 to π
    pub retardance: f64,      // 0 to 1 (normalized)
}

impl Voxel5D {
    pub fn new(
        x: u32,
        y: u32,
        z: u32,
        orientation_rad: f64,
        retardance: f64,
    ) -> Result<Self, VoxelMapperError> {
        if orientation_rad < 0.0 || orientation_rad > std::f64::consts::PI {
            return Err(VoxelMapperError::InvalidOrientation(orientation_rad));
        }
        if retardance < MIN_RETARDANCE || retardance > MAX_RETARDANCE {
            return Err(VoxelMapperError::InvalidRetardance(retardance));
        }

        Ok(Self {
            x,
            y,
            z,
            orientation_rad,
            retardance,
        })
    }

    /// Encode 8 bits into orientation and retardance
    pub fn from_bits(bits: u8, x: u32, y: u32, z: u32) -> Result<Self, VoxelMapperError> {
        // Split 8 bits: 4 for orientation (16 levels), 4 for retardance (16 levels)
        let orient_bits = (bits >> 4) & 0x0F;
        let retard_bits = bits & 0x0F;

        // Map to physical values
        let orientation_rad = (orient_bits as f64 / 15.0) * std::f64::consts::PI;
        let retardance = MIN_RETARDANCE + (retard_bits as f64 / 15.0) * (MAX_RETARDANCE - MIN_RETARDANCE);

        Self::new(x, y, z, orientation_rad, retardance)
    }

    /// Decode 8 bits from voxel properties
    pub fn to_bits(&self) -> u8 {
        let orient_bits = ((self.orientation_rad / std::f64::consts::PI) * 15.0).round() as u8 & 0x0F;
        let retard_bits = (((self.retardance - MIN_RETARDANCE) / (MAX_RETARDANCE - MIN_RETARDANCE)) * 15.0).round() as u8 & 0x0F;
        (orient_bits << 4) | retard_bits
    }
}

/// Laser pulse parameters for writing a voxel
#[derive(Debug, Clone, Copy)]
pub struct LaserPulseParams {
    pub pulse_energy_uj: f64,
    pub repetition_rate_khz: f64,
    pub polarization_angle_rad: f64,
    pub focus_depth_um: f64,
    pub exposure_time_ms: f64,
}

impl LaserPulseParams {
    /// Calculate optimal pulse energy for given retardance
    pub fn calculate_pulse_energy(retardance: f64, material_threshold_uj: f64) -> Result<f64, VoxelMapperError> {
        // Empirical relationship between pulse energy and induced retardance
        let energy = material_threshold_uj * (1.0 + retardance * 2.0);
        
        if energy < material_threshold_uj || energy > material_threshold_uj * 5.0 {
            return Err(VoxelMapperError::PulseEnergyOutOfRange(energy));
        }
        
        Ok(energy)
    }

    /// Calculate polarization angle from desired orientation
    pub fn polarization_from_orientation(orientation_rad: f64) -> f64 {
        // Polarization angle directly controls nano-grating orientation
        orientation_rad
    }
}

/// Pre-allocated buffer for voxel data
pub struct VoxelBuffer5D {
    data: Box<[Voxel5D]>,
    len: usize,
    capacity: usize,
}

impl VoxelBuffer5D {
    pub fn with_capacity(capacity: usize) -> Self {
        let default_voxel = Voxel5D {
            x: 0, y: 0, z: 0,
            orientation_rad: 0.0,
            retardance: MIN_RETARDANCE,
        };
        let data = vec![default_voxel; capacity].into_boxed_slice();
        Self { data, len: 0, capacity }
    }

    #[inline]
    pub fn push(&mut self, voxel: Voxel5D) -> Result<(), VoxelMapperError> {
        if self.len >= self.capacity {
            return Err(VoxelMapperError::BufferOverflow);
        }
        self.data[self.len] = voxel;
        self.len += 1;
        Ok(())
    }

    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn as_slice(&self) -> &[Voxel5D] {
        &self.data[..self.len]
    }
}

/// Femtosecond Voxel Mapper for 5D optical storage
pub struct FemtosecondVoxelMapper {
    buffer: VoxelBuffer5D,
    volume_width: u32,
    volume_height: u32,
    volume_depth: u32,
    material_threshold_uj: f64,
    voxel_spacing_um: f64,
}

impl FemtosecondVoxelMapper {
    /// Create a new mapper with specified volume dimensions
    pub fn new(
        width: u32,
        height: u32,
        depth: u32,
        voxel_spacing_um: f64,
        material_threshold_uj: f64,
    ) -> Result<Self, VoxelMapperError> {
        if voxel_spacing_um < MIN_VOXEL_SPACING_UM {
            return Err(VoxelMapperError::VoxelCollision);
        }

        let capacity = (width * height * depth) as usize;
        let buffer = VoxelBuffer5D::with_capacity(capacity.min(10_000_000));

        Ok(Self {
            buffer,
            volume_width: width,
            volume_height: height,
            volume_depth: depth,
            material_threshold_uj,
            voxel_spacing_um,
        })
    }

    /// Write a single byte as a 5D voxel
    pub fn write_byte(
        &mut self,
        byte: u8,
        x: u32,
        y: u32,
        z: u32,
    ) -> Result<LaserPulseParams, VoxelMapperError> {
        // Validate coordinates
        if x >= self.volume_width || y >= self.volume_height || z >= self.volume_depth {
            return Err(VoxelMapperError::CoordinateOutOfBounds);
        }

        // Create voxel from byte
        let voxel = Voxel5D::from_bits(byte, x, y, z)?;

        // Calculate laser parameters
        let pulse_energy = LaserPulseParams::calculate_pulse_energy(
            voxel.retardance,
            self.material_threshold_uj,
        )?;

        let params = LaserPulseParams {
            pulse_energy_uj: pulse_energy,
            repetition_rate_khz: 100.0, // Default 100 kHz
            polarization_angle_rad: LaserPulseParams::polarization_from_orientation(voxel.orientation_rad),
            focus_depth_um: z as f64 * self.voxel_spacing_um,
            exposure_time_ms: 1.0,
        };

        self.buffer.push(voxel)?;

        Ok(params)
    }

    /// Write a block of bytes as multiple voxels
    pub fn write_block(
        &mut self,
        data: &[u8],
        start_x: u32,
        start_y: u32,
        start_z: u32,
    ) -> Result<Vec<LaserPulseParams>, VoxelMapperError> {
        let mut params_vec = Vec::with_capacity(data.len());
        let mut x = start_x;
        let mut y = start_y;
        let z = start_z;

        for &byte in data {
            match self.write_byte(byte, x, y, z) {
                Ok(params) => {
                    params_vec.push(params);
                    
                    // Increment position (row-major order)
                    x += 1;
                    if x >= self.volume_width {
                        x = start_x;
                        y += 1;
                        if y >= self.volume_height {
                            y = start_y;
                            // z increment would go here for multi-layer
                        }
                    }
                }
                Err(e) => return Err(e),
            }
        }

        Ok(params_vec)
    }

    /// Read a byte from a voxel position (simulated)
    pub fn read_byte(&self, x: u32, y: u32, z: u32) -> Result<u8, VoxelMapperError> {
        if x >= self.volume_width || y >= self.volume_height || z >= self.volume_depth {
            return Err(VoxelMapperError::CoordinateOutOfBounds);
        }

        // Search buffer for voxel at this position
        for voxel in self.buffer.as_slice() {
            if voxel.x == x && voxel.y == y && voxel.z == z {
                return Ok(voxel.to_bits());
            }
        }

        // Return 0 if not found (would be actual read in hardware)
        Ok(0)
    }

    /// Read a block of bytes
    pub fn read_block(
        &self,
        start_x: u32,
        start_y: u32,
        start_z: u32,
        length: usize,
    ) -> Result<Vec<u8>, VoxelMapperError> {
        let mut data = Vec::with_capacity(length);
        let mut x = start_x;
        let mut y = start_y;
        let z = start_z;

        for _ in 0..length {
            let byte = self.read_byte(x, y, z)?;
            data.push(byte);

            x += 1;
            if x >= self.volume_width {
                x = start_x;
                y += 1;
            }
        }

        Ok(data)
    }

    /// Get the number of voxels written
    #[inline]
    pub fn voxels_written(&self) -> usize {
        self.buffer.len()
    }

    /// Clear the buffer
    #[inline]
    pub fn clear_buffer(&mut self) {
        self.buffer.clear();
    }

    /// Get volume dimensions
    pub fn volume_dimensions(&self) -> (u32, u32, u32) {
        (self.volume_width, self.volume_height, self.volume_depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_creation() {
        let voxel = Voxel5D::new(10, 20, 30, std::f64::consts::PI / 4.0, 0.5).unwrap();
        assert_eq!(voxel.x, 10);
        assert!(voxel.orientation_rad > 0.7);
    }

    #[test]
    fn test_byte_roundtrip() {
        let voxel = Voxel5D::from_bits(0xAB, 0, 0, 0).unwrap();
        let decoded = voxel.to_bits();
        assert_eq!(decoded, 0xAB);
    }

    #[test]
    fn test_mapper_write_read() {
        let mut mapper = FemtosecondVoxelMapper::new(100, 100, 10, 2.0, 0.5).unwrap();
        
        let data = vec![0x01, 0x23, 0x45, 0x67, 0x89];
        let _params = mapper.write_block(&data, 0, 0, 0).unwrap();
        
        let read_data = mapper.read_block(0, 0, 0, data.len()).unwrap();
        assert_eq!(data, read_data);
    }

    #[test]
    fn test_coordinate_bounds() {
        let mut mapper = FemtosecondVoxelMapper::new(10, 10, 5, 2.0, 0.5).unwrap();
        
        let result = mapper.write_byte(0xFF, 10, 0, 0); // Out of bounds
        assert!(result.is_err());
    }
}
