//! Nano-Grating Orientation Module
//! 
//! Manages the orientation and retardance properties of nano-gratings in 5D optical storage.
//! Uses Jones calculus for precise polarization control.

use thiserror::Error;

/// Number of discrete orientation levels
pub const ORIENTATION_LEVELS: usize = 16;

/// Number of discrete retardance levels  
pub const RETARDANCE_LEVELS: usize = 16;

#[derive(Error, Debug)]
pub enum OrientationError {
    #[error("Invalid orientation angle: {0} radians")]
    InvalidAngle(f64),
    #[error("Invalid retardance value: {0}")]
    InvalidRetardance(f64),
    #[error("Quantization error")]
    QuantizationError,
}

/// Represents a nano-grating's optical properties
#[derive(Debug, Clone, Copy)]
pub struct NanoGratingOrientation {
    /// Fast axis orientation angle (radians, 0 to π)
    pub orientation_rad: f64,
    /// Normalized retardance (0 to 1)
    pub retardance: f64,
    /// Discrete level for orientation (0 to ORIENTATION_LEVELS-1)
    pub orient_level: u8,
    /// Discrete level for retardance (0 to RETARDANCE_LEVELS-1)
    pub retard_level: u8,
}

impl NanoGratingOrientation {
    /// Create a new nano-grating with continuous values
    pub fn new(orientation_rad: f64, retardance: f64) -> Result<Self, OrientationError> {
        if orientation_rad < 0.0 || orientation_rad > std::f64::consts::PI {
            return Err(OrientationError::InvalidAngle(orientation_rad));
        }
        if retardance < 0.0 || retardance > 1.0 {
            return Err(OrientationError::InvalidRetardance(retardance));
        }

        let orient_level = Self::quantize_orientation(orientation_rad)?;
        let retard_level = Self::quantize_retardance(retardance)?;

        Ok(Self {
            orientation_rad,
            retardance,
            orient_level,
            retard_level,
        })
    }

    /// Create from discrete levels
    pub fn from_levels(orient_level: u8, retard_level: u8) -> Result<Self, OrientationError> {
        if orient_level >= ORIENTATION_LEVELS as u8 {
            return Err(OrientationError::QuantizationError);
        }
        if retard_level >= RETARDANCE_LEVELS as u8 {
            return Err(OrientationError::QuantizationError);
        }

        let orientation_rad = (orient_level as f64 / (ORIENTATION_LEVELS - 1) as f64) * std::f64::consts::PI;
        let retardance = retard_level as f64 / (RETARDANCE_LEVELS - 1) as f64;

        Ok(Self {
            orientation_rad,
            retardance,
            orient_level,
            retard_level,
        })
    }

    /// Quantize continuous orientation to discrete level
    fn quantize_orientation(orientation_rad: f64) -> Result<u8, OrientationError> {
        let level = ((orientation_rad / std::f64::consts::PI) * (ORIENTATION_LEVELS - 1) as f64).round() as u8;
        if level >= ORIENTATION_LEVELS as u8 {
            return Err(OrientationError::QuantizationError);
        }
        Ok(level)
    }

    /// Quantize continuous retardance to discrete level
    fn quantize_retardance(retardance: f64) -> Result<u8, OrientationError> {
        let level = (retardance * (RETARDANCE_LEVELS - 1) as f64).round() as u8;
        if level >= RETARDANCE_LEVELS as u8 {
            return Err(OrientationError::QuantizationError);
        }
        Ok(level)
    }

    /// Get de-quantized orientation angle
    pub fn dequantize_orientation(level: u8) -> Result<f64, OrientationError> {
        if level >= ORIENTATION_LEVELS as u8 {
            return Err(OrientationError::QuantizationError);
        }
        Ok((level as f64 / (ORIENTATION_LEVELS - 1) as f64) * std::f64::consts::PI)
    }

    /// Get de-quantized retardance
    pub fn dequantize_retardance(level: u8) -> Result<f64, OrientationError> {
        if level >= RETARDANCE_LEVELS as u8 {
            return Err(OrientationError::QuantizationError);
        }
        Ok(level as f64 / (RETARDANCE_LEVELS - 1) as f64)
    }

    /// Encode both levels into a single byte
    pub fn encode_to_byte(&self) -> u8 {
        (self.orient_level << 4) | self.retard_level
    }

    /// Decode from a single byte
    pub fn decode_from_byte(byte: u8) -> Result<Self, OrientationError> {
        let orient_level = (byte >> 4) & 0x0F;
        let retard_level = byte & 0x0F;
        Self::from_levels(orient_level, retard_level)
    }

    /// Calculate birefringence effect on polarized light
    pub fn calculate_jones_matrix(&self) -> [[f64; 2]; 2] {
        let theta = self.orientation_rad;
        let delta = self.retardance * std::f64::consts::PI; // Phase delay

        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let cos_delta = delta.cos();
        let sin_delta = delta.sin();

        // Jones matrix for linear retarder
        [
            [
                cos_t * cos_t + sin_t * sin_t * complex_exp(-delta),
                cos_t * sin_t * (1.0 - complex_exp(-delta)),
            ],
            [
                cos_t * sin_t * (1.0 - complex_exp(-delta)),
                sin_t * sin_t + cos_t * cos_t * complex_exp(-delta),
            ],
        ]
    }
}

/// Helper function for complex exponential (real part only for intensity)
fn complex_exp(phase: f64) -> f64 {
    phase.cos() // Real part of e^(i*phase)
}

/// Orientation mapper for batch processing
pub struct OrientationMapper {
    orientations: Box<[NanoGratingOrientation]>,
    len: usize,
    capacity: usize,
}

impl OrientationMapper {
    /// Create a new mapper with pre-allocated buffer
    pub fn with_capacity(capacity: usize) -> Self {
        let default_orient = NanoGratingOrientation {
            orientation_rad: 0.0,
            retardance: 0.0,
            orient_level: 0,
            retard_level: 0,
        };
        let orientations = vec![default_orient; capacity].into_boxed_slice();
        Self { orientations, len: 0, capacity }
    }

    /// Add an orientation to the buffer
    pub fn push(&mut self, orient: NanoGratingOrientation) -> Result<(), OrientationError> {
        if self.len >= self.capacity {
            return Err(OrientationError::QuantizationError);
        }
        self.orientations[self.len] = orient;
        self.len += 1;
        Ok(())
    }

    /// Encode a byte array into orientations
    pub fn encode_bytes(&mut self, data: &[u8]) -> Result<usize, OrientationError> {
        self.len = 0;
        
        for &byte in data {
            let orient = NanoGratingOrientation::decode_from_byte(byte)?;
            self.push(orient)?;
        }
        
        Ok(self.len)
    }

    /// Decode orientations back to bytes
    pub fn decode_to_bytes(&self) -> Vec<u8> {
        self.orientations[..self.len]
            .iter()
            .map(|o| o.encode_to_byte())
            .collect()
    }

    /// Get orientation at index
    pub fn get(&self, index: usize) -> Option<&NanoGratingOrientation> {
        self.orientations.get(index)
    }

    /// Get all orientations as slice
    pub fn as_slice(&self) -> &[NanoGratingOrientation] {
        &self.orientations[..self.len]
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Get count
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orientation_creation() {
        let orient = NanoGratingOrientation::new(std::f64::consts::PI / 4.0, 0.5).unwrap();
        assert!(orient.orientation_rad > 0.7);
        assert_eq!(orient.retardance, 0.5);
    }

    #[test]
    fn test_level_roundtrip() {
        let original = NanoGratingOrientation::from_levels(5, 10).unwrap();
        let encoded = original.encode_to_byte();
        let decoded = NanoGratingOrientation::decode_from_byte(encoded).unwrap();
        
        assert_eq!(original.orient_level, decoded.orient_level);
        assert_eq!(original.retard_level, decoded.retard_level);
    }

    #[test]
    fn test_byte_roundtrip() {
        let byte = 0xAB;
        let orient = NanoGratingOrientation::decode_from_byte(byte).unwrap();
        let encoded = orient.encode_to_byte();
        assert_eq!(byte, encoded);
    }

    #[test]
    fn test_jones_matrix() {
        let orient = NanoGratingOrientation::new(std::f64::consts::PI / 4.0, 0.5).unwrap();
        let matrix = orient.calculate_jones_matrix();
        
        // Matrix should be 2x2
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
    }

    #[test]
    fn test_mapper_encode_decode() {
        let mut mapper = OrientationMapper::with_capacity(256);
        let data = vec![0x00, 0x55, 0xAA, 0xFF];
        
        let count = mapper.encode_bytes(&data).unwrap();
        assert_eq!(count, data.len());
        
        let decoded = mapper.decode_to_bytes();
        assert_eq!(data, decoded);
    }
}
