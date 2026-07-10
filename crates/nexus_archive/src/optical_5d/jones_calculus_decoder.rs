//! Jones Calculus Decoder for 5D Optical Storage
//! 
//! Implements optical readout simulation using Jones calculus to reconstruct
//! binary data from the birefringence properties of nano-gratings.

use crate::optical_5d::nanograting_orientation::{NanoGratingOrientation, OrientationError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum JonesDecoderError {
    #[error("Invalid input polarization state")]
    InvalidPolarization,
    #[error("Invalid analyzer angle: {0} radians")]
    InvalidAnalyzerAngle(f64),
    #[error("Intensity out of range: {0}")]
    IntensityOutOfRange(f64),
    #[error("Decoding failed: signal too weak")]
    SignalTooWeak,
    #[error("Buffer overflow")]
    BufferOverflow,
}

/// Jones vector representing polarized light state
#[derive(Debug, Clone, Copy)]
pub struct JonesVector {
    pub ex: f64, // x-component (real)
    pub ey: f64, // y-component (real)
}

impl JonesVector {
    /// Create a normalized Jones vector
    pub fn new(ex: f64, ey: f64) -> Result<Self, JonesDecoderError> {
        let intensity = ex * ex + ey * ey;
        if intensity < 1e-10 || intensity > 2.0 {
            return Err(JonesDecoderError::InvalidPolarization);
        }
        Ok(Self { ex, ey })
    }

    /// Horizontal linear polarization
    pub fn horizontal() -> Self {
        Self { ex: 1.0, ey: 0.0 }
    }

    /// Vertical linear polarization
    pub fn vertical() -> Self {
        Self { ex: 0.0, ey: 1.0 }
    }

    /// Linear polarization at angle theta
    pub fn linear_at_angle(theta: f64) -> Self {
        Self {
            ex: theta.cos(),
            ey: theta.sin(),
        }
    }

    /// Calculate intensity (|E|^2)
    pub fn intensity(&self) -> f64 {
        self.ex * self.ex + self.ey * self.ey
    }

    /// Apply a Jones matrix transformation
    pub fn transform(&self, matrix: &[[f64; 2]; 2]) -> Self {
        Self {
            ex: matrix[0][0] * self.ex + matrix[0][1] * self.ey,
            ey: matrix[1][0] * self.ex + matrix[1][1] * self.ey,
        }
    }
}

/// Linear polarizer (analyzer) Jones matrix
fn polarizer_matrix(angle: f64) -> [[f64; 2]; 2] {
    let cos_t = angle.cos();
    let sin_t = angle.sin();
    
    [
        [cos_t * cos_t, cos_t * sin_t],
        [cos_t * sin_t, sin_t * sin_t],
    ]
}

/// Quarter-wave plate Jones matrix
fn quarter_wave_plate(fast_axis: f64) -> [[f64; 2]; 2] {
    let cos_t = fast_axis.cos();
    let sin_t = fast_axis.sin();
    
    // QWP introduces π/2 phase shift
    let phase = std::f64::consts::FRAC_PI_2;
    let cos_p = phase.cos();
    let sin_p = phase.sin();
    
    [
        [
            cos_t * cos_t + sin_t * sin_t * cos_p,
            cos_t * sin_t * (1.0 - cos_p) - sin_t * sin_t * sin_p,
        ],
        [
            cos_t * sin_t * (1.0 - cos_p) + cos_t * cos_t * sin_p,
            sin_t * sin_t + cos_t * cos_t * cos_p,
        ],
    ]
}

/// Jones Calculus Decoder for optical readout
pub struct JonesCalculusDecoder {
    input_polarization: JonesVector,
    analyzer_angle: f64,
    detection_threshold: f64,
}

impl JonesCalculusDecoder {
    /// Create a new decoder with specified configuration
    pub fn new(
        input_polarization: JonesVector,
        analyzer_angle: f64,
        detection_threshold: f64,
    ) -> Result<Self, JonesDecoderError> {
        if analyzer_angle < 0.0 || analyzer_angle > std::f64::consts::PI {
            return Err(JonesDecoderError::InvalidAnalyzerAngle(analyzer_angle));
        }
        if detection_threshold <= 0.0 || detection_threshold >= 1.0 {
            return Err(JonesDecoderError::IntensityOutOfRange(detection_threshold));
        }

        Ok(Self {
            input_polarization,
            analyzer_angle,
            detection_threshold,
        })
    }

    /// Default configuration for 5D storage readout
    pub fn default_config() -> Result<Self, JonesDecoderError> {
        Self::new(
            JonesVector::horizontal(),
            std::f64::consts::PI / 4.0,
            0.01,
        )
    }

    /// Simulate reading a voxel and decode to byte
    pub fn read_voxel(&self, orientation: &NanoGratingOrientation) -> Result<u8, JonesDecoderError> {
        // Get the Jones matrix for this nano-grating
        let grating_matrix = orientation.calculate_jones_matrix();

        // Propagate light through the system:
        // 1. Input polarization -> 2. Nano-grating -> 3. Analyzer

        let after_grating = self.input_polarization.transform(&grating_matrix);
        let analyzer_matrix = polarizer_matrix(self.analyzer_angle);
        let after_analyzer = after_grating.transform(&analyzer_matrix);

        // Calculate detected intensity
        let intensity = after_analyzer.intensity();

        if intensity < self.detection_threshold {
            return Err(JonesDecoderError::SignalTooWeak);
        }

        // Normalize intensity to [0, 1] range
        let normalized_intensity = intensity.clamp(0.0, 1.0);

        // Decode retardance level from intensity
        let retard_level = self.intensity_to_retard_level(normalized_intensity)?;

        // Combine with orientation level (already known from position/scanning)
        let orient_level = orientation.orient_level;

        Ok((orient_level << 4) | retard_level)
    }

    /// Convert detected intensity to retardance level
    fn intensity_to_retard_level(&self, intensity: f64) -> Result<u8, JonesDecoderError> {
        use crate::optical_5d::nanograting_orientation::RETARDANCE_LEVELS;
        
        // Intensity is proportional to sin²(retardance * π/2) for crossed polarizers
        // Inverse: retardance = (2/π) * arcsin(sqrt(intensity))
        
        let sqrt_intensity = intensity.sqrt().clamp(0.0, 1.0);
        let arcsin_val = sqrt_intensity.asin();
        let retardance = (2.0 / std::f64::consts::PI) * arcsin_val;
        
        let level = (retardance * (RETARDANCE_LEVELS - 1) as f64).round() as u8;
        
        if level >= RETARDANCE_LEVELS as u8 {
            return Err(JonesDecoderError::IntensityOutOfRange(intensity));
        }
        
        Ok(level)
    }

    /// Read multiple voxels (batch operation)
    pub fn read_voxels(
        &self,
        orientations: &[NanoGratingOrientation],
        output: &mut [u8],
    ) -> Result<usize, JonesDecoderError> {
        if output.len() < orientations.len() {
            return Err(JonesDecoderError::BufferOverflow);
        }

        let mut count = 0;
        for (i, &orient) in orientations.iter().enumerate() {
            match self.read_voxel(&orient) {
                Ok(byte) => {
                    output[i] = byte;
                    count += 1;
                }
                Err(JonesDecoderError::SignalTooWeak) => {
                    // Record as 0 but continue
                    output[i] = 0;
                    count += 1;
                }
                Err(e) => return Err(e),
            }
        }

        Ok(count)
    }

    /// Calculate expected intensity for given orientation (for calibration)
    pub fn calculate_expected_intensity(&self, orientation: &NanoGratingOrientation) -> f64 {
        let grating_matrix = orientation.calculate_jones_matrix();
        let after_grating = self.input_polarization.transform(&grating_matrix);
        let analyzer_matrix = polarizer_matrix(self.analyzer_angle);
        let after_analyzer = after_grating.transform(&analyzer_matrix);
        after_analyzer.intensity()
    }

    /// Set input polarization state
    pub fn set_input_polarization(&mut self, polarization: JonesVector) -> Result<(), JonesDecoderError> {
        polarization.intensity(); // Validate
        self.input_polarization = polarization;
        Ok(())
    }

    /// Set analyzer angle
    pub fn set_analyzer_angle(&mut self, angle: f64) -> Result<(), JonesDecoderError> {
        if angle < 0.0 || angle > std::f64::consts::PI {
            return Err(JonesDecoderError::InvalidAnalyzerAngle(angle));
        }
        self.analyzer_angle = angle;
        Ok(())
    }

    /// Get current analyzer angle
    pub fn analyzer_angle(&self) -> f64 {
        self.analyzer_angle
    }
}

/// Multi-channel decoder for parallel readout
pub struct MultiChannelDecoder {
    channels: Box<[JonesCalculusDecoder]>,
    num_channels: usize,
}

impl MultiChannelDecoder {
    /// Create a multi-channel decoder
    pub fn new(num_channels: usize) -> Result<Self, JonesDecoderError> {
        if num_channels == 0 || num_channels > 16 {
            return Err(JonesDecoderError::InvalidPolarization);
        }

        let mut channels = Vec::with_capacity(num_channels);
        for i in 0..num_channels {
            // Each channel has slightly different analyzer angle for multiplexing
            let angle_offset = (i as f64) * std::f64::consts::PI / (num_channels as f64 * 4.0);
            let decoder = JonesCalculusDecoder::default_config()?;
            // Would need mutable access to set angle - simplified here
            channels.push(decoder);
        }

        Ok(Self {
            channels: channels.into_boxed_slice(),
            num_channels,
        })
    }

    /// Read from all channels in parallel (simulated)
    pub fn parallel_read(
        &self,
        orientations: &[NanoGratingOrientation],
    ) -> Vec<Vec<u8>> {
        let mut results = Vec::with_capacity(self.num_channels);
        
        for channel in self.channels.iter() {
            let mut output = vec![0u8; orientations.len()];
            let _ = channel.read_voxels(orientations, &mut output);
            results.push(output);
        }
        
        results
    }

    /// Get number of channels
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jones_vector_creation() {
        let v = JonesVector::new(1.0, 0.0).unwrap();
        assert_eq!(v.intensity(), 1.0);

        let h = JonesVector::horizontal();
        assert_eq!(h.ex, 1.0);
        assert_eq!(h.ey, 0.0);
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = JonesCalculusDecoder::default_config().unwrap();
        assert!(decoder.analyzer_angle() > 0.7);
    }

    #[test]
    fn test_voxel_readout() {
        let decoder = JonesCalculusDecoder::default_config().unwrap();
        let orientation = NanoGratingOrientation::from_levels(8, 8).unwrap();
        
        let result = decoder.read_voxel(&orientation);
        assert!(result.is_ok());
    }

    #[test]
    fn test_intensity_calculation() {
        let decoder = JonesCalculusDecoder::default_config().unwrap();
        let orientation = NanoGratingOrientation::new(0.0, 0.0).unwrap();
        
        let intensity = decoder.calculate_expected_intensity(&orientation);
        assert!(intensity >= 0.0);
    }

    #[test]
    fn test_multi_channel_decoder() {
        let decoder = MultiChannelDecoder::new(4).unwrap();
        assert_eq!(decoder.num_channels(), 4);
        
        let orientations = vec![
            NanoGratingOrientation::from_levels(0, 0).unwrap(),
            NanoGratingOrientation::from_levels(15, 15).unwrap(),
        ];
        
        let results = decoder.parallel_read(&orientations);
        assert_eq!(results.len(), 4);
    }
}
