//! Nicoll-Dyson Phased Array Beam Simulator
//! 
//! Implements phased-array laser/microwave transmission for interstellar
//! energy beaming to Von Neumann probes and light-sail spacecraft.

use nalgebra::{SVector, Vector3};
use num_traits::{Float, Zero};
use thiserror::Error;

/// Beam array configuration parameters
#[derive(Clone, Debug)]
pub struct PhasedArrayConfig<T> {
    pub wavelength: T,           // Transmission wavelength (m)
    pub element_spacing: T,      // Spacing between array elements (m)
    pub element_count: u32,      // Number of elements per dimension
    pub total_elements: u32,
    pub element_diameter: T,     // Diameter of each element (m)
    pub max_power_per_element: T, // W
}

impl<T: Float + Copy + Zero> PhasedArrayConfig<T> {
    pub fn new(
        wavelength: T,
        element_spacing: T,
        elements_per_side: u32,
        element_diameter: T,
        max_power: T,
    ) -> Self {
        let total = elements_per_side * elements_per_side;
        Self {
            wavelength,
            element_spacing,
            element_count: elements_per_side,
            total_elements: total,
            element_diameter,
            max_power_per_element: max_power,
        }
    }
    
    /// Total array aperture diameter
    pub fn aperture_diameter(&self) -> T {
        let n = T::from(self.element_count as f64).unwrap();
        self.element_spacing * n
    }
    
    /// Maximum total power output
    pub fn max_total_power(&self) -> T {
        self.max_power_per_element * T::from(self.total_elements as f64).unwrap()
    }
}

/// Beam steering parameters
#[derive(Clone, Debug)]
pub struct BeamSteering<T> {
    pub azimuth: T,   // Horizontal angle (radians)
    pub elevation: T, // Vertical angle (radians)
    pub phase_offsets: Vec<T>,  // Per-element phase corrections
}

/// Errors in beam simulation
#[derive(Error, Debug)]
pub enum BeamError {
    #[error("Beam divergence exceeds target at distance {distance:?} m")]
    ExcessiveDivergence { distance: f64, spot_size: f64 },
    #[error("Phase error detected: rms_error={rms_error:?} radians")]
    PhaseError { rms_error: f64 },
    #[error("Pointing error exceeds tolerance: error={error:?} radians")]
    PointingError { error: f64 },
    #[error("Atmospheric attenuation too high: loss={loss_db:?} dB")]
    AtmosphericLoss { loss_db: f64 },
}

/// Nicoll-Dyson beam array simulator
pub struct NicollDysonArray<T> {
    config: PhasedArrayConfig<T>,
    position: SVector<T, 3>,
    steering: BeamSteering<T>,
}

impl<T: Float + Copy + Zero> NicollDysonArray<T> {
    pub fn new(config: PhasedArrayConfig<T>, position: SVector<T, 3>) -> Self {
        let zero = T::zero();
        Self {
            config,
            position,
            steering: BeamSteering {
                azimuth: zero,
                elevation: zero,
                phase_offsets: vec![zero; config.total_elements as usize],
            },
        }
    }
    
    /// Calculate diffraction-limited beam divergence angle
    /// θ ≈ 1.22 λ / D (radians)
    pub fn calculate_divergence_angle(&self) -> T {
        let one_point_22 = T::from(1.22).unwrap();
        let aperture = self.config.aperture_diameter();
        
        if aperture <= T::zero() {
            return T::one();  // Maximum divergence as fallback
        }
        
        one_point_22 * self.config.wavelength / aperture
    }
    
    /// Calculate beam spot size at target distance
    /// Spot radius ≈ θ * L (for small angles)
    pub fn calculate_spot_size(&self, distance: T) -> T {
        let divergence = self.calculate_divergence_angle();
        distance * divergence
    }
    
    /// Calculate power density at target (W/m²)
    pub fn calculate_power_density(&self, distance: T, total_power: T) -> Result<T, BeamError> {
        let spot_radius = self.calculate_spot_size(distance);
        
        if spot_radius <= T::zero() {
            return Err(BeamError::ExcessiveDivergence {
                distance: distance.to_f64().unwrap_or(0.0),
                spot_size: 0.0,
            });
        }
        
        // Assume Gaussian beam profile
        let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        let two = T::one() + T::one();
        
        let spot_area = pi * spot_radius * spot_radius;
        let peak_intensity = two * total_power / spot_area;  // Peak of Gaussian
        
        Ok(peak_intensity)
    }
    
    /// Calculate received power at target with receiver aperture
    pub fn calculate_received_power(
        &self,
        distance: T,
        receiver_aperture: T,
        total_power: T,
    ) -> Result<T, BeamError> {
        let intensity = self.calculate_power_density(distance, total_power)?;
        
        // Received power = intensity × receiver area
        let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        let receiver_radius = receiver_aperture / T::from(2.0).unwrap();
        let receiver_area = pi * receiver_radius * receiver_radius;
        
        Ok(intensity * receiver_area)
    }
    
    /// Compute phase offsets for beam steering to target direction
    pub fn compute_steering_phases(&mut self, azimuth: T, elevation: T) -> Result<(), BeamError> {
        self.steering.azimuth = azimuth;
        self.steering.elevation = elevation;
        
        let k = T::from(2.0).unwrap() * T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap()) 
              / self.config.wavelength;
        
        let sin_az = azimuth.sin();
        let cos_az = azimuth.cos();
        let sin_el = elevation.sin();
        
        let mut idx = 0;
        for row in 0..self.config.element_count {
            for col in 0..self.config.element_count {
                let x = T::from(col as f64) * self.config.element_spacing;
                let y = T::from(row as f64) * self.config.element_spacing;
                
                // Path difference for phased array
                let path_diff = x * cos_az * sin_el + y * sin_az;
                let phase = -(k * path_diff);
                
                // Wrap to [-π, π]
                let two_pi = T::from(2.0).unwrap() * T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
                let wrapped_phase = phase - two_pi * (phase / two_pi).floor();
                
                self.steering.phase_offsets[idx] = wrapped_phase;
                idx += 1;
            }
        }
        
        Ok(())
    }
    
    /// Calculate atmospheric attenuation (simplified model)
    pub fn calculate_atmospheric_loss(&self, elevation: T, wavelength_microns: f64) -> f64 {
        // Simplified atmospheric transmission model
        // Real implementation would use MODTRAN or similar
        
        let el_deg = elevation.to_f64().unwrap_or(1.0) * 180.0 / std::f64::consts::PI;
        
        if el_deg < 5.0 {
            return 20.0;  // High loss near horizon
        }
        
        // Approximate zenith transmission
        let airmass = 1.0 / el_deg.sin();
        
        // Wavelength-dependent attenuation (dB/km)
        let attenuation_db_km = match wavelength_microns {
            x if x < 0.3 => 0.5,  // UV
            x if x < 1.0 => 0.1,  // Visible
            x if x < 10.0 => 0.05,  // Near IR
            _ => 0.1,  // Far IR / microwave
        };
        
        // Scale by airmass and typical atmosphere height (~8 km)
        attenuation_db_km * 8.0 * airmass
    }
    
    /// Verify beam pointing accuracy
    pub fn verify_pointing_accuracy(&self, target_position: SVector<T, 3>, tolerance: T) -> Result<(), BeamError> {
        let direction = target_position - self.position;
        let distance = direction.norm();
        
        if distance <= T::zero() {
            return Err(BeamError::PointingError { error: f64::INFINITY });
        }
        
        let expected_azimuth = direction[0].atan2(direction[1]);
        let expected_elevation = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt().atan2(direction[2]);
        
        let az_error = (self.steering.azimuth - expected_azimuth).abs();
        let el_error = (self.steering.elevation - expected_elevation).abs();
        
        let total_error = (az_error * az_error + el_error * el_error).sqrt();
        
        if total_error > tolerance {
            return Err(BeamError::PointingError {
                error: total_error.to_f64().unwrap_or(f64::INFINITY),
            });
        }
        
        Ok(())
    }
}

/// Beam transmission result
#[derive(Debug, Clone)]
pub struct BeamTransmissionResult<T> {
    pub transmitted_power: T,
    pub received_power: T,
    pub transmission_efficiency: T,
    pub spot_size_at_target: T,
    pub travel_time_seconds: T,
}

/// Phased array performance metrics
#[derive(Debug, Clone)]
pub struct ArrayMetrics {
    pub gain_db: f64,
    pub beamwidth_degrees: f64,
    pub max_effective_range_ly: f64,
    pub power_transfer_efficiency: f64,
}

impl<T: Float + Copy + Zero> NicollDysonArray<T> {
    /// Calculate array gain
    pub fn calculate_gain(&self) -> T {
        let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        let four_pi = T::from(4.0).unwrap() * pi;
        
        let aperture_area = pi * (self.config.aperture_diameter() / T::from(2.0).unwrap()).powi(2);
        let wavelength_sq = self.config.wavelength * self.config.wavelength;
        
        // G = 4πA / λ² (ideal)
        let ideal_gain = four_pi * aperture_area / wavelength_sq;
        
        // Apply efficiency factor (~0.55 for realistic arrays)
        let efficiency = T::from(0.55).unwrap();
        ideal_gain * efficiency
    }
    
    /// Calculate maximum effective range for useful power transfer
    pub fn calculate_max_effective_range(&self, min_power_density: T) -> T {
        let total_power = self.config.max_total_power();
        let divergence = self.calculate_divergence_angle();
        
        // P_density = P_total / (π * (θ*L)²)
        // L = sqrt(P_total / (π * P_density * θ²))
        let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        
        let denominator = pi * min_power_density * divergence * divergence;
        if denominator <= T::zero() {
            return T::zero();
        }
        
        (total_power / denominator).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_array_creation() {
        type F = f64;
        let config = PhasedArrayConfig::new(
            F::from(1e-6).unwrap(),  // 1 micron wavelength
            F::from(2e-6).unwrap(),  // 2 micron spacing
            1000,                     // 1000x1000 elements
            F::from(1e-6).unwrap(),
            F::from(100.0).unwrap(),  // 100W per element
        );
        
        let position = SVector::zeros();
        let array = NicollDysonArray::new(config, position);
        
        assert!(array.config.total_elements > 0);
    }
    
    #[test]
    fn test_divergence_calculation() {
        type F = f64;
        let config = PhasedArrayConfig::new(
            F::from(1e-6).unwrap(),
            F::from(2e-6).unwrap(),
            100,
            F::from(1e-6).unwrap(),
            F::from(100.0).unwrap(),
        );
        
        let position = SVector::zeros();
        let array = NicollDysonArray::new(config, position);
        
        let divergence = array.calculate_divergence_angle();
        assert!(divergence > F::zero());
        assert!(divergence < F::from(0.1).unwrap());  // Should be very small
    }
    
    #[test]
    fn test_gain_calculation() {
        type F = f64;
        let config = PhasedArrayConfig::new(
            F::from(1e-6).unwrap(),
            F::from(2e-6).unwrap(),
            1000,
            F::from(1e-6).unwrap(),
            F::from(100.0).unwrap(),
        );
        
        let position = SVector::zeros();
        let array = NicollDysonArray::new(config, position);
        
        let gain = array.calculate_gain();
        assert!(gain > F::zero());
    }
}
