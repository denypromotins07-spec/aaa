//! Relativistic Doppler Attenuation Calculator for Interstellar Beaming
//! 
//! Computes relativistic effects on beamed energy transmission to probes
//! moving at significant fractions of light speed.

use nalgebra::SVector;
use num_traits::{Float, Zero};
use thiserror::Error;

/// Relativistic parameters for a moving probe
#[derive(Clone, Debug)]
pub struct RelativisticProbe<T> {
    pub position: SVector<T, 3>,
    pub velocity: SVector<T, 3>,
    pub proper_time: T,  // Time experienced on probe
    pub distance_from_source: T,
}

impl<T: Float + Copy + Zero> RelativisticProbe<T> {
    pub fn new(
        position: SVector<T, 3>,
        velocity: SVector<T, 3>,
        initial_time: T,
    ) -> Self {
        let distance = position.norm();
        Self {
            position,
            velocity,
            proper_time: initial_time,
            distance_from_source: distance,
        }
    }
    
    /// Calculate Lorentz factor: γ = 1/√(1 - v²/c²)
    pub fn lorentz_factor(&self, c: T) -> Result<T, RelativisticError> {
        let v_squared = self.velocity.dot(&self.velocity);
        let c_squared = c * c;
        
        if v_squared >= c_squared {
            return Err(RelativisticError::SuperluminalVelocity {
                velocity: v_squared.sqrt().to_f64().unwrap_or(f64::INFINITY),
                c: c.to_f64().unwrap_or(299792458.0),
            });
        }
        
        let one = T::one();
        let beta_squared = v_squared / c_squared;
        let gamma = one / (one - beta_squared).sqrt();
        
        Ok(gamma)
    }
    
    /// Calculate β = v/c
    pub fn beta(&self, c: T) -> T {
        let v = self.velocity.norm();
        v / c
    }
}

/// Errors in relativistic calculations
#[derive(Error, Debug)]
pub enum RelativisticError {
    #[error("Velocity {velocity:?} m/s exceeds speed of light {c:?} m/s")]
    SuperluminalVelocity { velocity: f64, c: f64 },
    #[error("Invalid redshift parameter z={z:?}")]
    InvalidRedshift { z: f64 },
    #[error("Beam missed target by {miss_distance:?} m")]
    BeamMiss { miss_distance: f64 },
}

/// Doppler shift calculator for relativistic beaming
pub struct RelativisticDopplerCalculator<T> {
    speed_of_light: T,
}

impl<T: Float + Copy + Zero> RelativisticDopplerCalculator<T> {
    pub fn new() -> Self {
        Self {
            speed_of_light: T::from(299792458.0).unwrap(),
        }
    }
    
    /// Calculate relativistic Doppler factor for longitudinal motion
    /// f_observed = f_emitted * √((1-β)/(1+β)) for receding source
    pub fn calculate_doppler_factor(&self, probe: &RelativisticProbe<T>, approaching: bool) -> Result<T, RelativisticError> {
        let beta = probe.beta(self.speed_of_light);
        let one = T::one();
        
        if beta >= one {
            return Err(RelativisticError::SuperluminalVelocity {
                velocity: beta.to_f64().unwrap_or(1.0) * self.speed_of_light.to_f64().unwrap_or(299792458.0),
                c: self.speed_of_light.to_f64().unwrap_or(299792458.0),
            });
        }
        
        let numerator = if approaching {
            one + beta
        } else {
            one - beta
        };
        
        let denominator = if approaching {
            one - beta
        } else {
            one + beta
        };
        
        Ok((numerator / denominator).sqrt())
    }
    
    /// Calculate received frequency after Doppler shift
    pub fn calculate_shifted_frequency(
        &self,
        emitted_frequency: T,
        probe: &RelativisticProbe<T>,
        approaching: bool,
    ) -> Result<T, RelativisticError> {
        let doppler_factor = self.calculate_doppler_factor(probe, approaching)?;
        Ok(emitted_frequency * doppler_factor)
    }
    
    /// Calculate relativistic aberration angle
    /// cos(θ') = (cos(θ) - β) / (1 - β*cos(θ))
    pub fn calculate_aberration(&self, emission_angle: T, beta: T) -> Result<T, RelativisticError> {
        let one = T::one();
        
        if beta.abs() >= one {
            return Err(RelativisticError::SuperluminalVelocity {
                velocity: beta.to_f64().unwrap_or(1.0) * self.speed_of_light.to_f64().unwrap_or(299792458.0),
                c: self.speed_of_light.to_f64().unwrap_or(299792458.0),
            });
        }
        
        let cos_theta = emission_angle.cos();
        let numerator = cos_theta - beta;
        let denominator = one - beta * cos_theta;
        
        if denominator.abs() <= T::from(1e-10).unwrap() {
            return Err(RelativisticError::InvalidRedshift { z: f64::INFINITY });
        }
        
        let cos_theta_prime = numerator / denominator;
        
        // Clamp to valid range for acos
        let cos_clamped = cos_theta_prime.max(T::from(-1.0).unwrap()).min(T::one());
        
        Ok(cos_clamped.acos())
    }
    
    /// Calculate time dilation effect on pulse reception
    pub fn calculate_time_dilation(&self, probe: &RelativisticProbe<T>) -> Result<T, RelativisticError> {
        let gamma = probe.lorentz_factor(self.speed_of_light)?;
        Ok(gamma)
    }
    
    /// Calculate relativistic beaming (headlight effect) concentration
    /// Power is concentrated into cone of half-angle ~1/γ
    pub fn calculate_beaming_cone_half_angle(&self, probe: &RelativisticProbe<T>) -> Result<T, RelativisticError> {
        let gamma = probe.lorentz_factor(self.speed_of_light)?;
        let one = T::one();
        
        // Cone half-angle ≈ 1/γ radians
        if gamma > T::zero() {
            Ok(one / gamma)
        } else {
            Ok(T::from(std::f64::consts::FRAC_PI_2).unwrap())
        }
    }
}

/// Complete attenuation calculation including all relativistic effects
pub struct AttenuationCalculator<T> {
    doppler_calc: RelativisticDopplerCalculator<T>,
}

impl<T: Float + Copy + Zero> AttenuationCalculator<T> {
    pub fn new() -> Self {
        Self {
            doppler_calc: RelativisticDopplerCalculator::new(),
        }
    }
    
    /// Calculate total power attenuation including:
    /// - Inverse square law
    /// - Relativistic Doppler shift
    /// - Time dilation
    /// - Aberration
    pub fn calculate_total_attenuation(
        &self,
        probe: &RelativisticProbe<T>,
        transmitted_power: T,
        receiver_area: T,
        wavelength: T,
    ) -> Result<AttenuationResult<T>, RelativisticError> {
        let c = self.doppler_calc.speed_of_light;
        let distance = probe.distance_from_source;
        
        // Classical inverse square loss
        let four_pi = T::from(4.0).unwrap() * T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        let sphere_area = four_pi * distance * distance;
        
        if sphere_area <= T::zero() {
            return Err(RelativisticError::InvalidRedshift { z: f64::INFINITY });
        }
        
        // Classical received power
        let classical_received = transmitted_power * receiver_area / sphere_area;
        
        // Relativistic corrections
        let beta = probe.beta(c);
        let gamma = probe.lorentz_factor(c)?;
        
        // Determine if approaching or receding
        let radial_velocity = probe.velocity.dot(&probe.position) / distance;
        let approaching = radial_velocity < T::zero();
        
        // Doppler factor affects both frequency and photon rate
        let doppler_factor = self.doppler_calc.calculate_doppler_factor(probe, approaching)?;
        
        // Received power scales as doppler_factor² (frequency shift × photon rate)
        let relativistic_received = classical_received * doppler_factor * doppler_factor;
        
        // Aberration correction for effective receiver area
        let emission_angle = T::from(std::f64::consts::FRAC_PI_2).unwrap();  // Assume perpendicular in probe frame
        let aberrated_angle = self.doppler_calc.calculate_aberration(emission_angle, beta)?;
        
        // Effective area reduction due to aberration
        let area_factor = aberrated_angle.cos().abs();
        let final_received = relativistic_received * area_factor;
        
        // Calculate attenuation in dB
        let attenuation_ratio = if transmitted_power > T::zero() {
            final_received / transmitted_power
        } else {
            T::zero()
        };
        
        let attenuation_db = if attenuation_ratio > T::zero() {
            T::from(10.0).unwrap() * attenuation_ratio.log10()
        } else {
            T::from(-1000.0).unwrap()  // Effectively zero
        };
        
        Ok(AttenuationResult {
            classical_received_power: classical_received,
            relativistic_received_power: final_received,
            doppler_factor: doppler_factor,
            lorentz_factor: gamma,
            aberration_angle: aberrated_angle,
            attenuation_db: attenuation_db.to_f64().unwrap_or(-1000.0),
            travel_time: distance / c,
        })
    }
    
    /// Calculate optimal beaming frequency for relativistic probe
    /// Account for Doppler shift to ensure probe receives correct frequency
    pub fn calculate_optimal_transmit_frequency(
        &self,
        probe: &RelativisticProbe<T>,
        desired_receive_frequency: T,
    ) -> Result<T, RelativisticError> {
        let beta = probe.beta(self.doppler_calc.speed_of_light);
        let radial_velocity = probe.velocity.dot(&probe.position) / probe.distance_from_source;
        let approaching = radial_velocity < T::zero();
        
        let doppler_factor = self.doppler_calc.calculate_doppler_factor(probe, approaching)?;
        
        // f_transmit = f_receive / doppler_factor
        if doppler_factor > T::zero() {
            Ok(desired_receive_frequency / doppler_factor)
        } else {
            Err(RelativisticError::InvalidRedshift { z: f64::INFINITY })
        }
    }
}

/// Attenuation calculation result
#[derive(Debug, Clone)]
pub struct AttenuationResult<T> {
    pub classical_received_power: T,
    pub relativistic_received_power: T,
    pub doppler_factor: T,
    pub lorentz_factor: T,
    pub aberration_angle: T,
    pub attenuation_db: f64,
    pub travel_time: T,
}

/// Light-year conversion utility
pub fn light_years_to_meters<T: Float>(ly: T) -> T {
    let meters_per_ly = T::from(9.461e15).unwrap();
    ly * meters_per_ly
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lorentz_factor() {
        type F = f64;
        let calc = RelativisticDopplerCalculator::<F>::new();
        
        // Probe at 0.5c
        let velocity = SVector::new(F::from(1.499e8).unwrap(), F::zero(), F::zero());
        let position = SVector::new(F::from(1e11).unwrap(), F::zero(), F::zero());
        let probe = RelativisticProbe::new(position, velocity, F::zero());
        
        let gamma = probe.lorentz_factor(calc.speed_of_light).unwrap();
        
        // γ at 0.5c ≈ 1.155
        assert!(gamma > F::from(1.1).unwrap());
        assert!(gamma < F::from(1.2).unwrap());
    }
    
    #[test]
    fn test_doppler_shift_receding() {
        type F = f64;
        let calc = RelativisticDopplerCalculator::<F>::new();
        
        // Receding probe at 0.1c
        let velocity = SVector::new(F::from(2.998e7).unwrap(), F::zero(), F::zero());
        let position = SVector::new(F::from(1e11).unwrap(), F::zero(), F::zero());
        let probe = RelativisticProbe::new(position, velocity, F::zero());
        
        let factor = calc.calculate_doppler_factor(&probe, false).unwrap();
        
        // Redshift: factor < 1 for receding
        assert!(factor < F::one());
        assert!(factor > F::from(0.8).unwrap());
    }
    
    #[test]
    fn test_superluminal_error() {
        type F = f64;
        let calc = RelativisticDopplerCalculator::<F>::new();
        
        // Impossible velocity > c
        let velocity = SVector::new(F::from(4e8).unwrap(), F::zero(), F::zero());
        let position = SVector::new(F::from(1e11).unwrap(), F::zero(), F::zero());
        let probe = RelativisticProbe::new(position, velocity, F::zero());
        
        let result = probe.lorentz_factor(calc.speed_of_light);
        assert!(result.is_err());
    }
}
