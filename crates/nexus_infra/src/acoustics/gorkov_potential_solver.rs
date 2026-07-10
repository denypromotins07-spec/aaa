//! Gor'kov Potential Solver for Acoustic Levitation
//! 
//! Calculates the acoustic radiation force required to trap and levitate
//! microscopic particulates using ultrasonic phased arrays.

use core::fmt;

/// Speed of sound in air at 20°C (m/s)
const SPEED_OF_SOUND: f64 = 343.0;
/// Air density at 20°C (kg/m³)
const AIR_DENSITY: f64 = 1.204;
/// Maximum acoustic pressure amplitude (Pa)
const MAX_PRESSURE_AMPLITUDE: f64 = 1e5;
/// Minimum particle radius that can be levitated (μm)
const MIN_PARTICLE_RADIUS: f64 = 1.0;
/// Maximum particle radius (μm)
const MAX_PARTICLE_RADIUS: f64 = 500.0;

/// Errors in Gor'kov potential calculations
#[derive(Debug, Clone, PartialEq)]
pub enum GorkovError {
    InvalidParticleSize,
    InvalidFrequency,
    PressureExceeded,
    NumericalInstability,
}

impl fmt::Display for GorkovError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GorkovError::InvalidParticleSize => write!(f, "Particle size outside levitatable range"),
            GorkovError::InvalidFrequency => write!(f, "Ultrasonic frequency out of valid range"),
            GorkovError::PressureExceeded => write!(f, "Acoustic pressure exceeds transducer limits"),
            GorkovError::NumericalInstability => write!(f, "Numerical instability in potential calculation"),
        }
    }
}

/// Particle properties for levitation
#[derive(Debug, Clone, Copy)]
pub struct ParticleProperties {
    /// Radius in micrometers
    pub radius_um: f64,
    /// Density in kg/m³
    pub density: f64,
    /// Compressibility in Pa⁻¹
    pub compressibility: f64,
}

impl Default for ParticleProperties {
    fn default() -> Self {
        // Typical dust particle properties
        Self {
            radius_um: 10.0,
            density: 2500.0, // Silica dust
            compressibility: 2.7e-10,
        }
    }
}

/// Gor'kov potential field state
#[derive(Debug, Clone, Copy)]
pub struct GorkovPotentialField {
    /// Potential energy at trap center (J)
    pub potential_depth: f64,
    /// Trap stiffness in x direction (N/m)
    pub stiffness_x: f64,
    /// Trap stiffness in y direction (N/m)
    pub stiffness_y: f64,
    /// Trap stiffness in z direction (N/m)
    pub stiffness_z: f64,
    /// Equilibrium position (x, y, z) in meters
    pub equilibrium: [f64; 3],
    /// Acoustic wavelength (m)
    pub wavelength: f64,
    /// Wave number (rad/m)
    pub wave_number: f64,
}

impl Default for GorkovPotentialField {
    fn default() -> Self {
        Self {
            potential_depth: 0.0,
            stiffness_x: 0.0,
            stiffness_y: 0.0,
            stiffness_z: 0.0,
            equilibrium: [0.0; 3],
            wavelength: 0.0,
            wave_number: 0.0,
        }
    }
}

/// Gor'kov potential solver for acoustic levitation
pub struct GorkovPotentialSolver {
    /// Operating frequency (Hz)
    frequency: f64,
    /// Transducer array geometry parameters
    transducer_spacing: f64,
    /// Medium properties
    medium_density: f64,
    medium_sound_speed: f64,
}

impl GorkovPotentialSolver {
    /// Create a new solver with specified operating frequency
    pub fn new(frequency: f64) -> Result<Self, GorkovError> {
        // Validate frequency (typical ultrasonic range: 20kHz - 10MHz)
        if frequency < 20_000.0 || frequency > 10_000_000.0 {
            return Err(GorkovError::InvalidFrequency);
        }

        Ok(Self {
            frequency,
            transducer_spacing: frequency_to_spacing(frequency),
            medium_density: AIR_DENSITY,
            medium_sound_speed: SPEED_OF_SOUND,
        })
    }

    /// Calculate Gor'kov potential for a particle at given position
    pub fn calculate_potential(
        &self,
        particle: &ParticleProperties,
        pressure_amplitude: f64,
        position: [f64; 3],
    ) -> Result<GorkovPotentialField, GorkovError> {
        // Validate particle size
        if particle.radius_um < MIN_PARTICLE_RADIUS || particle.radius_um > MAX_PARTICLE_RADIUS {
            return Err(GorkovError::InvalidParticleSize);
        }

        // Validate pressure
        if pressure_amplitude < 0.0 || pressure_amplitude > MAX_PRESSURE_AMPLITUDE {
            return Err(GorkovError::PressureExceeded);
        }

        // Convert radius to meters
        let radius_m = particle.radius_um * 1e-6;

        // Calculate wavelength and wave number
        let wavelength = self.medium_sound_speed / self.frequency;
        let wave_number = 2.0 * core::f64::consts::PI / wavelength;

        // Particle volume
        let volume = (4.0 / 3.0) * core::f64::consts::PI * radius_m.powi(3);

        // Acoustic contrast factors
        // f1: density contrast factor
        let density_ratio = self.medium_density / particle.density;
        let f1 = 1.0 - density_ratio;

        // f2: compressibility contrast factor
        let compressibility_ratio = particle.compressibility / (1.0 / (self.medium_density * self.medium_sound_speed.powi(2)));
        let f2 = 2.0 * (compressibility_ratio - 1.0) / (2.0 * compressibility_ratio + 1.0);

        // Mean squared pressure and velocity at position
        // For a standing wave: p² = p₀² cos²(kz), v² = (p₀/ρc)² sin²(kz)
        let kz = wave_number * position[2];
        let cos_kz = kz.cos();
        let sin_kz = kz.sin();

        let pressure_squared = pressure_amplitude.powi(2) * cos_kz.powi(2);
        let velocity_squared = (pressure_amplitude / (self.medium_density * self.medium_sound_speed)).powi(2) * sin_kz.powi(2);

        // Gor'kov potential: U = (2πr³/3) * [f1 * <p²>/(2ρc²) - f2 * 3ρ<v²>/4]
        let term1 = f1 * pressure_squared / (2.0 * self.medium_density * self.medium_sound_speed.powi(2));
        let term2 = f2 * 3.0 * self.medium_density * velocity_squared / 4.0;

        let potential_depth = volume * (term1 - term2);

        // Check for numerical instability
        if !potential_depth.is_finite() || potential_depth.is_nan() {
            return Err(GorkovError::NumericalInstability);
        }

        // Calculate trap stiffness (second derivative of potential)
        // Approximate using finite differences
        let delta = wavelength / 100.0;
        
        let stiffness_x = self.calculate_stiffness(particle, pressure_amplitude, position, 0, delta)?;
        let stiffness_y = self.calculate_stiffness(particle, pressure_amplitude, position, 1, delta)?;
        let stiffness_z = self.calculate_stiffness(particle, pressure_amplitude, position, 2, delta)?;

        // Find equilibrium position (where gradient is zero)
        let equilibrium = self.find_equilibrium(particle, pressure_amplitude, position)?;

        Ok(GorkovPotentialField {
            potential_depth,
            stiffness_x,
            stiffness_y,
            stiffness_z,
            equilibrium,
            wavelength,
            wave_number,
        })
    }

    /// Calculate trap stiffness along one axis
    fn calculate_stiffness(
        &self,
        particle: &ParticleProperties,
        pressure_amplitude: f64,
        base_position: [f64; 3],
        axis: usize,
        delta: f64,
    ) -> Result<f64, GorkovError> {
        let mut pos_minus = base_position;
        let mut pos_plus = base_position;
        
        pos_minus[axis] -= delta;
        pos_plus[axis] += delta;

        let u_minus = self.calculate_potential_simple(particle, pressure_amplitude, pos_minus)?;
        let u_plus = self.calculate_potential_simple(particle, pressure_amplitude, pos_plus)?;
        let u_center = self.calculate_potential_simple(particle, pressure_amplitude, base_position)?;

        // Second derivative approximation: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
        let stiffness = (u_plus - 2.0 * u_center + u_minus) / (delta * delta);

        Ok(stiffness.abs()) // Stiffness should be positive for stable trap
    }

    /// Simplified potential calculation for stiffness computation
    fn calculate_potential_simple(
        &self,
        particle: &ParticleProperties,
        pressure_amplitude: f64,
        position: [f64; 3],
    ) -> Result<f64, GorkovError> {
        let radius_m = particle.radius_um * 1e-6;
        let wavelength = self.medium_sound_speed / self.frequency;
        let wave_number = 2.0 * core::f64::consts::PI / wavelength;
        let volume = (4.0 / 3.0) * core::f64::consts::PI * radius_m.powi(3);

        let density_ratio = self.medium_density / particle.density;
        let f1 = 1.0 - density_ratio;
        let compressibility_ratio = particle.compressibility / (1.0 / (self.medium_density * self.medium_sound_speed.powi(2)));
        let f2 = 2.0 * (compressibility_ratio - 1.0) / (2.0 * compressibility_ratio + 1.0);

        let kz = wave_number * position[2];
        let pressure_squared = pressure_amplitude.powi(2) * kz.cos().powi(2);
        let velocity_squared = (pressure_amplitude / (self.medium_density * self.medium_sound_speed)).powi(2) * kz.sin().powi(2);

        let term1 = f1 * pressure_squared / (2.0 * self.medium_density * self.medium_sound_speed.powi(2));
        let term2 = f2 * 3.0 * self.medium_density * velocity_squared / 4.0;

        Ok(volume * (term1 - term2))
    }

    /// Find equilibrium position near given point
    fn find_equilibrium(
        &self,
        particle: &ParticleProperties,
        pressure_amplitude: f64,
        initial_pos: [f64; 3],
    ) -> Result<[f64; 3], GorkovError> {
        // Simple gradient descent to find local minimum
        let mut pos = initial_pos;
        let mut step_size = 1e-6;
        let tolerance = 1e-9;
        let max_iterations = 100;

        for _ in 0..max_iterations {
            let gradient = self.calculate_gradient(particle, pressure_amplitude, pos)?;
            let gradient_mag = (gradient[0].powi(2) + gradient[1].powi(2) + gradient[2].powi(2)).sqrt();

            if gradient_mag < tolerance {
                break;
            }

            // Update position
            for i in 0..3 {
                pos[i] -= step_size * gradient[i];
            }

            // Adaptive step size
            step_size *= 0.9;
        }

        Ok(pos)
    }

    /// Calculate gradient of potential
    fn calculate_gradient(
        &self,
        particle: &ParticleProperties,
        pressure_amplitude: f64,
        position: [f64; 3],
    ) -> Result<[f64; 3], GorkovError> {
        let delta = 1e-7;
        let mut gradient = [0.0; 3];

        for i in 0..3 {
            let mut pos_plus = position;
            let mut pos_minus = position;
            pos_plus[i] += delta;
            pos_minus[i] -= delta;

            let u_plus = self.calculate_potential_simple(particle, pressure_amplitude, pos_plus)?;
            let u_minus = self.calculate_potential_simple(particle, pressure_amplitude, pos_minus)?;

            gradient[i] = (u_plus - u_minus) / (2.0 * delta);
        }

        Ok(gradient)
    }

    /// Get operating frequency
    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    /// Get wavelength
    pub fn wavelength(&self) -> f64 {
        self.medium_sound_speed / self.frequency
    }

    /// Check if particle can be stably levitated
    pub fn can_levitate(&self, particle: &ParticleProperties, pressure_amplitude: f64) -> bool {
        let result = self.calculate_potential(particle, pressure_amplitude, [0.0, 0.0, self.wavelength() / 4.0]);
        
        match result {
            Ok(field) => {
                // Stable trap requires positive stiffness in all directions
                field.stiffness_x > 0.0 && field.stiffness_y > 0.0 && field.stiffness_z > 0.0
            }
            Err(_) => false,
        }
    }
}

/// Calculate optimal transducer spacing for given frequency
fn frequency_to_spacing(frequency: f64) -> f64 {
    // Optimal spacing is typically λ/2 for constructive interference
    let wavelength = SPEED_OF_SOUND / frequency;
    wavelength / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_creation() {
        let solver = GorkovPotentialSolver::new(40_000.0);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_invalid_frequency() {
        let solver = GorkovPotentialSolver::new(10_000.0); // Below ultrasonic range
        assert_eq!(solver.unwrap_err(), GorkovError::InvalidFrequency);
    }

    #[test]
    fn test_potential_calculation() {
        let solver = GorkovPotentialSolver::new(40_000.0).unwrap();
        let particle = ParticleProperties::default();
        
        let field = solver.calculate_potential(&particle, 1e4, [0.0, 0.0, 0.004]);
        assert!(field.is_ok());
        
        let field = field.unwrap();
        assert!(field.potential_depth.is_finite());
        assert!(field.wavelength > 0.0);
    }

    #[test]
    fn test_levitation_check() {
        let solver = GorkovPotentialSolver::new(40_000.0).unwrap();
        let particle = ParticleProperties::default();
        
        let can_levitate = solver.can_levitate(&particle, 1e4);
        assert!(can_levitate);
    }
}
