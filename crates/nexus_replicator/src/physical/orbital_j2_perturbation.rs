//! Orbital J2 Perturbation Simulator
//!
//! Calculates orbital mechanics including Earth's J2 zonal harmonic perturbations
//! for Low Earth Orbit (LEO) satellites.

use core::f64;

/// Error types for orbital calculations
#[derive(Debug, Clone, PartialEq)]
pub enum OrbitalError {
    InvalidAltitude,
    InvalidInclination,
    DecayTooRapid,
    CalculationOverflow,
}

/// Result type for orbital operations
pub type OrbitalResult<T> = Result<T, OrbitalError>;

/// Physical constants for orbital mechanics
pub mod orbital_constants {
    /// Gravitational parameter of Earth (m^3/s^2)
    pub const MU_EARTH: f64 = 3.986_004_418e14;
    
    /// Earth radius (m)
    pub const R_EARTH: f64 = 6_378_137.0;
    
    /// J2 zonal harmonic coefficient
    pub const J2: f64 = 1.082_626_68e-3;
    
    /// Earth rotation rate (rad/s)
    pub const OMEGA_EARTH: f64 = 7.292_115_0e-5;
    
    /// Minimum safe LEO altitude (km)
    pub const MIN_LEO_ALTITUDE_KM: f64 = 200.0;
    
    /// Maximum practical LEO altitude (km)
    pub const MAX_LEO_ALTITUDE_KM: f64 = 2000.0;
}

/// Orbital elements
#[derive(Debug, Clone)]
pub struct OrbitalElements {
    /// Semi-major axis (m)
    pub semi_major_axis: f64,
    /// Eccentricity
    pub eccentricity: f64,
    /// Inclination (rad)
    pub inclination: f64,
    /// Right ascension of ascending node (rad)
    pub raan: f64,
    /// Argument of periapsis (rad)
    pub arg_periapsis: f64,
    /// True anomaly (rad)
    pub true_anomaly: f64,
}

impl OrbitalElements {
    /// Create circular orbit from altitude and inclination
    pub fn circular(altitude_km: f64, inclination_deg: f64) -> OrbitalResult<Self> {
        use orbital_constants::*;

        if altitude_km < MIN_LEO_ALTITUDE_KM || altitude_km > MAX_LEO_ALTITUDE_KM {
            return Err(OrbitalError::InvalidAltitude);
        }

        let inclination_rad = inclination_deg.to_radians();
        if inclination_rad < 0.0 || inclination_rad > f64::consts::PI {
            return Err(OrbitalError::InvalidInclination);
        }

        let altitude_m = altitude_km * 1000.0;
        let semi_major_axis = R_EARTH + altitude_m;

        Ok(Self {
            semi_major_axis,
            eccentricity: 0.0, // Circular orbit
            inclination: inclination_rad,
            raan: 0.0,
            arg_periapsis: 0.0,
            true_anomaly: 0.0,
        })
    }

    /// Calculate orbital period (seconds)
    pub fn period(&self) -> f64 {
        use orbital_constants::MU_EARTH;
        // T = 2π * sqrt(a³/μ)
        2.0 * f64::consts::PI * (self.semi_major_axis.powi(3) / MU_EARTH).sqrt()
    }

    /// Calculate mean motion (rad/s)
    pub fn mean_motion(&self) -> f64 {
        use orbital_constants::MU_EARTH;
        // n = sqrt(μ/a³)
        (MU_EARTH / self.semi_major_axis.powi(3)).sqrt()
    }
}

/// J2 perturbation effects on orbital elements
#[derive(Debug, Clone)]
pub struct J2Perturbation {
    /// Rate of change of RAAN (rad/s)
    pub raan_dot: f64,
    /// Rate of change of argument of periapsis (rad/s)
    pub arg_periapsis_dot: f64,
    /// Rate of change of mean anomaly (rad/s)
    pub mean_anomaly_dot: f64,
}

impl J2Perturbation {
    /// Calculate J2 perturbation rates
    pub fn calculate(elements: &OrbitalElements) -> OrbitalResult<Self> {
        use orbital_constants::*;

        let a = elements.semi_major_axis;
        let e = elements.eccentricity;
        let i = elements.inclination;

        // Mean motion
        let n = elements.mean_motion();

        // Semi-latus rectum
        let p = a * (1.0 - e.powi(2));

        // Check for division by zero
        if p < R_EARTH {
            return Err(OrbitalError::CalculationOverflow);
        }

        // J2 perturbation formulas
        let j2_factor = 1.5 * J2 * n * (R_EARTH / p).powi(2);

        // RAAN precession rate
        let raan_dot = -j2_factor * i.cos();

        // Argument of periapsis precession rate
        let arg_periapsis_dot = j2_factor * (2.0 - 2.5 * i.sin().powi(2));

        // Mean anomaly rate correction
        let mean_anomaly_dot = n + j2_factor * (1.0 - 1.5 * i.sin().powi(2)).sqrt();

        Ok(Self {
            raan_dot,
            arg_periapsis_dot,
            mean_anomaly_dot,
        })
    }
}

/// Atmospheric drag model for orbital decay
#[derive(Debug, Clone)]
pub struct AtmosphericDrag {
    /// Ballistic coefficient (kg/m²)
    pub ballistic_coefficient: f64,
    /// Atmospheric density at reference altitude (kg/m³)
    pub rho_0: f64,
    /// Scale height (m)
    pub scale_height: f64,
}

impl AtmosphericDrag {
    /// Create standard atmospheric drag model
    pub fn new(ballistic_coefficient: f64) -> Self {
        Self {
            ballistic_coefficient,
            rho_0: 1.225, // Sea level density
            scale_height: 8500.0, // ~8.5 km
        }
    }

    /// Calculate atmospheric density at altitude
    pub fn density_at_altitude(&self, altitude_m: f64) -> f64 {
        use orbital_constants::R_EARTH;
        
        let h = altitude_m / 1000.0; // Convert to km
        if h < 0.0 {
            return self.rho_0;
        }
        
        // Exponential atmosphere model
        self.rho_0 * (-h / (self.scale_height / 1000.0)).exp()
    }

    /// Estimate delta-V per orbit due to drag
    pub fn delta_v_per_orbit(&self, elements: &OrbitalElements) -> f64 {
        let altitude = elements.semi_major_axis - orbital_constants::R_EARTH;
        let rho = self.density_at_altitude(altitude);
        
        // Simplified drag calculation
        let velocity = (orbital_constants::MU_EARTH / elements.semi_major_axis).sqrt();
        let period = elements.period();
        
        // ΔV ≈ (ρ * v² * A/m * T) / 2
        (rho * velocity.powi(2) / self.ballistic_coefficient * period) / 2.0
    }
}

/// Orbital simulator state
#[derive(Debug)]
pub struct OrbitalState {
    /// Current orbital elements
    pub elements: OrbitalElements,
    /// Time since epoch (seconds)
    pub time_since_epoch: f64,
    /// Cumulative delta-V from station keeping (m/s)
    pub cumulative_delta_v: f64,
}

impl OrbitalState {
    /// Create new orbital state
    pub fn new(elements: OrbitalElements) -> Self {
        Self {
            elements,
            time_since_epoch: 0.0,
            cumulative_delta_v: 0.0,
        }
    }

    /// Propagate orbit with J2 perturbations
    pub fn propagate_j2(&mut self, dt: f64) -> OrbitalResult<()> {
        let perturbation = J2Perturbation::calculate(&self.elements)?;
        
        // Update angular elements
        self.elements.raan += perturbation.raan_dot * dt;
        self.elements.arg_periapsis += perturbation.arg_periapsis_dot * dt;
        
        // Normalize angles
        self.elements.raan = self.elements.raan.rem_euclid(2.0 * f64::consts::PI);
        self.elements.arg_periapsis = self.elements.arg_periapsis.rem_euclid(2.0 * f64::consts::PI);
        
        self.time_since_epoch += dt;
        
        Ok(())
    }

    /// Calculate required station-keeping delta-V
    pub fn required_station_keeping_delta_v(&self, drag: &AtmosphericDrag) -> f64 {
        let decay_rate = drag.delta_v_per_orbit(&self.elements);
        
        // Station keeping must counteract decay
        decay_rate.abs()
    }
}

/// Main orbital simulator
pub struct OrbitalSimulator {
    /// Current state
    state: Option<OrbitalState>,
    /// Drag model
    drag_model: AtmosphericDrag,
}

impl OrbitalSimulator {
    /// Create new simulator
    pub fn new(ballistic_coefficient: f64) -> Self {
        Self {
            state: None,
            drag_model: AtmosphericDrag::new(ballistic_coefficient),
        }
    }

    /// Initialize with circular orbit
    pub fn init_circular_orbit(
        &mut self,
        altitude_km: f64,
        inclination_deg: f64,
    ) -> OrbitalResult<()> {
        let elements = OrbitalElements::circular(altitude_km, inclination_deg)?;
        self.state = Some(OrbitalState::new(elements));
        Ok(())
    }

    /// Get optimal LEO altitude for maximum solar exposure with minimal drag
    pub fn optimal_leo_altitude() -> f64 {
        // Balance between atmospheric drag and radiation belt exposure
        // Typically 500-600 km is optimal for small satellites
        550.0
    }

    /// Calculate orbital lifetime before re-entry
    pub fn estimate_orbital_lifetime_years(&self) -> OrbitalResult<f64> {
        let state = self.state.as_ref().ok_or(OrbitalError::InvalidAltitude)?;
        
        let decay_per_orbit = self.drag_model.delta_v_per_orbit(&state.elements);
        if decay_per_orbit <= 0.0 {
            return Ok(f64::INFINITY); // Stable orbit
        }
        
        let velocity = (orbital_constants::MU_EARTH / state.elements.semi_major_axis).sqrt();
        let orbits_to_decay = velocity / decay_per_orbit.abs();
        let period_seconds = state.elements.period();
        
        let total_seconds = orbits_to_decay * period_seconds;
        let years = total_seconds / (365.25 * 24.0 * 3600.0);
        
        Ok(years)
    }

    /// Get current state
    pub fn state(&self) -> Option<&OrbitalState> {
        self.state.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_orbit_creation() {
        let result = OrbitalElements::circular(500.0, 51.6);
        assert!(result.is_ok());
        
        let elements = result.unwrap();
        assert_eq!(elements.eccentricity, 0.0);
        assert!(elements.semi_major_axis > orbital_constants::R_EARTH);
    }

    #[test]
    fn test_invalid_altitude() {
        let low = OrbitalElements::circular(100.0, 45.0);
        assert_eq!(low, Err(OrbitalError::InvalidAltitude));
        
        let high = OrbitalElements::circular(3000.0, 45.0);
        assert_eq!(high, Err(OrbitalError::InvalidAltitude));
    }

    #[test]
    fn test_orbital_period() {
        let elements = OrbitalElements::circular(400.0, 51.6).unwrap();
        let period = elements.period();
        
        // ISS-like orbit should have ~92 minute period
        let expected_minutes = period / 60.0;
        assert!(expected_minutes > 90.0 && expected_minutes < 95.0);
    }

    #[test]
    fn test_j2_perturbation() {
        let elements = OrbitalElements::circular(500.0, 45.0).unwrap();
        let perturbation = J2Perturbation::calculate(&elements);
        assert!(perturbation.is_ok());
    }
}
