//! Delta-V Manifold Calculator for Asteroid Mining Missions
//! 
//! Calculates orbital transfer costs using Tsiolkovsky rocket equation
//! and Hohmann transfer manifolds with physical mass-ratio limits.

/// Error types for Delta-V calculations
#[derive(Debug, Clone, Copy)]
pub enum DeltaVError {
    InvalidMassRatio(f64),
    ExceedsPhysicalLimit(f64),
    ImpossibleOrbit(&'static str),
    NumericalInstability,
}

impl core::fmt::Display for DeltaVError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DeltaVError::InvalidMassRatio(r) => write!(f, "Invalid mass ratio: {}", r),
            DeltaVError::ExceedsPhysicalLimit(dv) => {
                write!(f, "Delta-V exceeds physical limit: {} m/s", dv)
            }
            DeltaVError::ImpossibleOrbit(reason) => {
                write!(f, "Impossible orbit: {}", reason)
            }
            DeltaVError::NumericalInstability => write!(f, "Numerical instability"),
        }
    }
}

/// Physical constants
pub const GRAVITY_SEA_LEVEL: f64 = 9.80665; // m/s²
pub const MU_SUN: f64 = 1.32712440018e20; // m³/s²
pub const AU_METERS: f64 = 1.495978707e11; // meters
pub const MAX_MASS_RATIO: f64 = 50.0; // Practical upper limit for chemical rockets

/// Orbital elements for celestial body
#[derive(Debug, Clone, Copy)]
pub struct OrbitalElements {
    pub semi_major_axis_au: f64,
    pub eccentricity: f64,
    pub inclination_deg: f64,
    pub longitude_asc_node_deg: f64,
    pub argument_periapsis_deg: f64,
}

/// Delta-V budget for complete mission
#[derive(Debug, Clone, Copy)]
pub struct DeltaVBudget {
    pub earth_departure_dv: f64,
    pub transfer_injection_dv: f64,
    pub plane_change_dv: f64,
    pub asteroid_arrival_dv: f64,
    pub return_dv: f64,
    pub total_dv: f64,
    pub required_mass_ratio: f64,
    pub mission_feasible: bool,
}

/// Delta-V manifold calculator
pub struct DeltaVManifoldCalculator {
    pub isp_chemical_s: f64,
    pub isp_electric_s: f64,
}

impl DeltaVManifoldCalculator {
    /// Create calculator with typical propulsion parameters
    pub fn new() -> Self {
        Self {
            isp_chemical_s: 450.0, // LH2/LOX upper stage
            isp_electric_s: 3000.0, // Ion thruster
        }
    }
    
    /// Calculate Hohmann transfer Delta-V from Earth to target
    pub fn hohmann_transfer_dv(
        &self,
        earth_orbit_au: f64,
        target_orbit_au: f64,
    ) -> Result<(f64, f64), DeltaVError> {
        if earth_orbit_au <= 0.0 || target_orbit_au <= 0.0 {
            return Err(DeltaVError::ImpossibleOrbit("Non-positive orbital radius"));
        }
        
        let r1 = earth_orbit_au * AU_METERS;
        let r2 = target_orbit_au * AU_METERS;
        
        // Circular orbit velocities
        let v1 = (MU_SUN / r1).sqrt();
        let v2 = (MU_SUN / r2).sqrt();
        
        // Transfer ellipse parameters
        let a_transfer = (r1 + r2) / 2.0;
        
        // Velocity at perihelion of transfer orbit
        let v_transfer_1 = (MU_SUN * (2.0 / r1 - 1.0 / a_transfer)).sqrt();
        
        // Velocity at aphelion of transfer orbit
        let v_transfer_2 = (MU_SUN * (2.0 / r2 - 1.0 / a_transfer)).sqrt();
        
        // Delta-V for departure burn
        let dv_departure = (v_transfer_1 - v1).abs();
        
        // Delta-V for arrival burn
        let dv_arrival = (v2 - v_transfer_2).abs();
        
        Ok((dv_departure, dv_arrival))
    }
    
    /// Calculate plane change Delta-V
    pub fn plane_change_dv(
        &self,
        velocity_ms: f64,
        inclination_change_deg: f64,
    ) -> f64 {
        let delta_i_rad = inclination_change_deg.to_radians();
        // Delta-V = 2 * v * sin(delta_i / 2)
        2.0 * velocity_ms * (delta_i_rad / 2.0).sin()
    }
    
    /// Calculate complete mission Delta-V budget
    pub fn calculate_mission_budget(
        &self,
        earth_elements: &OrbitalElements,
        asteroid_elements: &OrbitalElements,
        use_electric_propulsion: bool,
    ) -> Result<DeltaVBudget, DeltaVError> {
        let isp = if use_electric_propulsion {
            self.isp_electric_s
        } else {
            self.isp_chemical_s
        };
        
        let exhaust_velocity = isp * GRAVITY_SEA_LEVEL;
        
        // Hohmann transfer
        let (dv_departure, dv_arrival) = self.hohmann_transfer_dv(
            earth_elements.semi_major_axis_au,
            asteroid_elements.semi_major_axis_au,
        )?;
        
        // Plane change at arrival (most efficient at aphelion)
        let inclination_change = (asteroid_elements.inclination_deg - earth_elements.inclination_deg).abs();
        let v_aphelion = (MU_SUN * (2.0 / (asteroid_elements.semi_major_axis_au * AU_METERS) 
            - 1.0 / ((earth_elements.semi_major_axis_au + asteroid_elements.semi_major_axis_au) * AU_METERS / 2.0))).sqrt();
        
        let dv_plane_change = self.plane_change_dv(v_aphelion, inclination_change);
        
        // Total one-way Delta-V
        let total_one_way = dv_departure + dv_arrival + dv_plane_change;
        
        // Return Delta-V (similar magnitude)
        let return_dv = total_one_way * 0.8; // Slightly less due to gravity assist possibilities
        
        // Total mission Delta-V
        let total_dv = total_one_way + return_dv;
        
        // Check physical feasibility
        let max_possible_dv = exhaust_velocity * MAX_MASS_RATIO.ln();
        let mission_feasible = total_dv < max_possible_dv;
        
        // Required mass ratio from Tsiolkovsky equation
        let mass_ratio = (total_dv / exhaust_velocity).exp();
        
        // Cap mass ratio at physical limit
        let capped_mass_ratio = mass_ratio.min(MAX_MASS_RATIO);
        
        if !mission_feasible {
            return Err(DeltaVError::ExceedsPhysicalLimit(total_dv));
        }
        
        Ok(DeltaVBudget {
            earth_departure_dv: dv_departure,
            transfer_injection_dv: dv_arrival,
            plane_change_dv: dv_plane_change,
            asteroid_arrival_dv: dv_arrival,
            return_dv: return_dv,
            total_dv: total_dv,
            required_mass_ratio: capped_mass_ratio,
            mission_feasible,
        })
    }
    
    /// Calculate payload fraction for given dry mass and propellant
    pub fn payload_fraction(&self, dry_mass_kg: f64, propellant_mass_kg: f64, required_dv: f64) -> Result<f64, DeltaVError> {
        if dry_mass_kg <= 0.0 || propellant_mass_kg <= 0.0 {
            return Err(DeltaVError::InvalidMassRatio(dry_mass_kg));
        }
        
        let initial_mass = dry_mass_kg + propellant_mass_kg;
        let mass_ratio = initial_mass / dry_mass_kg;
        
        if mass_ratio > MAX_MASS_RATIO {
            return Err(DeltaVError::ExceedsPhysicalLimit(required_dv));
        }
        
        let exhaust_velocity = self.isp_chemical_s * GRAVITY_SEA_LEVEL;
        let achievable_dv = exhaust_velocity * mass_ratio.ln();
        
        if required_dv > achievable_dv {
            return Err(DeltaVError::ExceedsPhysicalLimit(required_dv));
        }
        
        // Payload fraction = dry_mass / initial_mass
        Ok(dry_mass_kg / initial_mass)
    }
}

impl Default for DeltaVManifoldCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hohmann_earth_to_mars() {
        let calc = DeltaVManifoldCalculator::new();
        
        // Earth at 1 AU, Mars at ~1.52 AU
        let result = calc.hohmann_transfer_dv(1.0, 1.52);
        assert!(result.is_ok());
        
        let (dv1, dv2) = result.unwrap();
        assert!(dv1 > 0.0 && dv2 > 0.0);
    }
    
    #[test]
    fn test_mission_budget() {
        let calc = DeltaVManifoldCalculator::new();
        
        let earth = OrbitalElements {
            semi_major_axis_au: 1.0,
            eccentricity: 0.0167,
            inclination_deg: 0.0,
            longitude_asc_node_deg: 0.0,
            argument_periapsis_deg: 0.0,
        };
        
        let asteroid = OrbitalElements {
            semi_major_axis_au: 2.5,
            eccentricity: 0.15,
            inclination_deg: 10.0,
            longitude_asc_node_deg: 45.0,
            argument_periapsis_deg: 30.0,
        };
        
        let budget = calc.calculate_mission_budget(&earth, &asteroid, false);
        assert!(budget.is_ok());
    }
}
