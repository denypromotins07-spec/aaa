//! Debris Cloud Expansion Model
//! 
//! Simulates the expansion of debris clouds after collision events.

use super::kessler_boltzmann_pde::{DebrisDensityField, KesslerBoltzmannSolver, KesslerError};

/// Error types for debris cloud model
#[derive(Debug, Clone, Copy)]
pub enum DebrisCloudError {
    InvalidInitialEnergy(f64),
    NegativeExpansionRate,
    NumericalInstability,
}

impl core::fmt::Display for DebrisCloudError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DebrisCloudError::InvalidInitialEnergy(e) => {
                write!(f, "Invalid initial energy: {}", e)
            }
            DebrisCloudError::NegativeExpansionRate => {
                write!(f, "Negative expansion rate")
            }
            DebrisCloudError::NumericalInstability => {
                write!(f, "Numerical instability")
            }
        }
    }
}

/// Debris cloud state after collision
#[derive(Debug, Clone, Copy)]
pub struct DebrisCloudState {
    pub center_altitude_km: f64,
    pub center_inclination_deg: f64,
    pub altitude_spread_km: f64,
    pub inclination_spread_deg: f64,
    pub total_debris_count: f64,
    pub mean_velocity_ms: f64,
    pub time_since_event: f64,
}

/// Debris cloud expansion model
pub struct DebrisCloudExpander {
    pub solver: KesslerBoltzmannSolver,
    pub drag_coefficient: f64,
    pub atmospheric_scale_height_km: f64,
}

impl DebrisCloudExpander {
    /// Create new expander with default parameters
    pub fn new() -> Self {
        Self {
            solver: KesslerBoltzmannSolver::new(),
            drag_coefficient: 2.2,
            atmospheric_scale_height_km: 50.0,
        }
    }
    
    /// Initialize debris cloud from collision event
    pub fn initialize_cloud(
        &self,
        collision_altitude_km: f64,
        collision_inclination_deg: f64,
        debris_count: f64,
        delta_v_ms: f64,
    ) -> Result<DebrisCloudState, DebrisCloudError> {
        if debris_count <= 0.0 {
            return Err(DebrisCloudError::NumericalInstability);
        }
        if delta_v_ms < 0.0 {
            return Err(DebrisCloudError::NegativeExpansionRate);
        }
        
        // Initial spread based on delta-V impulse
        let initial_spread_km = delta_v_ms * 10.0 / 1000.0; // Simplified conversion
        
        Ok(DebrisCloudState {
            center_altitude_km: collision_altitude_km,
            center_inclination_deg: collision_inclination_deg,
            altitude_spread_km: initial_spread_km,
            inclination_spread_deg: delta_v_ms / 100.0,
            total_debris_count: debris_count,
            mean_velocity_ms: delta_v_ms,
            time_since_event: 0.0,
        })
    }
    
    /// Evolve debris cloud forward in time
    pub fn evolve_cloud(
        &mut self,
        state: &mut DebrisCloudState,
        dt_seconds: f64,
    ) -> Result<(), DebrisCloudError> {
        if dt_seconds <= 0.0 {
            return Err(DebrisCloudError::NumericalInstability);
        }
        
        // Expand due to differential orbital mechanics
        let expansion_rate = self.compute_expansion_rate(state)?;
        
        // Update spreads
        state.altitude_spread_km += expansion_rate * dt_seconds;
        state.inclination_spread_deg += expansion_rate * 0.01 * dt_seconds;
        
        // Decay due to atmospheric drag (altitude dependent)
        let decay_rate = self.compute_drag_decay(state.center_altitude_km);
        state.total_debris_count *= (-decay_rate * dt_seconds).exp();
        
        // Update time
        state.time_since_event += dt_seconds;
        
        // Clamp values to physical bounds
        state.altitude_spread_km = state.altitude_spread_km.min(500.0);
        state.inclination_spread_deg = state.inclination_spread_deg.min(30.0);
        state.total_debris_count = state.total_debris_count.max(0.0);
        
        Ok(())
    }
    
    /// Compute expansion rate from orbital dynamics
    fn compute_expansion_rate(&self, state: &DebrisCloudState) -> Result<f64, DebrisCloudError> {
        // J2 perturbation causes nodal precession
        let j2 = 1.0826e-3;
        let earth_radius = 6371.0;
        let orbital_radius = earth_radius + state.center_altitude_km;
        
        // Nodal precession rate (rad/s)
        let n = (3.986e14 / (orbital_radius * 1000.0).powi(3)).sqrt();
        let inc_rad = state.center_inclination_deg.to_radians();
        
        let omega_dot = -1.5 * j2 * n * (earth_radius / orbital_radius).powi(2) * inc_rad.cos();
        
        // Expansion rate proportional to precession and initial velocity dispersion
        let expansion = omega_dot.abs() * state.mean_velocity_ms * 0.1;
        
        if expansion < 0.0 {
            return Err(DebrisCloudError::NegativeExpansionRate);
        }
        
        Ok(expansion.min(1.0)) // Cap at reasonable value
    }
    
    /// Compute atmospheric drag decay rate
    fn compute_drag_decay(&self, altitude_km: f64) -> f64 {
        // Exponential atmosphere model
        let sea_level_density = 1.225; // kg/m³
        let density = sea_level_density * (-altitude_km / self.atmospheric_scale_height_km).exp();
        
        // Decay rate proportional to density
        let decay = density * self.drag_coefficient * 1e-6;
        
        decay.min(1e-3) // Cap decay rate
    }
    
    /// Inject debris cloud into density field
    pub fn inject_into_field(
        &self,
        field: &mut DebrisDensityField,
        cloud: &DebrisCloudState,
    ) -> Result<(), KesslerError> {
        // Find grid cells affected by cloud
        let alt_bin_size = field.data.len() as f64 / (field.altitude_bins * field.inclination_bins);
        
        for alt_idx in 0..field.altitude_bins {
            for inc_idx in 0..field.inclination_bins {
                let alt_center = 200.0 + alt_idx as f64 * 50.0; // Assuming 50km bins
                
                // Gaussian distribution centered on cloud
                let alt_distance = (alt_center - cloud.center_altitude_km).abs();
                let weight = (-alt_distance.powi(2) / (2.0 * cloud.altitude_spread_km.powi(2))).exp();
                
                if weight > 0.01 {
                    let current = field.get(alt_idx, inc_idx);
                    let addition = cloud.total_debris_count * weight / field.data.len() as f64;
                    field.set(alt_idx, inc_idx, current + addition);
                }
            }
        }
        
        Ok(())
    }
}

impl Default for DebrisCloudExpander {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cloud_initialization() {
        let expander = DebrisCloudExpander::new();
        let cloud = expander.initialize_cloud(800.0, 45.0, 1000.0, 100.0);
        
        assert!(cloud.is_ok());
        let state = cloud.unwrap();
        assert!(state.altitude_spread_km > 0.0);
    }
    
    #[test]
    fn test_cloud_evolution() {
        let mut expander = DebrisCloudExpander::new();
        let mut cloud = expander.initialize_cloud(800.0, 45.0, 1000.0, 100.0).unwrap();
        
        let result = expander.evolve_cloud(&mut cloud, 3600.0);
        assert!(result.is_ok());
        assert!(cloud.altitude_spread_km > 0.0);
    }
}
