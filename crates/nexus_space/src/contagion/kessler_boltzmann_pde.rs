//! Kessler Syndrome Boltzmann PDE Solver
//! 
//! Models collisional cascade in orbital debris using Boltzmann kinetic equation.
//! Implements strict physical bounds to prevent infinite integration loops.

/// Error types for Kessler PDE solver
#[derive(Debug, Clone, Copy)]
pub enum KesslerError {
    NegativeDensity(f64),
    UnphysicalCollisionRate(f64),
    CFLViolation(f64),
    NumericalInstability,
}

impl core::fmt::Display for KesslerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            KesslerError::NegativeDensity(d) => write!(f, "Negative density: {}", d),
            KesslerError::UnphysicalCollisionRate(r) => {
                write!(f, "Unphysical collision rate: {}", r)
            }
            KesslerError::CFLViolation(cfl) => {
                write!(f, "CFL condition violated: {}", cfl)
            }
            KesslerError::NumericalInstability => write!(f, "Numerical instability"),
        }
    }
}

/// Physical constants for LEO environment
pub const EARTH_RADIUS_KM: f64 = 6371.0;
pub const LEO_MIN_ALT_KM: f64 = 200.0;
pub const LEO_MAX_ALT_KM: f64 = 2000.0;
pub const MAX_COLLISION_CROSS_SECTION: f64 = 1000.0; // m²
pub const MAX_DEBRIS_DENSITY: f64 = 1e6; // objects/km³
pub const MAX_ITERATIONS: usize = 10000;
pub const CFL_NUMBER_LIMIT: f64 = 0.5;

/// Debris size distribution
#[derive(Debug, Clone, Copy)]
pub struct DebrisSizeDistribution {
    pub n_lt_1cm: f64,      // objects per km³
    pub n_1cm_to_10cm: f64,
    pub n_gt_10cm: f64,
}

/// Spatial debris density field
pub struct DebrisDensityField {
    pub altitude_bins: usize,
    pub inclination_bins: usize,
    pub data: Box<[f64]>,
    pub dt: f64,
    pub time: f64,
}

impl DebrisDensityField {
    /// Create new density field
    pub fn new(alt_bins: usize, inc_bins: usize) -> Result<Self, KesslerError> {
        if alt_bins == 0 || inc_bins == 0 {
            return Err(KesslerError::NumericalInstability);
        }
        
        let size = alt_bins * inc_bins;
        Ok(Self {
            altitude_bins: alt_bins,
            inclination_bins: inc_bins,
            data: vec![0.0; size].into_boxed_slice(),
            dt: 0.0,
            time: 0.0,
        })
    }
    
    /// Get density at (altitude_idx, inclination_idx)
    #[inline]
    pub fn get(&self, alt_idx: usize, inc_idx: usize) -> f64 {
        if alt_idx >= self.altitude_bins || inc_idx >= self.inclination_bins {
            return 0.0;
        }
        self.data[alt_idx * self.inclination_bins + inc_idx]
    }
    
    /// Set density at (altitude_idx, inclination_idx)
    #[inline]
    pub fn set(&mut self, alt_idx: usize, inc_idx: usize, value: f64) {
        if alt_idx < self.altitude_bins && inc_idx < self.inclination_bins {
            // Clamp to physical bounds
            let clamped = value.max(0.0).min(MAX_DEBRIS_DENSITY);
            self.data[alt_idx * self.inclination_bins + inc_idx] = clamped;
        }
    }
    
    /// Reset field to zero
    pub fn reset(&mut self) {
        self.data.fill(0.0);
        self.time = 0.0;
    }
}

/// Kessler syndrome PDE solver
pub struct KesslerBoltzmannSolver {
    pub grid_altitude_km: f64,
    pub grid_inclination_deg: f64,
    pub collision_coefficient: f64,
    pub fragmentation_yield: f64,
}

impl KesslerBoltzmannSolver {
    /// Create new solver with physical parameters
    pub fn new() -> Self {
        Self {
            grid_altitude_km: 50.0, // Altitude bin size
            grid_inclination_deg: 10.0,
            collision_coefficient: 1e-9, // Simplified collision probability
            fragmentation_yield: 100.0, // Fragments per collision
        }
    }
    
    /// Calculate collision rate using Boltzmann collision integral
    pub fn collision_rate(&self, density: f64, relative_velocity_ms: f64) -> Result<f64, KesslerError> {
        if density < 0.0 {
            return Err(KesslerError::NegativeDensity(density));
        }
        
        // Collision rate = n² * σ * v_rel
        // where n is number density, σ is cross-section, v_rel is relative velocity
        let cross_section = MAX_COLLISION_CROSS_SECTION.min(100.0); // Conservative estimate
        
        let rate = density * density * cross_section * relative_velocity_ms * self.collision_coefficient;
        
        // Bound collision rate to prevent explosion
        let bounded_rate = rate.min(1e6);
        
        if !bounded_rate.is_finite() {
            return Err(KesslerError::UnphysicalCollisionRate(rate));
        }
        
        Ok(bounded_rate)
    }
    
    /// Time step with CFL condition enforcement
    pub fn timestep(&mut self, field: &mut DebrisDensityField) -> Result<(), KesslerError> {
        let mut iterations = 0;
        let max_dt = self.compute_max_dt(field)?;
        
        // Adaptive time stepping to satisfy CFL
        let dt = field.dt.min(max_dt);
        
        // Create temporary buffer for update (avoid aliasing)
        let mut new_data = vec![0.0; field.data.len()].into_boxed_slice();
        
        for alt_idx in 0..field.altitude_bins {
            for inc_idx in 0..field.inclination_bins {
                let density = field.get(alt_idx, inc_idx);
                
                // Compute source term (fragmentation from collisions)
                let v_rel = self.estimate_relative_velocity(alt_idx);
                let collision_rate = self.collision_rate(density, v_rel)?;
                let source = collision_rate * self.fragmentation_yield;
                
                // Compute diffusion term (orbital perturbations)
                let diffusion = self.compute_diffusion(field, alt_idx, inc_idx);
                
                // Update: dn/dt = source - sink + diffusion
                let sink = collision_rate; // Loss due to collisions
                let new_density = density + dt * (source - sink + diffusion);
                
                // Enforce physical bounds
                let bounded = new_density.max(0.0).min(MAX_DEBRIS_DENSITY);
                new_data[alt_idx * field.inclination_bins + inc_idx] = bounded;
                
                iterations += 1;
                if iterations > MAX_ITERATIONS {
                    return Err(KesslerError::NumericalInstability);
                }
            }
        }
        
        // Commit update
        field.data = new_data;
        field.dt = dt;
        field.time += dt;
        
        Ok(())
    }
    
    /// Compute maximum stable time step (CFL condition)
    fn compute_max_dt(&self, field: &DebrisDensityField) -> Result<f64, KesslerError> {
        // Find maximum density for stability analysis
        let max_density = field.data.iter().cloned().fold(0.0_f64, f64::max);
        
        if max_density <= 0.0 {
            return Ok(1.0); // No constraint if empty
        }
        
        // CFL: dt < dx / (n * σ * v)
        let dx = self.grid_altitude_km * 1000.0; // Convert to meters
        let v_rel = 7800.0; // Typical LEO velocity m/s
        let sigma = 10.0; // Cross-section m²
        
        let max_rate = max_density * sigma * v_rel * self.collision_coefficient;
        
        if max_rate <= 0.0 {
            return Ok(1.0);
        }
        
        let cfl_number = max_rate * 1.0; // Characteristic time scale
        let dt_max = CFL_NUMBER_LIMIT / cfl_number.max(1e-6);
        
        // Cap at reasonable value
        Ok(dt_max.min(86400.0)) // Max 1 day
    }
    
    /// Estimate relative velocity at altitude bin
    fn estimate_relative_velocity(&self, alt_idx: usize) -> f64 {
        let altitude = LEO_MIN_ALT_KM + alt_idx as f64 * self.grid_altitude_km;
        let orbital_radius = (EARTH_RADIUS_KM + altitude) * 1000.0;
        
        // Orbital velocity
        let mu = 3.986e14; // m³/s²
        let v_orbital = (mu / orbital_radius).sqrt();
        
        // Relative velocity is typically fraction of orbital velocity
        v_orbital * 0.1 // ~10% for random inclinations
    }
    
    /// Compute diffusion term from orbital perturbations
    fn compute_diffusion(&self, field: &DebrisDensityField, alt_idx: usize, inc_idx: usize) -> f64 {
        // Simplified diffusion model
        let diffusion_coef = 1e-6;
        
        let center = field.get(alt_idx, inc_idx);
        let left = if alt_idx > 0 { field.get(alt_idx - 1, inc_idx) } else { center };
        let right = if alt_idx + 1 < field.altitude_bins {
            field.get(alt_idx + 1, inc_idx)
        } else {
            center
        };
        
        // Second derivative approximation
        diffusion_coef * (left - 2.0 * center + right)
    }
    
    /// Check if Kessler syndrome threshold is exceeded
    pub fn check_kessler_threshold(&self, field: &DebrisDensityField, threshold: f64) -> bool {
        let total_density: f64 = field.data.iter().sum();
        let avg_density = total_density / field.data.len() as f64;
        avg_density > threshold
    }
}

impl Default for KesslerBoltzmannSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_collision_rate() {
        let solver = KesslerBoltzmannSolver::new();
        let rate = solver.collision_rate(1000.0, 7800.0);
        assert!(rate.is_ok());
        assert!(rate.unwrap() >= 0.0);
    }
    
    #[test]
    fn test_timestep() {
        let mut solver = KesslerBoltzmannSolver::new();
        let mut field = DebrisDensityField::new(20, 18).unwrap();
        
        // Initialize with some debris
        for i in 0..field.altitude_bins {
            for j in 0..field.inclination_bins {
                field.set(i, j, 100.0);
            }
        }
        
        field.dt = 3600.0; // 1 hour
        let result = solver.timestep(&mut field);
        assert!(result.is_ok());
    }
}
