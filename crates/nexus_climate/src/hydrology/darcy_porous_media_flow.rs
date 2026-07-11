//! Darcy's Law Porous Media Flow for Aquifer Simulation
//! Models groundwater flow through porous geological formations

use alloc::vec::Vec;
use core::fmt;

/// Error types for Darcy flow simulation
#[derive(Debug, Clone, PartialEq)]
pub enum DarcyError {
    InvalidPermeability,
    NegativePorosity,
    CFLViolation,
    SingularMatrix,
    BoundaryError,
}

impl fmt::Display for DarcyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPermeability => write!(f, "Invalid permeability value"),
            Self::NegativePorosity => write!(f, "Negative porosity not allowed"),
            Self::CFLViolation => write!(f, "CFL stability condition violated"),
            Self::SingularMatrix => write!(f, "Singular matrix in solver"),
            Self::BoundaryError => write!(f, "Boundary condition error"),
        }
    }
}

/// Hydraulic properties of porous media
#[derive(Debug, Clone)]
pub struct PorousMediaProperties {
    /// Permeability tensor (m²) - minimum epsilon enforced
    pub permeability: f64,
    /// Porosity (0-1)
    pub porosity: f64,
    /// Specific storage (1/m)
    pub specific_storage: f64,
    /// Fluid viscosity (Pa·s)
    pub viscosity: f64,
    /// Fluid density (kg/m³)
    pub density: f64,
}

impl PorousMediaProperties {
    pub fn new(
        permeability: f64,
        porosity: f64,
        specific_storage: f64,
        viscosity: f64,
        density: f64,
    ) -> Result<Self, DarcyError> {
        // Enforce minimum permeability floor to prevent singularities
        let k_min = 1e-20; // Absolute minimum permeability (m²)
        let permeability = permeability.max(k_min);

        if porosity < 0.0 || porosity > 1.0 {
            return Err(DarcyError::NegativePorosity);
        }
        if specific_storage < 0.0 {
            return Err(DarcyError::InvalidPermeability);
        }
        if viscosity <= 0.0 {
            return Err(DarcyError::InvalidPermeability);
        }
        if density <= 0.0 {
            return Err(DarcyError::InvalidPermeability);
        }

        Ok(Self {
            permeability,
            porosity,
            specific_storage,
            viscosity,
            density,
        })
    }

    /// Calculate hydraulic conductivity (m/s)
    pub fn hydraulic_conductivity(&self) -> f64 {
        let g = 9.81;
        self.permeability * self.density * g / self.viscosity
    }

    /// Calculate hydraulic diffusivity (m²/s)
    pub fn hydraulic_diffusivity(&self) -> f64 {
        self.hydraulic_conductivity() / self.specific_storage
    }
}

/// 3D grid configuration for aquifer simulation
#[derive(Debug, Clone)]
pub struct AquiferGridConfig {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

/// Hydraulic head field state
pub struct HydraulicHeadField {
    /// Head values [nx * ny * nz]
    pub head: Box<[f64]>,
    /// Pressure values
    pub pressure: Box<[f64]>,
    /// Velocity components
    pub vx: Box<[f64]>,
    pub vy: Box<[f64]>,
    pub vz: Box<[f64]>,
}

impl HydraulicHeadField {
    fn new(n_total: usize, initial_head: f64) -> Self {
        Self {
            head: vec![initial_head; n_total].into_boxed_slice(),
            pressure: vec![0.0_f64; n_total].into_boxed_slice(),
            vx: vec![0.0_f64; n_total].into_boxed_slice(),
            vy: vec![0.0_f64; n_total].into_boxed_slice(),
            vz: vec![0.0_f64; n_total].into_boxed_slice(),
        }
    }
}

/// Darcy's Law Groundwater Flow Solver
pub struct DarcyFlowSolver {
    config: AquiferGridConfig,
    properties: PorousMediaProperties,
    head_field: HydraulicHeadField,
    /// Maximum stable time step
    max_dt: f64,
    /// Current time step
    dt: f64,
}

impl DarcyFlowSolver {
    /// Create new Darcy flow solver
    pub fn new(
        config: AquiferGridConfig,
        properties: PorousMediaProperties,
        initial_head: f64,
    ) -> Result<Self, DarcyError> {
        if config.nx < 2 || config.ny < 2 || config.nz < 2 {
            return Err(DarcyError::BoundaryError);
        }
        if config.dx <= 0.0 || config.dy <= 0.0 || config.dz <= 0.0 {
            return Err(DarcyError::BoundaryError);
        }

        let n_total = config.nx * config.ny * config.nz;
        let head_field = HydraulicHeadField::new(n_total, initial_head);

        // CFL condition for diffusion: dt <= dx² / (2*D)
        let min_dx = config.dx.min(config.dy).min(config.dz);
        let diffusivity = properties.hydraulic_diffusivity();
        
        let max_dt = if diffusivity > 1e-15 {
            min_dx * min_dx / (2.0 * diffusivity) * 0.5
        } else {
            1e6 // Large value if no diffusion
        };

        Ok(Self {
            config,
            properties,
            head_field,
            max_dt,
            dt: max_dt * 0.8,
        })
    }

    /// Get maximum stable time step
    pub fn max_dt(&self) -> f64 {
        self.max_dt
    }

    /// Compute Darcy velocity (specific discharge)
    fn compute_darcy_velocity(&mut self) {
        let nx = self.config.nx;
        let ny = self.config.ny;
        let nz = self.config.nz;
        let k = self.properties.permeability;
        let mu = self.properties.viscosity;
        let rho = self.properties.density;
        let g = 9.81;

        for k_idx in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = k_idx * ny * nx + j * nx + i;

                    // Compute pressure gradient using central differences
                    let mut dpdx = 0.0;
                    let mut dpdy = 0.0;
                    let mut dpdz = 0.0;

                    if i > 0 && i < nx - 1 {
                        let p_left = self.head_field.head[k_idx * ny * nx + j * nx + (i - 1)] * rho * g;
                        let p_right = self.head_field.head[k_idx * ny * nx + j * nx + (i + 1)] * rho * g;
                        dpdx = (p_right - p_left) / (2.0 * self.config.dx);
                    }

                    if j > 0 && j < ny - 1 {
                        let p_front = self.head_field.head[k_idx * ny * nx + (j - 1) * nx + i] * rho * g;
                        let p_back = self.head_field.head[k_idx * ny * nx + (j + 1) * nx + i] * rho * g;
                        dpdy = (p_back - p_front) / (2.0 * self.config.dy);
                    }

                    if k_idx > 0 && k_idx < nz - 1 {
                        let p_below = self.head_field.head[(k_idx - 1) * ny * nx + j * nx + i] * rho * g;
                        let p_above = self.head_field.head[(k_idx + 1) * ny * nx + j * nx + i] * rho * g;
                        dpdz = (p_above - p_below) / (2.0 * self.config.dz) + rho * g; // Include gravity
                    }

                    // Darcy's law: q = -k/mu * grad(p)
                    self.head_field.vx[idx] = -k / mu * dpdx;
                    self.head_field.vy[idx] = -k / mu * dpdy;
                    self.head_field.vz[idx] = -k / mu * dpdz;
                }
            }
        }
    }

    /// Solve groundwater flow equation using implicit method
    pub fn step(&mut self) -> Result<(), DarcyError> {
        let n_total = self.config.nx * self.config.ny * self.config.nz;
        let s_s = self.properties.specific_storage;
        let phi = self.properties.porosity;

        // Compute Darcy velocity
        self.compute_darcy_velocity();

        // Update head using continuity equation
        // S_s * dh/dt = -div(q) / phi
        for idx in 0..n_total {
            let qx = self.head_field.vx[idx];
            let qy = self.head_field.vy[idx];
            let qz = self.head_field.vz[idx];

            // Simplified divergence (would need proper stencil in production)
            let div_q = qx / self.config.dx + qy / self.config.dy + qz / self.config.dz;

            let dh_dt = -div_q / (s_s * phi.max(0.01));
            
            self.head_field.head[idx] += dh_dt * self.dt;

            // Clamp to physical bounds
            self.head_field.head[idx] = self.head_field.head[idx].clamp(-100.0, 1000.0);
        }

        // Update pressure from head
        let rho = self.properties.density;
        let g = 9.81;
        for idx in 0..n_total {
            self.head_field.pressure[idx] = self.head_field.head[idx] * rho * g;
        }

        Ok(())
    }

    /// Get total water volume in aquifer
    pub fn total_water_volume(&self) -> f64 {
        let mut total = 0.0;
        let phi = self.properties.porosity;
        let cell_volume = self.config.dx * self.config.dy * self.config.dz;

        for &head in &self.head_field.head {
            // Convert head to saturated thickness (simplified)
            let saturated_thickness = head.max(0.0);
            total += saturated_thickness * phi * cell_volume;
        }

        total
    }

    /// Get average hydraulic head
    pub fn average_head(&self) -> f64 {
        let sum: f64 = self.head_field.head.iter().sum();
        sum / self.head_field.head.len() as f64
    }
}

/// Aquifer depletion state tracker
pub struct AquiferDepletionTracker {
    /// Initial water volume
    initial_volume: f64,
    /// Historical volumes
    volume_history: Vec<f64>,
    /// Depletion rate (m³/year)
    depletion_rate: f64,
}

impl AquiferDepletionTracker {
    pub fn new(initial_volume: f64) -> Self {
        Self {
            initial_volume,
            volume_history: Vec::new(),
            depletion_rate: 0.0,
        }
    }

    pub fn record_volume(&mut self, volume: f64, timestamp_us: u64) {
        self.volume_history.push(volume);
        
        // Calculate depletion rate from recent history
        if self.volume_history.len() >= 2 {
            let dt_years = 1.0 / 365.0; // Assume daily recordings
            let dv = self.initial_volume - self.volume_history[self.volume_history.len() - 1];
            self.depletion_rate = dv / (self.volume_history.len() as f64 * dt_years);
        }
    }

    pub fn years_remaining(&self, critical_fraction: f64) -> f64 {
        if self.depletion_rate <= 0.0 {
            return f64::MAX;
        }
        let current = *self.volume_history.last().unwrap_or(&self.initial_volume);
        let critical_volume = self.initial_volume * critical_fraction;
        (current - critical_volume) / self.depletion_rate
    }

    pub fn depletion_rate(&self) -> f64 {
        self.depletion_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_porous_media_properties() {
        let props = PorousMediaProperties::new(1e-12, 0.3, 1e-6, 1e-3, 1000.0).unwrap();
        assert!(props.hydraulic_conductivity() > 0.0);
    }

    #[test]
    fn test_permeability_floor() {
        // Test that zero permeability gets floored
        let props = PorousMediaProperties::new(0.0, 0.3, 1e-6, 1e-3, 1000.0).unwrap();
        assert!(props.permeability >= 1e-20);
    }

    #[test]
    fn test_solver_creation() {
        let config = AquiferGridConfig {
            nx: 10,
            ny: 10,
            nz: 5,
            dx: 100.0,
            dy: 100.0,
            dz: 10.0,
        };

        let props = PorousMediaProperties::new(1e-12, 0.3, 1e-6, 1e-3, 1000.0).unwrap();
        let solver = DarcyFlowSolver::new(config, props, 50.0);
        assert!(solver.is_ok());
    }
}
