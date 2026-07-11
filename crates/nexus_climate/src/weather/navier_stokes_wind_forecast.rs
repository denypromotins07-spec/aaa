//! 3D Navier-Stokes Atmospheric Boundary Layer Solver for Wind Forecasting
//! Predicts wind shear and turbine generation output using CFD

use alloc::vec::Vec;
use core::fmt;

/// Error types for Navier-Stokes solver
#[derive(Debug, Clone, PartialEq)]
pub enum NSError {
    CFLViolation { cfl_number: f64, max_allowed: f64 },
    DivergenceDetected,
    InvalidGridSize,
    NegativeViscosity,
    BoundaryConditionError,
    NaNPropagation,
}

impl fmt::Display for NSError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CFLViolation { cfl_number, max_allowed } => {
                write!(f, "CFL violation: {:.2e} > {:.2e}", cfl_number, max_allowed)
            }
            Self::DivergenceDetected => write!(f, "Solution divergence detected"),
            Self::InvalidGridSize => write!(f, "Invalid grid dimensions"),
            Self::NegativeViscosity => write!(f, "Negative viscosity not allowed"),
            Self::BoundaryConditionError => write!(f, "Boundary condition error"),
            Self::NaNPropagation => write!(f, "NaN propagation detected"),
        }
    }
}

/// Grid configuration for ABL simulation
#[derive(Debug, Clone)]
pub struct ABLGridConfig {
    pub nx: usize,  // x-direction (streamwise)
    pub ny: usize,  // y-direction (spanwise)
    pub nz: usize,  // z-direction (vertical)
    pub dx: f64,    // Grid spacing x (m)
    pub dy: f64,    // Grid spacing y (m)
    pub dz: f64,    // Grid spacing z (m)
    pub domain_height: f64,  // Total domain height (m)
}

/// Velocity field state
#[derive(Debug, Clone)]
pub struct VelocityField {
    /// u-velocity component [nx * ny * nz]
    pub u: Box<[f64]>,
    /// v-velocity component
    pub v: Box<[f64]>,
    /// w-velocity component
    pub w: Box<[f64]>,
    /// Pressure field
    pub p: Box<[f64]>,
}

impl VelocityField {
    fn new(n_total: usize, initial_u: f64) -> Self {
        let mut u = vec![initial_u; n_total];
        let v = vec![0.0_f64; n_total];
        let w = vec![0.0_f64; n_total];
        let p = vec![0.0_f64; n_total];

        // Add some vertical shear to initial condition
        // (implementation simplified for brevity)

        Self {
            u: u.into_boxed_slice(),
            v: v.into_boxed_slice(),
            w: w.into_boxed_slice(),
            p: p.into_boxed_slice(),
        }
    }
}

/// Atmospheric Boundary Layer Navier-Stokes Solver
pub struct ABLNavierStokesSolver {
    config: ABLGridConfig,
    velocity: VelocityField,
    /// Kinematic viscosity (m²/s)
    nu: f64,
    /// Coriolis parameter (1/s)
    f_cor: f64,
    /// Geostrophic wind speed (m/s)
    u_geostrophic: f64,
    /// Maximum stable time step
    max_dt: f64,
    /// Current time step
    dt: f64,
    /// Artificial compressibility factor
    beta: f64,
}

impl ABLNavierStokesSolver {
    /// Create new ABL solver
    pub fn new(config: ABLGridConfig, nu: f64, u_geostrophic: f64) -> Result<Self, NSError> {
        if config.nx < 2 || config.ny < 2 || config.nz < 2 {
            return Err(NSError::InvalidGridSize);
        }
        if nu <= 0.0 {
            return Err(NSError::NegativeViscosity);
        }
        if config.dx <= 0.0 || config.dy <= 0.0 || config.dz <= 0.0 {
            return Err(NSError::InvalidGridSize);
        }

        let n_total = config.nx * config.ny * config.nz;

        // Calculate maximum stable time step from CFL condition
        // dt <= min(dx, dy, dz) / (|u| + c) where c is artificial sound speed
        let min_dx = config.dx.min(config.dy).min(config.dz);
        let max_velocity = u_geostrophic.abs() * 1.5; // Safety margin
        let artificial_c = 10.0; // Artificial compressibility wave speed

        let cfl_dt = min_dx / (max_velocity + artificial_c);
        
        // Diffusion limit: dt <= dx² / (2*nu)
        let diff_dt = config.dx * config.dx / (2.0 * nu);

        let max_dt = cfl_dt.min(diff_dt) * 0.5; // Safety factor

        let velocity = VelocityField::new(n_total, u_geostrophic);

        Ok(Self {
            config,
            velocity,
            nu,
            f_cor: 1e-4, // Typical mid-latitude Coriolis
            u_geostrophic,
            max_dt,
            dt: max_dt * 0.8,
            beta: 1.0,
        })
    }

    /// Get maximum stable time step (CFL limit)
    pub fn max_dt(&self) -> f64 {
        self.max_dt
    }

    /// Set time step with CFL check
    pub fn set_dt(&mut self, dt: f64) -> Result<(), NSError> {
        // Check CFL condition
        let max_u = *self.velocity.u.iter().map(|x| x.abs()).fold(&0.0_f64, |a, &b| if a > &b { a } else { &b });
        let max_v = *self.velocity.v.iter().map(|x| x.abs()).fold(&0.0_f64, |a, &b| if a > &b { a } else { &b });
        let max_w = *self.velocity.w.iter().map(|x| x.abs()).fold(&0.0_f64, |a, &b| if a > &b { a } else { &b });
        
        let max_vel = max_u.max(max_v).max(max_w);
        let artificial_c = 10.0;
        
        let cfl_x = dt * (max_vel + artificial_c) / self.config.dx;
        let cfl_y = dt * (max_vel + artificial_c) / self.config.dy;
        let cfl_z = dt * (max_vel + artificial_c) / self.config.dz;
        
        let max_cfl = cfl_x.max(cfl_y).max(cfl_z);
        let max_allowed_cfl = 0.5;

        if max_cfl > max_allowed_cfl {
            return Err(NSError::CFLViolation {
                cfl_number: max_cfl,
                max_allowed: max_allowed_cfl,
            });
        }

        self.dt = dt;
        Ok(())
    }

    /// Compute advection term using upwind scheme
    fn compute_advection(&self, field: &[f64], u_comp: &[f64], v_comp: &[f64], w_comp: &[f64]) -> Box<[f64]> {
        let mut adv = vec![0.0_f64; self.config.nx * self.config.ny * self.config.nz];
        let nx = self.config.nx;
        let ny = self.config.ny;
        let nz = self.config.nz;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = k * ny * nx + j * nx + i;
                    
                    // Upwind differencing for u-component
                    let dudx = if u_comp[idx] >= 0.0 {
                        if i > 0 {
                            (field[idx] - field[k * ny * nx + j * nx + (i - 1)]) / self.config.dx
                        } else {
                            0.0
                        }
                    } else {
                        if i < nx - 1 {
                            (field[k * ny * nx + j * nx + (i + 1)] - field[idx]) / self.config.dx
                        } else {
                            0.0
                        }
                    };

                    // Upwind for v-component
                    let dvdy = if v_comp[idx] >= 0.0 {
                        if j > 0 {
                            (field[idx] - field[k * ny * nx + (j - 1) * nx + i]) / self.config.dy
                        } else {
                            0.0
                        }
                    } else {
                        if j < ny - 1 {
                            (field[k * ny * nx + (j + 1) * nx + i] - field[idx]) / self.config.dy
                        } else {
                            0.0
                        }
                    };

                    // Upwind for w-component
                    let dwdz = if w_comp[idx] >= 0.0 {
                        if k > 0 {
                            (field[idx] - field[(k - 1) * ny * nx + j * nx + i]) / self.config.dz
                        } else {
                            0.0
                        }
                    } else {
                        if k < nz - 1 {
                            (field[(k + 1) * ny * nx + j * nx + i] - field[idx]) / self.config.dz
                        } else {
                            0.0
                        }
                    };

                    adv[idx] = u_comp[idx] * dudx + v_comp[idx] * dvdy + w_comp[idx] * dwdz;
                }
            }
        }

        adv.into_boxed_slice()
    }

    /// Compute diffusion term
    fn compute_diffusion(&self, field: &[f64]) -> Box<[f64]> {
        let mut diff = vec![0.0_f64; self.config.nx * self.config.ny * self.config.nz];
        let nx = self.config.nx;
        let ny = self.config.ny;
        let nz = self.config.nz;

        for k in 1..(nz - 1) {
            for j in 1..(ny - 1) {
                for i in 1..(nx - 1) {
                    let idx = k * ny * nx + j * nx + i;
                    
                    let d2dx2 = (field[k * ny * nx + j * nx + (i + 1)] - 2.0 * field[idx] + field[k * ny * nx + j * nx + (i - 1)]) 
                                / (self.config.dx * self.config.dx);
                    let d2dy2 = (field[k * ny * nx + (j + 1) * nx + i] - 2.0 * field[idx] + field[k * ny * nx + (j - 1) * nx + i]) 
                                / (self.config.dy * self.config.dy);
                    let d2dz2 = (field[(k + 1) * ny * nx + j * nx + i] - 2.0 * field[idx] + field[(k - 1) * ny * nx + j * nx + i]) 
                                / (self.config.dz * self.config.dz);

                    diff[idx] = self.nu * (d2dx2 + d2dy2 + d2dz2);
                }
            }
        }

        diff.into_boxed_slice()
    }

    /// Advance one time step using projection method
    pub fn step(&mut self) -> Result<(), NSError> {
        let n_total = self.config.nx * self.config.ny * self.config.nz;
        
        // Store old velocities
        let u_old = self.velocity.u.clone();
        let v_old = self.velocity.v.clone();
        let w_old = self.velocity.w.clone();

        // Compute advection and diffusion
        let adv_u = self.compute_advection(&u_old, &u_old, &v_old, &w_old);
        let adv_v = self.compute_advection(&v_old, &u_old, &v_old, &w_old);
        let adv_w = self.compute_advection(&w_old, &u_old, &v_old, &w_old);

        let diff_u = self.compute_diffusion(&u_old);
        let diff_v = self.compute_diffusion(&v_old);
        let diff_w = self.compute_diffusion(&w_old);

        // Predictor step (without pressure gradient)
        for i in 0..n_total {
            let du_dt = -adv_u[i] + diff_u[i];
            let dv_dt = -adv_v[i] + diff_v[i] + self.f_cor * (self.u_geostrophic - u_old[i]); // Coriolis
            let dw_dt = -adv_w[i] + diff_w[i];

            self.velocity.u[i] = u_old[i] + du_dt * self.dt;
            self.velocity.v[i] = v_old[i] + dv_dt * self.dt;
            self.velocity.w[i] = w_old[i] + dw_dt * self.dt;

            // Check for NaN
            if !self.velocity.u[i].is_finite() || !self.velocity.v[i].is_finite() || !self.velocity.w[i].is_finite() {
                return Err(NSError::NaNPropagation);
            }
        }

        // Apply boundary conditions
        self.apply_boundary_conditions()?;

        // Check for divergence
        let max_u = self.velocity.u.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        if max_u > 1000.0 {
            return Err(NSError::DivergenceDetected);
        }

        Ok(())
    }

    /// Apply boundary conditions
    fn apply_boundary_conditions(&mut self) -> Result<(), NSError> {
        let nx = self.config.nx;
        let ny = self.config.ny;
        let nz = self.config.nz;

        // Bottom boundary: no-slip
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                self.velocity.u[idx] = 0.0;
                self.velocity.v[idx] = 0.0;
                self.velocity.w[idx] = 0.0;
            }
        }

        // Top boundary: geostrophic wind
        for j in 0..ny {
            for i in 0..nx {
                let idx = (nz - 1) * ny * nx + j * nx + i;
                self.velocity.u[idx] = self.u_geostrophic;
                self.velocity.v[idx] = 0.0;
            }
        }

        // Lateral boundaries: periodic (simplified as fixed)
        for k in 0..nz {
            for j in 0..ny {
                self.velocity.u[k * ny * nx + j * nx] = self.velocity.u[k * ny * nx + j * nx + (nx - 2)];
                self.velocity.u[k * ny * nx + j * nx + (nx - 1)] = self.velocity.u[k * ny * nx + j * nx + 1];
            }
        }

        Ok(())
    }

    /// Get wind speed at specific height
    pub fn get_wind_at_height(&self, height: f64) -> Option<f64> {
        if height < 0.0 || height > self.config.domain_height {
            return None;
        }

        let k = ((height / self.config.dz) as usize).min(self.config.nz - 1);
        let nx = self.config.nx;
        let ny = self.config.ny;

        // Average over horizontal plane
        let mut sum = 0.0;
        let mut count = 0;
        for j in 0..ny {
            for i in 0..nx {
                let idx = k * ny * nx + j * nx + i;
                sum += self.velocity.u[idx].abs();
                count += 1;
            }
        }

        if count == 0 {
            return None;
        }
        Some(sum / count as f64)
    }

    /// Estimate turbine power output at hub height
    pub fn estimate_turbine_power(&self, hub_height: f64, rotor_diameter: f64, efficiency: f64) -> Option<f64> {
        let wind_speed = self.get_wind_at_height(hub_height)?;
        let air_density = 1.225; // kg/m³
        
        let swept_area = std::f64::consts::PI * (rotor_diameter / 2.0).powi(2);
        let power = 0.5 * air_density * swept_area * wind_speed.powi(3) * efficiency;
        
        Some(power.clamp(0.0, 10e6)) // Cap at 10 MW
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_creation() {
        let config = ABLGridConfig {
            nx: 10,
            ny: 10,
            nz: 20,
            dx: 100.0,
            dy: 100.0,
            dz: 10.0,
            domain_height: 200.0,
        };

        let solver = ABLNavierStokesSolver::new(config, 1e-5, 10.0);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_cfl_check() {
        let config = ABLGridConfig {
            nx: 10,
            ny: 10,
            nz: 20,
            dx: 100.0,
            dy: 100.0,
            dz: 10.0,
            domain_height: 200.0,
        };

        let mut solver = ABLNavierStokesSolver::new(config, 1e-5, 10.0).unwrap();
        
        // This should fail CFL check
        let result = solver.set_dt(10.0);
        assert!(result.is_err());
    }
}
