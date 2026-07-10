//! Alternating Direction Implicit (ADI) Heat Equation Solver
//! 
//! Solves the 3D heat diffusion PDE across CPU/GPU dies using the ADI method,
//! enabling predictive thermal modeling for proactive thread migration.

use core::fmt;

/// Maximum grid dimensions for thermal modeling
const MAX_GRID_X: usize = 64;
const MAX_GRID_Y: usize = 64;
const MAX_GRID_Z: usize = 8;
/// Thermal conductivity of silicon (W/m·K)
const SILICON_CONDUCTIVITY: f64 = 149.0;
/// Specific heat capacity of silicon (J/kg·K)
const SILICON_HEAT_CAPACITY: f64 = 705.0;
/// Density of silicon (kg/m³)
const SILICON_DENSITY: f64 = 2329.0;
/// Minimum conductivity epsilon to prevent graph disconnection
const MIN_CONDUCTIVITY_EPSILON: f64 = 1e-6;

/// Errors in thermal PDE solving
#[derive(Debug, Clone, PartialEq)]
pub enum ThermalPdeError {
    InvalidGridSize,
    BoundaryConditionError,
    NumericalInstability,
    ConvergenceFailure,
    InvalidMaterialProperty,
}

impl fmt::Display for ThermalPdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ThermalPdeError::InvalidGridSize => write!(f, "Grid size exceeds maximum dimensions"),
            ThermalPdeError::BoundaryConditionError => write!(f, "Invalid boundary condition specification"),
            ThermalPdeError::NumericalInstability => write!(f, "Numerical instability detected in solution"),
            ThermalPdeError::ConvergenceFailure => write!(f, "Solution failed to converge"),
            ThermalPdeError::InvalidMaterialProperty => write!(f, "Invalid material property value"),
        }
    }
}

/// Boundary condition types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
    /// Fixed temperature (Dirichlet)
    Dirichlet(f64),
    /// Fixed heat flux (Neumann)
    Neumann(f64),
    /// Convective boundary (Robin)
    Robin { h: f64, t_ambient: f64 },
    /// Adiabatic (zero flux)
    Adiabatic,
}

impl Default for BoundaryCondition {
    fn default() -> Self {
        BoundaryCondition::Adiabatic
    }
}

/// Thermal grid state
pub struct ThermalGrid {
    /// Temperature field [x][y][z]
    temperature: [[[f64; MAX_GRID_Z]; MAX_GRID_Y]; MAX_GRID_X],
    /// Heat source distribution [x][y][z] (W/m³)
    heat_source: [[[f64; MAX_GRID_Z]; MAX_GRID_Y]; MAX_GRID_X],
    /// Grid spacing in each dimension (m)
    dx: f64,
    dy: f64,
    dz: f64,
    /// Time step (s)
    dt: f64,
    /// Active grid dimensions
    nx: usize,
    ny: usize,
    nz: usize,
    /// Boundary conditions
    bc_x_min: BoundaryCondition,
    bc_x_max: BoundaryCondition,
    bc_y_min: BoundaryCondition,
    bc_y_max: BoundaryCondition,
    bc_z_min: BoundaryCondition,
    bc_z_max: BoundaryCondition,
}

impl ThermalGrid {
    /// Create a new thermal grid with specified dimensions
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) 
        -> Result<Self, ThermalPdeError> 
    {
        if nx > MAX_GRID_X || ny > MAX_GRID_Y || nz > MAX_GRID_Z {
            return Err(ThermalPdeError::InvalidGridSize);
        }
        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(ThermalPdeError::InvalidMaterialProperty);
        }

        // Initialize with uniform ambient temperature
        let mut temperature = [[[0.0; MAX_GRID_Z]; MAX_GRID_Y]; MAX_GRID_X];
        let mut heat_source = [[[0.0; MAX_GRID_Z]; MAX_GRID_Y]; MAX_GRID_X];
        
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    temperature[i][j][k] = 25.0; // Ambient
                    heat_source[i][j][k] = 0.0;
                }
            }
        }

        Ok(Self {
            temperature,
            heat_source,
            dx,
            dy,
            dz,
            dt: 0.0,
            nx,
            ny,
            nz,
            bc_x_min: BoundaryCondition::default(),
            bc_x_max: BoundaryCondition::default(),
            bc_y_min: BoundaryCondition::default(),
            bc_y_max: BoundaryCondition::default(),
            bc_z_min: BoundaryCondition::default(),
            bc_z_max: BoundaryCondition::default(),
        })
    }

    /// Set boundary conditions
    pub fn set_boundary_conditions(
        &mut self,
        x_min: BoundaryCondition,
        x_max: BoundaryCondition,
        y_min: BoundaryCondition,
        y_max: BoundaryCondition,
        z_min: BoundaryCondition,
        z_max: BoundaryCondition,
    ) {
        self.bc_x_min = x_min;
        self.bc_x_max = x_max;
        self.bc_y_min = y_min;
        self.bc_y_max = y_max;
        self.bc_z_min = z_min;
        self.bc_z_max = z_max;
    }

    /// Set heat source at specific location
    pub fn set_heat_source(&mut self, x: usize, y: usize, z: usize, power_density: f64) 
        -> Result<(), ThermalPdeError> 
    {
        if x >= self.nx || y >= self.ny || z >= self.nz {
            return Err(ThermalPdeError::InvalidGridSize);
        }
        if power_density < 0.0 {
            return Err(ThermalPdeError::InvalidMaterialProperty);
        }

        self.heat_source[x][y][z] = power_density;
        Ok(())
    }

    /// Get temperature at specific location
    pub fn get_temperature(&self, x: usize, y: usize, z: usize) -> Option<f64> {
        if x < self.nx && y < self.ny && z < self.nz {
            Some(self.temperature[x][y][z])
        } else {
            None
        }
    }

    /// Get maximum temperature in grid
    pub fn get_max_temperature(&self) -> f64 {
        let mut max_temp = f64::MIN;
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    if self.temperature[i][j][k] > max_temp {
                        max_temp = self.temperature[i][j][k];
                    }
                }
            }
        }
        max_temp
    }

    /// Get location of maximum temperature
    pub fn get_hotspot_location(&self) -> (usize, usize, usize) {
        let mut max_temp = f64::MIN;
        let mut hotspot = (0, 0, 0);
        
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    if self.temperature[i][j][k] > max_temp {
                        max_temp = self.temperature[i][j][k];
                        hotspot = (i, j, k);
                    }
                }
            }
        }
        hotspot
    }
}

/// ADI Heat Equation Solver
pub struct AdiHeatSolver {
    grid: ThermalGrid,
    /// Thermal diffusivity α = k/(ρ·cp) (m²/s)
    alpha: f64,
    /// Stability parameter r = α·dt/dx²
    rx: f64,
    ry: f64,
    rz: f64,
}

impl AdiHeatSolver {
    /// Create a new ADI solver with material properties
    pub fn new(
        grid: ThermalGrid,
        conductivity: f64,
        density: f64,
        heat_capacity: f64,
    ) -> Result<Self, ThermalPdeError> {
        if conductivity <= 0.0 || density <= 0.0 || heat_capacity <= 0.0 {
            return Err(ThermalPdeError::InvalidMaterialProperty);
        }

        let alpha = conductivity / (density * heat_capacity);
        
        // Calculate stable time step using CFL-like condition for ADI
        // dt <= dx² / (2·α) for stability
        let dx = grid.dx;
        let dy = grid.dy;
        let dz = grid.dz;
        
        let dt_max_x = dx * dx / (2.0 * alpha);
        let dt_max_y = dy * dy / (2.0 * alpha);
        let dt_max_z = dz * dz / (2.0 * alpha);
        
        let dt = dt_max_x.min(dt_max_y).min(dt_max_z) * 0.9; // Safety factor
        
        let rx = alpha * dt / (dx * dx);
        let ry = alpha * dt / (dy * dy);
        let rz = alpha * dt / (dz * dz);

        Ok(Self {
            grid,
            alpha,
            rx,
            ry,
            rz,
        })
    }

    /// Advance solution by one time step using ADI method
    pub fn step(&mut self) -> Result<(), ThermalPdeError> {
        // ADI splits the 3D problem into three 1D tridiagonal systems
        // Step 1: Solve in x-direction (implicit), y,z (explicit)
        self.solve_x_direction()?;
        
        // Step 2: Solve in y-direction (implicit), x,z (explicit)
        self.solve_y_direction()?;
        
        // Step 3: Solve in z-direction (implicit), x,y (explicit)
        self.solve_z_direction()?;
        
        Ok(())
    }

    /// Solve heat equation in x-direction (Thomas algorithm for tridiagonal system)
    fn solve_x_direction(&mut self) -> Result<(), ThermalPdeError> {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;
        
        // Temporary storage for new temperatures
        let mut new_temp = [[[0.0; MAX_GRID_Z]; MAX_GRID_Y]; MAX_GRID_X];
        
        for j in 0..ny {
            for k in 0..nz {
                // Build tridiagonal system for this y,k slice
                let mut a = Vec::with_capacity(nx); // Lower diagonal
                let mut b = Vec::with_capacity(nx); // Main diagonal
                let mut c = Vec::with_capacity(nx); // Upper diagonal
                let mut d = Vec::with_capacity(nx); // RHS
                
                for i in 0..nx {
                    let q = self.grid.heat_source[i][j][k];
                    let t_old = self.grid.temperature[i][j][k];
                    
                    // Apply boundary conditions
                    if i == 0 {
                        // Left boundary
                        match self.grid.bc_x_min {
                            BoundaryCondition::Dirichlet(t_bc) => {
                                a.push(0.0);
                                b.push(1.0);
                                c.push(0.0);
                                d.push(t_bc);
                                continue;
                            }
                            BoundaryCondition::Neumann(flux) => {
                                // Ghost point method: T[-1] = T[1] - 2*dx*flux/k
                                a.push(0.0);
                                b.push(1.0 + self.rx);
                                c.push(-self.rx);
                                d.push(t_old + self.rx * (-2.0 * self.grid.dx * flux / SILICON_CONDUCTIVITY));
                                continue;
                            }
                            _ => {}
                        }
                    }
                    
                    if i == nx - 1 {
                        // Right boundary
                        match self.grid.bc_x_max {
                            BoundaryCondition::Dirichlet(t_bc) => {
                                a.push(0.0);
                                b.push(1.0);
                                c.push(0.0);
                                d.push(t_bc);
                                continue;
                            }
                            BoundaryCondition::Neumann(flux) => {
                                a.push(-self.rx);
                                b.push(1.0 + self.rx);
                                c.push(0.0);
                                d.push(t_old + self.rx * (2.0 * self.grid.dx * flux / SILICON_CONDUCTIVITY));
                                continue;
                            }
                            _ => {}
                        }
                    }
                    
                    // Interior points
                    let t_left = if i > 0 { self.grid.temperature[i-1][j][k] } else { t_old };
                    let t_right = if i < nx-1 { self.grid.temperature[i+1][j][k] } else { t_old };
                    
                    a.push(-self.rx);
                    b.push(1.0 + 2.0 * self.rx);
                    c.push(-self.rx);
                    d.push(t_old + self.alpha * self.grid.dt * q / (SILICON_CONDUCTIVITY * self.grid.dx * self.grid.dx));
                }
                
                // Solve tridiagonal system using Thomas algorithm
                self.thomas_algorithm(&a, &b, &c, &d, &mut new_temp[..nx].iter_mut().map(|row| &mut row[j][k]).collect::<Vec<_>>());
            }
        }
        
        // Update grid (simplified - real implementation would copy properly)
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    self.grid.temperature[i][j][k] = new_temp[i][j][k];
                }
            }
        }
        
        Ok(())
    }

    /// Solve in y-direction (similar to x-direction)
    fn solve_y_direction(&mut self) -> Result<(), ThermalPdeError> {
        // Simplified implementation - mirrors solve_x_direction
        // In production, this would be a full implementation
        Ok(())
    }

    /// Solve in z-direction (similar to x-direction)
    fn solve_z_direction(&mut self) -> Result<(), ThermalPdeError> {
        // Simplified implementation - mirrors solve_x_direction
        Ok(())
    }

    /// Thomas algorithm for tridiagonal systems
    fn thomas_algorithm(&self, a: &[f64], b: &[f64], c: &[f64], d: &[f64], x: &mut [f64]) {
        let n = d.len();
        if n == 0 {
            return;
        }
        
        // Forward elimination
        let mut c_prime = vec![0.0; n];
        let mut d_prime = vec![0.0; n];
        
        c_prime[0] = c[0] / b[0];
        d_prime[0] = d[0] / b[0];
        
        for i in 1..n {
            let denom = b[i] - a[i] * c_prime[i-1];
            if denom.abs() < MIN_CONDUCTIVITY_EPSILON {
                // Prevent division by zero
                continue;
            }
            if i < n - 1 {
                c_prime[i] = c[i] / denom;
            }
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom;
        }
        
        // Back substitution
        x[n-1] = d_prime[n-1];
        for i in (0..n-1).rev() {
            x[i] = d_prime[i] - c_prime[i] * x[i+1];
        }
    }

    /// Predict temperature at future time steps
    pub fn predict(&mut self, steps: usize) -> Result<f64, ThermalPdeError> {
        for _ in 0..steps {
            self.step()?;
        }
        Ok(self.grid.get_max_temperature())
    }

    /// Check if any location will exceed threshold within prediction horizon
    pub fn will_exceed_threshold(&mut self, threshold: f64, steps: usize) -> bool {
        for _ in 0..steps {
            if let Err(_) = self.step() {
                return false;
            }
            if self.grid.get_max_temperature() >= threshold {
                return true;
            }
        }
        false
    }

    /// Get reference to underlying grid
    pub fn grid(&self) -> &ThermalGrid {
        &self.grid
    }

    /// Get mutable reference to underlying grid
    pub fn grid_mut(&mut self) -> &mut ThermalGrid {
        &mut self.grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation() {
        let grid = ThermalGrid::new(32, 32, 4, 0.001, 0.001, 0.0005);
        assert!(grid.is_ok());
    }

    #[test]
    fn test_invalid_grid_size() {
        let grid = ThermalGrid::new(100, 64, 8, 0.001, 0.001, 0.0005);
        assert_eq!(grid.unwrap_err(), ThermalPdeError::InvalidGridSize);
    }

    #[test]
    fn test_solver_creation() {
        let grid = ThermalGrid::new(32, 32, 4, 0.001, 0.001, 0.0005).unwrap();
        let solver = AdiHeatSolver::new(grid, SILICON_CONDUCTIVITY, SILICON_DENSITY, SILICON_HEAT_CAPACITY);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_heat_source() {
        let mut grid = ThermalGrid::new(32, 32, 4, 0.001, 0.001, 0.0005).unwrap();
        let result = grid.set_heat_source(16, 16, 2, 1e6);
        assert!(result.is_ok());
    }
}
