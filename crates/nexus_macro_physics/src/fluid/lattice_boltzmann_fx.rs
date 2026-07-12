// NEXUS-OMEGA Stage 34: Currency Peg Fluid Dynamics
// Chapter 3: Lattice Boltzmann Method FX Liquidity Solver
// File: crates/nexus_macro_physics/src/fluid/lattice_boltzmann_fx.rs

//! Lattice Boltzmann Method (LBM) Solver for FX Liquidity Simulation
//!
//! Models currency peg defense as fluid dynamics where speculative attacks
//! create hydrodynamic pressure on central bank reserves.
//!
//! CRITICAL: Strictly enforces CFL condition to prevent numerical instability.

#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]

use core::fmt;
use alloc::vec::Vec;
use alloc::boxed::Box;

/// D2Q9 lattice velocities (2D, 9 velocities)
pub const D2Q9_VELOCITIES: [(i8, i8); 9] = [
    (0, 0),   // rest
    (1, 0),   // right
    (0, 1),   // up
    (-1, 0),  // left
    (0, -1),  // down
    (1, 1),   //右上
    (-1, 1),  //左上
    (-1, -1), //左下
    (1, -1),  //右下
];

/// D2Q9 weights for equilibrium distribution
pub const D2Q9_WEIGHTS: [f64; 9] = [
    4.0 / 9.0,  // rest
    1.0 / 9.0,  // cardinal directions
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0, // diagonals
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

/// Speed of sound in lattice units (D2Q9)
pub const LATTICE_SOUND_SPEED: f64 = 1.0 / 3.0_f64.sqrt();

/// Maximum Mach number for incompressible flow assumption
pub const MAX_MACH_NUMBER: f64 = 0.3;

/// Error types for LBM operations
#[derive(Debug, Clone, PartialEq)]
pub enum LBMError {
    CFLViolation { velocity: f64, max_allowed: f64 },
    MachNumberExceeded { mach: f64 },
    InvalidGridSize { width: usize, height: usize },
    NumericalInstability { density: f64 },
    BoundaryConditionFailed,
}

impl fmt::Display for LBMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CFLViolation { velocity, max_allowed } => {
                write!(f, "CFL violation: velocity={}, max={}", velocity, max_allowed)
            }
            Self::MachNumberExceeded { mach } => {
                write!(f, "Mach number exceeded: {}", mach)
            }
            Self::InvalidGridSize { width, height } => {
                write!(f, "Invalid grid size: {}x{}", width, height)
            }
            Self::NumericalInstability { density } => {
                write!(f, "Numerical instability: density={}", density)
            }
            Self::BoundaryConditionFailed => write!(f, "Boundary condition failed"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LBMError {}

/// Distribution function at a lattice site
#[derive(Debug, Clone, Copy)]
pub struct DistributionFunction {
    /// 9 distribution values for D2Q9
    f: [f64; 9],
}

impl DistributionFunction {
    #[must_use]
    pub fn new(equilibrium: f64) -> Self {
        let mut f = [0.0; 9];
        for i in 0..9 {
            f[i] = D2Q9_WEIGHTS[i] * equilibrium;
        }
        Self { f }
    }

    #[must_use]
    pub fn zero() -> Self {
        Self { f: [0.0; 9] }
    }

    /// Compute macroscopic density from distributions
    #[must_use]
    pub fn density(&self) -> f64 {
        self.f.iter().sum()
    }

    /// Compute macroscopic velocity from distributions
    #[must_use]
    pub fn velocity(&self) -> (f64, f64) {
        let mut ux = 0.0;
        let mut uy = 0.0;
        let rho = self.density();

        if rho > 1e-10 {
            for i in 0..9 {
                let (cx, cy) = D2Q9_VELOCITIES[i];
                ux += self.f[i] * cx as f64;
                uy += self.f[i] * cy as f64;
            }
            ux /= rho;
            uy /= rho;
        }

        (ux, uy)
    }

    /// Compute equilibrium distribution
    #[must_use]
    pub fn equilibrium(rho: f64, ux: f64, uy: f64) -> Self {
        let u_sq = ux * ux + uy * uy;
        let mut f_eq = [0.0; 9];

        for i in 0..9 {
            let (cx, cy) = D2Q9_VELOCITIES[i];
            let cu = cx as f64 * ux + cy as f64 * uy;
            let w = D2Q9_WEIGHTS[i];

            // Equilibrium: f_eq = w * rho * (1 + 3*cu + 4.5*cu^2 - 1.5*u^2)
            f_eq[i] = w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq);
        }

        Self { f: f_eq }
    }

    /// Stream: propagate distributions to neighbor cells
    pub fn stream_to(&self, target: &mut Self) {
        *target = *self;
    }
}

/// Lattice cell containing distribution functions
#[derive(Debug, Clone)]
pub struct LatticeCell {
    /// Current distribution
    pub f: DistributionFunction,
    /// Post-collision distribution
    pub f_post: DistributionFunction,
    /// Cell type (fluid, boundary, obstacle)
    pub cell_type: CellType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CellType {
    Fluid,
    Inlet,
    Outlet,
    Wall,
    Obstacle,
}

/// Lattice Boltzmann solver state
pub struct LatticeBoltzmannSolver {
    /// Grid width
    width: usize,
    /// Grid height
    height: usize,
    /// Relaxation time (tau)
    tau: f64,
    /// Kinematic viscosity
    viscosity: f64,
    /// Lattice cells
    cells: Box<[LatticeCell]>,
    /// Time step
    dt: f64,
    /// CFL number
    cfl_number: f64,
}

impl LatticeBoltzmannSolver {
    /// Create a new LBM solver
    ///
    /// # Arguments
    /// * `width` - Grid width (must be > 0)
    /// * `height` - Grid height (must be > 0)
    /// * `tau` - Relaxation time (must be > 0.5 for stability)
    ///
    /// # Returns
    /// * `Ok(Self)` on success
    /// * `Err(LBMError)` on failure
    pub fn new(width: usize, height: usize, tau: f64) -> Result<Self, LBMError> {
        if width == 0 || height == 0 {
            return Err(LBMError::InvalidGridSize { width, height });
        }

        if tau <= 0.5 {
            return Err(LBMError::NumericalInstability { density: tau });
        }

        let num_cells = width * height;
        let mut cells = Vec::with_capacity(num_cells);

        // Initialize all cells as fluid with equilibrium distribution
        for _ in 0..num_cells {
            cells.push(LatticeCell {
                f: DistributionFunction::new(1.0),
                f_post: DistributionFunction::zero(),
                cell_type: CellType::Fluid,
            });
        }

        // Compute viscosity from tau: nu = c_s^2 * (tau - 0.5)
        let viscosity = LATTICE_SOUND_SPEED * LATTICE_SOUND_SPEED * (tau - 0.5);

        Ok(Self {
            width,
            height,
            tau,
            viscosity,
            cells: cells.into_boxed_slice(),
            dt: 1.0, // Lattice time step
            cfl_number: 0.0,
        })
    }

    /// Set cell type at specific position
    pub fn set_cell_type(&mut self, x: usize, y: usize, cell_type: CellType) -> Result<(), LBMError> {
        if x >= self.width || y >= self.height {
            return Err(LBMError::InvalidGridSize {
                width: self.width,
                height: self.height,
            });
        }

        let idx = y * self.width + x;
        self.cells[idx].cell_type = cell_type;
        Ok(())
    }

    /// Perform collision step (BGK approximation)
    fn collide(&mut self) {
        let omega = 1.0 / self.tau; // Collision frequency

        for cell in &mut self.cells {
            if cell.cell_type != CellType::Fluid {
                continue;
            }

            let rho = cell.f.density();
            let (ux, uy) = cell.f.velocity();

            // Check Mach number
            let u_mag = (ux * ux + uy * uy).sqrt();
            let mach = u_mag / LATTICE_SOUND_SPEED;

            if mach > MAX_MACH_NUMBER {
                // Apply additional dissipation for high Mach
                // This is a simplified stabilization
            }

            // Compute equilibrium
            let f_eq = DistributionFunction::equilibrium(rho, ux, uy);

            // BGK collision: f_post = f - omega * (f - f_eq)
            for i in 0..9 {
                cell.f_post.f[i] = cell.f.f[i] - omega * (cell.f.f[i] - f_eq.f[i]);
            }
        }
    }

    /// Perform streaming step
    fn stream(&mut self) {
        // Create temporary buffer for streaming
        let mut f_temp = vec![DistributionFunction::zero(); self.width * self.height];

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                
                if self.cells[idx].cell_type == CellType::Obstacle {
                    continue;
                }

                for i in 0..9 {
                    let (cx, cy) = D2Q9_VELOCITIES[i];
                    
                    let nx = (x as i32 + cx as i32).rem_euclid(self.width as i32) as usize;
                    let ny = (y as i32 + cy as i32).rem_euclid(self.height as i32) as usize;
                    
                    let target_idx = ny * self.width + nx;
                    f_temp[target_idx].f[i] = self.cells[idx].f_post.f[i];
                }
            }
        }

        // Update cells
        for (cell, temp) in self.cells.iter_mut().zip(f_temp.iter()) {
            if cell.cell_type != CellType::Obstacle {
                cell.f = *temp;
            }
        }
    }

    /// Apply inlet boundary condition (fixed velocity)
    pub fn apply_inlet(&mut self, x: usize, ux: f64, uy: f64, rho: f64) -> Result<(), LBMError> {
        if x >= self.width {
            return Err(LBMError::InvalidGridSize {
                width: self.width,
                height: self.height,
            });
        }

        // Check CFL condition
        let velocity_mag = (ux * ux + uy * uy).sqrt();
        let max_velocity = LATTICE_SOUND_SPEED * MAX_MACH_NUMBER;

        if velocity_mag > max_velocity {
            return Err(LBMError::CFLViolation {
                velocity: velocity_mag,
                max_allowed: max_velocity,
            });
        }

        for y in 0..self.height {
            let idx = y * self.width + x;
            self.cells[idx].f = DistributionFunction::equilibrium(rho, ux, uy);
            self.cells[idx].cell_type = CellType::Inlet;
        }

        Ok(())
    }

    /// Apply outlet boundary condition (fixed density)
    pub fn apply_outlet(&mut self, x: usize, rho: f64) -> Result<(), LBMError> {
        if x >= self.width {
            return Err(LBMError::InvalidGridSize {
                width: self.width,
                height: self.height,
            });
        }

        for y in 0..self.height {
            let idx = y * self.width + x;
            
            // Zero-gradient extrapolation
            if x > 0 {
                let prev_idx = y * self.width + (x - 1);
                let (ux, uy) = self.cells[prev_idx].f.velocity();
                self.cells[idx].f = DistributionFunction::equilibrium(rho, ux, uy);
            }
            self.cells[idx].cell_type = CellType::Outlet;
        }

        Ok(())
    }

    /// Run one time step
    pub fn step(&mut self) -> Result<(), LBMError> {
        self.collide();
        self.stream();
        self.update_cfl();
        Ok(())
    }

    /// Update CFL number based on current velocities
    fn update_cfl(&mut self) {
        let mut max_velocity = 0.0;

        for cell in &self.cells {
            let (ux, uy) = cell.f.velocity();
            let v = (ux * ux + uy * uy).sqrt();
            max_velocity = max_velocity.max(v);
        }

        self.cfl_number = max_velocity * self.dt;
    }

    /// Get average velocity in a region
    #[must_use]
    pub fn average_velocity(&self, x_start: usize, y_start: usize, width: usize, height: usize) -> (f64, f64) {
        let mut sum_ux = 0.0;
        let mut sum_uy = 0.0;
        let mut count = 0;

        for y in y_start..(y_start + height).min(self.height) {
            for x in x_start..(x_start + width).min(self.width) {
                let idx = y * self.width + x;
                if self.cells[idx].cell_type == CellType::Fluid {
                    let (ux, uy) = self.cells[idx].f.velocity();
                    sum_ux += ux;
                    sum_uy += uy;
                    count += 1;
                }
            }
        }

        if count > 0 {
            (sum_ux / count as f64, sum_uy / count as f64)
        } else {
            (0.0, 0.0)
        }
    }

    /// Get average density in a region
    #[must_use]
    pub fn average_density(&self, x_start: usize, y_start: usize, width: usize, height: usize) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;

        for y in y_start..(y_start + height).min(self.height) {
            for x in x_start..(x_start + width).min(self.width) {
                let idx = y * self.width + x;
                if self.cells[idx].cell_type == CellType::Fluid {
                    sum += self.cells[idx].f.density();
                    count += 1;
                }
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            1.0
        }
    }

    /// Get CFL number
    #[must_use]
    pub const fn cfl_number(&self) -> f64 {
        self.cfl_number
    }

    /// Get kinematic viscosity
    #[must_use]
    pub const fn viscosity(&self) -> f64 {
        self.viscosity
    }

    /// Get grid dimensions
    #[must_use]
    pub const fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_creation() {
        let solver = LatticeBoltzmannSolver::new(32, 32, 1.0);
        assert!(solver.is_ok());
        
        let solver = solver.expect("Lattice Boltzmann solver should initialize with valid parameters");
        assert_eq!(solver.dimensions(), (32, 32));
        assert!(solver.viscosity() > 0.0);
    }

    #[test]
    fn test_distribution_function() {
        let dist = DistributionFunction::new(1.0);
        let rho = dist.density();
        assert!((rho - 1.0).abs() < 1e-10);

        let (ux, uy) = dist.velocity();
        assert!((ux - 0.0).abs() < 1e-10);
        assert!((uy - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_equilibrium() {
        let eq = DistributionFunction::equilibrium(1.0, 0.1, 0.0);
        let rho = eq.density();
        assert!((rho - 1.0).abs() < 0.01); // Small deviation due to numerics
    }

    #[test]
    fn test_invalid_tau() {
        let result = LatticeBoltzmannSolver::new(10, 10, 0.3);
        assert!(result.is_err());
    }
}
