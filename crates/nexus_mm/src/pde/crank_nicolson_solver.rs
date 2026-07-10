//! Crank-Nicolson Finite Difference Solver for HJB PDE.
//! Zero-allocation implementation using pre-allocated buffers.
//! Solves the Avellaneda-Stoikov HJB equation backward in time.

use crate::pde::hjb_equation::{
    AvellanedaStoikovParams, BoundaryConditions, GridConfig, PdeError,
};
use crate::pde::thomas_algorithm::solve_tridiagonal_zero_alloc;

/// Configuration for the Crank-Nicolson solver
#[derive(Debug, Clone)]
pub struct CrankNicolsonConfig {
    /// Damping factor for numerical stability (theta = 0.5 is pure CN)
    pub theta: f64,
    /// Maximum iterations for convergence check
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for CrankNicolsonConfig {
    fn default() -> Self {
        Self {
            theta: 0.5, // Pure Crank-Nicolson
            max_iterations: 1000,
            tolerance: 1e-8,
        }
    }
}

/// Workspace buffers for zero-allocation solving
pub struct SolverWorkspace {
    /// Current time slice of value function
    pub v_current: Vec<f64>,
    /// Next time slice of value function
    pub v_next: Vec<f64>,
    /// Lower diagonal of tridiagonal matrix
    pub lower_diag: Vec<f64>,
    /// Main diagonal of tridiagonal matrix
    pub main_diag: Vec<f64>,
    /// Upper diagonal of tridiagonal matrix
    pub upper_diag: Vec<f64>,
    /// Right-hand side vector
    pub rhs: Vec<f64>,
    /// Temporary workspace for Thomas algorithm
    pub thomas_c: Vec<f64>,
    pub thomas_d: Vec<f64>,
}

impl SolverWorkspace {
    pub fn new(n_inventory: usize) -> Self {
        Self {
            v_current: vec![0.0; n_inventory],
            v_next: vec![0.0; n_inventory],
            lower_diag: vec![0.0; n_inventory - 1],
            main_diag: vec![0.0; n_inventory],
            upper_diag: vec![0.0; n_inventory - 1],
            rhs: vec![0.0; n_inventory],
            thomas_c: vec![0.0; n_inventory - 1],
            thomas_d: vec![0.0; n_inventory],
        }
    }
}

/// Crank-Nicolson solver for the HJB PDE
pub struct CrankNicolsonSolver {
    config: CrankNicolsonConfig,
    grid: GridConfig,
    params: AvellanedaStoikovParams,
    workspace: SolverWorkspace,
}

impl CrankNicolsonSolver {
    pub fn new(
        grid: GridConfig,
        params: AvellanedaStoikovParams,
        config: CrankNicolsonConfig,
    ) -> Self {
        let workspace = SolverWorkspace::new(grid.n_inventory);
        
        Self {
            config,
            grid,
            params,
            workspace,
        }
    }
    
    /// Solve the HJB PDE backward from terminal time to t=0
    /// Returns the value function V(q, t) on the grid
    pub fn solve(&mut self, boundary: &BoundaryConditions) -> Result<Vec<f64>, PdeError> {
        let n_inv = self.grid.n_inventory;
        let n_time = self.grid.n_time;
        
        // Initialize with terminal condition
        self.workspace.v_current.copy_from_slice(&boundary.terminal_condition);
        
        // Time-stepping loop (backward in time)
        for t_idx in (0..n_time - 1).rev() {
            // Build tridiagonal system for this time step
            self.build_tridiagonal_system(t_idx, boundary)?;
            
            // Solve using Thomas algorithm (zero-allocation)
            solve_tridiagonal_zero_alloc(
                &self.workspace.lower_diag,
                &self.workspace.main_diag,
                &self.workspace.upper_diag,
                &self.workspace.rhs,
                &mut self.workspace.v_next,
                &mut self.workspace.thomas_c,
                &mut self.workspace.thomas_d,
            )?;
            
            // Enforce boundary conditions strictly to prevent probability mass leakage
            self.enforce_boundary_conditions(t_idx, boundary);
            
            // Check for NaN/Inf (numerical instability)
            if self.check_numerical_stability()? {
                return Err(PdeError::NumericalInstability);
            }
            
            // Swap buffers
            std::mem::swap(&mut self.workspace.v_current, &mut self.workspace.v_next);
        }
        
        Ok(self.workspace.v_current.clone())
    }
    
    /// Build the tridiagonal system for one time step
    fn build_tridiagonal_system(
        &mut self,
        t_idx: usize,
        boundary: &BoundaryConditions,
    ) -> Result<(), PdeError> {
        let n = self.grid.n_inventory;
        let dt = self.grid.dt;
        let dq = self.grid.dq;
        let theta = self.config.theta;
        
        // Coefficients from HJB equation
        // dV/dt + 0.5*sigma^2*d²V/dq² - gamma*q*sigma*dV/dq + ... = 0
        let sigma_sq = self.params.sigma * self.params.sigma;
        let gamma = self.params.gamma;
        
        // Precompute constant factors
        let dt_dq_sq = dt / (dq * dq);
        let dt_dq = dt / dq;
        
        // Build tridiagonal matrix coefficients
        for i in 0..n {
            let q = self.grid.inventory_from_index(i) as f64;
            
            // Diffusion coefficient: 0.5 * sigma^2
            let diff_coef = 0.5 * sigma_sq;
            
            // Convection coefficient: -gamma * q * sigma^2 (from inventory risk term)
            let conv_coef = -gamma * q * sigma_sq;
            
            // Upwind scheme for stability when convection dominates
            let peclet = (conv_coef * dq / diff_coef).abs();
            let upwind_factor = if peclet > 2.0 { 
                (peclet - 2.0) / peclet 
            } else { 
                0.0 
            };
            
            // Lower diagonal (i-1)
            if i > 0 {
                let mut a_i = theta * diff_coef * dt_dq_sq;
                
                // Add upwind contribution for convection
                if conv_coef > 0.0 {
                    a_i += theta * conv_coef.abs() * dt_dq * upwind_factor;
                }
                
                self.workspace.lower_diag[i - 1] = -a_i;
            }
            
            // Main diagonal (i)
            let mut b_i = 1.0 + 2.0 * theta * diff_coef * dt_dq_sq;
            
            // Add convection contribution
            if conv_coef > 0.0 {
                b_i += theta * conv_coef * dt_dq * (1.0 - upwind_factor);
            } else if conv_coef < 0.0 {
                b_i -= theta * conv_coef * dt_dq * (1.0 - upwind_factor);
            }
            
            self.workspace.main_diag[i] = b_i;
            
            // Upper diagonal (i+1)
            if i < n - 1 {
                let mut c_i = theta * diff_coef * dt_dq_sq;
                
                // Add upwind contribution for convection
                if conv_coef < 0.0 {
                    c_i += theta * conv_coef.abs() * dt_dq * upwind_factor;
                }
                
                self.workspace.upper_diag[i] = -c_i;
            }
            
            // Right-hand side: explicit part + previous time step
            let mut rhs_i = self.workspace.v_current[i];
            
            // Explicit diffusion term
            if i > 0 && i < n - 1 {
                let v_prev = self.workspace.v_current[i - 1];
                let v_curr = self.workspace.v_current[i];
                let v_next = self.workspace.v_current[i + 1];
                
                let laplacian = (v_prev - 2.0 * v_curr + v_next) / (dq * dq);
                let convection = if conv_coef > 0.0 {
                    (v_curr - v_prev) / dq
                } else if conv_coef < 0.0 {
                    (v_next - v_curr) / dq
                } else {
                    0.0
                };
                
                rhs_i -= (1.0 - theta) * dt * (diff_coef * laplacian + conv_coef * convection);
            }
            
            // Handle boundaries
            if i == 0 {
                rhs_i += theta * diff_coef * dt_dq_sq * boundary.left_boundary[t_idx];
            } else if i == n - 1 {
                rhs_i += theta * diff_coef * dt_dq_sq * boundary.right_boundary[t_idx];
            }
            
            self.workspace.rhs[i] = rhs_i;
        }
        
        Ok(())
    }
    
    /// Enforce boundary conditions strictly
    fn enforce_boundary_conditions(&mut self, t_idx: usize, boundary: &BoundaryConditions) {
        let n = self.grid.n_inventory;
        
        // Left boundary (minimum inventory)
        self.workspace.v_next[0] = boundary.left_boundary[t_idx];
        
        // Right boundary (maximum inventory)
        self.workspace.v_next[n - 1] = boundary.right_boundary[t_idx];
    }
    
    /// Check for numerical instability (NaN or Inf values)
    fn check_numerical_stability(&self) -> Result<bool, PdeError> {
        for &val in &self.workspace.v_next {
            if val.is_nan() || val.is_infinite() {
                return Ok(true); // Instability detected
            }
        }
        Ok(false)
    }
    
    /// Extract reservation price at current time for given inventory
    pub fn get_reservation_price(
        &self,
        solution: &[f64],
        inventory: i64,
        mid_price: f64,
    ) -> Option<f64> {
        let idx = self.grid.index_from_inventory(inventory)?;
        
        if idx >= solution.len() {
            return None;
        }
        
        // Reservation price includes skew from value function gradient
        let value = solution[idx];
        let skew = self.compute_skew(inventory, idx, solution);
        
        Some(mid_price + skew + value)
    }
    
    /// Compute optimal bid and ask prices
    pub fn get_optimal_quotes(
        &self,
        solution: &[f64],
        inventory: i64,
        mid_price: f64,
    ) -> Option<(f64, f64)> {
        let reservation = self.get_reservation_price(solution, inventory, mid_price)?;
        
        let half_spread = self.params.optimal_spread() / 2.0;
        
        Some((reservation - half_spread, reservation + half_spread))
    }
    
    /// Compute skew from value function gradient
    fn compute_skew(&self, inventory: i64, idx: usize, solution: &[f64]) -> f64 {
        if idx == 0 || idx >= solution.len() - 1 {
            return 0.0;
        }
        
        // Central difference for gradient
        let dq = self.grid.dq;
        let v_prev = solution[idx - 1];
        let v_next = solution[idx + 1];
        
        let delta = (v_next - v_prev) / (2.0 * dq);
        
        // Skew proportional to delta and inventory
        -delta * inventory as f64 * self.params.gamma
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_solver_initialization() {
        let params = AvellanedaStoikovParams::new(
            0.1, 0.02, 1.0, 0.5, 1.0, 10
        ).unwrap();
        let grid = GridConfig::new(10, 100, 1.0).unwrap();
        let config = CrankNicolsonConfig::default();
        
        let solver = CrankNicolsonSolver::new(grid, params, config);
        
        assert_eq!(solver.workspace.v_current.len(), 21);
    }
    
    #[test]
    fn test_boundary_enforcement() {
        let params = AvellanedaStoikovParams::new(
            0.1, 0.02, 1.0, 0.5, 1.0, 10
        ).unwrap();
        let grid = GridConfig::new(10, 100, 1.0).unwrap();
        let config = CrankNicolsonConfig::default();
        
        let mut solver = CrankNicolsonSolver::new(grid, params.clone(), config);
        let boundary = BoundaryConditions::new(&solver.grid, &params, 100.0);
        
        // Manually set some values
        solver.workspace.v_next.fill(0.5);
        solver.enforce_boundary_conditions(0, &boundary);
        
        // Boundaries should match exactly
        assert!((solver.workspace.v_next[0] - boundary.left_boundary[0]).abs() < 1e-15);
        assert!((solver.workspace.v_next[20] - boundary.right_boundary[0]).abs() < 1e-15);
    }
}
