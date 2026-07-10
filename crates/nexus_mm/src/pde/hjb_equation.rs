//! Hamilton-Jacobi-Bellman (HJB) Equation for Avellaneda-Stoikov Market Making.
//! Defines the PDE structure and boundary conditions for optimal quoting.

use crate::pde::crank_nicolson_solver::CrankNicolsonConfig;

/// Error types for PDE operations
#[derive(Debug, Clone, PartialEq)]
pub enum PdeError {
    DimensionMismatch,
    SingularMatrix,
    InvalidBoundary,
    NumericalInstability,
    NanDetected,
}

/// Parameters for the Avellaneda-Stoikov model
#[derive(Debug, Clone)]
pub struct AvellanedaStoikovParams {
    /// Risk aversion coefficient (gamma)
    pub gamma: f64,
    /// Volatility (sigma)
    pub sigma: f64,
    /// Order arrival intensity (lambda)
    pub lambda: f64,
    /// Price elasticity of order flow (kappa)
    pub kappa: f64,
    /// Time horizon (T)
    pub time_horizon: f64,
    /// Maximum inventory position (Q_max)
    pub max_inventory: i64,
}

impl AvellanedaStoikovParams {
    pub fn new(
        gamma: f64,
        sigma: f64,
        lambda: f64,
        kappa: f64,
        time_horizon: f64,
        max_inventory: i64,
    ) -> Result<Self, PdeError> {
        if gamma <= 0.0 || sigma <= 0.0 || lambda <= 0.0 || kappa <= 0.0 || time_horizon <= 0.0 {
            return Err(PdeError::InvalidBoundary);
        }
        
        Ok(Self {
            gamma,
            sigma,
            lambda,
            kappa,
            time_horizon,
            max_inventory,
        })
    }
    
    /// Calculate the drift term for the HJB equation
    #[inline(always)]
    pub fn drift(&self) -> f64 {
        0.5 * self.gamma * self.sigma * self.sigma
    }
    
    /// Calculate the optimal spread adjustment
    #[inline(always)]
    pub fn optimal_spread(&self) -> f64 {
        2.0 / self.gamma + (2.0 / (self.gamma * self.kappa)) * 
            (1.0 - (-self.gamma * self.kappa / (2.0 + self.gamma * self.kappa)).exp()).ln()
    }
}

/// Grid configuration for finite difference method
#[derive(Debug, Clone)]
pub struct GridConfig {
    /// Number of inventory grid points (must be odd for symmetry around 0)
    pub n_inventory: usize,
    /// Number of time grid points
    pub n_time: usize,
    /// Minimum inventory value
    pub min_inventory: i64,
    /// Maximum inventory value
    pub max_inventory: i64,
    /// Time step size
    pub dt: f64,
    /// Inventory step size (always 1 tick/unit)
    pub dq: f64,
}

impl GridConfig {
    pub fn new(max_inventory: i64, n_time: usize, time_horizon: f64) -> Result<Self, PdeError> {
        if max_inventory <= 0 || n_time < 2 || time_horizon <= 0.0 {
            return Err(PdeError::InvalidBoundary);
        }
        
        let n_inventory = (2 * max_inventory + 1) as usize;
        let dt = time_horizon / (n_time - 1) as f64;
        let dq = 1.0; // Inventory changes by 1 unit
        
        Ok(Self {
            n_inventory,
            n_time,
            min_inventory: -max_inventory,
            max_inventory,
            dt,
            dq,
        })
    }
    
    /// Convert inventory index to actual inventory value
    #[inline(always)]
    pub fn inventory_from_index(&self, idx: usize) -> i64 {
        self.min_inventory + idx as i64
    }
    
    /// Convert actual inventory to index
    #[inline(always)]
    pub fn index_from_inventory(&self, q: i64) -> Option<usize> {
        if q < self.min_inventory || q > self.max_inventory {
            return None;
        }
        Some((q - self.min_inventory) as usize)
    }
    
    /// Get time value at time index
    #[inline(always)]
    pub fn time_from_index(&self, idx: usize) -> f64 {
        idx as f64 * self.dt
    }
}

/// Boundary conditions for the HJB PDE
#[derive(Debug, Clone)]
pub struct BoundaryConditions {
    /// Value at minimum inventory boundary
    pub left_boundary: Vec<f64>,
    /// Value at maximum inventory boundary
    pub right_boundary: Vec<f64>,
    /// Terminal condition (at T)
    pub terminal_condition: Vec<f64>,
}

impl BoundaryConditions {
    /// Create boundary conditions for Avellaneda-Stoikov model
    /// Terminal condition: V(q, T) = -exp(-gamma * q * s_T) approximately
    /// For numerical stability, we use a quadratic approximation
    pub fn new(grid: &GridConfig, params: &AvellanedaStoikovParams, current_mid_price: f64) -> Self {
        let mut terminal_condition = vec![0.0; grid.n_inventory];
        let mut left_boundary = vec![0.0; grid.n_time];
        let mut right_boundary = vec![0.0; grid.n_time];
        
        // Terminal condition: quadratic penalty for inventory
        // V(q, T) ≈ -gamma * sigma^2 * q^2 * (T-t) / 2
        for (idx, val) in terminal_condition.iter_mut().enumerate() {
            let q = grid.inventory_from_index(idx);
            *val = -0.5 * params.gamma * params.sigma * params.sigma * (q as f64) * (q as f64);
        }
        
        // Boundary conditions: extrapolate from interior using linear growth
        // Left boundary (q = -Q_max): steep penalty
        for (idx, val) in left_boundary.iter_mut().enumerate() {
            let t = grid.time_from_index(idx);
            let tau = params.time_horizon - t;
            let q = grid.min_inventory as f64;
            *val = -0.5 * params.gamma * params.sigma * params.sigma * q * q * tau;
        }
        
        // Right boundary (q = +Q_max): steep penalty
        for (idx, val) in right_boundary.iter_mut().enumerate() {
            let t = grid.time_from_index(idx);
            let tau = params.time_horizon - t;
            let q = grid.max_inventory as f64;
            *val = -0.5 * params.gamma * params.sigma * params.sigma * q * q * tau;
        }
        
        Self {
            left_boundary,
            right_boundary,
            terminal_condition,
        }
    }
}

/// State variables for HJB solution at a point
#[derive(Debug, Clone, Copy)]
pub struct HjbState {
    /// Value function V(q, t)
    pub value: f64,
    /// First derivative dV/dq
    pub delta: f64,
    /// Second derivative d²V/dq²
    pub gamma_exposure: f64,
    /// Time derivative dV/dt
    pub theta: f64,
}

impl HjbState {
    pub fn new(value: f64, delta: f64, gamma_exposure: f64, theta: f64) -> Self {
        Self {
            value,
            delta,
            gamma_exposure,
            theta,
        }
    }
}

/// Compute the reservation price given the HJB solution
/// r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
#[inline(always)]
pub fn compute_reservation_price(
    mid_price: f64,
    inventory: i64,
    params: &AvellanedaStoikovParams,
    time_remaining: f64,
) -> f64 {
    let skew = inventory as f64 * params.gamma * params.sigma * params.sigma * time_remaining;
    mid_price - skew
}

/// Compute optimal bid-ask spread
#[inline(always)]
pub fn compute_optimal_spread(
    params: &AvellanedaStoikovParams,
    time_remaining: f64,
) -> f64 {
    // Base spread plus inventory adjustment
    let base_spread = 2.0 / params.gamma;
    let inventory_adjustment = (2.0 / (params.gamma * params.kappa)) 
        * (1.0 - (-params.gamma * params.kappa * time_remaining / params.time_horizon).exp());
    
    base_spread + inventory_adjustment
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_params_validation() {
        let params = AvellanedaStoikovParams::new(
            0.1, 0.02, 1.0, 0.5, 1.0, 10
        );
        assert!(params.is_ok());
        
        let invalid = AvellanedaStoikovParams::new(
            -0.1, 0.02, 1.0, 0.5, 1.0, 10
        );
        assert_eq!(invalid.unwrap_err(), PdeError::InvalidBoundary);
    }
    
    #[test]
    fn test_grid_config() {
        let grid = GridConfig::new(10, 100, 1.0).unwrap();
        assert_eq!(grid.n_inventory, 21);
        assert_eq!(grid.min_inventory, -10);
        assert_eq!(grid.max_inventory, 10);
        
        assert_eq!(grid.inventory_from_index(10), 0);
        assert_eq!(grid.inventory_from_index(0), -10);
        assert_eq!(grid.inventory_from_index(20), 10);
    }
    
    #[test]
    fn test_reservation_price() {
        let params = AvellanedaStoikovParams::new(
            0.1, 0.02, 1.0, 0.5, 1.0, 10
        ).unwrap();
        
        let mid_price = 100.0;
        let inventory = 5i64;
        let time_remaining = 0.5;
        
        let r = compute_reservation_price(mid_price, inventory, &params, time_remaining);
        
        // Should be below mid price for positive inventory
        assert!(r < mid_price);
    }
}
