//! Stochastic Temperature PDE Solver
//! 
//! Solves stochastic partial differential equations for temperature
//! and moisture evolution, fusing SAR soil-moisture with thermal feeds.

use std::time::SystemTime;
use thiserror::Error;

/// PDE solver errors
#[derive(Debug, Error)]
pub enum PdeError {
    #[error("Invalid grid dimensions: {0}")]
    InvalidGrid(String),
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),
    #[error("Boundary condition error: {0}")]
    BoundaryConditionError(String),
}

/// Grid point in the simulation domain
#[derive(Debug, Clone)]
pub struct GridPoint {
    pub latitude: f64,
    pub longitude: f64,
    pub depth_meters: f64,
}

/// State variables at each grid point
#[derive(Debug, Clone)]
pub struct StateVariables {
    pub temperature_kelvin: f64,
    pub soil_moisture_fraction: f64, // 0.0 to 1.0
    pub humidity_kg_per_kg: f64,
    pub wind_speed_ms: f64,
    pub solar_radiation_wm2: f64,
}

impl StateVariables {
    pub fn new(
        temperature_kelvin: f64,
        soil_moisture_fraction: f64,
        humidity_kg_per_kg: f64,
        wind_speed_ms: f64,
        solar_radiation_wm2: f64,
    ) -> Result<Self, PdeError> {
        if temperature_kelvin < 0.0 {
            return Err(PdeError::BoundaryConditionError(
                "Temperature must be positive".to_string(),
            ));
        }
        
        if soil_moisture_fraction < 0.0 || soil_moisture_fraction > 1.0 {
            return Err(PdeError::BoundaryConditionError(
                "Soil moisture must be between 0 and 1".to_string(),
            ));
        }
        
        Ok(StateVariables {
            temperature_kelvin,
            soil_moisture_fraction,
            humidity_kg_per_kg,
            wind_speed_ms,
            solar_radiation_wm2,
        })
    }
}

/// 3D simulation grid
pub struct SimulationGrid {
    pub lat_points: usize,
    pub lon_points: usize,
    pub depth_points: usize,
    pub lat_min: f64,
    pub lat_max: f64,
    pub lon_min: f64,
    pub lon_max: f64,
    pub depth_max: f64,
    /// Flattened 3D array of state variables
    pub states: Vec<StateVariables>,
}

impl SimulationGrid {
    pub fn new(
        lat_points: usize,
        lon_points: usize,
        depth_points: usize,
        lat_min: f64,
        lat_max: f64,
        lon_min: f64,
        lon_max: f64,
        depth_max: f64,
        initial_state: StateVariables,
    ) -> Result<Self, PdeError> {
        if lat_points < 2 || lon_points < 2 || depth_points < 1 {
            return Err(PdeError::InvalidGrid(
                "Grid dimensions must be at least 2x2x1".to_string(),
            ));
        }
        
        let total_points = lat_points * lon_points * depth_points;
        let states = vec![initial_state.clone(); total_points];
        
        Ok(SimulationGrid {
            lat_points,
            lon_points,
            depth_points,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            depth_max,
            states,
        })
    }

    /// Get index in flattened array
    fn get_index(&self, lat_idx: usize, lon_idx: usize, depth_idx: usize) -> Option<usize> {
        if lat_idx >= self.lat_points 
            || lon_idx >= self.lon_points 
            || depth_idx >= self.depth_points 
        {
            return None;
        }
        Some(depth_idx * self.lat_points * self.lon_points 
             + lat_idx * self.lon_points 
             + lon_idx)
    }

    /// Get state at grid coordinates
    pub fn get_state(&self, lat_idx: usize, lon_idx: usize, depth_idx: usize) -> Option<&StateVariables> {
        self.get_index(lat_idx, lon_idx, depth_idx)
            .and_then(|idx| self.states.get(idx))
    }

    /// Set state at grid coordinates
    pub fn set_state(
        &mut self,
        lat_idx: usize,
        lon_idx: usize,
        depth_idx: usize,
        state: StateVariables,
    ) -> Result<(), PdeError> {
        let idx = self.get_index(lat_idx, lon_idx, depth_idx)
            .ok_or_else(|| PdeError::InvalidGrid("Index out of bounds".to_string()))?;
        
        self.states[idx] = state;
        Ok(())
    }

    /// Convert physical coordinates to grid indices
    pub fn coords_to_indices(&self, lat: f64, lon: f64, depth: f64) -> (usize, usize, usize) {
        let lat_idx = ((lat - self.lat_min) / (self.lat_max - self.lat_min) 
                       * (self.lat_points - 1) as f64).round() as usize;
        let lon_idx = ((lon - self.lon_min) / (self.lon_max - self.lon_min) 
                       * (self.lon_points - 1) as f64).round() as usize;
        let depth_idx = (depth / self.depth_max * (self.depth_points - 1) as f64).round() as usize;
        
        (
            lat_idx.min(self.lat_points - 1),
            lon_idx.min(self.lon_points - 1),
            depth_idx.min(self.depth_points - 1),
        )
    }
}

/// Stochastic PDE solver using finite difference method
pub struct StochasticTemperaturePde {
    grid: SimulationGrid,
    /// Thermal diffusivity (m²/s)
    thermal_diffusivity: f64,
    /// Moisture diffusivity (m²/s)
    moisture_diffusivity: f64,
    /// Time step (seconds)
    dt: f64,
    /// Random seed for stochastic term
    rng_seed: u64,
}

impl StochasticTemperaturePde {
    pub fn new(
        grid: SimulationGrid,
        thermal_diffusivity: f64,
        moisture_diffusivity: f64,
        dt: f64,
    ) -> Result<Self, PdeError> {
        if dt <= 0.0 {
            return Err(PdeError::NumericalInstability(
                "Time step must be positive".to_string(),
            ));
        }
        
        // Check CFL stability condition
        let dx = (grid.lat_max - grid.lat_min) / (grid.lat_points - 1) as f64;
        let cfl_number = thermal_diffusivity * dt / (dx * dx);
        
        if cfl_number > 0.5 {
            return Err(PdeError::NumericalInstability(
                format!("CFL number {} exceeds stability limit 0.5", cfl_number),
            ));
        }
        
        Ok(StochasticTemperaturePde {
            grid,
            thermal_diffusivity,
            moisture_diffusivity,
            dt,
            rng_seed: 42,
        })
    }

    /// Advance simulation by one time step
    pub fn step(&mut self) -> Result<(), PdeError> {
        let mut new_states = self.grid.states.clone();
        
        for lat_idx in 1..self.grid.lat_points - 1 {
            for lon_idx in 1..self.grid.lon_points - 1 {
                for depth_idx in 0..self.grid.depth_points {
                    let idx = self.grid.get_index(lat_idx, lon_idx, depth_idx)
                        .ok_or_else(|| PdeError::InvalidGrid("Index error".to_string()))?;
                    
                    // Calculate Laplacian for temperature diffusion
                    let temp_laplacian = self.calculate_laplacian_temperature(
                        lat_idx, lon_idx, depth_idx,
                    );
                    
                    // Calculate Laplacian for moisture diffusion
                    let moisture_laplacian = self.calculate_laplacian_moisture(
                        lat_idx, lon_idx, depth_idx,
                    );
                    
                    // Add stochastic forcing term
                    let stochastic_temp = self.stochastic_forcing(idx);
                    
                    // Update temperature using heat equation with stochastic term
                    let current_temp = self.grid.states[idx].temperature_kelvin;
                    new_states[idx].temperature_kelvin = current_temp 
                        + self.dt * self.thermal_diffusivity * temp_laplacian
                        + stochastic_temp * self.dt.sqrt();
                    
                    // Update moisture using diffusion equation
                    let current_moisture = self.grid.states[idx].soil_moisture_fraction;
                    new_states[idx].soil_moisture_fraction = (current_moisture 
                        + self.dt * self.moisture_diffusivity * moisture_laplacian)
                        .clamp(0.0, 1.0);
                }
            }
        }
        
        // Apply boundary conditions
        self.apply_boundary_conditions(&mut new_states)?;
        
        self.grid.states = new_states;
        Ok(())
    }

    /// Calculate Laplacian for temperature
    fn calculate_laplacian_temperature(&self, lat: usize, lon: usize, depth: usize) -> f64 {
        let dx = (self.grid.lat_max - self.grid.lat_min) / (self.grid.lat_points - 1) as f64;
        let dy = (self.grid.lon_max - self.grid.lon_min) / (self.grid.lon_points - 1) as f64;
        
        let center = self.grid.get_state(lat, lon, depth)
            .map(|s| s.temperature_kelvin)
            .unwrap_or(288.0);
        
        let north = self.grid.get_state(lat + 1, lon, depth)
            .map(|s| s.temperature_kelvin)
            .unwrap_or(center);
        let south = self.grid.get_state(lat - 1, lon, depth)
            .map(|s| s.temperature_kelvin)
            .unwrap_or(center);
        let east = self.grid.get_state(lat, lon + 1, depth)
            .map(|s| s.temperature_kelvin)
            .unwrap_or(center);
        let west = self.grid.get_state(lat, lon - 1, depth)
            .map(|s| s.temperature_kelvin)
            .unwrap_or(center);
        
        // 2D Laplacian (horizontal diffusion)
        let laplacian_h = (north - 2.0 * center + south) / (dx * dx)
                        + (east - 2.0 * center + west) / (dy * dy);
        
        // Add vertical diffusion for non-surface layers
        let laplacian_v = if depth > 0 && depth < self.grid.depth_points - 1 {
            let dz = self.grid.depth_max / (self.grid.depth_points - 1) as f64;
            let above = self.grid.get_state(lat, lon, depth - 1)
                .map(|s| s.temperature_kelvin)
                .unwrap_or(center);
            let below = self.grid.get_state(lat, lon, depth + 1)
                .map(|s| s.temperature_kelvin)
                .unwrap_or(center);
            (above - 2.0 * center + below) / (dz * dz)
        } else {
            0.0
        };
        
        laplacian_h + laplacian_v
    }

    /// Calculate Laplacian for moisture
    fn calculate_laplacian_moisture(&self, lat: usize, lon: usize, depth: usize) -> f64 {
        let dx = (self.grid.lat_max - self.grid.lat_min) / (self.grid.lat_points - 1) as f64;
        let dy = (self.grid.lon_max - self.grid.lon_min) / (self.grid.lon_points - 1) as f64;
        
        let center = self.grid.get_state(lat, lon, depth)
            .map(|s| s.soil_moisture_fraction)
            .unwrap_or(0.3);
        
        let north = self.grid.get_state(lat + 1, lon, depth)
            .map(|s| s.soil_moisture_fraction)
            .unwrap_or(center);
        let south = self.grid.get_state(lat - 1, lon, depth)
            .map(|s| s.soil_moisture_fraction)
            .unwrap_or(center);
        let east = self.grid.get_state(lat, lon + 1, depth)
            .map(|s| s.soil_moisture_fraction)
            .unwrap_or(center);
        let west = self.grid.get_state(lat, lon - 1, depth)
            .map(|s| s.soil_moisture_fraction)
            .unwrap_or(center);
        
        (north - 2.0 * center + south) / (dx * dx)
            + (east - 2.0 * center + west) / (dy * dy)
    }

    /// Simple deterministic pseudo-random forcing (would use proper RNG in production)
    fn stochastic_forcing(&self, idx: usize) -> f64 {
        // Linear congruential generator for reproducibility
        let a = 1103515245u64;
        let c = 12345u64;
        let m = 2u64.pow(31);
        
        let seed = (a.wrapping_mul(self.rng_seed + idx as u64).wrapping_add(c)) % m;
        let normalized = (seed as f64 / m as f64) * 2.0 - 1.0; // Range [-1, 1]
        
        normalized * 0.01 // Scale factor for forcing magnitude
    }

    /// Apply boundary conditions
    fn apply_boundary_conditions(&self, states: &mut Vec<StateVariables>) -> Result<(), PdeError> {
        // Dirichlet boundary conditions at surface (top of atmosphere temperature)
        let surface_temp = 288.0; // K
        
        for lat_idx in 0..self.grid.lat_points {
            for lon_idx in 0..self.grid.lon_points {
                if let Some(idx) = self.grid.get_index(lat_idx, lon_idx, 0) {
                    // Relax surface temperature toward boundary value
                    states[idx].temperature_kelvin = 
                        0.9 * states[idx].temperature_kelvin + 0.1 * surface_temp;
                }
            }
        }
        
        Ok(())
    }

    /// Run simulation for specified duration
    pub fn run_simulation(&mut self, steps: usize) -> Result<Vec<f64>, PdeError> {
        let mut avg_temps = Vec::with_capacity(steps);
        
        for _ in 0..steps {
            self.step()?;
            
            // Calculate domain average temperature
            let avg_temp: f64 = self.grid.states.iter()
                .map(|s| s.temperature_kelvin)
                .sum::<f64>() / self.grid.states.len() as f64;
            
            avg_temps.push(avg_temp);
        }
        
        Ok(avg_temps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation() {
        let initial = StateVariables::new(288.0, 0.3, 0.01, 5.0, 200.0).unwrap();
        let grid = SimulationGrid::new(
            10, 10, 5,
            -90.0, 90.0, -180.0, 180.0, 100.0,
            initial,
        ).unwrap();
        
        assert_eq!(grid.lat_points, 10);
        assert_eq!(grid.lon_points, 10);
        assert_eq!(grid.depth_points, 5);
    }

    #[test]
    fn test_cfl_stability_check() {
        let initial = StateVariables::new(288.0, 0.3, 0.01, 5.0, 200.0).unwrap();
        let grid = SimulationGrid::new(
            10, 10, 5,
            -90.0, 90.0, -180.0, 180.0, 100.0,
            initial,
        ).unwrap();
        
        // Large dt should fail CFL check
        let result = StochasticTemperaturePde::new(grid, 1e-5, 1e-6, 1e6);
        assert!(result.is_err());
    }
}
