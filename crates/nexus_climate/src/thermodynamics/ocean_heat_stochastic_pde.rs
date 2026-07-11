//! Stochastic Ocean Heat Content PDE solver
//! Models heat diffusion in ocean layers with stochastic forcing

use alloc::vec::Vec;
use core::fmt;

/// Error types for ocean heat PDE operations
#[derive(Debug, Clone, PartialEq)]
pub enum OceanHeatError {
    InvalidGridSize,
    NegativeDiffusivity,
    CFLViolation { max_dt: f64, requested_dt: f64 },
    BoundaryError,
    NumericalInstability,
}

impl fmt::Display for OceanHeatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGridSize => write!(f, "Invalid grid size"),
            Self::NegativeDiffusivity => write!(f, "Negative diffusivity not allowed"),
            Self::CFLViolation { max_dt, requested_dt } => {
                write!(f, "CFL violation: max dt={:.2e}, requested={:.2e}", max_dt, requested_dt)
            }
            Self::BoundaryError => write!(f, "Boundary condition error"),
            Self::NumericalInstability => write!(f, "Numerical instability detected"),
        }
    }
}

/// Ocean layer configuration
#[derive(Debug, Clone)]
pub struct OceanLayerConfig {
    pub depth: f64,           // Layer depth in meters
    pub n_vertical: usize,    // Number of vertical grid points
    pub n_horizontal: usize,  // Number of horizontal grid points
    pub diffusivity: f64,     // Thermal diffusivity (m²/s)
    pub advection_velocity: f64, // Mean advection velocity (m/s)
}

/// Stochastic Ocean Heat Model state
pub struct StochasticOceanHeatModel {
    config: OceanLayerConfig,
    /// Temperature field [n_vertical * n_horizontal]
    temperature: Box<[f64]>,
    /// Stochastic forcing amplitude
    noise_amplitude: f64,
    /// Grid spacing
    dz: f64,
    dx: f64,
    /// Maximum stable time step (CFL limit)
    max_dt: f64,
}

impl StochasticOceanHeatModel {
    /// Create new ocean heat model
    pub fn new(config: OceanLayerConfig, initial_temp: f64) -> Result<Self, OceanHeatError> {
        if config.n_vertical < 2 || config.n_horizontal < 2 {
            return Err(OceanHeatError::InvalidGridSize);
        }
        if config.diffusivity < 0.0 {
            return Err(OceanHeatError::NegativeDiffusivity);
        }

        let n_total = config.n_vertical * config.n_horizontal;
        let mut temperature = vec![initial_temp; n_total];

        // Add some initial gradient (warmer at surface)
        for z in 0..config.n_vertical {
            let depth_frac = z as f64 / (config.n_vertical - 1) as f64;
            for x in 0..config.n_horizontal {
                temperature[z * config.n_horizontal + x] = 
                    initial_temp + 10.0 * (1.0 - depth_frac);
            }
        }

        let dz = config.depth / (config.n_vertical - 1) as f64;
        let dx = 1000.0 / (config.n_horizontal - 1) as f64; // Assume 1000km domain

        // CFL condition for explicit diffusion: dt <= dx² / (2*D)
        let dt_diff_z = dz * dz / (2.0 * config.diffusivity.max(1e-10));
        let dt_diff_x = dx * dx / (2.0 * config.diffusivity.max(1e-10));
        
        // CFL for advection: dt <= dx / |u|
        let dt_adv = if config.advection_velocity.abs() > 1e-10 {
            dx / config.advection_velocity.abs()
        } else {
            f64::MAX
        };

        let max_dt = dt_diff_z.min(dt_diff_x).min(dt_adv) * 0.8; // Safety factor

        Ok(Self {
            config,
            temperature: temperature.into_boxed_slice(),
            noise_amplitude: 0.0,
            dz,
            dx,
            max_dt,
        })
    }

    /// Set stochastic forcing amplitude
    pub fn set_noise_amplitude(&mut self, amplitude: f64) {
        self.noise_amplitude = amplitude.max(0.0);
    }

    /// Get maximum stable time step
    pub fn max_dt(&self) -> f64 {
        self.max_dt
    }

    /// Advance one time step with stochastic forcing
    /// Uses explicit finite difference with Euler-Maruyama for noise
    pub fn step(&mut self, dt: f64, rng_value: f64) -> Result<(), OceanHeatError> {
        if dt > self.max_dt * 1.01 {
            return Err(OceanHeatError::CFLViolation {
                max_dt: self.max_dt,
                requested_dt: dt,
            });
        }

        let n_v = self.config.n_vertical;
        let n_h = self.config.n_horizontal;
        let d = self.config.diffusivity;
        let u = self.config.advection_velocity;

        let mut new_temp = vec![0.0_f64; n_v * n_h];

        for z in 0..n_v {
            for x in 0..n_h {
                let idx = z * n_h + x;
                let t = self.temperature[idx];

                // Vertical diffusion (second derivative)
                let d2t_dz2 = if z == 0 {
                    // Surface boundary: fixed flux
                    let t_below = self.temperature[(z + 1) * n_h + x];
                    2.0 * (t_below - t) / (self.dz * self.dz)
                } else if z == n_v - 1 {
                    // Bottom boundary: no flux
                    let t_above = self.temperature[(z - 1) * n_h + x];
                    2.0 * (t_above - t) / (self.dz * self.dz)
                } else {
                    let t_above = self.temperature[(z - 1) * n_h + x];
                    let t_below = self.temperature[(z + 1) * n_h + x];
                    (t_above - 2.0 * t + t_below) / (self.dz * self.dz)
                };

                // Horizontal diffusion
                let d2t_dx2 = if x == 0 {
                    let t_right = self.temperature[z * n_h + x + 1];
                    2.0 * (t_right - t) / (self.dx * self.dx)
                } else if x == n_h - 1 {
                    let t_left = self.temperature[z * n_h + x - 1];
                    2.0 * (t_left - t) / (self.dx * self.dx)
                } else {
                    let t_left = self.temperature[z * n_h + x - 1];
                    let t_right = self.temperature[z * n_h + x + 1];
                    (t_left - 2.0 * t + t_right) / (self.dx * self.dx)
                };

                // Advection (upwind scheme)
                let dt_dx = if u > 0.0 {
                    if x == 0 {
                        0.0
                    } else {
                        let t_upwind = self.temperature[z * n_h + x - 1];
                        -u * (t - t_upwind) / self.dx
                    }
                } else if u < 0.0 {
                    if x == n_h - 1 {
                        0.0
                    } else {
                        let t_upwind = self.temperature[z * n_h + x + 1];
                        -u * (t_upwind - t) / self.dx
                    }
                } else {
                    0.0
                };

                // Stochastic forcing (additive noise scaled by sqrt(dt))
                let noise = self.noise_amplitude * rng_value * dt.sqrt();

                // Update
                new_temp[idx] = t + dt * (d * (d2t_dz2 + d2t_dx2) + dt_dx) + noise;

                // Clamp to physical bounds
                new_temp[idx] = new_temp[idx].clamp(-2.0, 35.0);
            }
        }

        self.temperature = new_temp.into_boxed_slice();
        Ok(())
    }

    /// Get temperature at specific location
    pub fn get_temperature(&self, z: usize, x: usize) -> Option<f64> {
        if z >= self.config.n_vertical || x >= self.config.n_horizontal {
            return None;
        }
        Some(self.temperature[z * self.config.n_horizontal + x])
    }

    /// Get vertically integrated heat content (relative to reference)
    pub fn integrated_heat_content(&self, reference_temp: f64) -> f64 {
        let rho = 1025.0; // kg/m³
        let cp = 3985.0; // J/(kg·K)
        
        let mut total = 0.0;
        for z in 0..self.config.n_vertical {
            for x in 0..self.config.n_horizontal {
                let delta_t = self.temperature[z * self.config.n_horizontal + x] - reference_temp;
                total += delta_t;
            }
        }
        
        // Scale by volume and properties
        total * rho * cp * self.dz * self.dx / 1e9 // Convert to GJ
    }

    /// Get full temperature field
    pub fn temperature_field(&self) -> &[f64] {
        &self.temperature
    }
}

/// AMOC (Atlantic Meridional Overturning Circulation) strength estimator
pub struct AmocEstimator {
    /// Salinity difference between North Atlantic and South Atlantic
    delta_s: f64,
    /// Temperature difference
    delta_t: f64,
    /// Reference AMOC strength (Sv)
    reference_strength: f64,
}

impl AmocEstimator {
    pub fn new(reference_strength: f64) -> Self {
        Self {
            delta_s: 0.0,
            delta_t: 0.0,
            reference_strength,
        }
    }

    /// Update salinity and temperature differences
    pub fn update(&mut self, delta_s: f64, delta_t: f64) {
        self.delta_s = delta_s;
        self.delta_t = delta_t;
    }

    /// Estimate AMOC strength using simple scaling relation
    /// Based on thermal wind balance
    pub fn estimate_strength(&self) -> f64 {
        // Simplified scaling: AMOC ~ g * alpha * delta_T - beta * delta_S
        let g = 9.81;
        let alpha = 2.07e-4; // Thermal expansion coefficient
        let beta = 7.6e-4;   // Haline contraction coefficient
        
        let buoyancy_forcing = g * (alpha * self.delta_t - beta * self.delta_s);
        
        // Linear scaling from reference
        let reference_buoyancy = 1e-6; // Reference value
        if reference_buoyancy.abs() < 1e-15 {
            return self.reference_strength;
        }
        
        self.reference_strength * (buoyancy_forcing / reference_buoyancy).clamp(0.1, 2.0)
    }

    /// Check for AMOC collapse threshold
    pub fn is_near_collapse(&self, threshold_fraction: f64) -> bool {
        let current = self.estimate_strength();
        current < self.reference_strength * threshold_fraction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocean_heat_model() {
        let config = OceanLayerConfig {
            depth: 4000.0,
            n_vertical: 20,
            n_horizontal: 50,
            diffusivity: 1e-4,
            advection_velocity: 0.01,
        };

        let mut model = StochasticOceanHeatModel::new(config, 10.0).unwrap();
        assert!(model.max_dt() > 0.0);

        // Step forward
        for _ in 0..10 {
            let result = model.step(model.max_dt() * 0.5, 0.0);
            assert!(result.is_ok());
        }
    }
}
