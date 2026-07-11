//! De Sitter Horizon Thermodynamics
//! 
//! Models the thermodynamic properties of the cosmological event horizon
//! in a De Sitter universe, including Gibbons-Hawking temperature and entropy.

use core::f64;

/// Physical constants for cosmological calculations (SI units)
pub struct CosmoConstants {
    /// Speed of light in vacuum [m/s]
    pub c: f64,
    /// Reduced Planck constant [J·s]
    pub hbar: f64,
    /// Boltzmann constant [J/K]
    pub k_b: f64,
    /// Gravitational constant [m³/(kg·s²)]
    pub g: f64,
    /// Cosmological constant [1/m²]
    pub lambda: f64,
}

impl Default for CosmoConstants {
    fn default() -> Self {
        Self {
            c: 299_792_458.0,
            hbar: 1.054_571_817e-34,
            k_b: 1.380_649e-23,
            g: 6.674_30e-11,
            lambda: 1.1056e-52, // Current observational estimate
        }
    }
}

/// De Sitter horizon thermodynamic state
#[derive(Debug, Clone, Copy)]
pub struct DeSitterHorizon {
    /// Hubble parameter H = c * sqrt(Λ/3) [1/s]
    pub hubble: f64,
    /// Cosmological event horizon radius [m]
    pub horizon_radius: f64,
    /// Gibbons-Hawking temperature [K]
    pub temperature: f64,
    /// Horizon entropy [J/K]
    pub entropy: f64,
    /// Horizon area [m²]
    pub area: f64,
}

impl DeSitterHorizon {
    /// Construct a new De Sitter horizon from the cosmological constant
    /// 
    /// # Arguments
    /// * `lambda` - Cosmological constant [1/m²]
    /// * `constants` - Physical constants
    /// 
    /// # Returns
    /// * `Result<Self, &'static str>` - Horizon state or error if lambda <= 0
    pub fn new(lambda: f64, constants: &CosmoConstants) -> Result<Self, &'static str> {
        if lambda <= 0.0 {
            return Err("Cosmological constant must be positive for De Sitter space");
        }

        // H = c * sqrt(Λ/3)
        let hubble = constants.c * (lambda / 3.0).sqrt();
        
        // Cosmological event horizon radius: r_H = c / H = sqrt(3/Λ)
        let horizon_radius = constants.c / hubble;
        
        // Gibbons-Hawking temperature: T_GH = ℏ * H / (2π * k_B * c)
        // Alternative form: T_GH = ℏ * c * sqrt(Λ/3) / (2π * k_B * c) = ℏ * c * sqrt(Λ) / (2π * k_B * sqrt(3))
        let temperature = (constants.hbar * hubble) / (2.0 * core::f64::consts::PI * constants.k_b);
        
        // Horizon area: A = 4π * r_H²
        let area = 4.0 * core::f64::consts::PI * horizon_radius.powi(2);
        
        // Horizon entropy: S = k_B * A / (4 * l_P²) where l_P² = ℏ*G/c³
        // Simplified: S = k_B * A * c³ / (4 * ℏ * G)
        let planck_area = constants.hbar * constants.g / constants.c.powi(3);
        let entropy = constants.k_b * area / (4.0 * planck_area);

        Ok(Self {
            hubble,
            horizon_radius,
            temperature,
            entropy,
            area,
        })
    }

    /// Calculate the maximum extractable work (exergy) from the horizon
    /// given a local system at temperature T_local
    /// 
    /// Uses Carnot efficiency: η = 1 - T_cold / T_hot
    /// Here T_hot is the local matter temperature, T_cold is the horizon temperature
    /// 
    /// # Arguments
    /// * `t_local` - Local system temperature [K]
    /// * `energy_available` - Available energy in local system [J]
    /// 
    /// # Returns
    /// * `Result<f64, &'static str>` - Extractable work [J] or error
    pub fn extractable_work(&self, t_local: f64, energy_available: f64) -> Result<f64, &'static str> {
        if t_local <= 0.0 {
            return Err("Local temperature must be positive");
        }
        if energy_available < 0.0 {
            return Err("Available energy cannot be negative");
        }
        
        // For heat death epoch, T_local approaches T_horizon from above
        // Carnot efficiency: η = 1 - T_GH / T_local
        let efficiency = 1.0 - self.temperature / t_local;
        
        // Efficiency must be non-negative (Second Law)
        let efficiency = efficiency.max(0.0).min(1.0);
        
        Ok(energy_available * efficiency)
    }

    /// Calculate the power available from Gibbons-Hawking radiation harvesting
    /// assuming a collector with effective area A_collector
    /// 
    /// Power = σ * T_GH⁴ * A_collector (Stefan-Boltzmann law)
    /// 
    /// # Arguments
    /// * `collector_area` - Effective collecting area [m²]
    /// 
    /// # Returns
    /// * `f64` - Available power [W]
    pub fn gibbons_hawking_power(&self, collector_area: f64) -> f64 {
        if collector_area <= 0.0 {
            return 0.0;
        }
        
        // Stefan-Boltzmann constant: σ = π² * k_B⁴ / (60 * ℏ³ * c²)
        let sigma = (core::f64::consts::PI.powi(2) * self.k_b().powi(4)) 
            / (60.0 * self.hbar().powi(3) * self.c().powi(2));
        
        // Power radiated per unit area
        let power_density = sigma * self.temperature.powi(4);
        
        power_density * collector_area
    }
    
    // Helper methods to access constants
    fn c(&self) -> f64 { 299_792_458.0 }
    fn hbar(&self) -> f64 { 1.054_571_817e-34 }
    fn k_b(&self) -> f64 { 1.380_649e-23 }
}

/// De Sitter Horizon Harvester State
#[derive(Debug, Clone)]
pub struct HorizonHarvester {
    /// The De Sitter horizon being harvested
    pub horizon: DeSitterHorizon,
    /// Collector array total area [m²]
    pub collector_area: f64,
    /// Current extraction rate [W]
    pub extraction_rate: f64,
    /// Cumulative extracted energy [J]
    pub cumulative_energy: f64,
}

impl HorizonHarvester {
    /// Create a new horizon harvester
    /// 
    /// # Arguments
    /// * `lambda` - Cosmological constant
    /// * `collector_area` - Total collector array area [m²]
    /// 
    /// # Returns
    /// * `Result<Self, &'static str>` - Harvester or error
    pub fn new(lambda: f64, collector_area: f64) -> Result<Self, &'static str> {
        let constants = CosmoConstants::default();
        let horizon = DeSitterHorizon::new(lambda, &constants)?;
        
        if collector_area <= 0.0 {
            return Err("Collector area must be positive");
        }
        
        let extraction_rate = horizon.gibbons_hawking_power(collector_area);
        
        Ok(Self {
            horizon,
            collector_area,
            extraction_rate,
            cumulative_energy: 0.0,
        })
    }
    
    /// Simulate harvesting for a time interval
    /// 
    /// # Arguments
    /// * `dt` - Time interval [s]
    /// 
    /// # Returns
    /// * `Result<f64, &'static str>` - Energy extracted in this interval [J]
    pub fn harvest(&mut self, dt: f64) -> Result<f64, &'static str> {
        if dt <= 0.0 {
            return Err("Time interval must be positive");
        }
        
        let energy_extracted = self.extraction_rate * dt;
        self.cumulative_energy += energy_extracted;
        
        Ok(energy_extracted)
    }
    
    /// Update horizon parameters if cosmological constant changes
    /// (e.g., due to dark energy evolution)
    /// 
    /// # Arguments
    /// * `new_lambda` - Updated cosmological constant
    /// 
    /// # Returns
    /// * `Result<(), &'static str>` - Success or error
    pub fn update_cosmology(&mut self, new_lambda: f64) -> Result<(), &'static str> {
        let constants = CosmoConstants::default();
        self.horizon = DeSitterHorizon::new(new_lambda, &constants)?;
        self.extraction_rate = self.horizon.gibbons_hawking_power(self.collector_area);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_desitter_horizon_creation() {
        let constants = CosmoConstants::default();
        let horizon = DeSitterHorizon::new(constants.lambda, &constants);
        assert!(horizon.is_ok());
        
        let h = horizon.unwrap();
        assert!(h.hubble > 0.0);
        assert!(h.horizon_radius > 0.0);
        assert!(h.temperature > 0.0);
        assert!(h.entropy > 0.0);
    }

    #[test]
    fn test_invalid_lambda() {
        let constants = CosmoConstants::default();
        let result = DeSitterHorizon::new(-1.0, &constants);
        assert!(result.is_err());
    }

    #[test]
    fn test_extractable_work() {
        let constants = CosmoConstants::default();
        let horizon = DeSitterHorizon::new(constants.lambda, &constants).unwrap();
        
        // CMB temperature ~2.7K, much warmer than GH temperature
        let work = horizon.extractable_work(2.7, 1e20);
        assert!(work.is_ok());
        assert!(work.unwrap() > 0.0);
    }

    #[test]
    fn test_harvester() {
        let mut harvester = HorizonHarvester::new(1.1056e-52, 1e12).unwrap();
        let energy = harvester.harvest(1.0);
        assert!(energy.is_ok());
        assert!(harvester.cumulative_energy > 0.0);
    }
}
