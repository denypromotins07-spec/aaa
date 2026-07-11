//! Gibbons-Hawking Radiation Harvester
//! 
//! Implements the extraction of useful work from Gibbons-Hawking radiation
//! emitted by the cosmological event horizon in a De Sitter universe.

use super::desitter_thermodynamics::{CosmoConstants, DeSitterHorizon};

/// Quantum field state near the cosmological horizon
#[derive(Debug, Clone, Copy)]
pub struct QFTState {
    /// Mode frequency [Hz]
    pub frequency: f64,
    /// Occupation number (Bose-Einstein distribution)
    pub occupation: f64,
    /// Energy per mode [J]
    pub energy_per_mode: f64,
}

impl QFTState {
    /// Create a new QFT state for a given frequency and temperature
    /// 
    /// # Arguments
    /// * `frequency` - Mode frequency [Hz]
    /// * `temperature` - Horizon temperature [K]
    /// 
    /// # Returns
    /// * `Self` - QFT state with Bose-Einstein occupation
    pub fn new(frequency: f64, temperature: f64) -> Result<Self, &'static str> {
        if frequency <= 0.0 {
            return Err("Frequency must be positive");
        }
        if temperature <= 0.0 {
            return Err("Temperature must be positive");
        }
        
        let constants = CosmoConstants::default();
        let hbar = constants.hbar;
        let k_b = constants.k_b;
        
        // Energy per mode: E = ℏω
        let omega = 2.0 * core::f64::consts::PI * frequency;
        let energy_per_mode = hbar * omega;
        
        // Bose-Einstein occupation: n = 1 / (exp(ℏω/kT) - 1)
        let beta = 1.0 / (k_b * temperature);
        let exponent = beta * energy_per_mode;
        
        // Prevent overflow in exp for high frequencies
        let occupation = if exponent > 700.0 {
            0.0 // Effectively zero occupation
        } else {
            1.0 / (exponent.exp() - 1.0).max(f64::EPSILON)
        };
        
        Ok(Self {
            frequency,
            occupation,
            energy_per_mode,
        })
    }
    
    /// Calculate the total energy in this mode
    pub fn total_energy(&self) -> f64 {
        self.occupation * self.energy_per_mode
    }
}

/// Gibbons-Hawking harvester configuration
#[derive(Debug, Clone)]
pub struct GHHarvesterConfig {
    /// Collector effective area [m²]
    pub collector_area: f64,
    /// Frequency band minimum [Hz]
    pub freq_min: f64,
    /// Frequency band maximum [Hz]
    pub freq_max: f64,
    /// Number of frequency bins for integration
    pub num_bins: usize,
    /// Conversion efficiency (0 to 1)
    pub efficiency: f64,
}

impl Default for GHHarvesterConfig {
    fn default() -> Self {
        Self {
            collector_area: 1e12, // 1 million km²
            freq_min: 1e3,        // 1 kHz
            freq_max: 1e15,       // 1 PHz (infrared)
            num_bins: 1000,
            efficiency: 0.85,     // 85% conversion efficiency
        }
    }
}

/// Main Gibbons-Hawking harvester structure
#[derive(Debug, Clone)]
pub struct GibbonsHawkingHarvester {
    /// Horizon being harvested
    pub horizon: DeSitterHorizon,
    /// Configuration
    pub config: GHHarvesterConfig,
    /// Cumulative harvested energy [J]
    pub cumulative_energy: f64,
    /// Operating time [s]
    pub operating_time: f64,
}

impl GibbonsHawkingHarvester {
    /// Create a new Gibbons-Hawking harvester
    /// 
    /// # Arguments
    /// * `lambda` - Cosmological constant [1/m²]
    /// * `config` - Harvester configuration
    /// 
    /// # Returns
    /// * `Result<Self, &'static str>` - Harvester or error
    pub fn new(lambda: f64, config: GHHarvesterConfig) -> Result<Self, &'static str> {
        let constants = CosmoConstants::default();
        let horizon = DeSitterHorizon::new(lambda, &constants)?;
        
        if config.collector_area <= 0.0 {
            return Err("Collector area must be positive");
        }
        if config.freq_min <= 0.0 || config.freq_max <= 0.0 {
            return Err("Frequencies must be positive");
        }
        if config.freq_min >= config.freq_max {
            return Err("freq_min must be less than freq_max");
        }
        if config.efficiency <= 0.0 || config.efficiency > 1.0 {
            return Err("Efficiency must be in (0, 1]");
        }
        
        Ok(Self {
            horizon,
            config,
            cumulative_energy: 0.0,
            operating_time: 0.0,
        })
    }
    
    /// Calculate spectral power density at a given frequency
    /// 
    /// Uses Planck's law adapted for Gibbons-Hawking radiation
    /// 
    /// # Arguments
    /// * `frequency` - Frequency [Hz]
    /// 
    /// # Returns
    /// * `f64` - Spectral power density [W/Hz/m²]
    pub fn spectral_power_density(&self, frequency: f64) -> f64 {
        if frequency <= 0.0 {
            return 0.0;
        }
        
        let hbar = 1.054_571_817e-34;
        let c = 299_792_458.0;
        let k_b = 1.380_649e-23;
        let t = self.horizon.temperature;
        
        let omega = 2.0 * core::f64::consts::PI * frequency;
        let energy = hbar * omega;
        let beta = 1.0 / (k_b * t);
        let exponent = beta * energy;
        
        // Handle numerical stability
        if exponent > 700.0 {
            return 0.0;
        }
        
        let occupation = 1.0 / (exponent.exp() - 1.0).max(f64::EPSILON);
        
        // Density of states in 3D: g(ω) = ω² / (π² * c³)
        let density_of_states = omega.powi(2) / (core::f64::consts::PI.powi(2) * c.powi(3));
        
        // Power per unit area per unit frequency
        // dP/dω = (ℏω) * occupation * g(ω) * c / 4
        let power_density = energy * occupation * density_of_states * c / 4.0;
        
        power_density
    }
    
    /// Integrate total power over the configured frequency band
    /// 
    /// Uses trapezoidal rule for numerical integration
    /// 
    /// # Returns
    /// * `f64` - Total harvested power [W]
    pub fn total_power(&self) -> f64 {
        let n = self.config.num_bins;
        let f_min = self.config.freq_min;
        let f_max = self.config.freq_max;
        let df = (f_max - f_min) / (n as f64);
        
        let mut integral = 0.0;
        
        for i in 0..=n {
            let f = f_min + (i as f64) * df;
            let weight = if i == 0 || i == n { 0.5 } else { 1.0 };
            integral += weight * self.spectral_power_density(f);
        }
        
        integral * df * self.config.collector_area * self.config.efficiency
    }
    
    /// Operate the harvester for a time interval
    /// 
    /// # Arguments
    /// * `dt` - Time interval [s]
    /// 
    /// # Returns
    /// * `Result<f64, &'static str>` - Energy harvested [J]
    pub fn operate(&mut self, dt: f64) -> Result<f64, &'static str> {
        if dt <= 0.0 {
            return Err("Time interval must be positive");
        }
        
        let power = self.total_power();
        let energy = power * dt;
        
        self.cumulative_energy += energy;
        self.operating_time += dt;
        
        Ok(energy)
    }
    
    /// Calculate the entropy production rate of the harvesting process
    /// 
    /// # Returns
    /// * `f64` - Entropy production rate [W/K]
    pub fn entropy_production_rate(&self) -> f64 {
        let power = self.total_power();
        // Entropy flow out of horizon: dS/dt = P / T_horizon
        power / self.horizon.temperature
    }
    
    /// Get current harvest statistics
    pub fn statistics(&self) -> HarvestStats {
        HarvestStats {
            power: self.total_power(),
            cumulative_energy: self.cumulative_energy,
            operating_time: self.operating_time,
            entropy_rate: self.entropy_production_rate(),
            avg_power: if self.operating_time > 0.0 {
                self.cumulative_energy / self.operating_time
            } else {
                0.0
            },
        }
    }
}

/// Statistics from harvester operation
#[derive(Debug, Clone, Copy)]
pub struct HarvestStats {
    /// Current power output [W]
    pub power: f64,
    /// Cumulative harvested energy [J]
    pub cumulative_energy: f64,
    /// Total operating time [s]
    pub operating_time: f64,
    /// Entropy production rate [W/K]
    pub entropy_rate: f64,
    /// Average power over operating lifetime [W]
    pub avg_power: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qft_state_creation() {
        let state = QFTState::new(1e10, 1e-29);
        assert!(state.is_ok());
        let s = state.unwrap();
        assert!(s.frequency > 0.0);
        assert!(s.occupation >= 0.0);
    }

    #[test]
    fn test_harvester_creation() {
        let config = GHHarvesterConfig::default();
        let harvester = GibbonsHawkingHarvester::new(1.1056e-52, config);
        assert!(harvester.is_ok());
    }

    #[test]
    fn test_spectral_power() {
        let config = GHHarvesterConfig::default();
        let harvester = GibbonsHawkingHarvester::new(1.1056e-52, config).unwrap();
        
        // Power should be positive for valid frequencies
        let power = harvester.spectral_power_density(1e10);
        assert!(power >= 0.0);
    }

    #[test]
    fn test_total_power() {
        let config = GHHarvesterConfig::default();
        let harvester = GibbonsHawkingHarvester::new(1.1056e-52, config).unwrap();
        
        let power = harvester.total_power();
        assert!(power >= 0.0);
    }

    #[test]
    fn test_operate() {
        let config = GHHarvesterConfig::default();
        let mut harvester = GibbonsHawkingHarvester::new(1.1056e-52, config).unwrap();
        
        let energy = harvester.operate(1.0);
        assert!(energy.is_ok());
        assert!(harvester.cumulative_energy >= 0.0);
    }
}
