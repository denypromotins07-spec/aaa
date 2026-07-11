//! Stefan-Boltzmann Thermodynamic Manifold for Matrioshka Brains
//! 
//! Models the temperature gradient across nested Dyson swarm shells,
//! computing radiative heat transfer and optimal shell configurations
//! for maximum computational efficiency.

use nalgebra::SVector;
use num_traits::{Float, Zero};
use thiserror::Error;

/// Physical constants for thermodynamic calculations
pub struct ThermodynamicConstants<T> {
    pub stefan_boltzmann: T,  // σ = 5.670374419e-8 W/(m²·K⁴)
    pub boltzmann: T,         // k_B
    pub speed_of_light: T,    // c
    pub planck: T,            // h
}

impl<T: Float + Zero> Default for ThermodynamicConstants<T> {
    fn default() -> Self {
        Self {
            stefan_boltzmann: T::from(5.670374419e-8).unwrap_or_else(|| T::zero()),
            boltzmann: T::from(1.380649e-23).unwrap_or_else(|| T::zero()),
            speed_of_light: T::from(2.99792458e8).unwrap_or_else(|| T::zero()),
            planck: T::from(6.62607015e-34).unwrap_or_else(|| T::zero()),
        }
    }
}

/// A single shell in the Matrioshka brain structure
#[derive(Clone, Debug)]
pub struct MatrioshkaShell<T> {
    pub shell_id: u32,
    pub radius: T,           // Distance from star center
    pub temperature: T,      // Operating temperature
    pub emissivity: T,       // Surface emissivity (0-1)
    pub absorptivity: T,     // Surface absorptivity (0-1)
    pub surface_area: T,     // Total collector area
    pub compute_efficiency: T, // ops/Joule at this temperature
}

impl<T: Float + Copy + Zero> MatrioshkaShell<T> {
    pub fn new(
        shell_id: u32,
        radius: T,
        temperature: T,
        emissivity: T,
        absorptivity: T,
        surface_area: T,
    ) -> Self {
        // Compute efficiency based on temperature
        // Landauer limit: k_B T ln(2) energy per irreversible bit operation
        let kb = T::from(1.380649e-23).unwrap_or_else(|| T::one());
        let ln2 = T::from(0.693147).unwrap_or_else(|| T::from(7).unwrap() / T::from(10).unwrap());
        let landauer_energy = kb * temperature * ln2;
        
        // Efficiency is inverse of minimum energy per op
        let compute_efficiency = if landauer_energy > T::zero() {
            T::one() / landauer_energy
        } else {
            T::zero()
        };
        
        Self {
            shell_id,
            radius,
            temperature,
            emissivity,
            absorptivity,
            surface_area,
            compute_efficiency,
        }
    }
    
    /// Compute radiated power using Stefan-Boltzmann law: P = εσAT⁴
    pub fn radiated_power(&self, constants: &ThermodynamicConstants<T>) -> T {
        let t_fourth = self.temperature.powi(4);
        constants.stefan_boltzmann * self.emissivity * self.surface_area * t_fourth
    }
    
    /// Compute absorbed power from inner source
    pub fn absorbed_power(&self, incident_flux: T) -> T {
        incident_flux * self.absorptivity * self.surface_area
    }
}

/// Errors in thermodynamic calculations
#[derive(Error, Debug)]
pub enum ThermoError {
    #[error("Temperature violation: T_hot={t_hot:?} <= T_cold={t_cold:?}")]
    TemperatureViolation { t_hot: f64, t_cold: f64 },
    #[error("Energy balance not achieved: imbalance={imbalance:?} W")]
    EnergyImbalance { imbalance: f64 },
    #[error("Invalid emissivity: value={value:?} outside [0,1]")]
    InvalidEmissivity { value: f64 },
    #[error("Carnot efficiency negative or > 1: eta={eta:?}")]
    InvalidCarnotEfficiency { eta: f64 },
}

/// Stefan-Boltzmann manifold calculator
pub struct StefanBoltzmannManifold<T> {
    shells: Vec<MatrioshkaShell<T>>,
    stellar_luminosity: T,
    constants: ThermodynamicConstants<T>,
}

impl<T: Float + Copy + Zero> StefanBoltzmannManifold<T> {
    pub fn new(stellar_luminosity: T) -> Self {
        Self {
            shells: Vec::new(),
            stellar_luminosity,
            constants: ThermodynamicConstants::default(),
        }
    }
    
    /// Add a shell to the manifold
    pub fn add_shell(&mut self, shell: MatrioshkaShell<T>) -> Result<(), ThermoError> {
        // Validate emissivity
        if shell.emissivity < T::zero() || shell.emissivity > T::one() {
            return Err(ThermoError::InvalidEmissivity { 
                value: shell.emissivity.to_f64().unwrap_or(0.0) 
            });
        }
        
        // Check ordering: outer shells must have larger radius
        if let Some(last) = self.shells.last() {
            if shell.radius <= last.radius {
                // Allow same radius for co-orbital shells
                if shell.radius < last.radius {
                    return Err(ThermoError::TemperatureViolation {
                        t_hot: last.radius.to_f64().unwrap_or(0.0),
                        t_cold: shell.radius.to_f64().unwrap_or(0.0),
                    });
                }
            }
        }
        
        self.shells.push(shell);
        Ok(())
    }
    
    /// Compute equilibrium temperature for each shell
    pub fn compute_equilibrium_temperatures(&mut self) -> Result<(), ThermoError> {
        if self.shells.is_empty() {
            return Ok(());
        }
        
        let two = T::one() + T::one();
        let four = two + two;
        
        // Innermost shell receives direct stellar radiation
        if let Some(first_shell) = self.shells.first_mut() {
            let sphere_area = T::from(4.0).unwrap() * T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap()) 
                            * first_shell.radius * first_shell.radius;
            
            // Incident flux = L / (4πr²)
            let incident_flux = self.stellar_luminosity / sphere_area;
            
            // Equilibrium: absorbed = radiated
            // α * L / (4πr²) * A = ε * σ * A * T⁴
            // T⁴ = α * L / (4πr² * ε * σ)
            
            let numerator = first_shell.absorptivity * self.stellar_luminosity;
            let denominator = sphere_area * first_shell.emissivity * self.constants.stefan_boltzmann;
            
            if denominator > T::zero() {
                let t_fourth = numerator / denominator;
                first_shell.temperature = t_fourth.powf(T::one() / four);
            }
        }
        
        // Outer shells receive radiation from inner shells
        for i in 1..self.shells.len() {
            let inner_shell = &self.shells[i - 1];
            let outer_shell = &mut self.shells[i];
            
            // Radiation from inner shell spreads over sphere at outer radius
            let inner_radiated = inner_shell.radiated_power(&self.constants);
            let sphere_area_outer = T::from(4.0).unwrap() * T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap()) 
                                  * outer_shell.radius * outer_shell.radius;
            
            // Fraction intercepted by outer shell
            let interception_fraction = if outer_shell.surface_area > T::zero() {
                outer_shell.surface_area / sphere_area_outer
            } else {
                T::one()
            };
            
            let absorbed = inner_radiated * interception_fraction * outer_shell.absorptivity;
            
            // Equilibrium: absorbed = radiated
            let radiated = outer_shell.radiated_power(&self.constants);
            
            // Solve for T: absorbed = εσAT⁴
            // T = (absorbed / (εσA))^(1/4)
            let denominator = outer_shell.emissivity * self.constants.stefan_boltzmann * outer_shell.surface_area;
            
            if denominator > T::zero() && absorbed > T::zero() {
                let t_fourth = absorbed / denominator;
                outer_shell.temperature = t_fourth.powf(T::one() / four);
            }
        }
        
        Ok(())
    }
    
    /// Verify second law compliance: T must decrease outward
    pub fn verify_second_law(&self) -> Result<(), ThermoError> {
        for i in 1..self.shells.len() {
            let inner_t = self.shells[i - 1].temperature;
            let outer_t = self.shells[i].temperature;
            
            if outer_t >= inner_t {
                return Err(ThermoError::TemperatureViolation {
                    t_hot: inner_t.to_f64().unwrap_or(0.0),
                    t_cold: outer_t.to_f64().unwrap_or(0.0),
                });
            }
        }
        Ok(())
    }
    
    /// Compute total system compute capacity
    pub fn compute_total_capacity(&self) -> ComputeCapacity<T> {
        let mut total_ops_per_sec = T::zero();
        let mut total_power_consumed = T::zero();
        let mut shell_capacities = Vec::new();
        
        for shell in &self.shells {
            // Power available = absorbed radiation
            let sphere_area = T::from(4.0).unwrap() * T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap()) 
                            * shell.radius * shell.radius;
            let incident_flux = self.stellar_luminosity / sphere_area;
            let absorbed_power = shell.absorbed_power(incident_flux);
            
            // Compute ops = efficiency * power
            let ops = shell.compute_efficiency * absorbed_power;
            
            shell_capacities.push(ShellCapacity {
                shell_id: shell.shell_id,
                ops_per_sec: ops,
                power_watts: absorbed_power,
                temperature: shell.temperature,
            });
            
            total_ops_per_sec = total_ops_per_sec + ops;
            total_power_consumed = total_power_consumed + absorbed_power;
        }
        
        ComputeCapacity {
            total_ops_per_sec,
            total_power_watts: total_power_consumed,
            shell_capacities,
        }
    }
    
    /// Get shell by ID
    pub fn get_shell(&self, shell_id: u32) -> Option<&MatrioshkaShell<T>> {
        self.shells.iter().find(|s| s.shell_id == shell_id)
    }
    
    /// Number of shells
    pub fn shell_count(&self) -> usize {
        self.shells.len()
    }
}

/// Compute capacity results
#[derive(Debug, Clone)]
pub struct ComputeCapacity<T> {
    pub total_ops_per_sec: T,
    pub total_power_watts: T,
    pub shell_capacities: Vec<ShellCapacity<T>>,
}

/// Individual shell capacity
#[derive(Debug, Clone)]
pub struct ShellCapacity<T> {
    pub shell_id: u32,
    pub ops_per_sec: T,
    pub power_watts: T,
    pub temperature: T,
}

/// Heat router for waste heat management
pub struct WasteHeatRouter<T> {
    manifold: StefanBoltzmannManifold<T>,
    max_temperature_gradient: T,
}

impl<T: Float + Copy + Zero> WasteHeatRouter<T> {
    pub fn new(manifold: StefanBoltzmannManifold<T>, max_gradient_kelvin: T) -> Self {
        Self {
            manifold,
            max_temperature_gradient: max_gradient_kelvin,
        }
    }
    
    /// Route waste heat from compute operations to appropriate shell
    pub fn route_waste_heat(&self, source_shell: u32, waste_heat_watts: T) -> Result<HeatRoute<T>, ThermoError> {
        let source = self.manifold.get_shell(source_shell)
            .ok_or_else(|| ThermoError::TemperatureViolation { t_hot: 0.0, t_cold: 0.0 })?;
        
        // Find next outer shell that can accept heat
        let mut target_shell = None;
        for shell in &self.manifold.shells {
            if shell.radius > source.radius {
                // Check temperature gradient constraint
                let temp_diff = source.temperature - shell.temperature;
                if temp_diff <= self.max_temperature_gradient {
                    target_shell = Some(shell);
                    break;
                }
            }
        }
        
        match target_shell {
            Some(target) => Ok(HeatRoute {
                source_shell: source_shell,
                target_shell: target.shell_id,
                waste_heat_watts,
                temperature_drop: source.temperature - target.temperature,
                radiative_efficiency: target.emissivity,
            }),
            None => Err(ThermoError::EnergyImbalance {
                imbalance: waste_heat_watts.to_f64().unwrap_or(0.0),
            }),
        }
    }
}

/// Heat routing result
#[derive(Debug, Clone)]
pub struct HeatRoute<T> {
    pub source_shell: u32,
    pub target_shell: u32,
    pub waste_heat_watts: T,
    pub temperature_drop: T,
    pub radiative_efficiency: T,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_manifold_creation() {
        type F = f64;
        let luminosity = F::from(3.828e26).unwrap();  // Solar luminosity
        let manifold = StefanBoltzmannManifold::new(luminosity);
        
        assert!(manifold.stellar_luminosity > F::zero());
    }
    
    #[test]
    fn test_shell_radiation() {
        type F = f64;
        let constants = ThermodynamicConstants::<F>::default();
        
        let shell = MatrioshkaShell::new(
            1,
            F::from(1e9).unwrap(),
            F::from(1000.0).unwrap(),
            F::from(0.9).unwrap(),
            F::from(0.95).unwrap(),
            F::from(1e12).unwrap(),
        );
        
        let power = shell.radiated_power(&constants);
        assert!(power > F::zero());
    }
    
    #[test]
    fn test_second_law_verification() {
        type F = f64;
        let luminosity = F::from(3.828e26).unwrap();
        let mut manifold = StefanBoltzmannManifold::new(luminosity);
        
        // Add shells with decreasing temperatures
        let shell1 = MatrioshkaShell::new(1, F::from(1e9).unwrap(), F::from(1000.0).unwrap(), F::from(0.9).unwrap(), F::from(0.95).unwrap(), F::from(1e12).unwrap());
        let shell2 = MatrioshkaShell::new(2, F::from(2e9).unwrap(), F::from(500.0).unwrap(), F::from(0.9).unwrap(), F::from(0.95).unwrap(), F::from(4e12).unwrap());
        
        manifold.add_shell(shell1).unwrap();
        manifold.add_shell(shell2).unwrap();
        
        assert!(manifold.verify_second_law().is_ok());
    }
}
