//! Waste Heat Router for Matrioshka Brain Thermodynamics
//! 
//! Manages heat flow between nested Dyson shells, ensuring
//! thermodynamic compliance and optimal heat dissipation.

use crate::thermodynamics::stefan_boltzmann_manifold::{MatrioshkaShell, StefanBoltzmannManifold, ThermoError, HeatRoute};
use num_traits::{Float, Zero};

/// Heat flow configuration between shells
#[derive(Debug, Clone)]
pub struct HeatFlowConfig<T> {
    pub source_shell: u32,
    pub target_shell: u32,
    pub max_heat_flux: T,  // W/m²
    pub thermal_conductance: T,  // W/K
}

/// Thermal link between two shells
#[derive(Clone, Debug)]
pub struct ThermalLink<T> {
    pub from_shell: u32,
    pub to_shell: u32,
    pub conductance: T,  // W/K
    pub current_heat_flow: T,  // W
}

impl<T: Float + Copy + Zero> ThermalLink<T> {
    pub fn new(from_shell: u32, to_shell: u32, conductance: T) -> Self {
        Self {
            from_shell,
            to_shell,
            conductance,
            current_heat_flow: T::zero(),
        }
    }
    
    /// Calculate heat flow given temperature difference: Q = G * ΔT
    pub fn calculate_heat_flow(&self, t_source: T, t_sink: T) -> Result<T, ThermoError> {
        if t_source <= t_sink {
            return Err(ThermoError::TemperatureViolation {
                t_hot: t_source.to_f64().unwrap_or(0.0),
                t_cold: t_sink.to_f64().unwrap_or(0.0),
            });
        }
        
        let delta_t = t_source - t_sink;
        Ok(self.conductance * delta_t)
    }
}

/// Waste heat router manager
pub struct WasteHeatRouter<T> {
    manifold: StefanBoltzmannManifold<T>,
    thermal_links: Vec<ThermalLink<T>>,
    background_temp: T,  // Deep space background (~2.7K)
}

impl<T: Float + Copy + Zero> WasteHeatRouter<T> {
    pub fn new(manifold: StefanBoltzmannManifold<T>, background_temp: T) -> Self {
        Self {
            manifold,
            thermal_links: Vec::new(),
            background_temp,
        }
    }
    
    /// Add a thermal link between shells
    pub fn add_thermal_link(&mut self, link: ThermalLink<T>) -> Result<(), ThermoError> {
        // Verify shells exist and ordering is correct (heat flows outward)
        let source = self.manifold.get_shell(link.from_shell);
        let target = self.manifold.get_shell(link.to_shell);
        
        match (source, target) {
            (Some(s), Some(t)) => {
                if s.radius >= t.radius {
                    return Err(ThermoError::TemperatureViolation {
                        t_hot: s.temperature.to_f64().unwrap_or(0.0),
                        t_cold: t.temperature.to_f64().unwrap_or(0.0),
                    });
                }
            }
            _ => return Err(ThermoError::EnergyImbalance { imbalance: 0.0 }),
        }
        
        self.thermal_links.push(link);
        Ok(())
    }
    
    /// Create default thermal links (each shell to next outer shell)
    pub fn create_default_links(&mut self, base_conductance: T) -> Result<(), ThermoError> {
        let shells = &self.manifold.shells;
        
        for i in 0..shells.len() - 1 {
            let link = ThermalLink::new(
                shells[i].shell_id,
                shells[i + 1].shell_id,
                base_conductance,
            );
            self.add_thermal_link(link)?;
        }
        
        // Link outermost shell to deep space
        if let Some(last) = shells.last() {
            // Virtual link to background (radiative)
            let radiative_conductance = last.emissivity 
                                      * T::from(5.67e-8).unwrap()  // σ
                                      * last.surface_area 
                                      * T::from(4.0).unwrap() 
                                      * last.temperature.powi(3);
            
            self.thermal_links.push(ThermalLink {
                from_shell: last.shell_id,
                to_shell: u32::MAX,  // Background marker
                conductance: radiative_conductance,
                current_heat_flow: T::zero(),
            });
        }
        
        Ok(())
    }
    
    /// Route waste heat through the thermal network
    pub fn route_waste_heat(&mut self, source_shell: u32, waste_heat: T) -> Result<Vec<HeatRoute<T>>, ThermoError> {
        let mut routes = Vec::new();
        let mut remaining_heat = waste_heat;
        let mut current_shell = source_shell;
        
        // Follow thermal links outward until heat is dissipated
        while remaining_heat > T::from(1e-6).unwrap() {
            // Find outgoing link from current shell
            let link = self.thermal_links.iter_mut()
                .find(|l| l.from_shell == current_shell);
            
            match link {
                Some(l) => {
                    let source_temp = self.manifold.get_shell(current_shell)
                        .map(|s| s.temperature)
                        .unwrap_or(self.background_temp);
                    
                    let target_temp = if l.to_shell == u32::MAX {
                        self.background_temp
                    } else {
                        self.manifold.get_shell(l.to_shell)
                            .map(|s| s.temperature)
                            .unwrap_or(self.background_temp)
                    };
                    
                    if source_temp <= target_temp {
                        return Err(ThermoError::TemperatureViolation {
                            t_hot: source_temp.to_f64().unwrap_or(0.0),
                            t_cold: target_temp.to_f64().unwrap_or(0.0),
                        });
                    }
                    
                    // Calculate how much heat can flow through this link
                    let max_flow = l.calculate_heat_flow(source_temp, target_temp)?;
                    let actual_flow = remaining_heat.min(max_flow);
                    
                    l.current_heat_flow = l.current_heat_flow + actual_flow;
                    
                    routes.push(HeatRoute {
                        source_shell: current_shell,
                        target_shell: l.to_shell,
                        waste_heat_watts: actual_flow,
                        temperature_drop: source_temp - target_temp,
                        radiative_efficiency: T::one(),  // Would get from shell
                    });
                    
                    remaining_heat = remaining_heat - actual_flow;
                    current_shell = l.to_shell;
                    
                    if l.to_shell == u32::MAX {
                        break;  // Reached deep space
                    }
                }
                None => {
                    // No outgoing link - heat trapped
                    return Err(ThermoError::EnergyImbalance {
                        imbalance: remaining_heat.to_f64().unwrap_or(0.0),
                    });
                }
            }
        }
        
        Ok(routes)
    }
    
    /// Check thermal equilibrium of the system
    pub fn check_thermal_equilibrium(&self, tolerance: T) -> EquilibriumStatus<T> {
        let mut total_imbalance = T::zero();
        let mut shell_balances = Vec::new();
        
        for shell in &self.manifold.shells {
            // Sum heat flowing into this shell
            let mut heat_in = T::zero();
            let mut heat_out = T::zero();
            
            for link in &self.thermal_links {
                if link.to_shell == shell.shell_id {
                    heat_in = heat_in + link.current_heat_flow;
                }
                if link.from_shell == shell.shell_id {
                    heat_out = heat_out + link.current_heat_flow;
                }
            }
            
            // Shell also receives stellar radiation
            let sphere_area = T::from(4.0).unwrap() * T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap()) 
                            * shell.radius * shell.radius;
            let incident_flux = T::from(3.828e26).unwrap() / sphere_area;  // Solar luminosity
            let absorbed = shell.absorbed_power(incident_flux);
            
            heat_in = heat_in + absorbed;
            
            // Shell radiates heat
            let radiated = shell.radiated_power(&self.manifold.constants);
            heat_out = heat_out + radiated;
            
            let balance = heat_in - heat_out;
            total_imbalance = total_imbalance + balance.abs();
            
            shell_balances.push(ShellBalance {
                shell_id: shell.shell_id,
                heat_in,
                heat_out,
                net_balance: balance,
            });
        }
        
        EquilibriumStatus {
            is_equilibrium: total_imbalance < tolerance,
            total_imbalance,
            shell_balances,
        }
    }
    
    /// Optimize thermal link conductances for maximum heat dissipation
    pub fn optimize_conductances(&mut self, max_iterations: usize) -> OptimizationResult {
        let mut improvements = 0;
        
        for _iter in 0..max_iterations {
            let status = self.check_thermal_equilibrium(T::from(1e-6).unwrap());
            
            if status.is_equilibrium {
                break;
            }
            
            // Adjust conductances based on imbalances
            for balance in &status.shell_balances {
                if balance.net_balance > T::zero() {
                    // Too much heat in - increase outgoing conductance
                    if let Some(link) = self.thermal_links.iter_mut()
                        .find(|l| l.from_shell == balance.shell_id) 
                    {
                        let adjustment = T::from(1.05).unwrap();
                        link.conductance = link.conductance * adjustment;
                        improvements += 1;
                    }
                }
            }
        }
        
        OptimizationResult {
            iterations_run: max_iterations,
            total_improvements: improvements,
            final_status: self.check_thermal_equilibrium(T::from(1e-6).unwrap()),
        }
    }
}

/// Shell heat balance
#[derive(Debug, Clone)]
pub struct ShellBalance<T> {
    pub shell_id: u32,
    pub heat_in: T,
    pub heat_out: T,
    pub net_balance: T,
}

/// Equilibrium status
#[derive(Debug, Clone)]
pub struct EquilibriumStatus<T> {
    pub is_equilibrium: bool,
    pub total_imbalance: T,
    pub shell_balances: Vec<ShellBalance<T>>,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub iterations_run: usize,
    pub total_improvements: usize,
    pub final_status: EquilibriumStatus<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thermodynamics::stefan_boltzmann_manifold::MatrioshkaShell;
    
    #[test]
    fn test_thermal_link_creation() {
        type F = f64;
        let link = ThermalLink::new(1, 2, F::from(1e9).unwrap());
        
        assert_eq!(link.from_shell, 1);
        assert_eq!(link.to_shell, 2);
        assert!(link.conductance > F::zero());
    }
    
    #[test]
    fn test_heat_flow_calculation() {
        type F = f64;
        let link = ThermalLink::new(1, 2, F::from(1e6).unwrap());
        
        let t_source = F::from(1000.0).unwrap();
        let t_sink = F::from(500.0).unwrap();
        
        let flow = link.calculate_heat_flow(t_source, t_sink).unwrap();
        
        // Q = G * ΔT = 1e6 * 500 = 5e8 W
        assert!(flow > F::zero());
    }
    
    #[test]
    fn test_heat_flow_violation() {
        type F = f64;
        let link = ThermalLink::new(1, 2, F::from(1e6).unwrap());
        
        let t_source = F::from(300.0).unwrap();
        let t_sink = F::from(500.0).unwrap();  // Sink hotter than source!
        
        let result = link.calculate_heat_flow(t_source, t_sink);
        assert!(result.is_err());
    }
}
