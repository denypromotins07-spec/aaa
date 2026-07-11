//! Interstellar Energy Carry Arbitrage Engine
//! 
//! Determines optimal energy delivery strategy for deep-space probes:
//! beamed power from Dyson swarm vs. local Bussard ramjet harvesting.

use crate::beaming::nicoll_dyson_phased_array::{NicollDysonArray, PhasedArrayConfig};
use crate::beaming::relativistic_doppler_attenuation::{AttenuationCalculator, RelativisticProbe};
use nalgebra::SVector;
use num_traits::{Float, Zero};

/// Bussard ramjet parameters
#[derive(Clone, Debug)]
pub struct BussardRamjet<T> {
    pub scoop_radius: T,           // Magnetic scoop radius (m)
    pub collection_efficiency: T,  // Fraction of intercepted ISM collected
    pub fusion_efficiency: T,      // Mass-to-energy conversion efficiency
    pub ism_density: T,            // Interstellar medium density (kg/m³)
}

impl<T: Float + Copy + Zero> BussardRamjet<T> {
    pub fn new(
        scoop_radius: T,
        collection_efficiency: T,
        fusion_efficiency: T,
        ism_density: T,
    ) -> Self {
        Self {
            scoop_radius,
            collection_efficiency,
            fusion_efficiency,
            ism_density,
        }
    }
    
    /// Calculate mass collection rate: ṁ = ρ * A * v
    pub fn calculate_mass_collection_rate(&self, velocity: T) -> T {
        let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        let scoop_area = pi * self.scoop_radius * self.scoop_radius;
        
        self.ism_density * scoop_area * velocity * self.collection_efficiency
    }
    
    /// Calculate power generation from collected mass: P = ṁ * c² * η
    pub fn calculate_power_generation(&self, velocity: T, c: T) -> T {
        let mass_rate = self.calculate_mass_collection_rate(velocity);
        let c_squared = c * c;
        
        mass_rate * c_squared * self.fusion_efficiency
    }
    
    /// Calculate drag force from scoop: F_drag = ½ * ρ * A * v²
    pub fn calculate_drag_force(&self, velocity: T) -> T {
        let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        let scoop_area = pi * self.scoop_radius * self.scoop_radius;
        
        T::from(0.5).unwrap() * self.ism_density * scoop_area * velocity * velocity
    }
    
    /// Net power after drag losses
    pub fn calculate_net_power(&self, velocity: T, c: T) -> T {
        let generated = self.calculate_power_generation(velocity, c);
        let drag_power_loss = self.calculate_drag_force(velocity) * velocity;
        
        if generated > drag_power_loss {
            generated - drag_power_loss
        } else {
            T::zero()
        }
    }
}

/// Energy delivery options comparison
#[derive(Debug, Clone)]
pub enum EnergyDeliveryOption {
    /// Beam power from Dyson swarm
    BeamedPower {
        received_power_watts: f64,
        transmission_efficiency: f64,
        cost_per_kwh: f64,
    },
    /// Local Bussard ramjet harvesting
    BussardHarvesting {
        net_power_watts: f64,
        drag_penalty: f64,
        ism_density_dependent: bool,
    },
    /// Hybrid approach
    Hybrid {
        beamed_fraction: f64,
        harvested_fraction: f64,
        total_power: f64,
    },
}

/// Arbitrage decision result
#[derive(Debug, Clone)]
pub struct ArbitrageDecision {
    pub recommended_option: String,
    pub beamed_power_cost: f64,
    pub bussard_net_value: f64,
    pub savings_from_optimal: f64,
    pub breakeven_distance_ly: f64,
}

/// Interstellar energy carry arbitrage engine
pub struct InterstellarEnergyArbitrage<T> {
    beam_array: NicollDysonArray<T>,
    ramjet: BussardRamjet<T>,
    attenuation_calc: AttenuationCalculator<T>,
    energy_price_at_source: T,  // $/kWh at Dyson swarm
}

impl<T: Float + Copy + Zero> InterstellarEnergyArbitrage<T> {
    pub fn new(
        beam_array: NicollDysonArray<T>,
        ramjet: BussardRamjet<T>,
        energy_price: T,
    ) -> Self {
        Self {
            beam_array,
            ramjet,
            attenuation_calc: AttenuationCalculator::new(),
            energy_price_at_source: energy_price,
        }
    }
    
    /// Compare beamed power vs. Bussard harvesting at given distance
    pub fn compare_delivery_options(
        &self,
        probe: &RelativisticProbe<T>,
        receiver_aperture: T,
        transmitted_power: T,
    ) -> Result<EnergyComparison<T>, crate::beaming::relativistic_doppler_attenuation::RelativisticError> {
        let c = T::from(299792458.0).unwrap();
        
        // Option 1: Beamed power
        let attenuation = self.attenuation_calc.calculate_total_attenuation(
            probe,
            transmitted_power,
            receiver_aperture,
            self.beam_array.config.wavelength,
        )?;
        
        let beamed_power = attenuation.relativistic_received_power;
        
        // Option 2: Bussard ramjet
        let velocity = probe.velocity.norm();
        let bussard_power = self.ramjet.calculate_net_power(velocity, c);
        
        Ok(EnergyComparison {
            beamed_power_watts: beamed_power,
            bussard_power_watts: bussard_power,
            preferred_option: if beamed_power > bussard_power {
                "beamed"
            } else {
                "bussard"
            }.to_string(),
            power_ratio: if bussard_power > T::zero() {
                beamed_power / bussard_power
            } else {
                T::from(f64::INFINITY).unwrap()
            },
        })
    }
    
    /// Calculate cost of carry for beamed energy over distance
    /// Cost includes transmission losses and opportunity cost
    pub fn calculate_cost_of_carry(
        &self,
        distance_light_years: f64,
        probe_velocity_fraction_c: f64,
    ) -> CarryCostAnalysis {
        let distance_meters = T::from(distance_light_years * 9.461e15).unwrap();
        let c = T::from(299792458.0).unwrap();
        let probe_velocity = c * T::from(probe_velocity_fraction_c).unwrap();
        
        // Create probe state
        let position = SVector::new(distance_meters, T::zero(), T::zero());
        let velocity = SVector::new(probe_velocity, T::zero(), T::zero());
        let probe = RelativisticProbe::new(position, velocity, T::zero());
        
        // Assume standard receiver
        let receiver_aperture = T::from(1000.0).unwrap();  // 1 km²
        let transmitted_power = self.beam_array.config.max_total_power();
        
        // Calculate received power
        match self.attenuation_calc.calculate_total_attenuation(
            &probe,
            transmitted_power,
            receiver_aperture,
            self.beam_array.config.wavelength,
        ) {
            Ok(attenuation) => {
                let received_power = attenuation.relativistic_received_power;
                let efficiency = if transmitted_power > T::zero() {
                    received_power / transmitted_power
                } else {
                    T::zero()
                };
                
                // Cost per useful kWh at probe
                let base_cost = self.energy_price_at_source;
                let delivered_cost = if efficiency > T::zero() {
                    base_cost / efficiency
                } else {
                    T::from(f64::INFINITY).unwrap()
                };
                
                // Travel time for energy (at light speed)
                let travel_time_seconds = distance_meters / c;
                let travel_time_years = travel_time_seconds / T::from(3.154e7).unwrap();
                
                CarryCostAnalysis {
                    distance_ly: distance_light_years,
                    transmission_efficiency: efficiency.to_f64().unwrap_or(0.0),
                    delivered_cost_per_kwh: delivered_cost.to_f64().unwrap_or(f64::INFINITY),
                    energy_travel_time_years: travel_time_years.to_f64().unwrap_or(f64::INFINITY),
                    recommendation: if efficiency > T::from(0.001).unwrap() {
                        "viable".to_string()
                    } else {
                        "switch_to_bussard".to_string()
                    },
                }
            }
            Err(_) => CarryCostAnalysis {
                distance_ly: distance_light_years,
                transmission_efficiency: 0.0,
                delivered_cost_per_kwh: f64::INFINITY,
                energy_travel_time_years: distance_light_years,
                recommendation: "impossible".to_string(),
            },
        }
    }
    
    /// Find breakeven distance where Bussard becomes more economical than beaming
    pub fn find_breakeven_distance(&self, ism_density_range: (f64, f64)) -> BreakevenResult {
        let mut breakeven_ly = f64::INFINITY;
        
        // Binary search for breakeven
        let mut low = 0.1;
        let mut high = 100.0;
        
        for _iteration in 0..50 {
            let mid = (low + high) / 2.0;
            
            let analysis = self.calculate_cost_of_carry(mid, 0.1);
            
            // Check if beaming is still viable
            if analysis.transmission_efficiency < 0.001 {
                high = mid;
                breakeven_ly = mid;
            } else {
                low = mid;
            }
        }
        
        BreakevenResult {
            breakeven_distance_ly: breakeven_ly,
            below_breakeven_recommendation: "beam_power",
            above_breakeven_recommendation: "bussard_ramjet",
        }
    }
}

/// Energy comparison result
#[derive(Debug, Clone)]
pub struct EnergyComparison<T> {
    pub beamed_power_watts: T,
    pub bussard_power_watts: T,
    pub preferred_option: String,
    pub power_ratio: T,
}

/// Carry cost analysis result
#[derive(Debug, Clone)]
pub struct CarryCostAnalysis {
    pub distance_ly: f64,
    pub transmission_efficiency: f64,
    pub delivered_cost_per_kwh: f64,
    pub energy_travel_time_years: f64,
    pub recommendation: String,
}

/// Breakeven analysis result
#[derive(Debug, Clone)]
pub struct BreakevenResult {
    pub breakeven_distance_ly: f64,
    pub below_breakeven_recommendation: &'static str,
    pub above_breakeven_recommendation: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beaming::nicoll_dyson_phased_array::PhasedArrayConfig;
    
    #[test]
    fn test_bussard_power_calculation() {
        type F = f64;
        let ramjet = BussardRamjet::new(
            F::from(1e6).unwrap(),     // 1000 km scoop
            F::from(0.5).unwrap(),     // 50% collection efficiency
            F::from(0.007).unwrap(),   // ~0.7% fusion efficiency (H->He)
            F::from(1.67e-21).unwrap(), // Typical ISM density
        );
        
        let c = F::from(3e8).unwrap();
        let velocity = F::from(1e7).unwrap();  // ~3% c
        
        let power = ramjet.calculate_power_generation(velocity, c);
        assert!(power > F::zero());
    }
    
    #[test]
    fn test_arbitrage_initialization() {
        type F = f64;
        let config = PhasedArrayConfig::new(
            F::from(1e-6).unwrap(),
            F::from(2e-6).unwrap(),
            100,
            F::from(1e-6).unwrap(),
            F::from(100.0).unwrap(),
        );
        
        let array = NicollDysonArray::new(config, SVector::zeros());
        let ramjet = BussardRamjet::new(
            F::from(1e6).unwrap(),
            F::from(0.5).unwrap(),
            F::from(0.007).unwrap(),
            F::from(1.67e-21).unwrap(),
        );
        
        let arbitrage = InterstellarEnergyArbitrage::new(array, ramjet, F::from(0.05).unwrap());
        
        assert!(arbitrage.energy_price_at_source > F::zero());
    }
}
