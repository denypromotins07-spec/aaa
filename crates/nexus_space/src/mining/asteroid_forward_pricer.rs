//! Asteroid Forward Pricer
//! 
//! Prices Delta-V discounted commodity forwards for asteroid mining missions.

use super::spectroscopic_yield::{MineralComposition, SpectralType, SpectroscopicYieldEstimator};
use super::delta_v_manifold_calculator::{DeltaVBudget, DeltaVManifoldCalculator, OrbitalElements};

/// Error types for forward pricing
#[derive(Debug, Clone, Copy)]
pub enum AsteroidForwardError {
    InvalidCommodityPrice(f64),
    InvalidMissionCost(f64),
    NegativePresentValue,
    NumericalInstability,
}

impl core::fmt::Display for AsteroidForwardError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AsteroidForwardError::InvalidCommodityPrice(p) => {
                write!(f, "Invalid commodity price: {}", p)
            }
            AsteroidForwardError::InvalidMissionCost(c) => {
                write!(f, "Invalid mission cost: {}", c)
            }
            AsteroidForwardError::NegativePresentValue => {
                write!(f, "Negative present value")
            }
            AsteroidForwardError::NumericalInstability => {
                write!(f, "Numerical instability")
            }
        }
    }
}

/// Commodity type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CommodityType {
    Platinum,
    Palladium,
    Rhodium,
    Iron,
    Nickel,
    WaterIce,
}

impl CommodityType {
    /// Get typical price per kg (USD)
    pub fn price_per_kg(&self) -> f64 {
        match self {
            CommodityType::Platinum => 30_000.0,
            CommodityType::Palladium => 70_000.0,
            CommodityType::Rhodium => 150_000.0,
            CommodityType::Iron => 0.1,
            CommodityType::Nickel => 15.0,
            CommodityType::WaterIce => 1000.0, // In-space value
        }
    }
}

/// Asteroid forward contract valuation
#[derive(Debug, Clone, Copy)]
pub struct AsteroidForwardValuation {
    pub gross_resource_value: f64,
    pub mission_cost_usd: f64,
    pub risk_adjusted_value: f64,
    pub npv_usd: f64,
    pub forward_price_per_kg: f64,
    pub mission_feasible: bool,
}

/// Asteroid forward pricer
pub struct AsteroidForwardPricer {
    yield_estimator: SpectroscopicYieldEstimator,
    dv_calculator: DeltaVManifoldCalculator,
    risk_free_rate: f64,
    launch_cost_per_kg_leo: f64,
}

impl AsteroidForwardPricer {
    /// Create new pricer with market parameters
    pub fn new(risk_free_rate: f64, launch_cost_per_kg_leo: f64) -> Result<Self, AsteroidForwardError> {
        if risk_free_rate < 0.0 || risk_free_rate > 1.0 {
            return Err(AsteroidForwardError::NumericalInstability);
        }
        if launch_cost_per_kg_leo <= 0.0 {
            return Err(AsteroidForwardError::InvalidMissionCost(launch_cost_per_kg_leo));
        }
        
        Ok(Self {
            yield_estimator: SpectroscopicYieldEstimator::new(),
            dv_calculator: DeltaVManifoldCalculator::new(),
            risk_free_rate,
            launch_cost_per_kg_leo,
        })
    }
    
    /// Price asteroid forward contract
    pub fn price_forward(
        &self,
        diameter_km: f64,
        density_g_cm3: f64,
        spectral_type: SpectralType,
        albedo: f64,
        earth_elements: &OrbitalElements,
        asteroid_elements: &OrbitalElements,
        mission_duration_years: f64,
        success_probability: f64,
    ) -> Result<AsteroidForwardValuation, AsteroidForwardError> {
        // Estimate composition
        let composition = self.yield_estimator.estimate_composition(spectral_type, albedo);
        
        // Calculate yield
        let pgm_tonnes = self.yield_estimator.estimate_yield(diameter_km, density_g_cm3, &composition)
            .map_err(|_| AsteroidForwardError::NumericalInstability)?;
        
        // Gross resource value (PGM only for simplicity)
        let pgm_kg = pgm_tonnes * 1000.0;
        let avg_pgm_price = (CommodityType::Platinum.price_per_kg() 
            + CommodityType::Palladium.price_per_kg() 
            + CommodityType::Rhodium.price_per_kg()) / 3.0;
        let gross_value = pgm_kg * avg_pgm_price;
        
        // Calculate Delta-V budget
        let dv_budget = self.dv_calculator.calculate_mission_budget(earth_elements, asteroid_elements, false)
            .map_err(|e| match e {
                super::delta_v_manifold_calculator::DeltaVError::ExceedsPhysicalLimit(_) => {
                    AsteroidForwardError::InvalidMissionCost(f64::INFINITY)
                }
                _ => AsteroidForwardError::NumericalInstability,
            })?;
        
        if !dv_budget.mission_feasible {
            return Ok(AsteroidForwardValuation {
                gross_resource_value: gross_value,
                mission_cost_usd: f64::INFINITY,
                risk_adjusted_value: 0.0,
                npv_usd: 0.0,
                forward_price_per_kg: 0.0,
                mission_feasible: false,
            });
        }
        
        // Mission cost estimation
        let payload_mass_kg = self.estimate_payload_mass(diameter_km, density_g_cm3);
        let launch_cost = payload_mass_kg * self.launch_cost_per_kg_leo;
        let spacecraft_cost = 500_000_000.0; // Simplified
        let operations_cost = 100_000_000.0 * mission_duration_years;
        let total_mission_cost = launch_cost + spacecraft_cost + operations_cost;
        
        // Risk adjustment
        let risk_adjusted_value = gross_value * success_probability;
        
        // NPV calculation
        let discount_factor = (1.0 + self.risk_free_rate).powf(-mission_duration_years);
        let expected_cashflow = risk_adjusted_value - total_mission_cost;
        let npv = expected_cashflow * discount_factor;
        
        if npv < 0.0 {
            return Ok(AsteroidForwardValuation {
                gross_resource_value: gross_value,
                mission_cost_usd: total_mission_cost,
                risk_adjusted_value: risk_adjusted_value,
                npv_usd: 0.0,
                forward_price_per_kg: 0.0,
                mission_feasible: true,
            });
        }
        
        // Forward price per kg of PGM
        let forward_price = npv / pgm_kg.max(1.0);
        
        Ok(AsteroidForwardValuation {
            gross_resource_value: gross_value,
            mission_cost_usd: total_mission_cost,
            risk_adjusted_value: risk_adjusted_value,
            npv_usd: npv,
            forward_price_per_kg: forward_price,
            mission_feasible: true,
        })
    }
    
    /// Estimate required payload mass for mining operation
    fn estimate_payload_mass(&self, diameter_km: f64, density_g_cm3: f64) -> f64 {
        // Simplified: scale with asteroid volume
        let volume_km3 = (4.0 / 3.0) * std::f64::consts::PI * (diameter_km / 2.0).powi(3);
        let mass_tonnes = volume_km3 * density_g_cm3 * 1e9;
        
        // Payload is small fraction of extracted mass
        mass_tonnes * 0.001 * 1000.0 // Convert to kg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_forward_pricing() {
        let pricer = AsteroidForwardPricer::new(0.05, 2000.0).unwrap();
        
        let earth = OrbitalElements {
            semi_major_axis_au: 1.0,
            eccentricity: 0.0167,
            inclination_deg: 0.0,
            longitude_asc_node_deg: 0.0,
            argument_periapsis_deg: 0.0,
        };
        
        let asteroid = OrbitalElements {
            semi_major_axis_au: 2.5,
            eccentricity: 0.15,
            inclination_deg: 10.0,
            longitude_asc_node_deg: 45.0,
            argument_periapsis_deg: 30.0,
        };
        
        let result = pricer.price_forward(
            1.0,    // diameter km
            5.0,    // density
            SpectralType::M,
            0.15,   // albedo
            &earth,
            &asteroid,
            5.0,    // years
            0.7,    // success prob
        );
        
        assert!(result.is_ok());
    }
}
