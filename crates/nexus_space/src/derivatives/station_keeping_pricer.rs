//! Station-Keeping Maneuver Derivative Pricer
//! 
//! Prices orbital slot derivatives and station-keeping fuel costs based on collision risk.

use crate::orbital::conjunction_covariance::ConjunctionData;

/// Error types for station-keeping pricer
#[derive(Debug, Clone, Copy)]
pub enum StationKeepingError {
    InvalidDeltaV(f64),
    InvalidFuelMass(f64),
    NegativePropellantFraction,
    NumericalInstability,
}

impl core::fmt::Display for StationKeepingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StationKeepingError::InvalidDeltaV(d) => write!(f, "Invalid delta-V: {}", d),
            StationKeepingError::InvalidFuelMass(m) => write!(f, "Invalid fuel mass: {}", m),
            StationKeepingError::NegativePropellantFraction => {
                write!(f, "Negative propellant fraction")
            }
            StationKeepingError::NumericalInstability => {
                write!(f, "Numerical instability in pricing")
            }
        }
    }
}

/// Tsiolkovsky rocket equation constants
pub const ISP_HYPERGOLIC: f64 = 320.0; // seconds
pub const ISP_ION: f64 = 3000.0; // seconds
pub const GRAVITY_SEA_LEVEL: f64 = 9.80665; // m/s²

/// Station-keeping maneuver parameters
#[derive(Debug, Clone, Copy)]
pub struct StationKeepingParams {
    pub satellite_mass_kg: f64,
    pub specific_impulse_s: f64,
    pub fuel_remaining_kg: f64,
    pub max_delta_v_ms: f64,
}

/// Orbital slot derivative valuation
#[derive(Debug, Clone, Copy)]
pub struct OrbitalSlotValue {
    pub slot_value_usd: f64,
    pub insurance_premium_usd: f64,
    pub maneuver_cost_usd: f64,
    pub risk_adjusted_value: f64,
}

impl StationKeepingParams {
    /// Create new station-keeping parameters with validation
    pub fn new(
        satellite_mass_kg: f64,
        specific_impulse_s: f64,
        fuel_remaining_kg: f64,
    ) -> Result<Self, StationKeepingError> {
        if satellite_mass_kg <= 0.0 {
            return Err(StationKeepingError::InvalidFuelMass(satellite_mass_kg));
        }
        if specific_impulse_s <= 0.0 {
            return Err(StationKeepingError::InvalidDeltaV(specific_impulse_s));
        }
        if fuel_remaining_kg < 0.0 {
            return Err(StationKeepingError::InvalidFuelMass(fuel_remaining_kg));
        }
        
        // Calculate maximum possible delta-V using Tsiolkovsky equation
        let dry_mass = satellite_mass_kg - fuel_remaining_kg;
        if dry_mass <= 0.0 {
            return Err(StationKeepingError::NegativePropellantFraction);
        }
        
        let exhaust_velocity = specific_impulse_s * GRAVITY_SEA_LEVEL;
        let max_delta_v = exhaust_velocity * (satellite_mass_kg / dry_mass).ln();
        
        Ok(Self {
            satellite_mass_kg,
            specific_impulse_s,
            fuel_remaining_kg,
            max_delta_v_ms: max_delta_v,
        })
    }
    
    /// Calculate delta-V required for avoidance maneuver
    pub fn delta_v_for_avoidance(&self, conjunction: &ConjunctionData) -> f64 {
        // Simplified model: delta-V proportional to collision probability and inverse miss distance
        let pc_factor = conjunction.collision_probability.sqrt();
        let distance_factor = 1.0 / (conjunction.miss_distance_km + 0.1);
        
        // Base delta-V scaled by risk factors (m/s)
        let base_dv = 0.5 * pc_factor * distance_factor;
        
        base_dv.min(self.max_delta_v_ms * 0.1) // Cap at 10% of total capability
    }
    
    /// Calculate fuel required for given delta-V
    pub fn fuel_for_delta_v(&self, delta_v_ms: f64) -> Result<f64, StationKeepingError> {
        if delta_v_ms < 0.0 {
            return Err(StationKeepingError::InvalidDeltaV(delta_v_ms));
        }
        
        let dry_mass = self.satellite_mass_kg - self.fuel_remaining_kg;
        if dry_mass <= 0.0 {
            return Err(StationKeepingError::NegativePropellantFraction);
        }
        
        // Tsiolkovsky: delta_v = Isp * g0 * ln((m_dry + m_fuel) / m_dry)
        // Solving for m_fuel: m_fuel = m_dry * (exp(delta_v / (Isp * g0)) - 1)
        let exhaust_velocity = self.specific_impulse_s * GRAVITY_SEA_LEVEL;
        let mass_ratio = (delta_v_ms / exhaust_velocity).exp();
        let fuel_required = dry_mass * (mass_ratio - 1.0);
        
        Ok(fuel_required.max(0.0))
    }
}

/// Station-keeping derivative pricer
pub struct StationKeepingPricer {
    pub fuel_price_per_kg: f64,
    pub insurance_base_rate: f64,
    pub slot_value_base_usd: f64,
}

impl StationKeepingPricer {
    /// Create new pricer with market parameters
    pub fn new(fuel_price_per_kg: f64, insurance_base_rate: f64, slot_value_base_usd: f64) -> Self {
        Self {
            fuel_price_per_kg,
            insurance_base_rate,
            slot_value_base_usd,
        }
    }
    
    /// Price station-keeping maneuver cost
    pub fn price_maneuver(
        &self,
        params: &StationKeepingParams,
        conjunction: &ConjunctionData,
    ) -> Result<f64, StationKeepingError> {
        let delta_v = params.delta_v_for_avoidance(conjunction);
        let fuel_required = params.fuel_for_delta_v(delta_v)?;
        
        Ok(fuel_required * self.fuel_price_per_kg)
    }
    
    /// Price insurance premium based on collision risk
    pub fn price_insurance(
        &self,
        conjunction: &ConjunctionData,
        satellite_value_usd: f64,
    ) -> f64 {
        // Base premium scaled by collision probability
        let risk_multiplier = 1.0 + conjunction.collision_probability * 100.0;
        let annual_rate = self.insurance_base_rate * risk_multiplier;
        
        satellite_value_usd * annual_rate
    }
    
    /// Calculate risk-adjusted orbital slot value
    pub fn valuate_orbital_slot(
        &self,
        params: &StationKeepingParams,
        conjunction: &ConjunctionData,
        satellite_value_usd: f64,
        remaining_lifetime_years: f64,
    ) -> Result<OrbitalSlotValue, StationKeepingError> {
        let maneuver_cost = self.price_maneuver(params, conjunction)?;
        let insurance_premium = self.price_insurance(conjunction, satellite_value_usd);
        
        // Discount future cash flows
        let discount_rate = 0.05; // 5% risk-free rate
        let npv_factor = (1.0 - (1.0 + discount_rate).powf(-remaining_lifetime_years)) / discount_rate;
        
        // Base slot value adjusted for risk
        let risk_factor = 1.0 - conjunction.collision_probability;
        let risk_adjusted_base = self.slot_value_base_usd * risk_factor;
        
        let slot_value = risk_adjusted_base * npv_factor - maneuver_cost - insurance_premium;
        
        Ok(OrbitalSlotValue {
            slot_value_usd: slot_value.max(0.0),
            insurance_premium_usd: insurance_premium,
            maneuver_cost_usd: maneuver_cost,
            risk_adjusted_value: slot_value / self.slot_value_base_usd,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_station_keeping_params() {
        let params = StationKeepingParams::new(1000.0, 320.0, 100.0);
        assert!(params.is_ok());
    }
    
    #[test]
    fn test_tsiolkovsky_equation() {
        let params = StationKeepingParams::new(1000.0, 320.0, 100.0).unwrap();
        let fuel = params.fuel_for_delta_v(100.0);
        assert!(fuel.is_ok());
        assert!(fuel.unwrap() > 0.0);
    }
}
