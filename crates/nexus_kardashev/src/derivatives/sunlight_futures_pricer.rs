//! Sunlight Futures Pricer for Dyson Swarm Orbital Economics
//! 
//! Implements pricing models for sunlight derivatives, orbital slot leases,
//! and shade compensation contracts in the Dyson swarm economy.

use crate::orbital::resonance_stabilizer::{OrbitalSlot, ResonanceStabilizer, ShadeDerivative, SunlightFuture};
use num_traits::{Float, Zero};

/// Types of sunlight-related derivatives
#[derive(Debug, Clone)]
pub enum SunlightDerivative {
    /// Direct sunlight delivery contract
    InsolationContract {
        contract_id: u64,
        slot_id: u64,
        kwh_quantity: f64,
        delivery_start_day: u32,
        duration_days: u32,
    },
    /// Shade avoidance insurance
    ShadeInsurance {
        contract_id: u64,
        protected_slot: u64,
        coverage_fraction: f64,
        premium_per_day: f64,
    },
    /// Orbital slot lease with sunlight rights
    SlotLease {
        slot_id: u64,
        lease_duration_days: u32,
        exclusivity_level: ExclusivityLevel,
    },
}

/// Levels of orbital slot exclusivity
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExclusivityLevel {
    /// Shared slot, time-multiplexed access
    TimeShared,
    /// Primary rights, secondary can be denied
    Primary,
    /// Exclusive use, no other collectors allowed
    Exclusive,
}

impl ExclusivityLevel {
    pub fn price_multiplier(&self) -> f64 {
        match self {
            ExclusivityLevel::TimeShared => 0.5,
            ExclusivityLevel::Primary => 1.0,
            ExclusivityLevel::Exclusive => 3.0,
        }
    }
}

/// Sunlight futures pricer
pub struct SunlightFuturesPricer<T> {
    base_solar_price: T,  // $/kWh at 1 AU
    risk_free_rate: T,
    volatility: T,
}

impl<T: Float + Copy + Zero> SunlightFuturesPricer<T> {
    pub fn new(base_price: T, risk_free_rate: T, volatility: T) -> Self {
        Self {
            base_solar_price: base_price,
            risk_free_rate,
            volatility,
        }
    }
    
    /// Price an insolation contract based on orbital parameters
    pub fn price_insolation_contract(
        &self,
        slot: &OrbitalSlot<T>,
        kwh_quantity: f64,
        duration_days: u32,
    ) -> ContractQuote {
        // Calculate effective solar constant at this orbit
        let au_distance = T::from(1.496e11).unwrap();
        let distance_ratio = au_distance / slot.semi_major_axis;
        let local_solar_flux = self.base_solar_price * distance_ratio * distance_ratio;
        
        // Adjust for eccentricity (varying distance over orbit)
        let ecc_factor = T::one() - slot.eccentricity * slot.eccentricity;
        let avg_flux = local_solar_flux * ecc_factor;
        
        // Eclipse losses
        let eclipse_fraction = self.compute_eclipse_fraction(slot);
        let effective_flux = avg_flux * (T::one() - eclipse_fraction);
        
        // Forward pricing with interest
        let rf = self.risk_free_rate;
        let dt = T::from(duration_days as f64).unwrap() / T::from(365.0).unwrap();
        let forward_factor = T::one() + rf * dt;
        
        // Total contract value
        let quantity_t = T::from(kwh_quantity).unwrap();
        let base_value = effective_flux * quantity_t;
        let forward_value = base_value * forward_factor;
        
        ContractQuote {
            contract_value: forward_value.to_f64().unwrap_or(0.0),
            effective_price_per_kwh: effective_flux.to_f64().unwrap_or(0.0),
            capacity_factor: (T::one() - eclipse_fraction).to_f64().unwrap_or(0.0),
            duration_days,
        }
    }
    
    /// Price shade insurance policy
    pub fn price_shade_insurance(
        &self,
        stabilizer: &ResonanceStabilizer<T>,
        protected_slot: u64,
        coverage_fraction: f64,
    ) -> Option<InsuranceQuote> {
        // Find potential shading slots
        let mut total_shade_risk = T::zero();
        
        for slot in &stabilizer.slots {
            if let Some(shade) = stabilizer.calculate_shade_derivative(
                slot.slot_id,
                protected_slot,
                self.base_solar_price * T::from(1000.0).unwrap(),  // Assume 1 MW collector
                T::from(1.0).unwrap(),
            ) {
                total_shade_risk = total_shade_risk + shade.shade_fraction;
            }
        }
        
        if total_shade_risk <= T::zero() {
            return Some(InsuranceQuote {
                daily_premium: 0.0,
                coverage_amount: 0.0,
                probability_of_shade: 0.0,
            });
        }
        
        // Expected daily loss
        let expected_loss = total_shade_risk * self.base_solar_price * T::from(1000.0).unwrap();
        let coverage_t = T::from(coverage_fraction).unwrap();
        let covered_loss = expected_loss * coverage_t;
        
        // Risk loading
        let risk_loading = T::from(1.3).unwrap();
        let premium = covered_loss * risk_loading;
        
        Some(InsuranceQuote {
            daily_premium: premium.to_f64().unwrap_or(0.0),
            coverage_amount: covered_loss.to_f64().unwrap_or(0.0),
            probability_of_shade: total_shade_risk.to_f64().unwrap_or(0.0),
        })
    }
    
    /// Price orbital slot lease
    pub fn price_slot_lease(
        &self,
        slot: &OrbitalSlot<T>,
        duration_days: u32,
        exclusivity: ExclusivityLevel,
    ) -> LeaseQuote {
        // Base value from sunlight capture potential
        let au_distance = T::from(1.496e11).unwrap();
        let distance_ratio = au_distance / slot.semi_major_axis;
        let local_solar_flux = self.base_solar_price * distance_ratio * distance_ratio;
        
        // Assume standard 1 km² collector array
        let collector_area = T::from(1e6).unwrap();  // m²
        let daily_energy = local_solar_flux * collector_area * T::from(24.0).unwrap();
        
        // Apply exclusivity multiplier
        let exclusivity_mult = T::from(exclusivity.price_multiplier()).unwrap();
        let daily_value = daily_energy * exclusivity_mult;
        
        // Lease present value
        let rf = self.risk_free_rate;
        let dt = T::from(1.0).unwrap() / T::from(365.0).unwrap();
        let discount_factor = T::one() / (T::one() + rf * dt);
        
        let mut pv = T::zero();
        for _ in 0..duration_days {
            pv = pv + daily_value * discount_factor;
        }
        
        // Orbital stability discount/premium
        let stability_factor = if slot.eccentricity < T::from(0.01).unwrap() {
            T::from(1.0).unwrap()  // Stable circular orbit
        } else {
            T::from(0.8).unwrap()  // Eccentric requires more station-keeping
        };
        
        LeaseQuote {
            total_lease_value: (pv * stability_factor).to_f64().unwrap_or(0.0),
            daily_rate: daily_value.to_f64().unwrap_or(0.0),
            exclusivity_level: exclusivity,
            duration_days,
        }
    }
    
    /// Compute eclipse fraction for a slot
    fn compute_eclipse_fraction(&self, slot: &OrbitalSlot<T>) -> T {
        let stellar_radius = T::from(6.9634e8).unwrap();
        let distance = slot.semi_major_axis;
        
        if distance <= stellar_radius {
            return T::one();
        }
        
        let angular_radius = (stellar_radius / distance).asin();
        let eclipse_arc = T::from(2.0).unwrap() * angular_radius;
        let two_pi = T::from(2.0).unwrap() * T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        
        eclipse_arc / two_pi
    }
    
    /// Calculate arbitrage opportunity between different orbital shells
    pub fn find_arbitrage_opportunities(
        &self,
        slots: &[OrbitalSlot<T>],
    ) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        
        for i in 0..slots.len() {
            for j in (i + 1)..slots.len() {
                let slot_a = &slots[i];
                let slot_b = &slots[j];
                
                // Compare effective prices
                let quote_a = self.price_insolation_contract(slot_a, 1000.0, 30);
                let quote_b = self.price_insolation_contract(slot_b, 1000.0, 30);
                
                let price_diff = quote_a.effective_price_per_kwh - quote_b.effective_price_per_kwh;
                let threshold = T::from(0.01).unwrap();
                
                if price_diff.abs() > threshold {
                    opportunities.push(ArbitrageOpportunity {
                        buy_slot: if price_diff < T::zero() { slot_a.slot_id } else { slot_b.slot_id },
                        sell_slot: if price_diff < T::zero() { slot_b.slot_id } else { slot_a.slot_id },
                        expected_profit_fraction: price_diff.abs().to_f64().unwrap_or(0.0),
                    });
                }
            }
        }
        
        opportunities
    }
}

/// Quote for insolation contract
#[derive(Debug, Clone)]
pub struct ContractQuote {
    pub contract_value: f64,
    pub effective_price_per_kwh: f64,
    pub capacity_factor: f64,
    pub duration_days: u32,
}

/// Quote for shade insurance
#[derive(Debug, Clone)]
pub struct InsuranceQuote {
    pub daily_premium: f64,
    pub coverage_amount: f64,
    pub probability_of_shade: f64,
}

/// Quote for orbital lease
#[derive(Debug, Clone)]
pub struct LeaseQuote {
    pub total_lease_value: f64,
    pub daily_rate: f64,
    pub exclusivity_level: ExclusivityLevel,
    pub duration_days: u32,
}

/// Arbitrage opportunity between slots
#[derive(Debug, Clone)]
pub struct ArbitrageOpportunity {
    pub buy_slot: u64,
    pub sell_slot: u64,
    pub expected_profit_fraction: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbital::resonance_stabilizer::ResonanceStabilizer;
    
    #[test]
    fn test_insolation_pricing() {
        type F = f64;
        let pricer = SunlightFuturesPricer::new(
            F::from(0.05).unwrap(),
            F::from(0.05).unwrap(),
            F::from(0.2).unwrap(),
        );
        
        let slot = OrbitalSlot::new(
            1,
            F::from(1.496e11).unwrap(),
            F::from(0.0).unwrap(),
            F::from(0.0).unwrap(),
            F::from(0.0).unwrap(),
            F::from(0.0).unwrap(),
            F::from(0.0).unwrap(),
        );
        
        let quote = pricer.price_insolation_contract(&slot, 1000.0, 30);
        
        assert!(quote.contract_value > 0.0);
        assert!(quote.capacity_factor > 0.9);  // Minimal eclipse at 1 AU
    }
    
    #[test]
    fn test_lease_pricing() {
        type F = f64;
        let pricer = SunlightFuturesPricer::new(
            F::from(0.05).unwrap(),
            F::from(0.05).unwrap(),
            F::from(0.2).unwrap(),
        );
        
        let slot = OrbitalSlot::new(
            1,
            F::from(1.496e11).unwrap(),
            F::from(0.0).unwrap(),
            F::from(0.0).unwrap(),
            F::from(0.0).unwrap(),
            F::from(0.0).unwrap(),
            F::from(0.0).unwrap(),
        );
        
        let quote = pricer.price_slot_lease(&slot, 365, ExclusivityLevel::Exclusive);
        
        assert!(quote.total_lease_value > 0.0);
        assert_eq!(quote.exclusivity_level, ExclusivityLevel::Exclusive);
    }
}
