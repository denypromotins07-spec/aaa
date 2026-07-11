//! Orbital Resonance Stabilizer for Dyson Swarm
//! 
//! Manages orbital slot allocation and resonance stabilization to prevent
//! collisions and optimize sunlight distribution across the swarm.

use crate::orbital::symplectic_nbody_integrator::{BodyState, GravitationalSystem, ResonanceCalculator};
use nalgebra::SVector;
use num_traits::{Float, Zero};

/// Orbital slot definition for Dyson swarm mirrors
#[derive(Debug, Clone)]
pub struct OrbitalSlot<T> {
    pub slot_id: u64,
    pub semi_major_axis: T,
    pub eccentricity: T,
    pub inclination: T,
    pub longitude_asc_node: T,
    pub arg_periapsis: T,
    pub mean_anomaly: T,
    pub assigned_to: Option<u64>,  // Replicator ID
}

impl<T: Float + Copy + Zero> OrbitalSlot<T> {
    pub fn new(
        slot_id: u64,
        sma: T,
        ecc: T,
        inc: T,
        lan: T,
        arg_p: T,
        m_anomaly: T,
    ) -> Self {
        Self {
            slot_id,
            semi_major_axis: sma,
            eccentricity: ecc,
            inclination: inc,
            longitude_asc_node: lan,
            arg_periapsis: arg_p,
            mean_anomaly: m_anomaly,
            assigned_to: None,
        }
    }
    
    /// Check if slot is in resonant configuration with another
    pub fn check_resonance_with(&self, other: &OrbitalSlot<T>, calc: &ResonanceCalculator<T>) -> Option<(u32, u32)> {
        // Check common resonances: 1:2, 2:3, 3:4, etc.
        let tolerances = [0.01, 0.02, 0.05];
        
        for (p, q) in [(1, 2), (2, 3), (3, 4), (2, 1), (3, 2), (4, 3)] {
            for &tol in &tolerances {
                let tol_t = T::from(tol).unwrap();
                if calc.check_resonance(self.semi_major_axis, other.semi_major_axis, p, q, tol_t) {
                    return Some((p, q));
                }
            }
        }
        
        None
    }
}

/// Shade derivative - cost of blocking sunlight to another collector
#[derive(Debug, Clone)]
pub struct ShadeDerivative<T> {
    pub shading_slot: u64,
    pub shaded_slot: u64,
    pub shade_fraction: T,
    pub power_loss_usd_per_hour: T,
    pub duration_hours: T,
}

impl<T: Float + Copy + Zero> ShadeDerivative<T> {
    pub fn calculate_compensation(&self) -> T {
        self.power_loss_usd_per_hour * self.duration_hours * self.shade_fraction
    }
}

/// Resonance stabilizer manager
pub struct ResonanceStabilizer<T> {
    slots: Vec<OrbitalSlot<T>>,
    calculator: ResonanceCalculator<T>,
    min_separation_angle: T,
}

impl<T: Float + Copy + Zero> ResonanceStabilizer<T> {
    pub fn new(mu: T, min_separation_degrees: T) -> Self {
        let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        let deg_to_rad = pi / T::from(180.0).unwrap();
        
        Self {
            slots: Vec::new(),
            calculator: ResonanceCalculator::new(mu),
            min_separation_angle: min_separation_degrees * deg_to_rad,
        }
    }
    
    /// Add an orbital slot to the registry
    pub fn register_slot(&mut self, slot: OrbitalSlot<T>) -> Result<(), RegistrationError> {
        // Check for conflicts with existing slots
        for existing in &self.slots {
            if self.check_collision_risk(&slot, existing) {
                return Err(RegistrationError::CollisionRisk(slot.slot_id, existing.slot_id));
            }
        }
        
        self.slots.push(slot);
        Ok(())
    }
    
    /// Check collision risk between two slots
    fn check_collision_risk(&self, a: &OrbitalSlot<T>, b: &OrbitalSlot<T>) -> bool {
        // Simplified: check if orbits intersect and phasing could cause close approach
        
        // Same orbital plane check
        let inc_diff = (a.inclination - b.inclination).abs();
        let lan_diff = (a.longitude_asc_node - b.longitude_asc_node).abs();
        
        let two = T::one() + T::one();
        let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        
        if inc_diff < self.min_separation_angle && lan_diff < self.min_separation_angle {
            // Nearly coplanar - check radial separation
            let sma_diff = (a.semi_major_axis - b.semi_major_axis).abs();
            let max_ecc_radius = a.eccentricity.max(b.eccentricity) * a.semi_major_axis.max(b.semi_major_axis);
            
            // If orbital bands overlap, collision risk exists
            if sma_diff < max_ecc_radius * two {
                return true;
            }
        }
        
        false
    }
    
    /// Compute optimal phase angles to minimize collision risk
    pub fn compute_optimal_phasing(&self, slot_id: u64) -> Option<PhasingSolution<T>> {
        let slot = self.slots.iter().find(|s| s.slot_id == slot_id)?;
        
        // Find neighbors in similar orbits
        let neighbors: Vec<&OrbitalSlot<T>> = self.slots.iter()
            .filter(|s| s.slot_id != slot_id)
            .filter(|s| {
                let sma_diff = (s.semi_major_axis - slot.semi_major_axis).abs();
                sma_diff < slot.semi_major_axis * T::from(0.01).unwrap()
            })
            .collect();
        
        if neighbors.is_empty() {
            return Some(PhasingSolution {
                slot_id,
                recommended_mean_anomaly: slot.mean_anomaly,
                stability_score: T::one(),
            });
        }
        
        // Compute mean anomaly that maximizes separation
        let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        let two_pi = T::from(2.0).unwrap() * pi;
        let n_neighbors = T::from(neighbors.len() as f64 + 1).unwrap();
        
        // Even spacing around orbit
        let target_spacing = two_pi / n_neighbors;
        
        // Find current neighbor anomalies
        let mut neighbor_anomalies: Vec<T> = neighbors.iter()
            .map(|n| n.mean_anomaly)
            .collect();
        neighbor_anomalies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Find largest gap
        let mut best_anomaly = T::zero();
        let mut max_gap = T::zero();
        
        for i in 0..neighbor_anomalies.len() {
            let next_i = (i + 1) % neighbor_anomalies.len();
            let mut gap = neighbor_anomalies[next_i] - neighbor_anomalies[i];
            
            if gap < T::zero() {
                gap = gap + two_pi;
            }
            
            if gap > max_gap {
                max_gap = gap;
                best_anomaly = neighbor_anomalies[i] + gap / T::from(2.0).unwrap();
            }
        }
        
        // Normalize to [0, 2π]
        while best_anomaly >= two_pi {
            best_anomaly = best_anomaly - two_pi;
        }
        while best_anomaly < T::zero() {
            best_anomaly = best_anomaly + two_pi;
        }
        
        let stability = max_gap / target_spacing;
        
        Some(PhasingSolution {
            slot_id,
            recommended_mean_anomaly: best_anomaly,
            stability_score: stability.min(T::one()),
        })
    }
    
    /// Calculate shade derivative between slots
    pub fn calculate_shade_derivative(
        &self,
        shading_slot: u64,
        shaded_slot: u64,
        solar_constant: T,
        collector_area: T,
    ) -> Option<ShadeDerivative<T>> {
        let shadier = self.slots.iter().find(|s| s.slot_id == shading_slot)?;
        let shaded = self.slots.iter().find(|s| s.slot_id == shaded_slot)?;
        
        // Check if shading geometry applies
        // Inner slot can shade outer slot if aligned
        
        if shadier.semi_major_axis >= shaded.semi_major_axis {
            return None;  // Outer can't shade inner
        }
        
        // Compute angular alignment probability
        let angle_diff = (shadier.mean_anomaly - shaded.mean_anomaly).abs();
        let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        
        // Fraction of time slots are aligned within shadow cone
        let shadow_cone_angle = T::from(0.01).unwrap();  // ~0.5 degrees
        let shade_fraction = if angle_diff < shadow_cone_angle || angle_diff > T::from(2.0).unwrap() * pi - shadow_cone_angle {
            T::one()
        } else {
            T::zero()
        };
        
        // Power loss calculation
        let power_intercepted = solar_constant * collector_area * shade_fraction;
        let electricity_value = T::from(0.1).unwrap();  // $/kWh equivalent
        
        Some(ShadeDerivative {
            shading_slot,
            shaded_slot,
            shade_fraction,
            power_loss_usd_per_hour: power_intercepted * electricity_value,
            duration_hours: T::from(1.0).unwrap(),
        })
    }
    
    /// Generate sunlight futures contract for orbital slot
    pub fn price_sunlight_future(
        &self,
        slot_id: u64,
        delivery_days: u32,
        base_price_per_kwh: f64,
    ) -> Option<SunlightFuture<T>> {
        let slot = self.slots.iter().find(|s| s.slot_id == slot_id)?;
        
        // Base insolation at this orbital distance
        let solar_constant_1au = T::from(1361.0).unwrap();  // W/m² at 1 AU
        let au_distance = T::from(1.496e11).unwrap();
        
        let distance_ratio = au_distance / slot.semi_major_axis;
        let local_solar_constant = solar_constant_1au * distance_ratio * distance_ratio;
        
        // Eclipse fraction (time in shadow)
        let eclipse_fraction = self.compute_eclipse_fraction(slot);
        
        // Effective capacity factor
        let capacity_factor = T::one() - eclipse_fraction;
        
        // Forward price with reliability adjustment
        let rf = T::from(0.05).unwrap() / T::from(365.0).unwrap();
        let dt = T::from(delivery_days as f64).unwrap();
        
        let forward_multiplier = T::one() + rf * dt;
        let reliability_adjustment = capacity_factor;
        
        let forward_price = T::from(base_price_per_kwh).unwrap() 
                          * forward_multiplier 
                          * reliability_adjustment;
        
        Some(SunlightFuture {
            slot_id,
            delivery_days,
            forward_price_per_kwh: forward_price,
            capacity_factor,
            eclipse_fraction,
        })
    }
    
    /// Compute fraction of orbit spent in eclipse
    fn compute_eclipse_fraction(&self, slot: &OrbitalSlot<T>) -> T {
        // Simplified: circular orbit, spherical star
        // Eclipse fraction ≈ (stellar angular diameter) / π
        
        let stellar_radius = T::from(6.9634e8).unwrap();  // Solar radius
        let distance = slot.semi_major_axis;
        
        if distance <= stellar_radius {
            return T::one();  // Inside star!
        }
        
        // Angular radius of star from orbit
        let angular_radius = (stellar_radius / distance).asin();
        
        // Eclipse arc length (radians)
        let eclipse_arc = T::from(2.0).unwrap() * angular_radius;
        
        // Fraction of 2π orbit
        let two_pi = T::from(2.0).unwrap() * T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        
        eclipse_arc / two_pi
    }
    
    /// Rebalance swarm to optimal resonant configuration
    pub fn optimize_swarm_configuration(&mut self) -> OptimizationReport<T> {
        let mut adjustments = Vec::new();
        let mut total_stability = T::zero();
        let mut n_adjusted = 0usize;
        
        for slot in &self.slots {
            if let Some(solution) = self.compute_optimal_phasing(slot.slot_id) {
                if solution.stability_score < T::from(0.8).unwrap() {
                    adjustments.push(Adjustment {
                        slot_id: slot.slot_id,
                        parameter: AdjustmentParameter::MeanAnomaly,
                        old_value: slot.mean_anomaly,
                        new_value: solution.recommended_mean_anomaly,
                    });
                    n_adjusted += 1;
                }
                total_stability = total_stability + solution.stability_score;
            }
        }
        
        let avg_stability = if !self.slots.is_empty() {
            total_stability / T::from(self.slots.len() as f64).unwrap()
        } else {
            T::zero()
        };
        
        OptimizationReport {
            adjustments,
            average_stability: avg_stability,
            slots_adjusted: n_adjusted,
        }
    }
}

/// Errors in slot registration
#[derive(Debug, Clone, thiserror::Error)]
pub enum RegistrationError {
    #[error("Collision risk between slots {0} and {1}")]
    CollisionRisk(u64, u64),
    #[error("Invalid orbital elements")]
    InvalidElements,
}

/// Phasing solution for a slot
#[derive(Debug, Clone)]
pub struct PhasingSolution<T> {
    pub slot_id: u64,
    pub recommended_mean_anomaly: T,
    pub stability_score: T,
}

/// Sunlight futures contract
#[derive(Debug, Clone)]
pub struct SunlightFuture<T> {
    pub slot_id: u64,
    pub delivery_days: u32,
    pub forward_price_per_kwh: T,
    pub capacity_factor: T,
    pub eclipse_fraction: T,
}

/// Adjustment recommendation
#[derive(Debug, Clone)]
pub struct Adjustment<T> {
    pub slot_id: u64,
    pub parameter: AdjustmentParameter,
    pub old_value: T,
    pub new_value: T,
}

#[derive(Debug, Clone, Copy)]
pub enum AdjustmentParameter {
    MeanAnomaly,
    SemiMajorAxis,
    Eccentricity,
    Inclination,
}

/// Optimization report
#[derive(Debug, Clone)]
pub struct OptimizationReport<T> {
    pub adjustments: Vec<Adjustment<T>>,
    pub average_stability: T,
    pub slots_adjusted: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_slot_registration() {
        type F = f64;
        let mu = F::from(1.327e20).unwrap();  // Solar μ
        let mut stabilizer = ResonanceStabilizer::new(mu, F::from(1.0).unwrap());
        
        let slot1 = OrbitalSlot::new(
            1,
            F::from(1.0e11).unwrap(),
            F::from(0.01).unwrap(),
            F::from(0.0).unwrap(),
            F::from(0.0).unwrap(),
            F::from(0.0).unwrap(),
            F::from(0.0).unwrap(),
        );
        
        assert!(stabilizer.register_slot(slot1).is_ok());
    }
    
    #[test]
    fn test_sunlight_future_pricing() {
        type F = f64;
        let mu = F::from(1.327e20).unwrap();
        let mut stabilizer = ResonanceStabilizer::new(mu, F::from(1.0).unwrap());
        
        let slot = OrbitalSlot::new(
            1,
            F::from(1.496e11).unwrap(),  // 1 AU
            F::from(0.0).unwrap(),
            F::from(0.0).unwrap(),
            F::from(0.0).unwrap(),
            F::from(0.0).unwrap(),
            F::from(0.0).unwrap(),
        );
        
        stabilizer.slots.push(slot);
        
        let future = stabilizer.price_sunlight_future(1, 30, 0.05);
        assert!(future.is_some());
        
        let f = future.unwrap();
        assert!(f.forward_price_per_kwh > F::zero());
        assert!(f.capacity_factor > F::zero());
    }
}
