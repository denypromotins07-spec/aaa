//! Magnetic Containment Field Controller for Stellar Lifting Operations
//! 
//! Implements magnetic field configurations for plasma extraction and containment.
//! Uses multipole expansion and real-time field optimization to maintain stable
//! plasma flow from stellar corona to collection manifolds.

use nalgebra::{SVector, SMatrix, Vector3};
use num_traits::{Float, Zero};
use crate::stellar::mhd_plasma_solver::{MHDState, MHDSolver, MHDError, StellarConstants};

/// Magnetic field configuration types for stellar lifting
#[derive(Debug, Clone, Copy)]
pub enum MagneticConfiguration {
    /// Dipole field for basic plasma guidance
    Dipole,
    /// Quadrupole for focused extraction
    Quadrupole,
    /// Octupole for fine control
    Octupole,
    /// Custom field defined by spherical harmonic coefficients
    Custom { l_max: u32 },
}

/// Magnetic containment field parameters
#[derive(Clone, Debug)]
pub struct MagneticContainmentField<T> {
    pub configuration: MagneticConfiguration,
    pub field_strength: T,        // B₀ in Tesla
    pub focal_point: SVector<T, 3>,
    pub orientation: SVector<T, 3>, // Magnetic axis direction
    pub active_coils: Vec<CoilParameters<T>>,
}

/// Parameters for superconducting coil elements
#[derive(Clone, Debug)]
pub struct CoilParameters<T> {
    pub position: SVector<T, 3>,
    pub normal: SVector<T, 3>,
    pub radius: T,
    pub current: T,  // Amperes
    pub turns: u32,
}

impl<T: Float + Copy + Zero> MagneticContainmentField<T> {
    pub fn new(
        configuration: MagneticConfiguration,
        field_strength: T,
        focal_point: SVector<T, 3>,
        orientation: SVector<T, 3>,
    ) -> Self {
        let norm = orientation.norm();
        let normalized_orientation = if norm > T::zero() {
            orientation / norm
        } else {
            SVector::new(T::one(), T::zero(), T::zero())
        };
        
        Self {
            configuration,
            field_strength,
            focal_point,
            orientation: normalized_orientation,
            active_coils: Vec::new(),
        }
    }
    
    /// Add a superconducting coil to the containment array
    pub fn add_coil(&mut self, coil: CoilParameters<T>) {
        self.active_coils.push(coil);
    }
    
    /// Compute magnetic field at a point using Biot-Savart law for coils
    pub fn compute_field_at(&self, point: SVector<T, 3>) -> SVector<T, 3> {
        let mut b_total = SVector::<T, 3>::zeros();
        
        // Contribution from base configuration
        b_total += self.compute_multipole_field(point);
        
        // Contribution from individual coils
        for coil in &self.active_coils {
            b_total += self.compute_coil_field(point, coil);
        }
        
        b_total
    }
    
    /// Compute multipole expansion field
    fn compute_multipole_field(&self, point: SVector<T, 3>) -> SVector<T, 3> {
        let r_vec = point - self.focal_point;
        let r_mag = r_vec.norm();
        
        if r_mag <= T::zero() {
            return SVector::zeros();
        }
        
        match self.configuration {
            MagneticConfiguration::Dipole => {
                self.compute_dipole_field(r_vec, r_mag)
            }
            MagneticConfiguration::Quadrupole => {
                self.compute_quadrupole_field(r_vec, r_mag)
            }
            MagneticConfiguration::Octupole => {
                self.compute_octupole_field(r_vec, r_mag)
            }
            MagneticConfiguration::Custom { .. } => {
                // Would use spherical harmonic expansion
                self.compute_dipole_field(r_vec, r_mag)
            }
        }
    }
    
    /// Dipole magnetic field: B = (μ₀/4πr³)[3(m·r̂)r̂ - m]
    fn compute_dipole_field(&self, r_vec: SVector<T, 3>, r_mag: T) -> SVector<T, 3> {
        let one = T::one();
        let three = one + one + one;
        
        let r_hat = r_vec / r_mag;
        let m_dot_r = self.orientation.dot(&r_hat);
        
        // B ∝ [3(m·r̂)r̂ - m] / r³
        let factor = self.field_strength / (r_mag * r_mag * r_mag);
        
        let term1 = r_hat.map(|x| x * three * m_dot_r);
        let term2 = self.orientation;
        
        (term1 - term2).map(|x| x * factor)
    }
    
    /// Quadrupole field for focused plasma extraction
    fn compute_quadrupole_field(&self, r_vec: SVector<T, 3>, r_mag: T) -> SVector<T, 3> {
        let two = T::one() + T::one();
        let four = two + two;
        
        // Simplified quadrupole: B ∝ ∇(m·r)/r⁴
        let factor = self.field_strength / (r_mag * r_mag * r_mag * r_mag);
        
        // Linear gradient approximation
        let grad = SMatrix::<T, 3, 3>::new(
            two * r_vec[0], r_vec[1], r_vec[2],
            r_vec[0], two * r_vec[1], r_vec[2],
            r_vec[0], r_vec[1], two * r_vec[2],
        );
        
        grad * self.orientation * factor * four
    }
    
    /// Octupole field for fine control
    fn compute_octupole_field(&self, r_vec: SVector<T, 3>, r_mag: T) -> SVector<T, 3> {
        // Higher-order multipole (simplified)
        let factor = self.field_strength / r_mag.powi(5);
        r_vec.map(|x| x * factor)
    }
    
    /// Compute field from a single circular coil using Biot-Savart
    fn compute_coil_field(&self, point: SVector<T, 3>, coil: &CoilParameters<T>) -> SVector<T, 3> {
        // Vector from coil center to observation point
        let r_vec = point - coil.position;
        
        // Distance along coil axis
        let z = r_vec.dot(&coil.normal);
        
        // Perpendicular distance from axis
        let r_perp_sq = r_vec.dot(&r_vec) - z * z;
        let r_perp = if r_perp_sq > T::zero() {
            r_perp_sq.sqrt()
        } else {
            T::zero()
        };
        
        // On-axis field approximation (valid for r_perp << radius)
        // B_z = (μ₀ N I R²) / [2(R² + z²)^(3/2)]
        let mu0_over_4pi = T::from(1e-7).unwrap_or_else(|| T::one() / T::from(1e7).unwrap());
        let n_turns = T::from(coil.turns as f64).unwrap_or_else(|| T::one());
        let r_sq = coil.radius * coil.radius;
        let z_sq = z * z;
        let denom = (r_sq + z_sq).powf(T::from(1.5).unwrap_or_else(|| T::from(3).unwrap() / T::from(2).unwrap()));
        
        if denom <= T::zero() {
            return SVector::zeros();
        }
        
        let b_magnitude = mu0_over_4pi * T::from(4.0).unwrap() 
                        * T::one() * n_turns * coil.current * r_sq / denom;
        
        coil.normal.map(|x| x * b_magnitude)
    }
    
    /// Calculate magnetic pressure at a point: P_B = B²/(2μ₀)
    pub fn compute_magnetic_pressure(&self, point: SVector<T, 3>) -> T {
        let b_field = self.compute_field_at(point);
        let b_squared = b_field.dot(&b_field);
        let mu0_inv = T::from(7.95774715459e5).unwrap_or_else(|| T::one());
        let two = T::one() + T::one();
        
        b_squared * mu0_inv / two
    }
    
    /// Optimize coil currents to achieve target field configuration
    pub fn optimize_currents(
        &mut self,
        target_points: &[SVector<T, 3>],
        target_fields: &[SVector<T, 3>],
        max_iterations: usize,
    ) -> Result<(), OptimizationError> {
        if target_points.len() != target_fields.len() {
            return Err(OptimizationError::MismatchedTargets);
        }
        
        // Gradient descent optimization (simplified)
        let learning_rate = T::from(0.01).unwrap_or_else(|| T::one() / T::from(100).unwrap());
        
        for _iter in 0..max_iterations {
            let mut gradients: Vec<T> = vec![T::zero(); self.active_coils.len()];
            
            // Compute error at each target point
            for (target_pos, target_field) in target_points.iter().zip(target_fields.iter()) {
                let actual_field = self.compute_field_at(*target_pos);
                let error = (actual_field - *target_field).norm();
                
                // Compute gradient with respect to each coil current
                for (coil_idx, coil) in self.active_coils.iter().enumerate() {
                    // Finite difference approximation
                    let delta = T::from(100.0).unwrap_or_else(|| T::one());
                    let mut perturbed_coil = coil.clone();
                    perturbed_coil.current = coil.current + delta;
                    
                    let perturbed_field = self.compute_coil_field(*target_pos, &perturbed_coil);
                    let derivative = (perturbed_field - actual_field).norm() / delta;
                    
                    gradients[coil_idx] = gradients[coil_idx] + error * derivative;
                }
            }
            
            // Update currents
            for (coil_idx, coil) in self.active_coils.iter_mut().enumerate() {
                coil.current = coil.current - learning_rate * gradients[coil_idx];
                
                // Clamp to physical limits
                let max_current = T::from(1e6).unwrap_or_else(|| T::one());
                if coil.current > max_current {
                    coil.current = max_current;
                } else if coil.current < -max_current {
                    coil.current = -max_current;
                }
            }
        }
        
        Ok(())
    }
}

/// Errors in magnetic field optimization
#[derive(Debug, Clone, thiserror::Error)]
pub enum OptimizationError {
    #[error("Target points and fields have mismatched lengths")]
    MismatchedTargets,
    #[error("Optimization failed to converge")]
    NonConvergence,
    #[error("Physical constraint violation")]
    ConstraintViolation,
}

/// Plasma extraction rate controller based on magnetic field strength
pub struct PlasmaExtractor<T> {
    containment_field: MagneticContainmentField<T>,
    extraction_rate: T,  // kg/s
    target_rate: T,
    constants: StellarConstants<T>,
}

impl<T: Float + Copy + Zero> PlasmaExtractor<T> {
    pub fn new(
        containment_field: MagneticContainmentField<T>,
        target_rate: T,
        constants: StellarConstants<T>,
    ) -> Self {
        Self {
            containment_field,
            extraction_rate: T::zero(),
            target_rate,
            constants,
        }
    }
    
    /// Calculate maximum sustainable extraction rate given field strength
    pub fn calculate_max_extraction_rate(&self, plasma_density: T, plasma_temp: T) -> T {
        // Based on magnetic confinement criterion: β = P_plasma / P_magnetic < β_crit
        // P_plasma = n k_B T
        // P_magnetic = B²/(2μ₀)
        
        let kb = self.constants.boltzmann_constant;
        let n_particles = plasma_density / self.constants.proton_mass;
        let plasma_pressure = n_particles * kb * plasma_temp;
        
        let b_field = self.containment_field.field_strength;
        let mu0_inv = T::from(7.95774715459e5).unwrap_or_else(|| T::one());
        let two = T::one() + T::one();
        let magnetic_pressure = b_field * b_field * mu0_inv / two;
        
        // Critical beta for stable confinement (~0.1 for tokamaks)
        let beta_crit = T::from(0.1).unwrap_or_else(|| T::one() / T::from(10).unwrap());
        
        // Maximum plasma pressure that can be confined
        let max_plasma_pressure = beta_crit * magnetic_pressure;
        
        if plasma_pressure <= T::zero() {
            return T::zero();
        }
        
        // Extraction rate proportional to how much we can safely extract
        let extraction_fraction = if max_plasma_pressure > plasma_pressure {
            T::from(0.5).unwrap_or_else(|| T::one() / T::from(2).unwrap())
        } else {
            max_plasma_pressure / plasma_pressure * T::from(0.3).unwrap_or_else(|| T::from(3).unwrap() / T::from(10).unwrap())
        };
        
        extraction_fraction * plasma_density * T::from(1e6).unwrap_or_else(|| T::one())
    }
    
    /// Adjust field strength to maintain target extraction rate
    pub fn adjust_field_for_target_rate(&mut self, plasma_density: T, plasma_temp: T) -> T {
        let current_max = self.calculate_max_extraction_rate(plasma_density, plasma_temp);
        
        if current_max >= self.target_rate {
            return self.extraction_rate;
        }
        
        // Increase field strength proportionally
        let ratio = self.target_rate / (current_max + T::from(1e-10).unwrap_or_else(|| T::one()));
        let adjustment_factor = T::from(1.1).unwrap_or_else(|| T::one() + T::from(0.1).unwrap());
        
        self.containment_field.field_strength = 
            self.containment_field.field_strength * adjustment_factor * ratio.min(T::from(2.0).unwrap());
        
        self.extraction_rate = self.target_rate.min(current_max * adjustment_factor);
        self.extraction_rate
    }
    
    /// Detect magnetic reconnection events that could disrupt extraction
    pub fn detect_reconnection_risk(
        &self,
        mhd_state: &MHDState<T>,
        solver: &MHDSolver<T>,
    ) -> ReconnectionRisk<T> {
        let b_field = mhd_state.magnetic_field;
        let external_field = self.containment_field.compute_field_at(SVector::new(
            mhd_state.momentum[0] / (mhd_state.density + T::from(1e-10).unwrap()),
            mhd_state.momentum[1] / (mhd_state.density + T::from(1e-10).unwrap()),
            mhd_state.momentum[2] / (mhd_state.density + T::from(1e-10).unwrap()),
        ));
        
        // Angle between plasma field and containment field
        let b_mag = b_field.norm();
        let ext_mag = external_field.norm();
        
        if b_mag <= T::zero() || ext_mag <= T::zero() {
            return ReconnectionRisk::Low { angle: T::zero() };
        }
        
        let cos_angle = b_field.dot(&external_field) / (b_mag * ext_mag);
        let angle = cos_angle.acos();
        
        // High risk if fields are anti-parallel (angle near π)
        let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        let threshold = pi * T::from(0.8).unwrap_or_else(|| T::from(8).unwrap() / T::from(10).unwrap());
        
        if angle > threshold {
            ReconnectionRisk::High { angle, probability: T::from(0.9).unwrap() }
        } else if angle > threshold / T::from(2.0).unwrap() {
            ReconnectionRisk::Medium { angle, probability: T::from(0.5).unwrap() }
        } else {
            ReconnectionRisk::Low { angle }
        }
    }
}

/// Risk level for magnetic reconnection
#[derive(Debug, Clone)]
pub enum ReconnectionRisk<T> {
    Low { angle: T },
    Medium { angle: T, probability: T },
    High { angle: T, probability: T },
}

/// Stellar wind deflector for protecting inner Dyson swarm
pub struct StellarWindDeflector<T> {
    deflector_position: SVector<T, 3>,
    deflector_radius: T,
    field_strength: T,
}

impl<T: Float + Copy + Zero> StellarWindDeflector<T> {
    pub fn new(position: SVector<T, 3>, radius: T, field_strength: T) -> Self {
        Self {
            deflector_position: position,
            deflector_radius: radius,
            field_strength,
        }
    }
    
    /// Calculate deflection angle for incoming plasma
    pub fn calculate_deflection_angle(
        &self,
        plasma_velocity: SVector<T, 3>,
        plasma_density: T,
    ) -> T {
        let v_mag = plasma_velocity.norm();
        if v_mag <= T::zero() {
            return T::zero();
        }
        
        // Magnetic mirror effect: particles reflected when B increases
        // Deflection depends on Larmor radius vs deflector size
        let mu0_inv = T::from(7.95774715459e5).unwrap_or_else(|| T::one());
        let proton_mass = T::from(1.67262192369e-27).unwrap_or_else(|| T::zero());
        let elementary_charge = T::from(1.602176634e-19).unwrap_or_else(|| T::zero());
        
        // Larmor radius: r_L = mv⊥/(qB)
        let larmor_radius = if self.field_strength > T::zero() {
            proton_mass * v_mag / (elementary_charge * self.field_strength)
        } else {
            T::zero()
        };
        
        // Deflection efficiency when r_L < deflector size
        let size_ratio = self.deflector_radius / (larmor_radius + T::from(1e-10).unwrap());
        
        // Maximum deflection angle (π for perfect reflection)
        let max_deflection = T::from(std::f64::consts::PI).unwrap_or_else(|| T::from(3.14159).unwrap());
        
        // Empirical deflection model
        let deflection = max_deflection * (T::one() - (-size_ratio).exp());
        
        deflection.min(max_deflection)
    }
    
    /// Check if orbital slot is protected from stellar wind
    pub fn is_protected(&self, orbital_position: SVector<T, 3>) -> bool {
        let displacement = orbital_position - self.deflector_position;
        let distance = displacement.norm();
        
        // Within deflector magnetosphere
        distance < self.deflector_radius * T::from(3.0).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dipole_field_creation() {
        type F = f64;
        let focal_point = SVector::<F, 3>::zeros();
        let orientation = SVector::new(0.0, 0.0, 1.0);
        
        let field = MagneticContainmentField::new(
            MagneticConfiguration::Dipole,
            F::from(1.0).unwrap(),
            focal_point,
            orientation,
        );
        
        assert_eq!(field.orientation, SVector::new(0.0, 0.0, 1.0));
    }
    
    #[test]
    fn test_dipole_field_decay() {
        type F = f64;
        let focal_point = SVector::<F, 3>::zeros();
        let orientation = SVector::new(0.0, 0.0, 1.0);
        
        let field = MagneticContainmentField::new(
            MagneticConfiguration::Dipole,
            F::from(1.0).unwrap(),
            focal_point,
            orientation,
        );
        
        // Field should decrease with distance
        let b_close = field.compute_field_at(SVector::new(1.0, 0.0, 0.0));
        let b_far = field.compute_field_at(SVector::new(2.0, 0.0, 0.0));
        
        assert!(b_close.norm() > b_far.norm());
    }
    
    #[test]
    fn test_plasma_extractor_initialization() {
        type F = f64;
        let focal_point = SVector::<F, 3>::zeros();
        let orientation = SVector::new(0.0, 0.0, 1.0);
        
        let containment = MagneticContainmentField::new(
            MagneticConfiguration::Dipole,
            F::from(0.1).unwrap(),
            focal_point,
            orientation,
        );
        
        let constants = StellarConstants::<F>::default();
        let extractor = PlasmaExtractor::new(containment, F::from(1e6).unwrap(), constants);
        
        assert!(extractor.target_rate > F::zero());
    }
}
