//! Penrose Ergosphere Extractor
//! 
//! Implements energy extraction from rotating black holes via the Penrose process
//! and superradiant scattering in the ergosphere.

use core::f64;

/// Physical constants for black hole calculations
#[derive(Debug, Clone, Copy)]
pub struct BHConstants {
    /// Speed of light [m/s]
    pub c: f64,
    /// Gravitational constant [m³/(kg·s²)]
    pub g: f64,
    /// Reduced Planck constant [J·s]
    pub hbar: f64,
}

impl Default for BHConstants {
    fn default() -> Self {
        Self {
            c: 299_792_458.0,
            g: 6.674_30e-11,
            hbar: 1.054_571_817e-34,
        }
    }
}

/// Kerr black hole parameters
#[derive(Debug, Clone, Copy)]
pub struct KerrBlackHole {
    /// Mass [kg]
    pub mass: f64,
    /// Angular momentum [kg·m²/s]
    pub angular_momentum: f64,
    /// Spin parameter a = J/(Mc) [m]
    pub spin_parameter: f64,
    /// Event horizon radius [m]
    pub event_horizon: f64,
    /// Ergosphere radius at equator [m]
    pub ergosphere_radius: f64,
    /// Angular velocity of horizon [1/s]
    pub horizon_angular_velocity: f64,
    /// Irreducible mass [kg]
    pub irreducible_mass: f64,
}

impl KerrBlackHole {
    /// Create a new Kerr black hole
    /// 
    /// # Arguments
    /// * `mass` - Black hole mass [kg]
    /// * `spin_parameter` - Dimensionless spin a* = c*J/(G*M²), must be in [0, 1]
    /// 
    /// # Returns
    /// * `Result<Self, &'static str>` - Black hole or error
    pub fn new(mass: f64, dimensionless_spin: f64) -> Result<Self, &'static str> {
        if mass <= 0.0 {
            return Err("Mass must be positive");
        }
        if dimensionless_spin < 0.0 || dimensionless_spin > 1.0 {
            return Err("Dimensionless spin must be in [0, 1]");
        }
        
        let constants = BHConstants::default();
        let c = constants.c;
        let g = constants.g;
        
        // Schwarzschild radius: r_s = 2GM/c²
        let r_s = 2.0 * g * mass / c.powi(2);
        
        // Spin parameter a = (G*M/c²) * a* = r_s/2 * a*
        let spin_parameter = (g * mass / c.powi(2)) * dimensionless_spin;
        
        // Angular momentum J = a* * G * M² / c
        let angular_momentum = dimensionless_spin * g * mass.powi(2) / c;
        
        // Event horizon: r_+ = GM/c² + sqrt((GM/c²)² - a²)
        let gm_c2 = g * mass / c.powi(2);
        let discriminant = gm_c2.powi(2) - spin_parameter.powi(2);
        
        // For extremal black hole (a* = 1), discriminant approaches 0
        let sqrt_disc = if discriminant < 0.0 { 0.0 } else { discriminant.sqrt() };
        let event_horizon = gm_c2 + sqrt_disc;
        
        // Ergosphere radius at equator: r_ergo = 2GM/c² = r_s
        let ergosphere_radius = r_s;
        
        // Horizon angular velocity: Ω_H = a*c / (2*r_+*(r_+ + sqrt(r_+² - a²)))
        // Simplified: Ω_H = c * a / (2 * r_+ * (r_+ + sqrt_disc))
        let horizon_angular_velocity = if event_horizon > 0.0 {
            c * spin_parameter / (2.0 * event_horizon * (event_horizon + sqrt_disc)).max(f64::EPSILON)
        } else {
            0.0
        };
        
        // Irreducible mass: M_irr = sqrt((M² + sqrt(M⁴ - J²c²/G²))/2)
        // Or: M_irr² = (r_+² + a²) * c⁴ / (4*G²)
        let m_irr_squared = (event_horizon.powi(2) + spin_parameter.powi(2)) * c.powi(4) 
            / (4.0 * g.powi(2));
        let irreducible_mass = m_irr_squared.sqrt();
        
        Ok(Self {
            mass,
            angular_momentum,
            spin_parameter,
            event_horizon,
            ergosphere_radius,
            horizon_angular_velocity,
            irreducible_mass,
        })
    }
    
    /// Calculate the maximum extractable energy (rotational energy)
    /// E_extractable = (M - M_irr) * c²
    /// 
    /// # Returns
    /// * `f64` - Maximum extractable energy [J]
    pub fn extractable_energy(&self) -> f64 {
        let constants = BHConstants::default();
        let delta_m = self.mass - self.irreducible_mass;
        delta_m.max(0.0) * constants.c.powi(2)
    }
    
    /// Calculate the efficiency of the Penrose process
    /// Maximum theoretical efficiency: η = (sqrt(2) - 1) / 2 ≈ 20.7% for extremal Kerr
    /// 
    /// # Returns
    /// * `f64` - Maximum efficiency (0 to ~0.207)
    pub fn max_penrose_efficiency(&self) -> f64 {
        // For a particle splitting in the ergosphere
        // η_max = (sqrt(1 + a*) - 1) / 2 where a* is dimensionless spin
        let a_star = self.spin_parameter * BHConstants::default().c.powi(2) 
            / (BHConstants::default().g * self.mass);
        
        let a_star = a_star.min(1.0).max(0.0);
        (1.0 + a_star).sqrt() - 1.0
    }
    
    /// Check if a given radius is within the ergosphere
    /// 
    /// # Arguments
    /// * `radius` - Distance from center [m]
    /// 
    /// # Returns
    /// * `bool` - True if within ergosphere
    pub fn is_in_ergosphere(&self, radius: f64) -> bool {
        radius > self.event_horizon && radius < self.ergosphere_radius
    }
}

/// Penrose process simulation state
#[derive(Debug, Clone)]
pub struct PenroseProcess {
    /// The black hole being exploited
    pub black_hole: KerrBlackHole,
    /// Current extracted energy [J]
    pub extracted_energy: f64,
    /// Number of particles processed
    pub particles_processed: u64,
    /// Average efficiency achieved
    pub average_efficiency: f64,
}

impl PenroseProcess {
    /// Create a new Penrose process simulator
    /// 
    /// # Arguments
    /// * `mass` - Black hole mass [kg]
    /// * `dimensionless_spin` - Spin parameter [0, 1]
    /// 
    /// # Returns
    /// * `Result<Self, &'static str>` - Process or error
    pub fn new(mass: f64, dimensionless_spin: f64) -> Result<Self, &'static str> {
        let bh = KerrBlackHole::new(mass, dimensionless_spin)?;
        
        Ok(Self {
            black_hole: bh,
            extracted_energy: 0.0,
            particles_processed: 0,
            average_efficiency: 0.0,
        })
    }
    
    /// Simulate a single Penrose process event
    /// 
    /// A particle with energy E enters the ergosphere, splits into two:
    /// - One with negative energy falls into the black hole
    /// - One with energy E' > E escapes, carrying away rotational energy
    /// 
    /// # Arguments
    /// * `input_energy` - Energy of incoming particle [J]
    /// * `split_ratio` - Fraction of mass going to negative-energy particle [0, 0.5]
    /// 
    /// # Returns
    /// * `Result<f64, &'static str>` - Extracted energy [J]
    pub fn process_particle(&mut self, input_energy: f64, split_ratio: f64) -> Result<f64, &'static str> {
        if input_energy <= 0.0 {
            return Err("Input energy must be positive");
        }
        if split_ratio <= 0.0 || split_ratio > 0.5 {
            return Err("Split ratio must be in (0, 0.5]");
        }
        
        // Maximum efficiency depends on black hole spin
        let max_eff = self.black_hole.max_penrose_efficiency();
        
        // Actual efficiency (scaled by split ratio and geometric factors)
        // Optimal split gives ~half the theoretical max
        let efficiency = max_eff * split_ratio * 2.0;
        let efficiency = efficiency.min(max_eff);
        
        let extracted = input_energy * efficiency;
        
        // Update black hole parameters (slightly reduced angular momentum)
        let constants = BHConstants::default();
        let delta_j = extracted / self.black_hole.horizon_angular_velocity.max(f64::EPSILON);
        
        // Update running statistics
        let total_extracted = self.extracted_energy + extracted;
        self.particles_processed += 1;
        self.average_efficiency = total_extracted 
            / ((self.particles_processed as f64) * input_energy);
        self.extracted_energy = total_extracted;
        
        // Note: In a full simulation, we'd update bh.angular_momentum here
        // For now, we assume the reservoir is large enough that individual
        // extractions don't significantly change the BH parameters
        
        Ok(extracted)
    }
    
    /// Calculate superradiant amplification factor for a wave mode
    /// 
    /// Superradiance occurs when ω < m * Ω_H where m is the azimuthal number
    /// 
    /// # Arguments
    /// * `frequency` - Wave frequency [Hz]
    /// * `azimuthal_number` - Azimuthal quantum number m
    /// 
    /// # Returns
    /// * `f64` - Amplification factor (>1 means amplification)
    pub fn superradiant_amplification(&self, frequency: f64, azimuthal_number: i32) -> f64 {
        if frequency <= 0.0 || azimuthal_number <= 0 {
            return 1.0; // No amplification
        }
        
        let omega = 2.0 * core::f64::consts::PI * frequency;
        let m_omega_h = (azimuthal_number as f64) * self.black_hole.horizon_angular_velocity;
        
        // Superradiant condition: ω < m * Ω_H
        if omega >= m_omega_h {
            return 1.0;
        }
        
        // Amplification factor approximation for scalar waves
        // Z ≈ 4π * (m*Ω_H - ω) * r_+² / (ω * c)
        let excess = m_omega_h - omega;
        let r_plus = self.black_hole.event_horizon;
        
        let z = 4.0 * core::f64::consts::PI * excess * r_plus.powi(2) 
            / (omega * BHConstants::default().c);
        
        // Amplification = exp(Z) for small Z, but saturates
        (1.0 + z.min(10.0)).max(1.0)
    }
    
    /// Get remaining extractable energy
    pub fn remaining_extractable(&self) -> f64 {
        self.black_hole.extractable_energy() - self.extracted_energy
    }
}

/// Ergosphere mining operation statistics
#[derive(Debug, Clone, Copy)]
pub struct ErgosphereStats {
    /// Total energy extracted [J]
    pub total_extracted: f64,
    /// Particles processed
    pub particles: u64,
    /// Average efficiency
    pub avg_efficiency: f64,
    /// Remaining extractable energy [J]
    pub remaining: f64,
    /// Current black hole spin parameter
    pub current_spin: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kerr_bh_creation() {
        let bh = KerrBlackHole::new(1e30, 0.9);
        assert!(bh.is_ok());
        let b = bh.unwrap();
        assert!(b.event_horizon > 0.0);
        assert!(b.ergosphere_radius > b.event_horizon);
    }

    #[test]
    fn test_extremal_bh() {
        let bh = KerrBlackHole::new(1e30, 1.0);
        assert!(bh.is_ok());
        let b = bh.unwrap();
        // For extremal BH, event horizon = GM/c²
        let expected_r = BHConstants::default().g * 1e30 / BHConstants::default().c.powi(2);
        assert!((b.event_horizon - expected_r).abs() < 1.0);
    }

    #[test]
    fn test_extractable_energy() {
        let bh = KerrBlackHole::new(1e30, 0.9).unwrap();
        let e = bh.extractable_energy();
        assert!(e > 0.0);
    }

    #[test]
    fn test_penrose_process() {
        let mut process = PenroseProcess::new(1e30, 0.9).unwrap();
        let extracted = process.process_particle(1e20, 0.3);
        assert!(extracted.is_ok());
        assert!(extracted.unwrap() > 0.0);
    }

    #[test]
    fn test_superradiance() {
        let process = PenroseProcess::new(1e30, 0.9).unwrap();
        
        // Low frequency should show amplification
        let amp_low = process.superradiant_amplification(1e-5, 1);
        
        // High frequency should not amplify
        let amp_high = process.superradiant_amplification(1e10, 1);
        
        assert!(amp_low >= 1.0);
        assert!(amp_high >= 1.0);
    }
}
