//! Poincaré Phase Space Analyzer
//! 
//! Analyzes the phase space volume of cosmological systems to determine
//! recurrence properties and epoch arbitrage opportunities.

use super::tetration_math_library::{HyperNumber, PoincareRecurrenceCalculator};

/// Phase space cell representation
#[derive(Debug, Clone, Copy)]
pub struct PhaseSpaceCell {
    /// Position coordinates [m]
    pub position: [f64; 3],
    /// Momentum coordinates [kg·m/s]
    pub momentum: [f64; 3],
    /// Cell volume in phase space [m³·(kg·m/s)³]
    pub volume: f64,
}

/// Phase space analysis result
#[derive(Debug, Clone)]
pub struct PhaseSpaceAnalysis {
    /// Total phase space volume [units depend on system]
    pub total_volume: HyperNumber,
    /// Number of accessible microstates
    pub microstates: HyperNumber,
    /// Estimated recurrence time
    pub recurrence_time: HyperNumber,
    /// System entropy [J/K]
    pub entropy: f64,
}

/// Poincaré phase space analyzer
#[derive(Debug, Clone)]
pub struct PhaseSpaceAnalyzer {
    /// Planck constant for quantum cell size
    hbar: f64,
    /// Boltzmann constant
    k_b: f64,
    /// Recurrence calculator
    recurrence_calc: PoincareRecurrenceCalculator,
}

impl Default for PhaseSpaceAnalyzer {
    fn default() -> Self {
        Self {
            hbar: 1.054_571_817e-34,
            k_b: 1.380_649e-23,
            recurrence_calc: PoincareRecurrenceCalculator::default(),
        }
    }
}

impl PhaseSpaceAnalyzer {
    /// Calculate phase space volume for an ideal gas
    /// 
    /// # Arguments
    /// * `n_particles` - Number of particles
    /// * `volume` - Spatial volume [m³]
    /// * `temperature` - Temperature [K]
    /// * `particle_mass` - Mass per particle [kg]
    /// 
    /// # Returns
    /// * `Result<PhaseSpaceAnalysis, &'static str>` - Analysis result
    pub fn analyze_ideal_gas(
        &self,
        n_particles: u64,
        volume: f64,
        temperature: f64,
        particle_mass: f64,
    ) -> Result<PhaseSpaceAnalysis, &'static str> {
        if n_particles == 0 {
            return Err("Must have at least one particle");
        }
        if volume <= 0.0 {
            return Err("Volume must be positive");
        }
        if temperature <= 0.0 {
            return Err("Temperature must be positive");
        }
        if particle_mass <= 0.0 {
            return Err("Particle mass must be positive");
        }
        
        // Thermal de Broglie wavelength: λ = h / sqrt(2πmkT)
        let lambda_db = self.hbar / (2.0 * core::f64::consts::PI * particle_mass * self.k_b * temperature).sqrt();
        
        // Single particle partition function: Z_1 = V / λ³
        let z1 = volume / lambda_db.powi(3);
        
        // N-particle phase space volume (classical, distinguishable)
        // Ω ≈ (V/λ³)^N / N!
        // For large N, use Stirling: ln(N!) ≈ N*ln(N) - N
        
        let n = n_particles as f64;
        let ln_omega = n * z1.ln() - (n * n.ln() - n);
        
        // Entropy: S = k_B * ln(Ω)
        let entropy = self.k_b * ln_omega;
        
        // Microstates: Ω = exp(S/k_B)
        let microstates = if ln_omega > 709.0 {
            // Use hypernumber representation
            HyperNumber::Arrow {
                base: Box::new(HyperNumber::literal(core::f64::consts::E)),
                arrows: 1,
                height: Box::new(HyperNumber::literal(ln_omega)),
            }
        } else {
            HyperNumber::literal(ln_omega.exp())
        };
        
        // Characteristic time: thermal velocity crossing time
        let v_thermal = (self.k_b * temperature / particle_mass).sqrt();
        let linear_size = volume.powf(1.0 / 3.0);
        let tau = linear_size / v_thermal;
        
        // Recurrence time
        let recurrence_time = self.recurrence_calc.calculate_recurrence_time(entropy, tau)?;
        
        Ok(PhaseSpaceAnalysis {
            total_volume: HyperNumber::literal(volume * (2.0 * core::f64::consts::PI * particle_mass * self.k_b * temperature).powf(1.5)),
            microstates,
            recurrence_time,
            entropy,
        })
    }
    
    /// Analyze a black hole's phase space using Bekenstein-Hawking entropy
    /// 
    /// # Arguments
    /// * `mass` - Black hole mass [kg]
    /// 
    /// # Returns
    /// * `Result<PhaseSpaceAnalysis, &'static str>` - Analysis result
    pub fn analyze_black_hole(&self, mass: f64) -> Result<PhaseSpaceAnalysis, &'static str> {
        if mass <= 0.0 {
            return Err("Mass must be positive");
        }
        
        let c = 299_792_458.0;
        let g = 6.674_30e-11;
        
        // Schwarzschild radius
        let r_s = 2.0 * g * mass / c.powi(2);
        
        // Horizon area
        let area = 4.0 * core::f64::consts::PI * r_s.powi(2);
        
        // Planck area
        let planck_area = self.hbar * g / c.powi(3);
        
        // Number of Planck cells on horizon
        let n_cells = area / planck_area;
        
        // Bekenstein-Hawking entropy: S = A/(4*l_P²) * k_B
        let entropy = (n_cells / 4.0) * self.k_b;
        
        // Microstates: Ω = exp(S/k_B) = exp(A/(4*l_P²))
        let ln_omega = n_cells / 4.0;
        let microstates = if ln_omega > 709.0 {
            HyperNumber::tetration(core::f64::consts::E, ln_omega.log10().log10().max(2.0))
        } else {
            HyperNumber::literal(ln_omega.exp())
        };
        
        // Characteristic time: light crossing time
        let tau = r_s / c;
        
        // Recurrence time
        let recurrence_time = self.recurrence_calc.calculate_recurrence_time(entropy, tau)?;
        
        Ok(PhaseSpaceAnalysis {
            total_volume: HyperNumber::literal(area * r_s),
            microstates,
            recurrence_time,
            entropy,
        })
    }
    
    /// Compare phase space volumes of two systems
    /// 
    /// # Arguments
    /// * `analysis1` - First system analysis
    /// * `analysis2` - Second system analysis
    /// 
    /// # Returns
    /// * `i32` - Comparison result (-1, 0, 1)
    pub fn compare_phase_spaces(&self, analysis1: &PhaseSpaceAnalysis, analysis2: &PhaseSpaceAnalysis) -> i32 {
        // Compare by entropy first (directly proportional to log of microstates)
        if analysis1.entropy > analysis2.entropy + 1e-10 {
            1
        } else if analysis2.entropy > analysis1.entropy + 1e-10 {
            -1
        } else {
            // Similar entropy, compare recurrence times
            analysis1.recurrence_time.compare(&analysis2.recurrence_time)
        }
    }
    
    /// Calculate the "epoch arbitrage" value - which cosmological epoch offers
    /// the best compute efficiency based on available phase space
    /// 
    /// # Arguments
    /// * `current_entropy` - Current universe entropy
    /// * `target_entropy` - Target epoch entropy
    /// 
    /// # Returns
    /// * `f64` - Arbitrage ratio (>1 means target is better)
    pub fn epoch_arbitrage_ratio(&self, current_entropy: f64, target_entropy: f64) -> f64 {
        if current_entropy <= 0.0 || target_entropy <= 0.0 {
            return 1.0;
        }
        
        // Lower entropy epochs have more available free energy
        // but less total phase space for computation
        // Optimal is a balance
        
        // Simple model: efficiency ∝ 1/S for low S, but drops for very low S
        let current_efficiency = 1.0 / current_entropy.max(1e-50);
        let target_efficiency = 1.0 / target_entropy.max(1e-50);
        
        target_efficiency / current_efficiency
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ideal_gas_analysis() {
        let analyzer = PhaseSpaceAnalyzer::default();
        
        // 1 mole of gas at STP
        let analysis = analyzer.analyze_ideal_gas(
            6e23, // Avogadro's number
            0.0224, // 22.4 L
            273.0, // 0°C
            4.65e-26, // N2 molecule mass
        );
        
        assert!(analysis.is_ok());
        let a = analysis.unwrap();
        assert!(a.entropy > 0.0);
    }

    #[test]
    fn test_black_hole_analysis() {
        let analyzer = PhaseSpaceAnalyzer::default();
        
        // Solar mass black hole
        let analysis = analyzer.analyze_black_hole(2e30);
        
        assert!(analysis.is_ok());
        let a = analysis.unwrap();
        assert!(a.entropy > 0.0);
    }

    #[test]
    fn test_epoch_arbitrage() {
        let analyzer = PhaseSpaceAnalyzer::default();
        
        // Current universe vs heat death
        let ratio = analyzer.epoch_arbitrage_ratio(1e104, 1e120);
        
        // Heat death has much lower efficiency
        assert!(ratio < 1.0);
    }
}
