//! Phantom Energy Big Rip Hedger
//! 
//! Implements hedging strategies for the Big Rip scenario where dark energy
//! with w < -1 causes the scale factor to diverge in finite time.
//! 
//! CRITICAL: As t -> t_rip, the Hubble parameter H diverges. We enforce a
//! strict temporal cutoff ε > 0 before the singularity to prevent division-by-zero.

use core::f64;

/// Dark energy equation of state parameters
#[derive(Debug, Clone, Copy)]
pub struct DarkEnergyEOS {
    /// Equation of state parameter w = p/ρ
    /// w = -1: cosmological constant
    /// w < -1: phantom energy (Big Rip)
    /// w > -1: quintessence
    pub w: f64,
    /// Current dark energy density [J/m³]
    pub rho_de: f64,
    /// Current Hubble parameter [1/s]
    pub hubble_0: f64,
}

impl Default for DarkEnergyEOS {
    fn default() -> Self {
        Self {
            w: -1.0, // ΛCDM default
            rho_de: 5.9e-10, // Current dark energy density
            hubble_0: 2.2e-18, // ~70 km/s/Mpc
        }
    }
}

/// Big Rip scenario analysis
#[derive(Debug, Clone, Copy)]
pub struct BigRipAnalysis {
    /// Time until Big Rip [s] (infinity if not phantom)
    pub time_to_rip: f64,
    /// Scale factor at current time (normalized to 1)
    pub current_scale: f64,
    /// Hubble parameter at current time [1/s]
    pub current_hubble: f64,
    /// Whether this is a phantom energy scenario
    pub is_phantom: bool,
    /// Safety margin before singularity [s]
    pub safety_margin: f64,
}

/// Migration strategy to false vacuum bubble
#[derive(Debug, Clone)]
pub struct MigrationStrategy {
    /// Required energy for migration [J]
    pub energy_required: f64,
    /// Time available for migration [s]
    pub time_available: f64,
    /// Success probability (0 to 1)
    pub success_probability: f64,
    /// Recommended action
    pub recommendation: MigrationAction,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MigrationAction {
    /// No action needed
    None,
    /// Begin preparation
    Prepare,
    /// Immediate migration required
    MigrateNow,
    /// Too late, situation hopeless
    Hopeless,
}

/// Phantom energy Big Rip hedger
#[derive(Debug, Clone)]
pub struct BigRipHedger {
    /// Dark energy parameters
    eos: DarkEnergyEOS,
    /// Minimum time before singularity (ε cutoff)
    epsilon_cutoff: f64,
    /// Speed of light
    c: f64,
}

impl Default for BigRipHedger {
    fn default() -> Self {
        Self {
            eos: DarkEnergyEOS::default(),
            // 1 Planck time as minimum cutoff
            epsilon_cutoff: 5.39e-44,
            c: 299_792_458.0,
        }
    }
}

impl BigRipHedger {
    /// Create a new hedger with custom parameters
    /// 
    /// # Arguments
    /// * `w` - Dark energy equation of state
    /// * `rho_de` - Dark energy density
    /// * `hubble_0` - Current Hubble parameter
    /// 
    /// # Returns
    /// * `Result<Self, &'static str>` - Hedger or error
    pub fn new(w: f64, rho_de: f64, hubble_0: f64) -> Result<Self, &'static str> {
        if rho_de <= 0.0 {
            return Err("Dark energy density must be positive");
        }
        if hubble_0 <= 0.0 {
            return Err("Hubble parameter must be positive");
        }
        
        Ok(Self {
            eos: DarkEnergyEOS { w, rho_de, hubble_0 },
            epsilon_cutoff: 5.39e-44,
            c: 299_792_458.0,
        })
    }
    
    /// Calculate time until Big Rip
    /// 
    /// For phantom energy (w < -1):
    /// t_rip - t_0 = -2 / (3 * |1+w| * H_0)
    /// 
    /// # Returns
    /// * `BigRipAnalysis` - Analysis result
    pub fn analyze(&self) -> BigRipAnalysis {
        let w = self.eos.w;
        let h_0 = self.eos.hubble_0;
        
        let is_phantom = w < -1.0;
        
        if !is_phantom {
            // No Big Rip for w >= -1
            return BigRipAnalysis {
                time_to_rip: f64::INFINITY,
                current_scale: 1.0,
                current_hubble: h_0,
                is_phantom: false,
                safety_margin: f64::INFINITY,
            };
        }
        
        // Time to Big Rip for phantom energy
        // t_rip = -2 / (3 * (1+w) * H_0) where (1+w) is negative
        let one_plus_w = 1.0 + w; // Negative for phantom
        
        // Avoid division by zero near w = -1
        if one_plus_w.abs() < 1e-15 {
            return BigRipAnalysis {
                time_to_rip: f64::INFINITY,
                current_scale: 1.0,
                current_hubble: h_0,
                is_phantom: false, // Treat as effectively Λ
                safety_margin: f64::INFINITY,
            };
        }
        
        let time_to_rip = -2.0 / (3.0 * one_plus_w * h_0);
        
        // Apply epsilon cutoff - we can't get closer than this to singularity
        let effective_time = time_to_rip.max(self.epsilon_cutoff);
        
        BigRipAnalysis {
            time_to_rip: effective_time,
            current_scale: 1.0,
            current_hubble: h_0,
            is_phantom: true,
            safety_margin: effective_time - self.epsilon_cutoff,
        }
    }
    
    /// Calculate scale factor evolution
    /// 
    /// a(t) = a_0 * [(t_rip - t) / (t_rip - t_0)]^(-2/(3*(1+w)))
    /// 
    /// # Arguments
    /// * `time_from_now` - Time elapsed from now [s]
    /// 
    /// # Returns
    /// * `Result<f64, &'static str>` - Scale factor (relative to now)
    pub fn scale_factor_at(&self, time_from_now: f64) -> Result<f64, &'static str> {
        if time_from_now < 0.0 {
            return Err("Time must be non-negative");
        }
        
        let analysis = self.analyze();
        
        if !analysis.is_phantom {
            // Standard exponential expansion for Λ
            return Ok((self.eos.hubble_0 * time_from_now).exp());
        }
        
        let t_rip = analysis.time_to_rip;
        let w = self.eos.w;
        
        // Ensure we don't evaluate at or past the singularity
        let remaining_time = (t_rip - time_from_now).max(self.epsilon_cutoff);
        
        if remaining_time <= self.epsilon_cutoff {
            // Approaching singularity - return maximum finite value
            return Ok(f64::MAX);
        }
        
        let exponent = -2.0 / (3.0 * (1.0 + w));
        let ratio = remaining_time / t_rip;
        
        Ok(ratio.powf(exponent))
    }
    
    /// Calculate Hubble parameter evolution
    /// 
    /// H(t) = H_0 * (a(t))^(-3*(1+w)/2)
    /// 
    /// # Arguments
    /// * `time_from_now` - Time elapsed from now [s]
    /// 
    /// # Returns
    /// * `Result<f64, &'static str>` - Hubble parameter [1/s]
    pub fn hubble_at(&self, time_from_now: f64) -> Result<f64, &'static str> {
        if time_from_now < 0.0 {
            return Err("Time must be non-negative");
        }
        
        let analysis = self.analyze();
        
        if !analysis.is_phantom {
            return Ok(self.eos.hubble_0); // Constant for Λ
        }
        
        let a = self.scale_factor_at(time_from_now)?;
        let w = self.eos.w;
        
        // H ∝ a^(-3(1+w)/2)
        // For phantom w < -1, this grows with a
        let exponent = -3.0 * (1.0 + w) / 2.0;
        
        // Cap Hubble to prevent overflow
        let h = self.eos.hubble_0 * a.powf(exponent);
        Ok(h.min(1e50)) // Reasonable cap
    }
    
    /// Determine migration strategy
    /// 
    /// # Arguments
    /// * `migration_energy` - Energy required for false vacuum migration [J]
    /// * `available_power` - Available power for migration [W]
    /// 
    /// # Returns
    /// * `MigrationStrategy` - Recommended action
    pub fn migration_strategy(
        &self,
        migration_energy: f64,
        available_power: f64,
    ) -> Result<MigrationStrategy, &'static str> {
        if migration_energy <= 0.0 {
            return Err("Migration energy must be positive");
        }
        if available_power <= 0.0 {
            return Err("Available power must be positive");
        }
        
        let analysis = self.analyze();
        
        if !analysis.is_phantom {
            return Ok(MigrationStrategy {
                energy_required: migration_energy,
                time_available: f64::INFINITY,
                success_probability: 1.0,
                recommendation: MigrationAction::None,
            });
        }
        
        let time_needed = migration_energy / available_power;
        let time_available = analysis.time_to_rip - self.epsilon_cutoff;
        
        let (recommendation, success_prob) = if time_available <= 0.0 {
            (MigrationAction::Hopeless, 0.0)
        } else if time_needed > time_available {
            (MigrationAction::Hopeless, 0.0)
        } else if time_needed > time_available * 0.5 {
            (MigrationAction::MigrateNow, 0.5)
        } else if time_needed > time_available * 0.1 {
            (MigrationAction::Prepare, 0.8)
        } else {
            (MigrationAction::Prepare, 0.95)
        };
        
        Ok(MigrationStrategy {
            energy_required: migration_energy,
            time_available,
            success_probability: success_prob,
            recommendation,
        })
    }
    
    /// Calculate the "rip time" for bound structures
    /// 
    /// Structures unbind when dark energy tidal force exceeds binding
    /// 
    /// # Arguments
    /// * `binding_energy_density` - Binding energy per unit volume [J/m³]
    /// 
    /// # Returns
    /// * `Result<f64, &'static str>` - Time until structure rips [s]
    pub fn structure_rip_time(&self, binding_energy_density: f64) -> Result<f64, &'static str> {
        if binding_energy_density <= 0.0 {
            return Err("Binding energy density must be positive");
        }
        
        let analysis = self.analyze();
        
        if !analysis.is_phantom {
            return Ok(f64::INFINITY); // Never rips for Λ
        }
        
        // Simple model: structure rips when ρ_DE * c² > binding_energy_density
        // This happens at some time before t_rip
        
        let critical_rho = binding_energy_density / self.c.powi(2);
        
        // Dark energy density evolves as ρ ∝ a^(-3(1+w))
        // For phantom, this increases with time
        
        let current_rho_eff = self.eos.rho_de;
        
        if current_rho_eff >= critical_rho {
            return Ok(0.0); // Already ripped
        }
        
        // Find when ρ reaches critical
        let ratio = (critical_rho / current_rho_eff).powf(1.0 / (-3.0 * (1.0 + self.eos.w)));
        
        // Convert scale factor ratio to time
        // This is approximate
        let h_0 = self.eos.hubble_0;
        let time_approx = (ratio - 1.0) / h_0;
        
        Ok(time_approx.min(analysis.time_to_rip))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_phantom() {
        let hedger = BigRipHedger::new(-1.0, 5.9e-10, 2.2e-18).unwrap();
        let analysis = hedger.analyze();
        
        assert!(!analysis.is_phantom);
        assert!(analysis.time_to_rip.is_infinite());
    }

    #[test]
    fn test_phantom_analysis() {
        let hedger = BigRipHedger::new(-1.5, 5.9e-10, 2.2e-18).unwrap();
        let analysis = hedger.analyze();
        
        assert!(analysis.is_phantom);
        assert!(analysis.time_to_rip.is_finite());
        assert!(analysis.time_to_rip > 0.0);
    }

    #[test]
    fn test_scale_factor_growth() {
        let hedger = BigRipHedger::new(-1.5, 5.9e-10, 2.2e-18).unwrap();
        
        let a_now = hedger.scale_factor_at(0.0).unwrap();
        let a_future = hedger.scale_factor_at(1e17).unwrap();
        
        assert!((a_now - 1.0).abs() < 0.1);
        assert!(a_future > a_now);
    }

    #[test]
    fn test_migration_strategy() {
        let hedger = BigRipHedger::new(-1.5, 5.9e-10, 2.2e-18).unwrap();
        
        let strategy = hedger.migration_strategy(1e50, 1e40).unwrap();
        assert!(strategy.time_available.is_finite());
        assert!(strategy.success_probability >= 0.0);
        assert!(strategy.success_probability <= 1.0);
    }

    #[test]
    fn test_epsilon_cutoff() {
        // Test that we don't divide by zero near singularity
        let hedger = BigRipHedger::new(-2.0, 5.9e-10, 2.2e-18).unwrap();
        
        let analysis = hedger.analyze();
        assert!(analysis.safety_margin >= 0.0);
        
        // Scale factor should cap at MAX, not panic
        let a = hedger.scale_factor_at(analysis.time_to_rip).unwrap();
        assert!(a.is_finite());
    }
}
