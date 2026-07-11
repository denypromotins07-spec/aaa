//! False Vacuum Bubble Migration Engine
//! 
//! Implements the physics and economics of migrating computational substrate
//! into a localized false vacuum bubble to survive cosmological catastrophes.

use super::big_rip_phantom_hedge::{BigRipHedger, MigrationAction};

/// False vacuum parameters
#[derive(Debug, Clone, Copy)]
pub struct FalseVacuumParams {
    /// Energy density of false vacuum [J/m³]
    pub rho_false: f64,
    /// Energy density of true vacuum [J/m³]
    pub rho_true: f64,
    /// Surface tension of bubble wall [J/m²]
    pub surface_tension: f64,
    /// Critical radius for nucleation [m]
    pub critical_radius: f64,
}

impl Default for FalseVacuumParams {
    fn default() -> Self {
        // GUT-scale false vacuum parameters (approximate)
        Self {
            rho_false: 1e45, // ~GUT scale
            rho_true: 0.0,   // True vacuum at zero energy
            surface_tension: 1e30, // Domain wall tension
            critical_radius: 1e-15, // Femtometer scale
        }
    }
}

/// Bubble nucleation state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BubbleState {
    /// Not yet nucleated
    Unnucleated,
    /// Sub-critical, will collapse
    SubCritical,
    /// Critical, metastable
    Critical,
    /// Super-critical, expanding
    Expanding,
    /// Collision with other bubbles
    Collided,
}

/// A false vacuum bubble
#[derive(Debug, Clone)]
pub struct FalseVacuumBubble {
    /// Unique identifier
    pub id: u64,
    /// Current radius [m]
    pub radius: f64,
    /// Wall velocity [m/s] (fraction of c)
    pub wall_velocity: f64,
    /// Internal volume [m³]
    pub volume: f64,
    /// State
    pub state: BubbleState,
    /// Parameters
    pub params: FalseVacuumParams,
    /// Contained data mass equivalent [kg]
    pub contained_mass: f64,
}

impl FalseVacuumBubble {
    /// Create a new bubble attempt
    /// 
    /// # Arguments
    /// * `id` - Bubble identifier
    /// * `initial_radius` - Initial bubble radius [m]
    /// * `params` - Vacuum parameters
    /// 
    /// # Returns
    /// * `Result<Self, &'static str>` - Bubble or error
    pub fn new(
        id: u64,
        initial_radius: f64,
        params: FalseVacuumParams,
    ) -> Result<Self, &'static str> {
        if initial_radius <= 0.0 {
            return Err("Initial radius must be positive");
        }
        if params.rho_false <= 0.0 {
            return Err("False vacuum density must be positive");
        }
        if params.surface_tension <= 0.0 {
            return Err("Surface tension must be positive");
        }
        
        // Calculate critical radius using Coleman-de Luccia formula
        // R_c = 2σ / (ρ_false - ρ_true)
        let delta_rho = params.rho_false - params.rho_true;
        let critical_radius = 2.0 * params.surface_tension / delta_rho.max(f64::EPSILON);
        
        let mut params = params;
        params.critical_radius = critical_radius;
        
        let volume = (4.0 / 3.0) * core::f64::consts::PI * initial_radius.powi(3);
        
        let state = if initial_radius < critical_radius * 0.9 {
            BubbleState::SubCritical
        } else if (initial_radius - critical_radius).abs() < critical_radius * 0.1 {
            BubbleState::Critical
        } else {
            BubbleState::Expanding
        };
        
        Ok(Self {
            id,
            radius: initial_radius,
            wall_velocity: 0.0,
            volume,
            state,
            params,
            contained_mass: 0.0,
        })
    }
    
    /// Evolve the bubble for one timestep
    /// 
    /// # Arguments
    /// * `dt` - Timestep [s]
    /// 
    /// # Returns
    /// * `Result<(), &'static str>` - Success or error
    pub fn evolve(&mut self, dt: f64) -> Result<(), &'static str> {
        if dt <= 0.0 {
            return Err("Timestep must be positive");
        }
        
        match self.state {
            BubbleState::SubCritical => {
                // Bubble collapses
                let collapse_speed = self.params.surface_tension / 
                    (self.radius * self.params.rho_false).sqrt().max(f64::EPSILON);
                self.radius -= collapse_speed * dt;
                
                if self.radius <= 0.0 {
                    self.radius = 0.0;
                    self.state = BubbleState::Unnucleated;
                    self.volume = 0.0;
                } else {
                    self.volume = (4.0 / 3.0) * core::f64::consts::PI * self.radius.powi(3);
                }
            }
            BubbleState::Critical => {
                // Metastable - small perturbations determine fate
                // For now, stay critical
            }
            BubbleState::Expanding => {
                // Bubble expands approaching speed of light
                // Acceleration from vacuum pressure difference
                let delta_rho = self.params.rho_false - self.params.rho_true;
                let pressure = delta_rho / 3.0; // Relativistic equation of state
                
                // Wall acceleration (simplified)
                let acceleration = pressure / self.params.surface_tension;
                self.wall_velocity += acceleration * dt;
                
                // Cap at near light speed
                let c = 299_792_458.0;
                self.wall_velocity = self.wall_velocity.min(c * 0.999);
                
                self.radius += self.wall_velocity * dt;
                self.volume = (4.0 / 3.0) * core::f64::consts::PI * self.radius.powi(3);
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Check if bubble can contain given mass
    /// 
    /// # Arguments
    /// * `mass` - Mass to contain [kg]
    /// 
    /// # Returns
    /// * `bool` - True if containment possible
    pub fn can_contain(&self, mass: f64) -> bool {
        if mass <= 0.0 {
            return true;
        }
        
        // Schwarzschild limit - bubble must be larger than its Schwarzschild radius
        let g = 6.674_30e-11;
        let c = 299_792_458.0;
        let schwarzschild_r = 2.0 * g * mass / c.powi(2);
        
        self.radius > schwarzschild_r * 1.1 // 10% safety margin
    }
    
    /// Load data into the bubble
    /// 
    /// # Arguments
    /// * `mass_equivalent` - Mass equivalent of data [kg]
    /// 
    /// # Returns
    /// * `Result<(), &'static str>` - Success or error
    pub fn load_data(&mut self, mass_equivalent: f64) -> Result<(), &'static str> {
        if mass_equivalent < 0.0 {
            return Err("Mass must be non-negative");
        }
        
        if !self.can_contain(mass_equivalent) {
            return Err("Bubble too small for given mass - would collapse to black hole");
        }
        
        self.contained_mass = mass_equivalent;
        Ok(())
    }
    
    /// Calculate tunneling probability for sub-critical bubble
    /// 
    /// Uses Coleman-de Luccia instanton action
    /// 
    /// # Returns
    /// * `f64` - Tunneling probability per unit time
    pub fn tunneling_rate(&self) -> f64 {
        if self.state == BubbleState::Expanding {
            return 1.0; // Already expanded
        }
        
        // Bounce action B ≈ 27π²σ⁴ / (2(Δρ)³)
        let delta_rho = (self.params.rho_false - self.params.rho_true).max(f64::EPSILON);
        let sigma = self.params.surface_tension;
        
        let bounce_action = 27.0 * core::f64::consts::PI.powi(2) * sigma.powi(4) 
            / (2.0 * delta_rho.powi(3));
        
        // Tunneling rate Γ ∝ exp(-B/ℏ)
        let hbar = 1.054_571_817e-34;
        let exponent = -bounce_action / hbar;
        
        if exponent < -745.0 {
            0.0 // Effectively zero
        } else {
            exponent.exp()
        }
    }
    
    /// Get proper time inside bubble vs outside
    /// 
    /// Due to domain wall tension, there's gravitational time dilation
    /// 
    /// # Returns
    /// * `f64` - Time dilation factor (inside/outside)
    pub fn time_dilation_factor(&self) -> f64 {
        // Simplified model based on domain wall gravity
        let g = 6.674_30e-11;
        let c = 299_792_458.0;
        
        // Effective mass of bubble wall
        let wall_mass = 4.0 * core::f64::consts::PI * self.radius.powi(2) 
            * self.params.surface_tension / c.powi(2);
        
        // Gravitational redshift factor
        let schwarzschild_factor = 1.0 - 2.0 * g * wall_mass / (self.radius * c.powi(2));
        
        schwarzschild_factor.sqrt().max(0.0).min(1.0)
    }
}

/// False vacuum migration manager
#[derive(Debug, Clone)]
pub struct FalseVacuumManager {
    /// Active bubbles
    bubbles: Vec<FalseVacuumBubble>,
    /// Bubble counter
    bubble_counter: u64,
    /// Default parameters
    default_params: FalseVacuumParams,
}

impl Default for FalseVacuumManager {
    fn default() -> Self {
        Self {
            bubbles: Vec::new(),
            bubble_counter: 0,
            default_params: FalseVacuumParams::default(),
        }
    }
}

impl FalseVacuumManager {
    /// Attempt to nucleate a new bubble
    /// 
    /// # Arguments
    /// * `initial_radius` - Initial bubble radius [m]
    /// 
    /// # Returns
    /// * `Result<u64, &'static str>` - Bubble ID
    pub fn nucleate_bubble(&mut self, initial_radius: f64) -> Result<u64, &'static str> {
        let bubble = FalseVacuumBubble::new(
            self.bubble_counter,
            initial_radius,
            self.default_params,
        )?;
        
        let id = bubble.id;
        self.bubbles.push(bubble);
        self.bubble_counter += 1;
        
        Ok(id)
    }
    
    /// Evolve all bubbles
    /// 
    /// # Arguments
    /// * `dt` - Timestep [s]
    pub fn evolve_all(&mut self, dt: f64) {
        for bubble in &mut self.bubbles {
            let _ = bubble.evolve(dt);
        }
    }
    
    /// Get bubble by ID
    pub fn get_bubble(&self, id: u64) -> Option<&FalseVacuumBubble> {
        self.bubbles.iter().find(|b| b.id == id)
    }
    
    /// Get mutable bubble by ID
    pub fn get_bubble_mut(&mut self, id: u64) -> Option<&mut FalseVacuumBubble> {
        self.bubbles.iter_mut().find(|b| b.id == id)
    }
    
    /// Remove collapsed bubbles
    pub fn cleanup_collapsed(&mut self) -> usize {
        let before = self.bubbles.len();
        self.bubbles.retain(|b| b.state != BubbleState::Unnucleated);
        before - self.bubbles.len()
    }
    
    /// Get statistics
    pub fn get_statistics(&self) -> VacuumStats {
        let expanding = self.bubbles.iter()
            .filter(|b| b.state == BubbleState::Expanding)
            .count();
        let critical = self.bubbles.iter()
            .filter(|b| b.state == BubbleState::Critical)
            .count();
        let subcritical = self.bubbles.iter()
            .filter(|b| b.state == BubbleState::SubCritical)
            .count();
        
        let total_volume: f64 = self.bubbles.iter()
            .map(|b| b.volume)
            .sum();
        
        VacuumStats {
            total_bubbles: self.bubbles.len(),
            expanding,
            critical,
            subcritical,
            total_volume,
        }
    }
}

/// Vacuum statistics
#[derive(Debug, Clone, Copy)]
pub struct VacuumStats {
    /// Total active bubbles
    pub total_bubbles: usize,
    /// Expanding bubbles
    pub expanding: usize,
    /// Critical bubbles
    pub critical: usize,
    /// Sub-critical bubbles
    pub subcritical: usize,
    /// Total internal volume [m³]
    pub total_volume: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bubble_creation() {
        let params = FalseVacuumParams::default();
        let bubble = FalseVacuumBubble::new(0, 1e-14, params);
        
        assert!(bubble.is_ok());
        let b = bubble.unwrap();
        assert!(b.radius > 0.0);
    }

    #[test]
    fn test_bubble_evolution() {
        let params = FalseVacuumParams::default();
        let mut bubble = FalseVacuumBubble::new(0, 1e-14, params).unwrap();
        
        let initial_radius = bubble.radius;
        bubble.evolve(1e-30).unwrap();
        
        // Should have evolved
        assert!(bubble.radius != initial_radius || bubble.state == BubbleState::Critical);
    }

    #[test]
    fn test_tunneling_rate() {
        let params = FalseVacuumParams::default();
        let bubble = FalseVacuumBubble::new(0, 1e-16, params).unwrap();
        
        let rate = bubble.tunneling_rate();
        assert!(rate >= 0.0);
        assert!(rate <= 1.0);
    }

    #[test]
    fn test_manager() {
        let mut manager = FalseVacuumManager::default();
        
        let id = manager.nucleate_bubble(1e-14).unwrap();
        assert_eq!(id, 0);
        
        let stats = manager.get_statistics();
        assert_eq!(stats.total_bubbles, 1);
    }
}
