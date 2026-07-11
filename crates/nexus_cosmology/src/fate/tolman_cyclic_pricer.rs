//! Tolman Cyclic Big Bounce Pricer
//! 
//! Implements pricing for "Epoch Futures" that hedge against entropy
//! accumulation across cyclic universe bounces.

use core::f64;

/// Parameters for a single cosmological cycle
#[derive(Debug, Clone, Copy)]
pub struct CycleParameters {
    /// Maximum scale factor of this cycle [dimensionless]
    pub max_scale: f64,
    /// Minimum scale factor (bounce point) [dimensionless]
    pub min_scale: f64,
    /// Duration of cycle [s]
    pub duration: f64,
    /// Entropy at start of cycle [J/K]
    pub initial_entropy: f64,
    /// Entropy production during cycle [J/K]
    pub entropy_production: f64,
}

impl Default for CycleParameters {
    fn default() -> Self {
        Self {
            max_scale: 1e30, // Current observable universe scale
            min_scale: 1e-35, // Planck scale bounce
            duration: 1e18, // ~current age of universe
            initial_entropy: 1e104, // Current universe entropy
            entropy_production: 1e90, // Entropy produced per cycle
        }
    }
}

/// State of the cyclic universe
#[derive(Debug, Clone, Copy)]
pub enum UniversePhase {
    /// Expanding from bounce
    Expansion,
    /// At maximum expansion
    Turnaround,
    /// Contracting toward bounce
    Contraction,
    /// At minimum (bounce point)
    Bounce,
}

/// An Epoch Future contract
#[derive(Debug, Clone)]
pub struct EpochFuture {
    /// Unique contract ID
    pub id: u64,
    /// Target epoch (cycle number)
    pub target_cycle: u64,
    /// Target time within cycle [s]
    pub target_time: f64,
    /// Strike entropy [J/K]
    pub strike_entropy: f64,
    /// Premium paid [in entropy units]
    pub premium: f64,
    /// Contract status
    pub status: ContractStatus,
    /// Payout multiplier
    pub payout_multiplier: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContractStatus {
    /// Active, waiting for epoch
    Active,
    /// Exercised successfully
    Exercised,
    /// Expired worthless
    Expired,
    /// Defaulted (entropy too high)
    Defaulted,
}

/// Tolman cyclic universe pricer
#[derive(Debug, Clone)]
pub struct TolmanPricer {
    /// Current cycle parameters
    current_cycle: CycleParameters,
    /// Number of previous cycles
    cycles_elapsed: u64,
    /// Cumulative entropy increase
    cumulative_entropy: f64,
    /// Contract counter
    contract_counter: u64,
}

impl Default for TolmanPricer {
    fn default() -> Self {
        Self {
            current_cycle: CycleParameters::default(),
            cycles_elapsed: 0,
            cumulative_entropy: 1e104,
            contract_counter: 0,
        }
    }
}

impl TolmanPricer {
    /// Create a new pricer with custom initial conditions
    /// 
    /// # Arguments
    /// * `initial_entropy` - Starting entropy [J/K]
    /// * `cycles_elapsed` - Number of previous bounces
    /// 
    /// # Returns
    /// * `Result<Self, &'static str>` - Pricer or error
    pub fn new(initial_entropy: f64, cycles_elapsed: u64) -> Result<Self, &'static str> {
        if initial_entropy <= 0.0 {
            return Err("Initial entropy must be positive");
        }
        
        Ok(Self {
            current_cycle: CycleParameters::default(),
            cycles_elapsed,
            cumulative_entropy: initial_entropy,
            contract_counter: 0,
        })
    }
    
    /// Calculate entropy at the next bounce
    /// 
    /// Using Tolman's formula: S_{n+1} = S_n + ΔS
    /// where ΔS is entropy production per cycle
    /// 
    /// # Returns
    /// * `f64` - Entropy at next bounce [J/K]
    pub fn entropy_at_next_bounce(&self) -> f64 {
        self.cumulative_entropy + self.current_cycle.entropy_production
    }
    
    /// Calculate maximum scale factor for next cycle
    /// 
    /// In Tolman's model, each cycle reaches a larger maximum size
    /// due to increased entropy: a_max ∝ S^(2/3) for radiation-dominated
    /// 
    /// # Returns
    /// * `f64` - Maximum scale factor
    pub fn next_cycle_max_scale(&self) -> f64 {
        let next_entropy = self.entropy_at_next_bounce();
        let current_entropy = self.cumulative_entropy;
        
        // a_max ∝ S^(2/3)
        let ratio = (next_entropy / current_entropy.max(f64::EPSILON)).powf(2.0 / 3.0);
        
        self.current_cycle.max_scale * ratio
    }
    
    /// Calculate duration of next cycle
    /// 
    /// Higher entropy cycles last longer
    /// 
    /// # Returns
    /// * `f64` - Cycle duration [s]
    pub fn next_cycle_duration(&self) -> f64 {
        let next_entropy = self.entropy_at_next_bounce();
        let current_entropy = self.cumulative_entropy;
        
        // Duration scales roughly as entropy^(1/2) for matter-dominated
        let ratio = (next_entropy / current_entropy.max(f64::EPSILON)).powf(0.5);
        
        self.current_cycle.duration * ratio
    }
    
    /// Price an Epoch Future contract
    /// 
    /// The price reflects the probability that entropy will be below
    /// the strike at the target epoch
    /// 
    /// # Arguments
    /// * `target_cycle` - Which future cycle
    /// * `strike_entropy` - Strike entropy level
    /// 
    /// # Returns
    /// * `Result<f64, &'static str>` - Premium in entropy units
    pub fn price_epoch_future(
        &self,
        target_cycle: u64,
        strike_entropy: f64,
    ) -> Result<f64, &'static str> {
        if strike_entropy <= 0.0 {
            return Err("Strike entropy must be positive");
        }
        if target_cycle == 0 {
            return Err("Target cycle must be >= 1");
        }
        
        // Project entropy to target cycle
        let projected_entropy = self.cumulative_entropy 
            + (target_cycle as f64) * self.current_cycle.entropy_production;
        
        // Probability that actual entropy < strike
        // Modeled as log-normal distribution around projection
        let volatility = 0.1; // 10% uncertainty per cycle
        
        // Black-Scholes-like formula for entropy options
        let d1 = (projected_entropy / strike_entropy).ln() 
            + 0.5 * volatility.powi(2) * (target_cycle as f64);
        let d2 = d1 - volatility * (target_cycle as f64).sqrt();
        
        // Cumulative normal approximation
        let n_d1 = cumulative_normal(d1);
        let n_d2 = cumulative_normal(d2);
        
        // Premium = strike * N(d2) - projected * N(d1)
        let premium = strike_entropy * n_d2 - projected_entropy * n_d1;
        
        Ok(premium.max(0.0))
    }
    
    /// Issue a new Epoch Future contract
    /// 
    /// # Arguments
    /// * `target_cycle` - Target cycle number
    /// * `target_time` - Time within target cycle
    /// * `strike_entropy` - Strike entropy
    /// 
    /// # Returns
    /// * `Result<u64, &'static str>` - Contract ID
    pub fn issue_contract(
        &mut self,
        target_cycle: u64,
        target_time: f64,
        strike_entropy: f64,
    ) -> Result<u64, &'static str> {
        let premium = self.price_epoch_future(target_cycle, strike_entropy)?;
        
        let contract = EpochFuture {
            id: self.contract_counter,
            target_cycle,
            target_time,
            strike_entropy,
            premium,
            status: ContractStatus::Active,
            payout_multiplier: 1.0,
        };
        
        let id = contract.id;
        self.contract_counter += 1;
        
        Ok(id)
    }
    
    /// Simulate advancing one complete cycle
    /// 
    /// Updates entropy and cycle parameters
    pub fn advance_cycle(&mut self) {
        self.cycles_elapsed += 1;
        self.cumulative_entropy += self.current_cycle.entropy_production;
        
        // Update cycle parameters for new cycle
        self.current_cycle.initial_entropy = self.cumulative_entropy;
        self.current_cycle.max_scale = self.next_cycle_max_scale();
        self.current_cycle.duration = self.next_cycle_duration();
    }
    
    /// Get the "Tolman limit" - when cycles become infinitely long
    /// 
    /// This occurs when entropy production causes turnaround before
    /// reaching a stable configuration
    /// 
    /// # Returns
    /// * `u64` - Estimated cycles until heat death
    pub fn cycles_until_heat_death(&self) -> u64 {
        // Simplified: when entropy exceeds critical value
        let critical_entropy = 1e150; // Arbitrary large value
        
        let remaining = (critical_entropy - self.cumulative_entropy)
            .max(0.0) / self.current_cycle.entropy_production.max(f64::EPSILON);
        
        remaining as u64
    }
    
    /// Calculate present value of capital preservation across bounce
    /// 
    /// # Arguments
    /// * `capital_amount` - Amount to preserve [arbitrary units]
    /// * `cycles` - Number of cycles to preserve through
    /// 
    /// # Returns
    /// * `f64` - Present value discount factor
    pub fn capital_preservation_factor(&self, capital_amount: f64, cycles: u64) -> f64 {
        if capital_amount <= 0.0 {
            return 0.0;
        }
        
        // Each cycle has some probability of information loss
        let survival_per_cycle = 0.999; // 99.9% survival rate
        
        // Total survival probability
        let total_survival = survival_per_cycle.powi(cycles as i32);
        
        // Discount factor
        total_survival
    }
}

/// Cumulative normal distribution function (approximation)
fn cumulative_normal(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989423 * (-x * x / 2.0).exp();
    let prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    
    if x > 0.0 {
        1.0 - prob
    } else {
        prob
    }
}

/// Statistics about issued contracts
#[derive(Debug, Clone, Copy)]
pub struct ContractStats {
    /// Total active contracts
    pub active: u64,
    /// Total exercised
    pub exercised: u64,
    /// Total expired
    pub expired: u64,
    /// Total defaulted
    pub defaulted: u64,
    /// Total premium collected
    pub total_premium: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pricer_creation() {
        let pricer = TolmanPricer::new(1e104, 0);
        assert!(pricer.is_ok());
    }

    #[test]
    fn test_entropy_projection() {
        let pricer = TolmanPricer::default();
        
        let next = pricer.entropy_at_next_bounce();
        assert!(next > pricer.cumulative_entropy);
    }

    #[test]
    fn test_max_scale_growth() {
        let pricer = TolmanPricer::default();
        
        let next_scale = pricer.next_cycle_max_scale();
        assert!(next_scale > pricer.current_cycle.max_scale);
    }

    #[test]
    fn test_pricing() {
        let pricer = TolmanPricer::default();
        
        let premium = pricer.price_epoch_future(1, 1e105);
        assert!(premium.is_ok());
        assert!(premium.unwrap() >= 0.0);
    }

    #[test]
    fn test_cycle_advancement() {
        let mut pricer = TolmanPricer::default();
        
        let initial_entropy = pricer.cumulative_entropy;
        pricer.advance_cycle();
        
        assert!(pricer.cumulative_entropy > initial_entropy);
        assert_eq!(pricer.cycles_elapsed, 1);
    }

    #[test]
    fn test_capital_preservation() {
        let pricer = TolmanPricer::default();
        
        let factor = pricer.capital_preservation_factor(1e20, 10);
        assert!(factor > 0.0);
        assert!(factor <= 1.0);
    }
}
