// NEXUS-OMEGA Stage 34: Currency Peg Fluid Dynamics
// Chapter 3: Central Bank Reserve Dam Model
// File: crates/nexus_macro_physics/src/fluid/central_bank_reserve_dam.rs

//! Central Bank Reserve Dam Model
//!
//! Models a central bank's FX reserves as a dam holding back speculative pressure.
//! When hydrodynamic pressure exceeds the dam's structural integrity, the peg breaks.

#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]

use core::fmt;
use alloc::vec::Vec;

use super::lattice_boltzmann_fx::{LatticeBoltzmannSolver, LBMError};

/// Error types for reserve dam operations
#[derive(Debug, Clone, PartialEq)]
pub enum ReserveDamError {
    InsufficientReserves { current: f64, required: f64 },
    DamFailure { pressure: f64, capacity: f64 },
    LBMError(LBMError),
    InvalidParameter(String),
}

impl fmt::Display for ReserveDamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientReserves { current, required } => {
                write!(f, "Insufficient reserves: {} < {}", current, required)
            }
            Self::DamFailure { pressure, capacity } => {
                write!(f, "Dam failure: pressure {} exceeds capacity {}", pressure, capacity)
            }
            Self::LBMError(e) => write!(f, "LBM error: {}", e),
            Self::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ReserveDamError {}

/// State of the central bank reserve dam
#[derive(Debug, Clone)]
pub struct ReserveDamState {
    /// Current FX reserves (in billions USD equivalent)
    pub reserves: f64,
    /// Maximum reserve capacity (theoretical limit)
    pub max_capacity: f64,
    /// Current pressure from speculative attacks
    pub pressure: f64,
    /// Dam integrity (0 = broken, 1 = intact)
    pub integrity: f64,
    /// Peg exchange rate (domestic per USD)
    pub peg_rate: f64,
    /// Market exchange rate (if floating)
    pub market_rate: f64,
    /// Interest rate differential (domestic - foreign)
    pub rate_differential: f64,
}

impl ReserveDamState {
    #[must_use]
    pub fn new(
        reserves: f64,
        max_capacity: f64,
        peg_rate: f64,
    ) -> Self {
        Self {
            reserves: reserves.max(0.0),
            max_capacity: max_capacity.max(reserves),
            pressure: 0.0,
            integrity: 1.0,
            peg_rate: peg_rate.max(0.0),
            market_rate: peg_rate,
            rate_differential: 0.0,
        }
    }

    /// Check if the peg is still defended
    #[must_use]
    pub fn is_peg_defended(&self) -> bool {
        self.integrity > 0.5 && self.reserves > self.max_capacity * 0.1
    }

    /// Compute pressure head (analogous to water height behind dam)
    #[must_use]
    pub fn pressure_head(&self) -> f64 {
        // Pressure proportional to deviation from peg
        let deviation = (self.market_rate - self.peg_rate).abs() / self.peg_rate;
        deviation * self.reserves
    }
}

/// Central Bank Reserve Dam Simulator
pub struct CentralBankReserveDam {
    /// Current state
    state: ReserveDamState,
    /// LBM solver for fluid dynamics simulation
    lbm_solver: Option<LatticeBoltzmannSolver>,
    /// Minimum reserve threshold before intervention
    min_reserve_threshold: f64,
    /// Speculative attack intensity
    attack_intensity: f64,
}

impl CentralBankReserveDam {
    /// Create a new reserve dam simulator
    #[must_use]
    pub fn new(
        initial_reserves: f64,
        max_capacity: f64,
        peg_rate: f64,
    ) -> Self {
        Self {
            state: ReserveDamState::new(initial_reserves, max_capacity, peg_rate),
            lbm_solver: None,
            min_reserve_threshold: max_capacity * 0.2,
            attack_intensity: 0.0,
        }
    }

    /// Set minimum reserve threshold
    pub fn with_min_reserve_threshold(mut self, threshold: f64) -> Self {
        self.min_reserve_threshold = threshold;
        self
    }

    /// Initialize LBM solver for detailed pressure simulation
    pub fn initialize_lbm(&mut self, grid_size: usize) -> Result<(), ReserveDamError> {
        let solver = LatticeBoltzmannSolver::new(grid_size, grid_size, 1.0)
            .map_err(ReserveDamError::LBMError)?;
        self.lbm_solver = Some(solver);
        Ok(())
    }

    /// Apply speculative attack pressure
    ///
    /// # Arguments
    /// * `attack_size` - Size of speculative attack (in billions)
    /// * `duration` - Expected duration of attack (time steps)
    pub fn apply_speculative_attack(&mut self, attack_size: f64, duration: u32) {
        self.attack_intensity = attack_size / duration as f64;
        
        // Increase pressure based on attack
        self.state.pressure += attack_size * 0.1;
        
        // Decrease reserves as CB defends peg
        let reserve_depletion = attack_size * 0.5;
        self.state.reserves = (self.state.reserves - reserve_depletion).max(0.0);
    }

    /// Simulate one time step of reserve dynamics
    ///
    /// # Returns
    /// * `Ok(())` if peg still holds
    /// * `Err(ReserveDamError)` if peg breaks
    pub fn step(&mut self) -> Result<(), ReserveDamError> {
        // Update pressure from LBM if available
        if let Some(ref mut solver) = self.lbm_solver {
            let _ = solver.step();
            
            // Get average velocity as proxy for capital flow
            let (avg_vx, avg_vy) = solver.average_velocity(0, 0, solver.dimensions().0, solver.dimensions().1);
            let flow_magnitude = (avg_vx * avg_vx + avg_vy * avg_vy).sqrt();
            
            self.state.pressure = flow_magnitude * self.state.reserves;
        }

        // Update integrity based on pressure vs reserves
        let capacity_ratio = self.state.reserves / self.state.max_capacity;
        let pressure_ratio = self.state.pressure / (self.state.reserves + 1.0);
        
        // Integrity decreases when pressure exceeds capacity
        if pressure_ratio > capacity_ratio {
            self.state.integrity -= 0.01 * (pressure_ratio - capacity_ratio);
            self.state.integrity = self.state.integrity.clamp(0.0, 1.0);
        } else {
            // Slow recovery when pressure is low
            self.state.integrity = (self.state.integrity + 0.001).min(1.0);
        }

        // Update market rate based on pressure
        let pressure_deviation = self.state.pressure / (self.state.reserves + 1.0);
        self.state.market_rate = self.state.peg_rate * (1.0 + pressure_deviation);

        // Check for dam failure
        if self.state.integrity <= 0.0 || self.state.reserves <= self.min_reserve_threshold {
            return Err(ReserveDamError::DamFailure {
                pressure: self.state.pressure,
                capacity: self.state.reserves,
            });
        }

        // Decay attack intensity
        self.attack_intensity *= 0.9;
        self.state.pressure *= 0.95;

        Ok(())
    }

    /// Get probability of peg break within N time steps
    #[must_use]
    pub fn break_probability(&self, horizon: u32) -> f64 {
        if !self.state.is_peg_defended() {
            return 1.0;
        }

        let reserve_buffer = self.state.reserves - self.min_reserve_threshold;
        if reserve_buffer <= 0.0 {
            return 1.0;
        }

        // Simplified probability model
        let pressure_rate = self.attack_intensity / (self.state.reserves + 1.0);
        let expected_depletion = pressure_rate * horizon as f64;
        
        let probability = (expected_depletion / reserve_buffer).clamp(0.0, 1.0);
        probability * (1.0 - self.state.integrity * 0.5)
    }

    /// Get current state
    #[must_use]
    pub const fn state(&self) -> &ReserveDamState {
        &self.state
    }

    /// Get attack intensity
    #[must_use]
    pub const fn attack_intensity(&self) -> f64 {
        self.attack_intensity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reserve_dam_creation() {
        let dam = CentralBankReserveDam::new(100.0, 200.0, 7.8);
        assert!(dam.state().is_peg_defended());
        assert_eq!(dam.state().reserves, 100.0);
    }

    #[test]
    fn test_speculative_attack() {
        let mut dam = CentralBankReserveDam::new(100.0, 200.0, 7.8);
        
        dam.apply_speculative_attack(50.0, 10);
        
        assert!(dam.state().reserves < 100.0);
        assert!(dam.state().pressure > 0.0);
    }

    #[test]
    fn test_break_probability() {
        let dam = CentralBankReserveDam::new(100.0, 200.0, 7.8);
        let prob = dam.break_probability(10);
        
        assert!(prob >= 0.0 && prob <= 1.0);
    }
}
