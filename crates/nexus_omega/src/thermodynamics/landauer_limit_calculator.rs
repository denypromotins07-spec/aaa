//! Landauer Limit Calculator for thermodynamic bounds on computation.
//! Calculates minimum energy required to erase one bit of information.

use core::fmt::Debug;

/// Physical constants for Landauer calculations
pub struct PhysicalConstants;

impl PhysicalConstants {
    /// Boltzmann constant (J/K)
    pub const K_B: f64 = 1.380649e-23;
    
    /// Planck constant (J*s)
    pub const H: f64 = 6.62607015e-34;
    
    /// Speed of light (m/s)
    pub const C: f64 = 299792458.0;
    
    /// Room temperature (K) - typical operating condition
    pub const T_ROOM: f64 = 300.0;
    
    /// Cryogenic temperature (K) - for quantum computers
    pub const T_CRYO: f64 = 0.01;
}

/// Fixed-point representation for femtojoule precision
#[derive(Debug, Clone, Copy)]
pub struct FemtoJoule(u128);

impl FemtoJoule {
    /// 1 femtojoule = 1e-15 joules
    const SCALE: u128 = 1_000_000_000_000_000; // 10^15

    pub fn from_joules(joules: f64) -> Self {
        let value = (joules * Self::SCALE as f64).round() as u128;
        Self(value)
    }

    pub fn to_joules(self) -> f64 {
        self.0 as f64 / Self::SCALE as f64
    }

    pub fn from_bits(bits: u64, temperature: f64) -> Self {
        let energy = bits as f64 * PhysicalConstants::K_B * temperature * 2.0_f64.ln();
        Self::from_joules(energy)
    }

    pub fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    pub fn sub(self, other: Self) -> Option<Self> {
        if other.0 > self.0 {
            None
        } else {
            Some(Self(self.0 - other.0))
        }
    }

    pub fn mul(self, factor: u64) -> Self {
        Self(self.0 * factor as u128)
    }

    pub fn div(self, divisor: u64) -> Option<Self> {
        if divisor == 0 {
            None
        } else {
            Some(Self(self.0 / divisor as u128))
        }
    }

    pub fn compare(self, other: Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }

    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

/// Result of Landauer limit calculation
#[derive(Debug, Clone)]
pub struct LandauerResult {
    /// Minimum energy per bit erased (in joules)
    pub energy_per_bit_fj: FemtoJoule,
    /// Energy per bit in joules (for reference)
    pub energy_per_bit_j: f64,
    /// Temperature used (K)
    pub temperature: f64,
    /// Number of bits considered
    pub num_bits: u64,
    /// Total minimum energy for all bits
    pub total_energy_fj: FemtoJoule,
}

impl LandauerResult {
    /// Check if current compute efficiency approaches Landauer limit
    pub fn efficiency_ratio(&self, actual_energy_j: f64) -> f64 {
        if actual_energy_j <= 0.0 {
            return f64::MAX;
        }
        self.energy_per_bit_j / actual_energy_j
    }

    /// Get bits per joule at theoretical limit
    pub fn max_bits_per_joule(&self) -> f64 {
        if self.energy_per_bit_j > 1e-30 {
            1.0 / self.energy_per_bit_j
        } else {
            f64::MAX
        }
    }
}

/// Landauer principle calculator
pub struct LandauerCalculator {
    temperature: f64,
}

impl LandauerCalculator {
    pub const fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    pub fn room_temperature() -> Self {
        Self::new(PhysicalConstants::T_ROOM)
    }

    pub fn cryogenic() -> Self {
        Self::new(PhysicalConstants::T_CRYO)
    }

    /// Calculate Landauer limit for single bit erasure
    /// E = k_B * T * ln(2)
    pub fn calculate_single_bit(&self) -> LandauerResult {
        let energy = PhysicalConstants::K_B * self.temperature * 2.0_f64.ln();
        let energy_fj = FemtoJoule::from_joules(energy);

        LandauerResult {
            energy_per_bit_fj: energy_fj,
            energy_per_bit_j: energy,
            temperature: self.temperature,
            num_bits: 1,
            total_energy_fj: energy_fj,
        }
    }

    /// Calculate Landauer limit for multiple bit erasures
    pub fn calculate_multi_bit(&self, num_bits: u64) -> LandauerResult {
        let energy_per_bit = PhysicalConstants::K_B * self.temperature * 2.0_f64.ln();
        let total_energy = energy_per_bit * num_bits as f64;
        
        let energy_fj = FemtoJoule::from_joules(energy_per_bit);
        let total_fj = FemtoJoule::from_joules(total_energy);

        LandauerResult {
            energy_per_bit_fj: energy_fj,
            energy_per_bit_j: energy_per_bit,
            temperature: self.temperature,
            num_bits,
            total_energy_fj: total_fj,
        }
    }

    /// Calculate minimum energy to process one arbitrage trade
    /// Assumes N bits of information must be erased/processed
    pub fn calculate_trade_energy(&self, order_book_depth: usize, precision_bits: u8) -> LandauerResult {
        // Estimate bits needed: log2(order_book_depth) + precision
        let depth_bits = (order_book_depth as f64).log2().ceil() as u64;
        let total_bits = depth_bits + precision_bits as u64;
        
        self.calculate_multi_bit(total_bits)
    }

    /// Compare actual compute energy to Landauer limit
    pub fn compute_efficiency_gap(&self, actual_energy_per_op_j: f64) -> ComputeEfficiency {
        let landauer = self.calculate_single_bit();
        let ratio = actual_energy_per_op_j / landauer.energy_per_bit_j;

        ComputeEfficiency {
            landauer_limit_j: landauer.energy_per_bit_j,
            actual_energy_j: actual_energy_per_op_j,
            ratio_to_landauer: ratio,
            orders_of_magnitude: ratio.log10(),
        }
    }
}

/// Compute efficiency metrics
#[derive(Debug, Clone)]
pub struct ComputeEfficiency {
    /// Theoretical minimum (Landauer limit)
    pub landauer_limit_j: f64,
    /// Actual energy consumption
    pub actual_energy_j: f64,
    /// Ratio of actual to theoretical minimum
    pub ratio_to_landauer: f64,
    /// Orders of magnitude above Landauer limit
    pub orders_of_magnitude: f64,
}

impl ComputeEfficiency {
    /// Check if approaching thermodynamic limits (< 10 orders of magnitude)
    pub const fn approaching_limit(&self) -> bool {
        self.orders_of_magnitude < 10.0
    }

    /// Check if at Omega Point (< 3 orders of magnitude)
    pub const fn at_omega_point(&self) -> bool {
        self.orders_of_magnitude < 3.0
    }
}

/// Market efficiency tracker based on Landauer principle
pub struct MarketEfficiencyTracker {
    calculator: LandauerCalculator,
    omega_threshold: f64,
}

impl MarketEfficiencyTracker {
    pub fn new(temperature: f64, omega_threshold: f64) -> Self {
        Self {
            calculator: LandauerCalculator::new(temperature),
            omega_threshold,
        }
    }

    /// Analyze if a trading strategy has reached thermodynamic efficiency limit
    pub fn analyze_strategy_efficiency(
        &self,
        energy_per_trade_j: f64,
        bits_processed: u64,
    ) -> StrategyEfficiency {
        let theoretical_min = self.calculator.calculate_multi_bit(bits_processed);
        let ratio = energy_per_trade_j / theoretical_min.total_energy_fj.to_joules();

        let status = if ratio.log10() < self.omega_threshold {
            EfficiencyStatus::AtOmegaPoint
        } else if ratio.log10() < 10.0 {
            EfficiencyStatus::NearLimit
        } else if ratio.log10() < 20.0 {
            EfficiencyStatus::ModeratelyEfficient
        } else {
            EfficiencyStatus::Inefficient
        };

        StrategyEfficiency {
            theoretical_minimum_j: theoretical_min.total_energy_fj.to_joules(),
            actual_energy_j: energy_per_trade_j,
            efficiency_ratio: ratio,
            status,
            recommendation: self.get_recommendation(&status),
        }
    }

    fn get_recommendation(&self, status: &EfficiencyStatus) -> &'static str {
        match status {
            EfficiencyStatus::AtOmegaPoint => "ABANDON STRATEGY - No further alpha possible",
            EfficiencyStatus::NearLimit => "Reduce position - Alpha nearly exhausted",
            EfficiencyStatus::ModeratelyEfficient => "Monitor closely - Efficiency improving",
            EfficiencyStatus::Inefficient => "Room for optimization - Continue trading",
        }
    }
}

/// Strategy efficiency analysis result
#[derive(Debug, Clone)]
pub struct StrategyEfficiency {
    pub theoretical_minimum_j: f64,
    pub actual_energy_j: f64,
    pub efficiency_ratio: f64,
    pub status: EfficiencyStatus,
    pub recommendation: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EfficiencyStatus {
    AtOmegaPoint,
    NearLimit,
    ModeratelyEfficient,
    Inefficient,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_landauer_room_temperature() {
        let calc = LandauerCalculator::room_temperature();
        let result = calc.calculate_single_bit();

        // At 300K, Landauer limit ~ 2.85e-21 J
        assert!(result.energy_per_bit_j > 2e-21);
        assert!(result.energy_per_bit_j < 4e-21);
    }

    #[test]
    fn test_femtojoule_precision() {
        let energy_j = 2.85e-21;
        let fj = FemtoJoule::from_joules(energy_j);
        let recovered = fj.to_joules();

        assert!((recovered - energy_j).abs() < 1e-30);
    }

    #[test]
    fn test_compute_efficiency() {
        let calc = LandauerCalculator::room_temperature();
        
        // Modern CPU uses ~1e-9 J per operation
        let efficiency = calc.compute_efficiency_gap(1e-9);
        
        assert!(efficiency.orders_of_magnitude > 10.0);
        assert!(!efficiency.approaching_limit());
    }

    #[test]
    fn test_omega_point_detection() {
        let calc = LandauerCalculator::cryogenic();
        let limit = calc.calculate_single_bit().energy_per_bit_j;
        
        // At Omega Point, within 1000x of limit
        let efficiency = calc.compute_efficiency_gap(limit * 100.0);
        
        assert!(efficiency.at_omega_point());
    }
}
