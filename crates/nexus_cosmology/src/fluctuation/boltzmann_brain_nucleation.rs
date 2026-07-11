//! Boltzmann Brain Nucleation Calculator
//! 
//! Implements log-probability arithmetic to handle the infinitesimally small
//! probabilities of spontaneous macroscopic fluctuation events without underflow.
//! 
//! CRITICAL: Standard f64 would underflow to 0.0 for probabilities like 10^(-10^50).
//! This module stores probabilities in log-space (natural log) and performs all
//! operations using log-arithmetic to prevent underflow.

use core::f64;

/// Log-probability representation to prevent underflow
/// 
/// Stores ln(P) instead of P directly. For P = 10^(-10^50), 
/// ln(P) ≈ -10^50 * ln(10) ≈ -2.3 * 10^50, which can be represented
/// as an f64 exponent with appropriate scaling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogProb {
    /// Natural logarithm of probability: ln(P)
    /// For very small probabilities, this is a large negative number
    pub ln_p: f64,
}

impl LogProb {
    /// Probability of 1 (ln(1) = 0)
    pub const ONE: LogProb = LogProb { ln_p: 0.0 };
    
    /// Probability of 0 (ln(0) = -∞)
    pub const ZERO: LogProb = LogProb { ln_p: f64::NEG_INFINITY };
    
    /// Create a new LogProb from a direct probability
    /// 
    /// # Arguments
    /// * `p` - Probability value [0, 1]
    /// 
    /// # Returns
    /// * `Result<Self, &'static str>` - LogProb or error
    pub fn from_probability(p: f64) -> Result<Self, &'static str> {
        if p < 0.0 || p > 1.0 {
            return Err("Probability must be in [0, 1]");
        }
        if p == 0.0 {
            return Ok(Self::ZERO);
        }
        if p == 1.0 {
            return Ok(Self::ONE);
        }
        Ok(Self { ln_p: p.ln() })
    }
    
    /// Create a LogProb from base-10 exponent: P = 10^exp
    /// 
    /// This is useful for cosmological probabilities like 10^(-10^50)
    /// where we store exp = -10^50 directly
    /// 
    /// # Arguments
    /// * `exp10` - Exponent such that P = 10^exp10
    /// 
    /// # Returns
    /// * `Self` - LogProb with ln(P) = exp10 * ln(10)
    pub fn from_base10_exponent(exp10: f64) -> Self {
        Self {
            ln_p: exp10 * core::f64::consts::LN_10,
        }
    }
    
    /// Create a LogProb from a power tower: P = 10^(-10^n)
    /// 
    /// For extreme cosmological scales
    /// 
    /// # Arguments
    /// * `n` - The power in 10^(-10^n)
    /// 
    /// # Returns
    /// * `Self` - LogProb
    pub fn from_power_tower(n: f64) -> Self {
        // ln(P) = ln(10^(-10^n)) = -10^n * ln(10)
        let exponent = -(10.0_f64).powf(n);
        Self {
            ln_p: exponent * core::f64::consts::LN_10,
        }
    }
    
    /// Convert back to probability (may underflow to 0 for very small values)
    /// 
    /// # Returns
    /// * `f64` - Probability value (0.0 if too small)
    pub fn to_probability(&self) -> f64 {
        if self.ln_p < -745.0 {
            // Below f64 minimum before underflow
            0.0
        } else {
            self.ln_p.exp()
        }
    }
    
    /// Get the base-10 exponent: log10(P)
    /// 
    /// # Returns
    /// * `f64` - log10(P)
    pub fn log10(&self) -> f64 {
        self.ln_p / core::f64::consts::LN_10
    }
    
    /// Multiply two probabilities: ln(P1 * P2) = ln(P1) + ln(P2)
    /// 
    /// # Arguments
    /// * `other` - Other LogProb
    /// 
    /// # Returns
    /// * `Self` - Product
    pub fn multiply(&self, other: &LogProb) -> Self {
        Self {
            ln_p: self.ln_p + other.ln_p,
        }
    }
    
    /// Divide two probabilities: ln(P1 / P2) = ln(P1) - ln(P2)
    /// 
    /// # Arguments
    /// * `other` - Other LogProb
    /// 
    /// # Returns
    /// * `Self` - Quotient
    pub fn divide(&self, other: &LogProb) -> Self {
        Self {
            ln_p: self.ln_p - other.ln_p,
        }
    }
    
    /// Add two probabilities using log-sum-exp trick:
    /// ln(P1 + P2) = ln(P1) + ln(1 + exp(ln(P2) - ln(P1)))
    /// 
    /// Assumes ln(P1) >= ln(P2) for numerical stability
    /// 
    /// # Arguments
    /// * `other` - Other LogProb
    /// 
    /// # Returns
    /// * `Self` - Sum
    pub fn add(&self, other: &LogProb) -> Self {
        // Handle special cases
        if self.ln_p.is_infinite() && self.ln_p < 0.0 {
            return *other;
        }
        if other.ln_p.is_infinite() && other.ln_p < 0.0 {
            return *self;
        }
        
        // Ensure self has larger (less negative) ln_p
        let (max, min) = if self.ln_p >= other.ln_p {
            (self.ln_p, other.ln_p)
        } else {
            (other.ln_p, self.ln_p)
        };
        
        // log-sum-exp: max + ln(1 + exp(min - max))
        let diff = min - max;
        if diff < -745.0 {
            // exp(diff) underflows, result is just max
            Self { ln_p: max }
        } else {
            Self {
                ln_p: max + (1.0 + diff.exp()).ln(),
            }
        }
    }
    
    /// Raise to a power: ln(P^n) = n * ln(P)
    /// 
    /// # Arguments
    /// * `n` - Exponent
    /// 
    /// # Returns
    /// * `Self` - P^n
    pub fn pow(&self, n: f64) -> Self {
        Self {
            ln_p: self.ln_p * n,
        }
    }
    
    /// Check if this probability is effectively zero
    pub fn is_effectively_zero(&self) -> bool {
        self.ln_p < -1e10 // Threshold for "practically impossible"
    }
    
    /// Compare if self is more probable than other
    pub fn more_probable_than(&self, other: &LogProb) -> bool {
        self.ln_p > other.ln_p
    }
}

/// Boltzmann brain nucleation parameters
#[derive(Debug, Clone, Copy)]
pub struct BoltzmannParams {
    /// Target entropy decrease ΔS [J/K]
    pub delta_s: f64,
    /// Background temperature [K]
    pub temperature: f64,
    /// Fluctuation volume [m³]
    pub volume: f64,
    /// Fluctuation timescale [s]
    pub timescale: f64,
}

impl Default for BoltzmannParams {
    fn default() -> Self {
        // Parameters for a minimal conscious observer (~human brain)
        Self {
            delta_s: 1e-2, // Minimal entropy decrease for neural state
            temperature: 1e-30, // Heat death temperature
            volume: 1e-3, // ~1 liter
            timescale: 1.0, // 1 second coherence time
        }
    }
}

/// Boltzmann brain nucleation rate calculator
#[derive(Debug, Clone)]
pub struct BoltzmannNucleationCalculator {
    /// Boltzmann constant
    k_b: f64,
    /// Planck constant
    hbar: f64,
    /// Speed of light
    c: f64,
}

impl Default for BoltzmannNucleationCalculator {
    fn default() -> Self {
        Self {
            k_b: 1.380_649e-23,
            hbar: 1.054_571_817e-34,
            c: 299_792_458.0,
        }
    }
}

impl BoltzmannNucleationCalculator {
    /// Calculate nucleation rate using fluctuation theorem
    /// 
    /// Γ = Γ_0 * exp(-ΔS / k_B)
    /// 
    /// where Γ_0 is the attempt frequency based on quantum uncertainty
    /// 
    /// # Arguments
    /// * `params` - Nucleation parameters
    /// 
    /// # Returns
    /// * `LogProb` - Log-probability of nucleation per unit spacetime volume
    pub fn calculate_nucleation_rate(&self, params: &BoltzmannParams) -> LogProb {
        if params.delta_s <= 0.0 {
            // Entropy increase is favored, probability ~1
            return LogProb::ONE;
        }
        
        if params.temperature <= 0.0 {
            // At absolute zero, no thermal fluctuations
            return LogProb::ZERO;
        }
        
        // Fluctuation theorem: P ∝ exp(-ΔS / k_B)
        let exponent = -params.delta_s / self.k_b;
        
        // For macroscopic fluctuations, this is extremely small
        // We return it as LogProb to avoid underflow
        LogProb { ln_p: exponent }
    }
    
    /// Calculate nucleation probability for a specific brain configuration
    /// 
    /// Uses the Einstein formula: P ∝ exp(-E_fluctuation / kT)
    /// where E_fluctuation is the energy needed to assemble the brain state
    /// 
    /// # Arguments
    /// * `brain_mass` - Mass of brain [kg]
    /// * `temperature` - Background temperature [K]
    /// * `coherence_time` - Required coherence time [s]
    /// 
    /// # Returns
    /// * `LogProb` - Probability of spontaneous nucleation
    pub fn brain_nucleation_probability(
        &self,
        brain_mass: f64,
        temperature: f64,
        coherence_time: f64,
    ) -> Result<LogProb, &'static str> {
        if brain_mass <= 0.0 {
            return Err("Brain mass must be positive");
        }
        if temperature <= 0.0 {
            return Ok(LogProb::ZERO);
        }
        if coherence_time <= 0.0 {
            return Err("Coherence time must be positive");
        }
        
        // Energy required: E = mc²
        let energy = brain_mass * self.c.powi(2);
        
        // Entropy decrease: ΔS = E / T (minimum)
        let delta_s = energy / temperature;
        
        // Probability: P ∝ exp(-ΔS / k_B) = exp(-E / kT)
        let exponent = -energy / (self.k_b * temperature);
        
        Ok(LogProb { ln_p: exponent })
    }
    
    /// Calculate expected waiting time for a Boltzmann brain to appear
    /// in a given comoving volume
    /// 
    /// # Arguments
    /// * `volume` - Comoving volume [m³]
    /// * `params` - Nucleation parameters
    /// 
    /// # Returns
    /// * `Result<f64, &'static str>` - Expected waiting time [s]
    pub fn expected_waiting_time(
        &self,
        volume: f64,
        params: &BoltzmannParams,
    ) -> Result<f64, &'static str> {
        if volume <= 0.0 {
            return Err("Volume must be positive");
        }
        
        let rate_prob = self.calculate_nucleation_rate(params);
        
        // Attempt frequency from quantum uncertainty:
        // f_0 ~ kT / ℏ (thermal) or c / λ_C (quantum)
        let attempt_freq = self.k_b * params.temperature / self.hbar;
        
        // Total rate = attempt_freq * volume * P
        // But P is in log-space, so we need to be careful
        
        if rate_prob.is_effectively_zero() {
            // Waiting time exceeds age of universe by many orders
            return Ok(f64::INFINITY);
        }
        
        let probability = rate_prob.to_probability();
        if probability == 0.0 {
            return Ok(f64::INFINITY);
        }
        
        let total_rate = attempt_freq * volume * probability;
        
        if total_rate <= 0.0 {
            return Ok(f64::INFINITY);
        }
        
        // Expected waiting time = 1 / rate
        Ok(1.0 / total_rate)
    }
    
    /// Compare nucleation rates for different brain configurations
    /// 
    /// # Arguments
    /// * `mass1` - Mass of first configuration
    /// * `mass2` - Mass of second configuration
    /// * `temperature` - Background temperature
    /// 
    /// # Returns
    /// * `(LogProb, LogProb)` - Probabilities for each configuration
    pub fn compare_configurations(
        &self,
        mass1: f64,
        mass2: f64,
        temperature: f64,
    ) -> Result<(LogProb, LogProb), &'static str> {
        let p1 = self.brain_nucleation_probability(mass1, temperature, 1.0)?;
        let p2 = self.brain_nucleation_probability(mass2, temperature, 1.0)?;
        Ok((p1, p2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_prob_creation() {
        let p = LogProb::from_probability(0.5).unwrap();
        assert!(p.ln_p < 0.0);
        assert!((p.ln_p - (-0.693)).abs() < 0.001);
    }

    #[test]
    fn test_log_prob_extreme() {
        // P = 10^(-100)
        let p = LogProb::from_base10_exponent(-100.0);
        assert!(p.ln_p < -200.0); // ln(10) ≈ 2.3
        assert_eq!(p.to_probability(), 0.0); // Underflows
    }

    #[test]
    fn test_log_prob_multiply() {
        let p1 = LogProb::from_probability(0.5).unwrap();
        let p2 = LogProb::from_probability(0.25).unwrap();
        let product = p1.multiply(&p2);
        
        // Should be 0.125
        let expected = LogProb::from_probability(0.125).unwrap();
        assert!((product.ln_p - expected.ln_p).abs() < 1e-10);
    }

    #[test]
    fn test_log_prob_add() {
        let p1 = LogProb::from_probability(0.5).unwrap();
        let p2 = LogProb::from_probability(0.5).unwrap();
        let sum = p1.add(&p2);
        
        // Should be 1.0
        assert!((sum.ln_p - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_nucleation_calculator() {
        let calc = BoltzmannNucleationCalculator::default();
        let params = BoltzmannParams::default();
        
        let rate = calc.calculate_nucleation_rate(&params);
        assert!(rate.ln_p < 0.0); // Very small probability
    }

    #[test]
    fn test_brain_probability() {
        let calc = BoltzmannNucleationCalculator::default();
        
        // Human brain ~1.4 kg at heat death temperature
        let prob = calc.brain_nucleation_probability(1.4, 1e-30, 1.0);
        assert!(prob.is_ok());
        let p = prob.unwrap();
        assert!(p.is_effectively_zero());
    }
}
