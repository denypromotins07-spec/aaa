//! Tetration Math Library for Cosmological Timescales
//! 
//! CRITICAL: Standard u128 or f64 cannot represent Poincaré recurrence times
//! like 10^(10^(10^(10^10))) without overflow. This library implements Knuth's
//! up-arrow notation as an abstract syntax tree, allowing comparison and limited
//! arithmetic on these extreme values without ever evaluating the full integer.
//! 
//! Key insight: We store expressions like a↑↑n as (base=a, height=n) and only
//! evaluate comparisons using asymptotic bounds, never the actual value.

use core::f64;

/// Representation of large numbers using Knuth's up-arrow notation
/// 
/// A number is represented as an expression tree rather than evaluated:
/// - Literal(n): A small literal that fits in f64
/// - Arrow(base, arrows, height): base ↑^arrows height
///   - arrows=1: exponentiation (a^n)
///   - arrows=2: tetration (a^^n = a^(a^(...^a)) with n copies)
///   - arrows=3: pentation, etc.
#[derive(Debug, Clone)]
pub enum HyperNumber {
    /// A literal value that fits in f64
    Literal(f64),
    /// Up-arrow expression: base ↑^arrows height
    Arrow {
        base: Box<HyperNumber>,
        arrows: u32,
        height: Box<HyperNumber>,
    },
    /// Logarithm of a hypernumber (for intermediate calculations)
    Log(Box<HyperNumber>),
    /// Infinity
    Infinity,
}

impl HyperNumber {
    /// Create a literal hypernumber
    pub fn literal(n: f64) -> Self {
        HyperNumber::Literal(n)
    }
    
    /// Create infinity
    pub fn infinity() -> Self {
        HyperNumber::Infinity
    }
    
    /// Create a tetration: base ^^ height
    /// This is base^(base^(base^...)) with 'height' copies of base
    pub fn tetration(base: f64, height: f64) -> Self {
        HyperNumber::Arrow {
            base: Box::new(HyperNumber::literal(base)),
            arrows: 2,
            height: Box::new(HyperNumber::literal(height)),
        }
    }
    
    /// Create a power tower of specified height
    /// tower(10, 5) = 10^10^10^10^10
    pub fn power_tower(base: f64, height: u32) -> Self {
        if height == 0 {
            return HyperNumber::literal(1.0);
        }
        if height == 1 {
            return HyperNumber::literal(base);
        }
        
        // Build right-associative: base^(base^(base^...))
        let mut result = HyperNumber::literal(base);
        for _ in 1..height {
            result = HyperNumber::Arrow {
                base: Box::new(HyperNumber::literal(base)),
                arrows: 1,
                height: Box::new(result),
            };
        }
        result
    }
    
    /// Get the approximate log10 of this number (as a HyperNumber)
    /// For a^^n, log10(a^^n) ≈ (a^^(n-1)) * log10(a)
    pub fn approx_log10(&self) -> HyperNumber {
        match self {
            HyperNumber::Literal(n) => {
                if *n > 0.0 {
                    HyperNumber::literal(n.log10())
                } else {
                    HyperNumber::Infinity // log(0) = -inf, but we use inf for "undefined"
                }
            }
            HyperNumber::Infinity => HyperNumber::Infinity,
            HyperNumber::Log(inner) => {
                // log(log(x)) - further reduce
                HyperNumber::Log(Box::new(inner.approx_log10()))
            }
            HyperNumber::Arrow { base, arrows, height } => {
                if *arrows == 1 {
                    // log(a^n) = n * log(a)
                    HyperNumber::Literal(height.approx_log10_scale() * base.approx_log10_scale())
                } else if *arrows >= 2 {
                    // log(a^^n) ≈ (a^^(n-1)) * log(a)
                    // For large towers, this is dominated by a^^(n-1)
                    let reduced_height = height.decrement();
                    HyperNumber::Arrow {
                        base: base.clone(),
                        arrows: *arrows - 1,
                        height: Box::new(reduced_height),
                    }
                } else {
                    HyperNumber::Infinity
                }
            }
        }
    }
    
    /// Get a rough scale factor for comparison purposes
    /// Returns the "effective exponent" - how many times you'd take log to get to ~1
    pub fn approx_log10_scale(&self) -> f64 {
        match self {
            HyperNumber::Literal(n) => n.log10().max(-308.0).min(308.0),
            HyperNumber::Infinity => f64::INFINITY,
            HyperNumber::Log(_) => -f64::INFINITY, // Already logged
            HyperNumber::Arrow { base, arrows, height } => {
                // For a^^n, the scale is roughly the height of the tower
                if *arrows >= 2 {
                    height.approx_tower_height()
                } else {
                    height.approx_log10_scale() * base.approx_log10_scale()
                }
            }
        }
    }
    
    /// Estimate the "tower height" - number of exponentiations
    fn approx_tower_height(&self) -> f64 {
        match self {
            HyperNumber::Literal(n) => {
                if *n < 2.0 { 0.0 } else { 1.0 }
            }
            HyperNumber::Infinity => f64::INFINITY,
            HyperNumber::Log(_) => 0.0,
            HyperNumber::Arrow { base: _, arrows, height } => {
                if *arrows >= 2 {
                    // Each level of arrow adds to tower height
                    1.0 + height.approx_tower_height()
                } else {
                    height.approx_tower_height()
                }
            }
        }
    }
    
    /// Decrement a hypernumber (for reducing tower heights)
    fn decrement(&self) -> HyperNumber {
        match self {
            HyperNumber::Literal(n) => HyperNumber::literal((n - 1.0).max(0.0)),
            HyperNumber::Infinity => HyperNumber::Infinity,
            HyperNumber::Log(inner) => HyperNumber::Log(Box::new(inner.decrement())),
            HyperNumber::Arrow { base, arrows, height } => {
                if *arrows >= 2 {
                    // Reducing the height of a tetration
                    let new_height = height.decrement();
                    if let HyperNumber::Literal(h) = *new_height {
                        if h <= 0.0 {
                            return HyperNumber::literal(1.0); // a^^0 = 1
                        }
                    }
                    HyperNumber::Arrow {
                        base: base.clone(),
                        arrows: *arrows,
                        height: Box::new(new_height),
                    }
                } else {
                    HyperNumber::Arrow {
                        base: base.clone(),
                        arrows: *arrows,
                        height: Box::new(height.decrement()),
                    }
                }
            }
        }
    }
    
    /// Compare two hypernumbers without full evaluation
    /// Returns: -1 if self < other, 0 if equal, 1 if self > other
    pub fn compare(&self, other: &HyperNumber) -> i32 {
        match (self, other) {
            (HyperNumber::Infinity, HyperNumber::Infinity) => 0,
            (HyperNumber::Infinity, _) => 1,
            (_, HyperNumber::Infinity) => -1,
            
            (HyperNumber::Literal(a), HyperNumber::Literal(b)) => {
                if a < b { -1 } else if a > b { 1 } else { 0 }
            }
            
            // Compare by effective scale (number of log operations to reach ~1)
            (a, b) => {
                let scale_a = a.approx_log10_scale();
                let scale_b = b.approx_log10_scale();
                
                if scale_a.is_infinite() && !scale_b.is_infinite() {
                    return 1;
                }
                if !scale_a.is_infinite() && scale_b.is_infinite() {
                    return -1;
                }
                
                // For finite scales, compare directly
                if scale_a > scale_b + 1.0 {
                    1
                } else if scale_b > scale_a + 1.0 {
                    -1
                } else {
                    // Scales are close, need more detailed comparison
                    // Fall back to comparing structure
                    a.structural_compare(b)
                }
            }
        }
    }
    
    /// Structural comparison for numbers with similar scales
    fn structural_compare(&self, other: &HyperNumber) -> i32 {
        match (self, other) {
            (HyperNumber::Literal(a), HyperNumber::Literal(b)) => {
                if a < b { -1 } else if a > b { 1 } else { 0 }
            }
            (HyperNumber::Arrow { base: ba, arrows: aa, height: ha },
             HyperNumber::Arrow { base: bb, arrows: ab, height: hb }) => {
                // Compare by arrows first (more arrows = larger)
                if aa != ab {
                    return if aa > ab { 1 } else { -1 };
                }
                // Same arrows, compare bases
                let base_cmp = ba.compare(bb);
                if base_cmp != 0 {
                    return base_cmp;
                }
                // Same base and arrows, compare heights
                ha.compare(hb)
            }
            // Mixed types: arrows > literals for large values
            (HyperNumber::Arrow { .. }, HyperNumber::Literal(_)) => 1,
            (HyperNumber::Literal(_), HyperNumber::Arrow { .. }) => -1,
            _ => 0,
        }
    }
    
    /// Check if this number is definitely larger than another
    pub fn definitely_larger_than(&self, other: &HyperNumber) -> bool {
        self.compare(other) > 0
    }
    
    /// Render as a string in up-arrow notation
    pub fn to_uparrow_string(&self) -> String {
        match self {
            HyperNumber::Literal(n) => format!("{:.2}", n),
            HyperNumber::Infinity => "∞".to_string(),
            HyperNumber::Log(inner) => format!("log({})", inner.to_uparrow_string()),
            HyperNumber::Arrow { base, arrows, height } => {
                let arrow_str = match arrows {
                    1 => "^",
                    2 => "↑↑",
                    3 => "↑↑↑",
                    n => format!("↑^{}", n),
                };
                format!("{}{}{}", base.to_uparrow_string(), arrow_str, height.to_uparrow_string())
            }
        }
    }
}

/// Poincaré recurrence time calculator
#[derive(Debug, Clone)]
pub struct PoincareRecurrenceCalculator {
    /// Boltzmann constant
    k_b: f64,
    /// Planck constant
    hbar: f64,
}

impl Default for PoincareRecurrenceCalculator {
    fn default() -> Self {
        Self {
            k_b: 1.380_649e-23,
            hbar: 1.054_571_817e-34,
        }
    }
}

impl PoincareRecurrenceCalculator {
    /// Calculate Poincaré recurrence time for a system
    /// 
    /// t_recurrence ≈ exp(S/k_B) where S is the entropy
    /// For a system with N microstates: t ≈ N * τ where τ is characteristic time
    /// 
    /// # Arguments
    /// * `entropy` - System entropy [J/K]
    /// * `characteristic_time` - Typical dynamical timescale [s]
    /// 
    /// # Returns
    /// * `HyperNumber` - Recurrence time (may be a tower)
    pub fn calculate_recurrence_time(
        &self,
        entropy: f64,
        characteristic_time: f64,
    ) -> Result<HyperNumber, &'static str> {
        if entropy <= 0.0 {
            return Err("Entropy must be positive");
        }
        if characteristic_time <= 0.0 {
            return Err("Characteristic time must be positive");
        }
        
        // Number of microstates: Ω = exp(S/k_B)
        let omega_exponent = entropy / self.k_b;
        
        // For cosmological systems, this exponent is enormous
        // e.g., for observable universe S ~ 10^104 J/K
        // omega_exponent ~ 10^127
        
        // Recurrence time: t = τ * Ω = τ * exp(S/k_B)
        // We represent this as a hypernumber
        
        if omega_exponent > 709.0 {
            // Would overflow f64, use tower representation
            // exp(x) for large x is approximately 10^(x/ln(10))
            let log10_omega = omega_exponent / core::f64::consts::LN_10;
            
            // Represent as 10^10^... with appropriate height
            if log10_omega > 1e10 {
                // Need multiple levels of tower
                HyperNumber::tetration(10.0, log10_omega.log10())
            } else {
                HyperNumber::Arrow {
                    base: Box::new(HyperNumber::literal(10.0)),
                    arrows: 1,
                    height: Box::new(HyperNumber::literal(log10_omega)),
                }
            }
        } else {
            // Can represent as literal
            let omega = omega_exponent.exp();
            HyperNumber::literal(characteristic_time * omega)
        }
    }
    
    /// Compare recurrence times for different systems
    /// 
    /// # Arguments
    /// * `entropy1` - First system entropy
    /// * `entropy2` - Second system entropy
    /// 
    /// # Returns
    /// * `i32` - Comparison result (-1, 0, 1)
    pub fn compare_recurrence_times(&self, entropy1: f64, entropy2: f64) -> i32 {
        if entropy1 <= 0.0 || entropy2 <= 0.0 {
            return 0;
        }
        
        let t1 = match self.calculate_recurrence_time(entropy1, 1.0) {
            Ok(t) => t,
            Err(_) => return 0,
        };
        let t2 = match self.calculate_recurrence_time(entropy2, 1.0) {
            Ok(t) => t,
            Err(_) => return 0,
        };
        
        t1.compare(&t2)
    }
    
    /// Calculate recurrence time for a black hole
    /// Using Bekenstein-Hawking entropy: S = A/(4*l_P²)
    /// 
    /// # Arguments
    /// * `mass` - Black hole mass [kg]
    /// 
    /// # Returns
    /// * `HyperNumber` - Recurrence time
    pub fn black_hole_recurrence(&self, mass: f64) -> Result<HyperNumber, &'static str> {
        if mass <= 0.0 {
            return Err("Mass must be positive");
        }
        
        let c = 299_792_458.0;
        let g = 6.674_30e-11;
        let planck_length_sq = self.hbar * g / c.powi(3);
        
        // Schwarzschild radius
        let r_s = 2.0 * g * mass / c.powi(2);
        
        // Horizon area
        let area = 4.0 * core::f64::consts::PI * r_s.powi(2);
        
        // Bekenstein-Hawking entropy
        let entropy = area / (4.0 * planck_length_sq) * self.k_b;
        
        // Characteristic time: light crossing time
        let tau = r_s / c;
        
        self.calculate_recurrence_time(entropy, tau)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_creation() {
        let n = HyperNumber::literal(42.0);
        assert!(matches!(n, HyperNumber::Literal(42.0)));
    }

    #[test]
    fn test_tetration_creation() {
        let t = HyperNumber::tetration(3.0, 4.0);
        assert!(matches!(t, HyperNumber::Arrow { arrows: 2, .. }));
    }

    #[test]
    fn test_power_tower() {
        let tower = HyperNumber::power_tower(10.0, 3);
        // 10^10^10
        assert!(matches!(tower, HyperNumber::Arrow { .. }));
    }

    #[test]
    fn test_comparison_literals() {
        let a = HyperNumber::literal(10.0);
        let b = HyperNumber::literal(100.0);
        assert_eq!(a.compare(&b), -1);
        assert_eq!(b.compare(&a), 1);
        assert_eq!(a.compare(&a), 0);
    }

    #[test]
    fn test_comparison_towers() {
        // 10^^3 vs 10^^2
        let t1 = HyperNumber::tetration(10.0, 3.0);
        let t2 = HyperNumber::tetration(10.0, 2.0);
        assert_eq!(t1.compare(&t2), 1);
    }

    #[test]
    fn test_poincare_calculator() {
        let calc = PoincareRecurrenceCalculator::default();
        
        // Small entropy - should give literal
        let t = calc.calculate_recurrence_time(1e-20, 1.0);
        assert!(t.is_ok());
        
        // Large entropy - should give tower
        let t_large = calc.calculate_recurrence_time(1e10, 1.0);
        assert!(t_large.is_ok());
    }

    #[test]
    fn test_uparrow_string() {
        let t = HyperNumber::tetration(10.0, 3.0);
        let s = t.to_uparrow_string();
        assert!(s.contains("↑↑"));
    }
}
