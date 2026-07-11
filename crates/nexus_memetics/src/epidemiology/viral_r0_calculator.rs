//! Viral Reproduction Number (R0) Calculator for Financial Narratives
//! 
//! Computes the basic reproduction number R0 = beta / gamma to determine
//! if a narrative will go viral (R0 > 1.0) or die out (R0 < 1.0).

use crate::epidemiology::financial_sir_ode::{SirParameters, SirOdeError};

/// Threshold constants for viral classification
pub const R0_VIRAL_THRESHOLD: f64 = 1.0;
pub const R0_SUPER_VIRAL_THRESHOLD: f64 = 2.5;
pub const R0_CRITICAL_THRESHOLD: f64 = 5.0;

/// Classification of narrative viral potential
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViralClassification {
    /// R0 < 1.0: Narrative will die out
    Dying,
    /// 1.0 <= R0 < 2.5: Moderate spread
    Spreading,
    /// 2.5 <= R0 < 5.0: Viral outbreak
    Viral,
    /// R0 >= 5.0: Pandemic-level spread
    Pandemic,
}

impl ViralClassification {
    pub fn from_r0(r0: f64) -> Self {
        if r0 < R0_VIRAL_THRESHOLD {
            Self::Dying
        } else if r0 < R0_SUPER_VIRAL_THRESHOLD {
            Self::Spreading
        } else if r0 < R0_CRITICAL_THRESHOLD {
            Self::Viral
        } else {
            Self::Pandemic
        }
    }

    pub fn is_viral(&self) -> bool {
        matches!(self, Self::Viral | Self::Pandemic)
    }

    pub fn urgency_score(&self) -> f64 {
        match self {
            Self::Dying => 0.0,
            Self::Spreading => 0.3,
            Self::Viral => 0.7,
            Self::Pandemic => 1.0,
        }
    }
}

/// Enhanced R0 calculator with time-varying parameters
pub struct ViralR0Calculator {
    base_beta: f64,
    base_gamma: f64,
    /// Social media amplification factor
    amplification: f64,
    /// Narrative fatigue coefficient (reduces beta over time)
    fatigue_rate: f64,
}

impl ViralR0Calculator {
    pub fn new(beta: f64, gamma: f64) -> Result<Self, SirOdeError> {
        if beta <= 0.0 || gamma <= 0.0 {
            return Err(SirOdeError::InvalidParameter);
        }
        Ok(Self {
            base_beta: beta,
            base_gamma: gamma,
            amplification: 1.0,
            fatigue_rate: 0.0,
        })
    }

    pub fn with_amplification(mut self, amp: f64) -> Self {
        self.amplification = amp.max(0.1);
        self
    }

    pub fn with_fatigue(mut self, rate: f64) -> Self {
        self.fatigue_rate = rate.max(0.0);
        self
    }

    /// Calculate instantaneous R0 at time t
    #[inline]
    pub fn r0_at_time(&self, t: f64) -> f64 {
        // Time-varying beta due to fatigue
        let effective_beta = self.base_beta * self.amplification * (-self.fatigue_rate * t).exp();
        effective_beta / self.base_gamma
    }

    /// Calculate peak R0 (at t=0)
    #[inline]
    pub fn peak_r0(&self) -> f64 {
        (self.base_beta * self.amplification) / self.base_gamma
    }

    /// Get viral classification based on current R0
    pub fn classify(&self, t: f64) -> ViralClassification {
        ViralClassification::from_r0(self.r0_at_time(t))
    }

    /// Calculate time until R0 drops below 1.0 (narrative death)
    pub fn time_to_extinction(&self) -> Option<f64> {
        if self.peak_r0() < R0_VIRAL_THRESHOLD {
            return Some(0.0); // Already dying
        }

        // Solve: R0(t) = 1.0
        // (beta * amp * exp(-fatigue * t)) / gamma = 1.0
        // exp(-fatigue * t) = gamma / (beta * amp)
        // -fatigue * t = ln(gamma / (beta * amp))
        // t = -ln(gamma / (beta * amp)) / fatigue
        
        let ratio = self.base_gamma / (self.base_beta * self.amplification);
        if ratio >= 1.0 {
            return Some(0.0);
        }

        if self.fatigue_rate > 1e-10 {
            let t = -ratio.ln() / self.fatigue_rate;
            Some(t.max(0.0))
        } else {
            None // Never goes extinct without fatigue
        }
    }

    /// Calculate the "herd immunity" threshold: fraction that must be immune/recovered
    /// to stop the narrative spread
    pub fn herd_immunity_threshold(&self) -> f64 {
        let r0 = self.peak_r0();
        if r0 <= 1.0 {
            0.0
        } else {
            (1.0 - 1.0 / r0).max(0.0).min(1.0)
        }
    }

    /// Estimate final epidemic size (total fraction infected over entire outbreak)
    /// Using the final size equation: R_inf = 1 - exp(-R0 * R_inf)
    pub fn final_epidemic_size(&self) -> f64 {
        let r0 = self.peak_r0();
        if r0 <= 1.0 {
            return 0.0;
        }

        // Newton-Raphson to solve: R = 1 - exp(-R0 * R)
        let mut r = 0.5; // Initial guess
        for _ in 0..50 {
            let f = r - 1.0 + (-r0 * r).exp();
            let df = 1.0 - r0 * (-r0 * r).exp();
            
            if df.abs() < 1e-15 {
                break;
            }
            
            let delta = f / df;
            r -= delta;
            
            if delta.abs() < 1e-12 {
                break;
            }
            r = r.clamp(0.0, 1.0);
        }
        
        r
    }

    /// Convert to SIR parameters
    pub fn to_sir_parameters(&self) -> Result<SirParameters, SirOdeError> {
        SirParameters::new(self.base_beta * self.amplification, self.base_gamma)
    }

    /// Calculate the growth rate during early epidemic phase
    pub fn initial_growth_rate(&self) -> f64 {
        let r0 = self.peak_r0();
        if r0 <= 1.0 {
            0.0
        } else {
            self.base_gamma * (r0 - 1.0)
        }
    }

    /// Doubling time during exponential growth phase
    pub fn doubling_time(&self) -> Option<f64> {
        let rate = self.initial_growth_rate();
        if rate <= 1e-15 {
            None
        } else {
            Some(std::f64::consts::LN_2 / rate)
        }
    }
}

/// Batch calculator for comparing multiple narratives
pub struct NarrativeComparison {
    calculators: Vec<(String, ViralR0Calculator)>,
}

impl NarrativeComparison {
    pub fn new() -> Self {
        Self { calculators: Vec::new() }
    }

    pub fn add_narrative(&mut self, name: String, calc: ViralR0Calculator) {
        self.calculators.push((name, calc));
    }

    /// Rank narratives by viral potential
    pub fn rank_by_peak_r0(&self) -> Vec<(String, f64, ViralClassification)> {
        let mut ranked: Vec<_> = self.calculators
            .iter()
            .map(|(name, calc)| {
                let r0 = calc.peak_r0();
                (name.clone(), r0, calc.classify(0.0))
            })
            .collect();
        
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Find the most urgent narrative to trade
    pub fn most_urgent(&self) -> Option<(String, f64)> {
        self.calculators
            .iter()
            .filter(|(_, calc)| calc.classify(0.0).is_viral())
            .max_by(|a, b| {
                a.1.peak_r0().partial_cmp(&b.1.peak_r0()).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, calc)| (name.clone(), calc.peak_r0()))
    }
}

impl Default for NarrativeComparison {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_r0_calculation() {
        let calc = ViralR0Calculator::new(0.5, 0.1).unwrap();
        assert!((calc.peak_r0() - 5.0).abs() < 1e-10);
        assert_eq!(calc.classify(0.0), ViralClassification::Pandemic);
    }

    #[test]
    fn test_fatigue_effect() {
        let calc = ViralR0Calculator::new(2.0, 0.5)
            .unwrap()
            .with_fatigue(0.1);
        
        let r0_initial = calc.r0_at_time(0.0);
        let r0_later = calc.r0_at_time(10.0);
        
        assert!(r0_later < r0_initial);
    }

    #[test]
    fn test_herd_immunity() {
        let calc = ViralR0Calculator::new(2.0, 0.5).unwrap();
        // For R0 = 4, HIT = 1 - 1/4 = 0.75
        assert!((calc.herd_immunity_threshold() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_doubling_time() {
        let calc = ViralR0Calculator::new(1.0, 0.1).unwrap();
        // R0 = 10, growth rate = 0.1 * 9 = 0.9
        // Doubling time = ln(2) / 0.9 ≈ 0.77
        let dt = calc.doubling_time().unwrap();
        assert!((dt - std::f64::consts::LN_2 / 0.9).abs() < 1e-10);
    }
}
