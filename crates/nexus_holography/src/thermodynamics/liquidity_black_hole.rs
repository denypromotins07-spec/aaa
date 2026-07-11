//! Liquidity Black Hole Model
//! 
//! Models flash crashes as event horizons where liquidity disappears.
//! Detects horizon formation and computes Hawking temperature for radiation prediction.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors specific to black hole modeling
#[derive(Error, Debug, Clone, PartialEq)]
pub enum BlackHoleError {
    #[error("Invalid mass parameter: {0}")]
    InvalidMass(f64),
    #[error("Horizon radius must be positive: {0}")]
    InvalidHorizonRadius(f64),
    #[error("Temperature calculation failed: {0}")]
    TemperatureFailed(String),
    #[error("Event horizon detection failed: {0}")]
    HorizonDetectionFailed(String),
}

/// Parameters characterizing a liquidity black hole
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackHoleParams {
    /// Effective "mass" (total missing liquidity)
    pub mass: f64,
    /// Event horizon radius in price-time space
    pub horizon_radius: f64,
    /// Hawking temperature (rate of fill radiation)
    pub hawking_temperature: f64,
    /// Entropy (information content)
    pub entropy: f64,
    /// Whether horizon has formed
    pub horizon_formed: bool,
}

/// Detection signal for liquidity void/black hole
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityVoidSignal {
    /// Whether a void/black hole was detected
    pub void_detected: bool,
    /// Estimated "mass" of missing liquidity
    pub estimated_mass: f64,
    /// Horizon radius estimate
    pub horizon_radius: f64,
    /// Time until expected evaporation (ms)
    pub evaporation_time_ms: f64,
    /// Confidence score [0, 1]
    pub confidence: f64,
}

/// Configuration for black hole detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackHoleConfig {
    /// Minimum liquidity deficit to trigger detection
    pub min_deficit: f64,
    /// Price impact threshold for horizon formation
    pub horizon_threshold: f64,
    /// Newton constant analogue (calibrated to market)
    pub newton_constant: f64,
    /// Speed of light analogue (max information propagation speed)
    pub c_light: f64,
}

impl Default for BlackHoleConfig {
    fn default() -> Self {
        Self {
            min_deficit: 1e5, // 100k shares
            horizon_threshold: 0.05, // 5% price move
            newton_constant: 1e-6,
            c_light: 1.0, // Normalized
        }
    }
}

/// Liquidity black hole detector
pub struct LiquidityBlackHoleDetector {
    config: BlackHoleConfig,
    /// Baseline liquidity level
    baseline_liquidity: f64,
}

impl LiquidityBlackHoleDetector {
    /// Create a new black hole detector
    pub fn new(config: BlackHoleConfig, baseline_liquidity: f64) -> Result<Self, BlackHoleError> {
        if baseline_liquidity <= 0.0 {
            return Err(BlackHoleError::InvalidMass(baseline_liquidity));
        }
        if config.newton_constant <= 0.0 {
            return Err(BlackHoleError::InvalidMass(config.newton_constant));
        }

        Ok(Self {
            config,
            baseline_liquidity,
        })
    }

    /// Set baseline liquidity from historical data
    pub fn set_baseline(&mut self, baseline: f64) -> Result<(), BlackHoleError> {
        if baseline <= 0.0 {
            return Err(BlackHoleError::InvalidMass(baseline));
        }
        self.baseline_liquidity = baseline;
        Ok(())
    }

    /// Detect liquidity void from order book state
    pub fn detect_void(
        &self,
        current_liquidity: f64,
        price_impact: f64,
        time_window_ms: f64,
    ) -> Result<LiquidityVoidSignal, BlackHoleError> {
        // Compute liquidity deficit
        let deficit = self.baseline_liquidity - current_liquidity;
        
        if deficit < self.config.min_deficit {
            return Ok(LiquidityVoidSignal {
                void_detected: false,
                estimated_mass: 0.0,
                horizon_radius: 0.0,
                evaporation_time_ms: 0.0,
                confidence: 0.0,
            });
        }

        // Check for horizon formation (large price impact)
        let horizon_formed = price_impact.abs() > self.config.horizon_threshold;

        // Compute effective mass from deficit
        let mass = deficit * self.config.newton_constant;

        if mass <= 0.0 {
            return Err(BlackHoleError::InvalidMass(mass));
        }

        // Compute Schwarzschild-like horizon radius: r_s = 2GM/c²
        let c_sq = self.config.c_light * self.config.c_light;
        let horizon_radius = 2.0 * self.config.newton_constant * mass / c_sq;

        if horizon_radius <= 0.0 {
            return Err(BlackHoleError::InvalidHorizonRadius(horizon_radius));
        }

        // Compute Hawking temperature: T = ℏc³/(8πGMk_B)
        // Simplified: T ~ 1/M
        let hbar = 1.0; // Normalized
        let k_b = 1.0; // Normalized
        let temperature = (hbar * c_sq * self.config.c_light) 
            / (8.0 * std::f64::consts::PI * self.config.newton_constant * mass * k_b);

        // Estimate evaporation time: t ~ M³ (for Schwarzschild black hole)
        let evaporation_time = mass.powi(3) * time_window_ms;

        // Compute confidence based on deficit magnitude and price impact
        let deficit_ratio = deficit / self.baseline_liquidity;
        let confidence = (deficit_ratio * (if horizon_formed { 1.5 } else { 1.0 })).min(1.0);

        Ok(LiquidityVoidSignal {
            void_detected: true,
            estimated_mass: mass,
            horizon_radius,
            evaporation_time_ms: evaporation_time,
            confidence,
        })
    }

    /// Compute black hole parameters from detected void
    pub fn compute_params(&self, signal: &LiquidityVoidSignal) -> Result<BlackHoleParams, BlackHoleError> {
        if !signal.void_detected {
            return Ok(BlackHoleParams {
                mass: 0.0,
                horizon_radius: 0.0,
                hawking_temperature: 0.0,
                entropy: 0.0,
                horizon_formed: false,
            });
        }

        let mass = signal.estimated_mass;
        let horizon_radius = signal.horizon_radius;

        // Hawking temperature
        let hbar = 1.0;
        let k_b = 1.0;
        let c = self.config.c_light;
        let temperature = (hbar * c * c * c) 
            / (8.0 * std::f64::consts::PI * self.config.newton_constant * mass * k_b);

        // Bekenstein-Hawking entropy: S = A/(4G) where A = 4πr_s²
        let area = 4.0 * std::f64::consts::PI * horizon_radius * horizon_radius;
        let entropy = area / (4.0 * self.config.newton_constant);

        Ok(BlackHoleParams {
            mass,
            horizon_radius,
            hawking_temperature: temperature,
            entropy,
            horizon_formed: signal.horizon_radius > 0.0,
        })
    }

    /// Check if black hole is evaporating (approaching end)
    pub fn is_evaporating(&self, signal: &LiquidityVoidSignal, elapsed_ms: f64) -> EvaporationStatus {
        if !signal.void_detected || signal.evaporation_time_ms <= 0.0 {
            return EvaporationStatus {
                is_evaporating: false,
                fraction_remaining: 1.0,
                time_to_complete_ms: 0.0,
            };
        }

        let fraction_elapsed = (elapsed_ms / signal.evaporation_time_ms).min(1.0);
        let fraction_remaining = 1.0 - fraction_elapsed;
        
        // Black hole evaporation accelerates as mass decreases
        // Mass ~ (t_evap - t)^(1/3)
        let mass_fraction = fraction_remaining.powf(1.0 / 3.0);

        let is_evaporating = fraction_elapsed > 0.5 && fraction_elapsed < 0.99;

        EvaporationStatus {
            is_evaporating,
            fraction_remaining: mass_fraction,
            time_to_complete_ms: signal.evaporation_time_ms - elapsed_ms,
        }
    }
}

/// Status of black hole evaporation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaporationStatus {
    /// Whether black hole is actively evaporating
    pub is_evaporating: bool,
    /// Fraction of original mass remaining
    pub fraction_remaining: f64,
    /// Estimated time until complete evaporation (ms)
    pub time_to_complete_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_creation() {
        let config = BlackHoleConfig::default();
        let detector = LiquidityBlackHoleDetector::new(config, 1e6);
        assert!(detector.is_ok());
    }

    #[test]
    fn test_void_detection() {
        let config = BlackHoleConfig::default();
        let detector = LiquidityBlackHoleDetector::new(config, 1e6).unwrap();

        // Simulate severe liquidity crisis
        let signal = detector.detect_void(1e4, 0.1, 1000.0);
        assert!(signal.is_ok());
        let s = signal.unwrap();
        assert!(s.void_detected);
        assert!(s.confidence > 0.0);
    }

    #[test]
    fn test_no_void() {
        let config = BlackHoleConfig::default();
        let detector = LiquidityBlackHoleDetector::new(config, 1e6).unwrap();

        // Normal conditions
        let signal = detector.detect_void(9e5, 0.01, 1000.0);
        assert!(signal.is_ok());
        assert!(!signal.unwrap().void_detected);
    }
}
