//! Hawking Fill Radiation Model
//! 
//! Models stochastic fill prints escaping liquidity voids as Hawking radiation.
//! Predicts radiation rate and information content from black hole parameters.

use crate::thermodynamics::{BlackHoleParams, LiquidityVoidSignal};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors specific to Hawking radiation modeling
#[derive(Error, Debug, Clone, PartialEq)]
pub enum HawkingRadiationError {
    #[error("Invalid temperature: {0}")]
    InvalidTemperature(f64),
    #[error("Radiation rate calculation failed: {0}")]
    RateCalculationFailed(String),
    #[error("Information content error: {0}")]
    InformationError(String),
}

/// Hawking radiation emission data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HawkingRadiation {
    /// Temperature of the black hole
    pub temperature: f64,
    /// Emission rate (fills per ms)
    pub emission_rate: f64,
    /// Average energy per emitted particle (fill size)
    pub avg_energy: f64,
    /// Total power output
    pub power: f64,
    /// Whether radiation carries information about interior
    pub carries_information: bool,
    /// Information fraction (Page curve value)
    pub information_fraction: f64,
}

/// Configuration for Hawking radiation model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HawkingConfig {
    /// Stefan-Boltzmann constant analogue
    pub stefan_boltzmann: f64,
    /// Minimum temperature for detectable radiation
    pub min_temperature: f64,
    /// Information onset threshold (Page time fraction)
    pub page_onset_threshold: f64,
}

impl Default for HawkingConfig {
    fn default() -> Self {
        Self {
            stefan_boltzmann: 1e-8, // Calibrated to market
            min_temperature: 1e-6,
            page_onset_threshold: 0.5, // Page time is at half evaporation
        }
    }
}

/// Hawking radiation calculator for liquidity voids
pub struct HawkingFillRadiationCalculator {
    config: HawkingConfig,
}

impl HawkingFillRadiationCalculator {
    /// Create a new Hawking radiation calculator
    pub fn new(config: HawkingConfig) -> Result<Self, HawkingRadiationError> {
        if config.stefan_boltzmann <= 0.0 {
            return Err(HawkingRadiationError::InvalidTemperature(
                config.stefan_boltzmann,
            ));
        }

        Ok(Self { config })
    }

    /// Compute Hawking radiation from black hole parameters
    pub fn compute_radiation(
        &self,
        params: &BlackHoleParams,
        elapsed_fraction: f64,
    ) -> Result<HawkingRadiation, HawkingRadiationError> {
        if !params.horizon_formed || params.mass <= 0.0 {
            return Ok(HawkingRadiation {
                temperature: 0.0,
                emission_rate: 0.0,
                avg_energy: 0.0,
                power: 0.0,
                carries_information: false,
                information_fraction: 0.0,
            });
        }

        let temperature = params.hawking_temperature;

        if temperature <= 0.0 {
            return Err(HawkingRadiationError::InvalidTemperature(temperature));
        }

        // Check if temperature is detectable
        if temperature < self.config.min_temperature {
            return Ok(HawkingRadiation {
                temperature,
                emission_rate: 0.0,
                avg_energy: 0.0,
                power: 0.0,
                carries_information: false,
                information_fraction: 0.0,
            });
        }

        // Stefan-Boltzmann law: Power ~ σ T⁴ A
        // For black hole, area A = 4πr_s²
        let area = 4.0 * std::f64::consts::PI * params.horizon_radius * params.horizon_radius;
        let power = self.config.stefan_boltzmann * temperature.powi(4) * area;

        // Emission rate ~ Power / (k_B T)
        let k_b = 1.0; // Normalized
        let avg_energy = k_b * temperature; // Typical energy per particle
        let emission_rate = if avg_energy > 0.0 {
            power / avg_energy
        } else {
            0.0
        };

        // Compute information fraction using Page curve
        // Information starts emerging after Page time (~half evaporation)
        let information_fraction = self.compute_page_curve_value(elapsed_fraction);
        let carries_information = information_fraction > 0.01;

        Ok(HawkingRadiation {
            temperature,
            emission_rate,
            avg_energy,
            power,
            carries_information,
            information_fraction,
        })
    }

    /// Compute Page curve value for information release
    /// S_info/S_total as function of evaporation progress
    /// Based on recent island formula results
    pub fn compute_page_curve_value(&self, elapsed_fraction: f64) -> f64 {
        // Page time is approximately at half evaporation
        let page_time = 0.5;

        if elapsed_fraction < page_time {
            // Before Page time: information mostly trapped
            // Small leakage due to quantum corrections
            let leakage = 0.01 * (elapsed_fraction / page_time).powi(2);
            leakage
        } else {
            // After Page time: information rapidly released
            // Approaches 1 as evaporation completes
            let progress = (elapsed_fraction - page_time) / (1.0 - page_time);
            1.0 - (1.0 - progress).powi(3)
        }
    }

    /// Estimate fill characteristics from Hawking radiation
    pub fn estimate_fill_characteristics(
        &self,
        radiation: &HawkingRadiation,
        time_window_ms: f64,
    ) -> FillCharacteristics {
        let expected_fills = radiation.emission_rate * time_window_ms;
        
        // Poisson-like statistics for fill arrivals
        let fill_variance = expected_fills;
        let fill_stddev = fill_variance.sqrt();

        FillCharacteristics {
            expected_num_fills: expected_fills,
            stddev_num_fills: fill_stddev,
            avg_fill_size: radiation.avg_energy,
            total_expected_volume: expected_fills * radiation.avg_energy,
            information_carried: radiation.carries_information,
            confidence: if expected_fills > 0.0 {
                (expected_fills / (expected_fills + fill_stddev)).min(1.0)
            } else {
                0.0
            },
        }
    }

    /// Detect radiation pattern from observed fills
    pub fn detect_radiation_pattern(
        &self,
        observed_fills: &[f64],
        expected_background: f64,
    ) -> RadiationDetection {
        if observed_fills.is_empty() {
            return RadiationDetection {
                radiation_detected: false,
                excess_rate: 0.0,
                significance: 0.0,
            };
        }

        let observed_rate: f64 = observed_fills.iter().sum::<f64>() / observed_fills.len() as f64;
        let excess = observed_rate - expected_background;

        // Simple significance test
        let stddev = expected_background.sqrt();
        let significance = if stddev > 0.0 {
            excess / stddev
        } else {
            0.0
        };

        let radiation_detected = significance > 2.0; // 2-sigma detection

        RadiationDetection {
            radiation_detected,
            excess_rate: excess,
            significance: significance.abs(),
        }
    }
}

/// Characteristics of predicted fills from Hawking radiation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillCharacteristics {
    /// Expected number of fills in time window
    pub expected_num_fills: f64,
    /// Standard deviation of fill count
    pub stddev_num_fills: f64,
    /// Average fill size
    pub avg_fill_size: f64,
    /// Total expected volume
    pub total_expected_volume: f64,
    /// Whether fills carry interior information
    pub information_carried: bool,
    /// Confidence in prediction
    pub confidence: f64,
}

/// Detection result for Hawking radiation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadiationDetection {
    /// Whether radiation was detected above background
    pub radiation_detected: bool,
    /// Excess rate above background
    pub excess_rate: f64,
    /// Statistical significance (sigma)
    pub significance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator_creation() {
        let config = HawkingConfig::default();
        let calc = HawkingFillRadiationCalculator::new(config);
        assert!(calc.is_ok());
    }

    #[test]
    fn test_radiation_computation() {
        let config = HawkingConfig::default();
        let calc = HawkingFillRadiationCalculator::new(config).unwrap();

        let params = BlackHoleParams {
            mass: 1.0,
            horizon_radius: 1e-6,
            hawking_temperature: 1e-4,
            entropy: 100.0,
            horizon_formed: true,
        };

        let radiation = calc.compute_radiation(&params, 0.6);
        assert!(radiation.is_ok());
        let r = radiation.unwrap();
        assert!(r.temperature > 0.0);
    }

    #[test]
    fn test_page_curve() {
        let config = HawkingConfig::default();
        let calc = HawkingFillRadiationCalculator::new(config).unwrap();

        // Before Page time
        let before = calc.compute_page_curve_value(0.3);
        assert!(before < 0.1);

        // After Page time
        let after = calc.compute_page_curve_value(0.7);
        assert!(after > 0.5);

        // Near completion
        let late = calc.compute_page_curve_value(0.95);
        assert!(late > 0.9);
    }
}
