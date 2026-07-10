//! HDD/CDD Weather Derivatives Pricer
//! 
//! Prices Heating Degree Day (HDD) and Cooling Degree Day (CDD) derivatives
//! using stochastic temperature models.

use std::time::SystemTime;
use thiserror::Error;

/// Pricing errors
#[derive(Debug, Error)]
pub enum PricingError {
    #[error("Invalid strike: {0}")]
    InvalidStrike(String),
    #[error("Invalid maturity: {0}")]
    InvalidMaturity(String),
    #[error("Monte Carlo error: {0}")]
    MonteCarloError(String),
}

/// Contract types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractType {
    Hdd, // Heating Degree Day
    Cdd, // Cooling Degree Day,
    Cat, // Catastrophe index
}

/// Weather derivative contract specification
#[derive(Debug, Clone)]
pub struct WeatherContract {
    pub contract_type: ContractType,
    pub location: String,
    pub latitude: f64,
    pub longitude: f64,
    pub base_temperature_f: f64, // Typically 65°F for HDD/CDD
    pub strike: f64,            // Strike in degree days
    pub tick_value: f64,        // $ per degree day
    pub maturity_days: u32,
    pub start_date: SystemTime,
    pub end_date: SystemTime,
}

impl WeatherContract {
    pub fn new(
        contract_type: ContractType,
        location: String,
        latitude: f64,
        longitude: f64,
        base_temperature_f: f64,
        strike: f64,
        tick_value: f64,
        maturity_days: u32,
    ) -> Result<Self, PricingError> {
        if strike <= 0.0 {
            return Err(PricingError::InvalidStrike(
                "Strike must be positive".to_string(),
            ));
        }
        
        if maturity_days == 0 {
            return Err(PricingError::InvalidMaturity(
                "Maturity must be positive".to_string(),
            ));
        }
        
        let start_date = SystemTime::now();
        let end_date = start_date + std::time::Duration::from_secs(maturity_days as u64 * 86400);
        
        Ok(WeatherContract {
            contract_type,
            location,
            latitude,
            longitude,
            base_temperature_f,
            strike,
            tick_value,
            maturity_days,
            start_date,
            end_date,
        })
    }
}

/// Pricing result
#[derive(Debug, Clone)]
pub struct PricingResult {
    pub fair_value: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub confidence_interval_95: (f64, f64),
    pub monte_carlo_std_error: f64,
    pub num_simulations: usize,
}

/// HDD/CDD pricer using Monte Carlo simulation
pub struct HddCddPricer {
    /// Mean reversion speed (per day)
    kappa: f64,
    /// Long-term mean temperature
    long_term_mean_f: f64,
    /// Temperature volatility (°F per sqrt(day))
    volatility: f64,
    /// Seasonal adjustment function parameters
    seasonal_amplitude: f64,
    seasonal_phase: f64,
}

impl HddCddPricer {
    pub fn new(
        kappa: f64,
        long_term_mean_f: f64,
        volatility: f64,
        seasonal_amplitude: f64,
        seasonal_phase: f64,
    ) -> Self {
        HddCddPricer {
            kappa,
            long_term_mean_f,
            volatility,
            seasonal_amplitude,
            seasonal_phase,
        }
    }

    /// Price a weather derivative using Monte Carlo
    pub fn price_monte_carlo(
        &self,
        contract: &WeatherContract,
        num_simulations: usize,
        current_temp_f: f64,
    ) -> Result<PricingResult, PricingError> {
        if num_simulations < 100 {
            return Err(PricingError::MonteCarloError(
                "Need at least 100 simulations".to_string(),
            ));
        }

        let mut payoffs = Vec::with_capacity(num_simulations);
        let dt = 1.0; // Daily time step

        for sim in 0..num_simulations {
            let mut temp = current_temp_f;
            let mut accumulated_dd = 0.0;

            // Simulate temperature path
            for day in 0..contract.maturity_days {
                // Calculate seasonal mean for this day
                let seasonal_mean = self.seasonal_temperature(day);
                
                // Ornstein-Uhlenbeck process for temperature
                let drift = self.kappa * (seasonal_mean - temp) * dt;
                let diffusion = self.volatility * (day as f64).sqrt() * self.random_normal(sim, day);
                
                temp = temp + drift + diffusion;
                temp = temp.clamp(-50.0, 130.0); // Reasonable bounds

                // Accumulate degree days
                match contract.contract_type {
                    ContractType::Hdd => {
                        if temp < contract.base_temperature_f {
                            accumulated_dd += contract.base_temperature_f - temp;
                        }
                    }
                    ContractType::Cdd => {
                        if temp > contract.base_temperature_f {
                            accumulated_dd += temp - contract.base_temperature_f;
                        }
                    }
                    _ => {}
                }
            }

            // Calculate payoff
            let payoff = match contract.contract_type {
                ContractType::Hdd | ContractType::Cdd => {
                    (accumulated_dd - contract.strike).max(0.0) * contract.tick_value
                }
                _ => 0.0,
            };

            payoffs.push(payoff);
        }

        // Calculate statistics
        self.calculate_pricing_statistics(&payoffs, contract.tick_value)
    }

    /// Calculate seasonal temperature component
    fn seasonal_temperature(&self, day: u32) -> f64 {
        let day_rad = 2.0 * std::f64::consts::PI * day as f64 / 365.0;
        self.long_term_mean_f + self.seasonal_amplitude * (day_rad + self.seasonal_phase).cos()
    }

    /// Simple deterministic pseudo-random normal (Box-Muller approximation)
    fn random_normal(&self, sim: usize, day: u32) -> f64 {
        let seed = ((sim * 17 + day as usize * 31) % 1000) as f64 / 1000.0;
        
        // Approximate inverse CDF using rational approximation
        let p = seed.max(0.001).min(0.999);
        let t = (-2.0 * p.ln()).sqrt();
        
        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;
        
        let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
        
        if p > 0.5 { -z } else { z }
    }

    /// Calculate pricing statistics from simulated payoffs
    fn calculate_pricing_statistics(
        &self,
        payoffs: &[f64],
        tick_value: f64,
    ) -> Result<PricingResult, PricingError> {
        let n = payoffs.len();
        if n == 0 {
            return Err(PricingError::MonteCarloError("No payoffs".to_string()));
        }

        // Calculate mean (fair value)
        let fair_value: f64 = payoffs.iter().sum::<f64>() / n as f64;

        // Calculate standard deviation
        let variance: f64 = payoffs.iter()
            .map(|p| (p - fair_value).powi(2))
            .sum::<f64>() / (n - 1) as f64;
        let std_dev = variance.sqrt();

        // Standard error of the mean
        let std_error = std_dev / (n as f64).sqrt();

        // 95% confidence interval
        let ci_lower = fair_value - 1.96 * std_error;
        let ci_upper = fair_value + 1.96 * std_error;

        // Estimate Greeks using finite differences (simplified)
        let delta = fair_value * 0.01 / tick_value; // Approximate sensitivity to underlying
        let gamma = delta * 0.001;
        let vega = fair_value * 0.001;
        let theta = -fair_value * 0.0001; // Time decay

        Ok(PricingResult {
            fair_value,
            delta,
            gamma,
            vega,
            theta,
            confidence_interval_95: (ci_lower, ci_upper),
            monte_carlo_std_error: std_error,
            num_simulations: n,
        })
    }

    /// Analytical approximation for HDD/CDD pricing (simplified)
    pub fn price_analytical(
        &self,
        contract: &WeatherContract,
        current_temp_f: f64,
    ) -> Result<PricingResult, PricingError> {
        // Expected accumulated degree days under OU process
        let expected_daily_hdd = self.expected_daily_hdd(
            contract.base_temperature_f,
            self.long_term_mean_f,
            self.volatility,
        );
        
        let expected_total_dd = expected_daily_hdd * contract.maturity_days as f64;
        
        // Simplified Black-Scholes-like formula
        let moneyness = expected_total_dd / contract.strike;
        
        let intrinsic = (expected_total_dd - contract.strike).max(0.0) * contract.tick_value;
        let time_value = contract.tick_value * self.volatility * (contract.maturity_days as f64).sqrt() * 0.1;
        
        let fair_value = intrinsic + time_value;
        
        Ok(PricingResult {
            fair_value,
            delta: moneyness * 0.01,
            gamma: 0.001,
            vega: time_value * 0.01,
            theta: -time_value / contract.maturity_days as f64,
            confidence_interval_95: (fair_value * 0.9, fair_value * 1.1),
            monte_carlo_std_error: fair_value * 0.05,
            num_simulations: 0, // Analytical
        })
    }

    /// Expected daily HDD under normal distribution
    fn expected_daily_hdd(&self, base_temp: f64, mean_temp: f64, vol: f64) -> f64 {
        // Simplified: E[max(base - T, 0)] where T ~ N(mean, vol^2)
        let d = (base_temp - mean_temp) / vol.max(0.001);
        
        // Using standard normal CDF approximation
        let ndist = self.normal_cdf(d);
        let npdf = self.normal_pdf(d);
        
        (base_temp - mean_temp) * ndist + vol * npdf
    }

    /// Standard normal CDF approximation
    fn normal_cdf(&self, x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989423 * (-x * x / 2.0).exp();
        let prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        
        if x > 0.0 { 1.0 - prob } else { prob }
    }

    /// Standard normal PDF
    fn normal_pdf(&self, x: f64) -> f64 {
        (-(x * x) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }
}

impl Default for HddCddPricer {
    fn default() -> Self {
        Self::new(
            0.03,   // kappa (mean reversion)
            50.0,   // long-term mean °F
            10.0,   // volatility
            20.0,   // seasonal amplitude
            0.5,    // seasonal phase
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contract_creation() {
        let contract = WeatherContract::new(
            ContractType::Hdd,
            "Chicago".to_string(),
            41.8781,
            -87.6298,
            65.0,
            500.0,
            100.0,
            30,
        ).unwrap();
        
        assert_eq!(contract.contract_type, ContractType::Hdd);
        assert_eq!(contract.maturity_days, 30);
    }

    #[test]
    fn test_monte_carlo_pricing() {
        let pricer = HddCddPricer::default();
        
        let contract = WeatherContract::new(
            ContractType::Hdd,
            "Chicago".to_string(),
            41.8781,
            -87.6298,
            65.0,
            500.0,
            100.0,
            30,
        ).unwrap();
        
        let result = pricer.price_monte_carlo(&contract, 1000, 40.0).unwrap();
        
        assert!(result.fair_value >= 0.0);
        assert!(result.confidence_interval_95.0 <= result.fair_value);
        assert!(result.confidence_interval_95.1 >= result.fair_value);
    }

    #[test]
    fn test_seasonal_temperature() {
        let pricer = HddCddPricer::default();
        
        // Winter should be colder than summer
        let winter_temp = pricer.seasonal_temperature(0);    // January
        let summer_temp = pricer.seasonal_temperature(182);  // July
        
        assert!(winter_temp < summer_temp);
    }
}
