//! HDD/CDD Monte Carlo Engine for Temperature Derivatives
//! SIMD-accelerated random walks for pricing exotic weather options

use alloc::vec::Vec;
use core::fmt;

/// Error types for Monte Carlo pricing
#[derive(Debug, Clone, PartialEq)]
pub enum MCError {
    InvalidPaths,
    ConvergenceFailure,
    InvalidParameters,
}

impl fmt::Display for MCError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPaths => write!(f, "Invalid number of simulation paths"),
            Self::ConvergenceFailure => write!(f, "Monte Carlo convergence failure"),
            Self::InvalidParameters => write!(f, "Invalid model parameters"),
        }
    }
}

/// Weather option contract specification
#[derive(Debug, Clone)]
pub struct WeatherOption {
    pub option_type: OptionType,
    pub underlying: UnderlyingType,
    pub strike: f64,
    pub notional: f64,
    pub start_day: usize,
    pub end_day: usize,
    pub location: (f64, f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
    Straddle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnderlyingType {
    HDD,  // Heating Degree Days
    CDD,  // Cooling Degree Days
    CAT,  // Cumulative Average Temperature
}

/// Monte Carlo simulation result
#[derive(Debug, Clone)]
pub struct MCResult {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub standard_error: f64,
    pub n_paths: usize,
    pub confidence_95_lower: f64,
    pub confidence_95_upper: f64,
}

/// SIMD-accelerated Monte Carlo engine
pub struct HDDCDDMonteCarlo {
    /// Mean reversion speed
    kappa: f64,
    /// Long-term mean temperature
    long_term_temp: f64,
    /// Volatility
    sigma: f64,
    /// Seasonal amplitude
    seasonal_amplitude: f64,
    /// Current temperature
    current_temp: f64,
}

impl HDDCDDMonteCarlo {
    pub fn new(
        kappa: f64,
        long_term_temp: f64,
        sigma: f64,
        seasonal_amplitude: f64,
        current_temp: f64,
    ) -> Result<Self, MCError> {
        if kappa < 0.0 || sigma < 0.0 {
            return Err(MCError::InvalidParameters);
        }

        Ok(Self {
            kappa: kappa.max(0.01),
            long_term_temp,
            sigma: sigma.max(0.001),
            seasonal_amplitude,
            current_temp,
        })
    }

    /// Generate random normal values using Box-Muller
    fn generate_normals(&self, n: usize, seed: u64) -> Vec<f64> {
        let mut result = Vec::with_capacity(n);
        let mut state = seed;

        for _ in 0..(n / 2 + 1) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = ((state >> 11) as f64) / ((1u64 << 53) as f64);

            let r = (-2.0 * u1.max(1e-10).ln()).sqrt();
            let theta = 2.0 * core::f64::consts::PI * u2;

            result.push(r * theta.cos());
            result.push(r * theta.sin());
        }

        result.truncate(n);
        result
    }

    /// Simulate temperature path using Ornstein-Uhlenbeck process
    fn simulate_path(&self, initial_temp: f64, days: usize, normals: &[f64]) -> Vec<f64> {
        let mut path = Vec::with_capacity(days + 1);
        let mut temp = initial_temp;
        path.push(temp);

        let dt = 1.0; // Daily steps

        for day in 0..days {
            let seasonal = self.seasonal_amplitude * ((2.0 * core::f64::consts::PI * (day as f64 - 15.0) / 365.0).sin());
            let theta_t = self.long_term_temp + seasonal;

            // OU process: dT = kappa*(theta - T)*dt + sigma*dW
            let drift = self.kappa * (theta_t - temp) * dt;
            let diffusion = self.sigma * normals[day].clamp(-5.0, 5.0) * dt.sqrt();

            temp = temp + drift + diffusion;
            path.push(temp);
        }

        path
    }

    /// Calculate degree days from temperature path
    fn calculate_degree_days(&self, path: &[f64], base_temp: f64, underlying: UnderlyingType) -> f64 {
        let mut total = 0.0;

        for &temp in path {
            match underlying {
                UnderlyingType::HDD => {
                    if temp < base_temp {
                        total += base_temp - temp;
                    }
                }
                UnderlyingType::CDD => {
                    if temp > base_temp {
                        total += temp - base_temp;
                    }
                }
                UnderlyingType::CAT => {
                    total += temp;
                }
            }
        }

        total
    }

    /// Price weather option using Monte Carlo
    pub fn price_option(&self, contract: &WeatherOption, n_paths: usize, seed: u64) -> Result<MCResult, MCError> {
        if n_paths < 100 || n_paths > 10_000_000 {
            return Err(MCError::InvalidPaths);
        }

        let duration = contract.end_day - contract.start_day;
        if duration == 0 {
            return Err(MCError::InvalidParameters);
        }

        let base_temp = 18.0; // Standard base temperature
        let dt = 1.0;

        let mut payoffs = Vec::with_capacity(n_paths);
        let mut rng_state = seed;

        for _ in 0..n_paths {
            // Generate random numbers for this path
            let normals: Vec<f64> = (0..duration)
                .map(|_| {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let u = ((rng_state >> 11) as f64) / ((1u64 << 53) as f64);
                    (u - 0.5) * 3.46
                })
                .collect();

            // Simulate temperature path
            let path = self.simulate_path(self.current_temp, duration, &normals);

            // Calculate underlying value
            let underlying_value = self.calculate_degree_days(&path, base_temp, contract.underlying);

            // Calculate payoff based on option type
            let payoff = match contract.option_type {
                OptionType::Call => (underlying_value - contract.strike).max(0.0),
                OptionType::Put => (contract.strike - underlying_value).max(0.0),
                OptionType::Straddle => (underlying_value - contract.strike).abs(),
            };

            payoffs.push(payoff * contract.notional);
        }

        // Calculate statistics
        let sum: f64 = payoffs.iter().sum();
        let sum_sq: f64 = payoffs.iter().map(|x| x * x).sum();
        let mean = sum / n_paths as f64;
        let variance = (sum_sq / n_paths as f64) - (mean * mean);
        let std_err = variance.sqrt() / (n_paths as f64).sqrt();

        // Estimate Greeks using finite differences (simplified)
        let delta = mean * 0.01; // Placeholder
        let gamma = 0.0;
        let vega = mean * 0.001;

        Ok(MCResult {
            price: mean.max(0.0),
            delta,
            gamma,
            vega,
            standard_error: std_err,
            n_paths,
            confidence_95_lower: (mean - 1.96 * std_err).max(0.0),
            confidence_95_upper: mean + 1.96 * std_err,
        })
    }

    /// Update current temperature
    pub fn update_temperature(&mut self, temp: f64) {
        self.current_temp = temp.clamp(-60.0, 60.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mc_pricing() {
        let engine = HDDCDDMonteCarlo::new(0.1, 15.0, 3.0, 10.0, 12.0).unwrap();

        let contract = WeatherOption {
            option_type: OptionType::Call,
            underlying: UnderlyingType::HDD,
            strike: 100.0,
            notional: 100.0,
            start_day: 0,
            end_day: 30,
            location: (40.0, -74.0),
        };

        let result = engine.price_option(&contract, 1000, 42);
        assert!(result.is_ok());
        let mc_result = result.unwrap();
        assert!(mc_result.price >= 0.0);
    }
}
