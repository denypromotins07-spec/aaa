//! Stochastic Enthalpy Pricer for Weather Derivatives
//! Models thermodynamic energy content of atmosphere for pricing HDD/CDD options

use alloc::vec::Vec;
use core::fmt;

/// Error types for enthalpy pricing
#[derive(Debug, Clone, PartialEq)]
pub enum EnthalpyError {
    InvalidTemperature,
    InvalidHumidity,
    NumericalOverflow,
    ConvergenceFailure,
    InvalidStrike,
}

impl fmt::Display for EnthalpyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTemperature => write!(f, "Invalid temperature value"),
            Self::InvalidHumidity => write!(f, "Invalid humidity value"),
            Self::NumericalOverflow => write!(f, "Numerical overflow in calculation"),
            Self::ConvergenceFailure => write!(f, "Monte Carlo convergence failure"),
            Self::InvalidStrike => write!(f, "Invalid strike price"),
        }
    }
}

/// Atmospheric state for enthalpy calculation
#[derive(Debug, Clone, Copy)]
pub struct AtmosphericState {
    /// Temperature (°C)
    pub temperature_c: f64,
    /// Relative humidity (0-1)
    pub relative_humidity: f64,
    /// Pressure (Pa)
    pub pressure_pa: f64,
    /// Specific humidity (kg/kg)
    pub specific_humidity: f64,
}

impl AtmosphericState {
    /// Create from basic measurements
    pub fn new(temperature_c: f64, relative_humidity: f64, pressure_pa: f64) -> Result<Self, EnthalpyError> {
        if temperature_c < -100.0 || temperature_c > 60.0 {
            return Err(EnthalpyError::InvalidTemperature);
        }
        if relative_humidity < 0.0 || relative_humidity > 1.0 {
            return Err(EnthalpyError::InvalidHumidity);
        }
        if pressure_pa < 80000.0 || pressure_pa > 110000.0 {
            return Err(EnthalpyError::InvalidTemperature);
        }

        // Calculate specific humidity from RH and temperature
        let e_sat = Self::saturation_vapor_pressure(temperature_c);
        let e_actual = e_sat * relative_humidity;
        let specific_humidity = 0.622 * e_actual / (pressure_pa - e_actual);

        Ok(Self {
            temperature_c,
            relative_humidity,
            pressure_pa,
            specific_humidity,
        })
    }

    /// Saturation vapor pressure (Tetens formula)
    fn saturation_vapor_pressure(temp_c: f64) -> f64 {
        610.78 * ((17.27 * temp_c) / (temp_c + 237.3)).exp()
    }

    /// Calculate specific enthalpy (J/kg dry air)
    /// h = c_pd * T + w * (L_v + c_pv * T)
    pub fn specific_enthalpy(&self) -> f64 {
        let c_pd = 1005.0;  // Specific heat of dry air (J/kg·K)
        let c_pv = 1846.0;  // Specific heat of water vapor (J/kg·K)
        let l_v = 2.501e6;  // Latent heat of vaporization (J/kg)
        let t_k = self.temperature_c + 273.15;

        c_pd * t_k + self.specific_humidity * (l_v + c_pv * t_k)
    }

    /// Calculate moist air density (kg/m³)
    pub fn moist_density(&self) -> f64 {
        let r_d = 287.05;  // Gas constant for dry air
        let r_v = 461.5;   // Gas constant for water vapor
        let t_k = self.temperature_c + 273.15;

        let e = self.relative_humidity * Self::saturation_vapor_pressure(self.temperature_c);
        let p_d = self.pressure_pa - e;

        (p_d / (r_d * t_k)) + (e / (r_v * t_k))
    }
}

/// Heating/Cooling Degree Day contract specification
#[derive(Debug, Clone)]
pub struct HddCddContract {
    /// Contract type
    pub is_hdd: bool,  // true = HDD, false = CDD
    /// Base temperature (°C)
    pub base_temp: f64,
    /// Strike level (degree days)
    pub strike: f64,
    /// Tick size ($ per degree day)
    pub tick_size: f64,
    /// Start date (microseconds timestamp)
    pub start_us: u64,
    /// End date (microseconds timestamp)
    pub end_us: u64,
    /// Location
    pub location: (f64, f64),
}

/// Monte Carlo simulation result
#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    /// Estimated option price
    pub price: f64,
    /// Standard error
    pub standard_error: f64,
    /// Number of paths simulated
    pub n_paths: usize,
    /// 95% confidence interval lower bound
    pub ci_lower: f64,
    /// 95% confidence interval upper bound
    pub ci_upper: f64,
}

/// Stochastic Enthalpy Pricer using Monte Carlo simulation
pub struct StochasticEnthalpyPricer {
    /// Mean reversion speed (per day)
    kappa: f64,
    /// Long-term mean enthalpy (J/kg)
    long_term_enthalpy: f64,
    /// Volatility of enthalpy
    volatility: f64,
    /// Seasonal amplitude
    seasonal_amplitude: f64,
    /// Current enthalpy state
    current_enthalpy: f64,
}

impl StochasticEnthalpyPricer {
    /// Create new pricer with Ornstein-Uhlenbeck dynamics
    /// dH = kappa*(theta(t) - H)*dt + sigma*dW
    pub fn new(
        kappa: f64,
        long_term_enthalpy: f64,
        volatility: f64,
        seasonal_amplitude: f64,
        initial_enthalpy: f64,
    ) -> Result<Self, EnthalpyError> {
        if kappa < 0.0 {
            return Err(EnthalpyError::InvalidTemperature);
        }
        if volatility < 0.0 {
            return Err(EnthalpyError::InvalidHumidity);
        }
        if initial_enthalpy.abs() > 1e8 {
            return Err(EnthalpyError::NumericalOverflow);
        }

        Ok(Self {
            kappa: kappa.max(0.01),
            long_term_enthalpy,
            volatility: volatility.max(0.001),
            seasonal_amplitude,
            current_enthalpy: initial_enthalpy,
        })
    }

    /// Get seasonal component for day of year
    fn seasonal_component(&self, day_of_year: usize) -> f64 {
        let phase = 2.0 * std::f64::consts::PI * (day_of_year as f64 - 15.0) / 365.0;
        self.seasonal_amplitude * phase.sin()
    }

    /// Simulate one path of enthalpy evolution using Euler-Maruyama
    fn simulate_path(&self, initial_h: f64, dt: f64, n_steps: usize, random_values: &[f64]) -> Vec<f64> {
        let mut path = Vec::with_capacity(n_steps + 1);
        let mut h = initial_h;
        path.push(h);

        for step in 0..n_steps {
            let day = step % 365;
            let theta_t = self.long_term_enthalpy + self.seasonal_component(day);
            
            // OU process: dH = kappa*(theta - H)*dt + sigma*dW
            let drift = self.kappa * (theta_t - h) * dt;
            let diffusion = self.volatility * random_values[step].max(-5.0).min(5.0) * dt.sqrt();
            
            h = h + drift + diffusion;
            
            // Clamp to physical bounds
            h = h.clamp(2.5e5, 3.5e5);
            path.push(h);
        }

        path
    }

    /// Convert enthalpy path to degree days
    fn enthalpy_to_degree_days(&self, enthalpy_path: &[f64], base_temp: f64, is_hdd: bool) -> f64 {
        let mut total_dd = 0.0;
        
        for &h in enthalpy_path {
            // Approximate temperature from enthalpy (simplified)
            let approx_temp = (h - 2.7e5) / 1000.0 + 15.0;
            
            if is_hdd && approx_temp < base_temp {
                total_dd += base_temp - approx_temp;
            } else if !is_hdd && approx_temp > base_temp {
                total_dd += approx_temp - base_temp;
            }
        }
        
        total_dd
    }

    /// Price European HDD/CDD option using Monte Carlo
    pub fn price_option(
        &self,
        contract: &HddCddContract,
        n_paths: usize,
        rng_seed: u64,
    ) -> Result<MonteCarloResult, EnthalpyError> {
        if n_paths < 100 || n_paths > 1_000_000 {
            return Err(EnthalpyError::ConvergenceFailure);
        }
        if contract.strike < 0.0 {
            return Err(EnthalpyError::InvalidStrike);
        }

        let dt = 1.0 / 24.0; // Daily steps
        let duration_days = ((contract.end_us - contract.start_us) / (24 * 3600 * 1_000_000)) as usize;
        let n_steps = duration_days.max(1);

        // Simple LCG for reproducibility
        let mut rng_state = rng_seed;
        let next_rng = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*state >> 11) as f64) / ((1u64 << 53) as f64)
        };

        let mut payoffs_sum = 0.0;
        let mut payoffs_sq_sum = 0.0;

        for _ in 0..n_paths {
            // Generate random values for this path
            let random_values: Vec<f64> = (0..n_steps)
                .map(|_| {
                    let u = next_rng(&mut rng_state);
                    // Box-Muller approximation (simplified)
                    (u - 0.5) * 3.46 // Approximate N(0,1)
                })
                .collect();

            // Simulate enthalpy path
            let enthalpy_path = self.simulate_path(self.current_enthalpy, dt, n_steps, &random_values);

            // Convert to degree days
            let total_dd = self.enthalpy_to_degree_days(&enthalpy_path, contract.base_temp, contract.is_hdd);

            // Calculate payoff (call option on degree days)
            let payoff = (total_dd - contract.strike).max(0.0) * contract.tick_size;

            payoffs_sum += payoff;
            payoffs_sq_sum += payoff * payoff;
        }

        // Discount to present (simplified: assume r = 0 for short duration)
        let mean_payoff = payoffs_sum / n_paths as f64;
        let variance = (payoffs_sq_sum / n_paths as f64) - (mean_payoff * mean_payoff);
        let std_dev = variance.sqrt();
        let standard_error = std_dev / (n_paths as f64).sqrt();

        let ci_lower = mean_payoff - 1.96 * standard_error;
        let ci_upper = mean_payoff + 1.96 * standard_error;

        Ok(MonteCarloResult {
            price: mean_payoff.max(0.0),
            standard_error,
            n_paths,
            ci_lower: ci_lower.max(0.0),
            ci_upper,
        })
    }

    /// Update current enthalpy state from observation
    pub fn update_state(&mut self, temperature_c: f64, humidity: f64, pressure_pa: f64) -> Result<(), EnthalpyError> {
        let state = AtmosphericState::new(temperature_c, humidity, pressure_pa)?;
        self.current_enthalpy = state.specific_enthalpy();
        Ok(())
    }

    /// Get current enthalpy
    pub fn current_enthalpy(&self) -> f64 {
        self.current_enthalpy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atmospheric_state() {
        let state = AtmosphericState::new(20.0, 0.5, 101325.0).unwrap();
        assert!(state.specific_enthalpy() > 0.0);
        assert!(state.moist_density() > 0.0);
    }

    #[test]
    fn test_enthalpy_pricer() {
        let pricer = StochasticEnthalpyPricer::new(
            0.1,      // kappa
            2.8e5,    // long term enthalpy
            5000.0,   // volatility
            20000.0,  // seasonal amplitude
            2.75e5,   // initial
        ).unwrap();

        let contract = HddCddContract {
            is_hdd: true,
            base_temp: 18.0,
            strike: 100.0,
            tick_size: 20.0,
            start_us: 0,
            end_us: 90 * 24 * 3600 * 1_000_000,
            location: (40.0, -74.0),
        };

        let result = pricer.price_option(&contract, 1000, 42);
        assert!(result.is_ok());
        let mc_result = result.unwrap();
        assert!(mc_result.price >= 0.0);
    }
}
