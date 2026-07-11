//! Cairns-Blake-Dowd (CBD) Extension for Older Ages
//! 
//! Extends mortality modeling to ages 65+ using the CBD framework,
//! which is critical for pricing longevity derivatives and annuities.

use crate::mortality::lee_carter_kalman::{MortalityModelError, LogMortalityRate};

/// Maximum age supported (up to 120)
pub const MAX_CBD_AGE: usize = 120;

/// Minimum age for CBD model (typically 65+)
pub const MIN_CBD_AGE: usize = 65;

/// Error types for CBD modeling
#[derive(Debug, Clone, PartialEq)]
pub enum CbdError {
    InvalidAge,
    ParameterEstimationFailure,
    NumericalInstability,
    DataInsufficient,
}

impl core::fmt::Display for CbdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidAge => write!(f, "Invalid age"),
            Self::ParameterEstimationFailure => write!(f, "Parameter estimation failure"),
            Self::NumericalInstability => write!(f, "Numerical instability"),
            Self::DataInsufficient => write!(f, "Insufficient data"),
        }
    }
}

/// CBD model parameters (kappa1, kappa2 for two-factor model)
pub struct CbdParams {
    /// Level factor (kappa1_t) - parallel shifts
    kappa1: [f64; 50],
    /// Slope factor (kappa2_t) - age-dependent changes
    kappa2: [f64; 50],
    /// Number of time periods
    n_periods: usize,
    /// Age range start
    age_start: usize,
    /// Age range end
    age_end: usize,
}

impl CbdParams {
    pub const fn new() -> Self {
        Self {
            kappa1: [0.0; 50],
            kappa2: [0.0; 50],
            n_periods: 0,
            age_start: MIN_CBD_AGE,
            age_end: MAX_CBD_AGE,
        }
    }

    #[inline]
    pub fn set_kappa1(&mut self, period: usize, value: f64) -> Result<(), CbdError> {
        if period >= 50 {
            return Err(CbdError::InvalidAge);
        }
        if !value.is_finite() {
            return Err(CbdError::NumericalInstability);
        }
        self.kappa1[period] = value;
        if period >= self.n_periods {
            self.n_periods = period + 1;
        }
        Ok(())
    }

    #[inline]
    pub fn set_kappa2(&mut self, period: usize, value: f64) -> Result<(), CbdError> {
        if period >= 50 {
            return Err(CbdError::InvalidAge);
        }
        if !value.is_finite() {
            return Err(CbdError::NumericalInstability);
        }
        self.kappa2[period] = value;
        Ok(())
    }

    #[inline]
    pub fn get_kappa1(&self, period: usize) -> Option<f64> {
        if period >= self.n_periods {
            return None;
        }
        Some(self.kappa1[period])
    }

    #[inline]
    pub fn get_kappa2(&self, period: usize) -> Option<f64> {
        if period >= self.n_periods {
            return None;
        }
        Some(self.kappa2[period])
    }
}

/// CBD mortality rate calculator
pub struct CbdMortalityModel {
    params: CbdParams,
    /// Age modulation function (x - mean_age)
    age_center: f64,
    /// Volatility parameters
    sigma1: f64,
    sigma2: f64,
    /// Correlation between kappa1 and kappa2
    rho: f64,
}

impl CbdMortalityModel {
    pub fn new() -> Self {
        Self {
            params: CbdParams::new(),
            age_center: 80.0, // Center around age 80
            sigma1: 0.05,
            sigma2: 0.03,
            rho: 0.3,
        }
    }

    /// Initialize parameters from historical mortality data
    pub fn initialize_from_data(
        &mut self,
        log_mortality_rates: &[Vec<f64>],
        ages: &[usize],
    ) -> Result<(), CbdError> {
        if log_mortality_rates.is_empty() || ages.is_empty() {
            return Err(CbdError::DataInsufficient);
        }

        let n_periods = log_mortality_rates.len();
        let n_ages = ages.len();

        if n_ages < 2 {
            return Err(CbdError::DataInsufficient);
        }

        // Compute age center
        let age_sum: usize = ages.iter().sum();
        self.age_center = age_sum as f64 / n_ages as f64;

        // Estimate kappa1 and kappa2 using regression
        // log(q_x,t) = kappa1_t + kappa2_t * (x - x_bar)
        
        for t in 0..n_periods.min(50) {
            let rates = &log_mortality_rates[t];
            
            if rates.len() != n_ages {
                return Err(CbdError::DataInsufficient);
            }

            // Simple OLS estimation
            let mut sum_y = 0.0;
            let mut sum_xy = 0.0;
            let mut sum_xx = 0.0;

            for (i, &age) in ages.iter().enumerate() {
                let x = age as f64 - self.age_center;
                let y = rates[i];
                
                if !y.is_finite() {
                    return Err(CbdError::NumericalInstability);
                }

                sum_y += y;
                sum_xy += x * y;
                sum_xx += x * x;
            }

            let kappa1 = sum_y / n_ages as f64;
            let kappa2 = if sum_xx > 1e-10 {
                sum_xy / sum_xx
            } else {
                0.0
            };

            self.params.set_kappa1(t, kappa1)?;
            self.params.set_kappa2(t, kappa2)?;
        }

        Ok(())
    }

    /// Compute mortality rate at given age and time
    pub fn compute_mortality_rate(
        &self,
        age: usize,
        period: usize,
    ) -> Result<f64, CbdError> {
        if age < MIN_CBD_AGE || age > MAX_CBD_AGE {
            return Err(CbdError::InvalidAge);
        }

        let kappa1 = self.params.get_kappa1(period)
            .ok_or(CbdError::InvalidAge)?;
        let kappa2 = self.params.get_kappa2(period)
            .ok_or(CbdError::InvalidAge)?;

        // CBD formula: logit(q_x) = kappa1 + kappa2 * (x - x_bar)
        let x_centered = age as f64 - self.age_center;
        let logit_q = kappa1 + kappa2 * x_centered;

        // Transform from logit to probability
        let q = 1.0 / (1.0 + (-logit_q).exp());

        if !q.is_finite() || q < 0.0 || q > 1.0 {
            return Err(CbdError::NumericalInstability);
        }

        Ok(q)
    }

    /// Project future kappa values (random walk with drift)
    pub fn project_kappa(
        &self,
        horizon: usize,
    ) -> Result<Vec<(f64, f64)>, CbdError> {
        if horizon == 0 || horizon > 30 {
            return Err(CbdError::InvalidAge);
        }

        let last_period = self.params.n_periods.saturating_sub(1);
        let kappa1_last = self.params.get_kappa1(last_period)
            .ok_or(CbdError::DataInsufficient)?;
        let kappa2_last = self.params.get_kappa2(last_period)
            .ok_or(CbdError::DataInsufficient)?;

        // Estimate drift from historical data
        let drift1 = if last_period > 0 {
            let kappa1_prev = self.params.get_kappa1(last_period - 1).unwrap_or(kappa1_last);
            (kappa1_last - kappa1_prev) / last_period as f64
        } else {
            0.0
        };

        let drift2 = if last_period > 0 {
            let kappa2_prev = self.params.get_kappa2(last_period - 1).unwrap_or(kappa2_last);
            (kappa2_last - kappa2_prev) / last_period as f64
        } else {
            0.0
        };

        let mut projections = Vec::with_capacity(horizon);
        let mut k1 = kappa1_last;
        let mut k2 = kappa2_last;

        for _ in 0..horizon {
            k1 += drift1;
            k2 += drift2;

            // Apply bounds to prevent explosion
            k1 = k1.clamp(-10.0, 10.0);
            k2 = k2.clamp(-1.0, 1.0);

            projections.push((k1, k2));
        }

        Ok(projections)
    }

    /// Set volatility parameters
    pub fn set_volatility(&mut self, sigma1: f64, sigma2: f64) -> Result<(), CbdError> {
        if sigma1 < 0.0 || sigma1 > 1.0 || sigma2 < 0.0 || sigma2 > 1.0 {
            return Err(CbdError::NumericalInstability);
        }
        self.sigma1 = sigma1;
        self.sigma2 = sigma2;
        Ok(())
    }

    /// Set correlation between factors
    pub fn set_correlation(&mut self, rho: f64) -> Result<(), CbdError> {
        if rho.abs() > 1.0 {
            return Err(CbdError::NumericalInstability);
        }
        self.rho = rho;
        Ok(())
    }
}

/// Life expectancy calculator using CBD mortality rates
pub struct CbdLifeExpectancyCalculator {
    cbd_model: CbdMortalityModel,
    /// Discount rate for present value calculations
    discount_rate: f64,
}

impl CbdLifeExpectancyCalculator {
    pub fn new(cbd_model: CbdMortalityModel) -> Self {
        Self {
            cbd_model,
            discount_rate: 0.03,
        }
    }

    /// Compute life expectancy at given age
    pub fn compute_life_expectancy(&self, age: usize, period: usize) -> Result<f64, CbdError> {
        if age < MIN_CBD_AGE || age > MAX_CBD_AGE {
            return Err(CbdError::InvalidAge);
        }

        let mut survival_prob = 1.0;
        let mut le = 0.0;

        for future_age in age..MAX_CBD_AGE {
            let q = self.cbd_model.compute_mortality_rate(future_age, period)?;
            let p = 1.0 - q; // Survival probability for one year
            
            survival_prob *= p;
            le += survival_prob;

            if survival_prob < 1e-6 {
                break; // Negligible survival probability
            }
        }

        Ok(le)
    }

    /// Compute annuity present value factor
    pub fn compute_annuity_pv(&self, age: usize, term: usize, period: usize) -> Result<f64, CbdError> {
        if age < MIN_CBD_AGE || age > MAX_CBD_AGE {
            return Err(CbdError::InvalidAge);
        }
        if term == 0 || term > 50 {
            return Err(CbdError::InvalidAge);
        }

        let mut pv = 0.0;
        let mut survival_prob = 1.0;
        let df_per_year = (-self.discount_rate).exp();

        for t in 0..term {
            let future_age = age + t;
            if future_age >= MAX_CBD_AGE {
                break;
            }

            let q = self.cbd_model.compute_mortality_rate(future_age, period)?;
            survival_prob *= (1.0 - q);

            let df = df_per_year.powi(t as i32);
            pv += survival_prob * df;
        }

        if !pv.is_finite() {
            return Err(CbdError::NumericalInstability);
        }

        Ok(pv)
    }

    /// Set discount rate
    pub fn set_discount_rate(&mut self, rate: f64) -> Result<(), CbdError> {
        if rate < 0.0 || rate > 0.2 {
            return Err(CbdError::NumericalInstability);
        }
        self.discount_rate = rate;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cbd_initialization() {
        let mut model = CbdMortalityModel::new();
        
        let ages = vec![65, 70, 75, 80, 85, 90];
        let rates = vec![
            vec![-4.0, -3.5, -3.0, -2.5, -2.0, -1.5],
            vec![-4.1, -3.6, -3.1, -2.6, -2.1, -1.6],
        ];
        
        assert!(model.initialize_from_data(&rates, &ages).is_ok());
    }

    #[test]
    fn test_mortality_rate_computation() {
        let mut model = CbdMortalityModel::new();
        
        // Set some parameters manually
        model.params.set_kappa1(0, -3.0).unwrap();
        model.params.set_kappa2(0, 0.1).unwrap();
        model.params.n_periods = 1;
        
        let q = model.compute_mortality_rate(75, 0).unwrap();
        assert!(q > 0.0 && q < 1.0);
    }

    #[test]
    fn test_life_expectancy() {
        let mut model = CbdMortalityModel::new();
        model.params.set_kappa1(0, -3.0).unwrap();
        model.params.set_kappa2(0, 0.1).unwrap();
        model.params.n_periods = 1;
        
        let calc = CbdLifeExpectancyCalculator::new(model);
        let le = calc.compute_life_expectancy(65, 0).unwrap();
        assert!(le > 10.0 && le < 30.0); // Reasonable range
    }
}
