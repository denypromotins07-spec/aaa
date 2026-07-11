//! Lee-Carter Stochastic Mortality Model with Kalman Filtering
//! 
//! Implements the Lee-Carter mortality model with time-varying parameters
//! estimated via Kalman filtering for real-time mortality forecasting.

use core::slice;

/// Maximum number of age groups supported
pub const MAX_AGE_GROUPS: usize = 120; // Ages 0-119

/// Maximum time periods in state vector
pub const MAX_TIME_PERIODS: usize = 100;

/// Error types for mortality modeling
#[derive(Debug, Clone, PartialEq)]
pub enum MortalityModelError {
    InvalidMortalityRate,
    ParameterDivergence,
    KalmanFilterFailure,
    NumericalInstability,
    DataMismatch,
    NonPositiveDefinite,
}

impl core::fmt::Display for MortalityModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidMortalityRate => write!(f, "Invalid mortality rate"),
            Self::ParameterDivergence => write!(f, "Model parameter divergence detected"),
            Self::KalmanFilterFailure => write!(f, "Kalman filter failure"),
            Self::NumericalInstability => write!(f, "Numerical instability"),
            Self::DataMismatch => write!(f, "Data dimension mismatch"),
            Self::NonPositiveDefinite => write!(f, "Non-positive definite matrix"),
        }
    }
}

/// Log mortality rate (prevents underflow for small rates)
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct LogMortalityRate(f64);

impl LogMortalityRate {
    #[inline]
    pub fn new(rate: f64) -> Result<Self, MortalityModelError> {
        if rate <= 0.0 || rate > 1.0 {
            return Err(MortalityModelError::InvalidMortalityRate);
        }
        Ok(Self(rate.ln()))
    }

    #[inline]
    pub fn from_rate_unchecked(rate: f64) -> Self {
        Self(rate.ln())
    }

    #[inline]
    pub fn get_rate(self) -> f64 {
        self.0.exp()
    }

    #[inline]
    pub fn get_log(self) -> f64 {
        self.0
    }
}

/// Lee-Carter model parameters
pub struct LeeCarterParams {
    /// Age-specific intercept (alpha_x)
    alpha: [f64; MAX_AGE_GROUPS],
    /// Age-specific sensitivity to period effect (beta_x)
    beta: [f64; MAX_AGE_GROUPS],
    /// Time-varying period index (kappa_t)
    kappa: [f64; MAX_TIME_PERIODS],
    /// Number of active age groups
    n_ages: usize,
    /// Number of time periods
    n_periods: usize,
}

impl LeeCarterParams {
    pub const fn new() -> Self {
        Self {
            alpha: [0.0; MAX_AGE_GROUPS],
            beta: [0.0; MAX_AGE_GROUPS],
            kappa: [0.0; MAX_TIME_PERIODS],
            n_ages: 0,
            n_periods: 0,
        }
    }

    #[inline]
    pub fn set_alpha(&mut self, age_idx: usize, value: f64) -> Result<(), MortalityModelError> {
        if age_idx >= MAX_AGE_GROUPS {
            return Err(MortalityModelError::DataMismatch);
        }
        if !value.is_finite() {
            return Err(MortalityModelError::NumericalInstability);
        }
        self.alpha[age_idx] = value;
        if age_idx >= self.n_ages {
            self.n_ages = age_idx + 1;
        }
        Ok(())
    }

    #[inline]
    pub fn set_beta(&mut self, age_idx: usize, value: f64) -> Result<(), MortalityModelError> {
        if age_idx >= MAX_AGE_GROUPS {
            return Err(MortalityModelError::DataMismatch);
        }
        if !value.is_finite() {
            return Err(MortalityModelError::NumericalInstability);
        }
        self.beta[age_idx] = value;
        Ok(())
    }

    #[inline]
    pub fn set_kappa(&mut self, period_idx: usize, value: f64) -> Result<(), MortalityModelError> {
        if period_idx >= MAX_TIME_PERIODS {
            return Err(MortalityModelError::DataMismatch);
        }
        if !value.is_finite() || value.abs() > 1e6 {
            return Err(MortalityModelError::ParameterDivergence);
        }
        self.kappa[period_idx] = value;
        if period_idx >= self.n_periods {
            self.n_periods = period_idx + 1;
        }
        Ok(())
    }

    #[inline]
    pub fn get_kappa(&self, period_idx: usize) -> Option<f64> {
        if period_idx >= self.n_periods {
            return None;
        }
        Some(self.kappa[period_idx])
    }

    /// Normalize beta coefficients (sum to 1 constraint)
    pub fn normalize_beta(&mut self) -> Result<(), MortalityModelError> {
        let sum: f64 = self.beta[..self.n_ages].iter().sum();
        if sum.abs() < 1e-10 {
            return Err(MortalityModelError::NumericalInstability);
        }
        for i in 0..self.n_ages {
            self.beta[i] /= sum;
        }
        Ok(())
    }
}

/// Kalman Filter state for Lee-Carter model
pub struct KalmanFilterState {
    /// State vector (kappa and its drift)
    state: [f64; 2],
    /// State covariance matrix (2x2, row-major)
    covariance: [f64; 4],
    /// Process noise variance
    process_noise: f64,
    /// Measurement noise variance
    measurement_noise: f64,
    /// Parameter bounds to prevent divergence
    kappa_min: f64,
    kappa_max: f64,
}

impl KalmanFilterState {
    pub const fn new() -> Self {
        Self {
            state: [0.0; 2],
            covariance: [1.0, 0.0, 0.0, 1.0], // Identity
            process_noise: 0.001,
            measurement_noise: 0.01,
            kappa_min: -100.0,
            kappa_max: 100.0,
        }
    }

    /// Predict step (time update)
    #[inline]
    pub fn predict(&mut self) -> Result<(), MortalityModelError> {
        // State transition: kappa_t = kappa_{t-1} + drift + noise
        // Drift follows random walk with drift model
        let drift = self.state[1];
        
        // Update state
        self.state[0] += drift;
        
        // Update covariance: P = F * P * F' + Q
        // F = [[1, 1], [0, 1]] for random walk with drift
        let p00 = self.covariance[0] + 2.0 * self.covariance[1] + self.covariance[3] + self.process_noise;
        let p01 = self.covariance[1] + self.covariance[3];
        let p10 = p01;
        let p11 = self.covariance[3] + self.process_noise * 0.1; // Smaller noise for drift

        self.covariance[0] = p00.clamp(0.0, 1e6);
        self.covariance[1] = p01.clamp(-1e5, 1e5);
        self.covariance[2] = p10.clamp(-1e5, 1e5);
        self.covariance[3] = p11.clamp(0.0, 1e6);

        // Apply hard bounds to prevent divergence
        self.state[0] = self.state[0].clamp(self.kappa_min, self.kappa_max);
        
        if !self.state.iter().all(|s| s.is_finite()) {
            return Err(MortalityModelError::ParameterDivergence);
        }

        Ok(())
    }

    /// Update step (measurement update)
    #[inline]
    pub fn update(&mut self, measurement: f64) -> Result<(), MortalityModelError> {
        if !measurement.is_finite() {
            return Err(MortalityModelError::NumericalInstability);
        }

        // Observation matrix H = [1, 0]
        // Innovation: y = z - H * x
        let innovation = measurement - self.state[0];

        // Innovation covariance: S = H * P * H' + R
        let s = self.covariance[0] + self.measurement_noise;
        
        if s < 1e-10 {
            return Err(MortalityModelError::NonPositiveDefinite);
        }

        // Kalman gain: K = P * H' / S
        let k0 = self.covariance[0] / s;
        let k1 = self.covariance[2] / s;

        // Update state: x = x + K * y
        self.state[0] += k0 * innovation;
        self.state[1] += k1 * innovation;

        // Update covariance: P = (I - K * H) * P
        let ik0 = 1.0 - k0;
        let p00_new = ik0 * self.covariance[0];
        let p01_new = ik0 * self.covariance[1];
        let p10_new = self.covariance[2] - k1 * self.covariance[0];
        let p11_new = self.covariance[3] - k1 * self.covariance[1];

        self.covariance[0] = p00_new.max(1e-6).min(1e6);
        self.covariance[1] = p01_new.clamp(-1e5, 1e5);
        self.covariance[2] = p10_new.clamp(-1e5, 1e5);
        self.covariance[3] = p11_new.max(1e-6).min(1e6);

        // Apply bounds
        self.state[0] = self.state[0].clamp(self.kappa_min, self.kappa_max);
        self.state[1] = self.state[1].clamp(-10.0, 10.0);

        // Verify positive definiteness
        let det = self.covariance[0] * self.covariance[3] - self.covariance[1] * self.covariance[2];
        if det < 1e-10 {
            // Restore to identity if non-positive definite
            self.covariance = [1.0, 0.0, 0.0, 1.0];
        }

        if !self.state.iter().all(|s| s.is_finite()) {
            return Err(MortalityModelError::ParameterDivergence);
        }

        Ok(())
    }

    /// Adaptive process noise tuning based on innovation
    pub fn adapt_process_noise(&mut self, innovation: f64) {
        let innov_sq = innovation * innovation;
        
        // Increase process noise if innovations are large (model misspecification)
        if innov_sq > 4.0 * self.measurement_noise {
            self.process_noise = (self.process_noise * 1.1).min(0.1);
        } else if innov_sq < 0.25 * self.measurement_noise {
            self.process_noise = (self.process_noise * 0.95).max(1e-6);
        }
    }

    #[inline]
    pub fn get_kappa_estimate(&self) -> f64 {
        self.state[0]
    }

    #[inline]
    pub fn get_drift_estimate(&self) -> f64 {
        self.state[1]
    }
}

/// Lee-Carter model with Kalman filtering
pub struct LeeCarterKalmanModel {
    params: LeeCarterParams,
    kalman_state: KalmanFilterState,
    /// Current time period index
    current_period: usize,
}

impl LeeCarterKalmanModel {
    pub fn new() -> Self {
        Self {
            params: LeeCarterParams::new(),
            kalman_state: KalmanFilterState::new(),
            current_period: 0,
        }
    }

    /// Initialize model parameters from historical data
    pub fn initialize_from_data(
        &mut self,
        log_mortality_rates: &[Vec<f64>],
    ) -> Result<(), MortalityModelError> {
        if log_mortality_rates.is_empty() {
            return Err(MortalityModelError::DataMismatch);
        }

        let n_periods = log_mortality_rates.len();
        let n_ages = log_mortality_rates[0].len();

        if n_ages > MAX_AGE_GROUPS || n_periods > MAX_TIME_PERIODS {
            return Err(MortalityModelError::DataMismatch);
        }

        // Compute alpha (average log mortality by age)
        for age in 0..n_ages {
            let mut sum = 0.0;
            for period in 0..n_periods {
                let rate = log_mortality_rates[period][age];
                if !rate.is_finite() {
                    return Err(MortalityModelError::NumericalInstability);
                }
                sum += rate;
            }
            let alpha = sum / n_periods as f64;
            self.params.set_alpha(age, alpha)?;
        }

        // Initial SVD-like decomposition for beta and kappa
        // Simplified: use first principal component approximation
        for age in 0..n_ages {
            self.params.set_beta(age, 1.0 / n_ages as f64)?;
        }
        self.params.normalize_beta()?;

        // Initialize kappa from first period
        for age in 0..n_ages {
            let log_rate = log_mortality_rates[0][age];
            let alpha = self.params.alpha[age];
            let beta = self.params.beta[age];
            if beta.abs() > 1e-10 {
                let kappa_init = (log_rate - alpha) / beta;
                self.kalman_state.state[0] += kappa_init / n_ages as f64;
            }
        }

        self.params.n_ages = n_ages;
        self.params.n_periods = n_periods;
        self.current_period = 0;

        Ok(())
    }

    /// Process new mortality observation and update kappa
    pub fn process_observation(
        &mut self,
        log_mortality_rates: &[f64],
    ) -> Result<f64, MortalityModelError> {
        if log_mortality_rates.len() != self.params.n_ages {
            return Err(MortalityModelError::DataMismatch);
        }

        // Predict next kappa
        self.kalman_state.predict()?;

        // Compute observation for kappa update
        // Using weighted average across ages
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        for age in 0..self.params.n_ages {
            let log_rate = log_mortality_rates[age];
            let alpha = self.params.alpha[age];
            let beta = self.params.beta[age];

            if beta.abs() > 1e-10 {
                let kappa_obs = (log_rate - alpha) / beta;
                weighted_sum += kappa_obs * beta.abs();
                weight_total += beta.abs();
            }
        }

        if weight_total < 1e-10 {
            return Err(MortalityModelError::NumericalInstability);
        }

        let kappa_observation = weighted_sum / weight_total;

        // Update Kalman filter
        let innovation = kappa_observation - self.kalman_state.get_kappa_estimate();
        self.kalman_state.adapt_process_noise(innovation);
        self.kalman_state.update(kappa_observation)?;

        // Store updated kappa
        if self.current_period < MAX_TIME_PERIODS {
            self.params.set_kappa(self.current_period, self.kalman_state.get_kappa_estimate())?;
            self.current_period += 1;
        }

        Ok(self.kalman_state.get_kappa_estimate())
    }

    /// Forecast future mortality rates
    pub fn forecast(&self, horizon: usize) -> Result<Vec<[f64; MAX_AGE_GROUPS]>, MortalityModelError> {
        if horizon == 0 || horizon > 50 {
            return Err(MortalityModelError::DataMismatch);
        }

        let mut forecasts = Vec::with_capacity(horizon);
        let mut kappa_forecast = self.kalman_state.get_kappa_estimate();
        let drift = self.kalman_state.get_drift_estimate();

        for h in 0..horizon {
            kappa_forecast += drift * (h + 1) as f64;
            
            // Clamp to prevent numerical explosion
            kappa_forecast = kappa_forecast.clamp(-500.0, 500.0);

            let mut period_forecast = [0.0; MAX_AGE_GROUPS];
            for age in 0..self.params.n_ages {
                let log_rate = self.params.alpha[age] + self.params.beta[age] * kappa_forecast;
                period_forecast[age] = log_rate.exp().clamp(0.0, 1.0);
            }
            forecasts.push(period_forecast);
        }

        Ok(forecasts)
    }

    /// Get current kappa estimate
    pub fn current_kappa(&self) -> f64 {
        self.kalman_state.get_kappa_estimate()
    }

    /// Check for parameter stability
    pub fn check_stability(&self) -> bool {
        let kappa = self.kalman_state.get_kappa_estimate();
        let drift = self.kalman_state.get_drift_estimate();
        
        // Check for explosive behavior
        kappa.abs() < 100.0 && drift.abs() < 5.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_filter_basic() {
        let mut kf = KalmanFilterState::new();
        assert!(kf.predict().is_ok());
        assert!(kf.update(0.5).is_ok());
    }

    #[test]
    fn test_parameter_bounds() {
        let mut kf = KalmanFilterState::new();
        kf.state[0] = 200.0; // Exceeds bounds
        
        kf.predict().unwrap();
        
        // Should be clamped
        assert!(kf.state[0] <= kf.kappa_max);
    }

    #[test]
    fn test_lee_carter_initialization() {
        let mut model = LeeCarterKalmanModel::new();
        
        // Create synthetic data
        let data = vec![
            vec![-4.0, -3.5, -3.0],
            vec![-4.1, -3.6, -3.1],
            vec![-4.2, -3.7, -3.2],
        ];
        
        assert!(model.initialize_from_data(&data).is_ok());
    }
}
