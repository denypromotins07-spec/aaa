//! Affine Term Structure Model for Longevity Bond Pricing
//! 
//! Prices longevity derivatives using affine term structure models where
//! bond prices depend on stochastic mortality intensity and interest rates.

use crate::mortality::lee_carter_kalman::{LeeCarterKalmanModel, MortalityModelError};

/// Maximum number of factors in affine model
pub const MAX_AFFINE_FACTORS: usize = 5;

/// Error types for longevity derivative pricing
#[derive(Debug, Clone, PartialEq)]
pub enum LongevityDerivativeError {
    InvalidMaturity,
    NegativeIntensity,
    NumericalInstability,
    CalibrationFailure,
    CorrelationError,
}

impl core::fmt::Display for LongevityDerivativeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidMaturity => write!(f, "Invalid maturity"),
            Self::NegativeIntensity => write!(f, "Negative mortality intensity"),
            Self::NumericalInstability => write!(f, "Numerical instability"),
            Self::CalibrationFailure => write!(f, "Calibration failure"),
            Self::CorrelationError => write!(f, "Correlation matrix error"),
        }
    }
}

/// State vector for affine model (mortality intensity + covariates)
#[repr(C)]
pub struct AffineState {
    /// Mortality intensity (lambda)
    lambda: f64,
    /// Interest rate (r)
    r: f64,
    /// Additional factors (e.g., economic growth, medical progress)
    factors: [f64; MAX_AFFINE_FACTORS - 2],
    /// Number of active factors
    n_factors: usize,
}

impl AffineState {
    #[inline]
    pub const fn new() -> Self {
        Self {
            lambda: 0.01, // Initial mortality intensity
            r: 0.03,      // Initial interest rate
            factors: [0.0; MAX_AFFINE_FACTORS - 2],
            n_factors: 2,
        }
    }

    #[inline]
    pub fn set_lambda(&mut self, value: f64) -> Result<(), LongevityDerivativeError> {
        if value < 0.0 {
            return Err(LongevityDerivativeError::NegativeIntensity);
        }
        if !value.is_finite() {
            return Err(LongevityDerivativeError::NumericalInstability);
        }
        self.lambda = value;
        Ok(())
    }

    #[inline]
    pub fn set_r(&mut self, value: f64) -> Result<(), LongevityDerivativeError> {
        if !value.is_finite() {
            return Err(LongevityDerivativeError::NumericalInstability);
        }
        self.r = value;
        Ok(())
    }

    #[inline]
    pub fn get_lambda(&self) -> f64 {
        self.lambda
    }

    #[inline]
    pub fn get_r(&self) -> f64 {
        self.r
    }
}

/// Affine model coefficients (A, B matrices)
pub struct AffineCoefficients {
    /// Mean reversion speeds (kappa)
    kappa: [f64; MAX_AFFINE_FACTORS],
    /// Long-term means (theta)
    theta: [f64; MAX_AFFINE_FACTORS],
    /// Volatilities (sigma)
    sigma: [f64; MAX_AFFINE_FACTORS],
    /// Correlation matrix (upper triangular, row-major)
    correlation: [f64; MAX_AFFINE_FACTORS * (MAX_AFFINE_FACTORS - 1) / 2],
    /// Loading coefficients for price formula
    B: [f64; MAX_AFFINE_FACTORS],
    /// Constant term for price formula
    A: f64,
}

impl AffineCoefficients {
    pub const fn new() -> Self {
        Self {
            kappa: [0.1; MAX_AFFINE_FACTORS],
            theta: [0.0; MAX_AFFINE_FACTORS],
            sigma: [0.01; MAX_AFFINE_FACTORS],
            correlation: [0.0; MAX_AFFINE_FACTORS * (MAX_AFFINE_FACTORS - 1) / 2],
            B: [0.0; MAX_AFFINE_FACTORS],
            A: 0.0,
        }
    }

    /// Set correlation between factors i and j
    pub fn set_correlation(&mut self, i: usize, j: usize, rho: f64) -> Result<(), LongevityDerivativeError> {
        if i >= MAX_AFFINE_FACTORS || j >= MAX_AFFINE_FACTORS || i == j {
            return Err(LongevityDerivativeError::CorrelationError);
        }
        if rho.abs() > 1.0 {
            return Err(LongevityDerivativeError::CorrelationError);
        }

        let idx = if i < j {
            i * (MAX_AFFINE_FACTORS - 1) - i * (i + 1) / 2 + (j - i - 1)
        } else {
            j * (MAX_AFFINE_FACTORS - 1) - j * (j + 1) / 2 + (i - j - 1)
        };

        self.correlation[idx] = rho;
        Ok(())
    }

    /// Get correlation between factors i and j
    pub fn get_correlation(&self, i: usize, j: usize) -> Option<f64> {
        if i >= MAX_AFFINE_FACTORS || j >= MAX_AFFINE_FACTORS || i == j {
            return None;
        }

        let idx = if i < j {
            i * (MAX_AFFINE_FACTORS - 1) - i * (i + 1) / 2 + (j - i - 1)
        } else {
            j * (MAX_AFFINE_FACTORS - 1) - j * (j + 1) / 2 + (i - j - 1)
        };

        Some(self.correlation[idx])
    }

    /// Compute B coefficients for given maturity
    pub fn compute_b_coefficients(&mut self, tau: f64) -> Result<(), LongevityDerivativeError> {
        if tau <= 0.0 || tau > 50.0 {
            return Err(LongevityDerivativeError::InvalidMaturity);
        }

        for i in 0..self.n_factors() {
            let k = self.kappa[i];
            if k.abs() < 1e-10 {
                // Zero mean reversion case
                self.B[i] = -tau;
            } else {
                self.B[i] = -(1.0 - (-k * tau).exp()) / k;
            }

            if !self.B[i].is_finite() {
                return Err(LongevityDerivativeError::NumericalInstability);
            }
        }

        Ok(())
    }

    /// Compute A coefficient for given maturity
    pub fn compute_a_coefficient(&mut self, tau: f64) -> Result<(), LongevityDerivativeError> {
        if tau <= 0.0 || tau > 50.0 {
            return Err(LongevityDerivativeError::InvalidMaturity);
        }

        let mut a = 0.0;

        for i in 0..self.n_factors() {
            let k = self.kappa[i];
            let theta = self.theta[i];
            let sigma = self.sigma[i];
            let b = self.B[i];

            if k.abs() < 1e-10 {
                a += theta * b + 0.5 * sigma * sigma * tau * tau / 3.0;
            } else {
                let kb = k * b;
                a += theta * (b - tau) + 0.5 * sigma * sigma / (k * k) * (kb - 2.0 * b - k * tau);
            }

            if !a.is_finite() {
                return Err(LongevityDerivativeError::NumericalInstability);
            }
        }

        self.A = a;
        Ok(())
    }

    #[inline]
    pub fn n_factors(&self) -> usize {
        self.n_factors
    }

    pub fn set_n_factors(&mut self, n: usize) -> Result<(), LongevityDerivativeError> {
        if n < 2 || n > MAX_AFFINE_FACTORS {
            return Err(LongevityDerivativeError::CalibrationFailure);
        }
        self.n_factors = n;
        Ok(())
    }
}

/// Longevity bond price result
#[derive(Debug, Clone)]
pub struct LongevityBondPrice {
    /// Present value of bond
    pub price: f64,
    /// Duration (interest rate sensitivity)
    pub duration: f64,
    /// Convexity
    pub convexity: f64,
    /// Mortality delta (sensitivity to lambda)
    pub mortality_delta: f64,
    /// Survival probability to maturity
    pub survival_prob: f64,
}

impl LongevityBondPrice {
    pub const fn new() -> Self {
        Self {
            price: 0.0,
            duration: 0.0,
            convexity: 0.0,
            mortality_delta: 0.0,
            survival_prob: 1.0,
        }
    }
}

/// Affine longevity bond pricer
pub struct AffineLongevityBondPricer {
    state: AffineState,
    coeffs: AffineCoefficients,
    /// Correlation between interest rates and mortality
    r_lambda_corr: f64,
}

impl AffineLongevityBondPricer {
    pub fn new() -> Self {
        let mut coeffs = AffineCoefficients::new();
        coeffs.theta[0] = 0.01; // Long-term mortality mean
        coeffs.theta[1] = 0.03; // Long-term rate mean
        
        Self {
            state: AffineState::new(),
            coeffs,
            r_lambda_corr: 0.3, // Positive correlation during pandemics
        }
    }

    /// Initialize from Lee-Carter model
    pub fn initialize_from_lee_carter(
        &mut self,
        lc_model: &LeeCarterKalmanModel,
    ) -> Result<(), LongevityDerivativeError> {
        let kappa = lc_model.current_kappa();
        
        // Map Lee-Carter kappa to mortality intensity
        let lambda = kappa.abs().exp() * 0.001;
        self.state.set_lambda(lambda)?;
        
        Ok(())
    }

    /// Price longevity bond with maturity T
    pub fn price_bond(&mut self, maturity_years: f64) -> Result<LongevityBondPrice, LongevityDerivativeError> {
        if maturity_years <= 0.0 || maturity_years > 50.0 {
            return Err(LongevityDerivativeError::InvalidMaturity);
        }

        // Compute affine coefficients
        self.coeffs.compute_b_coefficients(maturity_years)?;
        self.coeffs.compute_a_coefficient(maturity_years)?;

        // Bond price: P = exp(A + B' * X)
        let mut exponent = self.coeffs.A;
        exponent += self.coeffs.B[0] * self.state.lambda;
        exponent += self.coeffs.B[1] * self.state.r;

        // Add correlation adjustment
        let corr_adjustment = self.r_lambda_corr * self.coeffs.sigma[0] * self.coeffs.sigma[1] 
            * maturity_years * maturity_years * 0.5;
        exponent -= corr_adjustment;

        let price = exponent.exp();

        if !price.is_finite() || price < 0.0 {
            return Err(LongevityDerivativeError::NumericalInstability);
        }

        // Compute Greeks
        let duration = -self.coeffs.B[1] * price;
        let mortality_delta = -self.coeffs.B[0] * price;
        let convexity = self.coeffs.B[1] * self.coeffs.B[1] * price;

        // Survival probability
        let survival_prob = (-self.state.lambda * maturity_years).exp();

        Ok(LongevityBondPrice {
            price,
            duration,
            convexity,
            mortality_delta,
            survival_prob,
        })
    }

    /// Price q-forward (forward contract on survival probability)
    pub fn price_q_forward(
        &self,
        maturity_years: f64,
        forward_rate: f64,
    ) -> Result<f64, LongevityDerivativeError> {
        if maturity_years <= 0.0 {
            return Err(LongevityDerivativeError::InvalidMaturity);
        }

        // Expected survival probability under risk-neutral measure
        let expected_survival = (-self.state.lambda * maturity_years).exp();
        
        // Forward value = discount_factor * (expected_survival - forward_rate)
        let discount_factor = (-self.state.r * maturity_years).exp();
        let value = discount_factor * (expected_survival - forward_rate);

        if !value.is_finite() {
            return Err(LongevityDerivativeError::NumericalInstability);
        }

        Ok(value)
    }

    /// Update mortality intensity from new data
    pub fn update_mortality_intensity(&mut self, new_lambda: f64) -> Result<(), LongevityDerivativeError> {
        self.state.set_lambda(new_lambda)
    }

    /// Set correlation between rates and mortality
    pub fn set_rate_mortality_correlation(&mut self, rho: f64) -> Result<(), LongevityDerivativeError> {
        if rho.abs() > 1.0 {
            return Err(LongevityDerivativeError::CorrelationError);
        }
        self.r_lambda_corr = rho;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_state() {
        let mut state = AffineState::new();
        assert!(state.set_lambda(0.02).is_ok());
        assert!(state.set_lambda(-0.01).is_err());
    }

    #[test]
    fn test_b_coefficients() {
        let mut coeffs = AffineCoefficients::new();
        assert!(coeffs.compute_b_coefficients(10.0).is_ok());
    }

    #[test]
    fn test_bond_pricing() {
        let mut pricer = AffineLongevityBondPricer::new();
        let result = pricer.price_bond(10.0);
        assert!(result.is_ok());
        
        let price = result.unwrap();
        assert!(price.price > 0.0 && price.price < 1.0);
    }
}
