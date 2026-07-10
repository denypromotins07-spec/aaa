//! Reservation Price Calculator for Avellaneda-Stoikov Market Making.
//! Computes r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
//! Zero-allocation, no unwrap/expect in hot paths.

use crate::pde::hjb_equation::AvellanedaStoikovParams;

/// Error types for reservation price calculations
#[derive(Debug, Clone, PartialEq)]
pub enum ReservationPriceError {
    InvalidParameters,
    NumericalOverflow,
    TimeExceeded,
}

/// Reservation price calculator with pre-computed constants
pub struct ReservationPriceCalculator {
    params: AvellanedaStoikovParams,
    /// Pre-computed: gamma * sigma^2
    gamma_sigma_sq: f64,
}

impl ReservationPriceCalculator {
    pub fn new(params: AvellanedaStoikovParams) -> Result<Self, ReservationPriceError> {
        if params.gamma <= 0.0 || params.sigma <= 0.0 {
            return Err(ReservationPriceError::InvalidParameters);
        }
        
        let gamma_sigma_sq = params.gamma * params.sigma * params.sigma;
        
        Ok(Self {
            params,
            gamma_sigma_sq,
        })
    }
    
    /// Calculate reservation price
    /// 
    /// # Arguments
    /// * `mid_price` - Current mid-market price (s)
    /// * `inventory` - Current inventory position (q)
    /// * `time_remaining` - Time to horizon (T - t)
    /// 
    /// Returns None if time_remaining is negative or exceeds horizon
    #[inline(always)]
    pub fn calculate(
        &self,
        mid_price: f64,
        inventory: i64,
        time_remaining: f64,
    ) -> Option<f64> {
        // Validate time
        if time_remaining < 0.0 || time_remaining > self.params.time_horizon {
            return None;
        }
        
        // r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
        let skew = inventory as f64 * self.gamma_sigma_sq * time_remaining;
        
        // Check for numerical overflow before subtraction
        let result = mid_price - skew;
        
        if result.is_nan() || result.is_infinite() {
            return None;
        }
        
        Some(result)
    }
    
    /// Calculate reservation price with dynamic risk aversion
    /// Uses adjusted gamma instead of base gamma
    #[inline(always)]
    pub fn calculate_with_gamma(
        mid_price: f64,
        inventory: i64,
        time_remaining: f64,
        gamma: f64,
        sigma: f64,
    ) -> Option<f64> {
        if gamma <= 0.0 || sigma <= 0.0 || time_remaining < 0.0 {
            return None;
        }
        
        let gamma_sigma_sq = gamma * sigma * sigma;
        let skew = inventory as f64 * gamma_sigma_sq * time_remaining;
        
        let result = mid_price - skew;
        
        if result.is_nan() || result.is_infinite() {
            return None;
        }
        
        Some(result)
    }
    
    /// Calculate optimal bid-ask spread around reservation price
    /// Spread = 2/gamma + adjustment for time remaining
    #[inline(always)]
    pub fn calculate_spread(&self, time_remaining: f64) -> Option<(f64, f64)> {
        if time_remaining < 0.0 || time_remaining > self.params.time_horizon {
            return None;
        }
        
        let tau_ratio = time_remaining / self.params.time_horizon;
        
        // Base spread: 2/gamma
        let base_spread = 2.0 / self.params.gamma;
        
        // Adjustment term from Avellaneda-Stoikov solution
        let kappa = self.params.kappa;
        let gamma = self.params.gamma;
        
        // Avoid division by zero
        if gamma * kappa < 1e-15 {
            return Some((base_spread / 2.0, base_spread / 2.0));
        }
        
        let adjustment = (2.0 / (gamma * kappa)) 
            * (1.0 - (-gamma * kappa * tau_ratio).exp()).ln().max(0.0);
        
        let half_spread = (base_spread + adjustment) / 2.0;
        
        Some((half_spread, half_spread))
    }
    
    /// Calculate full quote band (bid, ask) given inventory and mid-price
    #[inline(always)]
    pub fn calculate_quotes(
        &self,
        mid_price: f64,
        inventory: i64,
        time_remaining: f64,
    ) -> Option<(f64, f64)> {
        let reservation = self.calculate(mid_price, inventory, time_remaining)?;
        let (bid_half, ask_half) = self.calculate_spread(time_remaining)?;
        
        Some((reservation - bid_half, reservation + ask_half))
    }
    
    /// Get the inventory skew factor (how much to adjust per unit inventory)
    #[inline(always)]
    pub fn skew_factor(&self, time_remaining: f64) -> f64 {
        self.gamma_sigma_sq * time_remaining
    }
    
    /// Get parameters
    pub const fn params(&self) -> &AvellanedaStoikovParams {
        &self.params
    }
}

/// Builder for creating reservation price calculators with validation
pub struct ReservationPriceBuilder {
    gamma: Option<f64>,
    sigma: Option<f64>,
    lambda: Option<f64>,
    kappa: Option<f64>,
    time_horizon: Option<f64>,
    max_inventory: Option<i64>,
}

impl Default for ReservationPriceBuilder {
    fn default() -> Self {
        Self {
            gamma: Some(0.1),
            sigma: Some(0.02),
            lambda: Some(1.0),
            kappa: Some(0.5),
            time_horizon: Some(1.0),
            max_inventory: Some(100),
        }
    }
}

impl ReservationPriceBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }
    
    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = Some(sigma);
        self
    }
    
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.lambda = Some(lambda);
        self
    }
    
    pub fn kappa(mut self, kappa: f64) -> Self {
        self.kappa = Some(kappa);
        self
    }
    
    pub fn time_horizon(mut self, time_horizon: f64) -> Self {
        self.time_horizon = Some(time_horizon);
        self
    }
    
    pub fn max_inventory(mut self, max_inventory: i64) -> Self {
        self.max_inventory = Some(max_inventory);
        self
    }
    
    pub fn build(self) -> Result<ReservationPriceCalculator, ReservationPriceError> {
        let params = AvellanedaStoikovParams::new(
            self.gamma.ok_or(ReservationPriceError::InvalidParameters)?,
            self.sigma.ok_or(ReservationPriceError::InvalidParameters)?,
            self.lambda.ok_or(ReservationPriceError::InvalidParameters)?,
            self.kappa.ok_or(ReservationPriceError::InvalidParameters)?,
            self.time_horizon.ok_or(ReservationPriceError::InvalidParameters)?,
            self.max_inventory.ok_or(ReservationPriceError::InvalidParameters)?,
        )?;
        
        ReservationPriceCalculator::new(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_reservation_price_basic() {
        let params = AvellanedaStoikovParams::new(
            0.1, 0.02, 1.0, 0.5, 1.0, 10
        ).unwrap();
        let calc = ReservationPriceCalculator::new(params).unwrap();
        
        let mid_price = 100.0;
        let inventory = 5i64;
        let time_remaining = 0.5;
        
        let reservation = calc.calculate(mid_price, inventory, time_remaining).unwrap();
        
        // Positive inventory should give reservation price below mid
        assert!(reservation < mid_price);
        
        // Verify formula: r = s - q * gamma * sigma^2 * tau
        let expected_skew = 5.0 * 0.1 * 0.0004 * 0.5;
        let expected = mid_price - expected_skew;
        
        assert!((reservation - expected).abs() < 1e-10);
    }
    
    #[test]
    fn test_zero_inventory() {
        let params = AvellanedaStoikovParams::new(
            0.1, 0.02, 1.0, 0.5, 1.0, 10
        ).unwrap();
        let calc = ReservationPriceCalculator::new(params).unwrap();
        
        let reservation = calc.calculate(100.0, 0, 0.5).unwrap();
        
        // Zero inventory means no skew
        assert!((reservation - 100.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_invalid_time() {
        let params = AvellanedaStoikovParams::new(
            0.1, 0.02, 1.0, 0.5, 1.0, 10
        ).unwrap();
        let calc = ReservationPriceCalculator::new(params).unwrap();
        
        // Negative time
        assert!(calc.calculate(100.0, 5, -0.1).is_none());
        
        // Time exceeding horizon
        assert!(calc.calculate(100.0, 5, 1.5).is_none());
    }
    
    #[test]
    fn test_negative_inventory() {
        let params = AvellanedaStoikovParams::new(
            0.1, 0.02, 1.0, 0.5, 1.0, 10
        ).unwrap();
        let calc = ReservationPriceCalculator::new(params).unwrap();
        
        let reservation = calc.calculate(100.0, -5, 0.5).unwrap();
        
        // Negative inventory should give reservation price above mid
        assert!(reservation > 100.0);
    }
}
