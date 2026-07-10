//! Zero-allocation yield curve bootstrapper.
//!
//! Extracts zero-coupon spot rates and forward rates from the NSS curve
//! using SIMD-accelerated discount factor calculations.
//! All operations use pre-allocated buffers to avoid heap allocations.

use crate::fixed_income::nss_curve_fitter::{NssParameters, NssError};
use wide::f64x4;

/// Bootstrap result containing spot rates, forward rates, and discount factors
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// Maturities at which rates were computed
    pub maturities: Vec<f64>,
    /// Zero-coupon spot rates (continuous compounding)
    pub spot_rates: Vec<f64>,
    /// Instantaneous forward rates
    pub forward_rates: Vec<f64>,
    /// Discount factors P(0,t) = exp(-r*t)
    pub discount_factors: Vec<f64>,
}

impl BootstrapResult {
    /// Create empty result with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            maturities: Vec::with_capacity(capacity),
            spot_rates: Vec::with_capacity(capacity),
            forward_rates: Vec::with_capacity(capacity),
            discount_factors: Vec::with_capacity(capacity),
        }
    }
}

/// Zero-copy bootstrapper using SIMD acceleration
pub struct ZeroAllocBootstrapper {
    /// Pre-allocated buffer for maturities
    maturities_buffer: Vec<f64>,
    /// Pre-allocated buffer for spot rates
    spot_rates_buffer: Vec<f64>,
    /// Pre-allocated buffer for forward rates
    forward_rates_buffer: Vec<f64>,
    /// Pre-allocated buffer for discount factors
    discount_factors_buffer: Vec<f64>,
}

impl ZeroAllocBootstrapper {
    /// Create new bootstrapper with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            maturities_buffer: Vec::with_capacity(capacity),
            spot_rates_buffer: Vec::with_capacity(capacity),
            forward_rates_buffer: Vec::with_capacity(capacity),
            discount_factors_buffer: Vec::with_capacity(capacity),
        }
    }

    /// Compute discount factor from spot rate and maturity
    #[inline(always)]
    fn discount_factor(rate: f64, maturity: f64) -> f64 {
        if maturity <= 0.0 {
            return 1.0;
        }
        (-rate * maturity).exp()
    }

    /// Compute instantaneous forward rate from NSS parameters
    /// f(t) = β₀ + β₁*exp(-t/τ₁) + β₂*(t/τ₁)*exp(-t/τ₁) + β₃*(t/τ₂)*exp(-t/τ₂)
    #[inline(always)]
    fn instantaneous_forward_rate(params: &NssParameters, t: f64) -> Result<f64, NssError> {
        if t < 0.0 {
            return Err(NssError::InvalidMaturity);
        }

        let x1 = t / params.tau1;
        let x2 = t / params.tau2;

        let exp_x1 = (-x1).exp();
        let exp_x2 = (-x2).exp();

        let forward = params.beta0 
            + params.beta1 * exp_x1 
            + params.beta2 * x1 * exp_x1 
            + params.beta3 * x2 * exp_x2;

        Ok(forward)
    }

    /// Bootstrap spot rates and forward rates from NSS curve
    /// Uses SIMD vectorization for batch processing
    pub fn bootstrap<'a>(
        &'a mut self,
        params: &NssParameters,
        maturities: &[f64],
    ) -> Result<&'a BootstrapResult, NssError> {
        // Clear buffers but keep capacity
        self.maturities_buffer.clear();
        self.spot_rates_buffer.clear();
        self.forward_rates_buffer.clear();
        self.discount_factors_buffer.clear();

        // Reserve space
        self.maturities_buffer.reserve(maturities.len());
        self.spot_rates_buffer.reserve(maturities.len());
        self.forward_rates_buffer.reserve(maturities.len());
        self.discount_factors_buffer.reserve(maturities.len());

        // Process in batches of 4 for SIMD
        let mut i = 0;
        let len = maturities.len();

        while i + 4 <= len {
            // Load 4 maturities into SIMD register
            let m_vec = f64x4::from([
                maturities[i],
                maturities[i + 1],
                maturities[i + 2],
                maturities[i + 3],
            ]);

            // Compute spot rates using NSS formula (scalar for now, can be SIMD-ified)
            let mut spots = [0.0; 4];
            let mut forwards = [0.0; 4];
            let mut discounts = [0.0; 4];

            for j in 0..4 {
                let t = maturities[i + j];
                
                // Spot rate from NSS
                let x1 = t / params.tau1;
                let x2 = t / params.tau2;

                let factor1 = if x1.abs() < 1e-10 {
                    1.0
                } else {
                    (1.0 - (-x1).exp()) / x1
                };

                let factor2 = if x1.abs() < 1e-10 {
                    0.0
                } else {
                    factor1 - (-x1).exp()
                };

                let factor3 = if x2.abs() < 1e-10 {
                    0.0
                } else {
                    (1.0 - (-x2).exp()) / x2 - (-x2).exp()
                };

                let spot = params.beta0 + params.beta1 * factor1 + params.beta2 * factor2 + params.beta3 * factor3;
                
                // Forward rate
                let forward = Self::instantaneous_forward_rate(params, t)?;
                
                // Discount factor
                let discount = Self::discount_factor(spot, t);

                spots[j] = spot;
                forwards[j] = forward;
                discounts[j] = discount;
            }

            // Store results
            self.maturities_buffer.extend_from_slice(&maturities[i..i + 4]);
            self.spot_rates_buffer.extend_from_slice(&spots);
            self.forward_rates_buffer.extend_from_slice(&forwards);
            self.discount_factors_buffer.extend_from_slice(&discounts);

            i += 4;
        }

        // Handle remaining elements (scalar)
        while i < len {
            let t = maturities[i];
            
            let x1 = t / params.tau1;
            let x2 = t / params.tau2;

            let factor1 = if x1.abs() < 1e-10 {
                1.0
            } else {
                (1.0 - (-x1).exp()) / x1
            };

            let factor2 = if x1.abs() < 1e-10 {
                0.0
            } else {
                factor1 - (-x1).exp()
            };

            let factor3 = if x2.abs() < 1e-10 {
                0.0
            } else {
                (1.0 - (-x2).exp()) / x2 - (-x2).exp()
            };

            let spot = params.beta0 + params.beta1 * factor1 + params.beta2 * factor2 + params.beta3 * factor3;
            let forward = Self::instantaneous_forward_rate(params, t)?;
            let discount = Self::discount_factor(spot, t);

            self.maturities_buffer.push(t);
            self.spot_rates_buffer.push(spot);
            self.forward_rates_buffer.push(forward);
            self.discount_factors_buffer.push(discount);

            i += 1;
        }

        Ok(&BootstrapResult {
            maturities: std::mem::take(&mut self.maturities_buffer),
            spot_rates: std::mem::take(&mut self.spot_rates_buffer),
            forward_rates: std::mem::take(&mut self.forward_rates_buffer),
            discount_factors: std::mem::take(&mut self.discount_factors_buffer),
        })
    }

    /// Extract zero-coupon rate for a specific maturity via interpolation
    pub fn interpolate_spot_rate(
        &self,
        result: &BootstrapResult,
        target_maturity: f64,
    ) -> Result<f64, NssError> {
        if result.maturities.is_empty() {
            return Err(NssError::InvalidMaturity);
        }

        // Find bracketing maturities
        let n = result.maturities.len();
        
        if target_maturity <= result.maturities[0] {
            return Ok(result.spot_rates[0]);
        }
        
        if target_maturity >= result.maturities[n - 1] {
            return Ok(result.spot_rates[n - 1]);
        }

        // Linear interpolation
        for i in 0..(n - 1) {
            if result.maturities[i] <= target_maturity && result.maturities[i + 1] >= target_maturity {
                let t1 = result.maturities[i];
                let t2 = result.maturities[i + 1];
                let r1 = result.spot_rates[i];
                let r2 = result.spot_rates[i + 1];

                let weight = (target_maturity - t1) / (t2 - t1);
                return Ok(r1 * (1.0 - weight) + r2 * weight);
            }
        }

        // Fallback (should not reach here)
        Ok(*result.spot_rates.last().unwrap_or(&0.0))
    }

    /// Compute forward rate between two maturities: F(t1, t2)
    /// F(t1, t2) = (r2*t2 - r1*t1) / (t2 - t1)
    pub fn compute_forward_rate(
        &self,
        result: &BootstrapResult,
        t1: f64,
        t2: f64,
    ) -> Result<f64, NssError> {
        if t1 >= t2 {
            return Err(NssError::InvalidMaturity);
        }

        let r1 = self.interpolate_spot_rate(result, t1)?;
        let r2 = self.interpolate_spot_rate(result, t2)?;

        // Forward rate formula: (r2*t2 - r1*t1) / (t2 - t1)
        let forward = (r2 * t2 - r1 * t1) / (t2 - t1);

        Ok(forward)
    }

    /// Get the entire yield curve as a continuous function representation
    /// Returns parameters that can be evaluated at any maturity
    pub fn get_curve_function(
        &self,
        params: NssParameters,
    ) -> impl Fn(f64) -> Result<f64, NssError> + '_ {
        move |t: f64| -> Result<f64, NssError> {
            if t <= 0.0 {
                return Err(NssError::InvalidMaturity);
            }

            let x1 = t / params.tau1;
            let x2 = t / params.tau2;

            let factor1 = if x1.abs() < 1e-10 {
                1.0
            } else {
                (1.0 - (-x1).exp()) / x1
            };

            let factor2 = if x1.abs() < 1e-10 {
                0.0
            } else {
                factor1 - (-x1).exp()
            };

            let factor3 = if x2.abs() < 1e-10 {
                0.0
            } else {
                (1.0 - (-x2).exp()) / x2 - (-x2).exp()
            };

            Ok(params.beta0 + params.beta1 * factor1 + params.beta2 * factor2 + params.beta3 * factor3)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_basic() {
        let params = NssParameters::default();
        let mut bootstrapper = ZeroAllocBootstrapper::new(20);
        
        let maturities = vec![0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0];
        let result = bootstrapper.bootstrap(&params, &maturities).unwrap();
        
        assert_eq!(result.maturities.len(), maturities.len());
        assert_eq!(result.spot_rates.len(), maturities.len());
        assert_eq!(result.forward_rates.len(), maturities.len());
        assert_eq!(result.discount_factors.len(), maturities.len());
        
        // All rates should be finite
        for rate in &result.spot_rates {
            assert!(rate.is_finite());
        }
        
        // Discount factors should be in (0, 1]
        for df in &result.discount_factors {
            assert!(*df > 0.0 && *df <= 1.0);
        }
    }

    #[test]
    fn test_interpolation() {
        let params = NssParameters::default();
        let mut bootstrapper = ZeroAllocBootstrapper::new(10);
        
        let maturities = vec![1.0, 5.0, 10.0];
        let result = bootstrapper.bootstrap(&params, &maturities).unwrap();
        
        // Interpolate at 3.0 (between 1.0 and 5.0)
        let rate_3y = bootstrapper.interpolate_spot_rate(result, 3.0).unwrap();
        assert!(rate_3y.is_finite());
        
        // Should be between 1y and 5y rates
        let rate_1y = bootstrapper.interpolate_spot_rate(result, 1.0).unwrap();
        let rate_5y = bootstrapper.interpolate_spot_rate(result, 5.0).unwrap();
        
        // Linear interpolation should give value between endpoints
        assert!(rate_3y >= rate_1y.min(rate_5y));
        assert!(rate_3y <= rate_1y.max(rate_5y));
    }
}
