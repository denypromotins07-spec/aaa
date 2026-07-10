//! Analytical Greeks calculation for options
//! 
//! Computes first and second-order sensitivities using closed-form formulas.
//! All calculations use fast math approximations for HFT performance.

use crate::math::fast_cdf_erf::{fast_cdf, fast_cdf_complement, fast_exp, fast_sqrt};
use crate::pricing::black_scholes_fast::{BSParams, OptionType};

/// Numerical stability constants
const EPSILON: f64 = 1e-12;
const MIN_TIME: f64 = 1.0 / (365.0 * 24.0 * 3600.0);

/// First-order Greeks
#[derive(Debug, Clone, Copy)]
pub struct FirstOrderGreeks {
    /// Delta: ∂V/∂S (sensitivity to spot)
    pub delta: f64,
    /// Vega: ∂V/∂σ (sensitivity to vol) - per 1% change
    pub vega: f64,
    /// Theta: ∂V/∂t (sensitivity to time decay) - per day
    pub theta: f64,
    /// Rho: ∂V/∂r (sensitivity to rates) - per 1% change
    pub rho: f64,
}

/// Second-order Greeks
#[derive(Debug, Clone, Copy)]
pub struct SecondOrderGreeks {
    /// Gamma: ∂²V/∂S² (curvature wrt spot)
    pub gamma: f64,
    /// Vanna: ∂²V/∂S∂σ (delta sensitivity to vol, or vega sensitivity to spot)
    pub vanna: f64,
    /// Volga (Vomma): ∂²V/∂σ² (vega sensitivity to vol)
    pub volga: f64,
    /// Charm: ∂²V/∂S∂t (delta decay)
    pub charm: f64,
    /// Veta: ∂²V/∂σ∂t (vega decay)
    pub veta: f64,
}

/// Complete Greeks package
#[derive(Debug, Clone, Copy)]
pub struct FullGreeks {
    pub first: FirstOrderGreeks,
    pub second: SecondOrderGreeks,
}

impl FullGreeks {
    #[inline]
    pub const fn new(first: FirstOrderGreeks, second: SecondOrderGreeks) -> Self {
        Self { first, second }
    }
}

/// Standard normal PDF: φ(x) = e^(-x²/2) / √(2π)
#[inline(always)]
fn norm_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.3989422804014326779399460599343818684758;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Calculate all Greeks for a European option
/// 
/// # Arguments
/// * `params` - Option parameters
/// * `option_type` - Call or Put
/// 
/// # Returns
/// * `FullGreeks` containing all first and second-order sensitivities
/// 
/// # Safety
/// - Handles zero time to expiry gracefully
/// - No heap allocations
#[inline(always)]
pub fn calculate_greeks(params: &BSParams, option_type: OptionType) -> FullGreeks {
    // Clamp time for numerical stability
    let t = params.time_to_expiry.max(MIN_TIME);
    let vol = params.volatility.max(EPSILON);
    
    // Precompute common terms
    let sqrt_t = fast_sqrt(t);
    let vol_sqrt_t = vol * sqrt_t;
    
    // Forward and discount
    let drift = params.risk_free_rate - params.dividend_yield;
    let forward = params.spot * fast_exp(drift * t);
    let disc = fast_exp(-params.risk_free_rate * t);
    let disc_div = fast_exp(-params.dividend_yield * t);
    
    // d1 and d2
    let ln_moneyness = (forward / params.strike).ln();
    let d1 = (ln_moneyness + 0.5 * vol * vol * t) / vol_sqrt_t;
    let d2 = d1 - vol_sqrt_t;
    
    // Normal PDF at d1 and d2
    let pdf_d1 = norm_pdf(d1);
    let pdf_d2 = norm_pdf(d2);
    
    // CDF values
    let cdf_d1 = fast_cdf(d1);
    let cdf_d2 = fast_cdf(d2);
    let cdf_neg_d1 = fast_cdf_complement(d1);
    let cdf_neg_d2 = fast_cdf_complement(d2);
    
    // === FIRST ORDER GREEKS ===
    
    let (delta, vega, theta, rho) = match option_type {
        OptionType::Call => {
            // Delta = e^(-qT) * Φ(d1)
            let delta = disc_div * cdf_d1;
            
            // Vega = S * e^(-qT) * φ(d1) * √T
            let vega = params.spot * disc_div * pdf_d1 * sqrt_t / 100.0; // Per 1%
            
            // Theta = -S * e^(-qT) * φ(d1) * σ / (2√T) 
            //         - r * K * e^(-rT) * Φ(d2)
            let term1 = -params.spot * disc_div * pdf_d1 * vol / (2.0 * sqrt_t);
            let term2 = -params.risk_free_rate * params.strike * disc * cdf_d2;
            let theta = (term1 + term2) / 365.0; // Per day
            
            // Rho = K * T * e^(-rT) * Φ(d2)
            let rho = params.strike * t * disc * cdf_d2 / 100.0; // Per 1%
            
            (delta, vega, theta, rho)
        }
        OptionType::Put => {
            // Delta = e^(-qT) * (Φ(d1) - 1) = -e^(-qT) * Φ(-d1)
            let delta = -disc_div * cdf_neg_d1;
            
            // Vega is same as call
            let vega = params.spot * disc_div * pdf_d1 * sqrt_t / 100.0;
            
            // Theta = -S * e^(-qT) * φ(d1) * σ / (2√T)
            //         + r * K * e^(-rT) * Φ(-d2)
            let term1 = -params.spot * disc_div * pdf_d1 * vol / (2.0 * sqrt_t);
            let term2 = params.risk_free_rate * params.strike * disc * cdf_neg_d2;
            let theta = (term1 + term2) / 365.0;
            
            // Rho = -K * T * e^(-rT) * Φ(-d2)
            let rho = -params.strike * t * disc * cdf_neg_d2 / 100.0;
            
            (delta, vega, theta, rho)
        }
        OptionType::DigitalCall | OptionType::DigitalPut => {
            // Digital options have different Greeks
            // Delta = e^(-rT) * φ(d2) / (S * σ * √T)
            let delta = disc * pdf_d2 / (params.spot * vol_sqrt_t);
            
            // Vega for digital
            let vega = disc * pdf_d2 * d1 * sqrt_t / 100.0;
            
            // Simplified theta for digital
            let theta = 0.0;
            
            // Simplified rho for digital
            let rho = 0.0;
            
            (delta, vega, theta, rho)
        }
    };
    
    // === SECOND ORDER GREEKS ===
    
    // Gamma = e^(-qT) * φ(d1) / (S * σ * √T)
    let gamma = disc_div * pdf_d1 / (params.spot * vol_sqrt_t);
    
    // Vanna = ∂Delta/∂σ = -φ(d1) * d2 * e^(-qT) / σ
    let vanna = -pdf_d1 * d2 * disc_div / vol;
    
    // Volga (Vomma) = ∂Vega/∂σ = Vega * d1 * d2 / σ
    let volga = (vega * 100.0) * d1 * d2 / vol / 100.0; // Per 1%
    
    // Charm = ∂Delta/∂t (delta decay)
    let q_plus_term = params.dividend_yield + (2.0 * drift - vol * vol) / (2.0 * vol_sqrt_t) * d2;
    let charm = match option_type {
        OptionType::Call => {
            disc_div * (pdf_d1 * q_plus_term - params.dividend_yield * cdf_d1)
        }
        OptionType::Put => {
            -disc_div * (pdf_d1 * q_plus_term - params.dividend_yield * cdf_neg_d1)
        }
        _ => 0.0,
    };
    
    // Veta = ∂Vega/∂t (vega decay)
    let veta = match option_type {
        OptionType::Call | OptionType::Put => {
            params.spot * disc_div * pdf_d1 * d1 * (drift / vol - 0.5 * vol - drift * d1 / vol_sqrt_t)
        }
        _ => 0.0,
    };
    
    let first = FirstOrderGreeks { delta, vega, theta, rho };
    let second = SecondOrderGreeks { gamma, vanna, volga, charm, veta };
    
    FullGreeks::new(first, second)
}

/// Calculate Delta only (optimized hot path)
#[inline(always)]
pub fn calculate_delta(params: &BSParams, option_type: OptionType) -> f64 {
    let t = params.time_to_expiry.max(MIN_TIME);
    let vol = params.volatility.max(EPSILON);
    
    let sqrt_t = fast_sqrt(t);
    let drift = params.risk_free_rate - params.dividend_yield;
    let forward = params.spot * fast_exp(drift * t);
    let disc_div = fast_exp(-params.dividend_yield * t);
    
    let ln_moneyness = (forward / params.strike).ln();
    let d1 = (ln_moneyness + 0.5 * vol * vol * t) / (vol * sqrt_t);
    
    match option_type {
        OptionType::Call => disc_div * fast_cdf(d1),
        OptionType::Put => -disc_div * fast_cdf_complement(d1),
        _ => 0.0,
    }
}

/// Calculate Gamma only (optimized hot path)
#[inline(always)]
pub fn calculate_gamma(params: &BSParams) -> f64 {
    let t = params.time_to_expiry.max(MIN_TIME);
    let vol = params.volatility.max(EPSILON);
    
    let sqrt_t = fast_sqrt(t);
    let vol_sqrt_t = vol * sqrt_t;
    let drift = params.risk_free_rate - params.dividend_yield;
    let forward = params.spot * fast_exp(drift * t);
    let disc_div = fast_exp(-params.dividend_yield * t);
    
    let ln_moneyness = (forward / params.strike).ln();
    let d1 = (ln_moneyness + 0.5 * vol * vol * t) / vol_sqrt_t;
    
    disc_div * norm_pdf(d1) / (params.spot * vol_sqrt_t)
}

/// Batch calculate Greeks for multiple options
/// Zero allocation - uses pre-allocated output buffers
pub fn batch_calculate_greeks(
    params: &[BSParams],
    option_types: &[OptionType],
    delta_out: &mut [f64],
    gamma_out: &mut [f64],
    vega_out: &mut [f64],
    theta_out: &mut [f64],
) -> usize {
    let len = params.len().min(option_types.len())
        .min(delta_out.len()).min(gamma_out.len())
        .min(vega_out.len()).min(theta_out.len());
    
    for i in 0..len {
        let greeks = calculate_greeks(&params[i], option_types[i]);
        delta_out[i] = greeks.first.delta;
        gamma_out[i] = greeks.second.gamma;
        vega_out[i] = greeks.first.vega;
        theta_out[i] = greeks.first.theta;
    }
    
    len
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_call_delta_range() {
        let params = BSParams {
            spot: 100.0,
            strike: 100.0,
            time_to_expiry: 0.25,
            risk_free_rate: 0.05,
            volatility: 0.2,
            dividend_yield: 0.0,
        };
        
        let greeks = calculate_greeks(&params, OptionType::Call);
        
        // ATM call delta should be ~0.55
        assert!(greeks.first.delta > 0.5 && greeks.first.delta < 0.6,
            "ATM call delta out of range: {}", greeks.first.delta);
    }
    
    #[test]
    fn test_put_delta_negative() {
        let params = BSParams {
            spot: 100.0,
            strike: 100.0,
            time_to_expiry: 0.25,
            risk_free_rate: 0.05,
            volatility: 0.2,
            dividend_yield: 0.0,
        };
        
        let greeks = calculate_greeks(&params, OptionType::Put);
        
        // ATM put delta should be ~-0.45
        assert!(greeks.first.delta < 0.0 && greeks.first.delta > -0.5,
            "ATM put delta out of range: {}", greeks.first.delta);
    }
    
    #[test]
    fn test_gamma_positive() {
        let params = BSParams::default();
        
        let greeks = calculate_greeks(&params, OptionType::Call);
        
        // Gamma should always be positive for long options
        assert!(greeks.second.gamma > 0.0, "Gamma should be positive");
    }
    
    #[test]
    fn test_vega_positive() {
        let params = BSParams::default();
        
        let greeks = calculate_greeks(&params, OptionType::Call);
        
        // Vega should be positive for long options
        assert!(greeks.first.vega > 0.0, "Vega should be positive");
    }
    
    #[test]
    fn test_theta_negative_long_option() {
        let params = BSParams {
            spot: 100.0,
            strike: 100.0,
            time_to_expiry: 0.25,
            risk_free_rate: 0.05,
            volatility: 0.2,
            dividend_yield: 0.0,
        };
        
        let greeks = calculate_greeks(&params, OptionType::Call);
        
        // Long option theta should be negative (time decay)
        assert!(greeks.first.theta < 0.0, "Long option theta should be negative");
    }
    
    #[test]
    fn test_batch_greeks() {
        let params = vec![
            BSParams::default(),
            BSParams::default(),
            BSParams::default(),
        ];
        let types = vec![OptionType::Call, OptionType::Put, OptionType::Call];
        
        let mut deltas = [0.0; 10];
        let mut gammas = [0.0; 10];
        let mut vegas = [0.0; 10];
        let mut thetas = [0.0; 10];
        
        let count = batch_calculate_greeks(
            &params, &types, &mut deltas, &mut gammas, &mut vegas, &mut thetas
        );
        
        assert_eq!(count, 3);
        assert!(deltas[0] > 0.0); // Call delta positive
        assert!(deltas[1] < 0.0); // Put delta negative
        assert!(gammas[0] > 0.0);
        assert!(gammas[1] > 0.0);
    }
}
