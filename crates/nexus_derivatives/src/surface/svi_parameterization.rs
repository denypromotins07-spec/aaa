//! SVI (Stochastic Volatility Inspired) parameterization for volatility surfaces
//! 
//! Implements the Gatheral SVI model for smooth, arbitrage-free volatility smiles.
//! SVI(w) = a + b * [ρ * (k - m) + sqrt((k - m)² + σ²)]
//! where w = total variance = σ² * T

use crate::math::fast_cdf_erf::{fast_exp, fast_ln, fast_sqrt};

/// Maximum iterations for SVI calibration
const MAX_ITERATIONS: usize = 100;

/// Convergence tolerance
const CONVERGENCE_TOL: f64 = 1e-8;

/// SVI parameters for a single expiry
#[derive(Debug, Clone, Copy)]
pub struct SVIParams {
    /// a: ATM variance level
    pub a: f64,
    /// b: Slope of the wings
    pub b: f64,
    /// ρ: Correlation (asymmetry parameter)
    pub rho: f64,
    /// m: Shift (location of minimum variance)
    pub m: f64,
    /// σ: Smoothness parameter (controls curvature at minimum)
    pub sigma: f64,
}

impl Default for SVIParams {
    fn default() -> Self {
        Self {
            a: 0.04,
            b: 0.1,
            rho: -0.3,
            m: 0.0,
            sigma: 0.1,
        }
    }
}

impl SVIParams {
    /// Create new SVI parameters with validation
    #[inline]
    pub fn new(a: f64, b: f64, rho: f64, m: f64, sigma: f64) -> Option<Self> {
        // Validate constraints for no-arbitrage
        if b < 0.0 || b > 2.0 {
            return None;
        }
        if rho <= -1.0 || rho >= 1.0 {
            return None;
        }
        if sigma <= 0.0 {
            return None;
        }
        
        Some(Self { a, b, rho, m, sigma })
    }
    
    /// Calculate total variance w(k) for log-moneyness k
    /// w(k) = a + b * [ρ * (k - m) + sqrt((k - m)² + σ²)]
    #[inline(always)]
    pub fn total_variance(&self, k: f64) -> f64 {
        let km = k - self.m;
        let sqrt_term = fast_sqrt(km * km + self.sigma * self.sigma);
        self.a + self.b * (self.rho * km + sqrt_term)
    }
    
    /// Calculate implied volatility from total variance
    /// σ(K, T) = sqrt(w(k) / T)
    #[inline(always)]
    pub fn implied_vol(&self, k: f64, time: f64) -> Option<f64> {
        if time <= 0.0 {
            return None;
        }
        
        let w = self.total_variance(k);
        if w <= 0.0 {
            return None;
        }
        
        Some(fast_sqrt(w / time))
    }
    
    /// Convert log-moneyness to strike
    #[inline(always)]
    pub fn k_from_strike(&self, strike: f64, forward: f64) -> f64 {
        if forward <= 0.0 || strike <= 0.0 {
            return 0.0;
        }
        fast_ln(strike / forward)
    }
    
    /// Convert strike to total variance
    #[inline(always)]
    pub fn variance_from_strike(&self, strike: f64, forward: f64) -> f64 {
        let k = self.k_from_strike(strike, forward);
        self.total_variance(k)
    }
}

/// Calibrate SVI parameters to market data
/// Uses Levenberg-Marquardt optimization
/// 
/// # Arguments
/// * `strikes` - Array of strikes
/// * `vols` - Array of implied volatilities
/// * `time` - Time to expiry
/// * `forward` - Forward price
/// 
/// # Returns
/// Calibrated SVI parameters or None if calibration fails
pub fn calibrate_svi(
    strikes: &[f64],
    vols: &[f64],
    time: f64,
    forward: f64,
) -> Option<SVIParams> {
    if strikes.len() != vols.len() || strikes.len() < 3 {
        return None;
    }
    
    // Initial guess based on data
    let mut params = initial_guess(strikes, vols, time, forward)?;
    
    // Levenberg-Marquardt iteration
    let mut lm_damping = 0.001;
    let mut prev_sse = f64::INFINITY;
    
    for _ in 0..MAX_ITERATIONS {
        let sse = compute_sse(&params, strikes, vols, time, forward);
        
        if sse < CONVERGENCE_TOL {
            break;
        }
        
        if sse >= prev_sse {
            lm_damping *= 10.0;
            if lm_damping > 1e6 {
                break; // Diverging
            }
        } else {
            lm_damping /= 10.0;
            prev_sse = sse;
        }
        
        // Compute gradient and update (simplified - full LM would need Jacobian)
        let gradient = compute_gradient(&params, strikes, vols, time, forward);
        
        // Update parameters with damping
        params.a -= lm_damping * gradient[0];
        params.b -= lm_damping * gradient[1];
        params.rho -= lm_damping * gradient[2];
        params.m -= lm_damping * gradient[3];
        params.sigma -= lm_damping * gradient[4];
        
        // Project to valid region
        params.b = params.b.clamp(0.0, 2.0);
        params.rho = params.rho.clamp(-0.99, 0.99);
        params.sigma = params.sigma.max(0.001);
        params.a = params.a.max(0.0001);
    }
    
    // Final validation
    let final_sse = compute_sse(&params, strikes, vols, time, forward);
    if final_sse.is_finite() && final_sse < 0.1 {
        Some(params)
    } else {
        None
    }
}

/// Initial parameter guess from market data
fn initial_guess(strikes: &[f64], vols: &[f64], time: f64, forward: f64) -> Option<SVIParams> {
    let n = strikes.len();
    if n < 3 {
        return None;
    }
    
    // Find ATM point (closest to forward)
    let mut atm_idx = 0;
    let mut min_diff = f64::INFINITY;
    
    for (i, &k) in strikes.iter().enumerate() {
        let diff = (k - forward).abs();
        if diff < min_diff {
            min_diff = diff;
            atm_idx = i;
        }
    }
    
    let atm_vol = vols[atm_idx];
    let atm_var = atm_vol * atm_vol * time;
    
    // Estimate wing slopes
    let left_vol = if atm_idx > 0 { vols[atm_idx - 1] } else { vols[0] };
    let right_vol = if atm_idx < n - 1 { vols[atm_idx + 1] } else { vols[n - 1] };
    
    let left_k = SVIParams::default().k_from_strike(strikes[atm_idx.saturating_sub(1).max(0)], forward);
    let right_k = SVIParams::default().k_from_strike(strikes[(atm_idx + 1).min(n - 1)], forward);
    
    // Rough slope estimates
    let b = fast_sqrt((right_vol - left_vol).abs() / time.max(0.001)).clamp(0.01, 1.0);
    let rho = if right_vol > left_vol { -0.3 } else { 0.3 };
    
    Some(SVIParams {
        a: atm_var * 0.5,
        b,
        rho,
        m: 0.0,
        sigma: 0.1,
    })
}

/// Compute sum of squared errors
fn compute_sse(params: &SVIParams, strikes: &[f64], vols: &[f64], time: f64, forward: f64) -> f64 {
    let mut sse = 0.0;
    
    for (&k, &v) in strikes.iter().zip(vols.iter()) {
        let log_moneyness = params.k_from_strike(k, forward);
        if let Some(model_vol) = params.implied_vol(log_moneyness, time) {
            let diff = model_vol - v;
            sse += diff * diff;
        }
    }
    
    sse
}

/// Compute gradient of SSE with respect to parameters (finite difference)
fn compute_gradient(
    params: &SVIParams,
    strikes: &[f64],
    vols: &[f64],
    time: f64,
    forward: f64,
) -> [f64; 5] {
    let eps = 1e-5;
    let base_sse = compute_sse(params, strikes, vols, time, forward);
    
    let mut grad = [0.0; 5];
    
    // Gradient for a
    let mut p_a = *params;
    p_a.a += eps;
    grad[0] = (compute_sse(&p_a, strikes, vols, time, forward) - base_sse) / eps;
    
    // Gradient for b
    let mut p_b = *params;
    p_b.b += eps;
    grad[1] = (compute_sse(&p_b, strikes, vols, time, forward) - base_sse) / eps;
    
    // Gradient for rho
    let mut p_rho = *params;
    p_rho.rho += eps;
    grad[2] = (compute_sse(&p_rho, strikes, vols, time, forward) - base_sse) / eps;
    
    // Gradient for m
    let mut p_m = *params;
    p_m.m += eps;
    grad[3] = (compute_sse(&p_m, strikes, vols, time, forward) - base_sse) / eps;
    
    // Gradient for sigma
    let mut p_sigma = *params;
    p_sigma.sigma += eps;
    grad[4] = (compute_sse(&p_sigma, strikes, vols, time, forward) - base_sse) / eps;
    
    grad
}

/// Apply SVI smoothing to an expiry bucket
/// Replaces raw vol points with SVI-fitted values
pub fn apply_svi_smoothing(surface: &mut VolatilitySurface, expiry_idx: usize) -> bool {
    if expiry_idx >= surface.expiry_count {
        return false;
    }
    
    let bucket = &surface.buckets[expiry_idx];
    if bucket.count < 3 {
        return false;
    }
    
    // Extract current data
    let strikes: Vec<f64> = bucket.strikes[..bucket.count].to_vec();
    let vols: Vec<f64> = bucket.vols[..bucket.count].to_vec();
    
    // Calibrate SVI
    if let Some(svi) = calibrate_svi(&strikes, &vols, bucket.time_to_expiry, bucket.forward) {
        // Replace vols with SVI-fitted values
        for i in 0..bucket.count {
            let k = svi.k_from_strike(bucket.strikes[i], bucket.forward);
            if let Some(fitted_vol) = svi.implied_vol(k, bucket.time_to_expiry) {
                bucket.vols[i] = fitted_vol;
            }
        }
        
        // Update ATM vol
        let atm_k = svi.k_from_strike(bucket.forward, bucket.forward);
        if let Some(atm_vol) = svi.implied_vol(atm_k, bucket.time_to_expiry) {
            bucket.atm_iv = atm_vol;
        }
        
        return true;
    }
    
    false
}

/// Check if SVI parameters satisfy no-arbitrage conditions
/// Based on Gatheral's conditions for no butterfly arbitrage
pub fn check_svi_no_arbitrage(params: &SVIParams) -> bool {
    // Condition 1: b > 0 (already enforced)
    if params.b <= 0.0 {
        return false;
    }
    
    // Condition 2: |ρ| < 1 (already enforced)
    if params.rho.abs() >= 1.0 {
        return false;
    }
    
    // Condition 3: σ > 0 (already enforced)
    if params.sigma <= 0.0 {
        return false;
    }
    
    // Condition 4: No butterfly arbitrage
    // Requires checking that d²w/dk² > -2/k² for all k
    // Simplified check at critical points
    let test_points = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    
    for &k in &test_points {
        let w = params.total_variance(k);
        if w <= 0.0 {
            return false;
        }
        
        // Second derivative approximation
        let h = 0.01;
        let w_minus = params.total_variance(k - h);
        let w_plus = params.total_variance(k + h);
        let d2w = (w_plus - 2.0 * w + w_minus) / (h * h);
        
        // Density condition: w'' > -2/w (simplified)
        if d2w < -2.0 / w.max(0.001) {
            return false;
        }
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_svi_total_variance() {
        let params = SVIParams::default();
        
        // ATM (k=0) should give reasonable variance
        let var_atm = params.total_variance(0.0);
        assert!(var_atm > 0.0, "ATM variance should be positive");
        
        // Variance should increase away from ATM for typical params
        let var_otm = params.total_variance(1.0);
        assert!(var_otm > 0.0, "OTM variance should be positive");
    }
    
    #[test]
    fn test_svi_calibration_basic() {
        // Generate synthetic data from known SVI params
        let true_params = SVIParams {
            a: 0.04,
            b: 0.1,
            rho: -0.3,
            m: 0.0,
            sigma: 0.1,
        };
        
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let forward = 100.0;
        let time = 0.5;
        
        let mut vols = Vec::new();
        for &k in &strikes {
            let log_k = true_params.k_from_strike(k, forward);
            let var = true_params.total_variance(log_k);
            vols.push(fast_sqrt(var / time));
        }
        
        // Calibrate
        let calibrated = calibrate_svi(&strikes, &vols, time, forward);
        
        assert!(calibrated.is_some(), "Calibration failed");
        
        let cal = calibrated.unwrap();
        
        // Check parameters are close to true values
        assert!((cal.a - true_params.a).abs() < 0.02, "a mismatch: {} vs {}", cal.a, true_params.a);
        assert!((cal.b - true_params.b).abs() < 0.05, "b mismatch: {} vs {}", cal.b, true_params.b);
    }
    
    #[test]
    fn test_no_arbitrage_check() {
        let valid_params = SVIParams {
            a: 0.04,
            b: 0.1,
            rho: -0.3,
            m: 0.0,
            sigma: 0.1,
        };
        
        assert!(check_svi_no_arbitrage(&valid_params), "Valid params failed arbitrage check");
        
        let invalid_params = SVIParams {
            a: 0.04,
            b: -0.1, // Invalid: negative b
            rho: -0.3,
            m: 0.0,
            sigma: 0.1,
        };
        
        assert!(!check_svi_no_arbitrage(&invalid_params), "Invalid params passed arbitrage check");
    }
    
    #[test]
    fn test_implied_vol_from_svi() {
        let params = SVIParams::default();
        let time = 0.5;
        
        let vol = params.implied_vol(0.0, time);
        assert!(vol.is_some(), "Should produce vol");
        assert!(vol.unwrap() > 0.0, "Vol should be positive");
        
        let vol_neg_time = params.implied_vol(0.0, -0.1);
        assert!(vol_neg_time.is_none(), "Negative time should return None");
    }
}
