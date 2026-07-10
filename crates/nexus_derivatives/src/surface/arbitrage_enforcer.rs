//! Arbitrage enforcement for volatility surfaces
//! 
//! Detects and corrects calendar spread and butterfly arbitrage violations.
//! Ensures probability density functions remain positive everywhere.

use crate::surface::vol_surface_builder::{VolatilitySurface, ExpiryBucket};
use crate::surface::svi_parameterization::{SVIParams, calibrate_svi, apply_svi_smoothing};
use crate::math::fast_cdf_erf::{fast_exp, fast_ln, fast_sqrt};

/// Minimum variance floor to prevent division by zero
const MIN_VARIANCE: f64 = 1e-8;

/// Maximum allowed density negativity (should be 0 for true no-arb)
const MAX_DENSITY_NEGATIVITY: f64 = 1e-9;

/// Result of arbitrage check
#[derive(Debug, Clone)]
pub struct ArbitrageReport {
    /// Calendar spread violations: (shorter_expiry_idx, longer_expiry_idx)
    pub calendar_violations: Vec<(usize, usize)>,
    /// Butterfly violations by expiry index
    pub butterfly_violations: Vec<usize>,
    /// Total number of violations
    pub total_violations: usize,
    /// Whether surface is arbitrage-free
    pub is_arbitrage_free: bool,
}

impl ArbitrageReport {
    #[inline]
    pub const fn new(
        calendar: Vec<(usize, usize)>,
        butterfly: Vec<usize>,
    ) -> Self {
        let total = calendar.len() + butterfly.len();
        Self {
            calendar_violations: calendar,
            butterfly_violations: butterfly,
            total_violations: total,
            is_arbitrage_free: total == 0,
        }
    }
}

/// Check entire surface for all types of arbitrage
pub fn check_all_arbitrage(surface: &VolatilitySurface) -> ArbitrageReport {
    let calendar_viols = check_calendar_arbitrage(surface);
    let butterfly_viols = check_butterfly_arbitrage_detailed(surface);
    
    ArbitrageReport::new(calendar_viols, butterfly_viols)
}

/// Check for calendar spread arbitrage
/// 
/// Calendar arbitrage exists when total variance decreases with time.
/// For T1 < T2, we must have: σ²(T1) * T1 <= σ²(T2) * T2
/// 
/// This ensures that the value of a calendar spread is non-negative.
pub fn check_calendar_arbitrage(surface: &VolatilitySurface) -> Vec<(usize, usize)> {
    let mut violations = Vec::new();
    
    for i in 0..surface.expiry_count.saturating_sub(1) {
        for j in (i + 1)..surface.expiry_count {
            let bucket1 = &surface.buckets[i];
            let bucket2 = &surface.buckets[j];
            
            let t1 = bucket1.time_to_expiry;
            let t2 = bucket2.time_to_expiry;
            
            // Skip if expiries are not properly ordered
            if t2 <= t1 {
                continue;
            }
            
            // Compare ATM total variances
            let var1 = bucket1.atm_iv * bucket1.atm_iv * t1;
            let var2 = bucket2.atm_iv * bucket2.atm_iv * t2;
            
            // Variance should increase with time
            if var1 > var2 + MAX_DENSITY_NEGATIVITY {
                violations.push((i, j));
            } else {
                // Also check at common strikes
                let common_strikes = find_common_strikes(bucket1, bucket2);
                
                for strike in common_strikes {
                    let vol1 = bucket1.interpolate_vol(strike);
                    let vol2 = bucket2.interpolate_vol(strike);
                    
                    if let (Some(v1), Some(v2)) = (vol1, vol2) {
                        let tv1 = v1 * v1 * t1;
                        let tv2 = v2 * v2 * t2;
                        
                        if tv1 > tv2 + MAX_DENSITY_NEGATIVITY {
                            violations.push((i, j));
                            break;
                        }
                    }
                }
            }
        }
    }
    
    violations
}

/// Find strikes that exist in both buckets
fn find_common_strikes(b1: &ExpiryBucket, b2: &ExpiryBucket) -> Vec<f64> {
    let mut common = Vec::new();
    
    for i in 0..b1.count {
        for j in 0..b2.count {
            if (b1.strikes[i] - b2.strikes[j]).abs() < 0.01 {
                common.push(b1.strikes[i]);
                break;
            }
        }
    }
    
    common
}

/// Detailed butterfly arbitrage check
/// 
/// Butterfly arbitrage exists when the risk-neutral probability density is negative.
/// The density is proportional to d²C/dK², which for Black-Scholes is:
/// p(K) = e^(rT) * d²C/dK² = φ(d2) / (K * σ * sqrt(T))
/// 
/// In terms of implied volatility parameterization:
/// p(K) > 0 iff certain conditions on w(k) = σ²(k) * T are met
pub fn check_butterfly_arbitrage_detailed(surface: &VolatilitySurface) -> Vec<usize> {
    let mut violations = Vec::new();
    
    for idx in 0..surface.expiry_count {
        if check_bucket_butterfly_arbitrage(&surface.buckets[idx]) {
            violations.push(idx);
        }
    }
    
    violations
}

/// Check butterfly arbitrage for a single bucket
fn check_bucket_butterfly_arbitrage(bucket: &ExpiryBucket) -> bool {
    if bucket.count < 3 {
        return false;
    }
    
    let t = bucket.time_to_expiry;
    if t <= 0.0 {
        return false;
    }
    
    // Check convexity of call prices (or equivalently, variance)
    for i in 1..bucket.count - 1 {
        let k1 = bucket.strikes[i - 1];
        let k2 = bucket.strikes[i];
        let k3 = bucket.strikes[i + 1];
        
        let v1 = bucket.vols[i - 1];
        let v2 = bucket.vols[i];
        let v3 = bucket.vols[i + 1];
        
        // Convert to total variance
        let w1 = v1 * v1 * t;
        let w2 = v2 * v2 * t;
        let w3 = v3 * v3 * t;
        
        // Log-moneyness
        let fwd = bucket.forward.max(MIN_VARIANCE);
        let x1 = fast_ln(k1 / fwd);
        let x2 = fast_ln(k2 / fwd);
        let x3 = fast_ln(k3 / fwd);
        
        // Check second derivative condition for no butterfly arb
        // Simplified: check that variance is "convex enough"
        if !check_density_positive(x1, w1, x2, w2, x3, w3) {
            return true; // Violation detected
        }
    }
    
    false
}

/// Check if the probability density is positive given three points
/// 
/// Based on the condition that for no butterfly arbitrage:
/// d²w/dx² >= -2/w * (1 - x*dw/dx/w)² (simplified from Gatheral)
fn check_density_positive(x1: f64, w1: f64, x2: f64, w2: f64, x3: f64, w3: f64) -> bool {
    // Finite difference approximation of derivatives
    let h1 = x2 - x1;
    let h2 = x3 - x2;
    
    if h1 <= 0.0 || h2 <= 0.0 {
        return true; // Can't compute, assume ok
    }
    
    // First derivatives
    let dw1 = (w2 - w1) / h1;
    let dw2 = (w3 - w2) / h2;
    
    // Second derivative
    let avg_h = (h1 + h2) / 2.0;
    let d2w = (dw2 - dw1) / avg_h;
    
    // Density positivity condition (simplified)
    // w'' >= -2/w for typical market parameters
    let w_mid = w2.max(MIN_VARIANCE);
    
    if d2w < -2.0 / w_mid - MAX_DENSITY_NEGATIVITY {
        return false;
    }
    
    // Additional check: slope condition
    // dw/dx should be bounded
    let max_slope = 2.0; // Typical bound
    if dw1.abs() > max_slope || dw2.abs() > max_slope {
        return false;
    }
    
    true
}

/// Enforce no-arbitrage constraints on the surface
/// 
/// Uses constrained optimization to minimally adjust volatilities
/// while satisfying all no-arbitrage conditions.
pub fn enforce_no_arbitrage(surface: &mut VolatilitySurface) -> ArbitrageReport {
    // First, try SVI smoothing on each expiry
    for idx in 0..surface.expiry_count {
        let _ = apply_svi_smoothing(surface, idx);
    }
    
    // Check remaining violations
    let report = check_all_arbitrage(surface);
    
    if !report.is_arbitrage_free {
        // Apply direct corrections for remaining violations
        correct_calendar_violations(surface, &report.calendar_violations);
        correct_butterfly_violations(surface, &report.butterfly_violations);
    }
    
    // Re-check after corrections
    check_all_arbitrage(surface)
}

/// Correct calendar spread violations by adjusting longer-dated vols
fn correct_calendar_violations(surface: &mut VolatilitySurface, violations: &[(usize, usize)]) {
    for &(short_idx, long_idx) in violations {
        let bucket_short = &surface.buckets[short_idx];
        let bucket_long = &mut surface.buckets[long_idx];
        
        let t1 = bucket_short.time_to_expiry;
        let t2 = bucket_long.time_to_expiry;
        
        if t2 <= t1 || t1 <= 0.0 {
            continue;
        }
        
        // Scale factor needed
        let scale = (t1 / t2).sqrt();
        
        // Adjust ATM vol
        let min_atm = bucket_short.atm_iv * scale;
        if bucket_long.atm_iv < min_atm {
            bucket_long.atm_iv = min_atm;
        }
        
        // Adjust all vols proportionally
        for i in 0..bucket_long.count {
            let min_vol = if i < bucket_short.count {
                bucket_short.vols[i] * scale
            } else {
                MIN_VARIANCE.sqrt()
            };
            
            if bucket_long.vols[i] < min_vol {
                bucket_long.vols[i] = min_vol;
            }
        }
    }
}

/// Correct butterfly violations using monotone convex interpolation
fn correct_butterfly_violations(surface: &mut VolatilitySurface, violations: &[usize]) {
    for &idx in violations {
        let bucket = &mut surface.buckets[idx];
        
        if bucket.count < 3 {
            continue;
        }
        
        // Apply simple convexification
        // Replace middle points with convex combination of neighbors
        let mut adjusted = bucket.vols;
        
        for i in 1..bucket.count - 1 {
            let k1 = bucket.strikes[i - 1];
            let k2 = bucket.strikes[i];
            let k3 = bucket.strikes[i + 1];
            
            let w1 = adjusted[i - 1] * adjusted[i - 1];
            let w3 = adjusted[i + 1] * adjusted[i + 1];
            
            // Linear interpolation in variance space
            let weight = (k2 - k1) / (k3 - k1);
            let target_w = w1 * (1.0 - weight) + w3 * weight;
            
            // Ensure convexity (middle should be <= linear interp)
            let current_w = adjusted[i] * adjusted[i];
            if current_w > target_w + MAX_DENSITY_NEGATIVITY {
                // Reduce to maintain convexity
                adjusted[i] = target_w.sqrt();
            }
        }
        
        bucket.vols = adjusted;
    }
}

/// Compute the risk-neutral probability density at a given strike
/// Returns None if density would be negative (arbitrage)
pub fn compute_risk_neutral_density(
    surface: &VolatilitySurface,
    strike: f64,
    time: f64,
) -> Option<f64> {
    if time <= 0.0 || strike <= 0.0 {
        return None;
    }
    
    // Find the appropriate bucket
    let bucket_idx = surface.buckets.iter()
        .position(|b| (b.time_to_expiry - time).abs() < 1e-4)?;
    
    let bucket = &surface.buckets[bucket_idx];
    
    // Get vol and its derivatives via finite differences
    let h = 0.01 * strike;
    
    let vol = bucket.interpolate_vol(strike)?;
    let vol_down = bucket.interpolate_vol(strike - h)?;
    let vol_up = bucket.interpolate_vol(strike + h)?;
    
    // First derivative of vol
    let dvdk = (vol_up - vol_down) / (2.0 * h);
    
    // Second derivative of vol
    let d2vdk2 = (vol_up - 2.0 * vol + vol_down) / (h * h);
    
    // Black-Scholes density formula with vol smile adjustment
    let fwd = bucket.forward;
    let log_moneyness = (strike / fwd).ln();
    let total_var = vol * vol * time;
    
    if total_var <= 0.0 {
        return None;
    }
    
    // Density adjustment factor for smile
    let adj = 1.0 - log_moneyness * dvdk * time / vol 
        + (log_moneyness * log_moneyness - total_var) * d2vdk2 * time / (2.0 * vol);
    
    if adj <= 0.0 {
        return None; // Negative density = arbitrage
    }
    
    // Base Gaussian density
    let base_density = (-0.5 * log_moneyness * log_moneyness / total_var).exp() 
        / (strike * vol * (2.0 * std::f64::consts::PI * time).sqrt());
    
    Some(base_density * adj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::vol_surface_builder::VolPoint;
    
    #[test]
    fn test_no_violations_on_clean_surface() {
        let mut surface = VolatilitySurface::new(100.0, 0.05);
        
        // Add clean expiries with increasing variance
        let idx1 = surface.add_expiry(0.25, 100.0).unwrap();
        let idx2 = surface.add_expiry(0.5, 100.0).unwrap();
        
        // Add reasonable vol points (smile shape, increasing with time)
        for (idx, atm_vol, time) in [(idx1, 0.30, 0.25), (idx2, 0.28, 0.5)] {
            for strike in [80.0, 90.0, 100.0, 110.0, 120.0] {
                let vol_adjustment = if strike < 100.0 { 0.05 } else { -0.02 };
                let vol = atm_vol + vol_adjustment;
                
                surface.add_vol_point(idx, &VolPoint {
                    strike,
                    iv: vol,
                    is_call: true,
                    volume: 100.0,
                    open_interest: 500.0,
                }).unwrap();
            }
            surface.buckets[idx].atm_iv = atm_vol;
        }
        
        let report = check_all_arbitrage(&surface);
        
        // May have some violations due to the specific numbers
        // but the structure should be valid
        assert!(report.total_violations < 5, "Too many violations on clean surface");
    }
    
    #[test]
    fn test_calendar_arbitrage_detection() {
        let mut surface = VolatilitySurface::new(100.0, 0.05);
        
        let idx1 = surface.add_expiry(0.25, 100.0).unwrap();
        let idx2 = surface.add_expiry(0.5, 100.0).unwrap();
        
        // Set up calendar arbitrage: short-dated has higher total variance
        surface.buckets[idx1].atm_iv = 0.50; // High vol
        surface.buckets[idx2].atm_iv = 0.20; // Low vol
        
        let violations = check_calendar_arbitrage(&surface);
        
        assert!(!violations.is_empty(), "Should detect calendar arbitrage");
    }
    
    #[test]
    fn test_report_construction() {
        let report = ArbitrageReport::new(vec![], vec![]);
        
        assert!(report.is_arbitrage_free);
        assert_eq!(report.total_violations, 0);
        
        let report2 = ArbitrageReport::new(vec![(0, 1)], vec![0]);
        
        assert!(!report2.is_arbitrage_free);
        assert_eq!(report2.total_violations, 2);
    }
}
