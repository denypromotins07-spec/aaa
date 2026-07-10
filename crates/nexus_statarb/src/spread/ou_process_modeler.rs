//! Ornstein-Uhlenbeck Process Modeler for Spread Dynamics
//! 
//! Models the price spread between cointegrated pairs as an OU
//! mean-reverting stochastic process: dX_t = θ(μ - X_t)dt + σdW_t
//! 
//! Provides real-time estimation of θ (mean reversion speed),
//! μ (long-term mean), and σ (volatility).

use core::sync::atomic::{AtomicU64, Ordering};

/// Maximum window size for rolling statistics
const MAX_WINDOW: usize = 2048;

/// OU Process parameters estimated from data
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OUParameters {
    /// Mean reversion speed (theta) - higher means faster reversion
    pub theta: f64,
    /// Long-term mean (mu)
    pub mu: f64,
    /// Volatility (sigma) of the diffusion term
    pub sigma: f64,
    /// Half-life of mean reversion: ln(2) / theta
    pub half_life: f64,
}

impl OUParameters {
    #[inline]
    pub const fn new(theta: f64, mu: f64, sigma: f64) -> Self {
        let half_life = if theta > 0.0 {
            core::f64::consts::LN_2 / theta
        } else {
            f64::INFINITY
        };
        
        Self {
            theta,
            mu,
            sigma,
            half_life,
        }
    }
}

impl Default for OUParameters {
    #[inline]
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

/// Ornstein-Uhlenbeck Process Modeler
/// 
/// Estimates OU parameters using maximum likelihood estimation
/// on rolling windows with zero allocations.
pub struct OUProcessModeler {
    /// Rolling window of spread values
    spreads: [f64; MAX_WINDOW],
    /// Rolling window of spread differences (for MLE)
    diff_spreads: [f64; MAX_WINDOW],
    /// Current window size
    window_size: usize,
    /// Current position in circular buffer
    head: usize,
    /// Last estimated parameters
    last_params: OUParameters,
    /// Update counter
    update_count: AtomicU64,
    /// Minimum observations required for estimation
    min_observations: usize,
}

impl OUProcessModeler {
    /// Create a new OU Process Modeler
    /// 
    /// # Arguments
    /// * `min_observations` - Minimum data points needed before estimating
    #[inline]
    pub fn new(min_observations: usize) -> Self {
        Self {
            spreads: [0.0; MAX_WINDOW],
            diff_spreads: [0.0; MAX_WINDOW],
            window_size: 0,
            head: 0,
            last_params: OUParameters::default(),
            update_count: AtomicU64::new(0),
            min_observations: min_observations.max(3), // Need at least 3 for MLE
        }
    }

    /// Add a new spread observation and update estimates
    /// 
    /// # Arguments
    /// * `spread` - Current spread value (price_A - hedge_ratio * price_B)
    /// 
    /// Returns updated OU parameters if enough data, None otherwise
    #[inline]
    pub fn update(&mut self, spread: f64) -> Option<OUParameters> {
        if !spread.is_finite() {
            return Some(self.last_params); // Return last good estimate
        }

        // Store current spread
        let prev_spread = self.spreads[self.head];
        self.spreads[self.head] = spread;

        // Compute difference if we have previous value
        if self.window_size > 0 {
            self.diff_spreads[self.head] = spread - prev_spread;
        }

        // Update circular buffer position
        self.head = (self.head + 1) % MAX_WINDOW;
        
        if self.window_size < MAX_WINDOW {
            self.window_size += 1;
        }

        self.update_count.fetch_add(1, Ordering::Relaxed);

        // Estimate parameters if we have enough data
        if self.window_size >= self.min_observations {
            let params = self.estimate_mle();
            self.last_params = params;
            Some(params)
        } else {
            None
        }
    }

    /// Maximum Likelihood Estimation of OU parameters
    /// 
    /// For discrete observations with time step Δt:
    /// X_{t+Δt} = X_t + θ(μ - X_t)Δt + σ√Δt * ε
    /// 
    /// Rearranging: ΔX_t = θ*μ*Δt - θ*X_t*Δt + σ√Δt * ε
    /// 
    /// This is a linear regression: ΔX_t = α + β*X_t + noise
    /// where α = θ*μ*Δt, β = -θ*Δt, σ² = Var(residuals)/Δt
    #[inline]
    fn estimate_mle(&self) -> OUParameters {
        let n = self.window_size - 1; // One less due to differencing
        
        if n < 2 {
            return OUParameters::default();
        }

        // Compute sums for OLS estimation
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_xx = 0.0f64;
        let mut sum_xy = 0.0f64;

        for i in 0..n {
            let x = self.spreads[i]; // X_t
            let y = self.diff_spreads[i]; // ΔX_t
            
            if !x.is_finite() || !y.is_finite() {
                continue;
            }

            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        let n_f64 = n as f64;
        
        // OLS formulas
        // β = (n*Σxy - Σx*Σy) / (n*Σxx - (Σx)²)
        // α = (Σy - β*Σx) / n
        let denom = n_f64 * sum_xx - sum_x * sum_x;
        
        if denom.abs() < 1e-15 {
            return OUParameters::default();
        }

        let beta = (n_f64 * sum_xy - sum_x * sum_y) / denom;
        let alpha = (sum_y - beta * sum_x) / n_f64;

        // Convert OLS coefficients to OU parameters
        // Assuming Δt = 1 (one tick)
        // β = -θ => θ = -β
        // α = θ*μ => μ = α/θ = α/(-β)
        
        let theta = -beta;
        
        // Ensure theta is positive (mean-reverting)
        if theta <= 0.0 {
            // Not mean-reverting, return conservative estimates
            return OUParameters::new(0.0, 0.0, self.compute_volatility(n));
        }

        let mu = alpha / theta;
        let sigma = self.compute_volatility(n);

        OUParameters::new(theta, mu, sigma)
    }

    /// Compute volatility (sigma) from residuals
    #[inline]
    fn compute_volatility(&self, n: usize) -> f64 {
        if n < 2 {
            return 0.0;
        }

        let params = self.last_params;
        let mut sum_sq_residuals = 0.0f64;
        let mut count = 0usize;

        for i in 0..n {
            let x = self.spreads[i];
            let dx = self.diff_spreads[i];
            
            if !x.is_finite() || !dx.is_finite() {
                continue;
            }

            // Predicted change: θ*(μ - x)
            let predicted_dx = params.theta * (params.mu - x);
            let residual = dx - predicted_dx;
            
            sum_sq_residuals += residual * residual;
            count += 1;
        }

        if count < 2 {
            return 0.0;
        }

        // σ² = Σ(residuals²) / (n-1)
        (sum_sq_residuals / (count - 1) as f64).sqrt()
    }

    /// Get the current Z-score of the spread
    /// 
    /// Z = (current_spread - μ) / (σ / √(2θ))
    /// The denominator is the stationary standard deviation of the OU process
    #[inline]
    pub fn compute_zscore(&self, current_spread: f64) -> f64 {
        let params = self.last_params;
        
        if params.sigma <= 0.0 || params.theta <= 0.0 {
            return 0.0;
        }

        // Stationary variance of OU process: σ² / (2θ)
        let stationary_std = params.sigma / (2.0 * params.theta).sqrt();
        
        if stationary_std < 1e-15 {
            return 0.0;
        }

        (current_spread - params.mu) / stationary_std
    }

    /// Get the last estimated parameters
    #[inline]
    pub fn parameters(&self) -> OUParameters {
        self.last_params
    }

    /// Get the number of updates performed
    #[inline]
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }

    /// Get the current window size
    #[inline]
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Reset the modeler
    #[inline]
    pub fn reset(&mut self) {
        self.window_size = 0;
        self.head = 0;
        self.last_params = OUParameters::default();
        self.update_count.store(0, Ordering::Relaxed);
    }

    /// Check if the process is currently mean-reverting
    #[inline]
    pub fn is_mean_reverting(&self) -> bool {
        self.last_params.theta > 0.01 // Threshold for meaningful mean reversion
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ou_parameter_estimation() {
        let mut modeler = OUProcessModeler::new(50);
        
        // Generate synthetic OU process
        // dX_t = 0.1 * (0.0 - X_t) + 0.05 * dW_t
        let true_theta = 0.1;
        let true_mu = 0.0;
        let mut x = 0.0f64;
        
        for i in 0..500 {
            // Euler-Maruyama discretization
            let dw = ((i % 100) as f64 - 50.0) * 0.01; // Pseudo-random
            x += true_theta * (true_mu - x) + 0.05 * dw;
            
            if let Some(params) = modeler.update(x) {
                // After convergence, check estimates
                if i > 400 {
                    assert!(params.theta > 0.05, "Theta should be positive");
                    assert!(params.half_life.is_finite(), "Half-life should be finite");
                }
            }
        }
        
        let final_params = modeler.parameters();
        assert!(final_params.theta > 0.0, "Should detect mean reversion");
    }

    #[test]
    fn test_nan_handling() {
        let mut modeler = OUProcessModeler::new(10);
        
        // Feed some valid data first
        for i in 0..20 {
            modeler.update(i as f64 * 0.1);
        }
        
        let params_before = modeler.parameters();
        
        // NaN should not change parameters
        modeler.update(f64::NAN);
        let params_after = modeler.parameters();
        
        assert_eq!(params_before.theta, params_after.theta);
        assert_eq!(params_before.mu, params_after.mu);
    }

    #[test]
    fn test_half_life_calculation() {
        let mut modeler = OUProcessModeler::new(20);
        
        // Create strongly mean-reverting process
        for i in 0..100 {
            let x = (i as f64 * 0.01).sin(); // Bounded oscillation
            modeler.update(x);
        }
        
        let params = modeler.parameters();
        
        if params.theta > 0.0 {
            let expected_half_life = core::f64::consts::LN_2 / params.theta;
            assert!((params.half_life - expected_half_life).abs() < 1e-10);
        }
    }

    #[test]
    fn test_min_observations() {
        let mut modeler = OUProcessModeler::new(30);
        
        // Should return None until min_observations reached
        for i in 0..29 {
            let result = modeler.update(i as f64 * 0.1);
            assert!(result.is_none(), "Should return None before min observations");
        }
        
        // Should return Some on 30th observation
        let result = modeler.update(3.0);
        assert!(result.is_some(), "Should return Some after min observations");
    }
}
