//! Recursive Least Squares with Sherman-Morrison Formula
//! 
//! Implements O(N) rank-1 updates to the inverse covariance matrix,
//! avoiding O(N^3) matrix inversions in the hot path.
//! Uses symmetric rounding to maintain numerical stability.

use core::sync::atomic::{AtomicU64, Ordering};

/// Maximum number of regressors supported (stack-allocated)
const MAX_REGRESSORS: usize = 8;

/// Recursive Least Squares estimator using Sherman-Morrison formula
/// 
/// For cointegration, typically uses 2 regressors: [price_B, 1] for
/// estimating beta and alpha in: price_A = beta * price_B + alpha
#[repr(C)]
pub struct RlsShermanMorrison {
    /// Number of active regressors
    n: usize,
    /// Inverse covariance matrix P (n x n, row-major, stack allocated)
    p: [[f64; MAX_REGRESSORS]; MAX_REGRESSORS],
    /// Parameter estimates theta (n elements)
    theta: [f64; MAX_REGRESSORS],
    /// Forgetting factor lambda (typically 0.99 - 1.0)
    lambda: f64,
    /// Regularization constant for numerical stability
    regularization: f64,
    /// Update counter
    update_count: AtomicU64,
    /// Last prediction error
    last_error: f64,
}

impl RlsShermanMorrison {
    /// Create a new RLS estimator
    /// 
    /// # Arguments
    /// * `n` - Number of regressors (must be <= MAX_REGRESSORS)
    /// * `lambda` - Forgetting factor (0 < lambda <= 1)
    /// * `initial_uncertainty` - Initial P matrix diagonal value
    #[inline]
    pub fn new(n: usize, lambda: f64, initial_uncertainty: f64) -> Option<Self> {
        if n == 0 || n > MAX_REGRESSORS {
            return None;
        }
        
        // Initialize P as diagonal matrix with large uncertainty
        let mut p = [[0.0; MAX_REGRESSORS]; MAX_REGRESSORS];
        for i in 0..n {
            p[i][i] = initial_uncertainty;
        }
        
        Some(Self {
            n,
            p,
            theta: [0.0; MAX_REGRESSORS],
            lambda: lambda.clamp(0.9, 1.0),
            regularization: 1e-10,
            update_count: AtomicU64::new(0),
            last_error: 0.0,
        })
    }

    /// Update the RLS estimator with a new observation
    /// 
    /// # Arguments
    /// * `phi` - Regressor vector (e.g., [price_B, 1])
    /// * `y` - Observed output (e.g., price_A)
    /// 
    /// Returns the updated parameter estimates slice
    #[inline]
    pub fn update(&mut self, phi: &[f64], y: f64) -> Option<&[f64]> {
        if phi.len() != self.n {
            return None;
        }

        // Validate inputs
        for &p in phi.iter() {
            if !p.is_finite() {
                return Some(&self.theta[..self.n]); // Return last good estimate
            }
        }
        
        if !y.is_finite() {
            return Some(&self.theta[..self.n]);
        }

        // === SHERMAN-MORRISON UPDATE ===
        // P_new = (P - (P * phi * phi^T * P) / (lambda + phi^T * P * phi)) / lambda
        
        // Step 1: Compute P * phi (matrix-vector product)
        let mut p_phi = [0.0; MAX_REGRESSORS];
        for i in 0..self.n {
            let mut sum = 0.0;
            for j in 0..self.n {
                sum += self.p[i][j] * phi[j];
            }
            p_phi[i] = sum;
        }

        // Step 2: Compute phi^T * P * phi (scalar)
        let mut phi_p_phi = 0.0;
        for i in 0..self.n {
            phi_p_phi += phi[i] * p_phi[i];
        }

        // Step 3: Compute denominator
        let denom = self.lambda + phi_p_phi;
        
        // Prevent division by zero
        if denom.abs() < self.regularization {
            return Some(&self.theta[..self.n]);
        }
        
        let inv_denom = 1.0 / denom;

        // Step 4: Compute gain vector K = P * phi / (lambda + phi^T * P * phi)
        let mut k = [0.0; MAX_REGRESSORS];
        for i in 0..self.n {
            k[i] = p_phi[i] * inv_denom;
        }

        // Step 5: Compute prediction error
        let mut y_pred = 0.0;
        for i in 0..self.n {
            y_pred += self.theta[i] * phi[i];
        }
        self.last_error = y - y_pred;

        // Step 6: Update parameters: theta_new = theta + K * error
        for i in 0..self.n {
            self.theta[i] += k[i] * self.last_error;
        }

        // Step 7: Update inverse covariance using Sherman-Morrison
        // P_new = (P - K * phi^T * P) / lambda
        // This is equivalent to: P - (P*phi)*(phi^T*P) / denom
        
        // Compute outer product K * (phi^T * P) = K * (P * phi)^T since P is symmetric
        // Actually need: K * phi^T * P, which is K * (P * phi)^T = K * p_phi^T
        for i in 0..self.n {
            for j in 0..self.n {
                self.p[i][j] = (self.p[i][j] - k[i] * p_phi[j]) / self.lambda;
            }
        }

        // === SYMMETRIC ROUNDING ===
        // Force symmetry to combat floating-point drift
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let avg = (self.p[i][j] + self.p[j][i]) * 0.5;
                self.p[i][j] = avg;
                self.p[j][i] = avg;
            }
        }

        // Ensure diagonal elements stay positive
        for i in 0..self.n {
            self.p[i][i] = self.p[i][i].max(self.regularization);
        }

        self.update_count.fetch_add(1, Ordering::Relaxed);
        
        Some(&self.theta[..self.n])
    }

    /// Get current parameter estimates
    #[inline]
    pub fn parameters(&self) -> &[f64] {
        &self.theta[..self.n]
    }

    /// Get the hedge ratio (first parameter)
    #[inline]
    pub fn hedge_ratio(&self) -> f64 {
        if self.n >= 1 {
            self.theta[0]
        } else {
            0.0
        }
    }

    /// Get the intercept (second parameter if exists)
    #[inline]
    pub fn intercept(&self) -> f64 {
        if self.n >= 2 {
            self.theta[1]
        } else {
            0.0
        }
    }

    /// Get the last prediction error
    #[inline]
    pub fn last_prediction_error(&self) -> f64 {
        self.last_error
    }

    /// Get the number of updates performed
    #[inline]
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }

    /// Reset the estimator
    #[inline]
    pub fn reset(&mut self, initial_uncertainty: f64) {
        for i in 0..self.n {
            for j in 0..self.n {
                self.p[i][j] = if i == j { initial_uncertainty } else { 0.0 };
            }
            self.theta[i] = 0.0;
        }
        self.update_count.store(0, Ordering::Relaxed);
        self.last_error = 0.0;
    }

    /// Get the condition number estimate (ratio of max to min diagonal of P)
    /// High condition number indicates numerical instability
    #[inline]
    pub fn condition_number_estimate(&self) -> f64 {
        let mut max_diag = f64::MIN;
        let mut min_diag = f64::MAX;
        
        for i in 0..self.n {
            let d = self.p[i][i];
            if d > max_diag { max_diag = d; }
            if d < min_diag { min_diag = d; }
        }
        
        if min_diag < self.regularization {
            f64::INFINITY
        } else {
            max_diag / min_diag
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rls_convergence() {
        let mut rls = RlsShermanMorrison::new(2, 0.995, 1000.0).unwrap();
        
        // True parameters: beta = 1.5, alpha = 0.1
        let true_beta = 1.5;
        let true_alpha = 0.1;
        
        for i in 1..=500 {
            let x = (i as f64) * 0.01;
            let noise = ((i % 50) as f64 - 25.0) * 0.001;
            let y = true_beta * x + true_alpha + noise;
            
            let phi = [x, 1.0];
            let _params = rls.update(&phi, y);
        }
        
        let beta_est = rls.hedge_ratio();
        let alpha_est = rls.intercept();
        
        assert!((beta_est - true_beta).abs() < 0.05, 
            "Beta should converge to {:?}, got {:?}", true_beta, beta_est);
        assert!((alpha_est - true_alpha).abs() < 0.05,
            "Alpha should converge to {:?}, got {:?}", true_alpha, alpha_est);
    }

    #[test]
    fn test_invalid_input_handling() {
        let mut rls = RlsShermanMorrison::new(2, 0.99, 100.0).unwrap();
        
        // First get a valid estimate
        let _ = rls.update(&[1.0, 1.0], 2.0);
        let initial_theta = rls.parameters()[0];
        
        // NaN in regressor should return unchanged estimate
        let result = rls.update(&[f64::NAN, 1.0], 2.0);
        assert!(result.is_some());
        assert_eq!(rls.parameters()[0], initial_theta);
        
        // Inf in output should return unchanged estimate
        let result = rls.update(&[1.0, 1.0], f64::INFINITY);
        assert!(result.is_some());
    }

    #[test]
    fn test_symmetry_maintenance() {
        let mut rls = RlsShermanMorrison::new(2, 0.99, 100.0).unwrap();
        
        for i in 1..=200 {
            let x = (i as f64) * 0.01;
            let y = 1.5 * x + 0.1;
            let phi = [x, 1.0];
            rls.update(&phi, y);
            
            // Check P is symmetric
            assert!((rls.p[0][1] - rls.p[1][0]).abs() < 1e-10,
                "P should remain symmetric");
        }
    }

    #[test]
    fn test_too_many_regressors() {
        let result = RlsShermanMorrison::new(MAX_REGRESSORS + 1, 0.99, 100.0);
        assert!(result.is_none());
    }
}
