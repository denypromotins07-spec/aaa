//! Kalman Filter for Dynamic Hedge Ratio Estimation
//! 
//! Implements a real-time Kalman Filter to estimate the hedge ratio (beta)
//! between asset pairs on every tick. Uses Joseph form stabilization to
//! maintain positive-definiteness of the covariance matrix.

use core::sync::atomic::{AtomicU64, Ordering};

/// Kalman Filter state for hedge ratio estimation
/// 
/// State vector: [beta, alpha]^T where beta is the hedge ratio
/// Observation: y_t = beta * x_t + alpha + noise
#[repr(C)]
pub struct KalmanHedgeRatio {
    /// State estimate [beta, alpha]
    x: [f64; 2],
    /// Error covariance matrix (2x2, stored row-major)
    p: [f64; 4],
    /// Process noise covariance (2x2)
    q: [f64; 4],
    /// Measurement noise variance
    r: f64,
    /// Observation matrix H = [1, 0] for observing beta directly
    /// or H = [x_t, 1] for full regression
    h_temp: f64, // Cached x_t value
    /// Update counter for diagnostics
    update_count: AtomicU64,
    /// Stabilization counter - forces symmetric update periodically
    stabilizaton_interval: u64,
}

impl KalmanHedgeRatio {
    /// Create a new Kalman Filter for hedge ratio estimation
    /// 
    /// # Arguments
    /// * `initial_beta` - Initial hedge ratio estimate
    /// * `initial_alpha` - Initial intercept (usually 0)
    /// * `process_noise` - Process noise standard deviation
    /// * `measurement_noise` - Measurement noise standard deviation
    /// * `stabilization_interval` - How often to force symmetric update
    #[inline]
    pub fn new(
        initial_beta: f64,
        initial_alpha: f64,
        process_noise: f64,
        measurement_noise: f64,
        stabilization_interval: u64,
    ) -> Self {
        // Initialize covariance with moderate uncertainty
        let p_init = 1.0;
        
        Self {
            x: [initial_beta, initial_alpha],
            p: [p_init, 0.0, 0.0, p_init], // Diagonal initial P
            q: [
                process_noise * process_noise,
                0.0,
                0.0,
                process_noise * process_noise,
            ],
            r: measurement_noise * measurement_noise,
            h_temp: 0.0,
            update_count: AtomicU64::new(0),
            stabilizaton_interval: stabilization_interval.max(1),
        }
    }

    /// Update the Kalman Filter with a new observation
    /// 
    /// # Arguments
    /// * `x_obs` - Independent variable (e.g., asset B price)
    /// * `y_obs` - Dependent variable (e.g., asset A price)
    /// 
    /// Returns the updated hedge ratio (beta)
    #[inline]
    pub fn update(&mut self, x_obs: f64, y_obs: f64) -> f64 {
        self.h_temp = x_obs;
        
        // Handle NaN/Inf inputs gracefully
        if !x_obs.is_finite() || !y_obs.is_finite() {
            return self.x[0]; // Return last known good estimate
        }

        // === PREDICTION STEP ===
        // State prediction: x_pred = x (no dynamics in simple model)
        // Covariance prediction: P_pred = P + Q
        let p_pred = [
            self.p[0] + self.q[0],
            self.p[1] + self.q[1],
            self.p[2] + self.q[2],
            self.p[3] + self.q[3],
        ];

        // === UPDATE STEP ===
        // Innovation: y = y_obs - H * x_pred
        // H = [x_obs, 1]
        let y_pred = self.h_temp * self.x[0] + self.x[1];
        let innovation = y_obs - y_pred;

        // Innovation covariance: S = H * P_pred * H^T + R
        // S = x_obs^2 * P[0,0] + 2*x_obs*P[0,1] + P[1,1] + R
        let s = self.h_temp * self.h_temp * p_pred[0]
            + 2.0 * self.h_temp * p_pred[1]
            + p_pred[3]
            + self.r;

        // Prevent division by zero or numerical instability
        let s_inv = if s.abs() < 1e-15 {
            1e15 // Clamp to prevent overflow
        } else {
            1.0 / s
        };

        // Kalman Gain: K = P_pred * H^T * S^-1
        // K = [P[0,0]*x + P[0,1], P[1,0]*x + P[1,1]]^T * S^-1
        let k = [
            (p_pred[0] * self.h_temp + p_pred[1]) * s_inv,
            (p_pred[2] * self.h_temp + p_pred[3]) * s_inv,
        ];

        // State update: x = x_pred + K * innovation
        self.x[0] += k[0] * innovation;
        self.x[1] += k[1] * innovation;

        // === COVARIANCE UPDATE (Joseph Form for Stability) ===
        // P = (I - K*H) * P_pred * (I - K*H)^T + K * R * K^T
        // This form guarantees positive semi-definiteness
        
        let count = self.update_count.fetch_add(1, Ordering::Relaxed);
        let do_stabilize = (count % self.stabilizaton_interval) == 0;

        if do_stabilize {
            // Full Joseph form update
            // I - K*H = [[1 - k[0]*x, -k[0]], [-k[1]*x, 1 - k[1]]]
            let kh00 = 1.0 - k[0] * self.h_temp;
            let kh01 = -k[0];
            let kh10 = -k[1] * self.h_temp;
            let kh11 = 1.0 - k[1];

            // (I - KH) * P_pred
            let temp00 = kh00 * p_pred[0] + kh01 * p_pred[2];
            let temp01 = kh00 * p_pred[1] + kh01 * p_pred[3];
            let temp10 = kh10 * p_pred[0] + kh11 * p_pred[2];
            let temp11 = kh10 * p_pred[1] + kh11 * p_pred[3];

            // (I - KH) * P_pred * (I - KH)^T
            let p_new_00 = temp00 * kh00 + temp01 * kh01 + k[0] * k[0] * self.r;
            let p_new_01 = temp00 * kh10 + temp01 * kh11 + k[0] * k[1] * self.r;
            let p_new_10 = temp10 * kh00 + temp11 * kh01 + k[1] * k[0] * self.r;
            let p_new_11 = temp10 * kh10 + temp11 * kh11 + k[1] * k[1] * self.r;

            // Force symmetry to combat floating-point drift
            let p_sym = (p_new_01 + p_new_10) * 0.5;
            
            self.p[0] = p_new_00.max(1e-10); // Ensure positive diagonal
            self.p[1] = p_sym;
            self.p[2] = p_sym;
            self.p[3] = p_new_11.max(1e-10);
        } else {
            // Simplified update for speed (standard form)
            // P = (I - K*H) * P_pred
            self.p[0] = (1.0 - k[0] * self.h_temp) * p_pred[0] - k[0] * p_pred[2];
            self.p[1] = (1.0 - k[0] * self.h_temp) * p_pred[1] - k[0] * p_pred[3];
            self.p[2] = -k[1] * self.h_temp * p_pred[0] + (1.0 - k[1]) * p_pred[2];
            self.p[3] = -k[1] * self.h_temp * p_pred[1] + (1.0 - k[1]) * p_pred[3];

            // Force symmetry
            let p_sym = (self.p[1] + self.p[2]) * 0.5;
            self.p[1] = p_sym;
            self.p[2] = p_sym;

            // Ensure diagonals stay positive
            self.p[0] = self.p[0].max(1e-10);
            self.p[3] = self.p[3].max(1e-10);
        }

        self.x[0]
    }

    /// Get the current hedge ratio estimate
    #[inline]
    pub fn hedge_ratio(&self) -> f64 {
        self.x[0]
    }

    /// Get the current intercept estimate
    #[inline]
    pub fn intercept(&self) -> f64 {
        self.x[1]
    }

    /// Get the standard error of the hedge ratio estimate
    #[inline]
    pub fn hedge_ratio_std_error(&self) -> f64 {
        self.p[0].sqrt()
    }

    /// Reset the filter to initial state
    #[inline]
    pub fn reset(&mut self, initial_beta: f64, initial_alpha: f64) {
        self.x = [initial_beta, initial_alpha];
        self.p = [1.0, 0.0, 0.0, 1.0];
        self.update_count.store(0, Ordering::Relaxed);
    }

    /// Get the number of updates performed
    #[inline]
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_convergence() {
        let mut kf = KalmanHedgeRatio::new(0.0, 0.0, 0.001, 0.01, 100);
        
        // Simulate data with true beta = 1.5
        let true_beta = 1.5;
        let true_alpha = 0.0;
        
        for i in 1..=1000 {
            let x = i as f64 * 0.01;
            let noise = (i % 100) as f64 * 0.001 - 0.05; // Deterministic "noise"
            let y = true_beta * x + true_alpha + noise;
            
            let _beta_est = kf.update(x, y);
        }
        
        let estimated_beta = kf.hedge_ratio();
        assert!((estimated_beta - true_beta).abs() < 0.1, 
            "Beta should converge to {:?}, got {:?}", true_beta, estimated_beta);
    }

    #[test]
    fn test_nan_handling() {
        let mut kf = KalmanHedgeRatio::new(1.0, 0.0, 0.001, 0.01, 100);
        
        let initial_beta = kf.hedge_ratio();
        let result = kf.update(f64::NAN, 100.0);
        
        // Should return unchanged estimate on NaN input
        assert_eq!(result, initial_beta);
    }

    #[test]
    fn test_positive_definite_covariance() {
        let mut kf = KalmanHedgeRatio::new(1.0, 0.0, 0.001, 0.01, 10);
        
        for i in 1..=500 {
            let x = i as f64 * 0.01;
            let y = 1.5 * x + 0.1;
            kf.update(x, y);
            
            // Check P is positive definite (diagonals > 0, det > 0)
            assert!(kf.p[0] > 0.0, "P[0,0] should be positive");
            assert!(kf.p[3] > 0.0, "P[1,1] should be positive");
            
            let det = kf.p[0] * kf.p[3] - kf.p[1] * kf.p[2];
            assert!(det > 0.0, "Determinant should be positive");
        }
    }
}
