//! Micro-Price Drift Estimator using Kalman Filter.
//! Predicts short-term price movement for quote adjustment.
//! Zero-allocation, no unwrap/expect in hot paths.

/// Error types for drift estimation
#[derive(Debug, Clone, PartialEq)]
pub enum DriftError {
    InvalidParameters,
    NumericalInstability,
}

/// Kalman Filter state for micro-price tracking
#[derive(Debug, Clone)]
pub struct KalmanState {
    /// Estimated price (state)
    pub x: f64,
    /// Estimated velocity/drift (state derivative)
    pub v: f64,
    /// Error covariance matrix elements (2x2 stored as 4 values)
    pub p_xx: f64,
    pub p_xv: f64,
    pub p_vv: f64,
}

impl KalmanState {
    pub fn new(initial_price: f64, initial_variance: f64) -> Self {
        Self {
            x: initial_price,
            v: 0.0,
            p_xx: initial_variance,
            p_xv: 0.0,
            p_vv: initial_variance,
        }
    }
}

/// Configuration for Kalman filter
#[derive(Debug, Clone)]
pub struct KalmanConfig {
    /// Process noise for position
    pub process_noise_pos: f64,
    /// Process noise for velocity
    pub process_noise_vel: f64,
    /// Measurement noise
    pub measurement_noise: f64,
    /// Time step (seconds)
    pub dt: f64,
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            process_noise_pos: 0.001,
            process_noise_vel: 0.0001,
            measurement_noise: 0.01,
            dt: 0.001, // 1ms
        }
    }
}

/// Micro-Price Drift Estimator
pub struct MicroPriceDriftEstimator {
    config: KalmanConfig,
    state: KalmanState,
    /// Prediction horizon (nanoseconds)
    prediction_horizon_ns: u64,
}

impl MicroPriceDriftEstimator {
    pub fn new(config: KalmanConfig, prediction_horizon_ns: u64) -> Result<Self, DriftError> {
        if config.dt <= 0.0 || config.measurement_noise <= 0.0 {
            return Err(DriftError::InvalidParameters);
        }
        
        Ok(Self {
            config,
            state: KalmanState::new(0.0, 1.0),
            prediction_horizon_ns,
        })
    }
    
    /// Initialize with current micro-price
    pub fn initialize(&mut self, initial_price: f64, initial_variance: f64) {
        self.state = KalmanState::new(initial_price, initial_variance);
    }
    
    /// Update filter with new micro-price observation
    #[inline(always)]
    pub fn update(&mut self, observed_price: f64) -> Result<(), DriftError> {
        let dt = self.config.dt;
        
        // === PREDICT STEP ===
        // State transition: x_new = x + v*dt, v_new = v
        let x_pred = self.state.x + self.state.v * dt;
        let v_pred = self.state.v;
        
        // Covariance prediction: P_new = F * P * F' + Q
        // F = [[1, dt], [0, 1]]
        let p_xx_pred = self.state.p_xx + 2.0 * dt * self.state.p_xv + dt * dt * self.state.p_vv 
            + self.config.process_noise_pos;
        let p_xv_pred = self.state.p_xv + dt * self.state.p_vv;
        let p_vv_pred = self.state.p_vv + self.config.process_noise_vel;
        
        // === UPDATE STEP ===
        // Innovation: y = z - H * x_pred, where H = [1, 0]
        let innovation = observed_price - x_pred;
        
        // Innovation covariance: S = H * P_pred * H' + R
        let s = p_xx_pred + self.config.measurement_noise;
        
        // Check for numerical instability
        if s < 1e-15 {
            return Err(DriftError::NumericalInstability);
        }
        
        // Kalman gain: K = P_pred * H' / S
        let k_x = p_xx_pred / s;
        let k_v = p_xv_pred / s;
        
        // State update: x = x_pred + K * y
        self.state.x = x_pred + k_x * innovation;
        self.state.v = v_pred + k_v * innovation;
        
        // Covariance update: P = (I - K*H) * P_pred
        self.state.p_xx = (1.0 - k_x) * p_xx_pred;
        self.state.p_xv = (1.0 - k_x) * p_xv_pred;
        self.state.p_vv = p_vv_pred - k_v * p_xv_pred;
        
        Ok(())
    }
    
    /// Predict price at future time
    #[inline(always)]
    pub fn predict_price(&self, horizon_ns: u64) -> f64 {
        let horizon_sec = horizon_ns as f64 / 1e9;
        self.state.x + self.state.v * horizon_sec
    }
    
    /// Get predicted drift direction (-1 to 1)
    #[inline(always)]
    pub fn drift_direction(&self) -> f64 {
        // Normalize velocity to [-1, 1] range
        let max_vel = 1000.0; // Maximum expected velocity (price units per second)
        (self.state.v / max_vel).clamp(-1.0, 1.0)
    }
    
    /// Get predicted price change over horizon
    #[inline(always)]
    pub fn predicted_change(&self) -> f64 {
        let horizon_sec = self.prediction_horizon_ns as f64 / 1e9;
        self.state.v * horizon_sec
    }
    
    /// Get current estimated velocity
    #[inline(always)]
    pub const fn velocity(&self) -> f64 {
        self.state.v
    }
    
    /// Get current estimated price
    #[inline(always)]
    pub const fn estimated_price(&self) -> f64 {
        self.state.x
    }
    
    /// Get uncertainty (standard deviation)
    #[inline(always)]
    pub fn uncertainty(&self) -> f64 {
        self.state.p_xx.sqrt()
    }
    
    /// Reset filter
    pub fn reset(&mut self) {
        self.state = KalmanState::new(self.state.x, 1.0);
    }
}

/// Quote adjustment based on drift prediction
#[derive(Debug, Clone, Copy)]
pub struct DriftAdjustedQuote {
    /// Adjusted bid price
    pub bid: f64,
    /// Adjusted ask price
    pub ask: f64,
    /// Predicted drift direction
    pub drift: f64,
    /// Confidence level (0 to 1)
    pub confidence: f64,
}

impl MicroPriceDriftEstimator {
    /// Calculate drift-adjusted quotes
    #[inline(always)]
    pub fn calculate_adjusted_quotes(
        &self,
        base_bid: f64,
        base_ask: f64,
        adjustment_factor: f64,
    ) -> DriftAdjustedQuote {
        let drift = self.drift_direction();
        
        // Adjust quotes based on predicted drift
        // If drifting up (positive), move quotes up
        let bid_adjustment = drift * adjustment_factor;
        let ask_adjustment = drift * adjustment_factor;
        
        // Calculate confidence based on uncertainty
        let max_uncertainty = (base_ask - base_bid) * 0.5;
        let confidence = (1.0 - self.uncertainty() / max_uncertainty.max(1e-15)).clamp(0.0, 1.0);
        
        DriftAdjustedQuote {
            bid: base_bid + bid_adjustment * confidence,
            ask: base_ask + ask_adjustment * confidence,
            drift,
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kalman_filter_basic() {
        let config = KalmanConfig::default();
        let mut estimator = MicroPriceDriftEstimator::new(config, 50_000_000).unwrap();
        
        estimator.initialize(100.0, 0.01);
        
        // Simulate upward trending prices
        for i in 0..100 {
            let price = 100.0 + i as f64 * 0.01;
            estimator.update(price).unwrap();
        }
        
        // Should detect positive drift
        assert!(estimator.velocity() > 0.0);
        assert!(estimator.drift_direction() > 0.0);
    }
    
    #[test]
    fn test_prediction() {
        let config = KalmanConfig::default();
        let mut estimator = MicroPriceDriftEstimator::new(config, 50_000_000).unwrap();
        
        estimator.initialize(100.0, 0.01);
        
        // Constant price
        for _ in 0..50 {
            estimator.update(100.0).unwrap();
        }
        
        // Prediction should be close to current estimate
        let pred = estimator.predict_price(10_000_000);
        assert!((pred - estimator.estimated_price()).abs() < 0.01);
    }
}
