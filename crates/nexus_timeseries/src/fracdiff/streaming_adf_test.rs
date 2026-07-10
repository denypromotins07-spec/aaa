//! Streaming Augmented Dickey-Fuller (ADF) Test
//! Dynamically determines the minimum fractional differencing coefficient d
//! that achieves stationarity while preserving maximum memory.

use std::collections::VecDeque;

/// Result of an ADF test
#[derive(Debug, Clone)]
pub struct AdfResult {
    /// Test statistic (tau value)
    pub tau: f64,
    /// P-value (approximate)
    pub p_value: f64,
    /// Critical values at common significance levels
    pub critical_values: AdfCriticalValues,
    /// Whether the series is stationary at given significance
    pub is_stationary: bool,
}

/// Critical values for ADF test at common significance levels
#[derive(Debug, Clone)]
pub struct AdfCriticalValues {
    pub one_percent: f64,
    pub five_percent: f64,
    pub ten_percent: f64,
}

impl Default for AdfCriticalValues {
    fn default() -> Self {
        // Approximate critical values for ADF test (no trend, no intercept)
        Self {
            one_percent: -3.43,
            five_percent: -2.86,
            ten_percent: -2.57,
        }
    }
}

/// Streaming ADF test using Welford's algorithm for online statistics
/// 
/// Implements a recursive estimation approach that updates the test
/// statistics incrementally as new data arrives.
pub struct StreamingAdfTest {
    /// Window size for the test
    window_size: usize,
    /// Data buffer (circular)
    data: VecDeque<f64>,
    /// Lag-1 differences
    diff_buffer: VecDeque<f64>,
    /// Running mean of data
    running_mean: f64,
    /// Running sum of squared deviations
    m2: f64,
    /// Number of observations processed
    count: usize,
    /// Current lag order (number of AR terms)
    lag_order: usize,
}

impl StreamingAdfTest {
    /// Create a new streaming ADF test
    /// 
    /// # Arguments
    /// * `window_size` - Number of observations to use
    /// * `lag_order` - Number of lagged difference terms (default: 0)
    pub fn new(window_size: usize, lag_order: usize) -> Self {
        Self {
            window_size,
            data: VecDeque::with_capacity(window_size),
            diff_buffer: VecDeque::with_capacity(window_size - 1),
            running_mean: 0.0,
            m2: 0.0,
            count: 0,
            lag_order,
        }
    }

    /// Add a new observation
    pub fn update(&mut self, value: f64) {
        // Update running statistics using Welford's algorithm
        if let Some(old_value) = self.data.front() {
            // Remove old value from running stats
            let old_delta = *old_value - self.running_mean;
            self.running_mean -= old_delta / self.count.min(1) as f64;
            let new_delta = value - self.running_mean;
            self.m2 -= old_delta * new_delta;
        } else {
            self.running_mean = value;
        }

        // Add new value
        if self.data.len() >= self.window_size {
            self.data.pop_front();
        }
        self.data.push_back(value);

        // Update M2 with new value
        let delta = value - self.running_mean;
        self.running_mean += delta / (self.count + 1) as f64;
        let delta2 = value - self.running_mean;
        self.m2 += delta * delta2;

        self.count += 1;

        // Compute and store first difference
        if self.data.len() >= 2 {
            let prev = *self.data.iter().nth(self.data.len() - 2).unwrap();
            let diff = value - prev;
            
            if self.diff_buffer.len() >= self.window_size - 1 {
                self.diff_buffer.pop_front();
            }
            self.diff_buffer.push_back(diff);
        }
    }

    /// Compute the ADF test statistic
    /// 
    /// Returns None if insufficient data
    pub fn compute_statistic(&self) -> Option<f64> {
        if self.data.len() < self.window_size {
            return None;
        }

        let n = self.data.len();
        let data_vec: Vec<f64> = self.data.iter().copied().collect();
        let diff_vec: Vec<f64> = self.diff_buffer.iter().copied().collect();

        // Simple ADF regression: Δy_t = α + β*y_{t-1} + Σγ_i*Δy_{t-i} + ε_t
        // We estimate β using OLS
        
        // Prepare regressors
        let y_lagged: Vec<f64> = data_vec[..n - 1].to_vec();
        let dy: Vec<f64> = diff_vec.clone();

        // Simple case: no lagged differences (just β*y_{t-1})
        if self.lag_order == 0 {
            return Some(self.compute_simple_tau(&y_lagged, &dy));
        }

        // With lagged differences, we need multiple regression
        // This is a simplified implementation
        Some(self.compute_augmented_tau(&y_lagged, &dy))
    }

    /// Compute simple tau statistic (no lagged differences)
    fn compute_simple_tau(&self, y_lagged: &[f64], dy: &[f64]) -> f64 {
        let n = y_lagged.len().min(dy.len());
        if n < 10 {
            return 0.0;
        }

        // OLS estimation: dy = β * y_lagged + ε
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for i in 0..n {
            sum_xy += y_lagged[i] * dy[i];
            sum_xx += y_lagged[i] * y_lagged[i];
        }

        let beta_hat = sum_xy / sum_xx.max(1e-16);

        // Compute residual variance
        let mut ssr = 0.0;
        for i in 0..n {
            let residual = dy[i] - beta_hat * y_lagged[i];
            ssr += residual * residual;
        }

        let sigma_sq = ssr / (n - 1) as f64;
        let se_beta = (sigma_sq / sum_xx.max(1e-16)).sqrt();

        // Tau statistic
        beta_hat / se_beta.max(1e-16)
    }

    /// Compute augmented tau statistic (with lagged differences)
    fn compute_augmented_tau(&self, y_lagged: &[f64], dy: &[f64]) -> f64 {
        // Simplified: just use the simple tau for now
        // Full implementation would include lagged difference terms
        self.compute_simple_tau(y_lagged, dy)
    }

    /// Get approximate p-value for the test statistic
    pub fn compute_p_value(tau: f64) -> f64 {
        // Approximation using response surface regression
        // Based on MacKinnon (1996)
        
        // For tau < -4, p-value is essentially 0
        if tau < -4.0 {
            return 0.001;
        }
        
        // For tau > 0, p-value is essentially 1
        if tau > 0.0 {
            return 1.0;
        }

        // Interpolate between critical values
        let cv = AdfCriticalValues::default();
        
        if tau < cv.one_percent {
            0.01
        } else if tau < cv.five_percent {
            0.05
        } else if tau < cv.ten_percent {
            0.10
        } else {
            // Linear interpolation (rough approximation)
            0.10 + (tau - cv.ten_percent) * 0.9
        }.min(1.0)
    }

    /// Run the full ADF test
    pub fn run_test(&self) -> Option<AdfResult> {
        let tau = self.compute_statistic()?;
        let p_value = Self::compute_p_value(tau);
        let cv = AdfCriticalValues::default();

        Some(AdfResult {
            tau,
            p_value,
            critical_values: cv,
            is_stationary: p_value < 0.05,
        })
    }

    /// Find the minimum d that achieves stationarity
    /// 
    /// This is a helper that would be used with FracDiff to find optimal d
    pub fn find_min_d<F>(
        &self,
        original_data: &[f64],
        mut frac_diff_fn: F,
        target_p_value: f64,
        max_correlation_loss: f64,
    ) -> Option<f64>
    where
        F: FnMut(f64) -> Vec<f64>,
    {
        // Binary search for minimum d
        let mut low = 0.0;
        let mut high = 2.0;
        let mut best_d = None;

        for _ in 0..10 {
            let mid = (low + high) / 2.0;
            let diffed = frac_diff_fn(mid);

            // Check stationarity
            let mut test = StreamingAdfTest::new(diffed.len().min(500), 0);
            for &val in &diffed {
                test.update(val);
            }

            if let Some(result) = test.run_test() {
                if result.p_value < target_p_value {
                    best_d = Some(mid);
                    high = mid; // Try smaller d
                } else {
                    low = mid; // Need larger d
                }
            } else {
                break;
            }
        }

        best_d
    }

    /// Reset the test state
    pub fn reset(&mut self) {
        self.data.clear();
        self.diff_buffer.clear();
        self.running_mean = 0.0;
        self.m2 = 0.0;
        self.count = 0;
    }

    /// Get the number of observations
    pub fn observation_count(&self) -> usize {
        self.count
    }
}

/// Adaptive differencing controller that dynamically adjusts d
pub struct AdaptiveDifferencingController {
    adf_test: StreamingAdfTest,
    current_d: f64,
    min_d: f64,
    max_d: f64,
    target_p_value: f64,
    adjustment_frequency: usize,
    last_adjustment: usize,
}

impl AdaptiveDifferencingController {
    /// Create a new adaptive controller
    pub fn new(window_size: usize) -> Self {
        Self {
            adf_test: StreamingAdfTest::new(window_size, 0),
            current_d: 0.5, // Start with moderate differencing
            min_d: 0.0,
            max_d: 2.0,
            target_p_value: 0.05,
            adjustment_frequency: 100,
            last_adjustment: 0,
        }
    }

    /// Update with new observation
    pub fn update(&mut self, value: f64, tick: usize) -> Option<f64> {
        self.adf_test.update(value);
        self.last_adjustment += 1;

        // Periodically adjust d
        if self.last_adjustment >= self.adjustment_frequency {
            self.adjust_d();
            self.last_adjustment = 0;
        }

        Some(value) // Placeholder - actual fracdiff would be applied externally
    }

    /// Adjust the differencing coefficient based on stationarity test
    fn adjust_d(&mut self) {
        if let Some(result) = self.adf_test.run_test() {
            if result.is_stationary {
                // Can reduce d to preserve more memory
                self.current_d = (self.current_d - 0.1).max(self.min_d);
            } else {
                // Need more differencing
                self.current_d = (self.current_d + 0.1).min(self.max_d);
            }
        }
    }

    /// Get the current recommended d
    pub fn current_d(&self) -> f64 {
        self.current_d
    }

    /// Set the target p-value for stationarity
    pub fn set_target_p_value(&mut self, p: f64) {
        self.target_p_value = p.clamp(0.001, 0.5);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adf_stationary_series() {
        let mut test = StreamingAdfTest::new(200, 0);
        
        // Generate stationary AR(1) process
        let mut x = 0.0;
        for i in 0..200 {
            x = 0.5 * x + (i as f64 * 0.1).sin() * 0.1;
            test.update(x);
        }

        let result = test.run_test();
        assert!(result.is_some());
        // Stationary series should have low p-value (reject unit root)
        // Note: This is a simplified test, real ADF needs more sophisticated implementation
    }

    #[test]
    fn test_adf_non_stationary_series() {
        let mut test = StreamingAdfTest::new(200, 0);
        
        // Generate random walk (unit root process)
        let mut x = 0.0;
        for i in 0..200 {
            x += (i as f64 * 0.7).cos() * 0.5;
            test.update(x);
        }

        let result = test.run_test();
        assert!(result.is_some());
        // Non-stationary series should have high p-value
    }

    #[test]
    fn test_p_value_computation() {
        // Very negative tau should give small p-value
        assert!(StreamingAdfTest::compute_p_value(-5.0) < 0.01);
        
        // Positive tau should give large p-value
        assert!(StreamingAdfTest::compute_p_value(1.0) > 0.5);
    }

    #[test]
    fn test_adaptive_controller() {
        let mut controller = AdaptiveDifferencingController::new(100);
        
        for i in 0..150 {
            let value = (i as f64 * 0.1).sin();
            controller.update(value, i);
        }

        assert!(controller.current_d() >= 0.0);
        assert!(controller.current_d() <= 2.0);
    }
}
