//! Hamilton Time Series Filter
//! A superior alternative to Hodrick-Prescott for trend extraction.
//! Based on "Why You Should Never Use the Hodrick-Prescott Filter" (Hamilton, 2017).

/// Hamilton filter parameters
#[derive(Debug, Clone)]
pub struct HamiltonFilterParams {
    /// Number of periods for trend horizon (e.g., 8 for quarterly, 52 for weekly)
    h: usize,
    /// Number of lags in regression (typically 4)
    p: usize,
}

impl Default for HamiltonFilterParams {
    fn default() -> Self {
        Self { h: 8, p: 4 }
    }
}

/// Hamilton filter state for streaming computation
pub struct HamiltonFilter {
    params: HamiltonFilterParams,
    /// Data buffer for regression
    data_buffer: Vec<f64>,
    /// Pre-allocated X'X matrix accumulator
    xt_x: Vec<f64>,
    /// Pre-allocated X'y vector accumulator  
    xt_y: Vec<f64>,
    /// Current observation count
    n_obs: usize,
}

impl HamiltonFilter {
    /// Create new Hamilton filter
    pub fn new(params: HamiltonFilterParams) -> Self {
        let matrix_size = (params.p + 1) * (params.p + 1);
        let vector_size = params.p + 1;
        
        Self {
            params,
            data_buffer: Vec::with_capacity(params.h + params.p),
            xt_x: vec![0.0; matrix_size],
            xt_y: vec![0.0; vector_size],
            n_obs: 0,
        }
    }

    /// Update with new observation
    pub fn update(&mut self, y: f64) {
        self.data_buffer.push(y);
        
        // Keep only necessary history
        let required_len = self.params.h + self.params.p;
        if self.data_buffer.len() > required_len {
            self.data_buffer.remove(0);
        }
        
        self.n_obs += 1;
    }

    /// Compute trend estimate using OLS regression
    /// 
    /// y_{t+h} = β_0 + β_1*y_t + β_2*y_{t-1} + ... + β_p*y_{t-p+1} + ε_{t+h}
    /// 
    /// Trend at time t is the fitted value.
    pub fn compute_trend(&self) -> Option<f64> {
        let n = self.data_buffer.len();
        if n < self.params.h + self.params.p {
            return None;
        }

        // Build regression matrices incrementally
        let mut sum_xx = vec![0.0; (self.params.p + 1) * (self.params.p + 1)];
        let mut sum_xy = vec![0.0; self.params.p + 1];

        // Iterate through valid observations
        for i in self.params.p..(n - self.params.h) {
            // Regressors: [1, y_i, y_{i-1}, ..., y_{i-p+1}]
            let mut x = vec![1.0];
            for j in 0..self.params.p {
                x.push(self.data_buffer[i - j]);
            }

            // Target: y_{i+h}
            let y_target = self.data_buffer[i + self.params.h];

            // Accumulate X'X and X'y
            for row in 0..x.len() {
                for col in 0..x.len() {
                    sum_xx[row * x.len() + col] += x[row] * x[col];
                }
                sum_xy[row] += x[row] * y_target;
            }
        }

        // Solve normal equations: (X'X)β = X'y
        let beta = self.solve_ols(&sum_xx, &sum_xy)?;

        // Compute trend for most recent observation
        let mut x_latest = vec![1.0];
        for j in 0..self.params.p {
            let idx = (n - 1) - j;
            x_latest.push(self.data_buffer[idx]);
        }

        let mut trend = 0.0;
        for (i, &b) in beta.iter().enumerate() {
            trend += b * x_latest[i];
        }

        Some(trend)
    }

    /// Solve OLS using Cholesky decomposition
    fn solve_ols(&self, xtx: &[f64], xty: &[f64]) -> Option<Vec<f64>> {
        let p = self.params.p + 1;
        
        // Add small ridge for numerical stability
        let mut xtx_reg = xtx.to_vec();
        for i in 0..p {
            xtx_reg[i * p + i] += 1e-8;
        }

        // Cholesky decomposition: X'X = L * L'
        let mut l = vec![0.0; p * p];
        
        for i in 0..p {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i * p + k] * l[j * p + k];
                }
                
                if i == j {
                    let val = xtx_reg[i * p + i] - sum;
                    if val <= 0.0 {
                        return None; // Not positive definite
                    }
                    l[i * p + j] = val.sqrt();
                } else {
                    l[i * p + j] = (xtx_reg[i * p + j] - sum) / l[j * p + j];
                }
            }
        }

        // Forward substitution: L * z = X'y
        let mut z = vec![0.0; p];
        for i in 0..p {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[i * p + j] * z[j];
            }
            z[i] = (xty[i] - sum) / l[i * p + i];
        }

        // Backward substitution: L' * β = z
        let mut beta = vec![0.0; p];
        for i in (0..p).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..p {
                sum += l[j * p + i] * beta[j];
            }
            beta[i] = (z[i] - sum) / l[i * p + i];
        }

        Some(beta)
    }

    /// Get the cyclical component (detrended series)
    pub fn compute_cycle(&self) -> Option<f64> {
        let latest = *self.data_buffer.last()?;
        let trend = self.compute_trend()?;
        Some(latest - trend)
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.data_buffer.clear();
        self.xt_x.fill(0.0);
        self.xt_y.fill(0.0);
        self.n_obs = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hammer_filter_trend_extraction() {
        let params = HamiltonFilterParams { h: 4, p: 4 };
        let mut filter = HamiltonFilter::new(params);

        // Generate trending series with noise
        for i in 0..100 {
            let trend = i as f64 * 0.5;
            let noise = (i as f64 * 0.3).sin() * 2.0;
            filter.update(trend + noise);
        }

        let estimated_trend = filter.compute_trend();
        assert!(estimated_trend.is_some());
        
        // Trend should be close to linear trend at end
        let trend_val = estimated_trend.unwrap();
        assert!(trend_val > 40.0 && trend_val < 60.0);
    }

    #[test]
    fn test_cycle_extraction() {
        let params = HamiltonFilterParams::default();
        let mut filter = HamiltonFilter::new(params);

        // Pure sinusoidal cycle (no trend)
        for i in 0..100 {
            let cycle = (i as f64 * 0.2).sin() * 10.0;
            filter.update(cycle);
        }

        let cycle = filter.compute_cycle();
        assert!(cycle.is_some());
        
        // Cycle should capture the oscillation
        let cycle_val = cycle.unwrap();
        assert!(cycle_val.abs() < 15.0); // Within reasonable bounds
    }

    #[test]
    fn test_insufficient_data() {
        let params = HamiltonFilterParams { h: 8, p: 4 };
        let filter = HamiltonFilter::new(params);

        // Not enough data
        assert!(filter.compute_trend().is_none());
    }
}
