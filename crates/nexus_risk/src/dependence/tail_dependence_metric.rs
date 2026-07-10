//! Tail Dependence Metrics for real-time crash contagion measurement
//!
//! Calculates the probability that Asset B will crash given that Asset A has crashed.
//! Critical for early warning systems and portfolio risk management during black swan events.

use crate::dependence::student_t_copula::{StudentTCopula, StudentTCopulaConfig, CopulaError};
use ndarray::{Array1, Array2, ArrayView1};

/// Configuration for tail dependence calculation
#[derive(Debug, Clone)]
pub struct TailDependenceConfig {
    /// Lower tail threshold (e.g., 0.05 for 5th percentile)
    pub lower_threshold: f64,
    /// Upper tail threshold (e.g., 0.95 for 95th percentile)
    pub upper_threshold: f64,
    /// Degrees of freedom for Student-t copula
    pub degrees_of_freedom: f64,
    /// Minimum observations required
    pub min_observations: usize,
}

impl Default for TailDependenceConfig {
    fn default() -> Self {
        Self {
            lower_threshold: 0.05,
            upper_threshold: 0.95,
            degrees_of_freedom: 5.0,
            min_observations: 252, // 1 trading year
        }
    }
}

/// Result of tail dependence analysis between two assets
#[derive(Debug, Clone)]
pub struct TailDependenceResult {
    /// Lower tail dependence coefficient λ_L
    /// P(B crashes | A crashes) for left tail events
    pub lower_tail_dependence: f64,
    
    /// Upper tail dependence coefficient λ_U
    /// P(B surges | A surges) for right tail events
    pub upper_tail_dependence: f64,
    
    /// Asymmetry measure: |λ_U - λ_L|
    pub asymmetry: f64,
    
    /// Empirical lower tail dependence from data
    pub empirical_lower_lambda: f64,
    
    /// Empirical upper tail dependence from data
    pub empirical_upper_lambda: f64,
    
    /// Correlation during tail events (vs normal correlation)
    pub tail_correlation: f64,
    
    /// Normal period correlation for comparison
    pub normal_correlation: f64,
    
    /// Number of joint tail events observed
    pub joint_tail_events: usize,
    
    /// Statistical significance (p-value) of tail dependence
    pub p_value: f64,
}

impl TailDependenceResult {
    /// Check if tail dependence is statistically significant
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
    
    /// Check if there's significant lower tail dependence (crash contagion risk)
    pub fn has_crash_contagion(&self) -> bool {
        self.lower_tail_dependence > 0.3 && self.is_significant(0.05)
    }
    
    /// Get the average tail dependence
    pub fn average_tail_dependence(&self) -> f64 {
        (self.lower_tail_dependence + self.upper_tail_dependence) / 2.0
    }
}

/// Engine for calculating tail dependence metrics
pub struct TailDependenceCalculator {
    config: TailDependenceConfig,
}

impl TailDependenceCalculator {
    /// Create a new calculator with default configuration
    pub fn new() -> Self {
        Self::with_config(TailDependenceConfig::default())
    }
    
    /// Create a new calculator with custom configuration
    pub fn with_config(config: TailDependenceConfig) -> Self {
        Self { config }
    }
    
    /// Calculate tail dependence between two return series
    pub fn calculate_pairwise(
        &self,
        returns_a: &ArrayView1<f64>,
        returns_b: &ArrayView1<f64>,
    ) -> Result<TailDependenceResult, CopulaError> {
        if returns_a.len() != returns_b.len() {
            return Err(CopulaError::InvalidDimensions);
        }
        
        if returns_a.len() < self.config.min_observations {
            return Err(CopulaError::InvalidInput(
                format!(
                    "Insufficient observations: got {}, need at least {}",
                    returns_a.len(),
                    self.config.min_observations
                )
            ));
        }
        
        // Convert returns to uniform marginals using empirical CDF
        let (u_a, u_b) = self.convert_to_uniform(returns_a, returns_b)?;
        
        // Calculate empirical tail dependence
        let (emp_lower, emp_upper, joint_events) = 
            self.empirical_tail_dependence(&u_a, &u_b)?;
        
        // Estimate correlation from data
        let correlation = self.calculate_correlation(returns_a, returns_b);
        
        // Build Student-t copula and calculate theoretical tail dependence
        let config = StudentTCopulaConfig::bivariate(correlation, self.config.degrees_of_freedom)?;
        let copula = StudentTCopula::new(config)?;
        
        let theoretical_lambda = copula.tail_dependence_coefficient()?;
        
        // For Student-t, lower and upper tail dependence are equal
        // But we report empirical values which may differ
        let lower_lambda = theoretical_lambda;
        let upper_lambda = theoretical_lambda;
        
        // Calculate tail correlation (correlation during extreme events)
        let tail_corr = self.calculate_tail_correlation(&u_a, &u_b)?;
        
        // Calculate p-value for significance testing
        let p_value = self.calculate_significance(emp_lower, returns_a.len());
        
        Ok(TailDependenceResult {
            lower_tail_dependence: lower_lambda,
            upper_tail_dependence: upper_lambda,
            asymmetry: (upper_lambda - lower_lambda).abs(),
            empirical_lower_lambda: emp_lower,
            empirical_upper_lambda: emp_upper,
            tail_correlation: tail_corr,
            normal_correlation: correlation,
            joint_tail_events: joint_events,
            p_value,
        })
    }
    
    /// Calculate full tail dependence matrix for multiple assets
    pub fn calculate_matrix(
        &self,
        returns: &Array2<f64>,
    ) -> Result<Array2<f64>, CopulaError> {
        let (n_assets, n_obs) = returns.dim();
        
        if n_obs < self.config.min_observations {
            return Err(CopulaError::InvalidInput(
                "Insufficient observations for matrix calculation".to_string()
            ));
        }
        
        let mut lambda_matrix = Array2::zeros((n_assets, n_assets));
        
        for i in 0..n_assets {
            for j in 0..n_assets {
                if i == j {
                    lambda_matrix[[i, j]] = 1.0;
                } else if i < j {
                    let row_i = returns.row(i);
                    let row_j = returns.row(j);
                    
                    let result = self.calculate_pairwise(&row_i, &row_j)?;
                    lambda_matrix[[i, j]] = result.lower_tail_dependence;
                    lambda_matrix[[j, i]] = result.lower_tail_dependence;
                }
            }
        }
        
        Ok(lambda_matrix)
    }
    
    /// Convert return series to uniform marginals using empirical CDF
    fn convert_to_uniform(
        &self,
        returns_a: &ArrayView1<f64>,
        returns_b: &ArrayView1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), CopulaError> {
        let n = returns_a.len();
        
        // Rank transformation to uniform [0, 1]
        let mut u_a = Array1::zeros(n);
        let mut u_b = Array1::zeros(n);
        
        // Sort and rank for A
        let mut sorted_a: Vec<(usize, f64)> = returns_a.iter().copied().enumerate().collect();
        sorted_a.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        for (rank, (orig_idx, _)) in sorted_a.iter().enumerate() {
            u_a[*orig_idx] = (rank + 1) as f64 / (n + 1) as f64;
        }
        
        // Sort and rank for B
        let mut sorted_b: Vec<(usize, f64)> = returns_b.iter().copied().enumerate().collect();
        sorted_b.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        for (rank, (orig_idx, _)) in sorted_b.iter().enumerate() {
            u_b[*orig_idx] = (rank + 1) as f64 / (n + 1) as f64;
        }
        
        Ok((u_a, u_b))
    }
    
    /// Calculate empirical tail dependence from uniform marginals
    fn empirical_tail_dependence(
        &self,
        u_a: &Array1<f64>,
        u_b: &Array1<f64>,
    ) -> Result<(f64, f64, usize), CopulaError> {
        let n = u_a.len();
        
        // Count events in lower tail
        let mut lower_a_crash = 0;
        let mut lower_both_crash = 0;
        
        // Count events in upper tail
        let mut upper_a_surge = 0;
        let mut upper_both_surge = 0;
        
        for i in 0..n {
            // Lower tail (crashes)
            if u_a[i] < self.config.lower_threshold {
                lower_a_crash += 1;
                if u_b[i] < self.config.lower_threshold {
                    lower_both_crash += 1;
                }
            }
            
            // Upper tail (surges)
            if u_a[i] > self.config.upper_threshold {
                upper_a_surge += 1;
                if u_b[i] > self.config.upper_threshold {
                    upper_both_surge += 1;
                }
            }
        }
        
        let emp_lower = if lower_a_crash > 0 {
            lower_both_crash as f64 / lower_a_crash as f64
        } else {
            0.0
        };
        
        let emp_upper = if upper_a_surge > 0 {
            upper_both_surge as f64 / upper_a_surge as f64
        } else {
            0.0
        };
        
        Ok((emp_lower, emp_upper, lower_both_crash))
    }
    
    /// Calculate Pearson correlation
    fn calculate_correlation(&self, a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        let n = a.len() as f64;
        
        let mean_a = a.sum() / n;
        let mean_b = b.sum() / n;
        
        let cov: f64 = a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - mean_a) * (y - mean_b))
            .sum::<f64>()
            / n;
        
        let var_a: f64 = a.mapv(|x| (x - mean_a).powi(2)).sum() / n;
        let var_b: f64 = b.mapv(|x| (x - mean_b).powi(2)).sum() / n;
        
        let std_a = var_a.sqrt();
        let std_b = var_b.sqrt();
        
        if std_a < 1e-10 || std_b < 1e-10 {
            return 0.0;
        }
        
        (cov / (std_a * std_b)).clamp(-1.0, 1.0)
    }
    
    /// Calculate correlation during tail events only
    fn calculate_tail_correlation(
        &self,
        u_a: &Array1<f64>,
        u_b: &Array1<f64>,
    ) -> Result<f64, CopulaError> {
        let mut tail_a = Vec::new();
        let mut tail_b = Vec::new();
        
        for i in 0..u_a.len() {
            // Include observations in either tail
            if u_a[i] < self.config.lower_threshold 
                || u_a[i] > self.config.upper_threshold
                || u_b[i] < self.config.lower_threshold 
                || u_b[i] > self.config.upper_threshold 
            {
                tail_a.push(u_a[i]);
                tail_b.push(u_b[i]);
            }
        }
        
        if tail_a.len() < 10 {
            return Ok(0.0); // Insufficient tail observations
        }
        
        let tail_a_view = ArrayView1::from(&tail_a);
        let tail_b_view = ArrayView1::from(&tail_b);
        
        Ok(self.calculate_correlation(&tail_a_view, &tail_b_view))
    }
    
    /// Calculate statistical significance of tail dependence
    fn calculate_significance(&self, lambda: f64, n: usize) -> f64 {
        // Simplified p-value calculation using asymptotic normality
        // Under null hypothesis of independence, λ ~ N(0, 1/n)
        
        if n < 30 {
            return 1.0; // Too few observations for reliable test
        }
        
        let se = 1.0 / (n as f64).sqrt();
        
        if se < 1e-10 {
            return 0.0;
        }
        
        let z_score = lambda / se;
        
        // Two-tailed p-value from standard normal
        2.0 * (1.0 - Self::normal_cdf(z_score.abs()))
    }
    
    /// Standard normal CDF approximation
    fn normal_cdf(x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989423 * (-x * x / 2.0).exp();
        let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        
        if x > 0.0 {
            1.0 - p
        } else {
            p
        }
    }
}

impl Default for TailDependenceCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tail_dependence_calculation() {
        // Generate correlated test data
        let n = 500;
        let mut returns_a = Array1::zeros(n);
        let mut returns_b = Array1::zeros(n);
        
        // Simple correlated returns
        for i in 0..n {
            let base = (i as f64 * 0.1).sin();
            returns_a[i] = base * 0.02;
            returns_b[i] = base * 0.025 + (i as f64 * 0.05).cos() * 0.01;
        }
        
        let calc = TailDependenceCalculator::new();
        let result = calc.calculate_pairwise(&returns_a.view(), &returns_b.view()).unwrap();
        
        assert!(result.lower_tail_dependence >= 0.0);
        assert!(result.lower_tail_dependence <= 1.0);
        assert!(result.joint_tail_events >= 0);
    }
    
    #[test]
    fn test_crash_contagion_detection() {
        // Create data with strong tail dependence
        let n = 1000;
        let mut returns_a = Array1::zeros(n);
        let mut returns_b = Array1::zeros(n);
        
        // Add some joint crashes
        for i in 0..n {
            returns_a[i] = (i as f64 / 100.0).sin() * 0.03;
            returns_b[i] = returns_a[i] * 0.8; // Highly correlated
        }
        
        // Add joint extreme events
        returns_a[100] = -0.15;
        returns_b[100] = -0.18;
        returns_a[200] = -0.12;
        returns_b[200] = -0.14;
        returns_a[300] = -0.20;
        returns_b[300] = -0.22;
        
        let calc = TailDependenceCalculator::with_config(TailDependenceConfig {
            lower_threshold: 0.1,
            ..Default::default()
        });
        
        let result = calc.calculate_pairwise(&returns_a.view(), &returns_b.view()).unwrap();
        
        // Should detect some level of tail dependence
        assert!(result.lower_tail_dependence > 0.0);
    }
}
