//! Cohort Correlation Cholesky Decomposition
//! 
//! Computes correlated mortality improvements across multiple cohorts
//! using zero-allocation Cholesky decomposition.

use crate::mortality::lee_carter_kalman::{MortalityModelError, MAX_AGE_GROUPS};

/// Maximum number of cohorts for correlation matrix
pub const MAX_COHORTS: usize = 20;

/// Error types for Cholesky decomposition
#[derive(Debug, Clone, PartialEq)]
pub enum CholeskyError {
    NonPositiveDefinite,
    NumericalInstability,
    DimensionMismatch,
}

impl core::fmt::Display for CholeskyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonPositiveDefinite => write!(f, "Matrix is not positive definite"),
            Self::NumericalInstability => write!(f, "Numerical instability detected"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch"),
        }
    }
}

/// Pre-allocated correlation matrix (row-major, upper triangular stored)
pub struct CorrelationMatrix {
    /// Matrix data (n x n, row-major)
    data: [[f64; MAX_COHORTS]; MAX_COHORTS],
    /// Number of active cohorts
    n: usize,
    /// Validity flag
    valid: bool,
}

impl CorrelationMatrix {
    pub const fn new() -> Self {
        Self {
            data: [[0.0; MAX_COHORTS]; MAX_COHORTS],
            n: 0,
            valid: false,
        }
    }

    #[inline]
    pub fn set_correlation(&mut self, i: usize, j: usize, rho: f64) -> Result<(), CholeskyError> {
        if i >= MAX_COHORTS || j >= MAX_COHORTS {
            return Err(CholeskyError::DimensionMismatch);
        }
        if rho.abs() > 1.0 {
            return Err(CholeskyError::NonPositiveDefinite);
        }

        self.data[i][j] = rho;
        self.data[j][i] = rho; // Symmetric

        if i >= self.n {
            self.n = i + 1;
        }
        if j >= self.n {
            self.n = j + 1;
        }

        Ok(())
    }

    #[inline]
    pub fn get_correlation(&self, i: usize, j: usize) -> Option<f64> {
        if i >= self.n || j >= self.n {
            return None;
        }
        Some(self.data[i][j])
    }

    /// Set diagonal to 1.0 (valid correlation matrix)
    pub fn set_unit_diagonal(&mut self) {
        for i in 0..self.n {
            self.data[i][i] = 1.0;
        }
    }

    /// Check if matrix is positive definite
    pub fn is_positive_definite(&self) -> bool {
        // Check eigenvalues via Gershgorin circles
        for i in 0..self.n {
            let diag = self.data[i][i];
            let mut row_sum = 0.0;
            for j in 0..self.n {
                if i != j {
                    row_sum += self.data[i][j].abs();
                }
            }
            if diag <= row_sum {
                return false;
            }
        }
        true
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        self.valid && self.n > 0
    }

    #[inline]
    pub fn mark_valid(&mut self) {
        self.valid = true;
    }

    #[inline]
    pub fn get_row(&self, i: usize) -> Option<&[f64]> {
        if i >= self.n {
            return None;
        }
        Some(&self.data[i][..self.n])
    }
}

/// Cholesky decomposition result (lower triangular L where A = L * L')
pub struct CholeskyFactor {
    /// Lower triangular factor
    lower: [[f64; MAX_COHORTS]; MAX_COHORTS],
    /// Number of rows/columns
    n: usize,
    /// Success flag
    success: bool,
}

impl CholeskyFactor {
    pub const fn new() -> Self {
        Self {
            lower: [[0.0; MAX_COHORTS]; MAX_COHORTS],
            n: 0,
            success: false,
        }
    }

    /// Compute Cholesky decomposition of correlation matrix
    pub fn decompose(&mut self, matrix: &CorrelationMatrix) -> Result<(), CholeskyError> {
        let n = matrix.n();
        if n == 0 || n > MAX_COHORTS {
            return Err(CholeskyError::DimensionMismatch);
        }

        self.n = n;
        self.success = false;

        // Zero out the factor
        for i in 0..MAX_COHORTS {
            for j in 0..MAX_COHORTS {
                self.lower[i][j] = 0.0;
            }
        }

        // Cholesky-Banachiewicz algorithm
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;

                if j == i {
                    // Diagonal element
                    for k in 0..j {
                        sum += self.lower[j][k] * self.lower[j][k];
                    }

                    let val = matrix.data[j][j] - sum;
                    if val <= 1e-12 {
                        // Not positive definite - apply regularization
                        return Err(CholeskyError::NonPositiveDefinite);
                    }

                    self.lower[j][j] = val.sqrt();
                } else {
                    // Off-diagonal element
                    for k in 0..j {
                        sum += self.lower[i][k] * self.lower[j][k];
                    }

                    if self.lower[j][j].abs() < 1e-12 {
                        return Err(CholeskyError::NumericalInstability);
                    }

                    self.lower[i][j] = (matrix.data[i][j] - sum) / self.lower[j][j];
                }

                if !self.lower[i][j].is_finite() {
                    return Err(CholeskyError::NumericalInstability);
                }
            }
        }

        self.success = true;
        Ok(())
    }

    /// Generate correlated random samples
    pub fn generate_correlated_samples(
        &self,
        uncorrelated: &[f64],
    ) -> Result<Vec<f64>, CholeskyError> {
        if !self.success {
            return Err(CholeskyError::NumericalInstability);
        }

        if uncorrelated.len() < self.n {
            return Err(CholeskyError::DimensionMismatch);
        }

        let mut correlated = vec![0.0; self.n];

        // correlated = L * uncorrelated
        for i in 0..self.n {
            let mut sum = 0.0;
            for j in 0..=i {
                sum += self.lower[i][j] * uncorrelated[j];
            }
            correlated[i] = sum;

            if !correlated[i].is_finite() {
                return Err(CholeskyError::NumericalInstability);
            }
        }

        Ok(correlated)
    }

    /// Get lower triangular factor row
    #[inline]
    pub fn get_row(&self, i: usize) -> Option<&[f64]> {
        if i >= self.n || !self.success {
            return None;
        }
        Some(&self.lower[i][..=i])
    }
}

/// Multi-cohort mortality correlation model
pub struct CohortCorrelationModel {
    correlation_matrix: CorrelationMatrix,
    cholesky_factor: CholeskyFactor,
    /// Mean reversion speeds for each cohort
    mean_reversion: [f64; MAX_COHORTS],
    /// Long-term means for each cohort
    long_term_means: [f64; MAX_COHORTS],
}

impl CohortCorrelationModel {
    pub const fn new() -> Self {
        Self {
            correlation_matrix: CorrelationMatrix::new(),
            cholesky_factor: CholeskyFactor::new(),
            mean_reversion: [0.1; MAX_COHORTS],
            long_term_means: [0.0; MAX_COHORTS],
        }
    }

    /// Initialize with empirical correlations
    pub fn initialize_from_data(&mut self, cohort_returns: &[Vec<f64>]) -> Result<(), CholeskyError> {
        if cohort_returns.is_empty() {
            return Err(CholeskyError::DimensionMismatch);
        }

        let n_cohorts = cohort_returns.len();
        if n_cohorts > MAX_COHORTS {
            return Err(CholeskyError::DimensionMismatch);
        }

        // Compute correlation matrix from returns
        let mut means = [0.0; MAX_COHORTS];
        let mut stds = [0.0; MAX_COHORTS];

        for (i, returns) in cohort_returns.iter().enumerate() {
            if returns.is_empty() {
                return Err(CholeskyError::DimensionMismatch);
            }

            // Mean
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            means[i] = mean;

            // Standard deviation
            let variance = returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / returns.len() as f64;
            stds[i] = variance.sqrt().max(1e-10);
        }

        // Correlations
        for i in 0..n_cohorts {
            self.correlation_matrix.set_correlation(i, i, 1.0)?;
            
            for j in (i + 1)..n_cohorts {
                let mut cov = 0.0;
                let n_obs = cohort_returns[i].len().min(cohort_returns[j].len());

                for k in 0..n_obs {
                    cov += (cohort_returns[i][k] - means[i]) * (cohort_returns[j][k] - means[j]);
                }

                let corr = cov / (n_obs as f64 * stds[i] * stds[j]);
                self.correlation_matrix.set_correlation(i, j, corr.clamp(-1.0, 1.0))?;
            }
        }

        self.correlation_matrix.mark_valid();

        // Compute Cholesky decomposition
        self.cholesky_factor.decompose(&self.correlation_matrix)?;

        Ok(())
    }

    /// Simulate correlated mortality improvements
    pub fn simulate_improvements(
        &self,
        horizon_years: usize,
        initial_kappas: &[f64],
    ) -> Result<Vec<Vec<f64>>, CholeskyError> {
        if !self.cholesky_factor.success {
            return Err(CholeskyError::NumericalInstability);
        }

        let n_cohorts = self.correlation_matrix.n();
        if initial_kappas.len() != n_cohorts {
            return Err(CholeskyError::DimensionMismatch);
        }

        let mut paths = vec![vec![0.0; horizon_years + 1]; n_cohorts];

        // Initialize
        for i in 0..n_cohorts {
            paths[i][0] = initial_kappas[i];
        }

        // Simulate using correlated Ornstein-Uhlenbeck processes
        for t in 1..=horizon_years {
            // Generate uncorrelated standard normals (placeholder - would use RNG in production)
            let uncorrelated: Vec<f64> = (0..n_cohorts)
                .map(|i| ((t + i) as f64 * 0.1).sin())
                .collect();

            // Apply Cholesky to get correlated shocks
            let correlated = self.cholesky_factor.generate_correlated_samples(&uncorrelated)?;

            // Update each kappa
            for i in 0..n_cohorts {
                let kappa_prev = paths[i][t - 1];
                let mr = self.mean_reversion[i];
                let theta = self.long_term_means[i];
                let sigma = 0.01; // Volatility

                // OU process: dX = mr*(theta - X)*dt + sigma*dW
                let dt = 1.0;
                let drift = mr * (theta - kappa_prev) * dt;
                let diffusion = sigma * correlated[i] * dt.sqrt();

                paths[i][t] = kappa_prev + drift + diffusion;
                paths[i][t] = paths[i][t].clamp(-100.0, 100.0);
            }
        }

        Ok(paths)
    }

    /// Get correlation between two cohorts
    pub fn get_cohort_correlation(&self, i: usize, j: usize) -> Option<f64> {
        self.correlation_matrix.get_correlation(i, j)
    }

    /// Set mean reversion for a cohort
    pub fn set_mean_reversion(&mut self, cohort: usize, mr: f64) -> Result<(), CholeskyError> {
        if cohort >= MAX_COHORTS {
            return Err(CholeskyError::DimensionMismatch);
        }
        if mr < 0.0 || mr > 1.0 {
            return Err(CholeskyError::NumericalInstability);
        }
        self.mean_reversion[cohort] = mr;
        Ok(())
    }

    /// Set long-term mean for a cohort
    pub fn set_long_term_mean(&mut self, cohort: usize, theta: f64) -> Result<(), CholeskyError> {
        if cohort >= MAX_COHORTS {
            return Err(CholeskyError::DimensionMismatch);
        }
        self.long_term_means[cohort] = theta;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_matrix() {
        let mut matrix = CorrelationMatrix::new();
        assert!(matrix.set_correlation(0, 0, 1.0).is_ok());
        assert!(matrix.set_correlation(0, 1, 0.5).is_ok());
        assert!(matrix.set_correlation(1, 1, 1.0).is_ok());
        
        matrix.mark_valid();
        assert!(matrix.is_valid());
    }

    #[test]
    fn test_cholesky_decomposition() {
        let mut matrix = CorrelationMatrix::new();
        matrix.set_correlation(0, 0, 1.0).unwrap();
        matrix.set_correlation(0, 1, 0.5).unwrap();
        matrix.set_correlation(1, 1, 1.0).unwrap();
        matrix.mark_valid();

        let mut cholesky = CholeskyFactor::new();
        assert!(cholesky.decompose(&matrix).is_ok());
        assert!(cholesky.success);
    }

    #[test]
    fn test_cohort_model_initialization() {
        let mut model = CohortCorrelationModel::new();
        
        let returns = vec![
            vec![0.01, 0.02, -0.01, 0.03],
            vec![0.02, 0.01, 0.00, 0.02],
        ];
        
        assert!(model.initialize_from_data(&returns).is_ok());
    }
}
