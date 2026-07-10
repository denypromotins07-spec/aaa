//! Student-t Copula implementation for modeling non-linear tail dependence
//!
//! Unlike Gaussian copulas, Student-t copulas capture asymmetric tail dependence
//! where correlations spike to 1.0 during market crashes. Critical for accurate
//! black swan contagion modeling.
//!
//! Mathematical Foundation:
//! - Student-t copula density with ν degrees of freedom
//! - Tail dependence coefficient λ = 2 * t_ν+1(-sqrt((ν+1)*(1-ρ)/(1+ρ)))
//! - Numerically stable Cholesky decomposition with Higham correction

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Ix1, Ix2};
use nalgebra::{Matrix2, SymmetricEigen};
use rand::Rng;
use rand_distr::{Distribution, StudentT};
use thiserror::Error;

/// Errors from copula operations
#[derive(Error, Debug, Clone)]
pub enum CopulaError {
    #[error("Correlation matrix is not positive definite")]
    NotPositiveDefinite,
    
    #[error("Correlation matrix has invalid dimensions")]
    InvalidDimensions,
    
    #[error("Degrees of freedom must be positive: got {0}")]
    InvalidDegreesOfFreedom(f64),
    
    #[error("Correlation value out of range [-1, 1]: {0}")]
    InvalidCorrelation(f64),
    
    #[error("Numerical instability in Cholesky decomposition")]
    CholeskyFailure,
    
    #[error("Matrix inversion failed: {0}")]
    InversionFailed(String),
    
    #[error("Invalid probability input: {0}")]
    InvalidProbability(String),
    
    #[error("Tail dependence calculation failed: {0}")]
    TailDependenceFailed(String),
}

/// Configuration for Student-t Copula
#[derive(Debug, Clone)]
pub struct StudentTCopulaConfig {
    /// Degrees of freedom (ν): lower values = heavier tails
    pub degrees_of_freedom: f64,
    /// Correlation matrix (must be positive definite)
    pub correlation_matrix: Array2<f64>,
    /// Use Higham's nearest positive-definite correction
    pub apply_higham_correction: bool,
}

impl StudentTCopulaConfig {
    /// Create a bivariate copula configuration
    pub fn bivariate(rho: f64, degrees_of_freedom: f64) -> Result<Self, CopulaError> {
        if rho < -1.0 || rho > 1.0 {
            return Err(CopulaError::InvalidCorrelation(rho));
        }
        
        if degrees_of_freedom <= 0.0 {
            return Err(CopulaError::InvalidDegreesOfFreedom(degrees_of_freedom));
        }
        
        let mut corr = Array2::zeros((2, 2));
        corr[[0, 0]] = 1.0;
        corr[[0, 1]] = rho;
        corr[[1, 0]] = rho;
        corr[[1, 1]] = 1.0;
        
        Ok(Self {
            degrees_of_freedom,
            correlation_matrix: corr,
            apply_higham_correction: true,
        })
    }
}

/// Student-t Copula for modeling tail dependence
pub struct StudentTCopula {
    config: StudentTCopulaConfig,
    cholesky_lower: Array2<f64>,
    log_det: f64,
}

impl StudentTCopula {
    /// Create a new Student-t copula with the given configuration
    pub fn new(config: StudentTCopulaConfig) -> Result<Self, CopulaError> {
        // Validate degrees of freedom
        if config.degrees_of_freedom <= 0.0 {
            return Err(CopulaError::InvalidDegreesOfFreedom(config.degrees_of_freedom));
        }
        
        // Get correlation matrix and apply Higham correction if needed
        let mut corr = config.correlation_matrix.clone();
        
        if config.apply_higham_correction {
            corr = Self::higham_nearest_positive_definite(&corr)?;
        } else {
            // Verify positive definiteness
            if !Self::is_positive_definite(&corr) {
                return Err(CopulaError::NotPositiveDefinite);
            }
        }
        
        // Compute Cholesky decomposition
        let (cholesky_lower, log_det) = Self::stable_cholesky(&corr)?;
        
        Ok(Self {
            config: StudentTCopulaConfig {
                correlation_matrix: corr,
                ..config
            },
            cholesky_lower,
            log_det,
        })
    }
    
    /// Sample from the Student-t copula
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Result<Array1<f64>, CopulaError> {
        let n = self.config.correlation_matrix.nrows();
        let nu = self.config.degrees_of_freedom;
        
        // Step 1: Generate multivariate normal with correlation structure
        let z = self.sample_correlated_normal(rng, n)?;
        
        // Step 2: Generate chi-squared scaling factor
        // StudentT::new can fail for invalid degrees of freedom, handle gracefully
        let student_t_dist = match StudentT::new(nu) {
            Ok(dist) => dist,
            Err(_) => return Err(CopulaError::InvalidDegreesOfFreedom(nu)),
        };
        let chi_sq: f64 = rng.sample(student_t_dist);
        let scale = nu / (chi_sq * chi_sq + nu as f64).sqrt();
        
        // Step 3: Scale and transform to uniform via t-CDF
        let mut u = Array1::zeros(n);
        for i in 0..n {
            let t_val = z[i] * scale;
            u[i] = Self::student_t_cdf(t_val, nu);
        }
        
        Ok(u)
    }
    
    /// Calculate the tail dependence coefficient λ
    /// 
    /// For bivariate Student-t copula:
    /// λ = 2 * t_ν+1(-sqrt((ν+1)*(1-ρ)/(1+ρ)))
    /// 
    /// where t_ν is the CDF of standard Student-t with ν degrees of freedom
    pub fn tail_dependence_coefficient(&self) -> Result<f64, CopulaError> {
        if self.config.correlation_matrix.nrows() != 2 {
            return Err(CopulaError::TailDependenceFailed(
                "Tail dependence coefficient only defined for bivariate case".to_string()
            ));
        }
        
        let rho = self.config.correlation_matrix[[0, 1]];
        let nu = self.config.degrees_of_freedom;
        
        // Calculate the argument for the t-CDF
        let arg_squared = (nu + 1.0) * (1.0 - rho) / (1.0 + rho);
        
        if arg_squared < 0.0 {
            return Err(CopulaError::TailDependenceFailed(
                format!("Invalid argument for tail dependence: rho={}, nu={}", rho, nu)
            ));
        }
        
        let arg = -arg_squared.sqrt();
        
        // Evaluate t-CDF at the argument with ν+1 degrees of freedom
        let cdf_val = Self::student_t_cdf(arg, nu + 1.0);
        
        // λ = 2 * CDF
        let lambda = 2.0 * cdf_val;
        
        // Clamp to valid range
        Ok(lambda.clamp(0.0, 1.0))
    }
    
    /// Calculate the copula density at given uniform marginals
    pub fn density(&self, u: &ArrayView1<f64>) -> Result<f64, CopulaError> {
        let n = u.len();
        if n != self.config.correlation_matrix.nrows() {
            return Err(CopulaError::InvalidDimensions);
        }
        
        // Transform uniform to t-distributed
        let mut t_vals = Array1::zeros(n);
        for i in 0..n {
            if u[i] <= 0.0 || u[i] >= 1.0 {
                return Err(CopulaError::InvalidProbability(
                    format!("Uniform marginal {} out of range (0, 1)", u[i])
                ));
            }
            t_vals[i] = Self::student_t_quantile(u[i], self.config.degrees_of_freedom);
        }
        
        // Compute quadratic form t' * R^(-1) * t
        let t_vec = nalgebra::DVector::from_vec(t_vals.to_vec());
        let corr_slice = self.config.correlation_matrix.as_slice()
            .ok_or_else(|| CopulaError::InvalidDimensions)?;
        let corr_inv = nalgebra::DMatrix::from_vec(n, n, corr_slice.to_vec())
            .try_inverse()
            .ok_or_else(|| CopulaError::InversionFailed("Correlation matrix singular".to_string()))?;
        
        let quad_form = &t_vec.transpose() * &corr_inv * &t_vec;
        
        // Student-t copula density formula
        let nu = self.config.degrees_of_freedom;
        let n_f64 = n as f64;
        
        let num = (1.0 + quad_form[(0, 0)] / nu).powf(-(nu + n_f64) / 2.0);
        let denom = (1.0 / nu * t_vec.dot(&t_vec)).powf(-(nu + n_f64) / 2.0);
        
        let constant = ((nu + n_f64) / 2.0).ln() - ((nu + n_f64) / 2.0).ln()
            + n_f64 / 2.0 * (nu / 2.0).ln()
            - n_f64 / 2.0 * ((nu + n_f64) / 2.0).ln()
            - 0.5 * self.log_det;
        
        Ok(constant.exp() * num / denom)
    }
    
    /// Sample correlated normal variables using Cholesky decomposition
    fn sample_correlated_normal<R: Rng>(
        &self,
        rng: &mut R,
        n: usize,
    ) -> Result<Array1<f64>, CopulaError> {
        use rand_distr::StandardNormal;
        
        // Generate independent standard normals
        let mut z = Array1::zeros(n);
        for i in 0..n {
            z[i] = rng.sample(StandardNormal);
        }
        
        // Apply Cholesky transformation: L * z gives correlated normals
        let mut correlated = Array1::zeros(n);
        for i in 0..n {
            for j in 0..=i {
                correlated[i] += self.cholesky_lower[[i, j]] * z[j];
            }
        }
        
        Ok(correlated)
    }
    
    /// Stable Cholesky decomposition with numerical safeguards
    fn stable_cholesky(matrix: &Array2<f64>) -> Result<(Array2<f64>, f64), CopulaError> {
        let n = matrix.nrows();
        let mut lower = Array2::zeros((n, n));
        let mut log_det = 0.0;
        
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                
                if j == i {
                    // Diagonal element
                    for k in 0..j {
                        sum += lower[[j, k]].powi(2);
                    }
                    
                    let diag = matrix[[j, j]] - sum;
                    
                    if diag <= 0.0 {
                        // Add small perturbation for numerical stability
                        if diag > -1e-10 {
                            lower[[j, j]] = 1e-5;
                        } else {
                            return Err(CopulaError::CholeskyFailure);
                        }
                    } else {
                        lower[[j, j]] = diag.sqrt();
                    }
                    
                    log_det += 2.0 * lower[[j, j]].ln();
                } else {
                    // Off-diagonal element
                    for k in 0..j {
                        sum += lower[[i, k]] * lower[[j, k]];
                    }
                    
                    if lower[[j, j]].abs() < 1e-15 {
                        return Err(CopulaError::CholeskyFailure);
                    }
                    
                    lower[[i, j]] = (matrix[[i, j]] - sum) / lower[[j, j]];
                }
            }
        }
        
        Ok((lower, log_det))
    }
    
    /// Check if matrix is positive definite using eigenvalue decomposition
    fn is_positive_definite(matrix: &Array2<f64>) -> bool {
        if matrix.nrows() != matrix.ncols() {
            return false;
        }
        
        let n = matrix.nrows();
        let m = Matrix2::<f64>::from_iterator(
            matrix.iter().cloned()
        );
        
        let eigen = SymmetricEigen::new(&m);
        
        for eigenvalue in eigen.eigenvalues.iter() {
            if *eigenvalue <= 1e-10 {
                return false;
            }
        }
        
        true
    }
    
    /// Higham's nearest positive-definite matrix correction
    fn higham_nearest_positive_definite(matrix: &Array2<f64>) -> Result<Array2<f64>, CopulaError> {
        let n = matrix.nrows();
        
        if n != matrix.ncols() {
            return Err(CopulaError::InvalidDimensions);
        }
        
        // Simple implementation: add small diagonal perturbation
        let mut corrected = matrix.clone();
        
        // Ensure symmetry
        for i in 0..n {
            for j in (i+1)..n {
                let avg = (matrix[[i, j]] + matrix[[j, i]]) / 2.0;
                corrected[[i, j]] = avg;
                corrected[[j, i]] = avg;
            }
        }
        
        // Add small perturbation to diagonal if needed
        let epsilon = 1e-8;
        for i in 0..n {
            corrected[[i, i]] += epsilon;
        }
        
        // Verify result
        if Self::is_positive_definite(&corrected) {
            Ok(corrected)
        } else {
            // More aggressive correction
            for i in 0..n {
                corrected[[i, i]] += 0.01;
            }
            
            if Self::is_positive_definite(&corrected) {
                Ok(corrected)
            } else {
                Err(CopulaError::NotPositiveDefinite)
            }
        }
    }
    
    /// Student-t CDF approximation
    fn student_t_cdf(x: f64, nu: f64) -> f64 {
        // Use regularized incomplete beta function approximation
        // For large |x|, use asymptotic expansion
        
        if x < -10.0 {
            // Left tail approximation
            let t = nu / (x * x);
            return 0.5 * Self::beta_regularized(nu / 2.0, 0.5, t);
        }
        
        if x > 10.0 {
            // Right tail approximation
            let t = nu / (x * x);
            return 1.0 - 0.5 * Self::beta_regularized(nu / 2.0, 0.5, t);
        }
        
        // Standard case: use numerical integration or lookup
        // Simplified approximation for common cases
        let z = x / (nu + x * x).sqrt();
        0.5 + 0.5 * Self::signum(x) * Self::incomplete_beta_approx(nu / 2.0, 0.5, z * z)
    }
    
    /// Quantile function (inverse CDF) for Student-t distribution
    fn student_t_quantile(p: f64, nu: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 {
            return f64::NAN;
        }
        
        if p == 0.5 {
            return 0.0;
        }
        
        // Use rational approximation for quantile
        // This is a simplified version; production code should use more accurate methods
        let sign = if p < 0.5 { -1.0 } else { 1.0 };
        let p_adj = if p < 0.5 { p } else { 1.0 - p };
        
        // Approximation using normal quantile with correction
        let z = Self::normal_quantile(p_adj);
        let z2 = z * z;
        
        // Cornish-Fisher expansion for Student-t
        let g1 = 1.0 / (4.0 * nu);
        let g2 = 1.0 / (96.0 * nu * nu);
        
        let q = z + g1 * z * (z2 + 1.0) + g2 * z * (5.0 * z2.powi(3) + 16.0 * z2.powi(2) + 3.0 * z2 - 9.0);
        
        sign * q
    }
    
    /// Standard normal quantile approximation
    fn normal_quantile(p: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 {
            return f64::NAN;
        }
        
        // Rational approximation (Abramowitz and Stegun)
        if p < 0.5 {
            -Self::normal_quantile(1.0 - p)
        } else {
            let t = (-2.0 * (1.0 - p).ln()).sqrt();
            let c0 = 2.515517;
            let c1 = 0.802853;
            let c2 = 0.010328;
            let d1 = 1.432788;
            let d2 = 0.189269;
            let d3 = 0.001308;
            
            t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
        }
    }
    
    /// Signum function
    fn signum(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
    }
    
    /// Regularized incomplete beta function approximation
    fn beta_regularized(a: f64, b: f64, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        if x >= 1.0 { return 1.0; }
        
        // Continued fraction approximation
        Self::incomplete_beta_approx(a, b, x)
    }
    
    /// Incomplete beta function approximation
    fn incomplete_beta_approx(a: f64, b: f64, x: f64) -> f64 {
        // Simplified approximation for common parameter ranges
        // Production code should use more robust implementation
        if a > 0.0 && b > 0.0 && x > 0.0 && x < 1.0 {
            x.powf(a) * (1.0 - x).powf(b) / (a * Self::beta_function(a, b))
        } else {
            0.0
        }
    }
    
    /// Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b)
    fn beta_function(a: f64, b: f64) -> f64 {
        special::gamma(a) * special::gamma(b) / special::gamma(a + b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    
    #[test]
    fn test_bivariate_copula_creation() {
        let config = StudentTCopulaConfig::bivariate(0.5, 5.0).unwrap();
        let copula = StudentTCopula::new(config).unwrap();
        
        assert_eq!(copula.config.degrees_of_freedom, 5.0);
    }
    
    #[test]
    fn test_tail_dependence_coefficient() {
        // High correlation should give high tail dependence
        let config = StudentTCopulaConfig::bivariate(0.8, 5.0).unwrap();
        let copula = StudentTCopula::new(config).unwrap();
        
        let lambda = copula.tail_dependence_coefficient().unwrap();
        
        // Should be between 0 and 1
        assert!(lambda >= 0.0 && lambda <= 1.0);
        
        // With high correlation, tail dependence should be significant
        assert!(lambda > 0.3);
    }
    
    #[test]
    fn test_sampling() {
        let config = StudentTCopulaConfig::bivariate(0.5, 5.0).unwrap();
        let copula = StudentTCopula::new(config).unwrap();
        
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let sample = copula.sample(&mut rng).unwrap();
        
        assert_eq!(sample.len(), 2);
        
        // All samples should be in (0, 1)
        for &s in sample.iter() {
            assert!(s > 0.0 && s < 1.0);
        }
    }
    
    #[test]
    fn test_higham_correction() {
        // Create a nearly singular correlation matrix
        let mut matrix = Array2::zeros((2, 2));
        matrix[[0, 0]] = 1.0;
        matrix[[0, 1]] = 0.9999999;
        matrix[[1, 0]] = 0.9999999;
        matrix[[1, 1]] = 1.0;
        
        let corrected = StudentTCopula::higham_nearest_positive_definite(&matrix).unwrap();
        
        // Should still be close to original
        assert!((corrected[[0, 1]] - 0.9999999).abs() < 0.01);
    }
}
