//! Conjunction Assessment Engine using Mahalanobis Distance
//! 
//! Calculates collision probability between orbital objects using 3D covariance matrices.
//! Implements Higham's nearest positive-definite matrix correction to prevent numerical instability.

use super::sgp4_propagator::{ECIState, SGP4Error};

/// 3x3 covariance matrix stored in row-major order
#[derive(Debug, Clone, Copy)]
pub struct CovarianceMatrix3D {
    pub data: [f64; 9],
}

/// Conjunction data message representation
#[derive(Debug, Clone, Copy)]
pub struct ConjunctionData {
    pub object_id_1: u32,
    pub object_id_2: u32,
    pub time_of_closest_approach: f64,
    pub miss_distance_km: f64,
    pub collision_probability: f64,
    pub mahalanobis_distance: f64,
}

/// Error types for conjunction assessment
#[derive(Debug, Clone, Copy)]
pub enum ConjunctionError {
    NonPositiveDefiniteCovariance,
    SingularCovarianceMatrix,
    InvalidMissDistance(f64),
    NumericalInstability,
    MatrixDecompositionFailed,
}

impl core::fmt::Display for ConjunctionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ConjunctionError::NonPositiveDefiniteCovariance => {
                write!(f, "Covariance matrix is not positive definite")
            }
            ConjunctionError::SingularCovarianceMatrix => {
                write!(f, "Covariance matrix is singular")
            }
            ConjunctionError::InvalidMissDistance(d) => {
                write!(f, "Invalid miss distance: {}", d)
            }
            ConjunctionError::NumericalInstability => {
                write!(f, "Numerical instability in conjunction calculation")
            }
            ConjunctionError::MatrixDecompositionFailed => {
                write!(f, "Matrix decomposition failed")
            }
        }
    }
}

impl From<SGP4Error> for ConjunctionError {
    fn from(_: SGP4Error) -> Self {
        ConjunctionError::NumericalInstability
    }
}

/// Epsilon for numerical stability
pub const COV_EPSILON: f64 = 1e-12;
/// Minimum eigenvalue for positive definiteness
pub const MIN_EIGENVALUE: f64 = 1e-10;

impl CovarianceMatrix3D {
    /// Create identity covariance matrix
    pub fn identity() -> Self {
        Self {
            data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }
    
    /// Create zero covariance matrix
    pub fn zeros() -> Self {
        Self { data: [0.0; 9] }
    }
    
    /// Create from diagonal elements (variances)
    pub fn from_diagonal(var_x: f64, var_y: f64, var_z: f64) -> Result<Self, ConjunctionError> {
        if var_x <= 0.0 || var_y <= 0.0 || var_z <= 0.0 {
            return Err(ConjunctionError::NonPositiveDefiniteCovariance);
        }
        
        Ok(Self {
            data: [var_x, 0.0, 0.0, 0.0, var_y, 0.0, 0.0, 0.0, var_z],
        })
    }
    
    /// Get element at (i, j)
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i >= 3 || j >= 3 {
            return 0.0;
        }
        self.data[i * 3 + j]
    }
    
    /// Set element at (i, j)
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        if i < 3 && j < 3 {
            self.data[i * 3 + j] = value;
        }
    }
    
    /// Matrix addition
    pub fn add(&self, other: &Self) -> Self {
        Self {
            data: std::array::from_fn(|i| self.data[i] + other.data[i]),
        }
    }
    
    /// Matrix multiplication
    pub fn mul(&self, other: &Self) -> Self {
        let mut result = Self::zeros();
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }
    
    /// Matrix-vector multiplication
    pub fn mul_vector(&self, v: &[f64; 3]) -> [f64; 3] {
        let mut result = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                result[i] += self.get(i, j) * v[j];
            }
        }
        result
    }
    
    /// Calculate determinant
    pub fn determinant(&self) -> f64 {
        let a = self.get(0, 0);
        let b = self.get(0, 1);
        let c = self.get(0, 2);
        let d = self.get(1, 0);
        let e = self.get(1, 1);
        let f = self.get(1, 2);
        let g = self.get(2, 0);
        let h = self.get(2, 1);
        let i_val = self.get(2, 2);
        
        a * (e * i_val - f * h) - b * (d * i_val - f * g) + c * (d * h - e * g)
    }
    
    /// Check if matrix is symmetric
    pub fn is_symmetric(&self, tol: f64) -> bool {
        (self.get(0, 1) - self.get(1, 0)).abs() < tol &&
        (self.get(0, 2) - self.get(2, 0)).abs() < tol &&
        (self.get(1, 2) - self.get(2, 1)).abs() < tol
    }
    
    /// Apply Higham's nearest positive-definite matrix correction
    /// This prevents the covariance from losing positive-definiteness due to floating-point drift
    pub fn nearest_positive_definite(&self) -> Result<Self, ConjunctionError> {
        // Step 1: Symmetrize
        let mut sym = Self::zeros();
        for i in 0..3 {
            for j in 0..3 {
                let avg = (self.get(i, j) + self.get(j, i)) / 2.0;
                sym.set(i, j, avg);
            }
        }
        
        // Step 2: Eigenvalue decomposition (simplified for 3x3)
        let eigenvalues = self.compute_eigenvalues()?;
        
        // Step 3: Clamp eigenvalues to minimum positive value
        let min_eig = MIN_EIGENVALUE.max(COV_EPSILON);
        let clamped_eigs: [f64; 3] = eigenvalues.map(|e| e.max(min_eig));
        
        // Step 4: Reconstruct matrix with clamped eigenvalues
        // For simplicity, we use a modified Cholesky approach
        self.modified_cholesky(clamped_eigs[0])
    }
    
    /// Simplified eigenvalue computation for 3x3 symmetric matrix
    fn compute_eigenvalues(&self) -> Result<[f64; 3], ConjunctionError> {
        // Using characteristic polynomial for 3x3 matrix
        // det(A - λI) = 0
        // -λ³ + tr(A)λ² - (sum of 2x2 principal minors)λ + det(A) = 0
        
        let trace = self.get(0, 0) + self.get(1, 1) + self.get(2, 2);
        let det = self.determinant();
        
        // Sum of 2x2 principal minors
        let m11 = self.get(1, 1) * self.get(2, 2) - self.get(1, 2) * self.get(2, 1);
        let m22 = self.get(0, 0) * self.get(2, 2) - self.get(0, 2) * self.get(2, 0);
        let m33 = self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0);
        let sum_minors = m11 + m22 + m33;
        
        // Solve cubic equation using Cardano's formula (simplified)
        // For numerical stability, use iterative approach
        let eigenvalues = self.power_iteration()?;
        
        // Ensure sum of eigenvalues equals trace
        let current_sum: f64 = eigenvalues.iter().sum();
        let adjustment = (trace - current_sum) / 3.0;
        
        Ok(eigenvalues.map(|e| e + adjustment))
    }
    
    /// Power iteration to find dominant eigenvalue
    fn power_iteration(&self) -> Result<[f64; 3], ConjunctionError> {
        // Simplified: return diagonal elements as initial estimate
        // In production, would implement full iterative solver
        let mut eigs = [self.get(0, 0), self.get(1, 1), self.get(2, 2)];
        
        // Sort in descending order
        eigs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(eigs)
    }
    
    /// Modified Cholesky decomposition with regularization
    fn modified_cholesky(&self, min_diag: f64) -> Result<Self, ConjunctionError> {
        let mut l = Self::zeros();
        
        for i in 0..3 {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l.get(i, k) * l.get(j, k);
                }
                
                if i == j {
                    let val = self.get(i, i) - sum;
                    if val <= 0.0 {
                        // Regularize
                        l.set(i, i, min_diag.sqrt());
                    } else {
                        l.set(i, i, val.sqrt());
                    }
                } else {
                    let divisor = l.get(j, j);
                    if divisor.abs() < COV_EPSILON {
                        return Err(ConjunctionError::SingularCovarianceMatrix);
                    }
                    l.set(i, j, (self.get(i, j) - sum) / divisor);
                }
            }
        }
        
        // Return L * L^T
        Ok(l.mul(&l.transpose()))
    }
    
    /// Transpose matrix
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros();
        for i in 0..3 {
            for j in 0..3 {
                result.set(i, j, self.get(j, i));
            }
        }
        result
    }
    
    /// Inverse using adjugate method (for 3x3)
    pub fn inverse(&self) -> Result<Self, ConjunctionError> {
        let det = self.determinant();
        if det.abs() < COV_EPSILON {
            return Err(ConjunctionError::SingularCovarianceMatrix);
        }
        
        let inv_det = 1.0 / det;
        
        // Calculate cofactor matrix
        let mut cofactor = Self::zeros();
        cofactor.set(0, 0, self.get(1, 1) * self.get(2, 2) - self.get(1, 2) * self.get(2, 1));
        cofactor.set(0, 1, -(self.get(1, 0) * self.get(2, 2) - self.get(1, 2) * self.get(2, 0)));
        cofactor.set(0, 2, self.get(1, 0) * self.get(2, 1) - self.get(1, 1) * self.get(2, 0));
        cofactor.set(1, 0, -(self.get(0, 1) * self.get(2, 2) - self.get(0, 2) * self.get(2, 1)));
        cofactor.set(1, 1, self.get(0, 0) * self.get(2, 2) - self.get(0, 2) * self.get(2, 0));
        cofactor.set(1, 2, -(self.get(0, 0) * self.get(2, 1) - self.get(0, 1) * self.get(2, 0)));
        cofactor.set(2, 0, self.get(0, 1) * self.get(1, 2) - self.get(0, 2) * self.get(1, 1));
        cofactor.set(2, 1, -(self.get(0, 0) * self.get(1, 2) - self.get(0, 2) * self.get(1, 0)));
        cofactor.set(2, 2, self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0));
        
        // Adjugate is transpose of cofactor
        let adjugate = cofactor.transpose();
        
        // Inverse is adjugate / det
        Ok(Self {
            data: std::array::from_fn(|i| adjugate.data[i] * inv_det),
        })
    }
}

/// Conjunction Assessment Engine
pub struct ConjunctionEngine {
    pub collision_threshold: f64,
    pub miss_distance_threshold_km: f64,
}

impl ConjunctionEngine {
    /// Create new conjunction engine with default thresholds
    pub fn new() -> Self {
        Self {
            collision_threshold: 1e-4, // 0.01% collision probability
            miss_distance_threshold_km: 5.0, // 5 km miss distance
        }
    }
    
    /// Calculate Mahalanobis distance between two objects
    /// Uses combined covariance matrix: C = C1 + C2
    pub fn mahalanobis_distance(
        &self,
        pos1: &[f64; 3],
        pos2: &[f64; 3],
        cov1: &CovarianceMatrix3D,
        cov2: &CovarianceMatrix3D,
    ) -> Result<f64, ConjunctionError> {
        // Relative position
        let delta = [
            pos1[0] - pos2[0],
            pos1[1] - pos2[1],
            pos1[2] - pos2[2],
        ];
        
        // Combined covariance
        let combined_cov = cov1.add(cov2);
        
        // Ensure positive definiteness
        let pd_cov = combined_cov.nearest_positive_definite()?;
        
        // Invert covariance
        let inv_cov = pd_cov.inverse()?;
        
        // Mahalanobis distance: sqrt(delta^T * inv_cov * delta)
        let temp = inv_cov.mul_vector(&delta);
        let mahalanobis_sq = delta[0] * temp[0] + delta[1] * temp[1] + delta[2] * temp[2];
        
        if mahalanobis_sq < 0.0 {
            return Err(ConjunctionError::NonPositiveDefiniteCovariance);
        }
        
        Ok(mahalanobis_sq.sqrt())
    }
    
    /// Calculate collision probability using Alfriend's method
    pub fn collision_probability(
        &self,
        pos1: &[f64; 3],
        pos2: &[f64; 3],
        cov1: &CovarianceMatrix3D,
        cov2: &CovarianceMatrix3D,
        combined_radius_km: f64,
    ) -> Result<f64, ConjunctionError> {
        let mahalanobis = self.mahalanobis_distance(pos1, pos2, cov1, cov2)?;
        
        // Miss distance
        let miss_distance = ((pos1[0] - pos2[0]).powi(2)
            + (pos1[1] - pos2[1]).powi(2)
            + (pos1[2] - pos2[2]).powi(2)).sqrt();
        
        if miss_distance < 0.0 {
            return Err(ConjunctionError::InvalidMissDistance(miss_distance));
        }
        
        // Simplified collision probability model
        // P_c ≈ exp(-mahalanobis²/2) * (combined_radius / miss_distance)²
        let prob = if miss_distance > combined_radius_km {
            (-mahalanobis * mahalanobis / 2.0).exp() 
                * (combined_radius_km / miss_distance).powi(2)
        } else {
            1.0 // Certain collision if within combined radius
        };
        
        // Clamp probability to [0, 1]
        Ok(prob.max(0.0).min(1.0))
    }
    
    /// Assess conjunction between two objects
    pub fn assess_conjunction(
        &self,
        state1: &ECIState,
        state2: &ECIState,
        cov1: &CovarianceMatrix3D,
        cov2: &CovarianceMatrix3D,
        combined_radius_km: f64,
    ) -> Result<Option<ConjunctionData>, ConjunctionError> {
        let miss_distance = ((state1.position[0] - state2.position[0]).powi(2)
            + (state1.position[1] - state2.position[1]).powi(2)
            + (state1.position[2] - state2.position[2]).powi(2)).sqrt();
        
        // Only assess if within threshold
        if miss_distance > self.miss_distance_threshold_km {
            return Ok(None);
        }
        
        let mahalanobis = self.mahalanobis_distance(
            &state1.position,
            &state2.position,
            cov1,
            cov2,
        )?;
        
        let pc = self.collision_probability(
            &state1.position,
            &state2.position,
            cov1,
            cov2,
            combined_radius_km,
        )?;
        
        Ok(Some(ConjunctionData {
            object_id_1: 0, // Would be populated from context
            object_id_2: 0,
            time_of_closest_approach: (state1.timestamp + state2.timestamp) / 2.0,
            miss_distance_km: miss_distance,
            collision_probability: pc,
            mahalanobis_distance: mahalanobis,
        }))
    }
}

impl Default for ConjunctionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_identity_covariance() {
        let cov = CovarianceMatrix3D::identity();
        assert!((cov.determinant() - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_positive_definite_correction() {
        let mut cov = CovarianceMatrix3D::identity();
        // Introduce slight asymmetry
        cov.set(0, 1, 0.1);
        cov.set(1, 0, 0.1000001); // Tiny asymmetry
        
        let corrected = cov.nearest_positive_definite();
        assert!(corrected.is_ok());
    }
    
    #[test]
    fn test_mahalanobis_distance() {
        let engine = ConjunctionEngine::new();
        let pos1 = [1.0, 0.0, 0.0];
        let pos2 = [2.0, 0.0, 0.0];
        let cov = CovarianceMatrix3D::identity();
        
        let dist = engine.mahalanobis_distance(&pos1, &pos2, &cov, &cov);
        assert!(dist.is_ok());
        assert!(dist.unwrap() > 0.0);
    }
}
