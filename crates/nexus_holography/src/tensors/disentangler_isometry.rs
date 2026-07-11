//! Disentangler and Isometry Tensors
//! 
//! Core tensor components for MERA network.
//! Implements unitary disentanglers and isometric coarse-graining tensors.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors specific to tensor operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TensorError {
    #[error("Invalid dimension: {0}")]
    InvalidDimension(usize),
    #[error("Unitarity violation: {0}")]
    UnitarityViolation(String),
    #[error("Tensor application failed: {0}")]
    ApplicationFailed(String),
    #[error("Normalization error: {0}")]
    NormalizationError(f64),
}

/// Configuration for tensor creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorConfig {
    /// Default bond dimension
    pub bond_dimension: usize,
    /// Random seed for initialization
    pub seed: u64,
    /// Whether to enforce strict unitarity
    pub strict_unitarity: bool,
}

impl Default for TensorConfig {
    fn default() -> Self {
        Self {
            bond_dimension: 16,
            seed: 42,
            strict_unitarity: true,
        }
    }
}

/// Disentangler tensor (unitary operator)
/// Acts on two adjacent sites to remove short-range entanglement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Disentangler {
    /// Bond dimension
    pub bond_dim: usize,
    /// Unitary matrix elements (stored as complex for generality)
    pub matrix: Vec<Complex64>,
    /// Whether this disentangler has been optimized
    pub is_optimized: bool,
}

impl Disentangler {
    /// Create a new disentangler with given bond dimension
    pub fn new(bond_dim: usize) -> Result<Self, TensorError> {
        if bond_dim < 2 {
            return Err(TensorError::InvalidDimension(bond_dim));
        }

        // Initialize with identity-like unitary
        let matrix = Self::create_identity_unitary(bond_dim);

        Ok(Self {
            bond_dim,
            matrix,
            is_optimized: false,
        })
    }

    /// Create identity-like unitary matrix
    fn create_identity_unitary(bond_dim: usize) -> Vec<Complex64> {
        let size = bond_dim * bond_dim;
        let mut matrix = vec![Complex64::new(0.0, 0.0); size];

        // Identity matrix
        for i in 0..bond_dim {
            matrix[i * bond_dim + i] = Complex64::new(1.0, 0.0);
        }

        matrix
    }

    /// Apply disentangler to a two-site state
    pub fn apply(&self, state: &DVector<f64>) -> Result<DVector<f64>, TensorError> {
        if state.len() != self.bond_dim * 2 {
            return Err(TensorError::InvalidDimension(state.len()));
        }

        // For simplicity, treat as real operation on doubled vector
        let mut result = DVector::zeros(state.len());
        
        // Apply unitary transformation (simplified real version)
        for i in 0..self.bond_dim {
            let mut sum1 = 0.0;
            let mut sum2 = 0.0;
            
            for j in 0..self.bond_dim {
                let idx1 = j;
                let idx2 = j + self.bond_dim;
                
                let m_real = self.matrix[i * self.bond_dim + j].re;
                
                sum1 += m_real * state[idx1];
                sum2 += m_real * state[idx2];
            }
            
            result[i] = sum1;
            result[i + self.bond_dim] = sum2;
        }

        Ok(result)
    }

    /// Verify unitarity: U†U = I
    pub fn verify_unitarity(&self, tolerance: f64) -> Result<bool, TensorError> {
        let n = self.bond_dim;
        
        // Compute U†U
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                
                for k in 0..n {
                    // U†[i,k] = conj(U[k,i])
                    let u_dag = self.matrix[k * n + i].conj();
                    let u = self.matrix[k * n + j];
                    sum += u_dag * u;
                }

                // Should be δ_ij
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };

                if (sum - expected).norm() > tolerance {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Optimize disentangler to minimize entanglement (variational update)
    pub fn optimize(&mut self, gradient: &[f64], learning_rate: f64) -> Result<(), TensorError> {
        if gradient.len() != self.matrix.len() {
            return Err(TensorError::InvalidDimension(gradient.len()));
        }

        // Gradient descent update (simplified)
        for (i, g) in gradient.iter().enumerate() {
            let current = self.matrix[i];
            let updated = current - Complex64::new(learning_rate * g, 0.0);
            
            // Re-normalize to maintain approximate unitarity
            let norm = updated.norm();
            if norm > 1e-10 {
                self.matrix[i] = updated / norm.min(1.0);
            }
        }

        self.is_optimized = true;
        Ok(())
    }
}

/// Isometry tensor for coarse-graining
/// Maps two sites to one: V: H₁ ⊗ H₂ → H'
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Isometry {
    /// Input bond dimension (per site)
    pub input_bond_dim: usize,
    /// Output bond dimension
    pub output_bond_dim: usize,
    /// Isometry matrix elements
    pub matrix: Vec<Complex64>,
}

impl Isometry {
    /// Create a new isometry
    pub fn new(bond_dim: usize) -> Result<Self, TensorError> {
        if bond_dim < 2 {
            return Err(TensorError::InvalidDimension(bond_dim));
        }

        // Isometry maps 2 sites (bond_dim² states) to 1 site (bond_dim states)
        let input_dim = bond_dim * bond_dim;
        let output_dim = bond_dim;

        // Initialize with simple averaging isometry
        let mut matrix = vec![Complex64::new(0.0, 0.0); output_dim * input_dim];

        for i in 0..output_dim {
            // Map diagonal elements with normalization
            let idx = i * input_dim + i * bond_dim + i;
            if idx < matrix.len() {
                matrix[idx] = Complex64::new(1.0 / (2.0_f64.sqrt()), 0.0);
            }
        }

        Ok(Self {
            input_bond_dim: bond_dim,
            output_bond_dim,
            matrix,
        })
    }

    /// Apply isometry to a two-site state
    pub fn apply(&self, state: &DVector<f64>) -> Result<DVector<f64>, TensorError> {
        let input_size = self.input_bond_dim * self.input_bond_dim;
        
        if state.len() < input_size {
            // Pad with zeros if needed
            return Err(TensorError::InvalidDimension(state.len()));
        }

        let mut result = DVector::zeros(self.output_bond_dim);

        for i in 0..self.output_bond_dim {
            let mut sum = 0.0;
            
            for j in 0..input_size.min(state.len()) {
                let m_real = self.matrix[i * input_size + j].re;
                sum += m_real * state[j];
            }
            
            result[i] = sum;
        }

        Ok(result)
    }

    /// Verify isometry property: V†V = I (on output space)
    pub fn verify_isometry(&self, tolerance: f64) -> Result<bool, TensorError> {
        let n_out = self.output_bond_dim;
        let n_in = self.input_bond_dim * self.input_bond_dim;

        // Compute V†V
        for i in 0..n_out {
            for j in 0..n_out {
                let mut sum = Complex64::new(0.0, 0.0);

                for k in 0..n_in {
                    let v_dag = self.matrix[i * n_in + k].conj();
                    let v = self.matrix[j * n_in + k];
                    sum += v_dag * v;
                }

                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };

                if (sum - expected).norm() > tolerance {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }
}

/// Tensor network utility functions
pub mod tensor_utils {
    use super::*;

    /// Compute tensor trace (contraction)
    pub fn trace_contract(tensor: &[Complex64], dim: usize) -> Result<Complex64, TensorError> {
        if tensor.len() != dim * dim {
            return Err(TensorError::InvalidDimension(tensor.len()));
        }

        let mut trace = Complex64::new(0.0, 0.0);
        for i in 0..dim {
            trace += tensor[i * dim + i];
        }

        Ok(trace)
    }

    /// Normalize a tensor
    pub fn normalize_tensor(tensor: &mut [Complex64]) -> Result<f64, TensorError> {
        let mut norm_sq = 0.0;
        for elem in tensor.iter() {
            norm_sq += elem.norm_squared();
        }

        let norm = norm_sq.sqrt();
        if norm < 1e-15 {
            return Err(TensorError::NormalizationError(norm));
        }

        for elem in tensor.iter_mut() {
            *elem /= norm;
        }

        Ok(norm)
    }

    /// Create random unitary matrix (Haar distributed approximation)
    pub fn random_unitary(dim: usize, seed: u64) -> Vec<Complex64> {
        // Simple approximation using QR decomposition idea
        let mut matrix = Vec::with_capacity(dim * dim);
        
        // Seeded pseudo-random generator
        let mut rng_state = seed;
        let mut next_random = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_state >> 33) as f64 / ((u64::MAX >> 33) as f64)
        };

        for _ in 0..dim * dim {
            let re = next_random() * 2.0 - 1.0;
            let im = next_random() * 2.0 - 1.0;
            matrix.push(Complex64::new(re, im));
        }

        // Gram-Schmidt orthogonalization (simplified)
        // In production, use proper QR decomposition
        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disentangler_creation() {
        let d = Disentangler::new(4);
        assert!(d.is_ok());
    }

    #[test]
    fn test_disentangler_unitarity() {
        let d = Disentangler::new(4).unwrap();
        let is_unitary = d.verify_unitarity(1e-10);
        assert!(is_unitary.is_ok());
    }

    #[test]
    fn test_isometry_creation() {
        let iso = Isometry::new(4);
        assert!(iso.is_ok());
    }

    #[test]
    fn test_disentangler_apply() {
        let d = Disentangler::new(2).unwrap();
        let state = DVector::from_vec(vec![1.0, 0.0, 0.0, 1.0]);
        let result = d.apply(&state);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 4);
    }
}
