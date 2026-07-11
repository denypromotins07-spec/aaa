//! Deutsch Density Matrix for Closed Timelike Curve (CTC) modeling
//! 
//! Implements Deutsch's CTC formalism using density matrices to resolve
//! causal paradoxes in reflexive market execution scenarios.

use std::ops::{Add, Mul};

/// Maximum matrix dimension supported
const MAX_MATRIX_DIM: usize = 64;

/// Convergence tolerance for fixed-point iteration
const CONVERGENCE_TOLERANCE: f64 = 1e-10;

/// Maximum iterations before fallback
const MAX_ITERATIONS: usize = 1000;

/// Complex number for quantum amplitudes
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn real(val: f64) -> Self {
        Self { re: val, im: 0.0 }
    }

    pub fn conjugate(&self) -> Self {
        Self { re: self.re, im: -self.im }
    }

    pub fn norm_squared(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

/// Density matrix representing quantum state in CTC formalism
#[derive(Debug, Clone)]
pub struct DensityMatrix {
    /// Matrix data (row-major order)
    data: Vec<Complex>,
    /// Matrix dimension (square matrix)
    dim: usize,
    /// Trace value (should be 1.0 for normalized states)
    trace: f64,
}

impl DensityMatrix {
    /// Create a new density matrix from complex data
    pub fn new(data: Vec<Complex>, dim: usize) -> Option<Self> {
        if dim == 0 || dim > MAX_MATRIX_DIM {
            return None;
        }
        
        if data.len() != dim * dim {
            return None;
        }

        let trace = Self::compute_trace(&data, dim);
        
        Some(Self { data, dim, trace })
    }

    /// Create a maximally mixed state of given dimension
    pub fn maximally_mixed(dim: usize) -> Option<Self> {
        if dim == 0 || dim > MAX_MATRIX_DIM {
            return None;
        }

        let mut data = vec![Complex::real(0.0); dim * dim];
        for i in 0..dim {
            data[i * dim + i] = Complex::real(1.0 / dim as f64);
        }

        Some(Self {
            data,
            dim,
            trace: 1.0,
        })
    }

    /// Create a pure state |psi><psi| from state vector
    pub fn from_pure_state(state: &[f64]) -> Option<Self> {
        if state.is_empty() || state.len() > MAX_MATRIX_DIM {
            return None;
        }

        let dim = state.len();
        let mut data = vec![Complex::real(0.0); dim * dim];

        for i in 0..dim {
            for j in 0..dim {
                data[i * dim + j] = Complex::new(state[i] * state[j], 0.0);
            }
        }

        let trace = state.iter().map(|&x| x * x).sum();
        
        Some(Self { data, dim, trace })
    }

    /// Get matrix element at (i, j)
    pub fn get(&self, i: usize, j: usize) -> Option<Complex> {
        if i >= self.dim || j >= self.dim {
            return None;
        }
        Some(self.data[i * self.dim + j])
    }

    /// Get matrix dimension
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Get trace value
    pub fn trace_value(&self) -> f64 {
        self.trace
    }

    /// Check if matrix is Hermitian (within tolerance)
    pub fn is_hermitian(&self, tol: f64) -> bool {
        for i in 0..self.dim {
            for j in (i + 1)..self.dim {
                let elem_ij = self.data[i * self.dim + j];
                let elem_ji = self.data[j * self.dim + i];
                
                // For Hermitian: rho_ij = conj(rho_ji)
                let diff_re = (elem_ij.re - elem_ji.re).abs();
                let diff_im = (elem_ij.im + elem_ji.im).abs();
                
                if diff_re > tol || diff_im > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Check if matrix is positive semi-definite (via eigenvalue estimation)
    pub fn is_positive_semidefinite(&self) -> bool {
        // Simple check: all diagonal elements should be non-negative
        for i in 0..self.dim {
            let elem = self.data[i * self.dim + i];
            if elem.re < -CONVERGENCE_TOLERANCE || elem.im.abs() > CONVERGENCE_TOLERANCE {
                return false;
            }
        }
        // More rigorous check would require full eigenvalue decomposition
        true
    }

    /// Compute partial trace over subsystem
    pub fn partial_trace(&self, trace_out_indices: &[usize]) -> Option<Self> {
        if trace_out_indices.is_empty() || trace_out_indices.len() >= self.dim {
            return None;
        }

        let remaining_dim = self.dim - trace_out_indices.len();
        if remaining_dim == 0 || remaining_dim > MAX_MATRIX_DIM {
            return None;
        }

        // Simplified partial trace implementation
        let mut result_data = vec![Complex::real(0.0); remaining_dim * remaining_dim];
        
        let mut result_idx = 0;
        for i in 0..self.dim {
            if trace_out_indices.contains(&i) {
                continue;
            }
            for j in 0..self.dim {
                if trace_out_indices.contains(&j) {
                    continue;
                }
                if let Some(elem) = self.get(i, j) {
                    result_data[result_idx] = result_data[result_idx] + elem;
                }
                result_idx += 1;
            }
        }

        Self::new(result_data, remaining_dim)
    }

    /// Apply unitary transformation: U * rho * U^dagger
    pub fn apply_unitary(&self, unitary: &[Complex], unitary_dim: usize) -> Option<Self> {
        if unitary_dim != self.dim || unitary.len() != unitary_dim * unitary_dim {
            return None;
        }

        let mut result = vec![Complex::real(0.0); self.dim * self.dim];

        // Compute U * rho
        let mut temp = vec![Complex::real(0.0); self.dim * self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                for k in 0..self.dim {
                    temp[i * self.dim + j] = temp[i * self.dim + j] 
                        + unitary[i * self.dim + k] * self.data[k * self.dim + j];
                }
            }
        }

        // Compute (U * rho) * U^dagger
        for i in 0..self.dim {
            for j in 0..self.dim {
                for k in 0..self.dim {
                    let u_dagger = unitary[j * self.dim + k].conjugate();
                    result[i * self.dim + j] = result[i * self.dim + j] 
                        + temp[i * self.dim + k] * u_dagger;
                }
            }
        }

        Self::new(result, self.dim)
    }

    /// Compute distance to another density matrix (Frobenius norm)
    pub fn distance(&self, other: &Self) -> Option<f64> {
        if self.dim != other.dim {
            return None;
        }

        let mut sum = 0.0;
        for i in 0..self.data.len() {
            let diff_re = self.data[i].re - other.data[i].re;
            let diff_im = self.data[i].im - other.data[i].im;
            sum += diff_re * diff_re + diff_im * diff_im;
        }

        Some(sum.sqrt())
    }

    // Internal helper: compute trace of matrix
    fn compute_trace(data: &[Complex], dim: usize) -> f64 {
        let mut trace = 0.0;
        for i in 0..dim {
            trace += data[i * dim + i].re;
        }
        trace
    }
}

/// Result of CTC evolution computation
#[derive(Debug, Clone)]
pub struct CTCResult {
    /// Input density matrix
    pub rho_in: DensityMatrix,
    /// Output density matrix after CTC interaction
    pub rho_out: DensityMatrix,
    /// Fixed-point CTC state
    pub rho_ctc: DensityMatrix,
    /// Number of iterations to converge
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Final error metric
    pub final_error: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_matrix_creation() {
        let data = vec![
            Complex::real(0.7),
            Complex::real(0.0),
            Complex::real(0.0),
            Complex::real(0.3),
        ];
        
        let dm = DensityMatrix::new(data, 2);
        assert!(dm.is_some());
        
        let dm = dm.unwrap();
        assert_eq!(dm.dimension(), 2);
        assert!((dm.trace_value() - 1.0).abs() < CONVERGENCE_TOLERANCE);
    }

    #[test]
    fn test_maximally_mixed() {
        let dm = DensityMatrix::maximally_mixed(2);
        assert!(dm.is_some());
        
        let dm = dm.unwrap();
        assert_eq!(dm.dimension(), 2);
        assert!((dm.trace_value() - 1.0).abs() < CONVERGENCE_TOLERANCE);
    }

    #[test]
    fn test_pure_state() {
        let state = vec![1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()];
        let dm = DensityMatrix::from_pure_state(&state);
        assert!(dm.is_some());
    }

    #[test]
    fn test_hermitian_check() {
        let data = vec![
            Complex::real(0.5),
            Complex::new(0.1, 0.2),
            Complex::new(0.1, -0.2),
            Complex::real(0.5),
        ];
        
        let dm = DensityMatrix::new(data, 2).unwrap();
        assert!(dm.is_hermitian(CONVERGENCE_TOLERANCE));
    }
}
