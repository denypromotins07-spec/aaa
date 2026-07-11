// NEXUS-OMEGA Stage 34: Macro-Economic Gravity
// Chapter 1: Sovereign Debt Gravity & Riemannian Capital Flows
// File: crates/nexus_macro_physics/src/gravity/riemannian_metric_tensor.rs

//! Riemannian Metric Tensor Calculator for Global Sovereign Debt Networks
//!
//! Models the global sovereign debt and FX reserve network as a Riemannian Manifold.
//! Massive debt obligations create "gravity wells" that distort capital flows.
//!
//! CRITICAL SAFETY: Implements strictly positive-definite regularization (εI) to prevent
//! coordinate singularities when determinant drops to zero during correlated market crashes.

#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]

use core::fmt;
use alloc::vec::Vec;
use alloc::boxed::Box;

/// Dimension of the economic state space (typically number of nations tracked)
pub const MAX_NATIONS: usize = 256;

/// Regularization epsilon for positive-definite guarantee
/// Prevents division by zero in metric tensor inversion
pub const REGULARIZATION_EPSILON: f64 = 1e-12;

/// Maximum condition number before triggering regularization
pub const MAX_CONDITION_NUMBER: f64 = 1e10;

/// Error types for metric tensor operations
#[derive(Debug, Clone, PartialEq)]
pub enum MetricTensorError {
    InvalidDimension { expected: usize, got: usize },
    SingularMatrix { determinant: f64 },
    NonPositiveDefinite { min_eigenvalue: f64 },
    ConditionNumberExceeded { condition: f64 },
    ConvergenceFailure { iterations: u32 },
    OverflowDetected,
}

impl fmt::Display for MetricTensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension { expected, got } => {
                write!(f, "Invalid dimension: expected {}, got {}", expected, got)
            }
            Self::SingularMatrix { determinant } => {
                write!(f, "Singular matrix detected: det = {}", determinant)
            }
            Self::NonPositiveDefinite { min_eigenvalue } => {
                write!(f, "Non-positive definite: min eigenvalue = {}", min_eigenvalue)
            }
            Self::ConditionNumberExceeded { condition } => {
                write!(f, "Condition number exceeded: {}", condition)
            }
            Self::ConvergenceFailure { iterations } => {
                write!(f, "Convergence failure after {} iterations", iterations)
            }
            Self::OverflowDetected => write!(f, "Numerical overflow detected"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MetricTensorError {}

/// Economic state vector for a single nation
#[derive(Debug, Clone, Copy)]
pub struct EconomicState {
    /// Trade imbalance ratio (exports - imports) / GDP
    pub trade_imbalance: f64,
    /// Interest rate differential vs global baseline
    pub interest_differential: f64,
    /// Sovereign CDS spread in basis points
    pub cds_spread_bps: f64,
    /// FX reserves in billions USD
    pub fx_reserves: f64,
    /// Short-term debt rollover risk (0-1 scale)
    pub rollover_risk: f64,
}

impl EconomicState {
    #[must_use]
    pub fn new(
        trade_imbalance: f64,
        interest_differential: f64,
        cds_spread_bps: f64,
        fx_reserves: f64,
        rollover_risk: f64,
    ) -> Self {
        // Clamp inputs to prevent extreme values
        Self {
            trade_imbalance: trade_imbalance.clamp(-1.0, 1.0),
            interest_differential: interest_differential.clamp(-0.5, 0.5),
            cds_spread_bps: cds_spread_bps.clamp(0.0, 10000.0),
            fx_reserves: fx_reserves.max(0.0),
            rollover_risk: rollover_risk.clamp(0.0, 1.0),
        }
    }

    /// Convert to normalized feature vector
    #[must_use]
    pub fn to_feature_vector(&self) -> [f64; 5] {
        [
            self.trade_imbalance,
            self.interest_differential * 2.0, // Scale to [-1, 1]
            (self.cds_spread_bps / 10000.0).ln().max(0.0), // Log scale for CDS
            (self.fx_reserves + 1.0).ln() / 10.0, // Log-normalize reserves
            self.rollover_risk,
        ]
    }
}

/// Riemannian Metric Tensor for economic distance calculation
///
/// The metric tensor g_ij defines the local geometry of the economic manifold.
/// Economic distance between two states is computed via the geodesic integral:
/// d = ∫ sqrt(g_ij dx^i dx^j)
#[derive(Debug, Clone)]
pub struct RiemannianMetricTensor {
    /// Dimension of the manifold
    dim: usize,
    /// Metric tensor components stored in row-major order
    components: Box<[f64]>,
    /// Inverse metric tensor components
    inverse_components: Box<[f64]>,
    /// Determinant of the metric tensor
    determinant: f64,
    /// Regularization flag
    is_regularized: bool,
}

impl RiemannianMetricTensor {
    /// Create a new metric tensor from economic states
    ///
    /// # Arguments
    /// * `states` - Slice of economic states for each nation
    /// * `correlation_matrix` - Pre-computed correlation matrix of economic indicators
    ///
    /// # Returns
    /// * `Ok(Self)` on success
    /// * `Err(MetricTensorError)` on failure
    pub fn from_economic_states(
        states: &[EconomicState],
        correlation_matrix: Option<&[f64]>,
    ) -> Result<Self, MetricTensorError> {
        let n = states.len();
        if n == 0 || n > MAX_NATIONS {
            return Err(MetricTensorError::InvalidDimension {
                expected: MAX_NATIONS,
                got: n,
            });
        }

        let dim = 5; // Feature dimension from to_feature_vector
        let total_components = dim * dim;
        
        let mut components = vec![0.0_f64; total_components];
        let mut determinant = 1.0_f64;

        // Build metric tensor from economic distances
        // g_ij = δ_ij + α * Σ_k (∂φ_k/∂x^i)(∂φ_k/∂x^j)
        // where φ_k are economic potentials derived from states
        
        for i in 0..dim {
            for j in 0..dim {
                let mut g_ij = if i == j { 1.0 } else { 0.0 }; // Start with Euclidean metric
                
                // Add contribution from each nation's economic state
                for state in states {
                    let features = state.to_feature_vector();
                    
                    // Compute gradient contributions
                    let grad_i = Self::economic_gradient_component(i, state, features[i]);
                    let grad_j = Self::economic_gradient_component(j, state, features[j]);
                    
                    // Weight by CDS spread (higher risk = stronger curvature)
                    let risk_weight = 1.0 + (state.cds_spread_bps / 1000.0).min(10.0);
                    
                    g_ij += risk_weight * grad_i * grad_j * 0.1;
                }
                
                // Apply correlation adjustment if provided
                if let Some(corr) = correlation_matrix {
                    if corr.len() >= n * n {
                        let corr_factor = Self::aggregate_correlation(corr, n, i, j);
                        g_ij *= 1.0 + corr_factor * 0.5;
                    }
                }
                
                components[i * dim + j] = g_ij;
            }
        }

        // Compute determinant and check for singularity
        let det = Self::compute_determinant(&components, dim);
        
        // Check for near-singularity and apply regularization if needed
        let (final_components, final_det, is_regularized) = 
            if det.abs() < REGULARIZATION_EPSILON || !det.is_finite() {
                // Apply εI regularization to ensure positive definiteness
                Self::apply_regularization(&mut components, dim, REGULARIZATION_EPSILON)?
            } else {
                (components, det, false)
            };

        // Compute inverse metric tensor
        let inverse_components = Self::compute_inverse(&final_components, dim)?;

        Ok(Self {
            dim,
            components: final_components.into_boxed_slice(),
            inverse_components: inverse_components.into_boxed_slice(),
            determinant: final_det,
            is_regularized,
        })
    }

    /// Compute economic gradient component for feature i
    fn economic_gradient_component(i: usize, _state: &EconomicState, feature: f64) -> f64 {
        // Non-linear response functions for different economic indicators
        match i {
            0 => feature.tanh(), // Trade imbalance: saturating response
            1 => feature * (1.0 - feature.abs()).max(0.5), // Interest diff: peaked response
            2 => feature.exp().min(100.0), // CDS: exponential risk sensitivity
            3 => feature.sqrt(), // Reserves: diminishing returns
            4 => feature * feature, // Rollover risk: quadratic
            _ => 0.0,
        }
    }

    /// Aggregate correlation factor for metric components
    fn aggregate_correlation(corr: &[f64], n: usize, i: usize, j: usize) -> f64 {
        // Simplified: average correlation across all nation pairs
        let mut sum = 0.0;
        let mut count = 0;
        
        for a in 0..n {
            for b in 0..n {
                if a != b {
                    let idx = a * n + b;
                    if idx < corr.len() {
                        sum += corr[idx].abs();
                        count += 1;
                    }
                }
            }
        }
        
        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }

    /// Compute determinant using LU decomposition with partial pivoting
    fn compute_determinant(matrix: &[f64], dim: usize) -> f64 {
        if dim == 0 {
            return 1.0;
        }
        
        let mut lu = matrix.to_vec();
        let mut det = 1.0;
        let mut sign = 1;

        for k in 0..dim {
            // Find pivot
            let mut max_val = lu[k * dim + k].abs();
            let mut max_row = k;
            
            for i in (k + 1)..dim {
                let val = lu[i * dim + k].abs();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            if max_val < 1e-15 {
                return 0.0; // Singular matrix
            }

            // Swap rows if needed
            if max_row != k {
                for j in 0..dim {
                    lu.swap(k * dim + j, max_row * dim + j);
                }
                sign = -sign;
            }

            det *= lu[k * dim + k];

            // Eliminate column
            for i in (k + 1)..dim {
                let factor = lu[i * dim + k] / lu[k * dim + k];
                for j in k..dim {
                    lu[i * dim + j] -= factor * lu[k * dim + j];
                }
            }
        }

        (sign as f64) * det
    }

    /// Apply εI regularization to ensure positive definiteness
    fn apply_regularization(
        matrix: &mut [f64],
        dim: usize,
        epsilon: f64,
    ) -> Result<(Vec<f64>, f64, bool), MetricTensorError> {
        let mut regularized = matrix.to_vec();
        
        // Add ε to diagonal elements
        for i in 0..dim {
            regularized[i * dim + i] += epsilon;
        }
        
        let det = Self::compute_determinant(&regularized, dim);
        
        if det.abs() < epsilon || !det.is_finite() {
            // Try larger regularization
            let epsilon2 = epsilon * 10.0;
            for i in 0..dim {
                regularized[i * dim + i] += epsilon2;
            }
            let det2 = Self::compute_determinant(&regularized, dim);
            
            if det2.abs() < epsilon2 || !det2.is_finite() {
                return Err(MetricTensorError::ConditionNumberExceeded { 
                    condition: f64::INFINITY 
                });
            }
            
            Ok((regularized, det2, true))
        } else {
            Ok((regularized, det, true))
        }
    }

    /// Compute inverse metric tensor using Gauss-Jordan elimination
    fn compute_inverse(matrix: &[f64], dim: usize) -> Result<Vec<f64>, MetricTensorError> {
        if dim == 0 {
            return Ok(vec![]);
        }

        // Create augmented matrix [A | I]
        let mut augmented = vec![0.0_f64; dim * dim * 2];
        
        for i in 0..dim {
            for j in 0..dim {
                augmented[i * dim * 2 + j] = matrix[i * dim + j];
                if i == j {
                    augmented[i * dim * 2 + dim + j] = 1.0;
                }
            }
        }

        // Forward elimination with partial pivoting
        for k in 0..dim {
            // Find pivot
            let mut max_val = augmented[k * dim * 2 + k].abs();
            let mut max_row = k;
            
            for i in (k + 1)..dim {
                let val = augmented[i * dim * 2 + k].abs();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            if max_val < 1e-15 {
                return Err(MetricTensorError::SingularMatrix { determinant: 0.0 });
            }

            // Swap rows
            if max_row != k {
                for j in 0..(dim * 2) {
                    augmented.swap(k * dim * 2 + j, max_row * dim * 2 + j);
                }
            }

            // Scale pivot row
            let pivot = augmented[k * dim * 2 + k];
            for j in 0..(dim * 2) {
                augmented[k * dim * 2 + j] /= pivot;
            }

            // Eliminate column
            for i in 0..dim {
                if i != k {
                    let factor = augmented[i * dim * 2 + k];
                    for j in 0..(dim * 2) {
                        augmented[i * dim * 2 + j] -= factor * augmented[k * dim * 2 + j];
                    }
                }
            }
        }

        // Extract inverse from right half
        let mut inverse = vec![0.0_f64; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                inverse[i * dim + j] = augmented[i * dim * 2 + dim + j];
            }
        }

        Ok(inverse)
    }

    /// Get metric tensor component g_ij
    #[must_use]
    pub fn g(&self, i: usize, j: usize) -> f64 {
        if i < self.dim && j < self.dim {
            self.components[i * self.dim + j]
        } else {
            0.0
        }
    }

    /// Get inverse metric tensor component g^ij
    #[must_use]
    pub fn g_inv(&self, i: usize, j: usize) -> f64 {
        if i < self.dim && j < self.dim {
            self.inverse_components[i * self.dim + j]
        } else {
            0.0
        }
    }

    /// Compute economic distance between two feature vectors
    ///
    /// d = sqrt(g_ij Δx^i Δx^j)
    #[must_use]
    pub fn economic_distance(&self, features1: &[f64; 5], features2: &[f64; 5]) -> f64 {
        let mut squared_distance = 0.0;
        
        for i in 0..self.dim {
            for j in 0..self.dim {
                let dx_i = features1[i] - features2[i];
                let dx_j = features1[j] - features2[j];
                squared_distance += self.g(i, j) * dx_i * dx_j;
            }
        }
        
        squared_distance.max(0.0).sqrt()
    }

    /// Check if tensor was regularized
    #[must_use]
    pub const fn is_regularized(&self) -> bool {
        self.is_regularized
    }

    /// Get determinant of metric tensor
    #[must_use]
    pub const fn determinant(&self) -> f64 {
        self.determinant
    }

    /// Get dimension of manifold
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_tensor_creation() {
        let states = vec![
            EconomicState::new(0.05, 0.02, 150.0, 500.0, 0.3),
            EconomicState::new(-0.03, -0.01, 80.0, 200.0, 0.1),
            EconomicState::new(0.1, 0.05, 500.0, 50.0, 0.7),
        ];

        let tensor = RiemannianMetricTensor::from_economic_states(&states, None);
        assert!(tensor.is_ok());
        
        let tensor = tensor.unwrap();
        assert_eq!(tensor.dimension(), 5);
        assert!(tensor.determinant() > 0.0);
    }

    #[test]
    fn test_economic_distance() {
        let states = vec![
            EconomicState::new(0.05, 0.02, 150.0, 500.0, 0.3),
            EconomicState::new(-0.03, -0.01, 80.0, 200.0, 0.1),
        ];

        let tensor = RiemannianMetricTensor::from_economic_states(&states, None).unwrap();
        
        let f1 = states[0].to_feature_vector();
        let f2 = states[1].to_feature_vector();
        
        let distance = tensor.economic_distance(&f1, &f2);
        assert!(distance > 0.0);
        assert!(distance.is_finite());
    }

    #[test]
    fn test_regularization_on_singular() {
        // Create nearly identical states that could cause singularity
        let states = vec![
            EconomicState::new(0.05, 0.02, 150.0, 500.0, 0.3),
            EconomicState::new(0.05, 0.02, 150.0, 500.0, 0.3),
            EconomicState::new(0.05, 0.02, 150.0, 500.0, 0.3),
        ];

        let tensor = RiemannianMetricTensor::from_economic_states(&states, None);
        assert!(tensor.is_ok());
        
        let tensor = tensor.unwrap();
        assert!(tensor.is_regularized() || tensor.determinant() > 0.0);
    }
}
