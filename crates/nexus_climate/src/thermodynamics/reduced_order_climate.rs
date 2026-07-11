//! Reduced-Order Climate Model (ROCM) using Proper Orthogonal Decomposition (POD)
//! Projects high-dimensional atmospheric/oceanic dynamics onto low-dimensional manifolds.

use alloc::vec::Vec;
use core::fmt;

/// Error types for ROCM operations
#[derive(Debug, Clone, PartialEq)]
pub enum RocmError {
    InsufficientSnapshots,
    DimensionMismatch { expected: usize, got: usize },
    EigenDecompositionFailed,
    SingularCovarianceMatrix,
    ProjectionError,
}

impl fmt::Display for RocmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientSnapshots => write!(f, "Insufficient snapshots for POD"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::EigenDecompositionFailed => write!(f, "Eigen decomposition failed"),
            Self::SingularCovarianceMatrix => write!(f, "Singular covariance matrix"),
            Self::ProjectionError => write!(f, "Projection error"),
        }
    }
}

/// Proper Orthogonal Decomposition engine for climate state reduction
pub struct ProperOrthogonalDecomposition {
    /// Number of spatial grid points
    n_spatial: usize,
    /// Number of retained modes
    n_modes: usize,
    /// POD basis vectors (modes) - column major: [n_spatial * n_modes]
    modes: Box<[f64]>,
    /// Eigenvalues corresponding to each mode
    eigenvalues: Box<[f64]>,
    /// Mean state subtracted before projection
    mean_state: Box<[f64]>,
}

impl ProperOrthogonalDecomposition {
    /// Create new POD from snapshot matrix
    /// snapshots: [n_snapshots * n_spatial] row-major
    pub fn compute(
        snapshots: &[f64],
        n_spatial: usize,
        n_modes: usize,
    ) -> Result<Self, RocmError> {
        let n_snapshots = snapshots.len() / n_spatial;
        if snapshots.len() % n_spatial != 0 {
            return Err(RocmError::DimensionMismatch {
                expected: n_spatial,
                got: snapshots.len(),
            });
        }
        if n_snapshots < n_modes {
            return Err(RocmError::InsufficientSnapshots);
        }

        // Compute mean state
        let mut mean_state = vec![0.0_f64; n_spatial];
        for i in 0..n_spatial {
            let mut sum = 0.0;
            for s in 0..n_snapshots {
                sum += snapshots[s * n_spatial + i];
            }
            mean_state[i] = sum / n_snapshots as f64;
        }

        // Build centered snapshot matrix and compute covariance
        // C = (1/(N-1)) * X^T * X where X is centered
        let mut covariance = vec![0.0_f64; n_spatial * n_spatial];
        let inv_n_minus_1 = 1.0 / (n_snapshots - 1) as f64;

        for i in 0..n_spatial {
            for j in 0..=i {
                let mut cov_ij = 0.0;
                for s in 0..n_snapshots {
                    let xi = snapshots[s * n_spatial + i] - mean_state[i];
                    let xj = snapshots[s * n_spatial + j] - mean_state[j];
                    cov_ij += xi * xj;
                }
                cov_ij *= inv_n_minus_1;
                covariance[i * n_spatial + j] = cov_ij;
                covariance[j * n_spatial + i] = cov_ij;
            }
        }

        // Regularize diagonal to prevent singularity
        let epsilon = 1e-12;
        for i in 0..n_spatial {
            covariance[i * n_spatial + i] += epsilon;
        }

        // Compute dominant eigenpairs using power iteration with deflation
        let mut modes = vec![0.0_f64; n_spatial * n_modes];
        let mut eigenvalues = vec![0.0_f64; n_modes];
        let mut cov_work = covariance.clone();

        for k in 0..n_modes {
            // Power iteration for largest eigenvalue
            let mut eigenvector = vec![0.0_f64; n_spatial];
            eigenvector[k % n_spatial] = 1.0; // Initial guess

            for _iter in 0..1000 {
                let mut new_vec = vec![0.0_f64; n_spatial];
                for i in 0..n_spatial {
                    for j in 0..n_spatial {
                        new_vec[i] += cov_work[i * n_spatial + j] * eigenvector[j];
                    }
                }

                // Normalize
                let norm: f64 = new_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < 1e-15 {
                    break;
                }
                for i in 0..n_spatial {
                    eigenvector[i] = new_vec[i] / norm;
                }

                // Check convergence
                let mut residual = 0.0;
                for i in 0..n_spatial {
                    let mut av_i = 0.0;
                    for j in 0..n_spatial {
                        av_i += cov_work[i * n_spatial + j] * eigenvector[j];
                    }
                    residual += (av_i - eigenvector[i] * norm).abs();
                }
                if residual < 1e-10 {
                    break;
                }
            }

            // Compute eigenvalue via Rayleigh quotient
            let mut eigenvalue = 0.0;
            for i in 0..n_spatial {
                for j in 0..n_spatial {
                    eigenvalue += eigenvector[i] * cov_work[i * n_spatial + j] * eigenvector[j];
                }
            }

            eigenvalues[k] = eigenvalue.max(0.0);

            // Store mode
            for i in 0..n_spatial {
                modes[k * n_spatial + i] = eigenvector[i];
            }

            // Deflate: remove this component from covariance
            for i in 0..n_spatial {
                for j in 0..n_spatial {
                    cov_work[i * n_spatial + j] -= eigenvalue * eigenvector[i] * eigenvector[j];
                }
            }
        }

        Ok(Self {
            n_spatial,
            n_modes,
            modes: modes.into_boxed_slice(),
            eigenvalues: eigenvalues.into_boxed_slice(),
            mean_state: mean_state.into_boxed_slice(),
        })
    }

    /// Project a full state onto the reduced basis
    pub fn project(&self, state: &[f64]) -> Result<Box<[f64]>, RocmError> {
        if state.len() != self.n_spatial {
            return Err(RocmError::DimensionMismatch {
                expected: self.n_spatial,
                got: state.len(),
            });
        }

        let mut coefficients = vec![0.0_f64; self.n_modes];
        for k in 0..self.n_modes {
            let mut coeff = 0.0;
            for i in 0..self.n_spatial {
                coeff += self.modes[k * self.n_spatial + i] * (state[i] - self.mean_state[i]);
            }
            coefficients[k] = coeff;
        }

        Ok(coefficients.into_boxed_slice())
    }

    /// Reconstruct full state from reduced coefficients
    pub fn reconstruct(&self, coefficients: &[f64]) -> Result<Box<[f64]>, RocmError> {
        if coefficients.len() != self.n_modes {
            return Err(RocmError::DimensionMismatch {
                expected: self.n_modes,
                got: coefficients.len(),
            });
        }

        let mut state = self.mean_state.to_vec();
        for k in 0..self.n_modes {
            for i in 0..self.n_spatial {
                state[i] += coefficients[k] * self.modes[k * self.n_spatial + i];
            }
        }

        Ok(state.into_boxed_slice())
    }

    /// Get explained variance ratio
    pub fn explained_variance_ratio(&self) -> Box<[f64]> {
        let total: f64 = self.eigenvalues.iter().sum();
        if total < 1e-15 {
            return vec![0.0_f64; self.n_modes].into_boxed_slice();
        }
        self.eigenvalues
            .iter()
            .map(|&e| e / total)
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

/// Reduced-Order Climate Model state
pub struct ReducedOrderClimateModel {
    pod: ProperOrthogonalDecomposition,
    /// Current reduced state coefficients
    coefficients: Box<[f64]>,
    /// Time derivative of coefficients
    coefficient_rates: Box<[f64]>,
    /// Linear dynamics matrix in reduced space
    dynamics_matrix: Box<[f64]>,
    /// Forcing terms
    forcing: Box<[f64]>,
}

impl ReducedOrderClimateModel {
    /// Initialize ROM from POD and initial full state
    pub fn new(pod: ProperOrthogonalDecomposition, initial_state: &[f64]) -> Result<Self, RocmError> {
        let coefficients = pod.project(initial_state)?;
        let n_modes = coefficients.len();

        Ok(Self {
            pod,
            coefficients,
            coefficient_rates: vec![0.0_f64; n_modes].into_boxed_slice(),
            dynamics_matrix: vec![0.0_f64; n_modes * n_modes].into_boxed_slice(),
            forcing: vec![0.0_f64; n_modes].into_boxed_slice(),
        })
    }

    /// Learn linear dynamics from time series of coefficients
    pub fn learn_dynamics(&mut self, coefficient_series: &[f64], dt: f64) -> Result<(), RocmError> {
        let n_modes = self.coefficients.len();
        let n_samples = coefficient_series.len() / n_modes;

        if n_samples < 2 {
            return Err(RocmError::InsufficientSnapshots);
        }

        // Estimate dynamics matrix A and forcing b from: dc/dt = A*c + b
        // Using least squares: minimize ||dc/dt - A*c - b||^2

        let mut sum_cc = vec![0.0_f64; n_modes * n_modes];
        let mut sum_c = vec![0.0_f64; n_modes];
        let mut sum_dc = vec![0.0_f64; n_modes];
        let mut sum_dc_c = vec![0.0_f64; n_modes * n_modes];

        for t in 0..(n_samples - 1) {
            let c_t = &coefficient_series[t * n_modes..(t + 1) * n_modes];
            let c_tp1 = &coefficient_series[(t + 1) * n_modes..(t + 2) * n_modes];

            // Finite difference for derivative
            let mut dc_dt = vec![0.0_f64; n_modes];
            for i in 0..n_modes {
                dc_dt[i] = (c_tp1[i] - c_t[i]) / dt;
            }

            // Accumulate sums
            for i in 0..n_modes {
                sum_c[i] += c_t[i];
                sum_dc[i] += dc_dt[i];
                for j in 0..n_modes {
                    sum_cc[i * n_modes + j] += c_t[i] * c_t[j];
                    sum_dc_c[i * n_modes + j] += dc_dt[i] * c_t[j];
                }
            }
        }

        let n_eff = (n_samples - 1) as f64;

        // Simple regression: A = (sum(dc*c^T) - sum(dc)*sum(c)^T/n) * inv(sum(c*c^T) - sum(c)*sum(c)^T/n)
        // Simplified: assume zero mean and use A ≈ sum(dc*c^T) / sum(c*c^T) element-wise with regularization

        for i in 0..n_modes {
            for j in 0..n_modes {
                let cov_dc_c = sum_dc_c[i * n_modes + j] - sum_dc[i] * sum_c[j] / n_eff;
                let cov_c_c = sum_cc[i * n_modes + j] - sum_c[i] * sum_c[j] / n_eff;

                // Regularized division
                let denom = cov_c_c.abs() + 1e-10;
                self.dynamics_matrix[i * n_modes + j] = cov_dc_c / denom;
            }
        }

        // Forcing term
        for i in 0..n_modes {
            let mean_dc = sum_dc[i] / n_eff;
            let mut mean_a_c = 0.0;
            for j in 0..n_modes {
                mean_a_c += self.dynamics_matrix[i * n_modes + j] * (sum_c[j] / n_eff);
            }
            self.forcing[i] = mean_dc - mean_a_c;
        }

        Ok(())
    }

    /// Advance model by one time step using explicit Euler
    pub fn step(&mut self, dt: f64) -> Result<(), RocmError> {
        let n_modes = self.coefficients.len();
        let mut new_coefficients = vec![0.0_f64; n_modes];

        for i in 0..n_modes {
            let mut rate = self.forcing[i];
            for j in 0..n_modes {
                rate += self.dynamics_matrix[i * n_modes + j] * self.coefficients[j];
            }
            new_coefficients[i] = self.coefficients[i] + rate * dt;
        }

        self.coefficient_rates.copy_from_slice(&new_coefficients);
        for i in 0..n_modes {
            self.coefficient_rates[i] = (new_coefficients[i] - self.coefficients[i]) / dt;
        }
        self.coefficients = new_coefficients.into_boxed_slice();

        Ok(())
    }

    /// Get reconstructed full state
    pub fn get_full_state(&self) -> Result<Box<[f64]>, RocmError> {
        self.pod.reconstruct(&self.coefficients)
    }

    /// Get current reduced coefficients
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

/// Tipping point detector monitoring critical slowing down
pub struct TippingPointDetector {
    /// Window size for statistics
    window_size: usize,
    /// History of coefficient values for each mode
    history: Vec<Box<[f64]>>,
    /// Previous autocorrelation at lag-1
    prev_autocorr: Box<[f64]>,
    /// Previous variance
    prev_variance: Box<[f64]>,
    /// Critical slowing down indicator (increasing = approaching bifurcation)
    csd_indicator: f64,
}

impl TippingPointDetector {
    pub fn new(n_modes: usize, window_size: usize) -> Self {
        Self {
            window_size,
            history: Vec::with_capacity(window_size),
            prev_autocorr: vec![0.0_f64; n_modes].into_boxed_slice(),
            prev_variance: vec![0.0_f64; n_modes].into_boxed_slice(),
            csd_indicator: 0.0,
        }
    }

    /// Add new observation
    pub fn observe(&mut self, coefficients: &[f64]) {
        self.history.push(coefficients.to_vec().into_boxed_slice());
        if self.history.len() > self.window_size {
            self.history.remove(0);
        }
    }

    /// Compute early warning signals
    pub fn compute_warnings(&mut self) -> Result<TippingPointWarnings, RocmError> {
        if self.history.len() < self.window_size / 2 {
            return Ok(TippingPointWarnings {
                autocorrelation_increase: 0.0,
                variance_increase: 0.0,
                csd_score: 0.0,
                is_critical: false,
            });
        }

        let n_modes = self.prev_autocorr.len();
        let half_window = self.window_size / 2;

        // Split into two halves
        let first_half = &self.history[..half_window.min(self.history.len())];
        let second_half = &self.history[half_window.min(self.history.len())..];

        if first_half.is_empty() || second_half.is_empty() {
            return Ok(TippingPointWarnings::default());
        }

        let mut autocorr_increase = 0.0;
        let mut variance_increase = 0.0;
        let mut total_csd_score = 0.0;

        for mode in 0..n_modes {
            // Compute variance in each half
            let var1 = Self::compute_variance(first_half, mode);
            let var2 = Self::compute_variance(second_half, mode);

            // Compute lag-1 autocorrelation in each half
            let ac1 = Self::compute_autocorr_lag1(first_half, mode);
            let ac2 = Self::compute_autocorr_lag1(second_half, mode);

            // Relative increases
            let var_inc = if var1 > 1e-15 { (var2 - var1) / var1 } else { 0.0 };
            let ac_inc = if ac1.abs() > 1e-15 { (ac2 - ac1) / ac1.abs() } else { 0.0 };

            autocorr_increase += ac_inc;
            variance_increase += var_inc;

            // CSD score: both should increase near bifurcation
            let mode_csd = (var_inc.max(0.0) + ac_inc.max(0.0)) / 2.0;
            total_csd_score += mode_csd;

            // Update previous values
            self.prev_variance[mode] = var2;
            self.prev_autocorr[mode] = ac2;
        }

        let n_modes_f = n_modes as f64;
        autocorr_increase /= n_modes_f;
        variance_increase /= n_modes_f;
        total_csd_score /= n_modes_f;

        // Critical if both indicators show significant increase
        let is_critical = autocorr_increase > 0.1 && variance_increase > 0.1 && total_csd_score > 0.15;

        self.csd_indicator = total_csd_score;

        Ok(TippingPointWarnings {
            autocorrelation_increase: autocorr_increase,
            variance_increase: variance_increase,
            csd_score: total_csd_score,
            is_critical,
        })
    }

    fn compute_variance(history: &[Box<[f64]>], mode: usize) -> f64 {
        if history.is_empty() {
            return 0.0;
        }
        let n = history.len() as f64;
        let mean: f64 = history.iter().map(|h| h[mode]).sum::<f64>() / n;
        let variance: f64 = history.iter().map(|h| (h[mode] - mean).powi(2)).sum::<f64>() / n;
        variance
    }

    fn compute_autocorr_lag1(history: &[Box<[f64]>], mode: usize) -> f64 {
        if history.len() < 2 {
            return 0.0;
        }

        let n = history.len() as f64;
        let mean: f64 = history.iter().map(|h| h[mode]).sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..(history.len() - 1) {
            let x_i = history[i][mode] - mean;
            let x_ip1 = history[i + 1][mode] - mean;
            numerator += x_i * x_ip1;
            denominator += x_i * x_i;
        }

        if denominator < 1e-15 {
            return 0.0;
        }
        numerator / denominator
    }
}

/// Warning signals from tipping point analysis
#[derive(Debug, Clone, Default)]
pub struct TippingPointWarnings {
    pub autocorrelation_increase: f64,
    pub variance_increase: f64,
    pub csd_score: f64,
    pub is_critical: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pod_computation() {
        // Simple test: sinusoidal patterns
        let n_spatial = 10;
        let n_snapshots = 20;
        let mut snapshots = vec![0.0_f64; n_spatial * n_snapshots];

        for s in 0..n_snapshots {
            for i in 0..n_spatial {
                let t = s as f64 * 0.1;
                let x = i as f64 / n_spatial as f64;
                snapshots[s * n_spatial + i] = (x * 6.28318).sin() * t.sin();
            }
        }

        let pod = ProperOrthogonalDecomposition::compute(&snapshots, n_spatial, 3);
        assert!(pod.is_ok());
    }
}
