//! Semantic Manifold Curvature Calculator using Randomized Dimensionality Reduction
//! 
//! Maps financial narrative embeddings to a Riemannian manifold and computes
//! Ricci curvature to detect paradigm shifts. Uses Johnson-Lindenstrauss transform
//! to avoid OOM on high-dimensional data.

use nalgebra::{DMatrix, DVector, Matrix2, SVD, SymmetricEigen};
use thiserror::Error;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

#[derive(Error, Debug)]
pub enum ManifoldError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Insufficient points for curvature estimation: need at least {min}, got {actual}")]
    InsufficientPoints { min: usize, actual: usize },
    #[error("SVD decomposition failed")]
    SVDFailure,
    #[error("Eigenvalue computation failed")]
    EigenFailure,
    #[error("Numerical instability detected")]
    NumericalInstability,
}

/// Configuration for manifold curvature computation
pub struct ManifoldConfig {
    /// Target dimension after JL projection (prevents OOM)
    pub target_dim: usize,
    /// Number of neighbors for local curvature estimation
    pub num_neighbors: usize,
    /// Epsilon for numerical stability
    pub epsilon: f64,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for ManifoldConfig {
    fn default() -> Self {
        Self {
            target_dim: 50, // Computationally tractable
            num_neighbors: 10,
            epsilon: 1e-10,
            seed: 42,
        }
    }
}

/// Point on the semantic manifold with embedding and metadata
#[derive(Clone, Debug)]
pub struct ManifoldPoint {
    pub id: usize,
    pub timestamp: f64,
    pub embedding: DVector<f64>,
}

/// Local curvature estimate at a point
#[derive(Clone, Debug)]
pub struct CurvatureEstimate {
    pub point_id: usize,
    /// Ricci scalar curvature (negative = hyperbolic/saddle, positive = spherical)
    pub ricci_scalar: f64,
    /// Sectional curvatures in principal directions
    pub sectional_curvatures: Vec<f64>,
    /// Confidence in estimate (based on neighbor quality)
    pub confidence: f64,
}

impl CurvatureEstimate {
    pub fn is_paradigm_shift(&self, threshold: f64) -> bool {
        // Sharp negative curvature indicates topological tearing/paradigm shift
        self.ricci_scalar < -threshold
    }
}

/// Johnson-Lindenstrauss random projection for dimensionality reduction
struct JLProjector {
    projection: DMatrix<f64>,
}

impl JLProjector {
    /// Create JL projector from high_dim to target_dim
    pub fn new(high_dim: usize, target_dim: usize, seed: u64) -> Self {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        
        // Use sparse JL transform with Achlioptas matrix
        // Entries are +sqrt(3), 0, -sqrt(3) with probabilities 1/6, 2/3, 1/6
        let mut projection = DMatrix::zeros(target_dim, high_dim);
        let sqrt3 = std::f64::consts::SQRT_3;

        for i in 0..target_dim {
            for j in 0..high_dim {
                let r: f64 = rng.gen();
                projection[(i, j)] = if r < 1.0 / 6.0 {
                    sqrt3
                } else if r < 0.5 {
                    -sqrt3
                } else {
                    0.0
                };
            }
        }

        // Normalize by sqrt(target_dim)
        projection /= (target_dim as f64).sqrt();

        Self { projection }
    }

    /// Project a high-dimensional vector to lower dimension
    pub fn project(&self, v: &DVector<f64>) -> Result<DVector<f64>, ManifoldError> {
        if v.len() != self.projection.ncols() {
            return Err(ManifoldError::DimensionMismatch {
                expected: self.projection.ncols(),
                actual: v.len(),
            });
        }
        Ok(self.projection * v)
    }

    /// Batch project multiple vectors
    pub fn project_batch(&self, vectors: &[DVector<f64>]) -> Result<Vec<DVector<f64>>, ManifoldError> {
        vectors.iter().map(|v| self.project(v)).collect()
    }
}

/// Local tangent space estimator using PCA
struct TangentSpaceEstimator {
    config: ManifoldConfig,
}

impl TangentSpaceEstimator {
    fn new(config: ManifoldConfig) -> Self {
        Self { config }
    }

    /// Estimate tangent space basis at a point using local neighbors
    fn estimate_basis(
        &self,
        center: &DVector<f64>,
        neighbors: &[DVector<f64>],
    ) -> Result<(DMatrix<f64>, Vec<f64>), ManifoldError> {
        if neighbors.len() < self.config.num_neighbors.min(neighbors.len()) {
            return Err(ManifoldError::InsufficientPoints {
                min: 3,
                actual: neighbors.len(),
            });
        }

        let n = neighbors.len().min(self.config.num_neighbors);
        let d = center.len();

        // Center the data
        let mut centered = Vec::with_capacity(n);
        let mut mean = DVector::zeros(d);

        for neighbor in neighbors.iter().take(n) {
            centered.push(neighbor - center);
            mean += neighbor;
        }
        mean /= n as f64;

        // Build covariance matrix
        let mut cov = Matrix2::zeros();
        if d == 2 {
            for v in &centered {
                cov += v.fixed_rows::<2>(0) * v.fixed_rows::<2>(0).transpose();
            }
        } else {
            // For higher dimensions, use randomized SVD
            return self.randomized_pca(&centered, self.config.target_dim.min(d - 1));
        }

        // Eigendecomposition for 2D case
        let eigen = SymmetricEigen::new(cov);
        let basis = DMatrix::from_columns(&[eigen.eigenvectors.column(0), eigen.eigenvectors.column(1)]);
        let eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();

        Ok((basis, eigenvalues))
    }

    fn randomized_pca(
        &self,
        data: &[DVector<f64>],
        k: usize,
    ) -> Result<(DMatrix<f64>, Vec<f64>), ManifoldError> {
        if data.is_empty() {
            return Err(ManifoldError::InsufficientPoints { min: 1, actual: 0 });
        }

        let n = data.len();
        let d = data[0].len();
        let k = k.min(n - 1).min(d);

        // Build data matrix
        let mut X = DMatrix::zeros(d, n);
        for (i, v) in data.iter().enumerate() {
            X.set_column(i, v);
        }

        // Randomized SVD
        let svd = SVD::new(X, true, true);
        
        let u = svd.u.ok_or(ManifoldError::SVDFailure)?;
        let s = svd.singular_values;

        // Take top k components
        let basis = u.columns(0, k.min(u.ncols()));
        let eigenvalues: Vec<f64> = s.iter().take(k).map(|&v| v * v).collect();

        Ok((basis, eigenvalues))
    }
}

/// Ricci curvature calculator for semantic manifolds
pub struct RicciCurvatureCalculator {
    config: ManifoldConfig,
    jl_projector: Option<JLProjector>,
    tangent_estimator: TangentSpaceEstimator,
}

impl RicciCurvatureCalculator {
    pub fn new(config: ManifoldConfig) -> Self {
        Self {
            jl_projector: None,
            tangent_estimator: TangentSpaceEstimator::new(config.clone()),
            config,
        }
    }

    /// Initialize with expected input dimension
    pub fn with_dimension(mut self, input_dim: usize) -> Self {
        self.jl_projector = Some(JLProjector::new(
            input_dim,
            self.config.target_dim,
            self.config.seed,
        ));
        self
    }

    /// Compute Ricci curvature at a point given its neighbors
    pub fn compute_ricci_at_point(
        &self,
        center: &ManifoldPoint,
        neighbors: &[ManifoldPoint],
    ) -> Result<CurvatureEstimate, ManifoldError> {
        // Project to lower dimension if JL is initialized
        let projected_center = if let Some(ref jl) = self.jl_projector {
            jl.project(&center.embedding)?
        } else {
            center.embedding.clone()
        };

        let projected_neighbors: Vec<_> = neighbors
            .iter()
            .filter_map(|n| {
                if let Some(ref jl) = self.jl_projector {
                    jl.project(&n.embedding).ok()
                } else {
                    Some(n.embedding.clone())
                }
            })
            .collect();

        if projected_neighbors.len() < 3 {
            return Err(ManifoldError::InsufficientPoints {
                min: 3,
                actual: projected_neighbors.len(),
            });
        }

        // Estimate tangent space
        let (_basis, eigenvalues) = self.tangent_estimator.estimate_basis(
            &projected_center,
            &projected_neighbors,
        )?;

        // Compute Ricci scalar from eigenvalues of shape operator
        // In 2D: Ricci scalar = 2 * Gaussian curvature = 2 * (lambda1 * lambda2)
        // Higher dimensions: trace of Ricci tensor
        let ricci_scalar = if eigenvalues.len() >= 2 {
            let lambda1 = eigenvalues[0].max(self.config.epsilon);
            let lambda2 = eigenvalues[1].max(self.config.epsilon);
            
            // Regularized curvature estimate
            let gaussian_curvature = (lambda1 * lambda2).sqrt() / (lambda1 + lambda2 + self.config.epsilon);
            2.0 * gaussian_curvature
        } else {
            0.0
        };

        // Sectional curvatures (simplified for 2D subspace)
        let sectional_curvatures = eigenvalues
            .windows(2)
            .map(|w| {
                let num = w[0] * w[1];
                let denom = (w[0] + w[1] + self.config.epsilon).powi(2);
                num / denom
            })
            .collect();

        // Confidence based on eigenvalue gap and neighbor count
        let confidence = if eigenvalues.len() >= 2 {
            let gap = (eigenvalues[0] - eigenvalues[1]).abs();
            let gap_factor = 1.0 / (1.0 + gap);
            let count_factor = (projected_neighbors.len() as f64 / self.config.num_neighbors as f64).min(1.0);
            gap_factor * count_factor
        } else {
            0.1
        };

        Ok(CurvatureEstimate {
            point_id: center.id,
            ricci_scalar,
            sectional_curvatures,
            confidence,
        })
    }

    /// Compute curvature evolution over time series of points
    pub fn compute_curvature_evolution(
        &self,
        points: &[ManifoldPoint],
    ) -> Result<Vec<CurvatureEstimate>, ManifoldError> {
        let mut estimates = Vec::with_capacity(points.len());

        for (i, center) in points.iter().enumerate() {
            // Get temporal neighbors (points within time window)
            let time_window = 10; // Use 10 adjacent points
            let start = i.saturating_sub(time_window / 2);
            let end = (i + time_window / 2).min(points.len() - 1);

            let neighbors: Vec<_> = points[start..=end]
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, p)| p.clone())
                .collect();

            if neighbors.len() >= 3 {
                if let Ok(est) = self.compute_ricci_at_point(center, &neighbors) {
                    estimates.push(est);
                }
            }
        }

        Ok(estimates)
    }

    /// Detect paradigm shifts from curvature time series
    pub fn detect_paradigm_shifts(
        &self,
        estimates: &[CurvatureEstimate],
        threshold: f64,
    ) -> Vec<usize> {
        estimates
            .iter()
            .enumerate()
            .filter(|(_, est)| est.is_paradigm_shift(threshold))
            .map(|(i, _)| i)
            .collect()
    }
}

/// Streaming curvature tracker for real-time paradigm shift detection
pub struct StreamingCurvatureTracker {
    calculator: RicciCurvatureCalculator,
    recent_points: Vec<ManifoldPoint>,
    max_buffer_size: usize,
    baseline_curvature: Option<f64>,
    curvature_history: Vec<f64>,
}

impl StreamingCurvatureTracker {
    pub fn new(config: ManifoldConfig, max_buffer_size: usize) -> Self {
        Self {
            calculator: RicciCurvatureCalculator::new(config),
            recent_points: Vec::new(),
            max_buffer_size,
            baseline_curvature: None,
            curvature_history: Vec::new(),
        }
    }

    /// Add a new point and update curvature estimate
    pub fn update(&mut self, point: ManifoldPoint) -> Option<CurvatureEstimate> {
        self.recent_points.push(point);
        
        // Maintain buffer size
        if self.recent_points.len() > self.max_buffer_size {
            self.recent_points.remove(0);
        }

        // Need minimum points for curvature estimation
        if self.recent_points.len() < 5 {
            return None;
        }

        let center_idx = self.recent_points.len() - 1;
        let center = &self.recent_points[center_idx];
        
        let neighbors: Vec<_> = self.recent_points[..center_idx]
            .iter()
            .cloned()
            .collect();

        if let Ok(est) = self.calculator.compute_ricci_at_point(center, &neighbors) {
            self.curvature_history.push(est.ricci_scalar);
            
            // Update baseline after collecting enough samples
            if self.baseline_curvature.is_none() && self.curvature_history.len() >= 20 {
                let sum: f64 = self.curvature_history.iter().sum();
                self.baseline_curvature = Some(sum / self.curvature_history.len() as f64);
            }

            Some(est)
        } else {
            None
        }
    }

    /// Check if current curvature indicates paradigm shift relative to baseline
    pub fn is_paradigm_shift(&self, threshold: f64) -> bool {
        if let Some(baseline) = self.baseline_curvature {
            if let Some(current) = self.curvature_history.last() {
                let deviation = current - baseline;
                deviation < -threshold
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Get curvature z-score (normalized deviation from baseline)
    pub fn curvature_zscore(&self) -> Option<f64> {
        if self.curvature_history.len() < 3 {
            return None;
        }

        let mean: f64 = self.curvature_history.iter().sum::<f64>() / self.curvature_history.len() as f64;
        let variance: f64 = self.curvature_history.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / self.curvature_history.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            return None;
        }

        self.curvature_history.last().map(|&current| (current - mean) / std_dev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jl_projection() {
        let projector = JLProjector::new(1000, 50, 42);
        let v = DVector::from_fn(1000, |i, _| i as f64);
        
        let projected = projector.project(&v).unwrap();
        assert_eq!(projected.len(), 50);
    }

    #[test]
    fn test_curvature_detection() {
        let config = ManifoldConfig::default();
        let calculator = RicciCurvatureCalculator::new(config).with_dimension(100);

        // Create synthetic manifold points
        let points: Vec<_> = (0..20)
            .map(|i| ManifoldPoint {
                id: i,
                timestamp: i as f64,
                embedding: DVector::from_fn(100, |j, _| ((i + j) as f64 * 0.1).sin()),
            })
            .collect();

        let estimates = calculator.compute_curvature_evolution(&points);
        assert!(estimates.is_ok());
        assert!(!estimates.unwrap().is_empty());
    }

    #[test]
    fn test_streaming_tracker() {
        let config = ManifoldConfig::default();
        let mut tracker = StreamingCurvatureTracker::new(config, 100);

        for i in 0..50 {
            let point = ManifoldPoint {
                id: i,
                timestamp: i as f64,
                embedding: DVector::from_fn(100, |j, _| ((i + j) as f64 * 0.1).sin()),
            };
            tracker.update(point);
        }

        let zscore = tracker.curvature_zscore();
        assert!(zscore.is_some());
    }
}
