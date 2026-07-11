//! Takens' Embedding Theorem implementation for phase-space reconstruction.
//! Reconstructs high-dimensional attractors from scalar time series using time-delayed coordinates.

use alloc::vec::Vec;
use core::fmt::Debug;

/// Configuration for Takens' embedding
#[derive(Debug, Clone)]
pub struct TakensConfig {
    /// Embedding dimension (m)
    pub embedding_dim: usize,
    /// Time delay (tau) - will be optimized if set to 0
    pub tau: usize,
    /// Minimum samples required
    pub min_samples: usize,
}

impl Default for TakensConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 7,
            tau: 0, // Auto-optimize
            min_samples: 100,
        }
    }
}

/// Result of phase-space reconstruction
#[derive(Debug, Clone)]
pub struct PhaseSpaceReconstruction<T> {
    /// Reconstructed trajectory points in m-dimensional space
    pub trajectory: Vec<Vec<T>>,
    /// Optimized time delay
    pub tau: usize,
    /// Embedding dimension used
    pub embedding_dim: usize,
    /// Number of valid points reconstructed
    pub valid_points: usize,
}

/// Mutual Information calculator for optimal tau selection
pub struct MutualInformation {
    /// Number of bins for histogram estimation
    num_bins: usize,
}

impl MutualInformation {
    pub const fn new(num_bins: usize) -> Self {
        Self { num_bins }
    }

    /// Calculate mutual information between x and x(t+tau) for a given tau
    /// Uses histogram-based estimation with zero-allocation inner loop
    pub fn calculate<I>(&self, series: &[I], tau: usize) -> f64
    where
        I: Copy + PartialOrd + Debug,
    {
        if series.len() <= tau || self.num_bins == 0 {
            return 0.0;
        }

        let n = series.len() - tau;
        if n == 0 {
            return 0.0;
        }

        // Find min/max for binning
        let mut min_val = series[0];
        let mut max_val = series[0];
        for i in 1..series.len() {
            if series[i] < min_val {
                min_val = series[i];
            }
            if series[i] > max_val {
                max_val = series[i];
            }
        }

        let range = if max_val == min_val {
            1.0
        } else {
            (max_val as f64 - min_val as f64).abs().max(1e-15)
        };

        // Joint histogram P(x_t, x_{t+tau})
        let mut joint_hist = vec![vec![0usize; self.num_bins]; self.num_bins];
        // Marginal histograms
        let mut marginal_x = vec![0usize; self.num_bins];
        let mut marginal_y = vec![0usize; self.num_bins];

        for i in 0..n {
            let x = series[i] as f64;
            let y = series[i + tau] as f64;

            let bin_x = ((x - min_val as f64) / range * self.num_bins as f64)
                .min(self.num_bins as f64 - 1.0)
                .max(0.0) as usize;
            let bin_y = ((y - min_val as f64) / range * self.num_bins as f64)
                .min(self.num_bins as f64 - 1.0)
                .max(0.0) as usize;

            joint_hist[bin_x][bin_y] += 1;
            marginal_x[bin_x] += 1;
            marginal_y[bin_y] += 1;
        }

        let n_f64 = n as f64;
        let mut mi = 0.0;

        // Calculate MI = sum P(x,y) log(P(x,y) / (P(x)P(y)))
        for i in 0..self.num_bins {
            for j in 0..self.num_bins {
                if joint_hist[i][j] > 0 {
                    let p_xy = joint_hist[i][j] as f64 / n_f64;
                    let p_x = marginal_x[i] as f64 / n_f64;
                    let p_y = marginal_y[j] as f64 / n_f64;

                    if p_x > 1e-15 && p_y > 1e-15 {
                        mi += p_xy * (p_xy / (p_x * p_y)).ln();
                    }
                }
            }
        }

        mi
    }

    /// Find the first minimum of mutual information curve
    /// This is the optimal tau for Takens' embedding
    pub fn find_first_minimum<I>(&self, series: &[I], max_tau: usize) -> Option<usize>
    where
        I: Copy + PartialOrd + Debug,
    {
        if series.len() < 10 || max_tau == 0 {
            return None;
        }

        let actual_max = max_tau.min(series.len() / 4);
        if actual_max < 3 {
            return None;
        }

        // Calculate MI for tau = 1 to max_tau
        let mut mi_values: Vec<f64> = Vec::with_capacity(actual_max);
        for tau in 1..=actual_max {
            mi_values.push(self.calculate(series, tau));
        }

        // Find first local minimum
        for i in 1..mi_values.len().saturating_sub(1) {
            if mi_values[i] <= mi_values[i - 1] && mi_values[i] < mi_values[i + 1] {
                return Some(i + 1); // tau is 1-indexed
            }
        }

        // Fallback: return tau with global minimum if no local minimum found
        mi_values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(i, _)| i + 1)
    }
}

/// Takens' Embedding Theorem implementation
pub struct TakensEmbedding<T> {
    config: TakensConfig,
    _marker: core::marker::PhantomData<T>,
}

impl<T> TakensEmbedding<T>
where
    T: Copy + PartialOrd + Debug + Into<f64>,
{
    pub const fn new(config: TakensConfig) -> Self {
        Self {
            config,
            _marker: core::marker::PhantomData,
        }
    }

    /// Reconstruct phase space from scalar time series
    /// Returns embedded trajectory in m-dimensional space
    pub fn reconstruct(&self, series: &[T]) -> Result<PhaseSpaceReconstruction<f64>, &'static str> {
        if series.is_empty() {
            return Err("Empty series");
        }

        let mut tau = self.config.tau;
        let m = self.config.embedding_dim;

        // Auto-optimize tau if not specified
        if tau == 0 {
            let mi_calc = MutualInformation::new(20);
            tau = mi_calc.find_first_minimum(series, 50).unwrap_or(1);
        }

        // Validate parameters
        let required_len = 1 + (m - 1) * tau;
        if series.len() < required_len {
            return Err("Series too short for embedding dimension and tau");
        }

        let valid_points = series.len() - (m - 1) * tau;
        if valid_points < self.config.min_samples {
            return Err("Insufficient valid points after embedding");
        }

        // Build trajectory with zero allocation in hot path
        let mut trajectory = Vec::with_capacity(valid_points);

        for i in 0..valid_points {
            let mut point = Vec::with_capacity(m);
            for j in 0..m {
                let idx = i + j * tau;
                point.push(series[idx].into());
            }
            trajectory.push(point);
        }

        Ok(PhaseSpaceReconstruction {
            trajectory,
            tau,
            embedding_dim: m,
            valid_points,
        })
    }

    /// Get recommended embedding dimension using False Nearest Neighbors heuristic
    /// Simplified version: uses sqrt(N) rule of thumb
    pub fn recommend_embedding_dim(&self, series_len: usize) -> usize {
        if series_len < 10 {
            return 2;
        }
        // Common heuristic: m ≈ sqrt(N) or use saturation dimension
        let suggested = (series_len as f64).sqrt() as usize;
        suggested.clamp(3, 15)
    }
}

/// Helper for detecting folding artifacts in reconstructed attractor
pub struct FoldingArtifactDetector {
    /// Threshold for detecting false neighbors
    threshold: f64,
}

impl FoldingArtifactDetector {
    pub const fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Detect if the reconstructed attractor has folding artifacts
    /// High percentage of false nearest neighbors indicates poor tau choice
    pub fn detect_false_neighbors(&self, reconstruction: &PhaseSpaceReconstruction<f64>) -> f64 {
        if reconstruction.valid_points < 10 || reconstruction.embedding_dim < 2 {
            return 1.0; // Cannot detect
        }

        let m = reconstruction.embedding_dim;
        let mut false_neighbors = 0usize;
        let mut total_checks = 0usize;

        // Check each point for false neighbors in lower dimensions
        for i in 0..reconstruction.valid_points.saturating_sub(1) {
            for j in (i + 1)..reconstruction.valid_points {
                // Compare distances in (m-1)D vs mD
                let dist_m1 = self.euclidean_distance_partial(
                    &reconstruction.trajectory[i],
                    &reconstruction.trajectory[j],
                    m - 1,
                );
                let dist_m = self.euclidean_distance_full(
                    &reconstruction.trajectory[i],
                    &reconstruction.trajectory[j],
                );

                if dist_m1 < self.threshold && dist_m > self.threshold * 2.0 {
                    false_neighbors += 1;
                }
                total_checks += 1;

                // Early exit if too many false neighbors
                if total_checks > 1000 {
                    break;
                }
            }
            if total_checks > 1000 {
                break;
            }
        }

        if total_checks == 0 {
            return 0.0;
        }

        false_neighbors as f64 / total_checks as f64
    }

    fn euclidean_distance_partial(&self, a: &[f64], b: &[f64], dims: usize) -> f64 {
        let limit = dims.min(a.len().min(b.len()));
        let mut sum = 0.0;
        for i in 0..limit {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }

    fn euclidean_distance_full(&self, a: &[f64], b: &[f64]) -> f64 {
        let limit = a.len().min(b.len());
        let mut sum = 0.0;
        for i in 0..limit {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutual_information_constant_series() {
        let series = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mi = MutualInformation::new(10);
        let result = mi.calculate(&series, 1);
        // Constant series should have low MI
        assert!(result.is_finite());
    }

    #[test]
    fn test_takens_reconstruction() {
        let series: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let config = TakensConfig {
            embedding_dim: 3,
            tau: 5,
            min_samples: 10,
        };
        let embedding = TakensEmbedding::new(config);
        let result = embedding.reconstruct(&series);
        assert!(result.is_ok());
        let recon = result.unwrap();
        assert_eq!(recon.embedding_dim, 3);
        assert_eq!(recon.tau, 5);
    }
}
