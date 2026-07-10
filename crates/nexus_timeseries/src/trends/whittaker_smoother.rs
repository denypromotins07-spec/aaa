//! Whittaker-Henderson Smoothing Spline
//! Zero-allocation implementation using sparse banded solver.

use super::sparse_banded_solver::SparseBandedSolver;

/// Whittaker smoother parameters
#[derive(Debug, Clone)]
pub struct WhittakerParams {
    /// Smoothing parameter (lambda) - higher = smoother
    pub lambda: f64,
    /// Order of difference penalty (typically 2)
    pub order: usize,
}

impl Default for WhittakerParams {
    fn default() -> Self {
        Self { lambda: 100.0, order: 2 }
    }
}

/// Whittaker-Henderson smoother for trend extraction
pub struct WhittakerSmoother {
    params: WhittakerParams,
    solver: Option<SparseBandedSolver>,
    buffer_size: usize,
}

impl WhittakerSmoother {
    pub fn new(params: WhittakerParams, max_size: usize) -> Self {
        let solver = SparseBandedSolver::new(max_size, params.order);
        Self {
            params,
            solver,
            buffer_size: max_size,
        }
    }

    /// Smooth a time series
    pub fn smooth(&mut self, data: &[f64]) -> Option<Vec<f64>> {
        if data.is_empty() {
            return None;
        }

        let n = data.len();
        let solver = self.solver.as_mut()?;
        
        // Solve penalized least squares: (I + λD'D)x = y
        solver.solve_whittaker(data, self.params.lambda)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoothing() {
        let params = WhittakerParams::default();
        let mut smoother = WhittakerSmoother::new(params, 1000);
        
        // Noisy signal
        let data: Vec<f64> = (0..100)
            .map(|i| (i as f64 * 0.1).sin() + (i as f64).cos() * 0.5)
            .collect();
        
        let smoothed = smoother.smooth(&data);
        assert!(smoothed.is_some());
        assert_eq!(smoothed.unwrap().len(), data.len());
    }
}
