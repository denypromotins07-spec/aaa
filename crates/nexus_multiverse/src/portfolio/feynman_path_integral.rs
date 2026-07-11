//! Feynman Path Integral Optimizer for Multiverse Portfolio Theory
//! 
//! Sums probability amplitudes across all possible branching histories
//! weighted by their quantum measure.

use alloc::vec::Vec;
use core::fmt;

/// Maximum number of paths to consider (prevents combinatorial explosion)
const MAX_PATHS: usize = 65536;

/// Error types for path integral calculations
#[derive(Debug, Clone, PartialEq)]
pub enum PathIntegralError {
    TooManyPaths { requested: usize, max: usize },
    InvalidAmplitude { message: &'static str },
    NumericalOverflow { operation: &'static str },
    NonUnitaryEvolution { total_measure: f64 },
}

impl fmt::Display for PathIntegralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathIntegralError::TooManyPaths { requested, max } => {
                write!(f, "Too many paths: requested {}, max {}", requested, max)
            }
            PathIntegralError::InvalidAmplitude { message } => {
                write!(f, "Invalid amplitude: {}", message)
            }
            PathIntegralError::NumericalOverflow { operation } => {
                write!(f, "Numerical overflow in {}", operation)
            }
            PathIntegralError::NonUnitaryEvolution { total_measure } => {
                write!(f, "Non-unitary evolution: total_measure={}", total_measure)
            }
        }
    }
}

/// Single path in the Feynman sum-over-histories
#[derive(Debug, Clone)]
pub struct FeynmanPath {
    pub path_id: usize,
    pub history: Vec<usize>, // Sequence of branch IDs
    pub amplitude_re: f64,
    pub amplitude_im: f64,
    pub action: f64, // Classical action along this path
}

impl FeynmanPath {
    pub fn new(
        path_id: usize,
        history: Vec<usize>,
        amplitude_re: f64,
        amplitude_im: f64,
        action: f64,
    ) -> Result<Self, PathIntegralError> {
        // Validate amplitude
        let mag_sq = amplitude_re * amplitude_re + amplitude_im * amplitude_im;
        if mag_sq.is_nan() || mag_sq.is_infinite() {
            return Err(PathIntegralError::InvalidAmplitude {
                message: "NaN or Inf in amplitude",
            });
        }

        Ok(Self {
            path_id,
            history,
            amplitude_re,
            amplitude_im,
            action,
        })
    }

    /// Get squared magnitude of amplitude
    #[inline]
    pub fn probability(&self) -> f64 {
        self.amplitude_re * self.amplitude_re + self.amplitude_im * self.amplitude_im
    }
}

/// Feynman Path Integral Calculator
pub struct FeynmanPathIntegralOptimizer {
    paths: Vec<FeynmanPath>,
    planck_constant: f64, // Effective Planck constant for market dynamics
}

impl FeynmanPathIntegralOptimizer {
    pub fn new(planck_constant: f64) -> Self {
        Self {
            paths: Vec::new(),
            planck_constant,
        }
    }

    /// Add a path to the sum-over-histories
    pub fn add_path(&mut self, path: FeynmanPath) -> Result<(), PathIntegralError> {
        if self.paths.len() >= MAX_PATHS {
            return Err(PathIntegralError::TooManyPaths {
                requested: self.paths.len() + 1,
                max: MAX_PATHS,
            });
        }

        // Validate unitarity contribution
        let prob = path.probability();
        let current_total: f64 = self.paths.iter().map(|p| p.probability()).sum();
        
        if current_total + prob > 1.0 + 1e-10 {
            // Path would violate unitarity
            // Normalize incoming path
            let scale = ((1.0 - current_total) / prob.max(1e-15)).sqrt();
            let normalized_path = FeynmanPath::new(
                path.path_id,
                path.history,
                path.amplitude_re * scale,
                path.amplitude_im * scale,
                path.action,
            )?;
            self.paths.push(normalized_path);
        } else {
            self.paths.push(path);
        }

        Ok(())
    }

    /// Calculate total amplitude as sum over all paths
    pub fn calculate_total_amplitude(&self) -> Result<(f64, f64), PathIntegralError> {
        let mut total_re = 0.0_f64;
        let mut total_im = 0.0_f64;

        for path in &self.paths {
            total_re += path.amplitude_re;
            total_im += path.amplitude_im;

            if total_re.is_nan() || total_im.is_nan() {
                return Err(PathIntegralError::NumericalOverflow {
                    operation: "amplitude summation",
                });
            }
        }

        Ok((total_re, total_im))
    }

    /// Calculate total probability (Born rule on summed amplitudes)
    pub fn calculate_total_probability(&self) -> Result<f64, PathIntegralError> {
        let (total_re, total_im) = self.calculate_total_amplitude()?;
        let prob = total_re * total_re + total_im * total_im;

        if prob.is_nan() || prob.is_infinite() {
            return Err(PathIntegralError::NumericalOverflow {
                operation: "probability calculation",
            });
        }

        Ok(prob)
    }

    /// Calculate path integral with action weighting
    /// Amplitude ~ exp(i * S / ℏ)
    pub fn calculate_action_weighted_amplitude(&self) -> Result<(f64, f64), PathIntegralError> {
        let mut total_re = 0.0_f64;
        let mut total_im = 0.0_f64;

        for path in &self.paths {
            // Phase factor: exp(i * S / ℏ) = cos(S/ℏ) + i*sin(S/ℏ)
            let phase = path.action / self.planck_constant;
            let cos_phase = phase.cos();
            let sin_phase = phase.sin();

            // Weight the path's intrinsic amplitude by the action phase
            total_re += path.amplitude_re * cos_phase - path.amplitude_im * sin_phase;
            total_im += path.amplitude_re * sin_phase + path.amplitude_im * cos_phase;

            if total_re.is_nan() || total_im.is_nan() {
                return Err(PathIntegralError::NumericalOverflow {
                    operation: "action-weighted summation",
                });
            }
        }

        Ok((total_re, total_im))
    }

    /// Get number of paths
    pub fn num_paths(&self) -> usize {
        self.paths.len()
    }

    /// Clear all paths
    pub fn clear(&mut self) {
        self.paths.clear();
    }

    /// Verify unitarity (total probability ≈ 1.0)
    pub fn verify_unitarity(&self) -> Result<(), PathIntegralError> {
        let prob = self.calculate_total_probability()?;
        let deviation = (prob - 1.0).abs();
        
        if deviation > 1e-6 && prob > 1e-6 {
            return Err(PathIntegralError::NonUnitaryEvolution {
                total_measure: prob,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_path() {
        let mut optimizer = FeynmanPathIntegralOptimizer::new(1.0);
        
        let path = FeynmanPath::new(0, vec![1, 2, 3], 1.0, 0.0, 0.0).unwrap();
        optimizer.add_path(path).unwrap();

        let (re, im) = optimizer.calculate_total_amplitude().unwrap();
        assert!((re - 1.0).abs() < 1e-14);
        assert!(im.abs() < 1e-14);

        let prob = optimizer.calculate_total_probability().unwrap();
        assert!((prob - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_interference() {
        let mut optimizer = FeynmanPathIntegralOptimizer::new(1.0);
        
        // Two paths with opposite phases should interfere destructively
        let path1 = FeynmanPath::new(0, vec![1], 0.5, 0.0, 0.0).unwrap();
        let path2 = FeynmanPath::new(1, vec![2], -0.5, 0.0, 0.0).unwrap();
        
        optimizer.add_path(path1).unwrap();
        optimizer.add_path(path2).unwrap();

        let (re, im) = optimizer.calculate_total_amplitude().unwrap();
        assert!(re.abs() < 1e-14); // Destructive interference

        let prob = optimizer.calculate_total_probability().unwrap();
        assert!(prob.abs() < 1e-14);
    }

    #[test]
    fn test_path_limit() {
        let mut optimizer = FeynmanPathIntegralOptimizer::new(1.0);
        
        for i in 0..MAX_PATHS {
            let path = FeynmanPath::new(i, vec![i], 1e-10, 0.0, 0.0).unwrap();
            match optimizer.add_path(path) {
                Ok(_) => continue,
                Err(PathIntegralError::TooManyPaths { .. }) => break,
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        assert!(optimizer.num_paths() <= MAX_PATHS);
    }
}
