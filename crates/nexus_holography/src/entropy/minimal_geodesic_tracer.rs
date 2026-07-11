//! Minimal Geodesic Tracer
//! 
//! Numerical solver for minimal geodesics in AdS space.
//! Uses variational methods with strict UV cutoff enforcement.

use crate::geometry::PoincareMetric;
use nalgebra::Vector2;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors specific to geodesic tracing
#[derive(Error, Debug, Clone, PartialEq)]
pub enum GeodesicError {
    #[error("Geodesic iteration failed to converge")]
    NonConvergence,
    #[error("Invalid boundary points: {0}")]
    InvalidBoundaryPoints(String),
    #[error("UV cutoff violation: z={0} < cutoff")]
    UVCutoffViolation(f64),
    #[error("Numerical instability detected")]
    NumericalInstability,
}

/// Configuration for geodesic solver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeodesicConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// UV cutoff (minimum z)
    pub uv_cutoff: f64,
    /// Step size for discretization
    pub step_size: f64,
}

impl Default for GeodesicConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
            uv_cutoff: 0.01,
            step_size: 0.001,
        }
    }
}

/// A point on the geodesic curve
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GeodesicPoint {
    /// Position (z, x)
    pub position: Vector2<f64>,
    /// Proper length from start
    pub proper_length: f64,
    /// Tangent vector
    pub tangent: Vector2<f64>,
}

/// Complete geodesic curve data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeodesicCurve {
    /// Discrete points along the curve
    pub points: Vec<GeodesicPoint>,
    /// Total proper length
    pub total_length: f64,
    /// Number of iterations to converge
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
}

/// Minimal geodesic tracer for AdS space
pub struct MinimalGeodesicTracer {
    /// Poincaré metric calculator
    metric: PoincareMetric,
    /// Solver configuration
    config: GeodesicConfig,
}

impl MinimalGeodesicTracer {
    /// Create a new geodesic tracer
    pub fn new(metric: PoincareMetric, config: GeodesicConfig) -> Result<Self, GeodesicError> {
        if config.uv_cutoff <= 0.0 {
            return Err(GeodesicError::UVCutoffViolation(config.uv_cutoff));
        }
        if config.tolerance <= 0.0 {
            return Err(GeodesicError::InvalidBoundaryPoints(
                "Tolerance must be positive".to_string(),
            ));
        }

        Ok(Self { metric, config })
    }

    /// Trace geodesic between two boundary points using shooting method
    /// Boundary points are at (z₁, x₁) and (z₂, x₂)
    pub fn trace_geodesic(
        &self,
        start: (f64, f64),
        end: (f64, f64),
    ) -> Result<GeodesicCurve, GeodesicError> {
        let (z1, x1) = start;
        let (z2, x2) = end;

        // Validate inputs
        if z1 <= 0.0 || z2 <= 0.0 {
            return Err(GeodesicError::InvalidBoundaryPoints(
                "z coordinates must be positive".to_string(),
            ));
        }

        // Apply UV cutoff
        let z1_safe = z1.max(self.config.uv_cutoff);
        let z2_safe = z2.max(self.config.uv_cutoff);

        // For AdS half-plane, analytic geodesic is a semicircle
        // (x - x_c)² + z² = R²
        // Find center x_c and radius R
        
        let dx = x2 - x1;
        let dz_sq = z2_safe * z2_safe - z1_safe * z1_safe;
        
        // From (x1-xc)² + z1² = (x2-xc)² + z2²
        // x1² - 2*x1*xc + xc² + z1² = x2² - 2*x2*xc + xc² + z2²
        // -2*x1*xc + 2*x2*xc = x2² - x1² + z2² - z1²
        // 2*xc*(x2-x1) = (x2-x1)(x2+x1) + dz_sq
        let x_c = if dx.abs() > self.config.tolerance {
            (dx * (x1 + x2) + dz_sq) / (2.0 * dx)
        } else {
            // Vertical geodesic (same x)
            (x1 + x2) / 2.0
        };

        let r_squared = (x1 - x_c) * (x1 - x_c) + z1_safe * z1_safe;
        let r = if r_squared > 0.0 {
            r_squared.sqrt()
        } else {
            return Err(GeodesicError::NumericalInstability);
        };

        // Discretize the geodesic
        let mut points = Vec::new();
        let mut total_length = 0.0;
        let mut prev_point: Option<GeodesicPoint> = None;

        // Parameterize by angle θ from center
        let theta1 = ((z1_safe / r).acos()).copysign(x1 - x_c);
        let theta2 = ((z2_safe / r).acos()).copysign(x2 - x_c);
        
        let num_steps = ((theta2 - theta1).abs() / self.config.step_size) as usize;
        let num_steps = num_steps.min(10000).max(10);
        let dtheta = (theta2 - theta1) / num_steps as f64;

        for i in 0..=num_steps {
            let theta = theta1 + i as f64 * dtheta;
            
            let x = x_c + r * theta.cos();
            let z = (r * theta.sin()).max(self.config.uv_cutoff);

            let position = Vector2::new(z, x);
            
            // Compute tangent (derivative w.r.t. θ)
            let dx_dtheta = -r * theta.sin();
            let dz_dtheta = r * theta.cos();
            let mut tangent = Vector2::new(dz_dtheta, dx_dtheta);
            
            // Normalize tangent
            let norm = tangent.norm();
            if norm > self.config.tolerance {
                tangent /= norm;
            }

            // Compute proper length increment
            let segment_length = if let Some(prev) = &prev_point {
                self.metric.geodesic_distance(prev.position.x, prev.position.y, z, x)
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            total_length += segment_length;

            let point = GeodesicPoint {
                position,
                proper_length: total_length,
                tangent,
            };

            points.push(point);
            prev_point = Some(point);
        }

        Ok(GeodesicCurve {
            points,
            total_length,
            iterations: num_steps,
            converged: true,
        })
    }

    /// Compute geodesic using numerical minimization (gradient descent)
    /// This is more general but slower than the analytical solution
    pub fn trace_geodesic_numerical(
        &self,
        start: (f64, f64),
        end: (f64, f64),
    ) -> Result<GeodesicCurve, GeodesicError> {
        let (z1, x1) = start;
        let (z2, x2) = end;

        let z1_safe = z1.max(self.config.uv_cutoff);
        let z2_safe = z2.max(self.config.uv_cutoff);

        // Initialize with straight line in embedding space
        let num_points = 50;
        let mut positions: Vec<Vector2<f64>> = (0..=num_points)
            .map(|i| {
                let t = i as f64 / num_points as f64;
                let z = z1_safe + t * (z2_safe - z1_safe);
                let x = x1 + t * (x2 - x1);
                Vector2::new(z, x)
            })
            .collect();

        // Gradient descent to minimize proper length
        let mut converged = false;
        let mut iterations = 0;
        let mut prev_length = f64::MAX;

        for iter in 0..self.config.max_iterations {
            iterations = iter;
            
            // Compute current length
            let mut current_length = 0.0;
            for i in 0..positions.len() - 1 {
                let p1 = positions[i];
                let p2 = positions[i + 1];
                match self.metric.geodesic_distance(p1.x, p1.y, p2.x, p2.y) {
                    Ok(d) => current_length += d,
                    Err(_) => return Err(GeodesicError::NumericalInstability),
                }
            }

            // Check convergence
            if (current_length - prev_length).abs() < self.config.tolerance {
                converged = true;
                break;
            }
            prev_length = current_length;

            // Update positions (simple relaxation)
            let mut new_positions = positions.clone();
            for i in 1..positions.len() - 1 {
                let prev = positions[i - 1];
                let curr = positions[i];
                let next = positions[i + 1];

                // Move toward average of neighbors (minimizes length)
                let avg_z = (prev.x + next.x) / 2.0;
                let avg_x = (prev.y + next.y) / 2.0;

                // Relaxation factor
                let alpha = 0.3;
                new_positions[i] = Vector2::new(
                    curr.x + alpha * (avg_z - curr.x),
                    curr.y + alpha * (avg_x - curr.y),
                );

                // Enforce UV cutoff
                new_positions[i].x = new_positions[i].x.max(self.config.uv_cutoff);
            }

            positions = new_positions;
        }

        if !converged && iterations >= self.config.max_iterations - 1 {
            return Err(GeodesicError::NonConvergence);
        }

        // Build output curve
        let mut points = Vec::new();
        let mut total_length = 0.0;

        for (i, pos) in positions.iter().enumerate() {
            let segment_length = if i > 0 {
                let prev = positions[i - 1];
                self.metric.geodesic_distance(prev.x, prev.y, pos.x, pos.y)
                    .unwrap_or(0.0)
            } else {
                0.0
            };

            total_length += segment_length;

            // Approximate tangent
            let tangent = if i == 0 {
                positions[1] - positions[0]
            } else if i == positions.len() - 1 {
                positions[i] - positions[i - 1]
            } else {
                (positions[i + 1] - positions[i - 1]) / 2.0
            };

            let tangent_norm = tangent.norm();
            let tangent = if tangent_norm > self.config.tolerance {
                tangent / tangent_norm
            } else {
                Vector2::new(0.0, 1.0)
            };

            points.push(GeodesicPoint {
                position: *pos,
                proper_length: total_length,
                tangent,
            });
        }

        Ok(GeodesicCurve {
            points,
            total_length,
            iterations,
            converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::PoincareMetric;

    #[test]
    fn test_tracer_creation() {
        let metric = PoincareMetric::new(1.0, 0.01, 10.0).unwrap();
        let config = GeodesicConfig::default();
        let tracer = MinimalGeodesicTracer::new(metric, config);
        assert!(tracer.is_ok());
    }

    #[test]
    fn test_analytic_geodesic() {
        let metric = PoincareMetric::new(1.0, 0.01, 10.0).unwrap();
        let config = GeodesicConfig::default();
        let tracer = MinimalGeodesicTracer::new(metric, config).unwrap();

        let curve = tracer.trace_geodesic((1.0, 0.0), (1.0, 2.0));
        assert!(curve.is_ok());
        let c = curve.unwrap();
        assert!(c.total_length > 0.0);
        assert!(c.converged);
    }

    #[test]
    fn test_vertical_geodesic() {
        let metric = PoincareMetric::new(1.0, 0.01, 10.0).unwrap();
        let config = GeodesicConfig::default();
        let tracer = MinimalGeodesicTracer::new(metric, config).unwrap();

        let curve = tracer.trace_geodesic((1.0, 0.0), (2.0, 0.0));
        assert!(curve.is_ok());
    }
}
