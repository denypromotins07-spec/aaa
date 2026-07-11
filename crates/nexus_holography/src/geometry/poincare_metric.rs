//! Poincaré Metric Tensor Calculator
//! 
//! Implements the Poincaré Half-Plane and Disk models for AdS space.
//! Handles hyperbolic geometry with strict boundary regularization to prevent singularities.

use nalgebra::{Matrix2, Vector2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors specific to Poincaré metric calculations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum PoincareError {
    #[error("Coordinate z must be positive, got {0}")]
    InvalidZCoordinate(f64),
    #[error("Point outside Poincaré disk: radius {0} > 1")]
    OutsideDisk(f64),
    #[error("Numerical overflow detected")]
    NumericalOverflow,
    #[error("Boundary singularity approached: z={0} below cutoff")]
    BoundarySingularity(f64),
}

/// Coordinate representation in AdS space
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AdsCoordinates {
    /// Poincaré half-plane: (z, x) where z > 0
    HalfPlane { z: f64, x: f64 },
    /// Poincaré disk: (r, θ) where r < 1
    Disk { r: f64, theta: f64 },
}

/// Metric tensor components g_μν at a point in AdS space
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MetricTensor {
    /// g_zz component
    pub g_zz: f64,
    /// g_xx component  
    pub g_xx: f64,
    /// g_zx off-diagonal component
    pub g_zx: f64,
    /// Determinant of the metric
    pub determinant: f64,
    /// Inverse metric g^μν (diagonal elements)
    pub g_inv_zz: f64,
    pub g_inv_xx: f64,
}

impl MetricTensor {
    /// Create a new metric tensor with validation
    pub fn new(g_zz: f64, g_xx: f64, g_zx: f64) -> Result<Self, PoincareError> {
        let det = g_zz * g_xx - g_zx * g_zx;
        
        if !det.is_finite() || det <= 0.0 {
            return Err(PoincareError::NumericalOverflow);
        }

        Ok(Self {
            g_zz,
            g_xx,
            g_zx,
            determinant: det,
            g_inv_zz: g_xx / det,
            g_inv_xx: g_zz / det,
        })
    }
}

/// Poincaré metric calculator for AdS space
pub struct PoincareMetric {
    /// AdS radius L
    pub ads_radius: f64,
    /// Minimum z cutoff to prevent boundary singularities (z → 0)
    pub z_min_cutoff: f64,
    /// Maximum z value (IR cutoff)
    pub z_max_cutoff: f64,
    /// Asymptotic expansion threshold
    pub asymptotic_threshold: f64,
}

impl PoincareMetric {
    /// Create a new Poincaré metric calculator
    pub fn new(ads_radius: f64, z_min: f64, z_max: f64) -> Result<Self, PoincareError> {
        if ads_radius <= 0.0 {
            return Err(PoincareError::InvalidZCoordinate(ads_radius));
        }
        if z_min <= 0.0 {
            return Err(PoincareError::InvalidZCoordinate(z_min));
        }
        if z_max <= z_min {
            return Err(PoincareError::InvalidZCoordinate(z_max));
        }

        Ok(Self {
            ads_radius,
            z_min_cutoff: z_min,
            z_max_cutoff: z_max,
            asymptotic_threshold: z_min * 10.0,
        })
    }

    /// Compute metric in Poincaré half-plane coordinates
    /// ds² = (L²/z²)(dz² + dx²)
    /// 
    /// Uses asymptotic series expansion near z → 0 to prevent overflow
    pub fn half_plane_metric(&self, z: f64, _x: f64) -> Result<MetricTensor, PoincareError> {
        // Enforce UV cutoff to prevent boundary singularity
        let z_safe = z.max(self.z_min_cutoff);
        
        if z_safe > self.z_max_cutoff {
            return Err(PoincareError::InvalidZCoordinate(z_safe));
        }

        // Check if we're in the asymptotic regime
        let conformal_factor = if z_safe < self.asymptotic_threshold {
            // Use asymptotic expansion for small z
            // L²/z² = L² * (1/z²) but regularized
            let l_over_z = self.ads_radius / z_safe;
            // Prevent overflow by capping
            if l_over_z > 1e150 {
                return Err(PoincareError::NumericalOverflow);
            }
            l_over_z * l_over_z
        } else {
            let l_over_z = self.ads_radius / z_safe;
            l_over_z * l_over_z
        };

        // For Poincaré half-plane: g_zz = g_xx = L²/z², g_zx = 0
        MetricTensor::new(conformal_factor, conformal_factor, 0.0)
    }

    /// Compute metric in Poincaré disk coordinates
    /// ds² = 4L²/(1-r²)² (dr² + r²dθ²)
    pub fn disk_metric(&self, r: f64, _theta: f64) -> Result<MetricTensor, PoincareError> {
        if r < 0.0 || r >= 1.0 {
            return Err(PoincareError::OutsideDisk(r));
        }

        // Regularize near boundary r → 1
        let r_safe = r.min(1.0 - self.z_min_cutoff);
        
        let one_minus_r2 = 1.0 - r_safe * r_safe;
        
        if one_minus_r2 < self.z_min_cutoff {
            // Near boundary, use regulated expression
            let conformal_factor = 4.0 * self.ads_radius * self.ads_radius 
                / (self.z_min_cutoff * self.z_min_cutoff);
            
            if !conformal_factor.is_finite() {
                return Err(PoincareError::NumericalOverflow);
            }
            
            return MetricTensor::new(
                conformal_factor,
                conformal_factor * r_safe * r_safe,
                0.0,
            );
        }

        let conformal_factor = 4.0 * self.ads_radius * self.ads_radius / (one_minus_r2 * one_minus_r2);
        
        if !conformal_factor.is_finite() {
            return Err(PoincareError::NumericalOverflow);
        }

        MetricTensor::new(
            conformal_factor,
            conformal_factor * r_safe * r_safe,
            0.0,
        )
    }

    /// Convert between half-plane and disk coordinates
    pub fn half_plane_to_disk(&self, z: f64, x: f64) -> Result<(f64, f64), PoincareError> {
        let z_safe = z.max(self.z_min_cutoff);
        
        // Standard map: w = (i - ζ)/(i + ζ) where ζ = x + iz
        let denom = x * x + (z_safe + self.ads_radius) * (z_safe + self.ads_radius);
        
        if denom < self.z_min_cutoff {
            return Err(PoincareError::NumericalOverflow);
        }

        let r_squared = (x * x + (z_safe - self.ads_radius) * (z_safe - self.ads_radius)) / denom;
        let r = r_squared.sqrt();
        
        let theta = ((-2.0 * x * self.ads_radius) / denom).atan2(
            (z_safe * z_safe + x * x - self.ads_radius * self.ads_radius) / denom
        );

        Ok((r, theta))
    }

    /// Compute geodesic distance between two points in half-plane
    /// d = L * arccosh(1 + |x₁-x₂|²/(2z₁z₂))
    pub fn geodesic_distance(&self, z1: f64, x1: f64, z2: f64, x2: f64) -> Result<f64, PoincareError> {
        let z1_safe = z1.max(self.z_min_cutoff);
        let z2_safe = z2.max(self.z_min_cutoff);
        
        let dx = x1 - x2;
        let dz = z1_safe - z2_safe;
        
        let chord_squared = dx * dx + dz * dz;
        let product = 2.0 * z1_safe * z2_safe;
        
        if product < self.z_min_cutoff {
            return Err(PoincareError::NumericalOverflow);
        }

        let arg = 1.0 + chord_squared / product;
        
        // arccosh(x) = ln(x + sqrt(x²-1))
        if arg < 1.0 {
            return Err(PoincareError::NumericalOverflow);
        }
        
        let distance = if arg > 1e150 {
            // Asymptotic: arccosh(x) ≈ ln(2x) for large x
            self.ads_radius * (2.0 * arg).ln()
        } else {
            self.ads_radius * (arg + (arg * arg - 1.0).sqrt()).ln()
        };

        if !distance.is_finite() {
            return Err(PoincareError::NumericalOverflow);
        }

        Ok(distance)
    }

    /// Compute proper volume element sqrt(-g) d²x
    pub fn volume_element(&self, z: f64, _x: f64) -> Result<f64, PoincareError> {
        let z_safe = z.max(self.z_min_cutoff);
        
        let l_over_z = self.ads_radius / z_safe;
        let sqrt_g = l_over_z * l_over_z; // sqrt(det g) for diagonal metric
        
        if !sqrt_g.is_finite() {
            return Err(PoincareError::NumericalOverflow);
        }

        Ok(sqrt_g)
    }

    /// Christoffel symbols Γ^μ_νρ for geodesic equation
    /// For Poincaré half-plane: non-zero are Γ^z_zz = -1/z, Γ^z_xx = 1/z, Γ^x_zx = -1/z
    pub fn christoffel_symbols(&self, z: f64) -> Result<ChristoffelSymbols, PoincareError> {
        let z_safe = z.max(self.z_min_cutoff);
        
        let inv_z = 1.0 / z_safe;
        
        if !inv_z.is_finite() {
            return Err(PoincareError::NumericalOverflow);
        }

        Ok(ChristoffelSymbols {
            gamma_z_zz: -inv_z,
            gamma_z_xx: inv_z,
            gamma_x_zx: -inv_z,
            gamma_x_xz: -inv_z,
        })
    }
}

/// Christoffel symbols for geodesic equations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ChristoffelSymbols {
    pub gamma_z_zz: f64,
    pub gamma_z_xx: f64,
    pub gamma_x_zx: f64,
    pub gamma_x_xz: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_creation() {
        let metric = PoincareMetric::new(1.0, 0.01, 10.0);
        assert!(metric.is_ok());
        assert!(PoincareMetric::new(-1.0, 0.01, 10.0).is_err());
    }

    #[test]
    fn test_half_plane_metric() {
        let metric = PoincareMetric::new(1.0, 0.01, 10.0).unwrap();
        let g = metric.half_plane_metric(0.5, 0.0);
        assert!(g.is_ok());
        let g_val = g.unwrap();
        // At z=0.5, L=1: g_zz = g_xx = 4
        assert!((g_val.g_zz - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_boundary_regularization() {
        let metric = PoincareMetric::new(1.0, 0.01, 10.0).unwrap();
        // Very small z should be regularized to cutoff
        let g = metric.half_plane_metric(1e-10, 0.0);
        assert!(g.is_ok());
    }

    #[test]
    fn test_geodesic_distance() {
        let metric = PoincareMetric::new(1.0, 0.01, 10.0).unwrap();
        let d = metric.geodesic_distance(1.0, 0.0, 1.0, 1.0);
        assert!(d.is_ok());
        assert!(d.unwrap() > 0.0);
    }
}
