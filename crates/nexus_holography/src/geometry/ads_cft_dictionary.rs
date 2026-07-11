//! AdS/CFT Dictionary: Maps boundary operators to bulk fields
//! 
//! Implements the holographic dictionary relating CFT boundary operators
//! to bulk fields in AdS space.

use nalgebra::{Matrix2, Vector2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors specific to AdS/CFT dictionary operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum AdsCftError {
    #[error("Invalid conformal dimension: {0}")]
    InvalidConformalDimension(f64),
    #[error("Bulk-boundary mapping failed: {0}")]
    MappingFailed(String),
    #[error("Operator dimension exceeds unitarity bound")]
    UnitarityViolation,
}

/// Conformal dimension of a boundary operator
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConformalDimension(pub f64);

impl ConformalDimension {
    /// Create a new conformal dimension with validation
    /// For scalar operators in 1+1D CFT, Δ >= 0 (unitarity bound)
    pub fn new(delta: f64) -> Result<Self, AdsCftError> {
        if delta < 0.0 {
            Err(AdsCftError::UnitarityViolation)
        } else {
            Ok(ConformalDimension(delta))
        }
    }

    /// Get the corresponding bulk mass via m²L² = Δ(Δ - d)
    /// where d is the boundary dimension (d=1 for 1+1D CFT)
    pub fn bulk_mass_squared(&self, ads_radius: f64) -> f64 {
        let delta = self.0;
        let d = 1.0; // 1+1D boundary
        let m2_l2 = delta * (delta - d);
        m2_l2 / (ads_radius * ads_radius)
    }
}

/// Boundary operator in the CFT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryOperator {
    /// Position on the boundary (x coordinate)
    pub position: f64,
    /// Conformal dimension
    pub dimension: ConformalDimension,
    /// Operator type (e.g., stress tensor, current, scalar)
    pub operator_type: OperatorType,
    /// Scaling amplitude
    pub amplitude: f64,
}

/// Types of boundary operators
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OperatorType {
    /// Stress-energy tensor T_μν
    StressTensor,
    /// Conserved current J_μ
    Current,
    /// Scalar operator O
    Scalar,
    /// Spinor operator ψ
    Spinor,
}

/// Bulk field dual to a boundary operator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkField {
    /// Position in bulk (z, x) where z is radial coordinate
    pub position: Vector2<f64>,
    /// Mass of the bulk field
    pub mass_squared: f64,
    /// Field value at this position
    pub field_value: f64,
    /// Decay rate into the bulk
    pub decay_rate: f64,
}

/// Holographic dictionary for AdS/CFT correspondence
pub struct AdsCftDictionary {
    /// AdS radius L
    pub ads_radius: f64,
    /// UV cutoff (minimum z value to avoid singularities)
    pub uv_cutoff: f64,
    /// IR cutoff (maximum z value)
    pub ir_cutoff: f64,
}

impl AdsCftDictionary {
    /// Create a new holographic dictionary
    pub fn new(ads_radius: f64, uv_cutoff: f64, ir_cutoff: f64) -> Result<Self, AdsCftError> {
        if ads_radius <= 0.0 {
            return Err(AdsCftError::InvalidConformalDimension(ads_radius));
        }
        if uv_cutoff <= 0.0 || ir_cutoff <= uv_cutoff {
            return Err(AdsCftError::MappingFailed(
                "Invalid cutoff values".to_string(),
            ));
        }
        Ok(Self {
            ads_radius,
            uv_cutoff,
            ir_cutoff,
        })
    }

    /// Map a boundary operator to its bulk field profile
    /// Uses the standard AdS/CFT prescription: ϕ(z,x) ~ z^Δ near boundary
    pub fn map_to_bulk(&self, operator: &BoundaryOperator, bulk_z: f64) -> Result<BulkField, AdsCftError> {
        if bulk_z < self.uv_cutoff || bulk_z > self.ir_cutoff {
            return Err(AdsCftError::MappingFailed(format!(
                "Bulk z={} outside valid range [{}, {}]",
                bulk_z, self.uv_cutoff, self.ir_cutoff
            )));
        }

        let delta = operator.dimension.0;
        let scaling = (bulk_z / self.ads_radius).powf(delta);
        
        // Bulk field profile decays as z^Δ
        let field_value = operator.amplitude * scaling;
        
        // Decay rate from bulk mass
        let mass_sq = operator.dimension.bulk_mass_squared(self.ads_radius);
        let decay_rate = if mass_sq > 0.0 {
            mass_sq.sqrt()
        } else {
            0.0
        };

        Ok(BulkField {
            position: Vector2::new(bulk_z, operator.position),
            mass_squared: mass_sq,
            field_value,
            decay_rate,
        })
    }

    /// Reconstruct boundary operator from bulk field near boundary
    pub fn reconstruct_boundary(&self, bulk_field: &BulkField) -> Result<f64, AdsCftError> {
        let z = bulk_field.position.x;
        if z < self.uv_cutoff {
            return Err(AdsCftError::MappingFailed(
                "Cannot reconstruct from below UV cutoff".to_string(),
            ));
        }

        // Inverse mapping: O(x) ~ lim_{z->0} z^{-Δ} ϕ(z,x)
        // We use the known mass to infer Δ
        let m2_l2 = bulk_field.mass_squared * self.ads_radius * self.ads_radius;
        
        // Solve Δ(Δ-1) = m²L² for Δ
        let discriminant = 1.0 + 4.0 * m2_l2;
        if discriminant < 0.0 {
            return Err(AdsCftError::MappingFailed(
                "Tachyonic bulk field (BF bound violation)".to_string(),
            ));
        }
        
        let delta_plus = (1.0 + discriminant.sqrt()) / 2.0;
        let scaling = (z / self.ads_radius).powf(-delta_plus);
        
        Ok(bulk_field.field_value * scaling)
    }

    /// Compute the two-point correlation function from bulk propagator
    /// ⟨O(x)O(y)⟩ ~ 1/|x-y|^{2Δ}
    pub fn two_point_function(&self, x1: f64, x2: f64, delta: f64) -> Result<f64, AdsCftError> {
        let separation = (x1 - x2).abs();
        if separation < self.uv_cutoff {
            // Regularize UV divergence
            return Ok((self.uv_cutoff).powf(-2.0 * delta));
        }
        Ok(separation.powf(-2.0 * delta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformal_dimension_creation() {
        assert!(ConformalDimension::new(0.5).is_ok());
        assert!(ConformalDimension::new(0.0).is_ok());
        assert!(ConformalDimension::new(-0.1).is_err());
    }

    #[test]
    fn test_dictionary_creation() {
        let dict = AdsCftDictionary::new(1.0, 0.01, 10.0);
        assert!(dict.is_ok());
        assert!(AdsCftDictionary::new(-1.0, 0.01, 10.0).is_err());
        assert!(AdsCftDictionary::new(1.0, 0.0, 10.0).is_err());
    }

    #[test]
    fn test_bulk_mapping() {
        let dict = AdsCftDictionary::new(1.0, 0.01, 10.0).unwrap();
        let op = BoundaryOperator {
            position: 0.0,
            dimension: ConformalDimension::new(1.0).unwrap(),
            operator_type: OperatorType::Scalar,
            amplitude: 1.0,
        };
        let bulk = dict.map_to_bulk(&op, 0.5);
        assert!(bulk.is_ok());
    }
}
