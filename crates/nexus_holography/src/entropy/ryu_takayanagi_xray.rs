//! Ryu-Takayanagi X-Ray Engine
//! 
//! Implements the RT formula: S_A = Area(γ_A) / (4G_N)
//! Uses boundary entropy to trace minimal surfaces into bulk for dark pool detection.

use crate::geometry::{AdsCftDictionary, PoincareMetric};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors specific to RT X-Ray calculations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum RyuTakayanagiError {
    #[error("Invalid Newton constant G_N: {0}")]
    InvalidNewtonConstant(f64),
    #[error("Region A has zero or negative measure")]
    InvalidRegionMeasure,
    #[error("Minimal surface computation failed: {0}")]
    MinimalSurfaceFailed(String),
    #[error("UV divergence detected despite cutoff")]
    UVDivergence,
    #[error("Entropy calculation overflow")]
    EntropyOverflow,
}

/// Boundary region A for entanglement entropy calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryRegion {
    /// Start position on boundary
    pub x_start: f64,
    /// End position on boundary
    pub x_end: f64,
    /// Associated time window (for dynamic regions)
    pub time_window_ns: u64,
}

impl BoundaryRegion {
    /// Create a new boundary region with validation
    pub fn new(x_start: f64, x_end: f64, time_window_ns: u64) -> Result<Self, RyuTakayanagiError> {
        if (x_end - x_start).abs() < 1e-15 {
            return Err(RyuTakayanagiError::InvalidRegionMeasure);
        }
        Ok(Self {
            x_start,
            x_end,
            time_window_ns,
        })
    }

    /// Get the length of the region
    pub fn length(&self) -> f64 {
        (self.x_end - self.x_start).abs()
    }
}

/// Minimal surface data extending into bulk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimalSurface {
    /// The boundary region this surface is anchored to
    pub region: BoundaryRegion,
    /// Maximum depth reached in bulk (z_max)
    pub max_depth: f64,
    /// Computed area of the minimal surface
    pub area: f64,
    /// Number of discretization points used
    pub num_points: usize,
}

/// Ryu-Takayanagi X-Ray engine for dark pool detection
pub struct RyuTakayanagiXRay {
    /// AdS/CFT dictionary
    dictionary: AdsCftDictionary,
    /// Poincaré metric calculator
    metric: PoincareMetric,
    /// Effective Newton constant in bulk (calibrated to market liquidity)
    newton_constant: f64,
    /// UV cutoff for regularization (must match metric cutoff)
    uv_cutoff: f64,
}

impl RyuTakayanagiXRay {
    /// Create a new RT X-Ray engine
    pub fn new(
        dictionary: AdsCftDictionary,
        metric: PoincareMetric,
        newton_constant: f64,
    ) -> Result<Self, RyuTakayanagiError> {
        if newton_constant <= 0.0 {
            return Err(RyuTakayanagiError::InvalidNewtonConstant(newton_constant));
        }

        let uv_cutoff = dictionary.uv_cutoff.min(metric.z_min_cutoff);

        Ok(Self {
            dictionary,
            metric,
            newton_constant,
            uv_cutoff,
        })
    }

    /// Compute entanglement entropy via RT formula
    /// S_A = Area(γ_A) / (4G_N)
    pub fn compute_entropy(&self, region: &BoundaryRegion) -> Result<f64, RyuTakayanagiError> {
        let surface = self.trace_minimal_surface(region)?;
        
        // RT formula: S = Area / (4 G_N)
        let entropy = surface.area / (4.0 * self.newton_constant);
        
        if !entropy.is_finite() {
            return Err(RyuTakayanagiError::EntropyOverflow);
        }

        Ok(entropy)
    }

    /// Trace the minimal surface γ_A anchored to boundary region A
    /// For 1+1D boundary, the minimal surface is a geodesic curve in 2+1D AdS
    /// In Poincaré coordinates: z(x) = sqrt((L/2)² - (x - x_c)²) for interval of length L
    pub fn trace_minimal_surface(&self, region: &BoundaryRegion) -> Result<MinimalSurface, RyuTakayanagiError> {
        let l = region.length();
        
        if l <= 0.0 {
            return Err(RyuTakayanagiError::InvalidRegionMeasure);
        }

        // Center of the interval
        let x_c = (region.x_start + region.x_end) / 2.0;
        
        // For a strip of width l in AdS_3, the minimal surface is a semi-circle
        // z(x) = sqrt((l/2)² - (x - x_c)²)
        // Maximum depth z_max = l/2
        let z_max = l / 2.0;

        // Discretize the surface for numerical integration
        let num_points = ((l / self.uv_cutoff) as usize).min(10000).max(100);
        let dx = l / num_points as f64;

        let mut total_length = 0.0;
        let mut prev_point: Option<(f64, f64)> = None;

        for i in 0..=num_points {
            let x = region.x_start + i as f64 * dx;
            
            // z(x) = sqrt(z_max² - (x - x_c)²)
            let arg = z_max * z_max - (x - x_c) * (x - x_c);
            
            // Apply UV cutoff: don't go below z_min
            let z = if arg > 0.0 {
                arg.sqrt().max(self.uv_cutoff)
            } else {
                self.uv_cutoff
            };

            if let Some((prev_x, prev_z)) = prev_point {
                // Compute proper length element using metric
                let ds = self.metric.geodesic_distance(prev_z, prev_x, z, x)
                    .map_err(|e| RyuTakayanagiError::MinimalSurfaceFailed(e.to_string()))?;
                total_length += ds;
            }

            prev_point = Some((x, z));
        }

        // For AdS_3, the "area" of the minimal surface is its length
        let area = total_length;

        if !area.is_finite() {
            return Err(RyuTakayanagiError::UVDivergence);
        }

        Ok(MinimalSurface {
            region: region.clone(),
            max_depth: z_max,
            area,
            num_points,
        })
    }

    /// Analytical result for entanglement entropy of an interval in CFT_2
    /// S = (c/3) ln(L/ε) where c is central charge, L is interval length, ε is UV cutoff
    pub fn analytical_entropy(&self, region: &BoundaryRegion, central_charge: f64) -> Result<f64, RyuTakayanagiError> {
        let l = region.length();
        
        if central_charge <= 0.0 {
            return Err(RyuTakayanagiError::InvalidNewtonConstant(central_charge));
        }

        // Regulated entropy
        let regulated_length = l.max(self.uv_cutoff);
        let entropy = (central_charge / 3.0) * (regulated_length / self.uv_cutoff).ln();

        if !entropy.is_finite() {
            return Err(RyuTakayanagiError::EntropyOverflow);
        }

        Ok(entropy)
    }

    /// Detect dark pool presence from entropy excess
    /// If observed entropy > expected from lit tape, hidden liquidity exists
    pub fn detect_dark_pool(
        &self,
        region: &BoundaryRegion,
        observed_entropy: f64,
    ) -> Result<DarkPoolSignal, RyuTakayanagiError> {
        let expected_entropy = self.compute_entropy(region)?;
        
        let entropy_excess = observed_entropy - expected_entropy;
        
        // Threshold for detection (10% of expected)
        let threshold = expected_entropy.abs() * 0.1;
        
        let dark_pool_detected = entropy_excess > threshold;
        
        // Estimate hidden volume from entropy excess
        // S ~ ln(Volume), so Volume ~ exp(S)
        let estimated_hidden_volume = if dark_pool_detected {
            (entropy_excess / expected_entropy.abs()).exp() * region.length()
        } else {
            0.0
        };

        Ok(DarkPoolSignal {
            expected_entropy,
            observed_entropy,
            entropy_excess,
            dark_pool_detected,
            estimated_hidden_volume,
            confidence: if dark_pool_detected {
                (entropy_excess / threshold).min(1.0)
            } else {
                0.0
            },
        })
    }
}

/// Signal indicating dark pool detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DarkPoolSignal {
    /// Expected entropy from visible tape
    pub expected_entropy: f64,
    /// Actually observed entropy
    pub observed_entropy: f64,
    /// Excess entropy suggesting hidden activity
    pub entropy_excess: f64,
    /// Whether dark pool was detected
    pub dark_pool_detected: bool,
    /// Estimated hidden volume
    pub estimated_hidden_volume: f64,
    /// Confidence score [0, 1]
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{AdsCftDictionary, PoincareMetric};

    #[test]
    fn test_xray_creation() {
        let dict = AdsCftDictionary::new(1.0, 0.01, 10.0).unwrap();
        let metric = PoincareMetric::new(1.0, 0.01, 10.0).unwrap();
        let xray = RyuTakayanagiXRay::new(dict, metric, 0.1);
        assert!(xray.is_ok());
    }

    #[test]
    fn test_entropy_computation() {
        let dict = AdsCftDictionary::new(1.0, 0.01, 10.0).unwrap();
        let metric = PoincareMetric::new(1.0, 0.01, 10.0).unwrap();
        let xray = RyuTakayanagiXRay::new(dict, metric, 0.1).unwrap();
        
        let region = BoundaryRegion::new(0.0, 1.0, 1000).unwrap();
        let entropy = xray.compute_entropy(&region);
        assert!(entropy.is_ok());
        assert!(entropy.unwrap() > 0.0);
    }

    #[test]
    fn test_analytical_entropy() {
        let dict = AdsCftDictionary::new(1.0, 0.01, 10.0).unwrap();
        let metric = PoincareMetric::new(1.0, 0.01, 10.0).unwrap();
        let xray = RyuTakayanagiXRay::new(dict, metric, 0.1).unwrap();
        
        let region = BoundaryRegion::new(0.0, 1.0, 1000).unwrap();
        let entropy = xray.analytical_entropy(&region, 1.0);
        assert!(entropy.is_ok());
    }
}
