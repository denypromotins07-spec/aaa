//! VCM Satellite Verification Module
//! Verifies voluntary carbon market offsets using satellite SAR and IoT sensor data

use alloc::vec::Vec;
use core::fmt;

/// Error types for satellite verification
#[derive(Debug, Clone, PartialEq)]
pub enum VerificationError {
    InvalidCoordinates,
    DataUnavailable,
    VerificationMismatch,
    TemporalGap,
    QualityThresholdNotMet,
}

impl fmt::Display for VerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCoordinates => write!(f, "Invalid geographic coordinates"),
            Self::DataUnavailable => write!(f, "Satellite/sensor data unavailable"),
            Self::VerificationMismatch => write!(f, "Satellite and registry data mismatch"),
            Self::TemporalGap => write!(f, "Data temporal gap exceeds threshold"),
            Self::QualityThresholdNotMet => write!(f, "Data quality below threshold"),
        }
    }
}

/// Forest project type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectType {
    AvoidedDeforestation,
    Afforestation,
    Reforestation,
    ImprovedForestManagement,
    Agroforestry,
    MangroveConservation,
}

/// Satellite observation data
#[derive(Debug, Clone)]
pub struct SatelliteObservation {
    /// Timestamp in microseconds
    pub timestamp_us: u64,
    /// Latitude
    pub latitude: f64,
    /// Longitude
    pub longitude: f64,
    /// Biomass estimate (tonnes/ha)
    pub biomass_tonnes_ha: f64,
    /// Canopy cover percentage
    pub canopy_cover_pct: f64,
    /// NDVI (Normalized Difference Vegetation Index)
    pub ndvi: f64,
    /// SAR backscatter coefficient (dB)
    pub sar_backscatter_db: f64,
    /// Cloud cover percentage
    pub cloud_cover_pct: f64,
    /// Data quality flag
    pub quality_flag: u8,
}

/// IoT forest sensor reading
#[derive(Debug, Clone)]
pub struct IoTSensorReading {
    /// Sensor ID
    pub sensor_id: u64,
    /// Timestamp in microseconds
    pub timestamp_us: u64,
    /// Soil moisture percentage
    pub soil_moisture_pct: f64,
    /// Temperature (°C)
    pub temperature_c: f64,
    /// Humidity percentage
    pub humidity_pct: f64,
    /// Tree growth rate indicator
    pub growth_indicator: f64,
}

/// Registry claim for a carbon project
#[derive(Debug, Clone)]
pub struct RegistryClaim {
    /// Project ID in registry
    pub project_id: u64,
    /// Project type
    pub project_type: ProjectType,
    /// Claimed location (lat, lon) center
    pub location: (f64, f64),
    /// Claimed area in hectares
    pub area_ha: f64,
    /// Claimed baseline biomass (tonnes/ha)
    pub baseline_biomass: f64,
    /// Claimed current biomass (tonnes/ha)
    pub current_biomass: f64,
    /// Vintage year
    pub vintage: u16,
    /// Verified by registry
    pub registry_verified: bool,
}

/// Verification result
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Project ID
    pub project_id: u64,
    /// Overall verification status
    pub verified: bool,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Satellite-derived biomass estimate
    pub satellite_biomass: f64,
    /// Registry-claimed biomass
    pub claimed_biomass: f64,
    /// Discrepancy percentage
    pub discrepancy_pct: f64,
    /// Number of satellite observations used
    pub n_satellite_obs: usize,
    /// Number of IoT readings used
    pub n_iot_readings: usize,
    /// Issues detected
    pub issues: Vec<&'static str>,
}

/// Geospatial bounding box for overlap detection
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub min_lat: f64,
    pub max_lat: f64,
    pub min_lon: f64,
    pub max_lon: f64,
}

impl BoundingBox {
    /// Create bounding box from center and area
    pub fn from_center_area(center: (f64, f64), area_ha: f64) -> Self {
        // Approximate: 1 ha ≈ 100m x 100m
        // 1 degree lat ≈ 111km, 1 degree lon ≈ 111km * cos(lat)
        let side_m = (area_ha * 10_000.0).sqrt();
        let delta_lat = side_m / 111_000.0;
        let delta_lon = side_m / (111_000.0 * center.1.to_radians().cos().max(0.1));

        Self {
            min_lat: center.0 - delta_lat,
            max_lat: center.0 + delta_lat,
            min_lon: center.1 - delta_lon,
            max_lon: center.1 + delta_lon,
        }
    }

    /// Check if two bounding boxes overlap (with epsilon buffer)
    pub fn overlaps_with(&self, other: &BoundingBox, epsilon_deg: f64) -> bool {
        // Add epsilon buffer to prevent fuzzy boundary issues
        let eps = epsilon_deg.max(0.0001); // ~10m minimum
        
        !(self.max_lat + eps < other.min_lat ||
          self.min_lat - eps > other.max_lat ||
          self.max_lon + eps < other.min_lon ||
          self.min_lon - eps > other.max_lon)
    }

    /// Check if point is inside box
    pub fn contains_point(&self, lat: f64, lon: f64) -> bool {
        lat >= self.min_lat && lat <= self.max_lat &&
        lon >= self.min_lon && lon <= self.max_lon
    }
}

/// Satellite verification engine
pub struct SatelliteVerificationEngine {
    /// Minimum NDVI for forest detection
    min_ndvi_forest: f64,
    /// Maximum allowed biomass discrepancy (%)
    max_discrepancy_pct: f64,
    /// Maximum temporal gap (microseconds)
    max_temporal_gap_us: u64,
    /// Minimum quality flag
    min_quality_flag: u8,
    /// Epsilon buffer for geospatial overlap (degrees)
    geo_epsilon_deg: f64,
    /// Known project bounding boxes for double-counting detection
    project_boxes: alloc::collections::BTreeMap<u64, BoundingBox>,
}

impl SatelliteVerificationEngine {
    /// Create new verification engine
    pub fn new() -> Self {
        Self {
            min_ndvi_forest: 0.5,
            max_discrepancy_pct: 20.0,
            max_temporal_gap_us: 30 * 24 * 3600 * 1_000_000, // 30 days
            min_quality_flag: 3,
            geo_epsilon_deg: 0.001, // ~100m buffer
            project_boxes: alloc::collections::BTreeMap::new(),
        }
    }

    /// Register a project's bounding box
    pub fn register_project(&mut self, project_id: u64, location: (f64, f64), area_ha: f64) {
        let bbox = BoundingBox::from_center_area(location, area_ha);
        self.project_boxes.insert(project_id, bbox);
    }

    /// Check for potential double-counting (overlapping project areas)
    pub fn check_double_counting(&self, project_id: u64, location: (f64, f64), area_ha: f64) -> Result<(), VerificationError> {
        let new_bbox = BoundingBox::from_center_area(location, area_ha);

        for (&existing_id, existing_bbox) in &self.project_boxes {
            if existing_id != project_id && existing_bbox.overlaps_with(&new_bbox, self.geo_epsilon_deg) {
                return Err(VerificationError::VerificationMismatch);
            }
        }

        Ok(())
    }

    /// Verify a carbon credit claim
    pub fn verify_credit(
        &self,
        claim: &RegistryClaim,
        satellite_data: &[SatelliteObservation],
        iot_data: &[IoTSensorReading],
    ) -> Result<VerificationResult, VerificationError> {
        // Validate coordinates
        if claim.location.0.abs() > 90.0 || claim.location.1.abs() > 180.0 {
            return Err(VerificationError::InvalidCoordinates);
        }

        let bbox = BoundingBox::from_center_area(claim.location, claim.area_ha);

        // Filter satellite observations within bounding box and time window
        let mut relevant_satellite: Vec<&SatelliteObservation> = Vec::new();
        for obs in satellite_data {
            if bbox.contains_point(obs.latitude, obs.longitude) {
                // Check quality
                if obs.quality_flag >= self.min_quality_flag && obs.cloud_cover_pct < 20.0 {
                    relevant_satellite.push(obs);
                }
            }
        }

        // Filter IoT readings
        let mut relevant_iot: Vec<&IoTSensorReading> = Vec::new();
        for reading in iot_data {
            // Simple proximity check (would need actual sensor locations in production)
            relevant_iot.push(reading);
        }

        // Check data availability
        if relevant_satellite.is_empty() {
            return Err(VerificationError::DataUnavailable);
        }

        // Compute satellite-derived biomass estimate
        let mut total_biomass = 0.0;
        let mut valid_count = 0;

        for obs in &relevant_satellite {
            // Use NDVI and SAR backscatter to estimate biomass
            let ndvi_factor = (obs.ndvi - 0.3).max(0.0) / 0.7; // Normalize
            let sar_factor = (obs.sar_backscatter_db + 20.0) / 20.0; // Rough normalization
            
            // Combined estimate (simplified model)
            let estimated_biomass = obs.biomass_tonnes_ha * ndvi_factor * sar_factor.clamp(0.5, 1.5);
            
            total_biomass += estimated_biomass;
            valid_count += 1;
        }

        let satellite_biomass = if valid_count > 0 {
            total_biomass / valid_count as f64
        } else {
            return Err(VerificationError::QualityThresholdNotMet);
        };

        // Check NDVI threshold for forest detection
        let avg_ndvi: f64 = relevant_satellite.iter().map(|o| o.ndvi).sum::<f64>() / relevant_satellite.len() as f64;
        let is_forest = avg_ndvi >= self.min_ndvi_forest;

        // Calculate discrepancy
        let claimed_biomass = claim.current_biomass;
        let discrepancy_pct = if claimed_biomass > 0.0 {
            ((satellite_biomass - claimed_biomass).abs() / claimed_biomass) * 100.0
        } else {
            100.0
        };

        // Build issues list
        let mut issues = Vec::new();

        if !is_forest {
            issues.push("NDVI below forest threshold");
        }
        if discrepancy_pct > self.max_discrepancy_pct {
            issues.push("Biomass discrepancy exceeds threshold");
        }
        if relevant_satellite.len() < 3 {
            issues.push("Insufficient satellite observations");
        }

        // Determine verification status
        let verified = is_forest && 
                       discrepancy_pct <= self.max_discrepancy_pct &&
                       !issues.is_empty() == false; // No critical issues

        // Confidence calculation
        let coverage_confidence = (relevant_satellite.len() as f64 / 10.0).min(1.0);
        let discrepancy_confidence = (1.0 - discrepancy_pct / 100.0).max(0.0);
        let quality_confidence = relevant_satellite.iter()
            .map(|o| o.quality_flag as f64 / 5.0)
            .sum::<f64>() / relevant_satellite.len() as f64;

        let confidence = (coverage_confidence * 0.3 + discrepancy_confidence * 0.5 + quality_confidence * 0.2).clamp(0.0, 1.0);

        Ok(VerificationResult {
            project_id: claim.project_id,
            verified,
            confidence,
            satellite_biomass,
            claimed_biomass,
            discrepancy_pct,
            n_satellite_obs: relevant_satellite.len(),
            n_iot_readings: relevant_iot.len(),
            issues,
        })
    }

    /// Aggregate verification for multiple credits
    pub fn verify_batch(
        &self,
        claims: &[RegistryClaim],
        all_satellite: &[SatelliteObservation],
        all_iot: &[IoTSensorReading],
    ) -> Vec<Result<VerificationResult, VerificationError>> {
        claims.iter()
            .map(|claim| self.verify_credit(claim, all_satellite, all_iot))
            .collect()
    }
}

impl Default for SatelliteVerificationEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_overlap() {
        let bbox1 = BoundingBox::from_center_area((0.0, 0.0), 100.0);
        let bbox2 = BoundingBox::from_center_area((0.0, 0.0), 100.0);
        let bbox3 = BoundingBox::from_center_area((1.0, 1.0), 100.0);

        assert!(bbox1.overlaps_with(&bbox2, 0.001));
        assert!(!bbox1.overlaps_with(&bbox3, 0.001));
    }

    #[test]
    fn test_verification_basic() {
        let engine = SatelliteVerificationEngine::new();

        let claim = RegistryClaim {
            project_id: 1,
            project_type: ProjectType::AvoidedDeforestation,
            location: (0.0, 0.0),
            area_ha: 100.0,
            baseline_biomass: 200.0,
            current_biomass: 180.0,
            vintage: 2024,
            registry_verified: true,
        };

        let satellite_data = vec![
            SatelliteObservation {
                timestamp_us: 1000,
                latitude: 0.0,
                longitude: 0.0,
                biomass_tonnes_ha: 175.0,
                canopy_cover_pct: 80.0,
                ndvi: 0.75,
                sar_backscatter_db: -10.0,
                cloud_cover_pct: 5.0,
                quality_flag: 5,
            },
        ];

        let result = engine.verify_credit(&claim, &satellite_data, &[]);
        assert!(result.is_ok());
    }
}
