//! Shadow Volume Calculator for Crude Oil Tank Level Estimation
//! 
//! Uses sun angle ephemeris and shadow pixel measurements to calculate
//! the exact height of floating roofs in crude oil storage tanks.

use crate::vision::simd_edge_detector::TankDetection;
use crate::satellite::sgp4_orbital_propagator::GeodeticPosition;
use std::time::SystemTime;

/// Errors in shadow volume calculation
#[derive(Debug, thiserror::Error)]
pub enum ShadowVolumeError {
    #[error("Invalid shadow measurement: {0}")]
    InvalidShadowMeasurement(String),
    #[error("Sun angle calculation failed: {0}")]
    SunAngleError(String),
    #[error("Tank geometry invalid: {0}")]
    InvalidTankGeometry(String),
}

/// Shadow measurement from SAR imagery
#[derive(Debug, Clone)]
pub struct ShadowMeasurement {
    pub tank_id: String,
    pub shadow_length_pixels: f64,
    pub shadow_width_pixels: f64,
    pub tank_diameter_pixels: f64,
    pub image_resolution_meters_per_pixel: f64,
}

/// Calculated tank fill level
#[derive(Debug, Clone)]
pub struct TankFillLevel {
    pub tank_id: String,
    pub roof_height_meters: f64,
    pub tank_capacity_m3: f64,
    pub current_volume_m3: f64,
    pub fill_percentage: f64,
    pub confidence: f64,
    pub timestamp: SystemTime,
}

impl TankFillLevel {
    pub fn new(
        tank_id: String,
        roof_height_meters: f64,
        tank_capacity_m3: f64,
        current_volume_m3: f64,
        confidence: f64,
        timestamp: SystemTime,
    ) -> Result<Self, ShadowVolumeError> {
        if tank_capacity_m3 <= 0.0 {
            return Err(ShadowVolumeError::InvalidTankGeometry(
                "Tank capacity must be positive".to_string(),
            ));
        }

        let fill_percentage = (current_volume_m3 / tank_capacity_m3 * 100.0).clamp(0.0, 100.0);

        Ok(TankFillLevel {
            tank_id,
            roof_height_meters,
            tank_capacity_m3,
            current_volume_m3,
            fill_percentage,
            confidence,
            timestamp,
        })
    }
}

/// Shadow volume calculator using sun angle geometry
pub struct ShadowVolumeCalculator {
    /// Standard tank heights for reference (meters)
    standard_tank_heights: Vec<f64>,
    /// Minimum valid shadow length (pixels)
    min_shadow_pixels: f64,
    /// Maximum valid shadow length (pixels)
    max_shadow_pixels: f64,
}

impl ShadowVolumeCalculator {
    pub fn new() -> Self {
        ShadowVolumeCalculator {
            // Common API 650 tank heights
            standard_tank_heights: vec![12.0, 15.0, 18.0, 21.0, 24.0],
            min_shadow_pixels: 2.0,
            max_shadow_pixels: 500.0,
        }
    }

    /// Calculate tank fill level from shadow measurement
    pub fn calculate_fill_level(
        &self,
        measurement: &ShadowMeasurement,
        sun_elevation_deg: f64,
        tank_detection: &TankDetection,
        timestamp: SystemTime,
    ) -> Result<TankFillLevel, ShadowVolumeError> {
        // Validate shadow measurement
        if measurement.shadow_length_pixels < self.min_shadow_pixels
            || measurement.shadow_length_pixels > self.max_shadow_pixels
        {
            return Err(ShadowVolumeError::InvalidShadowMeasurement(
                format!(
                    "Shadow length {} pixels out of valid range [{}, {}]",
                    measurement.shadow_length_pixels,
                    self.min_shadow_pixels,
                    self.max_shadow_pixels
                ),
            ));
        }

        // Validate sun elevation
        if sun_elevation_deg <= 0.0 || sun_elevation_deg >= 90.0 {
            return Err(ShadowVolumeError::SunAngleError(
                "Sun elevation must be between 0 and 90 degrees".to_string(),
            ));
        }

        // Convert shadow length from pixels to meters
        let shadow_length_meters =
            measurement.shadow_length_pixels * measurement.image_resolution_meters_per_pixel;

        // Calculate roof height using trigonometry
        // tan(sun_elevation) = roof_height / shadow_length
        let sun_elevation_rad = sun_elevation_deg.to_radians();
        let roof_height_meters = shadow_length_meters * sun_elevation_rad.tan();

        // Get tank diameter in meters
        let tank_diameter_meters =
            measurement.tank_diameter_pixels * measurement.image_resolution_meters_per_pixel;
        let tank_radius_meters = tank_diameter_meters / 2.0;

        // Calculate tank capacity (cylindrical approximation)
        let max_tank_height = self.estimate_max_tank_height(tank_diameter_meters);
        let tank_capacity_m3 = std::f64::consts::PI * tank_radius_meters.powi(2) * max_tank_height;

        // Calculate current volume
        let current_volume_m3 = std::f64::consts::PI * tank_radius_meters.powi(2) * roof_height_meters;

        // Calculate confidence based on measurement quality
        let confidence = self.calculate_confidence(
            measurement,
            sun_elevation_deg,
            roof_height_meters,
            max_tank_height,
        );

        TankFillLevel::new(
            measurement.tank_id.clone(),
            roof_height_meters,
            tank_capacity_m3,
            current_volume_m3,
            confidence,
            timestamp,
        )
    }

    /// Estimate maximum tank height based on diameter (API 650 standard)
    fn estimate_max_tank_height(&self, diameter_meters: f64) -> f64 {
        // Simplified estimation - real implementation would use detailed tank database
        if diameter_meters < 30.0 {
            12.0
        } else if diameter_meters < 50.0 {
            15.0
        } else if diameter_meters < 70.0 {
            18.0
        } else if diameter_meters < 90.0 {
            21.0
        } else {
            24.0
        }
    }

    /// Calculate confidence score for the measurement
    fn calculate_confidence(
        &self,
        measurement: &ShadowMeasurement,
        sun_elevation: f64,
        roof_height: f64,
        max_height: f64,
    ) -> f64 {
        let mut confidence = 1.0;

        // Reduce confidence for very low sun angles (long shadows, more error)
        if sun_elevation < 20.0 {
            confidence *= 0.7;
        } else if sun_elevation < 40.0 {
            confidence *= 0.85;
        }

        // Reduce confidence for very high sun angles (short shadows, hard to measure)
        if sun_elevation > 70.0 {
            confidence *= 0.8;
        }

        // Reduce confidence if shadow is very short or very long
        let shadow_ratio = measurement.shadow_length_pixels / measurement.tank_diameter_pixels;
        if shadow_ratio < 0.1 || shadow_ratio > 2.0 {
            confidence *= 0.7;
        }

        // Reduce confidence if calculated height seems unreasonable
        if roof_height < 0.0 || roof_height > max_height * 1.1 {
            confidence *= 0.5;
        }

        confidence.clamp(0.0, 1.0)
    }

    /// Batch calculate fill levels for multiple tanks
    pub fn batch_calculate(
        &self,
        measurements: &[ShadowMeasurement],
        sun_elevation_deg: f64,
        tank_detections: &[TankDetection],
        timestamp: SystemTime,
    ) -> Result<Vec<TankFillLevel>, ShadowVolumeError> {
        let mut results = Vec::with_capacity(measurements.len());

        for (i, measurement) in measurements.iter().enumerate() {
            let tank_detection = tank_detections
                .get(i)
                .ok_or_else(|| {
                    ShadowVolumeError::InvalidTankGeometry(
                        format!("No tank detection for measurement {}", measurement.tank_id),
                    )
                })?;

            let fill_level = self.calculate_fill_level(
                measurement,
                sun_elevation_deg,
                tank_detection,
                timestamp,
            )?;

            results.push(fill_level);
        }

        Ok(results)
    }

    /// Aggregate total volume for a facility
    pub fn aggregate_facility_volume(
        &self,
        fill_levels: &[TankFillLevel],
    ) -> FacilityVolumeSummary {
        let total_capacity: f64 = fill_levels.iter().map(|t| t.tank_capacity_m3).sum();
        let total_volume: f64 = fill_levels.iter().map(|t| t.current_volume_m3).sum();
        let avg_fill_percentage = if !fill_levels.is_empty() {
            fill_levels.iter().map(|t| t.fill_percentage).sum::<f64>() / fill_levels.len() as f64
        } else {
            0.0
        };

        let weighted_confidence = if !fill_levels.is_empty() {
            let sum_weights: f64 = fill_levels.iter().map(|t| t.confidence).sum();
            sum_weights / fill_levels.len() as f64
        } else {
            0.0
        };

        FacilityVolumeSummary {
            facility_id: "AGGREGATED".to_string(),
            tank_count: fill_levels.len(),
            total_capacity_m3: total_capacity,
            total_volume_m3: total_volume,
            avg_fill_percentage,
            weighted_confidence,
            timestamp: SystemTime::now(),
        }
    }
}

impl Default for ShadowVolumeCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated facility volume summary
#[derive(Debug, Clone)]
pub struct FacilityVolumeSummary {
    pub facility_id: String,
    pub tank_count: usize,
    pub total_capacity_m3: f64,
    pub total_volume_m3: f64,
    pub avg_fill_percentage: f64,
    pub weighted_confidence: f64,
    pub timestamp: SystemTime,
}

/// Historical volume tracking for trend analysis
pub struct VolumeTrendAnalyzer {
    history: Vec<(SystemTime, f64)>, // (timestamp, volume)
    max_history_size: usize,
}

impl VolumeTrendAnalyzer {
    pub fn new(max_history_size: usize) -> Self {
        VolumeTrendAnalyzer {
            history: Vec::with_capacity(max_history_size),
            max_history_size,
        }
    }

    pub fn add_measurement(&mut self, timestamp: SystemTime, volume_m3: f64) {
        self.history.push((timestamp, volume_m3));

        // Trim old entries
        while self.history.len() > self.max_history_size {
            self.history.remove(0);
        }
    }

    /// Calculate volume change rate (m3 per day)
    pub fn calculate_change_rate(&self) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }

        let first = self.history.first().unwrap();
        let last = self.history.last().unwrap();

        let time_diff_days = last
            .0
            .duration_since(first.0)
            .ok()?
            .as_secs_f64()
            / 86400.0;

        if time_diff_days < 0.001 {
            return None;
        }

        let volume_diff = last.1 - first.1;
        Some(volume_diff / time_diff_days)
    }

    /// Detect unusual volume changes (potential market-moving events)
    pub fn detect_anomaly(&self, threshold_percentage: f64) -> Option<VolumeAnomaly> {
        if self.history.len() < 3 {
            return None;
        }

        let recent = self.history.last().unwrap().1;
        let previous_avg: f64 = self.history.iter().rev().skip(1).take(2).map(|(_, v)| v).sum::<f64>() / 2.0;

        if previous_avg < 0.001 {
            return None;
        }

        let change_percentage = ((recent - previous_avg) / previous_avg * 100.0).abs();

        if change_percentage > threshold_percentage {
            Some(VolumeAnomaly {
                change_percentage,
                direction: if recent > previous_avg {
                    AnomalyDirection::Increase
                } else {
                    AnomalyDirection::Decrease
                },
                magnitude: (recent - previous_avg).abs(),
                timestamp: self.history.last().unwrap().0,
            })
        } else {
            None
        }
    }
}

/// Detected volume anomaly
#[derive(Debug, Clone)]
pub struct VolumeAnomaly {
    pub change_percentage: f64,
    pub direction: AnomalyDirection,
    pub magnitude: f64,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyDirection {
    Increase,
    Decrease,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_volume_calculation() {
        let calculator = ShadowVolumeCalculator::new();
        
        let measurement = ShadowMeasurement {
            tank_id: "TANK_001".to_string(),
            shadow_length_pixels: 50.0,
            shadow_width_pixels: 100.0,
            tank_diameter_pixels: 200.0,
            image_resolution_meters_per_pixel: 0.5,
        };

        let tank_detection = TankDetection {
            center_x: 100.0,
            center_y: 100.0,
            radius: 100.0,
            confidence: 0.9,
        };

        let fill_level = calculator
            .calculate_fill_level(&measurement, 45.0, &tank_detection, SystemTime::now())
            .unwrap();

        assert!(fill_level.roof_height_meters > 0.0);
        assert!(fill_level.fill_percentage >= 0.0);
        assert!(fill_level.fill_percentage <= 100.0);
    }

    #[test]
    fn test_invalid_shadow_measurement() {
        let calculator = ShadowVolumeCalculator::new();
        
        let measurement = ShadowMeasurement {
            tank_id: "TANK_001".to_string(),
            shadow_length_pixels: 1.0, // Too small
            shadow_width_pixels: 100.0,
            tank_diameter_pixels: 200.0,
            image_resolution_meters_per_pixel: 0.5,
        };

        let tank_detection = TankDetection {
            center_x: 100.0,
            center_y: 100.0,
            radius: 100.0,
            confidence: 0.9,
        };

        let result = calculator.calculate_fill_level(&measurement, 45.0, &tank_detection, SystemTime::now());
        assert!(result.is_err());
    }
}
