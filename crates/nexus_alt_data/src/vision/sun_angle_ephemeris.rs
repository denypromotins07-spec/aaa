//! Sun Angle Ephemeris Calculator
//! 
//! Calculates solar position (elevation and azimuth) for any location
//! and time, essential for shadow-based volume measurements.

use std::time::SystemTime;
use thiserror::Error;

/// Solar position calculation errors
#[derive(Debug, Error)]
pub enum SunAngleError {
    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(String),
    #[error("Invalid coordinates: {0}")]
    InvalidCoordinates(String),
    #[error("Calculation error: {0}")]
    CalculationError(String),
}

/// Solar position data
#[derive(Debug, Clone)]
pub struct SolarPosition {
    pub elevation_deg: f64,
    pub azimuth_deg: f64,
    pub declination_deg: f64,
    pub hour_angle_deg: f64,
    pub timestamp: SystemTime,
    pub latitude: f64,
    pub longitude: f64,
}

impl SolarPosition {
    pub fn new(
        elevation_deg: f64,
        azimuth_deg: f64,
        declination_deg: f64,
        hour_angle_deg: f64,
        timestamp: SystemTime,
        latitude: f64,
        longitude: f64,
    ) -> Result<Self, SunAngleError> {
        if latitude < -90.0 || latitude > 90.0 {
            return Err(SunAngleError::InvalidCoordinates(
                "Latitude must be between -90 and 90".to_string(),
            ));
        }

        Ok(SolarPosition {
            elevation_deg,
            azimuth_deg,
            declination_deg,
            hour_angle_deg,
            timestamp,
            latitude,
            longitude,
        })
    }
}

/// Sun angle ephemeris calculator using NOAA algorithms
pub struct SunAngleEphemeris;

impl SunAngleEphemeris {
    /// Calculate solar position for given location and time
    pub fn calculate_solar_position(
        latitude: f64,
        longitude: f64,
        timestamp: SystemTime,
    ) -> Result<SolarPosition, SunAngleError> {
        // Get Julian Date
        let jd = Self::system_time_to_julian(timestamp)?;
        
        // Calculate Julian Century
        let jc = (jd - 2451545.0) / 36525.0;
        
        // Calculate geometric mean longitude of sun (degrees)
        let l0 = (280.46646 + jc * (36000.76983 + 0.0003032 * jc)) % 360.0;
        let l0 = if l0 < 0.0 { l0 + 360.0 } else { l0 };
        
        // Calculate geometric mean anomaly of sun (degrees)
        let m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc);
        let m = m % 360.0;
        
        // Calculate eccentricity of earth orbit
        let e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc);
        
        // Calculate sun equation of center
        let c = (1.914602 - jc * (0.004817 + 0.000014 * jc)) * (m.to_radians()).sin()
              + (0.019993 - 0.000101 * jc) * (2.0 * m.to_radians()).sin()
              + 0.000289 * (3.0 * m.to_radians()).sin();
        
        // Calculate sun true longitude
        let sun_true_long = l0 + c;
        
        // Calculate sun apparent longitude
        let omega = 125.04 - 1934.136 * jc;
        let lambda = sun_true_long - 0.00569 - 0.00478 * (omega.to_radians()).sin();
        
        // Calculate obliquity of ecliptic
        let epsilon0 = 23.0 + (26.0 + ((21.448 - jc * (46.8150 + jc * (0.00059 - jc * 0.001813)))) / 60.0) / 60.0;
        let epsilon = epsilon0 + 0.00256 * (omega.to_radians()).cos();
        
        // Calculate sun declination
        let declination = ((epsilon.to_radians()).sin() * (lambda.to_radians()).sin()).asin().to_degrees();
        
        // Calculate equation of time (minutes)
        let y = (epsilon.to_radians() / 2.0).tan().powi(2);
        let eq_time = 4.0 * (
            y * (2.0 * l0.to_radians()).sin()
            - 2.0 * e * (m.to_radians()).sin()
            + 4.0 * e * y * (m.to_radians()).sin() * (2.0 * l0.to_radians()).cos()
            - 0.5 * y * y * (4.0 * l0.to_radians()).sin()
            - 1.25 * e * e * (2.0 * m.to_radians()).sin()
        );
        
        // Calculate hour angle
        let ha = Self::calculate_hour_angle(latitude, timestamp, eq_time, longitude)?;
        
        // Calculate solar zenith angle
        let cos_zenith = (latitude.to_radians()).sin() * (declination.to_radians()).sin()
                       + (latitude.to_radians()).cos() * (declination.to_radians()).cos() * (ha.to_radians()).cos();
        
        let zenith = cos_zenith.acos().to_degrees();
        let elevation = 90.0 - zenith;
        
        // Calculate solar azimuth
        let mut azimuth = if (latitude.to_radians()).cos() * (zenith.to_radians()).sin() == 0.0 {
            0.0
        } else {
            let cos_az = ((declination.to_radians()).sin() - (latitude.to_radians()).sin() * (zenith.to_radians()).cos())
                / ((latitude.to_radians()).cos() * (zenith.to_radians()).sin());
            
            let az_rad = cos_az.acos();
            if ha.to_degrees() > 0.0 {
                az_rad.to_degrees()
            } else {
                360.0 - az_rad.to_degrees()
            }
        };
        
        // Normalize azimuth to [0, 360)
        azimuth = azimuth % 360.0;
        if azimuth < 0.0 {
            azimuth += 360.0;
        }

        SolarPosition::new(
            elevation,
            azimuth,
            declination,
            ha,
            timestamp,
            latitude,
            longitude,
        )
    }

    /// Calculate hour angle
    fn calculate_hour_angle(
        latitude: f64,
        timestamp: SystemTime,
        eq_time: f64,
        longitude: f64,
    ) -> Result<f64, SunAngleError> {
        // Get UTC time
        let duration = timestamp
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| SunAngleError::InvalidTimestamp(e.to_string()))?;
        
        let utc_hours = (duration.as_secs() % 86400) as f64 / 3600.0
                      + duration.subsec_millis() as f64 / 3600000.0;
        
        // Calculate local solar time
        let lst = utc_hours + longitude / 15.0 + eq_time / 60.0;
        
        // Calculate hour angle (degrees)
        let ha = 15.0 * (lst - 12.0);
        
        Ok(ha)
    }

    /// Convert SystemTime to Julian Date
    fn system_time_to_julian(timestamp: SystemTime) -> Result<f64, SunAngleError> {
        let duration = timestamp
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_err(|e| SunAngleError::InvalidTimestamp(e.to_string()))?;
        
        let jd = 2440587.5 + duration.as_secs_f64() / 86400.0;
        Ok(jd)
    }

    /// Check if sun is above horizon
    pub fn is_sun_above_horizon(elevation_deg: f64) -> bool {
        elevation_deg > -0.833 // Account for atmospheric refraction
    }

    /// Get optimal imaging windows (when shadows are measurable)
    pub fn get_optimal_imaging_windows(
        latitude: f64,
        longitude: f64,
        date: SystemTime,
    ) -> Result<Vec<ImagingWindow>, SunAngleError> {
        let mut windows = Vec::new();
        
        // Check every 15 minutes for 24 hours
        let mut current_time = date;
        let step = std::time::Duration::from_secs(900); // 15 minutes
        
        let mut in_window = false;
        let mut window_start = current_time;
        
        for _ in 0..96 {
            let position = Self::calculate_solar_position(latitude, longitude, current_time)?;
            
            // Optimal shadow measurement: elevation between 20 and 60 degrees
            let is_optimal = position.elevation_deg >= 20.0 && position.elevation_deg <= 60.0;
            
            if is_optimal && !in_window {
                window_start = current_time;
                in_window = true;
            } else if !is_optimal && in_window {
                windows.push(ImagingWindow {
                    start: window_start,
                    end: current_time,
                    peak_elevation_time: window_start, // Would calculate actual peak
                });
                in_window = false;
            }
            
            current_time = current_time + step;
        }
        
        // Close final window if still open
        if in_window {
            windows.push(ImagingWindow {
                start: window_start,
                end: current_time,
                peak_elevation_time: window_start,
            });
        }
        
        Ok(windows)
    }
}

/// Optimal imaging window for shadow measurements
#[derive(Debug, Clone)]
pub struct ImagingWindow {
    pub start: SystemTime,
    pub end: SystemTime,
    pub peak_elevation_time: SystemTime,
}

impl ImagingWindow {
    pub fn duration_seconds(&self) -> u64 {
        self.end.duration_since(self.start)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solar_position_calculation() {
        let lat = 35.9848; // Cushing, OK
        let lon = -97.3942;
        let time = SystemTime::now();
        
        let position = SunAngleEphemeris::calculate_solar_position(lat, lon, time);
        
        // Just verify it doesn't error and produces reasonable values
        assert!(position.is_ok());
        let pos = position.unwrap();
        assert!(pos.elevation_deg >= -90.0 && pos.elevation_deg <= 90.0);
        assert!(pos.azimuth_deg >= 0.0 && pos.azimuth_deg <= 360.0);
    }

    #[test]
    fn test_invalid_coordinates() {
        let result = SunAngleEphemeris::calculate_solar_position(100.0, 0.0, SystemTime::now());
        assert!(result.is_err());
    }
}
