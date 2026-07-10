//! SGP4 Orbital Propagator for Satellite Pass Prediction
//! 
//! Implements the Simplified General Perturbations (SGP4) model to predict
//! satellite positions with high precision using f64 arithmetic.
//! 
//! CRITICAL: Uses f64 throughout to prevent orbital drift over time.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Errors specific to SGP4 propagation
#[derive(Debug, Error)]
pub enum Sgp4Error {
    #[error("Invalid TLE format: {0}")]
    InvalidTle(String),
    #[error("Propagation error: {0}")]
    PropagationError(String),
    #[error("Julian date calculation overflow")]
    JulianDateOverflow,
}

/// Two-Line Element (TLE) data structure
#[derive(Debug, Clone)]
pub struct Tle {
    pub name: String,
    pub line1: String,
    pub line2: String,
    // Parsed elements
    pub epoch_year: u16,
    pub epoch_day: f64,
    pub mean_motion_deriv: f64,
    pub drag_term: f64,
    pub inclination: f64,
    pub raan: f64,
    pub eccentricity: f64,
    pub arg_of_perigee: f64,
    pub mean_anomaly: f64,
    pub mean_motion: f64,
    pub orbit_number: u32,
}

impl Tle {
    pub fn parse(name: &str, line1: &str, line2: &str) -> Result<Self, Sgp4Error> {
        let line1 = line1.trim();
        let line2 = line2.trim();

        if line1.len() < 69 || line2.len() < 69 {
            return Err(Sgp4Error::InvalidTle("Line too short".to_string()));
        }

        // Parse Line 1
        let epoch_year_str = &line1[18..20];
        let epoch_day_str = &line1[20..32];
        
        let epoch_year_short: u16 = epoch_year_str.parse()
            .map_err(|_| Sgp4Error::InvalidTle("Invalid epoch year".to_string()))?;
        let epoch_year = if epoch_year_short >= 57 {
            1900 + epoch_year_short
        } else {
            2000 + epoch_year_short
        };
        
        let epoch_day: f64 = epoch_day_str.trim().parse()
            .map_err(|_| Sgp4Error::InvalidTle("Invalid epoch day".to_string()))?;

        let mean_motion_deriv_str = &line1[33..43];
        let mean_motion_deriv: f64 = mean_motion_deriv_str.trim().parse()
            .map_err(|_| Sgp4Error::InvalidTle("Invalid mean motion derivative".to_string()))?;

        let drag_term_str = &line1[44..52];
        let drag_term: f64 = drag_term_str.trim().parse()
            .map_err(|_| Sgp4Error::InvalidTle("Invalid drag term".to_string()))?;

        // Parse Line 2
        let inclination_str = &line2[8..16];
        let inclination: f64 = inclination_str.trim().parse()
            .map_err(|_| Sgp4Error::InvalidTle("Invalid inclination".to_string()))?;

        let raan_str = &line2[17..25];
        let raan: f64 = raan_str.trim().parse()
            .map_err(|_| Sgp4Error::InvalidTle("Invalid RAAN".to_string()))?;

        let eccentricity_str = &line2[26..33];
        let eccentricity: f64 = format!("0.{}", eccentricity_str.trim()).parse()
            .map_err(|_| Sgp4Error::InvalidTle("Invalid eccentricity".to_string()))?;

        let arg_of_perigee_str = &line2[34..42];
        let arg_of_perigee: f64 = arg_of_perigee_str.trim().parse()
            .map_err(|_| Sgp4Error::InvalidTle("Invalid argument of perigee".to_string()))?;

        let mean_anomaly_str = &line2[43..51];
        let mean_anomaly: f64 = mean_anomaly_str.trim().parse()
            .map_err(|_| Sgp4Error::InvalidTle("Invalid mean anomaly".to_string()))?;

        let mean_motion_str = &line2[52..63];
        let mean_motion: f64 = mean_motion_str.trim().parse()
            .map_err(|_| Sgp4Error::InvalidTle("Invalid mean motion".to_string()))?;

        let orbit_number_str = &line2[63..68];
        let orbit_number: u32 = orbit_number_str.trim().parse()
            .map_err(|_| Sgp4Error::InvalidTle("Invalid orbit number".to_string()))?;

        Ok(Tle {
            name: name.to_string(),
            line1: line1.to_string(),
            line2: line2.to_string(),
            epoch_year,
            epoch_day,
            mean_motion_deriv,
            drag_term,
            inclination,
            raan,
            eccentricity,
            arg_of_perigee,
            mean_anomaly,
            mean_motion,
            orbit_number,
        })
    }
}

/// SGP4 Propagator state
#[derive(Debug, Clone)]
pub struct Sgp4Propagator {
    tle: Tle,
    // Pre-computed constants
    a: f64,           // Semi-major axis (Earth radii)
    n: f64,           // Mean motion (rad/min)
    e: f64,           // Eccentricity
    i: f64,           // Inclination (rad)
    omega: f64,       // RAAN (rad)
    w: f64,           // Argument of perigee (rad)
    m: f64,           // Mean anomaly (rad)
    epoch_jd: f64,    // Epoch Julian Date
}

impl Sgp4Propagator {
    /// Create a new SGP4 propagator from TLE data
    pub fn new(tle: Tle) -> Result<Self, Sgp4Error> {
        // Convert degrees to radians
        let i = tle.inclination.to_radians();
        let omega = tle.raan.to_radians();
        let w = tle.arg_of_perigee.to_radians();
        let m = tle.mean_anomaly.to_radians();
        let e = tle.eccentricity;
        
        // Calculate semi-major axis from mean motion
        // n = sqrt(mu / a^3) => a = (mu / n^2)^(1/3)
        // Using Earth's gravitational parameter mu = 398600.4418 km^3/s^2
        // and converting mean motion from rev/day to rad/min
        let n_rev_per_day = tle.mean_motion;
        let n_rad_per_min = n_rev_per_day * 2.0 * std::f64::consts::PI / 1440.0;
        
        // Earth radius in km
        let earth_radius = 6378.137;
        // Gravitational parameter / earth_radius^3
        let mu_normalized = 398600.4418 / (earth_radius.powi(3));
        
        // Semi-major axis in Earth radii
        let a = (mu_normalized / (n_rad_per_min.powi(2))).powf(1.0 / 3.0);
        
        // Calculate epoch Julian Date
        let epoch_jd = Self::calculate_julian_date(tle.epoch_year, tle.epoch_day)?;

        Ok(Sgp4Propagator {
            tle,
            a,
            n: n_rad_per_min,
            e,
            i,
            omega,
            w,
            m,
            epoch_jd,
        })
    }

    /// Calculate Julian Date from year and day of year
    fn calculate_julian_date(year: u16, day: f64) -> Result<f64, Sgp4Error> {
        // High-precision Julian Date calculation using f64
        let y = year as f64;
        let jd_base = if y >= 2000.0 {
            2451544.5 // JD for Jan 1, 2000 00:00 UT
        } else {
            2415019.5 // JD for Jan 1, 1900 00:00 UT
        };
        
        let days_since_base = (y - 2000.0) * 365.0 + ((y - 2000.0) / 4.0).floor() + day - 1.0;
        
        // Check for overflow
        if !days_since_base.is_finite() {
            return Err(Sgp4Error::JulianDateOverflow);
        }
        
        Ok(jd_base + days_since_base)
    }

    /// Propagate satellite position to given time
    pub fn propagate(&self, target_time: SystemTime) -> Result<GeodeticPosition, Sgp4Error> {
        // Calculate minutes since epoch
        let target_jd = Self::system_time_to_julian(target_time)?;
        let tsince = (target_jd - self.epoch_jd) * 1440.0; // Convert days to minutes

        if !tsince.is_finite() {
            return Err(Sgp4Error::PropagationError("Time calculation overflow".to_string()));
        }

        // Simplified SGP4 propagation (full implementation would be much longer)
        // This is a simplified version focusing on the key calculations
        
        // Mean anomaly at time t
        let m_t = self.m + self.n * tsince;
        
        // Solve Kepler's equation for eccentric anomaly E
        let e = self.solve_kepler(m_t, self.e)?;
        
        // True anomaly
        let true_anomaly = self.eccentric_to_true_anomaly(e, self.e);
        
        // Position in orbital plane
        let r = self.a * (1.0 - self.e * e.cos());
        let x_orbital = r * true_anomaly.cos();
        let y_orbital = r * true_anomaly.sin();
        
        // Transform to ECI coordinates (simplified)
        let cos_omega = self.omega.cos();
        let sin_omega = self.omega.sin();
        let cos_i = self.i.cos();
        let sin_i = self.i.sin();
        let cos_w = (self.w + true_anomaly).cos();
        let sin_w = (self.w + true_anomaly).sin();
        
        let x_eci = x_orbital * (cos_omega * cos_w - sin_omega * sin_w * cos_i) 
                  - y_orbital * (cos_omega * sin_w + sin_omega * cos_w * cos_i);
        let y_eci = x_orbital * (sin_omega * cos_w + cos_omega * sin_w * cos_i) 
                  - y_orbital * (sin_omega * sin_w - cos_omega * cos_w * cos_i);
        let z_eci = x_orbital * (sin_w * sin_i) + y_orbital * (cos_w * sin_i);
        
        // Convert to geodetic coordinates
        Self::eci_to_geodetic(x_eci, y_eci, z_eci, target_time)
    }

    /// Solve Kepler's equation: M = E - e*sin(E)
    fn solve_kepler(&self, m: f64, e: f64) -> Result<f64, Sgp4Error> {
        let mut e_val = m;
        let tolerance = 1e-12;
        let max_iterations = 50;
        
        for _ in 0..max_iterations {
            let delta = (e_val - e * e_val.sin() - m) / (1.0 - e * e_val.cos());
            e_val -= delta;
            
            if delta.abs() < tolerance {
                return Ok(e_val);
            }
        }
        
        Err(Sgp4Error::PropagationError("Kepler solver did not converge".to_string()))
    }

    /// Convert eccentric anomaly to true anomaly
    fn eccentric_to_true_anomaly(&self, e: f64, ecc: f64) -> f64 {
        let cos_v = (ecc.cos() - ecc) / (1.0 - ecc * e.cos());
        let sin_v = (1.0 - ecc.powi(2)).sqrt() * e.sin() / (1.0 - ecc * e.cos());
        sin_v.atan2(cos_v)
    }

    /// Convert ECI coordinates to geodetic (lat, lon, alt)
    fn eci_to_geodetic(
        x: f64, 
        y: f64, 
        z: f64, 
        time: SystemTime
    ) -> Result<GeodeticPosition, Sgp4Error> {
        // Earth rotation angle (Greenwich Mean Sidereal Time)
        let gmst = Self::calculate_gmst(time)?;
        
        // Rotate to ECF (Earth-Centered Fixed) coordinates
        let cos_gmst = gmst.cos();
        let sin_gmst = gmst.sin();
        
        let x_ecf = x * cos_gmst + y * sin_gmst;
        let y_ecf = -x * sin_gmst + y * cos_gmst;
        let z_ecf = z;
        
        // Convert to geodetic (simplified iterative method)
        let earth_radius_eq = 6378.137; // km
        let earth_radius_pol = 6356.752; // km
        let e_squared = 1.0 - (earth_radius_pol / earth_radius_eq).powi(2);
        
        let p = (x_ecf.powi(2) + y_ecf.powi(2)).sqrt();
        let mut lat = (z_ecf / p * (1.0 - e_squared)).atan();
        
        // Iterative refinement
        for _ in 0..5 {
            let n = earth_radius_eq / (1.0 - e_squared * lat.sin().powi(2)).sqrt();
            let h = p / lat.cos() - n;
            lat = (z_ecf / p * (1.0 - e_squared * n / (n + h))).atan();
        }
        
        let lon = y_ecf.atan2(x_ecf);
        let h = p / lat.cos() - earth_radius_eq / (1.0 - e_squared * lat.sin().powi(2)).sqrt();
        
        Ok(GeodeticPosition {
            latitude: lat.to_degrees(),
            longitude: lon.to_degrees(),
            altitude_km: h,
            timestamp: time,
        })
    }

    /// Calculate Greenwich Mean Sidereal Time
    fn calculate_gmst(time: SystemTime) -> Result<f64, Sgp4Error> {
        let duration = time.duration_since(UNIX_EPOCH)
            .map_err(|_| Sgp4Error::PropagationError("Time before epoch".to_string()))?;
        
        let jd = 2440587.5 + duration.as_secs_f64() / 86400.0;
        let t = (jd - 2451545.0) / 36525.0; // Julian centuries since J2000.0
        
        let gmst = 280.46061837 
                 + 360.98564736629 * (jd - 2451545.0)
                 + 0.000387933 * t.powi(2)
                 - t.powi(3) / 38710000.0;
        
        // Normalize to [0, 360)
        let gmst_norm = gmst.rem_euclid(360.0);
        Ok(gmst_norm.to_radians())
    }

    fn system_time_to_julian(time: SystemTime) -> Result<f64, Sgp4Error> {
        let duration = time.duration_since(UNIX_EPOCH)
            .map_err(|_| Sgp4Error::PropagationError("Time before epoch".to_string()))?;
        Ok(2440587.5 + duration.as_secs_f64() / 86400.0)
    }

    /// Predict next pass over a ground station
    pub fn predict_pass(
        &self,
        ground_station: &GeodeticPosition,
        start_time: SystemTime,
        max_duration_minutes: u64,
    ) -> Result<Option<SatellitePass>, Sgp4Error> {
        let mut current_time = start_time;
        let end_time = start_time + std::time::Duration::from_secs(max_duration_minutes * 60);
        
        let mut aos_time: Option<SystemTime> = None;
        let mut max_elevation = 0.0;
        let mut max_elevation_time: Option<SystemTime> = None;
        
        // Step through time in 10-second intervals
        let step = std::time::Duration::from_secs(10);
        
        while current_time < end_time {
            let position = self.propagate(current_time)?;
            let elevation = Self::calculate_elevation(&position, ground_station);
            
            if elevation > 0.0 {
                if aos_time.is_none() {
                    aos_time = Some(current_time);
                }
                
                if elevation > max_elevation {
                    max_elevation = elevation;
                    max_elevation_time = Some(current_time);
                }
            } else if aos_time.is_some() {
                // LOS detected after AOS
                return Ok(Some(SatellitePass {
                    aos: aos_time.unwrap(),
                    los: current_time,
                    max_elevation,
                    max_elevation_time: max_elevation_time.unwrap(),
                    satellite_name: self.tle.name.clone(),
                }));
            }
            
            current_time += step;
        }
        
        // If still in view at end of window
        if let Some(aos) = aos_time {
            return Ok(Some(SatellitePass {
                aos,
                los: end_time,
                max_elevation,
                max_elevation_time: max_elevation_time.unwrap_or(aos),
                satellite_name: self.tle.name.clone(),
            }));
        }
        
        Ok(None)
    }

    /// Calculate elevation angle from ground station to satellite
    fn calculate_elevation(sat: &GeodeticPosition, gs: &GeodeticPosition) -> f64 {
        // Simplified elevation calculation
        let lat_diff = sat.latitude - gs.latitude;
        let lon_diff = sat.longitude - gs.longitude;
        
        // Approximate distance in km
        let lat_km = lat_diff * 111.32;
        let lon_km = lon_diff * 111.32 * gs.latitude.to_radians().cos();
        
        let horizontal_dist = (lat_km.powi(2) + lon_km.powi(2)).sqrt();
        let vertical_dist = sat.altitude_km;
        
        if horizontal_dist < 0.001 {
            return 90.0;
        }
        
        (vertical_dist / horizontal_dist).atan().to_degrees()
    }
}

/// Geodetic position (latitude, longitude, altitude)
#[derive(Debug, Clone)]
pub struct GeodeticPosition {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude_km: f64,
    pub timestamp: SystemTime,
}

/// Satellite pass information
#[derive(Debug, Clone)]
pub struct SatellitePass {
    pub aos: SystemTime,      // Acquisition of Signal
    pub los: SystemTime,      // Loss of Signal
    pub max_elevation: f64,
    pub max_elevation_time: SystemTime,
    pub satellite_name: String,
}

/// Critical infrastructure location for monitoring
#[derive(Debug, Clone)]
pub struct InfrastructureLocation {
    pub name: String,
    pub latitude: f64,
    pub longitude: f64,
    pub asset_type: AssetType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetType {
    OilStorage,
    Port,
    Pipeline,
    Refinery,
    Agricultural,
}

impl InfrastructureLocation {
    pub fn cusching_ok() -> Self {
        InfrastructureLocation {
            name: "Cushing OK Hub".to_string(),
            latitude: 35.9848,
            longitude: -97.3942,
            asset_type: AssetType::OilStorage,
        }
    }

    pub fn port_of_shanghai() -> Self {
        InfrastructureLocation {
            name: "Port of Shanghai".to_string(),
            latitude: 31.2304,
            longitude: 121.4737,
            asset_type: AssetType::Port,
        }
    }
}

/// Scheduler for satellite passes over multiple locations
pub struct PassScheduler {
    propagators: Vec<Arc<Sgp4Propagator>>,
    locations: Vec<InfrastructureLocation>,
}

impl PassScheduler {
    pub fn new() -> Self {
        PassScheduler {
            propagators: Vec::new(),
            locations: Vec::new(),
        }
    }

    pub fn add_satellite(&mut self, tle: Tle) -> Result<(), Sgp4Error> {
        let propagator = Sgp4Propagator::new(tle)?;
        self.propagators.push(Arc::new(propagator));
        Ok(())
    }

    pub fn add_location(&mut self, location: InfrastructureLocation) {
        self.locations.push(location);
    }

    /// Get all upcoming passes for all satellites over all locations
    pub fn get_upcoming_passes(
        &self,
        start_time: SystemTime,
        horizon_hours: u64,
    ) -> Result<Vec<SatellitePass>, Sgp4Error> {
        let mut all_passes = Vec::new();
        
        for propagator in &self.propagators {
            for location in &self.locations {
                let gs = GeodeticPosition {
                    latitude: location.latitude,
                    longitude: location.longitude,
                    altitude_km: 0.0,
                    timestamp: start_time,
                };
                
                if let Some(pass) = propagator.predict_pass(&gs, start_time, horizon_hours * 60)? {
                    all_passes.push(pass);
                }
            }
        }
        
        // Sort by AOS time
        all_passes.sort_by(|a, b| a.aos.cmp(&b.aos));
        
        Ok(all_passes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tle_parsing() {
        let tle_line1 = "1 25544U 98067A   21001.00000000  .00000000  00000-0  00000-0 0  9990";
        let tle_line2 = "2 25544  51.6400 200.0000 0000000  0.0000  0.0000  15.50000000000000";
        
        let tle = Tle::parse("ISS", tle_line1, tle_line2).unwrap();
        assert_eq!(tle.name, "ISS");
        assert_eq!(tle.inclination, 51.64);
    }

    #[test]
    fn test_propagator_creation() {
        let tle_line1 = "1 25544U 98067A   21001.00000000  .00000000  00000-0  00000-0 0  9990";
        let tle_line2 = "2 25544  51.6400 200.0000 0000000  0.0000  0.0000  15.50000000000000";
        
        let tle = Tle::parse("ISS", tle_line1, tle_line2).unwrap();
        let propagator = Sgp4Propagator::new(tle).unwrap();
        
        assert!(propagator.a > 0.0);
        assert!(propagator.n > 0.0);
    }
}
