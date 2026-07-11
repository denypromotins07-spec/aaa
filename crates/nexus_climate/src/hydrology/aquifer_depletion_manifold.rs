//! Aquifer Depletion Manifold using GRACE-FO Gravity Anomaly Data
//! Estimates aquifer volume from satellite gravity measurements

use alloc::vec::Vec;
use core::fmt;

/// Error types for aquifer manifold operations
#[derive(Debug, Clone, PartialEq)]
pub enum AquiferManifoldError {
    InvalidGravityData,
    CoordinateError,
    CalibrationError,
}

impl fmt::Display for AquiferManifoldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGravityData => write!(f, "Invalid gravity anomaly data"),
            Self::CoordinateError => write!(f, "Geographic coordinate error"),
            Self::CalibrationError => write!(f, "Calibration error"),
        }
    }
}

/// GRACE-FO gravity anomaly observation
#[derive(Debug, Clone)]
pub struct GraceObservation {
    /// Timestamp in microseconds
    pub timestamp_us: u64,
    /// Latitude
    pub latitude: f64,
    /// Longitude
    pub longitude: f64,
    /// Gravity anomaly (mGal)
    pub gravity_anomaly_mgal: f64,
    /// Uncertainty (mGal)
    pub uncertainty_mgal: f64,
    /// Quality flag
    pub quality_flag: u8,
}

/// Aquifer region definition
#[derive(Debug, Clone)]
pub struct AquiferRegion {
    pub name: &'static str,
    pub center_lat: f64,
    pub center_lon: f64,
    pub radius_km: f64,
    pub area_km2: f64,
    /// Initial water storage estimate (km³)
    pub initial_storage_km3: f64,
}

/// Known major aquifers
pub const OGALLALA_AQUIFER: AquiferRegion = AquiferRegion {
    name: "Ogallala",
    center_lat: 40.5,
    center_lon: -101.5,
    radius_km: 300.0,
    area_km2: 450_000.0,
    initial_storage_km3: 3700.0,
};

pub const CENTRAL_VALLEY_AQUIFER: AquiferRegion = AquiferRegion {
    name: "Central Valley",
    center_lat: 36.5,
    center_lon: -120.0,
    radius_km: 200.0,
    area_km2: 130_000.0,
    initial_storage_km3: 850.0,
};

/// Convert gravity anomaly to equivalent water height change
fn gravity_to_water_height(gravity_mgal: f64, area_km2: f64) -> f64 {
    // Simplified conversion: ~0.3 mm water per mGal over large areas
    let conversion_factor = 0.3; // mm/mGal
    gravity_mgal * conversion_factor * area_km2 / 1e6 // Convert to km³
}

/// Aquifer Depletion Manifold state
pub struct AquiferDepletionManifold {
    /// Target aquifer region
    region: AquiferRegion,
    /// Current water storage estimate (km³)
    current_storage_km3: f64,
    /// Historical storage values
    storage_history: Vec<(u64, f64)>,
    /// Baseline gravity (mGal)
    baseline_gravity: f64,
}

impl AquiferDepletionManifold {
    /// Create new manifold for specific aquifer
    pub fn new(region: AquiferRegion, initial_storage_km3: f64) -> Self {
        Self {
            region,
            current_storage_km3: initial_storage_km3,
            storage_history: Vec::new(),
            baseline_gravity: 0.0,
        }
    }

    /// Set baseline gravity from reference period
    pub fn set_baseline(&mut self, baseline_gravity: f64) {
        self.baseline_gravity = baseline_gravity;
    }

    /// Ingest GRACE-FO observation and update storage estimate
    pub fn ingest_observation(&mut self, obs: &GraceObservation) -> Result<(), AquiferManifoldError> {
        // Validate coordinates
        if obs.latitude.abs() > 90.0 || obs.longitude.abs() > 180.0 {
            return Err(AquiferManifoldError::CoordinateError);
        }

        // Check if observation is within aquifer region
        let distance = Self::haversine_distance(
            obs.latitude,
            obs.longitude,
            self.region.center_lat,
            self.region.center_lon,
        );

        if distance > self.region.radius_km {
            // Outside region, ignore
            return Ok(());
        }

        // Check quality
        if obs.quality_flag < 3 {
            return Ok(());
        }

        // Calculate gravity anomaly from baseline
        let delta_gravity = obs.gravity_anomaly_mgal - self.baseline_gravity;

        // Convert to water storage change
        let delta_storage = gravity_to_water_height(delta_gravity, self.region.area_km2);

        // Update current estimate
        self.current_storage_km3 = self.region.initial_storage_km3 + delta_storage;

        // Record in history
        self.storage_history.push((obs.timestamp_us, self.current_storage_km3));

        Ok(())
    }

    /// Batch ingest multiple observations
    pub fn ingest_batch(&mut self, observations: &[GraceObservation]) -> Result<usize, AquiferManifoldError> {
        let mut count = 0;
        for obs in observations {
            if self.ingest_observation(obs).is_ok() {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Get current depletion fraction (0 = full, 1 = empty)
    pub fn depletion_fraction(&self) -> f64 {
        let remaining = self.current_storage_km3 / self.region.initial_storage_km3;
        (1.0 - remaining).clamp(0.0, 1.0)
    }

    /// Get annual depletion rate (km³/year)
    pub fn annual_depletion_rate(&self) -> f64 {
        if self.storage_history.len() < 2 {
            return 0.0;
        }

        let first = self.storage_history.first().unwrap();
        let last = self.storage_history.last().unwrap();

        let dt_years = ((last.0 - first.0) as f64) / (365.0 * 24.0 * 3600.0 * 1_000_000.0);
        if dt_years <= 0.0 {
            return 0.0;
        }

        let delta_storage = last.1 - first.1;
        delta_storage / dt_years
    }

    /// Estimate years until critical depletion
    pub fn years_to_critical(&self, critical_fraction: f64) -> f64 {
        let rate = self.annual_depletion_rate();
        if rate >= 0.0 {
            return f64::MAX; // Not depleting or gaining
        }

        let critical_storage = self.region.initial_storage_km3 * (1.0 - critical_fraction);
        let remaining_until_critical = self.current_storage_km3 - critical_storage;

        if remaining_until_critical <= 0.0 {
            return 0.0;
        }

        remaining_until_critical / rate.abs()
    }

    /// Haversine distance between two points (km)
    fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        let r = 6371.0; // Earth radius in km
        let dlat = (lat2 - lat1).to_radians();
        let dlon = (lon2 - lon1).to_radians();

        let a = (dlat / 2.0).sin().powi(2)
            + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);

        2.0 * r * a.sqrt().asin()
    }

    /// Get current storage estimate
    pub fn current_storage(&self) -> f64 {
        self.current_storage_km3
    }

    /// Get all historical data
    pub fn history(&self) -> &[(u64, f64)] {
        &self.storage_history
    }
}

/// Water stress classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaterStressLevel {
    Low,
    Medium,
    High,
    Critical,
    Exhausted,
}

impl AquiferDepletionManifold {
    /// Get current water stress level
    pub fn stress_level(&self) -> WaterStressLevel {
        let depletion = self.depletion_fraction();
        
        match depletion {
            d if d < 0.2 => WaterStressLevel::Low,
            d if d < 0.4 => WaterStressLevel::Medium,
            d if d < 0.6 => WaterStressLevel::High,
            d if d < 0.8 => WaterStressLevel::Critical,
            _ => WaterStressLevel::Exhausted,
        }
    }

    /// Generate alpha signal for water futures trading
    pub fn generate_alpha_signal(&self) -> Option<WaterAlphaSignal> {
        let stress = self.stress_level();
        let depletion_rate = self.annual_depletion_rate();
        let years_remaining = self.years_to_critical(0.8);

        // Only generate signal if significant stress or rapid depletion
        if matches!(stress, WaterStressLevel::Low | WaterStressLevel::Medium) && depletion_rate.abs() < 1.0 {
            return None;
        }

        let signal_strength = match stress {
            WaterStressLevel::Low => 0.0,
            WaterStressLevel::Medium => 0.3,
            WaterStressLevel::High => 0.6,
            WaterStressLevel::Critical => 0.8,
            WaterStressLevel::Exhausted => 1.0,
        };

        // Adjust based on depletion acceleration
        let acceleration_factor = if depletion_rate < -5.0 {
            1.5
        } else if depletion_rate < -2.0 {
            1.2
        } else {
            1.0
        };

        Some(WaterAlphaSignal {
            aquifer_name: self.region.name,
            stress_level: stress,
            signal_strength: (signal_strength * acceleration_factor).clamp(0.0, 1.0),
            recommended_position: if depletion_rate < 0.0 {
                Position::LongWaterFutures
            } else {
                Position::ShortWaterFutures
            },
            confidence: (1.0 - years_remaining / 50.0).clamp(0.3, 0.95),
            years_to_crisis: years_remaining,
        })
    }
}

/// Water trading signal
#[derive(Debug, Clone)]
pub struct WaterAlphaSignal {
    pub aquifer_name: &'static str,
    pub stress_level: WaterStressLevel,
    pub signal_strength: f64,
    pub recommended_position: Position,
    pub confidence: f64,
    pub years_to_crisis: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    LongWaterFutures,
    ShortWaterFutures,
    Neutral,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aquifer_manifold() {
        let mut manifold = AquiferDepletionManifold::new(OGALLALA_AQUIFER, OGALLALA_AQUIFER.initial_storage_km3);
        
        // Add some observations
        let obs = GraceObservation {
            timestamp_us: 1_000_000_000_000,
            latitude: 40.5,
            longitude: -101.5,
            gravity_anomaly_mgal: -0.5,
            uncertainty_mgal: 0.1,
            quality_flag: 5,
        };

        manifold.set_baseline(0.0);
        let result = manifold.ingest_observation(&obs);
        assert!(result.is_ok());

        assert_eq!(manifold.history().len(), 1);
    }

    #[test]
    fn test_stress_levels() {
        let mut manifold = AquiferDepletionManifold::new(OGALLALA_AQUIFER, 1000.0);
        manifold.current_storage_km3 = 200.0; // 80% depleted
        
        assert_eq!(manifold.stress_level(), WaterStressLevel::Critical);
    }
}
