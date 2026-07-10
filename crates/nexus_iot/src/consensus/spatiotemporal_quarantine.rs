//! Spatiotemporal Quarantine for Suspicious IoT Sensors
//! 
//! Implements quarantine protocols for sensors detected as potentially compromised.
//! Uses gradual trust decay and geographic isolation.

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH, Duration};

#[derive(Debug, Clone)]
pub struct QuarantinedSensor {
    pub sensor_id: u64,
    pub quarantine_start_ns: u64,
    pub reason: QuarantineReason,
    pub trust_decay_factor: f64,
    pub last_verified_location: Option<(f64, f64)>,
}

#[derive(Debug, Clone)]
pub enum QuarantineReason {
    SybilClusterDetected,
    GeographicImpossibility,
    ValueAnomaly,
    TimestampManipulation,
    ReputationThresholdBreached,
}

#[derive(Debug)]
pub enum QuarantineError {
    SensorNotFound,
    AlreadyQuarantined,
    QuarantineNotExpired,
    VerificationFailed,
}

/// Spatiotemporal Quarantine Manager
pub struct SpatiotemporalQuarantine {
    quarantined_sensors: HashMap<u64, QuarantinedSensor>,
    quarantine_duration_ns: u64,
    min_trust_threshold: f64,
    geographic_fences: Vec<GeographicFence>,
}

#[derive(Debug, Clone)]
pub struct GeographicFence {
    pub center_lat: f64,
    pub center_lon: f64,
    pub radius_km: f64,
    pub allowed_sensor_ids: HashSet<u64>,
}

impl SpatiotemporalQuarantine {
    pub fn new(quarantine_duration_ns: u64, min_trust_threshold: f64) -> Self {
        Self {
            quarantined_sensors: HashMap::new(),
            quarantine_duration_ns,
            min_trust_threshold,
            geographic_fences: Vec::new(),
        }
    }

    /// Add a geographic fence for restricted areas
    pub fn add_geographic_fence(&mut self, fence: GeographicFence) {
        self.geographic_fences.push(fence);
    }

    /// Quarantine a sensor with specified reason
    pub fn quarantine_sensor(
        &mut self,
        sensor_id: u64,
        reason: QuarantineReason,
        current_time_ns: u64,
    ) -> Result<(), QuarantineError> {
        if self.quarantined_sensors.contains_key(&sensor_id) {
            return Err(QuarantineError::AlreadyQuarantined);
        }

        let quarantined = QuarantinedSensor {
            sensor_id,
            quarantine_start_ns: current_time_ns,
            reason,
            trust_decay_factor: 0.5, // Immediate 50% trust reduction
            last_verified_location: None,
        };

        self.quarantined_sensors.insert(sensor_id, quarantined);
        Ok(())
    }

    /// Check if a sensor is currently quarantined
    pub fn is_quarantined(&self, sensor_id: u64, current_time_ns: u64) -> bool {
        match self.quarantined_sensors.get(&sensor_id) {
            Some(qs) => {
                let elapsed = current_time_ns - qs.quarantine_start_ns;
                elapsed < self.quarantine_duration_ns
            },
            None => false,
        }
    }

    /// Attempt to release a sensor from quarantine
    pub fn release_from_quarantine(
        &mut self,
        sensor_id: u64,
        current_time_ns: u64,
        verification_result: bool,
    ) -> Result<bool, QuarantineError> {
        let qs = match self.quarantined_sensors.get(&sensor_id) {
            Some(q) => q,
            None => return Err(QuarantineError::SensorNotFound),
        };

        let elapsed = current_time_ns - qs.quarantine_start_ns;
        
        // Check if quarantine duration has expired
        if elapsed < self.quarantine_duration_ns {
            return Err(QuarantineError::QuarantineNotExpired);
        }

        // Verify sensor location against geographic fences
        if let Some((lat, lon)) = qs.last_verified_location {
            for fence in &self.geographic_fences {
                let distance = haversine_distance(
                    lat, lon,
                    fence.center_lat, fence.center_lon,
                );
                
                if distance <= fence.radius_km && !fence.allowed_sensor_ids.contains(&sensor_id) {
                    return Err(QuarantineError::VerificationFailed);
                }
            }
        }

        // Release if verification passed
        if verification_result {
            self.quarantined_sensors.remove(&sensor_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get effective trust score for a sensor (considering quarantine decay)
    pub fn get_effective_trust(&self, sensor_id: u64, base_trust: f64, current_time_ns: u64) -> f64 {
        match self.quarantined_sensors.get(&sensor_id) {
            Some(qs) => {
                let elapsed_ns = current_time_ns - qs.quarantine_start_ns;
                let elapsed_ratio = (elapsed_ns as f64) / (self.quarantine_duration_ns as f64);
                
                // Exponential trust decay during quarantine
                let decay_multiplier = (-2.0 * elapsed_ratio).exp();
                base_trust * qs.trust_decay_factor * decay_multiplier
            },
            None => base_trust,
        }
    }

    /// Check for geographic impossibility (sensor moved faster than physically possible)
    pub fn check_geographic_impossibility(
        &self,
        sensor_id: u64,
        previous_location: (f64, f64),
        previous_time_ns: u64,
        current_location: (f64, f64),
        current_time_ns: u64,
        max_speed_kmh: f64,
    ) -> Result<bool, QuarantineError> {
        let distance_km = haversine_distance(
            previous_location.0, previous_location.1,
            current_location.0, current_location.1,
        );

        let time_elapsed_hours = (current_time_ns - previous_time_ns) as f64 / 3_600_000_000_000.0;
        
        if time_elapsed_hours <= 0.0 {
            return Ok(false);
        }

        let required_speed = distance_km / time_elapsed_hours;

        if required_speed > max_speed_kmh {
            // Geographic impossibility detected - auto-quarantine
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get all currently quarantined sensors
    pub fn get_quarantined_sensors(&self) -> Vec<&QuarantinedSensor> {
        self.quarantined_sensors.values().collect()
    }

    /// Clear expired quarantines
    pub fn clear_expired(&mut self, current_time_ns: u64) {
        self.quarantined_sensors.retain(|_, qs| {
            let elapsed = current_time_ns - qs.quarantine_start_ns;
            elapsed < self.quarantine_duration_ns
        });
    }
}

fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS_KM: f64 = 6371.0;
    
    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    
    EARTH_RADIUS_KM * c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geographic_impossibility() {
        let quarantine = SpatiotemporalQuarantine::new(60_000_000_000, 0.5);
        
        // New York to London in 1 second (impossible)
        let ny = (40.7128, -74.0060);
        let london = (51.5074, -0.1278);
        
        let result = quarantine.check_geographic_impossibility(
            1,
            ny,
            0,
            london,
            1_000_000_000, // 1 second
            1000.0, // Max 1000 km/h
        ).unwrap();
        
        assert!(result); // Should detect impossibility
    }

    #[test]
    fn test_trust_decay() {
        let mut quarantine = SpatiotemporalQuarantine::new(60_000_000_000, 0.5);
        
        quarantine.quarantine_sensor(
            1,
            QuarantineReason::SybilClusterDetected,
            0,
        ).unwrap();

        let base_trust = 1.0;
        
        // At start of quarantine
        let trust_start = quarantine.get_effective_trust(1, base_trust, 0);
        assert!(trust_start < base_trust);
        
        // Mid-quarantine
        let trust_mid = quarantine.get_effective_trust(1, base_trust, 30_000_000_000);
        assert!(trust_mid < trust_start);
    }
}
