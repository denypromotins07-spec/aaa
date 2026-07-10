//! Reputation-Weighted Byzantine Fault Tolerant Sensor Consensus
//! 
//! Validates IoT sensor data using BFT consensus with reputation weighting.
//! Cross-references against neighboring nodes and satellite ground-truth.

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensorReading {
    pub sensor_id: u64,
    pub timestamp_ns: u64,
    pub value: f64,
    pub latitude: f64,
    pub longitude: f64,
}

#[derive(Debug, Clone)]
pub struct SensorReputation {
    pub sensor_id: u64,
    pub trust_score: f64,
    pub valid_count: u64,
    pub invalid_count: u64,
    pub last_verified: u64,
}

impl SensorReputation {
    pub fn new(sensor_id: u64) -> Self {
        Self {
            sensor_id,
            trust_score: 1.0,
            valid_count: 0,
            invalid_count: 0,
            last_verified: 0,
        }
    }

    pub fn update(&mut self, is_valid: bool) {
        if is_valid {
            self.valid_count += 1;
            // Exponential moving average for trust score
            self.trust_score = 0.9 * self.trust_score + 0.1 * 1.0;
        } else {
            self.invalid_count += 1;
            self.trust_score = 0.9 * self.trust_score + 0.1 * 0.0;
        }
    }
}

#[derive(Debug)]
pub enum ConsensusError {
    InsufficientNodes,
    QuorumNotReached,
    InvalidTimestamp,
    GeographicOutlier,
}

/// BFT Sensor Validation Engine
pub struct BftSensorValidator {
    reputations: HashMap<u64, SensorReputation>,
    quorum_threshold: f64,
    max_geographic_distance_km: f64,
    time_window_ns: u64,
}

impl BftSensorValidator {
    pub fn new(quorum_threshold: f64, max_distance_km: f64, time_window_ns: u64) -> Self {
        Self {
            reputations: HashMap::new(),
            quorum_threshold,
            max_geographic_distance_km: max_distance_km,
            time_window_ns,
        }
    }

    /// Validate a sensor reading using BFT consensus
    pub fn validate_reading(
        &mut self,
        reading: SensorReading,
        neighboring_readings: &[SensorReading],
        satellite_ground_truth: Option<f64>,
    ) -> Result<(f64, bool), ConsensusError> {
        // Ensure we have enough nodes for consensus
        if neighboring_readings.is_empty() {
            return Err(ConsensusError::InsufficientNodes);
        }

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Check timestamp validity (within window)
        if reading.timestamp_ns > current_time + self.time_window_ns 
            || reading.timestamp_ns < current_time - self.time_window_ns {
            return Err(ConsensusError::InvalidTimestamp);
        }

        // Filter neighbors within geographic bounds
        let valid_neighbors: Vec<&SensorReading> = neighboring_readings
            .iter()
            .filter(|n| {
                let distance = haversine_distance(
                    reading.latitude, reading.longitude,
                    n.latitude, n.longitude,
                );
                distance <= self.max_geographic_distance_km
            })
            .collect();

        if valid_neighbors.is_empty() {
            return Err(ConsensusError::GeographicOutlier);
        }

        // Calculate weighted consensus value
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for neighbor in &valid_neighbors {
            let reputation = self.reputations.entry(neighbor.sensor_id)
                .or_insert_with(|| SensorReputation::new(neighbor.sensor_id));
            
            weighted_sum += neighbor.value * reputation.trust_score;
            total_weight += reputation.trust_score;
        }

        // Add own reading with reputation weight
        let self_reputation = self.reputations.entry(reading.sensor_id)
            .or_insert_with(|| SensorReputation::new(reading.sensor_id));
        weighted_sum += reading.value * self_reputation.trust_score;
        total_weight += self_reputation.trust_score;

        if total_weight < self.quorum_threshold {
            return Err(ConsensusError::QuorumNotReached);
        }

        let consensus_value = weighted_sum / total_weight;

        // Cross-reference with satellite ground truth if available
        let is_valid = if let Some(satellite_value) = satellite_ground_truth {
            let deviation = (consensus_value - satellite_value).abs();
            // Allow 5% deviation from satellite truth
            deviation < satellite_value.abs() * 0.05
        } else {
            // Use statistical validation (within 2 standard deviations)
            let variance: f64 = valid_neighbors.iter()
                .map(|n| (n.value - consensus_value).powi(2))
                .sum::<f64>() / valid_neighbors.len() as f64;
            
            let std_dev = variance.sqrt();
            (reading.value - consensus_value).abs() < 2.0 * std_dev.max(0.001)
        };

        // Update reputations
        self_reputation.update(is_valid);
        for neighbor in &valid_neighbors {
            let rep = self.reputations.get_mut(&neighbor.sensor_id).unwrap();
            let deviation = (neighbor.value - consensus_value).abs();
            let is_neighbor_valid = deviation < 2.0 * (consensus_value.abs() * 0.05).max(0.001);
            rep.update(is_neighbor_valid);
        }

        Ok((consensus_value, is_valid))
    }

    /// Get reputation for a specific sensor
    pub fn get_reputation(&self, sensor_id: u64) -> Option<&SensorReputation> {
        self.reputations.get(&sensor_id)
    }
}

/// Haversine distance calculation between two GPS coordinates
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
    fn test_bft_consensus() {
        let mut validator = BftSensorValidator::new(0.6, 10.0, 60_000_000_000);
        
        let reading = SensorReading {
            sensor_id: 1,
            timestamp_ns: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            value: 25.0,
            latitude: 40.7128,
            longitude: -74.0060,
        };

        let neighbors = vec![
            SensorReading {
                sensor_id: 2,
                timestamp_ns: reading.timestamp_ns,
                value: 25.1,
                latitude: 40.7130,
                longitude: -74.0062,
            },
            SensorReading {
                sensor_id: 3,
                timestamp_ns: reading.timestamp_ns,
                value: 24.9,
                latitude: 40.7125,
                longitude: -74.0058,
            },
        ];

        let result = validator.validate_reading(reading, &neighbors, None);
        assert!(result.is_ok());
        
        let (consensus_value, is_valid) = result.unwrap();
        assert!((consensus_value - 25.0).abs() < 0.2);
        assert!(is_valid);
    }
}
