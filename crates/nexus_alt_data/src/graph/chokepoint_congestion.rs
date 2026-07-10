//! Chokepoint Congestion Tracker
//! 
//! Tracks ship counts and congestion levels at global chokepoints
//! using satellite vision data.

use std::time::SystemTime;
use thiserror::Error;

/// Chokepoint tracking errors
#[derive(Debug, Error)]
pub enum ChokepointError {
    #[error("Invalid ship count: {0}")]
    InvalidShipCount(String),
    #[error("Location not found: {0}")]
    LocationNotFound(String),
}

/// Ship detection from satellite imagery
#[derive(Debug, Clone)]
pub struct ShipDetection {
    pub latitude: f64,
    pub longitude: f64,
    pub length_meters: f64,
    pub width_meters: f64,
    pub confidence: f64,
    pub ship_type: ShipType,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShipType {
    Container,
    BulkCarrier,
    Tanker,
    LngCarrier,
    VehicleCarrier,
    Unknown,
}

/// Chokepoint congestion state
#[derive(Debug, Clone)]
pub struct ChokepointState {
    pub name: String,
    pub latitude: f64,
    pub longitude: f64,
    pub ships_waiting: u32,
    pub ships_in_transit: u32,
    pub avg_wait_time_hours: f64,
    pub congestion_level: f64, // 0.0 (free) to 1.0 (saturated)
    pub last_updated: SystemTime,
}

impl ChokepointState {
    pub fn new(
        name: String,
        latitude: f64,
        longitude: f64,
    ) -> Self {
        ChokepointState {
            name,
            latitude,
            longitude,
            ships_waiting: 0,
            ships_in_transit: 0,
            avg_wait_time_hours: 0.0,
            congestion_level: 0.0,
            last_updated: SystemTime::now(),
        }
    }

    /// Update congestion level based on ship counts
    pub fn update_from_ships(&mut self, waiting: u32, in_transit: u32, max_capacity: u32) {
        self.ships_waiting = waiting;
        self.ships_in_transit = in_transit;
        
        let total = waiting + in_transit;
        self.congestion_level = if max_capacity > 0 {
            (total as f64 / max_capacity as f64).min(1.0)
        } else {
            0.0
        };
        
        // Estimate wait time based on queue length and typical throughput
        let throughput_per_hour = 2.0; // Ships per hour typical
        self.avg_wait_time_hours = if throughput_per_hour > 0.0 {
            waiting as f64 / throughput_per_hour
        } else {
            0.0
        };
        
        self.last_updated = SystemTime::now();
    }
}

/// Chokepoint congestion tracker
pub struct ChokepointCongestionTracker {
    chokepoints: Vec<ChokepointState>,
    ship_detections: Vec<ShipDetection>,
}

impl ChokepointCongestionTracker {
    pub fn new() -> Self {
        let mut tracker = ChokepointCongestionTracker {
            chokepoints: Vec::new(),
            ship_detections: Vec::new(),
        };
        
        // Initialize with major chokepoints
        tracker.initialize_chokepoints();
        tracker
    }

    fn initialize_chokepoints(&mut self) {
        // Suez Canal
        self.chokepoints.push(ChokepointState::new(
            "Suez Canal".to_string(),
            30.5667,
            32.2667,
        ));
        
        // Panama Canal
        self.chokepoints.push(ChokepointState::new(
            "Panama Canal".to_string(),
            9.0833,
            -79.6833,
        ));
        
        // Strait of Hormuz
        self.chokepoints.push(ChokepointState::new(
            "Strait of Hormuz".to_string(),
            26.5667,
            56.2500,
        ));
        
        // Strait of Malacca
        self.chokepoints.push(ChokepointState::new(
            "Strait of Malacca".to_string(),
            2.5000,
            101.5000,
        ));
        
        // Port of Los Angeles anchorage
        self.chokepoints.push(ChokepointState::new(
            "Port of Los Angeles".to_string(),
            33.7361,
            -118.2639,
        ));
    }

    /// Add ship detections from satellite imagery
    pub fn add_ship_detections(&mut self, detections: Vec<ShipDetection>) {
        self.ship_detections.extend(detections);
    }

    /// Process detections and update chokepoint states
    pub fn process_detections(&mut self, detection_radius_km: f64) -> Result<(), ChokepointError> {
        for chokepoint in &mut self.chokepoints {
            let mut waiting = 0u32;
            let mut in_transit = 0u32;
            
            for detection in &self.ship_detections {
                let distance = Self::haversine_distance(
                    chokepoint.latitude,
                    chokepoint.longitude,
                    detection.latitude,
                    detection.longitude,
                );
                
                if distance < detection_radius_km {
                    // Determine if ship is waiting or in transit based on type and position
                    // Simplified: assume tankers/bulk carriers near chokepoints are waiting
                    match detection.ship_type {
                        ShipType::Tanker | ShipType::BulkCarrier | ShipType::Container => {
                            waiting += 1;
                        }
                        _ => {
                            in_transit += 1;
                        }
                    }
                }
            }
            
            // Set max capacity based on chokepoint
            let max_capacity = match chokepoint.name.as_str() {
                "Suez Canal" => 50,
                "Panama Canal" => 30,
                "Strait of Hormuz" => 100,
                "Strait of Malacca" => 150,
                "Port of Los Angeles" => 40,
                _ => 50,
            };
            
            chokepoint.update_from_ships(waiting, in_transit, max_capacity);
        }
        
        Ok(())
    }

    /// Get congestion state for a specific chokepoint
    pub fn get_chokepoint_state(&self, name: &str) -> Option<&ChokepointState> {
        self.chokepoints.iter().find(|c| c.name == name)
    }

    /// Get all congested chokepoints above threshold
    pub fn get_congested_chokepoints(&self, threshold: f64) -> Vec<&ChokepointState> {
        self.chokepoints
            .iter()
            .filter(|c| c.congestion_level >= threshold)
            .collect()
    }

    /// Calculate haversine distance between two points
    fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        let r = 6371.0; // Earth radius in km
        
        let lat1_rad = lat1.to_radians();
        let lat2_rad = lat2.to_radians();
        let delta_lat = (lat2 - lat1).to_radians();
        let delta_lon = (lon2 - lon1).to_radians();
        
        let a = (delta_lat / 2.0).sin().powi(2)
              + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
        
        let c = 2.0 * a.sqrt().atan();
        r * c
    }
}

impl Default for ChokepointCongestionTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_initialization() {
        let tracker = ChokepointCongestionTracker::new();
        assert_eq!(tracker.chokepoints.len(), 5);
    }

    #[test]
    fn test_haversine_distance() {
        // Distance between two close points
        let dist = ChokepointCongestionTracker::haversine_distance(
            0.0, 0.0, 0.0, 1.0
        );
        assert!(dist > 100.0 && dist < 120.0); // ~111 km per degree
    }

    #[test]
    fn test_congestion_update() {
        let mut state = ChokepointState::new("Test".to_string(), 0.0, 0.0);
        state.update_from_ships(10, 5, 50);
        
        assert_eq!(state.ships_waiting, 10);
        assert_eq!(state.ships_in_transit, 5);
        assert!(state.congestion_level > 0.0);
    }
}
