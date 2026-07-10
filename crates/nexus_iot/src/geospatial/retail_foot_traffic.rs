//! Retail Foot Traffic Estimator from Geospatial IoT Data
//! 
//! Aggregates anonymized mobile device pings to estimate retail activity.
//! Generates same-store sales alpha signals for equity trading.

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct GeoPing {
    pub device_hash: u64, // Anonymized device identifier
    pub latitude: f64,
    pub longitude: f64,
    pub timestamp_ns: u64,
    pub accuracy_meters: f32,
}

#[derive(Debug, Clone)]
pub struct CommercialZone {
    pub zone_id: String,
    pub center_lat: f64,
    pub center_lon: f64,
    pub radius_meters: f32,
    pub retailer_name: Option<String>,
}

#[derive(Debug)]
pub struct FootTrafficEstimate {
    pub zone_id: String,
    pub unique_devices: usize,
    pub avg_dwell_time_seconds: f64,
    pub entry_rate_per_minute: f64,
    pub confidence_score: f64,
}

#[derive(Debug)]
pub enum FootTrafficError {
    InsufficientData,
    LowAccuracyReadings,
    ZoneNotFound,
}

/// Retail Foot Traffic Estimator
pub struct RetailFootTrafficEstimator {
    zones: HashMap<String, CommercialZone>,
    recent_pings: Vec<GeoPing>,
    max_ping_window_ns: u64,
    min_accuracy_meters: f32,
}

impl RetailFootTrafficEstimator {
    pub fn new(max_ping_window_ns: u64, min_accuracy_meters: f32) -> Self {
        Self {
            zones: HashMap::new(),
            recent_pings: Vec::new(),
            max_ping_window_ns,
            min_accuracy_meters,
        }
    }

    /// Register a commercial zone for monitoring
    pub fn register_zone(&mut self, zone: CommercialZone) {
        self.zones.insert(zone.zone_id.clone(), zone);
    }

    /// Add geospatial ping data
    pub fn add_ping(&mut self, ping: GeoPing) {
        // Filter by accuracy
        if ping.accuracy_meters > self.min_accuracy_meters {
            return;
        }

        // Filter by timestamp window
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        if current_time - ping.timestamp_ns > self.max_ping_window_ns {
            return;
        }

        self.recent_pings.push(ping);

        // Prune old pings periodically
        if self.recent_pings.len() > 10000 {
            self.recent_pings.retain(|p| current_time - p.timestamp_ns <= self.max_ping_window_ns);
        }
    }

    /// Estimate foot traffic for a specific zone
    pub fn estimate_traffic(&self, zone_id: &str) -> Result<FootTrafficEstimate, FootTrafficError> {
        let zone = self.zones.get(zone_id)
            .ok_or(FootTrafficError::ZoneNotFound)?;

        let zone_radius_sq = (zone.radius_meters as f64).powi(2);
        
        // Filter pings within zone
        let zone_pings: Vec<&GeoPing> = self.recent_pings.iter()
            .filter(|p| {
                let dist_sq = haversine_distance_squared(
                    p.latitude, p.longitude,
                    zone.center_lat, zone.center_lon,
                );
                dist_sq <= zone_radius_sq
            })
            .collect();

        if zone_pings.is_empty() {
            return Err(FootTrafficError::InsufficientData);
        }

        // Count unique devices
        let unique_devices: HashSet<u64> = zone_pings.iter()
            .map(|p| p.device_hash)
            .collect();

        // Calculate dwell times per device
        let mut device_times: HashMap<u64, Vec<u64>> = HashMap::new();
        for ping in &zone_pings {
            device_times.entry(ping.device_hash)
                .or_insert_with(Vec::new)
                .push(ping.timestamp_ns);
        }

        let mut total_dwell_time = 0.0;
        let mut device_count = 0;

        for (_, timestamps) in &device_times {
            if timestamps.len() >= 2 {
                let min_time = *timestamps.iter().min().unwrap();
                let max_time = *timestamps.iter().max().unwrap();
                let dwell = (max_time - min_time) as f64 / 1_000_000_000.0; // Convert to seconds
                
                if dwell < 3600.0 { // Cap at 1 hour to filter outliers
                    total_dwell_time += dwell;
                    device_count += 1;
                }
            }
        }

        let avg_dwell_time = if device_count > 0 {
            total_dwell_time / device_count as f64
        } else {
            0.0
        };

        // Calculate entry rate (new devices per minute)
        let window_minutes = (self.max_ping_window_ns / 60_000_000_000) as f64;
        let entry_rate = unique_devices.len() as f64 / window_minutes.max(1.0);

        // Confidence score based on sample size and accuracy
        let sample_confidence = (unique_devices.len() as f64 / 100.0).min(1.0);
        let accuracy_confidence = 1.0 - (self.min_accuracy_meters / 50.0).min(1.0);
        let confidence_score = sample_confidence * accuracy_confidence;

        Ok(FootTrafficEstimate {
            zone_id: zone_id.to_string(),
            unique_devices: unique_devices.len(),
            avg_dwell_time_seconds: avg_dwell_time,
            entry_rate_per_minute: entry_rate,
            confidence_score,
        })
    }

    /// Generate same-store sales signal
    pub fn generate_sales_signal(&self, zone_id: &str, baseline_traffic: f64) -> Option<f64> {
        match self.estimate_traffic(zone_id) {
            Ok(estimate) => {
                let current_traffic = estimate.unique_devices as f64;
                let pct_change = (current_traffic - baseline_traffic) / baseline_traffic * 100.0;
                
                // Simple linear model: 1% traffic change ≈ 0.8% sales change
                Some(pct_change * 0.8)
            },
            Err(_) => None,
        }
    }
}

fn haversine_distance_squared(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS_M: f64 = 6371000.0;
    
    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    
    // Return squared distance for comparison
    (2.0 * a.sqrt().atan2((1.0 - a).sqrt()) * EARTH_RADIUS_M).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foot_traffic_estimation() {
        let mut estimator = RetailFootTrafficEstimator::new(3_600_000_000_000, 20.0);
        
        // Register a store zone
        estimator.register_zone(CommercialZone {
            zone_id: "STORE-001".to_string(),
            center_lat: 40.7128,
            center_lon: -74.0060,
            radius_meters: 100.0,
            retailer_name: Some("Test Store".to_string()),
        });

        // Add simulated pings
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
        for i in 0..50 {
            estimator.add_ping(GeoPing {
                device_hash: i as u64,
                latitude: 40.7128 + (i as f64 * 0.0001),
                longitude: -74.0060 + (i as f64 * 0.0001),
                timestamp_ns: current_time - (i * 60_000_000_000),
                accuracy_meters: 5.0,
            });
        }

        let estimate = estimator.estimate_traffic("STORE-001").unwrap();
        assert!(estimate.unique_devices > 0);
        assert!(estimate.confidence_score > 0.0);
    }
}
