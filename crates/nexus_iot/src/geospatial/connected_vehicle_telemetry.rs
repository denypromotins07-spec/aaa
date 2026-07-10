//! Connected Vehicle Telemetry Processor
//! 
//! Processes OBD-II dongle data from connected vehicles.
//! Estimates economic activity from vehicle movement patterns.

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct VehicleTelemetry {
    pub vehicle_hash: u64,
    pub latitude: f64,
    pub longitude: f64,
    pub speed_kmh: f32,
    pub engine_load_pct: f32,
    pub fuel_level_pct: f32,
    pub timestamp_ns: u64,
}

#[derive(Debug)]
pub struct EconomicActivityIndex {
    pub region_id: String,
    pub active_vehicles: usize,
    pub avg_speed_kmh: f64,
    pub commercial_vehicle_ratio: f64,
    pub activity_index: f64, // 0-100 scale
}

/// Connected Vehicle Telemetry Processor
pub struct ConnectedVehicleProcessor {
    recent_telemetry: Vec<VehicleTelemetry>,
    max_window_ns: u64,
    commercial_speed_threshold_kmh: f32,
}

impl ConnectedVehicleProcessor {
    pub fn new(max_window_ns: u64) -> Self {
        Self {
            recent_telemetry: Vec::new(),
            max_window_ns,
            commercial_speed_threshold_kmh: 80.0,
        }
    }

    /// Add vehicle telemetry reading
    pub fn add_reading(&mut self, reading: VehicleTelemetry) {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        if current_time - reading.timestamp_ns <= self.max_window_ns {
            self.recent_telemetry.push(reading);
            
            // Prune old readings
            if self.recent_telemetry.len() > 50000 {
                self.recent_telemetry.retain(|r| 
                    current_time - r.timestamp_ns <= self.max_window_ns
                );
            }
        }
    }

    /// Calculate economic activity index for a region
    pub fn calculate_activity_index(&self, region_id: &str, bounds: (f64, f64, f64, f64)) -> Option<EconomicActivityIndex> {
        let (min_lat, max_lat, min_lon, max_lon) = bounds;

        // Filter vehicles within region bounds
        let region_vehicles: Vec<&VehicleTelemetry> = self.recent_telemetry.iter()
            .filter(|v| {
                v.latitude >= min_lat && v.latitude <= max_lat &&
                v.longitude >= min_lon && v.longitude <= max_lon
            })
            .collect();

        if region_vehicles.is_empty() {
            return None;
        }

        // Count unique vehicles
        let unique_vehicles: HashSet<u64> = region_vehicles.iter()
            .map(|v| v.vehicle_hash)
            .collect();

        // Calculate average speed
        let total_speed: f64 = region_vehicles.iter()
            .map(|v| v.speed_kmh as f64)
            .sum();
        let avg_speed = total_speed / region_vehicles.len() as f64;

        // Estimate commercial vehicle ratio (high speed + high engine load)
        let commercial_count = region_vehicles.iter()
            .filter(|v| v.speed_kmh > self.commercial_speed_threshold_kmh && v.engine_load_pct > 70.0)
            .count();
        
        let commercial_ratio = commercial_count as f64 / unique_vehicles.len() as f64;

        // Calculate activity index (0-100)
        // Based on: vehicle density, speed distribution, commercial ratio
        let density_factor = (unique_vehicles.len() as f64 / 1000.0).min(1.0) * 40.0;
        let speed_factor = ((avg_speed / 60.0).min(1.0)) * 30.0;
        let commercial_factor = commercial_ratio * 30.0;

        let activity_index = (density_factor + speed_factor + commercial_factor).min(100.0);

        Some(EconomicActivityIndex {
            region_id: region_id.to_string(),
            active_vehicles: unique_vehicles.len(),
            avg_speed_kmh: avg_speed,
            commercial_vehicle_ratio: commercial_ratio,
            activity_index,
        })
    }

    /// Detect supply chain disruption from vehicle patterns
    pub fn detect_supply_chain_anomaly(&self, baseline_index: f64, current_index: f64) -> Option<f64> {
        if baseline_index <= 0.0 {
            return None;
        }

        let pct_change = (current_index - baseline_index) / baseline_index * 100.0;
        
        // Significant drop indicates potential disruption
        if pct_change < -20.0 {
            Some(pct_change)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activity_index_calculation() {
        let mut processor = ConnectedVehicleProcessor::new(3_600_000_000_000);
        
        // Add simulated vehicle readings
        for i in 0..100 {
            processor.add_reading(VehicleTelemetry {
                vehicle_hash: i as u64,
                latitude: 40.7 + (i as f64 * 0.001),
                longitude: -74.0 + (i as f64 * 0.001),
                speed_kmh: 50.0 + (i as f32 % 40),
                engine_load_pct: 40.0 + (i as f32 % 50),
                fuel_level_pct: 50.0,
                timestamp_ns: 0,
            });
        }

        let bounds = (40.7, 40.8, -74.0, -73.9);
        let index = processor.calculate_activity_index("NYC", bounds);
        
        assert!(index.is_some());
        let idx = index.unwrap();
        assert!(idx.activity_index > 0.0);
        assert!(idx.activity_index <= 100.0);
    }
}
