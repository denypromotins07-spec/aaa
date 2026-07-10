//! Sybil Attack Detection Graph for IoT Sensor Networks
//! 
//! Detects coordinated spoofing attacks by analyzing spatiotemporal correlation patterns.
//! Incorporates physical asset registries and satellite cross-validation to prevent false positives.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct SensorNode {
    pub sensor_id: u64,
    pub latitude: f64,
    pub longitude: f64,
    pub registered_asset_id: Option<String>, // Physical asset registry link
    pub last_seen_ns: u64,
}

#[derive(Debug, Clone)]
pub struct SpatiotemporalCluster {
    pub sensor_ids: HashSet<u64>,
    pub centroid_lat: f64,
    pub centroid_lon: f64,
    pub time_variance_ns: u64,
    pub value_correlation: f64,
}

#[derive(Debug)]
pub enum SybilError {
    InsufficientData,
    SatelliteValidationFailed,
    ClusterAnalysisFailed,
}

/// Sybil Detection Graph using spatiotemporal correlation analysis
pub struct SybilDetectionGraph {
    nodes: HashMap<u64, SensorNode>,
    recent_readings: HashMap<u64, VecDeque<(u64, f64)>>, // sensor_id -> (timestamp, value)
    max_window_size: usize,
    correlation_threshold: f64,
    min_cluster_size: usize,
}

impl SybilDetectionGraph {
    pub fn new(max_window_size: usize, correlation_threshold: f64, min_cluster_size: usize) -> Self {
        Self {
            nodes: HashMap::new(),
            recent_readings: HashMap::new(),
            max_window_size,
            correlation_threshold,
            min_cluster_size,
        }
    }

    /// Register a sensor node with optional physical asset ID
    pub fn register_node(&mut self, node: SensorNode) {
        self.nodes.insert(node.sensor_id, node);
    }

    /// Add a reading to the temporal window
    pub fn add_reading(&mut self, sensor_id: u64, timestamp_ns: u64, value: f64) {
        let readings = self.recent_readings.entry(sensor_id).or_insert_with(VecDeque::new);
        
        readings.push_back((timestamp_ns, value));
        
        // Maintain fixed window size
        while readings.len() > self.max_window_size {
            readings.pop_front();
        }
    }

    /// Detect potential Sybil clusters
    /// CRITICAL FIX: Cross-references with physical asset registries and satellite data
    /// to prevent false positives on legitimate synchronized fleets
    pub fn detect_sybil_clusters(
        &self,
        satellite_validated_assets: &HashSet<String>,
    ) -> Result<Vec<SpatiotemporalCluster>, SybilError> {
        if self.nodes.len() < self.min_cluster_size {
            return Err(SybilError::InsufficientData);
        }

        let mut clusters: Vec<SpatiotemporalCluster> = Vec::new();
        let mut visited = HashSet::new();

        // Group sensors by spatiotemporal proximity
        for &sensor_id in self.nodes.keys() {
            if visited.contains(&sensor_id) {
                continue;
            }

            let node = match self.nodes.get(&sensor_id) {
                Some(n) => n,
                None => continue,
            };

            // BFS to find correlated sensors
            let mut cluster_sensors = HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back(sensor_id);
            cluster_sensors.insert(sensor_id);

            while let Some(current_id) = queue.pop_front() {
                let current_node = match self.nodes.get(&current_id) {
                    Some(n) => n,
                    None => continue,
                };

                for (&other_id, other_node) in &self.nodes {
                    if visited.contains(&other_id) || cluster_sensors.contains(&other_id) {
                        continue;
                    }

                    // Check spatial proximity (< 100m)
                    let distance = haversine_distance(
                        current_node.latitude, current_node.longitude,
                        other_node.latitude, other_node.longitude,
                    );

                    if distance > 0.1 { // 100 meters
                        continue;
                    }

                    // Check temporal correlation
                    let correlation = self.calculate_value_correlation(current_id, other_id);
                    
                    if correlation > self.correlation_threshold {
                        cluster_sensors.insert(other_id);
                        queue.push_back(other_id);
                    }
                }
            }

            visited.extend(&cluster_sensors);

            if cluster_sensors.len() >= self.min_cluster_size {
                // Analyze cluster for Sybil indicators
                let cluster = self.build_cluster(&cluster_sensors)?;
                
                // CRITICAL: Check if cluster members are registered physical assets
                let has_physical_registry = cluster_sensors.iter()
                    .filter_map(|id| self.nodes.get(id))
                    .any(|n| n.registered_asset_id.as_ref()
                        .map(|aid| satellite_validated_assets.contains(aid))
                        .unwrap_or(false));

                // Only flag as Sybil if NOT validated by physical registry + satellite
                if !has_physical_registry && cluster.value_correlation > 0.95 {
                    clusters.push(cluster);
                }
            }
        }

        Ok(clusters)
    }

    /// Calculate Pearson correlation between two sensors' value streams
    fn calculate_value_correlation(&self, sensor_a: u64, sensor_b: u64) -> f64 {
        let readings_a = match self.recent_readings.get(&sensor_a) {
            Some(r) if !r.is_empty() => r,
            _ => return 0.0,
        };
        
        let readings_b = match self.recent_readings.get(&sensor_b) {
            Some(r) if !r.is_empty() => r,
            _ => return 0.0,
        };

        // Align by timestamp (simplified - assumes similar timestamps)
        let min_len = readings_a.len().min(readings_b.len());
        if min_len < 3 {
            return 0.0;
        }

        let values_a: Vec<f64> = readings_a.iter().take(min_len).map(|(_, v)| *v).collect();
        let values_b: Vec<f64> = readings_b.iter().take(min_len).map(|(_, v)| *v).collect();

        pearson_correlation(&values_a, &values_b)
    }

    /// Build cluster statistics
    fn build_cluster(&self, sensor_ids: &HashSet<u64>) -> Result<SpatiotemporalCluster, SybilError> {
        if sensor_ids.is_empty() {
            return Err(SybilError::InsufficientData);
        }

        let mut sum_lat = 0.0;
        let mut sum_lon = 0.0;
        let mut timestamps = Vec::new();

        for &sensor_id in sensor_ids {
            if let Some(node) = self.nodes.get(&sensor_id) {
                sum_lat += node.latitude;
                sum_lon += node.longitude;
                
                if let Some(readings) = self.recent_readings.get(&sensor_id) {
                    if let Some((ts, _)) = readings.back() {
                        timestamps.push(*ts);
                    }
                }
            }
        }

        let count = sensor_ids.len() as f64;
        let centroid_lat = sum_lat / count;
        let centroid_lon = sum_lon / count;

        // Calculate time variance
        let mean_time = timestamps.iter().sum::<u64>() as f64 / timestamps.len() as f64;
        let time_variance = timestamps.iter()
            .map(|t| ((*t as f64 - mean_time).powi(2)))
            .sum::<f64>() / timestamps.len() as f64;

        // Calculate average pairwise correlation
        let mut total_correlation = 0.0;
        let mut pair_count = 0;
        let sensor_vec: Vec<u64> = sensor_ids.iter().copied().collect();
        
        for i in 0..sensor_vec.len() {
            for j in (i + 1)..sensor_vec.len() {
                let corr = self.calculate_value_correlation(sensor_vec[i], sensor_vec[j]);
                total_correlation += corr;
                pair_count += 1;
            }
        }

        let avg_correlation = if pair_count > 0 {
            total_correlation / pair_count as f64
        } else {
            0.0
        };

        Ok(SpatiotemporalCluster {
            sensor_ids: sensor_ids.clone(),
            centroid_lat,
            centroid_lon,
            time_variance: time_variance as u64,
            value_correlation: avg_correlation,
        })
    }

    /// Quarantine a detected Sybil cluster
    pub fn quarantine_cluster(&mut self, cluster: &SpatiotemporalCluster) {
        for &sensor_id in &cluster.sensor_ids {
            self.nodes.remove(&sensor_id);
            self.recent_readings.remove(&sensor_id);
        }
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

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_x2: f64 = x.iter().map(|v| v.powi(2)).sum();
    let sum_y2: f64 = y.iter().map(|v| v.powi(2)).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x.powi(2)) * (n * sum_y2 - sum_y.powi(2))).sqrt();

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legitimate_fleet_not_flagged() {
        let mut graph = SybilDetectionGraph::new(100, 0.8, 5);
        
        // Register a legitimate fleet with physical asset IDs
        for i in 0..10 {
            let node = SensorNode {
                sensor_id: i,
                latitude: 40.7128,
                longitude: -74.0060,
                registered_asset_id: Some(format!("SHIP-{}", i)),
                last_seen_ns: 0,
            };
            graph.register_node(node);
            
            // Add highly correlated readings (legitimate fleet behavior)
            for j in 0..50 {
                graph.add_reading(i, j * 1000, 25.0 + (j as f64 * 0.01));
            }
        }

        // Satellite-validated assets
        let mut validated = HashSet::new();
        for i in 0..10 {
            validated.insert(format!("SHIP-{}", i));
        }

        let clusters = graph.detect_sybil_clusters(&validated).unwrap();
        
        // Should NOT detect any Sybil clusters because assets are validated
        assert!(clusters.is_empty());
    }
}
