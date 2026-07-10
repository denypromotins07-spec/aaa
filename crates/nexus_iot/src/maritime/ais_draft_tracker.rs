//! AIS Maritime Draft & Load Tracker
//! 
//! Processes Automatic Identification System (AIS) data from cargo ships.
//! Calculates commodity tonnage from vessel draft measurements.

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct AisMessage {
    pub mmsi: u32, // Maritime Mobile Service Identity
    pub latitude: f64,
    pub longitude: f64,
    pub speed_knots: f32,
    pub course_degrees: f32,
    pub draft_meters: f32,
    pub vessel_type: u8,
    pub length_meters: u16,
    pub beam_meters: u16,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone)]
pub struct VesselState {
    pub mmsi: u32,
    pub last_known_position: (f64, f64),
    pub avg_draft_meters: f32,
    pub max_observed_draft: f32,
    pub deadweight_tonnage: f64,
    pub cargo_type: CargoType,
    pub last_update_ns: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum CargoType {
    BulkCarrier,
    Tanker,
    Container,
    GeneralCargo,
    Unknown,
}

#[derive(Debug)]
pub struct CommodityFlowEstimate {
    pub port_id: String,
    pub inbound_tonnage: f64,
    pub outbound_tonnage: f64,
    pub net_flow: f64,
    pub primary_commodity: String,
}

/// AIS Draft Tracker for maritime commodity flow estimation
pub struct AisDraftTracker {
    vessels: HashMap<u32, VesselState>,
    recent_messages: Vec<AisMessage>,
    max_message_window_ns: u64,
    port_zones: HashMap<String, PortZone>,
}

#[derive(Debug, Clone)]
pub struct PortZone {
    pub port_id: String,
    pub center_lat: f64,
    pub center_lon: f64,
    pub radius_km: f64,
}

impl AisDraftTracker {
    pub fn new(max_window_ns: u64) -> Self {
        Self {
            vessels: HashMap::new(),
            recent_messages: Vec::new(),
            max_message_window_ns: max_window_ns,
            port_zones: HashMap::new(),
        }
    }

    /// Register a port zone for monitoring
    pub fn register_port(&mut self, port: PortZone) {
        self.port_zones.insert(port.port_id.clone(), port);
    }

    /// Process incoming AIS message
    pub fn process_message(&mut self, msg: AisMessage) {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Update vessel state
        let vessel = self.vessels.entry(msg.mmsi).or_insert_with(|| VesselState {
            mmsi: msg.mmsi,
            last_known_position: (msg.latitude, msg.longitude),
            avg_draft_meters: msg.draft_meters,
            max_observed_draft: msg.draft_meters,
            deadweight_tonnage: estimate_deadweight(msg.length_meters, msg.beam_meters, msg.vessel_type),
            cargo_type: classify_cargo_type(msg.vessel_type),
            last_update_ns: msg.timestamp_ns,
        });

        // Update draft tracking (exponential moving average)
        vessel.avg_draft_meters = 0.7 * vessel.avg_draft_meters + 0.3 * msg.draft_meters;
        vessel.max_observed_draft = vessel.max_observed_draft.max(msg.draft_meters);
        vessel.last_known_position = (msg.latitude, msg.longitude);
        vessel.last_update_ns = msg.timestamp_ns;

        // Store recent message
        if current_time - msg.timestamp_ns <= self.max_message_window_ns {
            self.recent_messages.push(msg);
            
            if self.recent_messages.len() > 100000 {
                self.recent_messages.retain(|m| 
                    current_time - m.timestamp_ns <= self.max_message_window_ns
                );
            }
        }
    }

    /// Calculate commodity flow for a port
    pub fn calculate_port_flow(&self, port_id: &str) -> Option<CommodityFlowEstimate> {
        let port = self.port_zones.get(port_id)?;
        let port_radius_sq = port.radius_km.powi(2);

        let mut inbound_tonnage = 0.0;
        let mut outbound_tonnage = 0.0;
        let mut commodity_counts: HashMap<String, usize> = HashMap::new();

        for vessel in self.vessels.values() {
            let dist_sq = haversine_distance_squared(
                vessel.last_known_position.0, vessel.last_known_position.1,
                port.center_lat, port.center_lon,
            );

            if dist_sq <= port_radius_sq {
                // Estimate cargo weight from draft
                let load_factor = (vessel.avg_draft_meters / vessel.max_observed_draft).min(1.0);
                let cargo_tonnage = vessel.deadweight_tonnage * load_factor as f64 * 0.8; // Assume 80% max capacity

                // Determine direction based on recent movement
                let is_inbound = self.is_vessel_inbound(vessel.mmsi, port);

                if is_inbound {
                    inbound_tonnage += cargo_tonnage;
                } else {
                    outbound_tonnage += cargo_tonnage;
                }

                // Track commodity type
                let commodity_key = format!("{:?}", vessel.cargo_type);
                *commodity_counts.entry(commodity_key).or_insert(0) += 1;
            }
        }

        // Determine primary commodity
        let primary_commodity = commodity_counts.iter()
            .max_by_key(|(_, count)| *count)
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "Unknown".to_string());

        Some(CommodityFlowEstimate {
            port_id: port_id.to_string(),
            inbound_tonnage,
            outbound_tonnage,
            net_flow: inbound_tonnage - outbound_tonnage,
            primary_commodity,
        })
    }

    /// Check if vessel is heading into port (simplified)
    fn is_vessel_inbound(&self, mmsi: u32, port: &PortZone) -> bool {
        // Find most recent message for this vessel
        let vessel_msg = self.recent_messages.iter()
            .filter(|m| m.mmsi == mmsi)
            .last();

        if let Some(msg) = vessel_msg {
            // Simple heuristic: vessel within port zone and speed < 5 knots = inbound/docked
            let dist = haversine_distance(
                msg.latitude, msg.longitude,
                port.center_lat, port.center_lon,
            );
            
            dist <= port.radius_km && msg.speed_knots < 5.0
        } else {
            false
        }
    }

    /// Get total global fleet tonnage by type
    pub fn get_fleet_tonnage_by_type(&self) -> HashMap<CargoType, f64> {
        let mut tonnage: HashMap<CargoType, f64> = HashMap::new();

        for vessel in self.vessels.values() {
            let load_factor = (vessel.avg_draft_meters / vessel.max_observed_draft).min(1.0);
            let cargo_tonnage = vessel.deadweight_tonnage * load_factor as f64;
            
            *tonnage.entry(vessel.cargo_type).or_insert(0.0) += cargo_tonnage;
        }

        tonnage
    }
}

fn estimate_deadweight(length_m: u16, beam_m: u16, vessel_type: u8) -> f64 {
    // Simplified deadweight estimation based on vessel dimensions
    let volume_factor = (length_m as f64) * (beam_m as f64) * (beam_m as f64) * 0.5;
    
    match vessel_type {
        70..=79 => volume_factor * 0.8, // Tanker
        80..=89 => volume_factor * 0.7, // Bulk carrier
        _ => volume_factor * 0.5, // Other
    }
}

fn classify_cargo_type(vessel_type: u8) -> CargoType {
    match vessel_type {
        70..=79 => CargoType::Tanker,
        80..=89 => CargoType::BulkCarrier,
        30..=39 => CargoType::Container,
        40..=49 => CargoType::GeneralCargo,
        _ => CargoType::Unknown,
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
    
    2.0 * a.sqrt().atan2((1.0 - a).sqrt()) * EARTH_RADIUS_KM
}

fn haversine_distance_squared(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    haversine_distance(lat1, lon1, lat2, lon2).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ais_processing() {
        let mut tracker = AisDraftTracker::new(3_600_000_000_000);
        
        // Register a port
        tracker.register_port(PortZone {
            port_id: "PORT-LA".to_string(),
            center_lat: 33.7362,
            center_lon: -118.2644,
            radius_km: 10.0,
        });

        // Add simulated AIS message (bulk carrier near LA port)
        tracker.process_message(AisMessage {
            mmsi: 123456789,
            latitude: 33.74,
            longitude: -118.26,
            speed_knots: 3.0,
            course_degrees: 90.0,
            draft_meters: 12.0,
            vessel_type: 80, // Bulk carrier
            length_meters: 200,
            beam_meters: 30,
            timestamp_ns: 0,
        });

        let flow = tracker.calculate_port_flow("PORT-LA");
        assert!(flow.is_some());
        
        let flow = flow.unwrap();
        assert!(flow.inbound_tonnage > 0.0 || flow.outbound_tonnage > 0.0);
    }
}
