//! Predictive Cooling Map for Thermal-Aware Thread Scheduling
//! 
//! Generates spatial thermal maps and cooling recommendations for
//! proactive workload distribution across CPU/GPU dies.

use core::fmt;
use crate::thermal::adi_heat_equation::{ThermalGrid, AdiHeatSolver, BoundaryCondition, ThermalPdeError};
use crate::thermal::ebpf_thread_migrator::{CoreThermalState, ThreadState};

/// Maximum map resolution
const MAX_MAP_SIZE: usize = 128;
/// Default ambient temperature (°C)
const DEFAULT_AMBIENT: f64 = 25.0;
/// Thermal conductivity threshold for hotspot detection
const HOTSPOT_THRESHOLD: f64 = 10.0;

/// Errors in cooling map generation
#[derive(Debug, Clone, PartialEq)]
pub enum CoolingMapError {
    InvalidResolution,
    GridMismatch,
    InterpolationFailure,
    SensorDataStale,
}

impl fmt::Display for CoolingMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoolingMapError::InvalidResolution => write!(f, "Invalid map resolution"),
            CoolingMapError::GridMismatch => write!(f, "Grid dimensions mismatch"),
            CoolingMapError::InterpolationFailure => write!(f, "Temperature interpolation failed"),
            CoolingMapError::SensorDataStale => write!(f, "Sensor data is stale"),
        }
    }
}

/// Cooling recommendation type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoolingRecommendation {
    /// No action needed
    None,
    /// Increase fan/pump speed
    IncreaseCooling(u8), // Percentage increase
    /// Migrate workload away
    MigrateWorkload,
    /// Reduce clock frequency
    ThrottleClock,
    /// Emergency shutdown
    EmergencyShutdown,
}

/// Thermal zone classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThermalZoneType {
    /// Normal operating temperature
    Normal,
    /// Elevated but safe temperature
    Warm,
    /// Approaching threshold
    Hot,
    /// Critical - immediate action needed
    Critical,
}

/// Cooling map cell
#[derive(Debug, Clone, Copy)]
pub struct CoolingMapCell {
    /// Temperature at this location (°C)
    pub temperature: f64,
    /// Temperature gradient magnitude (°C/mm)
    pub gradient: f64,
    /// Zone classification
    pub zone_type: ThermalZoneType,
    /// Recommended action
    pub recommendation: CoolingRecommendation,
    /// Confidence in prediction (0-1)
    pub confidence: f64,
}

impl Default for CoolingMapCell {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_AMBIENT,
            gradient: 0.0,
            zone_type: ThermalZoneType::Normal,
            recommendation: CoolingRecommendation::None,
            confidence: 1.0,
        }
    }
}

/// Predictive cooling map
pub struct PredictiveCoolingMap {
    /// Map data [x][y]
    map_data: [[CoolingMapCell; MAX_MAP_SIZE]; MAX_MAP_SIZE],
    /// Map dimensions
    width: usize,
    height: usize,
    /// Physical dimensions (mm)
    physical_width_mm: f64,
    physical_height_mm: f64,
    /// Timestamp of last update (ms)
    last_update_ms: u64,
    /// Data freshness threshold (ms)
    freshness_threshold_ms: u64,
}

impl PredictiveCoolingMap {
    /// Create a new cooling map
    pub fn new(width: usize, height: usize, phys_width_mm: f64, phys_height_mm: f64)
        -> Result<Self, CoolingMapError>
    {
        if width == 0 || height == 0 || width > MAX_MAP_SIZE || height > MAX_MAP_SIZE {
            return Err(CoolingMapError::InvalidResolution);
        }
        if phys_width_mm <= 0.0 || phys_height_mm <= 0.0 {
            return Err(CoolingMapError::InvalidResolution);
        }

        let mut map_data = [[CoolingMapCell::default(); MAX_MAP_SIZE]; MAX_MAP_SIZE];
        
        // Initialize with ambient temperature
        for y in 0..height {
            for x in 0..width {
                map_data[x][y].temperature = DEFAULT_AMBIENT;
            }
        }

        Ok(Self {
            map_data,
            width,
            height,
            physical_width_mm: phys_width_mm,
            physical_height_mm: phys_height_mm,
            last_update_ms: 0,
            freshness_threshold_ms: 100, // 100ms freshness requirement
        })
    }

    /// Update map from thermal grid
    pub fn update_from_grid(&mut self, grid: &ThermalGrid, timestamp_ms: u64) 
        -> Result<(), CoolingMapError>
    {
        // Check data freshness
        if timestamp_ms - self.last_update_ms > self.freshness_threshold_ms {
            // Data might be stale, but still use it
        }

        // Interpolate grid data to map
        for y in 0..self.height {
            for x in 0..self.width {
                // Calculate physical position
                let phys_x = (x as f64 / self.width as f64) * self.physical_width_mm;
                let phys_y = (y as f64 / self.height as f64) * self.physical_height_mm;

                // Map to grid coordinates (simplified interpolation)
                let grid_x = ((phys_x / self.physical_width_mm) * grid.nx as f64) as usize;
                let grid_y = ((phys_y / self.physical_height_mm) * grid.ny as f64) as usize;
                let grid_z = 0; // Top layer

                if let Some(temp) = grid.get_temperature(grid_x.min(grid.nx - 1), grid_y.min(grid.ny - 1), grid_z) {
                    self.map_data[x][y].temperature = temp;
                    
                    // Calculate gradient (simplified)
                    self.map_data[x][y].gradient = self.calculate_gradient(x, y);
                    
                    // Classify zone
                    self.map_data[x][y].zone_type = self.classify_zone(temp);
                    
                    // Generate recommendation
                    self.map_data[x][y].recommendation = self.generate_recommendation(
                        temp, 
                        self.map_data[x][y].gradient
                    );
                }
            }
        }

        self.last_update_ms = timestamp_ms;
        Ok(())
    }

    /// Calculate temperature gradient at location
    fn calculate_gradient(&self, x: usize, y: usize) -> f64 {
        let dx = self.physical_width_mm / self.width as f64;
        let dy = self.physical_height_mm / self.height as f64;

        let t_center = self.map_data[x][y].temperature;
        
        // Central difference for gradient estimation
        let t_right = if x + 1 < self.width {
            self.map_data[x + 1][y].temperature
        } else {
            t_center
        };
        
        let t_up = if y + 1 < self.height {
            self.map_data[x][y + 1].temperature
        } else {
            t_center
        };

        let dt_dx = (t_right - t_center) / dx;
        let dt_dy = (t_up - t_center) / dy;

        (dt_dx.powi(2) + dt_dy.powi(2)).sqrt()
    }

    /// Classify thermal zone based on temperature
    fn classify_zone(&self, temp: f64) -> ThermalZoneType {
        if temp < 60.0 {
            ThermalZoneType::Normal
        } else if temp < 75.0 {
            ThermalZoneType::Warm
        } else if temp < 85.0 {
            ThermalZoneType::Hot
        } else {
            ThermalZoneType::Critical
        }
    }

    /// Generate cooling recommendation
    fn generate_recommendation(&self, temp: f64, gradient: f64) -> CoolingRecommendation {
        if temp >= 95.0 {
            CoolingRecommendation::EmergencyShutdown
        } else if temp >= 85.0 {
            CoolingRecommendation::ThrottleClock
        } else if temp >= 75.0 && gradient > HOTSPOT_THRESHOLD {
            CoolingRecommendation::MigrateWorkload
        } else if temp >= 70.0 {
            CoolingRecommendation::IncreaseCooling(20)
        } else if temp >= 60.0 {
            CoolingRecommendation::IncreaseCooling(10)
        } else {
            CoolingRecommendation::None
        }
    }

    /// Get temperature at specific location
    pub fn get_temperature(&self, x: usize, y: usize) -> Option<f64> {
        if x < self.width && y < self.height {
            Some(self.map_data[x][y].temperature)
        } else {
            None
        }
    }

    /// Get maximum temperature in map
    pub fn get_max_temperature(&self) -> f64 {
        let mut max_temp = f64::MIN;
        for y in 0..self.height {
            for x in 0..self.width {
                if self.map_data[x][y].temperature > max_temp {
                    max_temp = self.map_data[x][y].temperature;
                }
            }
        }
        max_temp
    }

    /// Get hotspot locations (top N)
    pub fn get_hotspots(&self, count: usize) -> Vec<(usize, usize, f64)> {
        let mut hotspots: Vec<(usize, usize, f64)> = Vec::new();
        
        for y in 0..self.height {
            for x in 0..self.width {
                let cell = &self.map_data[x][y];
                if cell.zone_type == ThermalZoneType::Hot || cell.zone_type == ThermalZoneType::Critical {
                    hotspots.push((x, y, cell.temperature));
                }
            }
        }

        // Sort by temperature descending
        hotspots.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(core::cmp::Ordering::Equal));
        
        // Return top N
        hotspots.into_iter().take(count).collect()
    }

    /// Check if map data is fresh
    pub fn is_data_fresh(&self, current_time_ms: u64) -> bool {
        current_time_ms - self.last_update_ms <= self.freshness_threshold_ms
    }

    /// Get overall cooling recommendation for the chip
    pub fn get_overall_recommendation(&self) -> CoolingRecommendation {
        let max_temp = self.get_max_temperature();
        
        if max_temp >= 95.0 {
            CoolingRecommendation::EmergencyShutdown
        } else if max_temp >= 85.0 {
            CoolingRecommendation::ThrottleClock
        } else if max_temp >= 75.0 {
            CoolingRecommendation::MigrateWorkload
        } else if max_temp >= 65.0 {
            CoolingRecommendation::IncreaseCooling(30)
        } else {
            CoolingRecommendation::None
        }
    }

    /// Get map dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Get physical dimensions
    pub fn physical_dimensions(&self) -> (f64, f64) {
        (self.physical_width_mm, self.physical_height_mm)
    }
}

/// Thread placement optimizer using cooling map
pub struct ThreadPlacementOptimizer {
    cooling_map: PredictiveCoolingMap,
    /// Preferred temperature margin below threshold
    preferred_margin: f64,
}

impl ThreadPlacementOptimizer {
    /// Create a new optimizer
    pub fn new(cooling_map: PredictiveCoolingMap, preferred_margin: f64) -> Self {
        Self {
            cooling_map,
            preferred_margin,
        }
    }

    /// Find optimal core for thread placement
    pub fn find_optimal_core(&self, thread: &ThreadState, core_states: &[CoreThermalState]) 
        -> Option<usize>
    {
        let mut best_core: Option<usize> = None;
        let mut best_score = f64::MIN;

        for (core_id, state) in core_states.iter().enumerate() {
            if !state.available {
                continue;
            }

            // Score based on predicted temperature and margin
            let score = self.preferred_margin - state.predicted_temp;
            
            if score > best_score {
                best_score = score;
                best_core = Some(core_id);
            }
        }

        best_core
    }

    /// Get reference to cooling map
    pub fn cooling_map(&self) -> &PredictiveCoolingMap {
        &self.cooling_map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_creation() {
        let map = PredictiveCoolingMap::new(64, 64, 20.0, 20.0);
        assert!(map.is_ok());
    }

    #[test]
    fn test_invalid_resolution() {
        let map = PredictiveCoolingMap::new(0, 64, 20.0, 20.0);
        assert_eq!(map.unwrap_err(), CoolingMapError::InvalidResolution);

        let map = PredictiveCoolingMap::new(200, 64, 20.0, 20.0);
        assert_eq!(map.unwrap_err(), CoolingMapError::InvalidResolution);
    }

    #[test]
    fn test_zone_classification() {
        let map = PredictiveCoolingMap::new(64, 64, 20.0, 20.0).unwrap();
        
        assert_eq!(map.classify_zone(50.0), ThermalZoneType::Normal);
        assert_eq!(map.classify_zone(70.0), ThermalZoneType::Warm);
        assert_eq!(map.classify_zone(80.0), ThermalZoneType::Hot);
        assert_eq!(map.classify_zone(90.0), ThermalZoneType::Critical);
    }

    #[test]
    fn test_recommendations() {
        let map = PredictiveCoolingMap::new(64, 64, 20.0, 20.0).unwrap();
        
        assert_eq!(map.generate_recommendation(50.0, 1.0), CoolingRecommendation::None);
        assert_eq!(map.generate_recommendation(72.0, 1.0), CoolingRecommendation::IncreaseCooling(10));
        assert_eq!(map.generate_recommendation(80.0, 15.0), CoolingRecommendation::MigrateWorkload);
        assert_eq!(map.generate_recommendation(90.0, 5.0), CoolingRecommendation::ThrottleClock);
        assert_eq!(map.generate_recommendation(100.0, 5.0), CoolingRecommendation::EmergencyShutdown);
    }
}
