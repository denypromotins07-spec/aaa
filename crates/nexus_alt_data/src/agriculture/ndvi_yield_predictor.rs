//! NDVI Yield Predictor for Agricultural Commodities
//! 
//! Uses multi-spectral satellite vegetation indices (NDVI, EVI)
//! to predict crop yields for wheat, corn, and soybeans.

use std::collections::HashMap;
use std::time::SystemTime;
use thiserror::Error;

/// Yield prediction errors
#[derive(Debug, Error)]
pub enum YieldError {
    #[error("Invalid NDVI value: {0}")]
    InvalidNdvi(String),
    #[error("Unknown crop type: {0}")]
    UnknownCropType(String),
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
}

/// Crop types supported by the predictor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CropType {
    Wheat,
    Corn,
    Soybeans,
    Rice,
    Cotton,
    Barley,
}

impl CropType {
    pub fn growing_season_days(&self) -> u32 {
        match self {
            CropType::Wheat => 110,
            CropType::Corn => 120,
            CropType::Soybeans => 100,
            CropType::Rice => 150,
            CropType::Cotton => 180,
            CropType::Barley => 90,
        }
    }
}

/// NDVI observation from satellite imagery
#[derive(Debug, Clone)]
pub struct NdviObservation {
    pub latitude: f64,
    pub longitude: f64,
    pub ndvi_value: f64,      // Normalized Difference Vegetation Index [-1, 1]
    pub evi_value: f64,       // Enhanced Vegetation Index
    pub lai_value: f64,       // Leaf Area Index
    pub cloud_cover: f64,     // Cloud cover percentage [0, 100]
    pub timestamp: SystemTime,
    pub resolution_meters: u32,
}

impl NdviObservation {
    pub fn new(
        latitude: f64,
        longitude: f64,
        ndvi_value: f64,
        evi_value: f64,
        lai_value: f64,
        cloud_cover: f64,
        resolution_meters: u32,
    ) -> Result<Self, YieldError> {
        if ndvi_value < -1.0 || ndvi_value > 1.0 {
            return Err(YieldError::InvalidNdvi(
                format!("NDVI {} out of range [-1, 1]", ndvi_value),
            ));
        }
        
        if cloud_cover < 0.0 || cloud_cover > 100.0 {
            return Err(YieldError::InvalidNdvi(
                "Cloud cover must be between 0 and 100".to_string(),
            ));
        }
        
        Ok(NdviObservation {
            latitude,
            longitude,
            ndvi_value,
            evi_value,
            lai_value,
            cloud_cover,
            timestamp: SystemTime::now(),
            resolution_meters,
        })
    }

    /// Check if observation is usable (low cloud cover, valid NDVI)
    pub fn is_usable(&self) -> bool {
        self.cloud_cover < 20.0 && self.ndvi_value > 0.1
    }
}

/// Crop growth stage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowthStage {
    Planting,
    Emergence,
    Vegetative,
    Reproductive,
    Maturity,
    Harvest,
}

impl GrowthStage {
    pub fn from_ndvi(ndvi: f64, days_since_planting: u32, crop: CropType) -> Self {
        let season_length = crop.growing_season_days();
        let progress = days_since_planting as f64 / season_length as f64;
        
        if progress < 0.1 {
            GrowthStage::Planting
        } else if progress < 0.2 {
            GrowthStage::Emergence
        } else if progress < 0.5 {
            GrowthStage::Vegetative
        } else if progress < 0.8 {
            GrowthStage::Reproductive
        } else if progress < 0.95 {
            GrowthStage::Maturity
        } else {
            GrowthStage::Harvest
        }
    }
}

/// Field-level yield estimate
#[derive(Debug, Clone)]
pub struct FieldYieldEstimate {
    pub field_id: String,
    pub crop_type: CropType,
    pub area_hectares: f64,
    pub predicted_yield_tonnes_per_hectare: f64,
    pub total_predicted_yield_tonnes: f64,
    pub confidence: f64,
    pub growth_stage: GrowthStage,
    pub days_to_harvest: u32,
    pub last_updated: SystemTime,
}

/// Regional yield aggregation
#[derive(Debug, Clone)]
pub struct RegionalYieldForecast {
    pub region_name: String,
    pub crop_type: CropType,
    pub total_area_hectares: f64,
    pub avg_yield_tonnes_per_hectare: f64,
    pub total_production_tonnes: f64,
    pub year_over_year_change_pct: f64,
    pub historical_avg_yield: f64,
    pub confidence: f64,
    pub forecast_date: SystemTime,
}

/// NDVI-based yield predictor
pub struct NdviYieldPredictor {
    /// Historical yield data by region and crop
    historical_yields: HashMap<(String, CropType), Vec<f64>>,
    /// NDVI time series by field
    field_observations: HashMap<String, Vec<NdviObservation>>,
    /// Crop-specific model parameters
    model_params: HashMap<CropType, CropModelParams>,
}

/// Crop-specific model parameters
#[derive(Debug, Clone)]
struct CropModelParams {
    /// Peak NDVI value for maximum yield
    peak_ndvi: f64,
    /// NDVI integral threshold for maturity
    ndvi_integral_threshold: f64,
    /// Base yield (tonnes/hectare) at minimum viable NDVI
    base_yield: f64,
    /// Maximum potential yield (tonnes/hectare)
    max_yield: f64,
    /// Sensitivity to early-season NDVI
    early_season_weight: f64,
    /// Sensitivity to mid-season NDVI
    mid_season_weight: f64,
    /// Sensitivity to late-season NDVI
    late_season_weight: f64,
}

impl NdviYieldPredictor {
    pub fn new() -> Self {
        let mut model_params = HashMap::new();
        
        // Corn parameters
        model_params.insert(CropType::Corn, CropModelParams {
            peak_ndvi: 0.85,
            ndvi_integral_threshold: 50.0,
            base_yield: 2.0,
            max_yield: 12.0,
            early_season_weight: 0.2,
            mid_season_weight: 0.5,
            late_season_weight: 0.3,
        });
        
        // Soybeans parameters
        model_params.insert(CropType::Soybeans, CropModelParams {
            peak_ndvi: 0.75,
            ndvi_integral_threshold: 40.0,
            base_yield: 1.5,
            max_yield: 4.5,
            early_season_weight: 0.15,
            mid_season_weight: 0.55,
            late_season_weight: 0.3,
        });
        
        // Wheat parameters
        model_params.insert(CropType::Wheat, CropModelParams {
            peak_ndvi: 0.70,
            ndvi_integral_threshold: 35.0,
            base_yield: 1.0,
            max_yield: 8.0,
            early_season_weight: 0.25,
            mid_season_weight: 0.45,
            late_season_weight: 0.3,
        });
        
        NdviYieldPredictor {
            historical_yields: HashMap::new(),
            field_observations: HashMap::new(),
            model_params,
        }
    }

    /// Add NDVI observation for a field
    pub fn add_observation(&mut self, field_id: String, obs: NdviObservation) {
        if !obs.is_usable() {
            return; // Skip cloudy or invalid observations
        }
        
        let obs_list = self.field_observations.entry(field_id).or_insert_with(Vec::new);
        obs_list.push(obs);
        
        // Sort by timestamp
        obs_list.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    }

    /// Add historical yield data for a region
    pub fn add_historical_yield(&mut self, region: String, crop: CropType, yield_tph: f64) {
        let yields = self.historical_yields.entry((region, crop)).or_insert_with(Vec::new);
        yields.push(yield_tph);
        
        // Keep last 10 years of data
        while yields.len() > 10 {
            yields.remove(0);
        }
    }

    /// Predict yield for a specific field
    pub fn predict_field_yield(
        &self,
        field_id: &str,
        crop_type: CropType,
        area_hectares: f64,
        planting_date: SystemTime,
    ) -> Result<FieldYieldEstimate, YieldError> {
        let observations = self.field_observations.get(field_id)
            .ok_or_else(|| YieldError::InsufficientData(
                format!("No observations for field {}", field_id),
            ))?;
        
        if observations.is_empty() {
            return Err(YieldError::InsufficientData("No usable observations".to_string()));
        }
        
        let params = self.model_params.get(&crop_type)
            .ok_or_else(|| YieldError::UnknownCropType(format!("{:?}", crop_type)))?;
        
        // Calculate days since planting for each observation
        let now = SystemTime::now();
        let days_since_planting = now.duration_since(planting_date)
            .map(|d| d.as_secs() / 86400)
            .unwrap_or(0) as u32;
        
        // Weight NDVI values by growth stage importance
        let mut weighted_ndvi_sum = 0.0;
        let mut weight_sum = 0.0;
        
        for obs in observations {
            let obs_days = obs.timestamp.duration_since(planting_date)
                .map(|d| d.as_secs() / 86400)
                .unwrap_or(0) as u32;
            
            let weight = self.calculate_stage_weight(obs_days, crop_type, params);
            weighted_ndvi_sum += obs.ndvi_value * weight;
            weight_sum += weight;
        }
        
        let avg_weighted_ndvi = if weight_sum > 0.0 {
            weighted_ndvi_sum / weight_sum
        } else {
            0.0
        };
        
        // Convert NDVI to yield estimate
        let yield_estimate = self.ndvi_to_yield(avg_weighted_ndvi, params, days_since_planting, crop_type);
        
        // Calculate confidence based on number of observations and their quality
        let confidence = self.calculate_confidence(observations, days_since_planting, crop_type);
        
        let growth_stage = GrowthStage::from_ndvi(avg_weighted_ndvi, days_since_planting, crop_type);
        let days_to_harvest = crop_type.growing_season_days().saturating_sub(days_since_planting);
        
        Ok(FieldYieldEstimate {
            field_id: field_id.to_string(),
            crop_type,
            area_hectares,
            predicted_yield_tonnes_per_hectare: yield_estimate,
            total_predicted_yield_tonnes: yield_estimate * area_hectares,
            confidence,
            growth_stage,
            days_to_harvest,
            last_updated: now,
        })
    }

    /// Calculate weight based on growth stage
    fn calculate_stage_weight(&self, days: u32, crop: CropType, params: &CropModelParams) -> f64 {
        let season_length = crop.growing_season_days();
        let progress = days as f64 / season_length as f64;
        
        if progress < 0.33 {
            params.early_season_weight
        } else if progress < 0.66 {
            params.mid_season_weight
        } else {
            params.late_season_weight
        }
    }

    /// Convert NDVI to yield estimate
    fn ndvi_to_yield(&self, ndvi: f64, params: &CropModelParams, days: u32, crop: CropType) -> f64 {
        // Normalize NDVI relative to crop-specific peak
        let normalized_ndvi = (ndvi / params.peak_ndvi).clamp(0.0, 1.0);
        
        // Apply sigmoid transformation for realistic yield curve
        let yield_factor = 1.0 / (1.0 + (-10.0 * (normalized_ndvi - 0.5)).exp());
        
        // Scale to yield range
        let base_yield = params.base_yield + (params.max_yield - params.base_yield) * yield_factor;
        
        // Adjust for days into season (yield certainty increases with time)
        let season_progress = (days as f64 / crop.growing_season_days() as f64).clamp(0.0, 1.0);
        let maturity_factor = 0.5 + 0.5 * season_progress;
        
        base_yield * maturity_factor
    }

    /// Calculate prediction confidence
    fn calculate_confidence(&self, observations: &[NdviObservation], days: u32, crop: CropType) -> f64 {
        let mut confidence = 0.0;
        
        // More observations = higher confidence
        let obs_factor = (observations.len() as f64 / 10.0).min(1.0);
        confidence += 0.3 * obs_factor;
        
        // Later in season = higher confidence
        let season_factor = (days as f64 / crop.growing_season_days() as f64).min(1.0);
        confidence += 0.4 * season_factor;
        
        // Low cloud cover = higher confidence
        let avg_cloud_cover: f64 = observations.iter().map(|o| o.cloud_cover).sum::<f64>() 
            / observations.len() as f64;
        let cloud_factor = 1.0 - (avg_cloud_cover / 100.0);
        confidence += 0.3 * cloud_factor;
        
        confidence.clamp(0.0, 1.0)
    }

    /// Aggregate field predictions into regional forecast
    pub fn create_regional_forecast(
        &self,
        region: String,
        crop_type: CropType,
        field_estimates: &[FieldYieldEstimate],
    ) -> Result<RegionalYieldForecast, YieldError> {
        if field_estimates.is_empty() {
            return Err(YieldError::InsufficientData(
                "No field estimates for region".to_string(),
            ));
        }
        
        let total_area: f64 = field_estimates.iter().map(|f| f.area_hectares).sum();
        let total_production: f64 = field_estimates.iter().map(|f| f.total_predicted_yield_tonnes).sum();
        let avg_yield = if total_area > 0.0 {
            total_production / total_area
        } else {
            0.0
        };
        
        // Calculate historical average
        let historical_avg = self.historical_yields.get(&(region.clone(), crop_type))
            .and_then(|yields| {
                if yields.is_empty() { None } else {
                    Some(yields.iter().sum::<f64>() / yields.len() as f64)
                }
            })
            .unwrap_or(avg_yield);
        
        // Year-over-year change
        let yoy_change = if historical_avg > 0.0 {
            (avg_yield - historical_avg) / historical_avg * 100.0
        } else {
            0.0
        };
        
        // Average confidence
        let avg_confidence: f64 = field_estimates.iter().map(|f| f.confidence).sum::<f64>() 
            / field_estimates.len() as f64;
        
        Ok(RegionalYieldForecast {
            region_name: region,
            crop_type,
            total_area_hectares: total_area,
            avg_yield_tonnes_per_hectare: avg_yield,
            total_production_tonnes: total_production,
            year_over_year_change_pct: yoy_change,
            historical_avg_yield: historical_avg,
            confidence: avg_confidence,
            forecast_date: SystemTime::now(),
        })
    }
}

impl Default for NdviYieldPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndvi_observation_validation() {
        let obs = NdviObservation::new(
            40.0, -90.0, 0.7, 0.5, 3.0, 5.0, 10
        ).unwrap();
        assert!(obs.is_usable());
        
        let cloudy_obs = NdviObservation::new(
            40.0, -90.0, 0.7, 0.5, 3.0, 50.0, 10
        ).unwrap();
        assert!(!cloudy_obs.is_usable());
    }

    #[test]
    fn test_growth_stage_determination() {
        let stage = GrowthStage::from_ndvi(0.5, 30, CropType::Corn);
        assert_eq!(stage, GrowthStage::Emergence);
        
        let stage = GrowthStage::from_ndvi(0.7, 60, CropType::Corn);
        assert_eq!(stage, GrowthStage::Vegetative);
    }

    #[test]
    fn test_yield_predictor_creation() {
        let predictor = NdviYieldPredictor::new();
        assert!(predictor.model_params.contains_key(&CropType::Corn));
        assert!(predictor.model_params.contains_key(&CropType::Soybeans));
        assert!(predictor.model_params.contains_key(&CropType::Wheat));
    }
}
