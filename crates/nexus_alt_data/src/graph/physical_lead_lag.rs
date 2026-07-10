//! Physical Lead-Lag Alpha Calculator
//! 
//! Calculates time delays between physical supply chain disruptions
//! and resulting price movements in commodity futures.

use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use thiserror::Error;

/// Lead-lag calculation errors
#[derive(Debug, Error)]
pub enum LeadLagError {
    #[error("Insufficient data points")]
    InsufficientData,
    #[error("Invalid correlation: {0}")]
    InvalidCorrelation(String),
    #[error("Asset not found: {0}")]
    AssetNotFound(String),
}

/// Commodity asset types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommodityType {
    CrudeOil,
    NaturalGas,
    HeatingOil,
    Gasoline,
    Gold,
    Silver,
    Copper,
    Wheat,
    Corn,
    Soybeans,
    Coffee,
    Sugar,
}

/// Lead-lag relationship between physical event and price
#[derive(Debug, Clone)]
pub struct LeadLagRelationship {
    pub chokepoint: String,
    pub commodity: CommodityType,
    pub lag_hours: f64,
    pub correlation_strength: f64,
    pub confidence: f64,
    pub sample_size: usize,
    pub last_updated: SystemTime,
}

impl LeadLagRelationship {
    pub fn new(
        chokepoint: String,
        commodity: CommodityType,
        lag_hours: f64,
        correlation_strength: f64,
        confidence: f64,
        sample_size: usize,
    ) -> Result<Self, LeadLagError> {
        if correlation_strength < -1.0 || correlation_strength > 1.0 {
            return Err(LeadLagError::InvalidCorrelation(
                format!("Correlation {} out of range [-1, 1]", correlation_strength),
            ));
        }

        Ok(LeadLagRelationship {
            chokepoint,
            commodity,
            lag_hours,
            correlation_strength,
            confidence,
            sample_size,
            last_updated: SystemTime::now(),
        })
    }
}

/// Historical observation for lead-lag analysis
#[derive(Debug, Clone)]
pub struct Observation {
    pub timestamp: SystemTime,
    pub congestion_change: f64, // Change in congestion level
    pub price_change: f64,      // Price change percentage
    pub volume: f64,
}

/// Physical lead-lag alpha calculator
pub struct PhysicalLeadLagCalculator {
    /// Historical observations per (chokepoint, commodity) pair
    observations: HashMap<(String, CommodityType), Vec<Observation>>,
    /// Calculated lead-lag relationships
    relationships: HashMap<(String, CommodityType), LeadLagRelationship>,
    /// Maximum history size per pair
    max_history: usize,
    /// Minimum samples for statistical significance
    min_samples: usize,
}

impl PhysicalLeadLagCalculator {
    pub fn new(max_history: usize, min_samples: usize) -> Self {
        PhysicalLeadLagCalculator {
            observations: HashMap::new(),
            relationships: HashMap::new(),
            max_history,
            min_samples,
        }
    }

    /// Add an observation
    pub fn add_observation(
        &mut self,
        chokepoint: String,
        commodity: CommodityType,
        congestion_change: f64,
        price_change: f64,
        volume: f64,
    ) {
        let key = (chokepoint, commodity);
        
        let obs = Observation {
            timestamp: SystemTime::now(),
            congestion_change,
            price_change,
            volume,
        };
        
        let obs_list = self.observations.entry(key).or_insert_with(Vec::new);
        obs_list.push(obs);
        
        // Trim old observations
        while obs_list.len() > self.max_history {
            obs_list.remove(0);
        }
    }

    /// Calculate lead-lag relationship using cross-correlation
    pub fn calculate_lead_lag(
        &self,
        chokepoint: &str,
        commodity: CommodityType,
    ) -> Result<LeadLagRelationship, LeadLagError> {
        let key = (chokepoint.to_string(), commodity);
        
        let obs = self.observations.get(&key)
            .ok_or_else(|| LeadLagError::InsufficientData)?;
        
        if obs.len() < self.min_samples {
            return Err(LeadLagError::InsufficientData);
        }
        
        // Extract time series
        let congestion_series: Vec<f64> = obs.iter().map(|o| o.congestion_change).collect();
        let price_series: Vec<f64> = obs.iter().map(|o| o.price_change).collect();
        
        // Find optimal lag using cross-correlation
        let (best_lag, best_correlation) = 
            self.find_optimal_lag(&congestion_series, &price_series)?;
        
        // Calculate confidence based on sample size and correlation strength
        let confidence = self.calculate_confidence(obs.len(), best_correlation);
        
        LeadLagRelationship::new(
            chokepoint.to_string(),
            commodity,
            best_lag as f64, // Each observation represents 1 hour
            best_correlation,
            confidence,
            obs.len(),
        )
    }

    /// Find optimal lag using cross-correlation
    fn find_optimal_lag(
        &self,
        x: &[f64],
        y: &[f64],
    ) -> Result<(isize, f64), LeadLagError> {
        let n = x.len().min(y.len());
        if n < self.min_samples {
            return Err(LeadLagError::InsufficientData);
        }
        
        let mut best_lag: isize = 0;
        let mut best_corr = -2.0; // Correlations are >= -1
        
        // Test lags from -24 to +48 hours
        for lag in -24..=48 {
            let corr = self.cross_correlation(x, y, lag)?;
            
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }
        
        Ok((best_lag, best_corr))
    }

    /// Calculate cross-correlation at given lag
    fn cross_correlation(&self, x: &[f64], y: &[f64], lag: isize) -> Result<f64, LeadLagError> {
        let n = x.len();
        
        if lag >= n as isize || lag <= -(n as isize) {
            return Ok(0.0);
        }
        
        let (start_x, start_y, count) = if lag >= 0 {
            (lag as usize, 0, n - lag as usize)
        } else {
            (0, (-lag) as usize, n - (-lag) as usize)
        };
        
        if count < self.min_samples {
            return Ok(0.0);
        }
        
        // Calculate means
        let mean_x: f64 = (start_x..start_x + count).map(|i| x[i]).sum::<f64>() / count as f64;
        let mean_y: f64 = (start_y..start_y + count).map(|i| y[i]).sum::<f64>() / count as f64;
        
        // Calculate correlation
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        
        for i in 0..count {
            let dx = x[start_x + i] - mean_x;
            let dy = y[start_y + i] - mean_y;
            
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }
        
        let denominator = (sum_x2 * sum_y2).sqrt();
        
        if denominator < 1e-10 {
            return Ok(0.0);
        }
        
        Ok(sum_xy / denominator)
    }

    /// Calculate confidence score
    fn calculate_confidence(&self, sample_size: usize, correlation: f64) -> f64 {
        // Confidence increases with sample size and correlation strength
        let sample_factor = (sample_size as f64 / self.max_history as f64).min(1.0);
        let correlation_factor = correlation.abs();
        
        // Weighted combination
        0.4 * sample_factor + 0.6 * correlation_factor
    }

    /// Update relationship cache
    pub fn update_relationships(&mut self) -> Result<(), LeadLagError> {
        let keys: Vec<_> = self.observations.keys().cloned().collect();
        
        for (chokepoint, commodity) in keys {
            if let Ok(relationship) = self.calculate_lead_lag(&chokepoint, commodity) {
                self.relationships.insert((chokepoint, commodity), relationship);
            }
        }
        
        Ok(())
    }

    /// Get calculated relationship
    pub fn get_relationship(
        &self,
        chokepoint: &str,
        commodity: CommodityType,
    ) -> Option<&LeadLagRelationship> {
        self.relationships.get(&(chokepoint.to_string(), commodity))
    }

    /// Generate alpha signal based on current congestion and historical lag
    pub fn generate_alpha_signal(
        &self,
        chokepoint: &str,
        commodity: CommodityType,
        current_congestion_change: f64,
    ) -> Option<AlphaSignal> {
        let relationship = self.get_relationship(chokepoint, commodity)?;
        
        // Only generate signal if correlation is significant
        if relationship.correlation_strength.abs() < 0.3 {
            return None;
        }
        
        // Predict price movement based on congestion change and lag
        let predicted_price_change = current_congestion_change 
            * relationship.correlation_strength 
            * relationship.confidence;
        
        Some(AlphaSignal {
            chokepoint: chokepoint.to_string(),
            commodity,
            predicted_price_change,
            expected_lag_hours: relationship.lag_hours,
            confidence: relationship.confidence,
            timestamp: SystemTime::now(),
        })
    }
}

impl Default for PhysicalLeadLagCalculator {
    fn default() -> Self {
        Self::new(1000, 50)
    }
}

/// Alpha signal for trading
#[derive(Debug, Clone)]
pub struct AlphaSignal {
    pub chokepoint: String,
    pub commodity: CommodityType,
    pub predicted_price_change: f64,
    pub expected_lag_hours: f64,
    pub confidence: f64,
    pub timestamp: SystemTime,
}

impl AlphaSignal {
    /// Get signal strength (-1 to 1)
    pub fn signal_strength(&self) -> f64 {
        self.predicted_price_change.clamp(-1.0, 1.0)
    }
    
    /// Get recommended position direction
    pub fn recommended_direction(&self) -> PositionDirection {
        if self.predicted_price_change > 0.01 {
            PositionDirection::Long
        } else if self.predicted_price_change < -0.01 {
            PositionDirection::Short
        } else {
            PositionDirection::Neutral
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionDirection {
    Long,
    Short,
    Neutral,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator_creation() {
        let calc = PhysicalLeadLagCalculator::new(1000, 50);
        assert_eq!(calc.max_history, 1000);
        assert_eq!(calc.min_samples, 50);
    }

    #[test]
    fn test_cross_correlation() {
        let calc = PhysicalLeadLagCalculator::new(100, 10);
        
        // Perfect positive correlation at lag 0
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let corr = calc.cross_correlation(&x, &y, 0).unwrap();
        assert!((corr - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_insufficient_data() {
        let mut calc = PhysicalLeadLagCalculator::new(100, 50);
        
        // Add insufficient observations
        for _ in 0..10 {
            calc.add_observation(
                "Suez Canal".to_string(),
                CommodityType::CrudeOil,
                0.1,
                0.05,
                1000.0,
            );
        }
        
        let result = calc.calculate_lead_lag("Suez Canal", CommodityType::CrudeOil);
        assert!(result.is_err());
    }
}
