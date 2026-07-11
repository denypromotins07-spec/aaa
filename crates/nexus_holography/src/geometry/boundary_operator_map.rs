//! Boundary Operator Mapper
//! 
//! Maps L3 micro-events (fills, cancels) to CFT boundary operators.
//! Computes how micro-events warp the bulk geometry via holographic dictionary.

use crate::geometry::{AdsCftDictionary, BoundaryOperator, ConformalDimension, OperatorType};
use nalgebra::Vector2;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors specific to boundary operator mapping
#[derive(Error, Debug, Clone, PartialEq)]
pub enum BoundaryMapError {
    #[error("Invalid event type: {0}")]
    InvalidEventType(String),
    #[error("Event amplitude out of range: {0}")]
    AmplitudeOutOfRange(f64),
    #[error("Mapping failed: {0}")]
    MappingFailed(String),
}

/// Types of L3 market events
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum MarketEventType {
    /// Limit order placement
    LimitAdd,
    /// Order execution/fill
    Fill,
    /// Order cancellation
    Cancel,
    /// Order modification
    Modify,
    /// Trade initiation (aggressor)
    TradeInitiate,
}

/// L3 market event on the tape
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketEvent {
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Price level
    pub price: f64,
    /// Volume/size
    pub volume: f64,
    /// Event type
    pub event_type: MarketEventType,
    /// Exchange/venue identifier
    pub venue_id: u32,
}

/// Mapped boundary operator with bulk influence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappedOperator {
    /// Original market event
    pub event: MarketEvent,
    /// Corresponding CFT boundary operator
    pub boundary_operator: BoundaryOperator,
    /// Computed bulk field perturbation at reference depth
    pub bulk_perturbation: f64,
    /// Gravitational pull metric (how much it warps the bulk)
    pub gravitational_pull: f64,
}

/// Boundary operator mapper for market events
pub struct BoundaryOperatorMapper {
    /// Holographic dictionary
    dictionary: AdsCftDictionary,
    /// Reference depth for bulk perturbation calculation
    reference_depth: f64,
    /// Mapping from event types to operator types
    event_to_operator: std::collections::HashMap<MarketEventType, OperatorType>,
}

impl BoundaryOperatorMapper {
    /// Create a new boundary operator mapper
    pub fn new(
        dictionary: AdsCftDictionary,
        reference_depth: f64,
    ) -> Result<Self, BoundaryMapError> {
        if reference_depth <= 0.0 || reference_depth < dictionary.uv_cutoff {
            return Err(BoundaryMapError::MappingFailed(
                "Invalid reference depth".to_string(),
            ));
        }

        // Build default event-to-operator mapping
        let mut event_to_operator = std::collections::HashMap::new();
        event_to_operator.insert(MarketEventType::LimitAdd, OperatorType::Scalar);
        event_to_operator.insert(MarketEventType::Fill, OperatorType::StressTensor);
        event_to_operator.insert(MarketEventType::Cancel, OperatorType::Current);
        event_to_operator.insert(MarketEventType::Modify, OperatorType::Scalar);
        event_to_operator.insert(MarketEventType::TradeInitiate, OperatorType::StressTensor);

        Ok(Self {
            dictionary,
            reference_depth,
            event_to_operator,
        })
    }

    /// Map a single market event to a boundary operator
    pub fn map_event(&self, event: &MarketEvent) -> Result<MappedOperator, BoundaryMapError> {
        // Validate event
        if event.volume <= 0.0 {
            return Err(BoundaryMapError::AmplitudeOutOfRange(event.volume));
        }
        if event.price <= 0.0 {
            return Err(BoundaryMapError::AmplitudeOutOfRange(event.price));
        }

        // Determine operator type from event
        let op_type = self.event_to_operator.get(&event.event_type)
            .copied()
            .unwrap_or(OperatorType::Scalar);

        // Compute conformal dimension based on event characteristics
        // Larger volume → higher dimension (more relevant operator)
        let log_volume = event.volume.ln().max(0.0);
        let delta = 0.5 + 0.1 * log_volume.min(10.0); // Bounded dimension

        let dimension = ConformalDimension::new(delta)
            .map_err(|e| BoundaryMapError::MappingFailed(e.to_string()))?;

        // Create boundary operator
        let boundary_op = BoundaryOperator {
            position: event.price,
            dimension,
            operator_type: op_type,
            amplitude: event.volume,
        };

        // Map to bulk and compute perturbation
        let bulk_field = self.dictionary.map_to_bulk(&boundary_op, self.reference_depth)
            .map_err(|e| BoundaryMapError::MappingFailed(e.to_string()))?;

        // Gravitational pull ~ mass × amplitude / distance²
        let gravitational_pull = if bulk_field.mass_squared > 0.0 {
            bulk_field.mass_squared.abs().sqrt() * boundary_op.amplitude / self.reference_depth.powi(2)
        } else {
            boundary_op.amplitude / self.reference_depth
        };

        Ok(MappedOperator {
            event: event.clone(),
            boundary_operator: boundary_op,
            bulk_perturbation: bulk_field.field_value,
            gravitational_pull,
        })
    }

    /// Map multiple events and compute cumulative bulk warp
    pub fn map_events_batch(&self, events: &[MarketEvent]) -> Result<Vec<MappedOperator>, BoundaryMapError> {
        events.iter().map(|e| self.map_event(e)).collect()
    }

    /// Compute total bulk geometry distortion from a sequence of events
    pub fn compute_bulk_warp(&self, events: &[MarketEvent]) -> Result<f64, BoundaryMapError> {
        let mapped = self.map_events_batch(events)?;
        
        // Sum of gravitational pulls (superposition principle)
        let total_warp: f64 = mapped.iter().map(|m| m.gravitational_pull).sum();
        
        Ok(total_warp)
    }

    /// Detect hidden dark pool activity by analyzing bulk warp anomalies
    /// If bulk warp exceeds sum of boundary events, there's hidden liquidity
    pub fn detect_hidden_liquidity(
        &self,
        events: &[MarketEvent],
        observed_bulk_response: f64,
    ) -> Result<HiddenLiquiditySignal, BoundaryMapError> {
        let expected_warp = self.compute_bulk_warp(events)?;
        
        let excess_warp = observed_bulk_response - expected_warp;
        
        // If excess is significant, infer hidden liquidity
        let threshold = expected_warp * 0.1; // 10% tolerance
        
        let hidden_detected = excess_warp > threshold;
        
        // Estimate hidden volume from excess warp
        let estimated_hidden_volume = if hidden_detected {
            excess_warp * self.reference_depth.powi(2) / 0.1 // Inverse of gravitational formula
        } else {
            0.0
        };

        Ok(HiddenLiquiditySignal {
            expected_warp,
            observed_warp: observed_bulk_response,
            excess_warp,
            hidden_detected,
            estimated_hidden_volume,
            confidence: if hidden_detected {
                (excess_warp / threshold).min(1.0)
            } else {
                0.0
            },
        })
    }
}

/// Signal indicating hidden liquidity detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenLiquiditySignal {
    /// Expected warp from visible events
    pub expected_warp: f64,
    /// Actually observed bulk response
    pub observed_warp: f64,
    /// Excess warp suggesting hidden activity
    pub excess_warp: f64,
    /// Whether hidden liquidity was detected
    pub hidden_detected: bool,
    /// Estimated volume of hidden orders
    pub estimated_hidden_volume: f64,
    /// Confidence score [0, 1]
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::AdsCftDictionary;

    #[test]
    fn test_mapper_creation() {
        let dict = AdsCftDictionary::new(1.0, 0.01, 10.0).unwrap();
        let mapper = BoundaryOperatorMapper::new(dict, 0.5);
        assert!(mapper.is_ok());
    }

    #[test]
    fn test_event_mapping() {
        let dict = AdsCftDictionary::new(1.0, 0.01, 10.0).unwrap();
        let mapper = BoundaryOperatorMapper::new(dict, 0.5).unwrap();
        
        let event = MarketEvent {
            timestamp_ns: 1000000,
            price: 100.0,
            volume: 1000.0,
            event_type: MarketEventType::Fill,
            venue_id: 1,
        };
        
        let mapped = mapper.map_event(&event);
        assert!(mapped.is_ok());
        let m = mapped.unwrap();
        assert!(m.gravitational_pull > 0.0);
    }

    #[test]
    fn test_invalid_event() {
        let dict = AdsCftDictionary::new(1.0, 0.01, 10.0).unwrap();
        let mapper = BoundaryOperatorMapper::new(dict, 0.5).unwrap();
        
        let event = MarketEvent {
            timestamp_ns: 1000000,
            price: 100.0,
            volume: -1.0, // Invalid
            event_type: MarketEventType::Fill,
            venue_id: 1,
        };
        
        assert!(mapper.map_event(&event).is_err());
    }
}
