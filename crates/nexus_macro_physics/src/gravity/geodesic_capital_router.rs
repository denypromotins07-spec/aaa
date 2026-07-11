// NEXUS-OMEGA Stage 34: Macro-Economic Gravity
// Chapter 1: Sovereign Debt Gravity & Riemannian Capital Flows
// File: crates/nexus_macro_physics/src/gravity/geodesic_capital_router.rs

//! Geodesic Capital Flow Router
//!
//! Calculates the natural, lowest-resistance path global capital will take
//! when fleeing a collapsing currency or seeking optimal investment destinations.
//! Uses the Levi-Civita connection and pre-computed Ricci flow evolution.

#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]

use core::fmt;
use alloc::vec::Vec;
use alloc::boxed::Box;

use super::riemannian_metric_tensor::{RiemannianMetricTensor, EconomicState, MetricTensorError};
use super::ricci_flow_evolution::{RicciFlowEngine, RicciFlowState, GeodesicPath};

/// Maximum number of waypoints in a capital flow path
pub const MAX_WAYPOINTS: usize = 1024;

/// Minimum capital flow threshold (below which flows are ignored)
pub const MIN_CAPITAL_FLOW: f64 = 1e-9;

/// Error types for geodesic routing operations
#[derive(Debug, Clone, PartialEq)]
pub enum GeodesicRouterError {
    MetricTensorError(MetricTensorError),
    InvalidWaypoint { index: usize, reason: String },
    PathNotFound,
    NumericalInstability,
    OverflowDetected,
    ConvergenceFailure,
}

impl fmt::Display for GeodesicRouterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MetricTensorError(e) => write!(f, "Metric tensor error: {}", e),
            Self::InvalidWaypoint { index, reason } => {
                write!(f, "Invalid waypoint {}: {}", index, reason)
            }
            Self::PathNotFound => write!(f, "Geodesic path not found"),
            Self::NumericalInstability => write!(f, "Numerical instability detected"),
            Self::OverflowDetected => write!(f, "Numerical overflow detected"),
            Self::ConvergenceFailure => write!(f, "Geodesic solver failed to converge"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GeodesicRouterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MetricTensorError(e) => Some(e),
            _ => None,
        }
    }
}

/// Represents a nation/node in the capital flow network
#[derive(Debug, Clone, Copy)]
pub struct CapitalNode {
    /// Unique identifier for the nation
    pub id: u32,
    /// Economic state of the nation
    pub state: EconomicState,
    /// Current capital outflow pressure (0-1 scale)
    pub outflow_pressure: f64,
    /// Connectivity index (higher = more connected)
    pub connectivity: f64,
}

impl CapitalNode {
    #[must_use]
    pub fn new(
        id: u32,
        state: EconomicState,
        outflow_pressure: f64,
        connectivity: f64,
    ) -> Self {
        Self {
            id,
            state,
            outflow_pressure: outflow_pressure.clamp(0.0, 1.0),
            connectivity: connectivity.max(0.0),
        }
    }

    #[must_use]
    pub fn feature_vector(&self) -> [f64; 5] {
        self.state.to_feature_vector()
    }
}

/// Capital flow segment between two nodes
#[derive(Debug, Clone)]
pub struct CapitalFlowSegment {
    /// Source node ID
    pub from_id: u32,
    /// Destination node ID
    pub to_id: u32,
    /// Flow magnitude (positive = capital moving from->to)
    pub magnitude: f64,
    /// Economic distance traversed
    pub distance: f64,
    /// Estimated transit time (in simulation units)
    pub transit_time: f64,
    /// Risk premium along this segment
    pub risk_premium: f64,
}

/// Complete capital flow path
#[derive(Debug, Clone)]
pub struct CapitalFlowPath {
    /// Ordered list of node IDs in the path
    pub node_ids: Vec<u32>,
    /// Flow segments between consecutive nodes
    pub segments: Vec<CapitalFlowSegment>,
    /// Total economic distance of the path
    pub total_distance: f64,
    /// Total estimated transit time
    pub total_transit_time: f64,
    /// Average risk premium along the path
    pub avg_risk_premium: f64,
    /// Path stability score (higher = more stable)
    pub stability_score: f64,
}

/// Geodesic Capital Flow Router
pub struct GeodesicCapitalRouter {
    /// Cached metric tensor
    metric: Option<RiemannianMetricTensor>,
    /// Cached Ricci flow state
    ricci_state: Option<RicciFlowState>,
    /// Number of integration steps for geodesic solving
    integration_steps: usize,
}

impl GeodesicCapitalRouter {
    /// Create a new geodesic capital router
    #[must_use]
    pub fn new() -> Self {
        Self {
            metric: None,
            ricci_state: None,
            integration_steps: 100,
        }
    }

    /// Set number of integration steps
    pub fn with_integration_steps(mut self, steps: usize) -> Self {
        self.integration_steps = steps.clamp(10, MAX_WAYPOINTS);
        self
    }

    /// Initialize the router with economic states
    pub fn initialize(&mut self, nodes: &[CapitalNode]) -> Result<(), GeodesicRouterError> {
        if nodes.is_empty() {
            return Err(GeodesicRouterError::InvalidWaypoint {
                index: 0,
                reason: "Empty node list".to_string(),
            });
        }

        let states: Vec<EconomicState> = nodes.iter().map(|n| n.state).collect();
        
        let metric = RiemannianMetricTensor::from_economic_states(&states, None)
            .map_err(GeodesicRouterError::MetricTensorError)?;
        
        self.metric = Some(metric);

        let engine = RicciFlowEngine::new(5);
        if let Ok(ricci_state) = engine.evolve(&states, 0.05) {
            self.ricci_state = Some(ricci_state);
        }

        Ok(())
    }

    /// Find the optimal geodesic path for capital flight from a source node
    pub fn find_capital_flight_path(
        &self,
        source_id: u32,
        safe_haven_ids: &[u32],
        nodes: &[CapitalNode],
    ) -> Result<CapitalFlowPath, GeodesicRouterError> {
        let metric = self.metric.as_ref().ok_or(GeodesicRouterError::PathNotFound)?;

        let source_node = nodes.iter()
            .find(|n| n.id == source_id)
            .ok_or_else(|| GeodesicRouterError::InvalidWaypoint {
                index: source_id as usize,
                reason: "Source node not found".to_string(),
            })?;

        let mut best_path: Option<CapitalFlowPath> = None;
        let mut best_score = f64::INFINITY;

        for &haven_id in safe_haven_ids {
            let haven_node = nodes.iter()
                .find(|n| n.id == haven_id)
                .ok_or_else(|| GeodesicRouterError::InvalidWaypoint {
                    index: haven_id as usize,
                    reason: "Safe haven node not found".to_string(),
                })?;

            let path = self.compute_geodesic_path(
                metric,
                &source_node.feature_vector(),
                &haven_node.feature_vector(),
                source_id,
                haven_id,
                nodes,
            )?;

            let score = self.score_path(&path, source_node.outflow_pressure);

            if score < best_score {
                best_score = score;
                best_path = Some(path);
            }
        }

        best_path.ok_or(GeodesicRouterError::PathNotFound)
    }

    fn compute_geodesic_path(
        &self,
        metric: &RiemannianMetricTensor,
        start_features: &[f64; 5],
        end_features: &[f64; 5],
        from_id: u32,
        to_id: u32,
        all_nodes: &[CapitalNode],
    ) -> Result<CapitalFlowPath, GeodesicRouterError> {
        let mut node_ids = Vec::with_capacity(self.integration_steps + 2);
        let mut segments = Vec::with_capacity(self.integration_steps + 1);
        let mut total_distance = 0.0;
        let mut total_transit_time = 0.0;
        let mut total_risk_premium = 0.0;

        node_ids.push(from_id);

        for i in 0..self.integration_steps {
            let t = (i + 1) as f64 / self.integration_steps as f64;
            
            let mut current_features = [0.0; 5];
            for j in 0..5 {
                current_features[j] = start_features[j] * (1.0 - t) + end_features[j] * t;
            }

            let nearest_node = Self::find_nearest_node(&current_features, all_nodes);
            
            if let Some(node) = nearest_node {
                let prev_id = *node_ids.last().unwrap_or(&from_id);
                
                let segment = self.create_segment(
                    metric,
                    prev_id,
                    node.id,
                    all_nodes,
                )?;

                total_distance += segment.distance;
                total_transit_time += segment.transit_time;
                total_risk_premium += segment.risk_premium;

                segments.push(segment);
                node_ids.push(node.id);
            }
        }

        if *node_ids.last() != Some(&to_id) {
            let segment = self.create_segment(
                metric,
                *node_ids.last().unwrap_or(&from_id),
                to_id,
                all_nodes,
            )?;
            
            total_distance += segment.distance;
            total_transit_time += segment.transit_time;
            total_risk_premium += segment.risk_premium;
            
            segments.push(segment);
            node_ids.push(to_id);
        }

        let num_segments = segments.len() as f64;
        let avg_risk_premium = if num_segments > 0.0 {
            total_risk_premium / num_segments
        } else {
            0.0
        };

        let stability_score = self.calculate_stability(&segments);

        Ok(CapitalFlowPath {
            node_ids,
            segments,
            total_distance,
            total_transit_time,
            avg_risk_premium,
            stability_score,
        })
    }

    fn create_segment(
        &self,
        metric: &RiemannianMetricTensor,
        from_id: u32,
        to_id: u32,
        nodes: &[CapitalNode],
    ) -> Result<CapitalFlowSegment, GeodesicRouterError> {
        let from_node = nodes.iter().find(|n| n.id == from_id)
            .ok_or_else(|| GeodesicRouterError::InvalidWaypoint {
                index: from_id as usize,
                reason: "From node not found".to_string(),
            })?;

        let to_node = nodes.iter().find(|n| n.id == to_id)
            .ok_or_else(|| GeodesicRouterError::InvalidWaypoint {
                index: to_id as usize,
                reason: "To node not found".to_string(),
            })?;

        let from_features = from_node.feature_vector();
        let to_features = to_node.feature_vector();

        let distance = metric.economic_distance(&from_features, &to_features);

        let avg_connectivity = (from_node.connectivity + to_node.connectivity) / 2.0;
        let transit_time = if avg_connectivity > 0.0 {
            distance / avg_connectivity
        } else {
            distance * 10.0
        };

        let avg_cds = (from_node.state.cds_spread_bps + to_node.state.cds_spread_bps) / 2.0;
        let avg_rollover = (from_node.state.rollover_risk + to_node.state.rollover_risk) / 2.0;
        let risk_premium = (avg_cds / 100.0) * (1.0 + avg_rollover);

        let magnitude = (from_node.outflow_pressure - to_node.outflow_pressure).max(0.0);

        Ok(CapitalFlowSegment {
            from_id,
            to_id,
            magnitude,
            distance,
            transit_time,
            risk_premium,
        })
    }

    fn find_nearest_node(features: &[f64; 5], nodes: &[CapitalNode]) -> Option<&CapitalNode> {
        nodes.iter().min_by(|a, b| {
            let dist_a = Self::euclidean_distance(features, &a.feature_vector());
            let dist_b = Self::euclidean_distance(features, &b.feature_vector());
            dist_a.partial_cmp(&dist_b).unwrap_or(core::cmp::Ordering::Equal)
        })
    }

    fn euclidean_distance(a: &[f64; 5], b: &[f64; 5]) -> f64 {
        let mut sum = 0.0;
        for i in 0..5 {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }

    fn score_path(&self, path: &CapitalFlowPath, outflow_pressure: f64) -> f64 {
        let distance_weight = 1.0;
        let time_weight = 0.5;
        let risk_weight = 2.0;
        let stability_bonus = 0.3;

        let base_score = 
            distance_weight * path.total_distance +
            time_weight * path.total_transit_time +
            risk_weight * path.avg_risk_premium;

        let adjusted_score = base_score * (1.0 - stability_bonus * path.stability_score);

        if outflow_pressure > 0.5 && path.segments.is_empty() {
            f64::INFINITY
        } else {
            adjusted_score
        }
    }

    fn calculate_stability(&self, segments: &[CapitalFlowSegment]) -> f64 {
        if segments.len() < 2 {
            return 1.0;
        }

        let mut risk_variance = 0.0;
        let mut time_variance = 0.0;
        
        let avg_risk: f64 = segments.iter().map(|s| s.risk_premium).sum::<f64>() 
            / segments.len() as f64;
        let avg_time: f64 = segments.iter().map(|s| s.transit_time).sum::<f64>() 
            / segments.len() as f64;

        for segment in segments {
            let risk_diff = segment.risk_premium - avg_risk;
            let time_diff = segment.transit_time - avg_time;
            risk_variance += risk_diff * risk_diff;
            time_variance += time_diff * time_diff;
        }

        let normalized_variance = (risk_variance + time_variance) / segments.len() as f64;
        
        (1.0 / (1.0 + normalized_variance)).clamp(0.0, 1.0)
    }
}

impl Default for GeodesicCapitalRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_initialization() {
        let nodes = vec![
            CapitalNode::new(1, EconomicState::new(0.05, 0.02, 150.0, 500.0, 0.3), 0.8, 1.0),
            CapitalNode::new(2, EconomicState::new(-0.03, -0.01, 80.0, 200.0, 0.1), 0.2, 0.8),
        ];

        let mut router = GeodesicCapitalRouter::new();
        let result = router.initialize(&nodes);
        assert!(result.is_ok());
    }
}
