//! Ant Agent implementation for Ant Colony System (ACS) Smart Order Routing.
//! 
//! Each Ant Agent represents a meta-order execution task that explores
//! execution venues and deposits pheromones based on performance.

use crate::aco::stigmergic_routing_table::{VenueEdge, PheromoneValue};
use crate::aco::probabilistic_venue_selector::VenueSelectionProb;
use nexus_types::order::{MetaOrderId, VenueId, ExecutionQuality};
use nexus_types::market::MarketSnapshot;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Instant, Duration};

/// Unique identifier for an ant agent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AntAgentId(pub u64);

impl AntAgentId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// State of an ant agent during its tour
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntState {
    /// Ant is searching for the next venue to visit
    Searching,
    /// Ant is executing at a venue
    Executing,
    /// Ant has completed its tour and is returning pheromone info
    Completed,
    /// Ant failed to find a valid path
    Failed,
}

/// Local pheromone update applied by ants during tour construction
#[derive(Debug, Clone, Copy)]
pub struct LocalPheromoneUpdate {
    pub from_venue: VenueId,
    pub to_venue: VenueId,
    pub delta: f64, // Negative for local evaporation
    pub timestamp_ns: u64,
}

/// Global pheromone update applied by the best ant of the epoch
#[derive(Debug, Clone)]
pub struct GlobalPheromoneUpdate {
    pub ant_id: AntAgentId,
    pub path: Vec<VenueId>,
    pub quality_score: f64, // Higher is better (low slippage, high fill rate)
    pub timestamp_ns: u64,
}

/// Ant Agent for exploring execution venues
pub struct AntAgent {
    pub id: AntAgentId,
    pub meta_order_id: MetaOrderId,
    pub state: AntState,
    pub current_venue: Option<VenueId>,
    pub visited_venues: Vec<VenueId>,
    pub path_quality: f64,
    pub creation_time: Instant,
    pub last_action_time: Instant,
    pub max_tour_duration: Duration,
    /// Local pheromone updates to be applied
    pub pending_local_updates: Vec<LocalPheromoneUpdate>,
    /// Whether this ant has deposited global pheromone
    pub has_deposited_global: AtomicBool,
}

impl AntAgent {
    pub fn new(id: AntAgentId, meta_order_id: MetaOrderId, max_tour_ms: u64) -> Self {
        let now = Instant::now();
        Self {
            id,
            meta_order_id,
            state: AntState::Searching,
            current_venue: None,
            visited_venues: Vec::with_capacity(16), // Pre-allocate for typical venue count
            path_quality: 0.0,
            creation_time: now,
            last_action_time: now,
            max_tour_duration: Duration::from_millis(max_tour_ms),
            pending_local_updates: Vec::with_capacity(8),
            has_deposited_global: AtomicBool::new(false),
        }
    }

    /// Select next venue based on pheromone levels and heuristic visibility
    /// 
    /// Uses the ACS pseudo-random proportional rule:
    /// - With probability q0: exploit best known edge
    /// - With probability 1-q0: explore using roulette wheel selection
    pub fn select_next_venue(
        &self,
        routing_table: &crate::aco::stigmergic_routing_table::StigmergicRoutingTable,
        market_snapshot: &MarketSnapshot,
        exploitation_prob: f64,
    ) -> Result<Option<VenueId>, AntSelectionError> {
        if self.state != AntState::Searching {
            return Err(AntSelectionError::InvalidState(self.state));
        }

        let candidate_venues = routing_table.get_candidate_venues(self.current_venue)?;
        
        if candidate_venues.is_empty() {
            return Ok(None);
        }

        // Filter out already visited venues (no cycles in single tour)
        let available: Vec<VenueId> = candidate_venues
            .iter()
            .copied()
            .filter(|v| !self.visited_venues.contains(v))
            .collect();

        if available.is_empty() {
            return Ok(None);
        }

        // Get pheromone and heuristic values for available venues
        let mut scores: Vec<(VenueId, f64)> = Vec::with_capacity(available.len());
        
        for venue_id in &available {
            let edge = routing_table.get_edge(self.current_venue, Some(*venue_id))?;
            let pheromone = edge.pheromone.normalized();
            
            // Heuristic visibility: inverse of expected cost (spread + latency)
            let visibility = self.calculate_visibility(venue_id, market_snapshot, routing_table)?;
            
            // ACS transition rule parameters
            let alpha = 1.0; // Pheromone importance
            let beta = 2.0;  // Heuristic importance
            
            let score = pheromone.powf(alpha) * visibility.powf(beta);
            scores.push((*venue_id, score));
        }

        // Pseudo-random proportional rule
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let q: f64 = rng.gen();

        if q < exploitation_prob {
            // Exploitation: choose best edge deterministically
            let best = scores
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(id, _)| *id);
            Ok(best)
        } else {
            // Exploration: roulette wheel selection
            let selector = crate::aco::probabilistic_venue_selector::VenueSelector::new(&scores)?;
            Ok(Some(selector.select(&mut rng)))
        }
    }

    /// Calculate heuristic visibility for a venue
    fn calculate_visibility(
        &self,
        venue_id: &VenueId,
        market_snapshot: &MarketSnapshot,
        routing_table: &crate::aco::stigmergic_routing_table::StigmergicRoutingTable,
    ) -> Result<f64, AntSelectionError> {
        let venue_info = market_snapshot.get_venue_info(*venue_id)
            .ok_or(AntSelectionError::VenueNotFound(*venue_id))?;

        // Visibility = 1 / (expected_cost)
        // Expected cost combines spread, latency, and historical slippage
        let spread_cost = venue_info.spread_bps.max(0.01); // Minimum spread to avoid div by zero
        let latency_cost = routing_table.get_latency(*venue_id) as f64 / 1000.0; // ns to μs
        let slippage_estimate = venue_info.recent_slippage_bps.max(0.01);

        let total_cost = spread_cost + slippage_estimate + (latency_cost * 0.001);
        
        // Inverse relationship: lower cost = higher visibility
        Ok(1.0 / total_cost.max(0.001))
    }

    /// Move ant to a new venue and apply local pheromone update
    pub fn move_to_venue(
        &mut self,
        venue_id: VenueId,
        routing_table: &mut crate::aco::stigmergic_routing_table::StigmergicRoutingTable,
        local_evaporation_rate: f64,
        initial_pheromone: f64,
    ) -> Result<(), AntMovementError> {
        if self.state == AntState::Failed {
            return Err(AntMovementError::AntFailed);
        }

        let now = Instant::now();
        
        // Check tour duration limit
        if now.duration_since(self.creation_time) > self.max_tour_duration {
            self.state = AntState::Failed;
            return Err(AntMovementError::TourTimeout);
        }

        // Apply local pheromone update to the edge we're traversing
        if let Some(current) = self.current_venue {
            let delta = -local_evaporation_rate * initial_pheromone;
            self.pending_local_updates.push(LocalPheromoneUpdate {
                from_venue: current,
                to_venue: venue_id,
                delta,
                timestamp_ns: now.elapsed().as_nanos() as u64,
            });

            // Apply immediate local update to discourage crowding
            routing_table.apply_local_pheromone_update(current, venue_id, delta)?;
        }

        self.current_venue = Some(venue_id);
        self.visited_venues.push(venue_id);
        self.last_action_time = now;
        self.state = AntState::Executing;

        Ok(())
    }

    /// Record execution result and update path quality
    pub fn record_execution(
        &mut self,
        quality: ExecutionQuality,
    ) -> Result<(), AntExecutionError> {
        if self.state != AntState::Executing {
            return Err(AntExecutionError::InvalidState(self.state));
        }

        // Update cumulative path quality
        // Quality factors: fill_rate (higher better), slippage (lower better), latency (lower better)
        let execution_score = quality.fill_rate * (1.0 - quality.slippage_bps / 100.0);
        
        // Exponential moving average of quality
        let alpha = 0.3;
        self.path_quality = self.path_quality * (1.0 - alpha) + execution_score * alpha;

        self.state = AntState::Searching;
        Ok(())
    }

    /// Complete the ant's tour and prepare global pheromone update
    pub fn complete_tour(&mut self) -> Result<GlobalPheromoneUpdate, AntTourError> {
        if self.visited_venues.is_empty() {
            return Err(AntTourError::EmptyPath);
        }

        self.state = AntState::Completed;

        Ok(GlobalPheromoneUpdate {
            ant_id: self.id,
            path: self.visited_venues.clone(),
            quality_score: self.path_quality,
            timestamp_ns: self.last_action_time.elapsed().as_nanos() as u64,
        })
    }

    /// Reset ant for reuse (object pooling to avoid allocations)
    pub fn reset(&mut self, new_meta_order_id: MetaOrderId) {
        self.meta_order_id = new_meta_order_id;
        self.state = AntState::Searching;
        self.current_venue = None;
        self.visited_venues.clear();
        self.path_quality = 0.0;
        self.creation_time = Instant::now();
        self.last_action_time = Instant::now();
        self.pending_local_updates.clear();
        self.has_deposited_global.store(false, Ordering::Relaxed);
    }

    /// Check if ant tour has timed out
    pub fn is_timed_out(&self) -> bool {
        Instant::now().duration_since(self.creation_time) > self.max_tour_duration
    }
}

/// Errors for ant selection operations
#[derive(Debug, thiserror::Error)]
pub enum AntSelectionError {
    #[error("Invalid ant state: {0:?}")]
    InvalidState(AntState),
    #[error("Venue not found: {0:?}")]
    VenueNotFound(VenueId),
    #[error("Routing table error: {0}")]
    RoutingError(#[from] crate::aco::stigmergic_routing_table::RoutingTableError),
}

/// Errors for ant movement operations
#[derive(Debug, thiserror::Error)]
pub enum AntMovementError {
    #[error("Ant has failed")]
    AntFailed,
    #[error("Tour timeout exceeded")]
    TourTimeout,
    #[error("Routing table error: {0}")]
    RoutingError(#[from] crate::aco::stigmergic_routing_table::RoutingTableError),
}

/// Errors for ant execution operations
#[derive(Debug, thiserror::Error)]
pub enum AntExecutionError {
    #[error("Invalid ant state: {0:?}")]
    InvalidState(AntState),
}

/// Errors for ant tour operations
#[derive(Debug, thiserror::Error)]
pub enum AntTourError {
    #[error("Empty path - no venues visited")]
    EmptyPath,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ant_agent_creation() {
        let ant = AntAgent::new(AntAgentId::new(1), MetaOrderId::new(100), 5000);
        assert_eq!(ant.id, AntAgentId::new(1));
        assert_eq!(ant.state, AntState::Searching);
        assert!(ant.visited_venues.is_empty());
    }

    #[test]
    fn test_ant_reset_reuses_allocation() {
        let mut ant = AntAgent::new(AntAgentId::new(1), MetaOrderId::new(100), 5000);
        
        // Simulate some visits
        ant.visited_venues.push(VenueId::new(1));
        ant.visited_venues.push(VenueId::new(2));
        ant.path_quality = 0.85;

        // Reset
        ant.reset(MetaOrderId::new(200));
        
        assert_eq!(ant.meta_order_id, MetaOrderId::new(200));
        assert!(ant.visited_venues.is_empty());
        assert_eq!(ant.path_quality, 0.0);
        assert_eq!(ant.state, AntState::Searching);
    }
}
