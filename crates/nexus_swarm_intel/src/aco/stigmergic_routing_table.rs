//! Stigmergic Routing Table for Ant Colony System Smart Order Routing.
//! 
//! Implements a thread-safe pheromone matrix with bounded values to prevent
//! overflow and ensure stable probability calculations.

use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use nexus_types::market::VenueId;
use thiserror::Error;

/// Maximum number of venues supported (fixed size for zero-allocation)
pub const MAX_VENUES: usize = 64;

/// Minimum pheromone value to prevent complete evaporation
pub const MIN_PHEROMONE: f64 = 0.001;

/// Maximum pheromone value to prevent overflow
pub const MAX_PHEROMONE: f64 = 1.0;

/// Initial pheromone value for all edges
pub const INITIAL_PHEROMONE: f64 = 0.5;

/// Bounded pheromone value that stays within [MIN_PHEROMONE, MAX_PHEROMONE]
#[derive(Debug, Clone, Copy)]
pub struct PheromoneValue(f64);

impl PheromoneValue {
    pub fn new(value: f64) -> Self {
        Self(value.clamp(MIN_PHEROMONE, MAX_PHEROMONE))
    }

    pub fn raw(&self) -> f64 {
        self.0
    }

    pub fn normalized(&self) -> f64 {
        // Normalize to [0, 1] range for probability calculations
        (self.0 - MIN_PHEROMONE) / (MAX_PHEROMONE - MIN_PHEROMONE)
    }

    pub fn apply_evaporation(&mut self, rate: f64) {
        // τ ← (1 - ρ) * τ
        let evaporated = self.0 * (1.0 - rate);
        self.0 = evaporated.clamp(MIN_PHEROMONE, MAX_PHEROMONE);
    }

    pub fn deposit(&mut self, amount: f64) {
        self.0 = (self.0 + amount).clamp(MIN_PHEROMONE, MAX_PHEROMONE);
    }
}

/// Edge in the routing table connecting two venues
#[derive(Debug, Clone)]
pub struct VenueEdge {
    pub from_venue: Option<VenueId>,
    pub to_venue: VenueId,
    pub pheromone: PheromoneValue,
    pub latency_ns: u64,
    pub last_update_ns: u64,
}

impl VenueEdge {
    pub fn new(from: Option<VenueId>, to: VenueId, latency_ns: u64) -> Self {
        Self {
            from_venue: from,
            to_venue: to,
            pheromone: PheromoneValue::new(INITIAL_PHEROMONE),
            latency_ns,
            last_update_ns: 0,
        }
    }
}

/// Stigmergic routing table maintaining pheromone trails between venues
pub struct StigmergicRoutingTable {
    /// Adjacency matrix representation: [from][to] -> edge index
    /// Using fixed-size arrays for zero-allocation hot paths
    adjacency: [[Option<usize>; MAX_VENUES]; MAX_VENUES],
    /// Edge storage pool
    edges: Vec<VenueEdge>,
    /// Venue latencies in nanoseconds
    venue_latencies: [u64; MAX_VENUES],
    /// Number of active venues
    venue_count: usize,
    /// Atomic counter for edge indices
    edge_counter: AtomicU64,
    /// Global pheromone update pending flag
    global_update_pending: AtomicU64,
}

impl StigmergicRoutingTable {
    pub fn new() -> Self {
        Self {
            adjacency: [[None; MAX_VENUES]; MAX_VENUES],
            edges: Vec::with_capacity(MAX_VENUES * MAX_VENUES),
            venue_latencies: [0; MAX_VENUES],
            venue_count: 0,
            edge_counter: AtomicU64::new(0),
            global_update_pending: AtomicU64::new(0),
        }
    }

    /// Register a new venue in the routing table
    pub fn register_venue(&mut self, venue_id: VenueId, initial_latency_ns: u64) -> Result<(), RoutingTableError> {
        if self.venue_count >= MAX_VENUES {
            return Err(RoutingTableError::VenueLimitExceeded);
        }

        let idx = self.venue_count;
        self.venue_latencies[idx] = initial_latency_ns;
        self.venue_count += 1;

        // Initialize edges from all existing venues to this new venue
        for from_idx in 0..self.venue_count {
            let from_venue = if from_idx == idx {
                None
            } else {
                Some(VenueId::new(from_idx as u64))
            };
            
            let edge = VenueEdge::new(from_venue, VenueId::new(idx as u64), initial_latency_ns);
            let edge_idx = self.edges.len();
            self.edges.push(edge);
            self.adjacency[from_idx][idx] = Some(edge_idx);
        }

        Ok(())
    }

    /// Get candidate venues reachable from current venue
    pub fn get_candidate_venues(&self, current: Option<VenueId>) -> Result<Vec<VenueId>, RoutingTableError> {
        let mut candidates = Vec::with_capacity(self.venue_count);

        match current {
            None => {
                // Starting point: all venues are candidates
                for i in 0..self.venue_count {
                    candidates.push(VenueId::new(i as u64));
                }
            }
            Some(current_id) => {
                let current_idx = current_id.0 as usize;
                if current_idx >= self.venue_count {
                    return Err(RoutingTableError::InvalidVenue(current_id));
                }

                for to_idx in 0..self.venue_count {
                    if to_idx != current_idx && self.adjacency[current_idx][to_idx].is_some() {
                        candidates.push(VenueId::new(to_idx as u64));
                    }
                }
            }
        }

        Ok(candidates)
    }

    /// Get edge between two venues
    pub fn get_edge(
        &self,
        from: Option<VenueId>,
        to: VenueId,
    ) -> Result<&VenueEdge, RoutingTableError> {
        let from_idx = from.map(|v| v.0 as usize).unwrap_or(0);
        let to_idx = to.0 as usize;

        if from_idx >= self.venue_count || to_idx >= self.venue_count {
            return Err(RoutingTableError::InvalidVenue(to));
        }

        self.adjacency[from_idx][to_idx]
            .and_then(|idx| self.edges.get(idx))
            .ok_or(RoutingTableError::EdgeNotFound(from, to))
    }

    /// Get mutable edge between two venues
    pub fn get_edge_mut(
        &mut self,
        from: VenueId,
        to: VenueId,
    ) -> Result<&mut VenueEdge, RoutingTableError> {
        let from_idx = from.0 as usize;
        let to_idx = to.0 as usize;

        if from_idx >= self.venue_count || to_idx >= self.venue_count {
            return Err(RoutingTableError::InvalidVenue(to));
        }

        let edge_idx = self.adjacency[from_idx][to_idx]
            .ok_or(RoutingTableError::EdgeNotFound(Some(from), to))?;

        self.edges.get_mut(edge_idx)
            .ok_or(RoutingTableError::EdgeNotFound(Some(from), to))
    }

    /// Apply local pheromone update (immediate, by individual ants)
    pub fn apply_local_pheromone_update(
        &mut self,
        from: VenueId,
        to: VenueId,
        delta: f64,
    ) -> Result<(), RoutingTableError> {
        let edge = self.get_edge_mut(from, to)?;
        edge.pheromone.deposit(delta);
        edge.last_update_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Ok(())
    }

    /// Apply global pheromone update (by best ant of epoch)
    pub fn apply_global_pheromone_update(
        &mut self,
        path: &[VenueId],
        quality_score: f64,
        evaporation_rate: f64,
    ) -> Result<(), RoutingTableError> {
        if path.len() < 2 {
            return Err(RoutingTableError::InvalidPath);
        }

        // First, evaporate all pheromones
        self.evaporate_all(evaporation_rate)?;

        // Then, reinforce the best path
        let deposit_amount = quality_score; // Higher quality = more pheromone
        
        for window in path.windows(2) {
            let from = window[0];
            let to = window[1];
            let edge = self.get_edge_mut(from, to)?;
            edge.pheromone.deposit(deposit_amount);
        }

        self.global_update_pending.store(1, Ordering::Release);
        Ok(())
    }

    /// Evaporate pheromones on all edges
    pub fn evaporate_all(&mut self, rate: f64) -> Result<(), RoutingTableError> {
        for edge in &mut self.edges {
            edge.pheromone.apply_evaporation(rate);
        }
        Ok(())
    }

    /// Get latency for a specific venue
    pub fn get_latency(&self, venue_id: VenueId) -> u64 {
        let idx = venue_id.0 as usize;
        if idx < self.venue_count {
            self.venue_latencies[idx]
        } else {
            0
        }
    }

    /// Update latency for a venue
    pub fn update_latency(&mut self, venue_id: VenueId, latency_ns: u64) -> Result<(), RoutingTableError> {
        let idx = venue_id.0 as usize;
        if idx >= self.venue_count {
            return Err(RoutingTableError::InvalidVenue(venue_id));
        }
        self.venue_latencies[idx] = latency_ns;
        Ok(())
    }

    /// Get pheromone matrix for visualization/debugging
    pub fn get_pheromone_matrix(&self) -> Vec<Vec<f64>> {
        let mut matrix = Vec::with_capacity(self.venue_count);
        for i in 0..self.venue_count {
            let mut row = Vec::with_capacity(self.venue_count);
            for j in 0..self.venue_count {
                if i != j {
                    let pheromone = self.adjacency[i][j]
                        .and_then(|idx| self.edges.get(idx))
                        .map(|e| e.pheromone.raw())
                        .unwrap_or(MIN_PHEROMONE);
                    row.push(pheromone);
                } else {
                    row.push(0.0);
                }
            }
            matrix.push(row);
        }
        matrix
    }

    /// Check if global update is pending
    pub fn has_pending_global_update(&self) -> bool {
        self.global_update_pending.load(Ordering::Acquire) != 0
    }

    /// Clear global update flag
    pub fn clear_global_update_flag(&self) {
        self.global_update_pending.store(0, Ordering::Release);
    }
}

impl Default for StigmergicRoutingTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors for routing table operations
#[derive(Debug, Error)]
pub enum RoutingTableError {
    #[error("Venue limit exceeded (max: {MAX_VENUES})")]
    VenueLimitExceeded,
    #[error("Invalid venue: {0:?}")]
    InvalidVenue(VenueId),
    #[error("Edge not found: {0:?} -> {1:?}")]
    EdgeNotFound(Option<VenueId>, VenueId),
    #[error("Invalid path length")]
    InvalidPath,
    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pheromone_value_bounds() {
        // Test clamping at construction
        let p1 = PheromoneValue::new(-1.0);
        assert_eq!(p1.raw(), MIN_PHEROMONE);

        let p2 = PheromoneValue::new(100.0);
        assert_eq!(p2.raw(), MAX_PHEROMONE);

        // Test clamping during operations
        let mut p3 = PheromoneValue::new(0.5);
        p3.deposit(10.0);
        assert_eq!(p3.raw(), MAX_PHEROMONE);

        let mut p4 = PheromoneValue::new(0.5);
        p4.apply_evaporation(0.99);
        assert!(p4.raw() >= MIN_PHEROMONE);
    }

    #[test]
    fn test_routing_table_creation() {
        let mut table = StigmergicRoutingTable::new();
        
        // Register venues
        table.register_venue(VenueId::new(0), 1000).unwrap();
        table.register_venue(VenueId::new(1), 2000).unwrap();
        table.register_venue(VenueId::new(2), 1500).unwrap();

        // Check candidate venues from start
        let candidates = table.get_candidate_venues(None).unwrap();
        assert_eq!(candidates.len(), 3);

        // Check candidate venues from venue 0
        let candidates = table.get_candidate_venues(Some(VenueId::new(0))).unwrap();
        assert_eq!(candidates.len(), 2); // Venues 1 and 2
    }

    #[test]
    fn test_pheromone_update_cycle() {
        let mut table = StigmergicRoutingTable::new();
        table.register_venue(VenueId::new(0), 1000).unwrap();
        table.register_venue(VenueId::new(1), 2000).unwrap();

        // Get initial pheromone
        let edge = table.get_edge(None, VenueId::new(1)).unwrap();
        let initial = edge.pheromone.raw();
        assert_eq!(initial, INITIAL_PHEROMONE);

        // Apply local update
        table.apply_local_pheromone_update(VenueId::new(0), VenueId::new(1), 0.1).unwrap();
        
        let edge = table.get_edge(VenueId::new(0), VenueId::new(1)).unwrap();
        assert!(edge.pheromone.raw() > initial);

        // Apply evaporation
        table.evaporate_all(0.5).unwrap();
        
        let edge = table.get_edge(VenueId::new(0), VenueId::new(1)).unwrap();
        assert!(edge.pheromone.raw() < initial + 0.1);
        assert!(edge.pheromone.raw() >= MIN_PHEROMONE);
    }
}
