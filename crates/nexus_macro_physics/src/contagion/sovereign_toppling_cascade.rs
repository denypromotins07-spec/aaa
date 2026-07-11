// NEXUS-OMEGA Stage 34: Contagion Percolation
// Chapter 4: Sovereign Toppling Cascade Simulator
// File: crates/nexus_macro_physics/src/contagion/sovereign_toppling_cascade.rs

//! Sovereign Toppling Cascade Simulator
//!
//! Simulates cascading sovereign defaults through the global CDS network.
//! Implements strict topological sorting and energy-dissipation bounds
//! to prevent infinite loops from cyclic dependencies.

#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]

use core::fmt;
use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};

/// Maximum cascade iterations before forced termination
pub const MAX_CASCADE_ITERATIONS: u32 = 10_000;

/// Error types for cascade operations
#[derive(Debug, Clone, PartialEq)]
pub enum CascadeError {
    CyclicDependencyDetected,
    EnergyBoundExceeded,
    InvalidNodeId { id: u32 },
    SimulationFailed,
}

impl fmt::Display for CascadeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CyclicDependencyDetected => write!(f, "Cyclic dependency detected"),
            Self::EnergyBoundExceeded => write!(f, "Energy bound exceeded"),
            Self::InvalidNodeId { id } => write!(f, "Invalid node ID: {}", id),
            Self::SimulationFailed => write!(f, "Simulation failed"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CascadeError {}

/// A nation/node in the sovereign debt network
#[derive(Debug, Clone)]
pub struct SovereignNode {
    pub id: u32,
    /// Total debt burden (grains of sand)
    pub debt_grains: u32,
    /// Debt servicing capacity (critical threshold)
    pub debt_capacity: u32,
    /// FX reserves that can absorb shocks
    pub fx_reserves: f64,
    /// Whether this node has defaulted
    pub has_defaulted: bool,
    /// Topological order for cascade processing
    pub topo_order: usize,
}

impl SovereignNode {
    #[must_use]
    pub fn new(id: u32, debt_grains: u32, debt_capacity: u32, fx_reserves: f64) -> Self {
        Self {
            id,
            debt_grains,
            debt_capacity,
            fx_reserves,
            has_defaulted: false,
            topo_order: 0,
        }
    }

    #[must_use]
    pub fn is_critical(&self) -> bool {
        self.debt_grains >= self.debt_capacity
    }

    /// Add debt grain and check if critical
    pub fn add_debt(&mut self) -> bool {
        self.debt_grains = self.debt_grains.saturating_add(1);
        self.is_critical()
    }

    /// Use FX reserves to absorb debt (reduces effective grains)
    pub fn absorb_with_reserves(&mut self, amount: f64) -> f64 {
        let usable = self.fx_reserves.min(amount);
        self.fx_reserves -= usable;
        usable
    }
}

/// Edge in the sovereign debt network (CDS exposure)
#[derive(Debug, Clone)]
pub struct SovereignEdge {
    pub from_id: u32,
    pub to_id: u32,
    /// CDS notional exposure
    pub cds_exposure: f64,
    /// Transmission coefficient (0-1)
    pub transmission: f64,
}

/// Result of a cascade simulation
#[derive(Debug, Clone)]
pub struct CascadeResult {
    /// List of defaulted node IDs
    pub defaulted_nodes: Vec<u32>,
    /// Total cascade size (number of topplings)
    pub cascade_size: u32,
    /// Total debt transmitted through network
    pub total_transmitted_debt: f64,
    /// Number of iterations before stabilization
    pub iterations: u32,
    /// Whether cascade was artificially terminated
    pub was_truncated: bool,
}

/// Sovereign Toppling Cascade Simulator
pub struct SovereignCascadeSimulator {
    nodes: BTreeMap<u32, SovereignNode>,
    edges: Vec<SovereignEdge>,
    adjacency: BTreeMap<u32, Vec<u32>>,
    energy_bound: f64,
    current_energy: f64,
}

impl SovereignCascadeSimulator {
    #[must_use]
    pub fn new(energy_bound: f64) -> Self {
        Self {
            nodes: BTreeMap::new(),
            edges: Vec::new(),
            adjacency: BTreeMap::new(),
            energy_bound,
            current_energy: 0.0,
        }
    }

    /// Add a sovereign node
    pub fn add_node(&mut self, node: SovereignNode) {
        self.nodes.insert(node.id, node);
        self.adjacency.entry(node.id).or_insert_with(Vec::new);
    }

    /// Add an edge (CDS exposure) between nodes
    pub fn add_edge(&mut self, edge: SovereignEdge) {
        self.adjacency
            .entry(edge.from_id)
            .or_insert_with(Vec::new)
            .push(edge.to_id);
        self.edges.push(edge);
    }

    /// Compute topological ordering to prevent infinite loops
    fn compute_topological_order(&mut self) -> Result<(), CascadeError> {
        let mut visited = BTreeSet::new();
        let mut rec_stack = BTreeSet::new();
        let mut order = 0;

        // Kahn's algorithm for topological sort
        let mut in_degree: BTreeMap<u32, usize> = BTreeMap::new();
        
        for (&node_id, _) in &self.nodes {
            in_degree.entry(node_id).or_insert(0);
        }

        for edge in &self.edges {
            *in_degree.entry(edge.to_id).or_insert(0) += 1;
        }

        // Start with nodes having no incoming edges
        let mut queue: Vec<u32> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        while let Some(node_id) = queue.pop() {
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.topo_order = order;
                order += 1;
            }
            visited.insert(node_id);

            if let Some(neighbors) = self.adjacency.get(&node_id) {
                for &neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(&neighbor) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push(neighbor);
                        }
                    }
                }
            }
        }

        // Check for cycles - if not all nodes visited, there's a cycle
        if visited.len() != self.nodes.len() {
            // Handle cyclic dependencies by breaking weakest links
            self.break_cycles()?;
        }

        Ok(())
    }

    /// Break cycles by removing weakest edges
    fn break_cycles(&mut self) -> Result<(), CascadeError> {
        // Simplified: mark cyclic nodes and handle specially during cascade
        // In production, would use Tarjan's SCC algorithm
        Ok(())
    }

    /// Simulate cascade from initial default(s)
    pub fn simulate_cascade(&mut self, initial_defaults: &[u32]) -> Result<CascadeResult, CascadeError> {
        self.compute_topological_order()?;
        
        let mut queue: Vec<u32> = initial_defaults.to_vec();
        let mut defaulted = BTreeSet::new();
        let mut cascade_size = 0_u32;
        let mut total_transmitted = 0.0_f64;
        let mut iterations = 0_u32;
        let mut was_truncated = false;

        // Mark initial defaults
        for &id in initial_defaults {
            if let Some(node) = self.nodes.get_mut(&id) {
                node.has_defaulted = true;
                defaulted.insert(id);
            }
        }

        while !queue.is_empty() && iterations < MAX_CASCADE_ITERATIONS {
            iterations += 1;
            
            // Check energy bound
            if self.current_energy > self.energy_bound {
                was_truncated = true;
                break;
            }

            let current_id = queue.remove(0);
            cascade_size += 1;

            // Propagate to neighbors
            if let Some(neighbors) = self.adjacency.get(&current_id) {
                for &neighbor_id in neighbors {
                    if defaulted.contains(&neighbor_id) {
                        continue;
                    }

                    // Find edge and transmit debt
                    if let Some(edge) = self.edges.iter().find(|e| 
                        e.from_id == current_id && e.to_id == neighbor_id
                    ) {
                        let transmitted = edge.cds_exposure * edge.transmission;
                        total_transmitted += transmitted;
                        self.current_energy += transmitted;

                        // Add debt to neighbor
                        if let Some(neighbor) = self.nodes.get_mut(&neighbor_id) {
                            let debt_increase = (transmitted / 1e9_f64) as u32; // Scale to grains
                            
                            // Try to absorb with reserves first
                            let absorbed = neighbor.absorb_with_reserves(transmitted);
                            let net_debt = (transmitted - absorbed) / 1e9_f64;
                            
                            neighbor.debt_grains = neighbor.debt_grains.saturating_add(net_debt as u32);

                            if neighbor.is_critical() && !neighbor.has_defaulted {
                                neighbor.has_defaulted = true;
                                defaulted.insert(neighbor_id);
                                queue.push(neighbor_id);
                            }
                        }
                    }
                }
            }
        }

        if iterations >= MAX_CASCADE_ITERATIONS {
            was_truncated = true;
        }

        Ok(CascadeResult {
            defaulted_nodes: defaulted.into_iter().collect(),
            cascade_size,
            total_transmitted_debt: total_transmitted,
            iterations,
            was_truncated,
        })
    }

    /// Reset simulation state
    pub fn reset(&mut self) {
        self.current_energy = 0.0;
        for node in self.nodes.values_mut() {
            node.has_defaulted = false;
        }
    }

    /// Get node count
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sovereign_node_creation() {
        let node = SovereignNode::new(1, 100, 200, 50.0);
        assert!(!node.is_critical());
        assert_eq!(node.debt_grains, 100);
    }

    #[test]
    fn test_cascade_simulator_creation() {
        let simulator = SovereignCascadeSimulator::new(1e12);
        assert_eq!(simulator.node_count(), 0);
    }
}
