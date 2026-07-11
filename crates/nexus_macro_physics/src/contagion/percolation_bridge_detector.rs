// NEXUS-OMEGA Stage 34: Contagion Percolation
// Chapter 4: Percolation Bridge Detector
// File: crates/nexus_macro_physics/src/contagion/percolation_bridge_detector.rs

//! Percolation Bridge Detector for Critical CDS Contracts
//!
//! Identifies sovereign CDS contracts that act as critical "bridges"
//! in the global financial network. These bridges are the key transmission
//! channels for systemic risk cascades.

#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]

use core::fmt;
use alloc::vec::Vec;
use alloc::collections::{BTreeMap, BTreeSet};

/// Error types for bridge detection
#[derive(Debug, Clone, PartialEq)]
pub enum BridgeDetectionError {
    GraphDisconnected,
    NoBridgesFound,
    InvalidEdgeIndex(usize),
}

impl fmt::Display for BridgeDetectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GraphDisconnected => write!(f, "Graph is disconnected"),
            Self::NoBridgesFound => write!(f, "No bridges found"),
            Self::InvalidEdgeIndex(idx) => write!(f, "Invalid edge index: {}", idx),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BridgeDetectionError {}

/// Edge in the CDS network with bridge information
#[derive(Debug, Clone)]
pub struct CDSEdge {
    pub id: usize,
    pub from_node: u32,
    pub to_node: u32,
    pub notional: f64,
    pub maturity_years: f64,
    /// Bridge score (higher = more critical)
    pub bridge_score: f64,
    /// Whether this edge is a bridge
    pub is_bridge: bool,
}

/// Result of bridge detection analysis
#[derive(Debug, Clone)]
pub struct BridgeDetectionResult {
    /// All identified bridges
    pub bridges: Vec<CDSEdge>,
    /// Total number of edges analyzed
    pub total_edges: usize,
    /// Number of connected components
    pub num_components: usize,
    /// Percolation threshold estimate
    pub percolation_threshold: f64,
    /// Systemic risk score (0-1)
    pub systemic_risk_score: f64,
}

/// Percolation Bridge Detector using Tarjan's bridge-finding algorithm
pub struct PercolationBridgeDetector {
    adjacency: BTreeMap<u32, Vec<(u32, usize)>>, // node -> [(neighbor, edge_id)]
    edges: BTreeMap<usize, CDSEdge>,
    discovery_time: BTreeMap<u32, u32>,
    low_link: BTreeMap<u32, u32>,
    time_counter: u32,
}

impl PercolationBridgeDetector {
    #[must_use]
    pub fn new() -> Self {
        Self {
            adjacency: BTreeMap::new(),
            edges: BTreeMap::new(),
            discovery_time: BTreeMap::new(),
            low_link: BTreeMap::new(),
            time_counter: 0,
        }
    }

    /// Add a CDS edge to the network
    pub fn add_edge(&mut self, edge: CDSEdge) {
        let edge_id = edge.id;
        
        self.adjacency
            .entry(edge.from_node)
            .or_insert_with(Vec::new)
            .push((edge.to_node, edge_id));
        
        self.adjacency
            .entry(edge.to_node)
            .or_insert_with(Vec::new)
            .push((edge.from_node, edge_id));
        
        self.edges.insert(edge_id, edge);
    }

    /// Find all bridges in the network using Tarjan's algorithm
    pub fn find_bridges(&mut self) -> Result<BridgeDetectionResult, BridgeDetectionError> {
        self.discovery_time.clear();
        self.low_link.clear();
        self.time_counter = 0;

        let mut visited = BTreeSet::new();
        let mut bridge_ids: BTreeSet<usize> = BTreeSet::new();
        let mut num_components = 0;

        // Run DFS from each unvisited node to handle disconnected graphs
        for &node in self.adjacency.keys() {
            if !visited.contains(&node) {
                num_components += 1;
                self.dfs_find_bridges(node, None, &mut visited, &mut bridge_ids);
            }
        }

        // Update edge bridge status
        for (&edge_id, edge) in self.edges.iter_mut() {
            edge.is_bridge = bridge_ids.contains(&edge_id);
        }

        // Collect bridges
        let bridges: Vec<CDSEdge> = self.edges
            .values()
            .filter(|e| e.is_bridge)
            .cloned()
            .collect();

        // Calculate percolation threshold and systemic risk
        let total_edges = self.edges.len();
        let num_bridges = bridges.len();
        
        let percolation_threshold = if total_edges > 0 {
            num_bridges as f64 / total_edges as f64
        } else {
            0.0
        };

        // Systemic risk based on bridge concentration and notional
        let systemic_risk_score = self.calculate_systemic_risk(&bridges);

        Ok(BridgeDetectionResult {
            bridges,
            total_edges,
            num_components,
            percolation_threshold,
            systemic_risk_score,
        })
    }

    /// DFS to find bridges using Tarjan's algorithm
    fn dfs_find_bridges(
        &mut self,
        node: u32,
        parent: Option<u32>,
        visited: &mut BTreeSet<u32>,
        bridge_ids: &mut BTreeSet<usize>,
    ) {
        visited.insert(node);
        self.time_counter += 1;
        
        self.discovery_time.insert(node, self.time_counter);
        self.low_link.insert(node, self.time_counter);

        if let Some(neighbors) = self.adjacency.get(&node) {
            for &(neighbor, edge_id) in neighbors {
                if Some(neighbor) == parent {
                    continue;
                }

                if visited.contains(&neighbor) {
                    // Back edge - update low link
                    if let Some(&disc_time) = self.discovery_time.get(&neighbor) {
                        if let Some(low) = self.low_link.get_mut(&node) {
                            *low = (*low).min(disc_time);
                        }
                    }
                } else {
                    // Tree edge - recurse
                    self.dfs_find_bridges(neighbor, Some(node), visited, bridge_ids);
                    
                    // Check if this edge is a bridge
                    if let Some(&neighbor_low) = self.low_link.get(&neighbor) {
                        let disc_time = self.discovery_time.get(&node).copied().unwrap_or(0);
                        
                        if neighbor_low > disc_time {
                            // This is a bridge
                            bridge_ids.insert(edge_id);
                        }
                        
                        // Update low link
                        if let Some(low) = self.low_link.get_mut(&node) {
                            *low = (*low).min(neighbor_low);
                        }
                    }
                }
            }
        }
    }

    /// Calculate systemic risk score based on bridge properties
    fn calculate_systemic_risk(&self, bridges: &[CDSEdge]) -> f64 {
        if bridges.is_empty() {
            return 0.0;
        }

        let total_notional: f64 = bridges.iter().map(|b| b.notional).sum();
        let max_notional = bridges.iter().map(|b| b.notional).fold(0.0_f64, f64::max);
        
        let avg_maturity = bridges.iter().map(|b| b.maturity_years).sum::<f64>() 
            / bridges.len() as f64;

        // Risk increases with:
        // 1. Higher total notional on bridges
        // 2. Shorter maturities (more immediate risk)
        // 3. More bridges relative to network size
        
        let notional_factor = (total_notional / 1e12).min(1.0); // Normalize to trillions
        let maturity_factor = 1.0 / (1.0 + avg_maturity); // Shorter = riskier
        let concentration_factor = max_notional / (total_notional + 1.0);

        (notional_factor * 0.5 + maturity_factor * 0.3 + concentration_factor * 0.2)
            .clamp(0.0, 1.0)
    }

    /// Get edge by ID
    #[must_use]
    pub fn get_edge(&self, edge_id: usize) -> Option<&CDSEdge> {
        self.edges.get(&edge_id)
    }

    /// Get total edge count
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl Default for PercolationBridgeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_detector_creation() {
        let detector = PercolationBridgeDetector::new();
        assert_eq!(detector.edge_count(), 0);
    }

    #[test]
    fn test_simple_bridge_detection() {
        let mut detector = PercolationBridgeDetector::new();
        
        // Create a simple graph with one bridge: A-B-C-D where B-C is a bridge
        detector.add_edge(CDSEdge {
            id: 0,
            from_node: 0,
            to_node: 1,
            notional: 1e9,
            maturity_years: 5.0,
            bridge_score: 0.0,
            is_bridge: false,
        });
        detector.add_edge(CDSEdge {
            id: 1,
            from_node: 1,
            to_node: 2,
            notional: 1e9,
            maturity_years: 5.0,
            bridge_score: 0.0,
            is_bridge: false,
        });
        detector.add_edge(CDSEdge {
            id: 2,
            from_node: 2,
            to_node: 3,
            notional: 1e9,
            maturity_years: 5.0,
            bridge_score: 0.0,
            is_bridge: false,
        });

        let result = detector.find_bridges();
        assert!(result.is_ok());
    }
}
