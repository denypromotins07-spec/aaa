//! Global Supply Chain Graph
//! 
//! Lock-free directed graph representing global supply chain nodes
//! (ports, canals, rail hubs) and edges (shipping lanes, routes).
//! Uses arena allocation to prevent heap fragmentation.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thiserror::Error;

/// Supply chain graph errors
#[derive(Debug, Error)]
pub enum SupplyChainError {
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    #[error("Edge not found: {0}")]
    EdgeNotFound(String),
    #[error("Arena capacity exceeded")]
    ArenaCapacityExceeded,
    #[error("Invalid connection: {0}")]
    InvalidConnection(String),
}

/// Maximum number of nodes in the supply chain graph
const MAX_NODES: usize = 100_000;
/// Maximum number of edges in the supply chain graph
const MAX_EDGES: usize = 500_000;

/// Node types in the supply chain
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeType {
    Port,
    Canal,
    RailHub,
    Airport,
    Pipeline,
    Warehouse,
    Manufacturing,
    OilStorage,
    Agricultural,
}

/// Supply chain node
#[derive(Debug, Clone)]
pub struct SupplyNode {
    pub id: NodeId,
    pub node_type: NodeType,
    pub name: String,
    pub latitude: f64,
    pub longitude: f64,
    pub capacity_units_per_day: f64,
    pub current_congestion_level: f64, // 0.0 (empty) to 1.0 (saturated)
    pub metadata: NodeMetadata,
}

impl SupplyNode {
    pub fn new(
        id: NodeId,
        node_type: NodeType,
        name: String,
        latitude: f64,
        longitude: f64,
        capacity_units_per_day: f64,
    ) -> Self {
        SupplyNode {
            id,
            node_type,
            name,
            latitude,
            longitude,
            capacity_units_per_day,
            current_congestion_level: 0.0,
            metadata: NodeMetadata::default(),
        }
    }
}

/// Additional node metadata
#[derive(Debug, Clone, Default)]
pub struct NodeMetadata {
    pub country_code: Option<String>,
    pub timezone: Option<String>,
    pub operating_hours: Option<OperatingHours>,
    pub supported_cargo_types: Vec<CargoType>,
}

#[derive(Debug, Clone)]
pub struct OperatingHours {
    pub open_hour: u8,
    pub close_hour: u8,
    pub operates_24_7: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CargoType {
    Container,
    Bulk,
    Liquid,
    Gas,
    Vehicle,
    Refrigerated,
}

/// Unique node identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u64);

impl NodeId {
    pub fn new(id: u64) -> Self {
        NodeId(id)
    }

    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

/// Edge connecting two nodes
#[derive(Debug, Clone)]
pub struct SupplyEdge {
    pub id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
    pub edge_type: EdgeType,
    pub distance_km: f64,
    pub transit_time_hours: f64,
    pub cost_per_unit: f64,
    pub capacity_units_per_day: f64,
    pub current_flow: f64,
}

impl SupplyEdge {
    pub fn new(
        id: EdgeId,
        from: NodeId,
        to: NodeId,
        edge_type: EdgeType,
        distance_km: f64,
        transit_time_hours: f64,
    ) -> Self {
        SupplyEdge {
            id,
            from,
            to,
            edge_type,
            distance_km,
            transit_time_hours,
            cost_per_unit: 0.0,
            capacity_units_per_day: f64::INFINITY,
            current_flow: 0.0,
        }
    }
}

/// Edge types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    ShippingLane,
    RailRoute,
    TruckRoute,
    Pipeline,
    AirRoute,
    CanalPassage,
}

/// Unique edge identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(u64);

impl EdgeId {
    pub fn new(id: u64) -> Self {
        EdgeId(id)
    }
}

/// Lock-free supply chain graph using arena allocation
pub struct GlobalSupplyChainGraph {
    /// Arena for node storage (pre-allocated)
    nodes: Box<[Option<SupplyNode>; MAX_NODES]>,
    /// Arena for edge storage (pre-allocated)
    edges: Box<[Option<SupplyEdge>; MAX_EDGES]>,
    /// Atomic counter for node IDs
    next_node_id: AtomicUsize,
    /// Atomic counter for edge IDs
    next_edge_id: AtomicUsize,
    /// Adjacency list: node -> list of outgoing edges
    adjacency: Box<[Vec<EdgeId>; MAX_NODES]>,
    /// Reverse adjacency: node -> list of incoming edges
    reverse_adjacency: Box<[Vec<EdgeId>; MAX_NODES]>,
}

unsafe impl Send for GlobalSupplyChainGraph {}
unsafe impl Sync for GlobalSupplyChainGraph {}

impl GlobalSupplyChainGraph {
    /// Create a new empty supply chain graph
    pub fn new() -> Result<Self, SupplyChainError> {
        // Initialize arenas with None
        let nodes = Box::new([const { None }; MAX_NODES]);
        let edges = Box::new([const { None }; MAX_EDGES]);
        
        // Initialize adjacency lists
        let adjacency = Box::new(std::array::from_fn(|_| Vec::with_capacity(16)));
        let reverse_adjacency = Box::new(std::array::from_fn(|_| Vec::with_capacity(16)));
        
        Ok(GlobalSupplyChainGraph {
            nodes,
            edges,
            next_node_id: AtomicUsize::new(0),
            next_edge_id: AtomicUsize::new(0),
            adjacency,
            reverse_adjacency,
        })
    }

    /// Add a node to the graph
    pub fn add_node(&self, node: SupplyNode) -> Result<NodeId, SupplyChainError> {
        let idx = self.next_node_id.fetch_add(1, Ordering::SeqCst);
        
        if idx >= MAX_NODES {
            return Err(SupplyChainError::ArenaCapacityExceeded);
        }
        
        self.nodes[idx] = Some(node);
        Ok(NodeId::new(idx as u64))
    }

    /// Add an edge to the graph
    pub fn add_edge(&self, edge: SupplyEdge) -> Result<EdgeId, SupplyChainError> {
        // Validate nodes exist
        if self.nodes[edge.from.0 as usize].is_none() {
            return Err(SupplyChainError::NodeNotFound(format!(
                "Source node {} not found",
                edge.from.0
            )));
        }
        
        if self.nodes[edge.to.0 as usize].is_none() {
            return Err(SupplyChainError::NodeNotFound(format!(
                "Destination node {} not found",
                edge.to.0
            )));
        }
        
        let idx = self.next_edge_id.fetch_add(1, Ordering::SeqCst);
        
        if idx >= MAX_EDGES {
            return Err(SupplyChainError::ArenaCapacityExceeded);
        }
        
        self.edges[idx] = Some(edge.clone());
        
        // Update adjacency lists
        self.adjacency[edge.from.0 as usize].push(EdgeId::new(idx as u64));
        self.reverse_adjacency[edge.to.0 as usize].push(EdgeId::new(idx as u64));
        
        Ok(EdgeId::new(idx as u64))
    }

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&SupplyNode> {
        self.nodes.get(id.0 as usize).and_then(|n| n.as_ref())
    }

    /// Get an edge by ID
    pub fn get_edge(&self, id: EdgeId) -> Option<&SupplyEdge> {
        self.edges.get(id.0 as usize).and_then(|e| e.as_ref())
    }

    /// Get all outgoing edges from a node
    pub fn get_outgoing_edges(&self, node_id: NodeId) -> Vec<&SupplyEdge> {
        self.adjacency
            .get(node_id.0 as usize)
            .map(|edge_ids| {
                edge_ids
                    .iter()
                    .filter_map(|id| self.get_edge(*id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find path between two nodes using BFS
    pub fn find_path(&self, from: NodeId, to: NodeId) -> Option<Vec<NodeId>> {
        if from == to {
            return Some(vec![from]);
        }
        
        let mut visited = vec![false; MAX_NODES];
        let mut queue = std::collections::VecDeque::new();
        let mut parent = vec![None; MAX_NODES];
        
        queue.push_back(from);
        visited[from.0 as usize] = true;
        
        while let Some(current) = queue.pop_front() {
            for edge in self.get_outgoing_edges(current) {
                let next = edge.to;
                
                if !visited[next.0 as usize] {
                    visited[next.0 as usize] = true;
                    parent[next.0 as usize] = Some(current);
                    
                    if next == to {
                        // Reconstruct path
                        let mut path = vec![to];
                        let mut current = to;
                        
                        while let Some(p) = parent[current.0 as usize] {
                            path.push(p);
                            current = p;
                            
                            if p == from {
                                break;
                            }
                        }
                        
                        path.reverse();
                        return Some(path);
                    }
                    
                    queue.push_back(next);
                }
            }
        }
        
        None
    }

    /// Calculate congestion level for a node
    pub fn update_node_congestion(&self, node_id: NodeId, congestion: f64) -> Result<(), SupplyChainError> {
        if congestion < 0.0 || congestion > 1.0 {
            return Err(SupplyChainError::InvalidConnection(
                "Congestion must be between 0.0 and 1.0".to_string(),
            ));
        }
        
        if let Some(node) = &mut self.nodes[node_id.0 as usize] {
            node.current_congestion_level = congestion;
            Ok(())
        } else {
            Err(SupplyChainError::NodeNotFound(format!("Node {} not found", node_id.0)))
        }
    }
}

impl Default for GlobalSupplyChainGraph {
    fn default() -> Self {
        Self::new().expect("Failed to create default supply chain graph")
    }
}

/// Pre-built chokepoint locations
pub struct ChokepointRegistry;

impl ChokepointRegistry {
    pub fn suez_canal() -> SupplyNode {
        SupplyNode::new(
            NodeId::new(0),
            NodeType::Canal,
            "Suez Canal".to_string(),
            30.5667,
            32.2667,
            100_000_000.0, // tons per year capacity
        )
    }

    pub fn panama_canal() -> SupplyNode {
        SupplyNode::new(
            NodeId::new(1),
            NodeType::Canal,
            "Panama Canal".to_string(),
            9.0833,
            -79.6833,
            50_000_000.0,
        )
    }

    pub fn port_of_los_angeles() -> SupplyNode {
        SupplyNode::new(
            NodeId::new(2),
            NodeType::Port,
            "Port of Los Angeles".to_string(),
            33.7361,
            -118.2639,
            9_000_000.0, // TEU per year
        )
    }

    pub fn port_of_shanghai() -> SupplyNode {
        SupplyNode::new(
            NodeId::new(3),
            NodeType::Port,
            "Port of Shanghai".to_string(),
            31.2304,
            121.4737,
            47_000_000.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = GlobalSupplyChainGraph::new().unwrap();
        assert!(graph.nodes.len() == MAX_NODES);
        assert!(graph.edges.len() == MAX_EDGES);
    }

    #[test]
    fn test_add_node_and_edge() {
        let graph = GlobalSupplyChainGraph::new().unwrap();
        
        let node1 = SupplyNode::new(
            NodeId::new(0),
            NodeType::Port,
            "Test Port".to_string(),
            30.0,
            -90.0,
            1000.0,
        );
        
        let node2 = SupplyNode::new(
            NodeId::new(1),
            NodeType::RailHub,
            "Test Rail".to_string(),
            31.0,
            -91.0,
            500.0,
        );
        
        let id1 = graph.add_node(node1).unwrap();
        let id2 = graph.add_node(node2).unwrap();
        
        let edge = SupplyEdge::new(
            EdgeId::new(0),
            id1,
            id2,
            EdgeType::RailRoute,
            100.0,
            2.0,
        );
        
        let edge_id = graph.add_edge(edge).unwrap();
        
        assert!(graph.get_node(id1).is_some());
        assert!(graph.get_node(id2).is_some());
        assert!(graph.get_edge(edge_id).is_some());
    }
}
