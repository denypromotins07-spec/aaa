//! Sharded Knowledge Graph with Lock-Free Concurrent Access
//!
//! This module implements an in-memory knowledge graph using petgraph
//! with sharded, lock-free concurrent access patterns to enable
//! multiple threads to traverse the graph simultaneously without
//! mutex contention.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use dashmap::DashMap;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use tracing::{info, debug, warn};

/// Maximum number of shards for the graph
const NUM_SHARDS: usize = 16;

/// Hash a key to a shard index
#[inline]
fn shard_for(key: u64) -> usize {
    (key as usize) ^ (key as usize >> 7) ^ (key as usize >> 14) & (NUM_SHARDS - 1)
}

/// Types of nodes in the knowledge graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NodeType {
    Entity,      // Companies, people, organizations
    Event,       // News events, announcements
    Signal,      // Trading signals
    Indicator,   // Economic indicators
    Asset,       // Financial assets
}

/// Types of edges/relationships
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EdgeType {
    Owns,           // Entity owns asset
    Affects,        // Event affects entity/asset
    Correlates,     // Two entities correlate
    Causes,         // One event causes another
    Signals,        // Entity generates signal
    RelatedTo,      // General relationship
    DependsOn,      // Dependency relationship
}

/// Data associated with a node
#[derive(Debug, Clone)]
pub struct NodeData {
    /// Unique node ID
    pub id: u64,
    /// Node type
    pub node_type: NodeType,
    /// Display name
    pub name: String,
    /// Associated ticker/symbol (if applicable)
    pub symbol: Option<String>,
    /// Metadata payload
    pub metadata: serde_json::Value,
    /// Creation timestamp (nanoseconds)
    pub created_ns: u128,
    /// Last update timestamp (nanoseconds)
    pub updated_ns: u128,
    /// Version for optimistic locking
    pub version: u64,
}

impl NodeData {
    pub fn new(id: u64, node_type: NodeType, name: String) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        
        Self {
            id,
            node_type,
            name,
            symbol: None,
            metadata: serde_json::Value::Null,
            created_ns: now,
            updated_ns: now,
            version: 1,
        }
    }

    pub fn with_symbol(mut self, symbol: String) -> Self {
        self.symbol = Some(symbol);
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Data associated with an edge
#[derive(Debug, Clone)]
pub struct EdgeData {
    /// Edge type
    pub edge_type: EdgeType,
    /// Weight/strength of relationship (0.0 to 1.0)
    pub weight: f64,
    /// Timestamp when relationship was established
    pub created_ns: u128,
    /// Expiration timestamp (if applicable)
    pub expires_ns: Option<u128>,
    /// Metadata
    pub metadata: serde_json::Value,
}

impl EdgeData {
    pub fn new(edge_type: EdgeType) -> Self {
        Self {
            edge_type,
            weight: 1.0,
            created_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            expires_ns: None,
            metadata: serde_json::Value::Null,
        }
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }
}

/// A single shard of the knowledge graph
struct GraphShard {
    /// The actual graph
    graph: DashMap<NodeIndex, NodeData>,
    /// Edges stored separately for concurrent access
    edges: DashMap<(NodeIndex, NodeIndex), EdgeData>,
    /// Node ID to NodeIndex mapping
    id_to_index: DashMap<u64, NodeIndex>,
    /// Statistics
    node_count: AtomicUsize,
    edge_count: AtomicUsize,
}

impl GraphShard {
    fn new() -> Self {
        Self {
            graph: DashMap::new(),
            edges: DashMap::new(),
            id_to_index: DashMap::new(),
            node_count: AtomicUsize::new(0),
            edge_count: AtomicUsize::new(0),
        }
    }
}

impl Default for GraphShard {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the knowledge graph
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub nodes_by_type: Vec<(NodeType, usize)>,
    pub edges_by_type: Vec<(EdgeType, usize)>,
    pub avg_degree: f64,
}

/// Sharded, lock-free knowledge graph
pub struct ShardedKnowledgeGraph {
    /// Shards for concurrent access
    shards: Vec<Arc<GraphShard>>,
    /// Global node ID counter
    next_node_id: AtomicU64,
    /// Global statistics
    stats: Arc<AtomicGraphStats>,
}

/// Atomic statistics tracker
struct AtomicGraphStats {
    total_nodes: AtomicUsize,
    total_edges: AtomicUsize,
}

impl AtomicGraphStats {
    fn new() -> Self {
        Self {
            total_nodes: AtomicUsize::new(0),
            total_edges: AtomicUsize::new(0),
        }
    }
}

impl Default for AtomicGraphStats {
    fn default() -> Self {
        Self::new()
    }
}

impl ShardedKnowledgeGraph {
    /// Create a new sharded knowledge graph
    pub fn new() -> Self {
        let shards: Vec<Arc<GraphShard>> = (0..NUM_SHARDS)
            .map(|_| Arc::new(GraphShard::new()))
            .collect();

        info!("Created ShardedKnowledgeGraph with {} shards", NUM_SHARDS);

        Self {
            shards,
            next_node_id: AtomicU64::new(1),
            stats: Arc::new(AtomicGraphStats::new()),
        }
    }

    /// Get the shard for a given node ID
    #[inline]
    fn get_shard(&self, node_id: u64) -> &Arc<GraphShard> {
        &self.shards[shard_for(node_id)]
    }

    /// Add a node to the graph
    pub fn add_node(&self, mut data: NodeData) -> u64 {
        // Assign a unique ID if not already set
        if data.id == 0 {
            data.id = self.next_node_id.fetch_add(1, Ordering::Relaxed);
        }

        let shard = self.get_shard(data.id);
        
        // Create a node index (using ID as proxy for simplicity)
        let index = NodeIndex::new(data.id as usize);
        
        shard.graph.insert(index, data.clone());
        shard.id_to_index.insert(data.id, index);
        shard.node_count.fetch_add(1, Ordering::Relaxed);
        self.stats.total_nodes.fetch_add(1, Ordering::Relaxed);

        debug!("Added node {} ({})", data.id, data.name);
        data.id
    }

    /// Add an edge between two nodes
    pub fn add_edge(&self, from_id: u64, to_id: u64, edge_data: EdgeData) -> bool {
        let from_shard = self.get_shard(from_id);
        let to_shard = self.get_shard(to_id);

        // Verify both nodes exist
        if !from_shard.id_to_index.contains_key(&from_id) {
            warn!("Source node {} does not exist", from_id);
            return false;
        }
        if !to_shard.id_to_index.contains_key(&to_id) {
            warn!("Target node {} does not exist", to_id);
            return false;
        }

        let from_index = NodeIndex::new(from_id as usize);
        let to_index = NodeIndex::new(to_id as usize);

        // Store edge in the source node's shard
        from_shard.edges.insert((from_index, to_index), edge_data);
        from_shard.edge_count.fetch_add(1, Ordering::Relaxed);
        self.stats.total_edges.fetch_add(1, Ordering::Relaxed);

        debug!("Added edge {} -> {}", from_id, to_id);
        true
    }

    /// Get a node by ID
    pub fn get_node(&self, node_id: u64) -> Option<NodeData> {
        let shard = self.get_shard(node_id);
        let index = NodeIndex::new(node_id as usize);
        shard.graph.get(&index).map(|entry| entry.value().clone())
    }

    /// Get all neighbors of a node (outgoing edges)
    pub fn get_neighbors(&self, node_id: u64) -> Vec<(u64, EdgeData)> {
        let shard = self.get_shard(node_id);
        let from_index = NodeIndex::new(node_id as usize);
        let mut neighbors = Vec::new();

        for entry in shard.edges.iter() {
            let ((from, to), edge_data) = entry.pair();
            if *from == from_index {
                neighbors.push((to.index() as u64, edge_data.clone()));
            }
        }

        neighbors
    }

    /// Traverse the graph from a starting node up to max_depth
    pub fn traverse(&self, start_id: u64, max_depth: usize) -> Vec<u64> {
        let mut visited = std::collections::HashSet::new();
        let mut result = Vec::new();
        let mut queue = vec![(start_id, 0)];

        while let Some((current_id, depth)) = queue.pop() {
            if depth > max_depth || visited.contains(&current_id) {
                continue;
            }

            visited.insert(current_id);
            result.push(current_id);

            // Get neighbors and add to queue
            let neighbors = self.get_neighbors(current_id);
            for (neighbor_id, _) in neighbors {
                if !visited.contains(&neighbor_id) {
                    queue.push((neighbor_id, depth + 1));
                }
            }
        }

        result
    }

    /// Find paths between two nodes (BFS)
    pub fn find_paths(&self, from_id: u64, to_id: u64, max_length: usize) -> Vec<Vec<u64>> {
        let mut paths = Vec::new();
        let mut queue = vec![vec![from_id]];

        while let Some(path) = queue.pop() {
            if path.len() > max_length {
                continue;
            }

            let current = *path.last().unwrap();
            if current == to_id {
                paths.push(path);
                continue;
            }

            let neighbors = self.get_neighbors(current);
            for (neighbor_id, _) in neighbors {
                if !path.contains(&neighbor_id) {
                    let mut new_path = path.clone();
                    new_path.push(neighbor_id);
                    queue.push(new_path);
                }
            }
        }

        paths
    }

    /// Remove a node and its edges
    pub fn remove_node(&self, node_id: u64) -> bool {
        let shard = self.get_shard(node_id);
        let index = NodeIndex::new(node_id as usize);

        // Remove all edges involving this node
        let mut edges_to_remove = Vec::new();
        for entry in shard.edges.iter() {
            let ((from, to), _) = entry.pair();
            if *from == index || *to == index {
                edges_to_remove.push((*from, *to));
            }
        }

        for (from, to) in edges_to_remove {
            shard.edges.remove(&(from, to));
            shard.edge_count.fetch_sub(1, Ordering::Relaxed);
            self.stats.total_edges.fetch_sub(1, Ordering::Relaxed);
        }

        // Remove the node
        if shard.graph.remove(&index).is_some() {
            shard.id_to_index.remove(&node_id);
            shard.node_count.fetch_sub(1, Ordering::Relaxed);
            self.stats.total_nodes.fetch_sub(1, Ordering::Relaxed);
            return true;
        }

        false
    }

    /// Get graph statistics
    pub fn get_stats(&self) -> GraphStats {
        let mut total_nodes = 0;
        let mut total_edges = 0;
        let mut nodes_by_type: std::collections::HashMap<NodeType, usize> = std::collections::HashMap::new();

        for shard in &self.shards {
            total_nodes += shard.node_count.load(Ordering::Relaxed);
            total_edges += shard.edge_count.load(Ordering::Relaxed);

            for entry in shard.graph.iter() {
                let node_data = entry.value();
                *nodes_by_type.entry(node_data.node_type.clone()).or_insert(0) += 1;
            }
        }

        let avg_degree = if total_nodes > 0 {
            total_edges as f64 / total_nodes as f64
        } else {
            0.0
        };

        GraphStats {
            total_nodes,
            total_edges,
            nodes_by_type: nodes_by_type.into_iter().collect(),
            edges_by_type: Vec::new(), // Would need to track edge types similarly
            avg_degree,
        }
    }

    /// Clear the entire graph
    pub fn clear(&self) {
        for shard in &self.shards {
            shard.graph.clear();
            shard.edges.clear();
            shard.id_to_index.clear();
            shard.node_count.store(0, Ordering::Relaxed);
            shard.edge_count.store(0, Ordering::Relaxed);
        }
        self.stats.total_nodes.store(0, Ordering::Relaxed);
        self.stats.total_edges.store(0, Ordering::Relaxed);
    }
}

impl Default for ShardedKnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_node() {
        let graph = ShardedKnowledgeGraph::new();
        
        let node_data = NodeData::new(0, NodeType::Entity, "Apple".to_string());
        let id = graph.add_node(node_data);
        
        assert!(id > 0);
        
        let retrieved = graph.get_node(id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Apple");
    }

    #[test]
    fn test_add_edge() {
        let graph = ShardedKnowledgeGraph::new();
        
        let node1 = graph.add_node(NodeData::new(0, NodeType::Entity, "Company A".to_string()));
        let node2 = graph.add_node(NodeData::new(0, NodeType::Asset, "Stock A".to_string()));
        
        let edge = EdgeData::new(EdgeType::Owns);
        let success = graph.add_edge(node1, node2, edge);
        
        assert!(success);
        
        let neighbors = graph.get_neighbors(node1);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, node2);
    }

    #[test]
    fn test_traverse() {
        let graph = ShardedKnowledgeGraph::new();
        
        // Create a chain: A -> B -> C -> D
        let a = graph.add_node(NodeData::new(0, NodeType::Entity, "A".to_string()));
        let b = graph.add_node(NodeData::new(0, NodeType::Entity, "B".to_string()));
        let c = graph.add_node(NodeData::new(0, NodeType::Entity, "C".to_string()));
        let d = graph.add_node(NodeData::new(0, NodeType::Entity, "D".to_string()));
        
        graph.add_edge(a, b, EdgeData::new(EdgeType::RelatedTo));
        graph.add_edge(b, c, EdgeData::new(EdgeType::RelatedTo));
        graph.add_edge(c, d, EdgeData::new(EdgeType::RelatedTo));
        
        let visited = graph.traverse(a, 2);
        assert!(visited.contains(&a));
        assert!(visited.contains(&b));
        assert!(visited.contains(&c));
        assert!(!visited.contains(&d)); // Depth 2 shouldn't reach D
    }

    #[test]
    fn test_find_paths() {
        let graph = ShardedKnowledgeGraph::new();
        
        // Create multiple paths: A -> B -> D, A -> C -> D
        let a = graph.add_node(NodeData::new(0, NodeType::Entity, "A".to_string()));
        let b = graph.add_node(NodeData::new(0, NodeType::Entity, "B".to_string()));
        let c = graph.add_node(NodeData::new(0, NodeType::Entity, "C".to_string()));
        let d = graph.add_node(NodeData::new(0, NodeType::Entity, "D".to_string()));
        
        graph.add_edge(a, b, EdgeData::new(EdgeType::RelatedTo));
        graph.add_edge(a, c, EdgeData::new(EdgeType::RelatedTo));
        graph.add_edge(b, d, EdgeData::new(EdgeType::RelatedTo));
        graph.add_edge(c, d, EdgeData::new(EdgeType::RelatedTo));
        
        let paths = graph.find_paths(a, d, 3);
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;
        
        let graph = Arc::new(ShardedKnowledgeGraph::new());
        let mut handles = vec![];
        
        // Spawn multiple threads adding nodes
        for t in 0..10 {
            let graph_clone = graph.clone();
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let node_data = NodeData::new(0, NodeType::Entity, format!("Node_{}_{}", t, i));
                    graph_clone.add_node(node_data);
                }
            }));
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let stats = graph.get_stats();
        assert_eq!(stats.total_nodes, 1000);
    }
}
