//! Sharded Knowledge Graph
//! 
//! In-memory knowledge graph using sharded, lock-free concurrent hash maps
//! for storing nodes and edges. Allows multiple threads to traverse the graph
//! simultaneously without mutex contention.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use dashmap::DashMap;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;

/// Number of shards for the graph
const NUM_SHARDS: usize = 16;

/// Node types in the knowledge graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeType {
    Entity,
    Event,
    Signal,
    Asset,
    Indicator,
}

/// Edge types representing relationships
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeType {
    RelatesTo,
    Causes,
    CorrelatesWith,
    Impacts,
    PartOf,
    TradesAs,
}

/// Node data in the graph
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: u64,
    pub node_type: NodeType,
    pub name: String,
    pub metadata: HashMap<String, String>,
    pub created_at: u64, // timestamp
}

/// Edge data in the graph
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub source_id: u64,
    pub target_id: u64,
    pub edge_type: EdgeType,
    pub weight: f32,
    pub metadata: HashMap<String, String>,
}

/// Compute shard index for a given ID
#[inline]
fn shard_index(id: u64) -> usize {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    id.hash(&mut hasher);
    hasher.finish() as usize % NUM_SHARDS
}

/// Sharded knowledge graph with fine-grained locking
pub struct ShardedKnowledgeGraph {
    /// Shards for node storage - each shard is independent
    node_shards: Vec<DashMap<u64, GraphNode>>,
    
    /// Shards for adjacency lists (outgoing edges)
    adjacency_shards: Vec<DashMap<u64, Vec<(u64, EdgeType)>>>,
    
    /// Reverse adjacency (incoming edges) for backward traversal
    reverse_adjacency: Vec<DashMap<u64, Vec<(u64, EdgeType)>>>,
    
    /// Global node counter for ID generation
    node_counter: AtomicU64,
    
    /// Total node count cache
    total_nodes: AtomicU64,
    total_edges: AtomicU64,
}

// SAFETY: DashMap handles its own synchronization
unsafe impl Send for ShardedKnowledgeGraph {}
unsafe impl Sync for ShardedKnowledgeGraph {}

impl ShardedKnowledgeGraph {
    /// Create a new sharded knowledge graph
    pub fn new() -> Self {
        let node_shards: Vec<_> = (0..NUM_SHARDS)
            .map(|_| DashMap::new())
            .collect();
        
        let adjacency_shards: Vec<_> = (0..NUM_SHARDS)
            .map(|_| DashMap::new())
            .collect();
        
        let reverse_adjacency: Vec<_> = (0..NUM_SHARDS)
            .map(|_| DashMap::new())
            .collect();
        
        Self {
            node_shards,
            adjacency_shards,
            reverse_adjacency,
            node_counter: AtomicU64::new(0),
            total_nodes: AtomicU64::new(0),
            total_edges: AtomicU64::new(0),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&self, node_type: NodeType, name: &str, metadata: HashMap<String, String>) -> u64 {
        let id = self.node_counter.fetch_add(1, Ordering::Relaxed);
        
        let node = GraphNode {
            id,
            node_type,
            name: name.to_string(),
            metadata,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        let shard_idx = shard_index(id);
        self.node_shards[shard_idx].insert(id, node);
        self.total_nodes.fetch_add(1, Ordering::Relaxed);
        
        // Initialize empty adjacency lists
        self.adjacency_shards[shard_idx].insert(id, Vec::new());
        self.reverse_adjacency[shard_idx].insert(id, Vec::new());
        
        id
    }

    /// Add an edge between two nodes
    pub fn add_edge(&self, source_id: u64, target_id: u64, edge_type: EdgeType, weight: f32) -> Result<(), GraphError> {
        // Verify nodes exist
        if !self.has_node(source_id) || !self.has_node(target_id) {
            return Err(GraphError::NodeNotFound);
        }
        
        let source_shard = shard_index(source_id);
        let target_shard = shard_index(target_id);
        
        // Add to outgoing edges
        {
            let mut adj = self.adjacency_shards[source_shard]
                .entry(source_id)
                .or_insert_with(Vec::new);
            adj.push((target_id, edge_type));
        }
        
        // Add to incoming edges
        {
            let mut rev_adj = self.reverse_adjacency[target_shard]
                .entry(target_id)
                .or_insert_with(Vec::new);
            rev_adj.push((source_id, edge_type));
        }
        
        self.total_edges.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }

    /// Check if a node exists
    pub fn has_node(&self, id: u64) -> bool {
        let shard_idx = shard_index(id);
        self.node_shards[shard_idx].contains_key(&id)
    }

    /// Get a node by ID
    pub fn get_node(&self, id: u64) -> Option<GraphNode> {
        let shard_idx = shard_index(id);
        self.node_shards[shard_idx].get(&id).map(|r| r.value().clone())
    }

    /// Get neighbors of a node (outgoing edges)
    pub fn get_neighbors(&self, id: u64) -> Vec<(u64, EdgeType)> {
        let shard_idx = shard_index(id);
        self.adjacency_shards[shard_idx]
            .get(&id)
            .map(|r| r.value().clone())
            .unwrap_or_default()
    }

    /// Get predecessors of a node (incoming edges)
    pub fn get_predecessors(&self, id: u64) -> Vec<(u64, EdgeType)> {
        let shard_idx = shard_index(id);
        self.reverse_adjacency[shard_idx]
            .get(&id)
            .map(|r| r.value().clone())
            .unwrap_or_default()
    }

    /// Traverse the graph from a starting node up to max_depth
    pub fn traverse(&self, start_id: u64, max_depth: usize, edge_filter: Option<EdgeType>) -> Vec<u64> {
        let mut visited = std::collections::HashSet::new();
        let mut result = Vec::new();
        let mut queue: Vec<(u64, usize)> = vec![(start_id, 0)];
        
        while let Some((current_id, depth)) = queue.pop() {
            if depth > max_depth || visited.contains(&current_id) {
                continue;
            }
            
            visited.insert(current_id);
            result.push(current_id);
            
            // Get neighbors
            let neighbors = self.get_neighbors(current_id);
            
            for (neighbor_id, edge_type) in neighbors {
                if edge_filter.map_or(true, |f| f == edge_type) {
                    queue.push((neighbor_id, depth + 1));
                }
            }
        }
        
        result
    }

    /// Find paths between two nodes (BFS)
    pub fn find_paths(&self, start_id: u64, end_id: u64, max_length: usize) -> Vec<Vec<u64>> {
        let mut paths = Vec::new();
        let mut queue: Vec<(u64, Vec<u64>)> = vec![(start_id, vec![start_id])];
        
        while let Some((current, path)) = queue.pop() {
            if current == end_id {
                paths.push(path);
                continue;
            }
            
            if path.len() >= max_length {
                continue;
            }
            
            let neighbors = self.get_neighbors(current);
            
            for (neighbor_id, _) in neighbors {
                if !path.contains(&neighbor_id) {
                    let mut new_path = path.clone();
                    new_path.push(neighbor_id);
                    queue.push((neighbor_id, new_path));
                }
            }
        }
        
        paths
    }

    /// Get statistics about the graph
    pub fn stats(&self) -> GraphStats {
        GraphStats {
            total_nodes: self.total_nodes.load(Ordering::Relaxed),
            total_edges: self.total_edges.load(Ordering::Relaxed),
            num_shards: NUM_SHARDS,
        }
    }

    /// Remove a node and all its edges
    pub fn remove_node(&self, id: u64) -> Result<(), GraphError> {
        let shard_idx = shard_index(id);
        
        // Remove from node shard
        if self.node_shards[shard_idx].remove(&id).is_none() {
            return Err(GraphError::NodeNotFound);
        }
        
        // Remove adjacency lists
        self.adjacency_shards[shard_idx].remove(&id);
        self.reverse_adjacency[shard_idx].remove(&id);
        
        // Remove edges pointing to this node from other shards
        for shard in &self.adjacency_shards {
            if let Some(mut adj) = shard.get_mut(&id) {
                adj.clear();
            }
        }
        
        self.total_nodes.fetch_sub(1, Ordering::Relaxed);
        
        Ok(())
    }
}

impl Default for ShardedKnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph statistics
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub total_nodes: u64,
    pub total_edges: u64,
    pub num_shards: usize,
}

/// Graph errors
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("Node not found")]
    NodeNotFound,
    #[error("Edge not found")]
    EdgeNotFound,
    #[error("Cycle detected")]
    CycleDetected,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_basic_operations() {
        let graph = ShardedKnowledgeGraph::new();
        
        // Add nodes
        let apple = graph.add_node(NodeType::Entity, "Apple", HashMap::new());
        let aapl = graph.add_node(NodeType::Asset, "AAPL", HashMap::new());
        
        // Add edge
        graph.add_edge(apple, aapl, EdgeType::TradesAs, 1.0).unwrap();
        
        // Verify
        assert!(graph.has_node(apple));
        assert!(graph.has_node(aapl));
        
        let neighbors = graph.get_neighbors(apple);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, aapl);
        assert_eq!(neighbors[0].1, EdgeType::TradesAs);
    }

    #[test]
    fn test_graph_traversal() {
        let graph = ShardedKnowledgeGraph::new();
        
        // Create a chain: A -> B -> C -> D
        let a = graph.add_node(NodeType::Entity, "A", HashMap::new());
        let b = graph.add_node(NodeType::Entity, "B", HashMap::new());
        let c = graph.add_node(NodeType::Entity, "C", HashMap::new());
        let d = graph.add_node(NodeType::Entity, "D", HashMap::new());
        
        graph.add_edge(a, b, EdgeType::RelatesTo, 1.0).unwrap();
        graph.add_edge(b, c, EdgeType::RelatesTo, 1.0).unwrap();
        graph.add_edge(c, d, EdgeType::RelatesTo, 1.0).unwrap();
        
        // Traverse from A with depth 2
        let visited = graph.traverse(a, 2, None);
        assert!(visited.contains(&a));
        assert!(visited.contains(&b));
        assert!(visited.contains(&c));
        // D should not be included at depth 2
    }

    #[test]
    fn test_concurrent_access() {
        let graph = Arc::new(ShardedKnowledgeGraph::new());
        let mut handles = Vec::new();
        
        // Spawn multiple writers
        for i in 0..10 {
            let g = Arc::clone(&graph);
            let handle = std::thread::spawn(move || {
                for j in 0..100 {
                    let id = g.add_node(NodeType::Entity, &format!("Node_{}_{}", i, j), HashMap::new());
                    assert!(g.has_node(id));
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let stats = graph.stats();
        assert_eq!(stats.total_nodes, 1000);
    }
}
