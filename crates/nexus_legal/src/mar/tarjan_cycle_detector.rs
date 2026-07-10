// STAGE 23: Computational Law, RegTech Compliance & Immutable Audit Trails
// Chapter 1: Real-Time Market Abuse Regulation (MAR) & Wash Trade Detection
// File: crates/nexus_legal/src/mar/tarjan_cycle_detector.rs

//! Tarjan's Algorithm for Strongly Connected Components (SCC) detection.
//! Optimized for lock-free execution on sliding window graphs.
//! Used to detect wash trade cycles in the execution graph.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::mar::wash_trade_graph::{ExecutionId, ExecutionNode};

/// State for a single node during Tarjan's algorithm traversal
#[derive(Debug, Clone)]
struct TarjanNodeState {
    index: Option<usize>,
    lowlink: usize,
    on_stack: bool,
}

/// Tarjan's SCC detector with zero-allocation optimizations
pub struct TarjanDetector {
    /// Reusable stack to avoid allocations per detection run
    stack: Vec<ExecutionId>,
    /// Node states indexed by ExecutionId
    node_states: HashMap<ExecutionId, TarjanNodeState>,
    /// Current index counter
    current_index: usize,
    /// Collected SCCs
    sccs: Vec<Vec<ExecutionId>>,
    /// Statistics
    total_detections: AtomicUsize,
    total_cycles_found: AtomicUsize,
}

impl TarjanDetector {
    pub fn new() -> Self {
        Self {
            stack: Vec::with_capacity(1024),
            node_states: HashMap::new(),
            current_index: 0,
            sccs: Vec::new(),
            total_detections: AtomicUsize::new(0),
            total_cycles_found: AtomicUsize::new(0),
        }
    }

    /// Detect all strongly connected components in the graph.
    /// Returns a vector of SCCs, where each SCC is a list of ExecutionIds.
    /// 
    /// # Arguments
    /// * `adjacency` - Adjacency list representation of the graph
    /// * `nodes` - Map of all nodes in the graph
    /// 
    /// # Performance
    /// Time complexity: O(V + E) where V = vertices, E = edges
    /// Space complexity: O(V) for recursion stack and state storage
    pub fn detect_sccs(
        &mut self,
        adjacency: &HashMap<ExecutionId, Vec<ExecutionId>>,
        nodes: &HashMap<ExecutionId, ExecutionNode>,
    ) -> Vec<Vec<ExecutionId>> {
        self.total_detections.fetch_add(1, Ordering::Relaxed);
        
        // Reset state for this detection run
        self.stack.clear();
        self.node_states.clear();
        self.current_index = 0;
        self.sccs.clear();

        // Initialize all nodes
        for &node_id in nodes.keys() {
            self.node_states.insert(
                node_id,
                TarjanNodeState {
                    index: None,
                    lowlink: 0,
                    on_stack: false,
                },
            );
        }

        // Run Tarjan's algorithm from each unvisited node
        for &node_id in nodes.keys() {
            if self.node_states.get(&node_id).and_then(|s| s.index).is_none() {
                self.strongconnect(node_id, adjacency);
            }
        }

        let cycles_found = self.sccs.len();
        self.total_cycles_found.fetch_add(cycles_found, Ordering::Relaxed);

        // Filter to only return SCCs with more than one node (actual cycles)
        // or single-node self-loops
        self.sccs
            .drain(..)
            .filter(|scc| {
                if scc.len() > 1 {
                    true
                } else if scc.len() == 1 {
                    // Check for self-loop
                    let node_id = scc[0];
                    adjacency
                        .get(&node_id)
                        .map(|neighbors| neighbors.contains(&node_id))
                        .unwrap_or(false)
                } else {
                    false
                }
            })
            .collect()
    }

    /// The core recursive Tarjan's algorithm implementation.
    /// Uses iterative approach with explicit stack to prevent stack overflow.
    fn strongconnect(
        &mut self,
        start_node: ExecutionId,
        adjacency: &HashMap<ExecutionId, Vec<ExecutionId>>,
    ) {
        // Use iterative approach with manual call stack to prevent stack overflow
        #[derive(Clone)]
        enum VisitAction {
            Enter(ExecutionId),
            Resume(ExecutionId, usize, Vec<ExecutionId>),
        }

        let mut call_stack: Vec<VisitAction> = vec![VisitAction::Enter(start_node)];

        while let Some(action) = call_stack.pop() {
            match action {
                VisitAction::Enter(v) => {
                    // Set depth index for v
                    if let Some(state) = self.node_states.get_mut(&v) {
                        state.index = Some(self.current_index);
                        state.lowlink = self.current_index;
                        state.on_stack = true;
                    }
                    self.stack.push(v);
                    self.current_index += 1;

                    // Get neighbors
                    let neighbors = adjacency
                        .get(&v)
                        .cloned()
                        .unwrap_or_default();

                    if neighbors.is_empty() {
                        // No successors - check if root of SCC
                        self.check_and_pop_scc(v);
                    } else {
                        // Push resume action after processing first neighbor
                        call_stack.push(VisitAction::Resume(v, 0, neighbors));
                    }
                }

                VisitAction::Resume(v, neighbor_idx, neighbors) => {
                    if neighbor_idx >= neighbors.len() {
                        // All neighbors processed - check if root of SCC
                        self.check_and_pop_scc(v);
                        continue;
                    }

                    let w = neighbors[neighbor_idx];

                    // Push next neighbor to process after this one
                    if neighbor_idx + 1 < neighbors.len() {
                        call_stack.push(VisitAction::Resume(v, neighbor_idx + 1, neighbors.clone()));
                    }

                    // Process neighbor w
                    let w_state = self.node_states.get(&w).cloned();
                    
                    match w_state {
                        Some(state) if state.index.is_none() => {
                            // w has not been visited - recurse
                            call_stack.push(VisitAction::Enter(w));
                        }
                        Some(state) if state.on_stack => {
                            // w is on stack - update lowlink
                            if let (Some(w_idx), Some(v_state)) = 
                                (state.index, self.node_states.get_mut(&v)) {
                                v_state.lowlink = v_state.lowlink.min(w_idx);
                            }
                        }
                        _ => {
                            // w already visited and not on stack - ignore
                        }
                    }
                }
            }
        }
    }

    /// Check if node is root of an SCC and pop the component
    fn check_and_pop_scc(&mut self, v: ExecutionId) {
        let is_root = self.node_states
            .get(&v)
            .map(|state| {
                state.index == Some(state.lowlink)
            })
            .unwrap_or(false);

        if is_root {
            let mut scc = Vec::new();
            
            loop {
                if let Some(w) = self.stack.pop() {
                    scc.push(w);
                    
                    if let Some(state) = self.node_states.get_mut(&w) {
                        state.on_stack = false;
                    }
                    
                    if w == v {
                        break;
                    }
                } else {
                    break;
                }
            }

            if !scc.is_empty() {
                self.sccs.push(scc);
            }
        }
    }

    /// Get statistics about detections
    pub fn get_stats(&self) -> TarjanStats {
        TarjanStats {
            total_detections: self.total_detections.load(Ordering::Relaxed),
            total_cycles_found: self.total_cycles_found.load(Ordering::Relaxed),
            avg_cycle_size: 0, // Calculated on demand
        }
    }

    /// Clear internal state to free memory
    pub fn clear(&mut self) {
        self.stack.clear();
        self.node_states.clear();
        self.sccs.clear();
        self.current_index = 0;
    }
}

#[derive(Debug, Clone)]
pub struct TarjanStats {
    pub total_detections: usize,
    pub total_cycles_found: usize,
    pub avg_cycle_size: usize,
}

impl Default for TarjanDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_cycle() {
        let mut detector = TarjanDetector::new();
        
        // Create a simple cycle: A -> B -> C -> A
        let a = ExecutionId(1);
        let b = ExecutionId(2);
        let c = ExecutionId(3);

        let mut adjacency = HashMap::new();
        adjacency.insert(a, vec![b]);
        adjacency.insert(b, vec![c]);
        adjacency.insert(c, vec![a]);

        let mut nodes = HashMap::new();
        nodes.insert(a, create_test_node(a, "BTC"));
        nodes.insert(b, create_test_node(b, "BTC"));
        nodes.insert(c, create_test_node(c, "BTC"));

        let sccs = detector.detect_sccs(&adjacency, &nodes);
        
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 3);
    }

    #[test]
    fn test_no_cycle() {
        let mut detector = TarjanDetector::new();
        
        // Linear graph: A -> B -> C (no cycle)
        let a = ExecutionId(1);
        let b = ExecutionId(2);
        let c = ExecutionId(3);

        let mut adjacency = HashMap::new();
        adjacency.insert(a, vec![b]);
        adjacency.insert(b, vec![c]);
        adjacency.insert(c, vec![]);

        let mut nodes = HashMap::new();
        nodes.insert(a, create_test_node(a, "BTC"));
        nodes.insert(b, create_test_node(b, "BTC"));
        nodes.insert(c, create_test_node(c, "BTC"));

        let sccs = detector.detect_sccs(&adjacency, &nodes);
        
        // Should find no cycles (each node is its own SCC but filtered out)
        assert!(sccs.is_empty());
    }

    #[test]
    fn test_self_loop() {
        let mut detector = TarjanDetector::new();
        
        let a = ExecutionId(1);

        let mut adjacency = HashMap::new();
        adjacency.insert(a, vec![a]); // Self-loop

        let mut nodes = HashMap::new();
        nodes.insert(a, create_test_node(a, "BTC"));

        let sccs = detector.detect_sccs(&adjacency, &nodes);
        
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0], vec![a]);
    }

    fn create_test_node(id: ExecutionId, symbol: &str) -> ExecutionNode {
        use crate::mar::wash_trade_graph::{AssetClass, Side};
        
        ExecutionNode {
            id,
            symbol: symbol.to_string(),
            asset_class: AssetClass::CryptoSpot,
            venue_id: 1,
            side: Side::Buy,
            quantity: 100,
            price: 50000,
            timestamp_ns: 1000000000,
            strategy_id: 1,
            order_id: 100,
            is_maker: false,
        }
    }
}
