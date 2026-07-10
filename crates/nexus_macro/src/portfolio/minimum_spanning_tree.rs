//! Minimum Spanning Tree (MST) builder using Prim's algorithm.
//!
//! Used for Hierarchical Risk Parity to cluster correlated assets.
//! Handles disconnected graphs and invalid distances gracefully.

use ndarray::Array2;

/// Edge in the MST
#[derive(Debug, Clone)]
pub struct MstEdge {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
}

/// Result from MST construction
pub type MstResult = Result<Vec<MstEdge>, String>;

/// Minimum Spanning Tree builder using Prim's algorithm
pub struct MinimumSpanningTree {
    n_nodes: usize,
    /// Pre-allocated distance array
    min_dist: Vec<f64>,
    /// Pre-allocated parent array
    parent: Vec<usize>,
    /// Pre-allocated visited array
    visited: Vec<bool>,
}

impl MinimumSpanningTree {
    /// Create new MST builder for n nodes
    pub fn new(n_nodes: usize) -> Self {
        Self {
            n_nodes,
            min_dist: vec![f64::INFINITY; n_nodes],
            parent: vec![0; n_nodes],
            visited: vec![false; n_nodes],
        }
    }

    /// Build MST from distance matrix using Prim's algorithm
    /// 
    /// # Arguments
    /// * `dist_matrix` - Symmetric distance matrix [N x N]
    /// 
    /// # Returns
    /// Vector of MST edges, or error if graph is disconnected
    pub fn build(&mut self, dist_matrix: &Array2<f64>) -> MstResult {
        let n = dist_matrix.nrows();
        
        if dist_matrix.ncols() != n {
            return Err("Distance matrix must be square".to_string());
        }

        // Reset state
        self.min_dist.fill(f64::INFINITY);
        self.parent.fill(0);
        self.visited.fill(false);

        // Start from node 0
        self.min_dist[0] = 0.0;
        self.parent[0] = 0;

        let mut edges = Vec::with_capacity(n - 1);
        let mut nodes_added = 0usize;

        for _ in 0..n {
            // Find unvisited node with minimum distance
            let u = self.find_min_unvisited();
            
            if u.is_none() {
                break; // No more reachable nodes
            }
            
            let u = u.unwrap();
            self.visited[u] = true;
            nodes_added += 1;

            // Add edge to MST (except for root)
            if self.parent[u] != u {
                edges.push(MstEdge {
                    from: self.parent[u],
                    to: u,
                    weight: self.min_dist[u],
                });
            }

            // Update distances to neighbors
            for v in 0..n {
                if !self.visited[v] {
                    let dist = dist_matrix[[u, v]];
                    
                    // Handle invalid distances (NaN, Inf, negative)
                    if dist.is_finite() && dist >= 0.0 && dist < self.min_dist[v] {
                        self.min_dist[v] = dist;
                        self.parent[v] = u;
                    }
                }
            }
        }

        // Check for disconnected components
        if nodes_added < n {
            // Graph is disconnected - add remaining nodes with zero-weight edges
            // to connect them to the nearest component
            self.connect_disconnected_components(&mut edges, dist_matrix)?;
        }

        Ok(edges)
    }

    /// Find unvisited node with minimum distance
    fn find_min_unvisited(&self) -> Option<usize> {
        let mut min_node = None;
        let mut min_val = f64::INFINITY;

        for i in 0..self.n_nodes {
            if !self.visited[i] && self.min_dist[i] < min_val {
                min_val = self.min_dist[i];
                min_node = Some(i);
            }
        }

        min_node
    }

    /// Connect disconnected components by finding nearest neighbor
    fn connect_disconnected_components(
        &mut self,
        edges: &mut Vec<MstEdge>,
        dist_matrix: &Array2<f64>,
    ) -> Result<(), String> {
        let n = self.n_nodes;

        // Find all connected components
        let mut components: Vec<Vec<usize>> = Vec::new();
        let mut component_visited = vec![false; n];

        // Build adjacency from existing edges
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for edge in edges {
            adj[edge.from].push(edge.to);
            adj[edge.to].push(edge.from);
        }

        // DFS to find components
        for start in 0..n {
            if !component_visited[start] {
                let mut component = Vec::new();
                self.dfs_component(start, &adj, &mut component_visited, &mut component);
                components.push(component);
            }
        }

        // Connect components using minimum distance edges
        while components.len() > 1 {
            let mut best_dist = f64::INFINITY;
            let mut best_edge: Option<(usize, usize, usize, usize)> = None; // (comp_i, comp_j, node_i, node_j)

            for i in 0..components.len() {
                for j in (i + 1)..components.len() {
                    for &node_i in &components[i] {
                        for &node_j in &components[j] {
                            let dist = dist_matrix[[node_i, node_j]];
                            if dist.is_finite() && dist >= 0.0 && dist < best_dist {
                                best_dist = dist;
                                best_edge = Some((i, j, node_i, node_j));
                            }
                        }
                    }
                }
            }

            if let Some((comp_i, comp_j, node_i, node_j)) = best_edge {
                // Add connecting edge
                edges.push(MstEdge {
                    from: node_i,
                    to: node_j,
                    weight: best_dist,
                });

                // Merge components
                let merged = components.remove(comp_j);
                components[comp_i].extend(merged);
            } else {
                // No valid connection found - use zero-weight edge
                let node_i = components[0][0];
                let node_j = components[1][0];
                edges.push(MstEdge {
                    from: node_i,
                    to: node_j,
                    weight: 0.0,
                });
                
                let merged = components.remove(1);
                components[0].extend(merged);
            }
        }

        Ok(())
    }

    /// DFS to find connected component
    fn dfs_component(
        &self,
        node: usize,
        adj: &[Vec<usize>],
        visited: &mut [bool],
        component: &mut Vec<usize>,
    ) {
        if visited[node] {
            return;
        }
        visited[node] = true;
        component.push(node);

        for neighbor in &adj[node] {
            self.dfs_component(*neighbor, adj, visited, component);
        }
    }

    /// Get total MST weight
    pub fn total_weight(edges: &[MstEdge]) -> f64 {
        edges.iter().map(|e| e.weight).sum()
    }

    /// Validate MST structure
    pub fn validate_mst(edges: &[MstEdge], n_nodes: usize) -> Result<(), String> {
        if n_nodes == 0 {
            return Ok(()); // Empty graph is valid
        }

        if edges.len() < n_nodes - 1 {
            return Err(format!(
                "MST should have {} edges, got {}",
                n_nodes - 1,
                edges.len()
            ));
        }

        // Check for cycles using union-find
        let mut parent: Vec<usize> = (0..n_nodes).collect();

        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        fn union(parent: &mut [usize], x: usize, y: usize) -> bool {
            let px = find(parent, x);
            let py = find(parent, y);
            if px == py {
                return false; // Already connected (cycle)
            }
            parent[px] = py;
            true
        }

        for edge in edges {
            if edge.from >= n_nodes || edge.to >= n_nodes {
                return Err(format!("Invalid node index in edge: {} -> {}", edge.from, edge.to));
            }
            
            if !union(&mut parent, edge.from, edge.to) {
                return Err("Cycle detected in MST".to_string());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mst_basic() {
        // Simple triangle graph
        let dist = Array2::from_shape_vec((3, 3), vec![
            0.0, 1.0, 2.0,
            1.0, 0.0, 1.5,
            2.0, 1.5, 0.0,
        ]).unwrap();

        let mut mst = MinimumSpanningTree::new(3);
        let edges = mst.build(&dist).unwrap();

        assert_eq!(edges.len(), 2); // n-1 edges for connected graph
        
        // Validate no cycles
        assert!(MinimumSpanningTree::validate_mst(&edges, 3).is_ok());
    }

    #[test]
    fn test_mst_with_nan() {
        // Graph with NaN distances (should be treated as infinity)
        let dist = Array2::from_shape_vec((3, 3), vec![
            0.0, f64::NAN, 2.0,
            f64::NAN, 0.0, 1.5,
            2.0, 1.5, 0.0,
        ]).unwrap();

        let mut mst = MinimumSpanningTree::new(3);
        let edges = mst.build(&dist).unwrap();

        // Should still produce valid MST using available edges
        assert!(MinimumSpanningTree::validate_mst(&edges, 3).is_ok());
    }

    #[test]
    fn test_mst_disconnected() {
        // Disconnected graph (one node isolated)
        let dist = Array2::from_shape_vec((3, 3), vec![
            0.0, 1.0, f64::INFINITY,
            1.0, 0.0, f64::INFINITY,
            f64::INFINITY, f64::INFINITY, 0.0,
        ]).unwrap();

        let mut mst = MinimumSpanningTree::new(3);
        let edges = mst.build(&dist).unwrap();

        // Should connect all nodes via fallback mechanism
        assert_eq!(edges.len(), 2);
    }
}
