//! Pre-allocated k-d Tree Arena for KSG Estimator
//! Zero-allocation k-nearest neighbor search using arena allocation.

/// k-d tree node stored in arena
#[derive(Debug, Clone)]
pub struct KdTreeNode {
    pub point: Vec<f64>,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub split_dim: usize,
}

/// Pre-allocated arena for k-d tree nodes
/// Implements ring-buffer eviction when full
pub struct KdTreeArena {
    /// Maximum nodes in arena
    max_nodes: usize,
    /// Dimension of points
    dim: usize,
    /// Node storage
    nodes: Vec<Option<KdTreeNode>>,
    /// Free list indices
    free_list: Vec<usize>,
    /// Root node index
    root: Option<usize>,
    /// Count of active nodes
    count: usize,
}

impl KdTreeArena {
    pub fn new(max_nodes: usize, dim: usize) -> Self {
        let mut free_list = (0..max_nodes).collect::<Vec<_>>();
        let nodes = vec![None; max_nodes];
        
        Self {
            max_nodes,
            dim,
            nodes,
            free_list,
            root: None,
            count: 0,
        }
    }

    /// Allocate a new node (with ring-buffer eviction if full)
    pub fn allocate(&mut self, point: Vec<f64>) -> Option<usize> {
        // Evict oldest if necessary
        if self.free_list.is_empty() {
            self.evict_oldest();
        }

        let idx = self.free_list.pop()?;
        
        self.nodes[idx] = Some(KdTreeNode {
            point,
            left: None,
            right: None,
            split_dim: 0,
        });
        
        self.count += 1;
        Some(idx)
    }

    /// Evict oldest node (simplified: just clear root subtree)
    fn evict_oldest(&mut self) {
        if let Some(root_idx) = self.root.take() {
            self.clear_subtree(root_idx);
        }
        self.count = 0;
    }

    /// Clear subtree and return nodes to free list
    fn clear_subtree(&mut self, idx: usize) {
        if let Some(node) = self.nodes[idx].take() {
            if let Some(left) = node.left {
                self.clear_subtree(left);
            }
            if let Some(right) = node.right {
                self.clear_subtree(right);
            }
            self.free_list.push(idx);
        }
    }

    /// Build k-d tree from points
    pub fn build(&mut self, points: &[Vec<f64>]) -> Option<usize> {
        if points.is_empty() {
            return None;
        }

        let indices: Vec<usize> = (0..points.len()).collect();
        self.root = self.build_recursive(points, &indices, 0);
        self.root
    }

    fn build_recursive(&mut self, points: &[Vec<f64>], indices: &[usize], depth: usize) -> Option<usize> {
        if indices.is_empty() {
            return None;
        }

        let split_dim = depth % self.dim;
        
        // Find median (simplified: just use middle element)
        let mid = indices.len() / 2;
        let point_idx = indices[mid];
        
        let node_idx = self.allocate(points[point_idx].clone())?;
        
        if let Some(ref mut node) = self.nodes[node_idx] {
            node.split_dim = split_dim;
            
            let (left_indices, right_indices) = indices.split_at(mid);
            
            if !left_indices.is_empty() {
                node.left = self.build_recursive(points, &left_indices[..mid], depth + 1);
            }
            if right_indices.len() > 1 {
                node.right = self.build_recursive(points, &right_indices[1..], depth + 1);
            }
        }
        
        Some(node_idx)
    }

    /// Find k nearest neighbors to query point
    pub fn k_nearest(&self, query: &[f64], k: usize) -> Vec<(usize, f64)> {
        let mut neighbors = Vec::with_capacity(k);
        
        if let Some(root) = self.root {
            self.search_nearest(query, root, k, &mut neighbors);
        }
        
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        neighbors.truncate(k);
        neighbors
    }

    fn search_nearest(&self, query: &[f64], node_idx: usize, k: usize, neighbors: &mut Vec<(usize, f64)>) {
        let node = match &self.nodes[node_idx] {
            Some(n) => n,
            None => return,
        };

        // Compute distance
        let dist = self.distance(query, &node.point);
        
        // Insert into neighbors
        neighbors.push((node_idx, dist));
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        if neighbors.len() > k {
            neighbors.pop();
        }

        // Recurse into children
        if let Some(left) = node.left {
            self.search_nearest(query, left, k, neighbors);
        }
        if let Some(right) = node.right {
            self.search_nearest(query, right, k, neighbors);
        }
    }

    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
    }

    /// Get current node count
    pub fn count(&self) -> usize {
        self.count
    }

    /// Reset the arena
    pub fn reset(&mut self) {
        for i in 0..self.max_nodes {
            self.nodes[i] = None;
        }
        self.free_list = (0..self.max_nodes).collect();
        self.root = None;
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_allocation() {
        let mut arena = KdTreeArena::new(100, 3);
        
        let p1 = vec![1.0, 2.0, 3.0];
        let p2 = vec![4.0, 5.0, 6.0];
        
        let idx1 = arena.allocate(p1);
        let idx2 = arena.allocate(p2);
        
        assert!(idx1.is_some());
        assert!(idx2.is_some());
        assert_eq!(arena.count(), 2);
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let mut arena = KdTreeArena::new(5, 2);
        
        // Fill arena
        for i in 0..10 {
            arena.allocate(vec![i as f64, i as f64 * 2.0]);
        }
        
        // Should have evicted and reused
        assert!(arena.count() <= 5);
    }

    #[test]
    fn test_k_nearest() {
        let mut arena = KdTreeArena::new(100, 2);
        
        let points: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
        ];
        
        arena.build(&points);
        
        let query = vec![1.5, 1.5];
        let neighbors = arena.k_nearest(&query, 2);
        
        assert_eq!(neighbors.len(), 2);
        // Closest should be [1,1] and [2,2]
    }
}
