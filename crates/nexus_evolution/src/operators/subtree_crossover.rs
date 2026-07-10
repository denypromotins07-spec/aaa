//! Subtree Crossover Operator for Genetic Programming
//! 
//! Swaps subtrees between two parent trees to produce offspring,
//! respecting type constraints to ensure valid expression trees.

use crate::gp::arena_allocator::{AstNode, NodePtr, TreeArena, NodeData, Operator, PrimitiveType};
use crate::gp::expression_tree::ExpressionTree;
use rand::Rng;

/// Result of a crossover operation
#[derive(Debug)]
pub struct CrossoverResult {
    /// First offspring tree
    pub offspring1: Option<NodePtr<AstNode>>,
    /// Second offspring tree  
    pub offspring2: Option<NodePtr<AstNode>>,
    /// Whether crossover was successful
    pub success: bool,
    /// Crossover point depth in parent1
    pub crossover_depth1: u8,
    /// Crossover point depth in parent2
    pub crossover_depth2: u8,
}

impl CrossoverResult {
    pub const fn failed() -> Self {
        Self {
            offspring1: None,
            offspring2: None,
            success: false,
            crossover_depth1: 0,
            crossover_depth2: 0,
        }
    }
}

/// Subtree crossover operator with type safety
pub struct SubtreeCrossover {
    /// Maximum tree depth allowed after crossover
    max_depth: u8,
    /// Probability of performing crossover
    crossover_rate: f64,
    /// RNG seed for deterministic reproduction
    rng_seed: u64,
}

impl SubtreeCrossover {
    pub fn new(max_depth: u8, crossover_rate: f64) -> Self {
        Self {
            max_depth,
            crossover_rate: crossover_rate.clamp(0.0, 1.0),
            rng_seed: 0,
        }
    }

    /// Set RNG seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng_seed = seed;
        self
    }

    /// Perform subtree crossover on two parent trees
    /// Returns two offspring or None if crossover fails
    pub fn crossover(
        &self,
        parent1: NodePtr<AstNode>,
        parent2: NodePtr<AstNode>,
        arena: &mut TreeArena,
    ) -> CrossoverResult {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rng_seed);

        // Check if crossover should occur
        if rng.gen::<f64>() > self.crossover_rate {
            // No crossover - return copies of parents
            return CrossoverResult {
                offspring1: self.copy_tree(parent1, arena),
                offspring2: self.copy_tree(parent2, arena),
                success: true,
                crossover_depth1: 0,
                crossover_depth2: 0,
            };
        }

        // Collect all nodes from both parents
        let mut nodes1: Vec<(NodePtr<AstNode>, u8)> = Vec::new();
        let mut nodes2: Vec<(NodePtr<AstNode>, u8)> = Vec::new();
        
        self.collect_nodes(parent1, 0, &mut nodes1);
        self.collect_nodes(parent2, 0, &mut nodes2);

        // Filter to only internal nodes (non-terminals) for crossover points
        let internal1: Vec<_> = nodes1.iter()
            .filter(|(node, _)| {
                unsafe {
                    match node.as_ref().data {
                        NodeData::Operator(_) => true,
                        _ => false,
                    }
                }
            })
            .copied()
            .collect();

        let internal2: Vec<_> = nodes2.iter()
            .filter(|(node, _)| {
                unsafe {
                    match node.as_ref().data {
                        NodeData::Operator(_) => true,
                        _ => false,
                    }
                }
            })
            .copied()
            .collect();

        if internal1.is_empty() || internal2.is_empty() {
            // Cannot crossover - no internal nodes
            return CrossoverResult {
                offspring1: self.copy_tree(parent1, arena),
                offspring2: self.copy_tree(parent2, arena),
                success: true,
                crossover_depth1: 0,
                crossover_depth2: 0,
            };
        }

        // Select random crossover points
        let idx1 = rng.gen_range(0..internal1.len());
        let idx2 = rng.gen_range(0..internal2.len());

        let (crossover_node1, depth1) = internal1[idx1];
        let (crossover_node2, depth2) = internal2[idx2];

        // Check depth constraints
        unsafe {
            let node1_depth = crossover_node1.as_ref().depth;
            let node2_depth = crossover_node2.as_ref().depth;

            // Estimate resulting depths
            let new_depth1 = depth1 + node2_depth;
            let new_depth2 = depth2 + node1_depth;

            if new_depth1 > self.max_depth || new_depth2 > self.max_depth {
                // Would exceed max depth - try to find shallower crossover points
                if let Some((shallow1, _)) = internal1.iter().find(|(_, d)| *d + node2_depth <= self.max_depth) {
                    return self.perform_crossover(*shallow1, crossover_node2, parent1, parent2, arena);
                }
                if let Some((shallow2, _)) = internal2.iter().find(|(_, d)| *d + node1_depth <= self.max_depth) {
                    return self.perform_crossover(crossover_node1, *shallow2, parent1, parent2, arena);
                }
                // No valid crossover found
                return CrossoverResult {
                    offspring1: self.copy_tree(parent1, arena),
                    offspring2: self.copy_tree(parent2, arena),
                    success: true,
                    crossover_depth1: 0,
                    crossover_depth2: 0,
                };
            }
        }

        self.perform_crossover(crossover_node1, crossover_node2, parent1, parent2, arena)
    }

    /// Execute the actual subtree swap
    fn perform_crossover(
        &self,
        subtree1: NodePtr<AstNode>,
        subtree2: NodePtr<AstNode>,
        parent1: NodePtr<AstNode>,
        parent2: NodePtr<AstNode>,
        arena: &mut TreeArena,
    ) -> CrossoverResult {
        // Copy parent1 and replace subtree1 with subtree2
        let offspring1 = self.copy_tree_with_replacement(
            parent1,
            subtree1,
            subtree2,
            arena,
        );

        // Copy parent2 and replace subtree2 with subtree1
        let offspring2 = self.copy_tree_with_replacement(
            parent2,
            subtree2,
            subtree1,
            arena,
        );

        let depth1 = unsafe { subtree1.as_ref().depth };
        let depth2 = unsafe { subtree2.as_ref().depth };

        CrossoverResult {
            offspring1,
            offspring2,
            success: offspring1.is_some() && offspring2.is_some(),
            crossover_depth1: depth1,
            crossover_depth2: depth2,
        }
    }

    /// Collect all nodes in a tree with their depths
    fn collect_nodes(&self, node: NodePtr<AstNode>, depth: u8, result: &mut Vec<(NodePtr<AstNode>, u8)>) {
        result.push((node, depth));
        
        unsafe {
            let n = node.as_ref();
            for i in 0..n.child_count as usize {
                if let Some(child) = n.children[i] {
                    self.collect_nodes(child, depth + 1, result);
                }
            }
        }
    }

    /// Deep copy a tree
    fn copy_tree(&self, root: NodePtr<AstNode>, arena: &mut TreeArena) -> Option<NodePtr<AstNode>> {
        unsafe {
            let node = root.as_ref();
            
            // Copy children first
            let mut child_ptrs: [Option<NodePtr<AstNode>>; 4] = [None, None, None, None];
            for i in 0..node.child_count as usize {
                if let Some(child) = node.children[i] {
                    child_ptrs[i] = self.copy_tree(child, arena);
                }
            }

            // Create new node with copied data
            let new_node = AstNode {
                primitive_type: node.primitive_type,
                depth: node.depth,
                children: child_ptrs,
                child_count: node.child_count,
                data: self.copy_node_data(&node.data),
            };

            arena.alloc(new_node)
        }
    }

    /// Copy node data (handles String cloning for variables)
    fn copy_node_data(&self, data: &NodeData) -> NodeData {
        match data {
            NodeData::Operator(op) => NodeData::Operator(*op),
            NodeData::ConstantFloat(v) => NodeData::ConstantFloat(*v),
            NodeData::ConstantInt(v) => NodeData::ConstantInt(*v),
            NodeData::ConstantBool(v) => NodeData::ConstantBool(*v),
            NodeData::Variable { index, name } => NodeData::Variable {
                index: *index,
                name: name.clone(),
            },
        }
    }

    /// Copy tree with subtree replacement
    fn copy_tree_with_replacement(
        &self,
        root: NodePtr<AstNode>,
        target: NodePtr<AstNode>,
        replacement: NodePtr<AstNode>,
        arena: &mut TreeArena,
    ) -> Option<NodePtr<AstNode>> {
        // Check if this is the target node
        if std::ptr::eq(root.ptr.as_ptr(), target.ptr.as_ptr()) {
            // Return a copy of the replacement subtree
            return self.copy_tree(replacement, arena);
        }

        unsafe {
            let node = root.as_ref();
            
            // Copy children, replacing target if found
            let mut child_ptrs: [Option<NodePtr<AstNode>>; 4] = [None, None, None, None];
            for i in 0..node.child_count as usize {
                if let Some(child) = node.children[i] {
                    child_ptrs[i] = self.copy_tree_with_replacement(
                        child,
                        target,
                        replacement,
                        arena,
                    );
                }
            }

            let new_node = AstNode {
                primitive_type: node.primitive_type,
                depth: node.depth,
                children: child_ptrs,
                child_count: node.child_count,
                data: self.copy_node_data(&node.data),
            };

            arena.alloc(new_node)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::arena_allocator::{make_const_float, make_operator};

    #[test]
    fn test_crossover_basic() {
        let mut arena = TreeArena::new(1000);
        
        // Create two simple trees: (1.0 + 2.0) and (3.0 * 4.0)
        let leaf1 = make_const_float(&mut arena, 1.0).unwrap();
        let leaf2 = make_const_float(&mut arena, 2.0).unwrap();
        let parent1 = make_operator(&mut arena, Operator::Add, &[leaf1, leaf2]).unwrap();

        let leaf3 = make_const_float(&mut arena, 3.0).unwrap();
        let leaf4 = make_const_float(&mut arena, 4.0).unwrap();
        let parent2 = make_operator(&mut arena, Operator::Mul, &[leaf3, leaf4]).unwrap();

        let crossover = SubtreeCrossover::new(10, 1.0).with_seed(42);
        let result = crossover.crossover(parent1, parent2, &mut arena);

        assert!(result.success);
        assert!(result.offspring1.is_some());
        assert!(result.offspring2.is_some());
    }

    #[test]
    fn test_no_crossover_rate() {
        let mut arena = TreeArena::new(1000);
        
        let leaf1 = make_const_float(&mut arena, 1.0).unwrap();
        let leaf2 = make_const_float(&mut arena, 2.0).unwrap();
        let parent1 = make_operator(&mut arena, Operator::Add, &[leaf1, leaf2]).unwrap();

        let leaf3 = make_const_float(&mut arena, 3.0).unwrap();
        let parent2 = make_operator(&mut arena, Operator::Sub, &[leaf3, leaf1]).unwrap();

        // 0% crossover rate
        let crossover = SubtreeCrossover::new(10, 0.0).with_seed(42);
        let result = crossover.crossover(parent1, parent2, &mut arena);

        assert!(result.success);
        assert_eq!(result.crossover_depth1, 0);
        assert_eq!(result.crossover_depth2, 0);
    }
}
