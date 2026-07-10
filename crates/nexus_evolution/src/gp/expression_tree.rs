//! Expression Tree Implementation using Zero-Copy Arena Pointers
//! 
//! Provides tree traversal, evaluation, and manipulation utilities
//! without any heap allocation during hot-path operations.

use super::arena_allocator::{AstNode, NodePtr, PrimitiveType, NodeData, Operator};
use std::slice;

/// Immutable view of an expression tree rooted at a specific node
#[derive(Debug, Clone, Copy)]
pub struct ExpressionTree {
    root: Option<NodePtr<AstNode>>,
    size: usize, // Cached node count for quick fitness calculations
}

impl ExpressionTree {
    /// Create a new empty tree
    pub const fn empty() -> Self {
        Self {
            root: None,
            size: 0,
        }
    }

    /// Create a tree with the given root node
    pub fn new(root: NodePtr<AstNode>) -> Self {
        let size = Self::count_nodes_unsafe(root);
        Self {
            root: Some(root),
            size,
        }
    }

    /// Get the root node pointer (unsafe - caller must ensure arena validity)
    #[inline]
    pub fn root(&self) -> Option<NodePtr<AstNode>> {
        self.root
    }

    /// Get cached tree size (number of nodes)
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Check if tree is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Count nodes in tree (O(n) - use cached size() when possible)
    fn count_nodes_unsafe(root: NodePtr<AstNode>) -> usize {
        unsafe {
            let node = root.as_ref();
            let mut count = 1usize;
            for i in 0..node.child_count as usize {
                if let Some(child) = node.children[i] {
                    count += Self::count_nodes_unsafe(child);
                }
            }
            count
        }
    }

    /// Depth-first traversal visitor
    /// F is a closure that receives each node pointer
    pub fn visit_dfs<F>(&self, mut visitor: F)
    where
        F: FnMut(NodePtr<AstNode>),
    {
        if let Some(root) = self.root {
            Self::visit_dfs_recursive(root, &mut visitor);
        }
    }

    fn visit_dfs_recursive<F>(node: NodePtr<AstNode>, visitor: &mut F)
    where
        F: FnMut(NodePtr<AstNode>),
    {
        visitor(node);
        unsafe {
            let n = node.as_ref();
            for i in 0..n.child_count as usize {
                if let Some(child) = n.children[i] {
                    Self::visit_dfs_recursive(child, visitor);
                }
            }
        }
    }

    /// Find a random node at a specific depth level
    /// Returns None if no node exists at that depth
    pub fn find_node_at_depth(&self, target_depth: u8, rng_seed: u64) -> Option<NodePtr<AstNode>> {
        if self.root.is_none() {
            return None;
        }

        let mut candidates: [Option<NodePtr<AstNode>>; 256] = [None; 256];
        let mut count = 0usize;

        self.visit_dfs(|node| {
            unsafe {
                if node.as_ref().depth == target_depth && count < 256 {
                    candidates[count] = Some(node);
                    count += 1;
                }
            }
        });

        if count == 0 {
            return None;
        }

        // Simple deterministic selection based on seed
        let index = (rng_seed as usize) % count;
        candidates[index]
    }

    /// Validate tree type consistency
    /// Returns true if all operator inputs match expected types
    pub fn validate_types(&self) -> bool {
        let mut valid = true;
        self.visit_dfs(|node| {
            if !valid {
                return;
            }
            unsafe {
                let n = node.as_ref();
                match &n.data {
                    NodeData::Operator(op) => {
                        // Check children count matches arity
                        if n.child_count != op.arity() {
                            valid = false;
                            return;
                        }
                        // Type checking would go here for strict GP
                        // For now we trust the construction process
                    }
                    _ => {}
                }
            }
        });
        valid
    }

    /// Serialize tree to a compact string representation for logging
    pub fn to_string_repr(&self) -> String {
        let mut result = String::with_capacity(256);
        if let Some(root) = self.root {
            Self::build_string(root, &mut result);
        }
        result
    }

    fn build_string(node: NodePtr<AstNode>, out: &mut String) {
        unsafe {
            let n = node.as_ref();
            match &n.data {
                NodeData::ConstantFloat(v) => {
                    out.push_str(&format!("{:.6}", v));
                }
                NodeData::ConstantInt(v) => {
                    out.push_str(&format!("{}", v));
                }
                NodeData::ConstantBool(v) => {
                    out.push_str(if *v { "true" } else { "false" });
                }
                NodeData::Variable { name, .. } => {
                    out.push_str(name);
                }
                NodeData::Operator(op) => {
                    out.push('(');
                    out.push_str(&format!("{:?}", op));
                    for i in 0..n.child_count as usize {
                        out.push(' ');
                        if let Some(child) = n.children[i] {
                            Self::build_string(child, out);
                        }
                    }
                    out.push(')');
                }
            }
        }
    }
}

/// Iterator over all nodes in the tree (DFS order)
pub struct TreeIterator {
    stack: Vec<NodePtr<AstNode>>,
}

impl TreeIterator {
    pub fn new(root: Option<NodePtr<AstNode>>) -> Self {
        let mut stack = Vec::with_capacity(32);
        if let Some(r) = root {
            stack.push(r);
        }
        Self { stack }
    }
}

impl Iterator for TreeIterator {
    type Item = NodePtr<AstNode>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.stack.pop() {
            unsafe {
                let n = node.as_ref();
                // Push children in reverse order so they're popped in forward order
                for i in (0..n.child_count as usize).rev() {
                    if let Some(child) = n.children[i] {
                        self.stack.push(child);
                    }
                }
            }
            Some(node)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::arena_allocator::{TreeArena, make_const_float, make_operator};

    #[test]
    fn test_tree_creation() {
        let mut arena = TreeArena::new(1000);
        let leaf1 = make_const_float(&mut arena, 1.0).unwrap();
        let leaf2 = make_const_float(&mut arena, 2.0).unwrap();
        let root = make_operator(&mut arena, Operator::Add, &[leaf1, leaf2]).unwrap();
        
        let tree = ExpressionTree::new(root);
        assert_eq!(tree.size(), 3);
        assert!(tree.validate_types());
    }

    #[test]
    fn test_tree_traversal() {
        let mut arena = TreeArena::new(1000);
        let leaf1 = make_const_float(&mut arena, 1.0).unwrap();
        let leaf2 = make_const_float(&mut arena, 2.0).unwrap();
        let root = make_operator(&mut arena, Operator::Add, &[leaf1, leaf2]).unwrap();
        
        let tree = ExpressionTree::new(root);
        let mut count = 0;
        tree.visit_dfs(|_| { count += 1; });
        assert_eq!(count, 3);
    }

    #[test]
    fn test_empty_tree() {
        let tree = ExpressionTree::empty();
        assert!(tree.is_empty());
        assert_eq!(tree.size(), 0);
        assert!(tree.root().is_none());
    }
}
