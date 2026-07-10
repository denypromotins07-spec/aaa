//! Point Mutation Operator for Genetic Programming
//! 
//! Randomly alters nodes in an expression tree, including:
//! - Operator mutation (changing the operator type)
//! - Constant mutation (perturbing numeric constants)
//! - Subtree mutation (replacing a subtree with a new random one)

use crate::gp::arena_allocator::{AstNode, NodePtr, TreeArena, NodeData, Operator, PrimitiveType};
use crate::gp::primitive_set::PrimitiveSet;
use rand::Rng;

/// Type of mutation to apply
#[derive(Debug, Clone, Copy)]
pub enum MutationType {
    /// Change an operator to a different one of same arity
    OperatorChange,
    /// Perturb a constant value
    ConstantPerturbation,
    /// Replace a subtree with a new random tree
    SubtreeReplacement,
    /// Collapse a subtree to a single terminal
    SubtreeCollapse,
}

/// Result of a mutation operation
#[derive(Debug)]
pub struct MutationResult {
    /// Mutated tree root pointer
    pub mutated_tree: Option<NodePtr<AstNode>>,
    /// Whether mutation was applied
    pub mutation_applied: bool,
    /// Type of mutation that was applied
    pub mutation_type: Option<MutationType>,
    /// Number of nodes changed
    pub nodes_changed: usize,
}

impl MutationResult {
    pub const fn unchanged() -> Self {
        Self {
            mutated_tree: None,
            mutation_applied: false,
            mutation_type: None,
            nodes_changed: 0,
        }
    }
}

/// Point mutation operator with configurable rates
pub struct PointMutation {
    /// Probability of operator mutation per eligible node
    operator_mutation_rate: f64,
    /// Probability of constant perturbation per constant node
    constant_mutation_rate: f64,
    /// Probability of subtree replacement
    subtree_mutation_rate: f64,
    /// Magnitude of constant perturbation (as fraction of value)
    perturbation_magnitude: f64,
    /// Maximum depth for new subtrees
    max_new_subtree_depth: u8,
    /// RNG seed
    rng_seed: u64,
}

impl PointMutation {
    pub fn new(
        operator_rate: f64,
        constant_rate: f64,
        subtree_rate: f64,
    ) -> Self {
        Self {
            operator_mutation_rate: operator_rate.clamp(0.0, 1.0),
            constant_mutation_rate: constant_rate.clamp(0.0, 1.0),
            subtree_mutation_rate: subtree_rate.clamp(0.0, 1.0),
            perturbation_magnitude: 0.1, // 10% perturbation
            max_new_subtree_depth: 3,
            rng_seed: 0,
        }
    }

    /// Set perturbation magnitude for constant mutations
    pub fn with_perturbation(mut self, magnitude: f64) -> Self {
        self.perturbation_magnitude = magnitude.clamp(0.01, 1.0);
        self
    }

    /// Set maximum depth for generated subtrees
    pub fn with_max_subtree_depth(mut self, depth: u8) -> Self {
        self.max_new_subtree_depth = depth.min(10);
        self
    }

    /// Set RNG seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng_seed = seed;
        self
    }

    /// Apply mutation to a tree
    pub fn mutate(
        &self,
        tree: NodePtr<AstNode>,
        arena: &mut TreeArena,
        primitive_set: &PrimitiveSet,
    ) -> MutationResult {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rng_seed);

        // Decide mutation type based on rates
        let roll = rng.gen::<f64>();
        
        if roll < self.subtree_mutation_rate {
            return self.mutate_subtree(tree, arena, primitive_set, &mut rng);
        }

        // Otherwise, do point mutations
        self.mutate_points(tree, arena, primitive_set, &mut rng)
    }

    /// Perform subtree mutation - replace a random subtree with a new one
    fn mutate_subtree(
        &self,
        tree: NodePtr<AstNode>,
        arena: &mut TreeArena,
        primitive_set: &PrimitiveSet,
        rng: &mut impl Rng,
    ) -> MutationResult {
        // Collect all nodes
        let mut nodes: Vec<(NodePtr<AstNode>, u8)> = Vec::new();
        self.collect_nodes(tree, 0, &mut nodes);

        if nodes.is_empty() {
            return MutationResult::unchanged();
        }

        // Select random node to replace
        let idx = rng.gen_range(0..nodes.len());
        let (target_node, target_depth) = nodes[idx];

        // Check if replacement would exceed max depth
        unsafe {
            let target_type = target_node.as_ref().primitive_type;
            
            // Generate new random subtree of appropriate type
            let new_subtree = self.generate_random_tree(
                arena,
                primitive_set,
                rng,
                target_type,
                self.max_new_subtree_depth,
                0,
            );

            if let Some(new_sub) = new_subtree {
                // Copy original tree with replacement
                let mutated = self.copy_with_replacement(tree, target_node, new_sub, arena);
                
                return MutationResult {
                    mutated_tree: mutated,
                    mutation_applied: true,
                    mutation_type: Some(MutationType::SubtreeReplacement),
                    nodes_changed: 1,
                };
            }
        }

        MutationResult::unchanged()
    }

    /// Perform point mutations throughout the tree
    fn mutate_points(
        &self,
        tree: NodePtr<AstNode>,
        arena: &mut TreeArena,
        primitive_set: &PrimitiveSet,
        rng: &mut impl Rng,
    ) -> MutationResult {
        let mut changed = false;
        let mut change_count = 0usize;

        // First pass: determine what mutations to apply
        let mutations = self.plan_mutations(tree, primitive_set, rng);

        if mutations.is_empty() {
            return MutationResult::unchanged();
        }

        // Second pass: apply mutations while copying
        let mutated = self.apply_planned_mutations(tree, arena, primitive_set, &mutations, rng);

        MutationResult {
            mutated_tree: mutated,
            mutation_applied: true,
            mutation_type: Some(MutationType::OperatorChange),
            nodes_changed: mutations.len(),
        }
    }

    /// Plan which nodes to mutate
    fn plan_mutations(
        &self,
        tree: NodePtr<AstNode>,
        _primitive_set: &PrimitiveSet,
        rng: &mut impl Rng,
    ) -> Vec<(NodePtr<AstNode>, MutationType)> {
        let mut mutations = Vec::new();

        self.traverse_and_plan(tree, &mut mutations, rng);

        mutations
    }

    fn traverse_and_plan(
        &self,
        node: NodePtr<AstNode>,
        mutations: &mut Vec<(NodePtr<AstNode>, MutationType)>,
        rng: &mut impl Rng,
    ) {
        unsafe {
            let n = node.as_ref();
            
            match &n.data {
                NodeData::Operator(op) => {
                    // Consider operator mutation
                    if rng.gen::<f64>() < self.operator_mutation_rate {
                        mutations.push((node, MutationType::OperatorChange));
                    }
                }
                NodeData::ConstantFloat(_) | NodeData::ConstantInt(_) => {
                    // Consider constant perturbation
                    if rng.gen::<f64>() < self.constant_mutation_rate {
                        mutations.push((node, MutationType::ConstantPerturbation));
                    }
                }
                _ => {}
            }

            // Recurse into children
            for i in 0..n.child_count as usize {
                if let Some(child) = n.children[i] {
                    self.traverse_and_plan(child, mutations, rng);
                }
            }
        }
    }

    /// Apply planned mutations while copying the tree
    fn apply_planned_mutations(
        &self,
        tree: NodePtr<AstNode>,
        arena: &mut TreeArena,
        primitive_set: &PrimitiveSet,
        mutations: &[(NodePtr<AstNode>, MutationType)],
        rng: &mut impl Rng,
    ) -> Option<NodePtr<AstNode>> {
        unsafe {
            let node = tree.as_ref();
            
            // Check if this node should be mutated
            let new_data = mutations.iter()
                .find(|(ptr, _)| std::ptr::eq(ptr.ptr.as_ptr(), node))
                .map(|(_, mtype)| self.apply_mutation(&node.data, *mtype, primitive_set, rng))
                .unwrap_or_else(|| self.copy_node_data(&node.data));

            // Copy children
            let mut child_ptrs: [Option<NodePtr<AstNode>>; 4] = [None, None, None, None];
            for i in 0..node.child_count as usize {
                if let Some(child) = node.children[i] {
                    child_ptrs[i] = self.apply_planned_mutations(child, arena, primitive_set, mutations, rng);
                }
            }

            let new_node = AstNode {
                primitive_type: new_data.return_type(),
                depth: node.depth,
                children: child_ptrs,
                child_count: node.child_count,
                data: new_data,
            };

            arena.alloc(new_node)
        }
    }

    /// Apply a specific mutation to node data
    fn apply_mutation(
        &self,
        data: &NodeData,
        mtype: MutationType,
        primitive_set: &PrimitiveSet,
        rng: &mut impl Rng,
    ) -> NodeData {
        match (data, mtype) {
            (NodeData::Operator(op), MutationType::OperatorChange) => {
                // Find alternative operator with same arity and return type
                let arity = op.arity();
                let return_type = op.return_type();
                
                // Get compatible operators
                let candidates: Vec<Operator> = primitive_set
                    .get_ops_by_return_type(return_type)
                    .into_iter()
                    .filter(|o| o.arity() == arity && *o != *op)
                    .collect();

                if !candidates.is_empty() {
                    let idx = rng.gen_range(0..candidates.len());
                    NodeData::Operator(candidates[idx])
                } else {
                    self.copy_node_data(data)
                }
            }
            (NodeData::ConstantFloat(v), MutationType::ConstantPerturbation) => {
                let delta = v.abs() * self.perturbation_magnitude * (rng.gen::<f64>() * 2.0 - 1.0);
                NodeData::ConstantFloat(v + delta)
            }
            (NodeData::ConstantInt(v), MutationType::ConstantPerturbation) => {
                let delta = ((v.abs() as f64) * self.perturbation_magnitude * (rng.gen::<f64>() * 2.0 - 1.0)).round() as i64;
                NodeData::ConstantInt(v + delta)
            }
            _ => self.copy_node_data(data),
        }
    }

    /// Generate a random tree of specified type and max depth
    fn generate_random_tree(
        &self,
        arena: &mut TreeArena,
        primitive_set: &PrimitiveSet,
        rng: &mut impl Rng,
        target_type: PrimitiveType,
        max_depth: u8,
        current_depth: u8,
    ) -> Option<NodePtr<AstNode>> {
        if current_depth >= max_depth || rng.gen::<f64>() < 0.3 {
            // Generate terminal
            return self.generate_terminal(arena, primitive_set, rng, target_type);
        }

        // Generate operator node
        let ops = primitive_set.get_ops_by_return_type(target_type);
        if ops.is_empty() {
            return self.generate_terminal(arena, primitive_set, rng, target_type);
        }

        let op = ops[rng.gen_range(0..ops.len())];
        let arity = op.arity() as usize;

        let mut children: [Option<NodePtr<AstNode>>; 4] = [None, None, None, None];
        for i in 0..arity {
            // For simplicity, assume Float inputs
            children[i] = self.generate_random_tree(
                arena,
                primitive_set,
                rng,
                PrimitiveType::Float,
                max_depth,
                current_depth + 1,
            );
        }

        // Create operator node
        use crate::gp::arena_allocator::make_operator;
        make_operator(arena, op, &children[..arity])
    }

    /// Generate a random terminal node
    fn generate_terminal(
        &self,
        arena: &mut TreeArena,
        primitive_set: &PrimitiveSet,
        rng: &mut impl Rng,
        target_type: PrimitiveType,
    ) -> Option<NodePtr<AstNode>> {
        use crate::gp::arena_allocator::make_const_float;

        match target_type {
            PrimitiveType::Float => {
                let val = primitive_set.get_random_constant(rng.gen());
                make_const_float(arena, val)
            }
            PrimitiveType::Bool => {
                // Use 0.0 or 1.0 as boolean proxy
                let val = if rng.gen::<bool>() { 1.0 } else { 0.0 };
                make_const_float(arena, val)
            }
            _ => {
                let val = primitive_set.get_random_constant(rng.gen());
                make_const_float(arena, val)
            }
        }
    }

    /// Collect all nodes with depths
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

    /// Copy node data
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

    /// Copy tree with replacement
    fn copy_with_replacement(
        &self,
        root: NodePtr<AstNode>,
        target: NodePtr<AstNode>,
        replacement: NodePtr<AstNode>,
        arena: &mut TreeArena,
    ) -> Option<NodePtr<AstNode>> {
        if std::ptr::eq(root.ptr.as_ptr(), target.ptr.as_ptr()) {
            return self.copy_tree(replacement, arena);
        }

        unsafe {
            let node = root.as_ref();
            let mut child_ptrs: [Option<NodePtr<AstNode>>; 4] = [None, None, None, None];
            
            for i in 0..node.child_count as usize {
                if let Some(child) = node.children[i] {
                    child_ptrs[i] = self.copy_with_replacement(child, target, replacement, arena);
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

    /// Deep copy a tree
    fn copy_tree(&self, root: NodePtr<AstNode>, arena: &mut TreeArena) -> Option<NodePtr<AstNode>> {
        unsafe {
            let node = root.as_ref();
            let mut child_ptrs: [Option<NodePtr<AstNode>>; 4] = [None, None, None, None];
            
            for i in 0..node.child_count as usize {
                if let Some(child) = node.children[i] {
                    child_ptrs[i] = self.copy_tree(child, arena);
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

impl NodeData {
    fn return_type(&self) -> PrimitiveType {
        match self {
            NodeData::Operator(op) => op.return_type(),
            NodeData::ConstantFloat(_) => PrimitiveType::Float,
            NodeData::ConstantInt(_) => PrimitiveType::Float,
            NodeData::ConstantBool(_) => PrimitiveType::Bool,
            NodeData::Variable { .. } => PrimitiveType::Float,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::arena_allocator::{make_const_float, make_operator, TreeArena};

    #[test]
    fn test_constant_mutation() {
        let mut arena = TreeArena::new(1000);
        let leaf = make_const_float(&mut arena, 100.0).unwrap();
        
        let ps = PrimitiveSet::default();
        let mutation = PointMutation::new(0.0, 1.0, 0.0)
            .with_perturbation(0.5)
            .with_seed(42);
        
        let result = mutation.mutate(leaf, &mut arena, &ps);
        
        assert!(result.mutation_applied);
        assert_eq!(result.mutation_type, Some(MutationType::ConstantPerturbation));
    }

    #[test]
    fn test_no_mutation_at_zero_rate() {
        let mut arena = TreeArena::new(1000);
        let leaf1 = make_const_float(&mut arena, 1.0).unwrap();
        let leaf2 = make_const_float(&mut arena, 2.0).unwrap();
        let tree = make_operator(&mut arena, Operator::Add, &[leaf1, leaf2]).unwrap();
        
        let ps = PrimitiveSet::default();
        let mutation = PointMutation::new(0.0, 0.0, 0.0);
        
        let result = mutation.mutate(tree, &mut arena, &ps);
        
        assert!(!result.mutation_applied);
    }
}
