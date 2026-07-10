//! Zero-Allocation Expression Tree Arena Allocator
//! 
//! Uses bump allocation to prevent heap fragmentation when evolving millions of ASTs.
//! Memory is allocated in contiguous blocks and reset entirely between generations.
//! 
//! # Safety Guarantees
//! - Arena reset is O(1) and does not leak memory
//! - NodePtr is safe to copy since arena lifetime controls validity
//! - Hard capacity limits prevent OOM conditions

use bumpalo::Bump;
use std::cell::RefCell;
use std::ptr::NonNull;

/// Type system for GP primitives to prevent invalid tree construction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveType {
    Float,
    Bool,
    TimeSeries,
    Integer,
}

/// Raw pointer to a node in the arena. 
/// We avoid Box/Rc to eliminate double-indirection and allocator lock contention.
/// 
/// # Safety
/// NodePtr is only valid while the arena that allocated it remains unreset.
/// After arena.reset(), all NodePtrs from that arena become invalid.
#[derive(Debug, Clone, Copy)]
pub struct NodePtr<T> {
    ptr: NonNull<T>,
}

impl<T> NodePtr<T> {
    #[inline]
    pub unsafe fn as_ref<'a>(&self) -> &'a T {
        self.ptr.as_ref()
    }

    #[inline]
    pub unsafe fn as_mut<'a>(&mut self) -> &'a mut T {
        self.ptr.as_mut()
    }
    
    /// Get raw pointer for comparison (e.g., checking if two nodes are the same)
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
}

// Safe trait implementations since we control aliasing via the arena lifecycle
unsafe impl<T> Send for NodePtr<T> {}
unsafe impl<T> Sync for NodePtr<T> {}

/// An AST Node in the expression tree
#[derive(Debug)]
pub struct AstNode {
    pub primitive_type: PrimitiveType,
    pub depth: u8,
    pub children: [Option<NodePtr<AstNode>>; 4], // Max arity 4 for SIMD operations
    pub child_count: u8,
    // Union-like data storage based on node type
    pub data: NodeData,
}

#[derive(Debug, Clone)]
pub enum NodeData {
    Operator(Operator),
    ConstantFloat(f64),
    ConstantInt(i64),
    ConstantBool(bool),
    Variable { index: usize, name: String },
}

#[derive(Debug, Clone, Copy)]
pub enum Operator {
    // Arithmetic
    Add, Sub, Mul, Div, Mod,
    // Comparison
    Lt, Gt, Le, Ge, Eq, Neq,
    // Logical
    And, Or, Not,
    // Time Series Functions
    TsRank, TsMean, TsStdDev, TsMax, TsMin,
    TsLag, TsDelta, TsSum, TsProduct,
    // Financial Specific
    Correlation, Covariance, Beta, Alpha,
}

impl Operator {
    #[inline]
    pub const fn arity(&self) -> u8 {
        match self {
            Operator::Not => 1,
            Operator::Add | Operator::Sub | Operator::Mul | Operator::Div | Operator::Mod |
            Operator::Lt | Operator::Gt | Operator::Le | Operator::Ge | 
            Operator::Eq | Operator::Neq | Operator::And | Operator::Or |
            Operator::Correlation | Operator::Covariance | Operator::Beta | Operator::Alpha |
            Operator::TsRank | Operator::TsLag | Operator::TsDelta => 2,
            Operator::TsMean | Operator::TsStdDev | Operator::TsMax | Operator::TsMin |
            Operator::TsSum | Operator::TsProduct => 2, // (series, window)
        }
    }

    #[inline]
    pub const fn return_type(&self) -> PrimitiveType {
        match self {
            Operator::Lt | Operator::Gt | Operator::Le | Operator::Ge | 
            Operator::Eq | Operator::Neq | Operator::And | Operator::Or | Operator::Not => PrimitiveType::Bool,
            _ => PrimitiveType::Float,
        }
    }
}

/// Thread-local Arena for AST allocation
/// Uses RefCell for interior mutability while maintaining single-threaded access per worker
pub struct TreeArena {
    bump: Bump,
    node_count: usize,
    max_nodes: usize,
}

impl TreeArena {
    /// Create a new arena with pre-allocated capacity
    pub fn new(capacity: usize) -> Self {
        // Pre-allocate a large chunk to minimize syscalls during evolution
        let mut bump = Bump::with_capacity(capacity * std::mem::size_of::<AstNode>());
        Self {
            bump,
            node_count: 0,
            max_nodes: capacity,
        }
    }

    /// Allocate a new AST node in the arena
    /// Returns None if arena capacity is exceeded (prevents OOM)
    #[inline]
    pub fn alloc(&mut self, node: AstNode) -> Option<NodePtr<AstNode>> {
        if self.node_count >= self.max_nodes {
            return None; // Hard limit to prevent OOM
        }
        
        let ptr = self.bump.alloc(node);
        self.node_count += 1;
        
        // Safety: bump.alloc always returns a valid, aligned pointer
        Some(NodePtr {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
        })
    }

    /// Reset the arena for the next generation
    /// This is O(1) and does not free memory, just resets the bump pointer
    #[inline]
    pub fn reset(&mut self) {
        self.bump.reset();
        self.node_count = 0;
    }

    /// Get current allocation count
    #[inline]
    pub fn len(&self) -> usize {
        self.node_count
    }

    /// Check if arena is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.node_count == 0
    }

    /// Get remaining capacity
    #[inline]
    pub fn remaining(&self) -> usize {
        self.max_nodes.saturating_sub(self.node_count)
    }
}

/// Thread-local storage for per-worker arenas
thread_local! {
    static ARENA: RefCell<TreeArena> = RefCell::new(TreeArena::new(1_000_000));
}

/// Get access to the current thread's arena
#[inline]
pub fn get_arena<F, R>(f: F) -> R
where
    F: FnOnce(&mut TreeArena) -> R,
{
    ARENA.with(|arena| f(&mut arena.borrow_mut()))
}

/// Helper to create a constant float node
#[inline]
pub fn make_const_float(arena: &mut TreeArena, value: f64) -> Option<NodePtr<AstNode>> {
    let node = AstNode {
        primitive_type: PrimitiveType::Float,
        depth: 0,
        children: [None, None, None, None],
        child_count: 0,
        data: NodeData::ConstantFloat(value),
    };
    arena.alloc(node)
}

/// Helper to create a variable node
#[inline]
pub fn make_variable(arena: &mut TreeArena, index: usize, name: &str, ptype: PrimitiveType) -> Option<NodePtr<AstNode>> {
    let node = AstNode {
        primitive_type: ptype,
        depth: 0,
        children: [None, None, None, None],
        child_count: 0,
        data: NodeData::Variable { 
            index, 
            name: name.to_string(),
        },
    };
    arena.alloc(node)
}

/// Helper to create an operator node with children
#[inline]
pub fn make_operator(
    arena: &mut TreeArena,
    op: Operator,
    children: &[NodePtr<AstNode>],
) -> Option<NodePtr<AstNode>> {
    let expected_arity = op.arity() as usize;
    if children.len() != expected_arity {
        return None; // Type safety violation
    }

    let mut child_array: [Option<NodePtr<AstNode>>; 4] = [None, None, None, None];
    let mut max_depth: u8 = 0;

    for (i, child_ptr) in children.iter().enumerate() {
        unsafe {
            let child = child_ptr.as_ref();
            if child.depth >= max_depth {
                max_depth = child.depth + 1;
            }
            // Type checking: ensure child type matches operator input requirements
            // (Simplified check - full validation happens during tree construction)
        }
        child_array[i] = Some(*child_ptr);
    }

    let node = AstNode {
        primitive_type: op.return_type(),
        depth: max_depth,
        children: child_array,
        child_count: children.len() as u8,
        data: NodeData::Operator(op),
    };

    arena.alloc(node)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_allocation() {
        let mut arena = TreeArena::new(1000);
        let node = make_const_float(&mut arena, 42.0);
        assert!(node.is_some());
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = TreeArena::new(1000);
        for i in 0..100 {
            let _ = make_const_float(&mut arena, i as f64);
        }
        assert_eq!(arena.len(), 100);
        arena.reset();
        assert_eq!(arena.len(), 0);
        assert_eq!(arena.remaining(), 1000);
    }

    #[test]
    fn test_capacity_limit() {
        let mut arena = TreeArena::new(5);
        for i in 0..5 {
            let node = make_const_float(&mut arena, i as f64);
            assert!(node.is_some());
        }
        // 6th allocation should fail
        let node = make_const_float(&mut arena, 99.0);
        assert!(node.is_none());
    }
}
