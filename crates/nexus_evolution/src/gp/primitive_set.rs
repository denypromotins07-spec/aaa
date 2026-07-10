//! Primitive Set Definition for Genetic Programming
//! 
//! Defines the terminal and function sets available for AST evolution.
//! Enforces strict typing to prevent invalid expression generation.

use super::arena_allocator::{Operator, PrimitiveType};

/// Configuration for the primitive set
#[derive(Debug, Clone)]
pub struct PrimitiveSetConfig {
    pub max_tree_depth: u8,
    pub min_tree_depth: u8,
    pub constant_range: (f64, f64),
    pub available_operators: Vec<Operator>,
    pub variable_count: usize,
}

impl Default for PrimitiveSetConfig {
    fn default() -> Self {
        Self {
            max_tree_depth: 8,
            min_tree_depth: 2,
            constant_range: (-10.0, 10.0),
            available_operators: vec![
                Operator::Add, Operator::Sub, Operator::Mul, Operator::Div,
                Operator::Lt, Operator::Gt, Operator::Le, Operator::Ge,
                Operator::And, Operator::Or, Operator::Not,
                Operator::TsMean, Operator::TsStdDev, Operator::TsMax, Operator::TsMin,
                Operator::TsLag, Operator::TsDelta, Operator::TsRank,
            ],
            variable_count: 100, // Max number of input features
        }
    }
}

/// The complete set of primitives available for GP evolution
pub struct PrimitiveSet {
    config: PrimitiveSetConfig,
    /// Pre-computed lookup tables for fast random selection
    unary_ops: Vec<Operator>,
    binary_ops: Vec<Operator>,
    terminals_float: Vec<f64>, // Pre-generated constants
}

impl PrimitiveSet {
    pub fn new(config: PrimitiveSetConfig) -> Self {
        let mut unary_ops = Vec::new();
        let mut binary_ops = Vec::new();

        for &op in &config.available_operators {
            match op.arity() {
                1 => unary_ops.push(op),
                2 => binary_ops.push(op),
                _ => {} // Ignore unsupported arities
            }
        }

        // Pre-generate a pool of random constants for mutation
        let terminals_float: Vec<f64> = {
            let mut consts = Vec::with_capacity(64);
            let range = config.constant_range.1 - config.constant_range.0;
            for i in 0..64 {
                let val = config.constant_range.0 + (i as f64 / 64.0) * range;
                consts.push(val);
            }
            consts
        };

        Self {
            config,
            unary_ops,
            binary_ops,
            terminals_float,
        }
    }

    /// Get a random unary operator based on seed
    #[inline]
    pub fn get_unary_op(&self, seed: u64) -> Option<Operator> {
        if self.unary_ops.is_empty() {
            return None;
        }
        Some(self.unary_ops[(seed as usize) % self.unary_ops.len()])
    }

    /// Get a random binary operator based on seed
    #[inline]
    pub fn get_binary_op(&self, seed: u64) -> Option<Operator> {
        if self.binary_ops.is_empty() {
            return None;
        }
        Some(self.binary_ops[(seed as usize) % self.binary_ops.len()])
    }

    /// Get a random constant from the pre-generated pool
    #[inline]
    pub fn get_random_constant(&self, seed: u64) -> f64 {
        self.terminal_floats[(seed as usize) % self.terminal_floats.len()]
    }

    /// Check if an operator is available in this primitive set
    #[inline]
    pub fn is_operator_available(&self, op: Operator) -> bool {
        self.config.available_operators.contains(&op)
    }

    /// Get the maximum allowed tree depth
    #[inline]
    pub fn max_depth(&self) -> u8 {
        self.config.max_tree_depth
    }

    /// Get the minimum allowed tree depth
    #[inline]
    pub fn min_depth(&self) -> u8 {
        self.config.min_tree_depth
    }

    /// Get the number of available variables/features
    #[inline]
    pub fn variable_count(&self) -> usize {
        self.config.variable_count
    }

    /// Get operators that return a specific type
    pub fn get_ops_by_return_type(&self, ptype: PrimitiveType) -> Vec<Operator> {
        self.config
            .available_operators
            .iter()
            .filter(|op| op.return_type() == ptype)
            .copied()
            .collect()
    }

    /// Get operators that accept a specific input type
    pub fn get_ops_by_input_type(&self, ptype: PrimitiveType) -> Vec<Operator> {
        // Simplified: all numeric ops accept Float
        // In a full implementation, we'd track input types per argument position
        self.config
            .available_operators
            .iter()
            .filter(|op| {
                match op {
                    Operator::Not => ptype == PrimitiveType::Bool,
                    Operator::And | Operator::Or => ptype == PrimitiveType::Bool,
                    _ => ptype == PrimitiveType::Float || ptype == PrimitiveType::Integer,
                }
            })
            .copied()
            .collect()
    }
}

/// Builder for creating customized primitive sets
pub struct PrimitiveSetBuilder {
    config: PrimitiveSetConfig,
}

impl PrimitiveSetBuilder {
    pub fn new() -> Self {
        Self {
            config: PrimitiveSetConfig::default(),
        }
    }

    pub fn max_depth(mut self, depth: u8) -> Self {
        self.config.max_tree_depth = depth;
        self
    }

    pub fn min_depth(mut self, depth: u8) -> Self {
        self.config.min_tree_depth = depth;
        self
    }

    pub fn constant_range(mut self, min: f64, max: f64) -> Self {
        self.config.constant_range = (min, max);
        self
    }

    pub fn with_operators(mut self, ops: Vec<Operator>) -> Self {
        self.config.available_operators = ops;
        self
    }

    pub fn add_operator(mut self, op: Operator) -> Self {
        if !self.config.available_operators.contains(&op) {
            self.config.available_operators.push(op);
        }
        self
    }

    pub fn remove_operator(mut self, op: Operator) -> Self {
        self.config.available_operators.retain(|&o| o != op);
        self
    }

    pub fn variable_count(mut self, count: usize) -> Self {
        self.config.variable_count = count;
        self
    }

    pub fn build(self) -> PrimitiveSet {
        PrimitiveSet::new(self.config)
    }
}

impl Default for PrimitiveSetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_set_creation() {
        let ps = PrimitiveSet::default();
        assert!(ps.get_binary_op(42).is_some());
        assert_eq!(ps.max_depth(), 8);
    }

    #[test]
    fn test_builder_pattern() {
        let ps = PrimitiveSetBuilder::new()
            .max_depth(12)
            .min_depth(3)
            .constant_range(-100.0, 100.0)
            .add_operator(Operator::Correlation)
            .variable_count(50)
            .build();

        assert_eq!(ps.max_depth(), 12);
        assert_eq!(ps.min_depth(), 3);
        assert!(ps.is_operator_available(Operator::Correlation));
        assert_eq!(ps.variable_count(), 50);
    }

    #[test]
    fn test_default_primitive_set() {
        let ps = PrimitiveSet::default();
        assert!(ps.is_operator_available(Operator::Add));
        assert!(ps.is_operator_available(Operator::TsMean));
        assert!(!ps.is_operator_available(Operator::Beta)); // Not in default set
    }
}

impl Default for PrimitiveSet {
    fn default() -> Self {
        Self::new(PrimitiveSetConfig::default())
    }
}
