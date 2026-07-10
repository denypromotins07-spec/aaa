//! NEXUS-OMEGA Stage 10: Autonomous Strategy Evolution
//! 
//! Genetic Programming engine with:
//! - Zero-allocation AST arena allocator
//! - Combinatorial Purged Cross-Validation (CPCV)
//! - NSGA-II multi-objective optimization
//! - Cranelift JIT compilation to native x86_64

#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

pub mod gp {
    //! Genetic Programming core modules
    
    pub mod arena_allocator;
    pub mod expression_tree;
    pub mod primitive_set;
    
    pub use arena_allocator::*;
    pub use expression_tree::*;
    pub use primitive_set::*;
}

pub mod sandbox {
    //! Distributed evaluation sandbox with overfitting prevention
    
    pub mod ast_evaluator;
    pub mod cpcv_overfit_guard;
    
    pub use ast_evaluator::*;
    pub use cpcv_overfit_guard::*;
}

pub mod fitness {
    //! Multi-objective fitness calculation and NSGA-II sorting
    
    pub mod nsga2_sorter;
    pub mod orthogonality_penalty;
    pub mod crowding_distance;
    
    pub use nsga2_sorter::*;
    pub use orthogonality_penalty::*;
    pub use crowding_distance::*;
}

pub mod operators {
    //! Genetic operators for evolution
    
    pub mod subtree_crossover;
    pub mod point_mutation;
    
    pub use subtree_crossover::*;
    pub use point_mutation::*;
}

pub mod jit {
    //! JIT compilation of evolved strategies
    
    pub mod cranelift_compiler;
    
    pub use cranelift_compiler::*;
}

/// Re-export main types at crate root for convenience
pub use gp::{TreeArena, ExpressionTree, PrimitiveSet, NodePtr, AstNode, Operator};
pub use sandbox::{AstEvaluator, DataWindow, CpcvOverfitGuard, CpcvConfig, CpcvResult};
pub use fitness::{Nsga2Sorter, MultiObjectiveFitness, OrthogonalityCalculator};
pub use operators::{SubtreeCrossover, PointMutation, CrossoverResult, MutationResult};
pub use jit::{CraneliftCompiler, JitCompilationResult, CompiledFn};

/// Version of the nexus_evolution crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Create a complete GP evolution pipeline
pub struct EvolutionPipeline {
    arena: TreeArena,
    primitive_set: PrimitiveSet,
    evaluator: AstEvaluator,
    crossover: SubtreeCrossover,
    mutation: PointMutation,
    sorter: Nsga2Sorter,
    cpcv_guard: Option<CpcvOverfitGuard>,
    jit_compiler: Option<CraneliftCompiler>,
}

impl EvolutionPipeline {
    /// Create a new evolution pipeline with default configuration
    pub fn new(population_size: usize) -> Self {
        Self {
            arena: TreeArena::new(1_000_000),
            primitive_set: PrimitiveSet::default(),
            evaluator: AstEvaluator::new(10000, 100),
            crossover: SubtreeCrossover::new(10, 0.9),
            mutation: PointMutation::new(0.1, 0.1, 0.05),
            sorter: Nsga2Sorter::new(population_size, 4),
            cpcv_guard: None,
            jit_compiler: None,
        }
    }

    /// Enable CPCV overfitting prevention
    pub fn with_cpcv(mut self, config: CpcvConfig, dataset_length: usize) -> Self {
        self.cpcv_guard = Some(CpcvOverfitGuard::new(config, dataset_length));
        self
    }

    /// Enable JIT compilation
    pub fn with_jit(mut self) -> Self {
        self.jit_compiler = CraneliftCompiler::new().ok();
        self
    }

    /// Get reference to arena
    pub fn arena(&mut self) -> &mut TreeArena {
        &mut self.arena
    }

    /// Get reference to primitive set
    pub fn primitive_set(&self) -> &PrimitiveSet {
        &self.primitive_set
    }

    /// Get reference to evaluator
    pub fn evaluator(&mut self) -> &mut AstEvaluator {
        &mut self.evaluator
    }

    /// Get reference to crossover operator
    pub fn crossover(&self) -> &SubtreeCrossover {
        &self.crossover
    }

    /// Get reference to mutation operator
    pub fn mutation(&self) -> &PointMutation {
        &self.mutation
    }

    /// Get reference to NSGA-II sorter
    pub fn sorter(&self) -> &Nsga2Sorter {
        &self.sorter
    }

    /// Get reference to CPCV guard
    pub fn cpcv_guard(&self) -> Option<&CpcvOverfitGuard> {
        self.cpcv_guard.as_ref()
    }

    /// Get mutable reference to JIT compiler
    pub fn jit_compiler(&mut self) -> Option<&mut CraneliftCompiler> {
        self.jit_compiler.as_mut()
    }

    /// Reset arena for new generation
    pub fn reset_arena(&mut self) {
        self.arena.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = EvolutionPipeline::new(1000);
        assert_eq!(pipeline.arena().len(), 0);
    }

    #[test]
    fn test_pipeline_with_options() {
        let pipeline = EvolutionPipeline::new(500)
            .with_cpcv(CpcvConfig::default(), 10000)
            .with_jit();
        
        assert!(pipeline.cpcv_guard().is_some());
        assert!(pipeline.jit_compiler().is_some());
    }
}
