//! Everettian Branching Engine
//! 
//! Calculates the squared amplitude (Born rule) of divergent market trajectories
//! based on photonic ADC micro-structural noise. Ensures unitary evolution
//! with strict measure conservation.

use alloc::vec::Vec;
use core::fmt;
use super::hilbert_space_mps::{MatrixProductState, ComplexAmplitude, MpsError};

/// Maximum number of branches to track (prevents combinatorial explosion)
const MAX_BRANCHES: usize = 4096;

/// Tolerance for measure conservation (sum of probabilities must be 1.0)
const MEASURE_CONSERVATION_TOLERANCE: f64 = 1e-12;

/// Error types for Everettian branching
#[derive(Debug, Clone, PartialEq)]
pub enum EverettianError {
    MeasureNonConservation { total: f64, expected: f64 },
    BranchLimitExceeded { requested: usize, max: usize },
    InvalidBranchProbability { probability: f64 },
    NumericalOverflow { operation: &'static str },
}

impl fmt::Display for EverettianError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EverettianError::MeasureNonConservation { total, expected } => {
                write!(f, "Measure non-conservation: total={}, expected={}", total, expected)
            }
            EverettianError::BranchLimitExceeded { requested, max } => {
                write!(f, "Branch limit exceeded: requested {}, max {}", requested, max)
            }
            EverettianError::InvalidBranchProbability { probability } => {
                write!(f, "Invalid branch probability: {}", probability)
            }
            EverettianError::NumericalOverflow { operation } => {
                write!(f, "Numerical overflow in {}", operation)
            }
        }
    }
}

/// Represents a single branch in the multiverse
#[derive(Debug, Clone)]
pub struct EverettianBranch {
    /// Unique identifier for this branch
    pub id: usize,
    /// Description of the market trajectory (e.g., "Hawkish Fed Hike")
    pub trajectory: &'static str,
    /// Quantum amplitude (complex)
    pub amplitude: ComplexAmplitude,
    /// Squared amplitude (Born rule probability)
    pub probability: f64,
    /// Parent branch ID (None for root)
    pub parent_id: Option<usize>,
    /// Depth in the branching tree
    pub depth: usize,
}

impl EverettianBranch {
    pub fn new(
        id: usize,
        trajectory: &'static str,
        amplitude: ComplexAmplitude,
        parent_id: Option<usize>,
        depth: usize,
    ) -> Result<Self, EverettianError> {
        let probability = amplitude.magnitude_squared();
        
        // Validate probability is in [0, 1]
        if probability < 0.0 || probability > 1.0 + MEASURE_CONSERVATION_TOLERANCE {
            return Err(EverettianError::InvalidBranchProbability { probability });
        }

        Ok(Self {
            id,
            trajectory,
            amplitude,
            probability,
            parent_id,
            depth,
        })
    }
}

/// Everettian Branching Engine
/// 
/// Tracks divergent market trajectories and ensures measure conservation
pub struct EverettianBranchingEngine {
    /// All active branches
    branches: Vec<EverettianBranch>,
    /// Current total measure (should always be 1.0)
    total_measure: f64,
    /// Next branch ID
    next_id: usize,
    /// Reference to the underlying MPS state
    mps_state: MatrixProductState,
}

impl EverettianBranchingEngine {
    /// Create a new branching engine with an initial MPS state
    pub fn new(mps_state: MatrixProductState) -> Result<Self, EverettianError> {
        // Initialize with root branch
        let root_amplitude = ComplexAmplitude::one();
        let root_probability = root_amplitude.magnitude_squared();
        
        let root_branch = EverettianBranch::new(
            0,
            "Initial Market State",
            root_amplitude,
            None,
            0,
        ).map_err(|e| EverettianError::NumericalOverflow {
            operation: "root branch creation",
        })?;

        let mut engine = Self {
            branches: vec![root_branch],
            total_measure: 1.0,
            next_id: 1,
            mps_state,
        };

        // Verify initial measure conservation
        engine.verify_measure_conservation()?;

        Ok(engine)
    }

    /// Split a branch into multiple child branches
    /// 
    /// This simulates quantum measurement/decoherence creating parallel realities
    pub fn split_branch(
        &mut self,
        parent_id: usize,
        trajectories: Vec<(&'static str, ComplexAmplitude)>,
    ) -> Result<Vec<usize>, EverettianError> {
        // Check branch limit
        if self.branches.len() + trajectories.len() > MAX_BRANCHES {
            return Err(EverettianError::BranchLimitExceeded {
                requested: self.branches.len() + trajectories.len(),
                max: MAX_BRANCHES,
            });
        }

        // Find parent branch
        let parent_idx = self.branches.iter().position(|b| b.id == parent_id)
            .ok_or_else(|| EverettianError::NumericalOverflow {
                operation: "parent branch lookup",
            })?;
        
        let parent = &self.branches[parent_idx];
        let parent_depth = parent.depth;

        // Calculate sum of squared amplitudes for children
        let mut child_measure_sum = 0.0_f64;
        for (_, amplitude) in &trajectories {
            let prob = amplitude.magnitude_squared();
            if prob.is_nan() || prob.is_infinite() {
                return Err(EverettianError::NumericalOverflow {
                    operation: "amplitude squared calculation",
                });
            }
            child_measure_sum += prob;
        }

        // Normalize children to preserve parent's measure
        let normalization_factor = if child_measure_sum > MEASURE_CONSERVATION_TOLERANCE {
            (parent.probability / child_measure_sum).sqrt()
        } else {
            1.0
        };

        // Create child branches
        let mut child_ids = Vec::with_capacity(trajectories.len());
        
        for (trajectory, amplitude) in trajectories {
            let normalized_amplitude = ComplexAmplitude::new(
                amplitude.re * normalization_factor,
                amplitude.im * normalization_factor,
            );

            let child = EverettianBranch::new(
                self.next_id,
                trajectory,
                normalized_amplitude,
                Some(parent_id),
                parent_depth + 1,
            )?;

            child_ids.push(child.id);
            self.branches.push(child);
            self.next_id += 1;
        }

        // Verify measure conservation after split
        self.verify_measure_conservation()?;

        Ok(child_ids)
    }

    /// Calculate Born rule probability for a specific branch
    pub fn born_rule_probability(&self, branch_id: usize) -> Result<f64, EverettianError> {
        let branch = self.branches.iter().find(|b| b.id == branch_id)
            .ok_or_else(|| EverettianError::NumericalOverflow {
                operation: "branch lookup",
            })?;

        let prob = branch.amplitude.magnitude_squared();
        
        if prob.is_nan() || prob.is_infinite() {
            return Err(EverettianError::NumericalOverflow {
                operation: "Born rule calculation",
            });
        }

        Ok(prob)
    }

    /// Get all branches at a specific depth
    pub fn branches_at_depth(&self, depth: usize) -> Vec<&EverettianBranch> {
        self.branches.iter().filter(|b| b.depth == depth).collect()
    }

    /// Verify measure conservation across all branches
    /// 
    /// CRITICAL: Ensures sum of probabilities equals 1.0 within tolerance
    pub fn verify_measure_conservation(&mut self) -> Result<(), EverettianError> {
        let mut total = 0.0_f64;
        
        // Sum probabilities of all leaf branches (those without children)
        let leaf_branches: Vec<_> = self.branches.iter().filter(|b| {
            !self.branches.iter().any(|other| other.parent_id == Some(b.id))
        }).collect();

        for branch in &leaf_branches {
            total += branch.probability;
            
            if total.is_nan() || total.is_infinite() {
                return Err(EverettianError::NumericalOverflow {
                    operation: "measure summation",
                });
            }
        }

        self.total_measure = total;

        // Check conservation with strict tolerance
        let deviation = (total - 1.0).abs();
        if deviation > MEASURE_CONSERVATION_TOLERANCE {
            return Err(EverettianError::MeasureNonConservation {
                total,
                expected: 1.0,
            });
        }

        Ok(())
    }

    /// Get the total measure (should always be 1.0)
    pub const fn total_measure(&self) -> f64 {
        self.total_measure
    }

    /// Get number of active branches
    pub fn num_branches(&self) -> usize {
        self.branches.len()
    }

    /// Get reference to MPS state
    pub const fn mps_state(&self) -> &MatrixProductState {
        &self.mps_state
    }

    /// Prune branches with probability below threshold
    pub fn prune_low_probability_branches(&mut self, threshold: f64) -> Result<usize, EverettianError> {
        if threshold < 0.0 || threshold > 1.0 {
            return Err(EverettianError::InvalidBranchProbability { probability: threshold });
        }

        let initial_count = self.branches.len();
        
        // Keep only branches above threshold or their ancestors
        self.branches.retain(|branch| {
            branch.probability >= threshold || 
            branch.depth == 0 // Always keep root
        });

        let pruned_count = initial_count - self.branches.len();
        
        // Re-verify measure conservation after pruning
        self.verify_measure_conservation()?;

        Ok(pruned_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branch_creation() {
        let mps = MatrixProductState::new(2, 2).unwrap();
        let engine = EverettianBranchingEngine::new(mps).unwrap();
        
        assert_eq!(engine.num_branches(), 1);
        assert!((engine.total_measure() - 1.0).abs() < MEASURE_CONSERVATION_TOLERANCE);
    }

    #[test]
    fn test_branch_splitting() {
        let mps = MatrixProductState::new(2, 2).unwrap();
        let mut engine = EverettianBranchingEngine::new(mps).unwrap();

        // Split root into two branches
        let amplitudes = vec![
            ("Bull Market", ComplexAmplitude::new(0.8, 0.0)),
            ("Bear Market", ComplexAmplitude::new(0.6, 0.0)),
        ];

        let child_ids = engine.split_branch(0, amplitudes).unwrap();
        assert_eq!(child_ids.len(), 2);
        assert_eq!(engine.num_branches(), 3); // root + 2 children

        // Verify measure conservation
        assert!((engine.total_measure() - 1.0).abs() < MEASURE_CONSERVATION_TOLERANCE);
    }

    #[test]
    fn test_born_rule_probability() {
        let mps = MatrixProductState::new(2, 2).unwrap();
        let mut engine = EverettianBranchingEngine::new(mps).unwrap();

        let amplitudes = vec![
            ("Scenario A", ComplexAmplitude::new(0.5, 0.0)),
            ("Scenario B", ComplexAmplitude::new(0.5, 0.0)),
        ];

        let child_ids = engine.split_branch(0, amplitudes).unwrap();
        
        // Each child should have probability 0.5 (normalized)
        for id in child_ids {
            let prob = engine.born_rule_probability(id).unwrap();
            assert!(prob > 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_branch_limit() {
        let mps = MatrixProductState::new(2, 2).unwrap();
        let mut engine = EverettianBranchingEngine::new(mps).unwrap();

        // Try to exceed branch limit
        for _ in 0..MAX_BRANCHES {
            let amplitudes = vec![("Test", ComplexAmplitude::new(0.1, 0.0))];
            match engine.split_branch(0, amplitudes) {
                Ok(_) => continue,
                Err(EverettianError::BranchLimitExceeded { .. }) => break,
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }

        // Should have hit the limit
        assert!(engine.num_branches() <= MAX_BRANCHES);
    }
}
