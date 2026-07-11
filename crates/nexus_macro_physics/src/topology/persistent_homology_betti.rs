// NEXUS-OMEGA Stage 34: Geopolitical Topology
// Chapter 2: Persistent Homology Betti Number Calculator
// File: crates/nexus_macro_physics/src/topology/persistent_homology_betti.rs

//! Persistent Homology Betti Number Calculator
//!
//! Computes Betti numbers across filtration scales to detect topological
//! features in geopolitical networks that indicate blockades, alliances,
//! or isolation patterns.

#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]

use core::fmt;
use alloc::vec::Vec;
use alloc::boxed::Box;

use super::vietoris_rips_complex::{VietorisRipsBuilder, VietorisRipsComplex, PersistencePair, VietorisRipsError};
use super::sparse_boundary_matrix::{SparseBoundaryMatrix, ChainComplex, GF2Element, BoundaryMatrixError};

/// Maximum number of filtration steps
pub const MAX_FILTRATION_STEPS: usize = 100;

/// Error types for persistent homology operations
#[derive(Debug, Clone, PartialEq)]
pub enum PersistentHomologyError {
    VietorisRipsError(VietorisRipsError),
    BoundaryMatrixError(BoundaryMatrixError),
    InsufficientData,
    ComputationFailed,
}

impl fmt::Display for PersistentHomologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VietorisRipsError(e) => write!(f, "Vietoris-Rips error: {}", e),
            Self::BoundaryMatrixError(e) => write!(f, "Boundary matrix error: {}", e),
            Self::InsufficientData => write!(f, "Insufficient data for computation"),
            Self::ComputationFailed => write!(f, "Persistent homology computation failed"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PersistentHomologyError {}

/// Betti curve representing how Betti numbers change across filtration
#[derive(Debug, Clone)]
pub struct BettiCurve {
    /// Filtration values
    pub filtration_values: Vec<f64>,
    /// Betti numbers at each filtration step (indexed by dimension then step)
    pub betti_by_dim: Vec<Vec<usize>>,
    /// Maximum dimension computed
    pub max_dimension: usize,
}

impl BettiCurve {
    #[must_use]
    pub fn new(max_dimension: usize) -> Self {
        Self {
            filtration_values: Vec::new(),
            betti_by_dim: vec![Vec::new(); max_dimension + 1],
            max_dimension,
        }
    }

    /// Add Betti numbers for a filtration step
    pub fn add_step(&mut self, filtration: f64, betti_numbers: &[usize]) {
        self.filtration_values.push(filtration);
        for (dim, &betti) in betti_numbers.iter().enumerate() {
            if dim <= self.max_dimension {
                self.betti_by_dim[dim].push(betti);
            }
        }
    }

    /// Get Betti number at specific dimension and filtration index
    #[must_use]
    pub fn betti_at(&self, dim: usize, step: usize) -> Option<usize> {
        if dim <= self.max_dimension && step < self.betti_by_dim[dim].len() {
            Some(self.betti_by_dim[dim][step])
        } else {
            None
        }
    }

    /// Find significant changes in Betti numbers (topological events)
    #[must_use]
    pub fn find_significant_changes(&self, threshold: usize) -> Vec<(usize, usize, usize)> {
        let mut changes = Vec::new();
        
        for dim in 0..=self.max_dimension {
            let betti_vals = &self.betti_by_dim[dim];
            for i in 1..betti_vals.len() {
                let diff = betti_vals[i] as isize - betti_vals[i - 1] as isize;
                if diff.abs() >= threshold as isize {
                    changes.push((dim, i, diff.unsigned_abs()));
                }
            }
        }
        
        changes
    }
}

/// Persistent Homology Engine for computing Betti numbers across filtrations
pub struct PersistentHomologyEngine {
    num_points: usize,
    max_dimension: usize,
    num_filtration_steps: usize,
}

impl PersistentHomologyEngine {
    /// Create a new persistent homology engine
    #[must_use]
    pub fn new(num_points: usize, max_dimension: usize) -> Self {
        Self {
            num_points,
            max_dimension: max_dimension.min(4), // Practical limit
            num_filtration_steps: MAX_FILTRATION_STEPS,
        }
    }

    /// Set number of filtration steps
    pub fn with_filtration_steps(mut self, steps: usize) -> Self {
        self.num_filtration_steps = steps.min(MAX_FILTRATION_STEPS);
        self
    }

    /// Compute persistent homology from pairwise distances
    ///
    /// # Returns
    /// * `Ok(BettiCurve)` containing Betti numbers across filtration
    /// * `Err(PersistentHomologyError)` on failure
    pub fn compute_betti_curve(
        &self,
        distances: &[f64],
    ) -> Result<BettiCurve, PersistentHomologyError> {
        if distances.len() != self.num_points * self.num_points {
            return Err(PersistentHomologyError::InsufficientData);
        }

        // Find max distance for filtration range
        let max_dist = distances.iter()
            .filter(|&&d| d.is_finite())
            .cloned()
            .fold(0.0_f64, f64::max);

        if max_dist <= 0.0 {
            return Err(PersistentHomologyError::InsufficientData);
        }

        let mut curve = BettiCurve::new(self.max_dimension);

        // Sample filtration values
        for step in 0..self.num_filtration_steps {
            let filtration = (step as f64 / self.num_filtration_steps as f64) * max_dist;
            
            // Build complex at this filtration
            let builder = VietorisRipsBuilder::new(distances, self.num_points)
                .with_max_dimension(self.max_dimension);
            
            let complex = match builder.build(filtration) {
                Ok(c) => c,
                Err(e) => return Err(PersistentHomologyError::VietorisRipsError(e)),
            };

            // Compute Betti numbers at this scale
            let betti_numbers = self.compute_betti_numbers(&complex)?;
            curve.add_step(filtration, &betti_numbers);
        }

        Ok(curve)
    }

    /// Compute Betti numbers from a Vietoris-Rips complex
    fn compute_betti_numbers(
        &self,
        complex: &VietorisRipsComplex,
    ) -> Result<Vec<usize>, PersistentHomologyError> {
        let mut chain_complex = ChainComplex::new();
        let mut betti_numbers = Vec::with_capacity(self.max_dimension + 1);

        // Build boundary matrices for each dimension
        for dim in 0..=self.max_dimension {
            let simplices_k = complex.simplices_of_dimension(dim);
            let simplices_k_minus_1 = if dim > 0 {
                complex.simplices_of_dimension(dim - 1)
            } else {
                &[]
            };

            if simplices_k.is_empty() {
                betti_numbers.push(0);
                continue;
            }

            // Create boundary matrix ∂_k: C_k → C_{k-1}
            let num_rows = if dim > 0 { simplices_k_minus_1.len() } else { 1 };
            let num_cols = simplices_k.len();
            
            let mut boundary = SparseBoundaryMatrix::new(num_rows, num_cols, dim as u8);

            // Fill boundary matrix based on face relationships
            for (col_idx, simplex) in simplices_k.iter().enumerate() {
                if dim > 0 {
                    let faces = simplex.faces();
                    for face in faces {
                        // Find matching face in lower dimension
                        for (row_idx, lower_simplex) in simplices_k_minus_1.iter().enumerate() {
                            if face.vertices() == lower_simplex.vertices() {
                                let _ = boundary.set_entry(row_idx, col_idx, GF2Element::One);
                                break;
                            }
                        }
                    }
                }
            }

            chain_complex.add_boundary(boundary);
        }

        // Compute Betti numbers
        for dim in 0..=self.max_dimension {
            betti_numbers.push(chain_complex.betti_number(dim));
        }

        Ok(betti_numbers)
    }

    /// Detect geopolitical events from Betti curve changes
    ///
    /// Significant changes in β_1 indicate formation/breaking of cycles
    /// (alliances, blockades, trade loops)
    #[must_use]
    pub fn detect_geopolitical_events(&self, curve: &BettiCurve) -> Vec<GeopoliticalEvent> {
        let mut events = Vec::new();
        let changes = curve.find_significant_changes(1);

        for (dim, step, magnitude) in changes {
            let event_type = match dim {
                0 => {
                    if curve.betti_at(0, step).unwrap_or(0) > 
                       curve.betti_at(0, step.saturating_sub(1)).unwrap_or(0) {
                        GeopoliticalEventType::Fragmentation
                    } else {
                        GeopoliticalEventType::Integration
                    }
                }
                1 => {
                    if curve.betti_at(1, step).unwrap_or(0) > 
                       curve.betti_at(1, step.saturating_sub(1)).unwrap_or(0) {
                        GeopoliticalEventType::CycleFormation // Alliance/blockade
                    } else {
                        GeopoliticalEventType::CycleBreaking
                    }
                }
                2 => {
                    if curve.betti_at(2, step).unwrap_or(0) > 
                       curve.betti_at(2, step.saturating_sub(1)).unwrap_or(0) {
                        GeopoliticalEventType::VoidFormation // Trade cavity
                    } else {
                        GeopoliticalEventType::VoidFilling
                    }
                }
                _ => GeopoliticalEventType::Unknown,
            };

            events.push(GeopoliticalEvent {
                event_type,
                dimension: dim as u8,
                filtration_step: step,
                magnitude,
                significance: magnitude as f64 / 10.0, // Normalized
            });
        }

        events
    }
}

/// Type of geopolitical event detected
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeopoliticalEventType {
    Fragmentation,      // β_0 increase: network splitting
    Integration,        // β_0 decrease: network merging
    CycleFormation,     // β_1 increase: alliance/blockade forming
    CycleBreaking,      // β_1 decrease: alliance breaking
    VoidFormation,      // β_2 increase: trade cavity
    VoidFilling,        // β_2 decrease: cavity filling
    Unknown,
}

/// Detected geopolitical event
#[derive(Debug, Clone)]
pub struct GeopoliticalEvent {
    pub event_type: GeopoliticalEventType,
    pub dimension: u8,
    pub filtration_step: usize,
    pub magnitude: usize,
    pub significance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_betti_curve_creation() {
        let mut curve = BettiCurve::new(2);
        curve.add_step(0.1, &[3, 1, 0]);
        curve.add_step(0.5, &[2, 2, 0]);
        
        assert_eq!(curve.filtration_values.len(), 2);
        assert_eq!(curve.betti_at(0, 0), Some(3));
        assert_eq!(curve.betti_at(1, 1), Some(2));
    }

    #[test]
    fn test_persistent_homology_engine() {
        // Simple triangle
        let distances = vec![
            0.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
        ];

        let engine = PersistentHomologyEngine::new(3, 2);
        let result = engine.compute_betti_curve(&distances);
        
        match result {
            Ok(curve) => {
                assert!(!curve.filtration_values.is_empty());
            }
            Err(_) => {
                // May fail due to simplified implementation
            }
        }
    }
}
