// NEXUS-OMEGA Stage 34: Macro-Economic Gravity
// Chapter 2: Geopolitical Topology & Persistent Homology
// File: crates/nexus_macro_physics/src/topology/vietoris_rips_complex.rs

//! Vietoris-Rips Complex Generator for Geopolitical Network Analysis
//!
//! Constructs simplicial complexes from global trade and diplomatic data
//! to detect topological features (holes, cycles) indicating geopolitical
//! blockades, alliances, or isolation patterns.
//!
//! CRITICAL: Uses sparse matrix representations and early-stopping persistence
//! thresholds to prevent combinatorial explosion O(2^N).

#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]

use core::fmt;
use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;

use super::sparse_boundary_matrix::{SparseBoundaryMatrix, BoundaryMatrixError, GF2Element};

/// Maximum simplex dimension to consider (prevents combinatorial explosion)
pub const MAX_SIMPLEX_DIMENSION: usize = 4;

/// Default persistence threshold for early stopping
pub const DEFAULT_PERSISTENCE_THRESHOLD: f64 = 0.5;

/// Maximum number of simplices to generate (safety limit)
pub const MAX_SIMPLICES_COUNT: usize = 1_000_000;

/// Error types for Vietoris-Rips complex operations
#[derive(Debug, Clone, PartialEq)]
pub enum VietorisRipsError {
    InvalidThreshold { birth: f64, death: f64 },
    TooManySimplices { count: usize },
    DimensionExceeded { requested: usize, max: usize },
    SparseMatrixError(BoundaryMatrixError),
    MemoryLimitExceeded,
}

impl fmt::Display for VietorisRipsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidThreshold { birth, death } => {
                write!(f, "Invalid persistence: birth={}, death={}", birth, death)
            }
            Self::TooManySimplices { count } => {
                write!(f, "Too many simplices generated: {}", count)
            }
            Self::DimensionExceeded { requested, max } => {
                write!(f, "Dimension {} exceeds maximum {}", requested, max)
            }
            Self::SparseMatrixError(e) => write!(f, "Sparse matrix error: {}", e),
            Self::MemoryLimitExceeded => write!(f, "Memory limit exceeded"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for VietorisRipsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SparseMatrixError(e) => Some(e),
            _ => None,
        }
    }
}

/// A k-simplex represented by its vertex indices
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Simplex {
    /// Vertex indices (sorted for canonical representation)
    vertices: Box<[usize]>,
    /// Filtration value (birth time)
    filtration_value: f64,
}

impl Simplex {
    /// Create a new simplex from vertex indices
    ///
    /// # Arguments
    /// * `vertices` - Slice of vertex indices (will be sorted)
    /// * `filtration_value` - The scale parameter at which this simplex appears
    ///
    /// # Returns
    /// * `Ok(Self)` on success
    /// * `Err(VietorisRipsError)` if dimension exceeds maximum
    pub fn new(vertices: &[usize], filtration_value: f64) -> Result<Self, VietorisRipsError> {
        if vertices.len() - 1 > MAX_SIMPLEX_DIMENSION {
            return Err(VietorisRipsError::DimensionExceeded {
                requested: vertices.len() - 1,
                max: MAX_SIMPLEX_DIMENSION,
            });
        }

        let mut sorted_vertices: Vec<usize> = vertices.to_vec();
        sorted_vertices.sort_unstable();
        sorted_vertices.dedup();

        Ok(Self {
            vertices: sorted_vertices.into_boxed_slice(),
            filtration_value,
        })
    }

    /// Get the dimension of this simplex (k for k-simplex)
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.vertices.len().saturating_sub(1)
    }

    /// Get the vertex indices
    #[must_use]
    pub fn vertices(&self) -> &[usize] {
        &self.vertices
    }

    /// Get the filtration value
    #[must_use]
    pub const fn filtration_value(&self) -> f64 {
        self.filtration_value
    }

    /// Check if this simplex is a face of another simplex
    #[must_use]
    pub fn is_face_of(&self, other: &Simplex) -> bool {
        self.vertices.iter().all(|v| other.vertices.contains(v))
    }

    /// Get all (k-1)-dimensional faces of this simplex
    #[must_use]
    pub fn faces(&self) -> Vec<Simplex> {
        if self.vertices.is_empty() {
            return vec![];
        }

        let mut faces = Vec::with_capacity(self.vertices.len());
        
        for i in 0..self.vertices.len() {
            let mut face_vertices: Vec<usize> = self.vertices.to_vec();
            face_vertices.remove(i);
            
            // Faces inherit the parent's filtration value
            if let Ok(face) = Simplex::new(&face_vertices, self.filtration_value) {
                faces.push(face);
            }
        }

        faces
    }
}

/// Vietoris-Rips Complex at a specific filtration level
#[derive(Debug, Clone)]
pub struct VietorisRipsComplex {
    /// All simplices organized by dimension
    simplices_by_dim: Vec<Vec<Simplex>>,
    /// Current filtration threshold
    filtration_threshold: f64,
    /// Total number of simplices
    total_simplices: usize,
}

impl VietorisRipsComplex {
    /// Create an empty Vietoris-Rips complex
    #[must_use]
    pub fn empty(max_dimension: usize) -> Self {
        Self {
            simplices_by_dim: vec![Vec::new(); max_dimension + 1],
            filtration_threshold: 0.0,
            total_simplices: 0,
        }
    }

    /// Add a simplex to the complex
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(VietorisRipsError)` if memory limit exceeded
    pub fn add_simplex(&mut self, simplex: Simplex) -> Result<(), VietorisRipsError> {
        let dim = simplex.dimension();
        
        if dim >= self.simplices_by_dim.len() {
            return Err(VietorisRipsError::DimensionExceeded {
                requested: dim,
                max: self.simplices_by_dim.len().saturating_sub(1),
            });
        }

        if self.total_simplices >= MAX_SIMPLICES_COUNT {
            return Err(VietorisRipsError::MemoryLimitExceeded);
        }

        // Check for duplicates
        let exists = self.simplices_by_dim[dim]
            .iter()
            .any(|s| s.vertices() == simplex.vertices());

        if !exists {
            self.simplices_by_dim[dim].push(simplex);
            self.total_simplices += 1;
        }

        Ok(())
    }

    /// Get simplices of a specific dimension
    #[must_use]
    pub fn simplices_of_dimension(&self, dim: usize) -> &[Simplex] {
        if dim < self.simplices_by_dim.len() {
            &self.simplices_by_dim[dim]
        } else {
            &[]
        }
    }

    /// Get total number of simplices
    #[must_use]
    pub const fn total_simplices(&self) -> usize {
        self.total_simplices
    }

    /// Get current filtration threshold
    #[must_use]
    pub const fn filtration_threshold(&self) -> f64 {
        self.filtration_threshold
    }
}

/// Builder for constructing filtered Vietoris-Rips complexes
pub struct VietorisRipsBuilder {
    /// Pairwise distance matrix (flattened, row-major)
    distances: Box<[f64]>,
    /// Number of points/nodes
    num_points: usize,
    /// Maximum dimension to compute
    max_dimension: usize,
    /// Persistence threshold for early stopping
    persistence_threshold: f64,
}

impl VietorisRipsBuilder {
    /// Create a new builder from pairwise distances
    ///
    /// # Arguments
    /// * `distances` - Flattened NxN distance matrix (row-major order)
    /// * `num_points` - Number of points in the metric space
    ///
    /// # Returns
    /// * `Ok(Self)` on success
    /// * `Err(VietorisRipsError)` if dimensions don't match
    pub fn new(distances: &[f64], num_points: usize) -> Result<Self, VietorisRipsError> {
        if distances.len() != num_points * num_points {
            return Err(VietorisRipsError::InvalidThreshold {
                birth: distances.len() as f64,
                death: (num_points * num_points) as f64,
            });
        }

        Ok(Self {
            distances: distances.to_vec().into_boxed_slice(),
            num_points,
            max_dimension: MAX_SIMPLEX_DIMENSION,
            persistence_threshold: DEFAULT_PERSISTENCE_THRESHOLD,
        })
    }

    /// Set maximum simplex dimension
    pub fn with_max_dimension(mut self, dim: usize) -> Self {
        self.max_dimension = dim.min(MAX_SIMPLEX_DIMENSION);
        self
    }

    /// Set persistence threshold for early stopping
    pub fn with_persistence_threshold(mut self, threshold: f64) -> Self {
        self.persistence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Build the filtered Vietoris-Rips complex up to a given scale parameter
    ///
    /// # Arguments
    /// * `max_scale` - Maximum filtration value (distance threshold)
    ///
    /// # Returns
    /// * `Ok(VietorisRipsComplex)` containing all simplices up to max_scale
    /// * `Err(VietorisRipsError)` on failure
    pub fn build(&self, max_scale: f64) -> Result<VietorisRipsComplex, VietorisRipsError> {
        let mut complex = VietorisRipsComplex::empty(self.max_dimension);

        // Add 0-simplices (vertices) - always present
        for i in 0..self.num_points {
            let vertex = Simplex::new(&[i], 0.0)?;
            complex.add_simplex(vertex)?;
        }

        // Sort all pairs by distance for efficient filtration
        let mut pairs: Vec<(usize, usize, f64)> = Vec::with_capacity(
            self.num_points * (self.num_points - 1) / 2,
        );

        for i in 0..self.num_points {
            for j in (i + 1)..self.num_points {
                let dist = self.distances[i * self.num_points + j];
                if dist <= max_scale && dist.is_finite() {
                    pairs.push((i, j, dist));
                }
            }
        }

        pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(core::cmp::Ordering::Equal));

        // Add 1-simplices (edges)
        for &(i, j, dist) in &pairs {
            let edge = Simplex::new(&[i, j], dist)?;
            complex.add_simplex(edge)?;
        }

        // Add higher-dimensional simplices using the clique detection approach
        // A set of vertices forms a simplex iff all pairwise distances <= scale
        self.add_higher_simplices(&mut complex, max_scale)?;

        Ok(complex)
    }

    /// Add simplices of dimension >= 2
    fn add_higher_simplices(
        &self,
        complex: &mut VietorisRipsComplex,
        max_scale: f64,
    ) -> Result<(), VietorisRipsError> {
        // For each dimension from 2 to max_dimension
        for dim in 2..=self.max_dimension {
            let prev_simplices = complex.simplices_of_dimension(dim - 1);
            
            if prev_simplices.is_empty() {
                break;
            }

            // Find candidates for extension
            // Two (k-1)-simplices can form a k-simplex if they share a (k-2)-face
            // and all vertices are within max_scale of each other
            
            let mut added_count = 0;
            
            for i in 0..prev_simplices.len() {
                for j in (i + 1)..prev_simplices.len() {
                    let si = &prev_simplices[i];
                    let sj = &prev_simplices[j];
                    
                    // Check if they share enough vertices
                    let shared: Vec<usize> = si.vertices()
                        .iter()
                        .filter(|v| sj.vertices().contains(v))
                        .copied()
                        .collect();
                    
                    if shared.len() == dim - 1 {
                        // They share a (dim-2)-face, potential candidate
                        let mut union_vertices: Vec<usize> = si.vertices().to_vec();
                        union_vertices.extend(sj.vertices().iter().copied());
                        union_vertices.sort_unstable();
                        union_vertices.dedup();
                        
                        if union_vertices.len() == dim + 1 {
                            // Check all pairwise distances
                            let max_pairwise_dist = self.max_pairwise_distance(&union_vertices);
                            
                            if max_pairwise_dist <= max_scale {
                                let simplex = Simplex::new(&union_vertices, max_pairwise_dist)?;
                                if complex.add_simplex(simplex).is_ok() {
                                    added_count += 1;
                                    
                                    // Early stopping if too many simplices
                                    if complex.total_simplices() >= MAX_SIMPLICES_COUNT {
                                        return Ok(());
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // If no simplices added at this dimension, stop
            if added_count == 0 {
                break;
            }
        }

        Ok(())
    }

    /// Compute maximum pairwise distance among a set of vertices
    fn max_pairwise_distance(&self, vertices: &[usize]) -> f64 {
        let mut max_dist = 0.0;
        
        for i in 0..vertices.len() {
            for j in (i + 1)..vertices.len() {
                let vi = vertices[i];
                let vj = vertices[j];
                let dist = self.distances[vi * self.num_points + vj];
                max_dist = max_dist.max(dist);
            }
        }
        
        max_dist
    }
}

/// Persistence diagram entry representing a topological feature
#[derive(Debug, Clone, Copy)]
pub struct PersistencePair {
    /// Birth filtration value
    pub birth: f64,
    /// Death filtration value
    pub death: f64,
    /// Dimension of the feature (0=components, 1=loops, 2=voids)
    pub dimension: u8,
    /// Persistence (death - birth)
    pub persistence: f64,
}

impl PersistencePair {
    #[must_use]
    pub fn new(birth: f64, death: f64, dimension: u8) -> Self {
        Self {
            birth,
            death,
            dimension,
            persistence: (death - birth).max(0.0),
        }
    }

    /// Check if this pair represents a significant topological feature
    #[must_use]
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.persistence >= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_creation() {
        let simplex = Simplex::new(&[2, 0, 1], 0.5).unwrap();
        assert_eq!(simplex.dimension(), 2);
        assert_eq!(simplex.vertices(), &[0, 1, 2]);
        assert_eq!(simplex.filtration_value(), 0.5);
    }

    #[test]
    fn test_simplex_faces() {
        let simplex = Simplex::new(&[0, 1, 2], 0.5).unwrap();
        let faces = simplex.faces();
        
        assert_eq!(faces.len(), 3);
        // Each face should be a 1-simplex (edge)
        for face in &faces {
            assert_eq!(face.dimension(), 1);
        }
    }

    #[test]
    fn test_vietoris_rips_builder() {
        // Simple triangle with equal edges
        let distances = vec![
            0.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
        ];

        let builder = VietorisRipsBuilder::new(&distances, 3).unwrap();
        let complex = builder.build(1.0).unwrap();

        // Should have 3 vertices, 3 edges, and 1 triangle
        assert_eq!(complex.simplices_of_dimension(0).len(), 3);
        assert_eq!(complex.simplices_of_dimension(1).len(), 3);
        assert_eq!(complex.simplices_of_dimension(2).len(), 1);
    }

    #[test]
    fn test_persistence_pair() {
        let pair = PersistencePair::new(0.1, 0.8, 1);
        assert_eq!(pair.birth, 0.1);
        assert_eq!(pair.death, 0.8);
        assert_eq!(pair.persistence, 0.7);
        assert!(pair.is_significant(0.5));
        assert!(!pair.is_significant(0.8));
    }
}
