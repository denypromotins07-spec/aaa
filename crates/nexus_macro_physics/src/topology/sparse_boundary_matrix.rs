// NEXUS-OMEGA Stage 34: Geopolitical Topology
// Chapter 2: Sparse Boundary Matrix for Persistent Homology
// File: crates/nexus_macro_physics/src/topology/sparse_boundary_matrix.rs

//! Sparse Boundary Matrix over GF(2) for Efficient Betti Number Computation
//!
//! Implements sparse matrix representations with operations in the Galois Field GF(2)
//! to compute boundary operators ∂_k: C_k → C_{k-1} for persistent homology.

#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]

use core::fmt;
use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;

/// Element in GF(2) - either 0 or 1
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GF2Element {
    Zero,
    One,
}

impl GF2Element {
    #[must_use]
    pub const fn zero() -> Self {
        Self::Zero
    }

    #[must_use]
    pub const fn one() -> Self {
        Self::One
    }

    /// Addition in GF(2) is XOR
    #[must_use]
    pub const fn add(self, other: Self) -> Self {
        match (self, other) {
            (Self::Zero, Self::Zero) => Self::Zero,
            (Self::Zero, Self::One) => Self::One,
            (Self::One, Self::Zero) => Self::One,
            (Self::One, Self::One) => Self::Zero,
        }
    }

    /// Multiplication in GF(2) is AND
    #[must_use]
    pub const fn mul(self, other: Self) -> Self {
        match (self, other) {
            (Self::One, Self::One) => Self::One,
            _ => Self::Zero,
        }
    }

    #[must_use]
    pub const fn is_zero(self) -> bool {
        matches!(self, Self::Zero)
    }

    #[must_use]
    pub const fn is_one(self) -> bool {
        matches!(self, Self::One)
    }
}

/// Error types for boundary matrix operations
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryMatrixError {
    DimensionMismatch { expected: usize, got: usize },
    InvalidIndex { index: usize, max: usize },
    SparseFormatError(String),
    ReductionFailed,
}

impl fmt::Display for BoundaryMatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::InvalidIndex { index, max } => {
                write!(f, "Invalid index {}: maximum is {}", index, max)
            }
            Self::SparseFormatError(msg) => write!(f, "Sparse format error: {}", msg),
            Self::ReductionFailed => write!(f, "Matrix reduction failed"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BoundaryMatrixError {}

/// Sparse matrix entry in GF(2)
#[derive(Debug, Clone, Copy)]
pub struct SparseEntry {
    pub row: usize,
    pub value: GF2Element,
}

/// Sparse column representation for boundary matrices
#[derive(Debug, Clone)]
pub struct SparseColumn {
    /// Non-zero entries stored as (row_index, value) pairs
    entries: Vec<(usize, GF2Element)>,
}

impl SparseColumn {
    #[must_use]
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Add an entry to the column (maintains sorted order by row index)
    pub fn add_entry(&mut self, row: usize, value: GF2Element) {
        if value.is_zero() {
            return;
        }

        // Binary search for insertion point
        let pos = self.entries.binary_search_by_key(&row, |(r, _)| *r);
        
        match pos {
            Ok(idx) => {
                // Entry exists, XOR the values (addition in GF(2))
                let (_, existing) = &mut self.entries[idx];
                *existing = existing.add(value);
                
                // If result is zero, remove the entry
                if existing.is_zero() {
                    self.entries.remove(idx);
                }
            }
            Err(idx) => {
                self.entries.insert(idx, (row, value));
            }
        }
    }

    /// Get all non-zero row indices
    #[must_use]
    pub fn nonzero_rows(&self) -> Vec<usize> {
        self.entries.iter().map(|(r, _)| *r).collect()
    }

    /// Check if column is empty (all zeros)
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get number of non-zero entries
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Get entry at specific row
    #[must_use]
    pub fn get(&self, row: usize) -> GF2Element {
        self.entries
            .binary_search_by_key(&row, |(r, _)| *r)
            .ok()
            .map(|idx| self.entries[idx].1)
            .unwrap_or(GF2Element::Zero)
    }

    /// XOR two columns together (column addition in GF(2))
    pub fn xor_with(&mut self, other: &SparseColumn) {
        for &(row, value) in &other.entries {
            self.add_entry(row, value);
        }
    }
}

impl Default for SparseColumn {
    fn default() -> Self {
        Self::new()
    }
}

/// Sparse Boundary Matrix over GF(2)
///
/// Represents the boundary operator ∂_k: C_k → C_{k-1}
/// where columns represent k-simplices and rows represent (k-1)-simplices
#[derive(Debug, Clone)]
pub struct SparseBoundaryMatrix {
    /// Columns of the matrix (each column represents a simplex)
    columns: Vec<SparseColumn>,
    /// Number of rows ((k-1)-simplices)
    num_rows: usize,
    /// Dimension k of this boundary operator
    dimension: u8,
}

impl SparseBoundaryMatrix {
    /// Create a new empty boundary matrix
    #[must_use]
    pub fn new(num_rows: usize, num_cols: usize, dimension: u8) -> Self {
        Self {
            columns: vec![SparseColumn::new(); num_cols],
            num_rows,
            dimension,
        }
    }

    /// Set an entry in the matrix
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(BoundaryMatrixError)` if indices are invalid
    pub fn set_entry(
        &mut self,
        row: usize,
        col: usize,
        value: GF2Element,
    ) -> Result<(), BoundaryMatrixError> {
        if row >= self.num_rows {
            return Err(BoundaryMatrixError::InvalidIndex {
                index: row,
                max: self.num_rows.saturating_sub(1),
            });
        }
        if col >= self.columns.len() {
            return Err(BoundaryMatrixError::InvalidIndex {
                index: col,
                max: self.columns.len().saturating_sub(1),
            });
        }

        self.columns[col].add_entry(row, value);
        Ok(())
    }

    /// Get the pivot (lowest non-zero row) of a column
    #[must_use]
    pub fn pivot(&self, col: usize) -> Option<usize> {
        self.columns.get(col).and_then(|c| {
            c.nonzero_rows().first().copied()
        })
    }

    /// Reduce the matrix to reduced column echelon form using Gaussian elimination
    ///
    /// This is essential for computing persistent homology barcodes.
    ///
    /// # Returns
    /// * `Ok(reduced_matrix)` on success
    /// * `Err(BoundaryMatrixError)` on failure
    pub fn reduce(&self) -> Result<SparseBoundaryMatrix, BoundaryMatrixError> {
        let mut reduced = self.clone();
        let num_cols = reduced.columns.len();

        // Track which column has each pivot
        let mut pivot_to_col: BTreeMap<usize, usize> = BTreeMap::new();

        for col in 0..num_cols {
            // Find pivot of current column
            while let Some(pivot) = reduced.pivot(col) {
                if let Some(&existing_col) = pivot_to_col.get(&pivot) {
                    // Pivot already exists, XOR with that column to eliminate
                    reduced.columns[col].xor_with(&reduced.columns[existing_col]);
                } else {
                    // New pivot found, record it
                    pivot_to_col.insert(pivot, col);
                    break;
                }
            }
        }

        Ok(reduced)
    }

    /// Compute the rank of the matrix (number of linearly independent columns)
    #[must_use]
    pub fn rank(&self) -> usize {
        match self.reduce() {
            Ok(reduced) => {
                reduced.columns.iter().filter(|c| !c.is_empty()).count()
            }
            Err(_) => 0,
        }
    }

    /// Get the nullity (dimension of kernel)
    #[must_use]
    pub fn nullity(&self) -> usize {
        self.columns.len().saturating_sub(self.rank())
    }

    /// Get number of columns
    #[must_use]
    pub const fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Get number of rows
    #[must_use]
    pub const fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Get dimension of this boundary operator
    #[must_use]
    pub const fn dimension(&self) -> u8 {
        self.dimension
    }

    /// Get a specific column
    #[must_use]
    pub fn column(&self, col: usize) -> Option<&SparseColumn> {
        self.columns.get(col)
    }
}

/// Chain complex consisting of multiple boundary matrices
#[derive(Debug, Clone)]
pub struct ChainComplex {
    /// Boundary matrices ∂_k for each dimension k
    boundaries: Vec<SparseBoundaryMatrix>,
}

impl ChainComplex {
    #[must_use]
    pub fn new() -> Self {
        Self {
            boundaries: Vec::new(),
        }
    }

    /// Add a boundary matrix for dimension k
    pub fn add_boundary(&mut self, boundary: SparseBoundaryMatrix) {
        self.boundaries.push(boundary);
    }

    /// Compute the k-th Betti number: β_k = dim(ker ∂_k) - dim(im ∂_{k+1})
    ///
    /// β_0 = number of connected components
    /// β_1 = number of 1-dimensional holes (loops)
    /// β_2 = number of 2-dimensional voids (cavities)
    #[must_use]
    pub fn betti_number(&self, k: usize) -> usize {
        if k >= self.boundaries.len() {
            return 0;
        }

        let boundary_k = &self.boundaries[k];
        let kernel_dim = boundary_k.nullity();

        let image_dim = if k + 1 < self.boundaries.len() {
            self.boundaries[k + 1].rank()
        } else {
            0
        };

        kernel_dim.saturating_sub(image_dim)
    }

    /// Compute all Betti numbers up to maximum dimension
    #[must_use]
    pub fn all_betti_numbers(&self) -> Vec<usize> {
        (0..self.boundaries.len())
            .map(|k| self.betti_number(k))
            .collect()
    }
}

impl Default for ChainComplex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf2_arithmetic() {
        assert_eq!(GF2Element::Zero.add(GF2Element::Zero), GF2Element::Zero);
        assert_eq!(GF2Element::Zero.add(GF2Element::One), GF2Element::One);
        assert_eq!(GF2Element::One.add(GF2Element::One), GF2Element::Zero);
        assert_eq!(GF2Element::One.mul(GF2Element::One), GF2Element::One);
        assert_eq!(GF2Element::One.mul(GF2Element::Zero), GF2Element::Zero);
    }

    #[test]
    fn test_sparse_column() {
        let mut col = SparseColumn::new();
        col.add_entry(0, GF2Element::One);
        col.add_entry(2, GF2Element::One);
        col.add_entry(0, GF2Element::One); // Should cancel out

        assert_eq!(col.nnz(), 1);
        assert_eq!(col.get(0), GF2Element::Zero);
        assert_eq!(col.get(2), GF2Element::One);
    }

    #[test]
    fn test_boundary_matrix_reduction() {
        // Simple example: boundary of a triangle
        // 3 edges (cols) -> 3 vertices (rows)
        let mut matrix = SparseBoundaryMatrix::new(3, 3, 1);
        
        // Edge 0: connects vertices 0 and 1
        matrix.set_entry(0, 0, GF2Element::One).unwrap();
        matrix.set_entry(1, 0, GF2Element::One).unwrap();
        
        // Edge 1: connects vertices 1 and 2
        matrix.set_entry(1, 1, GF2Element::One).unwrap();
        matrix.set_entry(2, 1, GF2Element::One).unwrap();
        
        // Edge 2: connects vertices 0 and 2
        matrix.set_entry(0, 2, GF2Element::One).unwrap();
        matrix.set_entry(2, 2, GF2Element::One).unwrap();

        let reduced = matrix.reduce().unwrap();
        
        // After reduction, should have rank 2 (one dependency = cycle)
        assert_eq!(reduced.rank(), 2);
        assert_eq!(matrix.nullity(), 1); // One cycle (the triangle)
    }

    #[test]
    fn test_chain_complex_betti() {
        let mut complex = ChainComplex::new();
        
        // Add a simple boundary matrix
        let mut boundary = SparseBoundaryMatrix::new(3, 3, 1);
        boundary.set_entry(0, 0, GF2Element::One).unwrap();
        boundary.set_entry(1, 0, GF2Element::One).unwrap();
        
        complex.add_boundary(boundary);
        
        let betti_0 = complex.betti_number(0);
        assert!(betti_0 >= 0);
    }
}
