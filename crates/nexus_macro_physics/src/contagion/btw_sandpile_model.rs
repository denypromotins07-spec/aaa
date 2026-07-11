// NEXUS-OMEGA Stage 34: Contagion Percolation
// Chapter 4: Bak-Tang-Wiesenfeld Sandpile Model for Sovereign Debt
// File: crates/nexus_macro_physics/src/contagion/btw_sandpile_model.rs

//! Bak-Tang-Wiesenfeld (BTW) Sandpile Model for Sovereign Default Cascades
//!
//! Models the global CDS network as a Self-Organized Criticality (SOC) system.
//! Each nation holds "grains of sand" representing short-term debt rollover risk.
//! When a node exceeds its critical threshold, it topples, potentially triggering
//! an avalanche of defaults across the global banking system.

#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]

use core::fmt;
use alloc::vec::Vec;
use alloc::boxed::Box;
use alloc::collections::BTreeMap;

/// Maximum grid size for sandpile simulation
pub const MAX_GRID_SIZE: usize = 256;

/// Error types for sandpile operations
#[derive(Debug, Clone, PartialEq)]
pub enum SandpileError {
    InvalidGridSize { size: usize },
    OverflowDetected,
    InfiniteLoopDetected,
    InvalidNodeIndex { index: usize, max: usize },
}

impl fmt::Display for SandpileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGridSize { size } => write!(f, "Invalid grid size: {}", size),
            Self::OverflowDetected => write!(f, "Numerical overflow detected"),
            Self::InfiniteLoopDetected => write!(f, "Infinite loop detected in toppling"),
            Self::InvalidNodeIndex { index, max } => {
                write!(f, "Invalid node index {}: maximum is {}", index, max)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SandpileError {}

/// State of a single node in the sandpile
#[derive(Debug, Clone, Copy)]
pub struct SandpileNode {
    /// Number of grains (debt rollover risk units)
    pub grains: u32,
    /// Critical threshold for toppling
    pub critical_threshold: u32,
    /// Whether this node has toppled in current avalanche
    pub has_toppled: bool,
}

impl SandpileNode {
    #[must_use]
    pub fn new(critical_threshold: u32) -> Self {
        Self {
            grains: 0,
            critical_threshold,
            has_toppled: false,
        }
    }

    #[must_use]
    pub fn is_critical(&self) -> bool {
        self.grains >= self.critical_threshold
    }

    /// Add a grain and check if node becomes critical
    pub fn add_grain(&mut self) -> bool {
        self.grains = self.grains.saturating_add(1);
        self.is_critical()
    }
}

/// BTW Sandpile Model state
pub struct BTWSandpileModel {
    /// Grid dimension (width = height)
    dim: usize,
    /// Nodes arranged in row-major order
    nodes: Box<[SandpileNode]>,
    /// Total grains added
    total_grains: u64,
    /// Total topplings
    total_topplings: u64,
    /// Current avalanche size
    current_avalanche_size: u64,
}

impl BTWSandpileModel {
    /// Create a new sandpile model
    ///
    /// # Arguments
    /// * `dim` - Grid dimension (dim x dim grid)
    /// * `critical_threshold` - Threshold for toppling (typically 4 for 2D)
    ///
    /// # Returns
    /// * `Ok(Self)` on success
    /// * `Err(SandpileError)` on failure
    pub fn new(dim: usize, critical_threshold: u32) -> Result<Self, SandpileError> {
        if dim == 0 || dim > MAX_GRID_SIZE {
            return Err(SandpileError::InvalidGridSize { size: dim });
        }

        let num_nodes = dim * dim;
        let mut nodes = Vec::with_capacity(num_nodes);

        for _ in 0..num_nodes {
            nodes.push(SandpileNode::new(critical_threshold));
        }

        Ok(Self {
            dim,
            nodes: nodes.into_boxed_slice(),
            total_grains: 0,
            total_topplings: 0,
            current_avalanche_size: 0,
        })
    }

    /// Get node index from coordinates
    #[inline]
    fn get_index(&self, x: usize, y: usize) -> Option<usize> {
        if x < self.dim && y < self.dim {
            Some(y * self.dim + x)
        } else {
            None
        }
    }

    /// Get neighbor indices for a given position (4-connectivity)
    fn get_neighbors(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        let mut neighbors = Vec::with_capacity(4);

        if x > 0 { neighbors.push((x - 1, y)); }
        if x + 1 < self.dim { neighbors.push((x + 1, y)); }
        if y > 0 { neighbors.push((x, y - 1)); }
        if y + 1 < self.dim { neighbors.push((x, y + 1)); }

        neighbors
    }

    /// Add a grain at specified position
    ///
    /// # Returns
    /// * `Ok(avalanche_size)` - Number of topplings triggered
    /// * `Err(SandpileError)` on failure
    pub fn add_grain(&mut self, x: usize, y: usize) -> Result<u64, SandpileError> {
        let idx = self.get_index(x, y)
            .ok_or_else(|| SandpileError::InvalidNodeIndex { 
                index: x.max(y), 
                max: self.dim 
            })?;

        self.nodes[idx].add_grain();
        self.total_grains += 1;
        self.current_avalanche_size = 0;

        // Trigger relaxation if critical
        if self.nodes[idx].is_critical() {
            self.relax()?;
        }

        Ok(self.current_avalanche_size)
    }

    /// Relax the sandpile by toppling critical nodes
    fn relax(&mut self) -> Result<(), SandpileError> {
        let mut stack: Vec<(usize, usize)> = Vec::new();
        let mut iteration_count: u64 = 0;
        let max_iterations = (self.dim * self.dim * 100) as u64;

        // Find all critical nodes
        for y in 0..self.dim {
            for x in 0..self.dim {
                let idx = y * self.dim + x;
                if self.nodes[idx].is_critical() && !self.nodes[idx].has_toppled {
                    stack.push((x, y));
                }
            }
        }

        while let Some((x, y)) = stack.pop() {
            iteration_count += 1;
            if iteration_count > max_iterations {
                return Err(SandpileError::InfiniteLoopDetected);
            }

            let idx = y * self.dim + x;
            
            // Skip if already toppled or no longer critical
            if self.nodes[idx].has_toppled || !self.nodes[idx].is_critical() {
                continue;
            }

            // Topple this node
            self.topple(x, y)?;
            self.current_avalanche_size += 1;
            self.total_topplings += 1;

            // Check neighbors for new critical nodes
            for (nx, ny) in self.get_neighbors(x, y) {
                let nidx = ny * self.dim + nx;
                if self.nodes[nidx].is_critical() && !self.nodes[nidx].has_toppled {
                    stack.push((nx, ny));
                }
            }
        }

        // Reset toppled flags for next avalanche
        for node in &mut self.nodes {
            node.has_toppled = false;
        }

        Ok(())
    }

    /// Topple a single node, distributing grains to neighbors
    fn topple(&mut self, x: usize, y: usize) -> Result<(), SandpileError> {
        let idx = y * self.dim + x;
        let node = &mut self.nodes[idx];
        
        let threshold = node.critical_threshold;
        let grains_to_distribute = node.grains / threshold;
        node.grains %= threshold;
        node.has_toppled = true;

        // Distribute grains to neighbors
        let neighbors = self.get_neighbors(x, y);
        let grains_per_neighbor = grains_to_distribute;

        for (nx, ny) in neighbors {
            let nidx = ny * self.dim + nx;
            self.nodes[nidx].grains = self.nodes[nidx]
                .grains
                .saturating_add(grains_per_neighbor);
        }

        // Grains that fall off the edge are lost (dissipation)
        // This prevents infinite loops and models capital flight

        Ok(())
    }

    /// Get statistics about the sandpile state
    #[must_use]
    pub fn statistics(&self) -> SandpileStatistics {
        let mut min_grains = u32::MAX;
        let mut max_grains = 0_u32;
        let mut total_grains: u64 = 0;
        let mut critical_count = 0_usize;

        for node in &self.nodes {
            min_grains = min_grains.min(node.grains);
            max_grains = max_grains.max(node.grains);
            total_grains += node.grains as u64;
            if node.is_critical() {
                critical_count += 1;
            }
        }

        SandpileStatistics {
            dim: self.dim,
            total_nodes: self.dim * self.dim,
            total_grains_stored: total_grains,
            total_grains_added: self.total_grains,
            total_topplings: self.total_topplings,
            min_grains,
            max_grains,
            avg_grains: total_grains / (self.dim * self.dim) as u64,
            critical_nodes: critical_count,
            current_avalanche_size: self.current_avalanche_size,
        }
    }

    /// Get grains at specific position
    #[must_use]
    pub fn grains_at(&self, x: usize, y: usize) -> Option<u32> {
        self.get_index(x, y).map(|idx| self.nodes[idx].grains)
    }

    /// Get grid dimension
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dim
    }
}

/// Statistics about the sandpile state
#[derive(Debug, Clone)]
pub struct SandpileStatistics {
    pub dim: usize,
    pub total_nodes: usize,
    pub total_grains_stored: u64,
    pub total_grains_added: u64,
    pub total_topplings: u64,
    pub min_grains: u32,
    pub max_grains: u32,
    pub avg_grains: u64,
    pub critical_nodes: usize,
    pub current_avalanche_size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandpile_creation() {
        let model = BTWSandpileModel::new(10, 4);
        assert!(model.is_ok());
        
        let model = model.unwrap();
        assert_eq!(model.dimension(), 10);
    }

    #[test]
    fn test_add_grain_no_topple() {
        let mut model = BTWSandpileModel::new(5, 4).unwrap();
        
        // Add grains below threshold
        for _ in 0..3 {
            let avalanche = model.add_grain(2, 2).unwrap();
            assert_eq!(avalanche, 0);
        }
    }

    #[test]
    fn test_add_grain_with_topple() {
        let mut model = BTWSandpileModel::new(5, 4).unwrap();
        
        // Add grains until toppling
        for _ in 0..4 {
            model.add_grain(2, 2).unwrap();
        }
        
        let stats = model.statistics();
        assert!(stats.total_topplings > 0);
    }

    #[test]
    fn test_invalid_grid_size() {
        let result = BTWSandpileModel::new(0, 4);
        assert!(result.is_err());
        
        let result = BTWSandpileModel::new(MAX_GRID_SIZE + 1, 4);
        assert!(result.is_err());
    }
}
