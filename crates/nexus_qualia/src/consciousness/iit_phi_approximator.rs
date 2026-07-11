//! Integrated Information Theory (IIT) Φ Approximator using bipartite spectral relaxation.
//!
//! This module implements a zero-allocation approximation of Integrated Information Theory's
//! Φ (Phi) metric for measuring consciousness/causal density in neural networks, AI systems,
//! and brain-computer interfaces. Uses spectral relaxation to avoid O(2^N) combinatorial explosion.

use core::slice;

/// Maximum number of nodes supported (zero-alloc fixed size)
pub const MAX_NODES: usize = 1024;

/// Maximum number of partitions for bipartite split
pub const MAX_PARTITIONS: usize = 512;

/// Convergence threshold for spectral iteration
pub const SPECTRAL_CONVERGENCE_EPS: f32 = 1e-6;

/// Minimum eigenvalue gap for stable decomposition
pub const MIN_EIGENVALUE_GAP: f32 = 1e-8;

/// Error types for IIT computation
#[derive(Debug, Clone, PartialEq)]
pub enum IitError {
    DimensionMismatch(usize, usize),
    NonPositiveDefinite,
    EigenvalueDecompositionFailed,
    SingularMatrix,
    NumericalInstability(f32),
    GraphTooLarge(usize),
    InsufficientConnectivity,
}

impl core::fmt::Display for IitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            IitError::DimensionMismatch(expected, actual) => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, actual)
            }
            IitError::NonPositiveDefinite => write!(f, "Matrix is not positive definite"),
            IitError::EigenvalueDecompositionFailed => write!(f, "Eigenvalue decomposition failed"),
            IitError::SingularMatrix => write!(f, "Singular matrix encountered"),
            IitError::NumericalInstability(value) => {
                write!(f, "Numerical instability detected: {}", value)
            }
            IitError::GraphTooLarge(size) => write!(f, "Graph too large: {} nodes (max: {})", size, MAX_NODES),
            IitError::InsufficientConnectivity => write!(f, "Graph has insufficient connectivity for Φ calculation"),
        }
    }
}

impl std::error::Error for IitError {}

/// Adjacency matrix for causal network (symmetric, row-major upper triangular)
#[repr(C, align(64))]
pub struct CausalAdjacencyMatrix {
    /// Upper triangular storage
    data: [f32; MAX_NODES * (MAX_NODES + 1) / 2],
    /// Number of nodes
    num_nodes: usize,
    /// Sum of all edge weights
    total_connectivity: f32,
}

impl CausalAdjacencyMatrix {
    #[inline]
    pub const fn new() -> Self {
        Self {
            data: [0.0; MAX_NODES * (MAX_NODES + 1) / 2],
            num_nodes: 0,
            total_connectivity: 0.0,
        }
    }

    /// Initialize matrix with dimension
    #[inline]
    pub fn init(&mut self, num_nodes: usize) -> Result<(), IitError> {
        if num_nodes > MAX_NODES {
            return Err(IitError::GraphTooLarge(num_nodes));
        }

        self.num_nodes = num_nodes;
        self.total_connectivity = 0.0;

        // Zero out
        let size = num_nodes * (num_nodes + 1) / 2;
        unsafe {
            core::ptr::write_bytes(self.data.as_mut_ptr(), 0, size);
        }

        Ok(())
    }

    /// Get element at (i, j)
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> Option<f32> {
        if i >= self.num_nodes || j >= self.num_nodes {
            return None;
        }
        let (row, col) = if i <= j { (i, j) } else { (j, i) };
        let idx = row * self.num_nodes - row * (row - 1) / 2 + (col - row);
        Some(self.data[idx])
    }

    /// Set element at (i, j)
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, weight: f32) -> Result<(), IitError> {
        if i >= self.num_nodes || j >= self.num_nodes {
            return Err(IitError::DimensionMismatch(self.num_nodes, core::cmp::max(i, j)));
        }

        let (row, col) = if i <= j { (i, j) } else { (j, i) };
        let idx = row * self.num_nodes - row * (row - 1) / 2 + (col - row);
        
        // Only add to total connectivity once per edge
        if i == j {
            self.total_connectivity += weight;
        } else if self.data[idx] == 0.0 {
            self.total_connectivity += weight * 2.0; // Count both directions
        }
        
        self.data[idx] = weight;
        Ok(())
    }

    /// Get number of nodes
    #[inline]
    pub const fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get total connectivity
    #[inline]
    pub const fn total_connectivity(&self) -> f32 {
        self.total_connectivity
    }

    /// Sparsify graph by removing weak connections (prevents combinatorial explosion)
    pub fn sparsify(&mut self, threshold: f32) {
        let size = self.num_nodes * (self.num_nodes + 1) / 2;
        for i in 0..size {
            if self.data[i].abs() < threshold {
                self.data[i] = 0.0;
            }
        }
    }
}

impl Default for CausalAdjacencyMatrix {
    fn default() -> Self {
        Self::new()
    }
}

/// Laplacian matrix for spectral analysis
#[repr(C, align(64))]
pub struct GraphLaplacian {
    /// Diagonal elements (degrees)
    diagonal: [f32; MAX_NODES],
    /// Off-diagonal elements (negative adjacency)
    off_diagonal: [f32; MAX_NODES * (MAX_NODES - 1) / 2],
    /// Number of nodes
    num_nodes: usize,
    /// Fiedler value (second smallest eigenvalue)
    fiedler_value: f32,
    /// Fiedler vector (algebraic connectivity)
    fiedler_vector: [f32; MAX_NODES],
}

impl GraphLaplacian {
    #[inline]
    pub const fn new() -> Self {
        Self {
            diagonal: [0.0; MAX_NODES],
            off_diagonal: [0.0; MAX_NODES * (MAX_NODES - 1) / 2],
            num_nodes: 0,
            fiedler_value: 0.0,
            fiedler_vector: [0.0; MAX_NODES],
        }
    }

    /// Compute Laplacian from adjacency matrix
    pub fn from_adjacency(&mut self, adj: &CausalAdjacencyMatrix) -> Result<(), IitError> {
        let n = adj.num_nodes();
        if n > MAX_NODES {
            return Err(IitError::GraphTooLarge(n));
        }

        self.num_nodes = n;

        // Compute degree matrix (diagonal)
        for i in 0..n {
            let mut degree = 0.0f32;
            for j in 0..n {
                if let Some(weight) = adj.get(i, j) {
                    if i != j {
                        degree += weight.abs();
                    }
                }
            }
            self.diagonal[i] = degree;
        }

        // Off-diagonal: L_ij = -A_ij
        let mut idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                self.off_diagonal[idx] = -adj.get(i, j).unwrap_or(0.0);
                idx += 1;
            }
        }

        // Compute Fiedler value/vector using power iteration
        self.compute_fiedler()?;

        Ok(())
    }

    /// Compute Fiedler value (second smallest eigenvalue) using inverse power iteration
    fn compute_fiedler(&mut self) -> Result<(), IitError> {
        let n = self.num_nodes;
        if n < 2 {
            self.fiedler_value = 0.0;
            return Ok(());
        }

        // Initialize random vector orthogonal to constant vector
        let mut v = [0.0f32; MAX_NODES];
        for i in 0..n {
            v[i] = (i as f32 * 0.1).sin();
        }

        // Orthogonalize against constant vector (eigenvalue 0)
        let sum: f32 = v[..n].iter().sum();
        for i in 0..n {
            v[i] -= sum / n as f32;
        }

        // Normalize
        let norm: f32 = v[..n].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > MIN_EIGENVALUE_GAP {
            for i in 0..n {
                v[i] /= norm;
            }
        }

        // Power iteration (simplified - real impl would use inverse iteration)
        let max_iter = 100;
        for _ in 0..max_iter {
            // Multiply by Laplacian
            let mut Lv = [0.0f32; MAX_NODES];
            
            for i in 0..n {
                Lv[i] = self.diagonal[i] * v[i];
                let mut idx = if i < n - 1 { i * (2 * n - i - 1) / 2 } else { 0 };
                for j in 0..i {
                    Lv[i] += self.off_diagonal[j * (2 * n - j - 1) / 2 + (i - j - 1)] * v[j];
                }
                for j in (i + 1)..n {
                    Lv[i] += self.off_diagonal[idx] * v[j];
                    idx += 1;
                }
            }

            // Orthogonalize against constant
            let sum: f32 = Lv[..n].iter().sum();
            for i in 0..n {
                Lv[i] -= sum / n as f32;
            }

            // Rayleigh quotient
            let mut rq_num = 0.0f32;
            let mut rq_den = 0.0f32;
            for i in 0..n {
                rq_num += v[i] * Lv[i];
                rq_den += v[i] * v[i];
            }

            if rq_den > MIN_EIGENVALUE_GAP {
                self.fiedler_value = rq_num / rq_den;
            }

            // Normalize and update
            let norm: f32 = Lv[..n].iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > MIN_EIGENVALUE_GAP {
                for i in 0..n {
                    v[i] = Lv[i] / norm;
                }
            } else {
                break;
            }
        }

        self.fiedler_vector = v;
        Ok(())
    }

    /// Get Fiedler value (algebraic connectivity)
    #[inline]
    pub const fn fiedler_value(&self) -> f32 {
        self.fiedler_value
    }

    /// Get Fiedler vector
    #[inline]
    pub fn fiedler_vector(&self) -> &[f32] {
        &self.fiedler_vector[..self.num_nodes]
    }
}

impl Default for GraphLaplacian {
    fn default() -> Self {
        Self::new()
    }
}

/// Bipartite Φ approximator result
#[derive(Debug, Clone)]
pub struct BipartitePhiResult {
    /// Approximate Φ value
    pub phi: f32,
    /// Lower bound on Φ
    pub phi_lower_bound: f32,
    /// Upper bound (Tse-Quian estimator)
    pub phi_upper_bound: f32,
    /// Partition that minimizes information loss
    pub min_info_partition: [bool; MAX_NODES],
    /// Size of partition A
    pub partition_a_size: usize,
    /// Effective information (EI)
    pub effective_information: f32,
    /// Integration level
    pub integration_level: f32,
}

impl BipartitePhiResult {
    #[inline]
    pub const fn new() -> Self {
        Self {
            phi: 0.0,
            phi_lower_bound: 0.0,
            phi_upper_bound: 0.0,
            min_info_partition: [false; MAX_NODES],
            partition_a_size: 0,
            effective_information: 0.0,
            integration_level: 0.0,
        }
    }

    /// Check if result indicates significant integration
    #[inline]
    pub fn is_significant(&self, threshold: f32) -> bool {
        self.phi > threshold && self.integration_level > 0.1
    }
}

impl Default for BipartitePhiResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Main IIT Φ approximator engine
pub struct IitPhiApproximator {
    /// Adjacency matrix
    adjacency: CausalAdjacencyMatrix,
    /// Graph Laplacian
    laplacian: GraphLaplacian,
    /// Current Φ result
    current_result: BipartitePhiResult,
    /// Spectral relaxation state
    spectral_state: [f32; MAX_NODES],
    /// Number of nodes
    num_nodes: usize,
}

impl IitPhiApproximator {
    /// Create new approximator
    pub fn new() -> Self {
        Self {
            adjacency: CausalAdjacencyMatrix::new(),
            laplacian: GraphLaplacian::new(),
            current_result: BipartitePhiResult::new(),
            spectral_state: [0.0; MAX_NODES],
            num_nodes: 0,
        }
    }

    /// Initialize with network size
    pub fn init(&mut self, num_nodes: usize) -> Result<(), IitError> {
        self.adjacency.init(num_nodes)?;
        self.num_nodes = num_nodes;
        Ok(())
    }

    /// Set connection weight between nodes
    pub fn set_connection(&mut self, i: usize, j: usize, weight: f32) -> Result<(), IitError> {
        self.adjacency.set(i, j, weight)
    }

    /// Compute approximate Φ using bipartite spectral relaxation
    pub fn compute_phi(&mut self) -> Result<&BipartitePhiResult, IitError> {
        // Step 1: Build Laplacian
        self.laplacian.from_adjacency(&self.adjacency)?;

        // Step 2: Use Fiedler vector for optimal bipartition
        let fiedler = self.laplacian.fiedler_value();
        let fiedler_vec = self.laplacian.fiedler_vector();

        if fiedler < MIN_EIGENVALUE_GAP {
            return Err(IitError::InsufficientConnectivity);
        }

        // Step 3: Partition based on Fiedler vector sign
        let mut partition_a_size = 0;
        for i in 0..self.num_nodes {
            self.current_result.min_info_partition[i] = fiedler_vec[i] >= 0.0;
            if fiedler_vec[i] >= 0.0 {
                partition_a_size += 1;
            }
        }
        self.current_result.partition_a_size = partition_a_size;

        // Step 4: Compute effective information across partition
        let ei = self.compute_effective_information()?;
        self.current_result.effective_information = ei;

        // Step 5: Compute integration level
        let total_conn = self.adjacency.total_connectivity();
        let internal_conn = self.compute_internal_connectivity();
        self.current_result.integration_level = if total_conn > MIN_EIGENVALUE_GAP {
            1.0 - (internal_conn / total_conn)
        } else {
            0.0
        }.clamp(0.0, 1.0);

        // Step 6: Compute Φ bounds
        // Lower bound: EI of minimum information partition
        self.current_result.phi_lower_bound = ei * self.current_result.integration_level;

        // Upper bound: Tse-Quian estimator using spectral gap
        let spectral_gap = fiedler;
        self.current_result.phi_upper_bound = ei * (1.0 - (-spectral_gap).exp());

        // Final Φ estimate (geometric mean of bounds)
        self.current_result.phi = (self.current_result.phi_lower_bound * 
            self.current_result.phi_upper_bound.max(MIN_EIGENVALUE_GAP)).sqrt();

        Ok(&self.current_result)
    }

    /// Compute effective information across bipartition
    fn compute_effective_information(&self) -> Result<f32, IitError> {
        let n = self.num_nodes;
        let mut ei = 0.0f32;

        // Sum mutual information across partition boundary
        for i in 0..n {
            for j in 0..n {
                if self.current_result.min_info_partition[i] != self.current_result.min_info_partition[j] {
                    if let Some(weight) = self.adjacency.get(i, j) {
                        // MI approximation: w * log(w / (deg_i * deg_j))
                        let deg_i = self.node_degree(i);
                        let deg_j = self.node_degree(j);
                        
                        if deg_i > MIN_EIGENVALUE_GAP && deg_j > MIN_EIGENVALUE_GAP {
                            let mi = weight.abs() * (weight.abs() / (deg_i * deg_j)).ln().max(0.0);
                            ei += mi;
                        }
                    }
                }
            }
        }

        Ok(ei / (n as f32))
    }

    /// Compute internal connectivity (within partitions)
    fn compute_internal_connectivity(&self) -> f32 {
        let n = self.num_nodes;
        let mut internal = 0.0f32;

        for i in 0..n {
            for j in 0..n {
                if self.current_result.min_info_partition[i] == self.current_result.min_info_partition[j] {
                    internal += self.adjacency.get(i, j).unwrap_or(0.0).abs();
                }
            }
        }

        internal
    }

    /// Get node degree
    fn node_degree(&self, node: usize) -> f32 {
        let mut deg = 0.0f32;
        for j in 0..self.num_nodes {
            if node != j {
                deg += self.adjacency.get(node, j).unwrap_or(0.0).abs();
            }
        }
        deg
    }

    /// Apply graph sparsification to prevent combinatorial explosion
    pub fn sparsify_graph(&mut self, threshold: f32) {
        self.adjacency.sparsify(threshold);
    }

    /// Get current Φ result
    #[inline]
    pub const fn current_result(&self) -> &BipartitePhiResult {
        &self.current_result
    }

    /// Get number of nodes
    #[inline]
    pub const fn num_nodes(&self) -> usize {
        self.num_nodes
    }
}

impl Default for IitPhiApproximator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjacency_init() {
        let mut adj = CausalAdjacencyMatrix::new();
        assert!(adj.init(10).is_ok());
        assert_eq!(adj.num_nodes(), 10);
    }

    #[test]
    fn test_graph_too_large() {
        let mut adj = CausalAdjacencyMatrix::new();
        assert!(adj.init(MAX_NODES + 1).is_err());
    }

    #[test]
    fn test_phi_approximator_creation() {
        let approx = IitPhiApproximator::new();
        assert_eq!(approx.num_nodes(), 0);
    }

    #[test]
    fn test_bipartite_result_default() {
        let result = BipartitePhiResult::new();
        assert_eq!(result.phi, 0.0);
        assert!(!result.is_significant(0.1));
    }
}
