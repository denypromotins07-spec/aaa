//! Spectral Relaxation Bound for IIT Φ upper bound estimation (Tse-Quian estimator).
//! Zero-allocation implementation.

use super::iit_phi_approximator::{CausalAdjacencyMatrix, GraphLaplacian, IitError, MAX_NODES};

/// Spectral relaxation bound result
#[derive(Debug, Clone)]
pub struct SpectralBoundResult {
    pub upper_bound: f32,
    pub lower_bound: f32,
    pub spectral_gap: f32,
    pub algebraic_connectivity: f32,
    pub is_tight: bool,
}

impl SpectralBoundResult {
    pub const fn new() -> Self {
        Self {
            upper_bound: 0.0,
            lower_bound: 0.0,
            spectral_gap: 0.0,
            algebraic_connectivity: 0.0,
            is_tight: false,
        }
    }
}

impl Default for SpectralBoundResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Main spectral relaxation bound engine
pub struct SpectralRelaxationBound {
    adjacency: CausalAdjacencyMatrix,
    laplacian: GraphLaplacian,
    result: SpectralBoundResult,
    num_nodes: usize,
}

impl SpectralRelaxationBound {
    pub fn new() -> Self {
        Self {
            adjacency: CausalAdjacencyMatrix::new(),
            laplacian: GraphLaplacian::new(),
            result: SpectralBoundResult::new(),
            num_nodes: 0,
        }
    }

    pub fn init(&mut self, num_nodes: usize) -> Result<(), IitError> {
        self.adjacency.init(num_nodes)?;
        self.num_nodes = num_nodes;
        Ok(())
    }

    pub fn set_connection(&mut self, i: usize, j: usize, weight: f32) -> Result<(), IitError> {
        self.adjacency.set(i, j, weight)
    }

    pub fn compute_bounds(&mut self) -> Result<&SpectralBoundResult, IitError> {
        // Build Laplacian and compute eigenvalues
        self.laplacian.from_adjacency(&self.adjacency)?;

        let fiedler = self.laplacian.fiedler_value();
        self.result.spectral_gap = fiedler;
        self.result.algebraic_connectivity = fiedler;

        // Tse-Quian upper bound using spectral gap
        // Φ ≤ n * (1 - exp(-λ₂)) where λ₂ is Fiedler value
        self.result.upper_bound = self.num_nodes as f32 * (1.0 - (-fiedler).exp());

        // Lower bound from Cheeger inequality
        // Φ ≥ λ₂ / 2
        self.result.lower_bound = fiedler / 2.0;

        // Check if bounds are tight
        let gap = self.result.upper_bound - self.result.lower_bound;
        self.result.is_tight = gap < self.result.upper_bound * 0.2;

        Ok(&self.result)
    }

    pub fn get_spectral_gap(&self) -> f32 {
        self.result.spectral_gap
    }
}

impl Default for SpectralRelaxationBound {
    fn default() -> Self {
        Self::new()
    }
}
