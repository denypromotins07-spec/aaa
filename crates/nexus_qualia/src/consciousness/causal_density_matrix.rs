//! Causal Density Matrix for measuring information integration in neural/AI networks.
//! Zero-allocation implementation using fixed-size arrays.

use super::iit_phi_approximator::{CausalAdjacencyMatrix, GraphLaplacian, IitError, MAX_NODES};

/// Maximum time lags for temporal causality
pub const MAX_TIME_LAGS: usize = 16;

/// Causal density measurement result
#[derive(Debug, Clone)]
pub struct CausalDensityResult {
    pub global_causal_density: f32,
    pub local_causal_density: [f32; MAX_NODES],
    pub integrated_information: f32,
    pub differentiation: f32,
    pub timestamp_ns: u64,
}

impl CausalDensityResult {
    pub const fn new() -> Self {
        Self {
            global_causal_density: 0.0,
            local_causal_density: [0.0; MAX_NODES],
            integrated_information: 0.0,
            differentiation: 0.0,
            timestamp_ns: 0,
        }
    }
}

impl Default for CausalDensityResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Main causal density matrix engine
pub struct CausalDensityMatrix {
    adjacency: CausalAdjacencyMatrix,
    lag_matrices: [CausalAdjacencyMatrix; MAX_TIME_LAGS],
    result: CausalDensityResult,
    num_nodes: usize,
    num_lags: usize,
}

impl CausalDensityMatrix {
    pub fn new() -> Self {
        Self {
            adjacency: CausalAdjacencyMatrix::new(),
            lag_matrices: [CausalAdjacencyMatrix::new(); MAX_TIME_LAGS],
            result: CausalDensityResult::new(),
            num_nodes: 0,
            num_lags: 0,
        }
    }

    pub fn init(&mut self, num_nodes: usize, num_lags: usize) -> Result<(), IitError> {
        if num_lags > MAX_TIME_LAGS {
            return Err(IitError::DimensionMismatch(MAX_TIME_LAGS, num_lags));
        }
        self.adjacency.init(num_nodes)?;
        for i in 0..num_lags {
            self.lag_matrices[i].init(num_nodes)?;
        }
        self.num_nodes = num_nodes;
        self.num_lags = num_lags;
        Ok(())
    }

    pub fn set_contemporaneous_connection(&mut self, i: usize, j: usize, weight: f32) -> Result<(), IitError> {
        self.adjacency.set(i, j, weight)
    }

    pub fn set_lagged_connection(&mut self, lag: usize, i: usize, j: usize, weight: f32) -> Result<(), IitError> {
        if lag >= self.num_lags {
            return Err(IitError::DimensionMismatch(self.num_lags, lag));
        }
        self.lag_matrices[lag].set(i, j, weight)
    }

    pub fn compute_causal_density(&mut self) -> Result<&CausalDensityResult, IitError> {
        let mut total_cd = 0.0f32;
        
        for node in 0..self.num_nodes {
            let cd = self.compute_node_causal_density(node)?;
            self.result.local_causal_density[node] = cd;
            total_cd += cd;
        }

        self.result.global_causal_density = total_cd / self.num_nodes as f32;
        
        // Compute integrated information and differentiation
        let laplacian = GraphLaplacian::new();
        self.result.integrated_information = self.result.global_causal_density * 0.5;
        self.result.differentiation = self.compute_differentiation()?;

        Ok(&self.result)
    }

    fn compute_node_causal_density(&self, node: usize) -> Result<f32, IitError> {
        let mut cd = 0.0f32;
        
        // Sum incoming causal influences across all lags
        for lag in 0..self.num_lags {
            for source in 0..self.num_nodes {
                if source != node {
                    if let Some(weight) = self.lag_matrices[lag].get(source, node) {
                        cd += weight.abs() / (lag as f32 + 1.0);
                    }
                }
            }
        }

        Ok(cd)
    }

    fn compute_differentiation(&self) -> Result<f32, IitError> {
        let mean = self.result.global_causal_density;
        let mut variance = 0.0f32;

        for node in 0..self.num_nodes {
            let diff = self.result.local_causal_density[node] - mean;
            variance += diff * diff;
        }

        Ok((variance / self.num_nodes as f32).sqrt())
    }

    pub fn set_timestamp(&mut self, timestamp_ns: u64) {
        self.result.timestamp_ns = timestamp_ns;
    }
}

impl Default for CausalDensityMatrix {
    fn default() -> Self {
        Self::new()
    }
}
