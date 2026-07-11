//! MERA Renormalization Engine
//! 
//! Implements Multi-scale Entanglement Renormalization Ansatz (MERA)
//! for streaming tape data with strict depth limits to prevent RAM exhaustion.

use crate::tensors::{Disentangler, Isometry, TensorConfig};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors specific to MERA renormalization
#[derive(Error, Debug, Clone, PartialEq)]
pub enum MeraError {
    #[error("Maximum depth exceeded: {0} > {1}")]
    MaxDepthExceeded(usize, usize),
    #[error("Invalid tensor dimension: {0}")]
    InvalidTensorDimension(usize),
    #[error("Renormalization failed: {0}")]
    RenormalizationFailed(String),
    #[error("Memory limit exceeded")]
    MemoryLimitExceeded,
    #[error("Input size not power of 2: {0}")]
    InvalidInputSize(usize),
}

/// Configuration for MERA network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeraConfig {
    /// Maximum renormalization depth (strictly enforced)
    pub max_depth: usize,
    /// Bond dimension (controls accuracy vs memory)
    pub bond_dimension: usize,
    /// Whether to use disentanglers
    pub use_disentanglers: bool,
    /// Memory limit in MB
    pub memory_limit_mb: usize,
}

impl Default for MeraConfig {
    fn default() -> Self {
        Self {
            max_depth: 8, // Strict limit based on HFT time horizon
            bond_dimension: 16,
            use_disentanglers: true,
            memory_limit_mb: 512,
        }
    }
}

/// Single layer of MERA network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeraLayer {
    /// Optional disentangler tensors
    pub disentanglers: Option<Vec<Disentangler>>,
    /// Isometry tensors for coarse-graining
    pub isometries: Vec<Isometry>,
    /// Scale factor for this layer
    pub scale_factor: f64,
}

/// Complete MERA network state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeraNetwork {
    /// Layers from fine to coarse
    pub layers: Vec<MeraLayer>,
    /// Current depth
    pub current_depth: usize,
    /// Maximum allowed depth
    pub max_depth: usize,
    /// Original input size
    pub input_size: usize,
    /// Coarse-grained output at top level
    pub top_state: Option<DVector<f64>>,
}

/// MERA renormalization engine for tape data
pub struct MeraRenormalizer {
    /// Network configuration
    config: MeraConfig,
    /// Tensor configuration
    tensor_config: TensorConfig,
}

impl MeraRenormalizer {
    /// Create a new MERA renormalizer
    pub fn new(config: MeraConfig, tensor_config: TensorConfig) -> Result<Self, MeraError> {
        if config.max_depth == 0 {
            return Err(MeraError::InvalidTensorDimension(0));
        }
        if config.bond_dimension < 2 {
            return Err(MeraError::InvalidTensorDimension(config.bond_dimension));
        }

        Ok(Self {
            config,
            tensor_config,
        })
    }

    /// Build MERA network structure
    pub fn build_network(&self, input_size: usize) -> Result<MeraNetwork, MeraError> {
        // Validate input size (should be power of 2 for binary MERA)
        if !input_size.is_power_of_two() {
            return Err(MeraError::InvalidInputSize(input_size));
        }

        // Check memory constraints
        let estimated_memory = self.estimate_memory_usage(input_size);
        if estimated_memory > self.config.memory_limit_mb * 1024 * 1024 {
            return Err(MeraError::MemoryLimitExceeded);
        }

        let mut layers = Vec::new();
        let mut current_size = input_size;

        for depth in 0..self.config.max_depth {
            if current_size < 2 {
                break;
            }

            let num_tensors = current_size / 2;

            // Create disentanglers if enabled
            let disentanglers = if self.config.use_disentanglers && depth < self.config.max_depth - 1 {
                let mut d_list = Vec::with_capacity(num_tensors);
                for _ in 0..num_tensors {
                    d_list.push(Disentangler::new(self.config.bond_dimension)?);
                }
                Some(d_list)
            } else {
                None
            };

            // Create isometries
            let mut isometries = Vec::with_capacity(num_tensors);
            for _ in 0..num_tensors {
                isometries.push(Isometry::new(self.config.bond_dimension)?);
            }

            let layer = MeraLayer {
                disentanglers,
                isometries,
                scale_factor: 2.0_f64.powi(depth as i32),
            };

            layers.push(layer);
            current_size /= 2;
        }

        Ok(MeraNetwork {
            layers,
            current_depth: 0,
            max_depth: self.config.max_depth,
            input_size,
            top_state: None,
        })
    }

    /// Apply MERA renormalization to input data
    pub fn renormalize(&self, input: &[f64]) -> Result<DVector<f64>, MeraError> {
        let input_size = input.len();
        
        if !input_size.is_power_of_two() {
            return Err(MeraError::InvalidInputSize(input_size));
        }

        let mut network = self.build_network(input_size)?;
        let mut current_state = DVector::from_vec(input.to_vec());

        // Apply each layer
        for (depth, layer) in network.layers.iter().enumerate() {
            if depth >= self.config.max_depth {
                return Err(MeraError::MaxDepthExceeded(depth, self.config.max_depth));
            }

            // Apply disentanglers first (if present)
            if let Some(disentanglers) = &layer.disentanglers {
                current_state = self.apply_disentanglers(current_state, disentanglers)?;
            }

            // Apply isometries for coarse-graining
            current_state = self.apply_isometries(current_state, &layer.isometries)?;

            network.current_depth = depth + 1;
        }

        network.top_state = Some(current_state.clone());

        Ok(current_state)
    }

    /// Apply disentangling unitaries
    fn apply_disentanglers(
        &self,
        state: DVector<f64>,
        disentanglers: &[Disentangler],
    ) -> Result<DVector<f64>, MeraError> {
        let mut result = state.clone();

        for (i, d) in disentanglers.iter().enumerate() {
            let idx1 = i * 2;
            let idx2 = i * 2 + 1;

            if idx2 >= result.len() {
                break;
            }

            let pair = DVector::from_vec(vec![result[idx1], result[idx2]]);
            let transformed = d.apply(&pair)?;

            result[idx1] = transformed[0];
            result[idx2] = transformed[1];
        }

        Ok(result)
    }

    /// Apply isometries for coarse-graining
    fn apply_isometries(
        &self,
        state: DVector<f64>,
        isometries: &[Isometry],
    ) -> Result<DVector<f64>, MeraError> {
        let new_size = (state.len() + 1) / 2;
        let mut result = DVector::zeros(new_size);

        for (i, iso) in isometries.iter().enumerate() {
            let idx1 = i * 2;
            let idx2 = i * 2 + 1;

            if idx1 >= state.len() {
                break;
            }

            let input = if idx2 < state.len() {
                DVector::from_vec(vec![state[idx1], state[idx2]])
            } else {
                DVector::from_vec(vec![state[idx1], 0.0])
            };

            result[i] = iso.apply(&input)?[0];
        }

        Ok(result)
    }

    /// Estimate memory usage in bytes
    fn estimate_memory_usage(&self, input_size: usize) -> usize {
        let num_layers = (input_size as f64).log2() as usize;
        let bond = self.config.bond_dimension;
        
        // Each disentangler: O(bond²) parameters
        // Each isometry: O(bond²) parameters
        let params_per_layer = input_size * bond * bond;
        let total_params: usize = (0..num_layers.min(self.config.max_depth))
            .map(|l| params_per_layer / (1 << l))
            .sum();

        total_params * 8 // f64 is 8 bytes
    }

    /// Compute entanglement entropy at each scale
    pub fn compute_scale_entropies(&self, input: &[f64]) -> Result<Vec<f64>, MeraError> {
        let mut entropies = Vec::new();
        let mut current_state = DVector::from_vec(input.to_vec());

        while current_state.len() > 1 {
            // Compute von Neumann entropy for current scale
            let entropy = self.compute_von_neumann_entropy(&current_state);
            entropies.push(entropy);

            // Coarse-grain
            let new_size = (current_state.len() + 1) / 2;
            let mut next_state = DVector::zeros(new_size);
            
            for i in 0..new_size {
                let idx1 = i * 2;
                let idx2 = i * 2 + 1;
                
                if idx1 < current_state.len() {
                    if idx2 < current_state.len() {
                        next_state[i] = (current_state[idx1] + current_state[idx2]) / 2.0;
                    } else {
                        next_state[i] = current_state[idx1];
                    }
                }
            }

            current_state = next_state;
        }

        Ok(entropies)
    }

    /// Compute von Neumann entropy S = -Tr(ρ ln ρ)
    fn compute_von_neumann_entropy(&self, state: &DVector<f64>) -> f64 {
        let norm_sq = state.dot(state);
        if norm_sq < 1e-15 {
            return 0.0;
        }

        // Normalize and compute Shannon entropy (classical approximation)
        let mut entropy = 0.0;
        for &val in state.iter() {
            let p = (val * val) / norm_sq;
            if p > 1e-15 {
                entropy -= p * p.ln();
            }
        }

        entropy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mera_creation() {
        let config = MeraConfig::default();
        let tensor_config = TensorConfig::default();
        let mera = MeraRenormalizer::new(config, tensor_config);
        assert!(mera.is_ok());
    }

    #[test]
    fn test_renormalization() {
        let config = MeraConfig::default();
        let tensor_config = TensorConfig::default();
        let mera = MeraRenormalizer::new(config, tensor_config).unwrap();

        // Power of 2 input
        let input: Vec<f64> = vec![1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125];
        let result = mera.renormalize(&input);
        assert!(result.is_ok());
        assert!(result.unwrap().len() < input.len());
    }

    #[test]
    fn test_invalid_input_size() {
        let config = MeraConfig::default();
        let tensor_config = TensorConfig::default();
        let mera = MeraRenormalizer::new(config, tensor_config).unwrap();

        let input: Vec<f64> = vec![1.0, 2.0, 3.0]; // Not power of 2
        assert!(mera.renormalize(&input).is_err());
    }
}
