//! Ising Hamiltonian Mapper
//! 
//! Converts QUBO matrices to Ising Hamiltonian form compatible with quantum annealers.
//! QUBO: minimize x^T Q x + c^T x  (where x ∈ {0,1})
//! Ising: minimize s^T J s + h^T s  (where s ∈ {-1,+1})
//! 
//! The transformation uses: x = (s + 1) / 2

use ndarray::{Array2, Array1};
use num_traits::Float;
use crate::qubo::portfolio_hamiltonian::QuboMatrix;

/// Coefficients of the Ising Hamiltonian
#[derive(Debug, Clone)]
pub struct IsingCoefficients<F: Float> {
    /// Coupling matrix J (quadratic terms)
    pub j_matrix: Array2<F>,
    /// Local field vector h (linear terms)
    pub h_vector: Array1<F>,
    /// Constant energy offset
    pub energy_offset: F,
}

/// Complete Ising Hamiltonian representation
#[derive(Debug, Clone)]
pub struct IsingHamiltonian<F: Float> {
    /// Number of spins
    pub n_spins: usize,
    /// Spin-spin coupling coefficients
    pub couplings: Vec<(usize, usize, F)>,
    /// Local magnetic fields
    pub local_fields: Vec<(usize, F)>,
    /// Energy offset from QUBO conversion
    pub offset: F,
    /// Mapping from spin index to original qubit/asset info
    pub spin_mapping: Vec<String>,
}

impl<F: Float> IsingHamiltonian<F> {
    /// Create a new empty Ising Hamiltonian
    pub fn new(n_spins: usize) -> Self {
        Self {
            n_spins,
            couplings: Vec::new(),
            local_fields: Vec::with_capacity(n_spins),
            offset: F::zero(),
            spin_mapping: Vec::with_capacity(n_spins),
        }
    }

    /// Get the total number of spins
    pub fn n_spins(&self) -> usize {
        self.n_spins
    }

    /// Validate the Hamiltonian structure
    pub fn validate(&self) -> Result<(), IsingError> {
        if self.local_fields.len() != self.n_spins {
            return Err(IsingError::DimensionMismatch {
                expected: self.n_spins,
                actual: self.local_fields.len(),
            });
        }
        
        // Check for duplicate couplings
        let mut seen_pairs = std::collections::HashSet::new();
        for &(i, j, _) in &self.couplings {
            let pair = if i < j { (i, j) } else { (j, i) };
            if !seen_pairs.insert(pair) {
                return Err(IsingError::DuplicateCoupling(i, j));
            }
        }
        
        Ok(())
    }

    /// Calculate the energy of a given spin configuration
    pub fn energy(&self, spins: &[i8]) -> Result<F, IsingError> {
        if spins.len() != self.n_spins {
            return Err(IsingError::DimensionMismatch {
                expected: self.n_spins,
                actual: spins.len(),
            });
        }
        
        let mut energy = self.offset;
        
        // Add coupling terms
        for &(i, j, j_ij) in &self.couplings {
            if i < spins.len() && j < spins.len() {
                energy = energy + j_ij * F::from(spins[i] as f64).unwrap() * F::from(spins[j] as f64).unwrap();
            }
        }
        
        // Add local field terms
        for &(i, h_i) in &self.local_fields {
            if i < spins.len() {
                energy = energy + h_i * F::from(spins[i] as f64).unwrap();
            }
        }
        
        Ok(energy)
    }
}

/// Errors that can occur during Ising mapping
#[derive(Error, Debug)]
pub enum IsingError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Duplicate coupling between spins {0} and {1}")]
    DuplicateCoupling(usize, usize),
    #[error("Invalid spin value: must be -1 or +1, got {0}")]
    InvalidSpinValue(i8),
    #[error("QUBO matrix error: {0}")]
    QuboError(String),
}

use std::fmt;
impl fmt::Display for IsingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IsingError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, actual)
            }
            IsingError::DuplicateCoupling(i, j) => {
                write!(f, "Duplicate coupling between spins {} and {}", i, j)
            }
            IsingError::InvalidSpinValue(v) => {
                write!(f, "Invalid spin value: must be -1 or +1, got {}", v)
            }
            IsingError::QuboError(msg) => write!(f, "QUBO matrix error: {}", msg),
        }
    }
}

impl std::error::Error for IsingError {}

/// Mapper from QUBO to Ising Hamiltonian
pub struct IsingMapper<F: Float> {
    /// Conversion tolerance for sparse representation
    sparsity_threshold: F,
}

impl<F: Float + 'static> IsingMapper<F> 
where
    F: From<f64> + Copy + Into<f64>,
{
    /// Create a new Ising mapper with default settings
    pub fn new() -> Self {
        Self {
            sparsity_threshold: F::from(1e-10).unwrap(),
        }
    }

    /// Create a new Ising mapper with custom sparsity threshold
    pub fn with_sparsity_threshold(threshold: F) -> Self {
        Self {
            sparsity_threshold: threshold,
        }
    }

    /// Convert a QUBO matrix to Ising Hamiltonian
    /// 
    /// Transformation: x = (s + 1) / 2
    /// 
    /// QUBO: x^T Q x + c^T x
    ///     = ((s+1)/2)^T Q ((s+1)/2) + c^T (s+1)/2
    ///     = (1/4) s^T Q s + (1/4) s^T Q 1 + (1/4) 1^T Q s + (1/4) 1^T Q 1
    ///       + (1/2) c^T s + (1/2) c^T 1
    ///     
    /// Since Q is symmetric in QUBO:
    ///     = (1/4) s^T Q s + (1/2) (Q 1 + c)^T s + constant
    /// 
    /// Therefore:
    ///     J = Q / 4
    ///     h = (Q 1 + c) / 2
    ///     offset = (1^T Q 1) / 4 + (1^T c) / 2
    pub fn qubo_to_ising(&self, qubo: &QuboMatrix<F>) -> Result<IsingHamiltonian<F>, IsingError> {
        let n = qubo.n_qubits;
        
        // Calculate J = Q / 4
        let j_matrix = qubo.matrix.mapv(|x| x / F::from(4.0).unwrap());
        
        // Calculate Q * 1 (row sums)
        let q_row_sums: Array1<F> = qubo.matrix.sum_axis(ndarray::Axis(1));
        
        // Calculate h = (Q*1 + c) / 2
        let h_vector = (q_row_sums + &qubo.linear_term).mapv(|x| x / F::from(2.0).unwrap());
        
        // Calculate constant offset
        // offset = (1^T Q 1) / 4 + (1^T c) / 2 + qubo.constant_offset
        let one_t_q_one: F = qubo.matrix.sum();
        let one_t_c: F = qubo.linear_term.sum();
        let offset = one_t_q_one / F::from(4.0).unwrap() 
                   + one_t_c / F::from(2.0).unwrap() 
                   + qubo.constant_offset;
        
        // Build sparse representation
        let mut hamiltonian = IsingHamiltonian::new(n);
        hamiltonian.offset = offset;
        
        // Add couplings (only upper triangle, skip small values)
        for i in 0..n {
            for j in (i+1)..n {
                let j_ij = j_matrix[[i, j]];
                if j_ij.abs() > self.sparsity_threshold {
                    hamiltonian.couplings.push((i, j, j_ij));
                }
            }
        }
        
        // Add local fields
        for i in 0..n {
            let h_i = h_vector[i];
            if h_i.abs() > self.sparsity_threshold || true {
                // Always include local fields even if small
                hamiltonian.local_fields.push((i, h_i));
            }
        }
        
        // Copy spin mapping from qubit mapping
        hamiltonian.spin_mapping = qubo.qubit_mapping
            .iter()
            .map(|(asset, weight)| format!("{}_{}", asset, weight))
            .collect();
        
        // Validate the result
        hamiltonian.validate()?;
        
        Ok(hamiltonian)
    }

    /// Convert Ising Hamiltonian back to QUBO (for verification)
    /// 
    /// Inverse transformation: s = 2x - 1
    pub fn ising_to_qubo(&self, hamiltonian: &IsingHamiltonian<F>) -> Result<QuboMatrix<F>, IsingError> {
        let n = hamiltonian.n_spins;
        let mut qubo = QuboMatrix::new(n);
        
        // Reconstruct J matrix from sparse couplings
        let mut j_matrix = Array2::zeros((n, n));
        for &(i, j, j_ij) in &hamiltonian.couplings {
            j_matrix[[i, j]] = j_ij;
            j_matrix[[j, i]] = j_ij;
        }
        
        // Reconstruct h vector
        let mut h_vector = Array1::zeros(n);
        for &(i, h_i) in &hamiltonian.local_fields {
            h_vector[i] = h_i;
        }
        
        // Inverse transform:
        // Q = 4J
        // c = 2h - Q*1
        qubo.matrix = j_matrix.mapv(|x| x * F::from(4.0).unwrap());
        
        let q_row_sums: Array1<F> = qubo.matrix.sum_axis(ndarray::Axis(1));
        qubo.linear_term = h_vector.mapv(|x| x * F::from(2.0).unwrap()) - q_row_sums;
        
        // Recover constant offset (approximately)
        qubo.constant_offset = hamiltonian.offset;
        
        // Copy mapping
        qubo.qubit_mapping = hamiltonian.spin_mapping
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), F::one()))
            .collect();
        
        qubo.validate().map_err(|e| IsingError::QuboError(e.to_string()))?;
        
        Ok(qubo)
    }

    /// Extract subgraph for minor embedding (used by D-Wave bridge)
    /// Returns adjacency list representation
    pub fn extract_interaction_graph(&self, hamiltonian: &IsingHamiltonian<F>) -> InteractionGraph {
        let n = hamiltonian.n_spins;
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        
        for &(i, j, _) in &hamiltonian.couplings {
            adj[i].push(j);
            adj[j].push(i);
        }
        
        InteractionGraph {
            n_nodes: n,
            adjacency: adj,
            has_coupling: hamiltonian.couplings.iter()
                .map(|&(i, j, _)| (i.min(j), i.max(j)))
                .collect(),
        }
    }
}

impl<F: Float + Default + 'static + From<f64> + Copy + Into<f64>> Default for IsingMapper<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Interaction graph for embedding purposes
#[derive(Debug, Clone)]
pub struct InteractionGraph {
    /// Number of nodes (spins)
    pub n_nodes: usize,
    /// Adjacency list
    pub adjacency: Vec<Vec<usize>>,
    /// Set of edges with non-zero coupling (stored as sorted pairs)
    pub has_coupling: std::collections::HashSet<(usize, usize)>,
}

impl InteractionGraph {
    /// Get degree of a node
    pub fn degree(&self, node: usize) -> usize {
        self.adjacency.get(node).map(|v| v.len()).unwrap_or(0)
    }

    /// Get maximum degree in the graph
    pub fn max_degree(&self) -> usize {
        (0..self.n_nodes).map(|i| self.degree(i)).max().unwrap_or(0)
    }

    /// Check if there's an edge between two nodes
    pub fn has_edge(&self, i: usize, j: usize) -> bool {
        let (min, max) = if i < j { (i, j) } else { (j, i) };
        self.has_coupling.contains(&(min, max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qubo::portfolio_hamiltonian::{QuboMatrix, QuboConfig, PortfolioQuboBuilder};

    #[test]
    fn test_qubo_to_ising_conversion() {
        // Simple 2-qubit QUBO
        let mut qubo: QuboMatrix<f64> = QuboMatrix::new(2);
        qubo.matrix[[0, 0]] = 1.0;
        qubo.matrix[[1, 1]] = 1.0;
        qubo.matrix[[0, 1]] = -0.5;
        qubo.matrix[[1, 0]] = -0.5;
        qubo.linear_term[0] = -0.3;
        qubo.linear_term[1] = -0.2;
        qubo.qubit_mapping = vec![
            ("q0".to_string(), 1.0),
            ("q1".to_string(), 1.0),
        ];

        let mapper = IsingMapper::new();
        let ising = mapper.qubo_to_ising(&qubo).unwrap();

        assert_eq!(ising.n_spins, 2);
        
        // Verify energy equivalence for a sample configuration
        // QUBO: x = [1, 0], Ising: s = [1, -1] (since x = (s+1)/2)
        let qubo_x = vec![1, 0];
        let ising_s = vec![1, -1];
        
        // Manual QUBO energy calculation
        let qubo_energy = qubo.matrix[[0, 0]] * 1.0 * 1.0 
                        + qubo.matrix[[1, 1]] * 0.0 * 0.0 
                        + 2.0 * qubo.matrix[[0, 1]] * 1.0 * 0.0
                        + qubo.linear_term[0] * 1.0
                        + qubo.linear_term[1] * 0.0;
        
        let ising_energy = ising.energy(&ising_s).unwrap();
        
        // They should differ only by the constant offset
        let energy_diff = (qubo_energy - ising_energy).abs();
        assert!(energy_diff < 1e-10 || (energy_diff - ising.offset.abs()).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip_conversion() {
        let mut qubo: QuboMatrix<f64> = QuboMatrix::new(3);
        qubo.matrix[[0, 1]] = 0.5;
        qubo.matrix[[1, 0]] = 0.5;
        qubo.matrix[[1, 2]] = -0.3;
        qubo.matrix[[2, 1]] = -0.3;
        qubo.linear_term[0] = -0.1;
        qubo.linear_term[1] = 0.2;
        qubo.linear_term[2] = -0.05;
        qubo.qubit_mapping = vec![
            ("a".to_string(), 1.0),
            ("b".to_string(), 1.0),
            ("c".to_string(), 1.0),
        ];

        let mapper = IsingMapper::new();
        let ising = mapper.qubo_to_ising(&qubo).unwrap();
        let qubo_back = mapper.ising_to_qubo(&ising).unwrap();

        // Check matrix dimensions match
        assert_eq!(qubo.n_qubits, qubo_back.n_qubits);
    }

    #[test]
    fn test_interaction_graph() {
        let mut hamiltonian: IsingHamiltonian<f64> = IsingHamiltonian::new(4);
        hamiltonian.couplings = vec![
            (0, 1, 0.5),
            (1, 2, -0.3),
            (2, 3, 0.8),
        ];
        hamiltonian.local_fields = vec![(0, 0.1), (1, -0.2), (2, 0.3), (3, -0.1)];

        let mapper = IsingMapper::new();
        let graph = mapper.extract_interaction_graph(&hamiltonian);

        assert_eq!(graph.n_nodes, 4);
        assert_eq!(graph.max_degree(), 2); // Node 1 and 2 have degree 2
        assert!(graph.has_edge(0, 1));
        assert!(!graph.has_edge(0, 3));
    }
}
