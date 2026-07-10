//! Minor Embedding Heuristic
//! 
//! Maps logical QUBO/Ising graphs to physical quantum hardware topologies.
//! Quantum annealers have sparse connectivity (Pegasus, Zephyr), so logical qubits
//! must be mapped to chains of physical qubits.

use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;
use crate::qubo::ising_mapper::InteractionGraph;

/// Errors that can occur during embedding
#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Graph too large: {logical_nodes} nodes exceed hardware capacity {hardware_nodes}")]
    GraphTooLarge { logical_nodes: usize, hardware_nodes: usize },
    #[error("Node degree too high: {degree} exceeds hardware connectivity {max_degree}")]
    DegreeExceeded { degree: usize, max_degree: usize },
    #[error("Embedding failed after {attempts} attempts")]
    EmbeddingFailed { attempts: usize },
    #[error("Chain validation failed: chain {0} has disconnected qubits")]
    ChainDisconnected(usize),
    #[error("Invalid hardware topology: {0}")]
    InvalidTopology(String),
}

/// Hardware topology description
#[derive(Debug, Clone)]
pub struct HardwareTopology {
    /// Name of the topology (e.g., "Pegasus_P16", "Zephyr_Z4")
    pub name: String,
    /// Total number of physical qubits
    pub num_qubits: usize,
    /// Adjacency list representation of hardware connectivity
    pub adjacency: Vec<Vec<usize>>,
    /// Maximum degree of any qubit
    pub max_degree: usize,
}

/// Pegasus topology parameters
#[derive(Debug, Clone)]
pub struct PegasusTopology {
    /// Number of cells in each dimension (typically 16 for Advantage)
    pub m: usize,
    /// Total qubits = 8 * m^3
    pub total_qubits: usize,
}

impl PegasusTopology {
    /// Create a Pegasus topology with given size parameter m
    pub fn new(m: usize) -> Self {
        // Pegasus has 8*m^3 qubits
        let total_qubits = 8 * m * m * m;
        Self { m, total_qubits }
    }

    /// Generate the full hardware adjacency list
    pub fn generate_adjacency(&self) -> Vec<Vec<usize>> {
        let m = self.m;
        let n_qubits = self.total_qubits;
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_qubits];

        // Pegasus connectivity is complex - simplified generation here
        // Full implementation would follow D-Wave's Pegasus specification
        
        // Each qubit connects to:
        // 1. Within-cell connections
        // 2. Inter-cell horizontal connections  
        // 3. Inter-cell vertical connections
        // 4. Odd/even parity connections

        for q in 0..n_qubits {
            // Simplified neighbor generation
            // In production, implement full Pegasus specification
            
            // Cell index and position within cell
            let cell_idx = q / 8;
            let pos_in_cell = q % 8;
            
            // Add within-cell neighbors (simplified)
            for other_pos in 0..8 {
                if other_pos != pos_in_cell {
                    let neighbor = cell_idx * 8 + other_pos;
                    if neighbor < n_qubits && !adj[q].contains(&neighbor) {
                        adj[q].push(neighbor);
                    }
                }
            }
            
            // Add inter-cell neighbors based on position
            let row = cell_idx / m;
            let col = cell_idx % m;
            
            // Horizontal connections
            if col + 1 < m {
                let right_cell = cell_idx + 1;
                if right_cell * 8 + pos_in_cell < n_qubits {
                    adj[q].push(right_cell * 8 + pos_in_cell);
                }
            }
            
            // Vertical connections
            if row + 1 < m {
                let down_cell = cell_idx + m;
                if down_cell * 8 + pos_in_cell < n_qubits {
                    adj[q].push(down_cell * 8 + pos_in_cell);
                }
            }
        }

        // Remove duplicates and sort
        for neighbors in &mut adj {
            neighbors.sort();
            neighbors.dedup();
        }

        adj
    }

    /// Get the maximum degree in Pegasus topology
    pub fn max_degree(&self) -> usize {
        // Pegasus has maximum degree 20 for interior qubits
        if self.m >= 4 { 20 } else { 15 }
    }
}

/// Result of minor embedding
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// Mapping from logical qubit to chain of physical qubits
    pub logical_to_physical: HashMap<usize, Vec<usize>>,
    /// Reverse mapping from physical qubit to logical qubit (if any)
    pub physical_to_logical: HashMap<usize, Option<usize>>,
    /// Statistics about the embedding
    pub stats: EmbeddingStats,
}

/// Statistics about an embedding
#[derive(Debug, Clone, Default)]
pub struct EmbeddingStats {
    /// Number of logical qubits embedded
    pub logical_qubits: usize,
    /// Number of physical qubits used
    pub physical_qubits_used: usize,
    /// Average chain length
    pub avg_chain_length: f64,
    /// Maximum chain length
    pub max_chain_length: usize,
    /// Embedding attempt number that succeeded
    pub attempts: usize,
    /// Time taken in milliseconds
    pub time_ms: u64,
}

/// Minor embedding heuristic using various strategies
pub struct MinorEmbedder {
    /// Hardware topology to embed into
    hardware: HardwareTopology,
    /// Maximum number of embedding attempts
    max_attempts: usize,
    /// Random seed for reproducibility
    seed: u64,
}

impl MinorEmbedder {
    /// Create a new embedder for a specific hardware topology
    pub fn new(hardware: HardwareTopology) -> Self {
        Self {
            hardware,
            max_attempts: 100,
            seed: 42,
        }
    }

    /// Create embedder with custom settings
    pub fn with_settings(hardware: HardwareTopology, max_attempts: usize, seed: u64) -> Self {
        Self {
            hardware,
            max_attempts,
            seed,
        }
    }

    /// Create a Pegasus embedder with standard configuration
    pub fn pegasus(m: usize) -> Self {
        let pegasus = PegasusTopology::new(m);
        let adj = pegasus.generate_adjacency();
        let max_degree = pegasus.max_degree();
        
        let hardware = HardwareTopology {
            name: format!("Pegasus_P{}", m),
            num_qubits: pegasus.total_qubits,
            adjacency: adj,
            max_degree,
        };
        
        Self::new(hardware)
    }

    /// Embed a logical interaction graph into hardware
    /// 
    /// Uses a combination of greedy and randomized heuristics to find
    /// a valid minor embedding where each logical qubit maps to a
    /// connected chain of physical qubits.
    pub fn embed(&self, logical_graph: &InteractionGraph) -> Result<EmbeddingResult, EmbeddingError> {
        // Check basic feasibility
        if logical_graph.n_nodes > self.hardware.num_qubits {
            return Err(EmbeddingError::GraphTooLarge {
                logical_nodes: logical_graph.n_nodes,
                hardware_nodes: self.hardware.num_qubits,
            });
        }

        if logical_graph.max_degree() > self.hardware.max_degree {
            return Err(EmbeddingError::DegreeExceeded {
                degree: logical_graph.max_degree(),
                max_degree: self.hardware.max_degree,
            });
        }

        // Try multiple embedding attempts with different random seeds
        for attempt in 0..self.max_attempts {
            match self.try_embed(logical_graph, self.seed + attempt as u64) {
                Ok(result) => return Ok(result),
                Err(_) => continue,
            }
        }

        Err(EmbeddingError::EmbeddingFailed {
            attempts: self.max_attempts,
        })
    }

    /// Single embedding attempt with given seed
    fn try_embed(
        &self,
        logical_graph: &InteractionGraph,
        seed: u64,
    ) -> Result<EmbeddingResult, EmbeddingError> {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(seed);
        let n_logical = logical_graph.n_nodes;
        let n_physical = self.hardware.num_qubits;

        // Initialize mappings
        let mut logical_to_physical: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut physical_used: HashSet<usize> = HashSet::new();

        // Order logical qubits by degree (highest first - helps reduce chain lengths)
        let mut logical_order: Vec<usize> = (0..n_logical).collect();
        logical_order.sort_by(|&a, &b| {
            logical_graph.degree(b).cmp(&logical_graph.degree(a))
        });

        // Embed each logical qubit
        for &logical_q in &logical_order {
            // Find neighbors that are already embedded
            let embedded_neighbors: Vec<usize> = logical_graph.adjacency[logical_q]
                .iter()
                .filter(|&&n| n < logical_q || logical_to_physical.contains_key(&n))
                .copied()
                .collect();

            // Find candidate physical qubits
            let candidates = self.find_candidate_qubits(
                &embedded_neighbors,
                &logical_to_physical,
                &physical_used,
                &mut rng,
            );

            if candidates.is_empty() {
                return Err(EmbeddingError::EmbeddingFailed { attempts: 1 });
            }

            // Select best candidate chain
            let chain = self.select_chain(
                &candidates,
                &embedded_neighbors,
                &logical_to_physical,
                &mut rng,
            );

            if chain.is_empty() {
                return Err(EmbeddingError::EmbeddingFailed { attempts: 1 });
            }

            // Verify chain connectivity
            if !self.verify_chain_connectivity(&chain) {
                return Err(EmbeddingError::ChainDisconnected(logical_q));
            }

            // Assign chain to logical qubit
            for &phys_q in &chain {
                physical_used.insert(phys_q);
            }
            logical_to_physical.insert(logical_q, chain);
        }

        // Build reverse mapping
        let mut physical_to_logical: HashMap<usize, Option<usize>> = HashMap::new();
        for phys_q in 0..n_physical {
            physical_to_logical.insert(phys_q, None);
        }
        for (&logical, chain) in &logical_to_physical {
            for &phys in chain {
                physical_to_logical.insert(phys, Some(logical));
            }
        }

        // Calculate statistics
        let chain_lengths: Vec<usize> = logical_to_physical.values().map(|c| c.len()).collect();
        let avg_chain_length = chain_lengths.iter().sum::<usize>() as f64 / chain_lengths.len() as f64;
        let max_chain_length = *chain_lengths.iter().max().unwrap_or(&0);

        Ok(EmbeddingResult {
            logical_to_physical,
            physical_to_logical,
            stats: EmbeddingStats {
                logical_qubits: n_logical,
                physical_qubits_used: physical_used.len(),
                avg_chain_length,
                max_chain_length,
                attempts: 1,
                time_ms: 0, // Would track in production
            },
        })
    }

    /// Find candidate physical qubit sets for a logical qubit
    fn find_candidate_qubits<R: Rng>(
        &self,
        embedded_neighbors: &[usize],
        logical_to_physical: &HashMap<usize, Vec<usize>>,
        physical_used: &HashSet<usize>,
        rng: &mut R,
    ) -> Vec<Vec<usize>> {
        let mut candidates = Vec::new();

        // Strategy 1: Try to place adjacent to already-embedded neighbors
        for &neighbor_logical in embedded_neighbors {
            if let Some(neighbor_chain) = logical_to_physical.get(&neighbor_logical) {
                // Find unused physical qubits adjacent to neighbor's chain
                for &neighbor_phys in neighbor_chain {
                    for &adjacent in &self.hardware.adjacency[neighbor_phys] {
                        if !physical_used.contains(&adjacent) {
                            candidates.push(vec![adjacent]);
                        }
                    }
                }
            }
        }

        // Strategy 2: If no adjacent spots, find any available qubit
        if candidates.is_empty() {
            let available: Vec<usize> = (0..self.hardware.num_qubits)
                .filter(|q| !physical_used.contains(q))
                .collect();
            
            // Sample some candidates randomly
            let sample_size = std::cmp::min(10, available.len());
            for _ in 0..sample_size {
                let idx = rng.gen_range(0..available.len());
                candidates.push(vec![available[idx]]);
            }
        }

        candidates
    }

    /// Select the best chain from candidates
    fn select_chain<R: Rng>(
        &self,
        candidates: &[Vec<usize>],
        embedded_neighbors: &[usize],
        logical_to_physical: &HashMap<usize, Vec<usize>>,
        rng: &mut R,
    ) -> Vec<usize> {
        if candidates.is_empty() {
            return Vec::new();
        }

        // Score each candidate based on proximity to neighbors and chain length
        let mut scored: Vec<(f64, Vec<usize>)> = candidates
            .iter()
            .map(|chain| {
                let score = self.score_chain(chain, embedded_neighbors, logical_to_physical);
                (score, chain.clone())
            })
            .collect();

        // Sort by score (higher is better)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Return best candidate (with some randomness for exploration)
        if scored.len() > 1 && rng.gen_bool(0.1) {
            // 10% chance to pick a non-optimal candidate for diversity
            let idx = rng.gen_range(0..std::cmp::min(3, scored.len()));
            scored[idx].1.clone()
        } else {
            scored[0].1.clone()
        }
    }

    /// Score a candidate chain
    fn score_chain(
        &self,
        chain: &[usize],
        embedded_neighbors: &[usize],
        logical_to_physical: &HashMap<usize, Vec<usize>>,
    ) -> f64 {
        let mut score = 0.0;

        // Penalize long chains
        score -= chain.len() as f64 * 0.5;

        // Reward proximity to neighbors
        for &neighbor_logical in embedded_neighbors {
            if let Some(neighbor_chain) = logical_to_physical.get(&neighbor_logical) {
                for &phys in chain {
                    for &neighbor_phys in neighbor_chain {
                        // Check if directly connected
                        if self.hardware.adjacency[phys].contains(&neighbor_phys) {
                            score += 2.0;
                        }
                        // Check distance (BFS)
                        let dist = self.shortest_path_distance(phys, neighbor_phys);
                        if dist <= 2 {
                            score += 1.0 / dist as f64;
                        }
                    }
                }
            }
        }

        score
    }

    /// Verify that all qubits in a chain are connected
    fn verify_chain_connectivity(&self, chain: &[usize]) -> bool {
        if chain.is_empty() {
            return true;
        }
        if chain.len() == 1 {
            return true;
        }

        // BFS from first qubit
        let start = chain[0];
        let chain_set: HashSet<usize> = chain.iter().copied().collect();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        visited.insert(start);
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            for &neighbor in &self.hardware.adjacency[current] {
                if chain_set.contains(&neighbor) && !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        // All chain qubits should be reachable
        visited.len() == chain.len()
    }

    /// Find shortest path distance between two physical qubits
    fn shortest_path_distance(&self, start: usize, end: usize) -> usize {
        if start == end {
            return 0;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        visited.insert(start);
        queue.push_back((start, 0));

        while let Some((current, dist)) = queue.pop_front() {
            for &neighbor in &self.hardware.adjacency[current] {
                if neighbor == end {
                    return dist + 1;
                }
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back((neighbor, dist + 1));
                }
            }
        }

        usize::MAX // No path found
    }

    /// Validate an embedding result
    pub fn validate_embedding(&self, result: &EmbeddingResult) -> Result<(), EmbeddingError> {
        // Check that all chains are connected
        for (&logical, chain) in &result.logical_to_physical {
            if !self.verify_chain_connectivity(chain) {
                return Err(EmbeddingError::ChainDisconnected(logical));
            }
        }

        // Check that chains don't overlap
        let mut seen_physical = HashSet::new();
        for chain in result.logical_to_physical.values() {
            for &phys in chain {
                if !seen_physical.insert(phys) {
                    return Err(EmbeddingError::InvalidTopology(
                        format!("Physical qubit {} used in multiple chains", phys),
                    ));
                }
            }
        }

        Ok(())
    }
}

/// Create an embedding graph from Ising Hamiltonian
pub fn create_embedding_graph<F: Copy>(
    couplings: &[(usize, usize, F)],
    n_spins: usize,
) -> InteractionGraph {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_spins];
    let mut has_coupling = HashSet::new();

    for &(i, j, _) in couplings {
        if i < n_spins && j < n_spins {
            adj[i].push(j);
            adj[j].push(i);
            has_coupling.insert((i.min(j), i.max(j)));
        }
    }

    // Deduplicate adjacency lists
    for neighbors in &mut adj {
        neighbors.sort();
        neighbors.dedup();
    }

    InteractionGraph {
        n_nodes: n_spins,
        adjacency: adj,
        has_coupling,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pegasus_topology_generation() {
        let pegasus = PegasusTopology::new(4);
        assert_eq!(pegasus.total_qubits, 8 * 4 * 4 * 4); // 512 qubits
        
        let adj = pegasus.generate_adjacency();
        assert_eq!(adj.len(), pegasus.total_qubits);
        
        // Each qubit should have at least some connections
        for neighbors in &adj {
            assert!(!neighbors.is_empty());
        }
    }

    #[test]
    fn test_minor_embedder_basic() {
        // Create a simple logical graph (line of 4 nodes)
        let logical_graph = InteractionGraph {
            n_nodes: 4,
            adjacency: vec![
                vec![1],
                vec![0, 2],
                vec![1, 3],
                vec![2],
            ],
            has_coupling: [(0, 1), (1, 2), (2, 3)].into_iter().collect(),
        };

        // Create small hardware topology
        let hardware = HardwareTopology {
            name: "Test".to_string(),
            num_qubits: 8,
            adjacency: vec![
                vec![1, 2, 3],
                vec![0, 2, 3],
                vec![0, 1, 3],
                vec![0, 1, 2, 4, 5],
                vec![3, 5, 6],
                vec![3, 4, 6, 7],
                vec![4, 5, 7],
                vec![5, 6],
            ],
            max_degree: 5,
        };

        let embedder = MinorEmbedder::new(hardware);
        let result = embedder.embed(&logical_graph);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.logical_to_physical.len(), 4);
        
        // Validate the embedding
        assert!(embedder.validate_embedding(&result).is_ok());
    }

    #[test]
    fn test_chain_connectivity_verification() {
        let hardware = HardwareTopology {
            name: "Test".to_string(),
            num_qubits: 4,
            adjacency: vec![
                vec![1],
                vec![0, 2],
                vec![1, 3],
                vec![2],
            ],
            max_degree: 2,
        };

        let embedder = MinorEmbedder::new(hardware);

        // Connected chain
        assert!(embedder.verify_chain_connectivity(&[0, 1, 2]));
        
        // Disconnected chain
        assert!(!embedder.verify_chain_connectivity(&[0, 2]));
    }
}
