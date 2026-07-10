//! Physarum ODE Solver for Slime Mold Network Optimization.
//! 
//! Implements the Ordinary Differential Equation solver for tube conductivity
//! adaptation in the Physarum polycephalum (slime mold) algorithm.
//! 
//! The algorithm models protoplasmic tube networks that adapt their conductivity
//! based on flux, minimizing total transport cost while maintaining connectivity.

use std::collections::HashMap;
use nexus_types::network::NodeId;
use thiserror::Error;

/// Minimum conductivity to prevent graph disconnection (epsilon > 0)
pub const MIN_CONDUCTIVITY: f64 = 1e-6;

/// Maximum conductivity to prevent numerical instability
pub const MAX_CONDUCTIVITY: f64 = 1e6;

/// Edge in the Physarum network with adaptive conductivity
#[derive(Debug, Clone)]
pub struct PhysarumEdge {
    pub from: NodeId,
    pub to: NodeId,
    /// Current tube conductivity
    pub conductivity: f64,
    /// Tube length (cost)
    pub length: f64,
    /// Current flux through the tube
    pub flux: f64,
    /// Previous flux for convergence checking
    pub prev_flux: f64,
}

impl PhysarumEdge {
    pub fn new(from: NodeId, to: NodeId, length: f64) -> Self {
        Self {
            from,
            to,
            conductivity: 0.5, // Initial conductivity
            length,
            flux: 0.0,
            prev_flux: 0.0,
        }
    }

    /// Update conductivity based on flux using ODE solution
    pub fn update_conductivity(&mut self, flux: f64, dt: f64, adaptation_rate: f64) {
        self.prev_flux = self.flux;
        self.flux = flux;

        // ODE: dD/dt = f(|Q|) - gamma * D
        // Where D is conductivity, Q is flux, f is increasing function, gamma is decay
        // Using simplified form: dD/dt = |Q| - D
        
        let flux_magnitude = flux.abs();
        let decay = adaptation_rate * self.conductivity;
        
        // Euler method for ODE integration
        let delta = (flux_magnitude - decay) * dt;
        
        self.conductivity = (self.conductivity + delta)
            .clamp(MIN_CONDUCTIVITY, MAX_CONDUCTIVITY);
    }

    /// Check if edge has converged (flux stabilized)
    pub fn is_converged(&self, tolerance: f64) -> bool {
        (self.flux - self.prev_flux).abs() < tolerance * (self.flux.abs().max(1.0))
    }
}

/// Node pressure in the Physarum network
#[derive(Debug, Clone, Copy)]
pub struct NodePressure {
    pub node_id: NodeId,
    pub pressure: f64,
    /// Whether this is a nutrient source (fixed pressure)
    pub is_source: bool,
    /// Source strength (positive for food, negative for sink)
    pub source_strength: f64,
}

/// Configuration for the ODE solver
#[derive(Debug, Clone, Copy)]
pub struct ODESolverConfig {
    /// Time step for ODE integration
    pub dt: f64,
    /// Adaptation rate parameter (gamma in the ODE)
    pub adaptation_rate: f64,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum iterations before forcing convergence
    pub max_iterations: usize,
    /// Whether to enforce minimum conductivity (prevents disconnection)
    pub enforce_min_conductivity: bool,
}

impl Default for ODESolverConfig {
    fn default() -> Self {
        Self {
            dt: 0.01,
            adaptation_rate: 0.5,
            tolerance: 1e-6,
            max_iterations: 1000,
            enforce_min_conductivity: true,
        }
    }
}

/// Physarum ODE Solver for tube conductivity adaptation
pub struct PhysarumODESolver {
    config: ODESolverConfig,
    edges: HashMap<(NodeId, NodeId), PhysarumEdge>,
    node_pressures: HashMap<NodeId, NodePressure>,
    adjacency: HashMap<NodeId, Vec<NodeId>>,
    iteration_count: usize,
    converged: bool,
}

impl PhysarumODESolver {
    pub fn new(config: ODESolverConfig) -> Self {
        Self {
            config,
            edges: HashMap::new(),
            node_pressures: HashMap::new(),
            adjacency: HashMap::new(),
            iteration_count: 0,
            converged: false,
        }
    }

    /// Add an edge to the network
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, length: f64) {
        let edge = PhysarumEdge::new(from, to, length);
        self.edges.insert((from, to), edge);
        
        // Also add reverse edge for undirected graph
        let reverse_edge = PhysarumEdge::new(to, from, length);
        self.edges.insert((to, from), reverse_edge);

        // Update adjacency
        self.adjacency.entry(from).or_default().push(to);
        self.adjacency.entry(to).or_default().push(from);
    }

    /// Register a node with optional source/sink designation
    pub fn add_node(&mut self, node_id: NodeId, is_source: bool, source_strength: f64) {
        self.node_pressures.insert(node_id, NodePressure {
            node_id,
            pressure: 0.0,
            is_source,
            source_strength,
        });
        
        self.adjacency.entry(node_id).or_default();
    }

    /// Solve the Kirchhoff-Physarum system using iterative method
    /// 
    /// This solves for node pressures given current conductivities,
    /// then updates conductivities based on resulting fluxes.
    pub fn solve_step(&mut self) -> Result<(), SolverError> {
        if self.node_pressures.is_empty() {
            return Err(SolverError::NoNodes);
        }

        self.iteration_count += 1;

        // Step 1: Solve for node pressures using Gauss-Seidel iteration
        self.solve_pressures()?;

        // Step 2: Calculate fluxes from pressure differences
        self.calculate_fluxes();

        // Step 3: Update conductivities based on fluxes
        self.update_conductivities();

        // Step 4: Check convergence
        self.converged = self.check_convergence();

        Ok(())
    }

    /// Run solver until convergence or max iterations
    pub fn run_to_convergence(&mut self) -> Result<usize, SolverError> {
        while !self.converged && self.iteration_count < self.config.max_iterations {
            self.solve_step()?;
        }

        if !self.converged {
            return Err(SolverError::MaxIterationsReached(self.config.max_iterations));
        }

        Ok(self.iteration_count)
    }

    /// Solve for node pressures using Gauss-Seidel iteration
    fn solve_pressures(&mut self) -> Result<(), SolverError> {
        let num_nodes = self.node_pressures.len();
        let max_inner_iterations = 100;
        let inner_tolerance = 1e-8;

        for _ in 0..max_inner_iterations {
            let mut max_change = 0.0;

            for (&node_id, node) in &mut self.node_pressures {
                if node.is_source {
                    continue; // Fixed pressure for sources
                }

                // Kirchhoff's current law: sum of fluxes = source strength
                // Flux_ij = D_ij * (P_i - P_j) / L_ij
                // Sum_j(D_ij * (P_i - P_j) / L_ij) = Q_i
                
                let neighbors = self.adjacency.get(&node_id).cloned().unwrap_or_default();
                if neighbors.is_empty() {
                    continue;
                }

                let mut sum_conductance = 0.0;
                let mut weighted_sum = 0.0;

                for neighbor_id in neighbors {
                    let edge = self.edges.get(&(node_id, neighbor_id));
                    if let Some(e) = edge {
                        let conductance = e.conductivity / e.length;
                        sum_conductance += conductance;
                        
                        let neighbor_pressure = self.node_pressures
                            .get(&neighbor_id)
                            .map(|n| n.pressure)
                            .unwrap_or(0.0);
                        weighted_sum += conductance * neighbor_pressure;
                    }
                }

                if sum_conductance > 0.0 {
                    let old_pressure = node.pressure;
                    // P_i = (sum_j(D_ij * P_j / L_ij) + Q_i) / sum_j(D_ij / L_ij)
                    node.pressure = (weighted_sum + node.source_strength) / sum_conductance;
                    
                    let change = (node.pressure - old_pressure).abs();
                    max_change = max_change.max(change);
                }
            }

            if max_change < inner_tolerance {
                break;
            }
        }

        Ok(())
    }

    /// Calculate fluxes from pressure differences
    fn calculate_fluxes(&mut self) {
        for ((from, to), edge) in &mut self.edges {
            let from_pressure = self.node_pressures.get(from).map(|n| n.pressure).unwrap_or(0.0);
            let to_pressure = self.node_pressures.get(to).map(|n| n.pressure).unwrap_or(0.0);
            
            // Flux = conductivity * pressure_difference / length
            edge.flux = edge.conductivity * (from_pressure - to_pressure) / edge.length;
        }
    }

    /// Update conductivities based on fluxes using ODE
    fn update_conductivities(&mut self) {
        for (_, edge) in &mut self.edges {
            edge.update_conductivity(
                edge.flux,
                self.config.dt,
                self.config.adaptation_rate,
            );

            // Enforce minimum conductivity to prevent disconnection
            if self.config.enforce_min_conductivity {
                edge.conductivity = edge.conductivity.max(MIN_CONDUCTIVITY);
            }
        }
    }

    /// Check if the system has converged
    fn check_convergence(&self) -> bool {
        let mut max_change = 0.0;

        for (_, edge) in &self.edges {
            let change = (edge.flux - edge.prev_flux).abs();
            let normalized = change / (edge.flux.abs().max(1.0));
            max_change = max_change.max(normalized);
        }

        max_change < self.config.tolerance
    }

    /// Get total network cost (sum of length * conductivity)
    pub fn total_network_cost(&self) -> f64 {
        let mut cost = 0.0;
        let mut counted = std::collections::HashSet::new();

        for (_, edge) in &self.edges {
            let key = (edge.from.min(edge.to), edge.to.max(edge.from));
            if !counted.contains(&key) {
                cost += edge.length * edge.conductivity;
                counted.insert(key);
            }
        }

        cost
    }

    /// Get all edges with significant conductivity
    pub fn get_active_edges(&self, threshold: f64) -> Vec<&PhysarumEdge> {
        self.edges.values()
            .filter(|e| e.conductivity > threshold)
            .collect()
    }

    /// Check if graph remains connected
    pub fn is_connected(&self) -> bool {
        if self.node_pressures.is_empty() {
            return true;
        }

        // BFS from first node
        let start = *self.node_pressures.keys().next().unwrap();
        let mut visited = std::collections::HashSet::new();
        let mut queue = vec![start];
        visited.insert(start);

        while let Some(node) = queue.pop() {
            if let Some(neighbors) = self.adjacency.get(&node) {
                for &neighbor in neighbors {
                    // Only count edges with sufficient conductivity
                    if let Some(edge) = self.edges.get(&(node, neighbor)) {
                        if edge.conductivity > MIN_CONDUCTIVITY && !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            queue.push(neighbor);
                        }
                    }
                }
            }
        }

        visited.len() == self.node_pressures.len()
    }

    /// Get iteration count
    pub fn iteration_count(&self) -> usize {
        self.iteration_count
    }

    /// Check if converged
    pub fn is_converged(&self) -> bool {
        self.converged
    }
}

/// Errors for the ODE solver
#[derive(Debug, Error)]
pub enum SolverError {
    #[error("No nodes registered")]
    NoNodes,
    #[error("Maximum iterations ({0}) reached without convergence")]
    MaxIterationsReached(usize),
    #[error("Numerical instability detected")]
    NumericalInstability,
    #[error("Graph disconnected")]
    GraphDisconnected,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_creation() {
        let edge = PhysarumEdge::new(NodeId::new(0), NodeId::new(1), 10.0);
        
        assert_eq!(edge.from, NodeId::new(0));
        assert_eq!(edge.to, NodeId::new(1));
        assert!((edge.conductivity - 0.5).abs() < 1e-10);
        assert!((edge.length - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_conductivity_update() {
        let mut edge = PhysarumEdge::new(NodeId::new(0), NodeId::new(1), 10.0);
        
        // Positive flux should increase conductivity
        edge.update_conductivity(5.0, 0.1, 0.5);
        assert!(edge.conductivity > 0.5);
        
        // Zero flux should decrease conductivity (decay)
        let prev = edge.conductivity;
        edge.update_conductivity(0.0, 0.1, 0.5);
        assert!(edge.conductivity < prev);
        
        // Should never go below minimum
        for _ in 0..100 {
            edge.update_conductivity(0.0, 0.1, 0.5);
        }
        assert!(edge.conductivity >= MIN_CONDUCTIVITY);
    }

    #[test]
    fn test_solver_basic() {
        let config = ODESolverConfig::default();
        let mut solver = PhysarumODESolver::new(config);

        // Create simple network: 0 -- 1 -- 2
        solver.add_node(NodeId::new(0), true, 1.0);  // Source
        solver.add_node(NodeId::new(1), false, 0.0);  // Intermediate
        solver.add_node(NodeId::new(2), true, -1.0); // Sink

        solver.add_edge(NodeId::new(0), NodeId::new(1), 1.0);
        solver.add_edge(NodeId::new(1), NodeId::new(2), 1.0);

        // Run solver
        let result = solver.run_to_convergence();
        assert!(result.is_ok());
        
        // Verify connectivity is maintained
        assert!(solver.is_connected());
    }

    #[test]
    fn test_min_conductivity_enforcement() {
        let mut config = ODESolverConfig::default();
        config.enforce_min_conductivity = true;
        
        let mut solver = PhysarumODESolver::new(config);
        
        solver.add_node(NodeId::new(0), true, 1.0);
        solver.add_node(NodeId::new(1), true, -1.0);
        solver.add_edge(NodeId::new(0), NodeId::new(1), 1.0);

        solver.run_to_convergence().ok();

        // All edges should have at least minimum conductivity
        for edge in solver.edges.values() {
            assert!(edge.conductivity >= MIN_CONDUCTIVITY);
        }
    }
}
