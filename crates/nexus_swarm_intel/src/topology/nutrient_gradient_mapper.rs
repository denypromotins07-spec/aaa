//! Nutrient Gradient Mapper for Physarum Network Optimization.
//! 
//! Maps nutrient sources (data flow requirements) in the network topology
//! and computes gradient fields that guide tube growth toward high-demand nodes.

use std::collections::{HashMap, VecDeque};
use nexus_types::network::NodeId;
use thiserror::Error;

/// Represents a nutrient source/sink in the network
#[derive(Debug, Clone, Copy)]
pub struct NutrientSource {
    pub node_id: NodeId,
    /// Positive for food sources (data producers), negative for sinks (consumers)
    pub strength: f64,
    /// Priority weight for this source
    pub priority: f64,
}

impl NutrientSource {
    pub fn new(node_id: NodeId, strength: f64, priority: f64) -> Self {
        Self {
            node_id,
            strength,
            priority: priority.max(0.01), // Minimum priority
        }
    }

    /// Check if this is a sink (consumer)
    pub fn is_sink(&self) -> bool {
        self.strength < 0.0
    }

    /// Check if this is a source (producer)
    pub fn is_source(&self) -> bool {
        self.strength > 0.0
    }
}

/// Gradient field value at a node
#[derive(Debug, Clone, Copy)]
pub struct GradientField {
    /// Nutrient concentration at this node
    pub concentration: f64,
    /// Gradient magnitude (direction of steepest ascent)
    pub gradient_magnitude: f64,
    /// Direction to highest concentration neighbor
    pub gradient_direction: Option<NodeId>,
}

impl Default for GradientField {
    fn default() -> Self {
        Self {
            concentration: 0.0,
            gradient_magnitude: 0.0,
            gradient_direction: None,
        }
    }
}

/// Configuration for gradient computation
#[derive(Debug, Clone, Copy)]
pub struct GradientConfig {
    /// Diffusion rate for nutrient spread
    pub diffusion_rate: f64,
    /// Decay rate for nutrient over distance
    pub decay_rate: f64,
    /// Maximum propagation distance (hops)
    pub max_distance: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self {
            diffusion_rate: 0.5,
            decay_rate: 0.1,
            max_distance: 20,
            tolerance: 1e-6,
        }
    }
}

/// Nutrient gradient mapper for computing chemotactic fields
pub struct NutrientGradientMapper {
    config: GradientConfig,
    sources: HashMap<NodeId, NutrientSource>,
    gradients: HashMap<NodeId, GradientField>,
    adjacency: HashMap<NodeId, Vec<NodeId>>,
    edge_weights: HashMap<(NodeId, NodeId), f64>,
}

impl NutrientGradientMapper {
    pub fn new(config: GradientConfig) -> Self {
        Self {
            config,
            sources: HashMap::new(),
            gradients: HashMap::new(),
            adjacency: HashMap::new(),
            edge_weights: HashMap::new(),
        }
    }

    /// Add a nutrient source to the map
    pub fn add_source(&mut self, source: NutrientSource) {
        self.sources.insert(source.node_id, source);
    }

    /// Remove a nutrient source
    pub fn remove_source(&mut self, node_id: NodeId) {
        self.sources.remove(&node_id);
    }

    /// Register an edge in the network
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, weight: f64) {
        self.adjacency.entry(from).or_default().push(to);
        self.adjacency.entry(to).or_default().push(from);
        self.edge_weights.insert((from, to), weight);
        self.edge_weights.insert((to, from), weight);
        
        // Initialize gradients for new nodes
        self.gradients.entry(from).or_default();
        self.gradients.entry(to).or_default();
    }

    /// Compute nutrient gradient field using iterative diffusion
    pub fn compute_gradients(&mut self) -> Result<(), GradientError> {
        if self.sources.is_empty() {
            return Ok(()); // No sources, zero gradient everywhere
        }

        // Initialize concentrations from sources
        for (&node_id, source) in &self.sources {
            let base_concentration = source.strength.abs() * source.priority;
            self.gradients.insert(node_id, GradientField {
                concentration: base_concentration,
                ..Default::default()
            });
        }

        // Iterative diffusion using Jacobi method
        let max_iterations = 100;
        let mut iteration = 0;

        while iteration < max_iterations {
            let mut new_gradients = self.gradients.clone();
            let mut max_change = 0.0;

            for (&node_id, current) in &self.gradients {
                // Skip source nodes (fixed concentration)
                if self.sources.contains_key(&node_id) {
                    continue;
                }

                let neighbors = self.adjacency.get(&node_id).cloned().unwrap_or_default();
                if neighbors.is_empty() {
                    continue;
                }

                // Diffusion: average neighbor concentrations weighted by edge weights
                let mut sum_concentration = 0.0;
                let mut total_weight = 0.0;

                for neighbor_id in neighbors {
                    if let Some(neighbor_grad) = self.gradients.get(&neighbor_id) {
                        let weight = self.edge_weights.get(&(node_id, neighbor_id))
                            .copied()
                            .unwrap_or(1.0);
                        
                        // Apply distance decay
                        let decay = (-self.config.decay_rate).exp();
                        sum_concentration += neighbor_grad.concentration * weight * decay;
                        total_weight += weight;
                    }
                }

                if total_weight > 0.0 {
                    let new_concentration = sum_concentration / total_weight;
                    
                    // Apply diffusion rate
                    let old_concentration = current.concentration;
                    if let Some(target) = new_gradients.get_mut(&node_id) {
                        target.concentration =
                            old_concentration * (1.0 - self.config.diffusion_rate) +
                            new_concentration * self.config.diffusion_rate;
                    }

                    let change = (new_concentration - old_concentration).abs();
                    max_change = max_change.max(change);
                }
            }

            self.gradients = new_gradients;
            iteration += 1;

            if max_change < self.config.tolerance {
                break;
            }
        }

        // Compute gradient directions after convergence
        self.compute_gradient_directions();

        Ok(())
    }

    /// Compute gradient direction for each node
    fn compute_gradient_directions(&mut self) {
        for (&node_id, field) in &mut self.gradients {
            let neighbors = self.adjacency.get(&node_id).cloned().unwrap_or_default();
            
            let mut best_neighbor = None;
            let mut max_concentration = field.concentration;

            for neighbor_id in neighbors {
                if let Some(neighbor_grad) = self.gradients.get(&neighbor_id) {
                    if neighbor_grad.concentration > max_concentration {
                        max_concentration = neighbor_grad.concentration;
                        best_neighbor = Some(neighbor_id);
                    }
                }
            }

            field.gradient_direction = best_neighbor;
            field.gradient_magnitude = max_concentration - field.concentration;
        }
    }

    /// Get the gradient field at a specific node
    pub fn get_gradient(&self, node_id: NodeId) -> Option<&GradientField> {
        self.gradients.get(&node_id)
    }

    /// Find path following gradient ascent to nearest source
    pub fn find_gradient_ascent_path(&self, start: NodeId) -> Vec<NodeId> {
        let mut path = vec![start];
        let mut current = start;
        let mut visited = std::collections::HashSet::new();
        visited.insert(start);

        while let Some(field) = self.gradients.get(&current) {
            if let Some(next) = field.gradient_direction {
                if visited.contains(&next) {
                    break; // Cycle detected
                }
                
                path.push(next);
                visited.insert(next);
                current = next;

                // Stop if we reached a source
                if self.sources.contains_key(&current) {
                    break;
                }
            } else {
                break; // No gradient direction
            }
        }

        path
    }

    /// Get all nutrient sources
    pub fn sources(&self) -> impl Iterator<Item = (&NodeId, &NutrientSource)> {
        self.sources.iter()
    }

    /// Clear all gradients (for recomputation)
    pub fn clear_gradients(&mut self) {
        self.gradients.clear();
    }

    /// Update configuration
    pub fn update_config(&mut self, config: GradientConfig) {
        self.config = config;
    }
}

/// Errors for gradient mapping operations
#[derive(Debug, Error)]
pub enum GradientError {
    #[error("Node not found in graph")]
    NodeNotFound(NodeId),
    #[error("Invalid source strength")]
    InvalidStrength,
    #[error("Graph disconnected")]
    GraphDisconnected,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nutrient_source_creation() {
        let source = NutrientSource::new(NodeId::new(0), 10.0, 1.0);
        assert!(source.is_source());
        assert!(!source.is_sink());
        assert!((source.strength - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_sink_creation() {
        let sink = NutrientSource::new(NodeId::new(1), -5.0, 1.0);
        assert!(!sink.is_source());
        assert!(sink.is_sink());
    }

    #[test]
    fn test_gradient_mapper_basic() {
        let config = GradientConfig::default();
        let mut mapper = NutrientGradientMapper::new(config);

        // Create simple network: 0 -- 1 -- 2
        mapper.add_source(NutrientSource::new(NodeId::new(0), 10.0, 1.0));
        mapper.add_edge(NodeId::new(0), NodeId::new(1), 1.0);
        mapper.add_edge(NodeId::new(1), NodeId::new(2), 1.0);

        mapper.compute_gradients().unwrap();

        // Node 0 should have highest concentration (it's the source)
        let grad_0 = mapper.get_gradient(NodeId::new(0)).unwrap();
        let grad_1 = mapper.get_gradient(NodeId::new(1)).unwrap();
        let grad_2 = mapper.get_gradient(NodeId::new(2)).unwrap();

        assert!(grad_0.concentration >= grad_1.concentration);
        assert!(grad_1.concentration >= grad_2.concentration);
    }

    #[test]
    fn test_gradient_ascent_path() {
        let config = GradientConfig::default();
        let mut mapper = NutrientGradientMapper::new(config);

        // Create network with source at one end
        mapper.add_source(NutrientSource::new(NodeId::new(0), 10.0, 1.0));
        mapper.add_edge(NodeId::new(0), NodeId::new(1), 1.0);
        mapper.add_edge(NodeId::new(1), NodeId::new(2), 1.0);
        mapper.add_edge(NodeId::new(2), NodeId::new(3), 1.0);

        mapper.compute_gradients().unwrap();

        // Path from node 3 should lead back to source at node 0
        let path = mapper.find_gradient_ascent_path(NodeId::new(3));
        assert!(!path.is_empty());
        assert_eq!(path[0], NodeId::new(3));
        assert!(path.contains(&NodeId::new(0)));
    }

    #[test]
    fn test_priority_effect() {
        let config = GradientConfig::default();
        let mut mapper = NutrientGradientMapper::new(config);

        // Two sources with different priorities
        mapper.add_source(NutrientSource::new(NodeId::new(0), 10.0, 1.0));
        mapper.add_source(NutrientSource::new(NodeId::new(2), 10.0, 5.0)); // Higher priority
        mapper.add_edge(NodeId::new(0), NodeId::new(1), 1.0);
        mapper.add_edge(NodeId::new(1), NodeId::new(2), 1.0);

        mapper.compute_gradients().unwrap();

        // Node 1 should be more influenced by higher priority source
        let grad_1 = mapper.get_gradient(NodeId::new(1)).unwrap();
        // The gradient direction should point toward the higher priority source
        assert_eq!(grad_1.gradient_direction, Some(NodeId::new(2)));
    }
}
