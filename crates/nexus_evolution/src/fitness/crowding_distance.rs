//! Crowding Distance Calculator for NSGA-II
//! 
//! Maintains genetic diversity by calculating the density of solutions
//! surrounding a particular solution in the objective space.

use super::nsga2_sorter::MultiObjectiveFitness;

/// Extended crowding distance with per-objective breakdown
#[derive(Debug, Clone)]
pub struct CrowdingDistanceResult {
    /// Total crowding distance across all objectives
    pub total_distance: f64,
    /// Per-objective crowding distances
    pub per_objective: Vec<f64>,
    /// Rank within the front (0 = boundary, higher = more isolated)
    pub isolation_rank: usize,
}

impl CrowdingDistanceResult {
    pub const fn zero() -> Self {
        Self {
            total_distance: 0.0,
            per_objective: Vec::new(),
            isolation_rank: 0,
        }
    }
}

/// Calculator for crowding distance in multi-objective optimization
pub struct CrowdingDistanceCalculator {
    num_objectives: usize,
    /// Minimum distance to consider solutions as distinct
    epsilon: f64,
}

impl CrowdingDistanceCalculator {
    pub fn new(num_objectives: usize) -> Self {
        Self {
            num_objectives,
            epsilon: 1e-10,
        }
    }

    /// Set epsilon for numerical stability
    pub fn with_epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps.max(1e-15);
        self
    }

    /// Calculate crowding distance for all solutions in a front
    pub fn calculate_front(&self, front: &[usize], fitness_values: &[MultiObjectiveFitness]) -> Vec<f64> {
        let n = front.len();
        if n == 0 {
            return Vec::new();
        }

        let mut distances = vec![0.0f64; n];

        // Boundary points always get infinite distance (maximum priority)
        if n <= 2 {
            return vec![f64::INFINITY; n];
        }

        // Calculate contribution from each objective
        for obj in 0..self.num_objectives {
            self.calculate_objective_distance(obj, front, fitness_values, &mut distances);
        }

        distances
    }

    /// Calculate crowding distance contribution for a single objective
    fn calculate_objective_distance(
        &self,
        obj_idx: usize,
        front: &[usize],
        fitness_values: &[MultiObjectiveFitness],
        distances: &mut [f64],
    ) {
        let n = front.len();
        
        // Create sorted list of (front_index, objective_value)
        let mut sorted: Vec<(usize, f64)> = front
            .iter()
            .enumerate()
            .map(|(idx, &sol_idx)| {
                let val = fitness_values[sol_idx].as_array()[obj_idx.min(3)];
                (idx, val)
            })
            .collect();

        // Sort by objective value
        sorted.sort_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Get min and max for normalization
        let min_val = sorted.first().map(|(_, v)| *v).unwrap_or(0.0);
        let max_val = sorted.last().map(|(_, v)| *v).unwrap_or(0.0);
        let range = max_val - min_val;

        // Skip if no variation in this objective
        if range < self.epsilon {
            return;
        }

        // Boundary points get infinity
        if let Some(&(first_idx, _)) = sorted.first() {
            distances[first_idx] = f64::INFINITY;
        }
        if let Some(&(last_idx, _)) = sorted.last() {
            distances[last_idx] = f64::INFINITY;
        }

        // Calculate normalized distance for intermediate points
        for i in 1..(n - 1) {
            let prev_val = sorted[i - 1].1;
            let next_val = sorted[i + 1].1;
            let normalized_diff = (next_val - prev_val) / range;
            
            // Add to total distance (cumulative across objectives)
            if distances[sorted[i].0].is_finite() {
                distances[sorted[i].0] += normalized_diff;
            }
        }
    }

    /// Calculate crowding distance for a single solution within its front
    pub fn calculate_single(
        &self,
        solution_idx: usize,
        front: &[usize],
        fitness_values: &[MultiObjectiveFitness],
    ) -> f64 {
        let distances = self.calculate_front(front, fitness_values);
        
        front.iter()
            .position(|&idx| idx == solution_idx)
            .map(|pos| distances[pos])
            .unwrap_or(0.0)
    }

    /// Compare two solutions using crowded comparison operator
    /// Returns true if solution i is better than solution j
    pub fn crowded_compare(
        &self,
        i_rank: usize,
        i_distance: f64,
        j_rank: usize,
        j_distance: f64,
    ) -> bool {
        // Lower rank is better
        if i_rank != j_rank {
            return i_rank < j_rank;
        }
        
        // Same rank: higher crowding distance is better (more isolated)
        i_distance > j_distance
    }

    /// Sort solutions by crowding distance (descending)
    pub fn sort_by_crowding(&self, distances: &[f64]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..distances.len()).collect();
        
        indices.sort_by(|&a, &b| {
            distances[b].partial_cmp(&distances[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        indices
    }

    /// Select diverse subset of solutions using crowding distance
    pub fn select_diverse(
        &self,
        front: &[usize],
        fitness_values: &[MultiObjectiveFitness],
        count: usize,
    ) -> Vec<usize> {
        if count >= front.len() {
            return front.to_vec();
        }

        let distances = self.calculate_front(front, fitness_values);
        
        // Create (index, distance) pairs
        let mut indexed: Vec<(usize, f64)> = front.iter()
            .zip(distances.iter())
            .map(|(&idx, &dist)| (idx, dist))
            .collect();

        // Sort by distance descending
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top count
        indexed.into_iter()
            .take(count)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Calculate diversity metric for a front (average crowding distance)
    pub fn calculate_diversity_metric(&self, front: &[usize], fitness_values: &[MultiObjectiveFitness]) -> f64 {
        let distances = self.calculate_front(front, fitness_values);
        
        // Filter out infinite distances (boundary points)
        let finite_distances: Vec<f64> = distances
            .into_iter()
            .filter(|d| d.is_finite())
            .collect();

        if finite_distances.is_empty() {
            return f64::INFINITY; // All boundary points = maximum diversity
        }

        finite_distances.iter().sum::<f64>() / finite_distances.len() as f64
    }
}

/// Helper for creating calculators with standard configuration
pub fn create_default_calculator() -> CrowdingDistanceCalculator {
    CrowdingDistanceCalculator::new(4) // 4 objectives: Sharpe, DD, Turnover, Orthogonality
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_points_infinity() {
        let calc = CrowdingDistanceCalculator::new(4);
        
        let fitness_values = vec![
            MultiObjectiveFitness::new(1.0, 0.1, 0.2, 0.5),
            MultiObjectiveFitness::new(2.0, 0.2, 0.3, 0.6),
        ];
        let front = vec![0, 1];
        
        let distances = calc.calculate_front(&front, &fitness_values);
        
        assert_eq!(distances.len(), 2);
        assert!(distances[0].is_infinite());
        assert!(distances[1].is_infinite());
    }

    #[test]
    fn test_intermediate_distances() {
        let calc = CrowdingDistanceCalculator::new(4);
        
        let fitness_values = vec![
            MultiObjectiveFitness::new(1.0, 0.1, 0.2, 0.5), // Min sharpe
            MultiObjectiveFitness::new(3.0, 0.3, 0.4, 0.7), // Max sharpe
            MultiObjectiveFitness::new(2.0, 0.2, 0.3, 0.6), // Middle
        ];
        let front = vec![0, 1, 2];
        
        let distances = calc.calculate_front(&front, &fitness_values);
        
        // Boundaries should be infinite
        assert!(distances[0].is_infinite()); // Min
        assert!(distances[1].is_infinite()); // Max
        
        // Middle point should have finite positive distance
        assert!(distances[2].is_finite());
        assert!(distances[2] > 0.0);
    }

    #[test]
    fn test_crowded_comparison() {
        let calc = CrowdingDistanceCalculator::new(4);
        
        // Better rank wins
        assert!(calc.crowded_compare(0, 1.0, 1, 2.0));
        assert!(!calc.crowded_compare(1, 2.0, 0, 1.0));
        
        // Same rank: higher distance wins
        assert!(calc.crowded_compare(0, 2.0, 0, 1.0));
        assert!(!calc.crowded_compare(0, 1.0, 0, 2.0));
    }

    #[test]
    fn test_diverse_selection() {
        let calc = CrowdingDistanceCalculator::new(4);
        
        let fitness_values = vec![
            MultiObjectiveFitness::new(1.0, 0.1, 0.2, 0.5),
            MultiObjectiveFitness::new(1.5, 0.15, 0.25, 0.55),
            MultiObjectiveFitness::new(2.0, 0.2, 0.3, 0.6),
            MultiObjectiveFitness::new(2.5, 0.25, 0.35, 0.65),
            MultiObjectiveFitness::new(3.0, 0.3, 0.4, 0.7),
        ];
        let front = vec![0, 1, 2, 3, 4];
        
        // Select 3 most diverse
        let selected = calc.select_diverse(&front, &fitness_values, 3);
        
        assert_eq!(selected.len(), 3);
        // Should include boundary points (most diverse)
        assert!(selected.contains(&0) || selected.contains(&4));
    }

    #[test]
    fn test_diversity_metric() {
        let calc = CrowdingDistanceCalculator::new(4);
        
        // Uniformly spaced solutions
        let fitness_values: Vec<MultiObjectiveFitness> = (0..5)
            .map(|i| MultiObjectiveFitness::new(i as f64, 0.1 * i as f64, 0.1 * i as f64, 0.5))
            .collect();
        let front = vec![0, 1, 2, 3, 4];
        
        let metric = calc.calculate_diversity_metric(&front, &fitness_values);
        assert!(metric.is_finite());
        assert!(metric > 0.0);
    }
}
