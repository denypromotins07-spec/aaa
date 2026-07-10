//! NSGA-II Multi-Objective Fitness Sorter
//! 
//! Implements the Non-dominated Sorting Genetic Algorithm II for
//! multi-objective optimization of trading strategies.
//! 
//! Objectives (all to be maximized):
//! 1. Out-of-Sample Sharpe Ratio
//! 2. Negative Max Drawdown (minimize drawdown)
//! 3. Negative Turnover (minimize transaction costs)
//! 4. Orthogonality to existing portfolio alphas

use super::orthogonality_penalty::OrthogonalityCalculator;
use crate::sandbox::cpcv_overfit_guard::CpcvResult;
use std::collections::HashMap;

/// Multi-objective fitness values for a strategy
#[derive(Debug, Clone)]
pub struct MultiObjectiveFitness {
    /// Objective 1: Out-of-Sample Sharpe Ratio (maximize)
    pub oos_sharpe: f64,
    /// Objective 2: Negative Max Drawdown (maximize = minimize DD)
    pub neg_max_drawdown: f64,
    /// Objective 3: Negative Turnover (maximize = minimize turnover)
    pub neg_turnover: f64,
    /// Objective 4: Orthogonality score (maximize = less correlation)
    pub orthogonality: f64,
    /// Combined raw score (for quick comparisons)
    pub composite_score: f64,
}

impl MultiObjectiveFitness {
    pub fn new(
        oos_sharpe: f64,
        max_drawdown: f64,
        turnover: f64,
        orthogonality: f64,
    ) -> Self {
        let neg_max_dd = -max_drawdown;
        let neg_turn = -turnover;
        
        // Weighted composite (weights can be tuned)
        let composite = 
            oos_sharpe * 0.4 +
            neg_max_dd * 0.3 +
            neg_turn * 0.15 +
            orthogonality * 0.15;
        
        Self {
            oos_sharpe,
            neg_max_drawdown: neg_max_dd,
            neg_turnover: neg_turn,
            orthogonality,
            composite_score: composite,
        }
    }

    /// Get objectives as array for NSGA-II operations
    pub fn as_array(&self) -> [f64; 4] {
        [
            self.oos_sharpe,
            self.neg_max_drawdown,
            self.neg_turnover,
            self.orthogonality,
        ]
    }
}

/// Result of non-dominated sorting
#[derive(Debug, Clone)]
pub struct Nsga2Ranking {
    /// Pareto front rank (0 = best, non-dominated)
    pub rank: usize,
    /// Crowding distance within the front
    pub crowding_distance: f64,
    /// Number of solutions that dominate this one
    pub domination_count: usize,
    /// Indices of solutions dominated by this one
    pub dominated_solutions: Vec<usize>,
}

impl Nsga2Ranking {
    pub fn new() -> Self {
        Self {
            rank: 0,
            crowding_distance: 0.0,
            domination_count: 0,
            dominated_solutions: Vec::new(),
        }
    }
}

impl Default for Nsga2Ranking {
    fn default() -> Self {
        Self::new()
    }
}

/// NSGA-II sorter for multi-objective optimization
pub struct Nsga2Sorter {
    population_size: usize,
    num_objectives: usize,
}

impl Nsga2Sorter {
    pub fn new(population_size: usize, num_objectives: usize) -> Self {
        Self {
            population_size,
            num_objectives,
        }
    }

    /// Perform fast non-dominated sort
    /// Returns vector of fronts, where each front contains indices of solutions
    pub fn fast_non_dominated_sort(
        &self,
        fitness_values: &[MultiObjectiveFitness],
    ) -> Vec<Vec<usize>> {
        let n = fitness_values.len();
        let mut rankings: Vec<Nsga2Ranking> = vec![Nsga2Ranking::new(); n];

        // Compare all pairs
        for i in 0..n {
            for j in (i + 1)..n {
                let dominates_ij = self.dominates(&fitness_values[i], &fitness_values[j]);
                let dominates_ji = self.dominates(&fitness_values[j], &fitness_values[i]);

                if dominates_ij {
                    rankings[i].dominated_solutions.push(j);
                    rankings[j].domination_count += 1;
                } else if dominates_ji {
                    rankings[j].dominated_solutions.push(i);
                    rankings[i].domination_count += 1;
                }
            }
        }

        // Build fronts
        let mut fronts: Vec<Vec<usize>> = Vec::new();
        let mut current_front: Vec<usize> = Vec::new();

        // First front: solutions with domination_count == 0
        for i in 0..n {
            if rankings[i].domination_count == 0 {
                rankings[i].rank = 0;
                current_front.push(i);
            }
        }

        if !current_front.is_empty() {
            fronts.push(current_front);
        }

        // Build subsequent fronts
        let mut front_idx = 0;
        while front_idx < fronts.len() {
            let mut next_front: Vec<usize> = Vec::new();

            for &i in &fronts[front_idx] {
                for &j in &rankings[i].dominated_solutions {
                    rankings[j].domination_count -= 1;
                    if rankings[j].domination_count == 0 {
                        rankings[j].rank = front_idx + 1;
                        next_front.push(j);
                    }
                }
            }

            if !next_front.is_empty() {
                fronts.push(next_front);
            }

            front_idx += 1;
        }

        // Store rankings for crowding distance calculation
        self.store_rankings(rankings);

        fronts
    }

    /// Check if solution a dominates solution b
    /// (a is better than b in at least one objective and not worse in any)
    fn dominates(&self, a: &MultiObjectiveFitness, b: &MultiObjectiveFitness) -> bool {
        let a_vals = a.as_array();
        let b_vals = b.as_array();

        let mut at_least_one_better = false;

        for i in 0..self.num_objectives.min(4) {
            if a_vals[i] < b_vals[i] {
                return false; // a is worse in this objective
            }
            if a_vals[i] > b_vals[i] {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }

    /// Calculate crowding distance for each front
    pub fn calculate_crowding_distance(
        &self,
        front: &[usize],
        fitness_values: &[MultiObjectiveFitness],
    ) -> Vec<f64> {
        let n = front.len();
        if n == 0 {
            return Vec::new();
        }

        let mut distances = vec![0.0f64; n];

        if n <= 2 {
            // Boundary points get infinite distance
            for i in 0..n {
                distances[i] = f64::INFINITY;
            }
            return distances;
        }

        // Calculate distance for each objective
        for obj in 0..self.num_objectives.min(4) {
            // Sort front by this objective
            let mut sorted_indices: Vec<(usize, f64)> = front
                .iter()
                .enumerate()
                .map(|(idx, &sol_idx)| (idx, fitness_values[sol_idx].as_array()[obj]))
                .collect();

            sorted_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Boundary points get infinite distance
            distances[sorted_indices.first().unwrap().0] = f64::INFINITY;
            distances[sorted_indices.last().unwrap().0] = f64::INFINITY;

            // Calculate range for normalization
            let min_val = sorted_indices.first().unwrap().1;
            let max_val = sorted_indices.last().unwrap().1;
            let range = max_val - min_val;

            if range < 1e-10 {
                continue; // Skip if no variation
            }

            // Calculate distances for intermediate points
            for i in 1..(n - 1) {
                let prev_val = sorted_indices[i - 1].1;
                let next_val = sorted_indices[i + 1].1;
                distances[sorted_indices[i].0] += (next_val - prev_val) / range;
            }
        }

        distances
    }

    /// Select best individuals using tournament selection with crowding comparison
    pub fn tournament_select(
        &self,
        fitness_values: &[MultiObjectiveFitness],
        rankings: &[Nsga2Ranking],
        tournament_size: usize,
    ) -> usize {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut best_idx = rng.gen_range(0..fitness_values.len());

        for _ in 1..tournament_size {
            let challenger = rng.gen_range(0..fitness_values.len());

            // Compare by rank first, then by crowding distance
            let best_rank = rankings[best_idx].rank;
            let challenger_rank = rankings[challenger].rank;

            if challenger_rank < best_rank {
                best_idx = challenger;
            } else if challenger_rank == best_rank {
                let best_cd = rankings[best_idx].crowding_distance;
                let challenger_cd = rankings[challenger].crowding_distance;
                if challenger_cd > best_cd {
                    best_idx = challenger;
                }
            }
        }

        best_idx
    }

    /// Store rankings with crowding distances (simplified storage)
    fn store_rankings(&self, _rankings: Vec<Nsga2Ranking>) {
        // In a full implementation, this would store rankings for later access
        // For now, we compute on-demand
    }

    /// Sort population by NSGA-II criteria
    /// Returns indices sorted by preference (best first)
    pub fn sort_population(
        &self,
        fitness_values: &[MultiObjectiveFitness],
    ) -> Vec<usize> {
        let fronts = self.fast_non_dominated_sort(fitness_values);
        
        let mut sorted_indices = Vec::with_capacity(fitness_values.len());

        for front in fronts {
            let distances = self.calculate_crowding_distance(&front, fitness_values);
            
            // Create (index, distance) pairs and sort by distance descending
            let mut front_with_dist: Vec<(usize, f64)> = front
                .into_iter()
                .zip(distances.into_iter())
                .collect();

            front_with_dist.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            for (idx, _) in front_with_dist {
                sorted_indices.push(idx);
            }
        }

        sorted_indices
    }
}

/// Default sorter for standard 4-objective optimization
pub fn create_default_sorter(population_size: usize) -> Nsga2Sorter {
    Nsga2Sorter::new(population_size, 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fitness_creation() {
        let fitness = MultiObjectiveFitness::new(2.0, 0.15, 0.3, 0.8);
        assert!((fitness.oos_sharpe - 2.0).abs() < 1e-10);
        assert!((fitness.neg_max_drawdown - (-0.15)).abs() < 1e-10);
        assert!((fitness.neg_turnover - (-0.3)).abs() < 1e-10);
    }

    #[test]
    fn test_dominance() {
        let sorter = Nsga2Sorter::new(10, 4);
        
        // a dominates b (better in all objectives)
        let a = MultiObjectiveFitness::new(2.0, 0.1, 0.2, 0.9);
        let b = MultiObjectiveFitness::new(1.0, 0.2, 0.3, 0.5);
        
        assert!(sorter.dominates(&a, &b));
        assert!(!sorter.dominates(&b, &a));
    }

    #[test]
    fn test_non_dominated_sort() {
        let sorter = Nsga2Sorter::new(10, 4);
        
        let fitness_values = vec![
            MultiObjectiveFitness::new(2.0, 0.1, 0.2, 0.9), // Best
            MultiObjectiveFitness::new(1.0, 0.2, 0.3, 0.5), // Dominated by first
            MultiObjectiveFitness::new(1.5, 0.15, 0.25, 0.7), // Might be non-dominated
        ];

        let fronts = sorter.fast_non_dominated_sort(&fitness_values);
        assert!(!fronts.is_empty());
        assert!(fronts[0].contains(&0)); // Best should be in first front
    }
}
