//! Classical Simulated Annealing Fallback
//! 
//! Pure Rust implementation of simulated annealing for QUBO optimization.
//! Used as fallback when quantum APIs are unavailable or return invalid solutions.

use ndarray::{Array2, Array1};
use num_traits::Float;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use thiserror::Error;
use tokio::task::spawn_blocking;

use crate::qubo::portfolio_hamiltonian::QuboMatrix;

/// Errors that can occur during classical annealing
#[derive(Error, Debug)]
pub enum AnnealingError {
    #[error("Invalid problem size: {0}")]
    InvalidProblemSize(String),
    #[error("Numerical overflow in energy calculation")]
    NumericalOverflow,
    #[error("Failed to converge after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },
    #[error("Temperature schedule error: {0}")]
    TemperatureError(String),
}

/// Configuration for simulated annealing
#[derive(Debug, Clone)]
pub struct AnnealingConfig {
    /// Initial temperature
    pub initial_temperature: f64,
    /// Final temperature (stopping criterion)
    pub final_temperature: f64,
    /// Cooling rate (alpha in T_new = alpha * T_old)
    pub cooling_rate: f64,
    /// Number of iterations per temperature step
    pub iterations_per_temp: usize,
    /// Maximum total iterations
    pub max_total_iterations: usize,
    /// Random seed for reproducibility (None for random)
    pub seed: Option<u64>,
    /// Reheating enabled (restart from higher temp if stuck)
    pub enable_reheating: bool,
    /// Reheat multiplier
    pub reheat_multiplier: f64,
    /// Stagnation threshold for reheating
    pub stagnation_threshold: usize,
}

impl Default for AnnealingConfig {
    fn default() -> Self {
        Self {
            initial_temperature: 100.0,
            final_temperature: 0.01,
            cooling_rate: 0.95,
            iterations_per_temp: 100,
            max_total_iterations: 100_000,
            seed: None,
            enable_reheating: true,
            reheat_multiplier: 2.0,
            stagnation_threshold: 50,
        }
    }
}

/// Current state of the annealing process
#[derive(Debug, Clone)]
pub struct AnnealingState<F: Float> {
    /// Current solution (binary vector)
    pub solution: Vec<i8>,
    /// Current energy
    pub energy: F,
    /// Current temperature
    pub temperature: F,
    /// Iteration count
    pub iteration: usize,
    /// Best energy seen so far
    pub best_energy: F,
    /// Best solution seen so far
    pub best_solution: Vec<i8>,
    /// Iterations without improvement
    pub stagnation_count: usize,
}

/// Result of classical annealing
#[derive(Debug, Clone)]
pub struct AnnealingResult {
    /// Final binary solution
    pub solution: Vec<i8>,
    /// Energy of the solution
    pub energy: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Final temperature
    pub final_temperature: f64,
}

/// Classical Simulated Annealer for QUBO problems
pub struct ClassicalSimulatedAnnealer {
    config: AnnealingConfig,
}

impl ClassicalSimulatedAnnealer {
    /// Create a new annealer with default configuration
    pub fn new() -> Self {
        Self {
            config: AnnealingConfig::default(),
        }
    }

    /// Create a new annealer with custom configuration
    pub fn with_config(config: AnnealingConfig) -> Self {
        Self { config }
    }

    /// Solve a QUBO problem using simulated annealing
    /// 
    /// This is an async wrapper that runs the blocking computation
    /// on a thread pool to avoid blocking the Tokio runtime.
    pub async fn solve(&self, qubo: &QuboMatrix<f64>) -> Result<AnnealingResult, AnnealingError> {
        let config = self.config.clone();
        let matrix = qubo.matrix.clone();
        let linear = qubo.linear_term.clone();
        let n = qubo.n_qubits;
        
        // Run blocking computation on thread pool
        spawn_blocking(move || {
            run_simulated_annealing(n, &matrix, &linear, &config)
        }).await
            .map_err(|e| AnnealingError::ConvergenceFailure { iterations: 0 })?
    }

    /// Run annealing synchronously (for testing)
    pub fn solve_sync(&self, qubo: &QuboMatrix<f64>) -> Result<AnnealingResult, AnnealingError> {
        run_simulated_annealing(
            qubo.n_qubits,
            &qubo.matrix,
            &qubo.linear_term,
            &self.config,
        )
    }
}

impl Default for ClassicalSimulatedAnnealer {
    fn default() -> Self {
        Self::new()
    }
}

/// Core simulated annealing algorithm
fn run_simulated_annealing(
    n: usize,
    q_matrix: &Array2<f64>,
    linear: &Array1<f64>,
    config: &AnnealingConfig,
) -> Result<AnnealingResult, AnnealingError> {
    if n == 0 {
        return Err(AnnealingError::InvalidProblemSize("Problem has zero variables".to_string()));
    }

    // Initialize RNG
    let mut rng = match config.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    // Initialize random solution
    let mut solution: Vec<i8> = (0..n).map(|_| rng.gen_range(0..2) as i8).collect();
    let mut energy = calculate_energy(&solution, q_matrix, linear);
    
    let mut best_solution = solution.clone();
    let mut best_energy = energy;
    
    let mut temperature = config.initial_temperature;
    let mut iteration = 0;
    let mut stagnation_count = 0;
    let mut converged = false;

    // Main annealing loop
    while temperature > config.final_temperature && iteration < config.max_total_iterations {
        let mut improved_this_temp = false;
        
        for _ in 0..config.iterations_per_temp {
            if iteration >= config.max_total_iterations {
                break;
            }
            
            // Generate neighbor by flipping a random bit
            let flip_idx = rng.gen_range(0..n);
            let mut neighbor = solution.clone();
            neighbor[flip_idx] = 1 - neighbor[flip_idx];
            
            // Calculate energy difference efficiently (only affected terms)
            let delta_e = calculate_energy_delta(
                flip_idx,
                solution[flip_idx],
                1 - solution[flip_idx],
                &solution,
                q_matrix,
                linear,
            );
            
            // Metropolis acceptance criterion
            let accept = if delta_e < 0.0 {
                true // Always accept improvement
            } else if temperature > 0.0 {
                let prob = (-delta_e / temperature).exp();
                rng.gen::<f64>() < prob
            } else {
                false
            };
            
            if accept {
                solution = neighbor;
                energy += delta_e;
                improved_this_temp = true;
                
                // Update best if improved
                if energy < best_energy {
                    best_energy = energy;
                    best_solution = solution.clone();
                    stagnation_count = 0;
                } else {
                    stagnation_count += 1;
                }
            }
            
            iteration += 1;
        }
        
        // Check for stagnation and reheat if enabled
        if config.enable_reheating && stagnation_count >= config.stagnation_threshold {
            temperature *= config.reheat_multiplier;
            stagnation_count = 0;
        }
        
        // Cool down
        temperature *= config.cooling_rate;
        
        // Check convergence
        if !improved_this_temp && stagnation_count >= config.stagnation_threshold * 2 {
            converged = true;
            break;
        }
    }

    Ok(AnnealingResult {
        solution: best_solution,
        energy: best_energy,
        iterations: iteration,
        converged,
        final_temperature: temperature,
    })
}

/// Calculate total energy of a solution: x^T Q x + c^T x
fn calculate_energy(solution: &[i8], q_matrix: &Array2<f64>, linear: &Array1<f64>) -> f64 {
    let n = solution.len();
    let mut energy = 0.0;
    
    // Quadratic term
    for i in 0..n {
        for j in 0..n {
            energy += q_matrix[[i, j]] * solution[i] as f64 * solution[j] as f64;
        }
    }
    
    // Linear term
    for i in 0..n {
        energy += linear[i] * solution[i] as f64;
    }
    
    energy
}

/// Calculate energy change from flipping one bit (efficient O(n) update)
fn calculate_energy_delta(
    flip_idx: usize,
    old_value: i8,
    new_value: i8,
    solution: &[i8],
    q_matrix: &Array2<f64>,
    linear: &Array1<f64>,
) -> f64 {
    let n = solution.len();
    let diff = (new_value - old_value) as f64;
    
    let mut delta = 0.0;
    
    // Diagonal term (i == j)
    delta += q_matrix[[flip_idx, flip_idx]] * ((new_value * new_value) as f64 - (old_value * old_value) as f64);
    
    // Off-diagonal terms (i != j)
    for j in 0..n {
        if j != flip_idx {
            // Terms where flip_idx is first index
            delta += q_matrix[[flip_idx, j]] * diff * solution[j] as f64;
            // Terms where flip_idx is second index
            delta += q_matrix[[j, flip_idx]] * solution[j] as f64 * diff;
        }
    }
    
    // Linear term
    delta += linear[flip_idx] * diff;
    
    delta
}

/// Parallel tempering extension for better exploration
pub struct ParallelTemperingAnnealer {
    configs: Vec<AnnealingConfig>,
    num_replicas: usize,
    swap_frequency: usize,
}

impl ParallelTemperingAnnealer {
    /// Create a parallel tempering annealer with multiple temperature replicas
    pub fn new(num_replicas: usize, base_config: &AnnealingConfig) -> Self {
        let mut configs = Vec::with_capacity(num_replicas);
        
        for i in 0..num_replicas {
            let mut config = base_config.clone();
            // Higher replicas have higher temperatures
            config.initial_temperature *= (i + 1) as f64;
            config.final_temperature *= (i + 1) as f64;
            configs.push(config);
        }
        
        Self {
            configs,
            num_replicas,
            swap_frequency: 10,
        }
    }

    /// Run parallel tempering (simplified version)
    pub async fn solve(&self, qubo: &QuboMatrix<f64>) -> Result<AnnealingResult, AnnealingError> {
        // In production, this would run multiple replicas in parallel
        // and periodically attempt swaps between adjacent temperatures
        // For now, use the highest temperature replica for better exploration
        
        let annealer = ClassicalSimulatedAnnealer::with_config(self.configs[0].clone());
        annealer.solve(qubo).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_annealing_basic() {
        let config = AnnealingConfig {
            initial_temperature: 50.0,
            final_temperature: 0.1,
            cooling_rate: 0.9,
            iterations_per_temp: 50,
            max_total_iterations: 5000,
            seed: Some(42),
            ..Default::default()
        };
        
        let annealer = ClassicalSimulatedAnnealer::with_config(config);
        
        // Simple QUBO: minimize -x1 - x2 (should select both)
        let mut qubo = QuboMatrix::new(4);
        qubo.matrix = Array2::zeros((4, 4));
        qubo.linear_term = Array1::from_vec(vec![-1.0, -1.0, -1.0, -1.0]);
        qubo.qubit_mapping = vec![
            ("a".to_string(), 0.25),
            ("b".to_string(), 0.25),
            ("c".to_string(), 0.25),
            ("d".to_string(), 0.25),
        ];
        
        let result = annealer.solve_sync(&qubo).unwrap();
        
        assert!(result.energy <= 0.0); // Should find negative energy
        assert_eq!(result.solution.len(), 4);
    }

    #[test]
    fn test_energy_calculation() {
        let q_matrix = Array2::from_shape_vec((2, 2), vec![
            1.0, -0.5,
            -0.5, 1.0,
        ]).unwrap();
        let linear = Array1::from_vec(vec![-0.1, -0.1]);
        
        // Test solution [1, 0]
        let solution = vec![1, 0];
        let energy = calculate_energy(&solution, &q_matrix, &linear);
        
        // E = Q[0,0]*1*1 + Q[0,1]*1*0 + Q[1,0]*0*1 + Q[1,1]*0*0 + linear[0]*1 + linear[1]*0
        // E = 1.0 + 0 + 0 + 0 - 0.1 + 0 = 0.9
        assert!((energy - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_energy_delta() {
        let q_matrix = Array2::from_shape_vec((3, 3), vec![
            1.0, -0.5, 0.0,
            -0.5, 1.0, -0.5,
            0.0, -0.5, 1.0,
        ]).unwrap();
        let linear = Array1::from_vec(vec![-0.1, -0.1, -0.1]);
        
        let solution = vec![1, 0, 1];
        let energy_before = calculate_energy(&solution, &q_matrix, &linear);
        
        // Flip bit 1 from 0 to 1
        let delta = calculate_energy_delta(1, 0, 1, &solution, &q_matrix, &linear);
        
        let mut new_solution = solution.clone();
        new_solution[1] = 1;
        let energy_after = calculate_energy(&new_solution, &q_matrix, &linear);
        
        assert!((energy_before + delta - energy_after).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_async_annealing() {
        let annealer = ClassicalSimulatedAnnealer::new();
        
        let mut qubo = QuboMatrix::new(2);
        qubo.matrix = Array2::from_shape_vec((2, 2), vec![1.0, -0.5, -0.5, 1.0]).unwrap();
        qubo.linear_term = Array1::from_vec(vec![-0.1, -0.1]);
        qubo.qubit_mapping = vec![("a".to_string(), 0.5), ("b".to_string(), 0.5)];
        
        let result = annealer.solve(&qubo).await.unwrap();
        
        assert!(result.iterations > 0);
        assert_eq!(result.solution.len(), 2);
    }
}
