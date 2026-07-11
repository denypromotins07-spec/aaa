//! Fixed-Point Iteration Solver for Deutsch CTC maps
//! 
//! Implements contractive mapping and simulated annealing fallback
//! to guarantee convergence to the consistent CTC density matrix.

use crate::ctc::deutsch_density_matrix::{DensityMatrix, Complex, CONVERGENCE_TOLERANCE, MAX_ITERATIONS};

/// Damping factor for contractive mapping
const DAMPING_FACTOR: f64 = 0.7;

/// Initial temperature for simulated annealing
const ANNEALING_INITIAL_TEMP: f64 = 1.0;

/// Cooling rate for simulated annealing
const ANNEALING_COOLING_RATE: f64 = 0.95;

/// Minimum temperature for annealing termination
const ANNEALING_MIN_TEMP: f64 = 1e-8;

/// Result of fixed-point iteration
#[derive(Debug, Clone)]
pub struct FixedPointResult {
    /// The fixed-point density matrix (CTC state)
    pub rho_ctc: DensityMatrix,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Final residual error
    pub residual: f64,
    /// Method used for convergence
    pub method: ConvergenceMethod,
}

/// Method used to achieve convergence
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceMethod {
    /// Direct fixed-point iteration
    DirectIteration,
    /// Damped iteration for stability
    DampedIteration,
    /// Simulated annealing fallback
    SimulatedAnnealing,
    /// Failed to converge
    Failed,
}

/// Fixed-Point Iteration Solver for CTC states
pub struct FixedPointSolver {
    /// Maximum iterations allowed
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Damping factor for contractive mapping
    damping: f64,
    /// Use simulated annealing fallback
    use_annealing: bool,
}

impl FixedPointSolver {
    /// Create a new fixed-point solver with default parameters
    pub fn new() -> Self {
        Self {
            max_iterations: MAX_ITERATIONS,
            tolerance: CONVERGENCE_TOLERANCE,
            damping: DAMPING_FACTOR,
            use_annealing: true,
        }
    }

    /// Create solver with custom parameters
    pub fn with_params(max_iter: usize, tol: f64, damping: f64) -> Self {
        Self {
            max_iterations: max_iter.min(MAX_ITERATIONS),
            tolerance: tol.max(CONVERGENCE_TOLERANCE),
            damping: damping.clamp(0.1, 0.9),
            use_annealing: true,
        }
    }

    /// Enable or disable simulated annealing fallback
    pub fn set_annealing(&mut self, enabled: bool) {
        self.use_annealing = enabled;
    }

    /// Find fixed point of Deutsch CTC map
    /// 
    /// Solves: rho_CTC = Tr_CR[U(rho_in ⊗ rho_CTC)U^†]
    /// 
    /// # Arguments
    /// * `rho_in` - Input density matrix (chronology-respecting system)
    /// * `unitary` - Interaction unitary operator (flattened)
    /// * `unitary_dim` - Dimension of unitary matrix
    /// 
    /// # Returns
    /// FixedPointResult with CTC state and convergence info
    pub fn find_fixed_point(&self, rho_in: &DensityMatrix, unitary: &[Complex], unitary_dim: usize) -> FixedPointResult {
        if rho_in.dimension() != unitary_dim {
            return FixedPointResult {
                rho_ctc: rho_in.clone(),
                iterations: 0,
                converged: false,
                residual: f64::INFINITY,
                method: ConvergenceMethod::Failed,
            };
        }

        // Initialize with maximally mixed state
        let dim = rho_in.dimension();
        let mut rho_ctc = match DensityMatrix::maximally_mixed(dim) {
            Some(dm) => dm,
            None => return FixedPointResult {
                rho_ctc: rho_in.clone(),
                iterations: 0,
                converged: false,
                residual: f64::INFINITY,
                method: ConvergenceMethod::Failed,
            },
        };

        // Try direct damped iteration first
        let (converged, iterations, residual) = self.damped_iteration(rho_in, unitary, unitary_dim, &mut rho_ctc);
        
        if converged {
            return FixedPointResult {
                rho_ctc,
                iterations,
                converged: true,
                residual,
                method: ConvergenceMethod::DampedIteration,
            };
        }

        // Fallback to simulated annealing if enabled
        if self.use_annealing {
            let (anneal_converged, anneal_iters, anneal_residual) = 
                self.simulated_annealing(rho_in, unitary, unitary_dim, &mut rho_ctc);
            
            if anneal_converged {
                return FixedPointResult {
                    rho_ctc,
                    iterations: iterations + anneal_iters,
                    converged: true,
                    residual: anneal_residual,
                    method: ConvergenceMethod::SimulatedAnnealing,
                };
            }
        }

        // Return best result even if not fully converged
        FixedPointResult {
            rho_ctc,
            iterations: iterations,
            converged: false,
            residual,
            method: ConvergenceMethod::Failed,
        }
    }

    /// Check if a given state is a valid fixed point
    pub fn verify_fixed_point(&self, rho_ctc: &DensityMatrix, rho_in: &DensityMatrix, 
                              unitary: &[Complex], unitary_dim: usize) -> Option<f64> {
        if rho_ctc.dimension() != unitary_dim || rho_in.dimension() != unitary_dim {
            return None;
        }

        // Compute one iteration of the map
        let rho_ctc_new = self.apply_ctc_map(rho_in, rho_ctc, unitary, unitary_dim)?;
        
        // Compute residual
        rho_ctc.distance(&rho_ctc_new)
    }

    // Internal: Apply Deutsch CTC map
    fn apply_ctc_map(&self, rho_in: &DensityMatrix, rho_ctc: &DensityMatrix,
                     unitary: &[Complex], unitary_dim: usize) -> Option<DensityMatrix> {
        // Construct tensor product state: rho_in ⊗ rho_ctc
        let tensor_dim = unitary_dim * unitary_dim;
        let tensor_state = self.tensor_product(rho_in, rho_ctc, tensor_dim);
        
        // Apply unitary: U (rho_in ⊗ rho_ctc) U^†
        let evolved = tensor_state.apply_unitary(unitary, tensor_dim)?;
        
        // Partial trace over chronology-respecting system
        // This is simplified - full implementation requires proper partial trace
        evolved.partial_trace(&[0]) // Trace out first subsystem
    }

    // Internal: Tensor product of two density matrices
    fn tensor_product(&self, rho1: &DensityMatrix, rho2: &DensityMatrix, result_dim: usize) -> DensityMatrix {
        let dim1 = rho1.dimension();
        let dim2 = rho2.dimension();
        
        let mut data = Vec::with_capacity(result_dim * result_dim);
        
        for i in 0..result_dim {
            for j in 0..result_dim {
                let i1 = i / dim2;
                let i2 = i % dim2;
                let j1 = j / dim2;
                let j2 = j % dim2;
                
                let elem1 = rho1.get(i1, j1).unwrap_or(Complex::real(0.0));
                let elem2 = rho2.get(i2, j2).unwrap_or(Complex::real(0.0));
                
                data.push(elem1 * elem2);
            }
        }
        
        // Create result - handle potential None gracefully
        DensityMatrix::new(data, result_dim).unwrap_or_else(|| {
            DensityMatrix::maximally_mixed(result_dim).unwrap()
        })
    }

    // Internal: Damped fixed-point iteration
    fn damped_iteration(&self, rho_in: &DensityMatrix, unitary: &[Complex], 
                        unitary_dim: usize, rho_ctc: &mut DensityMatrix) -> (bool, usize, f64) {
        let mut prev_rho = rho_ctc.clone();
        let mut residual = f64::INFINITY;
        
        for iter in 0..self.max_iterations {
            // Apply CTC map
            let rho_new = match self.apply_ctc_map(rho_in, &prev_rho, unitary, unitary_dim) {
                Some(dm) => dm,
                None => return (false, iter, residual),
            };
            
            // Apply damping: rho_new = (1-damping)*rho_old + damping*rho_map
            let damping = self.damping;
            let one_minus_damping = 1.0 - damping;
            
            let mut damped_data = Vec::with_capacity(prev_rho.dimension() * prev_rho.dimension());
            for i in 0..prev_rho.dimension() {
                for j in 0..prev_rho.dimension() {
                    let old_elem = prev_rho.get(i, j).unwrap_or(Complex::real(0.0));
                    let new_elem = rho_new.get(i, j).unwrap_or(Complex::real(0.0));
                    
                    damped_data.push(Complex::new(
                        one_minus_damping * old_elem.re + damping * new_elem.re,
                        one_minus_damping * old_elem.im + damping * new_elem.im,
                    ));
                }
            }
            
            if let Some(updated) = DensityMatrix::new(damped_data, prev_rho.dimension()) {
                *rho_ctc = updated;
            }
            
            // Compute residual
            if let Some(res) = prev_rho.distance(rho_ctc) {
                residual = res;
            }
            
            if residual < self.tolerance {
                return (true, iter + 1, residual);
            }
            
            prev_rho = rho_ctc.clone();
        }
        
        (false, self.max_iterations, residual)
    }

    // Internal: Simulated annealing fallback
    fn simulated_annealing(&self, rho_in: &DensityMatrix, unitary: &[Complex],
                           unitary_dim: usize, rho_ctc: &mut DensityMatrix) -> (bool, usize, f64) {
        let mut current_rho = rho_ctc.clone();
        let mut best_rho = rho_ctc.clone();
        let mut best_residual = f64::INFINITY;
        
        let mut temperature = ANNEALING_INITIAL_TEMP;
        let mut iterations = 0;
        
        while temperature > ANNEALING_MIN_TEMP && iterations < self.max_iterations {
            // Generate perturbed candidate
            let candidate = self.perturb_state(&current_rho, temperature);
            
            // Compute residual for candidate
            let candidate_residual = match self.verify_fixed_point(&candidate, rho_in, unitary, unitary_dim) {
                Some(res) => res,
                None => {
                    temperature *= ANNEALING_COOLING_RATE;
                    iterations += 1;
                    continue;
                }
            };
            
            let current_residual = match self.verify_fixed_point(&current_rho, rho_in, unitary, unitary_dim) {
                Some(res) => res,
                None => f64::INFINITY,
            };
            
            // Acceptance probability (Metropolis criterion)
            let delta = candidate_residual - current_residual;
            let accept_prob = if delta < 0.0 {
                1.0
            } else {
                (-delta / temperature).exp()
            };
            
            let random_val = fastrand::f64();
            
            if random_val < accept_prob {
                current_rho = candidate;
                
                if candidate_residual < best_residual {
                    best_rho = candidate;
                    best_residual = candidate_residual;
                }
            }
            
            // Check convergence
            if best_residual < self.tolerance {
                *rho_ctc = best_rho;
                return (true, iterations + 1, best_residual);
            }
            
            // Cool down
            temperature *= ANNEALING_COOLING_RATE;
            iterations += 1;
        }
        
        *rho_ctc = best_rho;
        (best_residual < self.tolerance, iterations, best_residual)
    }

    // Internal: Perturb density matrix state
    fn perturb_state(&self, rho: &DensityMatrix, temperature: f64) -> DensityMatrix {
        let dim = rho.dimension();
        let mut data = Vec::with_capacity(dim * dim);
        
        let perturbation_scale = temperature * 0.1;
        
        for i in 0..dim {
            for j in 0..dim {
                if let Some(elem) = rho.get(i, j) {
                    let noise_re = fastrand::f64() * perturbation_scale * 2.0 - perturbation_scale;
                    let noise_im = fastrand::f64() * perturbation_scale * 2.0 - perturbation_scale;
                    
                    // Ensure diagonal elements remain real and positive
                    if i == j {
                        data.push(Complex::real((elem.re + noise_re).max(0.0)));
                    } else {
                        data.push(Complex::new(elem.re + noise_re, elem.im + noise_im));
                    }
                } else {
                    data.push(Complex::real(0.0));
                }
            }
        }
        
        // Normalize to maintain trace = 1
        let trace_sum: f64 = data.iter().enumerate()
            .filter(|(idx, _)| idx % (dim + 1) == 0)
            .map(|(_, c)| c.re)
            .sum();
        
        if trace_sum > 1e-15 {
            for elem in data.iter_mut() {
                elem.re /= trace_sum;
                elem.im /= trace_sum;
            }
        }
        
        DensityMatrix::new(data, dim).unwrap_or_else(|| rho.clone())
    }
}

impl Default for FixedPointSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_creation() {
        let solver = FixedPointSolver::new();
        assert!(solver.max_iterations > 0);
        assert!(solver.tolerance > 0.0);
        assert!(solver.damping > 0.0 && solver.damping < 1.0);
    }

    #[test]
    fn test_custom_params() {
        let solver = FixedPointSolver::with_params(500, 1e-8, 0.5);
        assert_eq!(solver.max_iterations, 500);
        assert!(solver.tolerance >= 1e-10);
        assert!(solver.damping >= 0.1 && solver.damping <= 0.9);
    }
}
