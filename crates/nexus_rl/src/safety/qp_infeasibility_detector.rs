//! NEXUS-OMEGA Stage 19: QP Infeasibility Detector
//!
//! This module detects when the Quadratic Programming action projection
//! becomes infeasible, indicating that no safe action exists within the
//! current constraints. This triggers the Recovery Policy (Chapter 4).
//!
//! Detection methods:
//! 1. Iteration limit exceeded without convergence
//! 2. Constraint conflict analysis
//! 3. Feasibility margin estimation
//! 4. Early warning based on constraint proximity
//!
//! Author: NEXUS-OMEGA Architecture
//! Stage: 19 of 50

use std::time::{Duration, Instant};

/// Maximum iterations before declaring potential infeasibility
const MAX_ITERATIONS_WARNING: usize = 80;
const MAX_ITERATIONS_CRITICAL: usize = 100;

/// Status of QP feasibility
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeasibilityStatus {
    /// Problem is feasible and solved
    Feasible,
    /// Problem may be infeasible (warning)
    PotentiallyInfeasible,
    /// Problem is definitely infeasible
    Infeasible,
    /// Solver encountered numerical issues
    NumericalError,
    /// Status unknown (not yet evaluated)
    Unknown,
}

/// Detailed infeasibility analysis result
#[derive(Debug, Clone)]
pub struct InfeasibilityAnalysis {
    /// Overall feasibility status
    pub status: FeasibilityStatus,
    
    /// Which constraints are conflicting
    pub conflicting_constraints: Vec<usize>,
    
    /// Estimated minimum constraint violation
    pub min_violation: f64,
    
    /// Suggested constraint relaxation
    pub suggested_relaxation: f64,
    
    /// Time since last feasible solution
    pub time_since_feasible_secs: f64,
}

/// Detector for QP infeasibility conditions
pub struct QPInfeasibilityDetector {
    // Configuration
    warning_iteration_threshold: usize,
    critical_iteration_threshold: usize,
    
    // State tracking
    consecutive_infeasible: u32,
    last_feasible_time: Option<Instant>,
    iteration_history: Vec<usize>,
    
    // Constraint tracking
    constraint_activity_history: Vec<Vec<bool>>,
    violation_magnitudes: Vec<f64>,
}

impl QPInfeasibilityDetector {
    /// Create a new infeasibility detector
    pub fn new() -> Self {
        Self {
            warning_iteration_threshold: MAX_ITERATIONS_WARNING,
            critical_iteration_threshold: MAX_ITERATIONS_CRITICAL,
            consecutive_infeasible: 0,
            last_feasible_time: None,
            iteration_history: Vec::with_capacity(100),
            constraint_activity_history: Vec::with_capacity(100),
            violation_magnitudes: Vec::with_capacity(100),
        }
    }
    
    /// Analyze QP solver result for infeasibility
    pub fn analyze(
        &mut self,
        iterations_used: usize,
        converged: bool,
        constraint_violations: &[f64],
        active_constraints: &[bool],
    ) -> FeasibilityStatus {
        // Record history
        self.iteration_history.push(iterations_used);
        if self.iteration_history.len() > 100 {
            self.iteration_history.remove(0);
        }
        
        self.constraint_activity_history.push(active_constraints.to_vec());
        if self.constraint_activity_history.len() > 100 {
            self.constraint_activity_history.remove(0);
        }
        
        let max_violation = violation_magnitudes.iter().cloned().fold(0.0_f64, f64::max);
        self.violation_magnitudes.push(max_violation);
        if self.violation_magnitudes.len() > 100 {
            self.violation_magnitudes.remove(0);
        }
        
        // Check for convergence
        if converged {
            self.last_feasible_time = Some(Instant::now());
            self.consecutive_infeasible = 0;
            return FeasibilityStatus::Feasible;
        }
        
        // Check iteration count
        if iterations_used >= self.critical_iteration_threshold {
            self.consecutive_infeasible += 1;
            return FeasibilityStatus::Infeasible;
        }
        
        if iterations_used >= self.warning_iteration_threshold {
            self.consecutive_infeasible += 1;
            return FeasibilityStatus::PotentiallyInfeasible;
        }
        
        // Check for numerical issues
        if max_violation.is_nan() || max_violation.is_infinite() {
            return FeasibilityStatus::NumericalError;
        }
        
        // Not enough information
        FeasibilityStatus::Unknown
    }
    
    /// Perform detailed infeasibility analysis
    pub fn analyze_infeasibility(
        &self,
        constraint_bounds: &[f64],
        constraint_values: &[f64],
    ) -> InfeasibilityAnalysis {
        let n_constraints = constraint_bounds.len();
        
        // Find most violated constraints
        let mut violations: Vec<(usize, f64)> = constraint_values
            .iter()
            .zip(constraint_bounds.iter())
            .enumerate()
            .map(|(i, (&val, &bound))| (i, val - bound))
            .filter(|(_, v)| *v > 0.0)
            .collect();
        
        violations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let conflicting: Vec<usize> = violations.iter().map(|(i, _)| *i).collect();
        let min_violation = violations.first().map(|(_, v)| *v).unwrap_or(0.0);
        
        // Estimate required relaxation
        let suggested_relaxation = if !violations.is_empty() {
            violations.iter().map(|(_, v)| *v).fold(0.0_f64.max) * 1.1
        } else {
            0.0
        };
        
        // Time since last feasible
        let time_since_feasible = self.last_feasible_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(f64::INFINITY);
        
        // Determine status
        let status = if self.consecutive_infeasible >= 3 {
            FeasibilityStatus::Infeasible
        } else if self.consecutive_infeasible >= 1 {
            FeasibilityStatus::PotentiallyInfeasible
        } else if time_since_feasible > 60.0 {
            FeasibilityStatus::PotentiallyInfeasible
        } else {
            FeasibilityStatus::Feasible
        };
        
        InfeasibilityAnalysis {
            status,
            conflicting_constraints: conflicting,
            min_violation,
            suggested_relaxation,
            time_since_feasible_secs: time_since_feasible,
        }
    }
    
    /// Get early warning score for impending infeasibility
    /// 
    /// Returns value in [0, 1] where:
    /// - 0 = No risk of infeasibility
    /// - 1 = Infeasibility imminent
    pub fn get_infeasibility_risk_score(&self) -> f64 {
        let mut score = 0.0;
        
        // Factor 1: Recent iteration trends
        if self.iteration_history.len() >= 5 {
            let recent_avg: f64 = self.iteration_history.iter().rev().take(5).sum::<usize>() as f64 / 5.0;
            let older_avg: f64 = if self.iteration_history.len() > 5 {
                self.iteration_history.iter().rev().skip(5).take(5).sum::<usize>() as f64 / 5.0
            } else {
                recent_avg
            };
            
            if older_avg > 0.0 {
                let trend = recent_avg / older_avg;
                if trend > 1.5 {
                    score += 0.3; // Increasing iterations
                }
            }
        }
        
        // Factor 2: Consecutive infeasible counts
        score += (self.consecutive_infeasible as f64 * 0.1).min(0.4);
        
        // Factor 3: Violation magnitude trends
        if self.violation_magnitudes.len() >= 3 {
            let recent_violation: f64 = self.violation_magnitudes.iter().rev().take(3).sum::<f64>() / 3.0;
            if recent_violation > 0.1 {
                score += 0.3;
            }
        }
        
        score.min(1.0)
    }
    
    /// Check if constraints are becoming increasingly active
    pub fn detect_constraint_saturation(&self) -> Vec<usize> {
        if self.constraint_activity_history.len() < 10 {
            return Vec::new();
        }
        
        let n_constraints = self.constraint_activity_history[0].len();
        let mut saturated = Vec::new();
        
        for c in 0..n_constraints {
            // Count how often this constraint was active recently
            let active_count: usize = self.constraint_activity_history
                .iter()
                .rev()
                .take(10)
                .filter(|history| history.get(c).copied().unwrap_or(false))
                .count();
            
            // Constraint is saturated if active > 80% of recent iterations
            if active_count >= 8 {
                saturated.push(c);
            }
        }
        
        saturated
    }
    
    /// Reset detector state
    pub fn reset(&mut self) {
        self.consecutive_infeasible = 0;
        self.last_feasible_time = Some(Instant::now());
        self.iteration_history.clear();
        self.constraint_activity_history.clear();
        self.violation_magnitudes.clear();
    }
    
    /// Get statistics
    pub fn get_statistics(&self) -> InfeasibilityStats {
        let avg_iterations = if self.iteration_history.is_empty() {
            0.0
        } else {
            self.iteration_history.iter().sum::<usize>() as f64 / self.iteration_history.len() as f64
        };
        
        InfeasibilityStats {
            consecutive_infeasible: self.consecutive_infeasible,
            avg_iterations,
            risk_score: self.get_infeasibility_risk_score(),
            time_since_feasible: self.last_feasible_time
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(f64::INFINITY),
        }
    }
}

impl Default for QPInfeasibilityDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from infeasibility detector
#[derive(Debug, Clone)]
pub struct InfeasibilityStats {
    pub consecutive_infeasible: u32,
    pub avg_iterations: f64,
    pub risk_score: f64,
    pub time_since_feasible: f64,
}

/// Helper function to compute maximum violation
fn violation_magnitude_max<'a>(violations: impl Iterator<Item = &'a f64>) -> f64 {
    violations.cloned().fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_detect_convergence() {
        let mut detector = QPInfeasibilityDetector::new();
        
        let status = detector.analyze(
            50,      // iterations
            true,    // converged
            &[0.0],  // violations
            &[true], // active
        );
        
        assert_eq!(status, FeasibilityStatus::Feasible);
    }
    
    #[test]
    fn test_detect_infeasibility_by_iterations() {
        let mut detector = QPInfeasibilityDetector::new();
        
        let status = detector.analyze(
            100,     // max iterations
            false,   // not converged
            &[0.5],  // violations
            &[true],
        );
        
        assert_eq!(status, FeasibilityStatus::Infeasible);
    }
    
    #[test]
    fn test_risk_score_increases_with_problems() {
        let mut detector = QPInfeasibilityDetector::new();
        
        // Simulate increasing problems
        for i in 0..5 {
            detector.analyze(
                70 + i * 5,
                false,
                &[0.1 * (i + 1) as f64],
                &[true],
            );
        }
        
        let score = detector.get_infeasibility_risk_score();
        assert!(score > 0.0);
    }
    
    #[test]
    fn test_constraint_saturation_detection() {
        let mut detector = QPInfeasibilityDetector::new();
        
        // Simulate constraint always active
        for _ in 0..15 {
            detector.constraint_activity_history.push(vec![true, false, true]);
        }
        
        let saturated = detector.detect_constraint_saturation();
        assert!(saturated.contains(&0));
        assert!(saturated.contains(&2));
        assert!(!saturated.contains(&1));
    }
}
