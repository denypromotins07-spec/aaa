//! Ricci Flow Evolution for Semantic Manifold Dynamics
//! 
//! Implements Hamilton's Ricci flow equation to track how narrative curvature
//! evolves over time, detecting critical singularities that indicate paradigm shifts.

use crate::topology::semantic_manifold_curvature::{ManifoldConfig, CurvatureEstimate, ManifoldPoint};
use nalgebra::{DMatrix, DVector};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RicciFlowError {
    #[error("Numerical instability in flow integration")]
    NumericalInstability,
    #[error("Curvature singularity detected at step {step}")]
    SingularityDetected { step: usize },
    #[error("Time step too large for stable integration")]
    TimeStepTooLarge,
}

/// Ricci flow evolution state
#[derive(Clone, Debug)]
pub struct RicciFlowState {
    pub time: f64,
    pub curvature: DVector<f64>,
    pub metric_trace: f64,
    /// Indicates if a singularity was approached
    pub near_singularity: bool,
}

/// Ricci flow integrator with adaptive stepping
pub struct RicciFlowIntegrator {
    config: ManifoldConfig,
    max_time: f64,
    epsilon: f64,
}

impl RicciFlowIntegrator {
    pub fn new(config: ManifoldConfig, max_time: f64) -> Self {
        Self {
            config,
            max_time,
            epsilon: config.epsilon,
        }
    }

    /// Integrate Ricci flow: ∂g/∂t = -2Ric
    /// where g is the metric and Ric is the Ricci curvature tensor
    pub fn integrate(
        &self,
        initial_curvature: &DVector<f64>,
        dt: f64,
    ) -> Result<Vec<RicciFlowState>, RicciFlowError> {
        if dt <= 0.0 || dt > 0.1 {
            return Err(RicciFlowError::TimeStepTooLarge);
        }

        let mut states = Vec::new();
        let mut current_curvature = initial_curvature.clone();
        let mut time = 0.0;
        let mut metric_trace = initial_curvature.norm();

        states.push(RicciFlowState {
            time,
            curvature: current_curvature.clone(),
            metric_trace,
            near_singularity: false,
        });

        while time < self.max_time {
            // Ricci flow ODE: dK/dt = ΔK + 2K² (simplified scalar form)
            // Using finite difference approximation
            let laplacian = self.approximate_laplacian(&current_curvature);
            let quadratic = self.elementwise_square(&current_curvature);
            
            let dcurvature = &laplacian + 2.0 * &quadratic;

            // Forward Euler step (use RK4 for production)
            let new_curvature = &current_curvature + dt * &dcurvature;

            // Check for singularity formation (curvature → ∞)
            let max_curv = new_curvature.iter().fold(0.0_f64, f64::max);
            if max_curv > 1e10 {
                return Err(RicciFlowError::SingularityDetected {
                    step: states.len(),
                });
            }

            // Check for numerical instability
            if !new_curvature.iter().all(|&x| x.is_finite()) {
                return Err(RicciFlowError::NumericalInstability);
            }

            current_curvature = new_curvature;
            time += dt;
            metric_trace = current_curvature.norm();

            let near_singularity = max_curv > 1e6;

            states.push(RicciFlowState {
                time,
                curvature: current_curvature.clone(),
                metric_trace,
                near_singularity,
            });
        }

        Ok(states)
    }

    fn approximate_laplacian(&self, curvature: &DVector<f64>) -> DVector<f64> {
        // Simple 1D Laplacian approximation for temporal evolution
        let n = curvature.len();
        let mut laplacian = DVector::zeros(n);

        for i in 1..n - 1 {
            laplacian[i] = curvature[i - 1] - 2.0 * curvature[i] + curvature[i + 1];
        }

        // Boundary conditions (Neumann: zero derivative)
        if n > 1 {
            laplacian[0] = curvature[1] - curvature[0];
            laplacian[n - 1] = curvature[n - 1] - curvature[n - 2];
        }

        laplacian
    }

    fn elementwise_square(&self, v: &DVector<f64>) -> DVector<f64> {
        v.map(|x| x * x)
    }

    /// Detect if flow approaches a singularity (paradigm shift indicator)
    pub fn detect_singularity_approach(&self, states: &[RicciFlowState]) -> Option<usize> {
        states.iter().position(|s| s.near_singularity)
    }

    /// Compute normalized curvature volume (monotonic under Ricci flow)
    pub fn compute_normalized_volume(&self, states: &[RicciFlowState]) -> Vec<f64> {
        states
            .iter()
            .map(|s| {
                let vol = s.curvature.iter().product::<f64>();
                vol.abs().ln().clamp(-1e10, 1e10)
            })
            .collect()
    }
}

/// Narrative paradigm shift detector using Ricci flow singularities
pub struct ParadigmShiftDetector {
    integrator: RicciFlowIntegrator,
    singularity_threshold: f64,
    history_window: usize,
}

impl ParadigmShiftDetector {
    pub fn new(config: ManifoldConfig, max_flow_time: f64) -> Self {
        Self {
            integrator: RicciFlowIntegrator::new(config, max_flow_time),
            singularity_threshold: 1e5,
            history_window: 50,
        }
    }

    pub fn with_singularity_threshold(mut self, threshold: f64) -> Self {
        self.singularity_threshold = threshold;
        self
    }

    /// Analyze curvature time series for paradigm shifts
    pub fn analyze(&self, curvature_history: &[CurvatureEstimate]) -> ParadigmShiftAnalysis {
        if curvature_history.is_empty() {
            return ParadigmShiftAnalysis::no_data();
        }

        // Extract curvature scalars
        let curvatures: DVector<f64> = DVector::from_vec(
            curvature_history.iter().map(|c| c.ricci_scalar).collect()
        );

        // Run Ricci flow
        let dt = 0.01;
        let flow_states = match self.integrator.integrate(&curvatures, dt) {
            Ok(states) => states,
            Err(RicciFlowError::SingularityDetected { step }) => {
                return ParadigmShiftAnalysis {
                    shift_detected: true,
                    confidence: 1.0,
                    singularity_step: Some(step),
                    shift_magnitude: f64::INFINITY,
                    recommended_action: ShiftAction::ImmediateExit,
                };
            }
            Err(_) => {
                return ParadigmShiftAnalysis {
                    shift_detected: true,
                    confidence: 0.8,
                    singularity_step: None,
                    shift_magnitude: f64::NAN,
                    recommended_action: ShiftAction::ReduceExposure,
                };
            }
        };

        // Analyze flow results
        let max_curvature = flow_states
            .iter()
            .map(|s| s.curvature.iter().fold(0.0_f64, f64::max))
            .fold(0.0_f64, f64::max);

        let singularity_approach = self.integrator.detect_singularity_approach(&flow_states);
        
        let shift_detected = max_curvature > self.singularity_threshold
            || singularity_approach.is_some();

        let confidence = if shift_detected {
            (max_curvature / self.singularity_threshold).min(1.0)
        } else {
            0.0
        };

        let action = if max_curvature > self.singularity_threshold * 10.0 {
            ShiftAction::ImmediateExit
        } else if max_curvature > self.singularity_threshold {
            ShiftAction::ReduceExposure
        } else if max_curvature > self.singularity_threshold * 0.5 {
            ShiftAction::Hedge
        } else {
            ShiftAction::Hold
        };

        ParadigmShiftAnalysis {
            shift_detected,
            confidence,
            singularity_step: singularity_approach,
            shift_magnitude: max_curvature,
            recommended_action: action,
        }
    }
}

/// Analysis result for paradigm shift detection
#[derive(Debug, Clone)]
pub struct ParadigmShiftAnalysis {
    pub shift_detected: bool,
    pub confidence: f64,
    pub singularity_step: Option<usize>,
    pub shift_magnitude: f64,
    pub recommended_action: ShiftAction,
}

impl ParadigmShiftAnalysis {
    fn no_data() -> Self {
        Self {
            shift_detected: false,
            confidence: 0.0,
            singularity_step: None,
            shift_magnitude: 0.0,
            recommended_action: ShiftAction::Hold,
        }
    }
}

/// Recommended trading action based on paradigm shift analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftAction {
    /// No action needed, regime stable
    Hold,
    /// Add hedges against regime change
    Hedge,
    /// Reduce exposure to narrative-sensitive assets
    ReduceExposure,
    /// Exit positions immediately
    ImmediateExit,
}

/// Batch analyzer for multiple narratives
pub struct MultiNarrativeAnalyzer {
    detector: ParadigmShiftDetector,
}

impl MultiNarrativeAnalyzer {
    pub fn new(config: ManifoldConfig) -> Self {
        Self {
            detector: ParadigmShiftDetector::new(config, 1.0),
        }
    }

    /// Analyze multiple narratives and rank by shift probability
    pub fn rank_by_shift_risk(
        &self,
        narratives: Vec<(String, Vec<CurvatureEstimate>)>,
    ) -> Vec<(String, ParadigmShiftAnalysis)> {
        let mut results: Vec<_> = narratives
            .into_iter()
            .map(|(name, history)| {
                let analysis = self.detector.analyze(&history);
                (name, analysis)
            })
            .collect();

        // Sort by confidence * magnitude (highest risk first)
        results.sort_by(|a, b| {
            let score_a = a.1.confidence * a.1.shift_magnitude;
            let score_b = b.1.confidence * b.1.shift_magnitude;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::semantic_manifold_curvature::CurvatureEstimate;

    #[test]
    fn test_ricci_flow_integration() {
        let config = ManifoldConfig::default();
        let integrator = RicciFlowIntegrator::new(config, 0.5);
        
        let initial = DVector::from_vec(vec![0.1, 0.2, 0.15, 0.1]);
        let states = integrator.integrate(&initial, 0.01);
        
        assert!(states.is_ok());
        assert!(!states.unwrap().is_empty());
    }

    #[test]
    fn test_paradigm_shift_detection() {
        let config = ManifoldConfig::default();
        let detector = ParadigmShiftDetector::new(config, 0.5);

        // Create curvature history with increasing magnitude
        let history: Vec<CurvatureEstimate> = (0..30)
            .map(|i| CurvatureEstimate {
                point_id: i,
                ricci_scalar: -0.1 * (i as f64),
                sectional_curvatures: vec![],
                confidence: 0.9,
            })
            .collect();

        let analysis = detector.analyze(&history);
        
        // Should detect shift due to rapidly decreasing curvature
        assert!(analysis.shift_detected || analysis.confidence > 0.0);
    }
}
