//! Jacobian Eigenvalue Analysis for Reflexivity System Stability
//! 
//! Computes eigenvalues of the Jacobian matrix to determine stability
//! boundaries and predict bubble/crash inflection points.

use crate::reflexivity::coupled_ode_solver::{ReflexivityState, ReflexivityParameters, ReflexivityError};
use nalgebra::{Matrix2, Complex, SymmetricEigen};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EigenvalueError {
    #[error("Matrix is singular")]
    SingularMatrix,
    #[error("Numerical overflow in computation")]
    NumericalOverflow,
}

/// Result of eigenvalue analysis
#[derive(Debug, Clone)]
pub struct EigenvalueAnalysis {
    /// Dominant (largest real part) eigenvalue
    pub dominant_eigenvalue: Complex<f64>,
    /// Second eigenvalue
    pub secondary_eigenvalue: Complex<f64>,
    /// Trace of Jacobian (sum of eigenvalues)
    pub trace: f64,
    /// Determinant of Jacobian (product of eigenvalues)
    pub determinant: f64,
    /// Stability classification
    pub stability: StabilityClass,
    /// Lyapunov exponent estimate
    pub lyapunov_exponent: f64,
}

impl EigenvalueAnalysis {
    /// Check if system is stable (all eigenvalues have negative real parts)
    pub fn is_stable(&self) -> bool {
        self.dominant_eigenvalue.re < 0.0 && self.secondary_eigenvalue.re < 0.0
    }

    /// Check if system exhibits oscillatory behavior (complex eigenvalues)
    pub fn is_oscillatory(&self) -> bool {
        self.dominant_eigenvalue.im.abs() > 1e-10
            || self.secondary_eigenvalue.im.abs() > 1e-10
    }

    /// Oscillation frequency if oscillatory
    pub fn oscillation_frequency(&self) -> Option<f64> {
        if self.is_oscillatory() {
            Some(self.dominant_eigenvalue.im.abs())
        } else {
            None
        }
    }

    /// Time constant for decay/growth
    pub fn time_constant(&self) -> f64 {
        let lambda = self.dominant_eigenvalue.re;
        if lambda.abs() < 1e-15 {
            f64::INFINITY
        } else {
            1.0 / lambda.abs()
        }
    }
}

/// Classification of system stability
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilityClass {
    /// All eigenvalues have negative real parts - stable fixed point
    StableNode,
    /// Complex eigenvalues with negative real parts - stable spiral
    StableSpiral,
    /// Positive real eigenvalue - unstable node (bubble formation)
    UnstableNode,
    /// Complex eigenvalues with positive real parts - unstable spiral
    UnstableSpiral,
    /// Mixed signs - saddle point (critical transition)
    SaddlePoint,
    /// Zero real parts - center (neutral stability)
    Center,
}

impl StabilityClass {
    pub fn is_critical(&self) -> bool {
        matches!(self, Self::SaddlePoint | Self::UnstableNode | Self::UnstableSpiral)
    }

    pub fn indicates_bubble(&self) -> bool {
        matches!(self, Self::UnstableNode | Self::UnstableSpiral)
    }

    pub fn indicates_crash_risk(&self) -> bool {
        matches!(self, Self::SaddlePoint)
    }
}

/// Jacobian eigenvalue analyzer
pub struct JacobianEigenAnalyzer {
    epsilon: f64,
}

impl JacobianEigenAnalyzer {
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }

    /// Compute Jacobian at a given state
    pub fn compute_jacobian(&self, state: &ReflexivityState, params: &ReflexivityParameters) -> Matrix2<f64> {
        let i = state.sir_state.infected.max(0.0);
        let s = state.sir_state.susceptible.max(0.0);
        let p = state.price_deviation;
        let n = params.sir_params.total_population;

        // Effective beta with feedback
        let beta_eff = params.sir_params.beta * (1.0 + params.price_feedback_beta * p.max(0.0));

        // Jacobian elements for [I, P] subsystem
        let j00 = beta_eff * s / n - params.sir_params.gamma;
        let j01 = params.sir_params.beta * params.price_feedback_beta * s * i / n;
        let j10 = params.narrative_price_impact;
        let j11 = -params.price_mean_reversion;

        Matrix2::new(j00, j01, j10, j11)
    }

    /// Perform full eigenvalue analysis
    pub fn analyze(&self, jacobian: &Matrix2<f64>) -> Result<EigenvalueAnalysis, EigenvalueError> {
        let trace = jacobian.trace();
        let det = jacobian.determinant();

        if det.abs() < self.epsilon {
            return Err(EigenvalueError::SingularMatrix);
        }

        // Characteristic polynomial: λ² - τλ + Δ = 0
        let discriminant = trace * trace - 4.0 * det;

        let (lambda1, lambda2) = if discriminant >= 0.0 {
            // Real eigenvalues
            let sqrt_disc = discriminant.sqrt();
            (
                Complex::new((trace + sqrt_disc) / 2.0, 0.0),
                Complex::new((trace - sqrt_disc) / 2.0, 0.0),
            )
        } else {
            // Complex conjugate eigenvalues
            let imag = (-discriminant).sqrt() / 2.0;
            let real = trace / 2.0;
            (Complex::new(real, imag), Complex::new(real, -imag))
        };

        // Check for numerical issues
        if !lambda1.re.is_finite() || !lambda1.im.is_finite() {
            return Err(EigenvalueError::NumericalOverflow);
        }

        // Determine dominant eigenvalue
        let dominant = if lambda1.re >= lambda2.re { lambda1 } else { lambda2 };
        let secondary = if lambda1.re >= lambda2.re { lambda2 } else { lambda1 };

        // Classify stability
        let stability = self.classify_stability(&lambda1, &lambda2);

        // Lyapunov exponent (dominant real part)
        let lyapunov = dominant.re;

        Ok(EigenvalueAnalysis {
            dominant_eigenvalue: dominant,
            secondary_eigenvalue: secondary,
            trace,
            determinant: det,
            stability,
            lyapunov_exponent: lyapunov,
        })
    }

    fn classify_stability(&self, l1: &Complex<f64>, l2: &Complex<f64>) -> StabilityClass {
        let re1 = l1.re;
        let re2 = l2.re;
        let im1 = l1.im.abs();
        let im2 = l2.im.abs();

        let has_positive = re1 > self.epsilon || re2 > self.epsilon;
        let has_negative = re1 < -self.epsilon && re2 < -self.epsilon;
        let has_complex = im1 > self.epsilon || im2 > self.epsilon;

        match (has_positive, has_negative, has_complex) {
            (false, true, false) => StabilityClass::StableNode,
            (false, true, true) => StabilityClass::StableSpiral,
            (true, false, false) => StabilityClass::UnstableNode,
            (true, false, true) => StabilityClass::UnstableSpiral,
            (true, false, _) if (re1 * re2) < 0.0 => StabilityClass::SaddlePoint,
            (false, false, _) => StabilityClass::Center,
            _ if (re1 * re2) < 0.0 => StabilityClass::SaddlePoint,
            _ => StabilityClass::UnstableNode,
        }
    }

    /// Analyze state directly
    pub fn analyze_state(
        &self,
        state: &ReflexivityState,
        params: &ReflexivityParameters,
    ) -> Result<EigenvalueAnalysis, EigenvalueError> {
        let j = self.compute_jacobian(state, params);
        self.analyze(&j)
    }

    /// Find critical parameter values where stability changes (bifurcation points)
    pub fn find_bifurcation_point(
        &self,
        base_params: &ReflexivityParameters,
        param_name: &str,
        range: (f64, f64),
        steps: usize,
    ) -> Option<(f64, EigenvalueAnalysis)> {
        let mut last_analysis: Option<EigenvalueAnalysis> = None;

        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let param_value = range.0 + t * (range.1 - range.0);

            let mut params = base_params.clone();
            
            // Modify parameter based on name
            match param_name {
                "price_feedback_beta" => params.price_feedback_beta = param_value,
                "narrative_price_impact" => params.narrative_price_impact = param_value,
                "price_mean_reversion" => params.price_mean_reversion = param_value,
                "beta" => {
                    params.sir_params = SirParameters::new(param_value, params.sir_params.gamma).ok()?;
                }
                _ => continue,
            }

            // Create a reference state for analysis
            let state = ReflexivityState::new(
                SirState::new(0.5, 0.1, 0.4).ok()?,
                0.1,
                0.8,
            );

            if let Ok(analysis) = self.analyze_state(&state, &params) {
                // Check for stability change
                if let Some(prev) = &last_analysis {
                    if prev.stability != analysis.stability {
                        return Some((param_value, analysis));
                    }
                }
                last_analysis = Some(analysis);
            }
        }

        None
    }
}

// Re-export types needed for bifurcation analysis
use crate::epidemiology::financial_sir_ode::{SirState, SirParameters};

/// Streaming eigenvalue tracker for real-time stability monitoring
pub struct StreamingEigenTracker {
    analyzer: JacobianEigenAnalyzer,
    history: Vec<EigenvalueAnalysis>,
    max_history: usize,
}

impl StreamingEigenTracker {
    pub fn new(max_history: usize) -> Self {
        Self {
            analyzer: JacobianEigenAnalyzer::new(1e-10),
            history: Vec::new(),
            max_history,
        }
    }

    /// Update with new state and track eigenvalue evolution
    pub fn update(&mut self, state: &ReflexivityState, params: &ReflexivityParameters) -> Option<EigenvalueAnalysis> {
        match self.analyzer.analyze_state(state, params) {
            Ok(analysis) => {
                self.history.push(analysis.clone());
                if self.history.len() > self.max_history {
                    self.history.remove(0);
                }
                Some(analysis)
            }
            Err(_) => None,
        }
    }

    /// Check if stability class has changed recently
    pub fn detect_stability_transition(&self) -> Option<(StabilityClass, StabilityClass)> {
        if self.history.len() < 2 {
            return None;
        }

        let current = self.history.last()?.stability;
        let previous = self.history[self.history.len() - 2].stability;

        if current != previous {
            Some((previous, current))
        } else {
            None
        }
    }

    /// Get trend in Lyapunov exponent
    pub fn lyapunov_trend(&self) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }

        let recent: Vec<f64> = self.history.iter().map(|a| a.lyapunov_exponent).collect();
        let slope = (recent.last()? - recent.first()?) / (recent.len() as f64 - 1.0);
        Some(slope)
    }

    /// Predict time to instability based on Lyapunov trend
    pub fn time_to_instability(&self) -> Option<f64> {
        let current_lyap = self.history.last()?.lyapunov_exponent;
        let trend = self.lyapunov_trend()?;

        if trend <= 1e-15 {
            return None; // Not trending toward instability
        }

        // Time until Lyapunov exponent crosses zero
        let time = -current_lyap / trend;
        if time > 0.0 {
            Some(time)
        } else {
            None // Already unstable or trend away from instability
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eigenvalue_computation() {
        let analyzer = JacobianEigenAnalyzer::new(1e-10);
        
        // Simple stable matrix
        let j = Matrix2::new(-1.0, 0.0, 0.0, -2.0);
        let analysis = analyzer.analyze(&j).unwrap();
        
        assert!(analysis.is_stable());
        assert_eq!(analysis.stability, StabilityClass::StableNode);
    }

    #[test]
    fn test_oscillatory_detection() {
        let analyzer = JacobianEigenAnalyzer::new(1e-10);
        
        // Matrix with complex eigenvalues
        let j = Matrix2::new(-0.5, 1.0, -1.0, -0.5);
        let analysis = analyzer.analyze(&j).unwrap();
        
        assert!(analysis.is_oscillatory());
        assert!(analysis.oscillation_frequency().is_some());
    }

    #[test]
    fn test_unstable_detection() {
        let analyzer = JacobianEigenAnalyzer::new(1e-10);
        
        // Unstable matrix
        let j = Matrix2::new(1.0, 0.0, 0.0, 0.5);
        let analysis = analyzer.analyze(&j).unwrap();
        
        assert!(!analysis.is_stable());
        assert!(analysis.stability.indicates_bubble());
    }
}
