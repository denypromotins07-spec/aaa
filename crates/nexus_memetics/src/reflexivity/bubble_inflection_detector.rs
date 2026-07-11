//! Bubble Inflection Detector using Lyapunov Analysis
//! 
//! Detects the exact moment when a reflexivity-driven bubble becomes unstable
//! and predicts the subsequent crash timing.

use crate::reflexivity::coupled_ode_solver::{
    ReflexivityState, ReflexivityParameters, CoupledODESolver, CrashPrediction,
};
use crate::reflexivity::jacobian_eigenvalue::{
    JacobianEigenAnalyzer, EigenvalueAnalysis, StabilityClass,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BubbleDetectionError {
    #[error("Trajectory too short for analysis")]
    TrajectoryTooShort,
    #[error("No inflection point found in trajectory")]
    NoInflectionFound,
    #[error("Numerical instability in detection")]
    NumericalInstability,
}

/// Detailed bubble phase classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BubblePhase {
    /// Early growth phase, stable
    Accumulation,
    /// Accelerating growth, approaching criticality
    Expansion,
    /// Critical instability reached, peak imminent
    Critical,
    /// Post-peak decline beginning
    Distribution,
    /// Rapid collapse underway
    Crash,
    /// Bottom formation
    Capitulation,
}

impl BubblePhase {
    pub fn from_stability(stability: &StabilityClass) -> Self {
        match stability {
            StabilityClass::StableNode => Self::Accumulation,
            StabilityClass::StableSpiral => Self::Expansion,
            StabilityClass::UnstableNode => Self::Critical,
            StabilityClass::UnstableSpiral => Self::Distribution,
            StabilityClass::SaddlePoint => Self::Crash,
            StabilityClass::Center => Self::Capitulation,
        }
    }

    pub fn is_dangerous(&self) -> bool {
        matches!(self, Self::Critical | Self::Distribution | Self::Crash)
    }

    pub fn urgency_score(&self) -> f64 {
        match self {
            Self::Accumulation => 0.1,
            Self::Expansion => 0.3,
            Self::Critical => 0.7,
            Self::Distribution => 0.85,
            Self::Crash => 1.0,
            Self::Capitulation => 0.5,
        }
    }
}

/// Bubble inflection signal with actionable intelligence
#[derive(Debug, Clone)]
pub struct BubbleInflectionSignal {
    /// Timestamp of signal
    pub timestamp: f64,
    /// Current bubble phase
    pub phase: BubblePhase,
    /// Confidence in detection [0, 1]
    pub confidence: f64,
    /// Estimated time to peak (if not yet peaked)
    pub time_to_peak: Option<f64>,
    /// Estimated time to crash (if past peak)
    pub time_to_crash: Option<f64>,
    /// Price deviation at signal
    pub price_deviation: f64,
    /// Recommended action
    pub action: BubbleAction,
}

/// Action recommendation for bubble trading
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BubbleAction {
    /// Continue holding, no action needed
    Hold,
    /// Begin reducing long exposure
    TrimLongs,
    /// Add hedges (puts, short futures)
    Hedge,
    /// Exit all longs immediately
    ExitLongs,
    /// Initiate short position
    Short,
    /// Cover shorts (bottom fishing)
    CoverShorts,
}

/// Main bubble inflection detector
pub struct BubbleInflectionDetector {
    eigen_analyzer: JacobianEigenAnalyzer,
    solver: CoupledODESolver,
    /// Threshold for "critical" classification
    critical_threshold: f64,
}

impl BubbleInflectionDetector {
    pub fn new(critical_threshold: f64) -> Self {
        Self {
            eigen_analyzer: JacobianEigenAnalyzer::new(1e-10),
            solver: CoupledODESolver::new(50, 1e-10),
            critical_threshold,
        }
    }

    /// Analyze current state for bubble conditions
    pub fn analyze_state(
        &self,
        state: &ReflexivityState,
        params: &ReflexivityParameters,
    ) -> Result<BubbleInflectionSignal, BubbleDetectionError> {
        let analysis = self.eigen_analyzer
            .analyze_state(state, params)
            .map_err(|_| BubbleDetectionError::NumericalInstability)?;

        let phase = BubblePhase::from_stability(&analysis.stability);
        
        // Calculate confidence based on distance from stability boundary
        let lyap = analysis.lyapunov_exponent;
        let confidence = if lyap.abs() < 0.01 {
            0.5 // Near boundary, uncertain
        } else if lyap > 0.0 {
            (lyap / self.critical_threshold).min(1.0)
        } else {
            ((-lyap) / self.critical_threshold).min(1.0)
        };

        // Estimate time scales
        let time_constant = analysis.time_constant();
        let time_to_peak = if phase == BubblePhase::Critical || phase == BubblePhase::Expansion {
            Some(time_constant * 2.0) // Rough estimate
        } else {
            None
        };

        let time_to_crash = if phase == BubblePhase::Distribution || phase == BubblePhase::Crash {
            Some(time_constant)
        } else {
            None
        };

        // Determine action
        let action = match phase {
            BubblePhase::Accumulation => BubbleAction::Hold,
            BubblePhase::Expansion => BubbleAction::TrimLongs,
            BubblePhase::Critical => BubbleAction::Hedge,
            BubblePhase::Distribution => BubbleAction::ExitLongs,
            BubblePhase::Crash => BubbleAction::Short,
            BubblePhase::Capitulation => BubbleAction::CoverShorts,
        };

        Ok(BubbleInflectionSignal {
            timestamp: 0.0, // Would be set by caller
            phase,
            confidence,
            time_to_peak,
            time_to_crash,
            price_deviation: state.price_deviation,
            action,
        })
    }

    /// Analyze full trajectory to find inflection point
    pub fn find_inflection_in_trajectory(
        &self,
        trajectory: &[(f64, ReflexivityState)],
        params: &ReflexivityParameters,
    ) -> Result<(f64, ReflexivityState, BubblePhase), BubbleDetectionError> {
        if trajectory.len() < 2 {
            return Err(BubbleDetectionError::TrajectoryTooShort);
        }

        let mut last_phase: Option<BubblePhase> = None;

        for (t, state) in trajectory.iter() {
            if let Ok(analysis) = self.eigen_analyzer.analyze_state(state, params) {
                let current_phase = BubblePhase::from_stability(&analysis.stability);
                
                // Detect phase transition
                if let Some(prev) = last_phase {
                    if prev != current_phase && current_phase.is_dangerous() {
                        return Ok((*t, state.clone(), current_phase));
                    }
                }
                last_phase = Some(current_phase);
            }
        }

        Err(BubbleDetectionError::NoInflectionFound)
    }

    /// Predict crash timing from trajectory
    pub fn predict_crash(
        &self,
        trajectory: &[(f64, ReflexivityState)],
        params: &ReflexivityParameters,
    ) -> Option<EnhancedCrashPrediction> {
        if trajectory.is_empty() {
            return None;
        }

        // Find peak
        let peak_idx = trajectory
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.1.price_deviation.partial_cmp(&b.1.price_deviation).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)?;

        let peak_time = trajectory[peak_idx].0;
        let peak_value = trajectory[peak_idx].1.price_deviation;

        // Check if we're past peak
        let current_time = trajectory.last()?.0;
        let is_past_peak = current_time > peak_time;

        // Analyze post-peak dynamics
        let post_peak_states: Vec<_> = trajectory[peak_idx..].to_vec();
        let crash_rate = if post_peak_states.len() >= 2 {
            let first_price = post_peak_states.first()?.1.price_deviation;
            let last_price = post_peak_states.last()?.1.price_deviation;
            let dt = post_peak_states.last()?.0 - post_peak_states.first()?.0;
            if dt > 0.0 {
                (first_price - last_price) / dt
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Estimate crash completion time
        let crash_completion = if crash_rate > 0.0 && is_past_peak {
            Some(peak_value / crash_rate)
        } else {
            None
        };

        Some(EnhancedCrashPrediction {
            peak_time,
            peak_value,
            is_past_peak,
            crash_rate,
            estimated_completion: crash_completion,
            confidence: if is_past_peak { 0.8 } else { 0.4 },
        })
    }
}

/// Enhanced crash prediction with rate information
#[derive(Debug, Clone)]
pub struct EnhancedCrashPrediction {
    pub peak_time: f64,
    pub peak_value: f64,
    pub is_past_peak: bool,
    pub crash_rate: f64,
    pub estimated_completion: Option<f64>,
    pub confidence: f64,
}

impl EnhancedCrashPrediction {
    /// Percentage of crash completed
    pub fn completion_percentage(&self, current_time: f64) -> Option<f64> {
        if !self.is_past_peak || self.estimated_completion.is_none() {
            return None;
        }
        
        let elapsed = current_time - self.peak_time;
        let total = self.estimated_completion?;
        
        Some((elapsed / total * 100.0).clamp(0.0, 100.0))
    }

    /// Time remaining until crash bottom
    pub fn time_remaining(&self, current_time: f64) -> Option<f64> {
        let completion = self.completion_percentage(current_time)?;
        if completion >= 100.0 {
            return Some(0.0);
        }
        
        let total = self.estimated_completion?;
        let elapsed = current_time - self.peak_time;
        
        Some((total - elapsed).max(0.0))
    }
}

/// Multi-asset bubble monitor
pub struct MultiAssetBubbleMonitor {
    detector: BubbleInflectionDetector,
    assets: std::collections::HashMap<String, ReflexivityState>,
}

impl MultiAssetBubbleMonitor {
    pub fn new(critical_threshold: f64) -> Self {
        Self {
            detector: BubbleInflectionDetector::new(critical_threshold),
            assets: std::collections::HashMap::new(),
        }
    }

    /// Register an asset for monitoring
    pub fn register_asset(&mut self, name: String, initial_state: ReflexivityState) {
        self.assets.insert(name, initial_state);
    }

    /// Update state for an asset
    pub fn update_asset(&mut self, name: &str, state: ReflexivityState) {
        self.assets.insert(name.to_string(), state);
    }

    /// Scan all assets for bubble risks
    pub fn scan_risks(
        &self,
        params: &ReflexivityParameters,
    ) -> Vec<(String, BubbleInflectionSignal)> {
        let mut risks: Vec<_> = self
            .assets
            .iter()
            .filter_map(|(name, state)| {
                self.detector.analyze_state(state, params).ok().map(|signal| {
                    (name.clone(), signal)
                })
            })
            .collect();

        // Sort by urgency
        risks.sort_by(|a, b| {
            b.1.confidence.partial_cmp(&a.1.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });

        risks
    }

    /// Get most dangerous bubble
    pub fn most_dangerous(&self, params: &ReflexivityParameters) -> Option<(String, BubbleInflectionSignal)> {
        self.scan_risks(params)
            .into_iter()
            .find(|(_, signal)| signal.phase.is_dangerous())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epidemiology::financial_sir_ode::SirState;

    #[test]
    fn test_bubble_phase_classification() {
        assert_eq!(BubblePhase::from_stability(&StabilityClass::StableNode), BubblePhase::Accumulation);
        assert_eq!(BubblePhase::from_stability(&StabilityClass::UnstableNode), BubblePhase::Critical);
        assert_eq!(BubblePhase::from_stability(&StabilityClass::SaddlePoint), BubblePhase::Crash);
    }

    #[test]
    fn test_inflection_detection() {
        let detector = BubbleInflectionDetector::new(0.5);
        
        let params = ReflexivityParameters::new(0.5, 0.1, 0.3, 0.2, 0.1, 0.05).unwrap();
        let state = ReflexivityState::new(
            SirState::new(0.8, 0.15, 0.05).unwrap(),
            0.5,
            0.7,
        );

        let signal = detector.analyze_state(&state, &params);
        assert!(signal.is_ok());
    }
}
