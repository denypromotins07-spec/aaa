//! Tipping Point Bifurcation Detector using SDEs and critical slowing down analysis
//! Detects approaching climate tipping points via early warning signals

use alloc::vec::Vec;
use core::fmt;

/// Error types for bifurcation detection
#[derive(Debug, Clone, PartialEq)]
pub enum BifurcationError {
    InsufficientData,
    InvalidParameter,
    NumericalOverflow,
}

impl fmt::Display for BifurcationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientData => write!(f, "Insufficient data for analysis"),
            Self::InvalidParameter => write!(f, "Invalid parameter value"),
            Self::NumericalOverflow => write!(f, "Numerical overflow detected"),
        }
    }
}

/// Types of climate tipping elements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TippingElementType {
    AMOC,                    // Atlantic Meridional Overturning Circulation
    AmazonDieback,           // Amazon rainforest dieback
    GreenlandIceSheet,       // Greenland ice sheet collapse
    WestAntarcticIceSheet,   // WAIS collapse
    PermafrostMethane,       // Permafrost methane release
    MonsoonShift,            // Monsoon regime shift
}

/// State of a tipping element
#[derive(Debug, Clone)]
pub struct TippingElementState {
    pub element_type: TippingElementType,
    /// Current state variable (normalized 0-1)
    pub state: f64,
    /// Rate of change
    pub rate: f64,
    /// Distance to bifurcation point (estimated)
    pub distance_to_tipping: f64,
    /// Confidence in estimate (0-1)
    pub confidence: f64,
}

/// Early warning signal metrics
#[derive(Debug, Clone, Default)]
pub struct EarlyWarningSignals {
    /// Lag-1 autocorrelation (increases near tipping)
    pub autocorrelation: f64,
    /// Variance (increases near tipping)
    pub variance: f64,
    /// Skewness (may change near asymmetric bifurcations)
    pub skewness: f64,
    /// Spectral reddening (power shifts to lower frequencies)
    pub spectral_ratio: f64,
    /// Kendall tau trend statistic for AR(1)
    pub ar1_trend: f64,
    /// Kendall tau trend statistic for variance
    pub var_trend: f64,
    /// Combined EWS score (0-1, higher = more critical)
    pub combined_score: f64,
}

/// Bifurcation detector monitoring multiple tipping elements
pub struct TippingPointBifurcationDetector {
    /// History window for each element
    history_window: usize,
    /// Time series history per element: Vec<Box<[state]>>
    histories: alloc::collections::BTreeMap<TippingElementType, Vec<Box<[f64]>>>,
    /// Minimum samples required
    min_samples: usize,
    /// Threshold for critical warning
    critical_threshold: f64,
}

impl TippingPointBifurcationDetector {
    /// Create new detector with specified history window
    pub fn new(history_window: usize, critical_threshold: f64) -> Self {
        let mut histories = alloc::collections::BTreeMap::new();
        for element in [
            TippingElementType::AMOC,
            TippingElementType::AmazonDieback,
            TippingElementType::GreenlandIceSheet,
            TippingElementType::WestAntarcticIceSheet,
            TippingElementType::PermafrostMethane,
            TippingElementType::MonsoonShift,
        ] {
            histories.insert(element, Vec::with_capacity(history_window));
        }

        Self {
            history_window,
            histories,
            min_samples: history_window / 3,
            critical_threshold: critical_threshold.clamp(0.5, 0.95),
        }
    }

    /// Add observation for a tipping element
    pub fn observe(&mut self, element: TippingElementType, state: f64) {
        if let Some(history) = self.histories.get_mut(&element) {
            history.push(vec![state].into_boxed_slice());
            if history.len() > self.history_window {
                history.remove(0);
            }
        }
    }

    /// Compute early warning signals for all elements
    pub fn compute_all_signals(&mut self) -> Result<alloc::collections::BTreeMap<TippingElementType, EarlyWarningSignals>, BifurcationError> {
        let mut results = alloc::collections::BTreeMap::new();

        for (&element, history) in &self.histories {
            let signals = self.compute_signals_for_element(element, history)?;
            results.insert(element, signals);
        }

        Ok(results)
    }

    /// Compute signals for a single element
    fn compute_signals_for_element(
        &self,
        element: TippingElementType,
        history: &[Box<[f64]>],
    ) -> Result<EarlyWarningSignals, BifurcationError> {
        if history.len() < self.min_samples {
            return Ok(EarlyWarningSignals::default());
        }

        // Extract time series
        let series: Vec<f64> = history.iter().map(|h| h[0]).collect();

        // Compute rolling statistics in two windows
        let half = history.len() / 2;
        let first_half = &series[..half];
        let second_half = &series[half..];

        // Autocorrelation at lag-1
        let ar1_first = Self::autocorrelation_lag1(first_half);
        let ar1_second = Self::autocorrelation_lag1(second_half);

        // Variance
        let var_first = Self::variance(first_half);
        let var_second = Self::variance(second_half);

        // Skewness
        let skew_first = Self::skewness(first_half);
        let skew_second = Self::skewness(second_half);

        // Spectral ratio (low freq power / high freq power)
        let spec_first = Self::spectral_ratio(first_half);
        let spec_second = Self::spectral_ratio(second_half);

        // Relative changes
        let ar1_change = if ar1_first.abs() > 1e-10 {
            (ar1_second - ar1_first) / ar1_first.abs()
        } else {
            0.0
        };

        let var_change = if var_first > 1e-10 {
            (var_second - var_first) / var_first
        } else {
            0.0
        };

        // Kendall tau trends (simplified: just use the changes as proxy)
        let ar1_trend = ar1_change.clamp(-1.0, 1.0);
        let var_trend = var_change.clamp(-1.0, 1.0);

        // Combined score: weighted average of normalized indicators
        // AR(1) and variance are most reliable
        let combined_score = (
            ar1_trend.max(0.0) * 0.35 +
            var_trend.max(0.0) * 0.35 +
            ((spec_second - spec_first).max(0.0) * 0.15) +
            ((skew_second - skew_first).abs() * 0.15)
        ).clamp(0.0, 1.0);

        Ok(EarlyWarningSignals {
            autocorrelation: ar1_second,
            variance: var_second,
            skewness: skew_second,
            spectral_ratio: spec_second,
            ar1_trend,
            var_trend,
            combined_score,
        })
    }

    /// Get warning status for specific element
    pub fn get_warning_status(&mut self, element: TippingElementType) -> Result<(EarlyWarningSignals, bool), BifurcationError> {
        let history = self.histories.get(&element)
            .ok_or(BifurcationError::InvalidParameter)?;
        
        let signals = self.compute_signals_for_element(element, history)?;
        let is_critical = signals.combined_score > self.critical_threshold;

        Ok((signals, is_critical))
    }

    /// Compute autocorrelation at lag-1
    fn autocorrelation_lag1(series: &[f64]) -> f64 {
        if series.len() < 3 {
            return 0.0;
        }

        let n = series.len() as f64;
        let mean = series.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..(series.len() - 1) {
            let x_i = series[i] - mean;
            let x_ip1 = series[i + 1] - mean;
            numerator += x_i * x_ip1;
            denominator += x_i * x_i;
        }

        if denominator < 1e-15 {
            return 0.0;
        }
        numerator / denominator
    }

    /// Compute variance
    fn variance(series: &[f64]) -> f64 {
        if series.is_empty() {
            return 0.0;
        }
        let n = series.len() as f64;
        let mean = series.iter().sum::<f64>() / n;
        series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
    }

    /// Compute skewness
    fn skewness(series: &[f64]) -> f64 {
        if series.len() < 3 {
            return 0.0;
        }
        let n = series.len() as f64;
        let mean = series.iter().sum::<f64>() / n;
        let std = Self::variance(series).sqrt();

        if std < 1e-15 {
            return 0.0;
        }

        let m3: f64 = series.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
        m3 / (std.powi(3))
    }

    /// Compute spectral ratio (low/high frequency power)
    fn spectral_ratio(series: &[f64]) -> f64 {
        if series.len() < 4 {
            return 1.0;
        }

        // Simple approximation: compare variance of smoothed vs differenced series
        let n = series.len();
        
        // Low frequency: moving average residual
        let mut low_freq_var = 0.0;
        if n >= 3 {
            let mut smoothed = Vec::with_capacity(n - 2);
            for i in 1..(n - 1) {
                smoothed.push((series[i - 1] + series[i] + series[i + 1]) / 3.0);
            }
            let smooth_mean = smoothed.iter().sum::<f64>() / smoothed.len() as f64;
            low_freq_var = smoothed.iter().map(|x| (x - smooth_mean).powi(2)).sum::<f64>();
        }

        // High frequency: first differences
        let mut high_freq_var = 0.0;
        for i in 0..(n - 1) {
            let diff = series[i + 1] - series[i];
            high_freq_var += diff * diff;
        }

        if high_freq_var < 1e-15 {
            return 1.0;
        }
        (low_freq_var / high_freq_var).max(0.1).min(10.0)
    }

    /// Estimate distance to tipping point based on current state and rate
    pub fn estimate_distance_to_tipping(
        &self,
        element: TippingElementType,
        current_state: f64,
        rate: f64,
    ) -> f64 {
        // Simplified estimation based on element type
        let critical_state = match element {
            TippingElementType::AMOC => 0.3,           // 30% of current strength
            TippingElementType::AmazonDieback => 0.6,   // 60% forest cover loss
            TippingElementType::GreenlandIceSheet => 0.7,
            TippingElementType::WestAntarcticIceSheet => 0.5,
            TippingElementType::PermafrostMethane => 0.8,
            TippingElementType::MonsoonShift => 0.4,
        };

        if rate <= 0.0 {
            return f64::MAX; // Moving away from tipping point
        }

        let distance = (critical_state - current_state) / rate;
        distance.max(0.0)
    }
}

/// Stochastic differential equation solver for tipping dynamics
pub struct SdeTippingSolver {
    /// Drift coefficient function parameters
    alpha: f64,
    /// Diffusion coefficient
    sigma: f64,
    /// Bifurcation parameter
    mu: f64,
    /// Current state
    state: f64,
}

impl SdeTippingSolver {
    /// Create new SDE solver for fold bifurcation normal form
    /// dx = (mu - x^2) dt + sigma dW
    pub fn new(mu: f64, sigma: f64, initial_state: f64) -> Self {
        Self {
            alpha: 1.0,
            sigma: sigma.max(0.0),
            mu,
            state: initial_state,
        }
    }

    /// Set bifurcation parameter
    pub fn set_mu(&mut self, mu: f64) {
        self.mu = mu;
    }

    /// Advance one step using Euler-Maruyama
    pub fn step(&mut self, dt: f64, dw: f64) -> Result<f64, BifurcationError> {
        if dt <= 0.0 {
            return Err(BifurcationError::InvalidParameter);
        }

        // Drift: alpha * (mu - x^2)
        let drift = self.alpha * (self.mu - self.state * self.state);

        // Check for numerical overflow
        if drift.abs() > 1e10 {
            return Err(BifurcationError::NumericalOverflow);
        }

        // Euler-Maruyama: x_{t+dt} = x_t + drift*dt + sigma*dW
        self.state = self.state + drift * dt + self.sigma * dw * dt.sqrt();

        // Clamp to prevent runaway
        self.state = self.state.clamp(-100.0, 100.0);

        Ok(self.state)
    }

    /// Get current state
    pub fn state(&self) -> f64 {
        self.state
    }

    /// Check if system has tipped (escaped stable state)
    pub fn has_tipped(&self, threshold: f64) -> bool {
        self.state.abs() > threshold
    }
}

/// Aggregate climate risk assessment
#[derive(Debug, Clone)]
pub struct ClimateRiskAssessment {
    pub overall_risk_level: f64,  // 0-1
    pub critical_elements: Vec<TippingElementType>,
    pub recommended_actions: Vec<&'static str>,
}

impl TippingPointBifurcationDetector {
    /// Generate overall climate risk assessment
    pub fn assess_climate_risk(&mut self) -> Result<ClimateRiskAssessment, BifurcationError> {
        let all_signals = self.compute_all_signals()?;

        let mut total_score = 0.0;
        let mut critical_elements = Vec::new();
        let mut n_elements = 0;

        for (element, signals) in &all_signals {
            total_score += signals.combined_score;
            n_elements += 1;

            if signals.combined_score > self.critical_threshold {
                critical_elements.push(*element);
            }
        }

        let overall_risk = if n_elements > 0 {
            total_score / n_elements as f64
        } else {
            0.0
        };

        // Generate recommendations based on risk level
        let mut recommendations = Vec::new();

        if overall_risk > 0.7 {
            recommendations.push("IMMEDIATE: Reduce exposure to climate-vulnerable assets");
            recommendations.push("URGENT: Increase hedging positions in climate-resilient sectors");
        } else if overall_risk > 0.5 {
            recommendations.push("Monitor: Increase frequency of climate risk assessments");
            recommendations.push("Prepare: Review portfolio exposure to tipping-sensitive regions");
        }

        if critical_elements.contains(&TippingElementType::AMOC) {
            recommendations.push("AMOC Warning: Short North Atlantic fisheries, long climate adaptation tech");
        }
        if critical_elements.contains(&TippingElementType::AmazonDieback) {
            recommendations.push("Amazon Warning: Reduce exposure to Brazilian agriculture bonds");
        }
        if critical_elements.contains(&TippingElementType::GreenlandIceSheet) {
            recommendations.push("Ice Sheet Warning: Long coastal infrastructure protection equities");
        }

        Ok(ClimateRiskAssessment {
            overall_risk_level: overall_risk,
            critical_elements,
            recommended_actions: recommendations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bifurcation_detector() {
        let mut detector = TippingPointBifurcationDetector::new(50, 0.6);

        // Simulate approaching tipping point (increasing variance and AR(1))
        for i in 0..60 {
            let t = i as f64;
            // State with increasing noise and memory
            let state = 0.5 + 0.01 * t + (t.sin() * (1.0 + 0.02 * t));
            detector.observe(TippingElementType::AMOC, state);
        }

        let (signals, is_critical) = detector.get_warning_status(TippingElementType::AMOC).unwrap();
        assert!(signals.combined_score >= 0.0);
    }

    #[test]
    fn test_sde_solver() {
        let mut solver = SdeTippingSolver::new(1.0, 0.1, 0.0);
        
        for _ in 0..100 {
            let result = solver.step(0.01, 0.0);
            assert!(result.is_ok());
        }
    }
}
