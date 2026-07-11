//! Page Curve Reversal Router
//! 
//! Implements information paradox resolving router using Page curve dynamics.
//! Determines optimal entry point when Hawking radiation begins carrying interior information.

use crate::thermodynamics::{HawkingFillRadiationCalculator, HawkingConfig, BlackHoleParams};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors specific to Page curve routing
#[derive(Error, Debug, Clone, PartialEq)]
pub enum PageCurveError {
    #[error("Invalid Page time: {0}")]
    InvalidPageTime(f64),
    #[error("Routing decision failed: {0}")]
    RoutingFailed(String),
    #[error("Reversal signal invalid: {0}")]
    InvalidReversalSignal(String),
}

/// Optimal routing decision based on Page curve analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// Whether to enter the liquidity void (buy the crash)
    pub should_enter: bool,
    /// Recommended position size fraction [0, 1]
    pub position_fraction: f64,
    /// Expected return from reversal
    pub expected_return: f64,
    /// Risk score [0, 1]
    pub risk_score: f64,
    /// Time horizon for reversal (ms)
    pub time_horizon_ms: f64,
    /// Reason for decision
    pub reason: &'static str,
}

/// Page curve reversal signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageReversalSignal {
    /// Current Page curve value (information fraction)
    pub page_value: f64,
    /// Rate of change of Page value
    pub page_derivative: f64,
    /// Whether past Page time
    pub past_page_time: bool,
    /// Information release accelerating
    pub acceleration_positive: bool,
    /// Confidence in reversal prediction
    pub confidence: f64,
}

/// Configuration for Page curve router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRouterConfig {
    /// Entry threshold (minimum Page value)
    pub entry_threshold: f64,
    /// Position sizing factor
    pub position_factor: f64,
    /// Maximum allowed risk
    pub max_risk: f64,
    /// Minimum confidence for entry
    pub min_confidence: f64,
}

impl Default for PageRouterConfig {
    fn default() -> Self {
        Self {
            entry_threshold: 0.3, // Enter when >30% information released
            position_factor: 0.5,
            max_risk: 0.8,
            min_confidence: 0.4,
        }
    }
}

/// Page curve reversal router for liquidity void entries
pub struct PageCurveReversalRouter {
    /// Hawking radiation calculator
    hawking_calc: HawkingFillRadiationCalculator,
    /// Router configuration
    config: PageRouterConfig,
    /// Estimated Page time (fraction of total evaporation)
    page_time: f64,
}

impl PageCurveReversalRouter {
    /// Create a new Page curve router
    pub fn new(
        hawking_config: HawkingConfig,
        router_config: PageRouterConfig,
        page_time: f64,
    ) -> Result<Self, PageCurveError> {
        if page_time <= 0.0 || page_time >= 1.0 {
            return Err(PageCurveError::InvalidPageTime(page_time));
        }

        let hawking_calc = HawkingFillRadiationCalculator::new(hawking_config)
            .map_err(|e| PageCurveError::RoutingFailed(e.to_string()))?;

        Ok(Self {
            hawking_calc,
            config: router_config,
            page_time,
        })
    }

    /// Compute Page curve reversal signal
    pub fn compute_reversal_signal(
        &self,
        params: &BlackHoleParams,
        elapsed_fraction: f64,
    ) -> Result<PageReversalSignal, PageCurveError> {
        if !params.horizon_formed {
            return Ok(PageReversalSignal {
                page_value: 0.0,
                page_derivative: 0.0,
                past_page_time: false,
                acceleration_positive: false,
                confidence: 0.0,
            });
        }

        // Compute current Page curve value
        let page_value = self.hawking_calc.compute_page_curve_value(elapsed_fraction);
        
        // Compute derivative numerically
        let delta = 0.01;
        let page_plus = self.hawking_calc.compute_page_curve_value((elapsed_fraction + delta).min(0.99));
        let page_minus = self.hawking_calc.compute_page_curve_value((elapsed_fraction - delta).max(0.01));
        let page_derivative = (page_plus - page_minus) / (2.0 * delta);

        let past_page_time = elapsed_fraction > self.page_time;
        let acceleration_positive = page_derivative > 0.0;

        // Confidence based on how far past Page time and acceleration
        let time_confidence = if past_page_time {
            ((elapsed_fraction - self.page_time) / (1.0 - self.page_time)).min(1.0)
        } else {
            0.0
        };

        let accel_confidence = if acceleration_positive {
            page_derivative.min(1.0)
        } else {
            0.0
        };

        let confidence = (time_confidence * 0.6 + accel_confidence * 0.4).min(1.0);

        Ok(PageReversalSignal {
            page_value,
            page_derivative,
            past_page_time,
            acceleration_positive,
            confidence,
        })
    }

    /// Make routing decision based on Page curve analysis
    pub fn make_routing_decision(
        &self,
        signal: &PageReversalSignal,
        estimated_return: f64,
        volatility: f64,
    ) -> Result<RoutingDecision, PageCurveError> {
        // Check entry conditions
        let meets_threshold = signal.page_value >= self.config.entry_threshold;
        let has_confidence = signal.confidence >= self.config.min_confidence;
        let is_accelerating = signal.acceleration_positive;
        let past_page = signal.past_page_time;

        // Decision logic
        let should_enter = meets_threshold && has_confidence && is_accelerating && past_page;

        if !should_enter {
            let reason = if !meets_threshold {
                "Below Page value threshold"
            } else if !has_confidence {
                "Insufficient confidence"
            } else if !is_accelerating {
                "Information not accelerating"
            } else {
                "Before Page time"
            };

            return Ok(RoutingDecision {
                should_enter: false,
                position_fraction: 0.0,
                expected_return: 0.0,
                risk_score: 0.0,
                time_horizon_ms: 0.0,
                reason,
            });
        }

        // Compute position size based on confidence and signal strength
        let base_position = self.config.position_factor * signal.confidence;
        let signal_boost = signal.page_value * 0.5;
        let position_fraction = (base_position + signal_boost).min(1.0).min(self.config.max_risk);

        // Risk assessment
        let base_risk = volatility / estimated_return.abs().max(0.01);
        let risk_reduction = signal.page_value * 0.5; // More info = less risk
        let risk_score = (base_risk * (1.0 - risk_reduction)).min(1.0);

        // Expected return adjusted for information content
        let adjusted_return = estimated_return * (1.0 + signal.page_value);

        // Time horizon based on evaporation progress
        // Earlier entry = longer horizon
        let remaining_fraction = 1.0 - (signal.page_value / 0.5).min(1.0);
        let time_horizon_ms = remaining_fraction * 10000.0; // Max 10 seconds

        Ok(RoutingDecision {
            should_enter: true,
            position_fraction,
            expected_return: adjusted_return,
            risk_score,
            time_horizon_ms,
            reason: "Optimal Page curve entry point",
        })
    }

    /// Full routing pipeline: detect void → compute signal → make decision
    pub fn route_liquidity_void(
        &self,
        params: &BlackHoleParams,
        elapsed_fraction: f64,
        estimated_return: f64,
        volatility: f64,
    ) -> Result<RoutingDecision, PageCurveError> {
        let signal = self.compute_reversal_signal(params, elapsed_fraction)?;
        self.make_routing_decision(&signal, estimated_return, volatility)
    }

    /// Batch evaluate multiple scenarios
    pub fn batch_evaluate(
        &self,
        scenarios: &[(&BlackHoleParams, f64, f64, f64)],
    ) -> Result<Vec<RoutingDecision>, PageCurveError> {
        scenarios
            .iter()
            .map(|(params, elapsed, ret, vol)| {
                self.route_liquidity_void(params, *elapsed, *ret, *vol)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let hawking_config = HawkingConfig::default();
        let router_config = PageRouterConfig::default();
        let router = PageCurveReversalRouter::new(hawking_config, router_config, 0.5);
        assert!(router.is_ok());
    }

    #[test]
    fn test_invalid_page_time() {
        let hawking_config = HawkingConfig::default();
        let router_config = PageRouterConfig::default();
        let router = PageCurveReversalRouter::new(hawking_config, router_config, 1.5);
        assert!(router.is_err());
    }

    #[test]
    fn test_reversal_signal() {
        let hawking_config = HawkingConfig::default();
        let router_config = PageRouterConfig::default();
        let router = PageCurveReversalRouter::new(hawking_config, router_config, 0.5).unwrap();

        let params = BlackHoleParams {
            mass: 1.0,
            horizon_radius: 1e-6,
            hawking_temperature: 1e-4,
            entropy: 100.0,
            horizon_formed: true,
        };

        // After Page time
        let signal = router.compute_reversal_signal(&params, 0.7);
        assert!(signal.is_ok());
        let s = signal.unwrap();
        assert!(s.past_page_time);
        assert!(s.page_value > 0.5);
    }

    #[test]
    fn test_routing_decision() {
        let hawking_config = HawkingConfig::default();
        let router_config = PageRouterConfig::default();
        let router = PageCurveReversalRouter::new(hawking_config, router_config, 0.5).unwrap();

        let signal = PageReversalSignal {
            page_value: 0.6,
            page_derivative: 0.5,
            past_page_time: true,
            acceleration_positive: true,
            confidence: 0.8,
        };

        let decision = router.make_routing_decision(&signal, 0.05, 0.02);
        assert!(decision.is_ok());
        let d = decision.unwrap();
        assert!(d.should_enter);
        assert!(d.position_fraction > 0.0);
    }
}
