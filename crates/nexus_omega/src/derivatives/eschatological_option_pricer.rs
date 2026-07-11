//! Eschatological Option Pricer - derivatives tied to terminal paradigm collapse.
//! Prices options based on final-state boundary conditions rather than initial conditions.

use alloc::vec::Vec;
use core::fmt::Debug;

use super::super::teleology::final_state_boundary::{
    FinalBoundaryType, FinalStateBoundarySolver, FinalStateConfig, FinalStateResult, OptionType,
};

/// Configuration for eschatological option pricing
#[derive(Debug, Clone)]
pub struct EschatologicalOptionConfig {
    /// Type of eschatological event
    pub event_type: EschatologicalEvent,
    /// Time horizon until potential event (years)
    pub time_horizon: f64,
    /// Confidence in event occurrence (0-1)
    pub event_probability: f64,
    /// Risk-free rate
    pub risk_free_rate: f64,
}

impl Default for EschatologicalOptionConfig {
    fn default() -> Self {
        Self {
            event_type: EschatologicalEvent::CurrencyCollapse,
            time_horizon: 30.0,
            event_probability: 0.1,
            risk_free_rate: 0.02,
        }
    }
}

/// Types of eschatological events
#[derive(Debug, Clone)]
pub enum EschatologicalEvent {
    /// Fiat currency purchasing power → 0
    CurrencyCollapse,
    /// Complete market regime change
    RegimeChange,
    /// Asset class extinction
    AssetExtinction,
    /// Technological singularity impact
    SingularityShock,
    /// Climate tipping point
    ClimateTippingPoint,
}

/// Result of eschatological option pricing
#[derive(Debug, Clone)]
pub struct EschatologicalOptionPrice {
    /// Call option price
    pub call_price: f64,
    /// Put option price
    pub put_price: f64,
    /// Probability-weighted expected payoff
    pub expected_payoff: f64,
    /// Implied probability of event
    pub implied_probability: f64,
    /// Sensitivity to event probability (rho)
    pub rho: f64,
    /// Sensitivity to time horizon (theta)
    pub theta: f64,
}

/// Eschatological option pricer
pub struct EschatologicalOptionPricer {
    config: EschatologicalOptionConfig,
    boundary_solver: FinalStateBoundarySolver,
}

impl EschatologicalOptionPricer {
    pub fn new(config: EschatologicalOptionConfig) -> Self {
        let boundary_type = match config.event_type {
            EschatologicalEvent::CurrencyCollapse => {
                FinalBoundaryType::CurrencyCollapse { decay_rate: 0.03 }
            }
            EschatologicalEvent::RegimeChange => FinalBoundaryType::PhaseTransition {
                critical_time: config.time_horizon,
            },
            EschatologicalEvent::AssetExtinction => FinalBoundaryType::VolatilityExtinction {
                timescale: config.time_horizon / 2.0,
            },
            EschatologicalEvent::SingularityShock => FinalBoundaryType::FiniteTimeSingularity {
                singularity_time: config.time_horizon,
            },
            EschatologicalEvent::ClimateTippingPoint => FinalBoundaryType::PhaseTransition {
                critical_time: config.time_horizon * 0.8,
            },
        };

        let boundary_config = FinalStateConfig {
            boundary_type,
            time_horizon: config.time_horizon,
            spatial_range: (0.01, 100.0),
            num_grid_points: 256,
        };

        Self {
            config,
            boundary_solver: FinalStateBoundarySolver::new(boundary_config),
        }
    }

    /// Price eschatological option with given strike
    pub fn price(&self, strike: f64, current_spot: f64) -> Result<EschatologicalOptionPrice, &'static str> {
        if strike <= 0.0 || current_spot <= 0.0 {
            return Err("Invalid strike or spot price");
        }

        if self.config.event_probability <= 0.0 || self.config.event_probability > 1.0 {
            return Err("Event probability must be in (0, 1]");
        }

        // Price call option (betting on event occurrence)
        let call_result = self.boundary_solver.price_eschatological_option(
            strike,
            OptionType::Call,
        )?;

        // Price put option (hedging against event)
        let put_result = self.boundary_solver.price_eschatological_option(
            strike,
            OptionType::Put,
        )?;

        // Extract prices from present value at current spot
        let call_price = self.extract_price_at_spot(&call_result.present_value, current_spot);
        let put_price = self.extract_price_at_spot(&put_result.present_value, current_spot);

        // Calculate expected payoff weighted by event probability
        let expected_payoff = (call_price + put_price) * self.config.event_probability;

        // Calculate Greeks
        let rho = self.calculate_rho(call_price, put_price);
        let theta = self.calculate_theta(call_price, put_price);

        Ok(EschatologicalOptionPrice {
            call_price,
            put_price,
            expected_payoff,
            implied_probability: self.config.event_probability,
            rho,
            theta,
        })
    }

    fn extract_price_at_spot(&self, present_value: &[f64], spot: f64) -> f64 {
        if present_value.is_empty() {
            return 0.0;
        }

        let n = present_value.len();
        let (x_min, x_max) = self.boundary_solver_spatial_range();
        let dx = (x_max - x_min) / (n - 1) as f64;

        // Find index corresponding to spot price
        let log_spot = spot.ln();
        let idx_f = (log_spot - x_min) / dx;
        let idx = idx_f as usize;

        if idx >= n {
            return present_value[n - 1];
        }
        if idx == 0 {
            return present_value[0];
        }

        // Linear interpolation
        let frac = idx_f - idx as f64;
        present_value[idx] * (1.0 - frac) + present_value[idx + 1] * frac
    }

    fn boundary_solver_spatial_range(&self) -> (f64, f64) {
        // Would need to expose this from FinalStateBoundarySolver
        (0.01_f64.ln(), 100.0_f64.ln())
    }

    fn calculate_rho(&self, call: f64, put: f64) -> f64 {
        // Sensitivity to event probability
        (call + put) / 100.0 // Approximate
    }

    fn calculate_theta(&self, call: f64, put: f64) -> f64 {
        // Sensitivity to time horizon
        -(call + put) / self.config.time_horizon / 365.0 // Per day
    }

    /// Price a basket of eschatological options
    pub fn price_basket(
        &self,
        strikes: &[f64],
        spots: &[f64],
        weights: &[f64],
    ) -> Result<BasketPrice, &'static str> {
        if strikes.len() != spots.len() || strikes.len() != weights.len() {
            return Err("Strikes, spots, and weights must have same length");
        }

        let mut total_call = 0.0;
        let mut total_put = 0.0;
        let mut total_expected = 0.0;

        for (&strike, &spot, &weight) in strikes.iter().zip(spots.iter()).zip(weights.iter()) {
            let price = self.price(strike, spot)?;
            total_call += price.call_price * weight;
            total_put += price.put_price * weight;
            total_expected += price.expected_payoff * weight;
        }

        Ok(BasketPrice {
            total_call_value: total_call,
            total_put_value: total_put,
            total_expected_value: total_expected,
            num_options: strikes.len(),
        })
    }
}

/// Basket of eschatological options
#[derive(Debug, Clone)]
pub struct BasketPrice {
    pub total_call_value: f64,
    pub total_put_value: f64,
    pub total_expected_value: f64,
    pub num_options: usize,
}

/// Builder for eschatological option configurations
pub struct EschatologicalOptionBuilder {
    config: EschatologicalOptionConfig,
}

impl EschatologicalOptionBuilder {
    pub fn new() -> Self {
        Self {
            config: EschatologicalOptionConfig::default(),
        }
    }

    pub fn with_event_type(mut self, event: EschatologicalEvent) -> Self {
        self.config.event_type = event;
        self
    }

    pub fn with_time_horizon(mut self, years: f64) -> Self {
        if years > 0.0 {
            self.config.time_horizon = years;
        }
        self
    }

    pub fn with_probability(mut self, prob: f64) -> Self {
        if prob > 0.0 && prob <= 1.0 {
            self.config.event_probability = prob;
        }
        self
    }

    pub fn build(self) -> EschatologicalOptionPricer {
        EschatologicalOptionPricer::new(self.config)
    }
}

impl Default for EschatologicalOptionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let pricer = EschatologicalOptionBuilder::new()
            .with_event_type(EschatologicalEvent::CurrencyCollapse)
            .with_time_horizon(20.0)
            .with_probability(0.15)
            .build();

        assert_eq!(pricer.config.time_horizon, 20.0);
        assert!((pricer.config.event_probability - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_basic_pricing() {
        let config = EschatologicalOptionConfig::default();
        let pricer = EschatologicalOptionPricer::new(config);

        let result = pricer.price(100.0, 100.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_inputs() {
        let pricer = EschatologicalOptionPricer::new(EschatologicalOptionConfig::default());

        assert!(pricer.price(-1.0, 100.0).is_err());
        assert!(pricer.price(100.0, 0.0).is_err());
    }
}
