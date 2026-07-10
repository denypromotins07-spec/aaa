//! Nexus Risk Management Engine - Stage 5 of 50
//! 
//! This crate implements:
//! - Chapter 1: Pre-Trade Risk Gatekeeper (Lock-Free Interceptor)
//! - Chapter 2: Real-Time Portfolio Risk Metrics (SIMD VaR & Greeks)
//! - Chapter 3: Velocity-Loss Detectors & Hardware Kill-Switches
//! - Chapter 4: Margin Cascade & Liquidation Prevention Engine

pub mod gatekeeper;
pub mod metrics;
pub mod breakers;
pub mod margin;

// Re-export main components for convenience
pub use gatekeeper::pre_trade_interceptor::PreTradeRiskInterceptor;
pub use gatekeeper::fat_finger_collars::FatFingerValidator;
pub use gatekeeper::lock_free_order_queue::LockFreeOrderQueue;

pub use metrics::simd_var_calculator::SimdVaRCalculator;
pub use metrics::covariance_matrix::CovarianceMatrixEngine;
pub use metrics::portfolio_greeks::PortfolioGreeksAggregator;

pub use breakers::velocity_loss_detector::VelocityLossDetector;
pub use breakers::global_state_machine::{SystemState, GlobalStateMachine};

pub use margin::liquidation_simulator::ShadowLiquidationSimulator;
pub use margin::auto_deleverager::AutoDeleveragingEngine;
pub use margin::cross_venue_aggregator::CrossVenueMarginAggregator;

/// Risk engine configuration
#[derive(Debug, Clone)]
pub struct RiskConfig {
    /// Maximum order size in base units
    pub max_order_size: u64,
    /// Fat finger price collar percentage (e.g., 200 = 2%)
    pub fat_finger_collar_bps: u16,
    /// Maximum open orders per symbol
    pub max_open_orders_per_symbol: u32,
    /// VaR confidence level (e.g., 0.99 for 99%)
    pub var_confidence_level: f64,
    /// Velocity loss threshold in USD per millisecond
    pub velocity_loss_threshold_usd_ms: f64,
    /// Margin utilization warning threshold (e.g., 0.9 = 90%)
    pub margin_warning_threshold: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_order_size: 1_000_000_000, // 1B base units
            fat_finger_collar_bps: 200,     // 2%
            max_open_orders_per_symbol: 100,
            var_confidence_level: 0.99,
            velocity_loss_threshold_usd_ms: 100.0, // $100k per second
            margin_warning_threshold: 0.9,
        }
    }
}
