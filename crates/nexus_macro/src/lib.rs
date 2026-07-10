//! Stage 12: Cross-Asset Macro Regimes, Yield Curve Modeling & Global Macro Alpha
//!
//! This crate implements:
//! - Nelson-Siegel-Svensson yield curve fitting with Levenberg-Marquardt optimization
//! - Hayashi-Yoshida estimator for non-synchronous cross-asset covariance
//! - Bayesian HMM for latent macro regime detection
//! - Hierarchical Risk Parity (HRP) with Ledoit-Wolf shrinkage

pub mod fixed_income;
pub mod lead_lag;
pub mod regimes;
pub mod portfolio;

// Re-exports for convenience
pub use fixed_income::nss_curve_fitter::*;
pub use fixed_income::levenberg_marquardt::*;
pub use fixed_income::zero_alloc_bootstrapper::*;

pub use lead_lag::hayashi_yoshida_estimator::*;
pub use lead_lag::thermodynamic_delay::*;
pub use lead_lag::non_sync_covariance_matrix::*;

pub use regimes::bayesian_hmm::*;
pub use regimes::baum_welch_em::*;
pub use regimes::regime_conditional_router::*;

pub use portfolio::hierarchical_risk_parity::*;
pub use portfolio::minimum_spanning_tree::*;
pub use portfolio::ledoit_wolf_shrinkage::*;
