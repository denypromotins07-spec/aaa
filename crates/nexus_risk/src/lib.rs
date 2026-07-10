//! NEXUS-OMEGA Stage 11: Tail-Risk Survival, Black Swan Modeling & Extreme Value Theory
//!
//! This crate implements advanced risk management for extreme market events:
//! - Chapter 1: Extreme Value Theory (EVT) & Fat Tail Modeling
//! - Chapter 2: Copulas & Non-Linear Tail Dependence  
//! - Chapter 3: Multivariate Hawkes Processes & Crash Contagion
//! - Chapter 4: Doomsday Portfolio Optimizer & Convex Hedging

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![forbid(unsafe_code)]

pub mod tail;
pub mod dependence;
pub mod contagion;
pub mod hedging;

/// Re-export commonly used types for Stage 11
pub use tail::extreme_value_theory::{ExtremeValueTheory, EvtConfig, EvtFitResult};
pub use tail::gpd_mle_solver::{GpdParameters, GpdSolver, MleFitResult};
pub use dependence::student_t_copula::{StudentTCopula, StudentTCopulaConfig};
pub use dependence::tail_dependence_metric::{TailDependenceCalculator, TailDependenceResult};
pub use contagion::multivariate_hawkes::{MultivariateHawkesProcess, HawkesConfig};
pub use contagion::ogata_thinning::{OgataThinningSimulator, SimulationResult};
pub use hedging::convex_hedge_optimizer::{ConvexHedgeOptimizer, HedgeInstrument, HedgeAllocation};
pub use hedging::doomsday_state_machine::{DoomsdayStateMachine, DoomsdayState, RiskSignals};
