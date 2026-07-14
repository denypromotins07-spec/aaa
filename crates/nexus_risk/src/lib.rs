//! NEXUS-OMEGA Stage 4: Pre-Trade Risk Gatekeeper, Circuit Breakers & Kill Switch
//!
//! This crate implements advanced risk management for live trading:
//! - Chapter 1: Pre-Trade Risk Gatekeeper (Zero-Alloc Hot Path)
//! - Chapter 2: Velocity Circuit Breakers & PnL Derivative Tracking
//! - Chapter 3: Global Kill Switch & Portfolio Flatten FSM
//! - Chapter 4: Tail-Risk Survival, Black Swan Modeling & Extreme Value Theory (EVT)

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![forbid(unsafe_code)]

pub mod gatekeeper;
pub mod breakers;
pub mod kill_switch;
pub mod tail;
pub mod dependence;
pub mod contagion;
pub mod hedging;

// Re-export Chapter 1 types
pub use gatekeeper::{PreTradeRiskGatekeeper, RiskApproved, RiskViolation, PriceCollarValidator, AtomicPositionAccumulator};

// Re-export Chapter 2 types
pub use breakers::{TokenBucketThrottle, VelocityCircuitBreaker, PnLDerivativeTracker};

// Re-export Chapter 3 types
pub use kill_switch::{GlobalKillSwitch, FlattenFSM, FlattenState, TWAPLiquidationRouter};

// Re-export Chapter 4 types (Stage 11)
pub use tail::extreme_value_theory::{ExtremeValueTheory, EvtConfig, EvtFitResult};
pub use tail::gpd_mle_solver::{GpdParameters, GpdSolver, MleFitResult};
pub use dependence::student_t_copula::{StudentTCopula, StudentTCopulaConfig};
pub use dependence::tail_dependence_metric::{TailDependenceCalculator, TailDependenceResult};
pub use contagion::multivariate_hawkes::{MultivariateHawkesProcess, HawkesConfig};
pub use contagion::ogata_thinning::{OgataThinningSimulator, SimulationResult};
pub use hedging::convex_hedge_optimizer::{ConvexHedgeOptimizer, HedgeInstrument, HedgeAllocation};
pub use hedging::doomsday_state_machine::{DoomsdayStateMachine, DoomsdayState, RiskSignals};
