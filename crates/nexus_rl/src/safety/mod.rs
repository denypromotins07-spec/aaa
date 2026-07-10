//! NEXUS-OMEGA Stage 19: Safe RL Module
//!
//! This module provides safety guarantees for reinforcement learning-based trading:
//! - Lyapunov stability verification
//! - QP-based action projection onto safe polytopes
//! - Forward invariance guards
//! - Recovery policy override
//! - QP infeasibility detection
//!
//! Author: NEXUS-OMEGA Architecture
//! Stage: 19 of 50

pub mod lyapunov_function;
pub mod qp_action_projector;
pub mod forward_invariance_guard;
pub mod recovery_policy_override;
pub mod qp_infeasibility_detector;

// Re-export main types for convenience
pub use lyapunov_function::{LyapunovFunction, LyapunovConfig, RiskMetrics, SafetyStatus};
pub use qp_action_projector::{QPActionProjector, QPProjectorConfig, ProjectionStatus};
pub use forward_invariance_guard::{ForwardInvarianceGuard, InvarianceGuardConfig, SafetyGuardResult};
pub use recovery_policy_override::{RecoveryPolicyManager, RecoveryPolicyConfig, RecoveryMode};
pub use qp_infeasibility_detector::{QPInfeasibilityDetector, FeasibilityStatus};
