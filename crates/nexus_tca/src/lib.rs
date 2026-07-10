//! Stage 13: Real-Time Transaction Cost Analysis (TCA)

pub mod metrics;
pub mod feedback;

pub use metrics::implementation_shortfall::*;
pub use metrics::slippage_decomposition::*;
pub use feedback::rl_reward_penalty::*;
