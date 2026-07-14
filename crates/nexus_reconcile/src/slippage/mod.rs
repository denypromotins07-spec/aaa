//! Slippage module - Implementation shortfall and market impact tracking
pub mod nanosecond_latency_tracker;
pub mod implementation_shortfall;
pub mod market_impact_feedback;

pub use nanosecond_latency_tracker::{NanosecondLatencyTracker, LatencyStats, LatencyStatsMicros};
pub use implementation_shortfall::{ImplementationShortfallTracker, SignalFillRecord, SignalContext, ShortfallStats};
pub use market_impact_feedback::{MarketImpactFeedback, SlippageProfile, FeedbackStats};
