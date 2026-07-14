//! Tap Module - Lock-free telemetry tap and frame aggregation
//!
//! This module provides the core telemetry tap functionality:
//! - Lock-free event subscription from trading engine
//! - Frame aggregation (microseconds to 60fps)
//! - Timing utilities

pub mod lock_free_event_subscriber;
pub mod frame_aggregator;
pub mod microsecond_to_fps;

pub use lock_free_event_subscriber::{
    LockFreeEventSubscriber, TelemetryEvent, EventSubscriberConfig,
};
pub use frame_aggregator::{
    FrameAggregator, FrameAggregatorConfig, SymbolState, ActiveAlphaSignal,
};
pub use microsecond_to_fps::{
    fps_to_frame_duration_ns, fps_to_frame_duration_us,
    micros_to_frames, nanos_to_frames,
    FrameTimer, TelemetryRateLimiter,
    FRAME_60FPS_NS, FRAME_60FPS_US,
};
