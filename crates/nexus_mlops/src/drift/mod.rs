//! Drift detection module

mod adwin_detector;
mod page_hinkley_test;
mod ks_divergence;
pub mod page_hinkley_detector;
pub mod streaming_ks_test;
pub mod drift_state_machine;

pub use adwin_detector::AdwinDetector;
pub use page_hinkley_test::PageHinkleyTest;
pub use ks_divergence::KSDivergence;
pub use page_hinkley_detector::{PageHinkleyTest as PHTest, StreamingKSTest};
pub use streaming_ks_test::{StreamingKSTest as KSTest, KSTestConfig};
pub use drift_state_machine::{DriftStateMachine, DriftStateMachineConfig, DriftState, DriftEvaluation, RecommendedAction};
