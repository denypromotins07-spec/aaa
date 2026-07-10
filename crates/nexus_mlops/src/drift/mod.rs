//! Drift detection module

mod adwin_detector;
mod page_hinkley_test;
mod ks_divergence;

pub use adwin_detector::AdwinDetector;
pub use page_hinkley_test::PageHinkleyTest;
pub use ks_divergence::KSDivergence;
