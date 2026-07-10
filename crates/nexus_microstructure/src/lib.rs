//! NEXUS-OMEGA Stage 17: Adversarial Microstructure, Spoofing Detection & Queue Dynamics
//! 
//! This crate provides real-time detection of spoofing, layering, and quote-stuffing attacks,
//! survival analysis for limit order queues, order flow toxicity decomposition, and
//! cryptographic stealth routing to evade adversarial HFT firms.

pub mod adversarial;
pub mod queue;
pub mod toxicity;
pub mod stealth;

// Re-export key types
pub use adversarial::hawkes_spoofer_detector::{HawkesSpoofingDetector, SpoofingSignal};
pub use adversarial::intensity_spike_classifier::IntensitySpikeClassifier;
pub use adversarial::quote_stuffing_filter::QuoteStuffingFilter;

pub use queue::survival_analysis::KaplanMeierEstimator;
pub use queue::iceberg_detector::IcebergDetector;
pub use queue::true_queue_depth::TrueQueueDepthCalculator;

pub use toxicity::glosten_milgrom_decomposition::GlostenMilgromDecomposer;
pub use toxicity::easley_ohara_vpin::EasleyOHaravPIN;
pub use toxicity::bayesian_informed_flow::BayesianInformedFlowUpdater;

pub use stealth::footprint_obfuscator::FootprintObfuscator;
pub use stealth::cryptographic_stealth_rng::CryptographicStealthRng;
pub use stealth::minimax_venue_router::MinimaxVenueRouter;
