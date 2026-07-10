//! NEXUS-OMEGA Stage 16: MLOps, Concept Drift Detection & Shadow Deployment Orchestration
//!
//! This crate provides:
//! - Streaming drift detection (ADWIN, Page-Hinkley, K-S divergence)
//! - Shadow deployment with zero-impact telemetry
//! - Atomic model hot-swapping via ArcSwap and Epoch-Based Reclamation
//! - Ray cluster orchestration for distributed GPU retraining

pub mod drift;
pub mod shadow;
pub mod checkpoint;
pub mod registry;
pub mod inference;
pub mod deployment;

pub use drift::{AdwinDetector, PageHinkleyTest, KSDivergence};
pub use shadow::{ShadowOrchestrator, TelemetryAggregator, PnLAttribution};
pub use checkpoint::AtomicWeightSaver;
pub use registry::ModelRegistry;
pub use inference::AtomicModelRouter;
pub use deployment::CanaryRouter;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MLOpsError {
    #[error("Drift detected: {0}")]
    DriftDetected(String),
    
    #[error("Shadow model validation failed: {0}")]
    ShadowValidationFailed(String),
    
    #[error("Checkpoint save failed: {0}")]
    CheckpointSaveFailed(String),
    
    #[error("Model swap failed: {0}")]
    ModelSwapFailed(String),
    
    #[error("Resource quota exceeded: {0}")]
    ResourceQuotaExceeded(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, MLOpsError>;
