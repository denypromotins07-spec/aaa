//! Shadow deployment module

mod telemetry_aggregator;
mod pnl_attribution;

pub use telemetry_aggregator::TelemetryAggregator;
pub use pnl_attribution::PnLAttribution;

use crate::MLOpsError;

/// Shadow orchestrator configuration
#[derive(Debug, Clone)]
pub struct ShadowConfig {
    /// Maximum concurrent shadow models
    pub max_shadow_models: usize,
    /// VRAM quota per shadow model (MB)
    pub vram_quota_mb: usize,
    /// Minimum samples before evaluation
    pub min_samples_eval: usize,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            max_shadow_models: 3,
            vram_quota_mb: 2048,
            min_samples_eval: 1000,
        }
    }
}

/// Shadow orchestrator for parallel model evaluation
pub struct ShadowOrchestrator {
    config: ShadowConfig,
    /// Active shadow models
    active_models: std::collections::HashMap<String, ShadowModel>,
}

struct ShadowModel {
    model_id: String,
    telemetry: TelemetryAggregator,
    start_time: u64,
    sample_count: u64,
}

impl ShadowOrchestrator {
    pub fn new(config: ShadowConfig) -> Self {
        Self {
            config,
            active_models: std::collections::HashMap::new(),
        }
    }

    /// Register a new shadow model
    pub fn register_shadow(&mut self, model_id: &str) -> Result<(), MLOpsError> {
        if self.active_models.len() >= self.config.max_shadow_models {
            // Evict oldest model (LRU)
            if let Some(oldest_id) = self.find_oldest_model() {
                self.active_models.remove(&oldest_id);
            }
        }

        let shadow = ShadowModel {
            model_id: model_id.to_string(),
            telemetry: TelemetryAggregator::new(),
            start_time: 0, // Would use actual timestamp
            sample_count: 0,
        };

        self.active_models.insert(model_id.to_string(), shadow);
        Ok(())
    }

    /// Process a sample through all shadow models
    pub fn process_sample(&mut self, features: &[f64], predictions: std::collections::HashMap<String, f64>) -> Result<(), MLOpsError> {
        for (model_id, prediction) in predictions {
            if let Some(shadow) = self.active_models.get_mut(&model_id) {
                shadow.telemetry.record_prediction(prediction)?;
                shadow.sample_count += 1;
            }
        }
        Ok(())
    }

    /// Get telemetry for a shadow model
    pub fn get_telemetry(&self, model_id: &str) -> Option<&TelemetryAggregator> {
        self.active_models.get(model_id).map(|s| &s.telemetry)
    }

    /// Find oldest model for LRU eviction
    fn find_oldest_model(&self) -> Option<String> {
        self.active_models
            .iter()
            .min_by_key(|(_, s)| s.start_time)
            .map(|(id, _)| id.clone())
    }
}
