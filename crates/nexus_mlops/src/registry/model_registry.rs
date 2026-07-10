//! Model Registry for Versioned Model Storage
//!
//! Tracks model versions, metadata, and lifecycle states.

use crate::MLOpsError;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Model lifecycle state
#[derive(Debug, Clone, PartialEq)]
pub enum ModelState {
    /// Model is being trained
    Training,
    /// Model completed training, pending validation
    PendingValidation,
    /// Model validated and ready for deployment
    Validated,
    /// Model deployed to production
    Production,
    /// Model in shadow mode (parallel testing)
    Shadow,
    /// Model deprecated
    Deprecated,
    /// Model archived
    Archived,
}

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub model_id: String,
    pub version: String,
    pub state: ModelState,
    pub created_at: u64,
    pub updated_at: u64,
    pub metrics: HashMap<String, f64>,
    pub checkpoint_path: Option<PathBuf>,
    pub parent_version: Option<String>,
    pub tags: Vec<String>,
}

impl ModelMetadata {
    pub fn new(model_id: &str, version: &str) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            model_id: model_id.to_string(),
            version: version.to_string(),
            state: ModelState::Training,
            created_at: now,
            updated_at: now,
            metrics: HashMap::new(),
            checkpoint_path: None,
            parent_version: None,
            tags: Vec::new(),
        }
    }
}

/// Model registry for tracking all models
pub struct ModelRegistry {
    /// Models indexed by (model_id, version)
    models: RwLock<HashMap<(String, String), ModelMetadata>>,
    /// Current production version per model
    production_versions: RwLock<HashMap<String, String>>,
    /// Maximum versions to keep per model
    max_versions_per_model: usize,
}

impl ModelRegistry {
    /// Create new model registry
    pub fn new(max_versions: usize) -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            production_versions: RwLock::new(HashMap::new()),
            max_versions_per_model: max_versions,
        }
    }

    /// Register a new model version
    pub fn register(&self, metadata: ModelMetadata) -> Result<(), MLOpsError> {
        let key = (metadata.model_id.clone(), metadata.version.clone());
        
        let mut models = self.models.write().map_err(|_| {
            MLOpsError::CheckpointSaveFailed("Registry lock poisoned".to_string())
        })?;

        // Check for duplicate
        if models.contains_key(&key) {
            return Err(MLOpsError::CheckpointSaveFailed(
                format!("Model {} version {} already exists", metadata.model_id, metadata.version)
            ));
        }

        models.insert(key, metadata);
        Ok(())
    }

    /// Update model state
    pub fn update_state(&self, model_id: &str, version: &str, state: ModelState) -> Result<(), MLOpsError> {
        let key = (model_id.to_string(), version.to_string());
        
        let mut models = self.models.write().map_err(|_| {
            MLOpsError::CheckpointSaveFailed("Registry lock poisoned".to_string())
        })?;

        if let Some(metadata) = models.get_mut(&key) {
            metadata.state = state;
            metadata.updated_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            Ok(())
        } else {
            Err(MLOpsError::CheckpointSaveFailed(
                format!("Model {} version {} not found", model_id, version)
            ))
        }
    }

    /// Promote model to production
    pub fn promote_to_production(&self, model_id: &str, version: &str) -> Result<(), MLOpsError> {
        // First validate the model exists and is validated
        {
            let models = self.models.read().map_err(|_| {
                MLOpsError::CheckpointSaveFailed("Registry lock poisoned".to_string())
            })?;

            let key = (model_id.to_string(), version.to_string());
            let metadata = models.get(&key).ok_or_else(|| {
                MLOpsError::CheckpointSaveFailed(
                    format!("Model {} version {} not found", model_id, version)
                )
            })?;

            if metadata.state != ModelState::Validated && metadata.state != ModelState::Shadow {
                return Err(MLOpsError::CheckpointSaveFailed(
                    "Model must be validated or in shadow before promotion".to_string()
                ));
            }
        }

        // Update state to production
        self.update_state(model_id, version, ModelState::Production)?;

        // Update production version mapping
        let mut prod_versions = self.production_versions.write().map_err(|_| {
            MLOpsError::CheckpointSaveFailed("Registry lock poisoned".to_string())
        })?;
        
        prod_versions.insert(model_id.to_string(), version.to_string());

        Ok(())
    }

    /// Get model metadata
    pub fn get_model(&self, model_id: &str, version: &str) -> Option<ModelMetadata> {
        let models = self.models.read().ok()?;
        let key = (model_id.to_string(), version.to_string());
        models.get(&key).cloned()
    }

    /// Get current production version
    pub fn get_production_version(&self, model_id: &str) -> Option<String> {
        let prod_versions = self.production_versions.read().ok()?;
        prod_versions.get(model_id).cloned()
    }

    /// List all versions of a model
    pub fn list_versions(&self, model_id: &str) -> Vec<ModelMetadata> {
        let models = self.models.read().unwrap_or_else(|e| e.into_inner());
        
        let mut versions: Vec<_> = models
            .iter()
            .filter(|((id, _), _)| id == model_id)
            .map(|(_, meta)| meta.clone())
            .collect();

        // Sort by creation time
        versions.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        versions
    }

    /// Add metric to model
    pub fn add_metric(&self, model_id: &str, version: &str, name: &str, value: f64) -> Result<(), MLOpsError> {
        let key = (model_id.to_string(), version.to_string());
        
        let mut models = self.models.write().map_err(|_| {
            MLOpsError::CheckpointSaveFailed("Registry lock poisoned".to_string())
        })?;

        if let Some(metadata) = models.get_mut(&key) {
            metadata.metrics.insert(name.to_string(), value);
            metadata.updated_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            Ok(())
        } else {
            Err(MLOpsError::CheckpointSaveFailed(
                format!("Model {} version {} not found", model_id, version)
            ))
        }
    }

    /// Archive old versions beyond max limit
    pub fn archive_old_versions(&self, model_id: &str) -> Result<Vec<String>, MLOpsError> {
        let mut models = self.models.write().map_err(|_| {
            MLOpsError::CheckpointSaveFailed("Registry lock poisoned".to_string())
        })?;

        let mut versions: Vec<_> = models
            .iter_mut()
            .filter(|((id, _), _)| id == model_id)
            .collect();

        // Sort by creation time descending
        versions.sort_by(|a, b| b.1.created_at.cmp(&a.1.created_at));

        let mut archived = Vec::new();

        // Archive versions beyond limit (except production)
        let prod_version = self.get_production_version(model_id);
        
        for (i, (_, metadata)) in versions.iter_mut().enumerate() {
            if i >= self.max_versions_per_model 
                && metadata.state != ModelState::Production
                && Some(metadata.version.clone()) != prod_version
            {
                metadata.state = ModelState::Archived;
                archived.push(metadata.version.clone());
            }
        }

        Ok(archived)
    }

    /// Get total model count
    pub fn model_count(&self) -> usize {
        let models = self.models.read().unwrap_or_else(|e| e.into_inner());
        models.len()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registration() {
        let registry = ModelRegistry::new(5);
        
        let mut metadata = ModelMetadata::new("test_model", "v1");
        metadata.state = ModelState::Validated;
        
        registry.register(metadata).unwrap();
        
        let retrieved = registry.get_model("test_model", "v1").unwrap();
        assert_eq!(retrieved.version, "v1");
        assert_eq!(retrieved.state, ModelState::Validated);
    }

    #[test]
    fn test_promotion_lifecycle() {
        let registry = ModelRegistry::new(5);
        
        let mut metadata = ModelMetadata::new("test_model", "v1");
        metadata.state = ModelState::Validated;
        registry.register(metadata).unwrap();
        
        registry.promote_to_production("test_model", "v1").unwrap();
        
        let prod_version = registry.get_production_version("test_model").unwrap();
        assert_eq!(prod_version, "v1");
        
        let model = registry.get_model("test_model", "v1").unwrap();
        assert_eq!(model.state, ModelState::Production);
    }

    #[test]
    fn test_archiving_old_versions() {
        let registry = ModelRegistry::new(2);
        
        for i in 1..=4 {
            let mut metadata = ModelMetadata::new("test_model", &format!("v{}", i));
            metadata.state = ModelState::Validated;
            registry.register(metadata).unwrap();
        }
        
        let archived = registry.archive_old_versions("test_model").unwrap();
        
        assert_eq!(archived.len(), 2); // v1 and v2 should be archived
    }
}
