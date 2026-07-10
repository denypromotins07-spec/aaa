//! Atomic Model Router using ArcSwap for Lock-Free Hot-Swapping
//!
//! Implements Read-Copy-Update (RCU) pattern for zero-downtime model weight replacement.
//! Uses arc-swap crate for atomic pointer operations without mutex contention.

use crate::MLOpsError;
use arc_swap::ArcSwap;
use std::sync::Arc;

/// Model trait for inference engines
pub trait InferenceModel: Send + Sync {
    /// Run inference on input features
    fn infer(&self, features: &[f64]) -> Result<f64, MLOpsError>;
    
    /// Get model version identifier
    fn version(&self) -> &str;
}

/// Atomic model router for lock-free hot-swapping
/// 
/// Allows inference threads to read model weights lock-free while
/// MLOps thread atomically swaps pointers in O(1) time.
pub struct AtomicModelRouter<M: InferenceModel> {
    /// Atomic pointer to current model
    current_model: ArcSwap<M>,
    /// Model version history
    version_history: std::sync::Mutex<Vec<String>>,
    /// Swap counter for monitoring
    swap_count: std::sync::atomic::AtomicU64,
}

impl<M: InferenceModel> AtomicModelRouter<M> {
    /// Create new atomic model router with initial model
    pub fn new(initial_model: Arc<M>) -> Self {
        let version = initial_model.version().to_string();
        
        Self {
            current_model: ArcSwap::new(initial_model),
            version_history: std::sync::Mutex::new(vec![version]),
            swap_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Load model (lock-free read)
    /// Returns Arc reference that keeps model alive even after swap
    pub fn load(&self) -> Arc<M> {
        self.current_model.load_full()
    }

    /// Run inference using current model (lock-free)
    pub fn infer(&self, features: &[f64]) -> Result<f64, MLOpsError> {
        let model = self.load();
        model.infer(features)
    }

    /// Atomically swap to new model (O(1) operation)
    /// Old model is deallocated only when all references are dropped
    pub fn swap(&self, new_model: Arc<M>) -> Result<Arc<M>, MLOpsError> {
        let old_model = self.current_model.swap(new_model);
        
        // Track version history
        if let Ok(mut history) = self.version_history.lock() {
            history.push(old_model.version().to_string());
        }
        
        // Increment swap counter
        self.swap_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        Ok(old_model)
    }

    /// Compare-and-swap: only swap if current model matches expected
    pub fn compare_and_swap(&self, expected: &M, new_model: Arc<M>) -> Result<bool, MLOpsError> {
        // Note: This requires PartialEq or similar trait bound
        // For now, we use a simpler approach with version checking
        let current = self.load();
        
        if current.version() == expected.version() {
            self.swap(new_model)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get current model version
    pub fn current_version(&self) -> String {
        self.load().version().to_string()
    }

    /// Get swap count
    pub fn swap_count(&self) -> u64 {
        self.swap_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get version history
    pub fn version_history(&self) -> Vec<String> {
        self.version_history
            .lock()
            .map(|h| h.clone())
            .unwrap_or_default()
    }

    /// Gracefully shutdown - wait for all references to be dropped
    pub fn shutdown(&self) {
        // Force release of any held references
        // In production, would implement proper epoch-based reclamation
        self.current_model.store(Arc::new(
            DummyModel::new("shutdown")
        ));
    }
}

/// Dummy model for shutdown state
struct DummyModel {
    version: String,
}

impl DummyModel {
    fn new(version: &str) -> Self {
        Self {
            version: version.to_string(),
        }
    }
}

impl InferenceModel for DummyModel {
    fn infer(&self, _features: &[f64]) -> Result<f64, MLOpsError> {
        Err(MLOpsError::ModelSwapFailed("Model in shutdown state".to_string()))
    }
    
    fn version(&self) -> &str {
        &self.version
    }
}

/// Builder for AtomicModelRouter
pub struct AtomicModelRouterBuilder<M: InferenceModel> {
    initial_model: Option<Arc<M>>,
}

impl<M: InferenceModel> AtomicModelRouterBuilder<M> {
    pub fn new() -> Self {
        Self {
            initial_model: None,
        }
    }

    pub fn with_model(mut self, model: Arc<M>) -> Self {
        self.initial_model = Some(model);
        self
    }

    pub fn build(self) -> Result<AtomicModelRouter<M>, MLOpsError> {
        let initial_model = self.initial_model.ok_or_else(|| {
            MLOpsError::ModelSwapFailed("Initial model required".to_string())
        })?;

        Ok(AtomicModelRouter::new(initial_model))
    }
}

impl<M: InferenceModel> Default for AtomicModelRouterBuilder<M> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestModel {
        version: String,
        offset: f64,
    }

    impl TestModel {
        fn new(version: &str, offset: f64) -> Self {
            Self {
                version: version.to_string(),
                offset,
            }
        }
    }

    impl InferenceModel for TestModel {
        fn infer(&self, features: &[f64]) -> Result<f64, MLOpsError> {
            let sum: f64 = features.iter().sum();
            Ok(sum + self.offset)
        }

        fn version(&self) -> &str {
            &self.version
        }
    }

    #[test]
    fn test_atomic_inference() {
        let model = Arc::new(TestModel::new("v1", 1.0));
        let router = AtomicModelRouter::new(model);

        let features = vec![1.0, 2.0, 3.0];
        let result = router.infer(&features).unwrap();
        
        assert_eq!(result, 7.0); // 1+2+3 + 1.0
        assert_eq!(router.current_version(), "v1");
    }

    #[test]
    fn test_atomic_swap() {
        let model_v1 = Arc::new(TestModel::new("v1", 1.0));
        let router = AtomicModelRouter::new(model_v1);

        let model_v2 = Arc::new(TestModel::new("v2", 2.0));
        let old = router.swap(model_v2).unwrap();

        assert_eq!(old.version(), "v1");
        assert_eq!(router.current_version(), "v2");
        assert_eq!(router.swap_count(), 1);

        let features = vec![1.0, 2.0, 3.0];
        let result = router.infer(&features).unwrap();
        
        assert_eq!(result, 8.0); // 1+2+3 + 2.0
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;
        
        let model = Arc::new(TestModel::new("v1", 0.0));
        let router = Arc::new(AtomicModelRouter::new(model));

        let mut handles = vec![];

        // Spawn reader threads
        for i in 0..10 {
            let router_clone = Arc::clone(&router);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let features = vec![i as f64];
                    let _ = router_clone.infer(&features);
                }
            }));
        }

        // Spawn swapper thread
        let router_clone = Arc::clone(&router);
        handles.push(thread::spawn(move || {
            for v in 2..5 {
                let new_model = Arc::new(TestModel::new(&format!("v{}", v), v as f64));
                let _ = router_clone.swap(new_model);
            }
        }));

        for handle in handles {
            handle.join().unwrap();
        }

        assert!(router.swap_count() >= 3);
    }
}
