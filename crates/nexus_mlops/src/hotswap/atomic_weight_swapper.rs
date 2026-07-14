//! Atomic Weight Swapper using Read-Copy-Update (RCU) pattern
//! 
//! Enables lock-free hot-swapping of model weights in O(1) time.
//! Uses arc-swap for atomic pointer operations with epoch-based reclamation.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use arc_swap::ArcSwap;

/// Model weights wrapper - can be any serializable weight structure
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Unique identifier for this weight version
    pub version: u64,
    /// Serialized weights (in production, this would be actual tensor data)
    pub data: Arc<[f64]>,
    /// Metadata about the model
    pub metadata: ModelMetadata,
}

/// Metadata about a model version
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub created_at_ns: u64,
    pub training_samples: u64,
    pub validation_score: f64,
    pub regime_type: &'static str,
}

impl ModelWeights {
    /// Create new model weights
    pub fn new(version: u64, data: Vec<f64>, metadata: ModelMetadata) -> Self {
        Self {
            version,
            data: data.into_boxed_slice().into(),
            metadata,
        }
    }
    
    /// Get weight count
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Result of a swap operation
#[derive(Debug)]
pub struct SwapResult {
    pub success: bool,
    pub old_version: u64,
    pub new_version: u64,
    pub swap_time_ns: u64,
}

/// Atomic Weight Swapper using RCU pattern
/// 
/// This allows multiple reader threads to access weights concurrently
/// while a writer can atomically swap in new weights without blocking readers.
pub struct AtomicWeightSwapper {
    /// Current active weights (RCU-protected via ArcSwap)
    current_weights: ArcSwap<ModelWeights>,
    /// Version counter for new weights
    version_counter: AtomicU64,
    /// Number of successful swaps
    swap_count: AtomicU64,
    /// Timestamp of last swap
    last_swap_timestamp_ns: AtomicU64,
    /// Whether swapper is initialized
    initialized: AtomicBool,
}

// Safety: ArcSwap provides thread-safe access
unsafe impl Send for AtomicWeightSwapper {}
unsafe impl Sync for AtomicWeightSwapper {}

impl AtomicWeightSwapper {
    /// Create a new atomic weight swapper with initial weights
    pub fn new(initial_weights: ModelWeights) -> Self {
        Self {
            current_weights: ArcSwap::new(Arc::new(initial_weights)),
            version_counter: AtomicU64::new(1),
            swap_count: AtomicU64::new(0),
            last_swap_timestamp_ns: AtomicU64::new(0),
            initialized: AtomicBool::new(true),
        }
    }
    
    /// Create an uninitialized swapper (weights must be loaded before use)
    pub fn uninit() -> Self {
        let dummy = ModelWeights::new(0, vec![0.0; 1], ModelMetadata {
            created_at_ns: 0,
            training_samples: 0,
            validation_score: 0.0,
            regime_type: "uninitialized",
        });
        
        Self {
            current_weights: ArcSwap::new(Arc::new(dummy)),
            version_counter: AtomicU64::new(0),
            swap_count: AtomicU64::new(0),
            last_swap_timestamp_ns: AtomicU64::new(0),
            initialized: AtomicBool::new(false),
        }
    }
    
    /// Load initial weights (one-time operation)
    pub fn load_initial(&self, weights: ModelWeights) -> bool {
        if self.initialized.load(Ordering::Relaxed) {
            return false; // Already initialized
        }
        
        self.current_weights.store(Arc::new(weights));
        self.initialized.store(true, Ordering::SeqCst);
        true
    }
    
    /// Atomically swap to new weights (RCU operation - O(1), zero blocking)
    /// 
    /// This is the core RCU operation:
    /// 1. Allocate new weights on heap (done before calling this)
    /// 2. Atomically update pointer to new weights
    /// 3. Old weights are kept alive until all readers finish (epoch-based)
    pub fn swap_weights(&self, new_weights: ModelWeights) -> SwapResult {
        let start_ns = timestamp_ns();
        let old_version = self.current_weights.load().version;
        
        // Atomic pointer swap - this is O(1) and non-blocking
        let new_arc = Arc::new(new_weights);
        let old_arc = self.current_weights.swap(new_arc.clone());
        
        let new_version = new_arc.version;
        self.version_counter.fetch_add(1, Ordering::Relaxed);
        self.swap_count.fetch_add(1, Ordering::Relaxed);
        self.last_swap_timestamp_ns.store(start_ns, Ordering::Relaxed);
        
        // ArcSwap handles epoch-based reclamation automatically
        // The old_arc will be dropped when no readers hold references
        
        let end_ns = timestamp_ns();
        
        SwapResult {
            success: true,
            old_version,
            new_version,
            swap_time_ns: end_ns - start_ns,
        }
    }
    
    /// Get current weights (read-only, lock-free)
    /// 
    /// Returns an Arc that keeps the weights alive for the duration of use.
    /// This is the "Read" part of RCU - completely lock-free.
    #[inline(always)]
    pub fn get_weights(&self) -> Arc<ModelWeights> {
        self.current_weights.load_full()
    }
    
    /// Get current version number
    #[inline(always)]
    pub fn current_version(&self) -> u64 {
        self.current_weights.load().version
    }
    
    /// Check if swapper is initialized
    #[inline(always)]
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Relaxed)
    }
    
    /// Get swap count
    #[inline(always)]
    pub fn swap_count(&self) -> u64 {
        self.swap_count.load(Ordering::Relaxed)
    }
    
    /// Get last swap timestamp
    #[inline(always)]
    pub fn last_swap_timestamp_ns(&self) -> u64 {
        self.last_swap_timestamp_ns.load(Ordering::Relaxed)
    }
    
    /// Force garbage collection of old epochs (optional, usually automatic)
    pub fn force_gc(&self) {
        // ArcSwap automatically manages epochs, but we can hint at GC
        // by dropping any cached loads
        self.current_weights.rcu(|w| w.clone());
    }
}

/// Helper function to get nanosecond timestamp
#[inline(always)]
fn timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Epoch-based reclamation manager for fine-grained control
/// 
/// This provides additional safety guarantees for scenarios where
/// readers may hold references for extended periods.
pub struct EpochManager {
    /// Current epoch
    current_epoch: AtomicU64,
    /// Number of active readers in current epoch
    active_readers: AtomicU64,
    /// Pending deallocations waiting for epoch advance
    pending_deallocs: std::sync::Mutex<Vec<Arc<dyn Send + Sync>>>,
    /// Epoch threshold before forced GC
    gc_threshold: u64,
}

impl EpochManager {
    /// Create a new epoch manager
    pub fn new(gc_threshold: u64) -> Self {
        Self {
            current_epoch: AtomicU64::new(0),
            active_readers: AtomicU64::new(0),
            pending_deallocs: std::sync::Mutex::new(Vec::new()),
            gc_threshold,
        }
    }
    
    /// Enter a read epoch (call before reading weights)
    #[inline(always)]
    pub fn enter_read(&self) -> EpochGuard {
        self.active_readers.fetch_add(1, Ordering::Acquire);
        EpochGuard {
            epoch: self.current_epoch.load(Ordering::Relaxed),
            manager: self,
        }
    }
    
    /// Advance epoch and attempt GC
    pub fn advance_epoch(&self) {
        let current = self.current_epoch.fetch_add(1, Ordering::SeqCst);
        
        // Only GC every gc_threshold epochs
        if current % self.gc_threshold == 0 {
            self.try_gc();
        }
    }
    
    /// Attempt to garbage collect old allocations
    fn try_gc(&self) {
        let active = self.active_readers.load(Ordering::Acquire);
        
        if active == 0 {
            // Safe to deallocate - no active readers
            let mut pending = self.pending_deallocs.lock().unwrap();
            pending.clear();
        }
        // If there are active readers, wait for next epoch
    }
    
    /// Queue an allocation for deferred deallocation
    pub fn defer_dealloc(&self, item: Arc<dyn Send + Sync>) {
        if let Ok(mut pending) = self.pending_deallocs.lock() {
            pending.push(item);
            
            // Force GC if queue is too large
            if pending.len() > 1000 {
                drop(pending);
                self.advance_epoch();
            }
        }
    }
}

/// Guard for epoch-based reads
pub struct EpochGuard<'a> {
    epoch: u64,
    manager: &'a EpochManager,
}

impl<'a> Drop for EpochGuard<'a> {
    fn drop(&mut self) {
        self.manager.active_readers.fetch_sub(1, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_atomic_swapper_creation() {
        let weights = ModelWeights::new(
            1,
            vec![1.0, 2.0, 3.0],
            ModelMetadata {
                created_at_ns: timestamp_ns(),
                training_samples: 1000,
                validation_score: 0.95,
                regime_type: "mean_reversion",
            },
        );
        
        let swapper = AtomicWeightSwapper::new(weights);
        assert!(swapper.is_initialized());
        assert_eq!(swapper.current_version(), 1);
    }
    
    #[test]
    fn test_lock_free_read() {
        let weights = ModelWeights::new(1, vec![1.0; 100], create_metadata());
        let swapper = AtomicWeightSwapper::new(weights);
        
        // Multiple concurrent reads should work
        let w1 = swapper.get_weights();
        let w2 = swapper.get_weights();
        let w3 = swapper.get_weights();
        
        assert_eq!(w1.len(), 100);
        assert_eq!(w2.len(), 100);
        assert_eq!(w3.len(), 100);
        assert_eq!(w1.version, w2.version);
    }
    
    #[test]
    fn test_atomic_swap() {
        let weights = ModelWeights::new(1, vec![1.0; 10], create_metadata());
        let swapper = AtomicWeightSwapper::new(weights);
        
        let new_weights = ModelWeights::new(
            2,
            vec![2.0; 10],
            ModelMetadata {
                created_at_ns: timestamp_ns(),
                training_samples: 2000,
                validation_score: 0.97,
                regime_type: "high_vol",
            },
        );
        
        let result = swapper.swap_weights(new_weights);
        
        assert!(result.success);
        assert_eq!(result.old_version, 1);
        assert_eq!(result.new_version, 2);
        assert_eq!(swapper.current_version(), 2);
        assert_eq!(swapper.swap_count(), 1);
    }
    
    #[test]
    fn test_concurrent_read_during_swap() {
        let weights = ModelWeights::new(1, vec![1.0; 100], create_metadata());
        let swapper = Arc::new(AtomicWeightSwapper::new(weights));
        
        let swapper_clone = Arc::clone(&swapper);
        
        // Spawn reader thread
        let reader_handle = std::thread::spawn(move || {
            let mut sum = 0.0;
            for _ in 0..1000 {
                let weights = swapper_clone.get_weights();
                sum += weights.data[0];
            }
            sum
        });
        
        // Perform swap while reader is active
        std::thread::sleep(std::time::Duration::from_millis(1));
        
        let new_weights = ModelWeights::new(2, vec![2.0; 100], create_metadata());
        let result = swapper.swap_weights(new_weights);
        
        assert!(result.success);
        
        // Reader should complete without panic
        let reader_sum = reader_handle.join().unwrap();
        assert!(reader_sum > 0.0); // Should have read some values
    }
    
    #[test]
    fn test_swap_latency_is_low() {
        let weights = ModelWeights::new(1, vec![1.0; 1000], create_metadata());
        let swapper = AtomicWeightSwapper::new(weights);
        
        let new_weights = ModelWeights::new(2, vec![2.0; 1000], create_metadata());
        
        let result = swapper.swap_weights(new_weights);
        
        // Swap should complete in sub-millisecond time
        assert!(result.swap_time_ns < 1_000_000, "Swap took too long: {}ns", result.swap_time_ns);
    }
    
    fn create_metadata() -> ModelMetadata {
        ModelMetadata {
            created_at_ns: timestamp_ns(),
            training_samples: 1000,
            validation_score: 0.95,
            regime_type: "test",
        }
    }
}
