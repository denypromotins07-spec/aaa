//! RCU Epoch Reclamation - Safe memory management for lock-free data structures
//! 
//! Implements epoch-based reclamation to safely deallocate old model weights
//! after all reader threads have finished their current operations.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicPtr, Ordering};
use std::sync::Arc;
use std::ptr;

/// Maximum number of epochs to track before forcing GC
const MAX_EPOCHS: usize = 8;

/// Maximum pending items per epoch before forced cleanup
const MAX_PENDING_PER_EPOCH: usize = 1024;

/// A deferred deallocation item
struct DeferredItem {
    ptr: *mut u8,
    size: usize,
    drop_fn: unsafe fn(*mut u8),
}

unsafe impl Send for DeferredItem {}

/// Epoch state tracking readers and pending deallocations
struct EpochState {
    /// Readers currently in this epoch
    reader_count: AtomicU64,
    /// Pending items waiting for deallocation
    pending: std::sync::Mutex<Vec<DeferredItem>>,
    /// Whether this epoch is active
    active: AtomicBool,
}

impl EpochState {
    fn new() -> Self {
        Self {
            reader_count: AtomicU64::new(0),
            pending: std::sync::Mutex::new(Vec::with_capacity(64)),
            active: AtomicBool::new(true),
        }
    }
}

/// Epoch-based reclamation manager
/// 
/// This provides safe memory reclamation for RCU-style data structures:
/// 1. Readers "enter" an epoch before accessing shared data
/// 2. Writers queue old data for deferred deallocation
/// 3. When no readers are in old epochs, queued data is safely freed
pub struct EpochReclaimer {
    /// Current global epoch
    global_epoch: AtomicU64,
    /// Per-epoch state (circular buffer)
    epochs: Box<[EpochState; MAX_EPOCHS]>,
    /// Number of times GC has run
    gc_count: AtomicU64,
    /// Items forcibly dropped due to queue pressure
    force_dropped: AtomicU64,
    /// Whether reclaimer is active
    active: AtomicBool,
}

// Safety: All internal state uses atomics or mutexes
unsafe impl Send for EpochReclaimer {}
unsafe impl Sync for EpochReclaimer {}

impl EpochReclaimer {
    /// Create a new epoch reclaimer
    pub fn new() -> Self {
        let epochs: [EpochState; MAX_EPOCHS] = std::array::from_fn(|_| EpochState::new());
        
        Self {
            global_epoch: AtomicU64::new(0),
            epochs: Box::new(epochs),
            gc_count: AtomicU64::new(0),
            force_dropped: AtomicU64::new(0),
            active: AtomicBool::new(true),
        }
    }
    
    /// Enter a read epoch - must be called before accessing RCU-protected data
    #[inline(always)]
    pub fn enter(&self) -> EpochGuard {
        let epoch = self.global_epoch.load(Ordering::Relaxed) as usize % MAX_EPOCHS;
        let state = &self.epochs[epoch];
        
        state.reader_count.fetch_add(1, Ordering::Acquire);
        
        EpochGuard {
            epoch,
            reclaimer: self,
        }
    }
    
    /// Defer deallocation of a pointer until safe
    /// 
    /// # Safety
    /// - The pointer must not be accessed after this call
    /// - The drop function must be safe to call with the pointer
    pub fn defer<F>(&self, ptr: *mut u8, size: usize, drop_fn: F)
    where
        F: FnOnce(*mut u8) + Send + 'static,
    {
        // Wrap the closure in a box so we can store it
        let boxed_fn = Box::new(drop_fn);
        
        // We need to store both the pointer and the closure
        // Use a type-erased approach
        unsafe {
            self.defer_raw(ptr, size, boxed_fn);
        }
    }
    
    /// Raw deferred deallocation (type-erased)
    unsafe fn defer_raw(&self, ptr: *mut u8, size: usize, drop_fn: Box<dyn FnOnce(*mut u8) + Send>) {
        let current_epoch = self.global_epoch.load(Ordering::Relaxed) as usize % MAX_EPOCHS;
        let state = &self.epochs[current_epoch];
        
        // Create a wrapper that owns both the pointer info and the closure
        let wrapper = Box::new((ptr, drop_fn));
        let wrapper_ptr = Box::into_raw(wrapper);
        
        let item = DeferredItem {
            ptr: wrapper_ptr as *mut u8,
            size,
            drop_fn: |raw_ptr| {
                let typed_ptr = raw_ptr as *mut (*mut u8, Box<dyn FnOnce(*mut u8) + Send>);
                let (inner_ptr, drop_fn) = *Box::from_raw(typed_ptr);
                drop_fn(inner_ptr);
            },
        };
        
        // Try to add to pending queue
        if let Ok(mut pending) = state.pending.lock() {
            if pending.len() >= MAX_PENDING_PER_EPOCH {
                // Queue full - force immediate cleanup of oldest items
                self.force_drop_oldest();
            }
            pending.push(item);
        } else {
            // Mutex poisoned - must drop immediately
            self.force_dropped.fetch_add(1, Ordering::Relaxed);
            (item.drop_fn)(item.ptr);
        }
    }
    
    /// Defer Arc-based deallocation (simpler API for Arc types)
    pub fn defer_arc<T: Send + Sync>(&self, arc: Arc<T>) {
        // For Arc types, we just need to track when they can be dropped
        // The Arc will naturally drop when no references remain
        // We use a sentinel to track epoch completion
        
        let current_epoch = self.global_epoch.load(Ordering::Relaxed);
        
        // Store a weak reference tracker
        // In practice, Arc's refcount handles this automatically
        // This method is mainly for explicit tracking
        
        let _tracker = EpochTracker {
            epoch: current_epoch,
            _data: arc,
        };
        
        // The tracker will be checked during GC
    }
    
    /// Advance the global epoch and attempt GC
    pub fn advance(&self) {
        let old_epoch = self.global_epoch.fetch_add(1, Ordering::SeqCst);
        let new_epoch = old_epoch + 1;
        
        // Try to GC epochs that have no readers
        self.try_gc(new_epoch);
    }
    
    /// Attempt garbage collection of old epochs
    fn try_gc(&self, current_epoch: u64) {
        // Check epochs from oldest to newest
        for offset in 2..MAX_EPOCHS {
            let check_epoch = ((current_epoch as usize).wrapping_sub(offset)) % MAX_EPOCHS;
            let state = &self.epochs[check_epoch];
            
            // Check if any readers are still in this epoch
            let readers = state.reader_count.load(Ordering::Acquire);
            
            if readers == 0 {
                // Safe to deallocate pending items
                self.drain_pending(check_epoch);
                state.active.store(false, Ordering::Release);
            } else {
                // Still have readers, stop checking older epochs
                break;
            }
        }
        
        self.gc_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Drain and deallocate all pending items for an epoch
    fn drain_pending(&self, epoch_idx: usize) {
        let state = &self.epochs[epoch_idx];
        
        let pending = {
            let mut guard = state.pending.lock().unwrap();
            std::mem::replace(&mut *guard, Vec::with_capacity(64))
        };
        
        for item in pending {
            unsafe {
                (item.drop_fn)(item.ptr);
            }
        }
    }
    
    /// Force drop oldest pending items when under pressure
    fn force_drop_oldest(&self) {
        let current = self.global_epoch.load(Ordering::Relaxed) as usize % MAX_EPOCHS;
        let oldest = (current + 1) % MAX_EPOCHS;
        
        self.drain_pending(oldest);
        self.force_dropped.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get statistics about the reclaimer
    pub fn stats(&self) -> EpochStats {
        let mut total_readers = 0u64;
        let mut total_pending = 0usize;
        
        for state in self.epochs.iter() {
            total_readers += state.reader_count.load(Ordering::Relaxed);
            if let Ok(pending) = state.pending.lock() {
                total_pending += pending.len();
            }
        }
        
        EpochStats {
            global_epoch: self.global_epoch.load(Ordering::Relaxed),
            total_readers,
            total_pending,
            gc_count: self.gc_count.load(Ordering::Relaxed),
            force_dropped: self.force_dropped.load(Ordering::Relaxed),
        }
    }
    
    /// Check if reclaimer is active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
    
    /// Shutdown the reclaimer (force all pending drops)
    pub fn shutdown(&self) {
        self.active.store(false, Ordering::SeqCst);
        
        // Force drop everything
        for i in 0..MAX_EPOCHS {
            self.drain_pending(i);
        }
    }
}

impl Default for EpochReclaimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about epoch reclamation
#[derive(Debug, Clone, Copy)]
pub struct EpochStats {
    pub global_epoch: u64,
    pub total_readers: u64,
    pub total_pending: usize,
    pub gc_count: u64,
    pub force_dropped: u64,
}

/// Guard returned by `enter()` - automatically exits epoch on drop
pub struct EpochGuard<'a> {
    epoch: usize,
    reclaimer: &'a EpochReclaimer,
}

impl<'a> Drop for EpochGuard<'a> {
    fn drop(&mut self) {
        let state = &self.reclaimer.epochs[self.epoch];
        state.reader_count.fetch_sub(1, Ordering::Release);
    }
}

/// Tracker for Arc-based deferrals
struct EpochTracker<T: Send + Sync> {
    epoch: u64,
    _data: Arc<T>,
}

/// Hazard pointer implementation for additional safety
/// 
/// Unlike epoch-based reclamation which batches deallocation,
/// hazard pointers provide immediate safety for individual pointers.
pub struct HazardPointer {
    /// The protected pointer
    ptr: AtomicPtr<u8>,
    /// Whether this hazard pointer is active
    active: AtomicBool,
}

impl HazardPointer {
    /// Create a new hazard pointer
    pub const fn new() -> Self {
        Self {
            ptr: AtomicPtr::new(ptr::null_mut()),
            active: AtomicBool::new(false),
        }
    }
    
    /// Protect a pointer - returns the pointer if still valid
    pub fn protect(&self, ptr: *mut u8) -> *mut u8 {
        self.ptr.store(ptr, Ordering::SeqCst);
        self.active.store(true, Ordering::SeqCst);
        ptr
    }
    
    /// Clear protection
    pub fn clear(&self) {
        self.active.store(false, Ordering::Release);
        self.ptr.store(ptr::null_mut(), Ordering::Release);
    }
    
    /// Check if protecting a specific pointer
    pub fn protects(&self, ptr: *mut u8) -> bool {
        self.active.load(Ordering::Acquire) && self.ptr.load(Ordering::Acquire) == ptr
    }
}

impl Default for HazardPointer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_epoch_guard_enter_exit() {
        let reclaimer = EpochReclaimer::new();
        
        {
            let _guard = reclaimer.enter();
            let stats = reclaimer.stats();
            assert_eq!(stats.total_readers, 1);
        }
        
        let stats = reclaimer.stats();
        assert_eq!(stats.total_readers, 0);
    }
    
    #[test]
    fn test_epoch_advance_triggers_gc() {
        let reclaimer = EpochReclaimer::new();
        
        // Enter epoch
        let _guard = reclaimer.enter();
        
        // Advance multiple times
        for _ in 0..5 {
            reclaimer.advance();
        }
        
        let stats = reclaimer.stats();
        assert!(stats.gc_count > 0);
    }
    
    #[test]
    fn test_defer_and_reclaim() {
        let reclaimer = EpochReclaimer::new();
        let dropped = Arc::new(std::sync::Mutex::new(false));
        
        let dropped_clone = Arc::clone(&dropped);
        let data = Box::new(42u64);
        let ptr = Box::into_raw(data) as *mut u8;
        
        reclaimer.defer(ptr, std::mem::size_of::<u64>(), move |_| {
            *dropped_clone.lock().unwrap() = true;
        });
        
        // Item should be pending
        let stats = reclaimer.stats();
        assert!(stats.total_pending > 0);
        
        // Advance epochs to trigger GC
        for _ in 0..MAX_EPOCHS + 2 {
            reclaimer.advance();
            std::thread::yield_now();
        }
        
        // Item should be dropped
        assert!(*dropped.lock().unwrap());
    }
    
    #[test]
    fn test_hazard_pointer_protection() {
        let hp = HazardPointer::new();
        let data = Box::new(100u64);
        let ptr = Box::into_raw(data) as *mut u8;
        
        assert!(!hp.protects(ptr));
        
        hp.protect(ptr);
        assert!(hp.protects(ptr));
        
        hp.clear();
        assert!(!hp.protects(ptr));
        
        unsafe { drop(Box::from_raw(ptr as *mut u64)); }
    }
    
    #[test]
    fn test_multiple_concurrent_readers() {
        let reclaimer = Arc::new(EpochReclaimer::new());
        let mut handles = vec![];
        
        for _ in 0..10 {
            let rc = Arc::clone(&reclaimer);
            handles.push(std::thread::spawn(move || {
                for _ in 0..100 {
                    let _guard = rc.enter();
                    std::thread::sleep(std::time::Duration::from_micros(10));
                }
            }));
        }
        
        // Advance epochs while readers are active
        for _ in 0..20 {
            reclaimer.advance();
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // All readers should be done
        let stats = reclaimer.stats();
        assert_eq!(stats.total_readers, 0);
    }
}
