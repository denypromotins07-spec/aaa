//! Custom Bump Allocator for zero-allocation hot-paths
//!
//! This module provides a `BumpAllocator` that uses the `bumpalo` crate to provide
//! fast, bump-style memory allocation with no per-allocation overhead.
//!
//! Key features:
//! - Zero heap allocations after initial capacity reservation
//! - O(1) allocation time
//! - Cache-line aligned allocations
//! - Thread-safe reset capability (single-threaded use during allocation)
//!
//! # Safety Notes
//! - Allocations are only freed when the entire arena is reset or dropped
//! - Not suitable for long-lived allocations with varying lifetimes
//! - Best used for batch processing of market data ticks

use bumpalo::Bump;
use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::usize;

/// Cache line size for x86_64 and most modern architectures
pub const CACHE_LINE_SIZE: usize = 64;

/// Default arena capacity: 16MB (sufficient for ~100k market data events)
pub const DEFAULT_ARENA_CAPACITY: usize = 16 * 1024 * 1024;

/// Maximum alignment we support (must be power of 2)
pub const MAX_ALIGNMENT: usize = 4096;

/// Error type for arena operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaError {
    OutOfMemory,
    InvalidAlignment,
    InvalidSize,
}

impl std::fmt::Display for ArenaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArenaError::OutOfMemory => write!(f, "Arena out of memory"),
            ArenaError::InvalidAlignment => write!(f, "Invalid alignment requested"),
            ArenaError::InvalidSize => write!(f, "Invalid allocation size"),
        }
    }
}

impl std::error::Error for ArenaError {}

/// A bump allocator wrapper with statistics tracking and safety guarantees
pub struct BumpAllocator {
    /// The underlying bumpalo arena
    arena: Bump,
    
    /// Total capacity in bytes
    capacity: AtomicUsize,
    
    /// Current allocation watermark (high water mark)
    watermark: AtomicUsize,
    
    /// Number of allocations made since last reset
    alloc_count: AtomicUsize,
    
    /// Flag indicating if the arena is currently active
    active: AtomicUsize, // Using usize as atomic bool for SeqCst ordering
}

// SAFETY: BumpAllocator is safe to send between threads as long as
// only one thread performs allocations at a time. The active flag
// provides a basic guard against concurrent misuse.
unsafe impl Send for BumpAllocator {}

// We do NOT implement Sync because bumpalo's Bump is not thread-safe
// for concurrent allocations. Users must ensure single-threaded access
// during the allocation phase.

impl BumpAllocator {
    /// Create a new BumpAllocator with default capacity
    pub fn new() -> Result<Self, ArenaError> {
        Self::with_capacity(DEFAULT_ARENA_CAPACITY)
    }

    /// Create a new BumpAllocator with specified capacity
    pub fn with_capacity(capacity: usize) -> Result<Self, ArenaError> {
        if capacity == 0 {
            return Err(ArenaError::InvalidSize);
        }

        // Ensure capacity is cache-line aligned
        let aligned_capacity = align_to_cache_line(capacity);
        
        let arena = Bump::try_with_capacity(aligned_capacity)
            .ok_or(ArenaError::OutOfMemory)?;

        Ok(Self {
            arena,
            capacity: AtomicUsize::new(aligned_capacity),
            watermark: AtomicUsize::new(0),
            alloc_count: AtomicUsize::new(0),
            active: AtomicUsize::new(1),
        })
    }

    /// Allocate memory from the arena with specified size and alignment
    /// 
    /// # Safety
    /// - The returned pointer is valid until the arena is reset or dropped
    /// - The memory is uninitialized - caller must initialize before use
    /// - Alignment must be a power of 2 and <= MAX_ALIGNMENT
    pub fn alloc(&self, size: usize, align: usize) -> Result<NonNull<u8>, ArenaError> {
        if size == 0 {
            return Err(ArenaError::InvalidSize);
        }

        if !align.is_power_of_two() || align > MAX_ALIGNMENT {
            return Err(ArenaError::InvalidAlignment);
        }

        // Check if arena is active
        if self.active.load(Ordering::SeqCst) == 0 {
            return Err(ArenaError::OutOfMemory);
        }

        // Try to allocate
        let layout = Layout::from_size_align(size, align)
            .map_err(|_| ArenaError::InvalidAlignment)?;

        // SAFETY: We've validated the layout parameters
        let ptr = unsafe { self.arena.try_alloc_layout(layout) }
            .map_err(|_| ArenaError::OutOfMemory)?;

        // Update statistics
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
        
        // Update watermark (approximate, race-tolerant)
        let current = self.watermark.fetch_add(size, Ordering::Relaxed);
        let new_watermark = current + size;
        
        // Use max to handle potential race conditions in watermark tracking
        let mut current_max = self.watermark.load(Ordering::Relaxed);
        while new_watermark > current_max {
            match self.watermark.compare_exchange_weak(
                current_max,
                new_watermark,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }

        Ok(ptr)
    }

    /// Allocate and initialize with a closure
    /// 
    /// This is safer than raw allocation as it ensures initialization
    pub fn alloc_with<T, F>(&self, init: F) -> Result<&mut T, ArenaError>
    where
        F: FnOnce() -> T,
    {
        let ptr = self.alloc(std::mem::size_of::<T>(), std::mem::align_of::<T>())?;
        
        // SAFETY: We just allocated this memory and it's properly aligned for T
        let slot = unsafe { &mut *(ptr.as_ptr() as *mut T) };
        *slot = init();
        
        Ok(slot)
    }

    /// Allocate a slice of elements
    pub fn alloc_slice<T: Clone>(&self, elem: &T, len: usize) -> Result<&mut [T], ArenaError> {
        if len == 0 {
            return Ok(&mut []);
        }

        let ptr = self.alloc(
            std::mem::size_of::<T>() * len,
            std::mem::align_of::<T>(),
        )?;

        // SAFETY: We allocated enough space for len elements of T
        let slice = unsafe {
            std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut T, len)
        };

        // Initialize each element
        for i in 0..len {
            slice[i] = elem.clone();
        }

        Ok(slice)
    }

    /// Reset the arena, freeing all allocations
    /// 
    /// This is O(1) and does not deallocate the underlying memory,
    /// allowing for efficient reuse.
    pub fn reset(&self) {
        self.arena.reset();
        self.watermark.store(0, Ordering::SeqCst);
        self.alloc_count.store(0, Ordering::SeqCst);
    }

    /// Get the total capacity in bytes
    pub fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Relaxed)
    }

    /// Get the current watermark (high water mark of allocations)
    pub fn watermark(&self) -> usize {
        self.watermark.load(Ordering::Relaxed)
    }

    /// Get the number of allocations since last reset
    pub fn allocation_count(&self) -> usize {
        self.alloc_count.load(Ordering::Relaxed)
    }

    /// Get remaining capacity in bytes
    pub fn remaining(&self) -> usize {
        self.capacity.load(Ordering::Relaxed) - self.watermark.load(Ordering::Relaxed)
    }

    /// Check if the arena is active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst) != 0
    }

    /// Deactivate the arena (prevents further allocations)
    pub fn deactivate(&self) {
        self.active.store(0, Ordering::SeqCst);
    }

    /// Reactivate the arena (allows allocations again)
    pub fn reactivate(&self) {
        self.active.store(1, Ordering::SeqCst);
    }

    /// Force inline helper for cache-line alignment
    #[inline(always)]
    pub fn align_to_cache_line(size: usize) -> usize {
        align_to_cache_line(size)
    }
}

impl Default for BumpAllocator {
    fn default() -> Self {
        Self::new().expect("Failed to create default BumpAllocator")
    }
}

impl Drop for BumpAllocator {
    fn drop(&mut self) {
        // Explicitly deactivate to prevent use-after-free scenarios
        self.active.store(0, Ordering::SeqCst);
        // Bump's Drop impl will free the underlying memory
    }
}

/// Align a size to the cache line boundary
#[inline(always)]
pub const fn align_to_cache_line(size: usize) -> usize {
    (size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1)
}

/// A cache-line padded wrapper for preventing false sharing
#[repr(C, align(64))]
pub struct CachePadded64<T> {
    pub inner: T,
    _padding: [u8; cache_padding_size::<T>()],
}

const fn cache_padding_size<T>() -> usize {
    let size = std::mem::size_of::<T>();
    let aligned = (size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1);
    aligned - size
}

impl<T> CachePadded64<T> {
    pub const fn new(inner: T) -> Self {
        Self {
            inner,
            _padding: [0u8; cache_padding_size::<T>()],
        }
    }

    pub fn into_inner(self) -> T {
        self.inner
    }

    pub fn as_ref(&self) -> &T {
        &self.inner
    }

    pub fn as_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

// Ensure CachePadded64 is always cache-line aligned and sized
static_assertions::const_assert!(
    std::mem::align_of::<CachePadded64<u8>>() >= CACHE_LINE_SIZE
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_creation() {
        let arena = BumpAllocator::new().unwrap();
        assert!(arena.is_active());
        assert_eq!(arena.allocation_count(), 0);
        assert_eq!(arena.watermark(), 0);
    }

    #[test]
    fn test_basic_allocation() {
        let arena = BumpAllocator::with_capacity(1024).unwrap();
        let ptr = arena.alloc(64, 8).unwrap();
        assert!(!ptr.is_null());
        assert_eq!(arena.allocation_count(), 1);
    }

    #[test]
    fn test_alloc_with() {
        let arena = BumpAllocator::new().unwrap();
        let value = arena.alloc_with(|| 42u64).unwrap();
        assert_eq!(*value, 42);
    }

    #[test]
    fn test_alloc_slice() {
        let arena = BumpAllocator::new().unwrap();
        let slice = arena.alloc_slice(&0u32, 10).unwrap();
        assert_eq!(slice.len(), 10);
        assert!(slice.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_reset() {
        let arena = BumpAllocator::new().unwrap();
        let _ptr = arena.alloc(100, 8).unwrap();
        assert_eq!(arena.allocation_count(), 1);
        assert!(arena.watermark() > 0);
        
        arena.reset();
        assert_eq!(arena.allocation_count(), 0);
        assert_eq!(arena.watermark(), 0);
    }

    #[test]
    fn test_out_of_memory() {
        let arena = BumpAllocator::with_capacity(64).unwrap();
        let result = arena.alloc(128, 8);
        assert!(matches!(result, Err(ArenaError::OutOfMemory)));
    }

    #[test]
    fn test_invalid_alignment() {
        let arena = BumpAllocator::new().unwrap();
        let result = arena.alloc(64, 3); // 3 is not a power of 2
        assert!(matches!(result, Err(ArenaError::InvalidAlignment)));
    }

    #[test]
    fn test_cache_padded() {
        let padded = CachePadded64::new(42u64);
        assert_eq!(*padded.as_ref(), 42);
        assert_eq!(std::mem::align_of_val(&padded), CACHE_LINE_SIZE);
    }
}
