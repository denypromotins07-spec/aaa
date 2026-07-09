//! Cache-line padding utilities for preventing false sharing
//!
//! False sharing occurs when multiple threads access different variables
//! that reside on the same cache line, causing unnecessary cache invalidations.
//!
//! This module provides:
//! - `CachePadded<T>`: A generic wrapper that pads any type to a full cache line
//! - `CachePadded64<T>`: Explicit 64-byte cache line padding (x86_64 standard)
//! - `CachePadded128<T>`: 128-byte padding for architectures with larger cache lines
//! - Alignment assertions and utilities
//!
//! # Usage Example
//!
//! ```rust
//! use nexus_core::memory::cache_padder::CachePadded64;
//! use std::sync::atomic::{AtomicUsize, Ordering};
//!
//! // Prevent false sharing between two atomic counters
//! struct Counters {
//!     producer_count: CachePadded64<AtomicUsize>,
//!     consumer_count: CachePadded64<AtomicUsize>,
//! }
//! ```

use std::fmt;
use std::ops::{Deref, DerefMut};
use std::ptr;

/// Standard cache line size for x86_64 (also used by most ARM64 implementations)
pub const CACHE_LINE_SIZE: usize = 64;

/// Extended cache line size for some ARM server processors
pub const CACHE_LINE_SIZE_LARGE: usize = 128;

/// Trait for types that can be cache-line padded
pub trait CachePadded: Sized {
    /// The padded wrapper type
    type Padded;
    
    /// Wrap self in cache-line padding
    fn padded(self) -> Self::Padded;
}

/// A value padded to occupy exactly one cache line (64 bytes)
/// 
/// This ensures that values of type `T` don't share cache lines with
/// other data, preventing false sharing in concurrent scenarios.
#[repr(C)]
pub struct CachePadded64<T> {
    /// The actual value, aligned to cache line boundary
    #[align(64)]
    pub inner: T,
    /// Padding to fill the rest of the cache line
    _pad: [u8; cache_padding_size_64::<T>()],
}

/// A value padded to occupy exactly 128 bytes (for larger cache lines)
#[repr(C)]
pub struct CachePadded128<T> {
    /// The actual value, aligned to 128-byte boundary
    #[align(128)]
    pub inner: T,
    /// Padding to fill the rest of the cache line
    _pad: [u8; cache_padding_size_128::<T>()],
}

/// Calculate padding needed for 64-byte cache lines
const fn cache_padding_size_64<T>() -> usize {
    let size = std::mem::size_of::<T>();
    let aligned_size = (size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1);
    aligned_size - size
}

/// Calculate padding needed for 128-byte cache lines
const fn cache_padding_size_128<T>() -> usize {
    let size = std::mem::size_of::<T>();
    let aligned_size = (size + CACHE_LINE_SIZE_LARGE - 1) & !(CACHE_LINE_SIZE_LARGE - 1);
    aligned_size - size
}

// Manual implementations since const generics in array sizes need special handling
impl<T> CachePadded64<T> {
    /// Create a new cache-padded value
    pub const fn new(value: T) -> Self 
    where
        T: Sized,
    {
        // We need to use unsafe here because const evaluation of padding
        // is not yet stable for all cases
        Self {
            inner: value,
            _pad: unsafe { create_zero_array::<{ cache_padding_size_64::<T>() }>() },
        }
    }
    
    /// Extract the inner value
    pub fn into_inner(self) -> T {
        self.inner
    }
    
    /// Get a reference to the inner value
    pub fn as_ref(&self) -> &T {
        &self.inner
    }
    
    /// Get a mutable reference to the inner value
    pub fn as_mut(&mut self) -> &mut T {
        &mut self.inner
    }
    
    /// Get the alignment of this structure
    pub const fn alignment() -> usize {
        CACHE_LINE_SIZE
    }
    
    /// Get the total size including padding
    pub const fn total_size() -> usize {
        let size = std::mem::size_of::<T>();
        (size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1)
    }
}

impl<T> CachePadded128<T> {
    /// Create a new cache-padded value with 128-byte alignment
    pub const fn new(value: T) -> Self
    where
        T: Sized,
    {
        Self {
            inner: value,
            _pad: unsafe { create_zero_array::<{ cache_padding_size_128::<T>() }>() },
        }
    }
    
    /// Extract the inner value
    pub fn into_inner(self) -> T {
        self.inner
    }
    
    /// Get a reference to the inner value
    pub fn as_ref(&self) -> &T {
        &self.inner
    }
    
    /// Get a mutable reference to the inner value
    pub fn as_mut(&mut self) -> &mut T {
        &mut self.inner
    }
    
    /// Get the alignment of this structure
    pub const fn alignment() -> usize {
        CACHE_LINE_SIZE_LARGE
    }
    
    /// Get the total size including padding
    pub const fn total_size() -> usize {
        let size = std::mem::size_of::<T>();
        (size + CACHE_LINE_SIZE_LARGE - 1) & !(CACHE_LINE_SIZE_LARGE - 1)
    }
}

/// Helper function to create a zero-initialized array at compile time
/// 
/// # Safety
/// This function uses transmute internally and should only be called
/// with valid array sizes.
#[inline(always)]
const unsafe fn create_zero_array<const N: usize>() -> [u8; N] {
    // SAFETY: Zero-initialization is valid for u8 arrays
    [0u8; N]
}

impl<T> Deref for CachePadded64<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for CachePadded64<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T> Deref for CachePadded128<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for CachePadded128<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<T: Clone> Clone for CachePadded64<T> {
    fn clone(&self) -> Self {
        Self::new(self.inner.clone())
    }
}

impl<T: Clone> Clone for CachePadded128<T> {
    fn clone(&self) -> Self {
        Self::new(self.inner.clone())
    }
}

impl<T: fmt::Debug> fmt::Debug for CachePadded64<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CachePadded64")
            .field("inner", &self.inner)
            .field("alignment", &Self::alignment())
            .field("total_size", &Self::total_size())
            .finish()
    }
}

impl<T: fmt::Debug> fmt::Debug for CachePadded128<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CachePadded128")
            .field("inner", &self.inner)
            .field("alignment", &Self::alignment())
            .field("total_size", &Self::total_size())
            .finish()
    }
}

impl<T: PartialEq> PartialEq for CachePadded64<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<T: PartialEq> PartialEq for CachePadded128<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<T: Eq> Eq for CachePadded64<T> {}
impl<T: Eq> Eq for CachePadded128<T> {}

// Ensure Send and Sync if T is Send/Sync
unsafe impl<T: Send> Send for CachePadded64<T> {}
unsafe impl<T: Sync> Sync for CachePadded64<T> {}
unsafe impl<T: Send> Send for CachePadded128<T> {}
unsafe impl<T: Sync> Sync for CachePadded128<T> {}

/// Align a pointer to cache line boundary
/// 
/// # Safety
/// The returned pointer may point to unallocated memory.
/// Caller must ensure proper allocation before use.
#[inline(always)]
pub unsafe fn align_ptr_to_cache_line<T>(ptr: *mut T) -> *mut T {
    let addr = ptr as usize;
    let aligned = (addr + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1);
    aligned as *mut T
}

/// Check if a reference is cache-line aligned
#[inline(always)]
pub fn is_cache_line_aligned<T>(val: &T) -> bool {
    let addr = val as *const T as usize;
    addr % CACHE_LINE_SIZE == 0
}

/// Assert at compile time that a type has the expected alignment
#[macro_export]
macro_rules! assert_cache_aligned {
    ($type:ty, $align:expr) => {
        const _: () = assert!(
            std::mem::align_of::<$type>() >= $align,
            concat!("Type ", stringify!($type), " does not have required alignment of ", stringify!($align))
        );
    };
    ($type:ty) => {
        assert_cache_aligned!($type, CACHE_LINE_SIZE);
    };
}

/// Runtime assertion that a value is cache-line aligned
#[inline(always)]
pub fn assert_cache_line_aligned_runtime<T>(val: &T, name: &str) {
    let addr = val as *const T as usize;
    if addr % CACHE_LINE_SIZE != 0 {
        panic!(
            "Value '{}' at address 0x{:x} is not cache-line aligned (expected multiple of {})",
            name, addr, CACHE_LINE_SIZE
        );
    }
}

/// Padding-only struct for explicit control over struct layout
/// 
/// Use this to manually pad structures to cache line boundaries:
/// 
/// ```rust
/// struct MyStruct {
///     important_value: u64,
///     _padding: CachePadding<56>, // Pad to 64 bytes total
/// }
/// ```
#[repr(C)]
pub struct CachePadding<const N: usize>([u8; N]);

impl<const N: usize> CachePadding<N> {
    /// Create a new cache padding instance
    pub const fn new() -> Self {
        Self([0u8; N])
    }
    
    /// Get the size of the padding
    pub const fn size() -> usize {
        N
    }
}

impl<const N: usize> Default for CachePadding<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_cache_padded64_alignment() {
        let padded = CachePadded64::new(42u64);
        assert!(is_cache_line_aligned(&padded));
        assert_eq!(std::mem::align_of_val(&padded), CACHE_LINE_SIZE);
    }

    #[test]
    fn test_cache_padded128_alignment() {
        let padded = CachePadded128::new(42u64);
        assert_eq!(std::mem::align_of_val(&padded), CACHE_LINE_SIZE_LARGE);
    }

    #[test]
    fn test_cache_padded64_small_type() {
        let padded = CachePadded64::new(1u8);
        assert_eq!(*padded, 1);
        assert_eq!(std::mem::size_of_val(&padded), CACHE_LINE_SIZE);
    }

    #[test]
    fn test_cache_padded64_large_type() {
        let padded = CachePadded64::new([0u64; 16]); // 128 bytes
        assert_eq!(std::mem::size_of_val(&padded), 128); // 2 cache lines
    }

    #[test]
    fn test_cache_padded_atomic() {
        let padded = CachePadded64::new(AtomicUsize::new(100));
        padded.fetch_add(1, Ordering::Relaxed);
        assert_eq!(padded.load(Ordering::Relaxed), 101);
    }

    #[test]
    fn test_deref() {
        let mut padded = CachePadded64::new(42u64);
        assert_eq!(*padded, 42);
        *padded = 100;
        assert_eq!(*padded, 100);
    }

    #[test]
    fn test_into_inner() {
        let padded = CachePadded64::new(42u64);
        let inner = padded.into_inner();
        assert_eq!(inner, 42);
    }

    #[test]
    fn test_clone() {
        let padded = CachePadded64::new(42u64);
        let cloned = padded.clone();
        assert_eq!(*cloned, 42);
    }

    #[test]
    fn test_debug() {
        let padded = CachePadded64::new(42u64);
        let debug_str = format!("{:?}", padded);
        assert!(debug_str.contains("CachePadded64"));
        assert!(debug_str.contains("42"));
    }

    #[test]
    fn test_partial_eq() {
        let a = CachePadded64::new(42u64);
        let b = CachePadded64::new(42u64);
        let c = CachePadded64::new(43u64);
        
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_align_ptr() {
        let mut buffer = [0u8; 256];
        let ptr = buffer.as_mut_ptr();
        
        // SAFETY: Buffer is properly allocated
        let aligned = unsafe { align_ptr_to_cache_line(ptr) };
        let offset = aligned as usize - ptr as usize;
        assert!(offset < CACHE_LINE_SIZE);
        assert_eq!(aligned as usize % CACHE_LINE_SIZE, 0);
    }

    #[test]
    fn test_cache_padding_explicit() {
        struct TestStruct {
            value: u64,
            _pad: CachePadding<56>,
        }
        
        assert_eq!(std::mem::size_of::<TestStruct>(), 64);
    }

    #[test]
    #[should_panic(expected = "not cache-line aligned")]
    fn test_runtime_assertion_fails() {
        let value = 42u64;
        // Force it to be potentially misaligned by taking address of field
        struct Misaligned {
            _byte: u8,
            value: u64,
        }
        let m = Misaligned { _byte: 0, value };
        assert_cache_line_aligned_runtime(&m.value, "test_value");
    }

    #[test]
    fn test_constants() {
        assert_eq!(CACHE_LINE_SIZE, 64);
        assert_eq!(CACHE_LINE_SIZE_LARGE, 128);
    }
}
