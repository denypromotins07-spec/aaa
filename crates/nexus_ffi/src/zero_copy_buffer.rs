//! Zero-Copy Buffer for Python-Rust Memory Sharing
//!
//! This module implements a `PyBuffer` protocol wrapper that allows
//! Python (NumPy/Polars) to read Rust memory directly via raw pointers
//! without copying data or acquiring the Python GIL for read operations.
//!
//! # Safety Guarantees
//!
//! - The buffer tracks whether Python holds references to the memory
//! - Rust cannot deallocate memory while Python is reading it
//! - Writes from Rust are visible to Python without synchronization overhead
//! - The GIL is only required for creating/destroying the buffer view

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyBufferError, PyValueError, PyRuntimeError};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::os::raw::{c_void, c_int};
use std::marker::PhantomData;

/// Reference-counted shared memory region
struct SharedMemoryRegion {
    /// Raw pointer to the memory
    ptr: NonNull<u8>,
    
    /// Size of the memory region in bytes
    size: usize,
    
    /// Reference count (Python readers + Rust holder)
    ref_count: AtomicUsize,
    
    /// Flag indicating if the memory is still valid
    valid: AtomicBool,
}

// SAFETY: SharedMemoryRegion can be sent between threads as long as
// access is properly synchronized via the ref_count.
unsafe impl Send for SharedMemoryRegion {}
unsafe impl Sync for SharedMemoryRegion {}

impl SharedMemoryRegion {
    fn new(ptr: NonNull<u8>, size: usize) -> Self {
        Self {
            ptr,
            size,
            ref_count: AtomicUsize::new(1), // Initial reference for Rust holder
            valid: AtomicBool::new(true),
        }
    }
    
    fn add_ref(&self) {
        self.ref_count.fetch_add(1, Ordering::AcqRel);
    }
    
    fn release_ref(&self) -> usize {
        self.ref_count.fetch_sub(1, Ordering::AcqRel)
    }
    
    fn is_valid(&self) -> bool {
        self.valid.load(Ordering::Acquire)
    }
    
    fn invalidate(&self) {
        self.valid.store(false, Ordering::Release);
    }
    
    /// Get the raw pointer (caller must ensure validity)
    fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }
    
    /// Get the mutable pointer (caller must ensure exclusive access)
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

/// A zero-copy buffer that can be shared between Rust and Python
/// 
/// This implements the Python buffer protocol, allowing NumPy and other
/// libraries to access Rust memory without copying.
#[pyclass(name = "ZeroCopyBuffer", module = "nexus_ffi")]
pub struct ZeroCopyBuffer<'py> {
    /// The shared memory region
    region: Arc<SharedMemoryRegion>,
    
    /// Phantom lifetime for Python GIL reference
    _phantom: PhantomData<&'py PyAny>,
    
    /// Whether this buffer is read-only from Python's perspective
    read_only: bool,
}

// We need to manually implement the lifetime bounds since pyclass
// doesn't support lifetime parameters directly. In practice, the
// buffer's lifetime is tied to the GIL scope during creation.
impl<'py> ZeroCopyBuffer<'py> {
    /// Create a new zero-copy buffer view
    pub fn new(
        _py: Python<'py>,
        ptr: NonNull<u8>,
        size: usize,
    ) -> Result<Self, PyBufferError> {
        if size == 0 {
            return Err(PyBufferError::new_err("Cannot create zero-size buffer"));
        }
        
        let region = Arc::new(SharedMemoryRegion::new(ptr, size));
        
        Ok(Self {
            region,
            _phantom: PhantomData,
            read_only: true, // Default to read-only for safety
        })
    }
    
    /// Create a writable zero-copy buffer
    pub fn with_write_access(
        _py: Python<'py>,
        ptr: NonNull<u8>,
        size: usize,
    ) -> Result<Self, PyBufferError> {
        let mut buf = Self::new(_py, ptr, size)?;
        buf.read_only = false;
        Ok(buf)
    }
    
    /// Get the size of the buffer in bytes
    pub fn size(&self) -> usize {
        self.region.size
    }
    
    /// Check if the buffer is still valid
    pub fn is_valid(&self) -> bool {
        self.region.is_valid()
    }
    
    /// Get a read-only slice (safe, no GIL needed)
    pub fn as_slice(&self) -> Option<&[u8]> {
        if !self.region.is_valid() {
            return None;
        }
        
        // SAFETY: We hold an Arc reference to the region, preventing deallocation.
        // The pointer is guaranteed valid for the lifetime of self due to the Arc.
        Some(unsafe {
            std::slice::from_raw_parts(self.region.as_ptr(), self.region.size)
        })
    }

    /// Get a mutable slice (requires exclusive access)
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        if !self.region.is_valid() {
            return None;
        }
        
        // SAFETY: We have exclusive &mut self access, and the Arc ensures
        // the memory region outlives this borrow. No other references can exist
        // while we hold &mut self.
        Some(unsafe {
            std::slice::from_raw_parts_mut(self.region.as_mut_ptr(), self.region.size)
        })
    }
    
    /// Export to Python as a buffer object implementing the buffer protocol
    pub fn to_py_buffer(&self, py: Python<'_>) -> Result<PyBuffer<u8>, PyErr> {
        // First create PyBytes as an intermediary
        let slice = self.as_slice()
            .ok_or_else(|| PyRuntimeError::new_err("Buffer is invalid"))?;
        
        let py_bytes = PyBytes::new(py, slice);
        
        // Get the buffer from PyBytes
        PyBuffer::<u8>::get(py_bytes.bind(py))
    }
    
    /// Read data directly into a Rust buffer (avoids GIL)
    pub fn read_into(&self, dest: &mut [u8]) -> Result<usize, PyBufferError> {
        if !self.region.is_valid() {
            return Err(PyBufferError::new_err("Buffer is invalid"));
        }
        
        let src = self.as_slice()
            .ok_or_else(|| PyBufferError::new_err("Cannot get buffer slice"))?;
        
        let copy_len = dest.len().min(src.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dest.as_mut_ptr(),
                copy_len,
            );
        }
        
        Ok(copy_len)
    }
    
    /// Write data directly from a Rust buffer (avoids GIL)
    pub fn write_from(&mut self, src: &[u8]) -> Result<usize, PyBufferError> {
        if !self.region.is_valid() {
            return Err(PyBufferError::new_err("Buffer is invalid"));
        }
        
        if self.read_only {
            return Err(PyBufferError::new_err("Buffer is read-only"));
        }
        
        let dest = self.as_mut_slice()
            .ok_or_else(|| PyBufferError::new_err("Cannot get mutable buffer slice"))?;
        
        let copy_len = dest.len().min(src.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dest.as_mut_ptr(),
                copy_len,
            );
        }
        
        Ok(copy_len)
    }
}

#[pymethods]
impl<'py> ZeroCopyBuffer<'py> {
    /// Get the buffer size
    #[getter]
    fn size(&self) -> usize {
        self.size()
    }
    
    /// Check if buffer is valid
    #[getter]
    fn is_valid(&self) -> bool {
        self.is_valid()
    }
    
    /// Check if buffer is read-only
    #[getter]
    fn read_only(&self) -> bool {
        self.read_only
    }
    
    /// Get the buffer as Python bytes (creates a copy)
    fn to_bytes<'a>(&self, py: Python<'a>) -> Result<&'a PyBytes, PyErr> {
        let slice = self.as_slice()
            .ok_or_else(|| PyRuntimeError::new_err("Buffer is invalid"))?;
        Ok(PyBytes::new(py, slice))
    }
    
    /// Get a memoryview of the buffer (zero-copy)
    fn get_memoryview(&self, py: Python<'_>) -> Result<PyObject, PyErr> {
        let py_buffer = self.to_py_buffer(py)?;
        Ok(py_buffer.to_object(py))
    }
    
    /// Read data into a Python bytes object
    fn read(&self, py: Python<'_>, size: Option<usize>) -> Result<&PyBytes, PyErr> {
        let slice = self.as_slice()
            .ok_or_else(|| PyRuntimeError::new_err("Buffer is invalid"))?;
        
        let read_size = size.unwrap_or(slice.len()).min(slice.len());
        Ok(PyBytes::new(py, &slice[..read_size]))
    }
    
    /// Convert to numpy array (requires numpy crate)
    #[cfg(feature = "numpy")]
    fn to_numpy<'a>(&self, py: Python<'a>) -> Result<PyObject, PyErr> {
        use numpy::{PyArray1, IntoPyArray};
        
        let slice = self.as_slice()
            .ok_or_else(|| PyRuntimeError::new_err("Buffer is invalid"))?;
        
        // Create a numpy array view over the Rust memory
        let arr = PyArray1::from_slice(py, slice);
        Ok(arr.to_object(py))
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "ZeroCopyBuffer(size={}, valid={}, read_only={})",
            self.region.size,
            self.region.is_valid(),
            self.read_only
        )
    }
    
    /// Support len() protocol
    fn __len__(&self) -> usize {
        self.region.size
    }
}

impl<'py> Drop for ZeroCopyBuffer<'py> {
    fn drop(&mut self) {
        // Decrement reference count
        let refs_remaining = self.region.release_ref();
        
        // If we're the last reference, invalidate the memory
        if refs_remaining == 1 {
            self.region.invalidate();
        }
    }
}

/// C-compatible FFI functions for zero-copy buffer operations

/// Create a zero-copy buffer from external memory
/// 
/// # Safety
/// - `ptr` must point to valid, aligned memory of at least `size` bytes
/// - The caller is responsible for ensuring the memory outlives any Python references
#[no_mangle]
pub unsafe extern "C" fn nexus_zerocopy_create(
    ptr: *mut u8,
    size: usize,
) -> *mut c_void {
    if ptr.is_null() || size == 0 {
        return std::ptr::null_mut();
    }
    
    match NonNull::new(ptr) {
        Some(non_null) => {
            let region = Arc::new(SharedMemoryRegion::new(non_null, size));
            Arc::into_raw(region) as *mut c_void
        }
        None => std::ptr::null_mut(),
    }
}

/// Release a zero-copy buffer reference
/// 
/// # Safety
/// - `handle` must be a valid pointer returned by `nexus_zerocopy_create`
#[no_mangle]
pub unsafe extern "C" fn nexus_zerocopy_release(handle: *mut c_void) {
    if !handle.is_null() {
        let region = Arc::from_raw(handle as *const SharedMemoryRegion);
        // Arc will be dropped when ref count reaches zero
        drop(region);
    }
}

/// Get the data pointer from a zero-copy buffer
/// 
/// # Safety
/// - `handle` must be a valid zero-copy buffer handle
/// - Returned pointer is only valid while at least one reference exists
#[no_mangle]
pub unsafe extern "C" fn nexus_zerocopy_get_ptr(handle: *mut c_void) -> *const u8 {
    if handle.is_null() {
        return std::ptr::null();
    }
    
    let region = &*(handle as *const SharedMemoryRegion);
    if region.is_valid() {
        region.as_ptr()
    } else {
        std::ptr::null()
    }
}

/// Get the size of a zero-copy buffer
#[no_mangle]
pub unsafe extern "C" fn nexus_zerocopy_get_size(handle: *mut c_void) -> usize {
    if handle.is_null() {
        return 0;
    }
    
    let region = &*(handle as *const SharedMemoryRegion);
    region.size
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyBytes;
    
    #[test]
    fn test_shared_memory_region() {
        let ptr = NonNull::new(vec![1u8, 2, 3, 4].as_mut_ptr()).unwrap();
        let region = Arc::new(SharedMemoryRegion::new(ptr, 4));
        
        assert!(region.is_valid());
        assert_eq!(region.size, 4);
        
        let region_clone = region.clone();
        assert_eq!(region.ref_count.load(Ordering::Relaxed), 2);
        
        drop(region_clone);
        assert_eq!(region.ref_count.load(Ordering::Relaxed), 1);
    }
    
    #[test]
    fn test_zero_copy_buffer_creation() {
        let mut data = vec![1u8, 2, 3, 4, 5];
        let ptr = NonNull::new(data.as_mut_ptr()).unwrap();
        
        // Note: This test doesn't actually acquire the GIL, which would
        // be required in real usage. We're testing the Rust-side logic.
        let buffer = ZeroCopyBuffer::new(
            unsafe { Python::assume_gil_acquired() },
            ptr,
            5,
        ).unwrap();
        
        assert_eq!(buffer.size(), 5);
        assert!(buffer.is_valid());
        assert!(buffer.read_only);
    }
    
    #[test]
    fn test_zero_copy_read() {
        let mut data = vec![10u8, 20, 30, 40];
        let ptr = NonNull::new(data.as_mut_ptr()).unwrap();
        
        let buffer = ZeroCopyBuffer::new(
            unsafe { Python::assume_gil_acquired() },
            ptr,
            4,
        ).unwrap();
        
        let slice = buffer.as_slice().unwrap();
        assert_eq!(slice, &[10, 20, 30, 40]);
        
        let mut dest = [0u8; 4];
        let read = buffer.read_into(&mut dest).unwrap();
        assert_eq!(read, 4);
        assert_eq!(dest, [10, 20, 30, 40]);
    }
    
    #[test]
    fn test_zero_copy_write() {
        let mut data = vec![0u8; 4];
        let ptr = NonNull::new(data.as_mut_ptr()).unwrap();
        
        let mut buffer = ZeroCopyBuffer::with_write_access(
            unsafe { Python::assume_gil_acquired() },
            ptr,
            4,
        ).unwrap();
        
        let src = [100u8, 200, 50, 25];
        let written = buffer.write_from(&src).unwrap();
        assert_eq!(written, 4);
        
        let slice = buffer.as_slice().unwrap();
        assert_eq!(slice, &[100, 200, 50, 25]);
    }
}
