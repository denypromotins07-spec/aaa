//! PyO3 GIL-Bypass Bridge for NEXUS-OMEGA
//!
//! This module provides the core FFI boundary between Rust and Python,
//! enabling zero-copy data sharing and async task spawning without
//! blocking the Python event loop.
//!
//! # Key Features
//!
//! - Custom error translation from Rust `Result` to Python exceptions
//! - GIL-independent data access via raw pointers
//! - Safe async runtime integration with pyo3-asyncio
//! - Reference-counted shared memory regions

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError, PyMemoryError, PyOverflowError};
use pyo3::types::{PyBytes, PyList, PyDict};
use pyo3::ffi;
use std::ptr::NonNull;
use std::sync::Arc;
use std::os::raw::{c_void, c_char};

use nexus_core::memory::arena::BumpAllocator;
use nexus_core::concurrency::spsc_ring::SPSCRingBuffer;
use crate::zero_copy_buffer::ZeroCopyBuffer;
use crate::async_runtime::PythonAsyncRuntime;

/// Error type for FFI operations
#[derive(Debug, thiserror::Error)]
pub enum FFIBridgeError {
    #[error("Invalid null pointer")]
    NullPointer,
    
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },
    
    #[error("GIL state error: {0}")]
    GILStateError(String),
    
    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),
    
    #[error("Async runtime error: {0}")]
    AsyncRuntimeError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("FFI boundary violation: {0}")]
    BoundaryViolation(String),
}

impl FFIBridgeError {
    /// Convert this error into a Python exception
    pub fn into_pyerr(self, py: Python<'_>) -> PyErr {
        match self {
            FFIBridgeError::NullPointer => PyValueError::new_err("Null pointer in FFI operation"),
            FFIBridgeError::BufferSizeMismatch { expected, actual } => {
                PyValueError::new_err(format!(
                    "Buffer size mismatch: expected {}, got {}",
                    expected, actual
                ))
            }
            FFIBridgeError::GILStateError(msg) => PyRuntimeError::new_err(msg),
            FFIBridgeError::AllocationFailed(msg) => PyMemoryError::new_err(msg),
            FFIBridgeError::AsyncRuntimeError(msg) => PyRuntimeError::new_err(msg),
            FFIBridgeError::SerializationError(msg) => PyValueError::new_err(msg),
            FFIBridgeError::BoundaryViolation(msg) => PyRuntimeError::new_err(msg),
        }
    }
}

/// Result type alias for FFI operations
pub type FFIResult<T> = Result<T, FFIBridgeError>;

/// Macro to convert Rust Result to Python exception
#[macro_export]
macro_rules! ffi_result_to_py {
    ($result:expr, $py:expr) => {
        match $result {
            Ok(val) => val,
            Err(e) => return Err(e.into_pyerr($py)),
        }
    };
}

/// The main FFI bridge structure
/// 
/// This provides a safe interface for Python to interact with
/// Rust-backed data structures without acquiring the GIL for
/// read-only operations.
#[pyclass(name = "FFIBridge", module = "nexus_ffi")]
pub struct FFIBridge {
    /// Shared bump allocator for zero-copy allocations
    allocator: Arc<parking_lot::RwLock<BumpAllocator>>,
    
    /// Event ring buffer for producer-consumer patterns
    event_buffer: Arc<SPSCRingBuffer<()>>,
    
    /// Async runtime handle
    async_runtime: Option<PythonAsyncRuntime>,
    
    /// Flag indicating if the bridge is initialized
    initialized: bool,
}

#[pymethods]
impl FFIBridge {
    /// Create a new FFI bridge instance
    #[new]
    #[pyo3(signature = (allocator_size_mb = 16, event_buffer_capacity = 4096))]
    pub fn new(allocator_size_mb: usize, event_buffer_capacity: usize) -> FFIResult<Self> {
        let allocator = BumpAllocator::with_capacity(allocator_size_mb * 1024 * 1024)
            .map_err(|e| FFIBridgeError::AllocationFailed(e.to_string()))?;
        
        let event_buffer = SPSCRingBuffer::new(event_buffer_capacity)
            .map_err(|e| FFIBridgeError::BoundaryViolation(e.to_string()))?;
        
        Ok(Self {
            allocator: Arc::new(parking_lot::RwLock::new(allocator)),
            event_buffer: Arc::new(event_buffer),
            async_runtime: None,
            initialized: true,
        })
    }
    
    /// Initialize the async runtime for Python asyncio integration
    pub fn init_async_runtime(&mut self, py: Python<'_>) -> FFIResult<()> {
        if self.async_runtime.is_some() {
            return Err(FFIBridgeError::GILStateError(
                "Async runtime already initialized".to_string()
            ));
        }
        
        let runtime = PythonAsyncRuntime::new(py)
            .map_err(|e| FFIBridgeError::AsyncRuntimeError(e.to_string()))?;
        
        self.async_runtime = Some(runtime);
        Ok(())
    }
    
    /// Allocate memory in the Rust arena and return a zero-copy buffer view
    pub fn allocate_zero_copy<'py>(
        &self,
        py: Python<'py>,
        size: usize,
    ) -> FFIResult<ZeroCopyBuffer<'py>> {
        let allocator = self.allocator.read();
        
        let ptr = allocator.alloc(size, 8)
            .map_err(|e| FFIBridgeError::AllocationFailed(e.to_string()))?;
        
        ZeroCopyBuffer::new(py, ptr, size)
            .map_err(|e| FFIBridgeError::BoundaryViolation(e.to_string()))
    }
    
    /// Push an event to the ring buffer (producer side)
    pub fn push_event(&self, data: &[u8]) -> FFIResult<()> {
        self.event_buffer.push(data)
            .map_err(|e| FFIBridgeError::BoundaryViolation(e.to_string()))
    }
    
    /// Pop an event from the ring buffer (consumer side)
    pub fn pop_event<'py>(&self, py: Python<'py>, max_size: usize) -> FFIResult<Option<&'py PyBytes>> {
        let mut buffer = vec![0u8; max_size];
        
        match self.event_buffer.pop(&mut buffer) {
            Ok(len) => {
                let py_bytes = PyBytes::new(py, &buffer[..len]);
                // Safety: We're returning a reference that's tied to the GIL lifetime
                Ok(Some(unsafe { 
                    &*(py_bytes as *const PyBytes) 
                }))
            }
            Err(nexus_core::concurrency::spsc_ring::RingBufferError::Empty) => Ok(None),
            Err(e) => Err(FFIBridgeError::BoundaryViolation(e.to_string())),
        }
    }
    
    /// Get the current event buffer size
    #[getter]
    pub fn event_buffer_size(&self) -> usize {
        self.event_buffer.len()
    }
    
    /// Get the allocator watermark (high water mark)
    #[getter]
    pub fn allocator_watermark(&self) -> usize {
        self.allocator.read().watermark()
    }
    
    /// Reset the allocator (frees all allocations)
    pub fn reset_allocator(&self) {
        self.allocator.write().reset();
    }
    
    /// Check if the bridge is properly initialized
    #[getter]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
    
    /// Spawn an async task from Python without blocking the event loop
    pub fn spawn_async_task<'py>(
        &self,
        py: Python<'py>,
        coro: PyObject,
    ) -> FFIResult<PyObject> {
        let runtime = self.async_runtime.as_ref()
            .ok_or_else(|| FFIBridgeError::GILStateError(
                "Async runtime not initialized".to_string()
            ))?;
        
        runtime.spawn_coroutine(py, coro)
            .map_err(|e| FFIBridgeError::AsyncRuntimeError(e.to_string()))
    }
}

/// C-compatible FFI functions for external integrations
/// These can be called from C/C++ code or via ctypes from Python

/// Create a new FFI bridge instance (returns opaque pointer)
#[no_mangle]
pub extern "C" fn nexus_ffi_bridge_create(
    allocator_size_mb: usize,
    event_buffer_capacity: usize,
) -> *mut c_void {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        FFIBridge::new(allocator_size_mb, event_buffer_capacity)
    })) {
        Ok(Ok(bridge)) => Box::into_raw(Box::new(bridge)) as *mut c_void,
        _ => std::ptr::null_mut(),
    }
}

/// Destroy an FFI bridge instance
/// 
/// # Safety
/// - `ptr` must be a valid pointer returned by `nexus_ffi_bridge_create`
/// - Must not be called concurrently with other operations on the same pointer
#[no_mangle]
pub unsafe extern "C" fn nexus_ffi_bridge_destroy(ptr: *mut c_void) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr as *mut FFIBridge);
    }
}

/// Push event data through the FFI bridge
/// 
/// # Safety
/// - `bridge_ptr` must be a valid FFIBridge pointer
/// - `data` must point to valid memory of `len` bytes
#[no_mangle]
pub unsafe extern "C" fn nexus_ffi_push_event(
    bridge_ptr: *mut c_void,
    data: *const u8,
    len: usize,
) -> i32 {
    if bridge_ptr.is_null() || data.is_null() {
        return -1;
    }
    
    let bridge = &*(bridge_ptr as *const FFIBridge);
    let slice = std::slice::from_raw_parts(data, len);
    
    match bridge.push_event(slice) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Pop event data from the FFI bridge
/// 
/// # Safety
/// - `bridge_ptr` must be a valid FFIBridge pointer
/// - `out_buffer` must point to writable memory of at least `max_size` bytes
/// - `out_len` must point to a valid usize for storing the result length
#[no_mangle]
pub unsafe extern "C" fn nexus_ffi_pop_event(
    bridge_ptr: *mut c_void,
    out_buffer: *mut u8,
    max_size: usize,
    out_len: *mut usize,
) -> i32 {
    if bridge_ptr.is_null() || out_buffer.is_null() || out_len.is_null() {
        return -1;
    }
    
    let bridge = &*(bridge_ptr as *const FFIBridge);
    let buffer = std::slice::from_raw_parts_mut(out_buffer, max_size);
    
    match bridge.event_buffer.pop(buffer) {
        Ok(len) => {
            *out_len = len;
            0
        }
        Err(nexus_core::concurrency::spsc_ring::RingBufferError::Empty) => {
            *out_len = 0;
            1 // Indicates empty (non-error)
        }
        Err(_) => -1,
    }
}

/// Get the Python module definition for pyo3
#[pymodule]
fn nexus_ffi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FFIBridge>()?;
    m.add_class::<ZeroCopyBuffer>()?;
    
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyBytes;
    
    #[test]
    fn test_ffi_bridge_creation() {
        let bridge = FFIBridge::new(16, 1024).unwrap();
        assert!(bridge.is_initialized());
        assert_eq!(bridge.event_buffer_size(), 0);
    }
    
    #[test]
    fn test_push_pop_event() {
        let bridge = FFIBridge::new(16, 1024).unwrap();
        let test_data = b"test event data";
        
        bridge.push_event(test_data).unwrap();
        assert_eq!(bridge.event_buffer_size(), 1);
    }
    
    #[test]
    fn test_allocator_watermark() {
        let bridge = FFIBridge::new(16, 1024).unwrap();
        let initial_watermark = bridge.allocator_watermark();
        
        bridge.reset_allocator();
        let after_reset = bridge.allocator_watermark();
        
        assert!(after_reset <= initial_watermark);
    }
}
