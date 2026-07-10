//! NEXUS-OMEGA FFI Layer
//!
//! This crate provides the Python-Rust FFI boundary for the NEXUS-OMEGA
//! high-frequency trading system. It enables:
//!
//! - Zero-copy memory sharing between Rust and Python
//! - Async task spawning without blocking Python's event loop
//! - Safe error translation between Rust and Python
//! - C-compatible FFI functions for external integrations

pub mod bridge;
pub mod zero_copy_buffer;
pub mod async_runtime;

// Re-export commonly used types
pub use bridge::{FFIBridge, FFIBridgeError, FFIResult};
pub use zero_copy_buffer::ZeroCopyBuffer;
pub use async_runtime::{PythonAsyncRuntime, AsyncRuntimeError, AsyncResult};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize logging for the FFI layer
pub fn init_logging() {
    tracing_subscriber::fmt::init();
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
