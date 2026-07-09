//! NEXUS-OMEGA High-Frequency Trading Module
//!
//! This crate provides HFT-specific functionality including:
//! - Market data feed handlers
//! - Order execution engines
//! - Low-latency networking

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
