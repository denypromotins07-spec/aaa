//! Memory Pinning with mlock
//! Pins memory pages to physical RAM to prevent swapping

use std::io;
use tracing::{warn, error};

/// Security level achieved by the system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityLevel {
    /// Full security: mlock successful, secrets pinned to RAM
    Production,
    /// Degraded security: mlock failed, secrets may be swapped
    Degraded,
}

/// MemoryPinner - Handles OS-level memory locking via mlock
pub struct MemoryPinner {
    page_size: usize,
    current_security_level: SecurityLevel,
}

impl MemoryPinner {
    pub fn new() -> Self {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        
        Self {
            page_size,
            current_security_level: SecurityLevel::Degraded, // Default until proven otherwise
        }
    }

    /// Get the page size of the system
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Get current security level
    pub fn security_level(&self) -> SecurityLevel {
        self.current_security_level
    }

    /// Attempt to pin a single page of memory
    /// 
    /// Returns Ok(SecurityLevel) on success, Err on failure
    pub fn pin_page(&mut self) -> Result<SecurityLevel, io::Error> {
        // Allocate aligned memory for pinning test
        let mut test_buffer: Vec<u8> = vec![0u8; self.page_size];
        let ptr = test_buffer.as_mut_ptr();
        
        // Attempt mlock
        let result = unsafe {
            libc::mlock(ptr as *const libc::c_void, self.page_size)
        };

        if result == 0 {
            // Success - unlock immediately after test
            unsafe {
                libc::munlock(ptr as *const libc::c_void, self.page_size);
            }
            self.current_security_level = SecurityLevel::Production;
            Ok(SecurityLevel::Production)
        } else {
            // Failed - get error code
            let err = io::Error::last_os_error();
            
            // Check specific error codes
            match err.raw_os_error() {
                Some(libc::EPERM) => {
                    warn!("mlock failed: EPERM - Insufficient privileges (missing IPC_LOCK capability)");
                    warn!("Run with: docker run --cap-add=IPC_LOCK ...");
                }
                Some(libc::ENOMEM) => {
                    warn!("mlock failed: ENOMEM - Not enough lockable memory");
                    warn!("Increase ulimit -l or /proc/sys/vm/max_map_count");
                }
                _ => {}
            }
            
            Err(err)
        }
    }

    /// Pin a buffer to physical RAM
    /// 
    /// The buffer must remain valid for the duration of the pin.
    /// Call `unpin_buffer` before dropping the buffer.
    pub fn pin_buffer(&self, buffer: &[u8]) -> Result<(), io::Error> {
        if buffer.is_empty() {
            return Ok(());
        }

        let ptr = buffer.as_ptr();
        let len = buffer.len();

        // Round up to page boundary
        let aligned_len = ((len + self.page_size - 1) / self.page_size) * self.page_size;

        let result = unsafe {
            libc::mlock(ptr as *const libc::c_void, aligned_len)
        };

        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Unpin a previously pinned buffer
    pub fn unpin_buffer(&self, buffer: &[u8]) -> Result<(), io::Error> {
        if buffer.is_empty() {
            return Ok(());
        }

        let ptr = buffer.as_ptr();
        let len = buffer.len();

        // Round up to page boundary
        let aligned_len = ((len + self.page_size - 1) / self.page_size) * self.page_size;

        let result = unsafe {
            libc::munlock(ptr as *const libc::c_void, aligned_len)
        };

        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    /// Pin a mutable buffer (for buffers that need to be modified)
    pub fn pin_buffer_mut(&self, buffer: &mut [u8]) -> Result<(), io::Error> {
        if buffer.is_empty() {
            return Ok(());
        }

        let ptr = buffer.as_mut_ptr();
        let len = buffer.len();

        // Round up to page boundary
        let aligned_len = ((len + self.page_size - 1) / self.page_size) * self.page_size;

        let result = unsafe {
            libc::mlock(ptr as *const libc::c_void, aligned_len)
        };

        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
}

impl Default for MemoryPinner {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MemoryPinner {
    fn drop(&mut self) {
        // Ensure no locked pages remain
        if self.current_security_level == SecurityLevel::Production {
            // All buffers should have been unlocked individually
            // This is just a sanity check
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pinner_creation() {
        let pinner = MemoryPinner::new();
        assert!(pinner.page_size() > 0);
        // Initial state is degraded until pin_page is called
        assert_eq!(pinner.security_level(), SecurityLevel::Degraded);
    }

    #[test]
    fn test_pin_page_result() {
        let mut pinner = MemoryPinner::new();
        let result = pinner.pin_page();
        
        // Result depends on system capabilities
        // Either succeeds with Production or fails with error
        match result {
            Ok(level) => {
                assert_eq!(level, SecurityLevel::Production);
                assert_eq!(pinner.security_level(), SecurityLevel::Production);
            }
            Err(_) => {
                assert_eq!(pinner.security_level(), SecurityLevel::Degraded);
            }
        }
    }
}
