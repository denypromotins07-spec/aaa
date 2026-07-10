//! Shared Memory Mapper for Zero-Copy RL State Transfer
//! 
//! This module handles POSIX shared memory creation, mapping, and synchronization
//! between Rust (writer) and Python (reader) processes.

use std::io;
use std::ptr;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

#[cfg(unix)]
use std::os::unix::io::RawFd;

/// Size of shared memory segment (1MB for state observations)
pub const SHARED_MEMORY_SIZE: usize = 1024 * 1024;
/// Maximum name length for shared memory segments
pub const MAX_SHM_NAME_LEN: usize = 64;

/// Error types for shared memory operations
#[derive(Debug)]
pub enum ShmError {
    CreationFailed(String),
    MappingFailed(String),
    InvalidName,
    AlreadyExists,
    NotFound,
}

impl std::fmt::Display for ShmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShmError::CreationFailed(msg) => write!(f, "Shared memory creation failed: {}", msg),
            ShmError::MappingFailed(msg) => write!(f, "Shared memory mapping failed: {}", msg),
            ShmError::InvalidName => write!(f, "Invalid shared memory name"),
            ShmError::AlreadyExists => write!(f, "Shared memory already exists"),
            ShmError::NotFound => write!(f, "Shared memory not found"),
        }
    }
}

impl std::error::Error for ShmError {}

/// POSIX Shared Memory Map wrapper
pub struct SharedMemoryMap {
    #[cfg(unix)]
    fd: RawFd,
    #[cfg(unix)]
    ptr: *mut u8,
    #[cfg(windows)]
    handle: *mut std::ffi::c_void,
    #[cfg(windows)]
    ptr: *mut u8,
    size: usize,
    name: String,
    _marker: std::marker::PhantomData<*mut ()>, // Not Send/Sync by default
}

unsafe impl Send for SharedMemoryMap {}
unsafe impl Sync for SharedMemoryMap {}

impl SharedMemoryMap {
    /// Create a new shared memory segment
    #[cfg(unix)]
    pub fn create(name: &str) -> Result<Self, ShmError> {
        use std::ffi::CString;
        
        if name.is_empty() || name.len() > MAX_SHM_NAME_LEN {
            return Err(ShmError::InvalidName);
        }
        
        let shm_name = format!("/{}", name.trim_start_matches('/'));
        let c_name = CString::new(shm_name.clone())
            .map_err(|_| ShmError::InvalidName)?;
        
        unsafe {
            // Create shared memory with read/write permissions
            let fd = libc::shm_open(
                c_name.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_EXCL,
                0o666,
            );
            
            if fd < 0 {
                let err = io::Error::last_os_error();
                if err.raw_os_error() == Some(libc::EEXIST) {
                    return Err(ShmError::AlreadyExists);
                }
                return Err(ShmError::CreationFailed(err.to_string()));
            }
            
            // Set size
            if libc::ftruncate(fd, SHARED_MEMORY_SIZE as i64) < 0 {
                libc::close(fd);
                libc::shm_unlink(c_name.as_ptr());
                return Err(ShmError::CreationFailed("ftruncate failed".to_string()));
            }
            
            // Map into memory
            let ptr = libc::mmap(
                ptr::null_mut(),
                SHARED_MEMORY_SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            
            if ptr == libc::MAP_FAILED {
                libc::close(fd);
                libc::shm_unlink(c_name.as_ptr());
                return Err(ShmError::MappingFailed(io::Error::last_os_error().to_string()));
            }
            
            // Zero initialize
            ptr::write_bytes(ptr, 0, SHARED_MEMORY_SIZE);
            
            Ok(Self {
                fd,
                ptr: ptr as *mut u8,
                size: SHARED_MEMORY_SIZE,
                name: shm_name,
                _marker: std::marker::PhantomData,
            })
        }
    }
    
    /// Open existing shared memory segment (for reader/Python side)
    #[cfg(unix)]
    pub fn open(name: &str) -> Result<Self, ShmError> {
        use std::ffi::CString;
        
        if name.is_empty() || name.len() > MAX_SHM_NAME_LEN {
            return Err(ShmError::InvalidName);
        }
        
        let shm_name = format!("/{}", name.trim_start_matches('/'));
        let c_name = CString::new(shm_name.clone())
            .map_err(|_| ShmError::InvalidName)?;
        
        unsafe {
            let fd = libc::shm_open(c_name.as_ptr(), libc::O_RDWR, 0o666);
            if fd < 0 {
                return Err(ShmError::NotFound);
            }
            
            let ptr = libc::mmap(
                ptr::null_mut(),
                SHARED_MEMORY_SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            
            if ptr == libc::MAP_FAILED {
                libc::close(fd);
                return Err(ShmError::MappingFailed(io::Error::last_os_error().to_string()));
            }
            
            Ok(Self {
                fd,
                ptr: ptr as *mut u8,
                size: SHARED_MEMORY_SIZE,
                name: shm_name,
                _marker: std::marker::PhantomData,
            })
        }
    }
    
    /// Windows implementation stub
    #[cfg(windows)]
    pub fn create(name: &str) -> Result<Self, ShmError> {
        unimplemented!("Windows support pending")
    }
    
    #[cfg(windows)]
    pub fn open(name: &str) -> Result<Self, ShmError> {
        unimplemented!("Windows support pending")
    }
    
    /// Get pointer to shared memory
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
    
    /// Get mutable pointer to shared memory
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
    
    /// Get size of shared memory
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Get name of shared memory segment
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Read atomic step counter from shared memory (with Acquire ordering)
    #[inline]
    pub fn read_step_counter(&self) -> u64 {
        unsafe {
            let step_ptr = self.ptr as *const AtomicU64;
            (*step_ptr).load(Ordering::Acquire)
        }
    }
    
    /// Read writing flag from shared memory (with Acquire ordering)
    #[inline]
    pub fn read_writing_flag(&self) -> u8 {
        unsafe {
            let flag_ptr = self.ptr.add(std::mem::size_of::<AtomicU64>()) as *const AtomicU8;
            (*flag_ptr).load(Ordering::Acquire)
        }
    }
    
    /// Check if data is ready to read
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.read_writing_flag() == 0
    }
}

impl Drop for SharedMemoryMap {
    fn drop(&mut self) {
        unsafe {
            #[cfg(unix)]
            {
                if !self.ptr.is_null() {
                    libc::munmap(self.ptr as *mut _, self.size);
                }
                if self.fd >= 0 {
                    libc::close(self.fd);
                }
            }
        }
    }
}

/// Unlink (delete) a shared memory segment
#[cfg(unix)]
pub fn unlink(name: &str) -> Result<(), ShmError> {
    use std::ffi::CString;
    
    let shm_name = format!("/{}", name.trim_start_matches('/'));
    let c_name = CString::new(shm_name)
        .map_err(|_| ShmError::InvalidName)?;
    
    unsafe {
        if libc::shm_unlink(c_name.as_ptr()) < 0 {
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::ENOENT) {
                return Err(ShmError::NotFound);
            }
            return Err(ShmError::CreationFailed(err.to_string()));
        }
    }
    Ok(())
}

#[cfg(windows)]
pub fn unlink(_name: &str) -> Result<(), ShmError> {
    unimplemented!("Windows support pending")
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[cfg(unix)]
    fn test_create_and_open_shm() {
        let name = "nexus_test_shm_1";
        
        // Clean up any existing
        let _ = unlink(name);
        
        // Create
        let shm = SharedMemoryMap::create(name).expect("Failed to create shared memory");
        assert_eq!(shm.size(), SHARED_MEMORY_SIZE);
        
        // Open from another "process"
        let shm_reader = SharedMemoryMap::open(name).expect("Failed to open shared memory");
        assert_eq!(shm_reader.size(), SHARED_MEMORY_SIZE);
        
        // Cleanup
        drop(shm);
        drop(shm_reader);
        let _ = unlink(name);
    }
    
    #[test]
    #[cfg(unix)]
    fn test_atomic_read_write() {
        let name = "nexus_test_shm_2";
        let _ = unlink(name);
        
        let mut shm = SharedMemoryMap::create(name).expect("Failed to create");
        
        // Write step counter
        unsafe {
            let step_ptr = shm.as_mut_ptr() as *mut AtomicU64;
            (*step_ptr).store(42, Ordering::Release);
        }
        
        // Read back
        assert_eq!(shm.read_step_counter(), 42);
        
        let _ = unlink(name);
    }
}
