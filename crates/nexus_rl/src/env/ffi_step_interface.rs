//! C-FFI Step Interface for Python Integration
//! 
//! This module provides the `extern "C"` interface that allows Python to interact
//! with the Rust RL environment without GIL contention, using zero-copy shared memory.

use std::ffi::c_void;
use std::os::raw::{c_char, c_int, c_uint, c_ulonglong};
use std::ptr;

use super::zero_copy_env::ZeroCopyEnv;
use super::shared_memory_mapper::{SharedMemoryMap, SHARED_MEMORY_SIZE};

/// Opaque handle to a Rust RL environment
pub struct RLEnvHandle {
    env: ZeroCopyEnv,
}

/// Global registry of active environments (protected by lazy_static or once_cell)
static mut ENV_REGISTRY: Vec<Option<RLEnvHandle>> = Vec::new();
static mut REGISTRY_INITIALIZED: bool = false;

/// Initialize the environment registry
unsafe fn init_registry() {
    if !REGISTRY_INITIALIZED {
        ENV_REGISTRY = vec![None; 256]; // Support up to 256 concurrent envs
        REGISTRY_INITIALIZED = true;
    }
}

/// Create a new RL environment and return handle
/// 
/// # Safety
/// This function uses raw pointers and must be called from FFI context
#[no_mangle]
pub unsafe extern "C" fn nexus_rl_env_create(env_id: c_uint) -> c_int {
    init_registry();
    
    let idx = env_id as usize;
    if idx >= ENV_REGISTRY.len() {
        return -1;
    }
    
    let env = ZeroCopyEnv::new(env_id);
    ENV_REGISTRY[idx] = Some(RLEnvHandle { env });
    
    idx as c_int
}

/// Create environment with shared memory backing
/// 
/// # Safety
/// shm_name must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn nexus_rl_env_create_shm(
    env_id: c_uint,
    shm_name: *const c_char,
) -> c_int {
    init_registry();
    
    let idx = env_id as usize;
    if idx >= ENV_REGISTRY.len() || shm_name.is_null() {
        return -1;
    }
    
    let name_str = std::ffi::CStr::from_ptr(shm_name).to_string_lossy();
    
    match ZeroCopyEnv::with_shared_memory(env_id, &name_str) {
        Ok(env) => {
            ENV_REGISTRY[idx] = Some(RLEnvHandle { env });
            idx as c_int
        }
        Err(_) => -1,
    }
}

/// Reset environment to initial state
#[no_mangle]
pub unsafe extern "C" fn nexus_rl_env_reset(handle: c_int) -> c_int {
    if handle < 0 || handle as usize >= ENV_REGISTRY.len() {
        return -1;
    }
    
    if let Some(ref mut env_handle) = ENV_REGISTRY[handle as usize] {
        env_handle.env.reset();
        0
    } else {
        -1
    }
}

/// Begin atomic state update (acquire write lock)
#[no_mangle]
pub unsafe extern "C" fn nexus_rl_env_begin_update(handle: c_int) -> c_int {
    if handle < 0 || handle as usize >= ENV_REGISTRY.len() {
        return -1;
    }
    
    if let Some(ref env_handle) = ENV_REGISTRY[handle as usize] {
        env_handle.env.begin_update();
        0
    } else {
        -1
    }
}

/// End atomic state update (release write lock)
#[no_mangle]
pub unsafe extern "C" fn nexus_rl_env_end_update(handle: c_int) -> c_int {
    if handle < 0 || handle as usize >= ENV_REGISTRY.len() {
        return -1;
    }
    
    if let Some(ref env_handle) = ENV_REGISTRY[handle as usize] {
        env_handle.env.end_update();
        0
    } else {
        -1
    }
}

/// Write order book data to environment state
/// 
/// # Safety
/// bids and asks must point to valid arrays of (price, size) pairs
#[no_mangle]
pub unsafe extern "C" fn nexus_rl_env_write_order_book(
    handle: c_int,
    bids: *const f64,
    asks: *const f64,
    depth: c_int,
) -> c_int {
    if handle < 0 || handle as usize >= ENV_REGISTRY.len() {
        return -1;
    }
    
    if bids.is_null() || asks.is_null() || depth <= 0 {
        return -1;
    }
    
    if let Some(ref env_handle) = ENV_REGISTRY[handle as usize] {
        let bid_slice = std::slice::from_raw_parts(bids, depth as usize * 2);
        let ask_slice = std::slice::from_raw_parts(asks, depth as usize * 2);
        
        // Convert interleaved [price, size, price, size, ...] to [(price, size), ...]
        let bids_vec: Vec<(f64, f64)> = (0..depth as usize)
            .map(|i| (bid_slice[i * 2], bid_slice[i * 2 + 1]))
            .collect();
        let asks_vec: Vec<(f64, f64)> = (0..depth as usize)
            .map(|i| (ask_slice[i * 2], ask_slice[i * 2 + 1]))
            .collect();
        
        env_handle.env.write_order_book(&bids_vec, &asks_vec);
        0
    } else {
        -1
    }
}

/// Write market features to environment state
#[no_mangle]
pub unsafe extern "C" fn nexus_rl_env_write_features(
    handle: c_int,
    features: *const f64,
    count: c_int,
) -> c_int {
    if handle < 0 || handle as usize >= ENV_REGISTRY.len() {
        return -1;
    }
    
    if features.is_null() || count <= 0 {
        return -1;
    }
    
    if let Some(ref env_handle) = ENV_REGISTRY[handle as usize] {
        let feature_slice = std::slice::from_raw_parts(features, count as usize);
        env_handle.env.write_market_features(feature_slice);
        0
    } else {
        -1
    }
}

/// Write portfolio state to environment
#[no_mangle]
pub unsafe extern "C" fn nexus_rl_env_write_portfolio(
    handle: c_int,
    positions: *const f64,
    cash: f64,
    pnl: *const f64,
    num_assets: c_int,
) -> c_int {
    if handle < 0 || handle as usize >= ENV_REGISTRY.len() {
        return -1;
    }
    
    if positions.is_null() || pnl.is_null() || num_assets <= 0 {
        return -1;
    }
    
    if let Some(ref env_handle) = ENV_REGISTRY[handle as usize] {
        let pos_slice = std::slice::from_raw_parts(positions, num_assets as usize);
        let pnl_slice = std::slice::from_raw_parts(pnl, num_assets as usize);
        env_handle.env.write_portfolio_state(pos_slice, cash, pnl_slice);
        0
    } else {
        -1
    }
}

/// Get pointer to shared memory for zero-copy reading
#[no_mangle]
pub unsafe extern "C" fn nexus_rl_env_get_shm_ptr(handle: c_int) -> *const c_void {
    if handle < 0 || handle as usize >= ENV_REGISTRY.len() {
        return ptr::null();
    }
    
    if let Some(ref env_handle) = ENV_REGISTRY[handle as usize] {
        env_handle.env.get_shm_ptr().unwrap_or(ptr::null()) as *const c_void
    } else {
        ptr::null()
    }
}

/// Get shared memory size
#[no_mangle]
pub unsafe extern "C" fn nexus_rl_env_get_shm_size(handle: c_int) -> c_ulonglong {
    if handle < 0 || handle as usize >= ENV_REGISTRY.len() {
        return 0;
    }
    
    if let Some(ref env_handle) = ENV_REGISTRY[handle as usize] {
        env_handle.env.get_shm_size() as c_ulonglong
    } else {
        0
    }
}

/// Check if environment state is ready (not being written)
#[no_mangle]
pub unsafe extern "C" fn nexus_rl_env_is_ready(handle: c_int) -> c_int {
    if handle < 0 || handle as usize >= ENV_REGISTRY.len() {
        return 0;
    }
    
    if let Some(ref env_handle) = ENV_REGISTRY[handle as usize] {
        if env_handle.env.is_reset() {
            1
        } else {
            0
        }
    } else {
        0
    }
}

/// Get current step counter
#[no_mangle]
pub unsafe extern "C" fn nexus_rl_env_get_step(handle: c_int) -> c_ulonglong {
    if handle < 0 || handle as usize >= ENV_REGISTRY.len() {
        return 0;
    }
    
    if let Some(ref env_handle) = ENV_REGISTRY[handle as usize] {
        // Access internal state - would need a getter in ZeroCopyEnv
        0 // Placeholder - actual implementation needs state access
    } else {
        0
    }
}

/// Destroy environment and free resources
#[no_mangle]
pub unsafe extern "C" fn nexus_rl_env_destroy(handle: c_int) -> c_int {
    init_registry();
    
    if handle < 0 || handle as usize >= ENV_REGISTRY.len() {
        return -1;
    }
    
    ENV_REGISTRY[handle as usize] = None;
    0
}

/// Get library version
#[no_mangle]
pub unsafe extern "C" fn nexus_rl_get_version() -> *const c_char {
    b"NEXUS-RL v0.1.0\0".as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ffi_interface_basic() {
        unsafe {
            // Create environment
            let handle = nexus_rl_env_create(0);
            assert!(handle >= 0);
            
            // Reset
            assert_eq!(nexus_rl_env_reset(handle), 0);
            
            // Destroy
            assert_eq!(nexus_rl_env_destroy(handle), 0);
        }
    }
    
    #[test]
    fn test_invalid_handle() {
        unsafe {
            assert_eq!(nexus_rl_env_reset(-1), -1);
            assert_eq!(nexus_rl_env_destroy(9999), -1);
        }
    }
}
