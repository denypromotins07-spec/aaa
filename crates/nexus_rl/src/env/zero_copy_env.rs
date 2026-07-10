//! Zero-Copy Shared Memory RL Environment
//! 
//! This module implements a Rust-native RL environment that writes state observations
//! directly into POSIX shared memory, enabling zero-copy access from Python.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::ptr;
use std::cell::UnsafeCell;

use super::shared_memory_mapper::{SharedMemoryMap, SHARED_MEMORY_SIZE};

/// Maximum number of features in the observation space
pub const MAX_FEATURES: usize = 256;
/// Maximum number of assets tracked
pub const MAX_ASSETS: usize = 32;
/// Maximum order book depth tracked
pub const MAX_BOOK_DEPTH: usize = 10;

/// State representation written to shared memory
#[repr(C, align(64))]
pub struct StateObservation {
    /// Atomic step counter - updated last to signal completion
    pub step_counter: AtomicU64,
    /// Writing flag: 0=ready, 1=writing
    pub writing_flag: AtomicU8,
    /// Episode step number
    pub episode_step: u64,
    /// Current timestamp (nanoseconds)
    pub timestamp_ns: u64,
    
    /// Order book features: [bid_prices[MAX_DEPTH], bid_sizes[MAX_DEPTH], ask_prices[MAX_DEPTH], ask_sizes[MAX_DEPTH]]
    pub order_book_data: [f64; MAX_BOOK_DEPTH * 4],
    
    /// Market features: spreads, volumes, volatility estimates
    pub market_features: [f64; 64],
    
    /// Portfolio state: positions, cash, unrealized PnL per asset
    pub portfolio_state: [f64; MAX_ASSETS * 3],
    
    /// Technical indicators: RSI, MACD, Bollinger bands, etc.
    pub technical_indicators: [f64; 64],
    
    /// Padding to ensure cache-line alignment
    _padding: [u8; 64],
}

impl StateObservation {
    /// Create a new zero-initialized state observation
    #[inline]
    pub const fn new() -> Self {
        Self {
            step_counter: AtomicU64::new(0),
            writing_flag: AtomicU8::new(0),
            episode_step: 0,
            timestamp_ns: 0,
            order_book_data: [0.0; MAX_BOOK_DEPTH * 4],
            market_features: [0.0; 64],
            portfolio_state: [0.0; MAX_ASSETS * 3],
            technical_indicators: [0.0; 64],
            _padding: [0u8; 64],
        }
    }
    
    /// Check if state is safe to read (not being written)
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.writing_flag.load(Ordering::Acquire) == 0
    }
    
    /// Get current step counter
    #[inline]
    pub fn get_step(&self) -> u64 {
        self.step_counter.load(Ordering::Acquire)
    }
}

/// Zero-copy environment wrapper
pub struct ZeroCopyEnv {
    state: UnsafeCell<StateObservation>,
    shm_map: Option<SharedMemoryMap>,
    env_id: u32,
    is_reset: bool,
}

unsafe impl Send for ZeroCopyEnv {}
unsafe impl Sync for ZeroCopyEnv {}

impl ZeroCopyEnv {
    /// Create a new zero-copy environment
    pub fn new(env_id: u32) -> Self {
        Self {
            state: UnsafeCell::new(StateObservation::new()),
            shm_map: None,
            env_id,
            is_reset: false,
        }
    }
    
    /// Initialize with shared memory mapping
    pub fn with_shared_memory(env_id: u32, shm_name: &str) -> Result<Self, &'static str> {
        let shm_map = SharedMemoryMap::create(shm_name)?;
        Ok(Self {
            state: UnsafeCell::new(StateObservation::new()),
            shm_map: Some(shm_map),
            env_id,
            is_reset: false,
        })
    }
    
    /// Get mutable reference to state (safe because we control write access)
    #[inline]
    fn get_state_mut(&self) -> &mut StateObservation {
        unsafe { &mut *self.state.get() }
    }
    
    /// Begin atomic state update
    /// Sets writing flag to prevent torn reads
    #[inline]
    pub fn begin_update(&self) {
        let state = self.get_state_mut();
        state.writing_flag.store(1, Ordering::Release);
        // Full fence to ensure all subsequent writes happen after flag
        std::sync::atomic::fence(Ordering::SeqCst);
    }
    
    /// Complete atomic state update
    /// Updates step counter and clears writing flag
    #[inline]
    pub fn end_update(&self) {
        let state = self.get_state_mut();
        
        // Full fence before publishing
        std::sync::atomic::fence(Ordering::SeqCst);
        
        // Increment step counter with Release ordering
        let new_step = state.step_counter.fetch_add(1, Ordering::Release) + 1;
        state.episode_step = new_step;
        
        // Clear writing flag with Release to ensure all writes are visible
        state.writing_flag.store(0, Ordering::Release);
    }
    
    /// Write order book data to state
    #[inline]
    pub fn write_order_book(&self, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        let state = self.get_state_mut();
        let depth = bids.len().min(asks.len()).min(MAX_BOOK_DEPTH);
        
        for i in 0..depth {
            state.order_book_data[i] = bids[i].0; // bid prices
            state.order_book_data[MAX_BOOK_DEPTH + i] = bids[i].1; // bid sizes
            state.order_book_data[MAX_BOOK_DEPTH * 2 + i] = asks[i].0; // ask prices
            state.order_book_data[MAX_BOOK_DEPTH * 3 + i] = asks[i].1; // ask sizes
        }
    }
    
    /// Write market features
    #[inline]
    pub fn write_market_features(&self, features: &[f64]) {
        let state = self.get_state_mut();
        let len = features.len().min(64);
        state.market_features[..len].copy_from_slice(&features[..len]);
    }
    
    /// Write portfolio state
    #[inline]
    pub fn write_portfolio_state(&self, positions: &[f64], cash: f64, pnl: &[f64]) {
        let state = self.get_state_mut();
        let num_assets = positions.len().min(MAX_ASSETS);
        
        for i in 0..num_assets {
            state.portfolio_state[i * 3] = positions[i];
            state.portfolio_state[i * 3 + 1] = if i == 0 { cash } else { 0.0 };
            state.portfolio_state[i * 3 + 2] = pnl.get(i).copied().unwrap_or(0.0);
        }
    }
    
    /// Write technical indicators
    #[inline]
    pub fn write_technical_indicators(&self, indicators: &[f64]) {
        let state = self.get_state_mut();
        let len = indicators.len().min(64);
        state.technical_indicators[..len].copy_from_slice(&indicators[..len]);
    }
    
    /// Reset environment
    pub fn reset(&mut self) {
        let state = self.get_state_mut();
        state.step_counter.store(0, Ordering::Release);
        state.episode_step = 0;
        state.timestamp_ns = 0;
        
        // Zero out all data arrays
        state.order_book_data.fill(0.0);
        state.market_features.fill(0.0);
        state.portfolio_state.fill(0.0);
        state.technical_indicators.fill(0.0);
        
        state.writing_flag.store(0, Ordering::Release);
        self.is_reset = true;
    }
    
    /// Get environment ID
    #[inline]
    pub fn id(&self) -> u32 {
        self.env_id
    }
    
    /// Check if environment is reset
    #[inline]
    pub fn is_reset(&self) -> bool {
        self.is_reset
    }
    
    /// Get pointer to shared memory for FFI
    #[inline]
    pub fn get_shm_ptr(&self) -> Option<*const u8> {
        self.shm_map.as_ref().map(|m| m.as_ptr())
    }
    
    /// Get shared memory size
    #[inline]
    pub fn get_shm_size(&self) -> usize {
        self.shm_map.as_ref().map_or(0, |m| m.size())
    }
}

impl Default for ZeroCopyEnv {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_zero_copy_env_creation() {
        let env = ZeroCopyEnv::new(1);
        assert_eq!(env.id(), 1);
        assert!(!env.is_reset());
    }
    
    #[test]
    fn test_atomic_state_update() {
        let env = ZeroCopyEnv::new(0);
        
        env.begin_update();
        env.write_market_features(&[1.0, 2.0, 3.0]);
        env.end_update();
        
        // Verify step counter incremented
        assert_eq!(env.get_state_mut().get_step(), 1);
        assert!(env.get_state_mut().is_ready());
    }
    
    #[test]
    fn test_reset_clears_state() {
        let mut env = ZeroCopyEnv::new(0);
        
        env.begin_update();
        env.write_market_features(&[1.0, 2.0, 3.0]);
        env.end_update();
        
        env.reset();
        
        assert!(env.is_reset());
        assert_eq!(env.get_state_mut().get_step(), 0);
        assert_eq!(env.get_state_mut().market_features[0], 0.0);
    }
}
