//! PagedAttention KV-Cache Memory Pool Manager
//!
//! This module implements a GPU VRAM block allocator for PagedAttention-based
//! LLM inference. It pre-allocates memory blocks and tracks free/used blocks
//! using atomic counters to prevent CUDA OOM errors during high-volatility events.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::{info, warn, error};

/// Default block size in bytes (1MB blocks)
const DEFAULT_BLOCK_SIZE_BYTES: usize = 1024 * 1024;

/// Default number of blocks to pre-allocate
const DEFAULT_NUM_BLOCKS: usize = 4096;

/// Maximum blocks per sequence
const MAX_BLOCKS_PER_SEQ: usize = 256;

/// Error types for memory pool operations
#[derive(Debug, thiserror::Error)]
pub enum PagedPoolError {
    #[error("Out of memory: no free blocks available")]
    OutOfMemory,
    #[error("Invalid block ID: {0}")]
    InvalidBlockId(usize),
    #[error("Block already allocated: {0}")]
    BlockAlreadyAllocated(usize),
    #[error("Block not allocated: {0}")]
    BlockNotAllocated(usize),
    #[error("Sequence limit exceeded")]
    SequenceLimitExceeded,
    #[error("CUDA error: {0}")]
    CudaError(String),
}

/// Result type for pool operations
pub type PoolResult<T> = Result<T, PagedPoolError>;

/// State of a memory block
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockState {
    Free,
    Allocated,
    Reserved,
}

/// A single memory block descriptor
#[derive(Debug)]
pub struct MemoryBlock {
    /// Block ID
    pub id: usize,
    /// GPU memory offset (bytes)
    pub gpu_offset: u64,
    /// Block size in bytes
    pub size_bytes: usize,
    /// Current state
    pub state: AtomicU8,
    /// Reference count (for shared attention)
    pub ref_count: AtomicUsize,
    /// Associated sequence ID (if allocated)
    pub seq_id: AtomicU64,
}

impl MemoryBlock {
    fn new(id: usize, gpu_offset: u64, size_bytes: usize) -> Self {
        Self {
            id,
            gpu_offset,
            size_bytes,
            state: AtomicU8::new(BlockState::Free as u8),
            ref_count: AtomicUsize::new(0),
            seq_id: AtomicU64::new(0),
        }
    }

    #[inline]
    fn get_state(&self) -> BlockState {
        match self.state.load(Ordering::Acquire) {
            0 => BlockState::Free,
            1 => BlockState::Allocated,
            2 => BlockState::Reserved,
            _ => BlockState::Free,
        }
    }

    #[inline]
    fn set_state(&self, state: BlockState) {
        self.state.store(state as u8, Ordering::Release);
    }

    #[inline]
    fn try_allocate(&self, seq_id: u64) -> PoolResult<()> {
        let expected = BlockState::Free as u8;
        match self.state.compare_exchange(
            expected,
            BlockState::Allocated as u8,
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                self.seq_id.store(seq_id, Ordering::Relaxed);
                self.ref_count.store(1, Ordering::Relaxed);
                Ok(())
            }
            Err(_) => Err(PagedPoolError::BlockAlreadyAllocated(self.id)),
        }
    }

    #[inline]
    fn release(&self) -> PoolResult<()> {
        let current = self.ref_count.fetch_sub(1, Ordering::AcqRel);
        if current == 1 {
            // Last reference, free the block
            self.set_state(BlockState::Free);
            self.seq_id.store(0, Ordering::Relaxed);
        }
        Ok(())
    }
}

// SAFETY: MemoryBlock uses atomics for all mutable state
unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

/// Configuration for the paged attention pool
#[derive(Debug, Clone)]
pub struct PagedPoolConfig {
    /// Total GPU memory budget (bytes)
    pub total_memory_bytes: u64,
    /// Block size (bytes)
    pub block_size_bytes: usize,
    /// Number of blocks to pre-allocate
    pub num_blocks: usize,
    /// Enable memory profiling
    pub enable_profiling: bool,
    /// Reserve percentage for emergency allocation
    pub reserve_percentage: f32,
}

impl Default for PagedPoolConfig {
    fn default() -> Self {
        Self {
            total_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            block_size_bytes: DEFAULT_BLOCK_SIZE_BYTES,
            num_blocks: DEFAULT_NUM_BLOCKS,
            enable_profiling: false,
            reserve_percentage: 0.1, // 10% reserve
        }
    }
}

/// Statistics for the memory pool
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total blocks
    pub total_blocks: usize,
    /// Free blocks
    pub free_blocks: usize,
    /// Allocated blocks
    pub allocated_blocks: usize,
    /// Reserved blocks
    pub reserved_blocks: usize,
    /// Total memory (bytes)
    pub total_memory_bytes: u64,
    /// Used memory (bytes)
    pub used_memory_bytes: u64,
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: u64,
    /// Allocation failures
    pub allocation_failures: u64,
}

/// PagedAttention KV-Cache memory pool manager
pub struct PagedAttentionPool {
    /// All memory blocks
    blocks: Vec<Arc<MemoryBlock>>,
    /// Free list (indices of free blocks)
    free_list: dashmap::DashMap<usize, ()>,
    /// Configuration
    config: PagedPoolConfig,
    /// Statistics
    stats: Arc<AtomicPoolStats>,
    /// Next sequence ID counter
    next_seq_id: AtomicU64,
}

/// Atomic statistics for lock-free updates
struct AtomicPoolStats {
    total_blocks: AtomicUsize,
    free_blocks: AtomicUsize,
    allocated_blocks: AtomicUsize,
    reserved_blocks: AtomicUsize,
    allocation_failures: AtomicU64,
    peak_memory_bytes: AtomicU64,
    used_memory_bytes: AtomicU64,
}

impl AtomicPoolStats {
    fn new(total_blocks: usize) -> Self {
        Self {
            total_blocks: AtomicUsize::new(total_blocks),
            free_blocks: AtomicUsize::new(total_blocks),
            allocated_blocks: AtomicUsize::new(0),
            reserved_blocks: AtomicUsize::new(0),
            allocation_failures: AtomicU64::new(0),
            peak_memory_bytes: AtomicU64::new(0),
            used_memory_bytes: AtomicU64::new(0),
        }
    }

    fn update_usage(&self, used_bytes: u64) {
        self.used_memory_bytes.store(used_bytes, Ordering::Relaxed);
        
        // Update peak if necessary
        let mut peak = self.peak_memory_bytes.load(Ordering::Relaxed);
        while used_bytes > peak {
            match self.peak_memory_bytes.compare_exchange_weak(
                peak,
                used_bytes,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    fn to_snapshot(&self, config: &PagedPoolConfig) -> PoolStats {
        let total = self.total_blocks.load(Ordering::Relaxed);
        let free = self.free_blocks.load(Ordering::Relaxed);
        let allocated = self.allocated_blocks.load(Ordering::Relaxed);
        let reserved = self.reserved_blocks.load(Ordering::Relaxed);
        
        PoolStats {
            total_blocks: total,
            free_blocks: free,
            allocated_blocks: allocated,
            reserved_blocks: reserved,
            total_memory_bytes: config.total_memory_bytes,
            used_memory_bytes: self.used_memory_bytes.load(Ordering::Relaxed),
            peak_memory_bytes: self.peak_memory_bytes.load(Ordering::Relaxed),
            allocation_failures: self.allocation_failures.load(Ordering::Relaxed),
        }
    }
}

impl PagedAttentionPool {
    /// Create a new paged attention pool
    pub fn new(config: PagedPoolConfig) -> PoolResult<Self> {
        let num_blocks = config.num_blocks;
        let block_size = config.block_size_bytes;
        
        // Validate configuration
        if num_blocks == 0 || num_blocks > 1_000_000 {
            return Err(PagedPoolError::InvalidBlockId(num_blocks));
        }
        
        let total_required = num_blocks as u64 * block_size as u64;
        if total_required > config.total_memory_bytes {
            return Err(PagedPoolError::CudaError(format!(
                "Requested memory {} exceeds budget {}",
                total_required, config.total_memory_bytes
            )));
        }

        // Pre-allocate all blocks
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut free_list = dashmap::DashMap::with_capacity(num_blocks);

        for i in 0..num_blocks {
            let gpu_offset = (i as u64) * (block_size as u64);
            let block = Arc::new(MemoryBlock::new(i, gpu_offset, block_size));
            free_list.insert(i, ());
            blocks.push(block);
        }

        let stats = Arc::new(AtomicPoolStats::new(num_blocks));

        info!(
            "PagedAttentionPool initialized: {} blocks, {} bytes total",
            num_blocks, total_required
        );

        Ok(Self {
            blocks,
            free_list,
            config,
            stats,
            next_seq_id: AtomicU64::new(1),
        })
    }

    /// Allocate a block for a sequence
    pub fn allocate_block(&self) -> PoolResult<Arc<MemoryBlock>> {
        // Get a free block from the free list
        let entry = self.free_list.iter().next()
            .map(|e| e.key().clone())
            .ok_or_else(|| {
                self.stats.allocation_failures.fetch_add(1, Ordering::Relaxed);
                PagedPoolError::OutOfMemory
            })?;

        // Remove from free list
        self.free_list.remove(&entry);

        // Get sequence ID
        let seq_id = self.next_seq_id.fetch_add(1, Ordering::Relaxed);

        // Allocate the block
        let block = &self.blocks[entry];
        block.try_allocate(seq_id)?;

        // Update statistics
        let allocated = self.stats.allocated_blocks.fetch_add(1, Ordering::Relaxed) + 1;
        let free = self.stats.free_blocks.fetch_sub(1, Ordering::Relaxed) - 1;
        self.stats.update_usage((allocated as u64) * (self.config.block_size_bytes as u64));

        info!("Allocated block {} for sequence {}", entry, seq_id);

        Ok(Arc::clone(block))
    }

    /// Allocate multiple blocks for a sequence
    pub fn allocate_blocks(&self, count: usize) -> PoolResult<Vec<Arc<MemoryBlock>>> {
        if count > MAX_BLOCKS_PER_SEQ {
            return Err(PagedPoolError::SequenceLimitExceeded);
        }

        let mut blocks = Vec::with_capacity(count);
        for _ in 0..count {
            let block = self.allocate_block()?;
            blocks.push(block);
        }

        Ok(blocks)
    }

    /// Release a block back to the pool
    pub fn release_block(&self, block: &MemoryBlock) -> PoolResult<()> {
        let block_id = block.id;
        
        if block_id >= self.blocks.len() {
            return Err(PagedPoolError::InvalidBlockId(block_id));
        }

        block.release()?;

        // Check if block is now free
        if block.get_state() == BlockState::Free {
            self.free_list.insert(block_id, ());
            
            let allocated = self.stats.allocated_blocks.fetch_sub(1, Ordering::Relaxed) - 1;
            let free = self.stats.free_blocks.fetch_add(1, Ordering::Relaxed) + 1;
            self.stats.update_usage((allocated as u64) * (self.config.block_size_bytes as u64));
            
            info!("Released block {} back to pool", block_id);
        }

        Ok(())
    }

    /// Get statistics snapshot
    pub fn get_stats(&self) -> PoolStats {
        self.stats.to_snapshot(&self.config)
    }

    /// Get current utilization ratio (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        let stats = self.get_stats();
        stats.allocated_blocks as f64 / stats.total_blocks as f64
    }

    /// Check if pool is running low on memory
    pub fn is_low_on_memory(&self) -> bool {
        let stats = self.get_stats();
        let threshold = (self.config.reserve_percentage * stats.total_blocks as f32) as usize;
        stats.free_blocks < threshold
    }

    /// Get number of available blocks
    pub fn available_blocks(&self) -> usize {
        self.stats.free_blocks.load(Ordering::Relaxed)
    }

    /// Reserve blocks for emergency use (e.g., high-priority trades)
    pub fn reserve_blocks(&self, count: usize) -> PoolResult<Vec<Arc<MemoryBlock>>> {
        let mut blocks = Vec::with_capacity(count);
        
        for _ in 0..count {
            let entry = self.free_list.iter().next()
                .map(|e| e.key().clone())
                .ok_or(PagedPoolError::OutOfMemory)?;

            self.free_list.remove(&entry);
            
            let block = &self.blocks[entry];
            block.set_state(BlockState::Reserved);
            
            blocks.push(Arc::clone(block));
        }

        let reserved = self.stats.reserved_blocks.fetch_add(count, Ordering::Relaxed) + count;
        
        Ok(blocks)
    }

    /// Release reserved blocks
    pub fn release_reserved(&self, blocks: &[Arc<MemoryBlock>]) -> PoolResult<()> {
        for block in blocks {
            if block.get_state() != BlockState::Reserved {
                return Err(PagedPoolError::BlockNotAllocated(block.id));
            }
            
            block.set_state(BlockState::Free);
            self.free_list.insert(block.id, ());
        }

        let reserved = self.stats.reserved_blocks.fetch_sub(blocks.len(), Ordering::Relaxed) - blocks.len();
        let free = self.stats.free_blocks.fetch_add(blocks.len(), Ordering::Relaxed) + blocks.len();

        Ok(())
    }
}

impl Drop for PagedAttentionPool {
    fn drop(&mut self) {
        let stats = self.get_stats();
        info!(
            "PagedAttentionPool shutting down: {} allocated, {} peak memory",
            stats.allocated_blocks, stats.peak_memory_bytes
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let config = PagedPoolConfig {
            num_blocks: 100,
            block_size_bytes: 1024,
            ..Default::default()
        };
        
        let pool = PagedAttentionPool::new(config);
        assert!(pool.is_ok());
        
        let pool = pool.unwrap();
        let stats = pool.get_stats();
        assert_eq!(stats.total_blocks, 100);
        assert_eq!(stats.free_blocks, 100);
        assert_eq!(stats.allocated_blocks, 0);
    }

    #[test]
    fn test_block_allocation() {
        let config = PagedPoolConfig {
            num_blocks: 10,
            ..Default::default()
        };
        
        let pool = PagedAttentionPool::new(config).unwrap();
        
        // Allocate a block
        let block = pool.allocate_block().unwrap();
        assert_eq!(block.get_state(), BlockState::Allocated);
        
        let stats = pool.get_stats();
        assert_eq!(stats.free_blocks, 9);
        assert_eq!(stats.allocated_blocks, 1);
    }

    #[test]
    fn test_block_release() {
        let config = PagedPoolConfig {
            num_blocks: 10,
            ..Default::default()
        };
        
        let pool = PagedAttentionPool::new(config).unwrap();
        
        // Allocate and release
        let block = pool.allocate_block().unwrap();
        let block_id = block.id;
        
        pool.release_block(&block).unwrap();
        
        let stats = pool.get_stats();
        assert_eq!(stats.free_blocks, 10);
        assert_eq!(stats.allocated_blocks, 0);
        
        // Block should be back in free list
        assert!(pool.free_list.contains_key(&block_id));
    }

    #[test]
    fn test_out_of_memory() {
        let config = PagedPoolConfig {
            num_blocks: 3,
            ..Default::default()
        };
        
        let pool = PagedAttentionPool::new(config).unwrap();
        
        // Allocate all blocks
        let _b1 = pool.allocate_block().unwrap();
        let _b2 = pool.allocate_block().unwrap();
        let _b3 = pool.allocate_block().unwrap();
        
        // Should fail
        let result = pool.allocate_block();
        assert!(matches!(result, Err(PagedPoolError::OutOfMemory)));
        
        let stats = pool.get_stats();
        assert_eq!(stats.allocation_failures, 1);
    }

    #[test]
    fn test_utilization() {
        let config = PagedPoolConfig {
            num_blocks: 100,
            ..Default::default()
        };
        
        let pool = PagedAttentionPool::new(config).unwrap();
        
        assert_eq!(pool.utilization(), 0.0);
        
        // Allocate 50 blocks
        for _ in 0..50 {
            let _ = pool.allocate_block().unwrap();
        }
        
        assert!((pool.utilization() - 0.5).abs() < 0.01);
    }
}
