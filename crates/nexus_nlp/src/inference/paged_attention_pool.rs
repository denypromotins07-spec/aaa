//! PagedAttention Pool Manager
//! 
//! Pre-allocates GPU VRAM blocks for KV cache management, ensuring
//! LLM inference never triggers CUDA OOM errors during high-volatility events.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Configuration for the paged attention pool
#[derive(Debug, Clone)]
pub struct PagedAttentionConfig {
    pub num_blocks: usize,
    pub block_size: usize, // tokens per block
    pub head_size: usize,
    pub num_heads: usize,
    pub dtype_bytes: usize, // bytes per element (2 for FP16, 4 for FP32)
}

impl Default for PagedAttentionConfig {
    fn default() -> Self {
        // Default: 1000 blocks, 16 tokens/block, 64 head size, 32 heads, FP16
        Self {
            num_blocks: 1000,
            block_size: 16,
            head_size: 64,
            num_heads: 32,
            dtype_bytes: 2,
        }
    }
}

/// Block state tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockState {
    Free,
    Allocated,
    Active,
}

/// Metadata for a single KV cache block
struct BlockMetadata {
    state: BlockState,
    ref_count: AtomicUsize,
    sequence_id: AtomicU64,
    token_count: AtomicUsize,
}

/// PagedAttention memory pool
pub struct PagedAttentionPool {
    config: PagedAttentionConfig,
    blocks: Vec<BlockMetadata>,
    free_list: AtomicUsize, // Index of first free block (linked list via next_free array)
    next_free: Vec<AtomicUsize>, // Next free block index
    allocated_count: AtomicUsize,
    total_memory_bytes: usize,
}

// SAFETY: All operations use atomics
unsafe impl Send for PagedAttentionPool {}
unsafe impl Sync for PagedAttentionPool {}

impl PagedAttentionPool {
    /// Create a new paged attention pool
    pub fn new(config: PagedAttentionConfig) -> Result<Self, PoolError> {
        if config.num_blocks == 0 {
            return Err(PoolError::InvalidConfig("num_blocks must be > 0"));
        }

        // Calculate memory requirements
        let kv_size_per_block = config.block_size * config.head_size * config.num_heads * config.dtype_bytes;
        let total_memory_bytes = config.num_blocks * kv_size_per_block * 2; // Key + Value

        // Initialize block metadata
        let mut blocks = Vec::with_capacity(config.num_blocks);
        let mut next_free = Vec::with_capacity(config.num_blocks);

        for i in 0..config.num_blocks {
            blocks.push(BlockMetadata {
                state: BlockState::Free,
                ref_count: AtomicUsize::new(0),
                sequence_id: AtomicU64::new(0),
                token_count: AtomicUsize::new(0),
            });
            next_free.push(AtomicUsize::new(i + 1));
        }

        Ok(Self {
            config,
            blocks,
            free_list: AtomicUsize::new(0),
            next_free,
            allocated_count: AtomicUsize::new(0),
            total_memory_bytes,
        })
    }

    /// Allocate a block for a sequence
    pub fn allocate(&self, sequence_id: u64) -> Option<usize> {
        loop {
            let current_head = self.free_list.load(Ordering::Acquire);
            
            if current_head >= self.config.num_blocks {
                // No free blocks available
                return None;
            }

            // Try to claim this block
            let next = self.next_free[current_head].load(Ordering::Relaxed);
            
            if self.free_list.compare_exchange_weak(
                current_head,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ).is_ok() {
                // Successfully claimed block
                let block = &self.blocks[current_head];
                block.state = BlockState::Allocated;
                block.ref_count.store(1, Ordering::Relaxed);
                block.sequence_id.store(sequence_id, Ordering::Relaxed);
                block.token_count.store(0, Ordering::Relaxed);
                
                self.allocated_count.fetch_add(1, Ordering::Relaxed);
                
                return Some(current_head);
            }
            // CAS failed, retry
        }
    }

    /// Free a block
    pub fn free(&self, block_idx: usize) -> Result<(), PoolError> {
        if block_idx >= self.config.num_blocks {
            return Err(PoolError::InvalidBlockIndex(block_idx));
        }

        let block = &self.blocks[block_idx];
        
        // Decrement ref count
        let prev_ref = block.ref_count.fetch_sub(1, Ordering::AcqRel);
        
        if prev_ref != 1 {
            // Still referenced elsewhere
            return Ok(());
        }

        // Reset block state
        block.state = BlockState::Free;
        block.sequence_id.store(0, Ordering::Relaxed);
        block.token_count.store(0, Ordering::Relaxed);

        // Add back to free list
        loop {
            let current_head = self.free_list.load(Ordering::Relaxed);
            self.next_free[block_idx].store(current_head, Ordering::Relaxed);
            
            if self.free_list.compare_exchange_weak(
                current_head,
                block_idx,
                Ordering::Release,
                Ordering::Relaxed,
            ).is_ok() {
                self.allocated_count.fetch_sub(1, Ordering::Relaxed);
                return Ok(());
            }
        }
    }

    /// Increment reference count for sharing blocks between sequences
    pub fn share(&self, block_idx: usize) -> Result<(), PoolError> {
        if block_idx >= self.config.num_blocks {
            return Err(PoolError::InvalidBlockIndex(block_idx));
        }

        let block = &self.blocks[block_idx];
        block.ref_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Get the number of tokens in a block
    pub fn get_token_count(&self, block_idx: usize) -> Result<usize, PoolError> {
        if block_idx >= self.config.num_blocks {
            return Err(PoolError::InvalidBlockIndex(block_idx));
        }
        Ok(self.blocks[block_idx].token_count.load(Ordering::Relaxed))
    }

    /// Set the number of tokens in a block
    pub fn set_token_count(&self, block_idx: usize, count: usize) -> Result<(), PoolError> {
        if block_idx >= self.config.num_blocks {
            return Err(PoolError::InvalidBlockIndex(block_idx));
        }
        if count > self.config.block_size {
            return Err(PoolError::TokenCountExceeded);
        }
        self.blocks[block_idx].token_count.store(count, Ordering::Relaxed);
        Ok(())
    }

    /// Get statistics about pool usage
    pub fn stats(&self) -> PoolStats {
        let allocated = self.allocated_count.load(Ordering::Relaxed);
        let free = self.config.num_blocks - allocated;
        
        PoolStats {
            total_blocks: self.config.num_blocks,
            allocated_blocks: allocated,
            free_blocks: free,
            utilization: allocated as f64 / self.config.num_blocks as f64,
            total_memory_bytes: self.total_memory_bytes,
        }
    }

    /// Check if pool is running low on free blocks
    pub fn is_low_on_blocks(&self, threshold: f64) -> bool {
        let stats = self.stats();
        stats.utilization >= threshold
    }
}

/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_blocks: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub utilization: f64,
    pub total_memory_bytes: usize,
}

/// Pool errors
#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    #[error("Invalid config: {0}")]
    InvalidConfig(&'static str),
    #[error("Invalid block index: {0}")]
    InvalidBlockIndex(usize),
    #[error("Token count exceeded block size")]
    TokenCountExceeded,
    #[error("Out of memory")]
    OutOfMemory,
}

/// Sequence manager that tracks block allocations per sequence
pub struct SequenceManager {
    sequence_blocks: dashmap::DashMap<u64, Vec<usize>>,
}

impl SequenceManager {
    /// Create a new sequence manager
    pub fn new() -> Self {
        Self {
            sequence_blocks: dashmap::DashMap::new(),
        }
    }

    /// Add a block to a sequence
    pub fn add_block(&self, sequence_id: u64, block_idx: usize) {
        self.sequence_blocks
            .entry(sequence_id)
            .or_insert_with(Vec::new)
            .push(block_idx);
    }

    /// Remove all blocks for a sequence and free them
    pub fn free_sequence(&self, sequence_id: u64, pool: &PagedAttentionPool) -> Result<(), PoolError> {
        if let Some((_, blocks)) = self.sequence_blocks.remove(&sequence_id) {
            for block_idx in blocks {
                pool.free(block_idx)?;
            }
        }
        Ok(())
    }

    /// Get blocks for a sequence
    pub fn get_blocks(&self, sequence_id: u64) -> Option<Vec<usize>> {
        self.sequence_blocks.get(&sequence_id).map(|r| r.clone())
    }
}

impl Default for SequenceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_allocation() {
        let config = PagedAttentionConfig {
            num_blocks: 10,
            block_size: 16,
            head_size: 64,
            num_heads: 8,
            dtype_bytes: 2,
        };

        let pool = PagedAttentionPool::new(config).unwrap();
        
        // Allocate blocks
        let block1 = pool.allocate(1).unwrap();
        let block2 = pool.allocate(2).unwrap();
        
        assert_ne!(block1, block2);
        
        let stats = pool.stats();
        assert_eq!(stats.allocated_blocks, 2);
        assert_eq!(stats.free_blocks, 8);
        
        // Free block1
        pool.free(block1).unwrap();
        
        let stats = pool.stats();
        assert_eq!(stats.allocated_blocks, 1);
    }

    #[test]
    fn test_pool_exhaustion() {
        let config = PagedAttentionConfig {
            num_blocks: 3,
            block_size: 16,
            head_size: 64,
            num_heads: 8,
            dtype_bytes: 2,
        };

        let pool = PagedAttentionPool::new(config).unwrap();
        
        // Allocate all blocks
        let _b1 = pool.allocate(1);
        let _b2 = pool.allocate(2);
        let _b3 = pool.allocate(3);
        
        // Should fail
        let b4 = pool.allocate(4);
        assert!(b4.is_none());
        
        // Free one and try again
        pool.free(_b1.unwrap()).unwrap();
        let b5 = pool.allocate(5);
        assert!(b5.is_some());
    }
}
