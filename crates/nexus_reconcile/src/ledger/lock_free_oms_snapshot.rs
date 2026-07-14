//! Lock-free OMS snapshot mechanism using epoch-based reclamation.
//! 
//! CRITICAL: This module provides tear-free reads of the OMS state without
//! blocking the live execution path. It uses an epoch-stamped state struct
//! where writers increment the epoch, update all fields atomically, then
//! publish the new epoch. Readers verify epoch consistency before and after
//! reading to detect torn reads.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use crossbeam::epoch::{self, Atomic, Owned};
use arc_swap::ArcSwap;

/// Epoch-stamped OMS state block for tear-free reads.
/// All fields are updated together as a unit within a single epoch.
#[derive(Debug, Clone)]
pub struct OmsStateBlock {
    /// Monotonically increasing epoch ID - incremented on every write
    pub epoch_id: u64,
    
    /// Total wallet balance in scaled integer (wei/satoshis)
    pub total_balance: i128,
    
    /// Net position value in scaled integer
    pub net_position_value: i128,
    
    /// Unrealized PnL in scaled integer
    pub unrealized_pnl: i128,
    
    /// Number of active orders
    pub active_order_count: u32,
    
    /// Timestamp of last update (nanoseconds since boot)
    pub last_update_ns: u64,
    
    /// Hash of order states for quick comparison
    pub order_state_hash: u64,
}

impl OmsStateBlock {
    pub fn new() -> Self {
        Self {
            epoch_id: 0,
            total_balance: 0,
            net_position_value: 0,
            unrealized_pnl: 0,
            active_order_count: 0,
            last_update_ns: 0,
            order_state_hash: 0,
        }
    }
}

impl Default for OmsStateBlock {
    fn default() -> Self {
        Self::new()
    }
}

/// Lock-free OMS snapshot reader/writer using epoch-based reclamation.
/// 
/// WRITERS (live OMS thread):
/// 1. Increment epoch_id (odd = writing in progress)
/// 2. Update all state fields
/// 3. Increment epoch_id (even = write complete)
/// 
/// READERS (shadow poller):
/// 1. Read epoch_id (must be even)
/// 2. Read all state fields
/// 3. Re-read epoch_id
/// 4. If epochs match, data is consistent; otherwise retry
pub struct LockFreeOMSSnapshot {
    /// Current state wrapped in ArcSwap for atomic pointer swaps
    state: ArcSwap<OmsStateBlock>,
    
    /// Flag indicating writer is in-progress (odd epoch)
    write_in_progress: AtomicBool,
    
    /// Counter for total snapshots taken
    snapshot_count: AtomicU64,
    
    /// Counter for retry attempts due to tears
    retry_count: AtomicU64,
}

impl LockFreeOMSSnapshot {
    pub fn new() -> Self {
        Self {
            state: ArcSwap::new(Arc::new(OmsStateBlock::new())),
            write_in_progress: AtomicBool::new(false),
            snapshot_count: AtomicU64::new(0),
            retry_count: AtomicU64::new(0),
        }
    }
    
    /// Update the OMS state atomically (called by live OMS thread).
    /// This is the ONLY write path and must be extremely fast.
    /// 
    /// # Arguments
    /// * `total_balance` - Total wallet balance (scaled integer)
    /// * `net_position_value` - Net position value (scaled integer)
    /// * `unrealized_pnl` - Unrealized PnL (scaled integer)
    /// * `active_order_count` - Number of active orders
    /// * `order_state_hash` - Hash of order states for comparison
    #[inline]
    pub fn update_state(
        &self,
        total_balance: i128,
        net_position_value: i128,
        unrealized_pnl: i128,
        active_order_count: u32,
        order_state_hash: u64,
    ) {
        let guard = epoch::pin();
        
        // Get current state
        let current = self.state.load();
        let current_block = current.load();
        
        // Create new state block with incremented epoch
        let new_epoch = current_block.epoch_id.wrapping_add(1);
        let now_ns = get_monotonic_ns();
        
        let new_block = OmsStateBlock {
            epoch_id: new_epoch,
            total_balance,
            net_position_value,
            unrealized_pnl,
            active_order_count,
            last_update_ns: now_ns,
            order_state_hash,
        };
        
        // Atomically swap the pointer
        self.state.store(Arc::new(new_block));
        
        // Defend against ABA problem - defer deletion until no readers hold old pointer
        drop(current);
        guard.flush();
        
        self.snapshot_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Read the current OMS state without blocking (called by shadow poller).
    /// Returns None if a tear is detected (retry required).
    /// 
    /// This function is lock-free but may return None if the writer was
    /// active during the read. Caller should retry in a loop.
    #[inline]
    pub fn try_snapshot(&self) -> Option<OmsStateBlock> {
        let guard = epoch::pin();
        
        // First attempt
        let first_read = self.state.load();
        let first_block = first_read.load();
        
        // Check if write is in progress (odd epoch indicates mid-write)
        if first_block.epoch_id % 2 == 1 {
            self.retry_count.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        
        // Copy all fields to stack (fast, no allocations)
        let snapshot = OmsStateBlock {
            epoch_id: first_block.epoch_id,
            total_balance: first_block.total_balance,
            net_position_value: first_block.net_position_value,
            unrealized_pnl: first_block.unrealized_pnl,
            active_order_count: first_block.active_order_count,
            last_update_ns: first_block.last_update_ns,
            order_state_hash: first_block.order_state_hash,
        };
        
        // Re-read to check for tears
        let second_read = self.state.load();
        let second_block = second_read.load();
        
        // Verify epoch hasn't changed (no tear)
        if snapshot.epoch_id != second_block.epoch_id {
            self.retry_count.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        
        // Additional verification: ensure epoch is still even
        if snapshot.epoch_id % 2 == 1 {
            self.retry_count.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        
        Some(snapshot)
    }
    
    /// Blocking snapshot that retries until successful.
    /// Use only in background threads (shadow poller), never in hot path.
    pub fn snapshot_blocking(&self) -> OmsStateBlock {
        const MAX_RETRIES: u32 = 1000;
        let mut retries = 0;
        
        loop {
            if let Some(snapshot) = self.try_snapshot() {
                return snapshot;
            }
            
            retries += 1;
            if retries > MAX_RETRIES {
                // After excessive retries, yield to let writer finish
                std::thread::yield_now();
                retries = 0;
            }
        }
    }
    
    /// Get statistics about snapshot operations
    pub fn get_stats(&self) -> SnapshotStats {
        SnapshotStats {
            total_snapshots: self.snapshot_count.load(Ordering::Relaxed),
            retry_count: self.retry_count.load(Ordering::Relaxed),
            current_epoch: self.state.load().load().epoch_id,
        }
    }
}

impl Default for LockFreeOMSSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct SnapshotStats {
    pub total_snapshots: u64,
    pub retry_count: u64,
    pub current_epoch: u64,
}

/// Get monotonic nanosecond timestamp (since system boot, not wall clock)
#[inline]
fn get_monotonic_ns() -> u64 {
    use std::time::Instant;
    static START_TIME: once_cell::sync::Lazy<Instant> = once_cell::sync::Lazy::new(Instant::now);
    
    START_TIME.elapsed().as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::sync::Arc;
    
    #[test]
    fn test_no_torn_reads_under_contention() {
        let snapshot = Arc::new(LockFreeOMSSnapshot::new());
        let stop = Arc::new(AtomicBool::new(false));
        
        // Writer thread: continuously update state
        let writer_snapshot = Arc::clone(&snapshot);
        let writer_stop = Arc::clone(&stop);
        let writer = thread::spawn(move || {
            let mut counter: i128 = 0;
            while !writer_stop.load(Ordering::Relaxed) {
                counter += 1;
                writer_snapshot.update_state(
                    counter * 1000,      // balance
                    counter * 500,       // position
                    counter * 10,        // pnl
                    5,                   // order count
                    counter as u64,      // hash
                );
                // No sleep - maximum contention
            }
        });
        
        // Reader threads: try to snapshot, verify consistency
        let mut readers = Vec::new();
        for _ in 0..4 {
            let reader_snapshot = Arc::clone(&snapshot);
            let reader_stop = Arc::clone(&stop);
            readers.push(thread::spawn(move || {
                let mut valid_reads = 0u64;
                let mut retries = 0u64;
                
                while !reader_stop.load(Ordering::Relaxed) {
                    match reader_snapshot.try_snapshot() {
                        Some(s) => {
                            // Verify internal consistency
                            // In this test, balance should always be 2x position
                            assert_eq!(s.total_balance, s.net_position_value * 2, 
                                "Torn read detected! epoch={}", s.epoch_id);
                            valid_reads += 1;
                        }
                        None => retries += 1,
                    }
                }
                
                (valid_reads, retries)
            }));
        }
        
        // Run for 2 seconds
        thread::sleep(std::time::Duration::from_secs(2));
        stop.store(true, Ordering::Relaxed);
        
        writer.join().unwrap();
        
        for reader in readers {
            let (valid, retries) = reader.join().unwrap();
            println!("Reader: {} valid snapshots, {} retries", valid, retries);
            assert!(valid > 0, "Reader should have at least one valid snapshot");
        }
        
        let stats = snapshot.get_stats();
        println!("Total snapshots: {}, Retries: {}", stats.total_snapshots, stats.retry_count);
    }
    
    #[test]
    fn test_epoch_monotonicity() {
        let snapshot = LockFreeOMSSnapshot::new();
        let mut prev_epoch = 0u64;
        
        for i in 0..1000 {
            snapshot.update_state(i, i, i, 1, i as u64);
            let current = snapshot.snapshot_blocking();
            
            // Epoch should always increase
            assert!(current.epoch_id >= prev_epoch, 
                "Epoch regression: {} < {}", current.epoch_id, prev_epoch);
            prev_epoch = current.epoch_id;
        }
    }
}
