//! Chapter 2: L3 Slab Allocator for Order-by-Order Tracking
//!
//! This module implements a pre-allocated slab allocator for tracking
//! individual order IDs, sizes, and queue positions without pointer chasing
//! or heap allocations during hot-path updates.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use nexus_core::memory::cache_padder::CachePadded64;

/// Maximum number of tracked orders in L3 book
pub const MAX_ORDER_RECORDS: usize = 65536;

/// Sentinel value for invalid order index
pub const INVALID_ORDER_INDEX: usize = usize::MAX;

/// Order record for L3 tracking
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OrderRecord {
    /// Unique order ID from exchange
    pub order_id: CachePadded64<AtomicU64>,
    /// Price in nanodollars
    pub price: CachePadded64<AtomicU64>,
    /// Volume in base units * 1e9
    pub volume: CachePadded64<AtomicU64>,
    /// Side (0=Bid, 1=Ask)
    pub side: CachePadded64<AtomicU8>,
    /// Queue position within price level
    pub queue_position: CachePadded64<AtomicU32>,
    /// Next order index in same price level (intrusive list)
    pub next_index: CachePadded64<AtomicUsize>,
    /// Previous order index in same price level
    pub prev_index: CachePadded64<AtomicUsize>,
    /// Whether this slot is occupied
    pub occupied: CachePadded64<AtomicBool>,
    /// Padding for cache alignment
    pub _padding: [u8; 38],
}

use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::AtomicU32;

// SAFETY: OrderRecord uses atomic operations
unsafe impl Send for OrderRecord {}
unsafe impl Sync for OrderRecord {}

impl OrderRecord {
    #[inline]
    pub const fn new() -> Self {
        Self {
            order_id: CachePadded64::new(AtomicU64::new(0)),
            price: CachePadded64::new(AtomicU64::new(0)),
            volume: CachePadded64::new(AtomicU64::new(0)),
            side: CachePadded64::new(AtomicU8::new(0)),
            queue_position: CachePadded64::new(AtomicU32::new(0)),
            next_index: CachePadded64::new(AtomicUsize::new(INVALID_ORDER_INDEX)),
            prev_index: CachePadded64::new(AtomicUsize::new(INVALID_ORDER_INDEX)),
            occupied: CachePadded64::new(AtomicBool::new(false)),
            _padding: [0; 38],
        }
    }

    #[inline]
    pub fn set(&self, order_id: u64, price: u64, volume: u64, side: u8) {
        self.order_id.0.store(order_id, Ordering::Release);
        self.price.0.store(price, Ordering::Release);
        self.volume.0.store(volume, Ordering::Release);
        self.side.0.store(side, Ordering::Release);
        self.queue_position.0.store(0, Ordering::Release);
        self.next_index.0.store(INVALID_ORDER_INDEX, Ordering::Release);
        self.prev_index.0.store(INVALID_ORDER_INDEX, Ordering::Release);
        self.occupied.0.store(true, Ordering::Release);
    }

    #[inline]
    pub fn clear(&self) {
        self.order_id.0.store(0, Ordering::Release);
        self.price.0.store(0, Ordering::Release);
        self.volume.0.store(0, Ordering::Release);
        self.side.0.store(0, Ordering::Release);
        self.queue_position.0.store(0, Ordering::Release);
        self.next_index.0.store(INVALID_ORDER_INDEX, Ordering::Release);
        self.prev_index.0.store(INVALID_ORDER_INDEX, Ordering::Release);
        self.occupied.0.store(false, Ordering::Release);
    }

    #[inline]
    pub fn get_order_id(&self) -> u64 {
        self.order_id.0.load(Ordering::Acquire)
    }

    #[inline]
    pub fn get_price(&self) -> u64 {
        self.price.0.load(Ordering::Acquire)
    }

    #[inline]
    pub fn get_volume(&self) -> u64 {
        self.volume.0.load(Ordering::Acquire)
    }

    #[inline]
    pub fn get_side(&self) -> u8 {
        self.side.0.load(Ordering::Acquire)
    }

    #[inline]
    pub fn is_occupied(&self) -> bool {
        self.occupied.0.load(Ordering::Acquire)
    }
}

impl Default for OrderRecord {
    fn default() -> Self {
        Self::new()
    }
}

/// L3 Slab Allocator for order tracking
pub struct L3SlabAllocator {
    /// Pre-allocated order records
    records: CachePadded64<[OrderRecord; MAX_ORDER_RECORDS]>,
    /// Free list head (index of first free slot)
    free_list_head: CachePadded64<AtomicUsize>,
    /// Count of allocated records
    allocated_count: CachePadded64<AtomicUsize>,
    /// Count of lookups by order ID
    lookup_count: CachePadded64<AtomicU64>,
}

// SAFETY: L3SlabAllocator uses atomic operations
unsafe impl Send for L3SlabAllocator {}
unsafe impl Sync for L3SlabAllocator {}

impl L3SlabAllocator {
    /// Create a new L3 slab allocator
    #[inline]
    pub fn new() -> Self {
        let records = CachePadded64::new(std::array::from_fn(|_| OrderRecord::new()));
        
        // Initialize free list (all slots linked)
        for i in 0..MAX_ORDER_RECORDS - 1 {
            records.0[i].next_index.0.store(i + 1, Ordering::Release);
        }
        records.0[MAX_ORDER_RECORDS - 1].next_index.0.store(INVALID_ORDER_INDEX, Ordering::Release);
        
        Self {
            records,
            free_list_head: CachePadded64::new(AtomicUsize::new(0)),
            allocated_count: CachePadded64::new(AtomicUsize::new(0)),
            lookup_count: CachePadded64::new(AtomicU64::new(0)),
        }
    }

    /// Allocate a new order record
    #[inline]
    pub fn allocate(&self, order_id: u64, price: u64, volume: u64, side: u8) -> Option<usize> {
        let head = self.free_list_head.0.swap(INVALID_ORDER_INDEX, Ordering::AcqRel);
        
        if head == INVALID_ORDER_INDEX {
            // Slab is full
            return None;
        }
        
        let record = &self.records.0[head];
        
        // Get next free slot before we overwrite it
        let next_free = record.next_index.0.load(Ordering::Acquire);
        
        // Update free list head
        self.free_list_head.0.store(next_free, Ordering::Release);
        
        // Initialize the record
        record.set(order_id, price, volume, side);
        
        // Update allocated count
        self.allocated_count.0.fetch_add(1, Ordering::AcqRel);
        
        Some(head)
    }

    /// Free an order record
    #[inline]
    pub fn free(&self, index: usize) -> Result<(), &'static str> {
        if index >= MAX_ORDER_RECORDS {
            return Err("Index out of bounds");
        }
        
        let record = &self.records.0[index];
        
        if !record.is_occupied() {
            return Err("Slot not occupied");
        }
        
        // Remove from intrusive list
        let prev = record.prev_index.0.load(Ordering::Acquire);
        let next = record.next_index.0.load(Ordering::Acquire);
        
        if prev != INVALID_ORDER_INDEX {
            let prev_record = &self.records.0[prev];
            prev_record.next_index.0.store(next, Ordering::Release);
        }
        
        if next != INVALID_ORDER_INDEX {
            let next_record = &self.records.0[next];
            next_record.prev_index.0.store(prev, Ordering::Release);
        }
        
        // Clear the record
        record.clear();
        
        // Add back to free list
        let current_head = self.free_list_head.0.load(Ordering::Acquire);
        record.next_index.0.store(current_head, Ordering::Release);
        
        // CAS loop to update head
        let mut old_head = current_head;
        loop {
            match self.free_list_head.0.compare_exchange_weak(
                old_head,
                index,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(new_head) => old_head = new_head,
            }
        }
        
        // Update allocated count
        self.allocated_count.0.fetch_sub(1, Ordering::AcqRel);
        
        Ok(())
    }

    /// Find order by ID (linear scan - optimized in production with hash map)
    #[inline]
    pub fn find_by_order_id(&self, order_id: u64) -> Option<usize> {
        self.lookup_count.0.fetch_add(1, Ordering::Relaxed);
        
        for i in 0..MAX_ORDER_RECORDS {
            let record = &self.records.0[i];
            if record.is_occupied() && record.get_order_id() == order_id {
                return Some(i);
            }
        }
        
        None
    }

    /// Get reference to order record at index
    #[inline]
    pub fn get_record(&self, index: usize) -> Option<&OrderRecord> {
        if index >= MAX_ORDER_RECORDS {
            return None;
        }
        
        let record = &self.records.0[index];
        if record.is_occupied() {
            Some(record)
        } else {
            None
        }
    }

    /// Insert order into price level's intrusive list
    #[inline]
    pub fn insert_into_level(&self, order_index: usize, tail_index: usize) -> Result<u32, &'static str> {
        let order_record = self.get_record(order_index)
            .ok_or("Invalid order index")?;
        
        if tail_index == INVALID_ORDER_INDEX {
            // First order in this level
            order_record.queue_position.0.store(1, Ordering::Release);
            return Ok(1);
        }
        
        let tail_record = self.get_record(tail_index)
            .ok_or("Invalid tail index")?;
        
        // Add to end of list
        let current_tail_next = tail_record.next_index.0.load(Ordering::Acquire);
        
        order_record.prev_index.0.store(tail_index, Ordering::Release);
        order_record.next_index.0.store(current_tail_next, Ordering::Release);
        
        tail_record.next_index.0.store(order_index, Ordering::Release);
        
        if current_tail_next != INVALID_ORDER_INDEX {
            let next_record = self.get_record(current_tail_next)
                .ok_or("Invalid next index")?;
            next_record.prev_index.0.store(order_index, Ordering::Release);
        }
        
        // Calculate queue position
        let tail_pos = tail_record.queue_position.0.load(Ordering::Acquire);
        let new_pos = tail_pos + 1;
        order_record.queue_position.0.store(new_pos, Ordering::Release);
        
        Ok(new_pos)
    }

    /// Remove order from intrusive list
    #[inline]
    pub fn remove_from_level(&self, order_index: usize) -> Result<(), &'static str> {
        let record = self.get_record(order_index)
            .ok_or("Invalid order index")?;
        
        let prev = record.prev_index.0.load(Ordering::Acquire);
        let next = record.next_index.0.load(Ordering::Acquire);
        
        if prev != INVALID_ORDER_INDEX {
            let prev_record = self.get_record(prev)
                .ok_or("Invalid prev index")?;
            prev_record.next_index.0.store(next, Ordering::Release);
        }
        
        if next != INVALID_ORDER_INDEX {
            let next_record = self.get_record(next)
                .ok_or("Invalid next index")?;
            next_record.prev_index.0.store(prev, Ordering::Release);
        }
        
        // Update queue positions for subsequent orders (simplified - just decrement)
        if next != INVALID_ORDER_INDEX {
            let mut current = next;
            while current != INVALID_ORDER_INDEX {
                let curr_record = self.get_record(current)
                    .ok_or("Invalid index in chain")?;
                let pos = curr_record.queue_position.0.load(Ordering::Acquire);
                if pos > 1 {
                    curr_record.queue_position.0.store(pos - 1, Ordering::Release);
                    current = curr_record.next_index.0.load(Ordering::Acquire);
                } else {
                    break;
                }
            }
        }
        
        Ok(())
    }

    /// Get allocated count
    #[inline]
    pub fn allocated_count(&self) -> usize {
        self.allocated_count.0.load(Ordering::Relaxed)
    }

    /// Get free count
    #[inline]
    pub fn free_count(&self) -> usize {
        MAX_ORDER_RECORDS - self.allocated_count()
    }

    /// Get lookup count
    #[inline]
    pub fn lookup_count(&self) -> u64 {
        self.lookup_count.0.load(Ordering::Relaxed)
    }

    /// Clear all records
    #[inline]
    pub fn clear(&self) {
        for i in 0..MAX_ORDER_RECORDS {
            self.records.0[i].clear();
            if i < MAX_ORDER_RECORDS - 1 {
                self.records.0[i].next_index.0.store(i + 1, Ordering::Release);
            } else {
                self.records.0[i].next_index.0.store(INVALID_ORDER_INDEX, Ordering::Release);
            }
        }
        
        self.free_list_head.0.store(0, Ordering::Release);
        self.allocated_count.0.store(0, Ordering::Release);
        self.lookup_count.0.store(0, Ordering::Release);
    }
}

impl Default for L3SlabAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slab_allocation() {
        let slab = L3SlabAllocator::new();
        assert_eq!(slab.allocated_count(), 0);
        assert_eq!(slab.free_count(), MAX_ORDER_RECORDS);
        
        let index = slab.allocate(12345, 100_000_000_000, 1_000_000_000, 0);
        assert!(index.is_some());
        assert_eq!(slab.allocated_count(), 1);
        
        let record = slab.get_record(index.unwrap()).unwrap();
        assert_eq!(record.get_order_id(), 12345);
        assert_eq!(record.get_price(), 100_000_000_000);
    }

    #[test]
    fn test_slab_free() {
        let slab = L3SlabAllocator::new();
        
        let index = slab.allocate(12345, 100_000_000_000, 1_000_000_000, 0).unwrap();
        assert!(slab.free(index).is_ok());
        assert_eq!(slab.allocated_count(), 0);
        
        // Should be able to allocate again
        let index2 = slab.allocate(67890, 100_100_000_000, 2_000_000_000, 1);
        assert!(index2.is_some());
    }

    #[test]
    fn test_find_by_order_id() {
        let slab = L3SlabAllocator::new();
        
        let idx1 = slab.allocate(111, 100_000_000_000, 1_000_000_000, 0).unwrap();
        let idx2 = slab.allocate(222, 100_100_000_000, 2_000_000_000, 1).unwrap();
        
        assert_eq!(slab.find_by_order_id(111), Some(idx1));
        assert_eq!(slab.find_by_order_id(222), Some(idx2));
        assert_eq!(slab.find_by_order_id(999), None);
    }

    #[test]
    fn test_intrusive_list() {
        let slab = L3SlabAllocator::new();
        
        let idx1 = slab.allocate(1, 100_000_000_000, 1_000_000_000, 0).unwrap();
        let idx2 = slab.allocate(2, 100_000_000_000, 2_000_000_000, 0).unwrap();
        let idx3 = slab.allocate(3, 100_000_000_000, 3_000_000_000, 0).unwrap();
        
        // Insert into level
        let pos1 = slab.insert_into_level(idx1, INVALID_ORDER_INDEX).unwrap();
        assert_eq!(pos1, 1);
        
        let pos2 = slab.insert_into_level(idx2, idx1).unwrap();
        assert_eq!(pos2, 2);
        
        let pos3 = slab.insert_into_level(idx3, idx2).unwrap();
        assert_eq!(pos3, 3);
        
        // Remove middle
        assert!(slab.remove_from_level(idx2).is_ok());
        
        // Verify idx1 -> idx3 linkage
        let rec1 = slab.get_record(idx1).unwrap();
        assert_eq!(rec1.next_index.0.load(Ordering::Acquire), idx3);
    }

    #[test]
    fn test_slab_exhaustion() {
        let slab = L3SlabAllocator::new();
        
        // Allocate all but one
        for i in 0..MAX_ORDER_RECORDS - 1 {
            let result = slab.allocate(i as u64, 100_000_000_000, 1_000_000_000, 0);
            assert!(result.is_some());
        }
        
        // Allocate last one
        let last = slab.allocate(MAX_ORDER_RECORDS as u64, 100_000_000_000, 1_000_000_000, 0);
        assert!(last.is_some());
        
        // Next should fail
        let overflow = slab.allocate(999999, 100_000_000_000, 1_000_000_000, 0);
        assert!(overflow.is_none());
    }
}
