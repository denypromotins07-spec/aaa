/*
 * NEXUS-OMEGA Stage 20: PCIe Ring Buffer for DMA Zero-Copy Bridge
 * 
 * Lock-free, cache-coherent ring buffer implementation for PCIe DMA
 * communication between host CPU and FPGA.
 * 
 * CRITICAL: Uses write-combining memory mappings to ensure
 * FPGA DMA controller sees writes immediately (no CPU cache staleness).
 * 
 * ZERO ALLOCATION in hot paths.
 * NO unwrap() or expect() - all errors handled gracefully.
 */

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};

/// PCIe DMA descriptor structure (64 bytes, cache-line aligned)
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct DmaDescriptor {
    /// Physical address of data buffer (IOVA)
    pub address: u64,
    /// Length of transfer in bytes
    pub length: u32,
    /// Control flags
    pub control: DmaControl,
    /// Status written by FPGA after completion
    pub status: DmaStatus,
    /// Reserved for alignment
    pub reserved: [u32; 5],
}

/// DMA control flags
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct DmaControl {
    bits: u32,
}

impl DmaControl {
    pub const fn new() -> Self {
        Self { bits: 0 }
    }
    
    /// Set transfer direction: 0=H2C (Host to Card), 1=C2H (Card to Host)
    pub fn with_direction(self, card_to_host: bool) -> Self {
        Self { bits: if card_to_host { self.bits | 0x1 } else { self.bits & !0x1 } }
    }
    
    /// Set end-of-packet marker
    pub fn with_eop(self, is_eop: bool) -> Self {
        Self { bits: if is_eop { self.bits | 0x2 } else { self.bits & !0x2 } }
    }
    
    /// Enable completion interrupt
    pub fn with_interrupt(self, enable: bool) -> Self {
        Self { bits: if enable { self.bits | 0x4 } else { self.bits & !0x4 } }
    }
    
    /// Mark descriptor as owned by hardware
    pub fn with_owner(self, is_hw: bool) -> Self {
        Self { bits: if is_hw { self.bits | 0x8000_0000 } else { self.bits & !0x8000_0000 } }
    }
    
    pub fn is_hw_owned(&self) -> bool {
        self.bits & 0x8000_0000 != 0
    }
    
    pub fn direction_card_to_host(&self) -> bool {
        self.bits & 0x1 != 0
    }
}

/// DMA status written by FPGA
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct DmaStatus {
    bits: u32,
}

impl DmaStatus {
    pub const fn new() -> Self {
        Self { bits: 0 }
    }
    
    /// Check if transfer completed successfully
    pub fn is_complete(&self) -> bool {
        self.bits & 0x1 != 0
    }
    
    /// Check for error condition
    pub fn has_error(&self) -> bool {
        self.bits & 0x2 != 0
    }
    
    /// Get actual bytes transferred
    pub fn bytes_transferred(&self) -> u32 {
        (self.bits >> 16) & 0xFFFF
    }
}

/// Scatter-Gather list entry for large transfers
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SgEntry {
    pub iova: u64,      // I/O virtual address
    pub len: u32,       // Length of this segment
    pub is_last: bool,  // Last entry in chain
}

/// TX Ring: Host produces commands, FPGA consumes via DMA
pub struct TxRingBuffer {
    /// Pointer to descriptor array (mmap'd, write-combining)
    descriptors: *mut DmaDescriptor,
    /// Producer index (host writes)
    producer: *mut AtomicU32,
    /// Consumer index (FPGA reads via DMA)
    consumer: *const AtomicU32,
    /// Ring size mask
    mask: u32,
    /// Cached indices to reduce MMIO reads
    cached_prod: u32,
    cached_cons: u32,
    /// Statistics
    total_submitted: AtomicU64,
    total_completed: AtomicU64,
}

// SAFETY: Single-producer single-consumer pattern
unsafe impl Send for TxRingBuffer {}
unsafe impl Sync for TxRingBuffer {}

impl TxRingBuffer {
    /// Create a new TX ring buffer
    /// 
    /// # Safety
    /// - `descriptors` must point to valid mmap'd write-combining memory
    /// - `producer` must be in host-accessible memory
    /// - `consumer` must be in FPGA-readable memory (DMA coherent)
    pub unsafe fn new(
        descriptors: *mut DmaDescriptor,
        producer: *mut AtomicU32,
        consumer: *const AtomicU32,
        num_descriptors: u32,
    ) -> Result<Self, &'static str> {
        if descriptors.is_null() || producer.is_null() || consumer.is_null() {
            return Err("Null pointer in TxRingBuffer construction");
        }
        
        if !num_descriptors.is_power_of_two() {
            return Err("TX ring size must be power of 2");
        }
        
        Ok(Self {
            descriptors,
            producer,
            consumer,
            mask: num_descriptors.wrapping_sub(1),
            cached_prod: 0,
            cached_cons: 0,
            total_submitted: AtomicU64::new(0),
            total_completed: AtomicU64::new(0),
        })
    }
    
    /// Submit a DMA transfer descriptor
    /// Returns true on success, false if ring is full
    pub fn submit(&mut self, desc: DmaDescriptor) -> bool {
        // Check space using cached values first
        let available = self.mask + 1 - (self.cached_prod - self.cached_cons);
        
        if available == 0 {
            // Reload consumer from FPGA
            self.cached_cons = unsafe { (*self.consumer).load(Ordering::Acquire) };
            
            let available = self.mask + 1 - (self.cached_prod - self.cached_cons);
            if available == 0 {
                return false; // Ring full
            }
        }
        
        // Write descriptor to ring
        let idx = (self.cached_prod & self.mask) as usize;
        unsafe {
            *self.descriptors.add(idx) = desc;
        }
        
        self.cached_prod = self.cached_prod.wrapping_add(1);
        self.total_submitted.fetch_add(1, Ordering::Relaxed);
        true
    }
    
    /// Ring doorbell to notify FPGA
    pub fn kick(&mut self) {
        // Memory barrier ensures all descriptor writes visible before index update
        unsafe {
            (*self.producer).store(self.cached_prod, Ordering::Release);
        }
    }
    
    /// Submit a scatter-gather transfer
    /// Returns number of descriptors used, or 0 on failure
    pub fn submit_sg(&mut self, sg_list: &[SgEntry]) -> u32 {
        if sg_list.is_empty() {
            return 0;
        }
        
        // Check if we have enough space
        let available = self.mask + 1 - (self.cached_prod - self.cached_cons);
        if available < sg_list.len() as u32 {
            self.cached_cons = unsafe { (*self.consumer).load(Ordering::Acquire) };
            
            let available = self.mask + 1 - (self.cached_prod - self.cached_cons);
            if available < sg_list.len() as u32 {
                return 0;
            }
        }
        
        // Chain descriptors
        for (i, entry) in sg_list.iter().enumerate() {
            let mut control = DmaControl::new()
                .with_direction(false) // H2C
                .with_eop(entry.is_last)
                .with_owner(true);
            
            if entry.is_last {
                control = control.with_interrupt(true);
            }
            
            let desc = DmaDescriptor {
                address: entry.iova,
                length: entry.len,
                control,
                status: DmaStatus::default(),
                reserved: [0; 5],
            };
            
            let idx = (self.cached_prod & self.mask) as usize;
            unsafe {
                *self.descriptors.add(idx) = desc;
            }
            
            self.cached_prod = self.cached_prod.wrapping_add(1);
        }
        
        self.total_submitted.fetch_add(sg_list.len() as u64, Ordering::Relaxed);
        sg_list.len() as u32
    }
    
    /// Check completions without consuming
    pub fn pending_completions(&mut self) -> u32 {
        let prod = self.cached_prod;
        let cons = unsafe { (*self.consumer).load(Ordering::Acquire) };
        self.cached_cons = cons;
        prod.wrapping_sub(cons)
    }
    
    /// Get statistics
    pub fn stats(&self) -> (u64, u64) {
        (
            self.total_submitted.load(Ordering::Relaxed),
            self.total_completed.load(Ordering::Relaxed),
        )
    }
}

/// RX Ring: FPGA produces data, host consumes
pub struct RxRingBuffer {
    descriptors: *mut DmaDescriptor,
    producer: *const AtomicU32,
    consumer: *mut AtomicU32,
    mask: u32,
    cached_prod: u32,
    cached_cons: u32,
    total_received: AtomicU64,
    dropped: AtomicU64,
}

unsafe impl Send for RxRingBuffer {}
unsafe impl Sync for RxRingBuffer {}

impl RxRingBuffer {
    pub unsafe fn new(
        descriptors: *mut DmaDescriptor,
        producer: *const AtomicU32,
        consumer: *mut AtomicU32,
        num_descriptors: u32,
    ) -> Result<Self, &'static str> {
        if descriptors.is_null() || producer.is_null() || consumer.is_null() {
            return Err("Null pointer in RxRingBuffer construction");
        }
        
        if !num_descriptors.is_power_of_two() {
            return Err("RX ring size must be power of 2");
        }
        
        Ok(Self {
            descriptors,
            producer,
            consumer,
            mask: num_descriptors.wrapping_sub(1),
            cached_prod: 0,
            cached_cons: 0,
            total_received: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        })
    }
    
    /// Consume a completed descriptor
    pub fn consume(&mut self) -> Option<DmaDescriptor> {
        if self.cached_cons >= self.cached_prod {
            self.cached_prod = unsafe { (*self.producer).load(Ordering::Acquire) };
            
            if self.cached_cons >= self.cached_prod {
                return None;
            }
        }
        
        let idx = (self.cached_cons & self.mask) as usize;
        let desc = unsafe { *self.descriptors.add(idx) };
        
        self.cached_cons = self.cached_cons.wrapping_add(1);
        self.total_received.fetch_add(1, Ordering::Relaxed);
        Some(desc)
    }
    
    /// Return descriptors to FPGA for reuse
    pub fn refill(&mut self, count: u32) {
        if count > 0 {
            unsafe {
                (*self.consumer).store(self.cached_cons, Ordering::Release);
            }
        }
    }
    
    pub fn stats(&self) -> (u64, u64) {
        (
            self.total_received.load(Ordering::Relaxed),
            self.dropped.load(Ordering::Relaxed),
        )
    }
}

/// Complete PCIe DMA Ring Buffer Set
pub struct PcieDmaRingSet {
    tx_ring: Option<TxRingBuffer>,
    rx_ring: Option<RxRingBuffer>,
    /// Doorbell register pointer (MMIO)
    tx_doorbell: *mut AtomicU32,
    rx_doorbell: *mut AtomicU32,
}

unsafe impl Send for PcieDmaRingSet {}
unsafe impl Sync for PcieDmaRingSet {}

impl PcieDmaRingSet {
    pub fn new(tx_db: *mut AtomicU32, rx_db: *mut AtomicU32) -> Self {
        Self {
            tx_ring: None,
            rx_ring: None,
            tx_doorbell: tx_db,
            rx_doorbell: rx_db,
        }
    }
    
    pub fn set_tx_ring(&mut self, ring: TxRingBuffer) {
        self.tx_ring = Some(ring);
    }
    
    pub fn set_rx_ring(&mut self, ring: RxRingBuffer) {
        self.rx_ring = Some(ring);
    }
    
    /// Kick TX doorbell
    pub fn kick_tx(&mut self) {
        if let Some(ref mut tx) = self.tx_ring {
            tx.kick();
        }
    }
    
    /// Process completions on both rings
    pub fn pump(&mut self) -> usize {
        let mut processed = 0usize;
        
        // Check TX completions
        if let Some(ref mut tx) = self.tx_ring {
            processed += tx.pending_completions() as usize;
        }
        
        // Consume RX data
        if let Some(ref mut rx) = self.rx_ring {
            while rx.consume().is_some() {
                processed += 1;
            }
        }
        
        processed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dma_descriptor_size() {
        // Verify descriptor is cache-line aligned
        assert_eq!(std::mem::size_of::<DmaDescriptor>(), 64);
        assert_eq!(std::mem::align_of::<DmaDescriptor>(), 64);
    }
    
    #[test]
    fn test_dma_control_flags() {
        let ctrl = DmaControl::new()
            .with_direction(true)
            .with_eop(true)
            .with_interrupt(true)
            .with_owner(true);
        
        assert!(ctrl.direction_card_to_host());
        assert!(ctrl.is_hw_owned());
    }
    
    #[test]
    fn test_power_of_two_validation() {
        assert!(256u32.is_power_of_two());
        assert!(1024u32.is_power_of_two());
        assert!(!255u32.is_power_of_two());
    }
}
