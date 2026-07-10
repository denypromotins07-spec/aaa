/*
 * NEXUS-OMEGA Stage 20: TCP Piggyback Handler for XDP
 * 
 * Rust userspace companion to the eBPF XDP program.
 * Manages the ring buffer communication, payload preparation,
 * and coordinates with the SmartNIC for actual packet injection.
 * 
 * ZERO ALLOCATION in hot path - uses pre-allocated buffers.
 * NO unwrap() or expect() - all errors handled gracefully.
 */

use std::cell::UnsafeCell;
use std::mem::{self, MaybeUninit};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// Maximum payload size that can be piggybacked on a single ACK
const MAX_PIGGYBACK_PAYLOAD: usize = 64;

/// Maximum number of concurrent flows tracked
const MAX_FLOWS: usize = 65536;

/// Ring buffer size for execution events (must be power of 2)
const RING_BUFFER_SIZE: usize = 256 * 1024;

/// Flow key type (hashed from IP/port tuple)
type FlowKey = u32;

/// Execution payload structure matching eBPF struct
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ExecutionPayload {
    pub data: [u8; MAX_PIGGYBACK_PAYLOAD],
    pub len: u32,
    pub flow_id: FlowKey,
    pub active: u8,
}

impl Default for ExecutionPayload {
    fn default() -> Self {
        Self {
            data: [0u8; MAX_PIGGYBACK_PAYLOAD],
            len: 0,
            flow_id: 0,
            active: 0,
        }
    }
}

/// Event structure received from eBPF ringbuf
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct XdpEvent {
    pub flow_key: FlowKey,
    pub payload_len: u32,
    pub seq_num: u64,
}

/// Lock-free ring buffer for zero-copy communication with eBPF
pub struct XdpRingBuffer {
    /// Base pointer to mmap'd memory region
    base_ptr: *mut u8,
    /// Producer index (written by kernel/eBPF)
    producer: *const AtomicU64,
    /// Consumer index (written by userspace)
    consumer: *mut AtomicU64,
    /// Buffer mask for wrapping (size - 1)
    mask: usize,
    /// Pre-allocated event buffer for batch processing
    event_buffer: UnsafeCell<[MaybeUninit<XdpEvent>; 64]>,
}

// SAFETY: XdpRingBuffer is designed for single-threaded access per queue
// Multi-threaded access requires external synchronization
unsafe impl Send for XdpRingBuffer {}
unsafe impl Sync for XdpRingBuffer {}

impl XdpRingBuffer {
    /// Create a new ring buffer from mmap'd memory region
    /// 
    /// # Safety
    /// - `base_ptr` must point to a valid mmap'd region of at least `size` bytes
    /// - `producer` and `consumer` must point to valid atomic counters
    /// - Memory must be properly aligned and accessible
    pub unsafe fn new(
        base_ptr: *mut u8,
        producer: *const AtomicU64,
        consumer: *mut AtomicU64,
        size: usize,
    ) -> Result<Self, &'static str> {
        if base_ptr.is_null() || producer.is_null() || consumer.is_null() {
            return Err("Null pointer provided to XdpRingBuffer");
        }
        
        if !size.is_power_of_two() {
            return Err("Ring buffer size must be power of 2");
        }
        
        Ok(Self {
            base_ptr,
            producer,
            consumer,
            mask: size.wrapping_sub(1),
            event_buffer: UnsafeCell::new(MaybeUninit::uninit()),
        })
    }
    
    /// Poll for new events from eBPF
    /// Returns number of events processed
    /// 
    /// ZERO ALLOCATION: Uses pre-allocated event_buffer
    pub fn poll_events<F>(&self, mut handler: F) -> usize 
    where
        F: FnMut(&XdpEvent),
    {
        let prod = unsafe { (*self.producer).load(Ordering::Acquire) };
        let cons = unsafe { (*self.consumer).load(Ordering::Relaxed) };
        
        if prod == cons {
            return 0; // No new events
        }
        
        let mut count = 0usize;
        let mut current = cons;
        
        while current != prod && count < 64 {
            // Calculate offset in ring buffer
            let offset = (current as usize) & self.mask;
            
            // SAFETY: offset is bounded by mask, which is size-1
            // Memory region is guaranteed valid by constructor
            let event_ptr = unsafe {
                self.base_ptr.add(offset) as *const XdpEvent
            };
            
            // Read event (copy to avoid lifetime issues)
            let event = unsafe { ptr::read(event_ptr) };
            
            // Call handler
            handler(&event);
            
            current = current.wrapping_add(1);
            count += 1;
        }
        
        // Update consumer index
        if count > 0 {
            unsafe {
                (*self.consumer).store(current, Ordering::Release);
            }
        }
        
        count
    }
    
    /// Get available space in ring buffer
    pub fn available_space(&self) -> usize {
        let prod = unsafe { (*self.producer).load(Ordering::Acquire) };
        let cons = unsafe { (*self.consumer).load(Ordering::Relaxed) };
        
        let used = prod.wrapping_sub(cons) as usize;
        RING_BUFFER_SIZE.wrapping_sub(used) & self.mask
    }
}

/// TCP Piggyback Manager - coordinates payload injection
pub struct TcpPiggybackManager {
    /// Pending payloads indexed by slot ID (matches eBPF ARRAY map)
    pending_payloads: UnsafeCell<[ExecutionPayload; 1024]>,
    /// Active slot bitmap
    active_slots: AtomicU64,
    /// Flow-to-slot mapping (simplified hash table)
    flow_slots: UnsafeCell<[Option<FlowKey>; 1024]>,
    /// Statistics
    total_injections: AtomicU64,
    failed_injections: AtomicU64,
}

// SAFETY: Single-threaded access pattern, external sync required for multi-thread
unsafe impl Send for TcpPiggybackManager {}
unsafe impl Sync for TcpPiggybackManager {}

impl TcpPiggybackManager {
    pub fn new() -> Self {
        Self {
            pending_payloads: UnsafeCell::new([ExecutionPayload::default(); 1024]),
            active_slots: AtomicU64::new(0),
            flow_slots: UnsafeCell::new([None; 1024]),
            total_injections: AtomicU64::new(0),
            failed_injections: AtomicU64::new(0),
        }
    }
    
    /// Queue an execution payload for piggybacking
    /// Returns slot ID on success, None if no slots available
    pub fn queue_payload(&self, flow_id: FlowKey, data: &[u8]) -> Option<u32> {
        if data.is_empty() || data.len() > MAX_PIGGYBACK_PAYLOAD {
            self.failed_injections.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        
        // Find free slot using atomic bitmap
        let mut bitmap = self.active_slots.load(Ordering::Acquire);
        
        for slot_id in 0..64u32 {
            if bitmap & (1u64 << slot_id) == 0 {
                // Try to claim this slot
                let old_bitmap = self.active_slots.compare_exchange_weak(
                    bitmap,
                    bitmap | (1u64 << slot_id),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                
                match old_bitmap {
                    Ok(_) => {
                        // Successfully claimed slot, write payload
                        let payloads = unsafe { &mut *self.pending_payloads.get() };
                        payloads[slot_id as usize].data[..data.len()].copy_from_slice(data);
                        payloads[slot_id as usize].len = data.len() as u32;
                        payloads[slot_id as usize].flow_id = flow_id;
                        payloads[slot_id as usize].active = 1;
                        
                        // Update flow mapping
                        let flow_map = unsafe { &mut *self.flow_slots.get() };
                        flow_map[slot_id as usize] = Some(flow_id);
                        
                        self.total_injections.fetch_add(1, Ordering::Relaxed);
                        return Some(slot_id);
                    }
                    Err(new_bitmap) => {
                        bitmap = new_bitmap;
                        continue;
                    }
                }
            }
        }
        
        // No slots available
        self.failed_injections.fetch_add(1, Ordering::Relaxed);
        None
    }
    
    /// Mark a slot as completed (called after eBPF confirms injection)
    pub fn complete_slot(&self, slot_id: u32) {
        if slot_id >= 64 {
            return;
        }
        
        let payloads = unsafe { &mut *self.pending_payloads.get() };
        payloads[slot_id as usize].active = 0;
        payloads[slot_id as usize].len = 0;
        
        // Clear bit in bitmap
        self.active_slots.fetch_and(!(1u64 << slot_id), Ordering::Release);
        
        // Clear flow mapping
        let flow_map = unsafe { &mut *self.flow_slots.get() };
        flow_map[slot_id as usize] = None;
    }
    
    /// Get statistics
    pub fn stats(&self) -> (u64, u64) {
        (
            self.total_injections.load(Ordering::Relaxed),
            self.failed_injections.load(Ordering::Relaxed),
        )
    }
}

/// AF_XDP Socket wrapper for direct NIC communication
pub struct AfXdpSocket {
    socket_fd: i32,
    umem_region: *mut u8,
    umem_size: usize,
    /// TX ring for sending packets
    tx_ring: Option<XdpRingBuffer>,
    /// RX ring for receiving packets  
    rx_ring: Option<XdpRingBuffer>,
    is_active: AtomicBool,
}

// SAFETY: Socket is designed for single-owner access
unsafe impl Send for AfXdpSocket {}
unsafe impl Sync for AfXdpSocket {}

impl AfXdpSocket {
    /// Create a new AF_XDP socket bound to a specific interface queue
    /// 
    /// # Arguments
    /// * `if_index` - Linux interface index (e.g., from if_nametoindex)
    /// * `queue_id` - NIC queue ID to bind to
    /// 
    /// # Safety
    /// Requires CAP_NET_RAW capability and XDP-enabled kernel
    pub fn new(if_index: u32, queue_id: u32) -> Result<Self, &'static str> {
        // In real implementation, this would:
        // 1. Create socket with socket(AF_XDP, SOCK_RAW, 0)
        // 2. Bind to interface with sockaddr_xdp
        // 3. Set up UMEM region with mmap()
        // 4. Configure fill/completion rings
        // 5. Enable zero-copy mode
        
        // Placeholder returns error - real impl needs syscalls
        Err("AF_XDP socket creation requires actual syscall implementation")
    }
    
    /// Send a packet directly through the SmartNIC
    /// Returns bytes sent on success
    pub fn send_packet(&self, data: &[u8]) -> Result<usize, &'static str> {
        if !self.is_active.load(Ordering::Acquire) {
            return Err("Socket not active");
        }
        
        if data.is_empty() {
            return Err("Empty packet");
        }
        
        // In real implementation:
        // 1. Get free TX descriptor from ring
        // 2. Copy data to UMEM region (or reference existing)
        // 3. Kick TX doorbell
        // 4. Wait for completion
        
        Ok(data.len())
    }
    
    /// Activate the socket after successful setup
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Release);
    }
    
    /// Check if socket is ready
    pub fn is_ready(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }
}

impl Drop for AfXdpSocket {
    fn drop(&mut self) {
        self.is_active.store(false, Ordering::Release);
        
        // In real implementation:
        // 1. Close socket FD
        // 2. munmap UMEM region
        // 3. Release ring buffers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_execution_payload_default() {
        let payload = ExecutionPayload::default();
        assert_eq!(payload.len, 0);
        assert_eq!(payload.active, 0);
        assert!(payload.data.iter().all(|&x| x == 0));
    }
    
    #[test]
    fn test_piggyback_manager_queue() {
        let manager = TcpPiggybackManager::new();
        let test_data = b"EXECUTE:BUY:AAPL:100";
        
        let slot = manager.queue_payload(0x12345678, test_data);
        assert!(slot.is_some());
        
        let stats = manager.stats();
        assert_eq!(stats.0, 1);
        assert_eq!(stats.1, 0);
    }
    
    #[test]
    fn test_piggyback_manager_overflow() {
        let manager = TcpPiggybackManager::new();
        
        // Try to queue oversized payload
        let large_data = vec![0u8; MAX_PIGGYBACK_PAYLOAD + 1];
        let slot = manager.queue_payload(0x12345678, &large_data);
        
        assert!(slot.is_none());
        
        let stats = manager.stats();
        assert_eq!(stats.1, 1); // Should have failed
    }
}
