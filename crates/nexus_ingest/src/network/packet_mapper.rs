//! Chapter 1: Packet Mapper for Zero-Copy Memory Mapping
//!
//! This module provides zero-copy memory mapping from network buffers
//! directly into the SPSC ring buffer, minimizing CPU cache misses and
//! memory allocations in the hot path.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use memmap2::{Mmap, MmapMut};
use tracing::{debug, error, info, warn};

use nexus_core::concurrency::spsc_ring::{Consumer, Producer, SpscRingBuffer};
use nexus_core::memory::cache_padder::CachePadded64;
use nexus_core::time::tsc_clock::MonotonicNanosClock;

use crate::network::kernel_bypass::{
    KernelBypassStats, PacketError, PacketMetadata, RawPacket, MAX_MTU_SIZE,
};

/// Memory-mapped packet buffer for zero-copy operations
#[repr(C)]
#[derive(Debug)]
pub struct MappedPacketBuffer {
    /// Memory map (if using mmap)
    mmap: Option<Mmap>,
    /// Mutable memory map (if using mmap)
    mmap_mut: Option<MmapMut>,
    /// Buffer pointer (raw access for zero-copy)
    ptr: *mut u8,
    /// Buffer length
    len: usize,
    /// Capacity
    capacity: usize,
    /// Current write position
    write_pos: CachePadded64<AtomicUsize>,
    /// Current read position
    read_pos: CachePadded64<AtomicUsize>,
    /// Statistics
    stats: CachePadded64<MappedBufferStats>,
}

// SAFETY: MappedPacketBuffer is used in single-producer context
unsafe impl Send for MappedPacketBuffer {}
unsafe impl Sync for MappedPacketBuffer {}

/// Statistics for mapped buffer operations
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct MappedBufferStats {
    /// Total bytes written
    pub bytes_written: CachePadded64<AtomicU64>,
    /// Total bytes read
    pub bytes_read: CachePadded64<AtomicU64>,
    /// Wrap-around count
    pub wrap_count: CachePadded64<AtomicU64>,
    /// Overflow events
    pub overflow_count: CachePadded64<AtomicU64>,
    /// Underflow events
    pub underflow_count: CachePadded64<AtomicU64>,
}

impl MappedBufferStats {
    #[inline]
    pub fn new() -> Self {
        Self {
            bytes_written: CachePadded64::new(AtomicU64::new(0)),
            bytes_read: CachePadded64::new(AtomicU64::new(0)),
            wrap_count: CachePadded64::new(AtomicU64::new(0)),
            overflow_count: CachePadded64::new(AtomicU64::new(0)),
            underflow_count: CachePadded64::new(AtomicU64::new(0)),
        }
    }
}

impl MappedPacketBuffer {
    /// Create a new anonymous mapped buffer
    pub fn new_anonymous(capacity: usize) -> Result<Self, PacketError> {
        // Allocate aligned memory
        let layout = std::alloc::Layout::from_size_align(capacity, 64)
            .map_err(|_| PacketError::InvalidFormat("Invalid layout".to_string()))?;
        
        unsafe {
            let ptr = std::alloc::alloc(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            
            // Initialize to zero
            std::ptr::write_bytes(ptr, 0, capacity);
            
            Ok(Self {
                mmap: None,
                mmap_mut: None,
                ptr,
                len: 0,
                capacity,
                write_pos: CachePadded64::new(AtomicUsize::new(0)),
                read_pos: CachePadded64::new(AtomicUsize::new(0)),
                stats: CachePadded64::new(MappedBufferStats::new()),
            })
        }
    }

    /// Create a memory-mapped buffer from a file
    pub fn new_mapped_file(path: &str, capacity: usize) -> Result<Self, PacketError> {
        use std::fs::File;
        use std::io::{Read, Write};
        
        // Create or open file
        let mut file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(|e| PacketError::NetworkInterface(e.to_string()))?;
        
        // Set file size if needed
        let metadata = file.metadata()
            .map_err(|e| PacketError::NetworkInterface(e.to_string()))?;
        
        if metadata.len() < capacity as u64 {
            file.set_len(capacity as u64)
                .map_err(|e| PacketError::NetworkInterface(e.to_string()))?;
        }
        
        // Memory map the file
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| PacketError::NetworkInterface(e.to_string()))?
        };
        
        let ptr = mmap.as_ptr() as *mut u8;
        
        Ok(Self {
            mmap: Some(mmap),
            mmap_mut: None,
            ptr,
            len: 0,
            capacity,
            write_pos: CachePadded64::new(AtomicUsize::new(0)),
            read_pos: CachePadded64::new(AtomicUsize::new(0)),
            stats: CachePadded64::new(MappedBufferStats::new()),
        })
    }

    /// Get current write position
    #[inline]
    pub fn write_position(&self) -> usize {
        self.write_pos.0.load(Ordering::Acquire)
    }

    /// Get current read position
    #[inline]
    pub fn read_position(&self) -> usize {
        self.read_pos.0.load(Ordering::Acquire)
    }

    /// Get available write space
    #[inline]
    pub fn available_write_space(&self) -> usize {
        let write = self.write_position();
        let read = self.read_position();
        
        if write >= read {
            self.capacity - (write - read) - 1
        } else {
            read - write - 1
        }
    }

    /// Get available read data
    #[inline]
    pub fn available_read_data(&self) -> usize {
        let write = self.write_position();
        let read = self.read_position();
        
        if write >= read {
            write - read
        } else {
            self.capacity - read + write
        }
    }

    /// Write data to buffer (zero-copy where possible)
    #[inline]
    pub fn write(&mut self, data: &[u8]) -> Result<usize, PacketError> {
        if data.is_empty() {
            return Ok(0);
        }
        
        let write = self.write_position();
        let available = self.available_write_space();
        
        if data.len() > available {
            self.stats.0.overflow_count.fetch_add(1, Ordering::Relaxed);
            return Err(PacketError::RingBufferFull);
        }
        
        let mut bytes_written = 0;
        let mut src_pos = 0;
        let mut dst_pos = write;
        
        while src_pos < data.len() {
            let chunk_end = std::cmp::min(dst_pos + (data.len() - src_pos), self.capacity);
            let chunk_size = chunk_end - dst_pos;
            
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().add(src_pos),
                    self.ptr.add(dst_pos),
                    chunk_size,
                );
            }
            
            src_pos += chunk_size;
            dst_pos = if chunk_end >= self.capacity {
                self.stats.0.wrap_count.fetch_add(1, Ordering::Relaxed);
                0
            } else {
                chunk_end
            };
            
            bytes_written += chunk_size;
        }
        
        self.write_pos.0.store(dst_pos, Ordering::Release);
        self.stats.0.bytes_written.fetch_add(bytes_written as u64, Ordering::Relaxed);
        
        Ok(bytes_written)
    }

    /// Read data from buffer (zero-copy)
    #[inline]
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize, PacketError> {
        if buf.is_empty() {
            return Ok(0);
        }
        
        let read = self.read_position();
        let available = self.available_read_data();
        
        if available == 0 {
            self.stats.0.underflow_count.fetch_add(1, Ordering::Relaxed);
            return Ok(0);
        }
        
        let to_read = std::cmp::min(buf.len(), available);
        let mut bytes_read = 0;
        let mut dst_pos = 0;
        let mut src_pos = read;
        
        while bytes_read < to_read {
            let chunk_end = std::cmp::min(src_pos + (to_read - bytes_read), self.capacity);
            let chunk_size = chunk_end - src_pos;
            
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.ptr.add(src_pos),
                    buf.as_mut_ptr().add(dst_pos),
                    chunk_size,
                );
            }
            
            dst_pos += chunk_size;
            src_pos = if chunk_end >= self.capacity {
                self.stats.0.wrap_count.fetch_add(1, Ordering::Relaxed);
                0
            } else {
                chunk_end
            };
            
            bytes_read += chunk_size;
        }
        
        self.read_pos.0.store(src_pos, Ordering::Release);
        self.stats.0.bytes_read.fetch_add(bytes_read as u64, Ordering::Relaxed);
        
        Ok(bytes_read)
    }

    /// Get raw pointer for zero-copy access (caller must ensure safety)
    #[inline]
    pub fn as_raw_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Get mutable raw pointer for zero-copy access
    #[inline]
    pub fn as_raw_ptr_mut(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Get buffer capacity
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> MappedBufferStats {
        self.stats.0.clone()
    }
}

impl Drop for MappedPacketBuffer {
    fn drop(&mut self) {
        if self.mmap.is_none() && !self.ptr.is_null() {
            unsafe {
                let layout = std::alloc::Layout::from_size_align(self.capacity, 64).unwrap();
                std::alloc::dealloc(self.ptr, layout);
            }
        }
    }
}

/// Direct memory mapper for network packets to ring buffer
pub struct DirectPacketMapper<'a> {
    /// Ring buffer producer
    producer: Producer<'a, RawPacket>,
    /// Clock for timestamping
    clock: MonotonicNanosClock,
    /// Statistics reference
    stats: &'a KernelBypassStats,
    /// Packets mapped counter
    packets_mapped: CachePadded64<AtomicUsize>,
}

// SAFETY: DirectPacketMapper is single-threaded producer
unsafe impl<'a> Send for DirectPacketMapper<'a> {}

impl<'a> DirectPacketMapper<'a> {
    /// Create a new direct packet mapper
    pub fn new(
        producer: Producer<'a, RawPacket>,
        clock: MonotonicNanosClock,
        stats: &'a KernelBypassStats,
    ) -> Self {
        Self {
            producer,
            clock,
            stats,
            packets_mapped: CachePadded64::new(AtomicUsize::new(0)),
        }
    }

    /// Map network packet directly to ring buffer slot
    #[inline]
    pub fn map_direct(&mut self, packet_data: &[u8]) -> Result<(), PacketError> {
        // Get timestamp first (minimize latency between arrival and timestamp)
        let timestamp_ns = self.clock.now_ns();
        
        // Try to get a slot in the ring buffer
        match self.producer.push_slot() {
            Some(mut slot) => {
                let packet = slot.as_mut();
                
                // Set timestamp
                packet.metadata.0.get_mut().timestamp_ns = timestamp_ns;
                
                // Set packet length
                packet.metadata.0.get_mut().packet_len = packet_data.len() as u16;
                
                // Copy packet data (minimal necessary copy)
                if let Err(e) = packet.set_data(packet_data) {
                    self.stats.increment_invalid();
                    return Err(e);
                }
                
                // Commit the slot (makes it visible to consumer)
                slot.commit();
                
                // Update statistics
                self.stats.increment_received(packet_data.len() as u64);
                self.stats.increment_forwarded();
                self.packets_mapped.0.fetch_add(1, Ordering::Relaxed);
                
                Ok(())
            }
            None => {
                self.stats.increment_overflow();
                Err(PacketError::RingBufferFull)
            }
        }
    }

    /// Map packet with extracted metadata
    #[inline]
    pub fn map_with_metadata(
        &mut self,
        packet_data: &[u8],
        metadata: PacketMetadata,
    ) -> Result<(), PacketError> {
        match self.producer.push_slot() {
            Some(mut slot) => {
                let packet = slot.as_mut();
                
                // Use provided metadata but update timestamp
                let mut meta = metadata;
                meta.timestamp_ns = self.clock.now_ns();
                meta.packet_len = packet_data.len() as u16;
                
                *packet.metadata.0.get_mut() = meta;
                
                if let Err(e) = packet.set_data(packet_data) {
                    self.stats.increment_invalid();
                    return Err(e);
                }
                
                slot.commit();
                
                self.stats.increment_received(packet_data.len() as u64);
                self.stats.increment_forwarded();
                self.packets_mapped.0.fetch_add(1, Ordering::Relaxed);
                
                Ok(())
            }
            None => {
                self.stats.increment_overflow();
                Err(PacketError::RingBufferFull)
            }
        }
    }

    /// Batch map multiple packets
    #[inline]
    pub fn map_batch(&mut self, packets: &[&[u8]]) -> Result<usize, PacketError> {
        let mut mapped = 0;
        
        for &packet_data in packets {
            match self.map_direct(packet_data) {
                Ok(()) => mapped += 1,
                Err(PacketError::RingBufferFull) => break,
                Err(e) => return Err(e),
            }
        }
        
        Ok(mapped)
    }

    /// Get total packets mapped
    #[inline]
    pub fn packets_mapped_count(&self) -> usize {
        self.packets_mapped.0.load(Ordering::Relaxed)
    }
}

/// Zero-copy packet slice for efficient processing
pub struct PacketSlice<'a> {
    /// Pointer to data
    ptr: *const u8,
    /// Length of data
    len: usize,
    /// Lifetime marker
    _marker: PhantomData<&'a [u8]>,
}

// SAFETY: PacketSlice is read-only and short-lived
unsafe impl<'a> Send for PacketSlice<'a> {}

impl<'a> PacketSlice<'a> {
    /// Create a new packet slice from raw pointer
    #[inline]
    pub unsafe fn from_raw(ptr: *const u8, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// Get slice as &[u8]
    #[inline]
    pub fn as_slice(&self) -> &'a [u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get length
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_core::concurrency::spsc_ring::SpscRingBuffer;
    use nexus_core::time::tsc_clock::MonotonicNanosClock;

    #[test]
    fn test_mapped_buffer_anonymous() {
        let mut buffer = MappedPacketBuffer::new_anonymous(4096).unwrap();
        assert_eq!(buffer.capacity(), 4096);
        
        let data = b"Hello, World!";
        let written = buffer.write(data).unwrap();
        assert_eq!(written, data.len());
        
        let mut read_buf = vec![0u8; data.len()];
        let read = buffer.read(&mut read_buf).unwrap();
        assert_eq!(read, data.len());
        assert_eq!(&read_buf, data);
    }

    #[test]
    fn test_mapped_buffer_wrap() {
        let mut buffer = MappedPacketBuffer::new_anonymous(64).unwrap();
        
        // Write almost full
        let data1 = vec![1u8; 60];
        buffer.write(&data1).unwrap();
        
        // Read half
        let mut read_buf = vec![0u8; 30];
        buffer.read(&mut read_buf).unwrap();
        
        // Write more (should wrap)
        let data2 = vec![2u8; 30];
        buffer.write(&data2).unwrap();
        
        let stats = buffer.get_stats();
        assert!(*stats.wrap_count.0.get() > 0);
    }

    #[test]
    fn test_direct_packet_mapper() {
        let ring = Box::leak(Box::new(SpscRingBuffer::<RawPacket>::new(1024)));
        let (producer, _) = ring.split();
        let clock = MonotonicNanosClock::new();
        let stats = KernelBypassStats::new();
        
        let mut mapper = DirectPacketMapper::new(producer, clock, &stats);
        
        let packet = vec![0x45u8, 0x00, 0x00, 0x28, 0x00, 0x00];
        assert!(mapper.map_direct(&packet).is_ok());
        
        assert_eq!(mapper.packets_mapped_count(), 1);
        assert_eq!(*stats.packets_forwarded.0.get(), 1);
    }

    #[test]
    fn test_packet_slice() {
        let data = [1u8, 2, 3, 4, 5];
        let slice = unsafe { PacketSlice::from_raw(data.as_ptr(), data.len()) };
        
        assert_eq!(slice.len(), 5);
        assert_eq!(slice.as_slice(), &data);
        assert!(!slice.is_empty());
    }

    #[test]
    fn test_buffer_overflow() {
        let mut buffer = MappedPacketBuffer::new_anonymous(32).unwrap();
        
        // Fill buffer
        let data = vec![0u8; 31];
        buffer.write(&data).unwrap();
        
        // Try to write more
        let extra = vec![1u8; 10];
        assert!(matches!(buffer.write(&extra), Err(PacketError::RingBufferFull)));
        
        let stats = buffer.get_stats();
        assert_eq!(*stats.overflow_count.0.get(), 1);
    }
}
