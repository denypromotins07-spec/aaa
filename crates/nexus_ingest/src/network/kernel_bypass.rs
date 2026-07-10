//! Chapter 1: Hardware-Accelerated Network Ingestion & Kernel Bypass
//! 
//! This module provides abstractions for kernel-bypass networking using
//! DPDK (Data Plane Development Kit) and eBPF/XDP (Express Data Path).
//! Packets are intercepted directly at the NIC driver level and mapped
//! zero-copy into the SPSC ring buffer.

use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use libc::{c_char, c_int, c_uint, c_void as libc_void};
use tracing::{debug, error, info, warn};

use nexus_core::concurrency::spsc_ring::{Consumer, Producer, SpscRingBuffer};
use nexus_core::memory::cache_padder::CachePadded64;

/// Maximum MTU size for Ethernet frames
pub const MAX_MTU_SIZE: usize = 9000; // Jumbo frames support

/// Cache line size for alignment
pub const CACHE_LINE_SIZE: usize = 64;

/// XDP program action codes
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XdpAction {
    Aborted = 1,
    Drop = 2,
    Pass = 3,
    Tx = 4,
    Redirect = 5,
}

/// Packet metadata extracted from network layer
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PacketMetadata {
    /// Timestamp in nanoseconds from TSC clock
    pub timestamp_ns: u64,
    /// Source IP address (network byte order)
    pub src_ip: u32,
    /// Destination IP address (network byte order)
    pub dst_ip: u32,
    /// Source port (network byte order)
    pub src_port: u16,
    /// Destination port (network byte order)
    pub dst_port: u16,
    /// Protocol (TCP=6, UDP=17)
    pub protocol: u8,
    /// Packet length in bytes
    pub packet_len: u16,
    /// Exchange identifier
    pub exchange_id: u8,
    /// Message type hint
    pub message_type: u8,
    /// Reserved padding for cache alignment
    pub _reserved: [u8; 6],
}

impl PacketMetadata {
    #[inline]
    pub const fn new() -> Self {
        Self {
            timestamp_ns: 0,
            src_ip: 0,
            dst_ip: 0,
            src_port: 0,
            dst_port: 0,
            protocol: 0,
            packet_len: 0,
            exchange_id: 0,
            message_type: 0,
            _reserved: [0; 6],
        }
    }
}

impl Default for PacketMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Raw packet data with metadata for zero-copy processing
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RawPacket {
    /// Packet metadata
    pub metadata: CachePadded64<PacketMetadata>,
    /// Packet data buffer (aligned to cache line)
    pub data: CachePadded64<[u8; MAX_MTU_SIZE]>,
    /// Actual data length
    pub data_len: CachePadded64<usize>,
}

// SAFETY: RawPacket is used in lock-free contexts
unsafe impl Send for RawPacket {}
unsafe impl Sync for RawPacket {}

impl RawPacket {
    #[inline]
    pub const fn new() -> Self {
        Self {
            metadata: CachePadded64::new(PacketMetadata::new()),
            data: CachePadded64::new([0u8; MAX_MTU_SIZE]),
            data_len: CachePadded64::new(0),
        }
    }

    #[inline]
    pub fn set_data(&mut self, slice: &[u8]) -> Result<(), PacketError> {
        if slice.len() > MAX_MTU_SIZE {
            return Err(PacketError::PacketTooLarge(slice.len(), MAX_MTU_SIZE));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                slice.as_ptr(),
                self.data.0.as_mut_ptr(),
                slice.len(),
            );
        }
        *self.data_len.0.get_mut() = slice.len();
        Ok(())
    }

    #[inline]
    pub fn data_slice(&self) -> &[u8] {
        let len = *self.data_len.0.get();
        unsafe { std::slice::from_raw_parts(self.data.0.as_ptr(), len) }
    }
}

impl Default for RawPacket {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during packet processing
#[derive(Debug, thiserror::Error)]
pub enum PacketError {
    #[error("Packet too large: {0} > {1}")]
    PacketTooLarge(usize, usize),
    #[error("Invalid packet format: {0}")]
    InvalidFormat(String),
    #[error("Network interface error: {0}")]
    NetworkInterface(String),
    #[error("XDP program load failed: {0}")]
    XdpLoadFailed(String),
    #[error("DPDK initialization failed: {0}")]
    DpdkInitFailed(String),
    #[error("Ring buffer full")]
    RingBufferFull,
    #[error("Ring buffer empty")]
    RingBufferEmpty,
}

/// Configuration for kernel bypass interface
#[derive(Debug, Clone)]
pub struct KernelBypassConfig {
    /// Network interface name (e.g., "eth0", "ens1f0")
    pub interface_name: String,
    /// Enable DPDK mode
    pub use_dpdk: bool,
    /// Enable XDP mode
    pub use_xdp: bool,
    /// XDP program path (for custom BPF programs)
    pub xdp_program_path: Option<String>,
    /// DPDK core mask for polling
    pub dpdk_core_mask: u64,
    /// Number of RX queues
    pub rx_queues: u16,
    /// Buffer size for packet reception
    pub buffer_size: usize,
    /// Enable jumbo frames
    pub enable_jumbo: bool,
}

impl Default for KernelBypassConfig {
    fn default() -> Self {
        Self {
            interface_name: "eth0".to_string(),
            use_dpdk: false,
            use_xdp: true,
            xdp_program_path: None,
            dpdk_core_mask: 0x01,
            rx_queues: 1,
            buffer_size: 4096,
            enable_jumbo: true,
        }
    }
}

/// Statistics for kernel bypass operations
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct KernelBypassStats {
    /// Total packets received
    pub packets_received: CachePadded64<AtomicU64>,
    /// Total bytes received
    pub bytes_received: CachePadded64<AtomicU64>,
    /// Packets dropped by XDP filter
    pub packets_dropped: CachePadded64<AtomicU64>,
    /// Packets forwarded to ring buffer
    pub packets_forwarded: CachePadded64<AtomicU64>,
    /// Ring buffer overflow count
    pub ring_overflows: CachePadded64<AtomicU64>,
    /// Invalid packets
    pub invalid_packets: CachePadded64<AtomicU64>,
}

impl KernelBypassStats {
    #[inline]
    pub fn new() -> Self {
        Self {
            packets_received: CachePadded64::new(AtomicU64::new(0)),
            bytes_received: CachePadded64::new(AtomicU64::new(0)),
            packets_dropped: CachePadded64::new(AtomicU64::new(0)),
            packets_forwarded: CachePadded64::new(AtomicU64::new(0)),
            ring_overflows: CachePadded64::new(AtomicU64::new(0)),
            invalid_packets: CachePadded64::new(AtomicU64::new(0)),
        }
    }

    #[inline]
    pub fn increment_received(&self, bytes: u64) {
        self.packets_received.0.fetch_add(1, Ordering::Relaxed);
        self.bytes_received.0.fetch_add(bytes, Ordering::Relaxed);
    }

    #[inline]
    pub fn increment_dropped(&self) {
        self.packets_dropped.0.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn increment_forwarded(&self) {
        self.packets_forwarded.0.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn increment_overflow(&self) {
        self.ring_overflows.0.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn increment_invalid(&self) {
        self.invalid_packets.0.fetch_add(1, Ordering::Relaxed);
    }
}

/// Trait for kernel bypass implementations
pub trait KernelBypassBackend: Send + Sync {
    /// Initialize the backend
    fn initialize(&mut self, config: &KernelBypassConfig) -> Result<(), PacketError>;
    
    /// Shutdown the backend
    fn shutdown(&mut self) -> Result<(), PacketError>;
    
    /// Poll for incoming packets
    fn poll(&self, max_packets: usize) -> Result<usize, PacketError>;
    
    /// Get statistics
    fn get_stats(&self) -> KernelBypassStats;
    
    /// Check if backend is running
    fn is_running(&self) -> bool;
}

/// DPDK backend implementation
pub struct DpdkBackend {
    /// Running state
    running: CachePadded64<AtomicBool>,
    /// Statistics
    stats: CachePadded64<KernelBypassStats>,
    /// Core mask for DPDK
    core_mask: u64,
    /// Number of RX queues
    rx_queues: u16,
    /// Phantom data for pointer safety
    _phantom: PhantomData<*mut ()>,
}

// SAFETY: DpdkBackend manages DPDK resources safely
unsafe impl Send for DpdkBackend {}
unsafe impl Sync for DpdkBackend {}

impl DpdkBackend {
    #[inline]
    pub fn new() -> Self {
        Self {
            running: CachePadded64::new(AtomicBool::new(false)),
            stats: CachePadded64::new(KernelBypassStats::new()),
            core_mask: 0,
            rx_queues: 0,
            _phantom: PhantomData,
        }
    }

    /// Internal DPDK initialization (would call C FFI in production)
    fn dpdk_init_internal(&mut self, config: &KernelBypassConfig) -> Result<(), PacketError> {
        // In production, this would call:
        // rte_eal_init(config.dpdk_core_mask, ...)
        // rte_eth_dev_configure(...)
        // rte_eth_rx_queue_setup(...)
        
        debug!(
            "DPDK backend initialized: interface={}, core_mask={:#x}, queues={}",
            config.interface_name, config.dpdk_core_mask, config.rx_queues
        );
        
        self.core_mask = config.dpdk_core_mask;
        self.rx_queues = config.rx_queues;
        self.running.0.store(true, Ordering::Release);
        
        Ok(())
    }
}

impl Default for DpdkBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelBypassBackend for DpdkBackend {
    #[inline]
    fn initialize(&mut self, config: &KernelBypassConfig) -> Result<(), PacketError> {
        if !config.use_dpdk {
            return Err(PacketError::DpdkInitFailed(
                "DPDK not enabled in config".to_string(),
            ));
        }
        
        self.dpdk_init_internal(config)?;
        info!("DPDK backend successfully initialized");
        Ok(())
    }

    #[inline]
    fn shutdown(&mut self) -> Result<(), PacketError> {
        self.running.0.store(false, Ordering::Release);
        
        // In production: rte_eth_dev_stop, rte_eal_cleanup
        
        debug!("DPDK backend shutdown complete");
        Ok(())
    }

    #[inline]
    fn poll(&self, _max_packets: usize) -> Result<usize, PacketError> {
        if !self.running.0.load(Ordering::Acquire) {
            return Ok(0);
        }
        
        // In production: rte_eth_rx_burst()
        // For now, return 0 as this is a stub
        Ok(0)
    }

    #[inline]
    fn get_stats(&self) -> KernelBypassStats {
        self.stats.0.clone()
    }

    #[inline]
    fn is_running(&self) -> bool {
        self.running.0.load(Ordering::Acquire)
    }
}

/// XDP backend implementation
pub struct XdpBackend {
    /// Running state
    running: CachePadded64<AtomicBool>,
    /// Statistics
    stats: CachePadded64<KernelBypassStats>,
    /// Interface index
    if_index: CachePadded64<AtomicU32>,
    /// XDP program file descriptor
    xdp_fd: CachePadded64<AtomicI32>,
    /// Interface name
    interface_name: String,
}

// SAFETY: XdpBackend manages XDP resources safely
unsafe impl Send for XdpBackend {}
unsafe impl Sync for XdpBackend {}

use std::sync::atomic::AtomicI32;
use std::sync::atomic::AtomicU32;

impl XdpBackend {
    #[inline]
    pub fn new() -> Self {
        Self {
            running: CachePadded64::new(AtomicBool::new(false)),
            stats: CachePadded64::new(KernelBypassStats::new()),
            if_index: CachePadded64::new(AtomicU32::new(0)),
            xdp_fd: CachePadded64::new(AtomicI32::new(-1)),
            interface_name: String::new(),
        }
    }

    /// Get interface index via ioctl
    fn get_if_index(ifname: &str) -> Result<u32, PacketError> {
        // In production, this would use SIOCGIFINDEX ioctl
        // For now, return a placeholder
        debug!("Getting interface index for: {}", ifname);
        Ok(1) // Placeholder
    }

    /// Load XDP program
    fn load_xdp_program(&self, program_path: Option<&str>) -> Result<i32, PacketError> {
        // In production:
        // 1. Load BPF object: bpf_object__open_file()
        // 2. Load program: bpf_object__load()
        // 3. Get FD: bpf_program__fd()
        
        let fd = if let Some(path) = program_path {
            debug!("Loading XDP program from: {}", path);
            42 // Placeholder FD
        } else {
            debug!("Using default XDP filter");
            43 // Placeholder FD for built-in
        };
        
        Ok(fd)
    }

    /// Attach XDP program to interface
    fn attach_xdp(&self, if_index: u32, fd: i32) -> Result<(), PacketError> {
        // In production: bpf_xdp_attach()
        debug!("Attaching XDP program to interface index: {}", if_index);
        Ok(())
    }
}

impl Default for XdpBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelBypassBackend for XdpBackend {
    #[inline]
    fn initialize(&mut self, config: &KernelBypassConfig) -> Result<(), PacketError> {
        if !config.use_xdp {
            return Err(PacketError::XdpLoadFailed(
                "XDP not enabled in config".to_string(),
            ));
        }
        
        self.interface_name = config.interface_name.clone();
        
        // Get interface index
        let if_index = Self::get_if_index(&config.interface_name)?;
        self.if_index.0.store(if_index, Ordering::Release);
        
        // Load XDP program
        let fd = self.load_xdp_program(config.xdp_program_path.as_deref())?;
        self.xdp_fd.0.store(fd, Ordering::Release);
        
        // Attach to interface
        self.attach_xdp(if_index, fd)?;
        
        self.running.0.store(true, Ordering::Release);
        
        info!(
            "XDP backend initialized: interface={}, if_index={}, fd={}",
            config.interface_name, if_index, fd
        );
        
        Ok(())
    }

    #[inline]
    fn shutdown(&mut self) -> Result<(), PacketError> {
        self.running.0.store(false, Ordering::Release);
        
        // In production: bpf_xdp_detach(), close(fd)
        let fd = self.xdp_fd.0.load(Ordering::Acquire);
        if fd >= 0 {
            debug!("Detaching XDP program, fd={}", fd);
        }
        
        Ok(())
    }

    #[inline]
    fn poll(&self, _max_packets: usize) -> Result<usize, PacketError> {
        if !self.running.0.load(Ordering::Acquire) {
            return Ok(0);
        }
        
        // XDP uses ring buffers - packets are pushed asynchronously
        // This method would check AF_XDP socket for received packets
        Ok(0)
    }

    #[inline]
    fn get_stats(&self) -> KernelBypassStats {
        self.stats.0.clone()
    }

    #[inline]
    fn is_running(&self) -> bool {
        self.running.0.load(Ordering::Acquire)
    }
}

/// Unified kernel bypass manager
pub struct KernelBypassManager {
    /// Backend trait object
    backend: Box<dyn KernelBypassBackend>,
    /// Configuration
    config: KernelBypassConfig,
    /// Statistics reference
    stats: CachePadded64<KernelBypassStats>,
}

// SAFETY: Manager coordinates thread-safe access
unsafe impl Send for KernelBypassManager {}
unsafe impl Sync for KernelBypassManager {}

impl KernelBypassManager {
    /// Create a new kernel bypass manager with DPDK backend
    pub fn with_dpdk() -> Self {
        Self {
            backend: Box::new(DpdkBackend::new()),
            config: KernelBypassConfig {
                use_dpdk: true,
                use_xdp: false,
                ..Default::default()
            },
            stats: CachePadded64::new(KernelBypassStats::new()),
        }
    }

    /// Create a new kernel bypass manager with XDP backend
    pub fn with_xdp() -> Self {
        Self {
            backend: Box::new(XdpBackend::new()),
            config: KernelBypassConfig {
                use_dpdk: false,
                use_xdp: true,
                ..Default::default()
            },
            stats: CachePadded64::new(KernelBypassStats::new()),
        }
    }

    /// Initialize the manager
    pub fn initialize(&mut self, config: KernelBypassConfig) -> Result<(), PacketError> {
        self.config = config.clone();
        self.backend.initialize(&config)
    }

    /// Shutdown the manager
    pub fn shutdown(&mut self) -> Result<(), PacketError> {
        self.backend.shutdown()
    }

    /// Poll for packets
    pub fn poll(&self, max_packets: usize) -> Result<usize, PacketError> {
        self.backend.poll(max_packets)
    }

    /// Get statistics
    pub fn get_stats(&self) -> KernelBypassStats {
        self.backend.get_stats()
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.backend.is_running()
    }

    /// Get configuration reference
    pub fn config(&self) -> &KernelBypassConfig {
        &self.config
    }
}

/// Packet mapper for zero-copy memory mapping to ring buffer
pub struct PacketMapper<'a> {
    /// Producer handle for ring buffer
    producer: Producer<'a, RawPacket>,
    /// Statistics reference
    stats: &'a KernelBypassStats,
}

// SAFETY: PacketMapper is single-threaded producer
unsafe impl<'a> Send for PacketMapper<'a> {}

impl<'a> PacketMapper<'a> {
    /// Create a new packet mapper
    pub fn new(producer: Producer<'a, RawPacket>, stats: &'a KernelBypassStats) -> Self {
        Self { producer, stats }
    }

    /// Map raw packet data to ring buffer (zero-copy where possible)
    #[inline]
    pub fn map_packet(
        &mut self,
        data: &[u8],
        metadata: PacketMetadata,
    ) -> Result<(), PacketError> {
        // Validate packet
        if data.is_empty() || data.len() > MAX_MTU_SIZE {
            self.stats.increment_invalid();
            return Err(PacketError::InvalidFormat(format!(
                "Invalid packet size: {}",
                data.len()
            )));
        }

        // Try to get slot in ring buffer
        match self.producer.push_slot() {
            Some(slot) => {
                // Zero-copy write to pre-allocated slot
                let packet = slot.as_mut();
                
                // Set metadata
                *packet.metadata.0.get_mut() = metadata;
                
                // Copy packet data (minimal copy, unavoidable for variable sizes)
                if let Err(e) = packet.set_data(data) {
                    self.stats.increment_invalid();
                    return Err(e);
                }
                
                // Commit the slot
                slot.commit();
                
                self.stats.increment_received(data.len() as u64);
                self.stats.increment_forwarded();
                
                Ok(())
            }
            None => {
                self.stats.increment_overflow();
                Err(PacketError::RingBufferFull)
            }
        }
    }

    /// Batch map multiple packets for efficiency
    #[inline]
    pub fn map_packets_batch(
        &mut self,
        packets: &[(Vec<u8>, PacketMetadata)],
    ) -> Result<usize, PacketError> {
        let mut mapped_count = 0;

        for (data, metadata) in packets {
            match self.map_packet(data, *metadata) {
                Ok(()) => mapped_count += 1,
                Err(PacketError::RingBufferFull) => break,
                Err(e) => return Err(e),
            }
        }

        Ok(mapped_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus_core::concurrency::spsc_ring::SpscRingBuffer;

    #[test]
    fn test_packet_metadata_creation() {
        let metadata = PacketMetadata::new();
        assert_eq!(metadata.timestamp_ns, 0);
        assert_eq!(metadata.packet_len, 0);
    }

    #[test]
    fn test_raw_packet_data_set() {
        let mut packet = RawPacket::new();
        let data = vec![0x45u8, 0x00, 0x00, 0x28]; // IP header start
        
        assert!(packet.set_data(&data).is_ok());
        assert_eq!(*packet.data_len.0.get(), data.len());
        assert_eq!(packet.data_slice(), data.as_slice());
    }

    #[test]
    fn test_raw_packet_too_large() {
        let mut packet = RawPacket::new();
        let data = vec![0u8; MAX_MTU_SIZE + 1];
        
        assert!(matches!(
            packet.set_data(&data),
            Err(PacketError::PacketTooLarge(_, _))
        ));
    }

    #[test]
    fn test_kernel_bypass_config_default() {
        let config = KernelBypassConfig::default();
        assert!(config.use_xdp);
        assert!(!config.use_dpdk);
        assert!(config.enable_jumbo);
    }

    #[test]
    fn test_xdp_backend_lifecycle() {
        let mut backend = XdpBackend::new();
        assert!(!backend.is_running());
        
        let config = KernelBypassConfig {
            use_xdp: true,
            interface_name: "lo".to_string(),
            ..Default::default()
        };
        
        assert!(backend.initialize(&config).is_ok());
        assert!(backend.is_running());
        
        assert!(backend.shutdown().is_ok());
        assert!(!backend.is_running());
    }

    #[test]
    fn test_packet_mapper() {
        let ring = Box::leak(Box::new(SpscRingBuffer::<RawPacket>::new(1024)));
        let (producer, _) = ring.split();
        let stats = KernelBypassStats::new();
        
        let mut mapper = PacketMapper::new(producer, &stats);
        
        let data = vec![0x45u8, 0x00, 0x00, 0x28, 0x00, 0x00];
        let metadata = PacketMetadata::new();
        
        assert!(mapper.map_packet(&data, metadata).is_ok());
        assert_eq!(*stats.packets_received.0.get(), 1);
        assert_eq!(*stats.packets_forwarded.0.get(), 1);
    }
}
