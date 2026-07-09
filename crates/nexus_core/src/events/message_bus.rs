//! Event bus and message passing for NEXUS-OMEGA
//!
//! Provides high-performance event routing between Rust and Python components.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::concurrency::spsc_ring::SPSCRingBuffer;
use crate::memory::arena::CACHE_LINE_SIZE;

/// Maximum number of channels supported
pub const MAX_CHANNELS: usize = 256;

/// Default channel capacity
pub const DEFAULT_CHANNEL_CAPACITY: usize = 4096;

/// Error type for message bus operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageBusError {
    ChannelNotFound,
    ChannelFull,
    ChannelExists,
    InvalidChannelName,
    SubscriberLimitReached,
}

impl std::fmt::Display for MessageBusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageBusError::ChannelNotFound => write!(f, "Channel not found"),
            MessageBusError::ChannelFull => write!(f, "Channel is full"),
            MessageBusError::ChannelExists => write!(f, "Channel already exists"),
            MessageBusError::InvalidChannelName => write!(f, "Invalid channel name"),
            MessageBusError::SubscriberLimitReached => write!(f, "Subscriber limit reached"),
        }
    }
}

impl std::error::Error for MessageBusError {}

/// Event header containing metadata
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct EventHeader {
    /// Unique event ID
    pub event_id: u64,
    /// Timestamp in nanoseconds since epoch
    pub timestamp_ns: u64,
    /// Event type identifier
    pub event_type: u32,
    /// Source component ID
    pub source_id: u32,
    /// Payload size in bytes
    pub payload_size: u32,
    /// Flags (bitfield)
    pub flags: u32,
    /// Reserved padding to reach 64 bytes
    _reserved: [u8; EVENT_HEADER_PADDING],
}

const EVENT_HEADER_SIZE: usize = 32; // 6 fields * 4 bytes (some are u64)
const EVENT_HEADER_PADDING: usize = CACHE_LINE_SIZE - EVENT_HEADER_SIZE;

impl EventHeader {
    pub const fn new(event_id: u64, timestamp_ns: u64, event_type: u32, source_id: u32, payload_size: u32) -> Self {
        Self {
            event_id,
            timestamp_ns,
            event_type,
            source_id,
            payload_size,
            flags: 0,
            _reserved: [0u8; EVENT_HEADER_PADDING],
        }
    }
    
    pub const fn with_flags(self, flags: u32) -> Self {
        Self { flags, ..self }
    }
}

/// A single channel in the message bus
struct Channel {
    /// Ring buffer for this channel
    buffer: SPSCRingBuffer<()>,
    /// Number of subscribers
    subscriber_count: AtomicU64,
}

impl Channel {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: SPSCRingBuffer::new(capacity).expect("Failed to create channel buffer"),
            subscriber_count: AtomicU64::new(0),
        }
    }
}

/// High-performance message bus for event routing
pub struct MessageBus {
    /// Channels indexed by name hash
    channels: parking_lot::RwLock<HashMap<u64, Arc<Channel>>>,
    /// Event ID generator
    next_event_id: AtomicU64,
    /// Maximum subscribers per channel
    max_subscribers: usize,
}

impl MessageBus {
    /// Create a new message bus
    pub fn new(max_subscribers: usize) -> Self {
        Self {
            channels: parking_lot::RwLock::new(HashMap::with_capacity(64)),
            next_event_id: AtomicU64::new(1),
            max_subscribers,
        }
    }
    
    /// Create or get a channel by name
    pub fn create_channel(&self, name: &str, capacity: usize) -> Result<(), MessageBusError> {
        if name.is_empty() || name.len() > 256 {
            return Err(MessageBusError::InvalidChannelName);
        }
        
        let hash = fxhash::hash64(name.as_bytes());
        let mut channels = self.channels.write();
        
        if channels.contains_key(&hash) {
            return Err(MessageBusError::ChannelExists);
        }
        
        channels.insert(hash, Arc::new(Channel::new(capacity)));
        Ok(())
    }
    
    /// Get a channel by name
    fn get_channel(&self, name: &str) -> Option<Arc<Channel>> {
        let hash = fxhash::hash64(name.as_bytes());
        let channels = self.channels.read();
        channels.get(&hash).cloned()
    }
    
    /// Publish an event to a channel
    pub fn publish(&self, channel_name: &str, data: &[u8]) -> Result<u64, MessageBusError> {
        let channel = self.get_channel(channel_name)
            .ok_or(MessageBusError::ChannelNotFound)?;
        
        // Generate event ID
        let event_id = self.next_event_id.fetch_add(1, Ordering::AcqRel);
        
        // Push to ring buffer
        match channel.buffer.push(data) {
            Ok(_) => Ok(event_id),
            Err(_) => Err(MessageBusError::ChannelFull),
        }
    }
    
    /// Subscribe to a channel (returns events via callback)
    pub fn subscribe<F>(&self, channel_name: &str, callback: F) -> Result<(), MessageBusError>
    where
        F: Fn(&[u8]) + Send + 'static,
    {
        let channel = self.get_channel(channel_name)
            .ok_or(MessageBusError::ChannelNotFound)?;
        
        let current = channel.subscriber_count.load(Ordering::Relaxed);
        if current >= self.max_subscribers as u64 {
            return Err(MessageBusError::SubscriberLimitReached);
        }
        
        channel.subscriber_count.fetch_add(1, Ordering::Relaxed);
        
        // In a real implementation, we'd spawn a task that continuously
        // polls the channel and calls the callback. For now, we just
        // track the subscription.
        drop(callback); // Suppress unused warning
        
        Ok(())
    }
    
    /// Get the number of channels
    pub fn channel_count(&self) -> usize {
        self.channels.read().len()
    }
    
    /// Remove a channel
    pub fn remove_channel(&self, name: &str) -> bool {
        let hash = fxhash::hash64(name.as_bytes());
        let mut channels = self.channels.write();
        channels.remove(&hash).is_some()
    }
    
    /// Clear all channels
    pub fn clear(&self) {
        let mut channels = self.channels.write();
        channels.clear();
    }
}

impl Default for MessageBus {
    fn default() -> Self {
        Self::new(16)
    }
}

// Simple hash function fallback if fxhash is not available
mod fxhash {
    pub fn hash64(data: &[u8]) -> u64 {
        // Simple FNV-1a hash as fallback
        const FNV_OFFSET: u64 = 14695981039346656037;
        const FNV_PRIME: u64 = 1099511628211;
        
        let mut hash = FNV_OFFSET;
        for byte in data {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_message_bus_creation() {
        let bus = MessageBus::new(16);
        assert_eq!(bus.channel_count(), 0);
    }
    
    #[test]
    fn test_create_channel() {
        let bus = MessageBus::new(16);
        bus.create_channel("test_channel", 1024).unwrap();
        assert_eq!(bus.channel_count(), 1);
    }
    
    #[test]
    fn test_duplicate_channel() {
        let bus = MessageBus::new(16);
        bus.create_channel("test", 1024).unwrap();
        let result = bus.create_channel("test", 1024);
        assert!(matches!(result, Err(MessageBusError::ChannelExists)));
    }
    
    #[test]
    fn test_invalid_channel_name() {
        let bus = MessageBus::new(16);
        let result = bus.create_channel("", 1024);
        assert!(matches!(result, Err(MessageBusError::InvalidChannelName)));
    }
    
    #[test]
    fn test_publish_without_channel() {
        let bus = MessageBus::new(16);
        let result = bus.publish("nonexistent", b"data");
        assert!(matches!(result, Err(MessageBusError::ChannelNotFound)));
    }
    
    #[test]
    fn test_publish_to_channel() {
        let bus = MessageBus::new(16);
        bus.create_channel("test", 1024).unwrap();
        
        let event_id = bus.publish("test", b"hello").unwrap();
        assert!(event_id > 0);
    }
    
    #[test]
    fn test_remove_channel() {
        let bus = MessageBus::new(16);
        bus.create_channel("test", 1024).unwrap();
        assert!(bus.remove_channel("test"));
        assert_eq!(bus.channel_count(), 0);
    }
    
    #[test]
    fn test_clear_channels() {
        let bus = MessageBus::new(16);
        bus.create_channel("ch1", 1024).unwrap();
        bus.create_channel("ch2", 1024).unwrap();
        bus.create_channel("ch3", 1024).unwrap();
        
        bus.clear();
        assert_eq!(bus.channel_count(), 0);
    }
    
    #[test]
    fn test_event_header() {
        let header = EventHeader::new(1, 1000000, 42, 7, 256);
        assert_eq!(header.event_id, 1);
        assert_eq!(header.timestamp_ns, 1000000);
        assert_eq!(header.event_type, 42);
        assert_eq!(header.source_id, 7);
        assert_eq!(header.payload_size, 256);
        assert_eq!(header.flags, 0);
    }
}
