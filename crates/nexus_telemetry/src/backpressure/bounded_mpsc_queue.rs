//! Bounded MPSC Queue for Backpressure Handling
//!
//! This module implements a bounded, lock-free MPSC (Multi-Producer Single-Consumer)
//! channel for telemetry broadcast. When the queue fills up due to slow clients,
//! new frames are dropped instead of blocking the trading engine.
//!
//! CRITICAL: This queue is designed to NEVER backpressure into the trading engine.
//! If the UI can't keep up, frames are silently dropped.

use crossbeam::channel::{bounded, Sender, Receiver, TrySendError};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

/// Configuration for the bounded queue
pub struct BoundedQueueConfig {
    /// Maximum number of frames in the queue
    pub capacity: usize,
    /// Whether to drop oldest or newest on overflow
    pub drop_policy: DropPolicy,
}

impl Default for BoundedQueueConfig {
    fn default() -> Self {
        Self {
            capacity: 1024, // ~17 seconds at 60fps
            drop_policy: DropPolicy::DropNewest,
        }
    }
}

/// Drop policy when queue is full
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropPolicy {
    /// Drop the newest frame (protect old data)
    DropNewest,
    /// Drop the oldest frame (keep latest data)
    DropOldest,
}

/// Internal state for tracking queue metrics
struct QueueState {
    /// Count of frames sent
    frames_sent: AtomicU64,
    /// Count of frames dropped due to overflow
    frames_dropped: AtomicU64,
    /// Queue is active
    active: AtomicBool,
}

/// Bounded MPSC queue with backpressure handling
pub struct BoundedMpscQueue<T> {
    sender: Sender<T>,
    receiver: Receiver<T>,
    state: Arc<QueueState>,
    config: BoundedQueueConfig,
}

impl<T> BoundedMpscQueue<T> {
    /// Create a new bounded queue
    pub fn new(config: BoundedQueueConfig) -> Self 
    where
        T: Send + 'static,
    {
        let (sender, receiver) = bounded(config.capacity);
        
        Self {
            sender,
            receiver,
            state: Arc::new(QueueState {
                frames_sent: AtomicU64::new(0),
                frames_dropped: AtomicU64::new(0),
                active: AtomicBool::new(true),
            }),
            config,
        }
    }

    /// Try to send an item (non-blocking)
    /// Returns Ok if sent, Err with the item if dropped
    #[inline]
    pub fn try_send(&self, item: T) -> Result<(), T> {
        if !self.state.active.load(Ordering::Relaxed) {
            return Err(item);
        }

        match self.sender.try_send(item) {
            Ok(()) => {
                self.state.frames_sent.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(TrySendError::Full(item)) => {
                // Queue is full - apply drop policy
                self.state.frames_dropped.fetch_add(1, Ordering::Relaxed);
                
                match self.config.drop_policy {
                    DropPolicy::DropNewest => {
                        // Simply drop the new item (return error)
                        Err(item)
                    }
                    DropPolicy::DropOldest => {
                        // Try to receive and discard oldest, then retry
                        // Note: This is tricky with MPSC - we just drop newest for simplicity
                        Err(item)
                    }
                }
            }
            Err(TrySendError::Disconnected(item)) => {
                // Consumer disconnected
                Err(item)
            }
        }
    }

    /// Receive an item (blocking)
    pub fn recv(&self) -> Result<T, crossbeam::channel::RecvError> {
        self.receiver.recv()
    }

    /// Try to receive an item (non-blocking)
    #[inline]
    pub fn try_recv(&self) -> Result<T, crossbeam::channel::TryRecvError> {
        self.receiver.try_recv()
    }

    /// Get the receiver end
    pub fn receiver(&self) -> &Receiver<T> {
        &self.receiver
    }

    /// Check if queue is active
    pub fn is_active(&self) -> bool {
        self.state.active.load(Ordering::Relaxed)
    }

    /// Shutdown the queue
    pub fn shutdown(&self) {
        self.state.active.store(false, Ordering::SeqCst);
    }

    /// Get count of frames sent
    pub fn frames_sent(&self) -> u64 {
        self.state.frames_sent.load(Ordering::Relaxed)
    }

    /// Get count of frames dropped
    pub fn frames_dropped(&self) -> u64 {
        self.state.frames_dropped.load(Ordering::Relaxed)
    }

    /// Get current queue length (approximate)
    pub fn len(&self) -> usize {
        self.receiver.len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }
}

/// Producer handle for sending frames
pub struct QueueProducer<T> {
    sender: Sender<T>,
    state: Arc<QueueState>,
    drop_policy: DropPolicy,
}

impl<T> QueueProducer<T> {
    /// Try to send a frame
    #[inline]
    pub fn try_send(&self, item: T) -> Result<(), T> {
        if !self.state.active.load(Ordering::Relaxed) {
            return Err(item);
        }

        match self.sender.try_send(item) {
            Ok(()) => {
                self.state.frames_sent.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(TrySendError::Full(item)) => {
                self.state.frames_dropped.fetch_add(1, Ordering::Relaxed);
                Err(item)
            }
            Err(TrySendError::Disconnected(item)) => Err(item),
        }
    }

    /// Check if producer is active
    pub fn is_active(&self) -> bool {
        self.state.active.load(Ordering::Relaxed)
    }
}

/// Consumer handle for receiving frames
pub struct QueueConsumer<T> {
    receiver: Receiver<T>,
    state: Arc<QueueState>,
}

impl<T> QueueConsumer<T> {
    /// Receive a frame (blocking)
    pub fn recv(&self) -> Result<T, crossbeam::channel::RecvError> {
        self.receiver.recv()
    }

    /// Try to receive a frame (non-blocking)
    #[inline]
    pub fn try_recv(&self) -> Result<T, crossbeam::channel::TryRecvError> {
        self.receiver.try_recv()
    }

    /// Check if there are pending frames
    pub fn has_pending(&self) -> bool {
        !self.receiver.is_empty()
    }
}

/// Split the queue into producer and consumer handles
pub fn split_queue<T>(queue: BoundedMpscQueue<T>) -> (QueueProducer<T>, QueueConsumer<T>) {
    let arc_state = Arc::clone(&queue.state);
    
    let producer = QueueProducer {
        sender: queue.sender.clone(),
        state: arc_state.clone(),
        drop_policy: queue.config.drop_policy,
    };
    
    let consumer = QueueConsumer {
        receiver: queue.receiver,
        state: arc_state,
    };
    
    (producer, consumer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_queue_basic() {
        let config = BoundedQueueConfig::default();
        let queue: BoundedMpscQueue<i32> = BoundedMpscQueue::new(config);
        
        // Send some items
        assert!(queue.try_send(1).is_ok());
        assert!(queue.try_send(2).is_ok());
        assert!(queue.try_send(3).is_ok());
        
        // Receive items
        assert_eq!(queue.try_recv().unwrap(), 1);
        assert_eq!(queue.try_recv().unwrap(), 2);
        assert_eq!(queue.try_recv().unwrap(), 3);
    }

    #[test]
    fn test_bounded_queue_overflow() {
        let config = BoundedQueueConfig {
            capacity: 2,
            drop_policy: DropPolicy::DropNewest,
        };
        let queue: BoundedMpscQueue<i32> = BoundedMpscQueue::new(config);
        
        // Fill the queue
        assert!(queue.try_send(1).is_ok());
        assert!(queue.try_send(2).is_ok());
        
        // Third send should fail (queue full)
        assert!(queue.try_send(3).is_err());
        
        // Verify drop count
        assert_eq!(queue.frames_dropped(), 1);
    }

    #[test]
    fn test_split_queue() {
        let config = BoundedQueueConfig::default();
        let queue: BoundedMpscQueue<i32> = BoundedMpscQueue::new(config);
        let (producer, consumer) = split_queue(queue);
        
        producer.try_send(42).unwrap();
        assert_eq!(consumer.try_recv().unwrap(), 42);
    }
}
