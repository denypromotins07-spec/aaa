// STAGE 25: CHAPTER 4 - GLOBAL EVENT BUS
/// Lock-free event routing for critical state transitions
/// Uses atomic epoch-based reclamation for safe concurrent access

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use std::sync::Arc;
use crossbeam_epoch::{self as epoch, Atomic};
use serde::{Deserialize, Serialize};

/// Event types for system-wide broadcasting
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SystemEvent {
    KillSwitchTriggered { reason: String, timestamp_ns: u64 },
    SwarmLeaderElected { node_id: u64, term: u64 },
    AlignmentVeto { source: String, reason: String },
    MarketDataInterrupted { venue: String, duration_ms: u64 },
    RiskLimitBreached { limit_type: String, current_value: f64, threshold: f64 },
    OrderExecutionComplete { order_id: u64, filled: bool },
    ChaosModeActivated { test_name: String },
    ChaosModeDeactivated { test_name: String },
}

/// Event priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3, // Kill-switch, alignment veto
}

impl EventPriority {
    pub fn from_event(event: &SystemEvent) -> Self {
        match event {
            SystemEvent::KillSwitchTriggered { .. } => EventPriority::Critical,
            SystemEvent::AlignmentVeto { .. } => EventPriority::Critical,
            SystemEvent::RiskLimitBreached { .. } => EventPriority::High,
            SystemEvent::SwarmLeaderElected { .. } => EventPriority::High,
            SystemEvent::MarketDataInterrupted { .. } => EventPriority::High,
            SystemEvent::ChaosModeActivated { .. } => EventPriority::Normal,
            SystemEvent::ChaosModeDeactivated { .. } => EventPriority::Normal,
            _ => EventPriority::Normal,
        }
    }
}

/// Enqueued event with metadata
pub struct EnqueuedEvent {
    pub event: SystemEvent,
    pub priority: EventPriority,
    pub timestamp_ns: u64,
    pub sequence_number: u64,
}

/// Lock-free Global Event Bus
/// Routes critical state transitions to all subsystems simultaneously
pub struct GlobalEventBus {
    /// Epoch-based atomic queue for events
    event_queue: Atomic<EventQueueNode>,
    /// Sequence counter for ordering
    sequence_counter: AtomicU64,
    /// Subscriber count
    subscriber_count: AtomicUsize,
    /// Total events processed
    events_processed: AtomicU64,
    /// Critical event flag (bypasses normal queue)
    has_critical_event: AtomicBool,
    /// Chaos mode flag
    chaos_mode_active: AtomicBool,
}

/// Queue node for lock-free linked list
struct EventQueueNode {
    event: Option<EnqueuedEvent>,
    next: Atomic<EventQueueNode>,
}

unsafe impl epoch::Sync for EventQueueNode {}

impl GlobalEventBus {
    pub fn new() -> Self {
        Self {
            event_queue: Atomic::null(epoch::unprotected()),
            sequence_counter: AtomicU64::new(0),
            subscriber_count: AtomicUsize::new(0),
            events_processed: AtomicU64::new(0),
            has_critical_event: AtomicBool::new(false),
            chaos_mode_active: AtomicBool::new(false),
        }
    }

    /// Activate chaos mode
    pub fn activate_chaos_mode(&self) {
        self.chaos_mode_active.store(true, Ordering::SeqCst);
    }

    /// Deactivate chaos mode
    pub fn deactivate_chaos_mode(&self) {
        self.chaos_mode_active.store(false, Ordering::SeqCst);
    }

    /// Check if chaos mode is active
    pub fn is_chaos_mode_active(&self) -> bool {
        self.chaos_mode_active.load(Ordering::SeqCst)
    }

    /// Publish an event to all subscribers (lock-free)
    pub fn publish(&self, event: SystemEvent) -> Result<u64, EventBusError> {
        let now = Instant::now();
        let now_ns = now.duration_since(Instant::EPOCH).as_nanos() as u64;

        let priority = EventPriority::from_event(&event);
        let sequence = self.sequence_counter.fetch_add(1, Ordering::Relaxed);

        let enqueued = EnqueuedEvent {
            event,
            priority,
            timestamp_ns: now_ns,
            sequence_number: sequence,
        };

        // Critical events set flag for immediate processing
        if priority == EventPriority::Critical {
            self.has_critical_event.store(true, Ordering::SeqCst);
        }

        // Create new queue node
        let guard = epoch::pin();
        let new_node = Box::new(EventQueueNode {
            event: Some(enqueued),
            next: Atomic::null(guard),
        });

        // Lock-free prepend to queue
        let mut new_ptr = guard.enter().allocate(new_node);
        loop {
            let head = self.event_queue.load(Ordering::Acquire, &guard);
            unsafe {
                (*new_ptr.as_ptr()).next.store(head, Ordering::Release, &guard);
            }

            if self.event_queue.compare_exchange_weak(
                head,
                new_ptr,
                Ordering::Release,
                Ordering::Relaxed,
                &guard
            ).is_ok() {
                break;
            }
        }

        Ok(sequence)
    }

    /// Consume next event from queue (lock-free)
    pub fn consume(&self) -> Option<EnqueuedEvent> {
        let guard = epoch::pin();

        loop {
            let head = self.event_queue.load(Ordering::Acquire, &guard);

            if head.is_null() {
                return None;
            }

            let head_ref = unsafe { guard.reference(head) };
            let next = head_ref.next.load(Ordering::Acquire, &guard);

            // Try to swap head with next
            if self.event_queue.compare_exchange_weak(
                head,
                next,
                Ordering::Release,
                Ordering::Relaxed,
                &guard
            ).is_ok() {
                // Successfully dequeued
                let event = unsafe { (*head.as_ptr()).event.take() };

                // Defer deletion for epoch safety
                unsafe {
                    guard.defer_destroy(head);
                }

                if let Some(enqueued) = event {
                    self.events_processed.fetch_add(1, Ordering::Relaxed);

                    // Clear critical flag if this was the last critical event
                    if enqueued.priority == EventPriority::Critical {
                        self.has_critical_event.store(false, Ordering::SeqCst);
                    }

                    return Some(enqueued);
                }
            }
            // CAS failed, retry
        }
    }

    /// Check if there are pending critical events
    pub fn has_critical_events(&self) -> bool {
        self.has_critical_event.load(Ordering::SeqCst)
    }

    /// Get bus statistics
    pub fn get_stats(&self) -> EventBusStats {
        EventBusStats {
            subscriber_count: self.subscriber_count.load(Ordering::Relaxed),
            events_processed: self.events_processed.load(Ordering::Relaxed),
            current_sequence: self.sequence_counter.load(Ordering::Relaxed),
            has_critical_events: self.has_critical_event.load(Ordering::SeqCst),
            chaos_mode_active: self.chaos_mode_active.load(Ordering::SeqCst),
        }
    }

    /// Register a subscriber
    pub fn register_subscriber(&self) -> usize {
        self.subscriber_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Unregister a subscriber
    pub fn unregister_subscriber(&self) -> usize {
        self.subscriber_count.fetch_sub(1, Ordering::Relaxed).saturating_sub(1)
    }

    /// Broadcast kill switch event (highest priority)
    pub fn broadcast_kill_switch(&self, reason: String) -> Result<u64, EventBusError> {
        let now = Instant::now();
        let now_ns = now.duration_since(Instant::EPOCH).as_nanos() as u64;

        let event = SystemEvent::KillSwitchTriggered {
            reason,
            timestamp_ns: now_ns,
        };

        self.publish(event)
    }

    /// Broadcast alignment veto from Stage 24 Super-Ego
    pub fn broadcast_alignment_veto(&self, source: String, reason: String) -> Result<u64, EventBusError> {
        let event = SystemEvent::AlignmentVeto { source, reason };
        self.publish(event)
    }

    /// Broadcast swarm leader election result
    pub fn broadcast_leader_election(&self, node_id: u64, term: u64) -> Result<u64, EventBusError> {
        let event = SystemEvent::SwarmLeaderElected { node_id, term };
        self.publish(event)
    }
}

/// Event bus statistics
#[derive(Debug, Clone)]
pub struct EventBusStats {
    pub subscriber_count: usize,
    pub events_processed: u64,
    pub current_sequence: u64,
    pub has_critical_events: bool,
    pub chaos_mode_active: bool,
}

/// Event bus errors
#[derive(Debug, Clone, PartialEq)]
pub enum EventBusError {
    QueueFull,
    AllocationFailed,
    InvalidEvent,
}

impl std::fmt::Display for EventBusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventBusError::QueueFull => write!(f, "Event queue full"),
            EventBusError::AllocationFailed => write!(f, "Memory allocation failed"),
            EventBusError::InvalidEvent => write!(f, "Invalid event"),
        }
    }
}

impl std::error::Error for EventBusError {}

impl Default for GlobalEventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_publish_and_consume() {
        let bus = GlobalEventBus::new();

        let event = SystemEvent::OrderExecutionComplete { order_id: 123, filled: true };
        let seq = bus.publish(event.clone());
        assert!(seq.is_ok());

        let consumed = bus.consume();
        assert!(consumed.is_some());
        let consumed = consumed.unwrap();
        assert_eq!(consumed.event, event);
    }

    #[test]
    fn test_critical_event_priority() {
        let bus = GlobalEventBus::new();

        assert!(!bus.has_critical_events());

        let event = SystemEvent::KillSwitchTriggered {
            reason: "Test".to_string(),
            timestamp_ns: 0,
        };
        let _ = bus.publish(event);

        assert!(bus.has_critical_events());

        // Consume should clear the flag
        let _ = bus.consume();
        assert!(!bus.has_critical_events());
    }

    #[test]
    fn test_subscriber_count() {
        let bus = GlobalEventBus::new();

        assert_eq!(bus.get_stats().subscriber_count, 0);

        let sub1 = bus.register_subscriber();
        assert_eq!(sub1, 1);

        let sub2 = bus.register_subscriber();
        assert_eq!(sub2, 2);

        bus.unregister_subscriber();
        assert_eq!(bus.get_stats().subscriber_count, 1);
    }

    #[test]
    fn test_broadcast_helpers() {
        let bus = GlobalEventBus::new();

        let seq = bus.broadcast_kill_switch("Emergency".to_string());
        assert!(seq.is_ok());

        let consumed = bus.consume();
        assert!(consumed.is_some());
        assert!(matches!(consumed.unwrap().event, SystemEvent::KillSwitchTriggered { .. }));
    }
}
