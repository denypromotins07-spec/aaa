//! Slow Client Guillotine - Backpressure Protection
//!
//! This module implements the "guillotine" mechanism that severs connections
//! to slow WebSocket clients. If a client's send buffer fills up and stays
//! full for too long, the connection is immediately terminated to protect
//! the trading engine from backpressure.
//!
//! ROOT CAUSE FIX: Implements a grace period / timeout threshold before severing
//! to avoid false positives from temporary TCP window scaling issues.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::Mutex;

/// Configuration for the slow client guillotine
pub struct GuillotineConfig {
    /// Maximum number of consecutive send failures before disconnection
    pub max_consecutive_failures: usize,
    /// Grace period before counting failures (allows for temporary network hiccups)
    pub grace_period: Duration,
    /// Maximum time a client can be unresponsive before disconnection
    pub max_unresponsive_time: Duration,
    /// Maximum pending messages in client buffer
    pub max_pending_messages: usize,
}

impl Default for GuillotineConfig {
    fn default() -> Self {
        Self {
            max_consecutive_failures: 10, // Allow 10 consecutive failures
            grace_period: Duration::from_millis(100), // 100ms grace period
            max_unresponsive_time: Duration::from_secs(5), // 5 seconds max unresponsive
            max_pending_messages: 2048, // Max 2K pending messages
        }
    }
}

/// Statistics for a single client
#[derive(Debug, Clone)]
pub struct ClientStats {
    /// Total messages sent
    pub total_sent: u64,
    /// Total messages dropped
    pub total_dropped: u64,
    /// Consecutive send failures
    pub consecutive_failures: usize,
    /// Last successful send timestamp
    pub last_send_time: Option<Instant>,
    /// First failure timestamp (for grace period tracking)
    pub first_failure_time: Option<Instant>,
    /// Connection established timestamp
    pub connected_at: Instant,
}

/// Internal state for tracking a client
struct ClientState {
    stats: Mutex<ClientStats>,
    /// Whether client has been marked for disconnection
    marked_for_disconnect: AtomicBool,
    /// Reason for disconnection
    disconnect_reason: Mutex<Option<DisconnectReason>>,
}

/// Reason for guillotine disconnection
#[derive(Debug, Clone)]
pub enum DisconnectReason {
    /// Too many consecutive send failures
    ConsecutiveFailures { count: usize },
    /// Client unresponsive for too long
    Unresponsive { duration: Duration },
    /// Buffer overflow
    BufferOverflow { pending_count: usize },
    /// Manual disconnect
    Manual,
}

/// The Guillotine - monitors and severs slow clients
pub struct SlowClientGuillotine {
    config: GuillotineConfig,
    /// Connected clients tracked by ID
    clients: Mutex<std::collections::HashMap<u64, Arc<ClientState>>>,
    /// Next client ID
    next_client_id: AtomicU64,
    /// Total disconnections
    total_disconnections: AtomicU64,
    /// Active flag
    active: AtomicBool,
}

impl SlowClientGuillotine {
    /// Create a new guillotine with the given configuration
    pub fn new(config: GuillotineConfig) -> Self {
        Self {
            config,
            clients: Mutex::new(std::collections::HashMap::new()),
            next_client_id: AtomicU64::new(1),
            total_disconnections: AtomicU64::new(0),
            active: AtomicBool::new(true),
        }
    }

    /// Register a new client and get its ID
    pub fn register_client(&self) -> u64 {
        let id = self.next_client_id.fetch_add(1, Ordering::Relaxed);
        
        let state = Arc::new(ClientState {
            stats: Mutex::new(ClientStats {
                total_sent: 0,
                total_dropped: 0,
                consecutive_failures: 0,
                last_send_time: None,
                first_failure_time: None,
                connected_at: Instant::now(),
            }),
            marked_for_disconnect: AtomicBool::new(false),
            disconnect_reason: Mutex::new(None),
        });
        
        self.clients.lock().insert(id, state);
        id
    }

    /// Record a successful send
    pub fn record_success(&self, client_id: u64) {
        if let Some(client) = self.clients.lock().get(&client_id) {
            let mut stats = client.stats.lock();
            stats.total_sent += 1;
            stats.consecutive_failures = 0;
            stats.first_failure_time = None;
            stats.last_send_time = Some(Instant::now());
        }
    }

    /// Record a send failure (buffer full)
    /// Returns true if the client should be disconnected
    pub fn record_failure(&self, client_id: u64) -> bool {
        if !self.active.load(Ordering::Relaxed) {
            return true; // Guillotine inactive, force disconnect
        }

        if let Some(client) = self.clients.lock().get(&client_id) {
            let mut stats = client.stats.lock();
            stats.total_dropped += 1;
            stats.consecutive_failures += 1;
            
            let now = Instant::now();
            
            // Set first failure time if this is the first failure
            if stats.first_failure_time.is_none() {
                stats.first_failure_time = Some(now);
            }
            
            // Check if grace period has elapsed
            let grace_elapsed = stats.first_failure_time
                .map(|t| t.elapsed() >= self.config.grace_period)
                .unwrap_or(false);
            
            // Check if we should disconnect
            if grace_elapsed && stats.consecutive_failures >= self.config.max_consecutive_failures {
                client.marked_for_disconnect.store(true, Ordering::SeqCst);
                *client.disconnect_reason.lock() = Some(DisconnectReason::ConsecutiveFailures {
                    count: stats.consecutive_failures,
                });
                return true;
            }
            
            // Check for unresponsive client
            if let Some(last_send) = stats.last_send_time {
                if last_send.elapsed() >= self.config.max_unresponsive_time {
                    client.marked_for_disconnect.store(true, Ordering::SeqCst);
                    *client.disconnect_reason.lock() = Some(DisconnectReason::Unresponsive {
                        duration: last_send.elapsed(),
                    });
                    return true;
                }
            }
            
            // Drop the lock before returning
            drop(stats);
            
            // Check pending message count
            let pending = client.stats.lock().total_dropped;
            if pending as usize >= self.config.max_pending_messages {
                client.marked_for_disconnect.store(true, Ordering::SeqCst);
                *client.disconnect_reason.lock() = Some(DisconnectReason::BufferOverflow {
                    pending_count: pending as usize,
                });
                return true;
            }
            
            false
        } else {
            // Client not found - should disconnect
            true
        }
    }

    /// Check if a client should be disconnected
    pub fn should_disconnect(&self, client_id: u64) -> bool {
        self.clients
            .lock()
            .get(&client_id)
            .map(|c| c.marked_for_disconnect.load(Ordering::SeqCst))
            .unwrap_or(true)
    }

    /// Get the disconnect reason for a client
    pub fn get_disconnect_reason(&self, client_id: u64) -> Option<DisconnectReason> {
        self.clients
            .lock()
            .get(&client_id)
            .and_then(|c| c.disconnect_reason.lock().clone())
    }

    /// Unregister a client (called after disconnection)
    pub fn unregister_client(&self, client_id: u64) -> Option<ClientStats> {
        self.clients
            .lock()
            .remove(&client_id)
            .map(|c| c.stats.lock().clone())
    }

    /// Manually mark a client for disconnection
    pub fn mark_for_disconnect(&self, client_id: u64, reason: DisconnectReason) {
        if let Some(client) = self.clients.lock().get(&client_id) {
            client.marked_for_disconnect.store(true, Ordering::SeqCst);
            *client.disconnect_reason.lock() = Some(reason);
        }
    }

    /// Get statistics for a client
    pub fn get_client_stats(&self, client_id: u64) -> Option<ClientStats> {
        self.clients
            .lock()
            .get(&client_id)
            .map(|c| c.stats.lock().clone())
    }

    /// Get count of connected clients
    pub fn connected_client_count(&self) -> usize {
        self.clients.lock().len()
    }

    /// Get total disconnection count
    pub fn total_disconnections(&self) -> u64 {
        self.total_disconnections.load(Ordering::Relaxed)
    }

    /// Increment disconnection counter (called when actually disconnecting)
    pub fn increment_disconnections(&self) {
        self.total_disconnections.fetch_add(1, Ordering::Relaxed);
    }

    /// Shutdown the guillotine (force all clients to disconnect)
    pub fn shutdown(&self) {
        self.active.store(false, Ordering::SeqCst);
        
        // Mark all clients for disconnect
        for client in self.clients.lock().values() {
            client.marked_for_disconnect.store(true, Ordering::SeqCst);
            *client.disconnect_reason.lock() = Some(DisconnectReason::Manual);
        }
    }

    /// Check if guillotine is active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Run periodic health check on all clients
    /// Returns list of client IDs that should be disconnected
    pub fn health_check(&self) -> Vec<u64> {
        let mut to_disconnect = Vec::new();
        let now = Instant::now();
        
        for (&id, client) in self.clients.lock().iter() {
            let stats = client.stats.lock();
            
            // Check for unresponsive clients
            if let Some(last_send) = stats.last_send_time {
                if last_send.elapsed() >= self.config.max_unresponsive_time {
                    to_disconnect.push(id);
                    client.marked_for_disconnect.store(true, Ordering::SeqCst);
                    *client.disconnect_reason.lock() = Some(DisconnectReason::Unresponsive {
                        duration: last_send.elapsed(),
                    });
                }
            }
            
            // Reset grace period if failures have stopped
            if stats.consecutive_failures == 0 {
                drop(stats);
                let mut stats = client.stats.lock();
                stats.first_failure_time = None;
            }
        }
        
        to_disconnect
    }
}

impl Default for SlowClientGuillotine {
    fn default() -> Self {
        Self::new(GuillotineConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guillotine_basic() {
        let guillotine = SlowClientGuillotine::default();
        
        let client_id = guillotine.register_client();
        assert!(guillotine.get_client_stats(client_id).is_some());
        
        // Record some successes
        guillotine.record_success(client_id);
        guillotine.record_success(client_id);
        
        let stats = guillotine.get_client_stats(client_id).unwrap();
        assert_eq!(stats.total_sent, 2);
        assert_eq!(stats.consecutive_failures, 0);
    }

    #[test]
    fn test_guillotine_consecutive_failures() {
        let config = GuillotineConfig {
            max_consecutive_failures: 3,
            grace_period: Duration::from_millis(10),
            ..Default::default()
        };
        let guillotine = SlowClientGuillotine::new(config);
        
        let client_id = guillotine.register_client();
        
        // Record failures (should trigger after grace period)
        for _ in 0..5 {
            let _ = guillotine.record_failure(client_id);
            std::thread::sleep(Duration::from_millis(5));
        }
        
        // Should be marked for disconnect
        assert!(guillotine.should_disconnect(client_id));
        
        let reason = guillotine.get_disconnect_reason(client_id);
        assert!(matches!(reason, Some(DisconnectReason::ConsecutiveFailures { .. })));
    }

    #[test]
    fn test_guillotine_recovery() {
        let guillotine = SlowClientGuillotine::default();
        let client_id = guillotine.register_client();
        
        // Record some failures
        for _ in 0..3 {
            let _ = guillotine.record_failure(client_id);
        }
        
        // Verify failures are tracked
        let stats = guillotine.get_client_stats(client_id).unwrap();
        assert_eq!(stats.consecutive_failures, 3);
        
        // Record success - should reset failure count
        guillotine.record_success(client_id);
        
        let stats = guillotine.get_client_stats(client_id).unwrap();
        assert_eq!(stats.consecutive_failures, 0);
        assert!(!guillotine.should_disconnect(client_id));
    }
}
