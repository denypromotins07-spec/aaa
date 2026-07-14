//! Global Atomic Kill Switch
//! 
//! An AtomicBool-based kill switch that is checked at the very bottom of
//! the execution stack. When tripped, all outbound messages are immediately
//! dropped and the WAPI connection halts.
//! 
//! This can be triggered by:
//! 1. Manual frontend command
//! 2. VelocityCircuitBreaker auto-trip
//! 3. System health monitor

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Result of kill switch check
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KillSwitchCheck {
    /// System is operational
    Clear,
    /// Kill switch is active - abort all operations
    Halted,
}

/// Global Kill Switch
/// 
/// Uses a single AtomicBool for zero-overhead state checking.
/// All threads can poll this with Relaxed ordering for minimal cost.
/// 
/// INTEGRATION POINT:
/// The WAPI signer must check `check()` before signing ANY message.
pub struct GlobalKillSwitch {
    /// The atomic halt flag
    halted: AtomicBool,
    /// Count of times switch was tripped
    trip_count: AtomicU64,
    /// Timestamp of last trip (nanoseconds)
    last_trip_timestamp_ns: AtomicU64,
    /// Whether auto-trip from velocity breaker is enabled
    auto_trip_enabled: AtomicBool,
}

unsafe impl Send for GlobalKillSwitch {}
unsafe impl Sync for GlobalKillSwitch {}

impl GlobalKillSwitch {
    /// Create a new kill switch in clear state
    pub fn new() -> Self {
        Self {
            halted: AtomicBool::new(false),
            trip_count: AtomicU64::new(0),
            last_trip_timestamp_ns: AtomicU64::new(0),
            auto_trip_enabled: AtomicBool::new(true),
        }
    }

    /// Check if system is halted.
    /// 
    /// This is the HOT PATH function called before every operation.
    /// Uses Relaxed ordering for minimum latency.
    #[inline]
    pub fn check(&self) -> KillSwitchCheck {
        if self.halted.load(Ordering::Relaxed) {
            KillSwitchCheck::Halted
        } else {
            KillSwitchCheck::Clear
        }
    }

    /// Quick boolean check for inline use
    #[inline]
    pub fn is_halted(&self) -> bool {
        self.halted.load(Ordering::Relaxed)
    }

    /// Trip the kill switch immediately.
    /// 
    /// This uses SeqCst ordering to ensure immediate visibility
    /// across all threads.
    /// 
    /// # Arguments
    /// * `timestamp_ns` - Timestamp of the trip event
    #[inline]
    pub fn trip(&self, timestamp_ns: u64) {
        self.halted.store(true, Ordering::SeqCst);
        self.trip_count.fetch_add(1, Ordering::Relaxed);
        self.last_trip_timestamp_ns.store(timestamp_ns, Ordering::Relaxed);
    }

    /// Reset the kill switch (after manual intervention).
    /// 
    /// WARNING: Only call after confirming it's safe to resume trading.
    #[inline]
    pub fn reset(&self) {
        self.halted.store(false, Ordering::SeqCst);
    }

    /// Enable or disable auto-trip from circuit breakers
    #[inline]
    pub fn set_auto_trip(&self, enabled: bool) {
        self.auto_trip_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if auto-trip is enabled
    #[inline]
    pub fn is_auto_trip_enabled(&self) -> bool {
        self.auto_trip_enabled.load(Ordering::Relaxed)
    }

    /// Get trip count
    #[inline]
    pub fn trip_count(&self) -> u64 {
        self.trip_count.load(Ordering::Relaxed)
    }

    /// Get last trip timestamp
    #[inline]
    pub fn last_trip_timestamp_ns(&self) -> u64 {
        self.last_trip_timestamp_ns.load(Ordering::Relaxed)
    }

    /// Check if kill switch has ever been tripped
    #[inline]
    pub fn has_been_tripped(&self) -> bool {
        self.trip_count.load(Ordering::Relaxed) > 0
    }
}

impl Default for GlobalKillSwitch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let ks = GlobalKillSwitch::new();
        assert_eq!(ks.check(), KillSwitchCheck::Clear);
        assert!(!ks.is_halted());
        assert_eq!(ks.trip_count(), 0);
    }

    #[test]
    fn test_trip_and_reset() {
        let ks = GlobalKillSwitch::new();
        
        ks.trip(1000);
        assert_eq!(ks.check(), KillSwitchCheck::Halted);
        assert!(ks.is_halted());
        assert_eq!(ks.trip_count(), 1);
        
        ks.reset();
        assert_eq!(ks.check(), KillSwitchCheck::Clear);
        assert!(!ks.is_halted());
    }

    #[test]
    fn test_concurrent_check() {
        use std::sync::Arc;
        use std::thread;
        
        let ks = Arc::new(GlobalKillSwitch::new());
        let checks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        
        let mut handles = vec![];
        
        // Multiple threads checking kill switch
        for _ in 0..10 {
            let ks_clone = Arc::clone(&ks);
            let checks_clone = Arc::clone(&checks);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    let _ = ks_clone.check();
                    checks_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }));
        }
        
        for h in handles {
            let _ = h.join();
        }
        
        assert_eq!(checks.load(std::sync::atomic::Ordering::Relaxed), 10000);
    }

    #[test]
    fn test_immediate_visibility() {
        use std::sync::Arc;
        use std::thread;
        use std::time::Duration;
        
        let ks = Arc::new(GlobalKillSwitch::new());
        let observed_halt = Arc::new(std::sync::atomic::AtomicBool::new(false));
        
        let ks_clone = Arc::clone(&ks);
        let observed_clone = Arc::clone(&observed_halt);
        
        // Thread that polls kill switch
        let handle = thread::spawn(move || {
            loop {
                if ks_clone.is_halted() {
                    observed_clone.store(true, std::sync::atomic::Ordering::Relaxed);
                    break;
                }
                thread::yield_now();
            }
        });
        
        // Main thread trips kill switch after short delay
        thread::sleep(Duration::from_millis(10));
        ks.trip(2000);
        
        // Wait for observer thread
        let _ = handle.join();
        
        // Observer should have seen the halt
        assert!(observed_halt.load(std::sync::atomic::Ordering::Relaxed));
    }
}
