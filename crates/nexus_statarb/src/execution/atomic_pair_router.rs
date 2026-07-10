//! Atomic Pair Execution Router
//! 
//! Tracks fill status of both legs in a stat-arb pair trade and
//! mitigates legging risk through aggressive hedging or immediate flattening.

use super::legging_risk_state_machine::{LegState, LeggingRiskState, PairExecutionState};
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};

/// Maximum number of retry attempts for failed leg
const MAX_RETRY_ATTEMPTS: u8 = 5;

/// Result of routing decision
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingDecision {
    /// Both legs ready to submit
    SubmitBoth,
    /// Submit only leg A (leg B pending)
    SubmitLegA,
    /// Submit only leg B (leg A pending)
    SubmitLegB,
    /// Leg A filled, waiting for leg B
    WaitForLegB,
    /// Leg B filled, waiting for leg A
    WaitForLegA,
    /// Leg A filled, leg B failed - hedge immediately
    HedgeLegA,
    /// Leg B filled, leg A failed - hedge immediately
    HedgeLegB,
    /// Both legs filled
    Complete,
    /// Error - cancel both and flatten
    CancelAll,
    /// Emergency flatten all positions
    EmergencyFlatten,
}

impl Default for RoutingDecision {
    #[inline]
    fn default() -> Self {
        Self::SubmitBoth
    }
}

/// Configuration for the atomic router
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AtomicRouterConfig {
    /// Maximum time (ms) to wait for second leg before hedging
    pub max_wait_time_ms: u64,
    /// Maximum slippage tolerance (basis points)
    pub max_slippage_bps: u16,
    /// Number of retry attempts before triggering hedge
    pub max_retries: u8,
    /// Whether to use proxy hedging
    pub use_proxy_hedge: bool,
}

impl AtomicRouterConfig {
    #[inline]
    pub const fn standard() -> Self {
        Self {
            max_wait_time_ms: 100,
            max_slippage_bps: 50,
            max_retries: MAX_RETRY_ATTEMPTS,
            use_proxy_hedge: true,
        }
    }

    #[inline]
    pub const fn aggressive() -> Self {
        Self {
            max_wait_time_ms: 50,
            max_slippage_bps: 25,
            max_retries: 3,
            use_proxy_hedge: true,
        }
    }

    #[inline]
    pub const fn conservative() -> Self {
        Self {
            max_wait_time_ms: 200,
            max_slippage_bps: 100,
            max_retries: MAX_RETRY_ATTEMPTS,
            use_proxy_hedge: false,
        }
    }
}

impl Default for AtomicRouterConfig {
    #[inline]
    fn default() -> Self {
        Self::standard()
    }
}

/// Fill information for a leg
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LegFillInfo {
    /// Whether the leg has been filled
    pub filled: bool,
    /// Fill quantity
    pub fill_qty: i64,
    /// Fill price
    pub fill_price: f64,
    /// Timestamp of fill (microseconds since epoch)
    pub fill_timestamp_us: u64,
    /// Number of retry attempts made
    pub retry_count: u8,
}

impl Default for LegFillInfo {
    #[inline]
    fn default() -> Self {
        Self {
            filled: false,
            fill_qty: 0,
            fill_price: 0.0,
            fill_timestamp_us: 0,
            retry_count: 0,
        }
    }
}

/// Atomic Pair Execution Router
/// 
/// Manages the execution of paired trades with strict legging risk controls.
pub struct AtomicPairRouter {
    /// Current state of leg A
    leg_a_state: LegState,
    /// Current state of leg B
    leg_b_state: LegState,
    /// Fill info for leg A
    leg_a_fill: LegFillInfo,
    /// Fill info for leg B
    leg_b_fill: LegFillInfo,
    /// Configuration
    config: AtomicRouterConfig,
    /// Start timestamp of pair execution (microseconds)
    start_timestamp_us: AtomicU64,
    /// Whether an emergency stop has been triggered
    emergency_stop: AtomicBool,
    /// State transition counter
    transition_count: AtomicU64,
    /// Last routing decision
    last_decision: AtomicU8,
}

impl AtomicPairRouter {
    /// Create a new atomic pair router
    #[inline]
    pub fn new(config: AtomicRouterConfig) -> Self {
        Self {
            leg_a_state: LegState::Pending,
            leg_b_state: LegState::Pending,
            leg_a_fill: LegFillInfo::default(),
            leg_b_fill: LegFillInfo::default(),
            config,
            start_timestamp_us: AtomicU64::new(0),
            emergency_stop: AtomicBool::new(false),
            transition_count: AtomicU64::new(0),
            last_decision: AtomicU8::new(RoutingDecision::SubmitBoth as u8),
        }
    }

    /// Initialize a new pair execution
    /// 
    /// # Arguments
    /// * `timestamp_us` - Current timestamp in microseconds
    #[inline]
    pub fn init_pair(&mut self, timestamp_us: u64) {
        self.leg_a_state = LegState::Pending;
        self.leg_b_state = LegState::Pending;
        self.leg_a_fill = LegFillInfo::default();
        self.leg_b_fill = LegFillInfo::default();
        self.start_timestamp_us.store(timestamp_us, Ordering::Relaxed);
        self.emergency_stop.store(false, Ordering::Relaxed);
        self.transition_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Update the state based on leg A fill status
    /// 
    /// # Arguments
    /// * `filled` - Whether leg A was filled
    /// * `qty` - Fill quantity (negative for short, positive for long)
    /// * `price` - Fill price
    /// * `timestamp_us` - Fill timestamp
    #[inline]
    pub fn update_leg_a(&mut self, filled: bool, qty: i64, price: f64, timestamp_us: u64) {
        if filled {
            self.leg_a_state = LegState::Filled;
            self.leg_a_fill = LegFillInfo {
                filled: true,
                fill_qty: qty,
                fill_price: price,
                fill_timestamp_us: timestamp_us,
                retry_count: 0,
            };
        } else if self.leg_a_fill.retry_count < self.config.max_retries {
            self.leg_a_state = LegState::Retrying;
            self.leg_a_fill.retry_count += 1;
        } else {
            self.leg_a_state = LegState::Failed;
        }
        
        self.transition_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Update the state based on leg B fill status
    #[inline]
    pub fn update_leg_b(&mut self, filled: bool, qty: i64, price: f64, timestamp_us: u64) {
        if filled {
            self.leg_b_state = LegState::Filled;
            self.leg_b_fill = LegFillInfo {
                filled: true,
                fill_qty: qty,
                fill_price: price,
                fill_timestamp_us: timestamp_us,
                retry_count: 0,
            };
        } else if self.leg_b_fill.retry_count < self.config.max_retries {
            self.leg_b_state = LegState::Retrying;
            self.leg_b_fill.retry_count += 1;
        } else {
            self.leg_b_state = LegState::Failed;
        }
        
        self.transition_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the current routing decision based on leg states
    /// 
    /// # Arguments
    /// * `current_timestamp_us` - Current timestamp for timeout calculation
    #[inline]
    pub fn get_routing_decision(&self, current_timestamp_us: u64) -> RoutingDecision {
        if self.emergency_stop.load(Ordering::Relaxed) {
            return RoutingDecision::EmergencyFlatten;
        }

        let elapsed_ms = current_timestamp_us.saturating_sub(
            self.start_timestamp_us.load(Ordering::Relaxed)
        ) / 1000;

        // Check for timeout condition
        let timed_out = elapsed_ms > self.config.max_wait_time_ms;

        match (self.leg_a_state, self.leg_b_state) {
            // Both pending - submit both
            (LegState::Pending, LegState::Pending) => RoutingDecision::SubmitBoth,
            
            // A pending, B filled - submit A
            (LegState::Pending, LegState::Filled) => RoutingDecision::SubmitLegA,
            
            // A filled, B pending - wait or hedge
            (LegState::Filled, LegState::Pending) => {
                if timed_out || !self.config.use_proxy_hedge {
                    RoutingDecision::HedgeLegA
                } else {
                    RoutingDecision::WaitForLegB
                }
            }
            
            // A filled, B failed - must hedge
            (LegState::Filled, LegState::Failed) => RoutingDecision::HedgeLegA,
            
            // B pending, A filled - wait or hedge
            (LegState::Pending, LegState::Filled) => {
                if timed_out || !self.config.use_proxy_hedge {
                    RoutingDecision::HedgeLegB
                } else {
                    RoutingDecision::WaitForLegA
                }
            }
            
            // B filled, A failed - must hedge
            (LegState::Failed, LegState::Filled) => RoutingDecision::HedgeLegB,
            
            // Both filled - complete
            (LegState::Filled, LegState::Filled) => RoutingDecision::Complete,
            
            // Both failed - cancel
            (LegState::Failed, LegState::Failed) => RoutingDecision::CancelAll,
            
            // One retrying, one pending - wait
            (LegState::Retrying, LegState::Pending) |
            (LegState::Pending, LegState::Retrying) => {
                if timed_out {
                    RoutingDecision::CancelAll
                } else {
                    RoutingDecision::SubmitBoth
                }
            }
            
            // One retrying, one filled
            (LegState::Retrying, LegState::Filled) => {
                if timed_out {
                    RoutingDecision::HedgeLegB
                } else {
                    RoutingDecision::WaitForLegA
                }
            }
            (LegState::Filled, LegState::Retrying) => {
                if timed_out {
                    RoutingDecision::HedgeLegA
                } else {
                    RoutingDecision::WaitForLegB
                }
            }
            
            // Retry + Failed combinations
            (LegState::Retrying, LegState::Failed) |
            (LegState::Failed, LegState::Retrying) => RoutingDecision::CancelAll,
        }
    }

    /// Trigger emergency stop
    #[inline]
    pub fn trigger_emergency_stop(&mut self) {
        self.emergency_stop.store(true, Ordering::Relaxed);
        self.transition_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Check if emergency stop is active
    #[inline]
    pub fn is_emergency_active(&self) -> bool {
        self.emergency_stop.load(Ordering::Relaxed)
    }

    /// Get the current pair execution state
    #[inline]
    pub fn execution_state(&self) -> PairExecutionState {
        PairExecutionState {
            leg_a_state: self.leg_a_state,
            leg_b_state: self.leg_b_state,
            leg_a_filled: self.leg_a_fill.filled,
            leg_b_filled: self.leg_b_fill.filled,
            leg_a_qty: self.leg_a_fill.fill_qty,
            leg_b_qty: self.leg_b_fill.fill_qty,
            leg_a_price: self.leg_a_fill.fill_price,
            leg_b_price: self.leg_b_fill.fill_price,
        }
    }

    /// Get the number of state transitions
    #[inline]
    pub fn transition_count(&self) -> u64 {
        self.transition_count.load(Ordering::Relaxed)
    }

    /// Reset the router to initial state
    #[inline]
    pub fn reset(&mut self) {
        self.leg_a_state = LegState::Pending;
        self.leg_b_state = LegState::Pending;
        self.leg_a_fill = LegFillInfo::default();
        self.leg_b_fill = LegFillInfo::default();
        self.emergency_stop.store(false, Ordering::Relaxed);
        self.start_timestamp_us.store(0, Ordering::Relaxed);
        self.transition_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Update configuration
    #[inline]
    pub fn set_config(&mut self, config: AtomicRouterConfig) {
        self.config = config;
    }
}

impl Default for AtomicPairRouter {
    #[inline]
    fn default() -> Self {
        Self::new(AtomicRouterConfig::standard())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_submission() {
        let mut router = AtomicPairRouter::new(AtomicRouterConfig::standard());
        router.init_pair(1000);

        let decision = router.get_routing_decision(1000);
        assert_eq!(decision, RoutingDecision::SubmitBoth);
    }

    #[test]
    fn test_leg_a_filled_wait_for_b() {
        let mut router = AtomicPairRouter::new(AtomicRouterConfig::standard());
        router.init_pair(1000);
        
        router.update_leg_a(true, 100, 50000.0, 1100);
        
        // Within timeout - should wait
        let decision = router.get_routing_decision(1050);
        assert_eq!(decision, RoutingDecision::WaitForLegB);
    }

    #[test]
    fn test_timeout_triggers_hedge() {
        let mut router = AtomicPairRouter::new(AtomicRouterConfig::standard());
        router.init_pair(1000);
        
        router.update_leg_a(true, 100, 50000.0, 1100);
        
        // Past timeout (100ms = 100000 us)
        let decision = router.get_routing_decision(200000);
        assert_eq!(decision, RoutingDecision::HedgeLegA);
    }

    #[test]
    fn test_both_filled_complete() {
        let mut router = AtomicPairRouter::new(AtomicRouterConfig::standard());
        router.init_pair(1000);
        
        router.update_leg_a(true, 100, 50000.0, 1100);
        router.update_leg_b(true, -100, 50050.0, 1150);
        
        let decision = router.get_routing_decision(1200);
        assert_eq!(decision, RoutingDecision::Complete);
    }

    #[test]
    fn test_emergency_stop() {
        let mut router = AtomicPairRouter::new(AtomicRouterConfig::standard());
        router.init_pair(1000);
        
        router.trigger_emergency_stop();
        
        let decision = router.get_routing_decision(1100);
        assert_eq!(decision, RoutingDecision::EmergencyFlatten);
        assert!(router.is_emergency_active());
    }

    #[test]
    fn test_leg_failed_cancel() {
        let mut router = AtomicPairRouter::new(AtomicRouterConfig::standard());
        router.init_pair(1000);
        
        // Simulate max retries exceeded
        for _ in 0..=MAX_RETRY_ATTEMPTS {
            router.update_leg_a(false, 0, 0.0, 1000);
        }
        
        let decision = router.get_routing_decision(1100);
        assert_eq!(decision, RoutingDecision::CancelAll);
    }
}
