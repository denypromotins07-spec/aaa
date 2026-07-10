//! Inventory Hard-Stop State Machine.
//! Halts quoting when position breaches risk limits from Stage 5.
//! Zero-allocation, no unwrap/expect in hot paths.

use crate::inventory::reservation_price::ReservationPriceCalculator;

/// Error types for hard-stop operations
#[derive(Debug, Clone, PartialEq)]
pub enum HardStopError {
    InvalidLimits,
    PositionBreached,
    QuoteHalted,
}

/// State of the hard-stop machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardStopState {
    /// Normal operation, quoting both sides
    Active,
    /// Long limit breached, only allow sells
    LongLimitReached,
    /// Short limit breached, only allow buys
    ShortLimitReached,
    /// Both limits breached or emergency halt
    Halted,
    /// Cooldown period after halt
    Cooldown,
}

/// Configuration for hard-stop state machine
#[derive(Debug, Clone)]
pub struct HardStopConfig {
    /// Maximum long position (positive)
    pub max_long: i64,
    /// Maximum short position (negative, stored as positive value)
    pub max_short: i64,
    /// Soft warning threshold (percentage of hard limit)
    pub warning_threshold: f64,
    /// Cooldown period after halt (in milliseconds)
    pub cooldown_ms: u64,
    /// Gradual reduction factor when approaching limits
    pub gradual_reduction_start: f64,
}

impl HardStopConfig {
    pub fn new(max_long: i64, max_short: i64, cooldown_ms: u64) -> Result<Self, HardStopError> {
        if max_long <= 0 || max_short <= 0 {
            return Err(HardStopError::InvalidLimits);
        }
        
        Ok(Self {
            max_long,
            max_short,
            warning_threshold: 0.8,
            cooldown_ms,
            gradual_reduction_start: 0.9,
        })
    }
}

/// Result of quote permission check
#[derive(Debug, Clone, Copy)]
pub struct QuotePermission {
    /// Can place bid quotes?
    pub can_bid: bool,
    /// Can place ask quotes?
    pub can_ask: bool,
    /// Current state
    pub state: HardStopState,
    /// Utilization ratio (0.0 to 1.0+)
    pub utilization: f64,
    /// Recommended size reduction factor (1.0 = full size, 0.0 = no quoting)
    pub size_factor: f64,
}

impl QuotePermission {
    pub const fn new(can_bid: bool, can_ask: bool, state: HardStopState, utilization: f64, size_factor: f64) -> Self {
        Self {
            can_bid,
            can_ask,
            state,
            utilization,
            size_factor,
        }
    }
}

/// Hard-Stop State Machine for inventory risk management
pub struct HardStopStateMachine {
    config: HardStopConfig,
    state: HardStopState,
    /// Current inventory position
    current_position: i64,
    /// Timestamp of last state change (milliseconds)
    last_state_change_ms: u64,
    /// Breach count for circuit breaker
    breach_count: u32,
    /// Maximum breaches before extended halt
    max_breaches: u32,
}

impl HardStopStateMachine {
    pub fn new(config: HardStopConfig) -> Self {
        Self {
            config,
            state: HardStopState::Active,
            current_position: 0,
            last_state_change_ms: 0,
            breach_count: 0,
            max_breaches: 5,
        }
    }
    
    /// Update current position and recalculate state
    #[inline(always)]
    pub fn update_position(&mut self, position: i64, timestamp_ms: u64) -> QuotePermission {
        self.current_position = position;
        self.evaluate_state(timestamp_ms)
    }
    
    /// Evaluate current state based on position
    #[inline(always)]
    fn evaluate_state(&mut self, timestamp_ms: u64) -> QuotePermission {
        let prev_state = self.state;
        
        // Check for position breaches
        if self.current_position >= self.config.max_long {
            if self.state != HardStopState::LongLimitReached 
                && self.state != HardStopState::Halted {
                self.breach_count += 1;
                
                if self.breach_count >= self.max_breaches {
                    self.state = HardStopState::Halted;
                } else {
                    self.state = HardStopState::LongLimitReached;
                }
                self.last_state_change_ms = timestamp_ms;
            }
        } else if self.current_position <= -(self.config.max_short as i64) {
            if self.state != HardStopState::ShortLimitReached 
                && self.state != HardStopState::Halted {
                self.breach_count += 1;
                
                if self.breach_count >= self.max_breaches {
                    self.state = HardStopState::Halted;
                } else {
                    self.state = HardStopState::ShortLimitReached;
                }
                self.last_state_change_ms = timestamp_ms;
            }
        } else if self.state == HardStopState::Cooldown {
            // Check if cooldown period has elapsed
            if timestamp_ms >= self.last_state_change_ms + self.config.cooldown_ms {
                self.state = HardStopState::Active;
                self.breach_count = self.breach_count.saturating_sub(1);
            }
        } else if self.state != HardStopState::Halted {
            // Back to active if within limits
            if self.state != HardStopState::Active {
                self.state = HardStopState::Active;
            }
        }
        
        // Calculate utilization and size factor
        let utilization = self.calculate_utilization();
        let size_factor = self.calculate_size_factor(utilization);
        
        // Determine quote permissions based on state
        let (can_bid, can_ask) = match self.state {
            HardStopState::Active => (true, true),
            HardStopState::LongLimitReached => (false, true),
            HardStopState::ShortLimitReached => (true, false),
            HardStopState::Halted => (false, false),
            HardStopState::Cooldown => (false, false),
        };
        
        // If state changed, record timestamp
        if self.state != prev_state {
            self.last_state_change_ms = timestamp_ms;
        }
        
        QuotePermission::new(can_bid, can_ask, self.state, utilization, size_factor)
    }
    
    /// Calculate position utilization ratio
    #[inline(always)]
    fn calculate_utilization(&self) -> f64 {
        if self.current_position > 0 {
            (self.current_position as f64 / self.config.max_long as f64).min(1.5)
        } else if self.current_position < 0 {
            (self.current_position.unsigned_abs() as f64 / self.config.max_short as f64).min(1.5)
        } else {
            0.0
        }
    }
    
    /// Calculate size reduction factor based on utilization
    #[inline(always)]
    fn calculate_size_factor(&self, utilization: f64) -> f64 {
        if utilization >= 1.0 {
            return 0.0;
        }
        
        if utilization < self.config.gradual_reduction_start {
            return 1.0;
        }
        
        // Linear reduction from gradual_reduction_start to 1.0
        let range = 1.0 - self.config.gradual_reduction_start;
        if range < 1e-15 {
            return 0.0;
        }
        
        let excess = utilization - self.config.gradual_reduction_start;
        1.0 - (excess / range).min(1.0)
    }
    
    /// Get current state
    #[inline(always)]
    pub const fn state(&self) -> HardStopState {
        self.state
    }
    
    /// Get current position
    #[inline(always)]
    pub const fn current_position(&self) -> i64 {
        self.current_position
    }
    
    /// Check if quoting is allowed on a specific side
    #[inline(always)]
    pub fn can_quote_bid(&self) -> bool {
        matches!(self.state, HardStopState::Active | HardStopState::ShortLimitReached)
    }
    
    #[inline(always)]
    pub fn can_quote_ask(&self) -> bool {
        matches!(self.state, HardStopState::Active | HardStopState::LongLimitReached)
    }
    
    /// Reset breach counter (called after successful risk management period)
    pub fn reset_breaches(&mut self) {
        self.breach_count = 0;
    }
    
    /// Force halt (emergency stop)
    pub fn force_halt(&mut self, timestamp_ms: u64) {
        self.state = HardStopState::Halted;
        self.last_state_change_ms = timestamp_ms;
        self.breach_count = self.max_breaches;
    }
    
    /// Attempt to resume from halt
    pub fn try_resume(&mut self, timestamp_ms: u64) -> Result<(), HardStopError> {
        if self.state != HardStopState::Halted {
            return Ok(());
        }
        
        if timestamp_ms >= self.last_state_change_ms + self.config.cooldown_ms * 2 {
            self.state = HardStopState::Cooldown;
            self.last_state_change_ms = timestamp_ms;
            Ok(())
        } else {
            Err(HardStopError::QuoteHalted)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_normal_operation() {
        let config = HardStopConfig::new(100, 100, 1000).unwrap();
        let mut fsm = HardStopStateMachine::new(config);
        
        let perm = fsm.update_position(50, 0);
        
        assert_eq!(perm.state, HardStopState::Active);
        assert!(perm.can_bid);
        assert!(perm.can_ask);
        assert!(perm.size_factor > 0.5);
    }
    
    #[test]
    fn test_long_limit_breach() {
        let config = HardStopConfig::new(100, 100, 1000).unwrap();
        let mut fsm = HardStopStateMachine::new(config);
        
        let perm = fsm.update_position(100, 0);
        
        assert_eq!(perm.state, HardStopState::LongLimitReached);
        assert!(!perm.can_bid);
        assert!(perm.can_ask);
    }
    
    #[test]
    fn test_short_limit_breach() {
        let config = HardStopConfig::new(100, 100, 1000).unwrap();
        let mut fsm = HardStopStateMachine::new(config);
        
        let perm = fsm.update_position(-100, 0);
        
        assert_eq!(perm.state, HardStopState::ShortLimitReached);
        assert!(perm.can_bid);
        assert!(!perm.can_ask);
    }
    
    #[test]
    fn test_size_reduction() {
        let config = HardStopConfig::new(100, 100, 1000).unwrap();
        let mut fsm = HardStopStateMachine::new(config);
        
        // At 95% utilization, should start reducing size
        let perm = fsm.update_position(95, 0);
        assert!(perm.size_factor < 1.0);
        assert!(perm.size_factor > 0.0);
    }
}
