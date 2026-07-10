// STAGE 24: EPISTEMIC HUMILITY MODE
// ====================================
//!
//! Implements epistemic humility mode - a safe operating state
//! triggered when ontological uncertainty exceeds acceptable bounds.
//!
//! Features:
//! - Automatic position size reduction
//! - Increased cash reserves
//! - Human operator confirmation requirement
//! - Conservative risk parameters

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Configuration for epistemic humility mode
#[derive(Debug, Clone)]
pub struct EpistemicHumilityConfig {
    /// Maximum position size as fraction of portfolio (default: 10%)
    pub max_position_size: f64,
    /// Minimum cash reserve as fraction of portfolio (default: 50%)
    pub min_cash_reserve: f64,
    /// Require human confirmation for new positions
    pub require_human_confirmation: bool,
    /// Maximum leverage allowed (default: 1.0 = no leverage)
    pub max_leverage: f64,
    /// Cooldown period before exiting humility mode
    pub cooldown_duration: Duration,
    /// Gradual re-entry factor (how quickly to return to normal)
    pub reentry_factor: f64,
}

impl Default for EpistemicHumilityConfig {
    fn default() -> Self {
        Self {
            max_position_size: 0.1,
            min_cash_reserve: 0.5,
            require_human_confirmation: true,
            max_leverage: 1.0,
            cooldown_duration: Duration::from_secs(3600), // 1 hour
            reentry_factor: 0.1,
        }
    }
}

/// Current state of epistemic humility mode
#[derive(Debug, Clone, PartialEq)]
pub enum HumilityState {
    Inactive,
    Entering,
    Active,
    Exiting,
}

/// Status report from epistemic humility mode
#[derive(Debug, Clone)]
pub struct HumilityStatus {
    pub state: HumilityState,
    pub activated_at: Option<Instant>,
    pub reason: String,
    pub current_position_limit: f64,
    pub current_cash_target: f64,
    pub pending_confirmations: u64,
    pub time_in_mode: Duration,
}

/// Epistemic Humility Mode controller
pub struct EpistemicHumilityMode {
    config: EpistemicHumilityConfig,
    state: HumilityState,
    is_active: AtomicBool,
    activation_time: AtomicU64,
    activation_reason: String,
    pending_confirmations: AtomicU64,
    entry_triggers: Vec<String>,
    last_status_check: Instant,
}

impl EpistemicHumilityMode {
    /// Create a new epistemic humility mode controller
    pub fn new(config: EpistemicHumilityConfig) -> Self {
        Self {
            config,
            state: HumilityState::Inactive,
            is_active: AtomicBool::new(false),
            activation_time: AtomicU64::new(0),
            activation_reason: String::new(),
            pending_confirmations: AtomicU64::new(0),
            entry_triggers: Vec::new(),
            last_status_check: Instant::now(),
        }
    }

    /// Activate epistemic humility mode
    pub fn activate(&mut self, reason: &str) {
        if self.is_active.load(Ordering::SeqCst) {
            return; // Already active
        }

        self.state = HumilityState::Entering;
        self.is_active.store(true, Ordering::SeqCst);
        self.activation_time.store(
            Instant::now().duration_since(Instant::now()).as_secs(),
            Ordering::SeqCst,
        );
        self.activation_reason = reason.to_string();
        self.entry_triggers.push(reason.to_string());

        self.state = HumilityState::Active;

        log::warn!(
            "EPISTEMIC HUMILITY ACTIVATED: {} at {:?}",
            reason,
            Instant::now()
        );
    }

    /// Deactivate epistemic humility mode (after cooldown)
    pub fn deactivate(&mut self) -> bool {
        if !self.is_active.load(Ordering::SeqCst) {
            return false;
        }

        let activation_secs = self.activation_time.load(Ordering::SeqCst);
        let elapsed = Duration::from_secs(activation_secs);

        if elapsed < self.config.cooldown_duration {
            self.state = HumilityState::Exiting;
            return false; // Still in cooldown
        }

        self.state = HumilityState::Exiting;
        self.is_active.store(false, Ordering::SeqCst);
        self.state = HumilityState::Inactive;

        log::info!("EPISTEMIC HUMILITY DEACTIVATED after {:?}", elapsed);

        true
    }

    /// Check if a proposed trade is allowed under humility mode
    pub fn is_trade_allowed(&self, position_size: f64, leverage: f64) -> TradeApproval {
        if !self.is_active.load(Ordering::SeqCst) {
            return TradeApproval::Allowed;
        }

        // Check position size limit
        if position_size > self.config.max_position_size {
            return TradeApproval::Rejected(Format::PositionSizeExceeded(
                position_size,
                self.config.max_position_size,
            ));
        }

        // Check leverage limit
        if leverage > self.config.max_leverage {
            return TradeApproval::Rejected(Format::LeverageExceeded(
                leverage,
                self.config.max_leverage,
            ));
        }

        // Check if human confirmation required
        if self.config.require_human_confirmation {
            self.pending_confirmations.fetch_add(1, Ordering::SeqCst);
            return TradeApproval::RequiresConfirmation;
        }

        TradeApproval::Allowed
    }

    /// Confirm a pending trade (human operator approval)
    pub fn confirm_trade(&self) -> bool {
        let pending = self.pending_confirmations.load(Ordering::SeqCst);
        if pending > 0 {
            self.pending_confirmations.fetch_sub(1, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    /// Get current status of humility mode
    pub fn get_status(&self) -> HumilityStatus {
        let activation_secs = self.activation_time.load(Ordering::SeqCst);
        let activation_time = if activation_secs > 0 {
            Some(Instant::now()) // Simplified
        } else {
            None
        };

        let time_in_mode = if self.is_active.load(Ordering::SeqCst) {
            Duration::from_secs(activation_secs)
        } else {
            Duration::from_secs(0)
        };

        HumilityStatus {
            state: self.state.clone(),
            activated_at: activation_time,
            reason: self.activation_reason.clone(),
            current_position_limit: if self.is_active.load(Ordering::SeqCst) {
                self.config.max_position_size
            } else {
                1.0
            },
            current_cash_target: if self.is_active.load(Ordering::SeqCst) {
                self.config.min_cash_reserve
            } else {
                0.0
            },
            pending_confirmations: self.pending_confirmations.load(Ordering::SeqCst),
            time_in_mode,
        }
    }

    /// Get the current maximum position size allowed
    pub fn current_position_limit(&self) -> f64 {
        if self.is_active.load(Ordering::SeqCst) {
            self.config.max_position_size
        } else {
            1.0
        }
    }

    /// Get the current minimum cash reserve required
    pub fn current_cash_target(&self) -> f64 {
        if self.is_active.load(Ordering::SeqCst) {
            self.config.min_cash_reserve
        } else {
            0.0
        }
    }

    /// Check if currently active
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::SeqCst)
    }

    /// Get list of triggers that activated humility mode
    pub fn get_entry_triggers(&self) -> &[String] {
        &self.entry_triggers
    }

    /// Reset the controller state
    pub fn reset(&mut self) {
        self.state = HumilityState::Inactive;
        self.is_active.store(false, Ordering::SeqCst);
        self.activation_time.store(0, Ordering::SeqCst);
        self.activation_reason.clear();
        self.pending_confirmations.store(0, Ordering::SeqCst);
        self.entry_triggers.clear();
    }
}

/// Result of a trade approval check
#[derive(Debug, Clone, PartialEq)]
pub enum TradeApproval {
    Allowed,
    RequiresConfirmation,
    Rejected(Format),
}

/// Format for rejection reasons
#[derive(Debug, Clone)]
pub enum Format {
    PositionSizeExceeded(f64, f64), // actual, limit
    LeverageExceeded(f64, f64),     // actual, limit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_humility_activation() {
        let config = EpistemicHumilityConfig::default();
        let mut mode = EpistemicHumilityMode::new(config);

        assert!(!mode.is_active());

        mode.activate("KL divergence spike detected");

        assert!(mode.is_active());
        assert_eq!(mode.get_status().state, HumilityState::Active);
    }

    #[test]
    fn test_trade_approval_under_humility() {
        let config = EpistemicHumilityConfig {
            max_position_size: 0.1,
            require_human_confirmation: true,
            ..Default::default()
        };
        let mut mode = EpistemicHumilityMode::new(config);

        // Test without humility
        let result = mode.is_trade_allowed(0.5, 1.0);
        assert_eq!(result, TradeApproval::Allowed);

        // Activate humility
        mode.activate("Test activation");

        // Test with excessive position
        let result = mode.is_trade_allowed(0.5, 1.0);
        assert!(matches!(result, TradeApproval::Rejected(_)));

        // Test with acceptable position but requires confirmation
        let result = mode.is_trade_allowed(0.05, 1.0);
        assert_eq!(result, TradeApproval::RequiresConfirmation);
    }
}
