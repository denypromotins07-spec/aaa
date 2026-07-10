// STAGE 25: CHAPTER 2 - FLASH CRASH INJECTOR
// Injects synthetic flash crashes into L2/L3 order book
// Tests Stage 5 Kill-Switch and Stage 19 Safe RL under extreme conditions

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Flash crash configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlashCrashConfig {
    pub drop_percentage: f64,      // e.g., 0.30 for 30% drop
    pub duration_ms: u64,          // Duration of the crash
    pub recovery_time_ms: u64,     // Time to recover
    pub target_symbols: Vec<String>,
    pub trigger_volume_multiplier: f64,
}

/// Flash crash phases
#[derive(Debug, Clone, PartialEq)]
pub enum CrashPhase {
    Normal,
    Initiation,      // Starting the crash
    Freefall,        // Rapid price decline
    Bottom,          // At lowest point
    Recovery,        // Price recovering
    Stabilized,      // Back to normal
}

/// Flash crash state
pub struct FlashCrashState {
    pub active: AtomicBool,
    pub current_phase: std::sync::Mutex<CrashPhase>,
    pub start_time_ns: AtomicU64,
    pub initial_price: AtomicU64, // Fixed point representation (price * 1e6)
    pub current_price: AtomicU64,
    pub chaos_mode_flag: AtomicBool, // CRITICAL: Prevents false-positive kill-switch
}

impl Default for FlashCrashState {
    fn default() -> Self {
        Self {
            active: AtomicBool::new(false),
            current_phase: std::sync::Mutex::new(CrashPhase::Normal),
            start_time_ns: AtomicU64::new(0),
            initial_price: AtomicU64::new(0),
            current_price: AtomicU64::new(0),
            chaos_mode_flag: AtomicBool::new(false),
        }
    }
}

/// Synthetic Flash Crash Injector
/// Hooks into Stage 2 SPSC Ring Buffer to inject market crashes
pub struct FlashCrashInjector {
    state: std::sync::Arc<FlashCrashState>,
    config: FlashCrashConfig,
}

impl FlashCrashInjector {
    pub fn new(config: FlashCrashConfig) -> Self {
        Self {
            state: std::sync::Arc::new(FlashCrashState::default()),
            config,
        }
    }

    /// Activate chaos mode - CRITICAL for preventing kill-switch false positives
    pub fn activate_chaos_mode(&self) {
        self.state.chaos_mode_flag.store(true, Ordering::SeqCst);
    }

    /// Deactivate chaos mode
    pub fn deactivate_chaos_mode(&self) {
        self.state.chaos_mode_flag.store(false, Ordering::SeqCst);
    }

    /// Check if chaos mode is active (used by Stage 5 Kill-Switch)
    pub fn is_chaos_mode_active(&self) -> bool {
        self.state.chaos_mode_flag.load(Ordering::SeqCst)
    }

    /// Initialize flash crash with a specific price
    pub fn initialize_crash(&self, initial_price_u64: u64) -> Result<(), CrashError> {
        if !self.state.chaos_mode_flag.load(Ordering::SeqCst) {
            return Err(CrashError::ChaosModeNotActive);
        }

        let now = Instant::now();
        let now_ns = now.duration_since(Instant::EPOCH).as_nanos() as u64;

        self.state.initial_price.store(initial_price_u64, Ordering::Relaxed);
        self.state.current_price.store(initial_price_u64, Ordering::Relaxed);
        self.state.start_time_ns.store(now_ns, Ordering::Relaxed);
        self.state.active.store(true, Ordering::SeqCst);

        *self.state.current_phase.lock().unwrap() = CrashPhase::Initiation;

        Ok(())
    }

    /// Get current simulated price after applying crash dynamics
    /// Returns price in fixed-point format (price * 1e6)
    pub fn get_simulated_price(&self) -> u64 {
        if !self.state.active.load(Ordering::Relaxed) {
            return self.state.current_price.load(Ordering::Relaxed);
        }

        let now = Instant::now();
        let now_ns = now.duration_since(Instant::EPOCH).as_nanos() as u64;
        let elapsed_ns = now_ns.saturating_sub(self.state.start_time_ns.load(Ordering::Relaxed));
        let elapsed_ms = elapsed_ns / 1_000_000;

        let initial = self.state.initial_price.load(Ordering::Relaxed) as f64;
        let drop_pct = self.config.drop_percentage;
        
        let new_price = if elapsed_ms < self.config.duration_ms / 2 {
            // Freefall phase - exponential decay
            let progress = elapsed_ms as f64 / (self.config.duration_ms / 2) as f64;
            let price = initial * (1.0 - drop_pct * progress);
            *self.state.current_phase.lock().unwrap() = CrashPhase::Freefall;
            price
        } else if elapsed_ms < self.config.duration_ms {
            // Bottom phase
            let price = initial * (1.0 - drop_pct);
            *self.state.current_phase.lock().unwrap() = CrashPhase::Bottom;
            price
        } else if elapsed_ms < self.config.duration_ms + self.config.recovery_time_ms {
            // Recovery phase
            let recovery_progress = (elapsed_ms - self.config.duration_ms) as f64 
                / self.config.recovery_time_ms as f64;
            let bottom_price = initial * (1.0 - drop_pct);
            let price = bottom_price + (initial - bottom_price) * recovery_progress;
            *self.state.current_phase.lock().unwrap() = CrashPhase::Recovery;
            price
        } else {
            // Stabilized
            *self.state.current_phase.lock().unwrap() = CrashPhase::Stabilized;
            self.state.active.store(false, Ordering::Relaxed);
            initial
        };

        let price_u64 = new_price as u64;
        self.state.current_price.store(price_u64, Ordering::Relaxed);
        price_u64
    }

    /// Get current crash phase
    pub fn get_current_phase(&self) -> CrashPhase {
        self.state.current_phase.lock().unwrap().clone()
    }

    /// Check if crash is currently active
    pub fn is_crash_active(&self) -> bool {
        self.state.active.load(Ordering::Relaxed)
    }

    /// Get crash statistics
    pub fn get_crash_stats(&self) -> CrashStats {
        let initial = self.state.initial_price.load(Ordering::Relaxed);
        let current = self.state.current_price.load(Ordering::Relaxed);
        
        let max_drop = if initial > 0 {
            (initial as f64 - current as f64) / initial as f64
        } else {
            0.0
        };

        CrashStats {
            initial_price: initial,
            current_price: current,
            max_drop_percentage: max_drop,
            phase: self.get_current_phase(),
            is_active: self.is_crash_active(),
            chaos_mode: self.state.chaos_mode_flag.load(Ordering::SeqCst),
        }
    }

    /// Reset crash state
    pub fn reset(&self) {
        self.state.active.store(false, Ordering::Relaxed);
        *self.state.current_phase.lock().unwrap() = CrashPhase::Normal;
        self.state.start_time_ns.store(0, Ordering::Relaxed);
    }
}

/// Crash statistics
#[derive(Debug, Clone)]
pub struct CrashStats {
    pub initial_price: u64,
    pub current_price: u64,
    pub max_drop_percentage: f64,
    pub phase: CrashPhase,
    pub is_active: bool,
    pub chaos_mode: bool,
}

/// Crash errors
#[derive(Debug, Clone, PartialEq)]
pub enum CrashError {
    ChaosModeNotActive,
    InvalidConfiguration,
    PriceOverflow,
}

impl std::fmt::Display for CrashError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CrashError::ChaosModeNotActive => write!(f, "Chaos mode not active"),
            CrashError::InvalidConfiguration => write!(f, "Invalid configuration"),
            CrashError::PriceOverflow => write!(f, "Price overflow"),
        }
    }
}

impl std::error::Error for CrashError {}

/// Builder for flash crash configurations
pub struct FlashCrashConfigBuilder {
    drop_percentage: f64,
    duration_ms: u64,
    recovery_time_ms: u64,
    target_symbols: Vec<String>,
    trigger_volume_multiplier: f64,
}

impl FlashCrashConfigBuilder {
    pub fn new() -> Self {
        Self {
            drop_percentage: 0.30, // 30% drop
            duration_ms: 100,       // 100ms freefall
            recovery_time_ms: 5000, // 5 second recovery
            target_symbols: vec!["BTC-PERP".to_string()],
            trigger_volume_multiplier: 10.0,
        }
    }

    pub fn drop_percentage(mut self, pct: f64) -> Self {
        self.drop_percentage = pct.clamp(0.0, 1.0);
        self
    }

    pub fn duration(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }

    pub fn recovery_time(mut self, ms: u64) -> Self {
        self.recovery_time_ms = ms;
        self
    }

    pub fn target_symbol(mut self, symbol: &str) -> Self {
        self.target_symbols.push(symbol.to_string());
        self
    }

    pub fn volume_multiplier(mut self, mult: f64) -> Self {
        self.trigger_volume_multiplier = mult;
        self
    }

    pub fn build(self) -> FlashCrashConfig {
        FlashCrashConfig {
            drop_percentage: self.drop_percentage,
            duration_ms: self.duration_ms,
            recovery_time_ms: self.recovery_time_ms,
            target_symbols: self.target_symbols,
            trigger_volume_multiplier: self.trigger_volume_multiplier,
        }
    }
}

impl Default for FlashCrashConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_crash_without_chaos_mode() {
        let config = FlashCrashConfigBuilder::new()
            .drop_percentage(0.30)
            .build();

        let injector = FlashCrashInjector::new(config);

        // Should fail without chaos mode
        let result = injector.initialize_crash(50000_000_000); // $50,000 in fixed point
        assert!(matches!(result, Err(CrashError::ChaosModeNotActive)));
    }

    #[test]
    fn test_flash_crash_dynamics() {
        let config = FlashCrashConfigBuilder::new()
            .drop_percentage(0.30)
            .duration_ms(100)
            .recovery_time_ms(500)
            .build();

        let injector = FlashCrashInjector::new(config);
        injector.activate_chaos_mode();

        let initial_price = 50000_000_000u64; // $50,000
        let _ = injector.initialize_crash(initial_price);

        // Initial price should match
        let price = injector.get_simulated_price();
        assert!(price > 0);

        // Verify crash becomes active
        assert!(injector.is_crash_active());
    }

    #[test]
    fn test_crash_stats() {
        let config = FlashCrashConfigBuilder::new()
            .drop_percentage(0.20)
            .build();

        let injector = FlashCrashInjector::new(config);
        injector.activate_chaos_mode();

        let stats = injector.get_crash_stats();
        assert!(!stats.is_active);
        assert!(stats.chaos_mode);
        assert_eq!(stats.phase, CrashPhase::Normal);
    }
}
