// STAGE 25: CHAPTER 2 - LIQUIDITY EVAPORATOR
// Instantly removes 99% of limit orders from L2/L3 order book
// Tests Stage 15 Market Maker adverse selection survival

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Liquidity level representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityLevel {
    pub price: u64,      // Fixed point (price * 1e6)
    pub size: u64,       // Base units
    pub order_count: u32,
}

/// Order book side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Bid,
    Ask,
}

/// Liquidity evaporation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaporationConfig {
    pub evaporation_rate: f64,    // 0.99 for 99% removal
    pub target_sides: Vec<Side>,
    pub min_levels_preserved: usize, // Minimum levels to keep
    pub chaos_mode_flag: bool,
}

/// Evaporation state
pub struct EvaporationState {
    pub active: AtomicBool,
    pub evaporated_volume: AtomicU64,
    pub total_volume_before: AtomicU64,
    pub evaporation_time_ns: AtomicU64,
}

impl Default for EvaporationState {
    fn default() -> Self {
        Self {
            active: AtomicBool::new(false),
            evaporated_volume: AtomicU64::new(0),
            total_volume_before: AtomicU64::new(0),
            evaporation_time_ns: AtomicU64::new(0),
        }
    }
}

/// Liquidity Evaporator
/// Simulates sudden liquidity withdrawal from order book
pub struct LiquidityEvaporator {
    state: std::sync::Arc<EvaporationState>,
    config: EvaporationConfig,
    original_book: std::sync::Mutex<HashMap<Side, Vec<LiquidityLevel>>>,
    current_book: std::sync::Mutex<HashMap<Side, Vec<LiquidityLevel>>>,
    chaos_mode_flag: AtomicBool,
}

impl LiquidityEvaporator {
    pub fn new(config: EvaporationConfig) -> Self {
        Self {
            state: std::sync::Arc::new(EvaporationState::default()),
            config,
            original_book: std::sync::Mutex::new(HashMap::new()),
            current_book: std::sync::Mutex::new(HashMap::new()),
            chaos_mode_flag: AtomicBool::new(false),
        }
    }

    /// Activate chaos mode
    pub fn activate_chaos_mode(&self) {
        self.chaos_mode_flag.store(true, Ordering::SeqCst);
    }

    /// Deactivate chaos mode
    pub fn deactivate_chaos_mode(&self) {
        self.chaos_mode_flag.store(false, Ordering::SeqCst);
    }

    /// Check if chaos mode is active
    pub fn is_chaos_mode_active(&self) -> bool {
        self.chaos_mode_flag.load(Ordering::SeqCst)
    }

    /// Load initial order book state
    pub fn load_order_book(&self, bids: Vec<LiquidityLevel>, asks: Vec<LiquidityLevel>) {
        let mut book = self.original_book.lock().unwrap();
        book.insert(Side::Bid, bids);
        book.insert(Side::Ask, asks);

        // Also set as current book
        let mut current = self.current_book.lock().unwrap();
        current.clone_from(&book);

        // Calculate total volume
        let total_vol: u64 = book.values()
            .flatten()
            .map(|l| l.size)
            .sum();
        self.state.total_volume_before.store(total_vol, Ordering::Relaxed);
    }

    /// Execute liquidity evaporation
    /// Returns the amount of volume evaporated
    pub fn evaporate_liquidity(&self) -> Result<u64, EvaporationError> {
        if !self.chaos_mode_flag.load(Ordering::SeqCst) {
            return Err(EvaporationError::ChaosModeNotActive);
        }

        let now = std::time::Instant::now();
        let now_ns = now.duration_since(std::time::Instant::EPOCH).as_nanos() as u64;

        let mut current_book = self.current_book.lock().unwrap();
        let original_book = self.original_book.lock().unwrap();

        let mut total_evaporated = 0u64;

        for (&side, levels) in original_book.iter() {
            if !self.config.target_sides.contains(&side) {
                continue;
            }

            // Preserve minimum levels
            let preserve_count = self.config.min_levels_preserved.min(levels.len());
            let evaporate_from = levels.len() - preserve_count;

            if evaporate_from == 0 {
                continue;
            }

            // Calculate volume to evaporate
            let mut preserved_levels = Vec::new();
            let mut evaporated_vol = 0u64;

            for (i, level) in levels.iter().enumerate() {
                if i < preserve_count {
                    preserved_levels.push(level.clone());
                } else {
                    // Apply evaporation rate
                    let keep_ratio = 1.0 - self.config.evaporation_rate;
                    let keep_size = (level.size as f64 * keep_ratio) as u64;
                    
                    if keep_size > 0 {
                        let mut new_level = level.clone();
                        new_level.size = keep_size;
                        preserved_levels.push(new_level);
                    }
                    
                    evaporated_vol += level.size - keep_size;
                }
            }

            current_book.insert(side, preserved_levels);
            total_evaporated += evaporated_vol;
        }

        self.state.evaporated_volume.store(total_evaporated, Ordering::Relaxed);
        self.state.evaporation_time_ns.store(now_ns, Ordering::Relaxed);
        self.state.active.store(true, Ordering::SeqCst);

        Ok(total_evaporated)
    }

    /// Get current order book after evaporation
    pub fn get_current_book(&self) -> HashMap<Side, Vec<LiquidityLevel>> {
        self.current_book.lock().unwrap().clone()
    }

    /// Restore original order book
    pub fn restore_liquidity(&self) {
        let original = self.original_book.lock().unwrap();
        let mut current = self.current_book.lock().unwrap();
        current.clone_from(&original);
        
        self.state.active.store(false, Ordering::Relaxed);
        self.state.evaporated_volume.store(0, Ordering::Relaxed);
    }

    /// Get evaporation statistics
    pub fn get_stats(&self) -> EvaporationStats {
        let before = self.state.total_volume_before.load(Ordering::Relaxed);
        let evaporated = self.state.evaporated_volume.load(Ordering::Relaxed);
        
        let remaining = self.current_book.lock().unwrap()
            .values()
            .flatten()
            .map(|l| l.size)
            .sum::<u64>();

        EvaporationStats {
            total_volume_before: before,
            evaporated_volume: evaporated,
            remaining_volume: remaining,
            evaporation_percentage: if before > 0 { 
                evaporated as f64 / before as f64 
            } else { 
                0.0 
            },
            is_active: self.state.active.load(Ordering::Relaxed),
            chaos_mode: self.chaos_mode_flag.load(Ordering::SeqCst),
        }
    }

    /// Check if evaporation is currently active
    pub fn is_active(&self) -> bool {
        self.state.active.load(Ordering::Relaxed)
    }
}

/// Evaporation statistics
#[derive(Debug, Clone)]
pub struct EvaporationStats {
    pub total_volume_before: u64,
    pub evaporated_volume: u64,
    pub remaining_volume: u64,
    pub evaporation_percentage: f64,
    pub is_active: bool,
    pub chaos_mode: bool,
}

/// Evaporation errors
#[derive(Debug, Clone, PartialEq)]
pub enum EvaporationError {
    ChaosModeNotActive,
    EmptyOrderBook,
    InvalidConfiguration,
}

impl std::fmt::Display for EvaporationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvaporationError::ChaosModeNotActive => write!(f, "Chaos mode not active"),
            EvaporationError::EmptyOrderBook => write!(f, "Empty order book"),
            EvaporationError::InvalidConfiguration => write!(f, "Invalid configuration"),
        }
    }
}

impl std::error::Error for EvaporationError {}

/// Builder for evaporation configurations
pub struct EvaporationConfigBuilder {
    evaporation_rate: f64,
    target_sides: Vec<Side>,
    min_levels_preserved: usize,
}

impl EvaporationConfigBuilder {
    pub fn new() -> Self {
        Self {
            evaporation_rate: 0.99, // 99% evaporation
            target_sides: vec![Side::Bid, Side::Ask],
            min_levels_preserved: 1,
        }
    }

    pub fn evaporation_rate(mut self, rate: f64) -> Self {
        self.evaporation_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn target_side(mut self, side: Side) -> Self {
        self.target_sides.push(side);
        self
    }

    pub fn min_levels_preserved(mut self, count: usize) -> Self {
        self.min_levels_preserved = count;
        self
    }

    pub fn build(self) -> EvaporationConfig {
        EvaporationConfig {
            evaporation_rate: self.evaporation_rate,
            target_sides: self.target_sides,
            min_levels_preserved: self.min_levels_preserved,
            chaos_mode_flag: false,
        }
    }
}

impl Default for EvaporationConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaporation_without_chaos_mode() {
        let config = EvaporationConfigBuilder::new()
            .evaporation_rate(0.99)
            .build();

        let evaporator = LiquidityEvaporator::new(config);

        // Load some test liquidity
        let bids = vec![
            LiquidityLevel { price: 49000_000_000, size: 1000, order_count: 10 },
            LiquidityLevel { price: 48000_000_000, size: 2000, order_count: 20 },
        ];
        let asks = vec![
            LiquidityLevel { price: 51000_000_000, size: 1000, order_count: 10 },
        ];
        evaporator.load_order_book(bids, asks);

        // Should fail without chaos mode
        let result = evaporator.evaporate_liquidity();
        assert!(matches!(result, Err(EvaporationError::ChaosModeNotActive)));
    }

    #[test]
    fn test_evaporation_with_chaos_mode() {
        let config = EvaporationConfigBuilder::new()
            .evaporation_rate(0.99)
            .min_levels_preserved(1)
            .build();

        let evaporator = LiquidityEvaporator::new(config);
        evaporator.activate_chaos_mode();

        // Load test liquidity
        let bids = vec![
            LiquidityLevel { price: 49000_000_000, size: 1000, order_count: 10 },
            LiquidityLevel { price: 48000_000_000, size: 2000, order_count: 20 },
            LiquidityLevel { price: 47000_000_000, size: 3000, order_count: 30 },
        ];
        let asks = vec![
            LiquidityLevel { price: 51000_000_000, size: 1000, order_count: 10 },
        ];
        evaporator.load_order_book(bids, asks);

        // Execute evaporation
        let evaporated = evaporator.evaporate_liquidity();
        assert!(evaporated.is_ok());

        let stats = evaporator.get_stats();
        assert!(stats.is_active);
        assert!(stats.evaporation_percentage > 0.9);
    }

    #[test]
    fn test_liquidity_restoration() {
        let config = EvaporationConfigBuilder::new()
            .build();

        let evaporator = LiquidityEvaporator::new(config);
        evaporator.activate_chaos_mode();

        let bids = vec![
            LiquidityLevel { price: 49000_000_000, size: 1000, order_count: 10 },
        ];
        let asks = vec![];
        evaporator.load_order_book(bids, asks);

        let _ = evaporator.evaporate_liquidity();
        evaporator.restore_liquidity();

        let stats = evaporator.get_stats();
        assert!(!stats.is_active);
        assert_eq!(stats.evaporated_volume, 0);
    }
}
