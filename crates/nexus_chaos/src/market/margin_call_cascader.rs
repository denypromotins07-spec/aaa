// STAGE 25: CHAPTER 2 - MARGIN CALL CASCADER
// Simulates Binance/Deribit liquidation engine logic
// Forces Stage 5 Risk Engine and Stage 19 Safe RL to execute deleveraging

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Position representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub size: i64,         // Positive = long, negative = short
    pub entry_price: u64,  // Fixed point (price * 1e6)
    pub leverage: u32,
    pub margin: u64,
    pub unrealized_pnl: i64,
}

/// Liquidation event
#[derive(Debug, Clone)]
pub struct LiquidationEvent {
    pub symbol: String,
    pub position_size: i64,
    pub liquidation_price: u64,
    pub pnl_at_liquidation: i64,
    pub timestamp_ns: u64,
}

/// Margin call configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginCallConfig {
    pub initial_margin_ratio: f64,    // e.g., 0.10 for 10x leverage
    pub maintenance_margin_ratio: f64, // e.g., 0.05
    pub liquidation_penalty: f64,      // Fee charged on liquidation
    pub chaos_mode_flag: bool,
}

/// Margin state tracking
pub struct MarginState {
    pub total_equity: AtomicU64,
    pub total_margin_used: AtomicU64,
    pub margin_ratio_bps: AtomicU64, // Basis points (ratio * 10000)
    pub liquidation_count: AtomicU64,
    pub is_liquidating: AtomicBool,
}

impl Default for MarginState {
    fn default() -> Self {
        Self {
            total_equity: AtomicU64::new(1_000_000_000), // $1M default
            total_margin_used: AtomicU64::new(0),
            margin_ratio_bps: AtomicU64::new(10000), // 100% default
            liquidation_count: AtomicU64::new(0),
            is_liquidating: AtomicBool::new(false),
        }
    }
}

/// Margin Call Cascader
/// Simulates exchange liquidation cascades
pub struct MarginCallCascader {
    state: std::sync::Arc<MarginState>,
    config: MarginCallConfig,
    positions: std::sync::Mutex<HashMap<String, Position>>,
    liquidation_events: std::sync::Mutex<Vec<LiquidationEvent>>,
    chaos_mode_flag: AtomicBool,
}

impl MarginCallCascader {
    pub fn new(config: MarginCallConfig) -> Self {
        Self {
            state: std::sync::Arc::new(MarginState::default()),
            config,
            positions: std::sync::Mutex::new(HashMap::new()),
            liquidation_events: std::sync::Mutex::new(Vec::new()),
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

    /// Add a position to the margin engine
    pub fn add_position(&self, position: Position) -> Result<(), MarginError> {
        let mut positions = self.positions.lock().unwrap();
        
        // Update margin used
        let current_margin = self.state.total_margin_used.load(Ordering::Relaxed);
        self.state.total_margin_used.store(
            current_margin + position.margin,
            Ordering::Relaxed,
        );

        positions.insert(position.symbol.clone(), position);
        self.update_margin_ratio();

        Ok(())
    }

    /// Update position PnL based on current market price
    pub fn update_position_pnl(&self, symbol: &str, current_price: u64) -> Result<i64, MarginError> {
        let mut positions = self.positions.lock().unwrap();
        
        let position = positions.get_mut(symbol)
            .ok_or(MarginError::PositionNotFound(symbol.to_string()))?;

        let price_diff = if position.size > 0 {
            // Long position
            current_price as i64 - position.entry_price as i64
        } else {
            // Short position
            position.entry_price as i64 - current_price as i64
        };

        // Calculate PnL: size * price_diff / entry_price (simplified)
        let pnl = (position.size.abs() as i64 * price_diff) / position.entry_price as i64;
        position.unrealized_pnl = pnl;

        // Update total equity
        let total_pnl: i64 = positions.values().map(|p| p.unrealized_pnl).sum();
        let base_equity = self.state.total_equity.load(Ordering::Relaxed) as i64;
        let new_equity = (base_equity + total_pnl).max(0) as u64;
        self.state.total_equity.store(new_equity, Ordering::Relaxed);

        self.update_margin_ratio();

        Ok(pnl)
    }

    /// Check if any positions should be liquidated
    /// Returns list of liquidation events
    pub fn check_liquidations(&self) -> Result<Vec<LiquidationEvent>, MarginError> {
        if !self.chaos_mode_flag.load(Ordering::SeqCst) {
            return Err(MarginError::ChaosModeNotActive);
        }

        let positions = self.positions.lock().unwrap();
        let mut events = Vec::new();

        let equity = self.state.total_equity.load(Ordering::Relaxed);
        let maintenance_ratio = self.config.maintenance_margin_ratio;

        for (symbol, position) in positions.iter() {
            // Calculate current margin ratio for this position
            let position_value = (position.size.abs() as u64 * position.entry_price) / 1_000_000;
            let current_margin_ratio = if position_value > 0 {
                position.margin as f64 / position_value as f64
            } else {
                1.0
            };

            // Check if below maintenance margin
            if current_margin_ratio < maintenance_ratio {
                let now = std::time::Instant::now();
                let now_ns = now.duration_since(std::time::Instant::EPOCH).as_nanos() as u64;

                // Calculate liquidation price
                let liq_price = self.calculate_liquidation_price(position);

                let event = LiquidationEvent {
                    symbol: symbol.clone(),
                    position_size: position.size,
                    liquidation_price: liq_price,
                    pnl_at_liquidation: position.unrealized_pnl,
                    timestamp_ns: now_ns,
                };

                events.push(event);
                self.state.liquidation_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        if !events.is_empty() {
            self.state.is_liquidating.store(true, Ordering::SeqCst);
            let mut event_log = self.liquidation_events.lock().unwrap();
            event_log.extend(events.clone());
        }

        Ok(events)
    }

    /// Calculate liquidation price for a position
    fn calculate_liquidation_price(&self, position: &Position) -> u64 {
        let maintenance_margin = position.margin as f64 * self.config.maintenance_margin_ratio;
        
        if position.size > 0 {
            // Long position: liq_price = entry_price * (1 - margin/position_value)
            let position_value = position.size as f64 * position.entry_price as f64 / 1_000_000.0;
            let liq_price = position.entry_price as f64 * 
                (1.0 - maintenance_margin / position_value);
            liq_price.max(0) as u64
        } else {
            // Short position: liq_price = entry_price * (1 + margin/position_value)
            let position_value = position.size.abs() as f64 * position.entry_price as f64 / 1_000_000.0;
            let liq_price = position.entry_price as f64 * 
                (1.0 + maintenance_margin / position_value);
            liq_price as u64
        }
    }

    /// Update margin ratio calculation
    fn update_margin_ratio(&self) {
        let equity = self.state.total_equity.load(Ordering::Relaxed);
        let margin_used = self.state.total_margin_used.load(Ordering::Relaxed);

        let ratio_bps = if margin_used > 0 {
            ((equity as f64 / margin_used as f64) * 10000.0) as u64
        } else {
            10000
        };

        self.state.margin_ratio_bps.store(ratio_bps, Ordering::Relaxed);
    }

    /// Execute forced deleveraging (iceberg algorithm trigger)
    pub fn execute_deleveraging(&self, target_reduction_pct: f64) -> Result<u64, MarginError> {
        if !self.chaos_mode_flag.load(Ordering::SeqCst) {
            return Err(MarginError::ChaosModeNotActive);
        }

        let positions = self.positions.lock().unwrap();
        let mut total_reduced = 0u64;

        for position in positions.values() {
            let reduction = (position.margin as f64 * target_reduction_pct) as u64;
            total_reduced += reduction;
        }

        // Reduce margin used
        let current_margin = self.state.total_margin_used.load(Ordering::Relaxed);
        let new_margin = current_margin.saturating_sub(total_reduced);
        self.state.total_margin_used.store(new_margin, Ordering::Relaxed);

        self.update_margin_ratio();
        self.state.is_liquidating.store(false, Ordering::SeqCst);

        Ok(total_reduced)
    }

    /// Get margin statistics
    pub fn get_stats(&self) -> MarginStats {
        let positions = self.positions.lock().unwrap();
        let total_pnl: i64 = positions.values().map(|p| p.unrealized_pnl).sum();

        MarginStats {
            total_equity: self.state.total_equity.load(Ordering::Relaxed),
            total_margin_used: self.state.total_margin_used.load(Ordering::Relaxed),
            margin_ratio_bps: self.state.margin_ratio_bps.load(Ordering::Relaxed),
            total_unrealized_pnl: total_pnl,
            liquidation_count: self.state.liquidation_count.load(Ordering::Relaxed),
            is_liquidating: self.state.is_liquidating.load(Ordering::Relaxed),
            position_count: positions.len(),
            chaos_mode: self.chaos_mode_flag.load(Ordering::SeqCst),
        }
    }

    /// Get all liquidation events
    pub fn get_liquidation_events(&self) -> Vec<LiquidationEvent> {
        self.liquidation_events.lock().unwrap().clone()
    }

    /// Reset state
    pub fn reset(&self) {
        self.positions.lock().unwrap().clear();
        self.liquidation_events.lock().unwrap().clear();
        self.state.total_margin_used.store(0, Ordering::Relaxed);
        self.state.liquidation_count.store(0, Ordering::Relaxed);
        self.state.is_liquidating.store(false, Ordering::Relaxed);
        self.state.margin_ratio_bps.store(10000, Ordering::Relaxed);
    }
}

/// Margin statistics
#[derive(Debug, Clone)]
pub struct MarginStats {
    pub total_equity: u64,
    pub total_margin_used: u64,
    pub margin_ratio_bps: u64,
    pub total_unrealized_pnl: i64,
    pub liquidation_count: u64,
    pub is_liquidating: bool,
    pub position_count: usize,
    pub chaos_mode: bool,
}

/// Margin errors
#[derive(Debug, Clone, PartialEq)]
pub enum MarginError {
    ChaosModeNotActive,
    PositionNotFound(String),
    InsufficientMargin,
    InvalidConfiguration,
}

impl std::fmt::Display for MarginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarginError::ChaosModeNotActive => write!(f, "Chaos mode not active"),
            MarginError::PositionNotFound(s) => write!(f, "Position {} not found", s),
            MarginError::InsufficientMargin => write!(f, "Insufficient margin"),
            MarginError::InvalidConfiguration => write!(f, "Invalid configuration"),
        }
    }
}

impl std::error::Error for MarginError {}

/// Builder for margin call configurations
pub struct MarginCallConfigBuilder {
    initial_margin_ratio: f64,
    maintenance_margin_ratio: f64,
    liquidation_penalty: f64,
}

impl MarginCallConfigBuilder {
    pub fn new() -> Self {
        Self {
            initial_margin_ratio: 0.10, // 10x leverage
            maintenance_margin_ratio: 0.05,
            liquidation_penalty: 0.005, // 0.5% penalty
        }
    }

    pub fn initial_margin_ratio(mut self, ratio: f64) -> Self {
        self.initial_margin_ratio = ratio.clamp(0.01, 1.0);
        self
    }

    pub fn maintenance_margin_ratio(mut self, ratio: f64) -> Self {
        self.maintenance_margin_ratio = ratio.clamp(0.01, 1.0);
        self
    }

    pub fn liquidation_penalty(mut self, penalty: f64) -> Self {
        self.liquidation_penalty = penalty.clamp(0.0, 0.1);
        self
    }

    pub fn build(self) -> MarginCallConfig {
        MarginCallConfig {
            initial_margin_ratio: self.initial_margin_ratio,
            maintenance_margin_ratio: self.maintenance_margin_ratio,
            liquidation_penalty: self.liquidation_penalty,
            chaos_mode_flag: false,
        }
    }
}

impl Default for MarginCallConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liquidation_without_chaos_mode() {
        let config = MarginCallConfigBuilder::new()
            .maintenance_margin_ratio(0.05)
            .build();

        let cascader = MarginCallCascader::new(config);

        // Add a position
        let position = Position {
            symbol: "BTC-PERP".to_string(),
            size: 100,
            entry_price: 50000_000_000,
            leverage: 10,
            margin: 500_000_000,
            unrealized_pnl: 0,
        };
        let _ = cascader.add_position(position);

        // Should fail without chaos mode
        let result = cascader.check_liquidations();
        assert!(matches!(result, Err(MarginError::ChaosModeNotActive)));
    }

    #[test]
    fn test_margin_stats() {
        let config = MarginCallConfigBuilder::new()
            .build();

        let cascader = MarginCallCascader::new(config);
        cascader.activate_chaos_mode();

        let stats = cascader.get_stats();
        assert_eq!(stats.position_count, 0);
        assert!(stats.chaos_mode);
        assert!(!stats.is_liquidating);
    }

    #[test]
    fn test_deleveraging() {
        let config = MarginCallConfigBuilder::new()
            .build();

        let cascader = MarginCallCascader::new(config);
        cascader.activate_chaos_mode();

        // Add a position
        let position = Position {
            symbol: "BTC-PERP".to_string(),
            size: 100,
            entry_price: 50000_000_000,
            leverage: 10,
            margin: 500_000_000,
            unrealized_pnl: 0,
        };
        let _ = cascader.add_position(position);

        // Execute deleveraging
        let reduced = cascader.execute_deleveraging(0.5); // 50% reduction
        assert!(reduced.is_ok());

        let stats = cascader.get_stats();
        assert!(!stats.is_liquidating);
    }
}
