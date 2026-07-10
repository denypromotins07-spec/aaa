//! Shadow Liquidation Simulator.
//! 
//! Reverse-engineers exchange margin calculation logic to predict
//! liquidation thresholds before the exchange's engine triggers.

use std::sync::atomic::{AtomicU64, Ordering};

/// Epsilon for floating-point comparisons
const EPSILON: f64 = 1e-9;

/// Exchange type for different margin models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeType {
    /// Binance futures model (cross/isolated margin)
    Binance,
    /// Bybit model (portfolio margin)
    Bybit,
    /// Deribit model (options-focused)
    Deribit,
    /// Generic conservative model
    Generic,
}

/// Position information for margin calculation
#[derive(Debug, Clone)]
pub struct PositionInfo {
    /// Symbol (e.g., "BTCUSD")
    pub symbol: String,
    /// Side (positive = long, negative = short)
    pub side: i8,
    /// Position size in base units
    pub size: u64,
    /// Entry price in quote units
    pub entry_price: f64,
    /// Current mark price in quote units
    pub mark_price: f64,
    /// Leverage used (e.g., 10 for 10x)
    pub leverage: u32,
    /// Margin mode
    pub is_isolated: bool,
}

impl PositionInfo {
    /// Calculate unrealized P&L
    #[inline]
    pub fn unrealized_pnl(&self) -> f64 {
        let price_diff = self.mark_price - self.entry_price;
        if self.side > 0 {
            price_diff * self.size as f64
        } else {
            -price_diff * self.size as f64
        }
    }

    /// Calculate position value in quote currency
    #[inline]
    pub fn notional_value(&self) -> f64 {
        self.mark_price * self.size as f64
    }
}

/// Margin calculation result
#[derive(Debug, Clone)]
pub struct MarginResult {
    /// Initial margin required
    pub initial_margin: f64,
    /// Maintenance margin required
    pub maintenance_margin: f64,
    /// Available margin (collateral - used)
    pub available_margin: f64,
    /// Margin ratio (used / total)
    pub margin_ratio: f64,
    /// Liquidation price estimate
    pub liquidation_price: Option<f64>,
    /// Distance to liquidation as percentage
    pub liquidation_buffer_pct: f64,
}

/// Shadow Liquidation Simulator
/// 
/// Simulates exchange margin calculations to predict liquidations
/// before they happen and trigger preemptive deleveraging.
pub struct ShadowLiquidationSimulator {
    /// Exchange model being simulated
    exchange: ExchangeType,
    /// Account collateral (margin balance + unrealized PnL)
    collateral: AtomicU64, // Stored as bits for atomic f64
    /// Count of simulations performed
    simulation_count: AtomicU64,
    /// Count of liquidation warnings issued
    liquidation_warnings: AtomicU64,
}

unsafe impl Send for ShadowLiquidationSimulator {}
unsafe impl Sync for ShadowLiquidationSimulator {}

impl ShadowLiquidationSimulator {
    /// Create a new simulator for the specified exchange model.
    pub fn new(exchange: ExchangeType) -> Self {
        Self {
            exchange,
            collateral: AtomicU64::new(0),
            simulation_count: AtomicU64::new(0),
            liquidation_warnings: AtomicU64::new(0),
        }
    }

    /// Update account collateral.
    #[inline]
    pub fn update_collateral(&self, collateral: f64) {
        self.collateral.store(f64::to_bits(collateral.max(0.0)), Ordering::Relaxed);
    }

    /// Get current collateral value.
    #[inline]
    pub fn get_collateral(&self) -> f64 {
        f64::from_bits(self.collateral.load(Ordering::Relaxed))
    }

    /// Calculate margin requirements for a single position.
    #[inline]
    pub fn calculate_position_margin(&self, position: &PositionInfo) -> MarginResult {
        match self.exchange {
            ExchangeType::Binance => self.calculate_binance_margin(position),
            ExchangeType::Bybit => self.calculate_bybit_margin(position),
            ExchangeType::Deribit => self.calculate_deribit_margin(position),
            ExchangeType::Generic => self.calculate_generic_margin(position),
        }
    }

    /// Binance-style margin calculation.
    /// 
    /// Uses tiered maintenance margin ratios based on notional value.
    fn calculate_binance_margin(&self, position: &PositionInfo) -> MarginResult {
        let notional = position.notional_value();
        
        // Binance initial margin = notional / leverage
        let initial_margin = notional / position.leverage as f64;
        
        // Binance maintenance margin rates (simplified tier structure)
        let mm_rate = self.get_binance_mm_rate(notional);
        let maintenance_margin = notional * mm_rate;
        
        // Calculate used margin
        let collateral = self.get_collateral();
        let used_margin = if position.is_isolated {
            initial_margin // Isolated uses fixed margin
        } else {
            maintenance_margin // Cross shares collateral
        };
        
        let available_margin = collateral - used_margin;
        let margin_ratio = if collateral > EPSILON {
            used_margin / collateral
        } else {
            1.0
        };
        
        // Estimate liquidation price
        let liq_price = self.estimate_liquidation_price_binance(position, collateral);
        
        // Calculate buffer to liquidation
        let liq_buffer = if let Some(lp) = liq_price {
            if position.side > 0 {
                // Long: how much can price drop before liq
                (position.mark_price - lp) / position.mark_price
            } else {
                // Short: how much can price rise before liq
                (lp - position.mark_price) / position.mark_price
            }
        } else {
            f64::INFINITY
        };
        
        // Track warnings
        if margin_ratio > 0.9 {
            self.liquidation_warnings.fetch_add(1, Ordering::Relaxed);
        }
        
        self.simulation_count.fetch_add(1, Ordering::Relaxed);
        
        MarginResult {
            initial_margin,
            maintenance_margin,
            available_margin,
            margin_ratio,
            liquidation_price: liq_price,
            liquidation_buffer_pct: liq_buffer,
        }
    }

    /// Get Binance maintenance margin rate based on notional tier.
    #[inline]
    fn get_binance_mm_rate(&self, notional: f64) -> f64 {
        // Simplified tier structure (actual Binance has more tiers)
        match notional {
            n if n < 50_000.0 => 0.004,   // 0.4% for small positions
            n if n < 250_000.0 => 0.005,  // 0.5%
            n if n < 1_000_000.0 => 0.01, // 1.0%
            n if n < 5_000_000.0 => 0.025,// 2.5%
            _ => 0.05,                     // 5% for whale positions
        }
    }

    /// Estimate liquidation price for Binance model.
    fn estimate_liquidation_price_binance(
        &self,
        position: &PositionInfo,
        collateral: f64,
    ) -> Option<f64> {
        if position.size == 0 {
            return None;
        }
        
        let mm_rate = self.get_binance_mm_rate(position.notional_value());
        let entry_price = position.entry_price;
        let size = position.size as f64;
        
        if position.side > 0 {
            // Long position liquidation price
            // LiqPrice = (Entry Price * Size - Collateral) / (Size * (1 - MM Rate))
            let numerator = entry_price * size - collateral;
            let denominator = size * (1.0 - mm_rate);
            if denominator > EPSILON {
                Some((numerator / denominator).max(0.0))
            } else {
                None
            }
        } else {
            // Short position liquidation price
            // LiqPrice = (Entry Price * Size + Collateral) / (Size * (1 + MM Rate))
            let numerator = entry_price * size + collateral;
            let denominator = size * (1.0 + mm_rate);
            if denominator > EPSILON {
                Some(numerator / denominator)
            } else {
                None
            }
        }
    }

    /// Bybit portfolio margin calculation.
    fn calculate_bybit_margin(&self, position: &PositionInfo) -> MarginResult {
        // Bybit uses portfolio margin with risk offsets
        let notional = position.notional_value();
        
        // Risk-based initial margin (simplified)
        let im_rate = 1.0 / position.leverage as f64;
        let initial_margin = notional * im_rate;
        
        // Maintenance margin (typically ~50% of IM)
        let maintenance_margin = initial_margin * 0.5;
        
        let collateral = self.get_collateral();
        let margin_ratio = if collateral > EPSILON {
            maintenance_margin / collateral
        } else {
            1.0
        };
        
        // Simplified liquidation price
        let liq_price = if position.side > 0 {
            Some(position.entry_price * (1.0 - 1.0 / position.leverage as f64))
        } else {
            Some(position.entry_price * (1.0 + 1.0 / position.leverage as f64))
        };
        
        let liq_buffer = if let Some(lp) = liq_price {
            if position.side > 0 {
                (position.mark_price - lp) / position.mark_price
            } else {
                (lp - position.mark_price) / position.mark_price
            }
        } else {
            f64::INFINITY
        };
        
        self.simulation_count.fetch_add(1, Ordering::Relaxed);
        
        MarginResult {
            initial_margin,
            maintenance_margin,
            available_margin: collateral - maintenance_margin,
            margin_ratio,
            liquidation_price: liq_price,
            liquidation_buffer_pct: liq_buffer,
        }
    }

    /// Deribit margin calculation (options-focused).
    fn calculate_deribit_margin(&self, position: &PositionInfo) -> MarginResult {
        // Deribit uses SPAN-like methodology for options
        // This is a simplified version
        
        let notional = position.notional_value();
        let initial_margin = notional / position.leverage as f64;
        let maintenance_margin = initial_margin * 0.7; // Higher MM for options
        
        let collateral = self.get_collateral();
        let margin_ratio = if collateral > EPSILON {
            maintenance_margin / collateral
        } else {
            1.0
        };
        
        self.simulation_count.fetch_add(1, Ordering::Relaxed);
        
        MarginResult {
            initial_margin,
            maintenance_margin,
            available_margin: collateral - maintenance_margin,
            margin_ratio,
            liquidation_price: None, // Options have complex liq mechanics
            liquidation_buffer_pct: f64::INFINITY,
        }
    }

    /// Generic conservative margin calculation.
    fn calculate_generic_margin(&self, position: &PositionInfo) -> MarginResult {
        let notional = position.notional_value();
        
        // Conservative: use highest margin requirements
        let initial_margin = notional / position.leverage as f64;
        let maintenance_margin = initial_margin * 0.8; // 80% of IM
        
        let collateral = self.get_collateral();
        let margin_ratio = if collateral > EPSILON {
            maintenance_margin / collateral
        } else {
            1.0
        };
        
        self.simulation_count.fetch_add(1, Ordering::Relaxed);
        
        MarginResult {
            initial_margin,
            maintenance_margin,
            available_margin: collateral - maintenance_margin,
            margin_ratio,
            liquidation_price: None,
            liquidation_buffer_pct: f64::INFINITY,
        }
    }

    /// Check if any position is at risk of liquidation.
    /// 
    /// Returns true if margin ratio exceeds the threshold.
    #[inline]
    pub fn is_at_risk(&self, positions: &[PositionInfo], threshold: f64) -> bool {
        for position in positions {
            let result = self.calculate_position_margin(position);
            if result.margin_ratio > threshold {
                return true;
            }
        }
        false
    }

    /// Get simulation statistics.
    pub fn stats(&self) -> LiquidationSimStats {
        LiquidationSimStats {
            exchange: self.exchange,
            collateral: self.get_collateral(),
            simulation_count: self.simulation_count.load(Ordering::Relaxed),
            liquidation_warnings: self.liquidation_warnings.load(Ordering::Relaxed),
        }
    }
}

/// Statistics from the liquidation simulator
#[derive(Debug, Clone)]
pub struct LiquidationSimStats {
    pub exchange: ExchangeType,
    pub collateral: f64,
    pub simulation_count: u64,
    pub liquidation_warnings: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binance_long_margin() {
        let sim = ShadowLiquidationSimulator::new(ExchangeType::Binance);
        sim.update_collateral(10_000.0);
        
        let position = PositionInfo {
            symbol: "BTCUSD".to_string(),
            side: 1, // Long
            size: 100, // 0.01 BTC if size is in sats, or 100 if in BTC
            entry_price: 50_000.0,
            mark_price: 50_000.0,
            leverage: 10,
            is_isolated: false,
        };
        
        let result = sim.calculate_position_margin(&position);
        
        assert!(result.initial_margin > 0.0);
        assert!(result.maintenance_margin > 0.0);
        assert!(result.maintenance_margin <= result.initial_margin);
        assert!(result.margin_ratio >= 0.0 && result.margin_ratio <= 1.0);
    }

    #[test]
    fn test_liquidation_price_long() {
        let sim = ShadowLiquidationSimulator::new(ExchangeType::Binance);
        sim.update_collateral(5_000.0);
        
        let position = PositionInfo {
            symbol: "BTCUSD".to_string(),
            side: 1,
            size: 1,
            entry_price: 50_000.0,
            mark_price: 50_000.0,
            leverage: 10,
            is_isolated: false,
        };
        
        let result = sim.calculate_position_margin(&position);
        
        // Long position should have liq price below entry
        if let Some(liq_price) = result.liquidation_price {
            assert!(liq_price < position.entry_price);
            assert!(liq_price > 0.0);
        } else {
            panic!("Expected liquidation price");
        }
    }

    #[test]
    fn test_liquidation_price_short() {
        let sim = ShadowLiquidationSimulator::new(ExchangeType::Binance);
        sim.update_collateral(5_000.0);
        
        let position = PositionInfo {
            symbol: "BTCUSD".to_string(),
            side: -1, // Short
            size: 1,
            entry_price: 50_000.0,
            mark_price: 50_000.0,
            leverage: 10,
            is_isolated: false,
        };
        
        let result = sim.calculate_position_margin(&position);
        
        // Short position should have liq price above entry
        if let Some(liq_price) = result.liquidation_price {
            assert!(liq_price > position.entry_price);
        } else {
            panic!("Expected liquidation price");
        }
    }

    #[test]
    fn test_margin_ratio_increases_with_loss() {
        let sim = ShadowLiquidationSimulator::new(ExchangeType::Binance);
        sim.update_collateral(10_000.0);
        
        let mut position = PositionInfo {
            symbol: "BTCUSD".to_string(),
            side: 1,
            size: 1,
            entry_price: 50_000.0,
            mark_price: 50_000.0,
            leverage: 10,
            is_isolated: false,
        };
        
        let result_initial = sim.calculate_position_margin(&position);
        
        // Now simulate price dropping (loss on long)
        position.mark_price = 45_000.0;
        let result_loss = sim.calculate_position_margin(&position);
        
        // Margin ratio should increase with losses
        assert!(result_loss.margin_ratio > result_initial.margin_ratio);
    }

    #[test]
    fn test_is_at_risk() {
        let sim = ShadowLiquidationSimulator::new(ExchangeType::Binance);
        sim.update_collateral(1_000.0); // Low collateral
        
        let positions = vec![
            PositionInfo {
                symbol: "BTCUSD".to_string(),
                side: 1,
                size: 1,
                entry_price: 50_000.0,
                mark_price: 50_000.0,
                leverage: 100, // High leverage
                is_isolated: false,
            },
        ];
        
        // With high leverage and low collateral, should be at risk
        assert!(sim.is_at_risk(&positions, 0.5));
    }

    #[test]
    fn test_unrealized_pnl() {
        let long_profit = PositionInfo {
            symbol: "BTCUSD".to_string(),
            side: 1,
            size: 1,
            entry_price: 50_000.0,
            mark_price: 55_000.0,
            leverage: 10,
            is_isolated: false,
        };
        assert_eq!(long_profit.unrealized_pnl(), 5_000.0);
        
        let short_profit = PositionInfo {
            symbol: "BTCUSD".to_string(),
            side: -1,
            size: 1,
            entry_price: 50_000.0,
            mark_price: 45_000.0,
            leverage: 10,
            is_isolated: false,
        };
        assert_eq!(short_profit.unrealized_pnl(), 5_000.0);
    }

    #[test]
    fn test_different_exchanges() {
        let exchanges = [
            ExchangeType::Binance,
            ExchangeType::Bybit,
            ExchangeType::Deribit,
            ExchangeType::Generic,
        ];
        
        let position = PositionInfo {
            symbol: "BTCUSD".to_string(),
            side: 1,
            size: 1,
            entry_price: 50_000.0,
            mark_price: 50_000.0,
            leverage: 10,
            is_isolated: false,
        };
        
        let mut results = Vec::new();
        for exchange in exchanges {
            let sim = ShadowLiquidationSimulator::new(exchange);
            sim.update_collateral(10_000.0);
            results.push(sim.calculate_position_margin(&position));
        }
        
        // Different exchanges should give different margin requirements
        // (though some might coincidentally be similar)
        assert!(results.iter().all(|r| r.initial_margin > 0.0));
    }
}
