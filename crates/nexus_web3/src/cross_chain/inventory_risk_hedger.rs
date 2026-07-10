//! Inventory Risk Hedger for Cross-Chain Arbitrage
//! Manages inventory exposure across multiple chains

use thiserror::Error;
use alloc::vec::Vec;

#[derive(Error, Debug)]
pub enum HedgingError {
    #[error("Invalid hedge ratio")]
    InvalidHedgeRatio,
    #[error("Insufficient inventory")]
    InsufficientInventory,
    #[error("Hedge execution failed")]
    HedgeExecutionFailed,
}

pub type Result<T> = core::result::Result<T, HedgingError>;

/// Inventory position on a single chain
#[derive(Clone, Debug)]
pub struct InventoryPosition {
    pub chain_id: u8,
    pub asset: [u8; 32],
    pub amount: i128, // Can be negative for short positions
    pub usd_value: f64,
}

/// Hedge instrument specification
#[derive(Clone, Debug)]
pub struct HedgeInstrument {
    pub instrument_type: InstrumentType,
    pub underlying: [u8; 32],
    pub notional: f64,
    pub delta: f64,
    pub expiry: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InstrumentType {
    SpotShort,
    PerpetualFuture,
    PutOption,
    CallOption,
    InverseETP,
}

/// Risk metrics for inventory management
#[derive(Clone, Debug)]
pub struct RiskMetrics {
    /// Total portfolio value in USD
    pub total_value_usd: f64,
    /// Net exposure (long - short)
    pub net_exposure_usd: f64,
    /// Gross exposure (|long| + |short|)
    pub gross_exposure_usd: f64,
    /// Current hedge ratio (hedged / total)
    pub hedge_ratio: f64,
    /// Target hedge ratio
    pub target_hedge_ratio: f64,
    /// Value at Risk (95% confidence, 1 day)
    pub var_95: f64,
    /// Maximum drawdown tolerance
    pub max_drawdown: f64,
}

/// Inventory Risk Hedger state machine
pub struct InventoryRiskHedger {
    /// Current inventory positions
    positions: Vec<InventoryPosition>,
    /// Active hedges
    hedges: Vec<HedgeInstrument>,
    /// Target hedge ratio (0.0 to 1.0)
    target_hedge_ratio: f64,
    /// Rebalance threshold (when to trigger rebalance)
    rebalance_threshold: f64,
    /// Maximum position size per asset
    max_position_usd: f64,
}

impl InventoryRiskHedger {
    /// Create a new hedger with default parameters
    pub fn new() -> Self {
        Self {
            positions: Vec::with_capacity(16),
            hedges: Vec::with_capacity(8),
            target_hedge_ratio: 0.8, // 80% hedged by default
            rebalance_threshold: 0.1, // Rebalance when drift > 10%
            max_position_usd: 1_000_000.0, // $1M max per position
        }
    }

    /// Create with custom parameters
    pub fn with_params(target_hedge: f64, rebalance_thresh: f64) -> Result<Self> {
        if target_hedge < 0.0 || target_hedge > 1.0 {
            return Err(HedgingError::InvalidHedgeRatio);
        }
        
        Ok(Self {
            positions: Vec::with_capacity(16),
            hedges: Vec::with_capacity(8),
            target_hedge_ratio: target_hedge,
            rebalance_threshold: rebalance_thresh,
            max_position_usd: 1_000_000.0,
        })
    }

    /// Add or update an inventory position
    pub fn update_position(&mut self, position: InventoryPosition) {
        // Check position limits
        if position.usd_value.abs() > self.max_position_usd {
            // Log warning but still accept (may need emergency unwind)
        }

        // Find existing position and update or add new
        let found = self.positions.iter_mut()
            .find(|p| p.chain_id == position.chain_id && p.asset == position.asset);
        
        match found {
            Some(existing) => {
                existing.amount = position.amount;
                existing.usd_value = position.usd_value;
            }
            None => {
                self.positions.push(position);
            }
        }
    }

    /// Add a hedge instrument
    pub fn add_hedge(&mut self, hedge: HedgeInstrument) {
        self.hedges.push(hedge);
    }

    /// Remove expired hedges
    pub fn cleanup_expired_hedges(&mut self, current_time: u64) {
        self.hedges.retain(|h| {
            h.expiry.map_or(true, |exp| exp > current_time)
        });
    }

    /// Calculate current risk metrics
    pub fn calculate_metrics(&self) -> RiskMetrics {
        let mut total_long = 0.0_f64;
        let mut total_short = 0.0_f64;

        for pos in &self.positions {
            if pos.amount > 0 {
                total_long += pos.usd_value;
            } else {
                total_short += pos.usd_value.abs();
            }
        }

        let total_value = total_long + total_short;
        let net_exposure = total_long - total_short;
        let gross_exposure = total_long + total_short;

        // Calculate effective hedge from instruments
        let hedge_value: f64 = self.hedges.iter()
            .map(|h| h.notional * h.delta)
            .sum();

        let hedge_ratio = if gross_exposure > 0.0 {
            hedge_value / gross_exposure
        } else {
            0.0
        };

        // Simple VaR approximation (would use historical simulation in production)
        let var_95 = gross_exposure * 0.05; // 5% daily VaR

        RiskMetrics {
            total_value_usd: total_value,
            net_exposure_usd: net_exposure,
            gross_exposure_usd: gross_exposure,
            hedge_ratio,
            target_hedge_ratio: self.target_hedge_ratio,
            var_95,
            max_drawdown: 0.1, // 10% max drawdown
        }
    }

    /// Check if rebalancing is needed
    pub fn needs_rebalance(&self) -> bool {
        let metrics = self.calculate_metrics();
        let drift = (metrics.hedge_ratio - metrics.target_hedge_ratio).abs();
        drift > self.rebalance_threshold
    }

    /// Calculate required hedge adjustment
    pub fn calculate_hedge_adjustment(&self) -> f64 {
        let metrics = self.calculate_metrics();
        
        let target_hedge_value = metrics.gross_exposure_usd * metrics.target_hedge_ratio;
        let current_hedge_value: f64 = self.hedges.iter()
            .map(|h| h.notional * h.delta)
            .sum();

        target_hedge_value - current_hedge_value
    }

    /// Generate hedge orders to reach target ratio
    pub fn generate_hedge_orders(&self) -> Vec<HedgeOrder> {
        let adjustment = self.calculate_hedge_adjustment();
        
        if adjustment.abs() < 1000.0 { // Minimum order size $1000
            return Vec::new();
        }

        let mut orders = Vec::new();
        
        // Distribute across available instruments
        let instrument_types = if adjustment > 0.0 {
            vec![InstrumentType::PerpetualFuture, InstrumentType::SpotShort]
        } else {
            vec![InstrumentType::InverseETP]
        };

        for instr_type in instrument_types {
            orders.push(HedgeOrder {
                instrument_type: instr_type,
                notional: adjustment.abs() / instrument_types.len() as f64,
                side: if adjustment > 0.0 { OrderSide::OpenShort } else { OrderSide::Close },
            });
        }

        orders
    }

    /// Execute a hedge order (simulation)
    pub fn execute_hedge(&mut self, order: &HedgeOrder) -> Result<()> {
        if order.notional <= 0.0 {
            return Err(HedgingError::InvalidHedgeRatio);
        }

        let delta = match order.instrument_type {
            InstrumentType::SpotShort => -1.0,
            InstrumentType::PerpetualFuture => -1.0,
            InstrumentType::PutOption => -0.5, // Approximate delta
            InstrumentType::CallOption => 0.5,
            InstrumentType::InverseETP => -1.0,
        };

        let hedge = HedgeInstrument {
            instrument_type: order.instrument_type,
            underlying: [0u8; 32], // Would be set based on asset
            notional: order.notional,
            delta,
            expiry: None,
        };

        self.add_hedge(hedge);
        Ok(())
    }

    /// Emergency unwind all positions
    pub fn emergency_unwind(&mut self) -> Vec<InventoryPosition> {
        let positions_to_close = self.positions.clone();
        self.positions.clear();
        self.hedges.clear();
        positions_to_close
    }

    /// Get current positions
    pub const fn positions(&self) -> &[InventoryPosition] {
        &self.positions
    }

    /// Get active hedges
    pub const fn hedges(&self) -> &[HedgeInstrument] {
        &self.hedges
    }

    /// Set target hedge ratio
    pub fn set_target_hedge_ratio(&mut self, ratio: f64) -> Result<()> {
        if ratio < 0.0 || ratio > 1.0 {
            return Err(HedgingError::InvalidHedgeRatio);
        }
        self.target_hedge_ratio = ratio;
        Ok(())
    }
}

impl Default for InventoryRiskHedger {
    fn default() -> Self {
        Self::new()
    }
}

/// Hedge order specification
#[derive(Clone, Debug)]
pub struct HedgeOrder {
    pub instrument_type: InstrumentType,
    pub notional: f64,
    pub side: OrderSide,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OrderSide {
    OpenShort,
    Close,
    BuyCover,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hedger_creation() {
        let hedger = InventoryRiskHedger::new();
        let metrics = hedger.calculate_metrics();
        assert_eq!(metrics.total_value_usd, 0.0);
        assert_eq!(metrics.hedge_ratio, 0.0);
    }

    #[test]
    fn test_invalid_hedge_ratio() {
        let result = InventoryRiskHedger::with_params(1.5, 0.1);
        assert!(matches!(result, Err(HedgingError::InvalidHedgeRatio)));
    }

    #[test]
    fn test_position_management() {
        let mut hedger = InventoryRiskHedger::new();
        
        hedger.update_position(InventoryPosition {
            chain_id: 1,
            asset: [1u8; 32],
            amount: 1_000_000_000,
            usd_value: 2000.0,
        });

        let metrics = hedger.calculate_metrics();
        assert_eq!(metrics.gross_exposure_usd, 2000.0);
    }

    #[test]
    fn test_needs_rebalance() {
        let mut hedger = InventoryRiskHedger::new();
        
        // Initially no positions, no rebalance needed
        assert!(!hedger.needs_rebalance());

        // Add unhedged position
        hedger.update_position(InventoryPosition {
            chain_id: 1,
            asset: [1u8; 32],
            amount: 1_000_000_000,
            usd_value: 100_000.0,
        });

        // Should need rebalance (0% hedged vs 80% target)
        assert!(hedger.needs_rebalance());
    }

    #[test]
    fn test_hedge_adjustment() {
        let mut hedger = InventoryRiskHedger::new();
        
        hedger.update_position(InventoryPosition {
            chain_id: 1,
            asset: [1u8; 32],
            amount: 1_000_000_000,
            usd_value: 100_000.0,
        });

        let adjustment = hedger.calculate_hedge_adjustment();
        assert!(adjustment > 0.0, "Should need positive hedge");
    }

    #[test]
    fn test_emergency_unwind() {
        let mut hedger = InventoryRiskHedger::new();
        
        hedger.update_position(InventoryPosition {
            chain_id: 1,
            asset: [1u8; 32],
            amount: 1_000_000_000,
            usd_value: 50_000.0,
        });

        let unwound = hedger.emergency_unwind();
        assert_eq!(unwound.len(), 1);
        assert!(hedger.positions().is_empty());
    }
}
