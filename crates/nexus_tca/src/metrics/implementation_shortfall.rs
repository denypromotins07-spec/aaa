//! Implementation Shortfall Calculator using Fixed-Point Math
//! 
//! Calculates Arrival Price vs Execution Price for meta-orders
//! in real-time without floating-point drift.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TCAError {
    #[error("Invalid price: {reason}")]
    InvalidPrice { reason: String },
    #[error("Invalid quantity: {reason}")]
    InvalidQuantity { reason: String },
    #[error("Division by zero in calculation")]
    DivisionByZero,
    #[error("Overflow detected in fixed-point math")]
    Overflow,
}

/// Implementation Shortfall result with all components
#[derive(Debug, Clone)]
pub struct ImplementationShortfallResult {
    /// Arrival price (fixed-point)
    pub arrival_price: i64,
    /// Average execution price (fixed-point)
    pub exec_price: i64,
    /// Total quantity executed
    pub exec_qty: i64,
    /// Remaining quantity unexecuted
    pub remaining_qty: i64,
    /// Price component of shortfall (fixed-point)
    pub price_component: i64,
    /// Delay cost component (fixed-point)
    pub delay_cost: i64,
    /// Market impact component (fixed-point)
    pub market_impact: i64,
    /// Spread cost component (fixed-point)
    pub spread_cost: i64,
    /// Total implementation shortfall in basis points
    pub shortfall_bps: i64,
    /// Sign: positive = unfavorable, negative = favorable
    pub is_unfavorable: bool,
}

/// Side of the order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Implementation Shortfall Calculator using i64 fixed-point arithmetic
pub struct ImplementationShortfallCalculator {
    /// Scale factor for fixed-point (e.g., 1_000_000 for micro-units)
    scale: i64,
    /// Total calculations performed
    calculation_count: AtomicU64,
    /// Cumulative shortfall in basis points
    cumulative_shortfall_bps: AtomicI64,
}

impl ImplementationShortfallCalculator {
    pub fn new(scale: i64) -> Self {
        Self {
            scale,
            calculation_count: AtomicU64::new(0),
            cumulative_shortfall_bps: AtomicI64::new(0),
        }
    }

    /// Calculate implementation shortfall for a buy order
    pub fn calc_buy_shortfall(
        &self,
        arrival_price: i64,
        exec_prices: &[i64],
        exec_qtys: &[i64],
        total_qty: i64,
        decision_price: Option<i64>,
        spread_bps: i64,
    ) -> Result<ImplementationShortfallResult, TCAError> {
        if arrival_price <= 0 {
            return Err(TCAError::InvalidPrice { 
                reason: "Arrival price must be positive".to_string() 
            });
        }
        if total_qty <= 0 {
            return Err(TCAError::InvalidQuantity { 
                reason: "Total quantity must be positive".to_string() 
            });
        }
        if exec_prices.len() != exec_qtys.len() {
            return Err(TCAError::InvalidQuantity { 
                reason: "Price and qty arrays must have same length".to_string() 
            });
        }

        // Calculate weighted average execution price
        let mut total_exec_value = 0i64;
        let mut total_exec_qty = 0i64;

        for (&price, &qty) in exec_prices.iter().zip(exec_qtys.iter()) {
            if price <= 0 || qty < 0 {
                return Err(TCAError::InvalidPrice { 
                    reason: "Invalid execution price or quantity".to_string() 
                });
            }
            
            // Check for overflow before multiplication
            total_exec_value = total_exec_value
                .checked_add(price.checked_mul(qty).ok_or(TCAError::Overflow)?)
                .ok_or(TCAError::Overflow)?;
            total_exec_qty = total_exec_qty.checked_add(qty).ok_or(TCAError::Overflow)?;
        }

        let exec_price = if total_exec_qty > 0 {
            total_exec_value / total_exec_qty
        } else {
            arrival_price // No executions yet
        };

        let remaining_qty = total_qty.saturating_sub(total_exec_qty);

        // Price component: (exec_price - arrival_price) * exec_qty
        let price_component = (exec_price - arrival_price) * total_exec_qty;

        // Delay cost: cost from waiting to execute (decision_price vs arrival_price)
        let delay_cost = if let Some(dec_price) = decision_price {
            (arrival_price - dec_price) * total_exec_qty
        } else {
            0
        };

        // Market impact: estimated permanent impact
        // Simplified: assume half of price component is permanent impact
        let market_impact = price_component / 2;

        // Spread cost: half-spread paid for crossing
        let spread_cost = arrival_price * spread_bps * total_exec_qty / (2 * 10_000);

        // Total shortfall
        let total_shortfall = price_component + delay_cost + spread_cost;

        // Convert to basis points relative to notional
        let notional = arrival_price * total_exec_qty;
        let shortfall_bps = if notional > 0 {
            (total_shortfall * 10_000 / notional).abs()
        } else {
            0
        };

        let is_unfavorable = exec_price > arrival_price; // For buys, higher exec price is bad

        self.calculation_count.fetch_add(1, Ordering::Relaxed);
        
        // Update cumulative shortfall (signed)
        let signed_bps = if is_unfavorable { shortfall_bps } else { -shortfall_bps };
        self.cumulative_shortfall_bps.fetch_add(signed_bps, Ordering::Relaxed);

        Ok(ImplementationShortfallResult {
            arrival_price,
            exec_price,
            exec_qty: total_exec_qty,
            remaining_qty,
            price_component,
            delay_cost,
            market_impact,
            spread_cost,
            shortfall_bps,
            is_unfavorable,
        })
    }

    /// Calculate implementation shortfall for a sell order
    pub fn calc_sell_shortfall(
        &self,
        arrival_price: i64,
        exec_prices: &[i64],
        exec_qtys: &[i64],
        total_qty: i64,
        decision_price: Option<i64>,
        spread_bps: i64,
    ) -> Result<ImplementationShortfallResult, TCAError> {
        if arrival_price <= 0 {
            return Err(TCAError::InvalidPrice { 
                reason: "Arrival price must be positive".to_string() 
            });
        }
        if total_qty <= 0 {
            return Err(TCAError::InvalidQuantity { 
                reason: "Total quantity must be positive".to_string() 
            });
        }

        // Calculate weighted average execution price
        let mut total_exec_value = 0i64;
        let mut total_exec_qty = 0i64;

        for (&price, &qty) in exec_prices.iter().zip(exec_qtys.iter()) {
            if price <= 0 || qty < 0 {
                return Err(TCAError::InvalidPrice { 
                    reason: "Invalid execution price or quantity".to_string() 
                });
            }
            
            total_exec_value = total_exec_value
                .checked_add(price.checked_mul(qty).ok_or(TCAError::Overflow)?)
                .ok_or(TCAError::Overflow)?;
            total_exec_qty = total_exec_qty.checked_add(qty).ok_or(TCAError::Overflow)?;
        }

        let exec_price = if total_exec_qty > 0 {
            total_exec_value / total_exec_qty
        } else {
            arrival_price
        };

        let remaining_qty = total_qty.saturating_sub(total_exec_qty);

        // Price component: (arrival_price - exec_price) * exec_qty (inverted for sells)
        let price_component = (arrival_price - exec_price) * total_exec_qty;

        // Delay cost
        let delay_cost = if let Some(dec_price) = decision_price {
            (dec_price - arrival_price) * total_exec_qty
        } else {
            0
        };

        // Market impact
        let market_impact = price_component / 2;

        // Spread cost
        let spread_cost = arrival_price * spread_bps * total_exec_qty / (2 * 10_000);

        // Total shortfall
        let total_shortfall = price_component + delay_cost + spread_cost;

        // Convert to basis points
        let notional = arrival_price * total_exec_qty;
        let shortfall_bps = if notional > 0 {
            (total_shortfall * 10_000 / notional).abs()
        } else {
            0
        };

        let is_unfavorable = exec_price < arrival_price; // For sells, lower exec price is bad

        self.calculation_count.fetch_add(1, Ordering::Relaxed);
        
        let signed_bps = if is_unfavorable { shortfall_bps } else { -shortfall_bps };
        self.cumulative_shortfall_bps.fetch_add(signed_bps, Ordering::Relaxed);

        Ok(ImplementationShortfallResult {
            arrival_price,
            exec_price,
            exec_qty: total_exec_qty,
            remaining_qty,
            price_component,
            delay_cost,
            market_impact,
            spread_cost,
            shortfall_bps,
            is_unfavorable,
        })
    }

    /// Calculate shortfall for either side
    pub fn calc_shortfall(
        &self,
        side: OrderSide,
        arrival_price: i64,
        exec_prices: &[i64],
        exec_qtys: &[i64],
        total_qty: i64,
        decision_price: Option<i64>,
        spread_bps: i64,
    ) -> Result<ImplementationShortfallResult, TCAError> {
        match side {
            OrderSide::Buy => self.calc_buy_shortfall(
                arrival_price, exec_prices, exec_qtys, total_qty,
                decision_price, spread_bps,
            ),
            OrderSide::Sell => self.calc_sell_shortfall(
                arrival_price, exec_prices, exec_qtys, total_qty,
                decision_price, spread_bps,
            ),
        }
    }

    /// Get calculation count
    pub fn get_calculation_count(&self) -> u64 {
        self.calculation_count.load(Ordering::Acquire)
    }

    /// Get cumulative shortfall in basis points
    pub fn get_cumulative_shortfall_bps(&self) -> i64 {
        self.cumulative_shortfall_bps.load(Ordering::Acquire)
    }

    /// Get average shortfall per calculation
    pub fn get_avg_shortfall_bps(&self) -> i64 {
        let count = self.calculation_count.load(Ordering::Acquire);
        if count == 0 {
            return 0;
        }
        self.cumulative_shortfall_bps.load(Ordering::Acquire) / count as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buy_shortfall_unfavorable() {
        let calc = ImplementationShortfallCalculator::new(1_000_000);
        
        // Arrival at 100.00, executed at 100.05 (worse for buy)
        let result = calc.calc_buy_shortfall(
            100_000_000, // arrival price (fixed-point)
            &[100_050_000], // exec prices
            &[1000], // exec qtys
            1000, // total qty
            None,
            10, // spread bps
        ).unwrap();

        assert!(result.is_unfavorable);
        assert!(result.shortfall_bps > 0);
        assert_eq!(result.exec_price, 100_050_000);
    }

    #[test]
    fn test_buy_shortfall_favorable() {
        let calc = ImplementationShortfallCalculator::new(1_000_000);
        
        // Arrival at 100.00, executed at 99.95 (better for buy)
        let result = calc.calc_buy_shortfall(
            100_000_000,
            &[99_950_000],
            &[1000],
            1000,
            None,
            10,
        ).unwrap();

        assert!(!result.is_unfavorable);
    }

    #[test]
    fn test_sell_shortfall_unfavorable() {
        let calc = ImplementationShortfallCalculator::new(1_000_000);
        
        // Arrival at 100.00, executed at 99.95 (worse for sell)
        let result = calc.calc_sell_shortfall(
            100_000_000,
            &[99_950_000],
            &[1000],
            1000,
            None,
            10,
        ).unwrap();

        assert!(result.is_unfavorable);
    }

    #[test]
    fn test_invalid_inputs() {
        let calc = ImplementationShortfallCalculator::new(1_000_000);
        
        // Zero arrival price
        assert!(matches!(
            calc.calc_buy_shortfall(0, &[100], &[1], 1, None, 10),
            Err(TCAError::InvalidPrice { .. })
        ));
        
        // Mismatched array lengths
        assert!(matches!(
            calc.calc_buy_shortfall(100, &[100, 101], &[1], 1, None, 10),
            Err(TCAError::InvalidQuantity { .. })
        ));
    }

    #[test]
    fn test_cumulative_tracking() {
        let calc = ImplementationShortfallCalculator::new(1_000_000);
        
        // First trade: unfavorable
        calc.calc_buy_shortfall(100_000_000, &[100_100_000], &[1000], 1000, None, 10).unwrap();
        
        // Second trade: favorable
        calc.calc_buy_shortfall(100_000_000, &[99_900_000], &[1000], 1000, None, 10).unwrap();
        
        assert_eq!(calc.get_calculation_count(), 2);
        // Cumulative should reflect net of both trades
    }
}
