//! Asymmetric Quote Router - Routes quotes based on toxicity and inventory.
//! Zero-allocation, no unwrap/expect in hot paths.

use crate::adverse::vpin_spread_adjuster::{AdjustedSpreads, ToxicityAction, VpinSpreadAdjuster};
use crate::inventory::hard_stop_state_machine::{HardStopStateMachine, QuotePermission};

/// Error types for quote routing
#[derive(Debug, Clone, PartialEq)]
pub enum QuoteRouterError {
    InvalidQuote,
    Halted,
    ToxicityTooHigh,
}

/// Quote side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuoteSide {
    Bid,
    Ask,
}

/// Quote action to take
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuoteAction {
    Place,
    Cancel,
    Replace,
    Skip,
}

/// Routing decision for a quote
#[derive(Debug, Clone, Copy)]
pub struct RoutingDecision {
    pub action: QuoteAction,
    pub price: f64,
    pub size: f64,
    pub reason: &'static str,
}

impl RoutingDecision {
    pub const fn new(action: QuoteAction, price: f64, size: f64, reason: &'static str) -> Self {
        Self { action, price, size, reason }
    }
}

/// Asymmetric Quote Router
pub struct AsymmetricQuoteRouter {
    /// Last bid price
    last_bid: f64,
    /// Last ask price
    last_ask: f64,
    /// Minimum price increment (tick size)
    tick_size: f64,
    /// Minimum size increment
    min_size: f64,
}

impl AsymmetricQuoteRouter {
    pub fn new(tick_size: f64, min_size: f64) -> Result<Self, QuoteRouterError> {
        if tick_size <= 0.0 || min_size <= 0.0 {
            return Err(QuoteRouterError::InvalidQuote);
        }
        
        Ok(Self {
            last_bid: 0.0,
            last_ask: 0.0,
            tick_size,
            min_size,
        })
    }
    
    /// Route quote based on VPIN adjustment and hard-stop permissions
    #[inline(always)]
    pub fn route_quote(
        &mut self,
        mid_price: f64,
        base_spread: f64,
        adjusted_spreads: AdjustedSpreads,
        permission: QuotePermission,
        base_size: f64,
        side: QuoteSide,
    ) -> RoutingDecision {
        // Check if quoting is allowed on this side
        let can_quote = match side {
            QuoteSide::Bid => permission.can_bid,
            QuoteSide::Ask => permission.can_ask,
        };
        
        if !can_quote {
            return RoutingDecision::new(
                QuoteAction::Cancel,
                0.0,
                0.0,
                "Hard-stop limit reached",
            );
        }
        
        // Apply size reduction from hard-stop
        let adjusted_size = (base_size * permission.size_factor).max(self.min_size);
        
        // Calculate quote price
        let quote_price = match side {
            QuoteSide::Bid => mid_price - adjusted_spreads.bid_spread,
            QuoteSide::Ask => mid_price + adjusted_spreads.ask_spread,
        };
        
        // Round to tick size
        let rounded_price = (quote_price / self.tick_size).round() * self.tick_size;
        
        // Determine action based on price change
        let action = match side {
            QuoteSide::Bid => {
                if self.last_bid == 0.0 {
                    QuoteAction::Place
                } else if (rounded_price - self.last_bid).abs() > self.tick_size {
                    QuoteAction::Replace
                } else {
                    QuoteAction::Skip
                }
            }
            QuoteSide::Ask => {
                if self.last_ask == 0.0 {
                    QuoteAction::Place
                } else if (rounded_price - self.last_ask).abs() > self.tick_size {
                    QuoteAction::Replace
                } else {
                    QuoteAction::Skip
                }
            }
        };
        
        // Update last prices
        match side {
            QuoteSide::Bid => self.last_bid = rounded_price,
            QuoteSide::Ask => self.last_ask = rounded_price,
        }
        
        let reason = match action {
            QuoteAction::Place => "New quote",
            QuoteAction::Replace => "Price update",
            QuoteAction::Skip => "No change",
            QuoteAction::Cancel => "Cancelled",
        };
        
        RoutingDecision::new(action, rounded_price, adjusted_size, reason)
    }
    
    /// Get routing decisions for both sides
    pub fn route_both_sides(
        &mut self,
        mid_price: f64,
        base_spread: f64,
        adjusted_spreads: AdjustedSpreads,
        permission: QuotePermission,
        base_size: f64,
    ) -> (RoutingDecision, RoutingDecision) {
        let bid_decision = self.route_quote(
            mid_price,
            base_spread,
            adjusted_spreads,
            permission,
            base_size,
            QuoteSide::Bid,
        );
        
        let ask_decision = self.route_quote(
            mid_price,
            base_spread,
            adjusted_spreads,
            permission,
            base_size,
            QuoteSide::Ask,
        );
        
        (bid_decision, ask_decision)
    }
    
    /// Reset last prices
    pub fn reset(&mut self) {
        self.last_bid = 0.0;
        self.last_ask = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_routing() {
        let mut router = AsymmetricQuoteRouter::new(0.01, 1.0).unwrap();
        
        let adjusted = AdjustedSpreads {
            bid_spread: 0.05,
            ask_spread: 0.05,
            multiplier: 1.0,
            toxicity_direction: 0.0,
        };
        
        let permission = QuotePermission::new(true, true, 
            crate::inventory::hard_stop_state_machine::HardStopState::Active, 
            1.0, 1.0);
        
        let (bid, ask) = router.route_both_sides(100.0, 0.05, adjusted, permission, 10.0);
        
        assert_eq!(bid.action, QuoteAction::Place);
        assert!((bid.price - 99.95).abs() < 0.01);
        assert_eq!(ask.action, QuoteAction::Place);
        assert!((ask.price - 100.05).abs() < 0.01);
    }
}
