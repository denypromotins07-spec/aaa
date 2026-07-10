//! Minimax Venue Router for adversarial-resistant execution venue selection.
//! 
//! Implements game-theoretic minimax routing that assumes worst-case adversarial
//! response from the market when selecting execution venues and order types.

use std::collections::HashMap;
use parking_lot::RwLock;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RouterError {
    #[error("No available venues")]
    NoVenues,
    #[error("Invalid payoff matrix")]
    InvalidPayoffMatrix,
}

/// Execution venue information
#[derive(Debug, Clone)]
pub struct Venue {
    pub id: usize,
    pub name: String,
    pub maker_fee_bps: f64,
    pub taker_fee_bps: f64,
    pub avg_latency_us: f64,
    pub fill_rate: f64, // 0-1
    pub dark_pool: bool,
}

/// Payoff matrix entry for game theory calculation
struct PayoffEntry {
    /// Our expected payoff for this strategy/venue combination
    expected_payoff: f64,
    /// Worst-case adversarial response payoff
    worst_case_payoff: f64,
    /// Probability of successful execution
    success_prob: f64,
}

/// Minimax routing decision
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Selected venue ID
    pub venue_id: usize,
    /// Order type to use (limit/market/pegged)
    pub order_type: OrderType,
    /// Expected cost (negative = profit)
    pub expected_cost_bps: f64,
    /// Worst-case cost
    pub worst_case_cost_bps: f64,
    /// Confidence in decision (0-1)
    pub confidence: f64,
}

/// Order type options
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderType {
    Limit,
    Market,
    PeggedMid,
    PeggedBid,
    PeggedAsk,
    Iceberg,
}

/// Adversarial state estimation
struct AdversaryState {
    /// Estimated probability adversary is monitoring this venue
    monitoring_prob: f64,
    /// Estimated adversary reaction speed (microseconds)
    reaction_time_us: f64,
    /// Historical exploitation count
    exploitation_count: u64,
}

impl AdversaryState {
    fn new() -> Self {
        Self {
            monitoring_prob: 0.5,
            reaction_time_us: 100.0,
            exploitation_count: 0,
        }
    }
}

/// Minimax Venue Router
pub struct MinimaxVenueRouter {
    venues: RwLock<Vec<Venue>>,
    /// Adversary state per venue
    adversary_states: RwLock<HashMap<usize, AdversaryState>>,
    /// Payoff history for learning
    payoff_history: RwLock<HashMap<(usize, OrderType), Vec<f64>>>,
    /// Risk aversion parameter (higher = more conservative)
    risk_aversion: f64,
}

impl MinimaxVenueRouter {
    /// Create a new minimax venue router
    pub fn new(risk_aversion: f64) -> Result<Self, RouterError> {
        if risk_aversion < 0.0 || risk_aversion > 1.0 {
            return Err(RouterError::InvalidPayoffMatrix);
        }
        
        Ok(Self {
            venues: RwLock::new(Vec::new()),
            adversary_states: RwLock::new(HashMap::new()),
            payoff_history: RwLock::new(HashMap::new()),
            risk_aversion,
        })
    }

    /// Register an execution venue
    pub fn add_venue(&self, venue: Venue) {
        let mut venues = self.venues.write();
        venues.push(venue);
        
        // Initialize adversary state
        let mut states = self.adversary_states.write();
        states.entry(venues.len() - 1).or_insert_with(AdversaryState::new);
    }

    /// Calculate optimal routing using minimax algorithm
    /// 
    /// For each venue/order-type combination, computes:
    /// - Expected payoff under normal conditions
    /// - Worst-case payoff assuming adversarial response
    /// 
    /// Selects the combination that maximizes the minimum (worst-case) payoff
    pub fn route(&self, order_size: u64, is_buy: bool) -> Result<RoutingDecision, RouterError> {
        let venues = self.venues.read();
        let states = self.adversary_states.read();
        
        if venues.is_empty() {
            return Err(RouterError::NoVenues);
        }
        
        let mut best_decision: Option<RoutingDecision> = None;
        let mut best_minimax_value = f64::NEG_INFINITY;
        
        // Evaluate each venue
        for venue in venues.iter() {
            // Evaluate each order type
            for order_type in self.get_applicable_order_types(venue.dark_pool) {
                let payoff = self.calculate_payoff(
                    venue,
                    order_type,
                    order_size,
                    is_buy,
                    states.get(&venue.id).map(|s| s.monitoring_prob).unwrap_or(0.5),
                );
                
                // Minimax: maximize the minimum (worst-case) payoff
                let minimax_value = payoff.worst_case_payoff * (1.0 - self.risk_aversion) 
                    + payoff.expected_payoff * self.risk_aversion;
                
                if minimax_value > best_minimax_value {
                    best_minimax_value = minimax_value;
                    best_decision = Some(RoutingDecision {
                        venue_id: venue.id,
                        order_type,
                        expected_cost_bps: -payoff.expected_payoff,
                        worst_case_cost_bps: -payoff.worst_case_payoff,
                        confidence: payoff.success_prob,
                    });
                }
            }
        }
        
        best_decision.ok_or(RouterError::NoVenues)
    }

    /// Get applicable order types for a venue
    fn get_applicable_order_types(&self, is_dark: bool) -> Vec<OrderType> {
        if is_dark {
            vec![OrderType::Limit, OrderType::PeggedMid, OrderType::Iceberg]
        } else {
            vec![OrderType::Limit, OrderType::Market, OrderType::PeggedBid, OrderType::PeggedAsk]
        }
    }

    /// Calculate payoff for a venue/order-type combination
    fn calculate_payoff(
        &self,
        venue: &Venue,
        order_type: OrderType,
        order_size: u64,
        is_buy: bool,
        adversary_monitoring_prob: f64,
    ) -> PayoffEntry {
        // Base cost from fees
        let base_fee = if matches!(order_type, OrderType::Market) {
            venue.taker_fee_bps
        } else {
            venue.maker_fee_bps
        };
        
        // Latency cost (expected slippage due to delay)
        let latency_cost = venue.avg_latency_us * 0.001; // ~0.001 bps per microsecond
        
        // Fill rate adjustment
        let fill_adjustment = 1.0 / venue.fill_rate.max(0.01);
        
        // Adversarial cost: probability of being front-run
        let adversarial_cost = self.calculate_adversarial_cost(
            order_type,
            adversary_monitoring_prob,
            is_buy,
        );
        
        // Expected payoff (negative = cost)
        let expected_payoff = -(base_fee + latency_cost + adversarial_cost * 0.5) * fill_adjustment;
        
        // Worst-case payoff (adversary successfully exploits)
        let worst_case_payoff = -(base_fee + latency_cost + adversarial_cost) * fill_adjustment * 1.5;
        
        // Success probability
        let success_prob = venue.fill_rate * (1.0 - adversary_monitoring_prob * 0.3);
        
        PayoffEntry {
            expected_payoff,
            worst_case_payoff,
            success_prob,
        }
    }

    /// Calculate adversarial exploitation cost
    fn calculate_adversarial_cost(
        &self,
        order_type: OrderType,
        monitoring_prob: f64,
        is_buy: bool,
    ) -> f64 {
        // Different order types have different vulnerability to adversarial exploitation
        let vulnerability = match order_type {
            OrderType::Market => 0.8, // High vulnerability - obvious intent
            OrderType::Limit => 0.3,  // Lower vulnerability
            OrderType::PeggedMid => 0.2, // Hidden until execution
            OrderType::PeggedBid | OrderType::PeggedAsk => 0.4,
            OrderType::Iceberg => 0.1, // Lowest vulnerability
        };
        
        // Cost increases with monitoring probability and vulnerability
        monitoring_prob * vulnerability * 5.0 // 5 bps max impact
    }

    /// Record actual payoff after execution for learning
    pub fn record_outcome(&self, venue_id: usize, order_type: OrderType, actual_payoff_bps: f64) {
        let mut history = self.payoff_history.write();
        let entry = history.entry((venue_id, order_type)).or_insert_with(Vec::new);
        entry.push(actual_payoff_bps);
        
        // Keep only recent history
        if entry.len() > 100 {
            entry.remove(0);
        }
        
        // Update adversary state based on outcome
        if actual_payoff_bps < -10.0 { // Significant negative outcome suggests exploitation
            let mut states = self.adversary_states.write();
            if let Some(state) = states.get_mut(&venue_id) {
                state.monitoring_prob = (state.monitoring_prob + 0.1).min(1.0);
                state.exploitation_count += 1;
            }
        }
    }

    /// Get all registered venues
    pub fn get_venues(&self) -> Vec<Venue> {
        self.venues.read().clone()
    }

    /// Remove a venue
    pub fn remove_venue(&self, venue_id: usize) {
        self.venues.write().retain(|v| v.id != venue_id);
        self.adversary_states.write().remove(&venue_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimax_routing() {
        let router = MinimaxVenueRouter::new(0.5).unwrap();
        
        // Add venues with different characteristics
        router.add_venue(Venue {
            id: 0,
            name: "LowFeeLit".to_string(),
            maker_fee_bps: 1.0,
            taker_fee_bps: 2.0,
            avg_latency_us: 50.0,
            fill_rate: 0.9,
            dark_pool: false,
        });
        
        router.add_venue(Venue {
            id: 1,
            name: "DarkPool".to_string(),
            maker_fee_bps: 0.5,
            taker_fee_bps: 1.0,
            avg_latency_us: 100.0,
            fill_rate: 0.7,
            dark_pool: true,
        });
        
        let decision = router.route(1000, true).unwrap();
        
        // Should select a valid venue
        assert!(decision.venue_id == 0 || decision.venue_id == 1);
        assert!(decision.confidence > 0.0);
    }

    #[test]
    fn test_no_venues_error() {
        let router = MinimaxVenueRouter::new(0.5).unwrap();
        let result = router.route(1000, true);
        assert!(result.is_err());
    }
}
