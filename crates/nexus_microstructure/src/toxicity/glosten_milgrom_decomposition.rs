//! Glosten-Milgrom spread decomposition model.
//! 
//! Decomposes the bid-ask spread into Order Processing Costs (harmless noise traders)
//! and Adverse Selection Costs (toxic, informed traders).

use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DecompositionError {
    #[error("Invalid spread value")]
    InvalidSpread,
    #[error("Insufficient data for decomposition")]
    InsufficientData,
}

/// Spread decomposition result
#[derive(Debug, Clone)]
pub struct SpreadDecomposition {
    /// Total bid-ask spread (in basis points)
    pub total_spread_bps: f64,
    /// Order processing cost component (harmless)
    pub order_processing_cost: f64,
    /// Adverse selection cost component (toxic)
    pub adverse_selection_cost: f64,
    /// Probability of informed trading (0-1)
    pub pin_estimate: f64,
    /// Timestamp
    pub timestamp_ns: u64,
}

/// Trade classification for Bayesian updating
#[derive(Debug, Clone, Copy)]
pub enum TradeType {
    BuyerInitiated,
    SellerInitiated,
    Unknown,
}

/// State for Glosten-Milgrom estimation
struct GmState {
    /// Prior probability of informed trader
    alpha_prior: f64,
    /// Probability signal is good given informed
    delta: f64,
    /// Number of buyer-initiated trades observed
    buy_count: u64,
    /// Number of seller-initiated trades observed
    sell_count: u64,
    /// Log-likelihood ratio for sequential updating
    log_likelihood_ratio: f64,
}

impl GmState {
    fn new(alpha_prior: f64, delta: f64) -> Self {
        Self {
            alpha_prior,
            delta,
            buy_count: 0,
            sell_count: 0,
            log_likelihood_ratio: 0.0,
        }
    }
}

/// Glosten-Milgrom Decomposer for real-time spread analysis
pub struct GlostenMilgromDecomposer {
    state: RwLock<GmState>,
    /// Minimum observations before valid decomposition
    min_observations: usize,
    /// Current spread estimate (basis points)
    current_spread_bps: AtomicU64,
    /// Decay factor for old observations
    decay_factor: f64,
}

impl GlostenMilgromDecomposer {
    /// Create a new Glosten-Milgrom decomposer
    pub fn new(
        alpha_prior: f64,
        delta: f64,
        min_observations: usize,
        decay_factor: f64,
    ) -> Result<Self, DecompositionError> {
        if alpha_prior < 0.0 || alpha_prior > 1.0 || delta < 0.0 || delta > 1.0 {
            return Err(DecompositionError::InvalidSpread);
        }
        if decay_factor < 0.0 || decay_factor > 1.0 {
            return Err(DecompositionError::InvalidSpread);
        }
        
        Ok(Self {
            state: RwLock::new(GmState::new(alpha_prior, delta)),
            min_observations,
            current_spread_bps: AtomicU64::new(0),
            decay_factor,
        })
    }

    /// Record a trade observation with classification
    #[inline]
    pub fn record_trade(&self, trade_type: TradeType, spread_bps: f64, timestamp_ns: u64) {
        let mut state = self.state.write();
        
        // Update spread estimate
        self.current_spread_bps.store(spread_bps as u64, Ordering::Relaxed);
        
        match trade_type {
            TradeType::BuyerInitiated => {
                state.buy_count += 1;
                // Update log-likelihood ratio
                // LLR += log(P(buy|informed) / P(buy|uninformed))
                let p_buy_informed = state.delta;
                let p_buy_uninformed = 0.5;
                if p_buy_uninformed > 0.0 && p_buy_informed > 0.0 {
                    state.log_likelihood_ratio += (p_buy_informed / p_buy_uninformed).ln();
                }
            }
            TradeType::SellerInitiated => {
                state.sell_count += 1;
                // LLR += log(P(sell|informed) / P(sell|uninformed))
                let p_sell_informed = 1.0 - state.delta;
                let p_sell_uninformed = 0.5;
                if p_sell_uninformed > 0.0 && p_sell_informed > 0.0 {
                    state.log_likelihood_ratio += (p_sell_informed / p_sell_uninformed).ln();
                }
            }
            TradeType::Unknown => {}
        }
        
        // Apply decay to counts periodically to prevent unbounded growth
        let total = state.buy_count + state.sell_count;
        if total > 10000 {
            state.buy_count = ((state.buy_count as f64) * self.decay_factor) as u64;
            state.sell_count = ((state.sell_count as f64) * self.decay_factor) as u64;
            state.log_likelihood_ratio *= self.decay_factor;
        }
    }

    /// Calculate spread decomposition
    pub fn decompose(&self, timestamp_ns: u64) -> Result<SpreadDecomposition, DecompositionError> {
        let state = self.state.read();
        
        let total_trades = state.buy_count + state.sell_count;
        if total_trades < self.min_observations as u64 {
            return Err(DecompositionError::InsufficientData);
        }
        
        // Calculate posterior probability of informed trading using Bayes' rule
        // P(informed|data) = P(data|informed) * P(informed) / P(data)
        // Using log-odds for numerical stability
        
        let log_odds = state.log_likelihood_ratio + (state.alpha_prior / (1.0 - state.alpha_prior)).ln();
        let pin_estimate = 1.0 / (1.0 + (-log_odds).exp());
        
        // Get current spread
        let total_spread_bps = self.current_spread_bps.load(Ordering::Relaxed) as f64;
        
        // Glosten-Milgrom decomposition:
        // Spread = OrderProcessingCost + AdverseSelectionCost
        // AdverseSelectionCost ≈ PIN * Spread * (1 - 2*delta) when delta != 0.5
        
        let adverse_selection_cost = if pin_estimate > 0.0 {
            total_spread_bps * pin_estimate * (2.0 * state.delta - 1.0).abs()
        } else {
            0.0
        };
        
        let order_processing_cost = total_spread_bps - adverse_selection_cost;
        
        Ok(SpreadDecomposition {
            total_spread_bps,
            order_processing_cost: order_processing_cost.max(0.0),
            adverse_selection_cost: adverse_selection_cost.max(0.0),
            pin_estimate,
            timestamp_ns,
        })
    }

    /// Get current adverse selection cost only (for quick toxicity check)
    pub fn get_adverse_selection_cost(&self) -> Option<f64> {
        self.decompose(0).ok().map(|d| d.adverse_selection_cost)
    }

    /// Check if market is currently toxic (high adverse selection)
    pub fn is_toxic(&self, threshold: f64) -> bool {
        if let Ok(decomp) = self.decompose(0) {
            decomp.adverse_selection_cost > threshold || decomp.pin_estimate > 0.5
        } else {
            false
        }
    }

    /// Reset estimator state
    pub fn reset(&self) {
        let mut state = self.state.write();
        state.buy_count = 0;
        state.sell_count = 0;
        state.log_likelihood_ratio = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glosten_milgrom_decomposition() {
        let decomposer = GlostenMilgromDecomposer::new(0.3, 0.7, 10, 0.99).unwrap();
        
        // Simulate mostly buyer-initiated trades (potential informed buying)
        for i in 0..50 {
            let trade_type = if i % 5 == 0 { TradeType::SellerInitiated } else { TradeType::BuyerInitiated };
            decomposer.record_trade(trade_type, 10.0, i * 1_000_000);
        }
        
        let decomp = decomposer.decompose(50_000_000).unwrap();
        assert!(decomp.total_spread_bps > 0.0);
        assert!(decomp.pin_estimate >= 0.0 && decomp.pin_estimate <= 1.0);
    }

    #[test]
    fn test_insufficient_data() {
        let decomposer = GlostenMilgromDecomposer::new(0.3, 0.7, 100, 0.99).unwrap();
        
        // Only a few trades
        for i in 0..5 {
            decomposer.record_trade(TradeType::BuyerInitiated, 10.0, i * 1_000_000);
        }
        
        let result = decomposer.decompose(5_000_000);
        assert!(result.is_err());
    }
}
