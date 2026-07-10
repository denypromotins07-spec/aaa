//! Bayesian Informed Flow Updater for real-time toxicity estimation.
//! 
//! Implements a sequential Bayesian updater to estimate the probability
//! that current order flow is driven by informed traders.

use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BayesianFlowError {
    #[error("Invalid prior probability")]
    InvalidPrior,
    #[error("Numerical instability detected")]
    NumericalInstability,
}

/// Bayesian informed flow state
#[derive(Debug, Clone)]
pub struct InformedFlowState {
    /// Posterior probability of informed trading
    pub posterior_prob: f64,
    /// Log-odds ratio for numerical stability
    pub log_odds: f64,
    /// Number of observations
    pub observation_count: u64,
    /// Timestamp
    pub timestamp_ns: u64,
}

/// Configuration for Bayesian updater
pub struct BayesianFlowConfig {
    /// Prior probability of informed trading (0-1)
    pub prior_alpha: f64,
    /// Probability of buy given informed trader has good news
    pub delta: f64,
    /// Arrival rate of informed traders
    pub mu_informed: f64,
    /// Arrival rate of uninformed traders
    pub mu_uninformed: f64,
    /// Decay factor for old observations
    pub decay_factor: f64,
}

impl Default for BayesianFlowConfig {
    fn default() -> Self {
        Self {
            prior_alpha: 0.2,
            delta: 0.7,
            mu_informed: 0.1,
            mu_uninformed: 0.5,
            decay_factor: 0.999,
        }
    }
}

/// Bayesian Informed Flow Updater
pub struct BayesianInformedFlowUpdater {
    config: BayesianFlowConfig,
    /// Current log-odds (for numerical stability)
    log_odds: RwLock<f64>,
    /// Observation count
    observation_count: AtomicU64,
    /// Last update timestamp
    last_update_ns: AtomicU64,
}

impl BayesianInformedFlowUpdater {
    /// Create a new Bayesian informed flow updater
    pub fn new(config: BayesianFlowConfig) -> Result<Self, BayesianFlowError> {
        if config.prior_alpha <= 0.0 || config.prior_alpha >= 1.0 {
            return Err(BayesianFlowError::InvalidPrior);
        }
        if config.delta <= 0.0 || config.delta >= 1.0 {
            return Err(BayesianFlowError::InvalidPrior);
        }
        
        let initial_log_odds = (config.prior_alpha / (1.0 - config.prior_alpha)).ln();
        
        Ok(Self {
            config,
            log_odds: RwLock::new(initial_log_odds),
            observation_count: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
        })
    }

    /// Update with a new trade observation
    /// 
    /// Uses sequential Bayes rule:
    /// log_odds_new = log_odds_old + log(P(data|informed) / P(data|uninformed))
    #[inline]
    pub fn update(&self, is_buy: bool, volume: u64, timestamp_ns: u64) -> Result<InformedFlowState, BayesianFlowError> {
        let mut log_odds = self.log_odds.write();
        
        // Calculate likelihood ratio for this observation
        // P(buy|informed) = δ, P(buy|uninformed) = 0.5
        // P(sell|informed) = 1-δ, P(sell|uninformed) = 0.5
        
        let (p_data_informed, p_data_uninformed) = if is_buy {
            (self.config.delta, 0.5)
        } else {
            (1.0 - self.config.delta, 0.5)
        };
        
        // Weight by volume (larger trades more informative)
        let volume_weight = (volume as f64).ln().max(1.0) / 10.0; // Normalize
        
        let likelihood_ratio = if p_data_uninformed > 1e-10 && p_data_informed > 1e-10 {
            (p_data_informed / p_data_uninformed).ln() * volume_weight
        } else {
            return Err(BayesianFlowError::NumericalInstability);
        };
        
        // Update log-odds
        *log_odds += likelihood_ratio;
        
        // Apply time decay based on time since last update
        let last_ts = self.last_update_ns.load(Ordering::Relaxed);
        if last_ts > 0 {
            let dt_ms = (timestamp_ns.saturating_sub(last_ts)) as f64 / 1_000_000.0;
            let time_decay = (-dt_ms / 1000.0).exp(); // 1 second half-life
            *log_odds *= time_decay.max(self.config.decay_factor);
        }
        
        // Update counters
        self.observation_count.fetch_add(1, Ordering::Relaxed);
        self.last_update_ns.store(timestamp_ns, Ordering::Relaxed);
        
        // Convert log-odds to probability
        let posterior_prob = 1.0 / (1.0 + (-*log_odds).exp());
        
        Ok(InformedFlowState {
            posterior_prob,
            log_odds: *log_odds,
            observation_count: self.observation_count.load(Ordering::Relaxed),
            timestamp_ns,
        })
    }

    /// Get current informed flow probability
    pub fn get_current_prob(&self) -> f64 {
        let log_odds = self.log_odds.read();
        1.0 / (1.0 + (-*log_odds).exp())
    }

    /// Check if flow is currently informed (above threshold)
    pub fn is_informed(&self, threshold: f64) -> bool {
        self.get_current_prob() > threshold
    }

    /// Get the current log-odds value
    pub fn get_log_odds(&self) -> f64 {
        *self.log_odds.read()
    }

    /// Reset to prior
    pub fn reset(&self) {
        let mut log_odds = self.log_odds.write();
        *log_odds = (self.config.prior_alpha / (1.0 - self.config.prior_alpha)).ln();
        self.observation_count.store(0, Ordering::Relaxed);
        self.last_update_ns.store(0, Ordering::Relaxed);
    }

    /// Get observation count
    pub fn get_observation_count(&self) -> u64 {
        self.observation_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_update_buy_sequence() {
        let updater = BayesianInformedFlowUpdater::new(BayesianFlowConfig::default()).unwrap();
        
        // Sequence of buys should increase informed probability
        let mut prob = updater.get_current_prob();
        
        for i in 0..20 {
            let state = updater.update(true, 1000, i * 1_000_000).unwrap();
            assert!(state.posterior_prob >= prob || i == 0);
            prob = state.posterior_prob;
        }
        
        // Final probability should be elevated
        assert!(prob > 0.5);
    }

    #[test]
    fn test_bayesian_update_mixed_sequence() {
        let updater = BayesianInformedFlowUpdater::new(BayesianFlowConfig::default()).unwrap();
        
        // Mixed sequence should keep probability near prior
        for i in 0..40 {
            let is_buy = i % 2 == 0;
            updater.update(is_buy, 100, i * 1_000_000).unwrap();
        }
        
        let prob = updater.get_current_prob();
        // Should be closer to prior (0.2) than extreme
        assert!(prob > 0.1 && prob < 0.5);
    }

    #[test]
    fn test_invalid_prior() {
        let config = BayesianFlowConfig {
            prior_alpha: 1.5, // Invalid
            ..Default::default()
        };
        let result = BayesianInformedFlowUpdater::new(config);
        assert!(result.is_err());
    }
}
