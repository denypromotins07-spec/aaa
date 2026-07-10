//! PnL Attribution for Shadow Models
//!
//! Decomposes hypothetical PnL into skill vs luck components.

use crate::MLOpsError;

/// PnL attribution results
#[derive(Debug, Clone)]
pub struct PnLAttribution {
    /// Total PnL
    pub total_pnl: f64,
    /// PnL from alpha (skill)
    pub alpha_pnl: f64,
    /// PnL from beta (market exposure)
    pub beta_pnl: f64,
    /// PnL from transaction costs
    pub cost_pnl: f64,
    /// Information ratio
    pub information_ratio: f64,
}

/// PnL attribution calculator
pub struct PnLAttributionEngine {
    /// Running sum of alpha PnL
    alpha_sum: f64,
    /// Running sum of beta PnL
    beta_sum: f64,
    /// Running sum of costs
    cost_sum: f64,
    /// Tracking error accumulator
    tracking_error_sq: f64,
    /// Sample count
    n_samples: u64,
    /// Benchmark returns for beta calculation
    benchmark_returns: Vec<f64>,
}

impl PnLAttributionEngine {
    /// Create new PnL attribution engine
    pub fn new() -> Self {
        Self {
            alpha_sum: 0.0,
            beta_sum: 0.0,
            cost_sum: 0.0,
            tracking_error_sq: 0.0,
            n_samples: 0,
            benchmark_returns: Vec::new(),
        }
    }

    /// Record a trade's PnL decomposition
    pub fn record_trade(
        &mut self,
        total_pnl: f64,
        market_return: f64,
        beta: f64,
        transaction_cost: f64,
    ) -> Result<(), MLOpsError> {
        // Beta PnL = beta * market_return
        let beta_pnl = beta * market_return;
        
        // Alpha PnL = total - beta - costs
        let alpha_pnl = total_pnl - beta_pnl - transaction_cost;

        self.alpha_sum += alpha_pnl;
        self.beta_sum += beta_pnl;
        self.cost_sum += transaction_cost;

        // Track tracking error (alpha volatility)
        self.tracking_error_sq += alpha_pnl * alpha_pnl;
        self.n_samples += 1;

        Ok(())
    }

    /// Set benchmark returns for analysis
    pub fn set_benchmark(&mut self, returns: &[f64]) {
        self.benchmark_returns = returns.to_vec();
    }

    /// Get attribution summary
    pub fn get_attribution(&self) -> PnLAttribution {
        let total_pnl = self.alpha_sum + self.beta_sum - self.cost_sum;
        
        let information_ratio = if self.n_samples > 0 {
            let mean_alpha = self.alpha_sum / self.n_samples as f64;
            let tracking_error = (self.tracking_error_sq / self.n_samples as f64).sqrt();
            
            if tracking_error > 0.0 {
                (mean_alpha * 252.0_f64.sqrt()) / (tracking_error * 252.0_f64.sqrt())
            } else {
                0.0
            }
        } else {
            0.0
        };

        PnLAttribution {
            total_pnl,
            alpha_pnl: self.alpha_sum,
            beta_pnl: self.beta_sum,
            cost_pnl: -self.cost_sum,
            information_ratio,
        }
    }

    /// Reset all accumulators
    pub fn reset(&mut self) {
        self.alpha_sum = 0.0;
        self.beta_sum = 0.0;
        self.cost_sum = 0.0;
        self.tracking_error_sq = 0.0;
        self.n_samples = 0;
    }
}

impl Default for PnLAttributionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pnl_attribution() {
        let mut engine = PnLAttributionEngine::new();

        // Record trades with positive alpha
        for _ in 0..100 {
            engine.record_trade(0.001, 0.0005, 0.8, 0.0001).unwrap();
        }

        let attr = engine.get_attribution();
        
        assert!(attr.total_pnl > 0.0);
        assert!(attr.alpha_pnl > 0.0);
        assert!(attr.beta_pnl > 0.0);
        assert!(attr.cost_pnl < 0.0);
    }
}
