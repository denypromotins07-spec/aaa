//! Regime-conditional alpha router.
//!
//! Dynamically allocates capital between different alpha strategies
//! based on the current HMM regime posterior probabilities.

use crate::regimes::bayesian_hmm::{RegimeType, RegimePosterior};

/// Alpha strategy types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlphaStrategy {
    /// Trend-following (works in trending regimes)
    SmcAlpha,
    /// Mean-reversion (works in range-bound regimes)
    StatArb,
    /// Momentum
    Momentum,
    /// Carry trade
    Carry,
    /// Volatility trading
    Volatility,
}

impl AlphaStrategy {
    pub fn all() -> Vec<Self> {
        vec![
            Self::SmcAlpha,
            Self::StatArb,
            Self::Momentum,
            Self::Carry,
            Self::Volatility,
        ]
    }
}

/// Allocation result for each strategy
#[derive(Debug, Clone)]
pub struct StrategyAllocation {
    pub strategy: AlphaStrategy,
    /// Weight in [0, 1]
    pub weight: f64,
    /// Expected Sharpe in current regime
    pub expected_sharpe: f64,
    /// Risk adjustment factor
    pub risk_adjustment: f64,
}

/// Regime-conditional alpha router
pub struct RegimeConditionalRouter {
    /// Strategy performance by regime: Map<(strategy, regime), sharpe>
    strategy_regime_performance: Vec<((AlphaStrategy, RegimeType), f64)>,
    /// Current regime posterior
    current_posterior: Option<RegimePosterior>,
    /// Minimum allocation per strategy
    min_allocation: f64,
    /// Maximum allocation per strategy
    max_allocation: f64,
    /// Turnover penalty (to reduce churning)
    turnover_penalty: f64,
    /// Previous allocations (for turnover calculation)
    previous_allocations: Vec<f64>,
}

impl RegimeConditionalRouter {
    /// Create new router with default parameters
    pub fn new() -> Self {
        Self {
            strategy_regime_performance: Vec::new(),
            current_posterior: None,
            min_allocation: 0.0,
            max_allocation: 1.0,
            turnover_penalty: 0.01,
            previous_allocations: vec![0.0; AlphaStrategy::all().len()],
        }
    }

    /// Set expected Sharpe for a strategy in a specific regime
    pub fn set_strategy_regime_performance(
        &mut self,
        strategy: AlphaStrategy,
        regime: RegimeType,
        expected_sharpe: f64,
    ) {
        // Remove existing entry if present
        self.strategy_regime_performance.retain(|((s, r), _)| {
            !(*s == strategy && *r == regime)
        });

        self.strategy_regime_performance.push(((strategy, regime), expected_sharpe));
    }

    /// Get expected Sharpe for strategy given regime posterior
    fn get_expected_sharpe(&self, strategy: AlphaStrategy, posterior: &RegimePosterior) -> f64 {
        let mut expected = 0.0;

        for ((strat, regime), sharpe) in &self.strategy_regime_performance {
            if *strat == strategy {
                let regime_prob = posterior.probabilities[regime.to_index()];
                expected += regime_prob * sharpe;
            }
        }

        expected
    }

    /// Compute optimal allocation based on current regime
    pub fn compute_allocation(&mut self, posterior: RegimePosterior) -> Vec<StrategyAllocation> {
        self.current_posterior = Some(posterior.clone());
        let posterior_ref = self.current_posterior.as_ref().unwrap();

        let strategies = AlphaStrategy::all();
        let n_strategies = strategies.len();

        // Compute raw scores based on expected Sharpe
        let mut scores: Vec<(AlphaStrategy, f64)> = Vec::with_capacity(n_strategies);

        for strategy in &strategies {
            let expected_sharpe = self.get_expected_sharpe(*strategy, posterior_ref);
            scores.push((*strategy, expected_sharpe));
        }

        // Apply softmax to convert scores to weights
        let weights = self.softmax_with_turnover(&scores);

        // Build allocation results
        let mut allocations = Vec::with_capacity(n_strategies);

        for (i, (strategy, expected_sharpe)) in scores.into_iter().enumerate() {
            let weight = weights[i];
            
            // Compute risk adjustment based on regime uncertainty
            let entropy = posterior_ref.entropy;
            let max_entropy = (posterior_ref.probabilities.len() as f64).ln();
            let risk_adjustment = 1.0 - (entropy / max_entropy.max(1e-10));

            allocations.push(StrategyAllocation {
                strategy,
                weight,
                expected_sharpe,
                risk_adjustment,
            });
        }

        // Store for next iteration's turnover calculation
        self.previous_allocations = allocations.iter().map(|a| a.weight).collect();

        allocations
    }

    /// Softmax with turnover penalty
    fn softmax_with_turnover(
        &self,
        scores: &[(AlphaStrategy, f64)],
    ) -> Vec<f64> {
        let temperature = 1.0;
        let n = scores.len();

        // Adjust scores based on turnover penalty
        let adjusted_scores: Vec<f64> = scores.iter().enumerate().map(|(i, (_, score))| {
            let prev_weight = if i < self.previous_allocations.len() {
                self.previous_allocations[i]
            } else {
                0.0
            };

            // Penalize large changes from previous allocation
            let turnover_adjustment = self.turnover_penalty * (1.0 - prev_weight).abs();
            score - turnover_adjustment
        }).collect();

        // Compute softmax
        let max_score = adjusted_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = adjusted_scores.iter()
            .map(|&s| ((s - max_score) / temperature).exp())
            .collect();

        let sum_exp: f64 = exp_scores.iter().sum();
        
        if sum_exp < 1e-15 {
            // Fallback to uniform
            return vec![1.0 / n as f64; n];
        }

        let mut weights: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

        // Apply min/max constraints
        for w in &mut weights {
            *w = w.clamp(self.min_allocation, self.max_allocation);
        }

        // Renormalize after clamping
        let sum: f64 = weights.iter().sum();
        if sum > 1e-15 {
            weights.iter_mut().for_each(|w| *w /= sum);
        }

        weights
    }

    /// Get current regime estimate
    pub fn current_regime(&self) -> Option<RegimeType> {
        self.current_posterior.as_ref().map(|p| p.dominant_regime)
    }

    /// Get regime uncertainty (entropy)
    pub fn regime_uncertainty(&self) -> Option<f64> {
        self.current_posterior.as_ref().map(|p| p.entropy)
    }

    /// Set turnover penalty
    pub fn set_turnover_penalty(&mut self, penalty: f64) {
        self.turnover_penalty = penalty.clamp(0.0, 1.0);
    }

    /// Set allocation bounds
    pub fn set_allocation_bounds(&mut self, min: f64, max: f64) {
        self.min_allocation = min.clamp(0.0, 1.0);
        self.max_allocation = max.clamp(0.0, 1.0);
    }
}

impl Default for RegimeConditionalRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_router_allocation() {
        let mut router = RegimeConditionalRouter::new();

        // SMC Alpha works well in Risk-On (trending)
        router.set_strategy_regime_performance(AlphaStrategy::SmcAlpha, RegimeType::RiskOn, 1.5);
        router.set_strategy_regime_performance(AlphaStrategy::SmcAlpha, RegimeType::RiskOff, -0.5);

        // Stat Arb works well in Goldilocks (stable)
        router.set_strategy_regime_performance(AlphaStrategy::StatArb, RegimeType::Goldilocks, 1.2);
        router.set_strategy_regime_performance(AlphaStrategy::StatArb, RegimeType::RiskOff, 0.3);

        // Create posterior favoring Risk-On
        let mut posterior = RegimePosterior::new(4);
        posterior.probabilities = Array1::from_vec(vec![0.8, 0.1, 0.05, 0.05]);
        posterior.dominant_regime = RegimeType::RiskOn;
        posterior.entropy = 0.5;

        let allocations = router.compute_allocation(posterior);

        // SMC Alpha should have highest weight in Risk-On
        let smc_alloc = allocations.iter()
            .find(|a| a.strategy == AlphaStrategy::SmcAlpha)
            .unwrap();

        assert!(smc_alloc.weight > 0.3); // Should be significant
        assert!(smc_alloc.expected_sharpe > 1.0); // Weighted average should be positive
    }
}
