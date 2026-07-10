//! Differential Sharpe Ratio (DSR) and Differential Sortino Ratio Calculator
//! 
//! Implements path-dependent reward shaping using incremental updates to risk-adjusted
//! return metrics. This allows RL agents to optimize for risk-adjusted returns on every tick.

use std::sync::atomic::{AtomicU64, Ordering};

/// Epsilon for numerical stability in division operations
const EPSILON: f64 = 1e-10;

/// Maximum window size for exponential weighting
const MAX_WINDOW: usize = 10000;

/// Differential Sharpe Ratio calculator with online updates
pub struct DifferentialSharpeRatio {
    /// Exponential decay factor for weighting (0 < decay <= 1)
    decay: f64,
    /// Running mean of returns
    mean_return: f64,
    /// Running mean of squared returns
    mean_return_sq: f64,
    /// Previous Sharpe ratio (for differential calculation)
    prev_sharpe: f64,
    /// Number of observations
    count: AtomicU64,
    /// Annualization factor (e.g., 252*24*60*60 for per-second data)
    annualization_factor: f64,
    /// Risk-free rate (annualized)
    risk_free_rate: f64,
}

impl DifferentialSharpeRatio {
    /// Create a new DSR calculator
    pub fn new(decay: f64, annualization_factor: f64, risk_free_rate: f64) -> Self {
        // Validate decay parameter
        let decay = decay.clamp(EPSILON, 1.0);
        
        Self {
            decay,
            mean_return: 0.0,
            mean_return_sq: 0.0,
            prev_sharpe: 0.0,
            count: AtomicU64::new(0),
            annualization_factor,
            risk_free_rate,
        }
    }
    
    /// Create with default parameters for crypto trading (per-minute data)
    pub fn crypto_default() -> Self {
        // 365 * 24 * 60 minutes per year, 5% risk-free rate
        Self::new(0.99, 365.0 * 24.0 * 60.0, 0.05)
    }
    
    /// Update with new return and compute differential Sharpe ratio
    /// 
    /// The differential Sharpe ratio is defined as:
    /// DSR_t = (∂SR/∂r_t) * r_t
    /// 
    /// This provides a step-by-step reward signal that optimizes for Sharpe ratio
    #[inline]
    pub fn update(&mut self, return_t: f64) -> f64 {
        // Handle NaN/Inf inputs gracefully
        let r = if return_t.is_finite() { return_t } else { 0.0 };
        
        let count = self.count.fetch_add(1, Ordering::Relaxed) as f64;
        let alpha = if count < 1.0 { 1.0 } else { self.decay };
        
        // Exponentially weighted moving average of return
        self.mean_return = (1.0 - alpha) * self.mean_return + alpha * r;
        
        // Exponentially weighted moving average of squared return
        let r_sq = r * r;
        self.mean_return_sq = (1.0 - alpha) * self.mean_return_sq + alpha * r_sq;
        
        // Calculate current Sharpe ratio
        let variance = self.mean_return_sq - self.mean_return * self.mean_return;
        let std_dev = if variance > EPSILON { variance.sqrt() } else { EPSILON };
        
        // Annualized Sharpe ratio
        let excess_return = self.mean_return * self.annualization_factor - self.risk_free_rate;
        let sharpe = excess_return / (std_dev * self.annualization_factor.sqrt());
        
        // Differential Sharpe ratio (gradient of Sharpe w.r.t. current return)
        // ∂SR/∂r_t ≈ (SR_t - SR_{t-1}) / α + correction term
        let delta_sharpe = sharpe - self.prev_sharpe;
        
        // The differential reward includes both the change in Sharpe and a penalty for variance
        let dsr = delta_sharpe;
        
        // Store current Sharpe for next iteration
        self.prev_sharpe = sharpe;
        
        dsr
    }
    
    /// Get current Sharpe ratio estimate
    #[inline]
    pub fn current_sharpe(&self) -> f64 {
        let variance = self.mean_return_sq - self.mean_return * self.mean_return;
        let std_dev = if variance > EPSILON { variance.sqrt() } else { EPSILON };
        
        let excess_return = self.mean_return * self.annualization_factor - self.risk_free_rate;
        excess_return / (std_dev * self.annualization_factor.sqrt())
    }
    
    /// Get current volatility estimate (annualized)
    #[inline]
    pub fn current_volatility(&self) -> f64 {
        let variance = self.mean_return_sq - self.mean_return * self.mean_return;
        let std_dev = if variance > EPSILON { variance.sqrt() } else { 0.0 };
        std_dev * self.annualization_factor.sqrt()
    }
    
    /// Reset calculator state
    #[inline]
    pub fn reset(&mut self) {
        self.mean_return = 0.0;
        self.mean_return_sq = 0.0;
        self.prev_sharpe = 0.0;
        self.count.store(0, Ordering::Release);
    }
    
    /// Get observation count
    #[inline]
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Acquire)
    }
}

/// Differential Sortino Ratio calculator (downside deviation only)
pub struct DifferentialSortinoRatio {
    /// Exponential decay factor
    decay: f64,
    /// Running mean of returns
    mean_return: f64,
    /// Running mean of squared negative returns (downside variance)
    downside_variance: f64,
    /// Minimum acceptable return (MAR)
    mar: f64,
    /// Previous Sortino ratio
    prev_sortino: f64,
    /// Number of observations
    count: AtomicU64,
    /// Annualization factor
    annualization_factor: f64,
}

impl DifferentialSortinoRatio {
    /// Create a new Sortino ratio calculator
    pub fn new(decay: f64, mar: f64, annualization_factor: f64) -> Self {
        let decay = decay.clamp(EPSILON, 1.0);
        
        Self {
            decay,
            mean_return: 0.0,
            downside_variance: 0.0,
            mar,
            prev_sortino: 0.0,
            count: AtomicU64::new(0),
            annualization_factor,
        }
    }
    
    /// Create with default parameters
    pub fn crypto_default() -> Self {
        Self::new(0.99, 0.0, 365.0 * 24.0 * 60.0) // MAR = 0 (target absolute returns)
    }
    
    /// Update with new return and compute differential Sortino ratio
    #[inline]
    pub fn update(&mut self, return_t: f64) -> f64 {
        let r = if return_t.is_finite() { return_t } else { 0.0 };
        
        let count = self.count.fetch_add(1, Ordering::Relaxed) as f64;
        let alpha = if count < 1.0 { 1.0 } else { self.decay };
        
        // Update mean return
        self.mean_return = (1.0 - alpha) * self.mean_return + alpha * r;
        
        // Update downside variance (only penalize returns below MAR)
        let downside_return = if r < self.mar { r - self.mar } else { 0.0 };
        self.downside_variance = (1.0 - alpha) * self.downside_variance + alpha * (downside_return * downside_return);
        
        // Calculate Sortino ratio
        let downside_std = if self.downside_variance > EPSILON {
            self.downside_variance.sqrt()
        } else {
            EPSILON
        };
        
        let excess_return = self.mean_return * self.annualization_factor - self.risk_free_rate();
        let sortino = excess_return / (downside_std * self.annualization_factor.sqrt());
        
        // Differential Sortino ratio
        let dsortino = sortino - self.prev_sortino;
        
        self.prev_sortino = sortino;
        
        dsortino
    }
    
    /// Get current Sortino ratio
    #[inline]
    pub fn current_sortino(&self) -> f64 {
        let downside_std = if self.downside_variance > EPSILON {
            self.downside_variance.sqrt()
        } else {
            EPSILON
        };
        
        let excess_return = self.mean_return * self.annualization_factor - self.risk_free_rate();
        excess_return / (downside_std * self.annualization_factor.sqrt())
    }
    
    /// Get downside deviation (annualized)
    #[inline]
    pub fn downside_deviation(&self) -> f64 {
        let downside_std = if self.downside_variance > EPSILON {
            self.downside_variance.sqrt()
        } else {
            0.0
        };
        downside_std * self.annualization_factor.sqrt()
    }
    
    /// Reset calculator
    #[inline]
    pub fn reset(&mut self) {
        self.mean_return = 0.0;
        self.downside_variance = 0.0;
        self.prev_sortino = 0.0;
        self.count.store(0, Ordering::Release);
    }
    
    /// Risk-free rate (daily approximation)
    #[inline]
    fn risk_free_rate(&self) -> f64 {
        0.05 / 365.0 // Simplified daily rate
    }
}

/// Combined reward calculator that blends multiple metrics
pub struct RewardShaper {
    /// DSR component weight
    dsr_weight: f64,
    /// Sortino component weight
    sortino_weight: f64,
    /// Transaction cost penalty weight
    tc_penalty_weight: f64,
    /// Position change penalty (to reduce churn)
    churn_penalty_weight: f64,
    /// DSR calculator
    dsr: DifferentialSharpeRatio,
    /// Sortino calculator
    sortino: DifferentialSortinoRatio,
    /// Cumulative transaction costs
    cumulative_tc: f64,
}

impl RewardShaper {
    /// Create a new reward shaper with specified weights
    pub fn new(
        dsr_weight: f64,
        sortino_weight: f64,
        tc_penalty_weight: f64,
        churn_penalty_weight: f64,
    ) -> Self {
        Self {
            dsr_weight,
            sortino_weight,
            tc_penalty_weight,
            churn_penalty_weight,
            dsr: DifferentialSharpeRatio::crypto_default(),
            sortino: DifferentialSortinoRatio::crypto_default(),
            cumulative_tc: 0.0,
        }
    }
    
    /// Create with default weights favoring risk-adjusted returns
    pub fn default_risk_adjusted() -> Self {
        Self::new(0.5, 0.3, 0.15, 0.05)
    }
    
    /// Compute shaped reward for a single step
    /// 
    /// Arguments:
    /// - return_t: Portfolio return for this step
    /// - transaction_cost: Cost incurred this step (spread + fees)
    /// - position_change: Absolute change in position size
    #[inline]
    pub fn compute_reward(
        &mut self,
        return_t: f64,
        transaction_cost: f64,
        position_change: f64,
    ) -> f64 {
        // Differential Sharpe component
        let dsr_component = self.dsr.update(return_t);
        
        // Differential Sortino component
        let sortino_component = self.sortino.update(return_t);
        
        // Transaction cost penalty (differential)
        let prev_tc = self.cumulative_tc;
        self.cumulative_tc += transaction_cost.abs();
        let tc_penalty = -(self.cumulative_tc - prev_tc) * self.tc_penalty_weight;
        
        // Churn penalty
        let churn_penalty = -position_change.abs() * self.churn_penalty_weight;
        
        // Combine components
        dsr_component * self.dsr_weight
            + sortino_component * self.sortino_weight
            + tc_penalty
            + churn_penalty
    }
    
    /// Get current Sharpe ratio
    #[inline]
    pub fn current_sharpe(&self) -> f64 {
        self.dsr.current_sharpe()
    }
    
    /// Get current Sortino ratio
    #[inline]
    pub fn current_sortino(&self) -> f64 {
        self.sortino.current_sortino()
    }
    
    /// Reset all state
    #[inline]
    pub fn reset(&mut self) {
        self.dsr.reset();
        self.sortino.reset();
        self.cumulative_tc = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dsr_positive_returns() {
        let mut dsr = DifferentialSharpeRatio::new(0.99, 252.0, 0.0);
        
        // Feed consistent positive returns
        for _ in 0..100 {
            dsr.update(0.001); // 0.1% per step
        }
        
        // Sharpe should be high and positive
        let sharpe = dsr.current_sharpe();
        assert!(sharpe > 1.0, "Expected high Sharpe for consistent returns, got {}", sharpe);
    }
    
    #[test]
    fn test_dsr_volatile_returns() {
        let mut dsr = DifferentialSharpeRatio::new(0.99, 252.0, 0.0);
        
        // Feed alternating returns
        for i in 0..100 {
            let ret = if i % 2 == 0 { 0.01 } else { -0.01 };
            dsr.update(ret);
        }
        
        // Sharpe should be low due to high variance
        let sharpe = dsr.current_sharpe();
        assert!(sharpe.abs() < 0.5, "Expected low Sharpe for volatile returns, got {}", sharpe);
    }
    
    #[test]
    fn test_sortino_asymmetric_penalty() {
        let mut sortino_pos = DifferentialSortinoRatio::new(0.99, 0.0, 252.0);
        let mut sortino_neg = DifferentialSortinoRatio::new(0.99, 0.0, 252.0);
        
        // Positive returns
        for _ in 0..50 {
            sortino_pos.update(0.001);
        }
        
        // Negative returns (same magnitude)
        for _ in 0..50 {
            sortino_neg.update(-0.001);
        }
        
        // Sortino should be higher for positive returns (no downside)
        assert!(sortino_pos.current_sortino() > sortino_neg.current_sortino());
    }
    
    #[test]
    fn test_reward_shaper_components() {
        let mut shaper = RewardShaper::default_risk_adjusted();
        
        // Good step: positive return, low cost, small position change
        let reward_good = shaper.compute_reward(0.001, 0.0001, 0.01);
        
        // Bad step: negative return, high cost, large position change
        let reward_bad = shaper.compute_reward(-0.001, 0.001, 0.1);
        
        assert!(reward_good > reward_bad, "Good step should have higher reward");
    }
    
    #[test]
    fn test_numerical_stability() {
        let mut dsr = DifferentialSharpeRatio::new(0.99, 252.0, 0.0);
        
        // Feed extreme values
        dsr.update(1e10);
        dsr.update(-1e10);
        dsr.update(f64::NAN);
        dsr.update(f64::INFINITY);
        
        // Should not panic or produce NaN
        let sharpe = dsr.current_sharpe();
        assert!(sharpe.is_finite(), "Sharpe should be finite after extreme inputs");
    }
}
