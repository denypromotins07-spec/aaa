//! Real-time Options Greeks aggregator.
//! 
//! Tracks Delta, Gamma, Vega, Theta, and Rho across the portfolio,
//! updating atomically as market conditions change.

use std::sync::atomic::{AtomicU64, Ordering};
use std::f64::consts::{E, PI};

/// Epsilon for numerical stability
const EPSILON: f64 = 1e-10;

/// Standard normal CDF approximation (Abramowitz & Stegun)
#[inline]
fn norm_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327 * (-x * x / 2.0).exp();
    let prob = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if x > 0.0 {
        1.0 - prob
    } else {
        prob
    }
}

/// Standard normal PDF
#[inline]
fn norm_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.3989422804014327;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Black-Scholes d1 and d2 calculations
#[inline]
fn black_scholes_d1_d2(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> (f64, f64) {
    let sqrt_t = t.max(EPSILON).sqrt();
    let d1 = (s / k).ln().max(-100.0).min(100.0) / (sigma * sqrt_t)
        + (r + 0.5 * sigma * sigma) * sqrt_t / sigma;
    let d2 = d1 - sigma * sqrt_t;
    (d1, d2)
}

/// Option type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
}

/// Greeks for a single option position
#[derive(Debug, Clone, Copy)]
pub struct OptionGreeks {
    /// Delta: rate of change of option price with respect to underlying price
    pub delta: f64,
    /// Gamma: rate of change of delta with respect to underlying price
    pub gamma: f64,
    /// Vega: rate of change of option price with respect to volatility
    pub vega: f64,
    /// Theta: rate of change of option price with respect to time (per day)
    pub theta: f64,
    /// Rho: rate of change of option price with respect to interest rate
    pub rho: f64,
}

impl OptionGreeks {
    /// Calculate Greeks for a single option using Black-Scholes model
    #[inline]
    pub fn calculate(
        option_type: OptionType,
        spot: f64,
        strike: f64,
        time_to_expiry: f64, // in years
        volatility: f64,
        risk_free_rate: f64,
        quantity: i64, // Positive for long, negative for short
    ) -> Self {
        let (d1, d2) = black_scholes_d1_d2(spot, strike, time_to_expiry, risk_free_rate, volatility);
        
        let sqrt_t = time_to_expiry.max(EPSILON).sqrt();
        let pdf_d1 = norm_pdf(d1);
        let cdf_d1 = norm_cdf(d1);
        let cdf_d2 = norm_cdf(d2);
        let cdf_neg_d1 = 1.0 - cdf_d1;
        let cdf_neg_d2 = 1.0 - cdf_d2;
        
        let q = quantity as f64;
        
        match option_type {
            OptionType::Call => {
                // Delta = N(d1) for long call
                let delta = q * cdf_d1;
                
                // Gamma = N'(d1) / (S * σ * √T)
                let gamma = q * pdf_d1 / (spot * volatility * sqrt_t);
                
                // Vega = S * N'(d1) * √T
                let vega = q * spot * pdf_d1 * sqrt_t / 100.0; // Per 1% vol change
                
                // Theta = -S * N'(d1) * σ / (2√T) - rK*e^(-rT)*N(d2)
                let term1 = -spot * pdf_d1 * volatility / (2.0 * sqrt_t);
                let term2 = -risk_free_rate * strike * (-risk_free_rate * time_to_expiry).exp() * cdf_d2;
                let theta = q * (term1 + term2) / 365.0; // Per day
                
                // Rho = K * T * e^(-rT) * N(d2)
                let rho = q * strike * time_to_expiry * (-risk_free_rate * time_to_expiry).exp() * cdf_d2 / 100.0;
                
                Self { delta, gamma, vega, theta, rho }
            }
            OptionType::Put => {
                // Delta = N(d1) - 1 for long put
                let delta = q * (cdf_d1 - 1.0);
                
                // Gamma is same as call
                let gamma = q * pdf_d1 / (spot * volatility * sqrt_t);
                
                // Vega is same as call
                let vega = q * spot * pdf_d1 * sqrt_t / 100.0;
                
                // Theta for put
                let term1 = -spot * pdf_d1 * volatility / (2.0 * sqrt_t);
                let term2 = risk_free_rate * strike * (-risk_free_rate * time_to_expiry).exp() * cdf_neg_d2;
                let theta = q * (term1 - term2) / 365.0;
                
                // Rho for put
                let rho = -q * strike * time_to_expiry * (-risk_free_rate * time_to_expiry).exp() * cdf_neg_d2 / 100.0;
                
                Self { delta, gamma, vega, theta, rho }
            }
        }
    }
}

/// Aggregated portfolio Greeks
#[derive(Debug, Clone, Copy)]
pub struct PortfolioGreeks {
    /// Total portfolio delta (equivalent units of underlying)
    pub total_delta: f64,
    /// Total portfolio gamma
    pub total_gamma: f64,
    /// Total portfolio vega (per 1% vol change)
    pub total_vega: f64,
    /// Total portfolio theta (per day)
    pub total_theta: f64,
    /// Total portfolio rho (per 1% rate change)
    pub total_rho: f64,
    /// Dollar delta (delta * spot price)
    pub dollar_delta: f64,
    /// Gamma P&L for 1% move: 0.5 * gamma * (0.01 * S)^2
    pub gamma_pnl_1pct: f64,
}

/// Atomic counters for lock-free updates
struct AtomicGreeks {
    delta_bits: AtomicU64,
    gamma_bits: AtomicU64,
    vega_bits: AtomicU64,
    theta_bits: AtomicU64,
    rho_bits: AtomicU64,
}

impl AtomicGreeks {
    fn new() -> Self {
        Self {
            delta_bits: AtomicU64::new(0),
            gamma_bits: AtomicU64::new(0),
            vega_bits: AtomicU64::new(0),
            theta_bits: AtomicU64::new(0),
            rho_bits: AtomicU64::new(0),
        }
    }
    
    #[inline]
    fn store(&self, greeks: &OptionGreeks) {
        self.delta_bits.store(f64::to_bits(greeks.delta), Ordering::Relaxed);
        self.gamma_bits.store(f64::to_bits(greeks.gamma), Ordering::Relaxed);
        self.vega_bits.store(f64::to_bits(greeks.vega), Ordering::Relaxed);
        self.theta_bits.store(f64::to_bits(greeks.theta), Ordering::Relaxed);
        self.rho_bits.store(f64::to_bits(greeks.rho), Ordering::Relaxed);
    }
    
    #[inline]
    fn load(&self) -> OptionGreeks {
        OptionGreeks {
            delta: f64::from_bits(self.delta_bits.load(Ordering::Relaxed)),
            gamma: f64::from_bits(self.gamma_bits.load(Ordering::Relaxed)),
            vega: f64::from_bits(self.vega_bits.load(Ordering::Relaxed)),
            theta: f64::from_bits(self.theta_bits.load(Ordering::Relaxed)),
            rho: f64::from_bits(self.rho_bits.load(Ordering::Relaxed)),
        }
    }
}

/// Real-time Options Greeks Aggregator
/// 
/// Maintains atomic snapshots of Greeks for each position and aggregates
/// them into portfolio-level metrics without locking.
pub struct PortfolioGreeksAggregator {
    /// Number of option positions tracked
    num_positions: usize,
    /// Atomic storage for each position's Greeks
    position_greeks: Vec<AtomicGreeks>,
    /// Current spot prices for each underlying
    spot_prices: Vec<f64>,
    /// Risk-free rate
    risk_free_rate: f64,
    /// Count of updates
    update_count: AtomicU64,
}

unsafe impl Send for PortfolioGreeksAggregator {}
unsafe impl Sync for PortfolioGreeksAggregator {}

impl PortfolioGreeksAggregator {
    /// Create a new Greeks aggregator for the given number of positions.
    pub fn new(num_positions: usize, risk_free_rate: f64) -> Self {
        let mut position_greeks = Vec::with_capacity(num_positions);
        for _ in 0..num_positions {
            position_greeks.push(AtomicGreeks::new());
        }
        
        Self {
            num_positions,
            position_greeks,
            spot_prices: vec![0.0; num_positions],
            risk_free_rate,
            update_count: AtomicU64::new(0),
        }
    }

    /// Update Greeks for a specific position.
    /// 
    /// # Arguments
    /// * `position_idx` - Index of the position to update
    /// * `option_type` - Call or Put
    /// * `spot` - Current spot price of underlying
    /// * `strike` - Option strike price
    /// * `time_to_expiry` - Time to expiry in years
    /// * `volatility` - Implied volatility (as decimal, e.g., 0.25 for 25%)
    /// * `quantity` - Position quantity (positive for long, negative for short)
    #[inline]
    pub fn update_position(
        &self,
        position_idx: usize,
        option_type: OptionType,
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        volatility: f64,
        quantity: i64,
    ) {
        assert!(position_idx < self.num_positions, "Position index out of bounds");
        
        let greeks = OptionGreeks::calculate(
            option_type,
            spot,
            strike,
            time_to_expiry,
            volatility,
            self.risk_free_rate,
            quantity,
        );
        
        self.position_greeks[position_idx].store(&greeks);
        self.spot_prices[position_idx] = spot;
        
        self.update_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Update spot price for a position (for delta recalculation).
    #[inline]
    pub fn update_spot(&self, position_idx: usize, spot: f64) {
        if position_idx < self.spot_prices.len() {
            self.spot_prices[position_idx] = spot;
        }
    }

    /// Aggregate all position Greeks into portfolio totals.
    #[inline]
    pub fn aggregate(&self) -> PortfolioGreeks {
        let mut total_delta = 0.0;
        let mut total_gamma = 0.0;
        let mut total_vega = 0.0;
        let mut total_theta = 0.0;
        let mut total_rho = 0.0;
        
        for i in 0..self.num_positions {
            let greeks = self.position_greeks[i].load();
            total_delta += greeks.delta;
            total_gamma += greeks.gamma;
            total_vega += greeks.vega;
            total_theta += greeks.theta;
            total_rho += greeks.rho;
        }
        
        // Calculate dollar delta using weighted average spot
        let mut total_notional = 0.0;
        let mut weighted_spot = 0.0;
        for i in 0..self.num_positions {
            let greeks = self.position_greeks[i].load();
            let spot = self.spot_prices[i];
            let notional = greeks.delta.abs() * spot;
            weighted_spot += greeks.delta * spot;
            total_notional += notional;
        }
        
        let dollar_delta = weighted_spot;
        let gamma_pnl_1pct = 0.5 * total_gamma * (0.01 * total_notional / total_delta.max(EPSILON)).powi(2);
        
        PortfolioGreeks {
            total_delta,
            total_gamma,
            total_vega,
            total_theta,
            total_rho,
            dollar_delta,
            gamma_pnl_1pct,
        }
    }

    /// Get aggregated Greeks with delta-neutral adjustment info.
    #[inline]
    pub fn delta_analysis(&self, portfolio_value: f64) -> DeltaAnalysis {
        let greeks = self.aggregate();
        
        // Calculate how many units of underlying needed to be delta-neutral
        let hedge_units = -greeks.total_delta;
        
        // Delta as percentage of portfolio
        let delta_pct = if portfolio_value > EPSILON {
            greeks.dollar_delta / portfolio_value
        } else {
            0.0
        };
        
        DeltaAnalysis {
            portfolio_greeks: greeks,
            hedge_units,
            delta_percentage: delta_pct,
            is_delta_neutral: delta_pct.abs() < 0.01, // Within 1%
        }
    }

    /// Get update statistics
    pub fn stats(&self) -> GreeksStats {
        GreeksStats {
            num_positions: self.num_positions,
            update_count: self.update_count.load(Ordering::Relaxed),
            risk_free_rate: self.risk_free_rate,
        }
    }
}

/// Delta analysis result
#[derive(Debug, Clone)]
pub struct DeltaAnalysis {
    pub portfolio_greeks: PortfolioGreeks,
    /// Units of underlying to buy/sell for delta neutrality
    pub hedge_units: f64,
    /// Delta as percentage of portfolio value
    pub delta_percentage: f64,
    /// Whether portfolio is approximately delta-neutral
    pub is_delta_neutral: bool,
}

/// Statistics from the Greeks aggregator
#[derive(Debug, Clone)]
pub struct GreeksStats {
    pub num_positions: usize,
    pub update_count: u64,
    pub risk_free_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_option_greeks() {
        let greeks = OptionGreeks::calculate(
            OptionType::Call,
            100.0,  // spot
            100.0,  // strike (ATM)
            0.25,   // 3 months to expiry
            0.20,   // 20% vol
            0.05,   // 5% risk-free rate
            1,      // long 1 contract
        );
        
        // ATM call should have delta ~0.5
        assert!(greeks.delta > 0.45 && greeks.delta < 0.55);
        
        // Gamma should be positive
        assert!(greeks.gamma > 0.0);
        
        // Vega should be positive
        assert!(greeks.vega > 0.0);
        
        // Theta should be negative (time decay)
        assert!(greeks.theta < 0.0);
    }

    #[test]
    fn test_put_option_greeks() {
        let greeks = OptionGreeks::calculate(
            OptionType::Put,
            100.0,
            100.0,
            0.25,
            0.20,
            0.05,
            1,
        );
        
        // ATM put should have delta ~-0.5
        assert!(greeks.delta > -0.55 && greeks.delta < -0.45);
        
        // Gamma should still be positive
        assert!(greeks.gamma > 0.0);
        
        // Vega should be positive
        assert!(greeks.vega > 0.0);
    }

    #[test]
    fn test_short_position() {
        let long_greeks = OptionGreeks::calculate(
            OptionType::Call,
            100.0, 100.0, 0.25, 0.20, 0.05, 1,
        );
        
        let short_greeks = OptionGreeks::calculate(
            OptionType::Call,
            100.0, 100.0, 0.25, 0.20, 0.05, -1,
        );
        
        // Short position should have opposite sign Greeks
        assert!((long_greeks.delta + short_greeks.delta).abs() < EPSILON);
        assert!((long_greeks.gamma + short_greeks.gamma).abs() < EPSILON);
        assert!((long_greeks.vega + short_greeks.vega).abs() < EPSILON);
    }

    #[test]
    fn test_portfolio_aggregation() {
        let aggregator = PortfolioGreeksAggregator::new(2, 0.05);
        
        // Long call
        aggregator.update_position(0, OptionType::Call, 100.0, 100.0, 0.25, 0.20, 1);
        
        // Long put (same underlying)
        aggregator.update_position(1, OptionType::Put, 100.0, 100.0, 0.25, 0.20, 1);
        
        let portfolio = aggregator.aggregate();
        
        // Straddle should have near-zero delta (call delta + put delta ≈ 0)
        assert!(portfolio.total_delta.abs() < 0.1);
        
        // But positive gamma and vega
        assert!(portfolio.total_gamma > 0.0);
        assert!(portfolio.total_vega > 0.0);
    }

    #[test]
    fn test_delta_neutral_hedge() {
        let aggregator = PortfolioGreeksAggregator::new(1, 0.05);
        aggregator.update_position(0, OptionType::Call, 100.0, 100.0, 0.25, 0.20, 10);
        
        let analysis = aggregator.delta_analysis(10000.0);
        
        // Should need to short underlying to hedge
        assert!(analysis.hedge_units < 0.0);
        
        // Hedge units magnitude should equal delta
        assert!((analysis.hedge_units + analysis.portfolio_greeks.total_delta).abs() < 0.1);
    }

    #[test]
    fn test_time_decay() {
        // Compare theta for different expiries
        let short_dte = OptionGreeks::calculate(
            OptionType::Call, 100.0, 100.0, 0.02, 0.20, 0.05, 1,
        );
        let long_dte = OptionGreeks::calculate(
            OptionType::Call, 100.0, 100.0, 0.50, 0.20, 0.05, 1,
        );
        
        // Shorter DTE should have more negative theta (faster decay)
        assert!(short_dte.theta < long_dte.theta);
    }

    #[test]
    fn test_volatility_sensitivity() {
        // Compare vega for different vols
        let low_vol = OptionGreeks::calculate(
            OptionType::Call, 100.0, 100.0, 0.25, 0.10, 0.05, 1,
        );
        let high_vol = OptionGreeks::calculate(
            OptionType::Call, 100.0, 100.0, 0.25, 0.40, 0.05, 1,
        );
        
        // Higher vol should have higher vega
        assert!(high_vol.vega > low_vol.vega);
    }

    #[test]
    fn test_norm_cdf_properties() {
        // CDF at 0 should be 0.5
        assert!((norm_cdf(0.0) - 0.5).abs() < 0.001);
        
        // CDF should approach 1 for large positive
        assert!(norm_cdf(5.0) > 0.999);
        
        // CDF should approach 0 for large negative
        assert!(norm_cdf(-5.0) < 0.001);
        
        // Symmetry: Φ(-x) = 1 - Φ(x)
        let x = 1.5;
        assert!((norm_cdf(-x) - (1.0 - norm_cdf(x))).abs() < 0.001);
    }
}
